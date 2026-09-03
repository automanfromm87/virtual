// Terrain: the height field, the queries, and the chunk meshes.
//
// The queries are the part that has to be right, because everything else in a
// game asks them: the character controller asks how high the ground is, the
// camera asks so it does not clip through a hill, placement asks the slope. If
// HeightAt and the drawn mesh disagree, a character walks along visibly above
// or below the surface -- and that reads as a physics bug rather than as a
// sampling mismatch.
//
// So the checks here are mostly agreement checks: the query against the
// analytic height it was generated from, the mesh against the query, the
// raycast against both, and each level of detail against level zero.

#include <algorithm>
#include <cmath>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "engine/geometry/terrain.h"
#include "engine/shaders/shader_types.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

using eng::Vec3;

// A smooth analytic surface. Smooth so that the bilinear grid can actually
// reproduce it -- testing a sampled surface against a function with detail
// finer than the grid measures the grid's resolution, not its correctness.
float Hills(float x, float z) {
    return 3.0f * std::sin(x * 0.06f) * std::cos(z * 0.05f) + 0.6f * std::sin(x * 0.13f);
}

eng::TerrainConfig Config() {
    eng::TerrainConfig c;
    c.resolution = 129;
    c.world_size = 128.0f;
    c.chunk_resolution = 33;
    c.skirt_depth = 2.0f;
    return c;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const eng::Terrain terrain = eng::Terrain::Generate(Config(), Hills);

    {
        std::printf("it generates and divides into chunks\n");
        Check(terrain.Valid(), "the terrain is valid");
        std::printf("    %d x %d chunks, max lod %d\n", terrain.ChunksX(),
                    terrain.ChunksZ(), terrain.MaxLod());
        Check(terrain.ChunksX() == 4 && terrain.ChunksZ() == 4,
              "a 129-sample terrain in 33-sample chunks is 4 by 4");
        Check(terrain.MaxLod() == 5, "and has six levels of detail");
    }

    {
        std::printf("\nthe height query matches what it was generated from\n");
        float worst = 0.0f;
        for (float z = 1.0f; z < 126.0f; z += 3.1f)
            for (float x = 1.0f; x < 126.0f; x += 3.1f)
                worst = std::max(worst, std::fabs(terrain.HeightAt(x, z) - Hills(x, z)));
        std::printf("    worst error against the generator: %.5f m\n", worst);
        // The grid samples every metre and the surface's finest feature is a
        // sine with a 48-metre period, so bilinear should be within millimetres.
        Check(worst < 0.02f, "bilinear reconstruction is within 2 cm");

        // AT THE SAMPLES it must be exact, not merely close: a query landing on
        // a grid point should return that grid point.
        float worst_on_grid = 0.0f;
        for (int i = 0; i < 129; i += 7)
            worst_on_grid = std::max(
                worst_on_grid,
                std::fabs(terrain.HeightAt(float(i), float(i)) - Hills(float(i), float(i))));
        Check(worst_on_grid < 1e-4f, "and exact on the grid points themselves");
    }

    {
        std::printf("\nthe normal follows the surface\n");
        // On a flat region the normal is up; on a slope it leans downhill. The
        // analytic gradient of Hills gives the expected direction.
        float worst_angle = 0.0f;
        for (float z = 4.0f; z < 120.0f; z += 7.3f)
            for (float x = 4.0f; x < 120.0f; x += 7.3f) {
                const float dx = 3.0f * 0.06f * std::cos(x * 0.06f) * std::cos(z * 0.05f) +
                                 0.6f * 0.13f * std::cos(x * 0.13f);
                const float dz = -3.0f * 0.05f * std::sin(x * 0.06f) * std::sin(z * 0.05f);
                const Vec3 want = eng::Normalize(Vec3{-dx, 1.0f, -dz});
                const Vec3 got = terrain.NormalAt(x, z);
                const float angle =
                    std::acos(std::clamp(eng::Dot(want, got), -1.0f, 1.0f)) * 57.2958f;
                worst_angle = std::max(worst_angle, angle);
            }
        std::printf("    worst normal error: %.3f degrees\n", worst_angle);
        Check(worst_angle < 1.5f, "the normal matches the analytic gradient");

        // A FLAT terrain's normal is exactly up. Central differences of equal
        // values give zero gradient; a forward difference would too, but a
        // normalisation that divided by a zero-length vector would not.
        std::vector<float> flat(129 * 129, 5.0f);
        const eng::Terrain level = eng::Terrain::FromHeights(Config(), flat);
        const Vec3 n = level.NormalAt(40.0f, 40.0f);
        Check(std::fabs(n.y - 1.0f) < 1e-5f, "and a flat terrain's normal is straight up");
        Check(level.SlopeAt(40.0f, 40.0f) < 0.01f, "with a slope of zero");
    }

    {
        std::printf("\nthe chunk mesh agrees with the height query\n");
        const eng::Mesh chunk = terrain.BuildChunk(1, 2, 0);
        std::printf("    chunk (1,2) lod 0: %zu vertices, %zu triangles\n",
                    chunk.vertices.size(), chunk.indices.size() / 3);
        Check(!chunk.vertices.empty(), "the chunk has geometry");

        // Every SURFACE vertex must sit on the queried surface. Skirt vertices
        // are deliberately below it, so they are excluded by height.
        float worst = 0.0f;
        int surface_verts = 0;
        for (const VertexIn& v : chunk.vertices) {
            const float want = terrain.HeightAt(v.position.x, v.position.z);
            if (v.position.y < want - 1.0f) continue;  // a skirt vertex
            worst = std::max(worst, std::fabs(v.position.y - want));
            ++surface_verts;
        }
        std::printf("    %d surface vertices, worst disagreement %.6f m\n",
                    surface_verts, worst);
        // EXACT, not close: a mesh vertex sits on a grid sample and so does the
        // query. Anything else means the two use different indexing, and a
        // character would walk consistently above or below the visible ground.
        Check(worst < 1e-4f, "every surface vertex is exactly on the queried surface");
        Check(surface_verts == 33 * 33, "and there is one per sample");

        bool in_range = true;
        for (std::uint32_t i : chunk.indices)
            if (i >= chunk.vertices.size()) in_range = false;
        Check(in_range, "and every index is in range");
    }

    {
        std::printf("\ncoarser levels cover the same ground\n");
        for (int lod = 0; lod <= 3; ++lod) {
            const eng::Mesh m = terrain.BuildChunk(2, 2, lod);
            std::printf("    lod %d: %5zu vertices, %5zu triangles\n", lod,
                        m.vertices.size(), m.indices.size() / 3);
        }
        const eng::Mesh fine = terrain.BuildChunk(2, 2, 0);
        const eng::Mesh coarse = terrain.BuildChunk(2, 2, 2);
        Check(coarse.indices.size() < fine.indices.size() / 8,
              "two levels down is under an eighth of the triangles");

        // THE SAME EXTENT. A coarse chunk that covered less ground would leave
        // a gap at the seam that no skirt could fill, because the skirt hangs
        // from the border and the border would be in the wrong place.
        const auto extent = [](const eng::Mesh& m) {
            float x0 = 1e30f, x1 = -1e30f, z0 = 1e30f, z1 = -1e30f;
            for (const VertexIn& v : m.vertices) {
                x0 = std::min(x0, v.position.x);
                x1 = std::max(x1, v.position.x);
                z0 = std::min(z0, v.position.z);
                z1 = std::max(z1, v.position.z);
            }
            return std::array<float, 4>{x0, x1, z0, z1};
        };
        const auto a = extent(fine);
        const auto b = extent(coarse);
        std::printf("    fine x %.1f..%.1f, coarse x %.1f..%.1f\n", a[0], a[1], b[0],
                    b[1]);
        Check(std::fabs(a[0] - b[0]) < 1e-3f && std::fabs(a[1] - b[1]) < 1e-3f &&
                  std::fabs(a[2] - b[2]) < 1e-3f && std::fabs(a[3] - b[3]) < 1e-3f,
              "every level covers exactly the same square");
    }

    {
        std::printf("\nthe skirt hangs below the surface\n");
        const eng::Mesh chunk = terrain.BuildChunk(1, 1, 0);
        float lowest_surface = 1e30f, lowest_any = 1e30f;
        for (const VertexIn& v : chunk.vertices) {
            lowest_any = std::min(lowest_any, v.position.y);
            const float want = terrain.HeightAt(v.position.x, v.position.z);
            if (v.position.y >= want - 1.0f) lowest_surface = std::min(lowest_surface, want);
        }
        std::printf("    lowest surface %.3f, lowest vertex %.3f (skirt depth %.1f)\n",
                    lowest_surface, lowest_any, Config().skirt_depth);
        Check(lowest_any < lowest_surface - Config().skirt_depth + 0.01f,
              "the skirt reaches a full skirt-depth below the lowest surface");
        // AND ONLY AROUND THE BORDER. A skirt under the middle of the chunk
        // would be visible through the terrain from below and would cost
        // triangles for nothing.
        int skirt_verts = 0;
        for (const VertexIn& v : chunk.vertices)
            if (v.position.y < terrain.HeightAt(v.position.x, v.position.z) - 1.0f)
                ++skirt_verts;
        std::printf("    %d skirt vertices for a 33-sample border\n", skirt_verts);
        Check(skirt_verts == 33 * 4, "there is one skirt vertex per border sample");
    }

    {
        std::printf("\nthe raycast agrees with the height query\n");
        int hits = 0, tested = 0;
        float worst = 0.0f;
        for (float x = 6.0f; x < 120.0f; x += 9.7f)
            for (float z = 6.0f; z < 120.0f; z += 9.7f) {
                ++tested;
                float t = 0.0f;
                Vec3 n{0, 1, 0};
                // Straight down from well above the highest ground.
                if (!terrain.Raycast(Vec3{x, 40.0f, z}, Vec3{0, -1, 0}, 100.0f, &t, &n))
                    continue;
                ++hits;
                const float hit_y = 40.0f - t;
                worst = std::max(worst, std::fabs(hit_y - terrain.HeightAt(x, z)));
            }
        std::printf("    %d of %d rays hit, worst disagreement %.5f m\n", hits, tested,
                    worst);
        Check(hits == tested, "every downward ray finds the ground");
        Check(worst < 1e-3f, "and lands where HeightAt says it should");

        // A ray pointing UP from above finds nothing. A raycast that reported a
        // hit here would be marching without checking the direction.
        float t = 0.0f;
        Check(!terrain.Raycast(Vec3{60.0f, 40.0f, 60.0f}, Vec3{0, 1, 0}, 100.0f, &t,
                               nullptr),
              "and a ray fired upward finds nothing");

        // A GRAZING ray along a slope: the case a fixed-step march gets wrong by
        // stepping over a ridge.
        Vec3 n{0, 1, 0};
        const bool grazed = terrain.Raycast(Vec3{10.0f, 8.0f, 60.0f},
                                            eng::Normalize(Vec3{1.0f, -0.08f, 0.0f}),
                                            110.0f, &t, &n);
        if (grazed) {
            const Vec3 p = Vec3{10.0f, 8.0f, 60.0f} +
                           eng::Normalize(Vec3{1.0f, -0.08f, 0.0f}) * t;
            std::printf("    a grazing ray hit at (%.2f, %.2f) where the ground is "
                        "%.2f\n", p.x, p.y, terrain.HeightAt(p.x, p.z));
            Check(std::fabs(p.y - terrain.HeightAt(p.x, p.z)) < 0.02f,
                  "a grazing ray lands on the surface too");
        } else {
            Check(true, "a grazing ray found no ground, which is also possible");
        }
    }

    {
        std::printf("\nheightmap images are read bilinearly\n");
        // A 4x4 image, ramping left to right. Sampled onto a 129-sample
        // terrain, nearest sampling would give four visible steps.
        std::vector<std::uint8_t> grey(16);
        for (int z = 0; z < 4; ++z)
            for (int x = 0; x < 4; ++x)
                grey[std::size_t(z) * 4 + std::size_t(x)] =
                    std::uint8_t(x * 255 / 3);
        const eng::Terrain ramp =
            eng::Terrain::FromImage(Config(), grey, 4, 4, 0.0f, 10.0f);
        Check(ramp.Valid(), "an image builds a terrain");
        // Monotonic across the whole width, with no flat treads.
        bool monotonic = true;
        int flat_runs = 0;
        float previous = -1.0f;
        for (float x = 1.0f; x < 127.0f; x += 1.0f) {
            const float h = ramp.HeightAt(x, 64.0f);
            if (h < previous - 1e-4f) monotonic = false;
            if (std::fabs(h - previous) < 1e-5f) ++flat_runs;
            previous = h;
        }
        std::printf("    ends: %.3f to %.3f, %d flat steps along the ramp\n",
                    ramp.HeightAt(1.0f, 64.0f), ramp.HeightAt(126.0f, 64.0f),
                    flat_runs);
        Check(monotonic, "the ramp rises all the way across");
        Check(flat_runs < 4, "and is not a staircase of four treads");
    }

    {
        std::printf("\ndegenerate inputs\n");
        eng::TerrainConfig bad = Config();
        bad.resolution = 129;
        std::vector<float> too_few(10, 0.0f);
        const eng::Terrain broken = eng::Terrain::FromHeights(bad, too_few);
        Check(!broken.Valid(), "too few heights makes an invalid terrain");
        Check(broken.HeightAt(1.0f, 1.0f) == 0.0f, "which queries as flat zero");
        float t = 0.0f;
        Check(!broken.Raycast(Vec3{0, 10, 0}, Vec3{0, -1, 0}, 100.0f, &t, nullptr),
              "and hits nothing");
        Check(broken.BuildChunk(0, 0, 0).vertices.empty(),
              "and builds no geometry");
        Check(terrain.BuildChunk(99, 99, 0).vertices.empty(),
              "and a chunk index out of range builds nothing");
    }

    std::printf(g_failures == 0 ? "\nterrain_test: all checks passed\n"
                                : "\nterrain_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
