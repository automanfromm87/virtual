// No test framework — from scratch means from scratch.
//
// The point of this file: winding and orientation are exactly what a
// pixel-coverage check CANNOT see (permuting the framebuffer does not change a
// colour histogram). On the CPU they are cheap and exact, so test them here and
// let the GPU harness worry about whether anything got drawn at all.
#include "engine/geometry/mesh.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <tuple>
#include <utility>

namespace {

int g_failures = 0;

void Fail(const char* what, const char* which, int line) {
    std::fprintf(stderr, "mesh_test.cc:%d  [%s] %s\n", line, which, what);
    ++g_failures;
}

#define CHECK(cond) \
    do { if (!(cond)) Fail(#cond, which, __LINE__); } while (0)

// Every closed mesh in this engine must satisfy these, whatever its shape.
void CheckClosedMesh(const eng::Mesh& m, const char* which) {
    using namespace eng;

    CHECK(!m.vertices.empty());
    CHECK(!m.indices.empty());
    CHECK(m.indices.size() % 3 == 0);
    CHECK(m.bounds.radius > 0.0f);

    for (std::uint32_t i : m.indices) CHECK(i < m.vertices.size());

    for (const VertexIn& v : m.vertices) {
        const Vec3 n{v.normal.x, v.normal.y, v.normal.z};
        CHECK(std::fabs(Length(n) - 1.0f) < 1e-4f);  // unit normals

        // Every vertex fits inside the declared bounding sphere, or frustum
        // culling will pop geometry off the edge of the screen.
        const Vec3 p{v.position.x, v.position.y, v.position.z};
        CHECK(Length(p - m.bounds.center) <= m.bounds.radius + 1e-4f);

        // uv inside the unit square. Out-of-range coordinates are invisible
        // with clamp addressing and catastrophic with repeat.
        CHECK(v.uv.x >= -1e-6f && v.uv.x <= 1.0f + 1e-6f);
        CHECK(v.uv.y >= -1e-6f && v.uv.y <= 1.0f + 1e-6f);
    }

    // THE important one: every triangle must wind counter-clockwise as seen
    // from outside, or back-face culling turns the object inside out. The
    // right-hand rule normal of a CCW triangle points away from the centre.
    int backwards = 0, degenerate = 0;
    for (std::size_t t = 0; t + 2 < m.indices.size(); t += 3) {
        const VertexIn& a = m.vertices[m.indices[t + 0]];
        const VertexIn& b = m.vertices[m.indices[t + 1]];
        const VertexIn& c = m.vertices[m.indices[t + 2]];
        const Vec3 pa{a.position.x, a.position.y, a.position.z};
        const Vec3 pb{b.position.x, b.position.y, b.position.z};
        const Vec3 pc{c.position.x, c.position.y, c.position.z};

        const Vec3 face = Cross(pb - pa, pc - pa);
        if (Length(face) < 1e-9f) { ++degenerate; continue; }
        // Outward direction at this triangle = its centroid, since both shapes
        // are centred on the origin.
        if (Dot(face, pa + pb + pc) <= 0.0f) ++backwards;
    }
    CHECK(backwards == 0);
    CHECK(degenerate == 0);  // pole quads should be skipped, not emitted

    // CLOSURE. A hole in the mesh passes every check above — normals, winding
    // and bounds are all per-triangle properties. On a closed orientable
    // surface each directed edge appears exactly once and its reverse exactly
    // once; a hole leaves an edge unmatched.
    //
    // Keyed on POSITION, not index: the UV sphere duplicates its seam and pole
    // vertices, so index-based edges would look unmatched even when the surface
    // is perfectly closed.
    {
        using Key = std::tuple<long long, long long, long long>;
        auto key = [](const Vec4& p) {
            auto q = [](float v) { return (long long)std::llround(v * 8192.0); };
            return Key{q(p.x), q(p.y), q(p.z)};
        };
        std::map<std::pair<Key, Key>, int> directed;
        for (std::size_t t = 0; t + 2 < m.indices.size(); t += 3) {
            const Key k[3] = {key(m.vertices[m.indices[t + 0]].position),
                              key(m.vertices[m.indices[t + 1]].position),
                              key(m.vertices[m.indices[t + 2]].position)};
            for (int e = 0; e < 3; ++e)
                ++directed[{k[e], k[(e + 1) % 3]}];
        }
        int unmatched = 0, duplicated = 0;
        for (const auto& [edge, count] : directed) {
            if (count != 1) ++duplicated;  // same edge, same direction, twice
            auto rev = directed.find({edge.second, edge.first});
            if (rev == directed.end()) ++unmatched;
        }
        CHECK(unmatched == 0);   // no holes
        CHECK(duplicated == 0);  // consistently oriented, no folded-over faces
    }

    // The face normal and the interpolated vertex normals must agree, or
    // lighting fights the silhouette.
    int flipped = 0;
    for (std::size_t t = 0; t + 2 < m.indices.size(); t += 3) {
        const VertexIn& a = m.vertices[m.indices[t + 0]];
        const VertexIn& b = m.vertices[m.indices[t + 1]];
        const VertexIn& c = m.vertices[m.indices[t + 2]];
        const Vec3 pa{a.position.x, a.position.y, a.position.z};
        const Vec3 pb{b.position.x, b.position.y, b.position.z};
        const Vec3 pc{c.position.x, c.position.y, c.position.z};
        const Vec3 face = Cross(pb - pa, pc - pa);
        const Vec3 avg{(a.normal.x + b.normal.x + c.normal.x),
                       (a.normal.y + b.normal.y + c.normal.y),
                       (a.normal.z + b.normal.z + c.normal.z)};
        if (Length(face) > 1e-9f && Dot(face, avg) <= 0.0f) ++flipped;
    }
    CHECK(flipped == 0);
}

}  // namespace

int main() {
    using namespace eng;

    {
        const char* which = "sphere";
        constexpr float kR = 2.0f;
        constexpr int kStacks = 16, kSlices = 24;
        const Mesh m =
            MakeUVSphere(kR, kStacks, kSlices, Vec4{1, 1, 1, 1}, Vec4{0, 0, 0, 1});
        CheckClosedMesh(m, which);

        CHECK(m.vertices.size() == std::size_t(kStacks + 1) * (kSlices + 1));
        // Poles contribute one degenerate triangle per quad, so two rows lose one.
        CHECK(m.indices.size() ==
              std::size_t(kStacks * kSlices * 2 - kSlices * 2) * 3);
        CHECK(std::fabs(m.bounds.radius - kR) < 1e-5f);

        // On a sphere the normal IS the normalised position. Checking only the
        // radius, as this used to, would pass for ANY unit normal — including
        // one pointing the wrong way.
        for (const VertexIn& v : m.vertices) {
            const Vec3 p{v.position.x, v.position.y, v.position.z};
            const Vec3 n{v.normal.x, v.normal.y, v.normal.z};
            CHECK(std::fabs(Length(p) - kR) < 1e-4f);
            CHECK(Dot(Normalize(p), n) > 1.0f - 1e-4f);
        }
    }

    {
        const char* which = "cube";
        constexpr float kSize = 2.0f;
        const Mesh m = MakeCube(kSize, Vec4{1, 1, 1, 1}, Vec4{0, 0, 0, 1});
        CheckClosedMesh(m, which);

        // 24, not 8: a hard edge needs a different normal per face.
        CHECK(m.vertices.size() == 24);
        CHECK(m.indices.size() == 36);
        // Bounding sphere is the half space-diagonal, not the half side.
        CHECK(std::fabs(m.bounds.radius - kSize * 0.5f * std::sqrt(3.0f)) < 1e-5f);

        // Every face covers the whole 0..1 uv square: 4 corners x 6 faces means
        // each of the four corner values appears exactly six times.
        int corners[4] = {0, 0, 0, 0};
        for (const VertexIn& v : m.vertices) {
            const int cu = v.uv.x > 0.5f ? 1 : 0;
            const int cv = v.uv.y > 0.5f ? 1 : 0;
            ++corners[cv * 2 + cu];
        }
        for (int i = 0; i < 4; ++i) CHECK(corners[i] == 6);

        // Exactly six distinct normals, one per face.
        int axis_hits = 0;
        for (const VertexIn& v : m.vertices) {
            const float ax = std::fabs(v.normal.x) + std::fabs(v.normal.y) +
                             std::fabs(v.normal.z);
            if (std::fabs(ax - 1.0f) < 1e-5f) ++axis_hits;  // axis-aligned
        }
        CHECK(axis_hits == 24);
    }

    {
        const char* which = "box";
        const Mesh m = MakeBox(Vec3{2.0f, 0.25f, 1.0f}, Vec4{1, 1, 1, 1});
        CheckClosedMesh(m, which);
        CHECK(m.vertices.size() == 24);
        // Bounding radius is the half-diagonal, not the largest half-extent.
        CHECK(std::fabs(m.bounds.radius - Length(Vec3{2.0f, 0.25f, 1.0f})) < 1e-5f);
        // Extents come out exactly as asked, on every axis independently.
        float mx = 0, my = 0, mz = 0;
        for (const VertexIn& v : m.vertices) {
            mx = std::fmax(mx, std::fabs(v.position.x));
            my = std::fmax(my, std::fabs(v.position.y));
            mz = std::fmax(mz, std::fabs(v.position.z));
        }
        CHECK(std::fabs(mx - 2.0f) < 1e-5f);
        CHECK(std::fabs(my - 0.25f) < 1e-5f);
        CHECK(std::fabs(mz - 1.0f) < 1e-5f);
    }

    {
        const char* which = "degenerate inputs";
        CHECK(MakeUVSphere(1.0f, 1, 8, Vec4{}, Vec4{}).vertices.empty());
        CHECK(MakeUVSphere(1.0f, 8, 2, Vec4{}, Vec4{}).vertices.empty());
        CHECK(MakeUVSphere(0.0f, 8, 8, Vec4{}, Vec4{}).vertices.empty());
        CHECK(MakeCube(0.0f, Vec4{}, Vec4{}).vertices.empty());
        CHECK(MakeCube(-1.0f, Vec4{}, Vec4{}).vertices.empty());
        CHECK(MakeBox(Vec3{0, 1, 1}, Vec4{}).vertices.empty());
        CHECK(MakeBox(Vec3{1, -1, 1}, Vec4{}).vertices.empty());
    }

    {
        // TANGENTS. Three properties, and each catches a different mistake:
        // unit length catches a missing normalise, perpendicularity catches a
        // missing Gram-Schmidt, and DIRECTION catches the one that matters --
        // a tangent that is a valid frame but points the wrong way makes a
        // normal map's x axis run backwards, which is invisible on a symmetric
        // pattern and obvious on lettering.
        const char* which = "tangents";
        const Mesh cube = MakeCube(2.0f, Vec4{1, 1, 1, 1}, Vec4{1, 1, 1, 1});
        CHECK(!cube.vertices.empty());

        float worst_len = 0.0f, worst_perp = 0.0f;
        for (const VertexIn& v : cube.vertices) {
            const Vec3 t{v.tangent.x, v.tangent.y, v.tangent.z};
            const Vec3 n{v.normal.x, v.normal.y, v.normal.z};
            worst_len = std::max(worst_len, std::fabs(Length(t) - 1.0f));
            worst_perp = std::max(worst_perp, std::fabs(Dot(Normalize(n), t)));
            CHECK(v.tangent.w == 1.0f || v.tangent.w == -1.0f);
        }
        std::printf("  cube: worst |T|-1 = %.2e, worst |dot(N,T)| = %.2e\n",
                    double(worst_len), double(worst_perp));
        CHECK(worst_len < 1e-5f);
        CHECK(worst_perp < 1e-5f);

        // DIRECTION, on a face whose axes are known. MakeCube's +Z face spans
        // u along +X and v along -Y (uv originates top-left), so the tangent
        // there must be +X and cross(N,T)*w must come out -Y.
        int checked = 0;
        for (const VertexIn& v : cube.vertices) {
            if (v.normal.z < 0.99f) continue;  // +Z face only
            const Vec3 t{v.tangent.x, v.tangent.y, v.tangent.z};
            CHECK(t.x > 0.99f);
            const Vec3 n{v.normal.x, v.normal.y, v.normal.z};
            const Vec3 b = Cross(n, t) * v.tangent.w;
            CHECK(b.y < -0.99f);
            ++checked;
        }
        CHECK(checked == 4);

        // HANDEDNESS flips with a mirrored uv shell. Same geometry, u running
        // backwards: the tangent reverses and so must the bitangent's sign, or
        // every dent on a mirrored half of a model becomes a bump.
        Mesh mirrored = cube;
        for (VertexIn& v : mirrored.vertices) v.uv.x = 1.0f - v.uv.x;
        GenerateTangents(mirrored);
        int flipped = 0, reversed = 0;
        for (std::size_t i = 0; i < cube.vertices.size(); ++i) {
            if (mirrored.vertices[i].tangent.w != cube.vertices[i].tangent.w)
                ++flipped;
            if (mirrored.vertices[i].tangent.x * cube.vertices[i].tangent.x +
                    mirrored.vertices[i].tangent.y * cube.vertices[i].tangent.y +
                    mirrored.vertices[i].tangent.z * cube.vertices[i].tangent.z <
                -0.99f)
                ++reversed;
        }
        std::printf("  mirrored uv: %d of %zu tangents reversed, %d signs flipped\n",
                    reversed, cube.vertices.size(), flipped);
        CHECK(reversed == int(cube.vertices.size()));
        CHECK(flipped == int(cube.vertices.size()));

        // A mesh with NO uv variation at all. Every triangle is degenerate in
        // uv space, so there is no correct tangent -- but there is a required
        // one: finite, unit length and perpendicular to the normal. A zero
        // tangent makes the shader's TBN singular and every fragment NaN,
        // which renders as a black surface rather than as an error.
        Mesh flat = MakeCube(1.0f, Vec4{1, 1, 1, 1}, Vec4{1, 1, 1, 1});
        for (VertexIn& v : flat.vertices) v.uv = Vec4{0.5f, 0.5f, 0.0f, 0.0f};
        GenerateTangents(flat);
        bool all_finite = true;
        for (const VertexIn& v : flat.vertices) {
            const Vec3 t{v.tangent.x, v.tangent.y, v.tangent.z};
            const Vec3 n{v.normal.x, v.normal.y, v.normal.z};
            if (!std::isfinite(t.x) || !std::isfinite(t.y) || !std::isfinite(t.z))
                all_finite = false;
            if (std::fabs(Length(t) - 1.0f) > 1e-5f) all_finite = false;
            if (std::fabs(Dot(Normalize(n), t)) > 1e-5f) all_finite = false;
        }
        CHECK(all_finite);

        // A SPHERE, where the accumulated tangent is genuinely not
        // perpendicular to the normal before orthogonalisation -- the normals
        // vary smoothly across each triangle, so the per-triangle tangent is
        // an average that leans. On a cube every face is flat and the raw
        // accumulation is already perpendicular, so a cube cannot tell whether
        // the Gram-Schmidt step ran.
        const Mesh sphere = MakeUVSphere(1.0f, 24, 32, Vec4{1, 1, 1, 1},
                                         Vec4{1, 1, 1, 1});
        float sphere_perp = 0.0f, sphere_len = 0.0f;
        for (const VertexIn& v : sphere.vertices) {
            const Vec3 t{v.tangent.x, v.tangent.y, v.tangent.z};
            const Vec3 nv{v.normal.x, v.normal.y, v.normal.z};
            sphere_perp = std::max(sphere_perp, std::fabs(Dot(Normalize(nv), t)));
            sphere_len = std::max(sphere_len, std::fabs(Length(t) - 1.0f));
        }
        std::printf("  sphere: worst |dot(N,T)| = %.2e, worst |T|-1 = %.2e\n",
                    double(sphere_perp), double(sphere_len));
        CHECK(sphere_perp < 1e-5f);
        CHECK(sphere_len < 1e-5f);

        // Every generator fills them in, not just the one that was remembered.
        CHECK(MakeUVSphere(1.0f, 8, 12, Vec4{}, Vec4{}).vertices[10].tangent.w != 0.0f);
        CHECK(MakeBox(Vec3{1, 2, 3}, Vec4{}).vertices[3].tangent.w != 0.0f);
    }

    if (g_failures == 0) std::printf("mesh_test: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
