// Meshlets, checked as arithmetic.
//
// Two of the three properties here are invariants that can be verified exactly:
// every triangle appears once, and the two-level indices reconstruct the
// original mesh. The third -- the bounding sphere and the normal cone -- is
// checked by the direction its errors go: a sphere or cone slightly too LARGE
// costs a meshlet that survives a cull it could have failed, and one slightly
// too SMALL is a hole in the model. So the tests are one-sided.
#include "engine/geometry/meshlet.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <array>
#include <set>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

}  // namespace

int main() {
    using namespace eng;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    {
        std::printf("the split loses nothing and duplicates nothing\n");
        const Mesh sphere = MakeUVSphere(1.0f, 40, 56, Vec4{1, 1, 1, 1},
                                         Vec4{1, 1, 1, 1});
        const MeshletBuild b = BuildMeshlets(sphere);
        std::printf("    %zu triangles -> %zu meshlets, %zu vertex refs, "
                    "%zu triangle bytes\n",
                    sphere.indices.size() / 3, b.meshlets.size(),
                    b.vertices.size(), b.triangles.size());
        Check(!b.Empty(), "a sphere splits into meshlets");

        // EVERY LIMIT respected. These are hardware limits, so exceeding one is
        // not a quality problem -- the mesh stage refuses to launch.
        bool within = true;
        std::size_t total_tris = 0;
        for (const Meshlet& m : b.meshlets) {
            if (m.vertex_count > std::uint32_t(kMeshletMaxVertices)) within = false;
            if (m.triangle_count > std::uint32_t(kMeshletMaxTriangles)) within = false;
            total_tris += m.triangle_count;
        }
        Check(within, "and every one is inside the 64-vertex, 124-triangle limit");
        Check(total_tris == sphere.indices.size() / 3,
              "with every triangle accounted for exactly once");

        // THE RECONSTRUCTION. Walking the two-level indices has to produce the
        // original triangle set -- as a SET, because the order changes and the
        // order does not matter. This is the check that catches a local index
        // written relative to the wrong meshlet, which is the mistake the whole
        // two-level scheme invites.
        std::multiset<std::array<std::uint32_t, 3>> want, got;
        for (std::size_t t = 0; t + 2 < sphere.indices.size(); t += 3) {
            std::array<std::uint32_t, 3> tri{sphere.indices[t], sphere.indices[t + 1],
                                             sphere.indices[t + 2]};
            // Rotated to a canonical starting vertex, which preserves winding
            // while making the comparison independent of which corner the
            // builder happened to emit first.
            const auto rot = std::size_t(std::min_element(tri.begin(), tri.end()) -
                                         tri.begin());
            want.insert({tri[rot], tri[(rot + 1) % 3], tri[(rot + 2) % 3]});
        }
        for (const Meshlet& m : b.meshlets)
            for (std::uint32_t t = 0; t < m.triangle_count; ++t) {
                std::array<std::uint32_t, 3> tri{};
                for (int k = 0; k < 3; ++k) {
                    const std::uint8_t local =
                        b.triangles[(m.triangle_offset + t) * 3 + std::size_t(k)];
                    tri[std::size_t(k)] = b.vertices[m.vertex_offset + local];
                }
                const auto rot = std::size_t(std::min_element(tri.begin(), tri.end()) -
                                             tri.begin());
                got.insert({tri[rot], tri[(rot + 1) % 3], tri[(rot + 2) % 3]});
            }
        std::printf("    reconstructed %zu triangles against the original %zu\n",
                    got.size(), want.size());
        Check(got == want,
              "and the two-level indices reconstruct the mesh exactly");

        // THE COMPRESSION, which is the point of the two levels. A triangle
        // costs three bytes instead of twelve, plus one 32-bit reference per
        // vertex per meshlet.
        const std::size_t plain = sphere.indices.size() * 4;
        const std::size_t meshlet_bytes = b.triangles.size() + b.vertices.size() * 4;
        std::printf("    %zu bytes of 32-bit indices against %zu as meshlets "
                    "(%.2fx)\n",
                    plain, meshlet_bytes, double(plain) / double(meshlet_bytes));
        Check(meshlet_bytes < plain,
              "and the two-level indices are smaller than 32-bit ones");
    }

    {
        std::printf("\nbounding spheres contain their meshlets\n");
        const Mesh m = MakeUVSphere(2.0f, 32, 48, Vec4{1, 1, 1, 1}, Vec4{1, 1, 1, 1});
        const MeshletBuild b = BuildMeshlets(m);
        double worst_excess = 0.0, worst_slack = 0.0;
        for (const Meshlet& ml : b.meshlets) {
            double furthest = 0.0;
            for (std::uint32_t i = 0; i < ml.vertex_count; ++i) {
                const Vec4& p = m.vertices[b.vertices[ml.vertex_offset + i]].position;
                furthest = std::max(furthest,
                                    double(Length(Vec3{p.x, p.y, p.z} - ml.center)));
            }
            worst_excess = std::max(worst_excess, furthest - double(ml.radius));
            worst_slack = std::max(worst_slack, double(ml.radius) - furthest);
        }
        std::printf("    worst vertex OUTSIDE its sphere: %.3e; worst unused "
                    "radius: %.4f\n",
                    worst_excess, worst_slack);
        // NOTHING outside, at all. A vertex outside its meshlet's sphere is a
        // meshlet that can be culled while part of it is on screen, and the
        // symptom is a triangle-shaped hole that flickers.
        Check(worst_excess <= 1e-5, "no vertex lies outside its meshlet's sphere");
        // And the sphere is not absurdly loose, or the culling buys nothing.
        // Ritter's method is within a few percent of minimal; the meshlets here
        // are about 0.3 units across, so a slack of more than a tenth of a unit
        // would mean the sphere had stopped tracking the geometry.
        Check(worst_slack < 0.15, "and no sphere is much larger than it needs to be");
    }

    {
        std::printf("\nthe normal cone contains every triangle's normal\n");
        const Mesh m = MakeUVSphere(1.0f, 24, 32, Vec4{1, 1, 1, 1}, Vec4{1, 1, 1, 1});
        const MeshletBuild b = BuildMeshlets(m);
        int with_cone = 0, violations = 0;
        for (const Meshlet& ml : b.meshlets) {
            if (ml.cone_cutoff >= 1.0f) continue;  // no cone fitted
            ++with_cone;
            // cone_cutoff is -sin(half-angle), so the half-angle's cosine is
            // sqrt(1 - cutoff^2). Every face normal has to be inside it.
            const float cos_half = std::sqrt(
                std::max(1.0f - ml.cone_cutoff * ml.cone_cutoff, 0.0f));
            for (std::uint32_t t = 0; t < ml.triangle_count; ++t) {
                Vec3 p[3];
                for (int k = 0; k < 3; ++k) {
                    const std::uint8_t local =
                        b.triangles[(ml.triangle_offset + t) * 3 + std::size_t(k)];
                    const Vec4& v =
                        m.vertices[b.vertices[ml.vertex_offset + local]].position;
                    p[k] = Vec3{v.x, v.y, v.z};
                }
                const Vec3 face = Cross(p[1] - p[0], p[2] - p[0]);
                if (Length(face) < 1e-9f) continue;
                const Vec3 n = Normalize(face);
                // A hair of tolerance: cos_half is derived through a square
                // root from a value that was itself a minimum over floats.
                if (Dot(ml.cone_axis, n) < cos_half - 1e-3f) ++violations;
            }
        }
        std::printf("    %d of %zu meshlets have a usable cone; %d normals fall "
                    "outside theirs\n",
                    with_cone, b.meshlets.size(), violations);
        Check(with_cone > 0, "a sphere's meshlets do get cones");
        // A normal outside its cone is a meshlet culled while some of its
        // triangles face the camera -- again a flickering hole.
        Check(violations == 0, "and every normal is inside its meshlet's cone");
    }

    {
        std::printf("\ndegenerate input\n");
        Check(BuildMeshlets(Mesh{}).Empty(), "an empty mesh gives no meshlets");
        Mesh no_tris;
        no_tris.vertices.resize(3);
        Check(BuildMeshlets(no_tris).Empty(),
              "vertices with no indices give no meshlets");

        // A mesh whose triangles face every direction at once: the average
        // normal cancels and there is no cone to fit. The builder has to say
        // "never cull" rather than fitting an arbitrary one -- a cone from
        // cancelling normals would reject the meshlet from directions it is
        // plainly visible in.
        Mesh cube = MakeCube(1.0f, Vec4{1, 1, 1, 1}, Vec4{1, 1, 1, 1});
        const MeshletBuild cb = BuildMeshlets(cube);
        Check(cb.meshlets.size() == 1, "a 12-triangle cube is one meshlet");
        std::printf("    the cube's cone cutoff is %.4f (1.0 means never cull)\n",
                    double(cb.meshlets[0].cone_cutoff));
        Check(cb.meshlets[0].cone_cutoff >= 1.0f,
              "and its normals cancel, so it refuses to fit a cone");
        // Its sphere still has to be right, cone or no cone.
        double furthest = 0.0;
        for (std::uint32_t i = 0; i < cb.meshlets[0].vertex_count; ++i) {
            const Vec4& p = cube.vertices[cb.vertices[i]].position;
            furthest = std::max(furthest, double(Length(Vec3{p.x, p.y, p.z} -
                                                       cb.meshlets[0].center)));
        }
        Check(furthest <= double(cb.meshlets[0].radius) + 1e-5,
              "while its bounding sphere still contains it");
    }

    std::printf(g_failures == 0 ? "\nmeshlet_test: all checks passed\n"
                                : "\nmeshlet_test: %d FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
