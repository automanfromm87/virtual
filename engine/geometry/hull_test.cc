// Convex hulls, against shapes whose volume and inertia are known on paper.
//
// A hull builder is easy to write in a way that looks right: it produces a
// closed mesh around roughly the correct points, and the failure mode -- a face
// buried inside the shape, or a vertex on the wrong side of one -- is invisible
// until something collides with it. So the assertions here are about
// PROPERTIES, not about vertex lists: every input point is inside, no vertex is
// outside any face, the volume matches the closed form, and the inertia matches
// the textbook tensor.

#include "engine/geometry/hull.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

bool Near(float a, float b, float tol = 1e-3f) { return std::fabs(a - b) <= tol; }

using eng::Vec3;
using eng::geom::ConvexHull;
using eng::geom::Hull;

// Every property a convex hull must have, checked at once. Returns false and
// names the first violation.
bool WellFormed(const Hull& h, std::span<const eng::Vec3> input,
                const char* label) {
    if (h.Empty()) {
        std::printf("    %s: hull is empty\n", label);
        return false;
    }
    // Euler's formula for a closed triangulated surface: V - E + F = 2, and
    // E = 3F/2 because every edge is shared by exactly two triangles. So
    // F = 2V - 4. A hull that is missing a face, or has one buried inside,
    // fails this without any geometry being examined.
    const std::size_t V = h.vertices.size(), F = h.FaceCount();
    if (F != 2 * V - 4) {
        std::printf("    %s: %zu vertices needs %zu faces, has %zu\n", label, V,
                    2 * V - 4, F);
        return false;
    }
    // No vertex outside any face. This is what "convex" means, and it is the
    // check that catches a horizon walk that kept an interior edge.
    for (std::size_t f = 0; f < F; ++f) {
        const Vec3 n = h.Normal(f);
        const Vec3& a = h.vertices[h.indices[f * 3]];
        for (const Vec3& v : h.vertices)
            if (Dot(n, v - a) > 1e-3f) {
                std::printf("    %s: vertex is %.4f outside face %zu\n", label,
                            Dot(n, v - a), f);
                return false;
            }
    }
    // Every INPUT point inside. A hull that drops a point is not the hull.
    for (const Vec3& p : input)
        if (!h.Contains(p, 1e-3f)) {
            std::printf("    %s: an input point is outside the hull\n", label);
            return false;
        }
    if (h.Volume() <= 0.0f) {
        std::printf("    %s: volume is %.6f (winding is inside out)\n", label,
                    h.Volume());
        return false;
    }
    return true;
}

}  // namespace

int main() {
    std::printf("convex hulls\n");

    // --- a cube, with interior points that must be discarded ----------------
    {
        std::vector<Vec3> pts;
        for (int i = 0; i < 8; ++i)
            pts.push_back(Vec3{(i & 1) ? 1.0f : -1.0f, (i & 2) ? 1.0f : -1.0f,
                               (i & 4) ? 1.0f : -1.0f});
        // Points strictly inside, and points on a face but not at a corner.
        // Both must vanish: a hull that keeps them still has the right shape
        // and has quietly doubled the cost of every support query.
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> in(-0.9f, 0.9f);
        for (int i = 0; i < 400; ++i) pts.push_back(Vec3{in(rng), in(rng), in(rng)});
        pts.push_back(Vec3{0.0f, 0.0f, 1.0f});   // on a face
        pts.push_back(Vec3{0.0f, 1.0f, 0.5f});   // on an edge-adjacent face
        pts.push_back(Vec3{1.0f, 1.0f, 0.0f});   // on an edge

        const Hull h = ConvexHull(pts);
        Check(WellFormed(h, pts, "cube"), "a cube's hull is well formed");
        std::printf("    %zu points in, %zu vertices out, %zu faces\n",
                    pts.size(), h.vertices.size(), h.FaceCount());
        Check(h.vertices.size() == 8, "only the 8 corners survive");
        Check(Near(h.Volume(), 8.0f), "volume is 2x2x2 = 8");
        const Vec3 c = h.Centroid();
        Check(Near(c.x, 0.0f) && Near(c.y, 0.0f) && Near(c.z, 0.0f),
              "centroid is at the origin");
        // A solid cube of side s and unit density: I = m s^2 / 6 on each axis,
        // with m = s^3 = 8, s = 2, so I = 8 * 4 / 6.
        const eng::Mat3 I = h.Inertia();
        const float want = 8.0f * 4.0f / 6.0f;
        std::printf("    inertia diagonal %.4f %.4f %.4f (textbook %.4f)\n",
                    I.col[0].x, I.col[1].y, I.col[2].z, want);
        Check(Near(I.col[0].x, want, 1e-2f) && Near(I.col[1].y, want, 1e-2f) &&
                  Near(I.col[2].z, want, 1e-2f),
              "inertia matches a solid cube");
        // A cube's products of inertia vanish about its own axes.
        Check(Near(I.col[0].y, 0.0f, 1e-3f) && Near(I.col[0].z, 0.0f, 1e-3f) &&
                  Near(I.col[1].z, 0.0f, 1e-3f),
              "the off-diagonal terms are zero for an axis-aligned cube");
    }

    // --- a long thin box, where the small moment is the one that matters ----
    {
        std::vector<Vec3> pts;
        for (int i = 0; i < 8; ++i)
            pts.push_back(Vec3{(i & 1) ? 3.0f : -3.0f, (i & 2) ? 0.25f : -0.25f,
                               (i & 4) ? 0.25f : -0.25f});
        const Hull h = ConvexHull(pts);
        Check(WellFormed(h, pts, "rod"), "a long thin box's hull is well formed");
        const float m = 6.0f * 0.5f * 0.5f;  // volume, unit density
        const eng::Mat3 I = h.Inertia();
        // I_xx = m(b^2 + c^2)/12 with b = c = 0.5; the other two use the long
        // side. The ratio between them is 145:1, so an approximation that gets
        // the big ones right can still be wildly wrong about the spin axis.
        const float ixx = m * (0.25f + 0.25f) / 12.0f;
        const float iyy = m * (36.0f + 0.25f) / 12.0f;
        std::printf("    rod inertia %.5f %.5f %.5f (textbook %.5f %.5f)\n",
                    I.col[0].x, I.col[1].y, I.col[2].z, ixx, iyy);
        Check(Near(I.col[0].x, ixx, 1e-3f), "the small moment is right");
        Check(Near(I.col[1].y, iyy, 1e-2f) && Near(I.col[2].z, iyy, 1e-2f),
              "the two large moments are right");
    }

    // --- a sphere's point cloud ---------------------------------------------
    //
    // The hull of points ON a sphere is an inscribed polyhedron, so its volume
    // is slightly LESS than the sphere's -- and converges upward as points are
    // added. Asserting it equals 4/3 pi r^3 would be wrong; asserting it
    // approaches it from below is the real property.
    {
        std::vector<Vec3> pts;
        std::mt19937 rng(11);
        std::normal_distribution<float> g(0.0f, 1.0f);
        for (int i = 0; i < 500; ++i) {
            Vec3 d{g(rng), g(rng), g(rng)};
            const float len = Length(d);
            if (len < 1e-3f) continue;
            pts.push_back(d * (2.0f / len));
        }
        const Hull h = ConvexHull(pts);
        Check(WellFormed(h, pts, "sphere"), "a sphere's hull is well formed");
        const float exact = 4.0f / 3.0f * 3.14159265f * 8.0f;
        std::printf("    hull volume %.3f, sphere %.3f (%.1f%%)\n", h.Volume(),
                    exact, 100.0f * h.Volume() / exact);
        Check(h.Volume() < exact, "an inscribed hull is smaller than the sphere");
        Check(h.Volume() > exact * 0.97f, "and close to it with 500 points");
        // Every point was on the sphere, so every point is a hull vertex.
        Check(h.vertices.size() == pts.size(), "no point on the sphere is interior");
    }

    // --- degenerate inputs ---------------------------------------------------
    //
    // Each returns an EMPTY hull rather than something flat. A flat hull has no
    // inside, no volume and no inertia, and every algorithm downstream would
    // divide by zero somewhere much further from the cause.
    {
        Check(ConvexHull({}).Empty(), "no points");
        const Vec3 three[3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
        Check(ConvexHull(three).Empty(), "three points cannot bound a volume");

        std::vector<Vec3> collinear;
        for (int i = 0; i < 20; ++i)
            collinear.push_back(Vec3{float(i), float(i) * 2.0f, float(i) * -1.0f});
        Check(ConvexHull(collinear).Empty(), "collinear points");

        std::vector<Vec3> coplanar;
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j)
                coplanar.push_back(Vec3{float(i), 3.0f, float(j)});
        Check(ConvexHull(coplanar).Empty(), "coplanar points");

        std::vector<Vec3> same(30, Vec3{2.0f, 2.0f, 2.0f});
        Check(ConvexHull(same).Empty(), "the same point many times");

        // Nearly coplanar, by less than the tolerance. This has to be refused
        // too: a hull one part in a million thick is not degenerate to the
        // builder but is to everything that uses it.
        std::vector<Vec3> nearly;
        for (int i = 0; i < 5; ++i)
            for (int j = 0; j < 5; ++j)
                nearly.push_back(
                    Vec3{float(i), float((i + j) % 2) * 1e-7f, float(j)});
        Check(ConvexHull(nearly).Empty(), "points coplanar to within the tolerance");
    }

    // --- a random cloud, which is where a horizon walk actually fails --------
    //
    // The shapes above are symmetric, and a builder can get them right by
    // accident. A few hundred random clouds cannot be got right by accident:
    // any leak in the horizon logic shows up as a face buried inside the hull
    // within a handful of seeds.
    {
        int bad = 0;
        std::size_t total_v = 0, total_f = 0;
        for (int seed = 0; seed < 200; ++seed) {
            std::mt19937 rng(std::uint32_t(seed) * 2654435761u);
            std::uniform_real_distribution<float> u(-1.0f, 1.0f);
            std::vector<Vec3> pts;
            const int n = 8 + int(seed % 40);
            for (int i = 0; i < n; ++i) pts.push_back(Vec3{u(rng), u(rng), u(rng)});
            const Hull h = ConvexHull(pts);
            if (h.Empty()) { ++bad; continue; }
            char label[32];
            std::snprintf(label, sizeof(label), "seed %d", seed);
            if (!WellFormed(h, pts, label)) { ++bad; continue; }
            total_v += h.vertices.size();
            total_f += h.FaceCount();
        }
        std::printf("    200 random clouds: %zu vertices and %zu faces total\n",
                    total_v, total_f);
        Check(bad == 0, "200 random point clouds all produce valid hulls");
    }

    // A cloud with many DUPLICATE points, which is what a real mesh looks like
    // once its vertices are split for uvs and normals.
    {
        std::vector<Vec3> pts;
        std::mt19937 rng(3);
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (int i = 0; i < 60; ++i) {
            const Vec3 p{u(rng), u(rng), u(rng)};
            pts.push_back(p);
            pts.push_back(p);
            pts.push_back(p);
        }
        const Hull h = ConvexHull(pts);
        Check(WellFormed(h, pts, "duplicated"),
              "a cloud with every point tripled still builds");
    }

    if (g_failures == 0) std::printf("hull_test: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
