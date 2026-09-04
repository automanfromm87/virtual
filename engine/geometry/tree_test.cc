// Procedural trees, checked for the things a generator has to get right.
//
// "Does it look like a tree" is not testable and is not what breaks. What
// breaks is: two seeds producing the same tree, the recursion exploding, the
// bounds not containing the geometry, and normals pointing inward so half the
// canopy renders black. All four are arithmetic.
#include "engine/geometry/tree.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

// How far the mesh reaches from the origin along each axis.
void Extent(const eng::Mesh& m, eng::Vec3* lo, eng::Vec3* hi) {
    *lo = eng::Vec3{1e30f, 1e30f, 1e30f};
    *hi = eng::Vec3{-1e30f, -1e30f, -1e30f};
    for (const VertexIn& v : m.vertices) {
        lo->x = std::min(lo->x, v.position.x);
        lo->y = std::min(lo->y, v.position.y);
        lo->z = std::min(lo->z, v.position.z);
        hi->x = std::max(hi->x, v.position.x);
        hi->y = std::max(hi->y, v.position.y);
        hi->z = std::max(hi->z, v.position.z);
    }
}

}  // namespace

int main() {
    using namespace eng;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    {
        std::printf("a tree is a tree\n");
        TreeParams p;
        const Tree t = MakeTree(p);
        std::printf("    trunk %zu verts / %zu tris, foliage %zu / %zu\n",
                    t.trunk.vertices.size(), t.trunk.indices.size() / 3,
                    t.foliage.vertices.size(), t.foliage.indices.size() / 3);
        Check(!t.trunk.vertices.empty() && !t.foliage.vertices.empty(),
              "both halves have geometry");
        Check(t.trunk.indices.size() % 3 == 0 && t.foliage.indices.size() % 3 == 0,
              "and both are whole triangles");

        bool in_range = true;
        for (std::uint32_t i : t.trunk.indices)
            if (i >= t.trunk.vertices.size()) in_range = false;
        for (std::uint32_t i : t.foliage.indices)
            if (i >= t.foliage.vertices.size()) in_range = false;
        Check(in_range, "with every index inside its own mesh");

        // IT GROWS UPWARD, and the canopy is above the trunk's base. A sign
        // error in the branching gives a tree that grows into the ground, which
        // is perfectly well-formed geometry.
        Vec3 tlo, thi, flo, fhi;
        Extent(t.trunk, &tlo, &thi);
        Extent(t.foliage, &flo, &fhi);
        std::printf("    trunk y %.2f..%.2f, foliage y %.2f..%.2f\n", double(tlo.y),
                    double(thi.y), double(flo.y), double(fhi.y));
        Check(thi.y > p.height * 0.8f, "the trunk reaches most of its height");
        Check(tlo.y > -p.height * 0.25f, "and does not grow into the ground");
        Check((flo.y + fhi.y) * 0.5f > thi.y * 0.4f,
              "and the canopy sits in the upper half of it");

        // WIDER THAN IT IS TALL is wrong for a tree and is what happens when
        // the branches spread without ever turning back up.
        const float width = std::max(fhi.x - flo.x, fhi.z - flo.z);
        std::printf("    canopy is %.2f wide and the tree is %.2f tall\n",
                    double(width), double(fhi.y));
        Check(width < fhi.y * 2.0f, "and the canopy is not an umbrella");
    }

    {
        std::printf("\nnormals point outward\n");
        // A blob lit from outside must be BRIGHT on the side facing the light.
        // Inverted normals produce geometry that is perfectly correct and
        // renders black, which reads as a shadow rather than as a bug.
        const Tree t = MakeTree(TreeParams{});
        // PER TRIANGLE, against its own winding -- not against the direction
        // from the mesh's centre, which is what this asked first and which is
        // meaningless here: the canopy is a scattered cloud of blobs, so a blob
        // on the far side has half its normals legitimately pointing back
        // toward the middle. Measured 6687 outward against 4977 inward, which
        // is what a cloud of convex blobs looks like and says nothing.
        //
        // The face normal from the winding and the interpolated vertex normals
        // have to agree. That catches an inverted normal AND a reversed
        // winding, it is local so the scatter does not matter, and both
        // failures render as a surface that is black from the front.
        const auto flipped = [](const Mesh& m) {
            // Degeneracy has to be judged AGAINST THE MESH'S SCALE. Area goes
            // as length squared, so an absolute epsilon is not a threshold, it
            // is an accident -- 1e-12 here passed a row of pole slivers whose
            // area was 3e-8 and whose orientation was rounding noise, and
            // reported them as seven real folded triangles.
            const float scale = m.bounds.radius * m.bounds.radius * 1e-6f;
            int bad = 0;
            for (std::size_t i = 0; i + 2 < m.indices.size(); i += 3) {
                const VertexIn& a = m.vertices[m.indices[i]];
                const VertexIn& b = m.vertices[m.indices[i + 1]];
                const VertexIn& c = m.vertices[m.indices[i + 2]];
                const Vec3 pa{a.position.x, a.position.y, a.position.z};
                const Vec3 pb{b.position.x, b.position.y, b.position.z};
                const Vec3 pc{c.position.x, c.position.y, c.position.z};
                const Vec3 face = Cross(pb - pa, pc - pa);
                if (Length(face) < scale) continue;  // degenerate: no opinion
                const Vec3 avg{a.normal.x + b.normal.x + c.normal.x,
                               a.normal.y + b.normal.y + c.normal.y,
                               a.normal.z + b.normal.z + c.normal.z};
                if (Dot(face, avg) <= 0.0f) ++bad;
            }
            return bad;
        };
        const int bad_trunk = flipped(t.trunk), bad_leaf = flipped(t.foliage);
        std::printf("    %d trunk and %d canopy triangles wound against their "
                    "own normals\n",
                    bad_trunk, bad_leaf);
        Check(bad_trunk == 0 && bad_leaf == 0,
              "every triangle's winding agrees with its vertex normals");

        // TANGENTS, which a normal-mapped bark needs and which are easy to
        // ship missing: the shader falls back to the geometric normal when the
        // frame is degenerate, so a tree with no tangents renders as a tree
        // whose normal map did not load.
        int no_tangent = 0, not_perpendicular = 0;
        for (const VertexIn& v : t.trunk.vertices) {
            const Vec3 tg{v.tangent.x, v.tangent.y, v.tangent.z};
            const Vec3 n{v.normal.x, v.normal.y, v.normal.z};
            if (Length(tg) < 0.5f) { ++no_tangent; continue; }
            if (std::fabs(Dot(Normalize(tg), n)) > 0.02f) ++not_perpendicular;
        }
        std::printf("    %d trunk vertices without a tangent, %d not perpendicular\n",
                    no_tangent, not_perpendicular);
        Check(no_tangent == 0, "every trunk vertex has a tangent frame");
        Check(not_perpendicular == 0, "and each one is perpendicular to its normal");
        bool handed = true;
        for (const VertexIn& v : t.trunk.vertices)
            if (std::fabs(std::fabs(v.tangent.w) - 1.0f) > 1e-3f) handed = false;
        Check(handed, "and records a handedness of plus or minus one");

        bool unit = true;
        for (const VertexIn& v : t.foliage.vertices) {
            const Vec3 n{v.normal.x, v.normal.y, v.normal.z};
            if (std::fabs(Length(n) - 1.0f) > 1e-3f) unit = false;
        }
        Check(unit, "and every one is unit length");
    }

    {
        std::printf("\nseeds produce different trees, and the same seed the same one\n");
        const auto vertex_sum = [](const Tree& t) {
            double s = 0.0;
            for (const VertexIn& v : t.trunk.vertices)
                s += double(v.position.x) + double(v.position.y) * 3.0 +
                     double(v.position.z) * 7.0;
            return s;
        };
        TreeParams a;
        a.seed = 12345;
        TreeParams b = a;
        b.seed = 999;
        const double sa = vertex_sum(MakeTree(a));
        const double sa2 = vertex_sum(MakeTree(a));
        const double sb = vertex_sum(MakeTree(b));
        std::printf("    seed 12345 -> %.4f (twice: %.4f), seed 999 -> %.4f\n", sa,
                    sa2, sb);
        // BIT-IDENTICAL for the same seed. A tree that differs between runs
        // makes every screenshot comparison meaningless, and the generator uses
        // its own xorshift for exactly this reason -- std::mt19937's sequence
        // is portable but its distributions are not.
        Check(sa == sa2, "the same seed gives a bit-identical tree");
        Check(std::fabs(sa - sb) > 1.0, "and a different seed a different one");
    }

    {
        std::printf("\nthe recursion is bounded\n");
        // levels and splits multiply: six levels of five splits is 15625
        // branches of 35 vertices each, over half a million for one tree. The
        // clamp is not taste, it is the difference between a tree and running
        // out of memory.
        TreeParams huge;
        huge.levels = 40;
        huge.splits = 40;
        const Tree t = MakeTree(huge);
        std::printf("    levels 40, splits 40 clamped to a %zu vertex tree\n",
                    t.trunk.vertices.size() + t.foliage.vertices.size());
        Check(t.trunk.vertices.size() + t.foliage.vertices.size() < 4000000,
              "absurd parameters are clamped rather than obeyed");

        TreeParams tiny;
        tiny.levels = 1;
        tiny.splits = 1;
        tiny.leaf_clusters = 1;
        const Tree small = MakeTree(tiny);
        Check(!small.trunk.vertices.empty(),
              "and the smallest possible tree is still a tree");
    }

    {
        std::printf("\nbounds and the forest helper\n");
        const Tree t = MakeTree(TreeParams{});
        float worst = 0.0f;
        for (const VertexIn& v : t.trunk.vertices)
            worst = std::max(worst,
                             Length(Vec3{v.position.x, v.position.y, v.position.z} -
                                    t.trunk.bounds.center) -
                                 t.trunk.bounds.radius);
        Check(worst <= 1e-4f, "no vertex lies outside the trunk's bounding sphere");

        // A FOREST AS ONE MESH is the difference between four hundred draw
        // calls and one, so the helper that builds it has to preserve every
        // triangle and place them where it was told.
        Mesh forest;
        for (int i = 0; i < 4; ++i)
            AppendTransformed(forest, t.trunk,
                              Mat4::Translation({float(i) * 10.0f, 0.0f, 0.0f}),
                              Vec4{1, 1, 1, 1});
        std::printf("    four trunks merged: %zu verts, %zu tris, bounds radius %.2f\n",
                    forest.vertices.size(), forest.indices.size() / 3,
                    double(forest.bounds.radius));
        Check(forest.vertices.size() == t.trunk.vertices.size() * 4,
              "the merged mesh has every vertex");
        Check(forest.indices.size() == t.trunk.indices.size() * 4,
              "and every triangle");
        bool merged_in_range = true;
        for (std::uint32_t i : forest.indices)
            if (i >= forest.vertices.size()) merged_in_range = false;
        // The one thing a merge gets wrong: forgetting to offset the indices,
        // so every copy points at the first copy's vertices. The mesh is then
        // four trees drawn on top of each other, which looks like one tree.
        Check(merged_in_range, "with every index rebased into the merged array");
        Vec3 lo, hi;
        Extent(forest, &lo, &hi);
        std::printf("    and spans x %.2f..%.2f, which is four trees apart\n",
                    double(lo.x), double(hi.x));
        Check(hi.x - lo.x > 30.0f, "and the copies are where they were placed");
    }

    std::printf(g_failures == 0 ? "\ntree_test: all checks passed\n"
                                : "\ntree_test: %d FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
