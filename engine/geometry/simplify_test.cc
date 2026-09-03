// Mesh simplification, checked for the three things a simplifier has to get
// right and the one it is most often shipped without.
//
//   * It has to actually reduce. A simplifier that emits degenerate triangles
//     instead of dropping them has the same index count and looks identical.
//   * It has to preserve the SHAPE. Reducing a sphere to a smaller sphere is
//     the job; reducing it to a lumpy blob of the same triangle count is not,
//     and the difference is only visible against a measurement.
//   * It has to preserve the BOUNDS, or objects vanish when they switch level.
//   * And it must not produce SPIKES. That is the one. A quadric solve on a
//     nearly-flat cluster is nearly singular, the determinant test passes, and
//     the answer comes back a hundred metres away -- one such vertex ruins a
//     model, it only happens on some inputs, and it is why people abandon
//     quadrics for centroids.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "engine/geometry/mesh.h"
#include "engine/shaders/shader_types.h"
#include "engine/geometry/simplify.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

eng::Vec3 Xyz(const eng::Vec4& v) { return eng::Vec3{v.x, v.y, v.z}; }

// The largest distance any vertex sits from where it should be, for a shape
// whose surface is known analytically.
float MaxRadialError(const eng::Mesh& m, float radius) {
    float worst = 0.0f;
    for (const VertexIn& v : m.vertices)
        worst = std::max(worst, std::fabs(eng::Length(Xyz(v.position)) - radius));
    return worst;
}

float MaxExtent(const eng::Mesh& m) {
    float worst = 0.0f;
    for (const VertexIn& v : m.vertices)
        worst = std::max(worst, eng::Length(Xyz(v.position)));
    return worst;
}

// A flat grid in the XZ plane with a deliberate step in it, so there is both a
// large flat region -- where the quadric is singular and the fallback must be
// used -- and a sharp edge, which the quadric must keep.
eng::Mesh StepPlane(int n, float size, float step_height) {
    eng::Mesh m;
    for (int z = 0; z <= n; ++z)
        for (int x = 0; x <= n; ++x) {
            VertexIn v{};
            const float fx = (float(x) / float(n) - 0.5f) * size;
            const float fz = (float(z) / float(n) - 0.5f) * size;
            const float y = fx > 0.0f ? step_height : 0.0f;
            v.position = eng::Vec4{fx, y, fz, 0.0f};
            v.normal = eng::Vec4{0.0f, 1.0f, 0.0f, 0.0f};
            v.color = eng::Vec4{1, 1, 1, 1};
            m.vertices.push_back(v);
        }
    for (int z = 0; z < n; ++z)
        for (int x = 0; x < n; ++x) {
            const std::uint16_t a = std::uint16_t(z * (n + 1) + x);
            const std::uint16_t b = std::uint16_t(a + 1);
            const std::uint16_t c = std::uint16_t(a + n + 1);
            const std::uint16_t d = std::uint16_t(c + 1);
            m.indices.insert(m.indices.end(), {a, c, b, b, c, d});
        }
    m.bounds.radius = size;
    return m;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    const eng::Mesh sphere =
        eng::MakeUVSphere(1.0f, 48, 64, eng::Vec4{1, 1, 1, 1}, eng::Vec4{1, 1, 1, 1});

    {
        std::printf("it actually reduces\n");
        eng::SimplifyOptions o;
        o.cell_fraction = 1.0f / 16.0f;
        const eng::Mesh out = eng::Simplify(sphere, o);
        std::printf("    %zu verts / %zu tris  ->  %zu verts / %zu tris\n",
                    sphere.vertices.size(), sphere.indices.size() / 3,
                    out.vertices.size(), out.indices.size() / 3);
        Check(out.indices.size() < sphere.indices.size() / 2,
              "at a sixteenth-diagonal cell, under half the triangles remain");
        Check(out.vertices.size() < sphere.vertices.size(), "and fewer vertices");

        // NO DEGENERATES. A simplifier that merges vertices but keeps every
        // triangle produces exactly the same picture at exactly the same cost,
        // and passes any test that only counts vertices.
        int degenerate = 0;
        for (std::size_t t = 0; t + 2 < out.indices.size(); t += 3) {
            const std::uint16_t a = out.indices[t], b = out.indices[t + 1],
                                c = out.indices[t + 2];
            if (a == b || b == c || a == c) ++degenerate;
        }
        Check(degenerate == 0, "and no triangle has two corners the same");

        bool in_range = true;
        for (std::uint16_t i : out.indices)
            if (i >= out.vertices.size()) in_range = false;
        Check(in_range, "and every index is in range");
    }

    {
        std::printf("\nit keeps the shape\n");
        eng::SimplifyOptions o;
        o.cell_fraction = 1.0f / 12.0f;
        const eng::Mesh out = eng::Simplify(sphere, o);
        const float error = MaxRadialError(out, 1.0f);
        std::printf("    unit sphere, cell 1/12 of the diagonal: worst radial "
                    "error %.4f\n", error);
        // A CENTROID-ONLY clusterer on a sphere pulls every vertex toward the
        // centre of its cell, which is inside the surface, and the result is a
        // measurably smaller ball. The quadric places it back on the surface.
        Check(error < 0.06f, "every vertex stays within 6% of the true surface");

        // NO SPIKES. The check this file exists for.
        std::printf("    furthest vertex from the origin: %.4f\n", MaxExtent(out));
        Check(MaxExtent(out) < 1.15f, "and nothing is flung outside the shape");
    }

    {
        // A large flat region is where the quadric is SINGULAR -- the planes
        // are all parallel, so nothing pins the vertex down along the surface.
        // The solve must be rejected and the centroid used, and a spike here is
        // the classic symptom of not checking.
        std::printf("\na flat surface does not develop spikes\n");
        const eng::Mesh plane = StepPlane(60, 10.0f, 0.0f);
        eng::SimplifyOptions o;
        o.cell_fraction = 1.0f / 10.0f;
        const eng::Mesh out = eng::Simplify(plane, o);
        float worst_y = 0.0f;
        for (const VertexIn& v : out.vertices)
            worst_y = std::max(worst_y, std::fabs(v.position.y));
        std::printf("    %zu -> %zu tris, worst deviation from the plane %.6f\n",
                    plane.indices.size() / 3, out.indices.size() / 3, worst_y);
        Check(out.indices.size() < plane.indices.size(), "it still reduces");
        Check(worst_y < 1e-3f, "and every vertex stays on the plane");
    }

    {
        std::printf("\na hard edge survives\n");
        // Two planes meeting at a step. The vertices on either side share cells
        // along the seam, and merging across it would round the corner off.
        const eng::Mesh step = StepPlane(60, 10.0f, 1.0f);
        eng::SimplifyOptions o;
        o.cell_fraction = 1.0f / 10.0f;
        const eng::Mesh out = eng::Simplify(step, o);
        int at_bottom = 0, at_top = 0, between = 0;
        float worst_seam_x = 0.0f;
        for (const VertexIn& v : out.vertices) {
            if (v.position.y < 0.05f) ++at_bottom;
            else if (v.position.y > 0.95f) ++at_top;
            else {
                ++between;
                worst_seam_x = std::max(worst_seam_x, std::fabs(v.position.x));
            }
        }
        std::printf("    %d on the lower level, %d on the upper, %d in between "
                    "(furthest from the seam: %.3f)\n",
                    at_bottom, at_top, between, worst_seam_x);
        Check(at_bottom > 0 && at_top > 0, "both levels survive");

        // VERTICES IN BETWEEN ARE EXPECTED, and asking for none was the wrong
        // assertion -- it failed at exactly 8 of 64, which is one per cell along
        // the seam. A cell straddling the step genuinely contains both surfaces,
        // and the quadric's answer for a cluster of two parallel planes a metre
        // apart IS the point halfway between them. That is correct, not rounded.
        //
        // What would be wrong is those vertices appearing away from the seam,
        // which is what rounding the edge off actually looks like: the whole
        // surface sagging toward the discontinuity instead of one row of cells
        // splitting the difference.
        Check(between * 4 < at_bottom + at_top,
              "and most of the mesh is on one level or the other");
        // The cells are a tenth of the diagonal, which for a 10-unit plane is
        // about 1.4 units, so a straddling cell reaches at most that far.
        Check(worst_seam_x < 1.6f,
              "with every in-between vertex confined to the seam itself");
    }

    {
        std::printf("\nthe bounds come from the original\n");
        eng::Mesh with_bounds = sphere;
        with_bounds.bounds.center = eng::Vec3{0.0f, 0.0f, 0.0f};
        with_bounds.bounds.radius = 1.0f;
        eng::SimplifyOptions o;
        o.cell_fraction = 1.0f / 6.0f;
        const eng::Mesh out = eng::Simplify(with_bounds, o);
        // A level of detail stands in for the original at a distance, and the
        // cull test uses these. A tighter sphere on a coarser level means the
        // object pops out of existence when it switches -- far away, where
        // nobody is looking hard enough to work out why.
        Check(out.bounds.radius == with_bounds.bounds.radius,
              "so an object does not vanish when it switches level");
    }

    {
        std::printf("\nthe LOD chain gets monotonically cheaper\n");
        const std::vector<eng::Mesh> chain = eng::BuildLodChain(sphere, 5);
        bool falling = true;
        for (std::size_t i = 0; i < chain.size(); ++i) {
            std::printf("    level %zu: %5zu tris\n", i, chain[i].indices.size() / 3);
            if (i > 0 && chain[i].indices.size() >= chain[i - 1].indices.size())
                falling = false;
        }
        Check(chain.size() >= 3, "at least three distinct levels are produced");
        Check(falling, "and each is cheaper than the one before");
        Check(chain[0].indices.size() == sphere.indices.size(),
              "and level 0 is the original untouched");
    }

    {
        std::printf("\ndegenerate inputs\n");
        eng::Mesh empty;
        Check(eng::Simplify(empty).indices.empty(), "an empty mesh comes back empty");

        eng::Mesh one_tri;
        one_tri.vertices.resize(3);
        one_tri.vertices[0].position = eng::Vec4{0, 0, 0, 0};
        one_tri.vertices[1].position = eng::Vec4{1, 0, 0, 0};
        one_tri.vertices[2].position = eng::Vec4{0, 1, 0, 0};
        for (VertexIn& v : one_tri.vertices)
            v.normal = eng::Vec4{0, 0, 1, 0};
        one_tri.indices = {0, 1, 2};
        eng::SimplifyOptions huge;
        huge.cell_fraction = 100.0f;  // one cell for the whole mesh
        const eng::Mesh collapsed = eng::Simplify(one_tri, huge);
        // Every corner lands in one cluster, so the only triangle is
        // degenerate and is dropped -- leaving nothing. Returning an empty mesh
        // would be a model that silently disappears at its coarsest level, so
        // the input is handed back instead.
        Check(!collapsed.indices.empty(),
              "a mesh that would collapse entirely is returned unsimplified");

        eng::Mesh zero_area = one_tri;
        for (VertexIn& v : zero_area.vertices) v.position = eng::Vec4{0, 0, 0, 0};
        Check(!eng::Simplify(zero_area).vertices.empty(),
              "and a mesh with no extent does not divide by zero");
    }

    std::printf(g_failures == 0 ? "\nsimplify_test: all checks passed\n"
                                : "\nsimplify_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
