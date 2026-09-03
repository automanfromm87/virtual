#include "engine/geometry/meshlet.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace eng {
namespace {

// The bounding sphere of a set of points, by Ritter's method: a first pass
// finds the two most separated points along each axis, and a second grows the
// sphere to cover anything still outside.
//
// Not minimal -- an exact minimum-enclosing sphere costs an order of magnitude
// more code -- but within a few percent, and the cost of a slightly large
// sphere is a meshlet that occasionally survives a cull it could have failed.
// The cost of a slightly SMALL one is a meshlet wrongly culled, which is a
// hole in the model, so erring large is the only safe direction.
void BoundingSphere(const std::vector<Vec3>& p, Vec3* center, float* radius) {
    if (p.empty()) {
        *center = Vec3{0, 0, 0};
        *radius = 0.0f;
        return;
    }
    int lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
    for (std::size_t i = 1; i < p.size(); ++i)
        for (int a = 0; a < 3; ++a) {
            const float v = (&p[i].x)[a];
            if (v < (&p[std::size_t(lo[a])].x)[a]) lo[a] = int(i);
            if (v > (&p[std::size_t(hi[a])].x)[a]) hi[a] = int(i);
        }
    int axis = 0;
    float best = -1.0f;
    for (int a = 0; a < 3; ++a) {
        const float d = Length(p[std::size_t(hi[a])] - p[std::size_t(lo[a])]);
        if (d > best) {
            best = d;
            axis = a;
        }
    }
    Vec3 c = (p[std::size_t(lo[axis])] + p[std::size_t(hi[axis])]) * 0.5f;
    float r = best * 0.5f;
    for (const Vec3& v : p) {
        const float d = Length(v - c);
        if (d <= r) continue;
        // Grow just enough, moving the centre half the excess so the far side
        // stays inside. Growing the radius alone would eventually cover
        // everything but would end up far larger than necessary.
        const float nr = (r + d) * 0.5f;
        c = c + (v - c) * ((nr - r) / d);
        r = nr;
    }
    *center = c;
    *radius = r * 1.0001f;  // a hair of slack against float rounding
}

}  // namespace

MeshletBuild BuildMeshlets(const Mesh& mesh) {
    MeshletBuild out;
    const std::size_t tri_count = mesh.indices.size() / 3;
    if (tri_count == 0 || mesh.vertices.empty()) return out;

    Meshlet current;
    current.vertex_offset = 0;
    current.triangle_offset = 0;
    // Original vertex index -> its position in the CURRENT meshlet's list.
    std::unordered_map<std::uint32_t, std::uint8_t> local;
    std::vector<Vec3> positions;   // of the current meshlet, for the sphere
    std::vector<Vec3> normals;     // face normals, for the cone

    const auto close = [&]() {
        if (current.triangle_count == 0) return;
        BoundingSphere(positions, &current.center, &current.radius);

        // THE NORMAL CONE. The axis is the normalised average of the face
        // normals; the cutoff is the SINE of the widest angle any of them makes
        // with it, which is the quantity the cull test compares against.
        Vec3 axis{0, 0, 0};
        for (const Vec3& n : normals) axis = axis + n;
        const float axis_len = Length(axis);
        if (axis_len < 1e-6f) {
            // The normals cancel out: the meshlet faces every way at once, so
            // there is no cone and no culling to be had. A cutoff above 1 is
            // unsatisfiable and says "never reject", which is the safe answer --
            // a cone fitted to cancelling normals would be arbitrary and would
            // cull the meshlet from directions it is plainly visible in.
            current.cone_axis = Vec3{0, 0, 1};
            current.cone_cutoff = 2.0f;  // unsatisfiable: never cull
            current.cone_apex_offset = 0.0f;
        } else {
            axis = axis * (1.0f / axis_len);
            float min_dot = 1.0f;
            for (const Vec3& n : normals) min_dot = std::min(min_dot, Dot(axis, n));
            current.cone_axis = axis;
            // A half-angle at or past 90 degrees spans more than a hemisphere
            // and can never reject anything, so it gets the same
            // unsatisfiable cutoff.
            if (min_dot <= 0.0f) {
                current.cone_cutoff = 2.0f;  // half-angle at 90 degrees or more
                current.cone_apex_offset = 0.0f;
            } else {
                // sin of the half-angle, POSITIVE.
                //
                // The cull test is dot(A, normalize(apex - eye)) >= sin(theta):
                // every normal lies within theta of the axis, so the least any
                // of them makes with a direction at angle phi from the axis is
                // cos(phi + theta), and that is non-negative -- every triangle
                // back-facing -- exactly when phi <= 90 - theta.
                //
                // Stored NEGATIVE at first, which made the test true for
                // almost every direction and culled the entire mesh.
                current.cone_cutoff = std::sqrt(1.0f - min_dot * min_dot);
                // THE APEX, pushed back along the axis until every triangle's
                // plane passes through or in front of it. That is what makes
                // the test conservative: without it, a cluster off to one side
                // of a curved object is culled from angles it is plainly
                // visible from, and the symptom is a hole that opens and closes
                // as the camera moves.
                //
                // The apex is center - offset * axis, and it has to sit behind
                // every triangle's plane: dot(n, apex - p) <= 0 for all of
                // them. Writing d = dot(n, center - p), that is
                // offset >= d / dot(n, axis) -- so the offset is the MAXIMUM
                // over the triangles, not the minimum, and d is measured from
                // the triangle toward the centre rather than the other way.
                // Both were backwards.
                float offset = 0.0f;
                for (std::size_t i = 0; i < normals.size(); ++i) {
                    const float along = Dot(normals[i], axis);
                    if (along <= 1e-4f) continue;
                    const float d = Dot(normals[i], current.center - positions[i * 3]);
                    offset = std::max(offset, d / along);
                }
                current.cone_apex_offset = offset;
            }
        }
        out.meshlets.push_back(current);
    };

    for (std::size_t t = 0; t < tri_count; ++t) {
        const std::uint32_t idx[3] = {mesh.indices[t * 3], mesh.indices[t * 3 + 1],
                                      mesh.indices[t * 3 + 2]};
        if (idx[0] >= mesh.vertices.size() || idx[1] >= mesh.vertices.size() ||
            idx[2] >= mesh.vertices.size())
            continue;

        // How many NEW vertices this triangle would add.
        int fresh = 0;
        for (std::uint32_t i : idx)
            if (local.find(i) == local.end()) ++fresh;
        // Duplicates within the triangle -- a degenerate one -- would be
        // counted twice above; harmless, since it only makes the estimate
        // conservative.

        const bool full = current.triangle_count >= std::uint32_t(kMeshletMaxTriangles) ||
                          current.vertex_count + std::uint32_t(fresh) >
                              std::uint32_t(kMeshletMaxVertices);
        if (full) {
            close();
            current = Meshlet{};
            current.vertex_offset = std::uint32_t(out.vertices.size());
            current.triangle_offset = std::uint32_t(out.triangles.size() / 3);
            local.clear();
            positions.clear();
            normals.clear();
        }

        std::uint8_t tri_local[3];
        for (int k = 0; k < 3; ++k) {
            auto it = local.find(idx[k]);
            if (it == local.end()) {
                const auto slot = std::uint8_t(current.vertex_count);
                local.emplace(idx[k], slot);
                out.vertices.push_back(idx[k]);
                ++current.vertex_count;
                tri_local[k] = slot;
            } else {
                tri_local[k] = it->second;
            }
        }
        for (int k = 0; k < 3; ++k) out.triangles.push_back(tri_local[k]);
        ++current.triangle_count;

        const auto pos = [&](std::uint32_t i) {
            const Vec4& p = mesh.vertices[i].position;
            return Vec3{p.x, p.y, p.z};
        };
        const Vec3 a = pos(idx[0]), b = pos(idx[1]), c = pos(idx[2]);
        positions.push_back(a);
        positions.push_back(b);
        positions.push_back(c);
        const Vec3 face = Cross(b - a, c - a);
        const float fl = Length(face);
        // A degenerate triangle has no normal, and inventing one for it would
        // widen the cone for no reason. Its VERTICES still count toward the
        // bounds, because they are still drawn.
        normals.push_back(fl > 1e-12f ? face * (1.0f / fl) : Vec3{0, 0, 0});
    }
    close();
    return out;
}

}  // namespace eng
