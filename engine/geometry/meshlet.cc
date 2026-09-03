#include "engine/geometry/meshlet.h"

#include <algorithm>
#include <array>
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

// Which triangles touch each vertex, as a flat CSR array.
//
// The growth loop needs "every triangle adjacent to the cluster", and the only
// cheap way to that is through the vertices: a triangle is adjacent when it
// shares a vertex. Building the table once is O(triangles); doing it by
// scanning would make the whole build quadratic.
//
// WELDED BY POSITION, not by index, and that is not a refinement. A UV sphere
// duplicates its seam column -- the same point needs u = 0 and u = 1, and one
// vertex can only carry one -- so the two sides of the seam share no index at
// all. Adjacency by index sees a wall there, the cluster stops growing, and it
// jumps somewhere else instead: measured at 14 spurious jumps on a 6016
// triangle sphere, which took the mean bounding radius from 0.25 to 0.33. The
// same is true of every hard edge and every uv island boundary in any real
// model.
struct VertexTriangles {
    std::vector<std::uint32_t> start;  // size = welded vertex count + 1
    std::vector<std::uint32_t> tris;
    std::vector<std::uint32_t> weld;   // original index -> welded index

    void Build(const Mesh& mesh) {
        const std::size_t nv = mesh.vertices.size();
        weld.resize(nv);
        // Quantised so that two positions that agree to a hundred-thousandth
        // are the same point. Exact float equality would miss vertices a
        // modelling package wrote through two different transforms.
        struct Key {
            std::int64_t x, y, z;
            bool operator==(const Key& o) const {
                return x == o.x && y == o.y && z == o.z;
            }
        };
        struct Hash {
            std::size_t operator()(const Key& k) const {
                std::size_t h = std::size_t(k.x) * 0x9E3779B97F4A7C15ull;
                h ^= std::size_t(k.y) + 0x9E3779B9u + (h << 6) + (h >> 2);
                h ^= std::size_t(k.z) + 0x85EBCA6Bu + (h << 6) + (h >> 2);
                return h;
            }
        };
        std::unordered_map<Key, std::uint32_t, Hash> seen;
        seen.reserve(nv);
        std::uint32_t welded = 0;
        for (std::size_t i = 0; i < nv; ++i) {
            const Vec4& p = mesh.vertices[i].position;
            const Key k{std::llround(double(p.x) * 100000.0),
                        std::llround(double(p.y) * 100000.0),
                        std::llround(double(p.z) * 100000.0)};
            auto [it, fresh] = seen.emplace(k, welded);
            if (fresh) ++welded;
            weld[i] = it->second;
        }

        start.assign(std::size_t(welded) + 1, 0);
        for (std::uint32_t i : mesh.indices)
            if (i < nv) ++start[weld[i] + 1];
        for (std::size_t i = 0; i < welded; ++i) start[i + 1] += start[i];
        tris.resize(start[welded]);
        std::vector<std::uint32_t> cursor(start.begin(), start.end() - 1);
        for (std::size_t t = 0; t + 2 < mesh.indices.size(); t += 3)
            for (int k = 0; k < 3; ++k) {
                const std::uint32_t v = mesh.indices[t + std::size_t(k)];
                if (v < nv) tris[cursor[weld[v]]++] = std::uint32_t(t / 3);
            }
    }
};

MeshletBuild BuildMeshlets(const Mesh& mesh) {
    MeshletBuild out;
    const std::size_t tri_count = mesh.indices.size() / 3;
    if (tri_count == 0 || mesh.vertices.empty()) return out;

    const auto pos_of = [&](std::uint32_t i) {
        const Vec4& p = mesh.vertices[i].position;
        return Vec3{p.x, p.y, p.z};
    };
    const auto tri_indices = [&](std::size_t t) {
        return std::array<std::uint32_t, 3>{mesh.indices[t * 3],
                                            mesh.indices[t * 3 + 1],
                                            mesh.indices[t * 3 + 2]};
    };

    // Precomputed per triangle: centroid and face normal. Both are read many
    // times by the scoring loop below and neither changes.
    std::vector<Vec3> centroid(tri_count), normal(tri_count);
    std::vector<std::uint8_t> valid(tri_count, 1);
    Vec3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
    for (std::size_t t = 0; t < tri_count; ++t) {
        const auto idx = tri_indices(t);
        if (idx[0] >= mesh.vertices.size() || idx[1] >= mesh.vertices.size() ||
            idx[2] >= mesh.vertices.size()) {
            valid[t] = 0;
            continue;
        }
        const Vec3 a = pos_of(idx[0]), b = pos_of(idx[1]), c = pos_of(idx[2]);
        centroid[t] = (a + b + c) * (1.0f / 3.0f);
        const Vec3 face = Cross(b - a, c - a);
        const float fl = Length(face);
        normal[t] = fl > 1e-12f ? face * (1.0f / fl) : Vec3{0, 0, 0};
        for (const Vec3& p : {a, b, c}) {
            lo = Vec3{std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
            hi = Vec3{std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
        }
    }
    // A length to measure distances against, so the weights below mean the same
    // thing on a mesh a millimetre across and one a kilometre across.
    const float mesh_radius = std::max(Length(hi - lo) * 0.5f, 1e-6f);

    VertexTriangles adjacency;
    adjacency.Build(mesh);

    Meshlet current;
    current.vertex_offset = 0;
    current.triangle_offset = 0;
    std::unordered_map<std::uint32_t, std::uint8_t> local;
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;

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


    // GROWTH, from a seed outward, rather than a walk down the index array.
    //
    // The index order is what the greedy version followed, and for a UV sphere
    // it runs along a whole latitude band before descending -- so a meshlet
    // came out as a RING around the sphere: 0.14 units tall, two units across,
    // with a bounding sphere of radius 1.05 on a sphere of radius 1.00. A
    // cluster whose bounds are larger than the whole model cannot be frustum
    // culled at all, and its normals span half the compass so its cone is
    // nearly useless too. Measured: 82 of 101 meshlets survived a cull that
    // should reject about half.
    //
    // So: start from one triangle and repeatedly take the best of its
    // neighbours, where "best" trades three things off.
    std::vector<std::uint8_t> claimed(tri_count, 0);
    std::vector<std::uint32_t> frontier;   // candidates, may contain duplicates
    std::vector<std::uint8_t> queued(tri_count, 0);
    std::size_t next_seed = 0;
    std::size_t placed = 0;

    // Running state of the cluster being grown.
    Vec3 sum_centroid{0, 0, 0};
    Vec3 sum_normal{0, 0, 0};

    const auto add_neighbours = [&](std::uint32_t t) {
        const auto idx = tri_indices(t);
        for (std::uint32_t raw : idx) {
            if (raw >= mesh.vertices.size()) continue;
            const std::uint32_t v = adjacency.weld[raw];
            for (std::uint32_t k = adjacency.start[v]; k < adjacency.start[v + 1];
                 ++k) {
                const std::uint32_t n = adjacency.tris[k];
                if (claimed[n] || queued[n] || !valid[n]) continue;
                queued[n] = 1;
                frontier.push_back(n);
            }
        }
    };

    const auto take = [&](std::uint32_t t) {
        const auto idx = tri_indices(t);
        std::uint8_t tri_local[3];
        for (int k = 0; k < 3; ++k) {
            auto it = local.find(idx[std::size_t(k)]);
            if (it == local.end()) {
                const auto slot = std::uint8_t(current.vertex_count);
                local.emplace(idx[std::size_t(k)], slot);
                out.vertices.push_back(idx[std::size_t(k)]);
                ++current.vertex_count;
                tri_local[k] = slot;
            } else {
                tri_local[k] = it->second;
            }
        }
        for (int k = 0; k < 3; ++k) out.triangles.push_back(tri_local[k]);
        ++current.triangle_count;
        for (std::uint32_t v : idx) positions.push_back(pos_of(v));
        normals.push_back(normal[t]);
        sum_centroid = sum_centroid + centroid[t];
        sum_normal = sum_normal + normal[t];
        claimed[t] = 1;
        ++placed;
        add_neighbours(t);
    };

    std::size_t valid_count = 0;
    for (std::uint8_t v : valid) valid_count += v;

    while (placed < valid_count) {
        // A fresh cluster. The seed is the first unclaimed triangle in index
        // order -- deterministic, and index order is at least locally coherent,
        // so consecutive clusters start near each other rather than jumping
        // about the model.
        while (next_seed < tri_count && (claimed[next_seed] || !valid[next_seed]))
            ++next_seed;
        if (next_seed >= tri_count) break;

        current = Meshlet{};
        current.vertex_offset = std::uint32_t(out.vertices.size());
        current.triangle_offset = std::uint32_t(out.triangles.size() / 3);
        local.clear();
        positions.clear();
        normals.clear();
        sum_centroid = Vec3{0, 0, 0};
        sum_normal = Vec3{0, 0, 0};
        std::fill(queued.begin(), queued.end(), std::uint8_t(0));
        frontier.clear();
        take(std::uint32_t(next_seed));

        for (;;) {
            if (current.triangle_count >= std::uint32_t(kMeshletMaxTriangles)) break;

            const Vec3 avg_c = sum_centroid * (1.0f / float(current.triangle_count));
            const float nl = Length(sum_normal);
            const Vec3 avg_n = nl > 1e-6f ? sum_normal * (1.0f / nl) : Vec3{0, 0, 1};

            int best = -1;
            float best_score = -1e30f;
            std::size_t write = 0;
            // Whether any neighbour existed at all, as distinct from whether
            // one FIT. The two mean opposite things: neighbours that do not fit
            // means the cluster is full and should close, and no neighbours at
            // all means it has run out of connected surface and should look
            // elsewhere. Treating them the same made every cluster on a sphere
            // jump when it hit the vertex limit, which packed them to 90
            // triangles and took the mean bounding radius from 0.25 to 0.33.
            bool had_neighbours = false;
            for (std::size_t i = 0; i < frontier.size(); ++i) {
                const std::uint32_t t = frontier[i];
                if (claimed[t]) continue;   // taken since it was queued
                frontier[write++] = t;

                had_neighbours = true;

                const auto idx = tri_indices(t);
                int extra = 0;
                for (std::uint32_t v : idx)
                    if (local.find(v) == local.end()) ++extra;
                if (current.vertex_count + std::uint32_t(extra) >
                    std::uint32_t(kMeshletMaxVertices))
                    continue;  // would not fit; leave it for the next cluster

                // THREE TERMS, and each one is a different failure if it is
                // missing.
                //
                //   REUSE. A triangle sharing two vertices with the cluster
                //   costs one vertex slot instead of three, so the cluster
                //   holds more triangles -- fewer clusters, fewer object
                //   threadgroups, and the 64-vertex limit stops being the
                //   binding constraint.
                //
                //   LOCALITY. Without it the cluster wanders and its bounding
                //   sphere grows without bound. This is the term the index-order
                //   version had none of.
                //
                //   CONE. Two triangles that face the same way can be rejected
                //   together; two that face opposite ways never can. Without
                //   this a cluster straddling a silhouette is uncullable
                //   however compact it is.
                const float reuse = float(3 - extra);
                const float dist = Length(centroid[t] - avg_c) / mesh_radius;
                const float align = Dot(normal[t], avg_n);
                const float score = reuse * 1.0f - dist * 8.0f + align * 2.0f;
                // Ties broken by the LOWER triangle index, so the build is
                // deterministic. Two candidates scoring identically is common
                // on a regular mesh -- a sphere's quads are all the same shape.
                if (score > best_score ||
                    (score == best_score && best >= 0 && t < std::uint32_t(best))) {
                    best_score = score;
                    best = int(t);
                }
            }
            frontier.resize(write);
            if (best < 0 && had_neighbours) break;  // full: close and reseed
            if (best < 0) {
                // NO NEIGHBOURS AT ALL: the cluster has reached the edge of a
                // connected component. Rather than close, jump to the nearest
                // unclaimed triangle -- a cluster is a spatial grouping, and
                // two pieces of surface that happen not to share an index are
                // still in the same place.
                //
                // Not an optimisation. A cube's six faces share NO vertex
                // indices at all -- 24 vertices for 12 triangles, because a
                // hard edge needs a different normal per side -- so closing on
                // an empty frontier turned a twelve-triangle cube into six
                // meshlets of two. Every one of them then costs its own object
                // threadgroup and its own copy of four vertices.
                const Vec3 avg = sum_centroid * (1.0f / float(current.triangle_count));
                int nearest = -1;
                float nearest_d = 1e30f;
                for (std::size_t t = 0; t < tri_count; ++t) {
                    if (claimed[t] || !valid[t]) continue;
                    const auto idx = tri_indices(t);
                    int extra = 0;
                    for (std::uint32_t v : idx)
                        if (local.find(v) == local.end()) ++extra;
                    if (current.vertex_count + std::uint32_t(extra) >
                        std::uint32_t(kMeshletMaxVertices))
                        continue;
                    const float d = Length(centroid[t] - avg);
                    // Ties by the lower index, so the build stays deterministic.
                    if (d < nearest_d) {
                        nearest_d = d;
                        nearest = int(t);
                    }
                }
                // A LINEAR SCAN, and it only runs when the frontier empties --
                // once per connected component on ordinary geometry. A mesh of
                // wholly disconnected triangles would make the build quadratic;
                // that wants a spatial index, and no mesh here is one.
                if (nearest < 0) break;
                // Only jump if the piece is CLOSE. Dragging in something across
                // the model would undo the whole point of clustering by
                // locality -- better to close a half-full cluster than to make
                // one whose bounds span the object.
                // Generous, deliberately: the jump only happens when the
                // connected surface has run out, and at that point the choice
                // is between one cluster holding two nearby pieces and two
                // clusters each holding one. The first is better -- half the
                // object threadgroups and no duplicated vertices -- and the
                // limits still cap how much it can absorb.
                // THE LIMIT SCALES with how much of the mesh one cluster ought
                // to cover. A cluster of k triangles out of n covers about k/n
                // of the surface, so its linear extent is about sqrt(k/n) of
                // the model's -- which is 100% of a twelve-triangle cube and
                // 14% of a six-thousand-triangle sphere. A fixed fraction of
                // the model cannot serve both: generous enough to merge a
                // cube's six faces, it lets the sphere drag in geometry half a
                // radius away.
                const float coverage =
                    std::sqrt(std::min(1.0f, float(kMeshletMaxTriangles) /
                                                 float(tri_count)));
                if (nearest_d > mesh_radius * coverage * 1.5f) break;
                take(std::uint32_t(nearest));
                continue;
            }
            take(std::uint32_t(best));
        }
        close();
    }
    return out;
}

}  // namespace eng
