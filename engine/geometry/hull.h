// The convex hull of a point cloud, and the mass properties that follow from it.
//
// WHY the engine needs one: a collision shape has to be convex for any of the
// usual algorithms to work, and art is not. The hull of a rock, a crate or a
// character's silhouette is the shape you actually collide against, and it is
// derived from the mesh rather than authored twice.
//
// Note what is NOT needed for collision: GJK's support function is "the vertex
// furthest along a direction", and the furthest point of a cloud is by
// definition a hull vertex -- so a support function over the raw cloud is
// already exactly right. Building the hull is worth it for three other reasons:
// it throws away the interior points, which is the whole cost of that support
// query; it gives faces, which is what volume, inertia and debug drawing need;
// and it turns a degenerate cloud into a detectable failure rather than a
// silently flat shape.
#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "engine/core/math.h"

namespace eng::geom {

struct Hull {
    std::vector<Vec3> vertices;
    // Triangles, wound counter-clockwise seen from OUTSIDE, so a face normal is
    // Cross(v1 - v0, v2 - v0) with no sign to remember.
    std::vector<std::uint32_t> indices;

    [[nodiscard]] bool Empty() const { return vertices.size() < 4; }
    [[nodiscard]] std::size_t FaceCount() const { return indices.size() / 3; }
    [[nodiscard]] Vec3 Normal(std::size_t face) const;

    // Signed volume by the divergence theorem, summed over faces. Positive when
    // the winding is right, which makes it a cheap check on the hull itself.
    [[nodiscard]] float Volume() const;
    [[nodiscard]] Vec3 Centroid() const;
    // The inertia tensor about the centroid, for unit density. Multiply by the
    // density to get the real one. Symmetric, so only six numbers matter, but
    // it is returned whole because that is what a rotation acts on.
    [[nodiscard]] Mat3 Inertia() const;

    [[nodiscard]] bool Contains(Vec3 p, float tolerance = 1e-4f) const;
};

// The convex hull of `points`, or an empty hull if they are degenerate --
// fewer than four, all collinear, or all coplanar. Degenerate is a real
// answer, not a failure to work around: a flat "hull" has no volume, no
// inertia and no inside, and every algorithm downstream would divide by zero
// somewhere further away from the cause.
//
// `tolerance` is the distance below which a point is considered to lie ON a
// face rather than outside it. Too small and floating-point noise adds
// near-duplicate vertices; too large and real detail is swallowed.
[[nodiscard]] Hull ConvexHull(std::span<const Vec3> points,
                              float tolerance = 1e-5f);

}  // namespace eng::geom
