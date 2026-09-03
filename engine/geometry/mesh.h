// Pure C++20 mesh generation. No platform code, no Metal — this produces plain
// arrays that any backend can upload.
#pragma once

#include <cstdint>
#include <vector>

#include "engine/core/math.h"
#include "engine/shaders/shader_types.h"

namespace eng {

// Object-space bounding sphere. Cheap, and rotation-invariant — which is why
// frustum culling uses it rather than a box: a box has to be re-fitted every
// time the object turns, a sphere never does.
struct Bounds {
    Vec3 center{0.0f, 0.0f, 0.0f};
    float radius = 0.0f;
};

// Indexed triangle mesh in the engine's one and only vertex format.
struct Mesh {
    std::vector<VertexIn> vertices;
    std::vector<std::uint16_t> indices;  // triangle list
    Bounds bounds;
};

// UV sphere centred at the origin, +Y up.
//
//   stacks = latitude bands (poles to poles), slices = longitude segments.
//   32/64 is a good default: smooth enough that the silhouette reads as round.
//
// Vertices alternate between `a` and `b` in an 8x8 lat/long checker. That is
// not decoration: a single-coloured sphere is rotationally symmetric on screen,
// so a spinning one looks frozen and a wrong model matrix looks correct.
//
// Winding is counter-clockwise when viewed from OUTSIDE, so the caller can turn
// on back-face culling. Vertex count is (stacks+1)*(slices+1), which must stay
// under 65536 because indices are uint16 — that caps you at roughly 255x255.
[[nodiscard]] Mesh MakeUVSphere(float radius, int stacks, int slices, Vec4 a,
                                Vec4 b);

// Axis-aligned cube of side `size` centred at the origin, faces alternating
// between `a` and `b`.
//
// 24 vertices, not 8: adjacent faces meet at a hard edge, so a corner needs a
// different normal per face and cannot be shared. Sharing them would give you a
// cube shaded like a very lumpy sphere.
[[nodiscard]] Mesh MakeCube(float size, Vec4 a, Vec4 b);

// Axis-aligned box with independent half-extents, centred at the origin.
//
// Furniture is not cubic, and Mat4::Scale is deliberately uniform — a
// non-uniform scale matrix would silently break every shader that transforms a
// normal by it. Baking the proportions into the MESH sidesteps that entirely.
[[nodiscard]] Mesh MakeBox(Vec3 half_extents, Vec4 color);

}  // namespace eng
