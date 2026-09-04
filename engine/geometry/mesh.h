// Pure C++20 mesh generation. No platform code, no Metal — this produces plain
// arrays that any backend can upload.
#pragma once

#include <cstdint>
#include <functional>
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
    std::vector<std::uint32_t> indices;  // triangle list
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
// on back-face culling. Vertex count is (stacks+1)*(slices+1), and indices are
// 32-bit, so there is no practical cap.
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

// Fills in every vertex's tangent from the positions, uvs and normals already
// there. Idempotent, and safe on a mesh whose tangents are already set — it
// overwrites them.
//
// This is the Lengyel construction: per triangle, solve the 2x2 system that
// maps the uv edge deltas onto the position edge deltas, which gives the two
// world directions in which u and v increase. Accumulate both per vertex, then
// orthogonalise the tangent against the normal and record the handedness of the
// bitangent in .w.
//
// It is a MESH-TIME job, not a shader-time one. Deriving the frame per fragment
// from screen-space derivatives costs four derivatives a pixel, breaks wherever
// a derivative straddles a uv seam, and produces frames that disagree between
// neighbouring triangles, so a flat wall shows facets.
//
// CONVENTION, and it has to be stated because both are in use: the bitangent is
// cross(N, T) * tangent.w, and the normal map's green channel runs along it.
// With uv originating at the TOP-LEFT -- Metal's convention, and glTF's -- that
// makes green point DOWN the image. This is exactly what glTF specifies, so a
// map exported for glTF is correct here with no channel flip.
void GenerateTangents(Mesh&);

// A mesh that LIES ON a surface: a patch of ground, lifted slightly, carrying a
// decal's uv.
//
// WHY THIS EXISTS WHEN THERE IS ALREADY A DECAL SYSTEM. The projected kind
// writes into the G-buffer's albedo, and a forward renderer shades each surface
// once as it draws it -- there is no moment between "the surface exists" and
// "the surface is lit" at which to intervene. decals.h says so in its first
// paragraph. So a forward path cannot have projected decals, and the technique
// it CAN have is this one: build geometry that follows the receiving surface
// and draw it as an ordinary transparent object.
//
// The trade is explicit. A projected decal needs no knowledge of what it lands
// on and costs nothing per receiver; this one has to be built against a
// specific surface and rebuilt if that surface changes. In exchange it works on
// a path where the other cannot, it is depth-sorted like anything else, and it
// takes the same normal maps and lighting as the ground it sits on.
//
// ALPHA IS IN THE VERTEX COLOUR, falling off toward the rim. The lit shader
// multiplies the vertex alpha by the albedo map's, so the mesh supplies the
// shape of the fade and the texture supplies its detail -- and a decal with a
// hard rectangular edge, which is what a texture alone would give against a
// clamped sampler, is the most recognisable way to get this wrong.
struct GroundDecalDesc {
    Vec3 centre{0.0f, 0.0f, 0.0f};  // y is ignored; the surface decides it
    float radius = 1.0f;
    float rotation = 0.0f;  // radians about y, so repeated decals are not clones
    // How far above the surface. Enough to clear depth-buffer noise at the
    // distances it will be seen from, and no more -- lift it too far and it
    // stops looking painted on and starts looking like a sticker hovering.
    float lift = 0.02f;
    // Grid resolution across the patch. It has to follow the ground, so this is
    // bounded below by how bumpy the ground is rather than by the decal.
    int segments = 20;
    Vec4 tint{1.0f, 1.0f, 1.0f, 1.0f};
    // Where the alpha starts falling off, as a fraction of the radius. 1 would
    // be a hard circular edge.
    float fade_from = 0.45f;
};

[[nodiscard]] Mesh MakeGroundDecal(const GroundDecalDesc&,
                                   const std::function<float(float, float)>& height);

}  // namespace eng
