// Pure C++20. Meshlets: a mesh cut into small, spatially tight clusters, each
// with its own bounds and its own cone of visible normals.
//
// WHY. The vertex pipeline's unit of work is a draw call, and its unit of
// culling is an object. A hundred-thousand-triangle rock is either drawn whole
// or not at all, so the half of it facing away from the camera is transformed,
// clipped and rasterised into nothing. Meshlets make the unit of work small
// enough that most of it can be rejected before a single vertex is fetched --
// by frustum, by a cone test that rejects a cluster whose every triangle faces
// away, and by an occlusion test against a depth pyramid.
//
// The mesh-shader pipeline is what makes them cheap to draw: one threadgroup
// per meshlet decides for itself whether to emit anything, so the culling
// happens on the GPU with no readback and no indirect-draw bookkeeping.
//
// LIMITS. 64 vertices and 124 triangles per meshlet, which is what Metal's
// mesh stage allows per threadgroup and close to what every other API allows
// too. They are a hardware fact rather than a tuning choice.
#pragma once

#include <cstdint>
#include <vector>

#include "engine/core/math.h"
#include "engine/geometry/mesh.h"

namespace eng {

// Hardware limits, not preferences.
constexpr int kMeshletMaxVertices = 64;
constexpr int kMeshletMaxTriangles = 124;

struct Meshlet {
    // Where this meshlet's slices of the two index arrays start.
    std::uint32_t vertex_offset = 0;
    std::uint32_t triangle_offset = 0;
    std::uint32_t vertex_count = 0;
    std::uint32_t triangle_count = 0;

    // Bounding sphere, for the frustum and occlusion tests.
    Vec3 center{0.0f, 0.0f, 0.0f};
    float radius = 0.0f;

    // THE NORMAL CONE: an axis and the cosine of the half-angle that contains
    // every triangle's normal, plus how far along the axis the cone's apex sits
    // behind the bounding sphere's centre.
    //
    // This is what rejects the back half of a closed object before any vertex
    // is read, and it needs the apex as well as the axis: a cluster of
    // outward-facing triangles on a curved surface is invisible from the far
    // side of the object, and testing only the axis against the view direction
    // gets that wrong wherever the cluster is off to one side.
    Vec3 cone_axis{0.0f, 0.0f, 1.0f};
    // SIN of the half-angle. The cull test is
    //     dot(axis, normalize(apex - eye)) >= cone_cutoff
    // because every normal lies within the half-angle of the axis, so the least
    // any of them makes with a direction at angle phi from the axis is
    // cos(phi + theta) -- non-negative, meaning every triangle back-facing,
    // exactly when phi <= 90 - theta.
    //
    // Above 1 is UNSATISFIABLE and means "never cull", which is what a meshlet
    // whose normals span more than a hemisphere gets.
    float cone_cutoff = 2.0f;
    // How far back along the axis the cone's apex sits from the sphere's
    // centre: apex = center - axis * cone_apex_offset, chosen so the apex is
    // behind every triangle's plane and the test above is conservative.
    float cone_apex_offset = 0.0f;
};

struct MeshletBuild {
    std::vector<Meshlet> meshlets;
    // Indices into the original mesh's vertex array, grouped by meshlet.
    std::vector<std::uint32_t> vertices;
    // Triangles as three indices into the OWNING MESHLET's vertex list, so
    // each fits in a byte. That is the whole reason for the two-level scheme:
    // a meshlet has at most 64 vertices, so its triangles cost three bytes
    // each instead of twelve.
    std::vector<std::uint8_t> triangles;

    [[nodiscard]] bool Empty() const { return meshlets.empty(); }
};

// Cuts `mesh` into meshlets.
//
// Greedy and spatially coherent: triangles are taken in order and a meshlet is
// closed when it fills up or when the next triangle would stretch it too far.
// A proper implementation clusters by locality first, which produces tighter
// bounds and tighter cones; this one relies on the index order already being
// coherent, which it is for every generator in this engine and for anything a
// modelling package exports.
[[nodiscard]] MeshletBuild BuildMeshlets(const Mesh&);

}  // namespace eng
