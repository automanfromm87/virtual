// Pure C++20. Opaque references to things the renderer owns.
//
// This is the seam that lets engine/scene say "draw the cube" without owning a
// cube, and lets engine/render own GPU buffers without knowing what a scene is.
// Both depend on this; neither depends on the other.
//
// Handles, not pointers. A pointer pins an address; a handle survives the
// resource being reuploaded, evicted, or hot-reloaded underneath it.
#pragma once

#include <cstdint>

namespace eng {

// 0 is always the null handle — never a valid resource.
//
// APPEND-ONLY tables: handles are indices into vectors that grow and never
// shrink or reorder, so an index always names the same resource and a stale
// handle can only go out of range (which every lookup checks) rather than
// silently alias a different one. That is why these stay bare indices instead
// of generation-counted ids like StreamId: nothing here is ever destroyed or
// recycled, so there is no lifetime for a generation to track. If a destroy
// API is ever added, these must become generation-counted with it.
struct MeshHandle {
    std::uint32_t v = 0;
};
struct MaterialHandle {
    std::uint32_t v = 0;
};

inline bool Valid(MeshHandle h) { return h.v != 0; }
inline bool Valid(MaterialHandle h) { return h.v != 0; }

inline bool operator==(MeshHandle a, MeshHandle b) { return a.v == b.v; }
inline bool operator==(MaterialHandle a, MaterialHandle b) { return a.v == b.v; }

// Built-in resources. The renderer registers these FIRST at construction, so
// their handles are fixed and usable before any Renderer exists. That is what
// lets engine/scene name a mesh without depending on engine/render — the same
// trick engines use for default/fallback/error resources.
inline constexpr MeshHandle kMeshSphere{1};
inline constexpr MeshHandle kMeshCube{2};

inline constexpr MaterialHandle kMaterialLit{1};    // shaded, depth-tested
inline constexpr MaterialHandle kMaterialFlat{2};   // unshaded, no depth
// Same shading and depth state as kMaterialLit, only the cull mode differs —
// so the pipeline cache must hand both of them the SAME pipeline object.
inline constexpr MaterialHandle kMaterialLitTwoSided{3};

}  // namespace eng
