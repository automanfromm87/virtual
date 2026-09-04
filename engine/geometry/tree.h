// Pure C++20. Procedural trees: a recursive skeleton swept into tapered tubes,
// with leaf clusters at the tips.
//
// WHY GENERATE THEM. This engine takes no third-party assets, so the only
// models it has are the ones it can compute -- and a landscape of nothing but
// boxes and spheres has no scale to it. A tree is the cheapest object that
// gives a scene scale, because everyone knows how big one is, and it is the
// cheapest thing that casts an interesting shadow.
//
// TRUNK AND FOLIAGE ARE SEPARATE MESHES. They want different materials: bark is
// rough and dark and lit like any other surface, and leaves are lighter, more
// translucent-looking and much rougher. Returning one mesh with a vertex colour
// would force one material on both and lose that.
#pragma once

#include <cstdint>

#include "engine/core/math.h"
#include "engine/geometry/mesh.h"

namespace eng {

struct TreeParams {
    // Any two trees with the same seed are identical, and two with different
    // seeds differ in every branch. A forest is one mesh per seed.
    std::uint32_t seed = 1;

    float height = 6.0f;          // trunk length before it starts branching
    float trunk_radius = 0.20f;
    // How many times a branch splits. Four is a recognisable tree; each level
    // multiplies the branch count, so six is thousands of tubes.
    int levels = 4;
    int splits = 3;               // children per branch

    // Each level's branches are this fraction of their parent's length and
    // radius. Below about 0.6 the tree looks stunted; above 0.85 it looks like
    // a bundle of sticks, because the children are nearly as thick as the
    // trunk.
    float length_falloff = 0.74f;
    float radius_falloff = 0.62f;
    // How far a child leans away from its parent, in radians, and how much that
    // angle varies between children.
    float spread = 0.62f;
    float spread_jitter = 0.22f;
    // How much a branch curves along its own length. Zero gives a tree made of
    // straight sticks, which reads as a diagram rather than as a tree.
    float droop = 0.28f;
    // PHOTOTROPISM: how strongly a branch curves back toward the sky as it
    // grows. Without it a tree spreads into a flat umbrella, because every
    // child leans away from its parent and nothing ever pulls it back up --
    // which is a recognisable shape, just not a tree's. Real branches lean out
    // and then turn upward, and that S-curve is most of what reads as growth.
    float upward = 0.55f;

    int sides = 7;                // ring resolution around a tube
    int segments = 4;             // rings along one branch

    // Leaf clusters at the tips: how many blobs, how big, and how much they
    // scatter around the tip.
    int leaf_clusters = 3;
    float leaf_size = 0.62f;
    float leaf_scatter = 0.45f;
    // Blobs are flattened, because a leaf mass is wider than it is tall and a
    // sphere reads as fruit.
    float leaf_flatten = 0.62f;

    Vec4 bark{0.29f, 0.21f, 0.15f, 1.0f};
    Vec4 leaf{0.20f, 0.42f, 0.16f, 1.0f};
    // Leaves vary in colour across the canopy -- new growth is lighter, and a
    // canopy of one flat green reads as plastic. This is the range, applied
    // per blob.
    float leaf_variation = 0.35f;
};

struct Tree {
    Mesh trunk;
    Mesh foliage;
    // The whole tree's extent, for placing it and for culling. The two meshes
    // carry their own bounds; this is the union.
    Bounds bounds;
};

[[nodiscard]] Tree MakeTree(const TreeParams&);

// Appends `src` to `dst`, transformed by `model` and tinted.
//
// Exposed because building a forest as ONE mesh is the difference between 400
// draw calls and one, and there is no other way to do that from outside.
//
// NORMALS are transformed by `model` and renormalised, which is correct for
// rotation and uniform scale and wrong for a non-uniform one -- that needs the
// inverse transpose. The tree generator only ever scales uniformly for this
// reason; the leaf blobs, which do not, build their normals directly.
void AppendTransformed(Mesh& dst, const Mesh& src, const Mat4& model, Vec4 tint);

}  // namespace eng
