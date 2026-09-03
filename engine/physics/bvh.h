// A bounding volume hierarchy over axis-aligned boxes.
//
// WHAT IT REPLACES. A brute-force broadphase tests N*(N-1)/2 pairs. At 200
// bodies that is 20,000 tests and cheaper than any tree; at 2,000 it is two
// million and dominates the step; at 20,000 it is two hundred million and the
// simulation is over. The crossover is real and it is not far away, which is
// why "brute force is fine for now" stays true right up until it is
// catastrophically false.
//
// WHAT KIND OF TREE. Rebuilt from scratch every step, top-down, splitting at
// the MEDIAN along the widest axis of the centroids.
//
// Not a surface-area-heuristic build, and not an incrementally refitted tree,
// and both of those are the usual choices -- so the reasons matter:
//
//   * SAH builds a better tree and costs several times as much to build. It
//     wins when the tree is built once and queried millions of times, which is
//     the ray tracing case. A broadphase rebuilt sixty times a second and
//     queried a few thousand times per build is the opposite trade.
//   * An incrementally refitted tree avoids the rebuild but degrades: after a
//     few hundred steps of things moving, the boxes overlap so much that the
//     traversal visits most of the tree. Keeping it good needs rotations and
//     re-insertions, which is more code than the rebuild it replaces.
//
// A median split is O(N log N) with a tiny constant, has no degenerate case,
// and produces the same tree for the same input every time -- which matters
// more than it sounds, because a physics engine whose broadphase is
// nondeterministic has a simulation that is nondeterministic.
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "engine/core/math.h"

namespace eng::physics {

struct Aabb {
    // Deliberately INVERTED when empty, so that Add() on a fresh box just works
    // and there is no separate "is this initialised" flag for every caller to
    // forget. An empty box overlaps nothing, which is the right answer.
    Vec3 lo{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
            std::numeric_limits<float>::max()};
    Vec3 hi{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
            -std::numeric_limits<float>::max()};

    void Add(Vec3 p) {
        lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
        hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
    }
    void Add(const Aabb& o) {
        lo.x = std::min(lo.x, o.lo.x); lo.y = std::min(lo.y, o.lo.y); lo.z = std::min(lo.z, o.lo.z);
        hi.x = std::max(hi.x, o.hi.x); hi.y = std::max(hi.y, o.hi.y); hi.z = std::max(hi.z, o.hi.z);
    }
    void Expand(float m) {
        lo = lo - Vec3{m, m, m};
        hi = hi + Vec3{m, m, m};
    }
    [[nodiscard]] bool Empty() const { return hi.x < lo.x; }
    [[nodiscard]] Vec3 Centre() const { return (lo + hi) * 0.5f; }
    [[nodiscard]] Vec3 Extent() const { return hi - lo; }
    [[nodiscard]] bool Overlaps(const Aabb& o) const {
        return lo.x <= o.hi.x && hi.x >= o.lo.x && lo.y <= o.hi.y &&
               hi.y >= o.lo.y && lo.z <= o.hi.z && hi.z >= o.lo.z;
    }
    [[nodiscard]] bool Contains(Vec3 p) const {
        return p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y &&
               p.z >= lo.z && p.z <= hi.z;
    }

    // Slab test. `inv` is the componentwise reciprocal of the direction --
    // computed ONCE by the caller and reused down the whole traversal, which is
    // the only reason this is fast enough to be worth having.
    [[nodiscard]] bool Hits(Vec3 origin, Vec3 inv, float tmax) const {
        float tmin = 0.0f, tout = tmax;
        Slab(lo.x, hi.x, origin.x, inv.x, &tmin, &tout);
        Slab(lo.y, hi.y, origin.y, inv.y, &tmin, &tout);
        Slab(lo.z, hi.z, origin.z, inv.z, &tmin, &tout);
        return tmin <= tout;
    }

    // One axis of the slab test, and the whole reason it is a function is the
    // NaN case, which is not a hypothetical.
    //
    // A direction component of exactly zero gives an infinite reciprocal. Times
    // a non-zero numerator that is a signed infinity, which behaves correctly:
    // an origin inside the slab gets [-inf, +inf] and the axis constrains
    // nothing, an origin outside gets [-inf, -inf] or [+inf, +inf] and the axis
    // rejects. But when the origin lies EXACTLY on one of the two planes the
    // numerator is zero too, and 0 * inf is NaN.
    //
    // That is not an exotic input. It is a character standing at y = 1 in a
    // level built on a one-metre grid, firing a horizontal ray.
    //
    // The tempting fix -- relying on fmin/fmax to ignore NaN -- is wrong, and
    // silently: fmin(-inf, NaN) is -inf, which is right, but fmax(-inf, NaN) is
    // ALSO -inf, which collapses the exit distance to negative infinity and
    // turns every such ray into a miss. Detecting the NaN and dropping the axis
    // is the correct answer as well as the obvious one: NaN here means the ray
    // is parallel to the slab with its origin on the boundary, so it is inside
    // the closed slab for all t and imposes no constraint at all.
    static void Slab(float lo1, float hi1, float o, float inv, float* tmin,
                     float* tout) {
        const float a = (lo1 - o) * inv;
        const float b = (hi1 - o) * inv;
        if (std::isnan(a) || std::isnan(b)) return;
        *tmin = std::fmax(*tmin, std::fmin(a, b));
        *tout = std::fmin(*tout, std::fmax(a, b));
    }
};

class Bvh {
  public:
    // Builds over `boxes`. Leaf payloads are indices INTO THAT ARRAY, so the
    // caller keeps whatever mapping it likes from index to object and this
    // knows nothing about bodies.
    void Build(const std::vector<Aabb>& boxes);
    void Clear();

    [[nodiscard]] bool Empty() const { return nodes_.empty(); }
    [[nodiscard]] int NodeCount() const { return int(nodes_.size()); }
    [[nodiscard]] int LeafCount() const { return int(order_.size()); }
    // The deepest leaf. A balanced tree over N items is about log2(N/leaf_size);
    // a tree that has degenerated into a list is N/leaf_size, and the difference
    // between those two numbers is the difference between a working broadphase
    // and a slower one than brute force. Tests assert on it.
    [[nodiscard]] int MaxDepth() const { return max_depth_; }
    [[nodiscard]] const Aabb& Bounds() const { return root_bounds_; }

    // Calls fn(index) for EXACTLY the boxes that overlap `box` -- not for every
    // member of every leaf whose node box overlaps.
    //
    // The distinction is the difference between an index and a hint. A leaf
    // holds up to four boxes and its node box is their union, so reporting
    // whole leaves would hand the caller up to three false positives per hit.
    // Every caller would then have to redo the AABB test, which means every
    // caller needs the boxes, which means the tree may as well own them and do
    // it once -- and it is a two-line test against a cache line that has just
    // been touched anyway.
    //
    // Order is the TREE's, not the caller's. A caller that needs a stable order
    // must sort; World::OverlapShape does.
    template <class F>
    void QueryBox(const Aabb& box, F&& fn) const {
        if (nodes_.empty() || box.Empty()) return;
        int stack[64];
        int top = 0;
        stack[top++] = 0;
        while (top > 0) {
            const Node& n = nodes_[std::size_t(stack[--top])];
            if (!n.box.Overlaps(box)) continue;
            if (n.count > 0) {
                for (int k = 0; k < n.count; ++k) {
                    const int item = order_[std::size_t(n.first + k)];
                    if (boxes_[std::size_t(item)].Overlaps(box)) fn(item);
                }
            } else {
                // Depth 64 is not a guess: the build refuses to recurse past
                // kMaxDepth, which is 48, and each level pushes at most one
                // node. Overflowing this would be a build bug, not a deep scene.
                stack[top++] = n.first;
                stack[top++] = n.first + 1;
            }
        }
    }

    // Calls fn(index) for exactly the boxes the ray meets within `tmax`.
    // `direction` need not be normalised; `tmax` is in its units.
    template <class F>
    void QueryRay(Vec3 origin, Vec3 direction, float tmax, F&& fn) const {
        if (nodes_.empty()) return;
        // Reciprocals ONCE. Recomputing them per node turns three multiplies
        // into three divides at every level of every traversal, and a divide is
        // twenty times the latency.
        const Vec3 inv{1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z};
        int stack[64];
        int top = 0;
        stack[top++] = 0;
        while (top > 0) {
            const Node& n = nodes_[std::size_t(stack[--top])];
            if (!n.box.Hits(origin, inv, tmax)) continue;
            if (n.count > 0) {
                for (int k = 0; k < n.count; ++k) {
                    const int item = order_[std::size_t(n.first + k)];
                    if (boxes_[std::size_t(item)].Hits(origin, inv, tmax)) fn(item);
                }
            } else {
                stack[top++] = n.first;
                stack[top++] = n.first + 1;
            }
        }
    }

    // The box of leaf `i`, as the tree holds it. Callers that already keep
    // their own copy do not need this; it exists so that having the tree own
    // the boxes does not force a second array on anyone.
    [[nodiscard]] const Aabb& Box(int i) const { return boxes_[std::size_t(i)]; }

  private:
    struct Node {
        Aabb box;
        // For a LEAF (count > 0): the first index into order_.
        // For an INTERNAL node (count == 0): the index of the left child, whose
        // sibling is always the very next node. Storing one child index rather
        // than two is the standard trick and it halves the node's pointer
        // payload; it works because the build emits children as a pair.
        int first = 0;
        int count = 0;
    };

    // Four. Below that the tree is all pointer-chasing and the leaves cost more
    // in traversal than they save in tests; above about eight the linear scan
    // inside a leaf starts to dominate. Four is the flat part of that curve.
    static constexpr int kLeafSize = 4;
    // A tree deeper than this over any realistic input means the split has
    // stopped separating anything -- thousands of boxes with identical
    // centroids, which is what a scene of stacked identical crates at the same
    // position looks like. Stopping produces a big leaf, which is slow; not
    // stopping produces a stack overflow, which is not recoverable.
    static constexpr int kMaxDepth = 48;

    // Fills an ALREADY-ALLOCATED node. Allocation happens in the parent, in
    // pairs, which is what makes siblings adjacent.
    void Fill(int node, int begin, int end, int depth);

    std::vector<Node> nodes_;
    // Primitive indices, PERMUTED by the build so that every leaf's members are
    // contiguous. This is what lets a leaf be (first, count) instead of a list.
    std::vector<int> order_;
    // A COPY of the input boxes, in the caller's original order. Copied rather
    // than pointed at: the exact per-item test in the queries reads them long
    // after Build returned, and a tree holding a pointer into an array the
    // caller was free to resize is a dangling read waiting for the first person
    // who reuses the tree across a frame.
    std::vector<Aabb> boxes_;
    Aabb root_bounds_;
    int max_depth_ = 0;
};

}  // namespace eng::physics
