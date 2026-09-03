#include "engine/physics/bvh.h"

namespace eng::physics {

void Bvh::Clear() {
    nodes_.clear();
    order_.clear();
    boxes_.clear();
    root_bounds_ = Aabb{};
    max_depth_ = 0;
}

void Bvh::Build(const std::vector<Aabb>& boxes) {
    nodes_.clear();
    order_.clear();
    max_depth_ = 0;
    root_bounds_ = Aabb{};
    boxes_ = boxes;
    if (boxes_.empty()) return;

    order_.resize(boxes_.size());
    for (std::size_t i = 0; i < boxes_.size(); ++i) order_[i] = int(i);
    // Generous, because the exact count depends on how the medians fall. The
    // point is not to save the last allocation; it is that a per-step rebuild
    // should not walk the reallocation ladder from 1 to N every single step.
    nodes_.reserve(4 * boxes_.size() / kLeafSize + 8);

    nodes_.push_back(Node{});
    Fill(0, 0, int(boxes_.size()), 0);
    root_bounds_ = nodes_[0].box;
}

void Bvh::Fill(int node, int begin, int end, int depth) {
    const std::vector<Aabb>& boxes = boxes_;
    max_depth_ = std::max(max_depth_, depth);

    Aabb bounds;
    for (int i = begin; i < end; ++i)
        bounds.Add(boxes[std::size_t(order_[std::size_t(i)])]);
    nodes_[std::size_t(node)].box = bounds;

    const int count = end - begin;
    const auto make_leaf = [&] {
        nodes_[std::size_t(node)].first = begin;
        nodes_[std::size_t(node)].count = count;
    };
    if (count <= kLeafSize || depth >= kMaxDepth) {
        make_leaf();
        return;
    }

    // The widest axis of the CENTROIDS, not of the bounds. Those are different
    // and the difference matters: a row of long thin boxes lying along X has
    // bounds that are widest in X, but if the boxes are stacked in Y then their
    // centroids only vary in Y, and splitting on X separates nothing.
    Aabb centroids;
    for (int i = begin; i < end; ++i)
        centroids.Add(boxes[std::size_t(order_[std::size_t(i)])].Centre());
    const Vec3 spread = centroids.Extent();
    int axis = 0;
    if (spread.y > spread.x) axis = 1;
    if (spread.z > (&spread.x)[axis]) axis = 2;

    // Every centroid at the same place -- a hundred identical crates spawned at
    // the origin, which is a thing level scripts do. No split separates them, so
    // recursing would re-derive the same range at every level down to the depth
    // cap. One big leaf is slow; a stack overflow is not recoverable.
    if ((&spread.x)[axis] <= 1e-9f) {
        make_leaf();
        return;
    }

    const int mid = begin + count / 2;
    // nth_element, not sort: the build only needs to know which half each item
    // belongs to, and partial selection is O(N) against O(N log N). Over the
    // whole recursion that is the difference between an O(N log N) build and an
    // O(N log^2 N) one.
    std::nth_element(order_.begin() + begin, order_.begin() + mid,
                     order_.begin() + end, [&](int a, int b) {
                         const Vec3 ca = boxes[std::size_t(a)].Centre();
                         const Vec3 cb = boxes[std::size_t(b)].Centre();
                         const float va = (&ca.x)[axis], vb = (&cb.x)[axis];
                         // Ties broken by INDEX, so the ordering is total and
                         // the tree is identical for identical input. Without
                         // it nth_element's arrangement among equal centroids is
                         // unspecified -- and a broadphase that reports pairs in
                         // a different order gives the solver its contacts in a
                         // different order, which resolves to a different
                         // answer. A physics engine that is not deterministic
                         // cannot be regression-tested at all.
                         return va != vb ? va < vb : a < b;
                     });

    // BOTH CHILDREN ALLOCATED NOW, before either subtree is built. That is what
    // makes them adjacent, and their adjacency is what lets a node store one
    // child index instead of two. Recursing first and allocating after would put
    // the whole left subtree between them.
    const int left = int(nodes_.size());
    nodes_.push_back(Node{});
    nodes_.push_back(Node{});
    nodes_[std::size_t(node)].first = left;
    nodes_[std::size_t(node)].count = 0;

    Fill(left, begin, mid, depth + 1);
    Fill(left + 1, mid, end, depth + 1);
}

}  // namespace eng::physics
