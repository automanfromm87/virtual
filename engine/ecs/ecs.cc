#include "engine/ecs/ecs.h"

#include <algorithm>

namespace eng::ecs {

Entity World::Create() {
    ++alive_count_;
    if (!free_list_.empty()) {
        const std::uint32_t index = free_list_.back();
        free_list_.pop_back();
        return Entity{index, generations_[index]};
    }
    const auto index = std::uint32_t(generations_.size());
    generations_.push_back(1);  // start at 1 so a default Entity{} is never live
    return Entity{index, 1};
}

void World::Destroy(Entity e) {
    if (!Alive(e)) return;
    transforms.Remove(e);
    renderables.Remove(e);
    bodies.Remove(e);
    names.Remove(e);
    // Bumping the generation is what makes every outstanding handle to this
    // entity detectably stale, including ones stored in other components.
    ++generations_[e.index];
    free_list_.push_back(e.index);
    --alive_count_;
}

bool World::Alive(Entity e) const {
    return e.index < generations_.size() && generations_[e.index] == e.generation &&
           e.generation != 0;
}

bool World::SetParent(Entity child, Entity parent) {
    if (!Alive(child)) return false;
    Transform* t = transforms.Get(child);
    if (!t) return false;
    if (parent == kNoEntity || !Alive(parent)) {
        t->parent = kNoEntity;
        return true;
    }
    // Walk up from the proposed parent: if we reach the child, this edge would
    // close a cycle. Rejecting here is the only cheap place to catch it —
    // afterwards it is an infinite loop in UpdateTransforms.
    Entity cursor = parent;
    int guard = int(generations_.size()) + 1;
    while (Alive(cursor) && guard-- > 0) {
        if (cursor == child) return false;
        const Transform* ct = transforms.Get(cursor);
        if (!ct) break;
        cursor = ct->parent;
    }
    if (guard <= 0) return false;  // already cyclic somehow; refuse to add more
    t->parent = parent;
    return true;
}

int World::Depth(Entity e) const {
    int depth = 0;
    Entity cursor = e;
    int guard = int(generations_.size()) + 1;
    while (guard-- > 0) {
        const Transform* t = transforms.Get(cursor);
        if (!t || !Alive(t->parent)) break;
        cursor = t->parent;
        ++depth;
    }
    return depth;
}

Quat World::WorldRotationOf(Entity e) const {
    Quat q;
    Entity cursor = e;
    int guard = int(generations_.size()) + 1;
    while (Alive(cursor) && guard-- > 0) {
        const Transform* t = transforms.Get(cursor);
        if (!t) break;
        q = t->rotation * q;  // the parent applies after the child, as with T*R*S
        cursor = t->parent;
    }
    return Normalize(q);
}

float World::WorldScaleOf(Entity e) const {
    float s = 1.0f;
    Entity cursor = e;
    int guard = int(generations_.size()) + 1;
    while (Alive(cursor) && guard-- > 0) {
        const Transform* t = transforms.Get(cursor);
        if (!t) break;
        s *= t->scale;
        cursor = t->parent;
    }
    return s;
}

void World::SetWorldPose(Entity e, Vec3 position, Quat rotation) {
    Transform* t = transforms.Get(e);
    if (!t) return;
    if (!Alive(t->parent)) {  // a root's local pose IS its world pose
        t->position = position;
        t->rotation = rotation;
        return;
    }
    const Mat4 pw = WorldOf(t->parent);
    const Vec3 parent_pos{pw.col[3].x, pw.col[3].y, pw.col[3].z};
    const Quat parent_rot = WorldRotationOf(t->parent);
    const float parent_scale = WorldScaleOf(t->parent);
    const float inv_scale = parent_scale != 0.0f ? 1.0f / parent_scale : 1.0f;

    t->position = RotateInverse(parent_rot, position - parent_pos) * inv_scale;
    t->rotation = Conjugate(parent_rot) * rotation;
}

void World::UpdateTransforms() {
    world_.assign(generations_.size(), Mat4::Identity());

    // Sort by depth, then evaluate in that order. A parent is always shallower
    // than its child, so one pass suffices — and unlike a recursive walk this
    // touches each entity exactly once no matter how the tree is shaped.
    struct Item {
        int depth;
        std::size_t slot;
    };
    std::vector<Item> order;
    order.reserve(transforms.Size());
    for (std::size_t i = 0; i < transforms.Size(); ++i)
        order.push_back({Depth(transforms.Owner(i)), i});
    std::sort(order.begin(), order.end(),
              [](const Item& a, const Item& b) { return a.depth < b.depth; });

    for (const Item& it : order) {
        const Entity e = transforms.Owner(it.slot);
        const Transform& t = transforms.At(it.slot);
        const Mat4 local = t.Local();
        world_[e.index] = Alive(t.parent) ? world_[t.parent.index] * local : local;
    }
}

Mat4 World::WorldOf(Entity e) const {
    if (e.index >= world_.size()) return Mat4::Identity();
    return world_[e.index];
}

}  // namespace eng::ecs
