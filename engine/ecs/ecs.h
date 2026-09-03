// Pure C++20. Entities, components, and a transform hierarchy.
//
// WHY a handle-and-sparse-set design rather than objects with pointers:
//
//  * Deletion. A pointer to a destroyed object is indistinguishable from a live
//    one. An Entity carries a GENERATION, so a stale handle is detectably stale
//    — the single most common source of crashes in a scene that changes.
//  * Iteration. Each component type lives in one contiguous array, so a system
//    that touches transforms walks transforms and nothing else.
//  * Composition. A thing is whatever components it has, rather than whatever
//    it inherited. Adding "this also has physics" is one line, not a new class.
//
// NOT here: archetypes, queries over component combinations, parallel systems.
// This is the storage and the hierarchy, which is what the rest of the engine
// actually needs; the query machinery earns its place when there are systems to
// run, and right now there are three.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "engine/core/math.h"
#include "engine/resource/handles.h"

namespace eng::ecs {

// index identifies the slot; generation says which occupant. Recycling a slot
// bumps the generation, so every handle to the previous occupant goes stale at
// once instead of silently aliasing the new one.
struct Entity {
    std::uint32_t index = 0;
    std::uint32_t generation = 0;
};

inline bool operator==(Entity a, Entity b) {
    return a.index == b.index && a.generation == b.generation;
}
inline bool operator!=(Entity a, Entity b) { return !(a == b); }

inline constexpr Entity kNoEntity{0xFFFFFFFFu, 0};

// Local transform plus a parent link. Stored as TRS rather than a matrix so a
// parent can be rotated without decomposing anything, and so rotations can be
// interpolated — a matrix cannot be blended without leaving the rotation group.
struct Transform {
    Vec3 position{0.0f, 0.0f, 0.0f};
    Quat rotation;
    float scale = 1.0f;
    Entity parent = kNoEntity;

    [[nodiscard]] Mat4 Local() const {
        return Mat4::Translation(position) * QuatToMat4(rotation) * Mat4::Scale(scale);
    }
};

// What to draw. The renderer's business; the ECS only stores it.
struct Renderable {
    MeshHandle mesh;
    MaterialHandle material;
    Vec4 tint{1.0f, 1.0f, 1.0f, 1.0f};
    bool visible = true;
};

// A human-readable label. Nothing in the engine reads it — it exists so a saved
// scene is legible and so a tool can ask for "the front door" instead of
// entity 47. A component rather than a field on Entity, because most entities
// in a real scene do not need one and an std::string each would not be free.
struct Name {
    std::string value;
};

// Links an entity to a body in the physics world. The physics world owns the
// simulation state; this is only the correspondence.
struct RigidBody {
    int body = -1;  // index into physics::World
};

// A contiguous array of T with a sparse index from entity slot to position.
//
// Removal swaps the last element into the hole, so iteration stays dense and
// there are no gaps to skip — at the cost of iteration order being arbitrary,
// which nothing here depends on.
template <class T>
class Pool {
  public:
    void Set(Entity e, const T& value) {
        if (e.index >= sparse_.size()) sparse_.resize(e.index + 1, kEmpty);
        if (sparse_[e.index] != kEmpty) {
            dense_[sparse_[e.index]] = value;
            return;
        }
        sparse_[e.index] = std::uint32_t(dense_.size());
        dense_.push_back(value);
        owners_.push_back(e);
    }

    [[nodiscard]] bool Has(Entity e) const {
        return e.index < sparse_.size() && sparse_[e.index] != kEmpty;
    }

    [[nodiscard]] T* Get(Entity e) {
        return Has(e) ? &dense_[sparse_[e.index]] : nullptr;
    }
    [[nodiscard]] const T* Get(Entity e) const {
        return Has(e) ? &dense_[sparse_[e.index]] : nullptr;
    }

    void Remove(Entity e) {
        if (!Has(e)) return;
        const std::uint32_t at = sparse_[e.index];
        const std::uint32_t last = std::uint32_t(dense_.size() - 1);
        if (at != last) {
            dense_[at] = dense_[last];
            owners_[at] = owners_[last];
            sparse_[owners_[at].index] = at;
        }
        dense_.pop_back();
        owners_.pop_back();
        sparse_[e.index] = kEmpty;
    }

    [[nodiscard]] std::size_t Size() const { return dense_.size(); }
    [[nodiscard]] T& At(std::size_t i) { return dense_[i]; }
    [[nodiscard]] const T& At(std::size_t i) const { return dense_[i]; }
    [[nodiscard]] Entity Owner(std::size_t i) const { return owners_[i]; }
    void Clear() {
        dense_.clear();
        owners_.clear();
        sparse_.clear();
    }

  private:
    static constexpr std::uint32_t kEmpty = 0xFFFFFFFFu;
    std::vector<T> dense_;
    std::vector<Entity> owners_;
    std::vector<std::uint32_t> sparse_;
};

class World {
  public:
    [[nodiscard]] Entity Create();
    void Destroy(Entity e);
    [[nodiscard]] bool Alive(Entity e) const;
    [[nodiscard]] int AliveCount() const { return alive_count_; }

    Pool<Transform> transforms;
    Pool<Renderable> renderables;
    Pool<RigidBody> bodies;
    Pool<Name> names;

    // Sets `child`'s parent, rejecting anything that would make a cycle. A
    // cycle here is not a crash you find later — it is an infinite loop the
    // first time world transforms are computed.
    bool SetParent(Entity child, Entity parent);

    // Recomputes every world matrix, parents before children. Call once per
    // frame after transforms change; results are read with WorldOf().
    void UpdateTransforms();
    [[nodiscard]] Mat4 WorldOf(Entity e) const;

    // Depth of `e` in the hierarchy; 0 for a root. Diagnostic.
    [[nodiscard]] int Depth(Entity e) const;

    // Accumulated rotation and scale from the root down to `e`. Composed from
    // the chain rather than pulled out of the world matrix, which has the scale
    // baked into its basis.
    [[nodiscard]] Quat WorldRotationOf(Entity e) const;
    [[nodiscard]] float WorldScaleOf(Entity e) const;

    // Writes a WORLD pose into `e`'s LOCAL transform, converting through its
    // parents.
    //
    // This exists because physics is world-space and a Transform is not. A
    // caller that assigns a body's position straight into a child's Transform
    // is writing world coordinates into a local field: a child one metre from a
    // parent five metres out ends up at x = 1 instead of 6. It looks right for
    // every root entity, which is why it survives until the first time someone
    // parents something.
    //
    // Reads the parent's pose as of the last UpdateTransforms, so a parent that
    // moved this frame is one frame stale. That is the usual trade for not
    // resolving the hierarchy twice per frame.
    void SetWorldPose(Entity e, Vec3 position, Quat rotation);

  private:
    std::vector<std::uint32_t> generations_;
    std::vector<std::uint32_t> free_list_;
    std::vector<Mat4> world_;
    int alive_count_ = 0;
};

}  // namespace eng::ecs
