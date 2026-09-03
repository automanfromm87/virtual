// Pure C++20. Saves and loads an ECS world plus its rigid bodies as JSON.
//
// WHY this exists: until now every scene in this engine is a C++ function, so
// moving a ball two metres is a recompile. That is the slowest loop in the
// project and it gets slower as scenes grow.
//
// WHY it stores names, not handles: a MeshHandle is a slot index in one run of
// one Renderer. Writing it down would produce a file that loads into a
// different scene tomorrow — or into a valid-looking wrong one, which is worse.
// The caller supplies a Registry mapping names to whatever it uploaded.
//
// NOT here: meshes and materials themselves. This saves the ARRANGEMENT of a
// scene, not its assets; those come from engine/asset or the mesh generators.
#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "engine/ecs/ecs.h"
#include "engine/physics/physics.h"
#include "engine/resource/handles.h"

namespace eng::serialize {

// Two-way name/handle table. Both directions are stored because saving needs
// handle -> name and loading needs name -> handle, and deriving one from the
// other at every call would be a linear scan per entity.
class Registry {
  public:
    void AddMesh(std::string name, MeshHandle h);
    void AddMaterial(std::string name, MaterialHandle h);

    [[nodiscard]] MeshHandle Mesh(std::string_view name) const;
    [[nodiscard]] MaterialHandle Material(std::string_view name) const;
    // Empty string when the handle was never registered.
    [[nodiscard]] const std::string& MeshName(MeshHandle) const;
    [[nodiscard]] const std::string& MaterialName(MaterialHandle) const;

  private:
    std::unordered_map<std::string, std::uint32_t> mesh_by_name_, material_by_name_;
    std::unordered_map<std::uint32_t, std::string> mesh_by_id_, material_by_id_;
};

// One entity as it appears in a file. Exposed because a caller often wants to
// look at what it loaded — which entity is named "player" — without walking
// component pools.
struct Named {
    std::string name;
    ecs::Entity entity;
};

struct Scene {
    ecs::World* ecs = nullptr;
    physics::World* physics = nullptr;
};

// Writes every live entity. Entities without a Transform are skipped: they have
// no place in the world and nothing in this engine creates one.
[[nodiscard]] std::string Save(const ecs::World&, const physics::World&,
                               const Registry&);

// Appends into `out`, which does NOT have to be empty — loading two files into
// one world is how a level and its props stay separate. Returns false and fills
// `error` on malformed input or an unknown mesh/material name; a partial load
// is still visible in `out` for debugging, so callers that care should load
// into a scratch world first.
[[nodiscard]] bool Load(std::string_view json, const Registry&, Scene out,
                        std::string& error, std::vector<Named>* names = nullptr);

// Reads a file and calls Load.
[[nodiscard]] bool LoadFile(const std::string& path, const Registry&, Scene out,
                            std::string& error, std::vector<Named>* names = nullptr);

}  // namespace eng::serialize
