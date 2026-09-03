#include "engine/serialize/scene_io.h"

#include <cstdio>
#include <unordered_map>
#include <vector>

#include "engine/asset/json.h"

namespace eng::serialize {
namespace {

constexpr int kVersion = 1;

const std::string& Empty() {
    static const std::string kEmpty;
    return kEmpty;
}

Vec3 ReadVec3(const json::Value& v, Vec3 fallback) {
    if (!v.IsArray() || v.Size() < 3) return fallback;
    return Vec3{float(v[0].Number()), float(v[1].Number()), float(v[2].Number())};
}

Vec4 ReadVec4(const json::Value& v, Vec4 fallback) {
    if (!v.IsArray() || v.Size() < 4) return fallback;
    return Vec4{float(v[0].Number()), float(v[1].Number()), float(v[2].Number()),
                float(v[3].Number())};
}

Quat ReadQuat(const json::Value& v) {
    if (!v.IsArray() || v.Size() < 4) return Quat{};
    return Quat{float(v[0].Number()), float(v[1].Number()), float(v[2].Number()),
                float(v[3].Number(1.0))};
}

void WriteVec3(json::Writer& w, Vec3 v) {
    const float a[3] = {v.x, v.y, v.z};
    w.Vec(a, 3);
}

void WriteVec4(json::Writer& w, Vec4 v) {
    const float a[4] = {v.x, v.y, v.z, v.w};
    w.Vec(a, 4);
}

void WriteQuat(json::Writer& w, Quat q) {
    const float a[4] = {q.x, q.y, q.z, q.w};
    w.Vec(a, 4);
}

}  // namespace

void Registry::AddMesh(std::string name, MeshHandle h) {
    mesh_by_name_[name] = h.v;
    mesh_by_id_[h.v] = std::move(name);
}

void Registry::AddMaterial(std::string name, MaterialHandle h) {
    material_by_name_[name] = h.v;
    material_by_id_[h.v] = std::move(name);
}

MeshHandle Registry::Mesh(std::string_view name) const {
    const auto it = mesh_by_name_.find(std::string(name));
    return it == mesh_by_name_.end() ? MeshHandle{} : MeshHandle{it->second};
}

MaterialHandle Registry::Material(std::string_view name) const {
    const auto it = material_by_name_.find(std::string(name));
    return it == material_by_name_.end() ? MaterialHandle{} : MaterialHandle{it->second};
}

const std::string& Registry::MeshName(MeshHandle h) const {
    const auto it = mesh_by_id_.find(h.v);
    return it == mesh_by_id_.end() ? Empty() : it->second;
}

const std::string& Registry::MaterialName(MaterialHandle h) const {
    const auto it = material_by_id_.find(h.v);
    return it == material_by_id_.end() ? Empty() : it->second;
}

std::string Save(const ecs::World& ecs, const physics::World& phys,
                 const Registry& reg) {
    // Entities are written in transform-pool order and referenced by their
    // POSITION in that list, not by Entity::index. An index is a slot in this
    // run's world and would collide the moment a file is loaded into a world
    // that already has entities in it.
    std::unordered_map<std::uint32_t, int> slot_of;
    std::vector<ecs::Entity> order;
    order.reserve(ecs.transforms.Size());
    for (std::size_t i = 0; i < ecs.transforms.Size(); ++i) {
        const ecs::Entity e = ecs.transforms.Owner(i);
        slot_of[e.index] = int(order.size());
        order.push_back(e);
    }

    json::Writer w;
    w.BeginObject();
    w.Key("version");
    w.Value(kVersion);

    w.Key("physics");
    w.BeginObject();
    w.Key("gravity");
    WriteVec3(w, phys.gravity);
    w.Key("fixed_dt");
    w.Value(double(phys.fixed_dt));
    w.Key("solver_iterations");
    w.Value(phys.solver_iterations);
    w.Key("linear_damping");
    w.Value(double(phys.linear_damping));
    w.Key("angular_damping");
    w.Value(double(phys.angular_damping));
    w.EndObject();

    w.Key("entities");
    w.BeginArray();
    for (const ecs::Entity e : order) {
        w.BeginObject();
        if (const ecs::Name* n = ecs.names.Get(e)) {
            w.Key("name");
            w.Value(n->value);
        }

        const ecs::Transform& t = *ecs.transforms.Get(e);
        w.Key("position");
        WriteVec3(w, t.position);
        // Skipped when they are the defaults. A scene file people are meant to
        // edit should not be nine tenths identity quaternions.
        if (t.rotation.x != 0.0f || t.rotation.y != 0.0f || t.rotation.z != 0.0f ||
            t.rotation.w != 1.0f) {
            w.Key("rotation");
            WriteQuat(w, t.rotation);
        }
        if (t.scale != 1.0f) {
            w.Key("scale");
            w.Value(double(t.scale));
        }
        if (const auto it = slot_of.find(t.parent.index);
            ecs.Alive(t.parent) && it != slot_of.end()) {
            w.Key("parent");
            w.Value(it->second);
        }

        if (const ecs::Renderable* r = ecs.renderables.Get(e)) {
            w.Key("renderable");
            w.BeginObject();
            w.Key("mesh");
            w.Value(reg.MeshName(r->mesh));
            w.Key("material");
            w.Value(reg.MaterialName(r->material));
            if (r->tint.x != 1.0f || r->tint.y != 1.0f || r->tint.z != 1.0f ||
                r->tint.w != 1.0f) {
                w.Key("tint");
                WriteVec4(w, r->tint);
            }
            if (!r->visible) {
                w.Key("visible");
                w.Value(false);
            }
            w.EndObject();
        }

        if (const ecs::RigidBody* rb = ecs.bodies.Get(e);
            rb && rb->body >= 0 && rb->body < phys.Count()) {
            const physics::Body& b = phys[rb->body];
            w.Key("body");
            w.BeginObject();
            if (b.shape.type == physics::ShapeType::Sphere) {
                w.Key("shape");
                w.Value("sphere");
                w.Key("radius");
                w.Value(double(b.shape.radius));
            } else {
                w.Key("shape");
                w.Value("box");
                w.Key("half_extents");
                WriteVec3(w, b.shape.half_extents);
            }
            // MASS, not inverse mass. Zero reads as "static", which is what the
            // person editing the file means; 0 for inverse mass means the
            // opposite and is a trap.
            w.Key("mass");
            w.Value(b.IsStatic() ? 0.0 : double(1.0f / b.inverse_mass));
            w.Key("restitution");
            w.Value(double(b.restitution));
            w.Key("friction");
            w.Value(double(b.friction));
            // Motion is only written when there is some, so a fresh scene file
            // stays quiet and a saved-mid-simulation one round-trips exactly.
            if (Dot(b.velocity, b.velocity) > 0.0f) {
                w.Key("velocity");
                WriteVec3(w, b.velocity);
            }
            if (Dot(b.angular_velocity, b.angular_velocity) > 0.0f) {
                w.Key("angular_velocity");
                WriteVec3(w, b.angular_velocity);
            }
            w.EndObject();
        }
        w.EndObject();
    }
    w.EndArray();
    w.EndObject();
    return w.Take();
}

bool Load(std::string_view text, const Registry& reg, Scene out, std::string& error,
          std::vector<Named>* names) {
    if (!out.ecs) {
        error = "Load needs an ecs world";
        return false;
    }
    const json::Value doc = json::Parse(text, error);
    if (!error.empty()) return false;
    if (!doc.IsObject()) {
        error = "scene root is not an object";
        return false;
    }
    const int version = doc["version"].Int(-1);
    if (version != kVersion) {
        error = "unsupported scene version " + std::to_string(version);
        return false;
    }

    if (out.physics && doc.Has("physics")) {
        const json::Value& p = doc["physics"];
        out.physics->gravity = ReadVec3(p["gravity"], out.physics->gravity);
        out.physics->fixed_dt = float(p["fixed_dt"].Number(out.physics->fixed_dt));
        out.physics->solver_iterations =
            p["solver_iterations"].Int(out.physics->solver_iterations);
        out.physics->linear_damping =
            float(p["linear_damping"].Number(out.physics->linear_damping));
        out.physics->angular_damping =
            float(p["angular_damping"].Number(out.physics->angular_damping));
    }

    const json::Value& list = doc["entities"];
    if (!list.IsArray()) {
        error = "scene has no entities array";
        return false;
    }

    // Two passes. Parenting can point forward as easily as backward, so every
    // entity has to exist before any link is made.
    std::vector<ecs::Entity> created;
    created.reserve(list.Size());
    for (std::size_t i = 0; i < list.Size(); ++i) {
        const json::Value& src = list[i];
        const ecs::Entity e = out.ecs->Create();
        created.push_back(e);

        ecs::Transform t;
        t.position = ReadVec3(src["position"], Vec3{0, 0, 0});
        t.rotation = ReadQuat(src["rotation"]);
        t.scale = float(src["scale"].Number(1.0));
        out.ecs->transforms.Set(e, t);

        if (src.Has("name")) {
            out.ecs->names.Set(e, ecs::Name{src["name"].Str()});
            if (names) names->push_back({src["name"].Str(), e});
        }

        if (src.Has("renderable")) {
            const json::Value& r = src["renderable"];
            const std::string& mesh_name = r["mesh"].Str();
            const std::string& mat_name = r["material"].Str();
            ecs::Renderable rc;
            rc.mesh = reg.Mesh(mesh_name);
            rc.material = reg.Material(mat_name);
            // An unregistered name is a hard error. Silently substituting the
            // null handle produces a scene that loads, renders nothing, and
            // reports no problem.
            if (!Valid(rc.mesh)) {
                error = "entity " + std::to_string(i) + ": unknown mesh '" +
                        mesh_name + "'";
                return false;
            }
            if (!Valid(rc.material)) {
                error = "entity " + std::to_string(i) + ": unknown material '" +
                        mat_name + "'";
                return false;
            }
            rc.tint = ReadVec4(r["tint"], Vec4{1, 1, 1, 1});
            rc.visible = r["visible"].Bool(true);
            out.ecs->renderables.Set(e, rc);
        }

        if (src.Has("body") && !out.physics) {
            error = "scene has bodies but no physics world was supplied";
            return false;
        }
    }

    for (std::size_t i = 0; i < list.Size(); ++i) {
        const json::Value& src = list[i];
        if (!src.Has("parent")) continue;
        const int p = src["parent"].Int(-1);
        if (p < 0 || p >= int(created.size())) {
            error = "entity " + std::to_string(i) + ": parent index out of range";
            return false;
        }
        if (!out.ecs->SetParent(created[i], created[std::size_t(p)])) {
            error = "entity " + std::to_string(i) + ": parent would make a cycle";
            return false;
        }
    }
    out.ecs->UpdateTransforms();

    // Bodies LAST, and from the WORLD transform.
    //
    // Physics is world-space throughout; an entity's Transform is relative to
    // its parent. Reading position straight off the Transform put a child's
    // body at its local coordinates — a child one metre from a parent five
    // metres out landed at x = 1 instead of x = 6, and the first step then
    // dragged the visible object back to meet it. Parents are only known after
    // the second pass, so this cannot happen any earlier.
    for (std::size_t i = 0; i < list.Size(); ++i) {
        const json::Value& src = list[i];
        if (!src.Has("body")) continue;
        const json::Value& bs = src["body"];
        const ecs::Entity e = created[i];

        physics::Body b;
        if (bs["shape"].Str() == "box")
            b.shape = physics::Shape::MakeBox(
                ReadVec3(bs["half_extents"], Vec3{0.5f, 0.5f, 0.5f}));
        else
            b.shape = physics::Shape::MakeSphere(float(bs["radius"].Number(0.5)));

        const Mat4 world = out.ecs->WorldOf(e);
        b.position = Vec3{world.col[3].x, world.col[3].y, world.col[3].z};
        // Composed from the chain of quaternions rather than extracted from the
        // matrix: the matrix has scale baked in, and pulling a rotation back out
        // of it means undoing that first.
        b.orientation = out.ecs->WorldRotationOf(e);
        b.velocity = ReadVec3(bs["velocity"], Vec3{0, 0, 0});
        b.angular_velocity = ReadVec3(bs["angular_velocity"], Vec3{0, 0, 0});
        b.restitution = float(bs["restitution"].Number(0.35));
        b.friction = float(bs["friction"].Number(0.5));
        b.SetMass(float(bs["mass"].Number(1.0)));
        out.ecs->bodies.Set(e, ecs::RigidBody{out.physics->Add(b)});
    }
    return true;
}

bool LoadFile(const std::string& path, const Registry& reg, Scene out,
              std::string& error, std::vector<Named>* names) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        error = "cannot open " + path;
        return false;
    }
    std::string text;
    char buf[4096];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
    std::fclose(f);
    return Load(text, reg, out, error, names);
}

}  // namespace eng::serialize
