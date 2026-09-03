// Where the subsystems meet.
//
// This header is the point of the exercise. A glTF import supplies geometry, an
// ECS owns what exists and how it is parented, physics decides where the
// dynamic things end up, and the renderer draws whatever the ECS says is
// visible. None of the four knows about the others — the wiring lives up here,
// which is what "layered" is supposed to buy you.
//
// The scene is chosen so each subsystem is falsifiable on its own:
//   * The imported quad can only appear if the glTF reader worked.
//   * The carousel's children have NO physics — they move only because the
//     hierarchy carries them, which a flat instance list cannot do.
//   * The balls have no animation and no starting spin, so every degree they
//     turn came from friction at a contact point.
//
// The ARRANGEMENT is not in this file. Entities, transforms, bodies and
// parenting all come from world.scene.json; what stays here is the part that
// genuinely cannot be data — uploading meshes to the GPU and compiling
// materials — plus the per-frame wiring between the four subsystems.
#pragma once

#include <cmath>
#include <string>
#include <string_view>
#include <vector>

#include "engine/asset/gltf.h"
#include "engine/ecs/ecs.h"
#include "engine/geometry/mesh.h"
#include "engine/physics/physics.h"
#include "engine/render/renderer.h"
#include "engine/scene/scene.h"
#include "engine/serialize/scene_io.h"

namespace demo {

inline constexpr float kBallRadius = 0.42f;
inline constexpr eng::Vec3 kFloorHalf{9.0f, 0.5f, 9.0f};
// Top face of the floor slab. Balls rest with their centres a radius above it.
inline constexpr float kFloorTop = 0.0f;

inline constexpr eng::Vec3 kRampHalf{3.5f, 0.25f, 2.2f};
inline constexpr eng::Vec3 kWallHalf{9.0f, 0.45f, 0.3f};

struct World {
    eng::ecs::World ecs;
    eng::physics::World physics;
    // Name -> handle, so a scene file can say "sphere" instead of a slot index
    // that only means something inside this run's Renderer.
    eng::serialize::Registry registry;

    eng::MeshHandle sphere_mesh, box_mesh, ramp_mesh, wall_mesh, gltf_mesh;
    eng::rhi::TextureId gltf_albedo;  // decoded from the PNG inside the glTF
    eng::MaterialHandle floor_mat, ball_mat, arm_mat, ramp_mat, gltf_mat;

    eng::ecs::Entity carousel = eng::ecs::kNoEntity;
    eng::ecs::Entity floor_entity = eng::ecs::kNoEntity;
    eng::ecs::Entity ramp = eng::ecs::kNoEntity;
    eng::ecs::Entity sign = eng::ecs::kNoEntity;
    std::vector<eng::ecs::Entity> arms;
    std::vector<eng::ecs::Entity> balls;
    std::vector<eng::ecs::Entity> walls;
    // Captured at load, so `r` puts the balls back where the FILE said they
    // start rather than where a constant in this header says they do.
    std::vector<eng::Vec3> ball_start;

    float spin = 0.0f;
    bool ok = false;
};

inline eng::ecs::Entity FindByName(const eng::ecs::World& w, std::string_view name) {
    for (std::size_t i = 0; i < w.names.Size(); ++i)
        if (w.names.At(i).value == name) return w.names.Owner(i);
    return eng::ecs::kNoEntity;
}

// Everything that needs a GPU: meshes uploaded, materials compiled, and every
// one of them registered under the name a scene file will use.
inline bool BuildAssets(eng::rhi::Device& dev, eng::Renderer& r, World& w,
                        const char* gltf_json, std::string& error) {
    // TWO shades, not one. A uniformly coloured sphere is rotationally
    // symmetric on screen: it can be spinning at any rate and look frozen, so a
    // rolling ball would be indistinguishable from a sliding one — which is the
    // one thing this scene exists to show.
    w.sphere_mesh = r.UploadMesh(eng::MakeUVSphere(
        kBallRadius, 32, 64, eng::Vec4{1, 1, 1, 1}, eng::Vec4{0.5f, 0.5f, 0.55f, 1}));
    w.box_mesh = r.UploadMesh(eng::MakeBox(kFloorHalf, eng::Vec4{1, 1, 1, 1}));
    w.ramp_mesh = r.UploadMesh(eng::MakeBox(kRampHalf, eng::Vec4{1, 1, 1, 1}));
    w.wall_mesh = r.UploadMesh(eng::MakeBox(kWallHalf, eng::Vec4{1, 1, 1, 1}));

    // IMPORTED geometry, not generated. Its material comes from the file too,
    // so a regression in either half of the reader shows up on screen.
    const eng::gltf::Document doc = eng::gltf::ParseGltf(gltf_json, {}, error);
    if (!error.empty()) return false;
    if (doc.primitives.empty()) {
        error = "gltf document contains no primitives";
        return false;
    }
    w.gltf_mesh = r.UploadMesh(doc.primitives[0].mesh);

    // The imported IMAGE, decoded from a PNG that was base64'd into the glTF
    // document. Nothing about this texture exists in the source tree as pixels.
    if (!doc.images.empty() && !doc.images[0].Empty()) {
        const eng::Texture2D& img = doc.images[0];
        w.gltf_albedo = dev.CreateTexture2D(img.width, img.height, img.rgba.data());
    }

    eng::MaterialDesc md;
    md.base_color = eng::Vec4{0.55f, 0.56f, 0.58f, 1.0f};
    md.roughness = 0.8f;
    w.floor_mat = r.CreateMaterial(md, error);

    md.base_color = eng::Vec4{0.86f, 0.34f, 0.24f, 1.0f};
    md.roughness = 0.35f;
    w.ball_mat = r.CreateMaterial(md, error);

    md.base_color = eng::Vec4{0.28f, 0.52f, 0.86f, 1.0f};
    md.roughness = 0.5f;
    w.arm_mat = r.CreateMaterial(md, error);

    md.base_color = eng::Vec4{0.42f, 0.46f, 0.40f, 1.0f};
    md.roughness = 0.65f;
    w.ramp_mat = r.CreateMaterial(md, error);

    if (!doc.materials.empty()) {
        const eng::gltf::MaterialDef& g = doc.materials[0];
        eng::MaterialDesc gm;
        gm.base_color = g.base_color;
        gm.metallic = g.metallic;
        gm.roughness = g.roughness;
        // The imported quad is a single sheet with no back face of its own.
        gm.cull = eng::rhi::Cull::None;
        // A material's baseColorTexture is an index into the document's images;
        // the reader already resolved glTF's texture/sampler indirection.
        if (g.base_color_image >= 0 && Valid(w.gltf_albedo)) gm.albedo = w.gltf_albedo;
        w.gltf_mat = r.CreateMaterial(gm, error);
    } else {
        w.gltf_mat = w.arm_mat;
    }
    if (!error.empty()) return false;

    w.registry.AddMesh("sphere", w.sphere_mesh);
    w.registry.AddMesh("floor_slab", w.box_mesh);
    w.registry.AddMesh("ramp", w.ramp_mesh);
    w.registry.AddMesh("wall", w.wall_mesh);
    w.registry.AddMesh("panel", w.gltf_mesh);
    w.registry.AddMaterial("floor", w.floor_mat);
    w.registry.AddMaterial("ball", w.ball_mat);
    w.registry.AddMaterial("arm", w.arm_mat);
    w.registry.AddMaterial("ramp", w.ramp_mat);
    w.registry.AddMaterial("panel", w.gltf_mat);

    return Valid(w.sphere_mesh) && Valid(w.box_mesh) && Valid(w.ramp_mesh) &&
           Valid(w.wall_mesh) && Valid(w.gltf_mesh);
}

// Loads the arrangement and picks out the entities this demo drives by hand.
// Anything the file does not name is still loaded and still drawn — the lookups
// below are for the carousel and the reset key, not for rendering.
inline bool LoadScene(World& w, std::string_view scene_json, std::string& error) {
    if (!eng::serialize::Load(scene_json, w.registry,
                              eng::serialize::Scene{&w.ecs, &w.physics}, error))
        return false;

    w.floor_entity = FindByName(w.ecs, "floor");
    w.ramp = FindByName(w.ecs, "ramp");
    w.carousel = FindByName(w.ecs, "carousel");
    w.sign = FindByName(w.ecs, "sign");
    for (int i = 0;; ++i) {
        const eng::ecs::Entity e = FindByName(w.ecs, "arm" + std::to_string(i));
        if (!w.ecs.Alive(e)) break;
        w.arms.push_back(e);
    }
    for (int i = 0;; ++i) {
        const eng::ecs::Entity e = FindByName(w.ecs, "wall" + std::to_string(i));
        if (!w.ecs.Alive(e)) break;
        w.walls.push_back(e);
    }
    for (int i = 0;; ++i) {
        const eng::ecs::Entity e = FindByName(w.ecs, "ball" + std::to_string(i));
        if (!w.ecs.Alive(e)) break;
        w.balls.push_back(e);
        w.ball_start.push_back(w.ecs.transforms.Get(e)->position);
    }

    if (!w.ecs.Alive(w.carousel) || w.arms.empty() || w.balls.empty()) {
        error = "scene is missing carousel/arms/balls";
        return false;
    }
    return true;
}

inline World Build(eng::rhi::Device& dev, eng::Renderer& r, const char* gltf_json,
                   std::string_view scene_json, std::string& error) {
    World w;
    if (!BuildAssets(dev, r, w, gltf_json, error)) {
        if (error.empty()) error = "asset build failed";
        return w;
    }
    if (!LoadScene(w, scene_json, error)) return w;
    w.ok = true;
    return w;
}

// Drops the balls back where the scene file put them, at rest.
inline void Reset(World& w) {
    for (std::size_t i = 0; i < w.balls.size(); ++i) {
        const eng::ecs::RigidBody* rb = w.ecs.bodies.Get(w.balls[i]);
        if (!rb || rb->body < 0) continue;
        eng::physics::Body& b = w.physics[rb->body];
        b.position = w.ball_start[i];
        b.velocity = eng::Vec3{0.0f, 0.0f, 0.0f};
        b.orientation = eng::Quat{};
        b.angular_velocity = eng::Vec3{0.0f, 0.0f, 0.0f};
    }
}

// Advances physics, copies the results into the ECS, spins the carousel, then
// recomputes world matrices. Order matters: transforms are derived LAST, from
// everything that moved this frame.
inline void Update(World& w, float dt, bool simulate = true) {
    if (simulate) {
        w.physics.Step(dt);
        for (std::size_t i = 0; i < w.ecs.bodies.Size(); ++i) {
            const eng::ecs::Entity e = w.ecs.bodies.Owner(i);
            const int body = w.ecs.bodies.At(i).body;
            if (body < 0) continue;
            // SetWorldPose, not a direct assignment: a body's pose is world
            // space and a Transform is relative to its parent. Writing one into
            // the other is correct for every root entity and silently wrong the
            // moment a body hangs off something.
            //
            // Orientation as well as position, now that bodies spin — copying
            // only the position is what made a rolling ball look like a sliding
            // one.
            w.ecs.SetWorldPose(e, w.physics[body].position,
                               w.physics[body].orientation);
        }
        w.spin += dt * 0.7f;
        if (eng::ecs::Transform* t = w.ecs.transforms.Get(w.carousel)) {
            const float half = w.spin * 0.5f;  // quaternion is half-angle
            t->rotation = eng::Quat{0.0f, std::sin(half), 0.0f, std::cos(half)};
        }
    }
    w.ecs.UpdateTransforms();
}

// Walks the ECS and produces something the renderer understands. The renderer
// never sees an entity; the ECS never sees a draw call.
inline eng::Scene ToScene(const World& w) {
    eng::Scene s;
    s.lightDir = eng::Vec4{-0.42f, 0.74f, -0.52f, 0.0f};
    s.lightColor = eng::Vec4{4.4f, 4.2f, 3.9f, 1.0f};
    s.shadowExtent = 11.0f;
    for (std::size_t i = 0; i < w.ecs.renderables.Size(); ++i) {
        const eng::ecs::Renderable& r = w.ecs.renderables.At(i);
        if (!r.visible) continue;
        const eng::ecs::Entity e = w.ecs.renderables.Owner(i);
        s.instances.push_back({r.mesh, r.material, w.ecs.WorldOf(e), r.tint});
    }
    return s;
}

}  // namespace demo
