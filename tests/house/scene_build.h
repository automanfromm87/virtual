// Uploads the apartment's meshes and materials once, then hands back a Scene
// for a given camera/cut configuration.
//
// Building the scene is the APPLICATION's job — engine/scene cannot create
// materials (that is the renderer's table) and engine/render does not know what
// a building is.
#pragma once

#include <string>
#include <vector>

#include "tests/house/plan.h"
#include "engine/render/renderer.h"
#include "engine/scene/scene.h"

namespace house {

struct Assets {
    eng::MeshHandle walls, floor, glass, roof, doors, frames, ground;
    std::vector<eng::MeshHandle> furniture;
    eng::MaterialHandle wall_mat, floor_mat, glass_mat, furniture_mat;
    eng::MaterialHandle roof_mat, door_mat, frame_mat, ground_mat;
    bool ok = false;
};

inline Assets Build(eng::Renderer& r, std::string& error) {
    Assets a;
    const eng::FloorPlan plan = Plan();

    a.walls = r.UploadMesh(eng::MakeWalls(plan, eng::Vec4{1, 1, 1, 1}));
    a.floor = r.UploadMesh(eng::MakeFloorSlab(plan, eng::Vec4{1, 1, 1, 1}));
    a.glass = r.UploadMesh(eng::MakeGlass(plan, eng::Vec4{1, 1, 1, 1}));
    // rise 1.9 m over a 7 m span is about a 28-degree pitch: steep enough to
    // read as a roof from any angle, shallow enough not to dwarf the building.
    a.roof = r.UploadMesh(eng::MakeGableRoof(plan, 1.9f, 0.45f, eng::Vec4{1, 1, 1, 1}));
    a.doors = r.UploadMesh(eng::MakeDoorLeaves(plan, 0.55f, eng::Vec4{1, 1, 1, 1}));
        // Frame depth must EXCEED the wall's half-thickness (0.08), or its face
    // lands exactly on the wall's and the two z-fight into a dashed mess
    // around every opening.
    a.frames = r.UploadMesh(eng::MakeWindowFrames(plan, 0.098f, eng::Vec4{1, 1, 1, 1}));
    // Site. Its top sits flush with the underside of the floor slab.
    a.ground = r.UploadMesh(eng::MakeBox(eng::Vec3{30.0f, 0.5f, 30.0f},
                                         eng::Vec4{1, 1, 1, 1}));
    for (const Block& b : Furniture())
        a.furniture.push_back(r.UploadMesh(eng::MakeBox(b.half, eng::Vec4{1, 1, 1, 1})));

    eng::MaterialDesc wall;
    wall.base_color = eng::Vec4{0.88f, 0.87f, 0.84f, 1.0f};
    wall.roughness = 0.85f;  // matte paint
    a.wall_mat = r.CreateMaterial(wall, error);

    eng::MaterialDesc floor;
    floor.base_color = eng::Vec4{0.52f, 0.40f, 0.29f, 1.0f};
    floor.roughness = 0.35f;  // varnished timber picks up a highlight
    a.floor_mat = r.CreateMaterial(floor, error);

    eng::MaterialDesc glass;
    glass.base_color = eng::Vec4{0.75f, 0.85f, 0.92f, 1.0f};
    glass.roughness = 0.05f;
    glass.transparent = true;
    // Double sided: a window has to be visible from indoors too, and a pane
    // this thin would otherwise vanish from one side.
    glass.cull = eng::rhi::Cull::None;
    a.glass_mat = r.CreateMaterial(glass, error);

    eng::MaterialDesc furn;
    furn.roughness = 0.6f;
    a.furniture_mat = r.CreateMaterial(furn, error);

    eng::MaterialDesc roof;
    roof.base_color = eng::Vec4{0.26f, 0.27f, 0.30f, 1.0f};  // slate
    roof.roughness = 0.75f;
    a.roof_mat = r.CreateMaterial(roof, error);

    eng::MaterialDesc door;
    door.base_color = eng::Vec4{0.42f, 0.28f, 0.17f, 1.0f};  // stained timber
    door.roughness = 0.45f;
    a.door_mat = r.CreateMaterial(door, error);

    eng::MaterialDesc frame;
    frame.base_color = eng::Vec4{0.94f, 0.94f, 0.92f, 1.0f};
    frame.roughness = 0.5f;
    a.frame_mat = r.CreateMaterial(frame, error);

    eng::MaterialDesc ground;
    ground.base_color = eng::Vec4{0.30f, 0.38f, 0.24f, 1.0f};  // grass
    ground.roughness = 0.95f;
    a.ground_mat = r.CreateMaterial(ground, error);

    a.ok = Valid(a.walls) && Valid(a.floor) && Valid(a.glass) && Valid(a.roof) &&
           Valid(a.doors) && Valid(a.frames) && Valid(a.ground) &&
           Valid(a.wall_mat) && Valid(a.floor_mat) && Valid(a.glass_mat) &&
           Valid(a.furniture_mat) && Valid(a.roof_mat) && Valid(a.door_mat) &&
           Valid(a.frame_mat) && Valid(a.ground_mat);
    for (eng::MeshHandle h : a.furniture)
        if (!Valid(h)) a.ok = false;
    if (!a.ok && error.empty()) error = "failed to build the apartment assets";
    return a;
}

// `clip_y` is the section-cut height; pass a huge value for no cut.
inline eng::Scene MakeScene(const Assets& a, float clip_y) {
    eng::Scene s;
    // Sun to the NORTH-WEST, opposite the default camera, and lower in the sky.
    //
    // Both choices are about being able to SEE the shadows. With the sun behind
    // the viewer every shadow falls on the far side of what casts it, so a
    // perfectly working shadow map produces a frame with no visible shadow in
    // it — which reads as a bug and is not one. A lower sun also lengthens the
    // shadows and lets sunlight reach further in through the windows.
    s.lightDir = eng::Vec4{-0.48f, 0.62f, -0.62f, 0.0f};
    s.lightColor = eng::Vec4{4.2f, 4.0f, 3.6f, 1.0f};
    s.shadowExtent = 9.0f;
    s.clipY = clip_y;

    const eng::Vec4 kWhite{1, 1, 1, 1};
    s.instances.push_back({a.ground, a.ground_mat,
                           eng::Mat4::Translation({0.0f, -0.65f, 0.0f}), kWhite});
    s.instances.push_back({a.floor, a.floor_mat, eng::Mat4::Identity(), kWhite});
    s.instances.push_back({a.walls, a.wall_mat, eng::Mat4::Identity(), kWhite});
    s.instances.push_back({a.roof, a.roof_mat, eng::Mat4::Identity(), kWhite});
    s.instances.push_back({a.doors, a.door_mat, eng::Mat4::Identity(), kWhite});
    s.instances.push_back({a.frames, a.frame_mat, eng::Mat4::Identity(), kWhite});

    const std::vector<Block> blocks = Furniture();
    for (std::size_t i = 0; i < blocks.size() && i < a.furniture.size(); ++i) {
        s.instances.push_back({a.furniture[i], a.furniture_mat,
                               eng::Mat4::Translation(blocks[i].at),
                               blocks[i].color});
    }

    // Glass LAST in the list, though the renderer sorts it into its own
    // back-to-front batch regardless — order here is only for readability.
    s.instances.push_back({a.glass, a.glass_mat, eng::Mat4::Identity(),
                           eng::Vec4{1, 1, 1, 0.34f}});
    return s;
}

}  // namespace house
