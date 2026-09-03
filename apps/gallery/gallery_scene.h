// A gallery at night: the scene the renderer was built for.
//
// Everything before this was a test rig — one object, one light, enough to
// prove a feature worked. This is the one that has to LOOK like something, and
// what makes that possible is having more than one light. A single directional
// sun lights every surface from the same angle with the same colour, which is
// why every earlier picture came out flat: there is nothing for the eye to read
// as shape except the silhouette.
//
// So: a dark room, a dim key light for shadows, three warm spots that pool on
// the plinths, and two coloured accents that only exist to put a different
// colour on the two sides of a curved surface. That last one is the whole
// trick — a sphere lit warm from the left and cool from the right reads as
// round; the same sphere lit by one white light reads as a disc.
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "apps/skinned/skinned_scene.h"
#include "engine/anim/anim.h"
#include "engine/asset/gltf.h"
#include "engine/geometry/mesh.h"
#include "engine/render/renderer.h"
#include "engine/scene/scene.h"
#include "engine/texture/texture.h"

namespace gallery {

// Three DIFFERENT heights. Not decoration: identical plinths under identical
// spots produce identical shadow maps, and a renderer that handed every light
// the same atlas tile would render exactly the right picture. The gate cannot
// tell those apart unless the maps differ.
inline constexpr float kPlinthTop = 0.95f;
inline constexpr float kPlinthX[3] = {-2.5f, 0.0f, 2.5f};
inline constexpr float kPlinthH[3] = {1.42f, 0.95f, 0.62f};

struct Assets {
    eng::MeshHandle sphere, floor, wall, lamp, flag, pole;
    eng::MeshHandle plinths[3];
    eng::MaterialHandle gold, copper, ceramic, stone, floor_mat, wall_mat,
        lamp_mat, flag_mat, pole_mat;
    demo::Flag flag_rig;
    bool ok = false;
};

inline Assets Build(eng::rhi::Device& dev, eng::Renderer& r, std::string& error) {
    Assets a;
    a.sphere = r.UploadMesh(eng::MakeUVSphere(0.55f, 48, 96, eng::Vec4{1, 1, 1, 1},
                                              eng::Vec4{1, 1, 1, 1}));
    for (int i = 0; i < 3; ++i)
        a.plinths[i] = r.UploadMesh(eng::MakeBox(
            eng::Vec3{0.42f, kPlinthH[i] * 0.5f, 0.42f}, eng::Vec4{1, 1, 1, 1}));
    a.floor = r.UploadMesh(
        eng::MakeBox(eng::Vec3{9.0f, 0.2f, 7.0f}, eng::Vec4{1, 1, 1, 1}));
    a.wall = r.UploadMesh(
        eng::MakeBox(eng::Vec3{9.0f, 3.6f, 0.2f}, eng::Vec4{1, 1, 1, 1}));
    a.lamp = r.UploadMesh(eng::MakeUVSphere(0.075f, 12, 20, eng::Vec4{1, 1, 1, 1},
                                            eng::Vec4{1, 1, 1, 1}));

    a.flag_rig = demo::MakeFlag();
    a.flag = r.UploadSkinnedMesh(a.flag_rig.mesh, a.flag_rig.skin, demo::kJoints);
    a.pole = r.UploadMesh(eng::MakeBox(eng::Vec3{0.035f, 1.55f, 0.035f},
                                       eng::Vec4{1, 1, 1, 1}));

    eng::MaterialDesc md;
    // POLISHED GOLD. Metals have no diffuse lobe at all: everything you see is
    // the specular reflection of a light, so a metal in a scene with one light
    // shows exactly one highlight and looks like plastic. It needs the others.
    md.base_color = eng::Vec4{1.00f, 0.78f, 0.34f, 1.0f};
    md.metallic = 1.0f;
    md.roughness = 0.20f;
    a.gold = r.CreateMaterial(md, error);

    md.base_color = eng::Vec4{0.95f, 0.55f, 0.42f, 1.0f};
    md.metallic = 1.0f;
    md.roughness = 0.38f;  // brushed: the same highlights, spread out
    a.copper = r.CreateMaterial(md, error);

    md.base_color = eng::Vec4{0.92f, 0.90f, 0.88f, 1.0f};
    md.metallic = 0.0f;
    md.roughness = 0.22f;  // glazed ceramic: a diffuse body under a tight sheen
    a.ceramic = r.CreateMaterial(md, error);

    md = eng::MaterialDesc{};
    md.base_color = eng::Vec4{0.20f, 0.20f, 0.22f, 1.0f};
    md.roughness = 0.80f;
    a.stone = r.CreateMaterial(md, error);

    // A slightly glossy floor, so the pools of light have a sheen and the
    // spots read as lights rather than as painted circles.
    md.base_color = eng::Vec4{0.16f, 0.16f, 0.18f, 1.0f};
    md.roughness = 0.35f;
    a.floor_mat = r.CreateMaterial(md, error);

    md.base_color = eng::Vec4{0.19f, 0.185f, 0.20f, 1.0f};
    md.roughness = 0.92f;
    a.wall_mat = r.CreateMaterial(md, error);

    md.base_color = eng::Vec4{0.30f, 0.32f, 0.36f, 1.0f};
    md.roughness = 0.30f;
    md.metallic = 0.8f;
    a.pole_mat = r.CreateMaterial(md, error);

    md = eng::MaterialDesc{};
    md.base_color = eng::Vec4{1.0f, 1.0f, 1.0f, 1.0f};
    md.roughness = 0.62f;
    a.flag_mat = r.CreateMaterial(md, error);

    // The visible bulbs. Unlit-bright so they read as the source rather than as
    // small white balls that happen to be near one.
    md = eng::MaterialDesc{};
    md.shading = eng::Shading::Flat;
    a.lamp_mat = r.CreateMaterial(md, error);

    a.ok = error.empty() && Valid(a.sphere) && Valid(a.flag) && Valid(a.floor) &&
           Valid(a.plinths[0]) && Valid(a.plinths[2]);
    return a;
}

// `flicker` is 0..1: how much the accents pulse. Zero makes the frame
// deterministic, which the offscreen gate needs.
inline eng::Scene MakeScene(const Assets& a, float time, float flicker = 1.0f) {
    eng::Scene s;

    // A DIM key light. It exists for the shadows and for a little shape on the
    // upward faces; if it did the lighting the spots would be invisible.
    s.lightDir = eng::Vec4{-0.35f, 0.86f, 0.37f, 0.0f};
    s.lightColor = eng::Vec4{0.42f, 0.44f, 0.52f, 1.0f};
    s.shadowExtent = 8.5f;
    // Nearly black. It is night and the room is lit by lamps; the default
    // outdoor ambient would light every wall on its own and leave the lamps
    // with nothing to do.
    s.ambientSky = eng::Vec3{0.045f, 0.052f, 0.075f};
    s.ambientGround = eng::Vec3{0.014f, 0.012f, 0.011f};

    eng::anim::Pose pose;
    a.flag_rig.clip.Sample(time, a.flag_rig.skeleton, &pose);
    eng::anim::ComputeJointMatrices(a.flag_rig.skeleton, pose, &s.joint_matrices);

    auto add = [&](eng::MeshHandle m, eng::MaterialHandle mat, eng::Mat4 model) {
        eng::Instance i;
        i.mesh = m;
        i.material = mat;
        i.model = model;
        s.instances.push_back(i);
    };

    add(a.floor, a.floor_mat, eng::Mat4::Translation(eng::Vec3{0, -0.2f, 0}));
    add(a.wall, a.wall_mat, eng::Mat4::Translation(eng::Vec3{0, 3.6f, -6.6f}));
    add(a.wall, a.wall_mat,
        eng::Mat4::Translation(eng::Vec3{-8.6f, 3.6f, 0}) *
            QuatToMat4(eng::QuatFromAxisAngle(eng::Vec3{0, 1, 0}, 1.5707963f)));

    const eng::MaterialHandle tops[3] = {a.gold, a.ceramic, a.copper};
    for (int i = 0; i < 3; ++i) {
        add(a.plinths[i], a.stone,
            eng::Mat4::Translation(eng::Vec3{kPlinthX[i], kPlinthH[i] * 0.5f, 0}));
        // A slow turn, so the highlights travel across the metals — a still
        // metal is the one case where a great BRDF looks like flat paint.
        add(a.sphere, tops[i],
            eng::Mat4::Translation(eng::Vec3{kPlinthX[i], kPlinthH[i] + 0.55f, 0}) *
                QuatToMat4(eng::QuatFromAxisAngle(eng::Vec3{0, 1, 0},
                                                  time * 0.35f + float(i))));
    }

    // The flag, off to one side and back, lit only by the accents.
    const eng::Vec3 flag_at{-5.6f, 1.55f, -2.2f};
    add(a.pole, a.pole_mat, eng::Mat4::Translation(flag_at));
    {
        eng::Instance f;
        f.mesh = a.flag;
        f.material = a.flag_mat;
        f.model = eng::Mat4::Translation(
                      eng::Vec3{flag_at.x + 0.05f, flag_at.y + 1.05f, flag_at.z}) *
                  QuatToMat4(eng::QuatFromAxisAngle(eng::Vec3{0, 1, 0}, 0.55f)) *
                  eng::Mat4::Scale(0.62f);
        f.palette = 0;
        s.instances.push_back(f);
    }

    // --- the lights ----------------------------------------------------------
    // Three spots, one per plinth, hung high and aimed straight down. A spot is
    // what makes a gallery read as a gallery: the pool of light has an edge.
    for (int i = 0; i < 3; ++i) {
        eng::Light spot;
        spot.type = eng::LightType::Spot;
        spot.position = eng::Vec3{kPlinthX[i], 4.4f, -0.95f};
        // Tilted toward the viewer, so the highlight it puts on a sphere
        // faces the camera instead of the ceiling.
        spot.direction = eng::Vec3{0.0f, -1.0f, 0.28f};
        // Radiance at ONE metre, and the plinth is three and a half metres
        // below the lamp — so 1/d² has already divided this by twelve by the
        // time it lands. Lamp intensities look absurd until you remember that.
        spot.color = eng::Vec3{170.0f, 148.0f, 116.0f};
        spot.range = 9.0f;
        spot.inner_degrees = 13.0f;
        spot.outer_degrees = 24.0f;
        spot.casts_shadow = true;
        s.lights.push_back(spot);
    }

    // Two coloured accents, low and to the sides. These are the ones doing the
    // work: they put a different colour on each side of every curved surface,
    // which is the cue that says "round".
    const float pulse = 1.0f + std::sin(time * 1.7f) * 0.12f * flicker;
    eng::Light cool;
    cool.position = eng::Vec3{-4.6f, 1.5f, 2.6f};
    cool.color = eng::Vec3{5.0f, 16.0f, 44.0f * pulse};
    cool.range = 11.0f;
    // A point light with shadows: six cube faces, so the plinths block it the
    // way they block the spots.
    cool.casts_shadow = true;
    cool.shadow_near = 0.35f;
    s.lights.push_back(cool);

    eng::Light warm;
    warm.position = eng::Vec3{4.8f, 1.2f, 2.2f};
    warm.color = eng::Vec3{46.0f, 15.0f, 4.0f};
    warm.range = 11.0f;
    s.lights.push_back(warm);

    // A dim fill behind the plinths, to separate them from the back wall.
    eng::Light rim;
    rim.position = eng::Vec3{0.0f, 2.1f, -4.4f};
    rim.color = eng::Vec3{9.0f, 10.0f, 15.0f};
    rim.range = 8.0f;
    s.lights.push_back(rim);

    // The visible bulbs for the two accents. Drawn flat, so they are the one
    // thing in the frame that is not shaded.
    for (const eng::Light& l : {cool, warm, rim}) {
        eng::Instance bulb;
        bulb.mesh = a.lamp;
        bulb.material = a.lamp_mat;
        bulb.model = eng::Mat4::Translation(l.position);
        // The bulb is the SOURCE, so it has to be the brightest thing in the
        // frame — otherwise the pools of light it casts bloom and the lamp
        // itself does not, which reads as backwards.
        //
        // Flat shading writes the tint straight out, and the scene target is
        // linear HDR, so a value above one is meaningful here in a way it was
        // not when every shader tone mapped its own output.
        const float peak = std::fmax(std::fmax(l.color.x, l.color.y), l.color.z);
        const float k = 7.0f / peak;
        bulb.tint = eng::Vec4{l.color.x * k, l.color.y * k, l.color.z * k, 1};
        s.instances.push_back(bulb);
    }
    return s;
}

}  // namespace gallery
