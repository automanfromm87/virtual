// A flag on a pole, waving.
//
// Generated rather than imported, for two reasons. A checked-in character would
// be tens of thousands of vertices of content in a repository that has none;
// and a generated mesh has a KNOWN answer — every vertex's joint weights come
// from a formula here, so the gate can compute where each one should end up
// instead of trusting the picture.
//
// A flag is the honest demonstration of skinning rather than a decorative one.
// The mesh is a single rigid vertex buffer that never changes; every ripple in
// it is a chain of joint rotations blended per vertex. The failure modes are
// legible too: weights that are wrong tear the flag at the pole, and a stale
// palette leaves it hanging flat.
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "engine/anim/anim.h"
#include "engine/geometry/mesh.h"
#include "engine/render/renderer.h"
#include "engine/scene/scene.h"

namespace demo {

inline constexpr int kJoints = 10;         // bones from the hoist to the fly
inline constexpr float kFlagWidth = 3.0f;  // pole to trailing edge
inline constexpr float kFlagHeight = 1.85f;
inline constexpr int kSpanCols = 40;  // grid resolution along the wave
inline constexpr int kSpanRows = 14;
inline constexpr float kPoleHeight = 4.0f;
inline constexpr float kFlagTop = 3.7f;

inline constexpr float Spacing() { return kFlagWidth / float(kJoints - 1); }

struct Flag {
    eng::Mesh mesh;
    std::vector<eng::anim::SkinVertex> skin;
    eng::anim::Skeleton skeleton;
    eng::anim::Clip clip;
};

// A grid in the local xy plane, x running from the hoist (0) to the fly
// (kFlagWidth), jointed every Spacing() along x.
//
// Built as TWO sheets a hair apart with opposite winding and opposite normals,
// rather than one sheet drawn two-sided. A single sheet lit from behind has a
// normal pointing away from the light and comes out black — and a flag spends
// half of every wave with its back to the sun.
inline Flag MakeFlag() {
    Flag f;

    for (int j = 0; j < kJoints; ++j) {
        eng::anim::Joint joint;
        joint.name = "bone" + std::to_string(j);
        joint.parent = j == 0 ? -1 : j - 1;
        joint.rest.translation = eng::Vec3{j == 0 ? 0.0f : Spacing(), 0.0f, 0.0f};
        // Bind pose = rest pose, so the inverse bind is the walk back along the
        // chain to the origin.
        joint.inverse_bind =
            eng::Mat4::Translation(eng::Vec3{-Spacing() * float(j), 0.0f, 0.0f});
        f.skeleton.joints.push_back(joint);
    }
    f.skeleton.Finalize();

    const int verts_per_sheet = (kSpanCols + 1) * (kSpanRows + 1);
    for (int side = 0; side < 2; ++side) {
        const float z = side == 0 ? 0.008f : -0.008f;
        const float nz = side == 0 ? 1.0f : -1.0f;
        for (int r = 0; r <= kSpanRows; ++r) {
            for (int c = 0; c <= kSpanCols; ++c) {
                const float u = float(c) / float(kSpanCols);
                const float v = float(r) / float(kSpanRows);
                const float x = u * kFlagWidth;

                VertexIn vert{};
                vert.position = eng::Vec4{x, (v - 0.5f) * kFlagHeight, z, 0.0f};
                vert.normal = eng::Vec4{0.0f, 0.0f, nz, 0.0f};
                // Vertical bars, so the travelling wave is legible: they
                // compress and stretch as it passes. A plain colour would leave
                // the flag looking like a sheet of paper flexing.
                const bool bar = int(u * 6.0f) % 2 == 0;
                vert.color = bar ? eng::Vec4{0.92f, 0.30f, 0.26f, 1.0f}
                                 : eng::Vec4{0.97f, 0.95f, 0.90f, 1.0f};
                vert.uv = eng::Vec4{u, 1.0f - v, 0.0f, 0.0f};
                f.mesh.vertices.push_back(vert);

                // Weights from the position along x: every column between two
                // bones is a blend of the pair, which is what makes the wave
                // smooth rather than a row of hinges.
                const float along = x / Spacing();
                const int lo = std::min(int(along), kJoints - 1);
                const int hi = std::min(lo + 1, kJoints - 1);
                eng::anim::SkinVertex sv;
                sv.joints[0] = std::uint16_t(lo);
                sv.joints[1] = std::uint16_t(hi);
                sv.weights[0] = 1.0f - (along - float(lo));
                sv.weights[1] = along - float(lo);
                eng::anim::NormalizeWeights(&sv);
                f.skin.push_back(sv);
            }
        }
        const int base = side * verts_per_sheet;
        for (int r = 0; r < kSpanRows; ++r) {
            for (int c = 0; c < kSpanCols; ++c) {
                const auto a = std::uint16_t(base + r * (kSpanCols + 1) + c);
                const auto b = std::uint16_t(a + 1);
                const auto d = std::uint16_t(a + kSpanCols + 1);
                const auto e = std::uint16_t(d + 1);
                // a->e->d is counter-clockwise seen from +z: the cross product
                // of (e-a) and (d-a) is +z, which is what side 0's vertex
                // normal claims. Wound the other way the sheet whose normal
                // faces the light is the one back-face culling throws away, and
                // the flag renders as its own unlit reverse — dark, plausible,
                // and nothing to do with the skinning.
                if (side == 0)
                    f.mesh.indices.insert(f.mesh.indices.end(), {a, e, d, a, b, e});
                else
                    f.mesh.indices.insert(f.mesh.indices.end(), {a, d, e, a, e, b});
            }
        }
    }

    // Bounds cover the BIND pose. A waving flag reaches outside them, so they
    // are padded — the renderer culls on these, and a flag that vanishes as it
    // ripples toward the screen edge is worse than one drawn a frame too long.
    f.mesh.bounds.center = eng::Vec3{kFlagWidth * 0.5f, 0.0f, 0.0f};
    f.mesh.bounds.radius = kFlagWidth;

    // The wave. Each bone turns about the flag's vertical axis a little later
    // than the one before, which is what sends a ripple down the flag instead
    // of swinging the whole thing at once.
    //
    // Bone 0 is left alone: it is the hoist, lashed to the pole. Animating it
    // would swing the flag off its own pole, which is the most obvious way for
    // a flag to look wrong.
    f.clip.name = "wave";
    f.clip.duration = 1.6f;
    constexpr int kKeys = 24;
    for (int j = 1; j < kJoints; ++j) {
        eng::anim::Channel ch;
        ch.joint = j;
        ch.path = eng::anim::Path::Rotation;
        ch.interp = eng::anim::Interp::Linear;
        // Rotations COMPOUND down the chain, so a constant per-bone angle
        // already gives a tip that swings far. The ramp only stops the first
        // couple of bones from snapping at the hoist.
        const float ramp = std::min(1.0f, float(j) / 3.0f);
        const float amplitude = 0.42f * ramp;
        const float phase = float(j) * 0.92f;
        for (int k = 0; k <= kKeys; ++k) {
            const float time = float(k) / float(kKeys) * f.clip.duration;
            ch.times.push_back(time);
            const float theta = time / f.clip.duration * 6.2831853f;
            const eng::Quat yaw = eng::QuatFromAxisAngle(
                eng::Vec3{0, 1, 0}, std::sin(theta - phase) * amplitude);
            // A little roll as well, at half the rate, so the fly edge lifts
            // and drops instead of staying dead flat.
            const eng::Quat roll = eng::QuatFromAxisAngle(
                eng::Vec3{1, 0, 0}, std::sin(theta * 0.5f - phase) * amplitude * 0.5f);
            const eng::Quat q = yaw * roll;
            ch.values.insert(ch.values.end(), {q.x, q.y, q.z, q.w});
        }
        f.clip.channels.push_back(std::move(ch));
    }
    return f;
}

struct Assets {
    Flag flag;
    eng::MeshHandle mesh;
    eng::MeshHandle pole;
    eng::MeshHandle finial;
    eng::MeshHandle ground;
    eng::MaterialHandle flag_mat, pole_mat, finial_mat, ground_mat;
    bool ok = false;
};

inline Assets Build(eng::Renderer& r, std::string& error) {
    Assets a;
    a.flag = MakeFlag();
    a.mesh = r.UploadSkinnedMesh(a.flag.mesh, a.flag.skin, kJoints);
    a.pole = r.UploadMesh(eng::MakeBox(eng::Vec3{0.045f, kPoleHeight * 0.5f, 0.045f},
                                       eng::Vec4{1, 1, 1, 1}));
    a.finial = r.UploadMesh(eng::MakeUVSphere(0.10f, 16, 24, eng::Vec4{1, 1, 1, 1},
                                              eng::Vec4{1, 1, 1, 1}));
    a.ground = r.UploadMesh(
        eng::MakeBox(eng::Vec3{7.0f, 0.25f, 7.0f}, eng::Vec4{1, 1, 1, 1}));

    eng::MaterialDesc md;
    // The bars come from vertex colour, so the material stays white and lets
    // them through.
    md.base_color = eng::Vec4{1.0f, 1.0f, 1.0f, 1.0f};
    md.roughness = 0.62f;
    a.flag_mat = r.CreateMaterial(md, error);

    md.base_color = eng::Vec4{0.34f, 0.35f, 0.38f, 1.0f};
    md.roughness = 0.35f;
    md.metallic = 0.7f;
    a.pole_mat = r.CreateMaterial(md, error);

    md.base_color = eng::Vec4{0.85f, 0.68f, 0.24f, 1.0f};
    md.roughness = 0.22f;
    md.metallic = 1.0f;
    a.finial_mat = r.CreateMaterial(md, error);

    md = eng::MaterialDesc{};
    md.base_color = eng::Vec4{0.38f, 0.50f, 0.32f, 1.0f};
    md.roughness = 0.9f;
    a.ground_mat = r.CreateMaterial(md, error);

    a.ok = error.empty() && Valid(a.mesh) && Valid(a.pole) && Valid(a.ground);
    return a;
}

// The flag hangs from the top of the pole, its hoist a whisker clear of it.
inline eng::Mat4 FlagModel() {
    return eng::Mat4::Translation(
        eng::Vec3{0.06f, kFlagTop - kFlagHeight * 0.5f, 0.0f});
}

inline eng::Scene MakeScene(const Assets& a, float time) {
    eng::Scene s;
    s.lightDir = eng::Vec4{-0.44f, 0.72f, 0.53f, 0.0f};
    s.lightColor = eng::Vec4{4.0f, 3.9f, 3.7f, 1.0f};
    s.shadowExtent = 6.5f;
    s.shadowCascades = 3;
    s.shadowDistance = 18.0f;

    eng::anim::Pose pose;
    a.flag.clip.Sample(time, a.flag.skeleton, &pose);
    // The palette lives in the scene, not in the renderer: a scene describes
    // what to draw, and a pose is part of that description.
    eng::anim::ComputeJointMatrices(a.flag.skeleton, pose, &s.joint_matrices);

    eng::Instance ground;
    ground.mesh = a.ground;
    ground.material = a.ground_mat;
    ground.model = eng::Mat4::Translation(eng::Vec3{0.0f, -0.25f, 0.0f});
    s.instances.push_back(ground);

    eng::Instance pole;
    pole.mesh = a.pole;
    pole.material = a.pole_mat;
    pole.model = eng::Mat4::Translation(eng::Vec3{0.0f, kPoleHeight * 0.5f, 0.0f});
    s.instances.push_back(pole);

    eng::Instance finial;
    finial.mesh = a.finial;
    finial.material = a.finial_mat;
    finial.model = eng::Mat4::Translation(eng::Vec3{0.0f, kPoleHeight + 0.06f, 0.0f});
    s.instances.push_back(finial);

    eng::Instance flag;
    flag.mesh = a.mesh;
    flag.material = a.flag_mat;
    flag.model = FlagModel();
    flag.palette = 0;  // offset into s.joint_matrices
    s.instances.push_back(flag);
    return s;
}

}  // namespace demo
