// A skinned tentacle, generated rather than imported.
//
// Two reasons it is procedural. A checked-in character would be tens of
// thousands of vertices of content in a repository that has none; and a
// generated one has a KNOWN answer — every vertex's joint weights are decided
// by a formula here, so the test can compute where each one should end up
// rather than trusting the picture.
//
// The imported fixture is exercised separately, in the offscreen gate: import
// and skinning are different things to get wrong.
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include "engine/anim/anim.h"
#include "engine/geometry/mesh.h"
#include "engine/render/renderer.h"
#include "engine/scene/scene.h"

namespace demo {

inline constexpr int kJoints = 8;
inline constexpr float kSegmentLength = 0.62f;
inline constexpr int kRingsPerSegment = 3;
inline constexpr int kRingVerts = 12;

struct Tentacle {
    eng::Mesh mesh;
    std::vector<eng::anim::SkinVertex> skin;
    eng::anim::Skeleton skeleton;
    eng::anim::Clip clip;
};

// A tube along +x, jointed every kSegmentLength.
//
// Weights come from the distance along the tube, so every ring between two
// joints is a smooth blend of the pair. That is what makes the bend look like
// a bend rather than a hinge, and it is also the case that exposes a palette
// built in the wrong order — a rigid one-joint-per-vertex mesh does not.
inline Tentacle MakeTentacle() {
    Tentacle t;

    for (int j = 0; j < kJoints; ++j) {
        eng::anim::Joint joint;
        joint.name = "bone" + std::to_string(j);
        joint.parent = j == 0 ? -1 : j - 1;
        // Local: each bone sits one segment along its parent. The root sits at
        // the origin, so the whole chain runs from x = 0.
        joint.rest.translation =
            eng::Vec3{j == 0 ? 0.0f : kSegmentLength, 0.0f, 0.0f};
        // Bind pose = rest pose, so the inverse bind is a translation back to
        // the origin by however far along the tube the joint sits.
        joint.inverse_bind =
            eng::Mat4::Translation(eng::Vec3{-kSegmentLength * float(j), 0, 0});
        t.skeleton.joints.push_back(joint);
    }
    t.skeleton.Finalize();

    const int rings = (kJoints - 1) * kRingsPerSegment + 1;
    for (int r = 0; r < rings; ++r) {
        const float along = float(r) / float(kRingsPerSegment);  // in joint units
        const float x = along * kSegmentLength;
        // Tapered, so the far end is visibly thinner and the silhouette says
        // which way round the tentacle is.
        const float radius = 0.30f * (1.0f - 0.55f * float(r) / float(rings - 1));

        const int lo = std::min(int(along), kJoints - 1);
        const int hi = std::min(lo + 1, kJoints - 1);
        const float frac = along - float(lo);

        for (int k = 0; k < kRingVerts; ++k) {
            const float a = float(k) / float(kRingVerts) * 6.2831853f;
            VertexIn v{};
            v.position = eng::Vec4{x, std::cos(a) * radius, std::sin(a) * radius, 0};
            v.normal = eng::Vec4{0.0f, std::cos(a), std::sin(a), 0.0f};
            // Two shades around the ring, so a rotating tentacle does not look
            // frozen — the same reason the rolling balls are chequered.
            v.color = (k % 2) ? eng::Vec4{1, 1, 1, 1} : eng::Vec4{0.62f, 0.66f, 0.72f, 1};
            v.uv = eng::Vec4{float(k) / float(kRingVerts), along, 0, 0};
            t.mesh.vertices.push_back(v);

            eng::anim::SkinVertex sv;
            sv.joints[0] = std::uint16_t(lo);
            sv.joints[1] = std::uint16_t(hi);
            sv.weights[0] = 1.0f - frac;
            sv.weights[1] = frac;
            eng::anim::NormalizeWeights(&sv);
            t.skin.push_back(sv);
        }
    }
    for (int r = 0; r + 1 < rings; ++r) {
        for (int k = 0; k < kRingVerts; ++k) {
            const auto a = std::uint16_t(r * kRingVerts + k);
            const auto b = std::uint16_t(r * kRingVerts + (k + 1) % kRingVerts);
            const auto c = std::uint16_t((r + 1) * kRingVerts + k);
            const auto d = std::uint16_t((r + 1) * kRingVerts + (k + 1) % kRingVerts);
            t.mesh.indices.insert(t.mesh.indices.end(), {a, c, d, a, d, b});
        }
    }
    // End caps. An open tube shows its own unlit interior down the barrel,
    // which reads as a black hole at the base.
    {
        const auto first_centre = std::uint16_t(t.mesh.vertices.size());
        VertexIn c{};
        c.position = eng::Vec4{0, 0, 0, 0};
        c.normal = eng::Vec4{-1, 0, 0, 0};
        c.color = eng::Vec4{1, 1, 1, 1};
        t.mesh.vertices.push_back(c);
        t.skin.push_back(t.skin[0]);
        for (int k = 0; k < kRingVerts; ++k)
            t.mesh.indices.insert(t.mesh.indices.end(),
                                  {first_centre, std::uint16_t((k + 1) % kRingVerts),
                                   std::uint16_t(k)});

        const int last_ring = (rings - 1) * kRingVerts;
        const auto last_centre = std::uint16_t(t.mesh.vertices.size());
        VertexIn e{};
        e.position = eng::Vec4{float(rings - 1) / float(kRingsPerSegment) *
                                   kSegmentLength, 0, 0, 0};
        e.normal = eng::Vec4{1, 0, 0, 0};
        e.color = eng::Vec4{1, 1, 1, 1};
        t.mesh.vertices.push_back(e);
        t.skin.push_back(t.skin[std::size_t(last_ring)]);
        for (int k = 0; k < kRingVerts; ++k)
            t.mesh.indices.insert(
                t.mesh.indices.end(),
                {last_centre, std::uint16_t(last_ring + k),
                 std::uint16_t(last_ring + (k + 1) % kRingVerts)});
    }

    // Bounds cover the BIND pose only. A posed tentacle reaches outside them,
    // which is why the renderer must not frustum-cull a skinned mesh on them —
    // see the note in the offscreen gate.
    t.mesh.bounds.center = eng::Vec3{kSegmentLength * float(kJoints - 1) * 0.5f, 0, 0};
    t.mesh.bounds.radius = kSegmentLength * float(kJoints);

    // The clip: every joint after the root sways, each a little later than the
    // one before, which is what makes it read as a travelling wave rather than
    // the whole thing rotating at once.
    t.clip.name = "wave";
    t.clip.duration = 2.0f;
    for (int j = 1; j < kJoints; ++j) {
        eng::anim::Channel ch;
        ch.joint = j;
        ch.path = eng::anim::Path::Rotation;
        ch.interp = eng::anim::Interp::Linear;
        const float phase = float(j) * 0.55f;
        const float amplitude = 0.38f;
        for (int k = 0; k <= 8; ++k) {
            const float time = float(k) / 8.0f * t.clip.duration;
            ch.times.push_back(time);
            const float angle =
                std::sin(time / t.clip.duration * 6.2831853f - phase) * amplitude;
            const eng::Quat q = eng::QuatFromAxisAngle(eng::Vec3{0, 0, 1}, angle);
            ch.values.insert(ch.values.end(), {q.x, q.y, q.z, q.w});
        }
        t.clip.channels.push_back(std::move(ch));
    }
    return t;
}

struct Assets {
    Tentacle tentacle;
    eng::MeshHandle mesh;
    eng::MeshHandle ground;
    eng::MaterialHandle skin_mat, ground_mat;
    bool ok = false;
};

inline Assets Build(eng::Renderer& r, std::string& error) {
    Assets a;
    a.tentacle = MakeTentacle();
    a.mesh = r.UploadSkinnedMesh(a.tentacle.mesh, a.tentacle.skin, kJoints);
    a.ground = r.UploadMesh(
        eng::MakeBox(eng::Vec3{6.0f, 0.25f, 6.0f}, eng::Vec4{1, 1, 1, 1}));

    eng::MaterialDesc md;
    md.base_color = eng::Vec4{0.85f, 0.45f, 0.30f, 1.0f};
    md.roughness = 0.42f;
    a.skin_mat = r.CreateMaterial(md, error);

    md.base_color = eng::Vec4{0.52f, 0.54f, 0.57f, 1.0f};
    md.roughness = 0.85f;
    a.ground_mat = r.CreateMaterial(md, error);

    a.ok = error.empty() && Valid(a.mesh) && Valid(a.ground);
    return a;
}

// The tentacle stands upright: the chain is modelled along +x, so a quarter
// turn about -z points it at the sky.
inline eng::Mat4 TentacleModel() {
    return eng::Mat4::Translation(eng::Vec3{0.0f, 0.0f, 0.0f}) *
           QuatToMat4(eng::QuatFromAxisAngle(eng::Vec3{0, 0, 1}, 1.5707963f));
}

inline eng::Scene MakeScene(const Assets& a, float time) {
    eng::Scene s;
    s.lightDir = eng::Vec4{-0.38f, 0.80f, -0.46f, 0.0f};
    s.lightColor = eng::Vec4{4.2f, 4.1f, 3.9f, 1.0f};
    s.shadowExtent = 7.0f;

    eng::anim::Pose pose;
    a.tentacle.clip.Sample(time, a.tentacle.skeleton, &pose);
    // The palette goes into the scene, not the renderer: the scene describes
    // what to draw, and a pose is part of that description.
    eng::anim::ComputeJointMatrices(a.tentacle.skeleton, pose, &s.joint_matrices);

    eng::Instance ground;
    ground.mesh = a.ground;
    ground.material = a.ground_mat;
    ground.model = eng::Mat4::Translation(eng::Vec3{0.0f, -0.25f, 0.0f});
    s.instances.push_back(ground);

    eng::Instance tentacle;
    tentacle.mesh = a.mesh;
    tentacle.material = a.skin_mat;
    tentacle.model = TentacleModel();
    tentacle.palette = 0;  // offset into s.joint_matrices
    s.instances.push_back(tentacle);
    return s;
}

}  // namespace demo
