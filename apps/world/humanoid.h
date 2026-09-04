// A person: skeleton, skinned body, and the clips to move them. Generated.
//
// WHY GENERATED. Every other asset in this engine is computed rather than
// loaded -- the terrain, the trees, the bark, the sky -- and a character is the
// one place where that is usually given up, because a rigged humanoid is
// thought of as something an artist makes in a DCC tool and an engine imports.
// It is worth not giving up. A checked-in glTF would make the animation system
// untestable without a binary blob, and would hide the thing that actually
// matters here: a skeleton is twenty-one transforms and a walk cycle is a
// handful of sine waves, and neither becomes more correct for having been
// exported from somewhere.
//
// It is not a character an artist would ship. It is a proportioned figure with
// a plausible gait, which is exactly enough to show that the skinning palette,
// the blend space, the phase synchronisation and the foot IK are all doing
// their jobs -- and those are invisible on a flag, which is what this engine
// had before.
//
// THE RIG IS IN BIND POSE WITH EVERY LOCAL ROTATION AT IDENTITY. That makes
// inverse_bind exactly Translation(-world_position) and makes a clip's rotation
// keys mean "rotate this joint from rest", which is what a clip author would
// assume. It also means a bug in the palette shows up as the body exploding
// rather than as a subtle drift, which is a much easier failure to see.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "engine/anim/anim.h"
#include "engine/core/math.h"
#include "engine/geometry/mesh.h"

namespace world {

// Joint indices. Ordered parents-first so Skeleton::order is the identity and
// one forward pass poses the whole figure -- see Skeleton::ParentsFirst.
enum HumanJoint : int {
    kPelvis = 0, kSpine, kChest, kNeck, kHead,
    kClavicleL, kUpperArmL, kForearmL, kHandL,
    kClavicleR, kUpperArmR, kForearmR, kHandR,
    kThighL, kShinL, kFootL, kToeL,
    kThighR, kShinR, kFootR, kToeR,
    kHumanJoints
};

// Bind-pose position of each joint, in metres, in a model space with +Y up and
// +Z the direction the figure faces. The top of the skull lands at about 1.76 m,
// which is a hair under the character controller's 1.8 m capsule -- deliberate,
// because a body that pokes out of its own collider is the first thing anyone
// notices.
struct HumanBone {
    const char* name;
    int parent;
    eng::Vec3 at;
};
inline constexpr HumanBone kHumanRig[kHumanJoints] = {
    {"pelvis",     -1,          { 0.000f, 0.980f,  0.000f}},
    {"spine",      kPelvis,     { 0.000f, 1.160f,  0.000f}},
    {"chest",      kSpine,      { 0.000f, 1.350f,  0.000f}},
    {"neck",       kChest,      { 0.000f, 1.540f,  0.000f}},
    {"head",       kNeck,       { 0.000f, 1.640f,  0.000f}},
    {"clavicle.L", kChest,      { 0.045f, 1.500f,  0.000f}},
    {"upperarm.L", kClavicleL,  { 0.175f, 1.500f,  0.000f}},
    {"forearm.L",  kUpperArmL,  { 0.175f, 1.225f,  0.000f}},
    {"hand.L",     kForearmL,   { 0.175f, 0.965f,  0.000f}},
    {"clavicle.R", kChest,      {-0.045f, 1.500f,  0.000f}},
    {"upperarm.R", kClavicleR,  {-0.175f, 1.500f,  0.000f}},
    {"forearm.R",  kUpperArmR,  {-0.175f, 1.225f,  0.000f}},
    {"hand.R",     kForearmR,   {-0.175f, 0.965f,  0.000f}},
    {"thigh.L",    kPelvis,     { 0.092f, 0.930f,  0.000f}},
    {"shin.L",     kThighL,     { 0.092f, 0.510f,  0.000f}},
    {"foot.L",     kShinL,      { 0.092f, 0.090f,  0.000f}},
    {"toe.L",      kFootL,      { 0.092f, 0.042f,  0.135f}},
    {"thigh.R",    kPelvis,     {-0.092f, 0.930f,  0.000f}},
    {"shin.R",     kThighR,     {-0.092f, 0.510f,  0.000f}},
    {"foot.R",     kShinR,      {-0.092f, 0.090f,  0.000f}},
    {"toe.R",      kFootR,      {-0.092f, 0.042f,  0.135f}},
};

// Standing height, so the caller can size a collider to the body rather than
// the other way round.
inline constexpr float kHumanHeight = 1.78f;

inline eng::anim::Skeleton BuildHumanSkeleton() {
    eng::anim::Skeleton s;
    s.joints.resize(kHumanJoints);
    for (int i = 0; i < kHumanJoints; ++i) {
        const HumanBone& b = kHumanRig[i];
        eng::anim::Joint& j = s.joints[std::size_t(i)];
        j.name = b.name;
        j.parent = b.parent;
        // Local rest is the offset from the parent, with no rotation: the whole
        // rig is axis-aligned in bind pose.
        j.rest.translation = b.parent < 0 ? b.at : b.at - kHumanRig[b.parent].at;
        j.rest.rotation = eng::Quat{0.0f, 0.0f, 0.0f, 1.0f};
        j.rest.scale = eng::Vec3{1.0f, 1.0f, 1.0f};
        // Every bind rotation is identity, so the bind world matrix is a pure
        // translation and its inverse is the negated one. Written out rather
        // than inverted numerically because an exact inverse here is the
        // difference between a mesh that does not move in bind pose and one
        // that drifts by a rounding error per joint down the chain.
        j.inverse_bind = eng::Mat4::Translation(b.at * -1.0f);
    }
    s.Finalize();
    return s;
}

// --- the body -----------------------------------------------------------------

struct HumanBody {
    eng::Mesh mesh;
    std::vector<eng::anim::SkinVertex> skin;
};

namespace detail {

// One vertex, weighted to at most two joints.
inline eng::anim::SkinVertex TwoWay(int a, float wa, int b, float wb) {
    eng::anim::SkinVertex v{};
    v.joints[0] = std::uint16_t(a);
    v.weights[0] = wa;
    v.joints[1] = std::uint16_t(b < 0 ? a : b);
    v.weights[1] = b < 0 ? 0.0f : wb;
    v.joints[2] = v.joints[3] = 0;
    v.weights[2] = v.weights[3] = 0.0f;
    eng::anim::NormalizeWeights(&v);
    return v;
}

inline float Smooth(float e0, float e1, float x) {
    const float t = std::clamp((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// A tapered elliptical tube from `a` to `b`, skinned to the joint that owns it.
//
// THE WEIGHTS ARE THE WHOLE POINT of doing this by hand rather than by nearest
// bone. A limb segment belongs to the joint at its near end -- the upper arm
// moves with the shoulder -- so the ring at the far end has to reach exactly
// half way to the child joint, and the first ring of the CHILD's segment has to
// come half way back. Any other split puts the crease off the joint, and an
// elbow that creases above the elbow is the classic look of a rig nobody
// checked. Both ends are handled here so the two segments meet at 50/50 by
// construction rather than by two matching constants in different places.
inline void AppendLimb(HumanBody& body, eng::Vec3 a, eng::Vec3 b, float ra_x,
                       float ra_z, float rb_x, float rb_z, int joint, int parent,
                       int child, eng::Vec4 color, int slices = 12, int rings = 6) {
    const eng::Vec3 axis = b - a;
    const float len = eng::Length(axis);
    if (len < 1e-5f) return;
    const eng::Vec3 dir = axis * (1.0f / len);
    // Any two vectors perpendicular to the bone. Chosen against whichever world
    // axis the bone is least aligned with, so a vertical bone (which every one
    // of these nearly is) does not degenerate.
    const eng::Vec3 guide = std::fabs(dir.y) > 0.9f ? eng::Vec3{0, 0, 1} : eng::Vec3{0, 1, 0};
    const eng::Vec3 u = eng::Normalize(eng::Cross(guide, dir));
    const eng::Vec3 v = eng::Cross(dir, u);

    const std::uint32_t base = std::uint32_t(body.mesh.vertices.size());
    for (int r = 0; r <= rings; ++r) {
        const float t = float(r) / float(rings);
        const eng::Vec3 centre = a + axis * t;
        const float rx = ra_x + (rb_x - ra_x) * t, rz = ra_z + (rb_z - ra_z) * t;

        // Half a unit of weight handed to the neighbour at each end.
        const float to_parent = parent < 0 ? 0.0f : 0.5f * (1.0f - Smooth(0.0f, 0.35f, t));
        const float to_child = child < 0 ? 0.0f : 0.5f * Smooth(0.65f, 1.0f, t);
        for (int s = 0; s < slices; ++s) {
            const float th = float(s) / float(slices) * 6.28318530718f;
            const eng::Vec3 radial = u * (std::cos(th) * rx) + v * (std::sin(th) * rz);
            ::VertexIn vert{};
            const eng::Vec3 p = centre + radial;
            vert.position = {p.x, p.y, p.z};
            // The surface normal of a TAPERED tube is not the radial direction:
            // the wall slopes, and using the radial makes a thigh light like a
            // cylinder while its silhouette is a cone. The correction is the
            // taper rate along the bone.
            const float taper = (rb_x + rb_z - ra_x - ra_z) * 0.5f / len;
            const eng::Vec3 n = eng::Normalize(eng::Normalize(radial) - dir * taper);
            vert.normal = {n.x, n.y, n.z};
            vert.color = {color.x, color.y, color.z, color.w};
            vert.uv = {float(s) / float(slices) * 2.0f, t * 1.5f, 0.0f, 0.0f};
            body.mesh.vertices.push_back(vert);
            body.skin.push_back(TwoWay(joint, 1.0f - to_parent - to_child,
                                       to_child > to_parent ? child : parent,
                                       std::max(to_child, to_parent)));
        }
    }
    for (int r = 0; r < rings; ++r)
        for (int s = 0; s < slices; ++s) {
            const std::uint32_t i0 = base + std::uint32_t(r * slices + s);
            const std::uint32_t i1 = base + std::uint32_t(r * slices + (s + 1) % slices);
            const std::uint32_t i2 = i0 + std::uint32_t(slices), i3 = i1 + std::uint32_t(slices);
            body.mesh.indices.insert(body.mesh.indices.end(), {i0, i2, i1, i1, i2, i3});
        }
}

// A rigid blob -- skull, hand -- entirely owned by one joint.
inline void AppendBlob(HumanBody& body, eng::Vec3 centre, eng::Vec3 radii, int joint,
                       eng::Vec4 color, int stacks = 10, int slices = 14) {
    const std::uint32_t base = std::uint32_t(body.mesh.vertices.size());
    for (int y = 0; y <= stacks; ++y) {
        const float phi = float(y) / float(stacks) * 3.14159265358979f;
        for (int s = 0; s <= slices; ++s) {
            const float th = float(s) / float(slices) * 6.28318530718f;
            const eng::Vec3 unit{std::sin(phi) * std::cos(th), std::cos(phi),
                                 std::sin(phi) * std::sin(th)};
            const eng::Vec3 p = centre + eng::Vec3{unit.x * radii.x, unit.y * radii.y,
                                                   unit.z * radii.z};
            ::VertexIn vert{};
            vert.position = {p.x, p.y, p.z};
            // An ellipsoid's normal is the unit vector divided by the radii,
            // not the unit vector: the two agree only on a sphere.
            const eng::Vec3 n = eng::Normalize(
                eng::Vec3{unit.x / radii.x, unit.y / radii.y, unit.z / radii.z});
            vert.normal = {n.x, n.y, n.z};
            vert.color = {color.x, color.y, color.z, color.w};
            vert.uv = {float(s) / float(slices), float(y) / float(stacks), 0.0f, 0.0f};
            body.mesh.vertices.push_back(vert);
            body.skin.push_back(TwoWay(joint, 1.0f, -1, 0.0f));
        }
    }
    for (int y = 0; y < stacks; ++y)
        for (int s = 0; s < slices; ++s) {
            const std::uint32_t i0 = base + std::uint32_t(y * (slices + 1) + s);
            const std::uint32_t i1 = i0 + 1;
            const std::uint32_t i2 = i0 + std::uint32_t(slices + 1), i3 = i2 + 1;
            // THE POLE ROWS ARE FANS, not quads. Every vertex of the top ring
            // is the same point, so one triangle of each quad there has exactly
            // zero area. A zero-area triangle is not harmless: it has no normal,
            // so tangent generation divides by zero and poisons the tangents of
            // every vertex it touches, and the rasteriser still pays for it.
            if (y > 0) body.mesh.indices.insert(body.mesh.indices.end(), {i0, i2, i1});
            if (y + 1 < stacks) body.mesh.indices.insert(body.mesh.indices.end(), {i1, i2, i3});
        }
}

}  // namespace detail

// The figure: cloth-coloured body, darker legs, skin at the head and hands.
inline HumanBody BuildHumanBody(const eng::anim::Skeleton& s) {
    using detail::AppendBlob;
    using detail::AppendLimb;
    HumanBody b;
    const auto at = [&](int j) { return kHumanRig[j].at; };
    const eng::Vec4 coat{0.32f, 0.38f, 0.46f, 1.0f};
    const eng::Vec4 trouser{0.20f, 0.22f, 0.27f, 1.0f};
    const eng::Vec4 skin{0.78f, 0.62f, 0.52f, 1.0f};

    // Torso. Elliptical rather than round -- a person is wider than they are
    // deep, and a cylindrical torso reads as a robot from any angle but front.
    AppendLimb(b, at(kPelvis), at(kSpine), 0.135f, 0.098f, 0.128f, 0.092f, kPelvis, -1,
               kSpine, trouser, 14);
    AppendLimb(b, at(kSpine), at(kChest), 0.128f, 0.092f, 0.152f, 0.105f, kSpine, kPelvis,
               kChest, coat, 14);
    AppendLimb(b, at(kChest), at(kNeck), 0.152f, 0.105f, 0.062f, 0.058f, kChest, kSpine,
               kNeck, coat, 14);
    AppendLimb(b, at(kNeck), at(kHead), 0.058f, 0.055f, 0.056f, 0.053f, kNeck, kChest,
               kHead, skin, 10, 2);
    AppendBlob(b, at(kHead) + eng::Vec3{0.0f, 0.040f, 0.008f},
               eng::Vec3{0.088f, 0.100f, 0.098f}, kHead, skin);

    for (int side = 0; side < 2; ++side) {
        const int clav = side ? kClavicleR : kClavicleL;
        const int arm = side ? kUpperArmR : kUpperArmL;
        const int fore = side ? kForearmR : kForearmL;
        const int hand = side ? kHandR : kHandL;
        const int thigh = side ? kThighR : kThighL;
        const int shin = side ? kShinR : kShinL;
        const int foot = side ? kFootR : kFootL;
        const int toe = side ? kToeR : kToeL;

        AppendLimb(b, at(clav), at(arm), 0.062f, 0.062f, 0.052f, 0.052f, clav, kChest, arm,
                   coat, 10, 2);
        AppendLimb(b, at(arm), at(fore), 0.055f, 0.055f, 0.044f, 0.044f, arm, clav, fore, coat);
        AppendLimb(b, at(fore), at(hand), 0.044f, 0.044f, 0.033f, 0.033f, fore, arm, hand, coat);
        AppendBlob(b, at(hand) + eng::Vec3{0.0f, -0.035f, 0.0f},
                   eng::Vec3{0.034f, 0.052f, 0.026f}, hand, skin, 6, 8);

        AppendLimb(b, at(thigh), at(shin), 0.092f, 0.092f, 0.062f, 0.062f, thigh, kPelvis,
                   shin, trouser);
        AppendLimb(b, at(shin), at(foot), 0.062f, 0.062f, 0.042f, 0.045f, shin, thigh, foot,
                   trouser);
        // The foot runs FORWARD from the ankle, not down: it is the one bone in
        // the rig that is not roughly vertical, and getting it wrong puts the
        // sole where the shin is.
        AppendLimb(b, at(foot) + eng::Vec3{0.0f, -0.045f, -0.035f},
                   at(toe) + eng::Vec3{0.0f, 0.0f, 0.030f}, 0.045f, 0.048f, 0.032f, 0.042f,
                   foot, shin, toe, trouser, 8, 3);
    }

    eng::GenerateTangents(b.mesh);
    // A bounding SPHERE, which is what the renderer culls against. Centred on
    // the middle of the figure's own extent rather than on the origin: a person
    // standing at the origin has their feet there, so an origin-centred sphere
    // has to be twice as big to hold them and the character is drawn from
    // behind every hill in the world.
    eng::Vec3 lo = b.mesh.vertices.empty()
                       ? eng::Vec3{}
                       : eng::Vec3{b.mesh.vertices[0].position.x, b.mesh.vertices[0].position.y,
                                   b.mesh.vertices[0].position.z};
    eng::Vec3 hi = lo;
    for (const ::VertexIn& v : b.mesh.vertices) {
        lo = eng::Vec3{std::min(lo.x, v.position.x), std::min(lo.y, v.position.y),
                       std::min(lo.z, v.position.z)};
        hi = eng::Vec3{std::max(hi.x, v.position.x), std::max(hi.y, v.position.y),
                       std::max(hi.z, v.position.z)};
    }
    b.mesh.bounds.center = (lo + hi) * 0.5f;
    float r = 0.0f;
    for (const ::VertexIn& v : b.mesh.vertices)
        r = std::max(r, eng::Length(eng::Vec3{v.position.x, v.position.y, v.position.z} -
                                    b.mesh.bounds.center));
    // ANIMATION MOVES VERTICES OUTSIDE THE BIND POSE. A run's arm swing and the
    // pelvis bob both reach past the T-pose extent, and a sphere sized to bind
    // pose pops the character out of existence at the edge of the screen on
    // exactly the frames where they are moving most. A fifth is generous for
    // the gaits here and costs a few pixels of overdraw.
    b.mesh.bounds.radius = r * 1.2f;
    (void)s;
    return b;
}

// --- the clips ----------------------------------------------------------------

namespace detail {

inline void PushRot(eng::anim::Clip& c, int joint, const std::vector<float>& times,
                    const std::vector<eng::Quat>& q) {
    eng::anim::Channel ch;
    ch.joint = joint;
    ch.path = eng::anim::Path::Rotation;
    ch.times = times;
    ch.values.reserve(q.size() * 4);
    for (const eng::Quat& r : q) {
        ch.values.push_back(r.x);
        ch.values.push_back(r.y);
        ch.values.push_back(r.z);
        ch.values.push_back(r.w);
    }
    c.channels.push_back(std::move(ch));
}

inline void PushTrans(eng::anim::Clip& c, int joint, const std::vector<float>& times,
                      const std::vector<eng::Vec3>& p) {
    eng::anim::Channel ch;
    ch.joint = joint;
    ch.path = eng::anim::Path::Translation;
    ch.times = times;
    ch.values.reserve(p.size() * 3);
    for (const eng::Vec3& v : p) {
        ch.values.push_back(v.x);
        ch.values.push_back(v.y);
        ch.values.push_back(v.z);
    }
    c.channels.push_back(std::move(ch));
}

inline eng::Quat Pitch(float deg) {
    return eng::QuatFromAxisAngle({1, 0, 0}, deg * 3.14159265358979f / 180.0f);
}
inline eng::Quat Yaw(float deg) {
    return eng::QuatFromAxisAngle({0, 1, 0}, deg * 3.14159265358979f / 180.0f);
}
inline eng::Quat Roll(float deg) {
    return eng::QuatFromAxisAngle({0, 0, 1}, deg * 3.14159265358979f / 180.0f);
}

}  // namespace detail

// How a gait is shaped. One struct rather than eight arguments, because walk
// and run differ only in these numbers and writing them side by side is how you
// can see that a run is not just a fast walk.
struct GaitDesc {
    float duration = 1.0f;    // seconds for a full two-step cycle
    float thigh = 22.0f;      // hip swing amplitude, degrees
    float knee = 48.0f;       // peak knee bend during swing
    float arm = 22.0f;        // shoulder swing, opposite the leg on that side
    float elbow = 14.0f;      // resting elbow bend, which grows with the arm swing
    float bob = 0.022f;       // pelvis rise at mid-stance, metres
    float sway = 3.0f;        // pelvis roll toward the stance leg, degrees
    float twist = 5.0f;       // pelvis yaw with the leading leg; chest counters it
    float lean = 3.0f;        // constant forward pitch of the whole spine
};

// THE SPEED EACH GAIT ACTUALLY TRAVELS AT, in m/s, which is the position it
// must occupy on the blend space's axis.
//
// Not a preference and not a tuning value: a clip with no root motion holds the
// body still and drives the stance foot backwards underneath it, and the speed
// the foot goes backwards is the only speed at which the character can be moved
// forward for that foot to look planted. Put the sample at any other position
// and the feet skate -- the legs cycling at one speed while the body travels at
// another -- which is then usually blamed on the blend rather than on the axis.
//
// Derived from the clips themselves by tests/humanoid, which measures the
// stance travel and fails if these two numbers have drifted from it. They are
// declared here rather than computed at startup because the character
// controller's speeds have to match them, and a constant both sides can read is
// what makes that a fact rather than a coincidence.
inline constexpr float kWalkSpeed = 1.35f;
inline constexpr float kRunSpeed = 3.18f;

inline constexpr GaitDesc kWalkGait{};
inline constexpr GaitDesc kRunGait{0.62f, 38.0f, 88.0f, 44.0f, 34.0f,
                                   0.055f, 4.5f, 9.0f, 11.0f};

// A locomotion cycle, sampled from continuous curves into keys.
//
// SAMPLED RATHER THAN HAND-KEYED, at a rate fine enough that linear
// interpolation between keys is indistinguishable from the curve. That is the
// honest way round: the gait is defined by the formula, and the keys are a
// representation of it, so the clip cannot disagree with the intent. Hand
// keying four poses and letting the sampler interpolate gives a cycle whose
// feet slide, and the slide is then blamed on the blend space.
inline eng::anim::Clip MakeGait(std::string name, const GaitDesc& g) {
    constexpr int kKeys = 33;  // 32 intervals, so phase 0 and 1 coincide exactly
    eng::anim::Clip c;
    c.name = std::move(name);
    c.duration = g.duration;

    std::vector<float> t(kKeys);
    for (int k = 0; k < kKeys; ++k) t[std::size_t(k)] = float(k) / float(kKeys - 1) * g.duration;
    const float kTau = 6.28318530718f;

    std::vector<eng::Quat> pelvis_r(kKeys), chest_r(kKeys), spine_r(kKeys);
    std::vector<eng::Vec3> pelvis_t(kKeys);
    for (int k = 0; k < kKeys; ++k) {
        const float p = float(k) / float(kKeys - 1);
        // The pelvis rises TWICE per cycle -- once over each stance leg -- and
        // is at its lowest at the two double-support moments. A single rise per
        // cycle is the classic tell of a walk built without watching one.
        pelvis_t[std::size_t(k)] =
            kHumanRig[kPelvis].at + eng::Vec3{0.0f, -g.bob * std::cos(2.0f * kTau * p), 0.0f};
        pelvis_r[std::size_t(k)] = detail::Pitch(g.lean) *
                                   detail::Yaw(g.twist * std::cos(kTau * p)) *
                                   detail::Roll(g.sway * std::sin(kTau * p));
        // The shoulders counter-rotate against the hips. This is what stops a
        // walk reading as a shuffle: without it the whole torso swings as one
        // block and there is no counter-rotation to drive the arms.
        chest_r[std::size_t(k)] = detail::Yaw(-1.6f * g.twist * std::cos(kTau * p));
        spine_r[std::size_t(k)] = detail::Pitch(g.lean * 0.5f);
    }
    detail::PushTrans(c, kPelvis, t, pelvis_t);
    detail::PushRot(c, kPelvis, t, pelvis_r);
    detail::PushRot(c, kSpine, t, spine_r);
    detail::PushRot(c, kChest, t, chest_r);

    for (int side = 0; side < 2; ++side) {
        const float offset = side ? 0.5f : 0.0f;  // the right leg is half a cycle behind
        std::vector<eng::Quat> thigh(kKeys), shin(kKeys), foot(kKeys), arm(kKeys), fore(kKeys);
        for (int k = 0; k < kKeys; ++k) {
            const float p = float(k) / float(kKeys - 1) + offset;
            const float hip = g.thigh * std::cos(kTau * p);
            // The knee bends during SWING and is nearly straight at heel
            // strike. Peak at three quarters of the cycle, which is mid-swing,
            // plus a small constant bend so the leg is never locked -- a locked
            // knee is both wrong and the thing that makes IK pop.
            const float bend = -g.knee * (0.5f + 0.5f * std::cos(kTau * (p - 0.75f))) -
                               g.knee * 0.10f;
            thigh[std::size_t(k)] = detail::Pitch(hip);
            shin[std::size_t(k)] = detail::Pitch(bend);
            // The ankle mostly undoes the two above it, so the sole stays near
            // level through stance; foot IK adjusts the rest against the actual
            // ground, and giving it a foot that is already close is what keeps
            // its correction small enough not to be seen.
            foot[std::size_t(k)] = detail::Pitch(-(hip + bend) * 0.45f);
            // Arms swing against the leg on the SAME side.
            const float swing = -g.arm * std::cos(kTau * p);
            arm[std::size_t(k)] = detail::Pitch(swing);
            fore[std::size_t(k)] = detail::Pitch(-g.elbow - 0.35f * std::fabs(swing));
        }
        detail::PushRot(c, side ? kThighR : kThighL, t, thigh);
        detail::PushRot(c, side ? kShinR : kShinL, t, shin);
        detail::PushRot(c, side ? kFootR : kFootL, t, foot);
        detail::PushRot(c, side ? kUpperArmR : kUpperArmL, t, arm);
        detail::PushRot(c, side ? kForearmR : kForearmL, t, fore);
    }
    return c;
}

// Standing. NOT a single held pose: a character frozen between footsteps is the
// most obviously dead thing on screen, and the fix is small -- breathing, a
// slow weight shift, and arms that hang rather than stick out.
inline eng::anim::Clip MakeIdle() {
    constexpr int kKeys = 25;
    eng::anim::Clip c;
    c.name = "idle";
    c.duration = 4.6f;  // deliberately not a round number against the gait cycles
    std::vector<float> t(kKeys);
    for (int k = 0; k < kKeys; ++k) t[std::size_t(k)] = float(k) / float(kKeys - 1) * c.duration;
    const float kTau = 6.28318530718f;

    std::vector<eng::Quat> chest(kKeys), pelvis(kKeys), arm_l(kKeys), arm_r(kKeys),
        fore_l(kKeys), fore_r(kKeys), head(kKeys);
    std::vector<eng::Vec3> pelvis_t(kKeys);
    for (int k = 0; k < kKeys; ++k) {
        const float p = float(k) / float(kKeys - 1);
        // Breathing is twice the sway period, so the two never line up and the
        // loop does not read as a loop.
        const float breath = std::sin(2.0f * kTau * p);
        pelvis_t[std::size_t(k)] =
            kHumanRig[kPelvis].at + eng::Vec3{0.0f, 0.006f * breath, 0.0f};
        pelvis[std::size_t(k)] = detail::Roll(1.8f * std::sin(kTau * p));
        chest[std::size_t(k)] = detail::Pitch(-1.4f * breath) *
                                detail::Yaw(-1.2f * std::sin(kTau * p));
        head[std::size_t(k)] = detail::Yaw(2.5f * std::sin(kTau * p + 1.1f));
        // A few degrees out from the body, or the arms intersect the hips.
        arm_l[std::size_t(k)] = detail::Roll(-4.0f) * detail::Pitch(1.5f * breath);
        arm_r[std::size_t(k)] = detail::Roll(4.0f) * detail::Pitch(1.5f * breath);
        fore_l[std::size_t(k)] = detail::Pitch(-9.0f);
        fore_r[std::size_t(k)] = detail::Pitch(-9.0f);
    }
    detail::PushTrans(c, kPelvis, t, pelvis_t);
    detail::PushRot(c, kPelvis, t, pelvis);
    detail::PushRot(c, kChest, t, chest);
    detail::PushRot(c, kHead, t, head);
    detail::PushRot(c, kUpperArmL, t, arm_l);
    detail::PushRot(c, kUpperArmR, t, arm_r);
    detail::PushRot(c, kForearmL, t, fore_l);
    detail::PushRot(c, kForearmR, t, fore_r);
    return c;
}

// HOW MUCH A FOOT SHOULD BE PINNED TO THE GROUND, from how far the animation
// has already lifted it.
//
// Foot IK that runs at full weight on both feet every frame does not work, and
// the way it fails is instructive: at the extreme of a stride the swing leg is
// nearly straight and its foot is at its farthest forward, so dragging that
// foot down to the ground asks for more leg than there is. The solver stops
// short -- correctly, it will not stretch a bone -- and the result is a pair of
// legs that reach for the floor on every step and never get there. Measured, it
// was 118 mm off; on screen it read as the legs having collapsed, which is a
// different bug and would have been looked for in a different place.
//
// A planted foot is one the animation has already put on the ground, so the
// gate is the foot's own clearance and nothing has to be authored per clip: a
// contact flag on the clip would be another thing to keep in sync with the
// curves that already say it.
inline float FootPlantWeight(float clearance_metres) {
    return 1.0f - detail::Smooth(0.02f, 0.12f, clearance_metres);
}

// Where the sole sits, in the ankle's own space. The foot mesh is a tube of
// radius 0.045 whose centre starts 45 mm below the ankle, so its underside is
// exactly this far down; the IK needs it because placing the ANKLE on the
// ground buries the foot by its own height.
inline constexpr eng::Vec3 kAnkleToSole{0.0f, -0.09f, 0.0f};

}  // namespace world
