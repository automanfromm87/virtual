// No test framework — from scratch means from scratch.
//
// Skinning fails by looking ALMOST right. A transposed matrix, a forgotten
// inverse bind, a lerp where a slerp belongs: every one of them produces a mesh
// that still moves, still follows the skeleton, and is wrong in a way nobody
// catches from a screenshot. So the checks here are exact positions of known
// vertices under known poses, not "the frame changed".
#include "engine/anim/anim.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace {

int g_failures = 0;

void Fail(const char* what, int line) {
    std::fprintf(stderr, "anim_test.cc:%d  %s\n", line, what);
    ++g_failures;
}

#define CHECK(cond) \
    do { if (!(cond)) Fail(#cond, __LINE__); } while (0)

using namespace eng;
using namespace eng::anim;

bool Near(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }
bool NearV(Vec3 a, Vec3 b, float eps = 1e-4f) {
    return Near(a.x, b.x, eps) && Near(a.y, b.y, eps) && Near(a.z, b.z, eps);
}

// Two joints in a line along +x: root at the origin, child one metre out.
// Bind pose is the rest pose, so every inverse bind is the inverse of the
// joint's world matrix at rest.
Skeleton TwoBoneArm() {
    Skeleton s;
    Joint root;
    root.name = "root";
    root.parent = -1;
    root.rest.translation = Vec3{0, 0, 0};
    root.inverse_bind = Mat4::Identity();
    s.joints.push_back(root);

    Joint elbow;
    elbow.name = "elbow";
    elbow.parent = 0;
    elbow.rest.translation = Vec3{1, 0, 0};  // local, so world is also (1,0,0)
    elbow.inverse_bind = Mat4::Translation(Vec3{-1, 0, 0});
    s.joints.push_back(elbow);
    return s;
}

}  // namespace

int main() {
    // --- the rest pose must be the identity transform --------------------------
    {
        // The single most important property. If posing a skeleton at rest moves
        // anything, every animation is wrong by that offset and it looks like a
        // modelling problem.
        const Skeleton s = TwoBoneArm();
        CHECK(s.ParentsFirst());

        Pose pose;
        Clip empty;  // no channels at all
        empty.Sample(0.0f, s, &pose);

        std::vector<Mat4> palette;
        ComputeJointMatrices(s, pose, &palette);
        CHECK(palette.size() == 2);

        SkinVertex v;
        v.joints[0] = 1;
        v.weights[0] = 1.0f;
        const Vec3 p{1.5f, 0.2f, 0.0f};
        CHECK(NearV(SkinPosition(p, v, palette), p));

        // ...and so must a vertex bound to the root.
        SkinVertex r;
        r.joints[0] = 0;
        r.weights[0] = 1.0f;
        CHECK(NearV(SkinPosition(p, r, palette), p));
    }

    // --- the inverse bind is what stops the mesh flying away --------------------
    {
        // Deleting it leaves palette[j] = world(j), which for the elbow is a
        // translation of +1: every vertex it owns jumps a metre down the arm.
        const Skeleton s = TwoBoneArm();
        Pose pose;
        Clip{}.Sample(0.0f, s, &pose);
        std::vector<Mat4> world, palette;
        ComputeJointWorld(s, pose, &world);
        ComputeJointMatrices(s, pose, &palette);

        const Vec4 wo = world[1] * Vec4{0, 0, 0, 1};
        CHECK(Near(wo.x, 1.0f));  // the joint really is a metre out
        const Vec4 po = palette[1] * Vec4{0, 0, 0, 1};
        CHECK(Near(po.x, 0.0f));  // but the palette entry is the identity
    }

    // --- bending the elbow moves only what the elbow owns -----------------------
    {
        Skeleton s = TwoBoneArm();
        Pose pose;
        Clip{}.Sample(0.0f, s, &pose);
        // A quarter turn about +z at the elbow: +x becomes +y.
        pose.local[1].rotation = QuatFromAxisAngle(Vec3{0, 0, 1}, 1.5707963f);

        std::vector<Mat4> palette;
        ComputeJointMatrices(s, pose, &palette);

        // A vertex half a metre past the elbow, fully weighted to it, must swing
        // to (1, 0.5, 0): the joint stays put and the offset rotates.
        SkinVertex e;
        e.joints[0] = 1;
        e.weights[0] = 1.0f;
        CHECK(NearV(SkinPosition(Vec3{1.5f, 0, 0}, e, palette), Vec3{1.0f, 0.5f, 0}));

        // A vertex owned by the root does not move at all.
        SkinVertex r;
        r.joints[0] = 0;
        r.weights[0] = 1.0f;
        CHECK(NearV(SkinPosition(Vec3{0.5f, 0, 0}, r, palette), Vec3{0.5f, 0, 0}));

        // Half and half lands exactly between the two answers, which is what
        // blending IS. A vertex bound 50/50 at the joint is the case that
        // exposes a palette built in the wrong order.
        SkinVertex mid;
        mid.joints[0] = 0;
        mid.joints[1] = 1;
        mid.weights[0] = 0.5f;
        mid.weights[1] = 0.5f;
        const Vec3 got = SkinPosition(Vec3{1.5f, 0, 0}, mid, palette);
        CHECK(NearV(got, Vec3{(1.5f + 1.0f) * 0.5f, 0.25f, 0.0f}));
    }

    // --- a parent's motion carries its children ---------------------------------
    {
        Skeleton s = TwoBoneArm();
        Pose pose;
        Clip{}.Sample(0.0f, s, &pose);
        pose.local[0].rotation = QuatFromAxisAngle(Vec3{0, 0, 1}, 1.5707963f);

        std::vector<Mat4> world;
        ComputeJointWorld(s, pose, &world);
        // Rotating the root swings the elbow from (1,0,0) to (0,1,0).
        const Vec4 e = world[1] * Vec4{0, 0, 0, 1};
        CHECK(Near(e.x, 0.0f) && Near(e.y, 1.0f));
    }

    // --- clips: keys, holds and looping -----------------------------------------
    {
        Skeleton s = TwoBoneArm();
        Clip c;
        c.duration = 2.0f;
        Channel ch;
        ch.joint = 1;
        ch.path = Path::Translation;
        ch.interp = Interp::Linear;
        ch.times = {0.0f, 1.0f, 2.0f};
        ch.values = {1, 0, 0, /**/ 1, 4, 0, /**/ 1, 0, 0};
        c.channels.push_back(ch);
        CHECK(c.channels[0].Valid(s.joints.size()));

        Pose pose;
        c.Sample(0.0f, s, &pose);
        CHECK(Near(pose.local[1].translation.y, 0.0f));
        c.Sample(0.5f, s, &pose);
        CHECK(Near(pose.local[1].translation.y, 2.0f));  // halfway up
        c.Sample(1.0f, s, &pose);
        CHECK(Near(pose.local[1].translation.y, 4.0f));
        c.Sample(1.5f, s, &pose);
        CHECK(Near(pose.local[1].translation.y, 2.0f));  // back down

        // Looping wraps rather than clamping.
        c.Sample(2.5f, s, &pose, /*loop=*/true);
        CHECK(Near(pose.local[1].translation.y, 2.0f));
        c.Sample(-0.5f, s, &pose, /*loop=*/true);
        CHECK(Near(pose.local[1].translation.y, 2.0f));  // wraps to 1.5

        // Not looping clamps to the ends rather than extrapolating.
        c.Sample(99.0f, s, &pose, /*loop=*/false);
        CHECK(Near(pose.local[1].translation.y, 0.0f));

        // A channel touching ONE joint leaves the others at rest. Starting from
        // identity instead would collapse the untouched half of a skeleton.
        CHECK(NearV(pose.local[0].translation, Vec3{0, 0, 0}));
        CHECK(Near(pose.local[1].translation.x, 1.0f));  // x came from the key
    }

    // --- STEP holds, it does not ramp -------------------------------------------
    {
        Skeleton s = TwoBoneArm();
        Clip c;
        c.duration = 2.0f;
        Channel ch;
        ch.joint = 1;
        ch.path = Path::Translation;
        ch.interp = Interp::Step;
        ch.times = {0.0f, 1.0f};
        ch.values = {0, 0, 0, /**/ 0, 10, 0};
        c.channels.push_back(ch);

        Pose pose;
        c.Sample(0.99f, s, &pose);
        CHECK(Near(pose.local[1].translation.y, 0.0f));  // still the first key
        c.Sample(1.0f, s, &pose);
        CHECK(Near(pose.local[1].translation.y, 10.0f));
    }

    // --- rotations are SLERPed, not lerped ---------------------------------------
    {
        // 170 degrees apart, sampled at the midpoint. A lerp of the components
        // gives a shorter, non-unit quaternion, so the joint would rotate to
        // the right angle but scale as it went.
        Skeleton s = TwoBoneArm();
        Clip c;
        c.duration = 1.0f;
        Channel ch;
        ch.joint = 1;
        ch.path = Path::Rotation;
        ch.times = {0.0f, 1.0f};
        const Quat a = QuatFromAxisAngle(Vec3{0, 0, 1}, 0.0f);
        const Quat b = QuatFromAxisAngle(Vec3{0, 0, 1}, 2.9670597f);  // 170 deg
        ch.values = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        c.channels.push_back(ch);

        Pose pose;
        // Sampled at a QUARTER, not the midpoint. A normalised component lerp
        // and a slerp agree exactly at t = 0.5 — the chord's midpoint projects
        // onto the arc's midpoint — so testing there cannot tell them apart at
        // all. At a quarter of the way they differ by nearly seven degrees.
        c.Sample(0.25f, s, &pose);
        const Quat m = pose.local[1].rotation;
        CHECK(Near(m.x * m.x + m.y * m.y + m.z * m.z + m.w * m.w, 1.0f));
        const float angle = 2.0f * std::acos(std::fabs(m.w));
        CHECK(Near(angle, 2.9670597f * 0.25f, 1e-3f));  // 42.5 deg; nlerp gives 35.8

        // The midpoint still has to be right, it just proves less.
        c.Sample(0.5f, s, &pose);
        const Quat h = pose.local[1].rotation;
        CHECK(Near(2.0f * std::acos(std::fabs(h.w)), 2.9670597f * 0.5f, 1e-3f));
    }

    // --- CUBICSPLINE reads tangents, and passes through its keys ------------------
    {
        Skeleton s = TwoBoneArm();
        Clip c;
        c.duration = 1.0f;
        Channel ch;
        ch.joint = 1;
        ch.path = Path::Translation;
        ch.interp = Interp::CubicSpline;
        ch.times = {0.0f, 1.0f};
        // Per key: in-tangent, value, out-tangent. Both values are zero and
        // the tangents arc the curve up and back — leaving key 0 rising, and
        // ARRIVING at key 1 falling. Giving key 1 a rising in-tangent instead
        // makes the two cancel exactly at the midpoint, which reads as "the
        // spline did nothing" and is a fixture bug rather than a code one.
        ch.values = {
            0,  0, 0, /*v*/ 0, 0, 0, /*out*/ 0, 6, 0,   // key 0
            0, -6, 0, /*v*/ 0, 0, 0, /*out*/ 0, 0, 0,   // key 1
        };
        c.channels.push_back(ch);
        CHECK(c.channels[0].Valid(s.joints.size()));

        Pose pose;
        // A spline must still hit its keys exactly, whatever the tangents do.
        c.Sample(0.0f, s, &pose);
        CHECK(Near(pose.local[1].translation.y, 0.0f));
        c.Sample(1.0f, s, &pose, /*loop=*/false);
        CHECK(Near(pose.local[1].translation.y, 0.0f));
        // ...and it must bulge in between, or the tangents were ignored and
        // this is just a linear interpolation wearing a different name.
        c.Sample(0.5f, s, &pose);
        CHECK(Near(pose.local[1].translation.y, 1.5f));
        // Symmetric tangents give a symmetric arc. A sign error in either
        // tangent term breaks this while leaving the midpoint plausible.
        Pose q;
        c.Sample(0.25f, s, &pose);
        c.Sample(0.75f, s, &q);
        CHECK(Near(pose.local[1].translation.y, q.local[1].translation.y));
        CHECK(pose.local[1].translation.y > 0.0f);
    }

    // --- weights are normalised, and bad input does not read memory ---------------
    {
        SkinVertex v;
        v.joints[0] = 0;
        v.joints[1] = 1;
        v.weights[0] = 0.6f;
        v.weights[1] = 0.6f;  // sums to 1.2: an exporter rounding badly
        NormalizeWeights(&v);
        CHECK(Near(v.weights[0] + v.weights[1], 1.0f));

        SkinVertex zero;  // no influences at all
        NormalizeWeights(&zero);
        CHECK(Near(zero.weights[0], 1.0f));

        // An out-of-range joint index must be skipped, not indexed.
        const Skeleton s = TwoBoneArm();
        Pose pose;
        Clip{}.Sample(0.0f, s, &pose);
        std::vector<Mat4> palette;
        ComputeJointMatrices(s, pose, &palette);
        SkinVertex bad;
        bad.joints[0] = 9999;
        bad.weights[0] = 1.0f;
        const Vec3 p{2, 3, 4};
        CHECK(NearV(SkinPosition(p, bad, palette), p));  // left where it was
    }

    // --- normals rotate but do not translate --------------------------------------
    {
        Skeleton s = TwoBoneArm();
        Pose pose;
        Clip{}.Sample(0.0f, s, &pose);
        pose.local[1].rotation = QuatFromAxisAngle(Vec3{0, 0, 1}, 1.5707963f);
        std::vector<Mat4> palette;
        ComputeJointMatrices(s, pose, &palette);

        SkinVertex e;
        e.joints[0] = 1;
        e.weights[0] = 1.0f;
        // +x turns into +y. If the translation leaked in, this would be a
        // vector of length 1.4 pointing somewhere off-axis.
        const Vec3 n = SkinNormal(Vec3{1, 0, 0}, e, palette);
        CHECK(NearV(n, Vec3{0, 1, 0}));
        CHECK(Near(Length(n), 1.0f));
    }

    // --- a skeleton whose parents come later is rejected ---------------------------
    {
        Skeleton bad;
        Joint a;
        a.parent = 1;  // forward reference
        Joint b;
        b.parent = -1;
        bad.joints = {a, b};
        CHECK(!bad.ParentsFirst());

        Skeleton loop;
        Joint self;
        self.parent = 0;  // its own parent
        loop.joints = {self};
        CHECK(!loop.ParentsFirst());

        CHECK(TwoBoneArm().ParentsFirst());
        CHECK(TwoBoneArm().Find("elbow") == 1);
        CHECK(TwoBoneArm().Find("nope") == -1);
    }

    // --- a malformed channel is skipped, not applied --------------------------------
    {
        Skeleton s = TwoBoneArm();
        Clip c;
        c.duration = 1.0f;
        Channel ch;
        ch.joint = 1;
        ch.path = Path::Translation;
        ch.times = {0.0f, 1.0f};
        ch.values = {1, 2, 3};  // one key's worth of data for two keys
        c.channels.push_back(ch);
        CHECK(!c.channels[0].Valid(s.joints.size()));

        Pose pose;
        c.Sample(0.5f, s, &pose);  // must not read past the end
        CHECK(NearV(pose.local[1].translation, Vec3{1, 0, 0}));  // still at rest

        Channel oob;
        oob.joint = 50;
        oob.times = {0.0f};
        oob.values = {0, 0, 0};
        CHECK(!oob.Valid(s.joints.size()));
    }

    // --- a skeleton stored children-first still evaluates correctly --------------
    {
        // glTF does not require a skin's joint array to be topologically
        // sorted, and JOINTS_0 indexes into it as written — so the joints
        // cannot be reordered without rewriting every vertex. Finalize()
        // reorders the EVALUATION instead.
        Skeleton s;
        Joint child;               // index 0, but the child
        child.name = "child";
        child.parent = 1;
        child.rest.translation = Vec3{1, 0, 0};
        Joint root;                // index 1, the parent
        root.name = "root";
        root.parent = -1;
        s.joints = {child, root};

        CHECK(!s.ParentsFirst());
        CHECK(s.Finalize());
        CHECK(s.order.size() == 2);
        CHECK(s.order[0] == 1 && s.order[1] == 0);  // root evaluated first

        Pose pose;
        Clip{}.Sample(0.0f, s, &pose);
        pose.local[1].rotation = QuatFromAxisAngle(Vec3{0, 0, 1}, 1.5707963f);

        std::vector<Mat4> world;
        ComputeJointWorld(s, pose, &world);
        // The child must have been carried by the root's rotation. Evaluated in
        // index order it would use the root's IDENTITY and stay at (1,0,0).
        const Vec4 c = world[0] * Vec4{0, 0, 0, 1};
        CHECK(Near(c.x, 0.0f) && Near(c.y, 1.0f));

        // A cycle is refused rather than half-ordered.
        Skeleton cyclic;
        Joint a, b;
        a.parent = 1;
        b.parent = 0;
        cyclic.joints = {a, b};
        CHECK(!cyclic.Finalize());
        CHECK(cyclic.order.empty());
    }

    if (g_failures == 0) std::printf("anim_test: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
