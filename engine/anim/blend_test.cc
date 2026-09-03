// Pose blending, crossfades and root motion.
//
// Everything here fails by looking SLIGHTLY wrong, which is the hardest kind of
// bug to find by playing the game. A crossfade that takes the long way round a
// quaternion spins a limb through 350 degrees in one frame -- at 60 Hz that is
// one frame of nonsense and reads as a glitch in the model. A blend space that
// does not synchronise phase makes a character's legs stop moving in the middle
// of a transition. Root motion applied without being removed from the pose
// moves the character at double speed, which reads as a tuning problem.
//
// All of them are numbers.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "engine/anim/anim.h"
#include "engine/anim/blend.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

using eng::Quat;
using eng::Vec3;

// A three-joint chain: root, then two children in a line up the Y axis.
eng::anim::Skeleton Chain() {
    eng::anim::Skeleton s;
    s.joints.resize(3);
    s.joints[0].name = "root";
    s.joints[0].parent = -1;
    s.joints[1].name = "mid";
    s.joints[1].parent = 0;
    s.joints[1].rest.translation = Vec3{0.0f, 1.0f, 0.0f};
    s.joints[2].name = "tip";
    s.joints[2].parent = 1;
    s.joints[2].rest.translation = Vec3{0.0f, 1.0f, 0.0f};
    s.Finalize();
    return s;
}

// A clip that rotates one joint about Y from `from` to `to` over `duration`,
// and optionally translates the root forward.
eng::anim::Clip Spin(const char* name, int joint, float from, float to,
                     float duration, float forward = 0.0f) {
    eng::anim::Clip c;
    c.name = name;
    c.duration = duration;
    eng::anim::Channel rot;
    rot.joint = joint;
    rot.path = eng::anim::Path::Rotation;
    rot.times = {0.0f, duration};
    for (float angle : {from, to}) {
        const Quat q = eng::QuatFromAxisAngle(Vec3{0, 1, 0}, angle);
        rot.values.insert(rot.values.end(), {q.x, q.y, q.z, q.w});
    }
    c.channels.push_back(rot);
    if (forward != 0.0f) {
        eng::anim::Channel t;
        t.joint = 0;
        t.path = eng::anim::Path::Translation;
        t.times = {0.0f, duration};
        t.values = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, forward};
        c.channels.push_back(t);
    }
    return c;
}

float AngleOf(const Quat& q) {
    return 2.0f * std::acos(std::clamp(std::fabs(q.w), -1.0f, 1.0f));
}

float AngleBetween(const Quat& a, const Quat& b) {
    const float d = std::fabs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
    return 2.0f * std::acos(std::clamp(d, -1.0f, 1.0f));
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const eng::anim::Skeleton skeleton = Chain();

    {
        std::printf("blending takes the short way round\n");
        // Two rotations 20 degrees apart, but expressed with opposite
        // quaternion signs -- which is what a real exporter produces, because
        // q and -q are the same rotation and nothing normalises the sign.
        eng::anim::Pose a, b;
        a.local.resize(3);
        b.local.resize(3);
        const Quat qa = eng::QuatFromAxisAngle(Vec3{0, 1, 0}, -0.1745f);  // -10 deg
        const Quat qb0 = eng::QuatFromAxisAngle(Vec3{0, 1, 0}, 0.1745f);  // +10 deg
        const Quat qb = Quat{-qb0.x, -qb0.y, -qb0.z, -qb0.w};             // same, negated
        a.local[1].rotation = qa;
        b.local[1].rotation = qb;

        eng::anim::Pose mid;
        eng::anim::BlendPoses(a, b, 0.5f, &mid);
        const float travelled = AngleBetween(qa, mid.local[1].rotation) * 57.2958f;
        std::printf("    halfway between -10 and +10 degrees: %.2f degrees from "
                    "the start\n", travelled);
        // 10 degrees is the short way. 170 would be the long way round, which
        // is what a slerp without the sign fix gives.
        Check(std::fabs(travelled - 10.0f) < 0.5f, "it travels 10 degrees, not 170");

        // And the ends are exact: a blend at 0 or 1 must reproduce its input,
        // or every animation is subtly filtered.
        eng::anim::Pose at0, at1;
        eng::anim::BlendPoses(a, b, 0.0f, &at0);
        eng::anim::BlendPoses(a, b, 1.0f, &at1);
        Check(AngleBetween(at0.local[1].rotation, qa) < 1e-4f, "t=0 is exactly a");
        Check(AngleBetween(at1.local[1].rotation, qb) < 1e-4f, "t=1 is exactly b");
    }

    {
        std::printf("\na mask keeps the two halves apart\n");
        eng::anim::Pose a, b;
        a.local.resize(3);
        b.local.resize(3);
        b.local[1].rotation = eng::QuatFromAxisAngle(Vec3{0, 1, 0}, 1.0f);
        b.local[2].rotation = eng::QuatFromAxisAngle(Vec3{0, 1, 0}, 1.0f);

        std::vector<float> mask;
        eng::anim::SetSubtreeWeight(skeleton, 2, 1.0f, &mask);
        std::printf("    mask: %.0f %.0f %.0f\n", mask[0], mask[1], mask[2]);
        Check(mask[0] == 0.0f && mask[1] == 0.0f && mask[2] == 1.0f,
              "a subtree mask covers the joint and its children only");

        eng::anim::Pose out;
        eng::anim::BlendPosesMasked(a, b, 1.0f, mask, &out);
        Check(AngleOf(out.local[1].rotation) < 1e-4f, "the unmasked joint is untouched");
        Check(AngleBetween(out.local[2].rotation, b.local[2].rotation) < 1e-4f,
              "and the masked one takes the layer fully");

        // A whole subtree, from the middle down.
        std::vector<float> upper;
        eng::anim::SetSubtreeWeight(skeleton, 1, 1.0f, &upper);
        Check(upper[0] == 0.0f && upper[1] == 1.0f && upper[2] == 1.0f,
              "and masking a parent takes its descendants with it");
    }

    {
        std::printf("\nan additive pose modifies rather than replaces\n");
        eng::anim::Pose base, additive, reference;
        base.local.resize(3);
        additive.local.resize(3);
        reference.local.resize(3);
        // The base has the joint at 30 degrees; the additive is 10 degrees away
        // from its own reference. The result should be 40, not 10.
        base.local[1].rotation = eng::QuatFromAxisAngle(Vec3{0, 1, 0}, 0.5236f);
        additive.local[1].rotation = eng::QuatFromAxisAngle(Vec3{0, 1, 0}, 0.1745f);
        reference.local[1].rotation = Quat{};

        eng::anim::Pose out;
        eng::anim::AddPose(base, additive, reference, 1.0f, &out);
        const float result = AngleOf(out.local[1].rotation) * 57.2958f;
        std::printf("    30 degrees of base plus 10 of additive: %.2f degrees\n",
                    result);
        Check(std::fabs(result - 40.0f) < 0.5f, "the two compose");

        // At half weight it is 35, not 20 -- an additive at half weight is half
        // the DELTA, not a blend toward the additive pose.
        eng::anim::AddPose(base, additive, reference, 0.5f, &out);
        const float half = AngleOf(out.local[1].rotation) * 57.2958f;
        std::printf("    at half weight: %.2f degrees\n", half);
        Check(std::fabs(half - 35.0f) < 0.5f, "and half weight is half the delta");

        // Zero weight leaves the base exactly alone.
        eng::anim::AddPose(base, additive, reference, 0.0f, &out);
        Check(AngleBetween(out.local[1].rotation, base.local[1].rotation) < 1e-4f,
              "and zero weight is a no-op");
    }

    {
        std::printf("\na crossfade is gradual and lands exactly\n");
        const eng::anim::Clip left = Spin("left", 1, -1.0f, -1.0f, 1.0f);
        const eng::anim::Clip right = Spin("right", 1, 1.0f, 1.0f, 1.0f);

        eng::anim::Animator anim(skeleton);
        eng::anim::StateDesc a;
        a.name = "left";
        a.clip = &left;
        eng::anim::StateDesc b;
        b.name = "right";
        b.clip = &right;
        const int sa = anim.AddState(a);
        const int sb = anim.AddState(b);

        anim.PlayImmediate(sa);
        anim.Update(0.0f);
        const float start = AngleOf(anim.CurrentPose().local[1].rotation) * 57.2958f;

        anim.Play(sb, 0.4f);
        std::vector<float> path;
        for (int i = 0; i < 40; ++i) {
            anim.Update(1.0f / 60.0f);
            path.push_back(AngleOf(anim.CurrentPose().local[1].rotation) * 57.2958f);
        }
        std::printf("    from %.1f degrees, after 40 frames: %.1f degrees\n", start,
                    path.back());
        Check(!anim.Transitioning(), "the fade has finished after its duration");
        Check(std::fabs(path.back() - 57.3f) < 1.0f,
              "and lands exactly on the target pose");

        // GRADUAL. A fade that snapped would show one large jump; the largest
        // single-frame change has to be a small fraction of the total.
        float biggest = 0.0f;
        for (std::size_t i = 1; i < path.size(); ++i)
            biggest = std::max(biggest, std::fabs(path[i] - path[i - 1]));
        std::printf("    largest single-frame change: %.2f degrees\n", biggest);
        Check(biggest < 12.0f, "and no single frame carries most of the move");
        // SMOOTHSTEP, not linear: the first and last frames of the fade should
        // move less than the middle ones. A linear ramp starts at full speed
        // and stops dead, which reads as a flinch at each end.
        Check(std::fabs(path[1] - path[0]) < biggest * 0.6f,
              "the fade eases in rather than starting at full speed");
    }

    {
        std::printf("\ninterrupting a fade does not snap\n");
        const eng::anim::Clip left = Spin("left", 1, -1.2f, -1.2f, 1.0f);
        const eng::anim::Clip right = Spin("right", 1, 1.2f, 1.2f, 1.0f);
        eng::anim::Animator anim(skeleton);
        eng::anim::StateDesc a;
        a.name = "l";
        a.clip = &left;
        eng::anim::StateDesc b;
        b.name = "r";
        b.clip = &right;
        const int sa = anim.AddState(a), sb = anim.AddState(b);

        anim.PlayImmediate(sa);
        anim.Update(0.0f);
        anim.Play(sb, 0.5f);
        for (int i = 0; i < 10; ++i) anim.Update(1.0f / 60.0f);
        const float before = AngleOf(anim.CurrentPose().local[1].rotation);
        // Reverse, part way through.
        anim.Play(sa, 0.5f);
        anim.Update(1.0f / 60.0f);
        const float after = AngleOf(anim.CurrentPose().local[1].rotation);
        const float jump = std::fabs(after - before) * 57.2958f;
        std::printf("    reversing mid-fade moved the pose %.2f degrees in one "
                    "frame\n", jump);
        // A naive implementation snaps to the old target before starting the
        // new fade, which here would be a jump of tens of degrees.
        Check(jump < 5.0f, "the pose continues from where it was");
    }

    {
        std::printf("\na blend space interpolates and synchronises\n");
        // A "walk" of 1.2 s and a "run" of 0.6 s. Their joint angles differ, so
        // the blend is measurable, and their durations differ, which is what
        // makes synchronisation matter.
        const eng::anim::Clip walk = Spin("walk", 1, 0.0f, 0.6f, 1.2f);
        const eng::anim::Clip run = Spin("run", 1, 0.0f, 1.4f, 0.6f);
        eng::anim::BlendSpaceDesc space;
        space.name = "locomotion";
        space.samples = {{&walk, 1.5f, 1.0f}, {&run, 5.0f, 1.0f}};

        eng::anim::Animator anim(skeleton);
        const int s = anim.AddBlendSpace(space);
        anim.PlayImmediate(s);

        // At the walk's own speed, the pose must be the walk's.
        anim.SetParameter(1.5f);
        anim.Update(0.3f);
        const float phase = anim.Phase();
        const float at_walk = AngleOf(anim.CurrentPose().local[1].rotation);
        // The walk at this phase, sampled directly.
        eng::anim::Pose direct;
        walk.Sample(phase * 1.2f, skeleton, &direct, true);
        const float expected = AngleOf(direct.local[1].rotation);
        std::printf("    at the walk's parameter: %.4f rad, walk alone %.4f rad\n",
                    at_walk, expected);
        Check(std::fabs(at_walk - expected) < 1e-3f,
              "at a sample's own position the blend is that sample");

        // PHASE SYNCHRONISATION. Both clips are sampled at the same fraction of
        // their own length, so the blend is between two comparable poses rather
        // than between opposite halves of a stride. Checked by asking whether
        // the blended angle stays between the two clips' angles at the same
        // phase -- which it cannot if they are being sampled out of step.
        anim.SetParameter(3.25f);  // exactly halfway
        bool always_between = true;
        for (int i = 0; i < 60; ++i) {
            anim.Update(1.0f / 60.0f);
            const float p = anim.Phase();
            eng::anim::Pose pw, pr;
            walk.Sample(p * 1.2f, skeleton, &pw, true);
            run.Sample(p * 0.6f, skeleton, &pr, true);
            const float a = AngleOf(pw.local[1].rotation);
            const float b = AngleOf(pr.local[1].rotation);
            const float blended = AngleOf(anim.CurrentPose().local[1].rotation);
            if (blended < std::min(a, b) - 1e-3f || blended > std::max(a, b) + 1e-3f)
                always_between = false;
        }
        Check(always_between,
              "and halfway it stays between the two clips at the same phase");

        // The cycle RATE interpolates too: at the run's parameter a cycle takes
        // 0.6 s, at the walk's 1.2 s. Otherwise the feet slide at one end.
        anim.PlayImmediate(s);
        anim.SetParameter(5.0f);
        int frames = 0;
        float last = anim.Phase();
        for (; frames < 300; ++frames) {
            anim.Update(1.0f / 120.0f);
            if (anim.Phase() < last) break;  // wrapped
            last = anim.Phase();
        }
        const float cycle = float(frames + 1) / 120.0f;
        std::printf("    a cycle at the run's parameter took %.3f s (clip is "
                    "0.600 s)\n", cycle);
        Check(std::fabs(cycle - 0.6f) < 0.03f,
              "and the cycle rate follows the parameter");
    }

    {
        std::printf("\nroot motion is extracted and removed\n");
        // A clip that walks the root two metres forward over one second.
        const eng::anim::Clip walk = Spin("walk", 1, 0.0f, 0.4f, 1.0f, 2.0f);
        eng::anim::Animator anim(skeleton);
        eng::anim::StateDesc d;
        d.name = "walk";
        d.clip = &walk;
        d.loop = false;
        const int s = anim.AddState(d);

        eng::anim::RootMotionConfig rc;
        rc.joint = 0;
        rc.horizontal = true;
        rc.vertical = false;
        rc.rotation = false;
        anim.SetRootMotion(rc);
        anim.PlayImmediate(s);

        float travelled = 0.0f;
        float worst_root_z = 0.0f;
        for (int i = 0; i < 60; ++i) {
            anim.Update(1.0f / 60.0f);
            travelled += anim.TakeRootMotion().translation.z;
            worst_root_z = std::max(worst_root_z,
                                    std::fabs(anim.CurrentPose().local[0].translation.z));
        }
        std::printf("    extracted %.3f m over one second (clip travels 2.000 m)\n",
                    travelled);
        Check(std::fabs(travelled - 2.0f) < 0.05f, "the whole travel is reported");
        // AND REMOVED FROM THE POSE. Leaving it in means the character moves
        // and the skeleton moves, so everything travels at double speed and
        // drifts out of its own collision capsule.
        std::printf("    largest root translation left in the pose: %.6f m\n",
                    worst_root_z);
        Check(worst_root_z < 1e-5f, "and taken out of the pose so it is not doubled");

        // TAKEN, not peeked. Reading it twice must not move the character twice.
        Check(anim.TakeRootMotion().translation.z == 0.0f,
              "and taking it twice yields nothing the second time");
    }

    {
        std::printf("\na looping clip's wrap is not reported as motion\n");
        const eng::anim::Clip walk = Spin("walk", 1, 0.0f, 0.4f, 1.0f, 2.0f);
        eng::anim::Animator anim(skeleton);
        eng::anim::StateDesc d;
        d.name = "walk";
        d.clip = &walk;
        d.loop = true;
        const int s = anim.AddState(d);
        eng::anim::RootMotionConfig rc;
        rc.joint = 0;
        rc.rotation = false;
        anim.SetRootMotion(rc);
        anim.PlayImmediate(s);

        // Three full cycles. At each wrap the root jumps from 2 m back to 0,
        // and a naive delta would report -2 m -- so the character would walk
        // forward for a second and teleport backwards, forever.
        float total = 0.0f, worst_step = 0.0f;
        for (int i = 0; i < 180; ++i) {
            anim.Update(1.0f / 60.0f);
            const float step = anim.TakeRootMotion().translation.z;
            total += step;
            worst_step = std::max(worst_step, std::fabs(step));
        }
        std::printf("    three cycles: %.3f m total, largest single step %.4f m\n",
                    total, worst_step);
        Check(total > 5.5f, "three cycles of a 2 m clip travel about six metres");
        Check(worst_step < 0.1f, "and no single frame reports a whole cycle");
    }

    std::printf(g_failures == 0 ? "\nblend_test: all checks passed\n"
                                : "\nblend_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
