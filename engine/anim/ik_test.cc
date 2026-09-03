// Two-bone IK, foot placement and look-at.
//
// An IK solver has one obvious property -- the end lands on the target -- and
// three that are easy to miss and are what separate a usable solver from one
// that has to be turned off:
//
//   * The BEND DIRECTION must be stable. A knee that flips as the leg passes
//     through straight is one frame of the joint bending backwards, every step.
//   * An UNREACHABLE target must degrade, not explode. The law of cosines
//     produces a value outside -1..1 there, and acos of that is a NaN that
//     propagates into the pose and makes the whole character vanish.
//   * The result must be LOCAL rotations. Writing world-space ones works
//     perfectly until the character turns round, at which point the limb keeps
//     pointing where it was solved.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "engine/anim/anim.h"
#include "engine/anim/ik.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

using eng::Mat4;
using eng::Quat;
using eng::Vec3;

// A leg: hip at the origin, knee a metre down, ankle another metre down. Bone
// lengths of 1 make the reachable range exactly 0 to 2, so every bound in the
// solver has a known value to be checked against.
eng::anim::Skeleton Leg() {
    eng::anim::Skeleton s;
    s.joints.resize(4);
    s.joints[0].name = "root";
    s.joints[0].parent = -1;
    s.joints[1].name = "hip";
    s.joints[1].parent = 0;
    s.joints[2].name = "knee";
    s.joints[2].parent = 1;
    s.joints[2].rest.translation = Vec3{0.0f, -1.0f, 0.0f};
    s.joints[3].name = "ankle";
    s.joints[3].parent = 2;
    s.joints[3].rest.translation = Vec3{0.0f, -1.0f, 0.0f};
    s.Finalize();
    return s;
}

eng::anim::Pose RestPose(const eng::anim::Skeleton& s) {
    eng::anim::Pose p;
    p.local.resize(s.joints.size());
    for (std::size_t i = 0; i < s.joints.size(); ++i) p.local[i] = s.joints[i].rest;
    return p;
}

Vec3 JointAt(const eng::anim::Skeleton& s, const eng::anim::Pose& p, int j) {
    std::vector<Mat4> world;
    ComputeJointWorld(s, p, &world);
    return Vec3{world[std::size_t(j)].col[3].x, world[std::size_t(j)].col[3].y,
                world[std::size_t(j)].col[3].z};
}

bool Finite(const eng::anim::Pose& p) {
    for (const eng::anim::Transform& t : p.local) {
        const float v[7] = {t.translation.x, t.translation.y, t.translation.z,
                            t.rotation.x,    t.rotation.y,    t.rotation.z,
                            t.rotation.w};
        for (float f : v)
            if (!std::isfinite(f)) return false;
    }
    return true;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const eng::anim::Skeleton leg = Leg();

    {
        std::printf("the end reaches the target\n");
        eng::anim::TwoBoneIk ik;
        ik.root = 1;
        ik.mid = 2;
        ik.end = 3;
        ik.pole = Vec3{0.0f, 0.0f, 1.0f};

        float worst = 0.0f;
        // A grid of reachable targets. One target proves the solve runs; a grid
        // proves the trigonometry is right rather than right at one place.
        for (float x = -1.2f; x <= 1.2f; x += 0.3f)
            for (float y = -1.9f; y <= -0.4f; y += 0.3f)
                for (float z = -1.0f; z <= 1.0f; z += 0.5f) {
                    const Vec3 target{x, y, z};
                    if (eng::Length(target) > 1.95f) continue;
                    eng::anim::Pose pose = RestPose(leg);
                    ik.target = target;
                    if (!eng::anim::SolveTwoBoneIk(leg, ik, &pose)) {
                        Check(false, "the solve reported failure on a valid chain");
                        break;
                    }
                    worst = std::max(worst, eng::Length(JointAt(leg, pose, 3) - target));
                }
        std::printf("    worst miss over the reachable volume: %.5f m\n", worst);
        Check(worst < 1e-3f, "every reachable target is hit to within a millimetre");
    }

    {
        std::printf("\nthe bone lengths are preserved\n");
        eng::anim::TwoBoneIk ik;
        ik.root = 1;
        ik.mid = 2;
        ik.end = 3;
        ik.target = Vec3{0.8f, -1.2f, 0.3f};
        eng::anim::Pose pose = RestPose(leg);
        (void)eng::anim::SolveTwoBoneIk(leg, ik, &pose);
        const float upper = eng::Length(JointAt(leg, pose, 2) - JointAt(leg, pose, 1));
        const float lower = eng::Length(JointAt(leg, pose, 3) - JointAt(leg, pose, 2));
        std::printf("    thigh %.5f m, shin %.5f m (both should be 1)\n", upper, lower);
        // An IK solver that moved the joints rather than rotating them would hit
        // every target perfectly and stretch the limb to do it.
        Check(std::fabs(upper - 1.0f) < 1e-4f && std::fabs(lower - 1.0f) < 1e-4f,
              "the limb rotates rather than stretching");
    }

    {
        std::printf("\nthe pole controls which way the knee bends\n");
        eng::anim::TwoBoneIk ik;
        ik.root = 1;
        ik.mid = 2;
        ik.end = 3;
        ik.target = Vec3{0.0f, -1.6f, 0.0f};

        ik.pole = Vec3{0.0f, 0.0f, 1.0f};
        eng::anim::Pose forward = RestPose(leg);
        (void)eng::anim::SolveTwoBoneIk(leg, ik, &forward);
        const Vec3 knee_forward = JointAt(leg, forward, 2);

        ik.pole = Vec3{0.0f, 0.0f, -1.0f};
        eng::anim::Pose backward = RestPose(leg);
        (void)eng::anim::SolveTwoBoneIk(leg, ik, &backward);
        const Vec3 knee_backward = JointAt(leg, backward, 2);

        std::printf("    knee z: pole forward %.4f, pole backward %.4f\n",
                    knee_forward.z, knee_backward.z);
        Check(knee_forward.z > 0.1f, "a forward pole puts the knee forward");
        Check(knee_backward.z < -0.1f, "and a backward one puts it backward");
    }

    {
        // THE FLIP. A leg sweeping through nearly straight and back is what a
        // walk cycle does at every step, and a solver whose bend direction is
        // derived from the current pose rather than from a pole flips there --
        // one frame of the knee bending backwards, once per stride.
        std::printf("\nthe knee does not flip as the leg passes through straight\n");
        eng::anim::TwoBoneIk ik;
        ik.root = 1;
        ik.mid = 2;
        ik.end = 3;
        ik.pole = Vec3{0.0f, 0.0f, 1.0f};

        float previous_z = 0.0f;
        bool flipped = false;
        bool first = true;
        for (int i = 0; i <= 80; ++i) {
            // Sweeping the target from bent, through almost fully extended, and
            // back.
            const float t = float(i) / 80.0f;
            const float reach = 1.2f + 0.78f * std::sin(t * 3.14159f);
            eng::anim::Pose pose = RestPose(leg);
            ik.target = Vec3{0.0f, -reach, 0.0f};
            (void)eng::anim::SolveTwoBoneIk(leg, ik, &pose);
            const float z = JointAt(leg, pose, 2).z;
            if (!first && z * previous_z < 0.0f && std::fabs(z) > 1e-4f)
                flipped = true;
            previous_z = z;
            first = false;
        }
        Check(!flipped, "the knee stays on the pole's side throughout the sweep");
    }

    {
        std::printf("\nan unreachable target degrades instead of exploding\n");
        eng::anim::TwoBoneIk ik;
        ik.root = 1;
        ik.mid = 2;
        ik.end = 3;
        ik.pole = Vec3{0.0f, 0.0f, 1.0f};

        // Far beyond the two-metre reach. The law of cosines here produces a
        // cosine greater than one, and acos of that is a NaN that spreads
        // through the quaternion into every child joint -- the character does
        // not bend wrongly, it disappears.
        ik.target = Vec3{0.0f, -50.0f, 0.0f};
        eng::anim::Pose pose = RestPose(leg);
        (void)eng::anim::SolveTwoBoneIk(leg, ik, &pose);
        Check(Finite(pose), "an unreachable target produces no NaN");
        const Vec3 ankle = JointAt(leg, pose, 3);
        std::printf("    reaching for 50 m: the ankle got to %.4f m\n",
                    eng::Length(ankle));
        Check(eng::Length(ankle) > 1.9f, "and the limb extends as far as it can");
        // NOT COMPLETELY straight, which is the other half: a fully locked limb
        // has no bend left, so the last millimetre of target travel swings the
        // knee through several degrees and the leg visibly snaps.
        Check(eng::Length(ankle) < 2.0f, "without locking completely straight");

        // A target INSIDE the fold -- closer than the difference of the bone
        // lengths -- is the other end of the same problem. With equal bones
        // that limit is zero, so the target at the hip itself is the case.
        ik.target = Vec3{0.0f, 0.0f, 0.0f};
        eng::anim::Pose folded = RestPose(leg);
        (void)eng::anim::SolveTwoBoneIk(leg, ik, &folded);
        Check(Finite(folded), "and a target at the joint itself is finite too");
    }

    {
        std::printf("\nthe solver writes LOCAL rotations\n");
        // The test that catches writing world-space rotations, which works
        // perfectly until the character turns round.
        eng::anim::TwoBoneIk ik;
        ik.root = 1;
        ik.mid = 2;
        ik.end = 3;
        ik.pole = Vec3{0.0f, 0.0f, 1.0f};
        ik.target = Vec3{0.7f, -1.4f, 0.0f};

        eng::anim::Pose upright = RestPose(leg);
        (void)eng::anim::SolveTwoBoneIk(leg, ik, &upright);
        const Vec3 solved = JointAt(leg, upright, 3);

        // The same solve on a character rotated 90 degrees about Y, with the
        // target rotated with them. The ankle must end up in the rotated place.
        eng::anim::Pose turned = RestPose(leg);
        const Quat spin = eng::QuatFromAxisAngle(Vec3{0, 1, 0}, 1.5708f);
        turned.local[0].rotation = spin;
        eng::anim::TwoBoneIk rotated = ik;
        rotated.target = eng::Rotate(spin, ik.target);
        rotated.pole = eng::Rotate(spin, ik.pole);
        (void)eng::anim::SolveTwoBoneIk(leg, rotated, &turned);
        const Vec3 got = JointAt(leg, turned, 3);
        const Vec3 want = eng::Rotate(spin, solved);
        std::printf("    solved at (%.3f, %.3f, %.3f), rotated solve landed at "
                    "(%.3f, %.3f, %.3f)\n", want.x, want.y, want.z, got.x, got.y, got.z);
        Check(eng::Length(got - want) < 1e-3f,
              "the same solve under a rotated parent gives the rotated answer");
    }

    {
        std::printf("\nweight fades the solve in\n");
        eng::anim::TwoBoneIk ik;
        ik.root = 1;
        ik.mid = 2;
        ik.end = 3;
        ik.pole = Vec3{0.0f, 0.0f, 1.0f};
        ik.target = Vec3{0.9f, -1.3f, 0.0f};

        eng::anim::Pose rest = RestPose(leg);
        const Vec3 unsolved = JointAt(leg, rest, 3);
        float previous = eng::Length(unsolved - ik.target);
        bool monotonic = true;
        // An INTEGER loop. Accumulating 0.1f ten times overshoots one, so a
        // `for (float w = 0; w <= 1; w += 0.1f)` never runs at full weight --
        // it stops at 0.9, and the "full weight reaches the target" check then
        // measures a 90% solve and fails by 0.13 m against code that is right.
        for (int step = 0; step <= 10; ++step) {
            const float w = float(step) / 10.0f;
            eng::anim::Pose pose = RestPose(leg);
            ik.weight = w;
            (void)eng::anim::SolveTwoBoneIk(leg, ik, &pose);
            const float miss = eng::Length(JointAt(leg, pose, 3) - ik.target);
            if (miss > previous + 1e-3f) monotonic = false;
            previous = miss;
        }
        std::printf("    at full weight the miss is %.5f m\n", previous);
        Check(monotonic, "raising the weight never moves the end further away");
        Check(previous < 1e-3f, "and full weight reaches the target");

        ik.weight = 0.0f;
        eng::anim::Pose untouched = RestPose(leg);
        (void)eng::anim::SolveTwoBoneIk(leg, ik, &untouched);
        Check(eng::Length(JointAt(leg, untouched, 3) - unsolved) < 1e-5f,
              "and zero weight leaves the pose exactly alone");
    }

    {
        std::printf("\na chain that is not a chain is refused\n");
        eng::anim::TwoBoneIk bad;
        bad.root = 1;
        bad.mid = 3;  // the ankle's parent is the knee, not the hip
        bad.end = 2;
        eng::anim::Pose pose = RestPose(leg);
        Check(!eng::anim::SolveTwoBoneIk(leg, bad, &pose),
              "joints that are not connected are rejected");
        eng::anim::TwoBoneIk oob;
        oob.root = 1;
        oob.mid = 2;
        oob.end = 99;
        Check(!eng::anim::SolveTwoBoneIk(leg, oob, &pose),
              "and so is an index off the end of the skeleton");
    }

    {
        std::printf("\nfoot placement puts the sole on the ground\n");
        eng::anim::FootIkConfig config;
        config.limb.root = 1;
        config.limb.mid = 2;
        config.limb.end = 3;
        config.limb.pole = Vec3{0.0f, 0.0f, 1.0f};
        config.ankle_to_sole = Vec3{0.0f, -0.1f, 0.0f};

        // The character stands with the hip at y = 2, so the rest pose puts the
        // ankle at y = 0 and the sole at y = -0.1.
        const Mat4 to_world = Mat4::Translation(Vec3{0.0f, 2.0f, 0.0f});
        eng::anim::Pose pose = RestPose(leg);

        // Ground 0.25 m higher than the animation expects: the foot has to come
        // up, and the leg has to bend.
        eng::anim::GroundHit ground;
        ground.hit = true;
        ground.point = Vec3{0.0f, 0.15f, 0.0f};
        ground.normal = Vec3{0.0f, 1.0f, 0.0f};
        const float lower = eng::anim::SolveFootIk(leg, config, to_world, ground, &pose);

        std::vector<Mat4> world;
        ComputeJointWorld(leg, pose, &world);
        const Mat4 ankle = to_world * world[3];
        const float sole_y = ankle.col[3].y + config.ankle_to_sole.y;
        std::printf("    ground at %.3f, sole ended at %.3f, hip lowered by %.3f\n",
                    ground.point.y, sole_y, lower);
        // THE SOLE, not the ankle. Putting the ankle on the ground sinks the
        // foot into it by the ankle's height, which reads as standing in mud --
        // and is the mistake that makes people conclude foot IK does not work.
        Check(std::fabs(sole_y - ground.point.y) < 5e-3f, "the sole lands on the ground");
        Check(lower == 0.0f, "and raising a foot does not lower the hips");

        // Ground BELOW the animation: the leg would have to stretch, so the
        // solver reports how far the hips should come down instead.
        eng::anim::Pose low = RestPose(leg);
        ground.point = Vec3{0.0f, -0.4f, 0.0f};
        const float drop = eng::anim::SolveFootIk(leg, config, to_world, ground, &low);
        std::printf("    ground 0.4 m lower: hips asked to drop %.3f m\n", drop);
        Check(drop < -0.1f,
              "a foot below the animation asks the hips to come down");
    }

    {
        std::printf("\nlook-at turns the chain and respects its limit\n");
        eng::anim::Skeleton spine;
        spine.joints.resize(4);
        spine.joints[0].parent = -1;
        for (int i = 1; i < 4; ++i) {
            spine.joints[std::size_t(i)].parent = i - 1;
            spine.joints[std::size_t(i)].rest.translation = Vec3{0.0f, 0.4f, 0.0f};
        }
        spine.Finalize();

        eng::anim::LookAtConfig look;
        look.joints = {1, 2, 3};
        look.forward_axis = Vec3{0.0f, 0.0f, 1.0f};
        look.target = Vec3{3.0f, 1.2f, 3.0f};  // 45 degrees to the right

        eng::anim::Pose pose = RestPose(spine);
        eng::anim::SolveLookAt(spine, look, &pose);
        Check(Finite(pose), "the solve produces finite rotations");

        std::vector<Mat4> world;
        ComputeJointWorld(spine, pose, &world);
        const Mat4& head = world[3];
        const Vec3 head_pos{head.col[3].x, head.col[3].y, head.col[3].z};
        const Vec3 facing =
            eng::Normalize(Vec3{head.col[2].x, head.col[2].y, head.col[2].z});
        const Vec3 want = eng::Normalize(look.target - head_pos);
        const float error = std::acos(std::clamp(eng::Dot(facing, want), -1.0f, 1.0f)) *
                            57.2958f;
        std::printf("    the head is %.1f degrees off the target\n", error);
        Check(error < 12.0f, "the head ends up looking at the target");

        // THE LIMIT. A target directly behind, with a 55-degree cap: the chain
        // must give up rather than rotate a neck through half a turn.
        look.target = Vec3{0.0f, 1.2f, -6.0f};
        eng::anim::Pose behind = RestPose(spine);
        eng::anim::SolveLookAt(spine, look, &behind);
        ComputeJointWorld(spine, behind, &world);
        float worst = 0.0f;
        for (int j : look.joints) {
            const Quat q = behind.local[std::size_t(j)].rotation;
            worst = std::max(worst, 2.0f * std::acos(std::clamp(std::fabs(q.w), -1.0f,
                                                                1.0f)) * 57.2958f);
        }
        std::printf("    target behind: the most any one joint turned is %.1f "
                    "degrees (limit %.0f)\n", worst, look.max_degrees);
        Check(worst <= look.max_degrees + 1.0f, "no joint exceeds its limit");
    }

    std::printf(g_failures == 0 ? "\nik_test: all checks passed\n"
                                : "\nik_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
