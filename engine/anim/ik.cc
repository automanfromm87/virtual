#include "engine/anim/ik.h"

#include <algorithm>
#include <cmath>

namespace eng::anim {
namespace {

Vec3 Translation(const Mat4& m) { return Vec3{m.col[3].x, m.col[3].y, m.col[3].z}; }

// The rotation part of a matrix, as a quaternion. Assumes no scale, which holds
// for a joint chain that has not been scaled -- and if one has been, the
// alternative is to normalise the columns first and lose the scale anyway.
Quat RotationOf(const Mat4& m) {
    Vec3 x{m.col[0].x, m.col[0].y, m.col[0].z};
    Vec3 y{m.col[1].x, m.col[1].y, m.col[1].z};
    Vec3 z{m.col[2].x, m.col[2].y, m.col[2].z};
    const float lx = Length(x), ly = Length(y), lz = Length(z);
    if (lx > 1e-9f) x = x * (1.0f / lx);
    if (ly > 1e-9f) y = y * (1.0f / ly);
    if (lz > 1e-9f) z = z * (1.0f / lz);

    // Shepperd's method: pick the largest diagonal term so the divisor is never
    // small. The textbook single-branch formula divides by sqrt(1 + trace),
    // which goes to zero for a 180-degree rotation and produces a quaternion of
    // infinities -- and a joint rotated exactly half a turn is not a rare input.
    const float m00 = x.x, m01 = y.x, m02 = z.x;
    const float m10 = x.y, m11 = y.y, m12 = z.y;
    const float m20 = x.z, m21 = y.z, m22 = z.z;
    const float trace = m00 + m11 + m22;
    Quat q;
    if (trace > 0.0f) {
        const float s = std::sqrt(std::max(trace + 1.0f, 0.0f)) * 2.0f;
        q.w = 0.25f * s;
        q.x = (m21 - m12) / s;
        q.y = (m02 - m20) / s;
        q.z = (m10 - m01) / s;
    } else if (m00 > m11 && m00 > m22) {
        const float s = std::sqrt(std::max(1.0f + m00 - m11 - m22, 0.0f)) * 2.0f;
        q.w = (m21 - m12) / s;
        q.x = 0.25f * s;
        q.y = (m01 + m10) / s;
        q.z = (m02 + m20) / s;
    } else if (m11 > m22) {
        const float s = std::sqrt(std::max(1.0f + m11 - m00 - m22, 0.0f)) * 2.0f;
        q.w = (m02 - m20) / s;
        q.x = (m01 + m10) / s;
        q.y = 0.25f * s;
        q.z = (m12 + m21) / s;
    } else {
        const float s = std::sqrt(std::max(1.0f + m22 - m00 - m11, 0.0f)) * 2.0f;
        q.w = (m10 - m01) / s;
        q.x = (m02 + m20) / s;
        q.y = (m12 + m21) / s;
        q.z = 0.25f * s;
    }
    return Normalize(q);
}

// The rotation taking `from` onto `to`, both unit vectors.
Quat FromTo(Vec3 from, Vec3 to) {
    from = Normalize(from);
    to = Normalize(to);
    const float d = Dot(from, to);
    if (d > 0.99999f) return Quat{};
    if (d < -0.99999f) {
        // OPPOSITE. Any perpendicular axis is a valid half turn, and the cross
        // product is zero here so it cannot supply one. Picking the axis least
        // aligned with `from` keeps the cross product well conditioned; without
        // this branch the result is a quaternion of NaNs and the joint
        // disappears.
        Vec3 axis = Cross(from, Vec3{1.0f, 0.0f, 0.0f});
        if (Dot(axis, axis) < 1e-6f) axis = Cross(from, Vec3{0.0f, 1.0f, 0.0f});
        axis = Normalize(axis);
        return Quat{axis.x, axis.y, axis.z, 0.0f};
    }
    const Vec3 axis = Cross(from, to);
    Quat q{axis.x, axis.y, axis.z, 1.0f + d};
    return Normalize(q);
}

}  // namespace

bool SolveTwoBoneIk(const Skeleton& skeleton, const TwoBoneIk& ik, Pose* pose) {
    if (!pose) return false;
    const int n = int(skeleton.joints.size());
    if (ik.root < 0 || ik.mid < 0 || ik.end < 0) return false;
    if (ik.root >= n || ik.mid >= n || ik.end >= n) return false;
    if (int(pose->local.size()) < n) return false;
    // A REAL CHAIN. Solving through joints that are not actually connected
    // bends the limb somewhere it has no joint, which looks like bad animation
    // rather than like a wiring mistake.
    if (skeleton.joints[std::size_t(ik.mid)].parent != ik.root) return false;
    if (skeleton.joints[std::size_t(ik.end)].parent != ik.mid) return false;
    if (ik.weight <= 0.0f) return true;

    std::vector<Mat4> world;
    ComputeJointWorld(skeleton, *pose, &world);
    const Vec3 a = Translation(world[std::size_t(ik.root)]);
    const Vec3 b = Translation(world[std::size_t(ik.mid)]);
    const Vec3 c = Translation(world[std::size_t(ik.end)]);

    const float upper = Length(b - a);
    const float lower = Length(c - b);
    if (upper < 1e-6f || lower < 1e-6f) return false;

    Vec3 target = ik.target;
    Vec3 to_target = target - a;
    float reach = Length(to_target);
    if (reach < 1e-6f) return true;
    const Vec3 dir = to_target * (1.0f / reach);

    // CLAMPED SHORT OF FULL EXTENSION. At exactly the sum of the bone lengths
    // the knee angle is 180 degrees and its derivative with respect to reach is
    // infinite -- so the last millimetre of travel swings the knee through
    // several degrees, and a foot target crossing that boundary makes the leg
    // snap. Stopping half a percent short keeps a little bend in.
    const float max_reach = (upper + lower) * std::clamp(ik.max_extension, 0.1f, 1.0f);
    // And a MINIMUM: the limb cannot fold shorter than the difference of its
    // bones, and asking it to gives a cosine outside -1..1 and a NaN angle.
    const float min_reach = std::fabs(upper - lower) + 1e-4f;
    reach = std::clamp(reach, min_reach, max_reach);
    target = a + dir * reach;

    // The law of cosines, twice: the angle at the root between the upper bone
    // and the line to the target, and the interior angle at the mid joint.
    const float cos_root =
        std::clamp((upper * upper + reach * reach - lower * lower) /
                       (2.0f * upper * reach), -1.0f, 1.0f);
    const float root_angle = std::acos(cos_root);

    // The bend plane, chosen by the pole. Its normal is perpendicular to the
    // line to the target.
    Vec3 pole_dir = ik.pole_is_direction ? ik.pole : (ik.pole - a);
    // A pole parallel to the limb defines no plane. Falling back to the CURRENT
    // bend keeps the knee where the animation had it, which is the least
    // surprising answer -- and it is a real input, because a leg pointing
    // straight down with a pole directly in front of the character is nearly
    // parallel whenever the character leans.
    Vec3 axis = Cross(dir, Normalize(pole_dir));
    if (Dot(axis, axis) < 1e-8f) {
        const Vec3 current_bend = b - (a + dir * Dot(b - a, dir));
        axis = Cross(dir, current_bend);
    }
    if (Dot(axis, axis) < 1e-8f) axis = Cross(dir, Vec3{0.0f, 1.0f, 0.0f});
    if (Dot(axis, axis) < 1e-8f) axis = Vec3{1.0f, 0.0f, 0.0f};
    axis = Normalize(axis);

    // Where the mid joint has to be. Rotating the direction-to-target about the
    // bend axis by the root angle and stepping the upper bone's length along it.
    //
    // POSITIVE, not negative. With the axis built as cross(dir, pole), a
    // positive rotation carries the limb TOWARD the pole and a negative one
    // away from it -- so the sign is what decides whether the pole attracts the
    // knee or repels it. Getting it backwards is not a subtle error and it is
    // not obvious either: the limb still reaches every target and the bend is
    // still stable, it is simply on the wrong side, and a leg whose knee bends
    // backwards reads as a broken rig rather than as a sign error here.
    const Quat swing = QuatFromAxisAngle(axis, root_angle);
    const Vec3 new_b = a + Rotate(swing, dir) * upper;

    // --- write the rotations back --------------------------------------------
    //
    // The joints' LOCAL rotations are what a pose holds, so each world-space
    // correction has to be moved back into its parent's frame. Setting a world
    // rotation directly is the usual mistake and it works until the character
    // turns, at which point the limb stays pointing the way it was solved.
    const Quat root_world = RotationOf(world[std::size_t(ik.root)]);
    const Quat mid_world = RotationOf(world[std::size_t(ik.mid)]);

    const Quat root_fix = FromTo(b - a, new_b - a);
    Quat root_target_world = Normalize(root_fix * root_world);
    // Weighted by slerping from where it was, which is what lets IK fade in.
    const Quat root_final = Normalize(Slerp(root_world, root_target_world,
                                            std::clamp(ik.weight, 0.0f, 1.0f)));

    const int root_parent = skeleton.joints[std::size_t(ik.root)].parent;
    const Quat root_parent_world =
        root_parent >= 0 ? RotationOf(world[std::size_t(root_parent)]) : Quat{};
    pose->local[std::size_t(ik.root)].rotation =
        Normalize(Conjugate(root_parent_world) * root_final);

    // The mid joint, solved against where the end now is after the root moved.
    // Recomputing the world matrices is the honest way to get that; the
    // alternative is to compose the correction by hand and it is the same
    // arithmetic with more chances to get a frame wrong.
    ComputeJointWorld(skeleton, *pose, &world);
    const Vec3 b2 = Translation(world[std::size_t(ik.mid)]);
    const Vec3 c2 = Translation(world[std::size_t(ik.end)]);
    const Quat mid_world2 = RotationOf(world[std::size_t(ik.mid)]);
    const Quat mid_fix = FromTo(c2 - b2, target - b2);
    const Quat mid_target_world = Normalize(mid_fix * mid_world2);
    const Quat mid_final = Normalize(
        Slerp(mid_world2, mid_target_world, std::clamp(ik.weight, 0.0f, 1.0f)));
    const Quat mid_parent_world = RotationOf(world[std::size_t(ik.root)]);
    pose->local[std::size_t(ik.mid)].rotation =
        Normalize(Conjugate(mid_parent_world) * mid_final);
    (void)mid_world;

    if (ik.set_end_rotation) {
        ComputeJointWorld(skeleton, *pose, &world);
        const Quat end_parent = RotationOf(world[std::size_t(ik.mid)]);
        const Quat want = Normalize(
            Slerp(RotationOf(world[std::size_t(ik.end)]), ik.end_rotation,
                  std::clamp(ik.weight, 0.0f, 1.0f)));
        pose->local[std::size_t(ik.end)].rotation =
            Normalize(Conjugate(end_parent) * want);
    }
    return true;
}

FootTrace FootTraceFor(const Skeleton& skeleton, const FootIkConfig& config,
                       const Mat4& to_world, const Pose& pose) {
    FootTrace trace;
    const int end = config.limb.end;
    if (end < 0 || std::size_t(end) >= skeleton.joints.size()) return trace;
    std::vector<Mat4> world;
    ComputeJointWorld(skeleton, pose, &world);
    const Mat4 ankle = to_world * world[std::size_t(end)];
    const Vec3 p = Translation(ankle);
    trace.origin = p + Vec3{0.0f, config.trace_up, 0.0f};
    trace.length = config.trace_up + config.trace_down;
    return trace;
}

float SolveFootIk(const Skeleton& skeleton, const FootIkConfig& config,
                  const Mat4& to_world, const GroundHit& ground, Pose* pose) {
    if (!pose || !ground.hit) return 0.0f;
    const int end = config.limb.end;
    if (end < 0 || std::size_t(end) >= skeleton.joints.size()) return 0.0f;

    std::vector<Mat4> world;
    ComputeJointWorld(skeleton, *pose, &world);
    const Mat4 ankle_world = to_world * world[std::size_t(end)];
    const Vec3 animated = Translation(ankle_world);

    // The SOLE, not the ankle. Placing the ankle on the ground sinks the foot
    // into it by the ankle's height, which reads as standing in mud -- and it
    // is the mistake that makes people conclude foot IK "does not work".
    const Vec3 sole_offset = Rotate(RotationOf(ankle_world), config.ankle_to_sole);
    const Vec3 animated_sole = animated + sole_offset;
    const float lift = ground.point.y - animated_sole.y;

    TwoBoneIk ik = config.limb;
    // The ankle goes wherever it has to for the SOLE to land on the ground.
    ik.target = animated + Vec3{0.0f, lift, 0.0f};

    // The target is in WORLD space and the solver works in the pose's space, so
    // it has to come back. Inverting the whole matrix rather than assuming it
    // is a rigid transform: a character scaled by its instance transform is
    // ordinary, and the assumption would place the foot at the wrong distance.
    const Mat4 to_model = Inverse(to_world);
    const Vec4 local = to_model * Vec4{ik.target.x, ik.target.y, ik.target.z, 1.0f};
    ik.target = Vec3{local.x, local.y, local.z};

    // FLATTEN THE FOOT to the slope, but only on ground it could stand on. A
    // foot rotated to match a cliff face looks far worse than one that ignores
    // it, and the animated orientation is a reasonable answer there.
    const float cos_limit =
        std::cos(config.max_slope_degrees * 3.14159265f / 180.0f);
    if (ground.normal.y >= cos_limit) {
        const Quat align = FromTo(Vec3{0.0f, 1.0f, 0.0f}, Normalize(ground.normal));
        // Applied to the ANIMATED orientation rather than replacing it, so the
        // foot keeps whatever the clip drew and is only tilted by the slope.
        const Quat animated_rot = RotationOf(ankle_world);
        ik.set_end_rotation = true;
        ik.end_rotation = Normalize(align * animated_rot);
    }

    (void)SolveTwoBoneIk(skeleton, ik, pose);
    // NEGATIVE lift means the ground is BELOW where the clip put the foot, and
    // the leg has to stretch. Returning it lets the caller lower the hips
    // instead, which is the only correct answer when one foot is on a step: the
    // pelvis comes down to the lower foot, or the higher leg locks straight.
    return lift < 0.0f ? lift : 0.0f;
}

void SolveLookAt(const Skeleton& skeleton, const LookAtConfig& config, Pose* pose) {
    if (!pose || config.joints.empty() || config.weight <= 0.0f) return;
    const float max_radians = config.max_degrees * 3.14159265f / 180.0f;

    std::vector<Mat4> world;
    for (std::size_t k = 0; k < config.joints.size(); ++k) {
        const int j = config.joints[k];
        if (j < 0 || std::size_t(j) >= skeleton.joints.size()) continue;
        // RECOMPUTED PER JOINT, because turning the spine moves the head. A
        // single pass over cached matrices would aim every joint at the target
        // from where it was BEFORE its parent turned, so the chain overshoots by
        // the sum of everything below it.
        ComputeJointWorld(skeleton, *pose, &world);
        const Mat4& m = world[std::size_t(j)];
        const Vec3 origin = Translation(m);
        const Quat rot = RotationOf(m);

        const Vec3 want = config.target - origin;
        if (Dot(want, want) < 1e-8f) continue;
        const Vec3 forward = Rotate(rot, config.forward_axis);
        Quat turn = FromTo(forward, Normalize(want));

        // SHARE THE TURN. Each joint takes a growing fraction, so a spine
        // contributes a little and the head most -- which is what makes it read
        // as a body turning rather than a head on a swivel.
        const float share = float(k + 1) / float(config.joints.size());
        float amount = config.weight * share;

        // CLAMPED PER JOINT. A neck that can rotate 180 degrees to follow
        // something behind the character is the classic failure, and clamping
        // the total instead lets one joint take all of it.
        const float angle = 2.0f * std::acos(std::clamp(std::fabs(turn.w), -1.0f, 1.0f));
        if (angle > max_radians && angle > 1e-6f)
            amount = std::min(amount, max_radians / angle);

        turn = Normalize(Slerp(Quat{}, turn, std::clamp(amount, 0.0f, 1.0f)));
        const Quat target_world = Normalize(turn * rot);
        const int parent = skeleton.joints[std::size_t(j)].parent;
        const Quat parent_world =
            parent >= 0 ? RotationOf(world[std::size_t(parent)]) : Quat{};
        pose->local[std::size_t(j)].rotation =
            Normalize(Conjugate(parent_world) * target_world);
    }
}

}  // namespace eng::anim
