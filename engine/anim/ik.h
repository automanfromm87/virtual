// Inverse kinematics: posing a limb from where its END should be rather than
// from where its joints are.
//
// WHAT IT IS FOR, concretely. An animator draws a walk cycle on flat ground. The
// character walks up a ramp, and the feet now pass through it on the up-slope
// and hang above it on the down-slope, because the clip has no idea what the
// ground is doing. No amount of blending fixes that -- the information is not
// in any clip. IK is how the pose is corrected against the world it is standing
// in, and foot placement is the case that makes the difference between "the
// character is walking" and "the character is sliding along a surface".
//
// TWO-BONE, ANALYTIC. A limb is an upper bone, a lower bone and an end -- an
// arm or a leg -- and for that specific shape the answer is trigonometry rather
// than iteration: the two bone lengths and the distance to the target form a
// triangle, and the law of cosines gives the knee angle directly. One
// evaluation, exact, no convergence to tune.
//
// WHY NOT a general solver (CCD, FABRIK). Those handle any chain length, cost
// several iterations, and converge -- so they can be short of the target, and
// they wander when the target is unreachable. Nearly every limb in a character
// is two bones. The general case is worth having when there is a tail or a
// tentacle; it is the wrong default for a leg.
#pragma once

#include <vector>

#include "engine/anim/anim.h"
#include "engine/core/math.h"

namespace eng::anim {

struct TwoBoneIk {
    // The three joints, root to tip. For a leg: hip, knee, ankle.
    int root = -1;
    int mid = -1;
    int end = -1;

    // Where the end should be, in the same space the pose's world matrices are
    // in -- model space, not world space, unless the caller has already put the
    // pose there.
    Vec3 target{0.0f, 0.0f, 0.0f};

    // WHICH WAY THE KNEE BENDS. A triangle with two known sides and a known
    // base has a whole circle of solutions, and this is what picks one: the mid
    // joint is placed on the side of the root-to-target line nearest this
    // point.
    //
    // Not optional and not derivable. Without it the solver has to guess, and
    // the usual guess -- keep the current bend direction -- flips whenever the
    // limb passes through straight, which is a knee snapping backwards once per
    // step.
    Vec3 pole{0.0f, 0.0f, 1.0f};
    bool pole_is_direction = true;

    // 0 leaves the pose alone, 1 fully solves. For fading IK in and out --
    // which is required, not decoration: a foot that snaps onto the ground the
    // instant the ray finds it is worse than one that never moves.
    float weight = 1.0f;

    // How much of the limb's length may be used. Below 1 the limb never
    // straightens completely, which is what stops the visible pop as it locks:
    // a fully extended leg has no bend left, so the last few centimetres of
    // reach change the knee angle very fast.
    float max_extension = 0.995f;

    // Also rotate the END joint to this orientation once solved. For planting a
    // foot flat on a slope rather than leaving it at the angle the clip drew.
    bool set_end_rotation = false;
    Quat end_rotation;
};

// Solves in place. `pose` is modified; `skeleton` supplies the hierarchy and
// the bone lengths.
//
// Returns false when the limb indices do not describe a chain -- the mid's
// parent must be the root and the end's the mid. Checked rather than assumed:
// solving through a joint that is not actually in the chain produces a limb
// that bends in the wrong place, which reads as bad animation.
bool SolveTwoBoneIk(const Skeleton&, const TwoBoneIk&, Pose* pose);

// --- foot placement -------------------------------------------------------

struct FootIkConfig {
    TwoBoneIk limb;
    // The offset from the ankle joint to the sole, in the ankle's own space.
    // Without it the ANKLE is placed on the ground and the foot sinks into it
    // by the height of the ankle, which looks like the character is standing in
    // soft mud.
    Vec3 ankle_to_sole{0.0f, -0.08f, 0.0f};
    // How far above and below the animated foot to look for ground.
    float trace_up = 0.5f;
    float trace_down = 0.6f;
    // The steepest ground the foot will be rotated to match. Beyond it the foot
    // keeps its animated orientation -- a foot flattened against a cliff face
    // looks worse than one that ignores it.
    float max_slope_degrees = 50.0f;
};

// What a caller's ground query has to answer.
struct GroundHit {
    bool hit = false;
    Vec3 point{0.0f, 0.0f, 0.0f};
    Vec3 normal{0.0f, 1.0f, 0.0f};
};

// Places one foot on the ground.
//
// The GROUND QUERY IS A CALLBACK, so this module does not depend on the physics
// module. It is the only thing IK needs from the world, and taking a
// physics::World here would make the animation layer depend on the collision
// layer -- for one raycast, and in a direction nothing else needs.
//
// `to_world` puts the pose's model space into the world the callback answers
// in. Returns the amount the hip should be lowered, which the caller applies:
// when one foot is on a step and the other on the floor, the pelvis has to come
// down to the lower of the two or the higher leg over-extends and locks
// straight. That value cannot be applied here because it depends on BOTH feet.
float SolveFootIk(const Skeleton&, const FootIkConfig&, const Mat4& to_world,
                  const GroundHit& ground, Pose* pose);

// Runs a ground trace for a foot and reports where it should go. Separate from
// the solve so a caller can batch its raycasts.
struct FootTrace {
    Vec3 origin{0.0f, 0.0f, 0.0f};   // start of the downward ray, in world space
    float length = 0.0f;
};
[[nodiscard]] FootTrace FootTraceFor(const Skeleton&, const FootIkConfig&,
                                     const Mat4& to_world, const Pose&);

// --- look-at ---------------------------------------------------------------

struct LookAtConfig {
    // The chain from the lowest joint that should contribute up to the head.
    // Spreading the turn over several joints is what makes it read as a
    // character looking rather than a head on a swivel.
    std::vector<int> joints;
    Vec3 target{0.0f, 0.0f, 0.0f};
    // The joint's own axis that should end up pointing at the target, and the
    // one that should stay up. In the joint's local space.
    Vec3 forward_axis{0.0f, 0.0f, 1.0f};
    Vec3 up_axis{0.0f, 1.0f, 0.0f};
    // How far any one joint may turn. A head that can rotate 180 degrees to
    // follow something behind it is the classic failure.
    float max_degrees = 55.0f;
    float weight = 1.0f;
};

// Turns a chain of joints toward a point. Later joints in the list get more of
// the turn, so a spine contributes a little and the head most.
void SolveLookAt(const Skeleton&, const LookAtConfig&, Pose*);

}  // namespace eng::anim
