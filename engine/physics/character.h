// A kinematic capsule character controller.
//
// WHY this is not just a rigid body: a rigid body with a character's shape does
// all the things a character must not. It tips over. It slides down ramps it
// should stand on. It bounces. It catches on the lip of every step. It
// conserves momentum, so walking into a wall and stopping means an impulse went
// somewhere. And it cannot be told "go here" -- only "here is a force", which
// is not how a player's input works.
//
// So the controller does not participate in the solver at all. It moves where
// it is told, discovers what is in the way, and slides along it -- which is
// the behaviour every game wants and no physically correct simulation produces.
//
// It reads the physics world and never writes to it. A character that pushed
// crates would need to apply impulses back, and that is a separate decision
// with its own tuning; this one walks around them.
//
// KNOWN COST: it is discrete, not swept. Each move happens, then discovers what
// it overlapped, then is pushed back out, and only the residue is progress.
// That is fine for walking and sliding, and it is why mounting a kerb takes
// about a second at walking pace rather than the few frames a swept controller
// needs -- the capsule has to advance its own radius before its underside is
// over the edge, and it only gains a fraction of each attempt. Substepping
// bounds the other consequence, which is passing through thin geometry at
// speed; a 5 cm wall holds at 62 m/s.
#pragma once

#include <cstdint>

#include "engine/core/math.h"
#include "engine/physics/physics.h"

namespace eng::physics {

struct CharacterConfig {
    float radius = 0.35f;
    // Feet to head, INCLUDING the caps. A capsule of total height h has a
    // segment of h - 2r, so a height below twice the radius is a sphere and is
    // clamped to one rather than producing a negative segment.
    float height = 1.8f;

    // The tallest lip the character walks over without jumping. Without it a
    // kerb, a doorway threshold or the seam between two floor tiles stops
    // someone dead, and the seam is the one that gets reported as a bug.
    float step_height = 0.35f;

    // The steepest surface that counts as GROUND rather than as a wall. Above
    // it the character slides back down instead of standing. Without a limit,
    // a character walks up anything short of vertical.
    float slope_limit_degrees = 50.0f;

    // How far outside the surface the capsule is kept. Resolving to exactly
    // touching leaves the next frame's query finding an overlap again from
    // floating-point noise, and the character buzzes.
    float skin = 0.015f;

    // Which layers it collides with. A character that collides with its own
    // trigger volumes, or with the projectiles it fires, is the first thing
    // anyone hits.
    std::uint32_t mask = 0xFFFFFFFFu;
};

class CharacterController {
  public:
    explicit CharacterController(const CharacterConfig& config = {});

    // The bottom of the capsule -- where the feet are, which is what a level
    // designer places and what an animation is authored around. The centre is
    // available too but is a worse thing to build on: it moves when the
    // character's height changes.
    void Teleport(Vec3 feet);
    [[nodiscard]] Vec3 Feet() const { return feet_; }
    [[nodiscard]] Vec3 Centre() const;

    // Moves by `displacement`, sliding along whatever it meets, stepping over
    // anything shorter than step_height, and refusing to climb anything steeper
    // than the slope limit.
    //
    // The motion is SUBSTEPPED so that no single move exceeds the capsule's
    // radius: this is a discrete move-and-resolve, and a discrete move longer
    // than the shape is how a character passes through a wall.
    void Move(const World&, Vec3 displacement);

    [[nodiscard]] bool Grounded() const { return grounded_; }
    // Touching something from below, whether or not it is walkable. A character
    // wedged on a steep edge is supported but not grounded -- which is the
    // difference between "cannot fall" and "can walk".
    [[nodiscard]] bool Supported() const { return supported_; }
    // Only meaningful while grounded. The surface actually stood on, so a
    // caller can align a character to a ramp or work out which way is downhill.
    [[nodiscard]] Vec3 GroundNormal() const { return ground_normal_; }
    [[nodiscard]] int GroundBody() const { return ground_body_; }
    // How far the last Move actually travelled. Less than it was asked for
    // whenever something was in the way, which is what a caller needs to know
    // to stop playing a walk cycle against a wall.
    [[nodiscard]] Vec3 LastMotion() const { return last_motion_; }

    [[nodiscard]] const CharacterConfig& Config() const { return config_; }
    [[nodiscard]] Shape Capsule() const;

  private:
    // Pushes the capsule out of anything it overlaps, and records the best
    // ground it found doing so. Returns how many bodies it had to resolve.
    int Depenetrate(const World&);
    // Nudges down to find out whether there is a floor. Needed because a
    // character standing on one does not overlap it.
    bool ProbeGround(const World&);

    CharacterConfig config_;
    Vec3 feet_{0.0f, 0.0f, 0.0f};
    Vec3 last_motion_{0.0f, 0.0f, 0.0f};
    Vec3 ground_normal_{0.0f, 1.0f, 0.0f};
    int ground_body_ = -1;
    bool grounded_ = false;
    // Whether the last Depenetrate was pushed back by something too steep to
    // stand on. A walkable ramp blocks horizontal motion too, and stepping over
    // one instead of walking up it is a different and much worse behaviour.
    bool hit_wall_ = false;
    // Whether anything at all is holding the capsule up, walkable or not. The
    // step-up needs this rather than grounded_: a capsule part-way up a kerb
    // rests on the kerb's EDGE, which is far too steep to stand on.
    bool supported_ = false;
    // How much more height step-ups may borrow before the character has to
    // find walkable ground. See the comment at the step-up.
    float climb_budget_ = 0.0f;
};

}  // namespace eng::physics
