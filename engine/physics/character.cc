#include "engine/physics/character.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace eng::physics {
namespace {

// How many times a single substep will try to push itself free. A character
// wedged into a corner is being pushed by two surfaces at once and resolving
// one re-enters the other, so it takes a few passes; it does not take twenty,
// and a wedge that will not resolve must not become an infinite loop.
constexpr int kMaxDepenetrations = 6;

}  // namespace

CharacterController::CharacterController(const CharacterConfig& config)
    : config_(config) {
    config_.radius = std::max(config_.radius, 1e-3f);
    // A capsule shorter than two radii is a sphere. Clamping here rather than
    // rejecting means a caller who sets height and forgets radius gets a small
    // character instead of a negative segment length and a support function
    // that returns points inside itself.
    config_.height = std::max(config_.height, 2.0f * config_.radius);
    climb_budget_ = config_.step_height;
}

Shape CharacterController::Capsule() const {
    return Shape::MakeCapsule(config_.radius,
                              config_.height * 0.5f - config_.radius);
}

Vec3 CharacterController::Centre() const {
    return feet_ + Vec3{0.0f, config_.height * 0.5f, 0.0f};
}

void CharacterController::Teleport(Vec3 feet) {
    feet_ = feet;
    climb_budget_ = config_.step_height;
    last_motion_ = Vec3{0, 0, 0};
    grounded_ = false;
    ground_body_ = -1;
}

int CharacterController::Depenetrate(const World& world) {
    const Shape capsule = Capsule();
    const float cos_limit =
        std::cos(config_.slope_limit_degrees * 3.14159265f / 180.0f);

    Body probe;
    probe.shape = capsule;
    probe.inverse_mass = 0.0f;

    // These are SET here and never cleared here. Clearing them would throw away
    // the footing established before the move: a character standing on a floor
    // does not overlap it, so the only thing this call discovers is the
    // obstacle in front -- and for a step tall enough to be met by the
    // capsule's widest point, that contact is horizontal and says nothing about
    // support. The character then cannot step up onto anything taller than its
    // own radius, which is most of what a step height is for.
    int resolved = 0;
    for (int pass = 0; pass < kMaxDepenetrations; ++pass) {
        probe.position = Centre();

        std::vector<int> hits;
        QueryFilter filter;
        filter.mask = config_.mask;
        world.OverlapShape(capsule, probe.position, Quat{}, &hits, filter);
        if (hits.empty()) break;

        bool moved = false;
        for (int index : hits) {
            const Body& other = world[index];
            Contact c;
            // CollideAny, not CollideConvex: a height field is not convex, and
            // asking GJK about one returns nothing -- so a character on terrain
            // would find no contact, never be grounded, and walk through the
            // ground with no error anywhere.
            if (!CollideAny(probe, other, &c)) continue;
            if (c.depth <= 0.0f) continue;

            // The normal comes back pointing from the probe toward the body, so
            // pushing OUT is along its negation.
            const Vec3 out = c.normal * -1.0f;
            // The skin is added to the push rather than subtracted from it: the
            // aim is to end up clear of the surface, not exactly on it, because
            // exactly on it is an overlap again next frame.
            feet_ = feet_ + out * (c.depth + config_.skin);
            probe.position = Centre();
            moved = true;
            ++resolved;

            // GROUND is whatever pushed us upward within the slope limit. Not
            // "the thing below us": on a ramp the surface below is a ramp, and
            // its normal is what decides whether this is a floor to stand on or
            // a wall to slide down.
            // SUPPORTED means something is holding us up at all, walkable or
            // not. It is a different question from grounded_, and the step-up
            // needs this one: a capsule part-way up a kerb is resting on the
            // kerb's edge, whose normal is far too steep to stand on -- so
            // gating the step-up on grounded_ stops the climb half way, with
            // the character stuck against the step at the height it reached.
            if (out.y > 0.1f) supported_ = true;
            if (out.y > cos_limit) {
                if (out.y > ground_normal_.y * 0.999f) {
                    grounded_ = true;
                    ground_normal_ = Normalize(out);
                    ground_body_ = index;
                }
            } else {
                // A WALL: too steep to stand on. Recorded separately because it
                // is what distinguishes "step over this" from "walk up this".
                // A walkable ramp also blocks horizontal motion -- that is what
                // walking uphill IS -- and treating that as an obstacle to step
                // over makes the character hop up every slope instead of
                // climbing it, which measured as a 45 degree ramp climbing to
                // 0.72 m instead of 2.42.
                hit_wall_ = true;
            }
        }
        if (!moved) break;
    }
    return resolved;
}

bool CharacterController::ProbeGround(const World& world) {
    // Nudge down, see what pushes back, and put the horizontal position back
    // exactly. A character resting on a surface is NOT overlapping it -- the
    // skin keeps it clear -- so nothing is found by asking where it is. The
    // only way to know there is a floor is to try to move into it.
    const Vec3 saved = feet_;
    const float reach = config_.skin * 4.0f;
    feet_ = feet_ - Vec3{0.0f, reach, 0.0f};
    Depenetrate(world);
    // Vertical only. Letting the probe move us sideways would be a second,
    // unasked-for slide every frame -- downhill, forever.
    feet_ = Vec3{saved.x, std::max(feet_.y, saved.y - reach), saved.z};
    if (!grounded_) feet_ = saved;
    return grounded_;
}

void CharacterController::Move(const World& world, Vec3 displacement) {
    const Vec3 start = feet_;
    grounded_ = false;
    supported_ = false;
    hit_wall_ = false;
    ground_normal_ = Vec3{0.0f, 0.0f, 0.0f};
    ground_body_ = -1;

    // Establish footing BEFORE moving, because the step-up below is only
    // allowed from the ground and would otherwise never fire: a character
    // walking on a floor is not overlapping it, so nothing in the substep loop
    // discovers the floor, and grounded_ stays false until the probe at the
    // very end. The symptom is a character who is unable to walk over a kerb
    // while walking, and steps over it perfectly the moment they land on it.
    ProbeGround(world);

    // SUBSTEPPED so no single move exceeds the capsule's radius. This is a
    // discrete move-and-resolve: it moves first and asks questions afterwards,
    // and a move longer than the shape can end up on the far side of a wall
    // with nothing overlapping to push it back.
    const float distance = Length(displacement);
    const int steps =
        std::clamp(int(std::ceil(distance / (config_.radius * 0.5f))), 1, 16);
    Vec3 remaining = displacement;

    for (int step = 0; step < steps; ++step) {
        Vec3 move = remaining * (1.0f / float(steps - step));
        remaining = remaining - move;

        const Vec3 before = feet_;
        feet_ = feet_ + move;
        // Per substep, unlike grounded_ and supported_: "was I blocked by a
        // wall just now" is a question about this move, not about the state the
        // character is in.
        hit_wall_ = false;
        const int hit = Depenetrate(world);
        if (hit == 0) continue;

        // How much of this substep actually survived. If a wall took most of
        // it, try going over instead of through.
        const Vec3 achieved = feet_ - before;
        const bool horizontal_block =
            hit_wall_ && Length(Vec3{move.x, 0.0f, move.z}) > 1e-5f &&
            Length(Vec3{achieved.x, 0.0f, achieved.z}) <
                Length(Vec3{move.x, 0.0f, move.z}) * 0.5f;

        // THE STEP-UP, and it is a separate manoeuvre rather than a taller
        // slope limit: a kerb is vertical, so no slope limit that lets a
        // character over one would stop them walking up a wall.
        //
        // Lift, move, drop -- then keep whichever of the two attempts got
        // FURTHER. Deciding in advance whether the step-up "worked" needs a
        // rule for what landing on a step looks like, and every such rule is
        // wrong somewhere: a capsule coming up onto a kerb rests on its EDGE,
        // and an edge contact has a normal 68 degrees off vertical, which any
        // honest slope limit calls a wall and refuses to stand on. Comparing
        // outcomes needs no such rule. Before this the step-up ran, cleared,
        // dropped, was pushed back out by the edge it had just climbed, and
        // left the character at exactly the step's face minus its radius.
        //
        // Only from the ground: stepping up in mid-air is how a character
        // climbs a wall by jumping at it repeatedly.
        // A CLIMB BUDGET, and it is what separates a kerb from a ramp.
        //
        // Neither is distinguishable in a single frame: at walking speed a 70
        // degree slope rises 9 cm per frame, well inside any step height, so
        // "did stepping up get me further" says yes to both and the character
        // walks up the wall. The difference is cumulative -- a kerb stops
        // rising once you are on it, and a slope does not.
        //
        // So a step-up may lift the character by at most step_height IN TOTAL
        // before it has to find walkable ground, and finding it refills the
        // budget. Getting onto a 30 cm kerb costs 30 cm of a 35 cm budget and
        // then refills; a 70 degree ramp spends it in four frames and stops.
        // Height rather than a frame count, so it does not change with the
        // frame rate.
        if (horizontal_block && supported_ && config_.step_height > 0.0f &&
            climb_budget_ > 1e-4f) {
            const Vec3 unstepped = feet_;
            const bool unstepped_supported = supported_;
            const bool unstepped_grounded = grounded_;
            const Vec3 unstepped_normal = ground_normal_;
            const int unstepped_body = ground_body_;

            feet_ = before + Vec3{0.0f, config_.step_height, 0.0f};
            Depenetrate(world);
            // HEADROOM: if lifting alone put us inside something, there is no
            // step to take and the attempt is over before it starts.
            //
            // An EARLY-OUT rather than a correctness guard, and measurably so:
            // removing it changes no answer, because the outcome comparison at
            // the bottom rejects a step-up that ended worse anyway. It saves
            // two depenetration passes every time a character walks into a
            // wall, which is often.
            const bool has_headroom =
                std::fabs(feet_.y - (before.y + config_.step_height)) < 1e-3f;
            if (has_headroom) {
                feet_ = feet_ + Vec3{move.x, 0.0f, move.z};
                Depenetrate(world);
                // Down, in small increments, stopping at the FIRST thing met.
                // Dropping the whole height at once lands the capsule beside
                // the step rather than on it, so the deepest overlap is the
                // step's vertical face and depenetration pushes it straight
                // back out.
                grounded_ = false;
                ground_normal_ = Vec3{0.0f, 0.0f, 0.0f};
                ground_body_ = -1;
                constexpr int kDrops = 8;
                for (int d = 0; d < kDrops; ++d) {
                    const Vec3 above = feet_;
                    feet_ = feet_ - Vec3{0.0f, config_.step_height / kDrops, 0.0f};
                    if (Depenetrate(world) > 0) {
                        // Landed. Do not let the landing shove us sideways --
                        // that is the edge pushing back, and it would undo the
                        // progress the lift just bought.
                        feet_ = Vec3{above.x, feet_.y, above.z};
                        break;
                    }
                }
            }

            // Whichever went further along the direction asked for. A step-up
            // that made things worse loses and is discarded.
            const Vec3 want = Normalize(Vec3{move.x, 0.0f, move.z});
            const float stepped_gain = Dot(feet_ - before, want);
            const float plain_gain = Dot(unstepped - before, want);
            if (!has_headroom || stepped_gain <= plain_gain + 1e-4f) {
                feet_ = unstepped;
                supported_ = unstepped_supported;
                grounded_ = unstepped_grounded;
                ground_normal_ = unstepped_normal;
                ground_body_ = unstepped_body;
            } else {
                climb_budget_ -= std::max(0.0f, feet_.y - before.y);
            }
        }
    }

    // And again at the end, because the last substep may have left the
    // character clear of everything: walking off a kerb leaves them a
    // centimetre above the floor, and reporting "not grounded" there makes them
    // fall for a frame and play a landing animation on the flat.
    if (!grounded_) ProbeGround(world);

    // Walkable ground refills the budget. Anything else -- an edge, a steep
    // face, thin air -- does not, so a character can climb one step and then
    // must stand somewhere before climbing another.
    if (grounded_) climb_budget_ = config_.step_height;
    last_motion_ = feet_ - start;
}

}  // namespace eng::physics
