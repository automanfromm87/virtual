// Pure C++20. Rigid body dynamics and collision.
//
// This is the line between a renderer and an engine: nothing above here has any
// notion of things affecting each other, and no amount of shading adds one.
//
// SCOPE, chosen so the whole thing stays verifiable on the CPU:
//   * Spheres and ORIENTED boxes against each other and against static geometry.
//   * Impulse resolution with restitution and Coulomb friction, applied at the
//     contact POINT, so friction produces torque and a ball rolls instead of
//     sliding.
//   * Linear and ANGULAR motion. Semi-implicit Euler at a FIXED timestep,
//     decoupled from the frame rate.
//   * Distance and ball-socket joints.
//   * Sleeping, so a settled pile stops costing anything.
//   * A BVH broadphase, rebuilt per step, shared with the queries.
// Not here: convex hulls, continuous collision.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "engine/core/math.h"
#include "engine/geometry/hull.h"
#include "engine/physics/bvh.h"

namespace eng::physics {

enum class ShapeType : std::uint8_t { Sphere, Box, Hull, Capsule };

struct Shape {
    ShapeType type = ShapeType::Sphere;
    // Sphere: the radius. Capsule: the radius of its caps. Meaningless for a
    // box or a hull.
    float radius = 0.5f;
    // The BOUNDING sphere, which the cheap rejects in raycasts, overlap
    // queries and the CCD broadphase all read without asking what kind of
    // shape it is.
    //
    // A separate field, and it was not at first: `radius` served as both, which
    // works right up to a capsule -- whose cap radius and bounding radius are
    // different numbers, and which would have had to lie about one of them.
    // Each factory below sets it; a shape built by hand must too.
    float bounds_radius = 0.5f;
    Vec3 half_extents{0.5f, 0.5f, 0.5f};  // Box, in the body's own frame
    // Hull: the vertices, in the body's own frame, already reduced by
    // geom::ConvexHull. Only the vertices are kept -- GJK never asks anything
    // else of a convex shape, and carrying the faces here would mean keeping
    // them in step with a shape that never changes.
    //
    // They are stored RE-CENTRED on the hull's centre of mass, because a body
    // rotates about its centre of mass and nothing else. `centre_offset` is
    // where that centre was in the vertices as supplied, so a caller can put
    // the body where the art expects it.
    std::vector<Vec3> points;
    Vec3 centre_offset{0.0f, 0.0f, 0.0f};
    // Inertia per unit MASS, about the centre of mass, in the body frame.
    // Negative x means "not known", and SetMass falls back to the bounding box.
    Vec3 unit_inertia{-1.0f, -1.0f, -1.0f};

    [[nodiscard]] static Shape MakeSphere(float r) {
        Shape s;
        s.type = ShapeType::Sphere;
        s.radius = r;
        s.bounds_radius = r;
        return s;
    }
    [[nodiscard]] static Shape MakeBox(Vec3 half) {
        Shape s;
        s.type = ShapeType::Box;
        s.half_extents = half;
        // The BOUNDING SPHERE, which every cheap reject in the engine reads off
        // `radius` regardless of shape -- MakeHull has always set it and this
        // did not, so a box carried the default 0.5 whatever its size.
        //
        // It was invisible because the two places that use it both happened to
        // survive: continuous collision's broadphase reject only ever ran
        // against small bullets, and nothing else asked. A raycast asks
        // immediately, and a ray passing 1.5 m above the centre of a 4 m box
        // was rejected before the exact test ever ran.
        s.bounds_radius = Length(half);
        return s;
    }
    // From a built hull, which is the form that also knows its own volume and
    // inertia. Preferred over the raw-vertex overload for anything that will
    // actually tumble.
    [[nodiscard]] static Shape MakeHull(const geom::Hull& hull);

    // `vertices` must already be a hull. Passing a raw cloud is not wrong --
    // the support function of a cloud and of its hull are identical -- but
    // every interior point is then paid for on every query, forever.
    //
    // Mass properties fall back to the bounding box, which OVERESTIMATES the
    // inertia of anything that is not a box: the shape resists spinning more
    // than it should. Use the geom::Hull overload where that matters.
    // A CAPSULE: a segment of length 2*half_height along the body's own Y,
    // swept by a sphere of `r`. The shape a character is, because a box catches
    // on every corner it passes and a sphere rolls.
    [[nodiscard]] static Shape MakeCapsule(float r, float half_height) {
        Shape s;
        s.type = ShapeType::Capsule;
        s.radius = r;
        // Stored in half_extents so the whole shape is described by the fields
        // that already exist; y is the SEGMENT's half length, not including the
        // caps, which is what the support function wants.
        s.half_extents = Vec3{r, half_height, r};
        s.bounds_radius = half_height + r;
        return s;
    }

    [[nodiscard]] static Shape MakeHull(std::vector<Vec3> vertices) {
        Shape s;
        s.type = ShapeType::Hull;
        s.points = std::move(vertices);
        // Kept in step so that broadphase and culling do not need to know
        // which kind of shape they are looking at.
        float r = 0.0f;
        for (const Vec3& p : s.points) r = std::max(r, Length(p));
        s.radius = r;
        Vec3 half{0, 0, 0};
        for (const Vec3& p : s.points) {
            half.x = std::max(half.x, std::fabs(p.x));
            half.y = std::max(half.y, std::fabs(p.y));
            half.z = std::max(half.z, std::fabs(p.z));
        }
        s.half_extents = half;
        return s;
    }
};

struct Body {
    Shape shape;
    Vec3 position{0.0f, 0.0f, 0.0f};
    Quat orientation;  // body -> world
    Vec3 velocity{0.0f, 0.0f, 0.0f};
    Vec3 angular_velocity{0.0f, 0.0f, 0.0f};  // world frame, rad/s, axis-angle

    // Zero mass means STATIC: infinite inertia, never moved by a contact. The
    // floor is a body like any other, it just has no give.
    float inverse_mass = 1.0f;
    // Diagonal of the inverse inertia tensor in the BODY's frame. Diagonal is
    // exact rather than an approximation: a sphere is isotropic, and a box's
    // principal axes are its own axes. Zero on an axis locks rotation about it,
    // which is how a static body and a rotation constraint are both expressed.
    Vec3 inverse_inertia{0.0f, 0.0f, 0.0f};

    // CONTINUOUS collision for this body. Off by default and deliberately so:
    // it costs a sweep against every candidate, and for anything moving less
    // than its own size per step the ordinary discrete test already sees the
    // overlap. Turn it on for the things that do not -- a bullet, a thrown
    // object, anything small and fast -- where the discrete test samples the
    // start and the end of the step and the wall was only ever in between.
    bool bullet = false;

    // A BIT MASK, for filtering queries and (later) collisions. One bit per
    // layer: the player, the world, projectiles, triggers. A raycast that
    // cannot be told to ignore the shooter hits the shooter, and every game
    // that has ever been written needed that on the first day.
    std::uint32_t layer = 1u;

    // A TRIGGER generates contacts and no impulses: things pass through it, and
    // the world reports that they did. A door's threshold, a checkpoint, a
    // damage volume. Without it the only way to ask "is the player in this
    // region" is to test it by hand every frame against every region.
    bool trigger = false;

    float restitution = 0.35f;  // 0 = dead drop, 1 = never loses energy
    float friction = 0.5f;

    // SLEEPING. A body that has been nearly still for long enough stops being
    // integrated at all. Not only a saving: a resting stack never settles
    // exactly, and the residual jitter is what makes a pile of crates shiver
    // forever. Freezing it is how that stops looking wrong.
    //
    // Woken by any contact with a moving body, and by anything that moves it
    // from outside — see World::Wake.
    bool sleeping = false;
    float still_for = 0.0f;  // seconds below the threshold
    // Opaque to the physics; the caller uses it to find its own object again.
    std::uint32_t user = 0;

    // Fills inverse_mass and inverse_inertia from a mass and the current shape.
    // mass <= 0 makes the body static. Call after setting `shape`.
    void SetMass(float mass);

    // The inverse inertia tensor rotated into world space, applied to `v`.
    // R * diag * Rᵀ * v, without ever materialising the matrix.
    [[nodiscard]] Vec3 ApplyInverseInertia(Vec3 v) const;

    [[nodiscard]] bool IsStatic() const { return inverse_mass <= 0.0f; }

    // Velocity of the material point at world position `p`. Not the same as
    // `velocity` once a body spins, and using the wrong one is why a rolling
    // ball would keep sliding.
    [[nodiscard]] Vec3 PointVelocity(Vec3 p) const {
        return velocity + Cross(angular_velocity, p - position);
    }
};

struct Contact {
    int a = -1, b = -1;
    Vec3 normal{0.0f, 1.0f, 0.0f};  // unit, points from a toward b
    Vec3 point{0.0f, 0.0f, 0.0f};   // world space, on the overlap
    float depth = 0.0f;             // positive when overlapping
};

// A pair of bodies starting, continuing or stopping touching.
//
// The transitions are what gameplay wants and what a contact list cannot give:
// "the player ENTERED the checkpoint" is a different event from "the player is
// still in the checkpoint", and deriving one from the other means every caller
// keeping its own record of last frame.
enum class TouchPhase : std::uint8_t { Begin, Stay, End };

struct TouchEvent {
    int a = -1, b = -1;
    TouchPhase phase = TouchPhase::Begin;
    // Meaningless for End: the bodies are apart, so there is no contact point
    // to report. Zeroed rather than left stale, because a stale one from two
    // steps ago is the kind of value that looks usable.
    Vec3 point{0.0f, 0.0f, 0.0f};
    Vec3 normal{0.0f, 1.0f, 0.0f};
};

// Where a ray met a body.
struct RayHit {
    int body = -1;  // index into the world, or -1 for nothing
    float t = 0.0f;  // distance along the ray, in the direction's own units
    Vec3 point{0.0f, 0.0f, 0.0f};
    // Points OUT of the surface that was hit, so a bullet decal or a bounce
    // uses it directly. Undefined when the ray starts inside the body.
    Vec3 normal{0.0f, 1.0f, 0.0f};
    [[nodiscard]] bool Hit() const { return body >= 0; }
};

// What a query is allowed to see.
struct QueryFilter {
    // A body is considered when `body.layer & mask` is non-zero.
    std::uint32_t mask = 0xFFFFFFFFu;
    // Skipped outright. The common case is the body doing the querying: a
    // character casting a ray to find the ground must not find itself.
    int ignore = -1;
    // Triggers are invisible to queries by DEFAULT. A ray looking for a wall
    // to stop a bullet should not stop at a checkpoint volume; a query that
    // wants to find triggers asks for them.
    bool hit_triggers = false;
};

struct WorldStats {
    int steps = 0;
    int contacts = 0;
    // Pairs the narrowphase was actually asked about. With the BVH this is the
    // number that shows whether the broadphase is doing its job: on a scattered
    // scene it is a small multiple of the body count, and if it ever approaches
    // N^2/2 the tree has degenerated into a list.
    int pairs_tested = 0;
    // How many times a bullet's step was cut short by an impact. Zero on a
    // scene with nothing fast in it, and the number to look at when something
    // fast still goes through a wall.
    int toi_clamps = 0;
    // Broadphase tree shape. `bvh_depth` against log2(bodies/4) is the health
    // check: equal means balanced, much larger means the split has stopped
    // separating things and the traversal is walking a list.
    int bvh_nodes = 0;
    int bvh_depth = 0;
    // How many times the tree has been rebuilt. One per step plus one per
    // query batch that follows an external mutation; a number climbing much
    // faster than the step count means something is dirtying the tree between
    // every query, which turns each query into an O(N log N) build.
    int bvh_rebuilds = 0;
};

// A body's world-space bounding box, expanded by `margin`.
//
// Exposed because the broadphase, the queries and the CCD sweep all need the
// same answer, and three copies of "what is the AABB of a rotated capsule"
// would be three chances to write it differently.
[[nodiscard]] Aabb BodyBounds(const Body&, float margin = 0.0f);

// A constraint between two bodies, solved alongside the contacts.
//
// Distance and ball-socket only. Both are the same equation — hold two anchor
// points a fixed distance apart — with a ball socket being the case where that
// distance is zero, which is why they are one type rather than two.
struct Joint {
    int a = -1, b = -1;
    // Anchors in each body's OWN frame, so they follow it as it turns.
    Vec3 anchor_a{0.0f, 0.0f, 0.0f};
    Vec3 anchor_b{0.0f, 0.0f, 0.0f};
    // Distance to hold. Zero is a ball socket: the two anchors coincide.
    float distance = 0.0f;
    // A rope rather than a rod: pulls when stretched, does nothing when slack.
    bool rope = false;
    // Fraction of the remaining positional error corrected per step. Below 1
    // the joint is springy; at 1 it is rigid and fights the contact solver.
    float stiffness = 0.4f;
};

class World {
  public:
    // Gravity in metres per second squared, and the fixed step the solver runs
    // at. A fixed step is not a detail: variable-dt integration makes
    // restitution and penetration recovery frame-rate dependent, so the same
    // scene behaves differently on a faster machine.
    Vec3 gravity{0.0f, -9.81f, 0.0f};
    float fixed_dt = 1.0f / 120.0f;
    int solver_iterations = 8;
    // Fraction of remaining penetration pushed out per iteration, and the slop
    // left in deliberately. Resolving to exactly zero makes resting contacts
    // jitter, because floating point puts them back in contact next frame.
    float penetration_slop = 0.005f;
    float penetration_correction = 0.4f;
    // Velocity below which restitution is dropped. Without it a resting body
    // bounces forever on the numerical noise of its own weight.
    float restitution_threshold = 1.0f;

    // Velocity lost per second, as a rate. Zero by DEFAULT, deliberately: with
    // damping on, none of the conservation laws this module is tested against
    // hold, and a default fudge factor is the kind of thing that quietly makes
    // a physics engine untestable.
    //
    // But a perfectly rigid sphere on a perfectly rigid plane never stops
    // rolling — nothing in the model deforms, so there is nothing to dissipate.
    // Real balls stop. Anything that wants that behaviour turns angular_damping
    // up and is choosing plausibility over the conservation law, knowingly.
    float linear_damping = 0.0f;
    float angular_damping = 0.0f;
    // A master switch for continuous collision, so it can be turned off
    // wholesale to see what it was costing -- or what it was hiding.
    bool ccd_enabled = true;

    // Below these speeds a body counts as still. After `sleep_after` seconds of
    // that, it sleeps. Zero disables sleeping entirely, which is what the
    // conservation tests want: a sleeping body stops conserving anything.
    float sleep_linear = 0.06f;
    float sleep_angular = 0.12f;
    float sleep_after = 0.6f;

    int Add(const Body& b);
    // Wakes a body and resets its stillness timer. Anything that moves a body
    // from outside the solver — teleporting it, hitting it, changing gravity —
    // has to call this, or it stays frozen where it was.
    void Wake(int i);
    void WakeAll();
    [[nodiscard]] int SleepingCount() const;
    // The NON-CONST accessor invalidates the broadphase, on the assumption that
    // a caller who asked for a mutable body is about to move it. That is
    // pessimistic -- reading through it costs a rebuild too -- and it is the
    // only sound rule available: this class hands out a raw reference, so there
    // is no other moment at which it could learn that a body moved. The
    // alternative, a Dirty() the caller must remember to call, fails silently
    // and intermittently, which is the worst way for a spatial index to be
    // wrong.
    //
    // Reading through a `const World&` -- which is what every query and the
    // character controller take -- does not invalidate anything.
    [[nodiscard]] Body& operator[](int i) {
        bvh_dirty_ = true;
        return bodies_[std::size_t(i)];
    }
    [[nodiscard]] const Body& operator[](int i) const { return bodies_[std::size_t(i)]; }
    [[nodiscard]] int Count() const { return int(bodies_.size()); }
    void Clear();

    int AddJoint(const Joint& j);
    [[nodiscard]] int JointCount() const { return int(joints_.size()); }
    [[nodiscard]] const Joint& GetJoint(int i) const { return joints_[std::size_t(i)]; }

    // Advances by `dt` in whole fixed steps, carrying the remainder to the next
    // call. Returns how many steps ran.
    int Step(float dt);
    // One fixed step. Exposed so tests can drive it deterministically.
    void StepFixed();

    // The contacts the SOLVER will resolve. Trigger pairs are not here: they
    // are reported through Touches() and never produce an impulse.
    [[nodiscard]] const std::vector<Contact>& Contacts() const { return contacts_; }

    // Every pair that began, continued or stopped touching during the last
    // step -- triggers and solid contacts alike. A solid collision beginning is
    // as useful as a trigger being entered: it is when the impact sound plays.
    [[nodiscard]] const std::vector<TouchEvent>& Touches() const { return touch_events_; }

    // --- queries ---------------------------------------------------------------
    //
    // The half of a physics engine gameplay is actually built on. Simulation
    // answers "where does everything end up"; queries answer "what is under the
    // crosshair", "is there ground beneath my feet", "who is inside the blast
    // radius" -- and nothing above can be asked without them.
    //
    // `direction` need not be normalised; `t` comes back in its units, so a
    // direction scaled to the ray's length makes `t` a fraction from 0 to 1.
    [[nodiscard]] bool Raycast(Vec3 origin, Vec3 direction, float max_distance,
                               RayHit* out, const QueryFilter& = {}) const;

    // Every body overlapping a sphere or a box, appended to `out`. Returns how
    // many were added. The box is axis-aligned; an oriented one is an overlap
    // test against a temporary body, which is what OverlapShape is for.
    int OverlapSphere(Vec3 centre, float radius, std::vector<int>* out,
                      const QueryFilter& = {}) const;
    int OverlapBox(Vec3 centre, Vec3 half_extents, std::vector<int>* out,
                   const QueryFilter& = {}) const;
    // The general form: anything the narrowphase understands, at any pose.
    int OverlapShape(const Shape&, Vec3 position, Quat orientation,
                     std::vector<int>* out, const QueryFilter& = {}) const;
    [[nodiscard]] const WorldStats& Stats() const { return stats_; }

    // Kinetic (linear + rotational) plus gravitational potential energy,
    // relative to y = 0. Tests use it: a solver that injects energy is the
    // classic instability, and it shows up here long before it is visible.
    [[nodiscard]] float Energy() const;
    // Total angular momentum about the origin. Conserved when nothing external
    // exerts a torque, which is the sharpest available check on whether the
    // inertia tensor and the contact impulses agree.
    [[nodiscard]] Vec3 AngularMomentum() const;

  private:
    void Collide();
    void Resolve();

    void SolveJoints();
    void UpdateSleep();

    // Rebuilds the broadphase if anything has invalidated it. Const because
    // every query needs it and a query does not conceptually change the world;
    // the tree is a cache of the bodies, not part of their state.
    void RefreshBroadphase() const;

    std::vector<Body> bodies_;
    std::vector<Joint> joints_;
    std::vector<Contact> contacts_;
    // Trigger pairs, kept apart so the solver cannot see them. Filtering them
    // out inside Resolve would work until something else iterated contacts_
    // and forgot to.
    std::vector<Contact> trigger_contacts_;
    std::vector<TouchEvent> touch_events_;
    // Pairs touching as of the last step, packed and sorted, so this step's set
    // can be differenced against it. Sorted rather than hashed because the
    // difference is the whole operation and a linear merge is the cheapest way
    // to do it.
    std::vector<std::uint64_t> touching_;
    std::vector<std::uint64_t> touching_next_;
    void BuildTouchEvents();
    // Contacts touching each body this step. Kept as a member so a steady-state
    // step allocates nothing.
    // How many contacts each body is in this step, so the positional
    // correction can be divided between them. Named for what it holds:
    // it sat next to touch_events_ as "touches_" and the two are not
    // remotely the same thing.
    std::vector<int> contacts_per_body_;
    // Union-find parents for the sleeping islands. A member so a settled scene
    // allocates nothing per step.
    std::vector<int> islands_;
    // MUTABLE only so a const query can record that it rebuilt the tree. The
    // simulation half of these is written from non-const code as before.
    mutable WorldStats stats_;
    float accumulator_ = 0.0f;

    // --- broadphase ------------------------------------------------------------
    //
    // MUTABLE, and the const-correctness argument is worth stating because it
    // is the usual excuse for a bug: the tree carries no information that is not
    // already in bodies_. Rebuilding it changes what this object COMPUTES WITH
    // and not what it MEANS, which is exactly the case mutable is for. A query
    // that had to be non-const to work would push the problem to every caller,
    // and CharacterController -- which takes a `const World&` precisely so it
    // cannot disturb the simulation -- could not run one at all.
    mutable Bvh bvh_;
    mutable std::vector<Aabb> body_boxes_;
    mutable bool bvh_dirty_ = true;
    // Scratch for the broadphase's per-body query, so a steady-state step does
    // not allocate. Mutable for the same reason as the tree.
    mutable std::vector<int> query_scratch_;
    // Broadphase pairs, packed as (low << 32 | high) so one sort orders them
    // the way the old double loop produced them. A member so a steady-state
    // step allocates nothing.
    std::vector<std::uint64_t> pairs_;
    // The margin every body's box is grown by when the tree is built. It has to
    // cover a step's worth of motion, because the tree built at the START of a
    // step is queried against positions the integrator has ALREADY advanced.
    // Without it a body moving 30 cm per step can leave its own leaf box and
    // stop being reported against things it is now overlapping.
    float broadphase_margin_ = 0.05f;
};

// Narrowphase, exposed for testing. Returns false when the pair is apart.
// Every one of these respects Body::orientation.
[[nodiscard]] bool CollideSphereSphere(const Body& a, const Body& b, Contact* out);
[[nodiscard]] bool CollideSphereBox(const Body& sphere, const Body& box, Contact* out);
[[nodiscard]] bool CollideBoxBox(const Body& a, const Body& b, Contact* out);

// The general pair, for anything involving a hull.
//
// GJK decides whether two convex shapes touch by asking only "which point of
// you is furthest in this direction" -- so it needs no case analysis at all,
// and one implementation covers hull-hull, hull-box and hull-sphere. EPA then
// grows a polytope inside the Minkowski difference until it reaches the
// boundary, which is where the penetration depth and normal come from.
//
// Slower than the specialised sphere and box routines, and they are kept for
// exactly that reason: a scene is mostly spheres and boxes, and this is the
// path for the shapes that are neither.
[[nodiscard]] bool CollideConvex(const Body& a, const Body& b, Contact* out);

// The furthest point of `body` along `dir`, in WORLD space. Exposed because it
// is the whole interface GJK has to a shape, and a test that can call it
// directly can check a shape's support function without going through a
// collision at all.
[[nodiscard]] Vec3 Support(const Body& body, Vec3 dir);

// The distance between two convex bodies, and the nearest points on each.
// Zero when they overlap -- the depth is CollideConvex's job, and conflating
// the two would mean every distance query paid for an EPA it did not need.
[[nodiscard]] float Distance(const Body& a, const Body& b, Vec3* normal = nullptr,
                             Vec3* point_a = nullptr, Vec3* point_b = nullptr);

// The fraction of the given motion at which two bodies first touch, or 1 if
// they do not touch within it. Rotation is ignored, which is what makes the
// bound conservative and cheap; for the fast-moving thing CCD exists for, the
// translation dominates by orders of magnitude.
[[nodiscard]] float TimeOfImpact(const Body& a, const Body& b, Vec3 motion_a,
                                 Vec3 motion_b, float tolerance = 1e-3f);

}  // namespace eng::physics
