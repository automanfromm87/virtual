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
//   * Brute-force broadphase.
// Not here: convex hulls, joints, continuous collision, sleeping.
#pragma once

#include <cstdint>
#include <vector>

#include "engine/core/math.h"

namespace eng::physics {

enum class ShapeType : std::uint8_t { Sphere, Box };

struct Shape {
    ShapeType type = ShapeType::Sphere;
    float radius = 0.5f;                  // Sphere
    Vec3 half_extents{0.5f, 0.5f, 0.5f};  // Box, in the body's own frame

    [[nodiscard]] static Shape MakeSphere(float r) {
        Shape s;
        s.type = ShapeType::Sphere;
        s.radius = r;
        return s;
    }
    [[nodiscard]] static Shape MakeBox(Vec3 half) {
        Shape s;
        s.type = ShapeType::Box;
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

    float restitution = 0.35f;  // 0 = dead drop, 1 = never loses energy
    float friction = 0.5f;
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

struct WorldStats {
    int steps = 0;
    int contacts = 0;
    int pairs_tested = 0;
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

    int Add(const Body& b);
    [[nodiscard]] Body& operator[](int i) { return bodies_[std::size_t(i)]; }
    [[nodiscard]] const Body& operator[](int i) const { return bodies_[std::size_t(i)]; }
    [[nodiscard]] int Count() const { return int(bodies_.size()); }
    void Clear();

    // Advances by `dt` in whole fixed steps, carrying the remainder to the next
    // call. Returns how many steps ran.
    int Step(float dt);
    // One fixed step. Exposed so tests can drive it deterministically.
    void StepFixed();

    [[nodiscard]] const std::vector<Contact>& Contacts() const { return contacts_; }
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

    std::vector<Body> bodies_;
    std::vector<Contact> contacts_;
    // Contacts touching each body this step. Kept as a member so a steady-state
    // step allocates nothing.
    std::vector<int> touches_;
    WorldStats stats_;
    float accumulator_ = 0.0f;
};

// Narrowphase, exposed for testing. Returns false when the pair is apart.
// Every one of these respects Body::orientation.
[[nodiscard]] bool CollideSphereSphere(const Body& a, const Body& b, Contact* out);
[[nodiscard]] bool CollideSphereBox(const Body& sphere, const Body& box, Contact* out);
[[nodiscard]] bool CollideBoxBox(const Body& a, const Body& b, Contact* out);

}  // namespace eng::physics
