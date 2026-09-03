// No test framework — from scratch means from scratch.
//
// Physics is the one subsystem where the right tests are CONSERVATION LAWS
// rather than sampled values. A solver that gains energy, lets bodies sink, or
// never comes to rest is broken in a way that individual position checks miss.
#include "engine/physics/physics.h"

#include <cmath>
#include <cstdio>

namespace {

int g_failures = 0;

void Fail(const char* what, int line) {
    std::fprintf(stderr, "physics_test.cc:%d  %s\n", line, what);
    ++g_failures;
}

#define CHECK(cond) \
    do { if (!(cond)) Fail(#cond, __LINE__); } while (0)

using namespace eng;
using namespace eng::physics;

// A static box whose top face is at y = 0.
Body Floor(float restitution = 0.4f) {
    Body f;
    f.shape = Shape::MakeBox(Vec3{50.0f, 1.0f, 50.0f});
    f.position = Vec3{0.0f, -1.0f, 0.0f};
    f.inverse_mass = 0.0f;  // static
    f.restitution = restitution;
    return f;
}

Body Ball(Vec3 at, float r = 0.5f, float restitution = 0.4f) {
    Body b;
    b.shape = Shape::MakeSphere(r);
    b.position = at;
    b.restitution = restitution;
    return b;
}

}  // namespace

int main() {
    // --- narrowphase ---------------------------------------------------------
    {
        Contact c;
        const Body a = Ball(Vec3{0, 0, 0}, 1.0f);
        CHECK(!CollideSphereSphere(a, Ball(Vec3{3, 0, 0}, 1.0f), &c));  // apart
        CHECK(CollideSphereSphere(a, Ball(Vec3{1.5f, 0, 0}, 1.0f), &c));
        CHECK(std::fabs(c.depth - 0.5f) < 1e-5f);
        CHECK(std::fabs(c.normal.x - 1.0f) < 1e-5f);  // a -> b

        // Concentric spheres must not produce a NaN normal. Normalising a zero
        // vector here poisons the solver and every body goes to NaN with it.
        CHECK(CollideSphereSphere(a, Ball(Vec3{0, 0, 0}, 1.0f), &c));
        CHECK(std::isfinite(c.normal.x) && std::isfinite(c.normal.y));
        CHECK(std::fabs(Length(Vec3{c.normal.x, c.normal.y, c.normal.z}) - 1.0f) < 1e-5f);
    }
    {
        Contact c;
        Body box;
        box.shape = Shape::MakeBox(Vec3{1, 1, 1});
        box.inverse_mass = 0.0f;
        CHECK(!CollideSphereBox(Ball(Vec3{0, 3, 0}), box, &c));
        CHECK(CollideSphereBox(Ball(Vec3{0, 1.3f, 0}), box, &c));
        CHECK(std::fabs(c.depth - 0.2f) < 1e-5f);
        CHECK(c.normal.y < -0.99f);  // from the sphere DOWN toward the box

        // Sphere centre fully inside the box: still has to produce a sane,
        // unit-length escape direction rather than a zero vector.
        CHECK(CollideSphereBox(Ball(Vec3{0, 0.5f, 0}), box, &c));
        CHECK(std::fabs(Length(Vec3{c.normal.x, c.normal.y, c.normal.z}) - 1.0f) < 1e-5f);
        CHECK(c.depth > 0.0f);
    }
    {
        Contact c;
        Body a, b;
        a.shape = Shape::MakeBox(Vec3{1, 1, 1});
        b.shape = Shape::MakeBox(Vec3{1, 1, 1});
        b.position = Vec3{1.5f, 0.2f, 0.0f};
        CHECK(CollideBoxBox(a, b, &c));
        // Least-overlap axis is X (0.5 deep) not Y (1.8), so that is the normal.
        CHECK(std::fabs(c.normal.x - 1.0f) < 1e-5f);
        CHECK(std::fabs(c.depth - 0.5f) < 1e-5f);
        b.position = Vec3{2.5f, 0, 0};
        CHECK(!CollideBoxBox(a, b, &c));
    }

    // --- free fall matches the closed form -----------------------------------
    {
        World w;
        const int ball = w.Add(Ball(Vec3{0, 100.0f, 0}));
        const float t = 1.0f;
        for (int i = 0; i < 120; ++i) w.StepFixed();  // 120 * 1/120 s
        // Semi-implicit Euler overshoots the analytic drop by exactly
        // 0.5*g*dt*t, which at this step size is about 4 cm. Asserting the
        // closed form to within a hair would be asserting the wrong thing.
        const float expect = 100.0f - 0.5f * 9.81f * t * t;
        CHECK(std::fabs(w[ball].position.y - expect) < 0.1f);
        CHECK(std::fabs(w[ball].velocity.y + 9.81f * t) < 0.05f);
    }

    // --- a ball settles ON the floor, and stays there ------------------------
    {
        World w;
        w.Add(Floor());
        const int ball = w.Add(Ball(Vec3{0, 4.0f, 0}, 0.5f, 0.3f));
        for (int i = 0; i < 1200; ++i) w.StepFixed();  // 10 seconds

        // Resting on top: centre one radius above the floor's surface, within
        // the slop the solver deliberately leaves.
        CHECK(std::fabs(w[ball].position.y - 0.5f) < 0.02f);
        // ...and actually at rest, not buzzing. A solver that fights its own
        // positional correction leaves a permanent millimetre-scale jitter.
        CHECK(std::fabs(w[ball].velocity.y) < 0.05f);
        // Never sank through. This is what positional correction is for.
        CHECK(w[ball].position.y > 0.4f);
    }

    // --- restitution does what it says ---------------------------------------
    {
        auto bounce_height = [](float restitution) {
            World w;
            // A BOUNCY floor, so min(floor, ball) is the ball's value. With the
            // default 0.4 floor the min clamps both cases to 0.4 and the test
            // compares the ball against itself.
            w.Add(Floor(0.95f));
            const int ball = w.Add(Ball(Vec3{0, 3.0f, 0}, 0.5f, restitution));
            float highest_after_hit = 0.0f;
            bool hit = false;
            for (int i = 0; i < 600; ++i) {
                w.StepFixed();
                if (w[ball].position.y < 0.55f) hit = true;
                if (hit) highest_after_hit = std::fmax(highest_after_hit, w[ball].position.y);
            }
            return highest_after_hit;
        };
        const float dead = bounce_height(0.0f);
        const float lively = bounce_height(0.85f);
        CHECK(lively > dead + 0.5f);
        // A perfectly inelastic drop must not bounce at all beyond the slop.
        CHECK(dead < 0.6f);
    }

    // --- ENERGY: the solver must never create any ----------------------------
    // The classic instability. A stack that slowly gains energy looks fine for
    // a second and then launches itself across the scene.
    {
        World w;
        w.Add(Floor());
        for (int i = 0; i < 12; ++i)
            w.Add(Ball(Vec3{float(i % 4) * 0.9f - 1.35f, 1.0f + float(i) * 1.1f,
                            float(i / 4) * 0.9f - 0.9f},
                       0.45f, 0.5f));
        const float start = w.Energy();
        float peak = start;
        for (int i = 0; i < 2400; ++i) {  // 20 seconds
            w.StepFixed();
            peak = std::fmax(peak, w.Energy());
        }
        std::printf("  energy: start %.2f  peak %.2f  end %.2f  contacts %d\n",
                    double(start), double(peak), double(w.Energy()),
                    w.Stats().contacts);
        // Positional correction does add a little, so allow a small margin —
        // but nothing like the runaway growth a broken solver produces.
        CHECK(peak < start * 1.05f + 1.0f);
        // And it has to LOSE energy overall: restitution below 1 is a promise.
        CHECK(w.Energy() < start * 0.9f);
        // Everything ended up above the floor rather than tunnelling through.
        for (int i = 1; i < w.Count(); ++i) CHECK(w[i].position.y > 0.3f);
    }

    // --- static bodies never move --------------------------------------------
    {
        World w;
        const int floor = w.Add(Floor());
        const Vec3 before = w[floor].position;
        for (int i = 0; i < 10; ++i) w.Add(Ball(Vec3{0, 1.0f + float(i), 0}));
        for (int i = 0; i < 600; ++i) w.StepFixed();
        CHECK(w[floor].position.y == before.y);
        CHECK(w[floor].velocity.y == 0.0f);
    }

    // --- the fixed timestep is actually fixed --------------------------------
    {
        World w;
        w.fixed_dt = 1.0f / 100.0f;
        w.Add(Ball(Vec3{0, 10, 0}));
        CHECK(w.Step(0.005f) == 0);   // less than one step: accumulate
        CHECK(w.Step(0.005f) == 1);   // now it adds up to exactly one
        CHECK(w.Step(0.025f) == 2);   // two whole steps, 0.005 carried over
        // A huge dt must be capped, not turned into a thousand-step burst.
        CHECK(w.Step(100.0f) <= 8);
    }

    // --- inertia tensors are the textbook ones -------------------------------
    {
        Body s;
        s.shape = Shape::MakeSphere(2.0f);
        s.SetMass(3.0f);
        // I = 2/5 m r² = 0.4 * 3 * 4 = 4.8, isotropic.
        CHECK(std::fabs(1.0f / s.inverse_inertia.x - 4.8f) < 1e-4f);
        CHECK(std::fabs(s.inverse_inertia.x - s.inverse_inertia.z) < 1e-6f);

        Body b;
        b.shape = Shape::MakeBox(Vec3{1.0f, 2.0f, 3.0f});
        b.SetMass(6.0f);
        // I_x = m(h_y² + h_z²)/3 = 6*(4+9)/3 = 26; I_z = 6*(1+4)/3 = 10.
        CHECK(std::fabs(1.0f / b.inverse_inertia.x - 26.0f) < 1e-3f);
        CHECK(std::fabs(1.0f / b.inverse_inertia.z - 10.0f) < 1e-3f);
        // A long box is HARDER to spin about a short axis. Getting the tensor
        // transposed would swap these and nothing else would notice.
        CHECK(b.inverse_inertia.x < b.inverse_inertia.z);

        b.SetMass(0.0f);
        CHECK(b.inverse_mass == 0.0f);
        CHECK(Dot(b.inverse_inertia, b.inverse_inertia) == 0.0f);
    }

    // --- the inverse inertia tensor follows the body around -------------------
    {
        // A box that is hard to spin about its own x stays hard to spin about
        // that axis after it turns, which is now the world's z. A tensor left
        // in body space (or rotated the wrong way) fails this and only this.
        Body b;
        b.shape = Shape::MakeBox(Vec3{0.2f, 2.0f, 2.0f});
        b.SetMass(4.0f);
        const Vec3 unrotated = b.ApplyInverseInertia(Vec3{1, 0, 0});
        b.orientation = QuatFromAxisAngle(Vec3{0, 1, 0}, 1.5707963f);
        const Vec3 rotated = b.ApplyInverseInertia(Vec3{0, 0, -1});
        CHECK(std::fabs(Length(rotated) - Length(unrotated)) < 1e-4f);
        // And the response is still parallel to the axis it was applied about.
        CHECK(std::fabs(std::fabs(rotated.z) - Length(rotated)) < 1e-4f);
    }

    // --- a free spin conserves angular momentum -------------------------------
    {
        World w;
        w.gravity = Vec3{0, 0, 0};
        Body b = Ball(Vec3{0, 0, 0});
        b.angular_velocity = Vec3{1.3f, -0.7f, 2.1f};
        w.Add(b);
        const Vec3 l0 = w.AngularMomentum();
        const float e0 = w.Energy();
        for (int i = 0; i < 2000; ++i) w.StepFixed();
        const Vec3 l1 = w.AngularMomentum();
        // Nothing exerts a torque, so L must not drift. This is the sharpest
        // check available on the integrator and the tensor together.
        CHECK(Length(l1 - l0) < 1e-4f);
        CHECK(std::fabs(w.Energy() - e0) < 1e-4f);
        // And the orientation stayed a unit quaternion rather than drifting.
        const Quat q = w[0].orientation;
        CHECK(std::fabs(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w - 1.0f) < 1e-4f);
        // It actually turned, rather than the integrator doing nothing.
        CHECK(std::fabs(q.w) < 0.9999f);
    }

    // --- a ball on the ground ROLLS, it does not slide ------------------------
    {
        World w;
        w.Add(Floor(0.0f));
        Body b = Ball(Vec3{0, 0.5f, 0}, 0.5f, 0.0f);
        b.friction = 0.6f;
        b.velocity = Vec3{4.0f, 0.0f, 0.0f};  // launched, not spun
        const int ball = w.Add(b);
        for (int i = 0; i < 400; ++i) w.StepFixed();

        const Vec3 v = w[ball].velocity;
        const Vec3 om = w[ball].angular_velocity;
        // Rolling without slipping: the contact point is instantaneously still,
        // so v = ω × r with r pointing from the centre down to the ground.
        const Vec3 surface = v + Cross(om, Vec3{0.0f, -0.5f, 0.0f});
        CHECK(Length(surface) < 0.1f);
        // Moving +x on the ground means spinning about -z. A sign error here
        // gives a ball that rolls backwards, which the magnitude check misses.
        CHECK(om.z < -1.0f);
        CHECK(v.x > 1.0f);  // friction converted motion to spin, not to heat
    }

    // --- rolling keeps its energy, sliding burns it ---------------------------
    {
        // Same ball, same speed, same floor. The only difference is that one is
        // already spinning at the rolling rate and the other is not.
        auto run = [](bool prespun) {
            World w;
            w.Add(Floor(0.0f));
            Body b = Ball(Vec3{0, 0.5f, 0}, 0.5f, 0.0f);
            b.friction = 0.6f;
            b.velocity = Vec3{4.0f, 0.0f, 0.0f};
            if (prespun) b.angular_velocity = Vec3{0.0f, 0.0f, -8.0f};  // v/r
            w.Add(b);
            const float e0 = w.Energy();
            for (int s = 0; s < 400; ++s) w.StepFixed();
            return e0 - w.Energy();
        };
        const float lost_sliding = run(false);
        const float lost_rolling = run(true);
        // A ball that starts rolling has nothing sliding at the contact, so
        // friction does no work on it.
        CHECK(lost_rolling < lost_sliding * 0.25f);
        CHECK(lost_sliding > 0.5f);
    }

    // --- a box tips over its own edge ------------------------------------------
    {
        // Balanced on a corner, it must fall onto a face rather than balance
        // forever. This needs contact points, a manifold and torque all at once
        // — with a single centre-of-mass impulse it just sinks straight down.
        World w;
        w.Add(Floor(0.0f));
        Body box;
        box.shape = Shape::MakeBox(Vec3{0.5f, 0.5f, 0.5f});
        box.position = Vec3{0.0f, 0.9f, 0.0f};
        box.orientation = QuatFromAxisAngle(Vec3{0, 0, 1}, 0.6f);
        box.restitution = 0.0f;
        box.friction = 0.8f;
        const int b = w.Add(box);
        for (int i = 0; i < 900; ++i) w.StepFixed();

        // Landed flat: its up axis is back to vertical, and it is at rest.
        const Vec3 up = Rotate(w[b].orientation, Vec3{0, 1, 0});
        CHECK(std::fabs(up.y) > 0.99f);
        CHECK(Length(w[b].velocity) < 0.05f);
        CHECK(Length(w[b].angular_velocity) < 0.2f);
        // Resting ON the floor, not sunk into it.
        CHECK(w[b].position.y > 0.45f && w[b].position.y < 0.56f);
    }

    // --- an oriented box is a real obstacle, not its bounding box -------------
    {
        // The old axis-aligned test would have called these two overlapping.
        Body a, b;
        a.shape = Shape::MakeBox(Vec3{1.0f, 1.0f, 1.0f});
        b.shape = Shape::MakeBox(Vec3{1.0f, 1.0f, 1.0f});
        b.orientation = QuatFromAxisAngle(Vec3{0, 0, 1}, 0.7853981f);
        Contact c;
        // Turned 45 degrees, the cube reaches sqrt(2) = 1.414 along x instead
        // of 1.0 — a corner leads now, not a face. At a centre distance of 2.2
        // that corner sits at 0.79, which is inside a's face at 1.0.
        b.position = Vec3{2.2f, 0.0f, 0.0f};
        CHECK(CollideBoxBox(a, b, &c));
        // The orientation-blind test this replaced compared half-extents only:
        // 1 + 1 < 2.2, so it would have called this pair separated and let the
        // two boxes pass through each other.
        b.position = Vec3{2.6f, 0.0f, 0.0f};  // corner at 1.19, clear of a
        CHECK(!CollideBoxBox(a, b, &c));
    }

    // --- a ball rolls DOWN a ramp, and the ramp is oriented -------------------
    {
        World w;
        Body ramp;
        ramp.shape = Shape::MakeBox(Vec3{6.0f, 0.25f, 3.0f});
        ramp.position = Vec3{0.0f, 0.0f, 0.0f};
        ramp.orientation = QuatFromAxisAngle(Vec3{0, 0, 1}, 0.35f);  // ~20 deg
        ramp.inverse_mass = 0.0f;
        ramp.friction = 0.9f;
        w.Add(ramp);

        Body b = Ball(Vec3{0.0f, 0.9f, 0.0f}, 0.35f, 0.0f);
        b.friction = 0.9f;
        const int ball = w.Add(b);
        for (int i = 0; i < 240; ++i) w.StepFixed();

        // Downhill is -x for a ramp tilted +0.35 about +z.
        CHECK(w[ball].velocity.x < -0.5f);
        CHECK(w[ball].angular_velocity.z > 0.5f);  // rolling, not sliding
        // Still on the ramp surface rather than through it.
        const Vec3 rel = w[ball].position - w[0].position;
        const float above = Dot(rel, Rotate(w[0].orientation, Vec3{0, 1, 0}));
        CHECK(above > 0.25f + 0.35f - 0.05f);
    }

    // --- penetration recovery does not scale with the manifold ----------------
    {
        // Regression test. Positional correction used to run once per contact,
        // so a box with four corners in the floor was pushed out four times as
        // far as it had sunk in — 0.312 of lift for 0.2 of overlap — and got
        // ejected. A sphere has one contact and never showed it, which is why
        // it survived: the fix is only visible if the two are compared.
        auto lift = [](bool box) {
            World w;
            w.Add(Floor(0.0f));
            Body b;
            if (box) b.shape = Shape::MakeBox(Vec3{0.5f, 0.5f, 0.5f});
            else b.shape = Shape::MakeSphere(0.5f);
            b.position = Vec3{0, 0.30f, 0};  // 0.2 deep
            b.restitution = 0.0f;
            const int i = w.Add(b);
            const float before = w[i].position.y;
            w.StepFixed();
            return w[i].position.y - before;
        };
        const float sphere = lift(false);
        const float box = lift(true);
        // Never more than the overlap being corrected, whatever the shape.
        CHECK(box <= 0.2f);
        CHECK(sphere <= 0.2f);
        // And the two shapes recover at comparable rates rather than one of
        // them being multiplied by its corner count.
        CHECK(box < sphere * 1.5f);
        CHECK(box > sphere * 0.5f);
    }

    if (g_failures == 0) std::printf("physics_test: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
