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

    // --- sleeping ---------------------------------------------------------------
    {
        World w;
        w.Add(Floor(0.0f));
        const int ball = w.Add(Ball(Vec3{0, 2.0f, 0}, 0.4f, 0.1f));

        // It has to actually settle first: a body still bouncing is not still.
        for (int i = 0; i < 600; ++i) w.StepFixed();
        CHECK(w.SleepingCount() == 1);
        CHECK(w[ball].sleeping);

        // And once asleep it stops moving AT ALL — not "moves very slowly".
        // The residual jitter of a resting contact is exactly what this is for.
        const Vec3 where = w[ball].position;
        for (int i = 0; i < 600; ++i) w.StepFixed();
        CHECK(Length(w[ball].position - where) == 0.0f);

        // A moving body wakes it. Without this a ball rolls into a settled
        // crate and passes through where it should have hit: the crate is
        // still solid, but nothing ever hands it an impulse.
        Body hit = Ball(Vec3{-3.0f, 0.4f, 0}, 0.4f, 0.1f);
        hit.velocity = Vec3{6.0f, 0.0f, 0.0f};
        w.Add(hit);
        for (int i = 0; i < 200; ++i) w.StepFixed();
        CHECK(!w[ball].sleeping);
        CHECK(Length(w[ball].position - where) > 0.05f);
    }

    // --- sleeping can be switched off, and then nothing sleeps -------------------
    {
        World w;
        w.sleep_after = 0.0f;
        w.Add(Floor(0.0f));
        w.Add(Ball(Vec3{0, 2.0f, 0}, 0.4f, 0.1f));
        for (int i = 0; i < 900; ++i) w.StepFixed();
        CHECK(w.SleepingCount() == 0);
    }

    // --- a body IN CONTACT with an awake one cannot sleep ------------------------
    {
        // The bottom of a stack must not freeze while the top is still settling
        // onto it, or the top lands on something that has stopped responding.
        //
        // Note what this does NOT claim: a still body with nothing touching it
        // sleeps regardless of what else is falling elsewhere in the scene, and
        // is woken by the contact when it arrives. That is the first test above.
        World w;
        w.Add(Floor(0.0f));
        const int lower = w.Add(Ball(Vec3{0, 0.4f, 0}, 0.4f, 0.0f));
        const int upper = w.Add(Ball(Vec3{0, 1.19f, 0}, 0.4f, 0.0f));

        // The upper ball is held awake and jostled, so it stays in contact and
        // stays moving. The lower one is doing nothing at all.
        for (int i = 0; i < 400; ++i) {
            w[upper].velocity = Vec3{0.0f, -0.4f, 0.0f};
            w.Wake(upper);
            w.StepFixed();
            if (w[lower].sleeping) {
                Fail("the bottom of the stack slept under a moving neighbour",
                     __LINE__);
                break;
            }
        }
        // ...and once the neighbour stops being jostled, it settles normally.
        for (int i = 0; i < 900; ++i) w.StepFixed();
        CHECK(w[lower].sleeping && w[upper].sleeping);
    }

    // --- joints ------------------------------------------------------------------
    {
        // A pendulum: one static anchor, one free bob, held a metre apart.
        World w;
        Body anchor;
        anchor.shape = Shape::MakeSphere(0.1f);
        anchor.position = Vec3{0, 5.0f, 0};
        anchor.SetMass(0.0f);
        const int top = w.Add(anchor);

        Body bob = Ball(Vec3{1.0f, 5.0f, 0}, 0.2f, 0.0f);
        const int swing = w.Add(bob);
        w.sleep_after = 0.0f;  // a pendulum that falls asleep at the bottom is not one

        Joint j;
        j.a = top;
        j.b = swing;
        j.distance = 1.0f;
        j.stiffness = 1.0f;
        w.AddJoint(j);
        CHECK(w.JointCount() == 1);

        for (int i = 0; i < 1200; ++i) {
            w.StepFixed();
            // The constraint has to hold at EVERY step, not just at the end.
            // A joint that is only correct once things settle is a spring.
            const float d = Length(w[swing].position - w[top].position);
            if (std::fabs(d - 1.0f) > 0.06f) {
                Fail("joint length drifted mid-swing", __LINE__);
                break;
            }
        }
        // And it actually swung rather than hanging where it started.
        CHECK(w[swing].position.y < 4.6f);
    }

    // --- a rope pulls but does not push -------------------------------------------
    {
        World w;
        w.gravity = Vec3{0, 0, 0};
        w.sleep_after = 0.0f;
        Body anchor;
        anchor.shape = Shape::MakeSphere(0.1f);
        anchor.SetMass(0.0f);
        const int top = w.Add(anchor);
        const int bob = w.Add(Ball(Vec3{0.3f, 0, 0}, 0.1f, 0.0f));

        Joint j;
        j.a = top;
        j.b = bob;
        j.distance = 1.0f;
        j.rope = true;
        j.stiffness = 1.0f;
        w.AddJoint(j);

        // Slack: a rod would yank it out to a metre, a rope leaves it alone.
        for (int i = 0; i < 300; ++i) w.StepFixed();
        CHECK(Length(w[bob].position - w[top].position) < 0.35f);

        // Now stretched past its length: it must be pulled back.
        w[bob].position = Vec3{3.0f, 0, 0};
        w.Wake(bob);
        for (int i = 0; i < 600; ++i) w.StepFixed();
        CHECK(Length(w[bob].position - w[top].position) < 1.05f);
    }

    // --- a ball socket is a joint of zero length -----------------------------------
    {
        World w;
        w.gravity = Vec3{0, -9.81f, 0};
        w.sleep_after = 0.0f;
        Body anchor;
        anchor.shape = Shape::MakeBox(Vec3{0.2f, 0.2f, 0.2f});
        anchor.position = Vec3{0, 3.0f, 0};
        anchor.SetMass(0.0f);
        const int top = w.Add(anchor);
        const int arm = w.Add(Ball(Vec3{0, 2.4f, 0}, 0.3f, 0.0f));

        Joint j;
        j.a = top;
        j.b = arm;
        // Anchors in each body's OWN frame: the socket sits below the anchor
        // and at the top of the arm, so the two points coincide.
        j.anchor_a = Vec3{0.0f, -0.3f, 0.0f};
        j.anchor_b = Vec3{0.0f, 0.3f, 0.0f};
        j.distance = 0.0f;
        j.stiffness = 1.0f;
        w.AddJoint(j);

        for (int i = 0; i < 1200; ++i) w.StepFixed();
        const Vec3 pa = w[top].position + Rotate(w[top].orientation, j.anchor_a);
        const Vec3 pb = w[arm].position + Rotate(w[arm].orientation, j.anchor_b);
        CHECK(Length(pb - pa) < 0.05f);
        // Gravity is still pulling on it, so it hangs rather than floating.
        CHECK(w[arm].position.y < 2.75f);
    }

    if (g_failures == 0) std::printf("physics_test: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
