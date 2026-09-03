// No test framework — from scratch means from scratch.
//
// Physics is the one subsystem where the right tests are CONSERVATION LAWS
// rather than sampled values. A solver that gains energy, lets bodies sink, or
// never comes to rest is broken in a way that individual position checks miss.
#include "engine/physics/physics.h"

#include <cmath>
#include <cstdio>
#include <vector>

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

    // --- convex hulls, through GJK and EPA -----------------------------------
    //
    // The strongest available check on GJK/EPA is that it must AGREE with the
    // closed-form routines. A box expressed as a hull of its eight corners is
    // the same shape as a box; if the general path and the special-cased one
    // disagree about depth or normal, one of them is wrong, and the closed
    // forms are already tested above.
    {
        std::printf("convex hulls\n");
        std::vector<Vec3> corners;
        for (int i = 0; i < 8; ++i)
            corners.push_back(Vec3{(i & 1) ? 1.0f : -1.0f, (i & 2) ? 1.0f : -1.0f,
                                   (i & 4) ? 1.0f : -1.0f});

        Body as_box;
        as_box.shape = Shape::MakeBox(Vec3{1, 1, 1});
        Body as_hull;
        as_hull.shape = Shape::MakeHull(corners);

        // Support functions first: everything else in GJK is built on this one
        // question, so a disagreement here explains any disagreement later.
        const Vec3 dirs[7] = {{1, 0, 0},   {0, 1, 0},        {0, 0, 1},
                              {1, 1, 1},   {-0.3f, 0.9f, 0}, {0.5f, -0.5f, 0.7f},
                              {-1, -1, -1}};
        // The support VALUE, not the point. Along (0,1,0) a cube has four
        // equally furthest corners, and which one comes back is arbitrary --
        // GJK only ever uses the dot product, so that is the only part that is
        // defined. Comparing the points instead fails on every axis direction
        // for a shape that is perfectly correct.
        bool support_agrees = true;
        for (const Vec3& d : dirs) {
            const float sb = Dot(Support(as_box, d), d);
            const float sh = Dot(Support(as_hull, d), d);
            if (std::fabs(sb - sh) > 1e-5f) support_agrees = false;
        }
        CHECK(support_agrees);

        // The reference for a penetration depth is not the box routine -- it
        // is the DEFINITION. The depth is the smallest, over all directions, of
        // the shapes' overlap along that direction, and both shapes answer
        // "how far do you reach along n" directly. Sampling the sphere finely
        // gives that minimum to whatever accuracy is wanted, with no shared
        // code and no shared assumptions.
        //
        // This matters here because the box path is NOT a neutral reference: it
        // biases toward face axes on purpose, because an edge-edge normal is
        // unstable and a crate resting on one jitters. So it deliberately
        // reports the face answer where the true minimum is an edge one, and
        // asserting that EPA matches it would be asserting the bias.
        const auto true_depth = [](const Body& x, const Body& y, Vec3* normal) {
            float best = 1e30f;
            // A Fibonacci sphere: even coverage with no clustering at the
            // poles, which a lat/long grid has and which would starve exactly
            // the equatorial directions an edge-edge contact lives on.
            constexpr int kN = 20000;
            const float golden = 3.14159265f * (3.0f - std::sqrt(5.0f));
            for (int i = 0; i < kN; ++i) {
                const float yy = 1.0f - 2.0f * (float(i) + 0.5f) / float(kN);
                const float r = std::sqrt(std::fmax(0.0f, 1.0f - yy * yy));
                const float th = golden * float(i);
                const Vec3 n{std::cos(th) * r, yy, std::sin(th) * r};
                const float d = Dot(Support(x, n) - Support(y, n * -1.0f), n);
                if (d < best) { best = d; *normal = n; }
            }
            // Refine around the winner: 20000 directions is 1.4 degrees apart,
            // and the last fraction of a percent of the depth lives inside that.
            for (int pass = 0; pass < 3; ++pass) {
                const float step = 0.02f / float(1 << pass);
                Vec3 centre = *normal;
                for (int i = 0; i < 400; ++i) {
                    const float a1 = float(i % 20) - 9.5f, a2 = float(i / 20) - 9.5f;
                    Vec3 n = Normalize(centre + Vec3{a1 * step, a2 * step,
                                                     (a1 - a2) * step * 0.5f});
                    const float d = Dot(Support(x, n) - Support(y, n * -1.0f), n);
                    if (d < best) { best = d; *normal = n; }
                }
            }
            return best;
        };

        // The same, for a ROTATED box. At identity the rotation in a box's
        // support function is the identity too, so a version that ignores
        // orientation entirely passes every axis-aligned test there is.
        {
            Body rot_box, rot_hull;
            rot_box.shape = Shape::MakeBox(Vec3{1, 1, 1});
            rot_hull.shape = Shape::MakeHull(corners);
            const Quat q = QuatFromAxisAngle(Normalize(Vec3{1.0f, 0.4f, -0.2f}), 0.9f);
            rot_box.orientation = rot_hull.orientation = q;
            rot_box.position = rot_hull.position = Vec3{0.7f, -0.3f, 0.2f};
            bool rotated_agrees = true;
            float worst_support = 0.0f;
            for (const Vec3& d : dirs) {
                const Vec3 u = Normalize(d);
                const float sb = Dot(Support(rot_box, u), u);
                const float sh = Dot(Support(rot_hull, u), u);
                worst_support = std::fmax(worst_support, std::fabs(sb - sh));
                if (std::fabs(sb - sh) > 1e-5f) rotated_agrees = false;
            }
            std::printf("    rotated box vs the same shape as a hull: support "
                        "differs by at most %.6f\n", worst_support);
            CHECK(rotated_agrees);
        }

        // Now the collisions, over a sweep of offsets and orientations.
        float worst_depth = 0.0f, worst_normal = 0.0f;
        float worst_vs_sat = 0.0f;
        int compared = 0, disagreed = 0;
        for (int t = 0; t < 60; ++t) {
            const float ang = float(t) * 0.11f;
            Body a_box, a_hull, b_box, b_hull;
            a_box.shape = Shape::MakeBox(Vec3{1, 1, 1});
            a_hull.shape = Shape::MakeHull(corners);
            b_box.shape = Shape::MakeBox(Vec3{1, 1, 1});
            b_hull.shape = Shape::MakeHull(corners);
            const Vec3 at{1.2f + 0.02f * float(t), 0.3f * std::sin(ang),
                          0.4f * std::cos(ang * 1.7f)};
            const Quat q = QuatFromAxisAngle(Normalize(Vec3{0.3f, 1.0f, 0.2f}), ang);
            b_box.position = b_hull.position = at;
            b_box.orientation = b_hull.orientation = q;

            Contact cb, ch;
            const bool hit_box = CollideBoxBox(a_box, b_box, &cb);
            const bool hit_hull = CollideConvex(a_hull, b_hull, &ch);
            if (hit_box != hit_hull) { ++disagreed; continue; }
            if (!hit_box) continue;
            ++compared;
            Vec3 ref_normal;
            const float ref = true_depth(a_hull, b_hull, &ref_normal);
            worst_depth = std::fmax(worst_depth, std::fabs(ref - ch.depth));
            // SELF-CONSISTENCY: the overlap measured along the normal EPA
            // reports must be the depth EPA reports. This is the sharp version
            // of the normal check. Comparing the two normals by angle is not,
            // because the depth is at a minimum there and therefore FLAT in the
            // direction: a few degrees of tilt costs a thousandth of a unit, so
            // the angle is barely determined even when the answer is right.
            const float along =
                Dot(Support(a_hull, ch.normal) - Support(b_hull, ch.normal * -1.0f),
                    ch.normal);
            worst_normal = std::fmax(worst_normal, std::fabs(along - ch.depth));
            worst_vs_sat = std::fmax(worst_vs_sat, std::fabs(cb.depth - ch.depth));
        }
        std::printf("    %d overlapping box pairs, solved as hulls:\n"
                    "      vs the sampled true minimum: depth off by at most "
                    "%.5f\n"
                    "      overlap along its own reported normal differs from "
                    "its reported depth by %.5f\n"
                    "      vs the box routine's face-biased answer: %.5f\n"
                    "      %d disagreed on whether they touch at all\n",
                    compared, worst_depth, worst_normal, worst_vs_sat, disagreed);
        CHECK(compared > 40);
        // Whether two shapes touch is not a matter of bias or tolerance, so
        // this one is exact.
        CHECK(disagreed == 0);
        CHECK(worst_depth < 5e-3f);
        CHECK(worst_normal < 1e-3f);
        // And the face bias is real but bounded -- worth pinning, because if it
        // ever grew large it would mean the box routine had started reporting a
        // separating axis that is not close to the true one.
        CHECK(worst_vs_sat < 0.05f);

        // A hull against a SPHERE, where the answer is arithmetic: the sphere's
        // centre is 1.3 from the box face at x = 1, so a radius 0.5 sphere
        // overlaps by 0.2.
        Body sphere = Ball(Vec3{1.3f, 0, 0}, 0.5f);
        Contact c;
        CHECK(CollideConvex(as_hull, sphere, &c));
        std::printf("    hull vs sphere: depth %.4f, contact point "
                    "(%.4f %.4f %.4f)\n", c.depth, c.point.x, c.point.y,
                    c.point.z);
        CHECK(std::fabs(c.depth - 0.2f) < 2e-3f);
        // The contact point sits in the MIDDLE of the overlap. The hull's face
        // is at x=1 and the sphere reaches back to x=0.8, so it belongs at 0.9.
        // Taking either witness alone puts it on one surface -- which is a
        // plausible-looking answer that biases every impulse and every torque
        // by half the penetration depth, in a direction that depends on which
        // body happened to be listed first.
        CHECK(std::fabs(c.point.x - 0.9f) < 0.02f);
        CHECK(std::fabs(c.point.y) < 0.02f && std::fabs(c.point.z) < 0.02f);
        CHECK(c.normal.x > 0.99f);  // hull -> sphere, along +x
        CHECK(!CollideConvex(as_hull, Ball(Vec3{2.0f, 0, 0}, 0.5f), &c));

        // Just touching, and just separated. The boundary is where a
        // penetration algorithm is least stable, and reporting a contact of
        // depth zero as a hit is what makes two resting bodies buzz.
        CHECK(!CollideConvex(as_hull, Ball(Vec3{1.5001f, 0, 0}, 0.5f), &c));

        // Deep overlap: one hull entirely inside another. EPA has to expand
        // right across the polytope here, and a fixed iteration count that is
        // too low returns the first face it looked at.
        Body big;
        std::vector<Vec3> big_corners;
        for (const Vec3& p : corners) big_corners.push_back(p * 3.0f);
        big.shape = Shape::MakeHull(big_corners);
        CHECK(CollideConvex(big, as_hull, &c));
        CHECK(c.depth > 3.9f && c.depth < 4.1f);  // 3 + 1 along the nearest face
        CHECK(std::isfinite(c.point.x) && std::isfinite(c.depth));

        // Mass properties from a real hull, against the box they describe.
        const eng::geom::Hull built = eng::geom::ConvexHull(corners);
        CHECK(!built.Empty());
        Body h;
        h.shape = Shape::MakeHull(built);
        h.SetMass(4.0f);
        Body bx;
        bx.shape = Shape::MakeBox(Vec3{1, 1, 1});
        bx.SetMass(4.0f);
        std::printf("    hull inverse inertia %.5f vs the same box %.5f\n",
                    h.inverse_inertia.x, bx.inverse_inertia.x);
        CHECK(std::fabs(h.inverse_inertia.x - bx.inverse_inertia.x) < 1e-3f);

        // An OFF-CENTRE hull is re-centred on its centre of mass, because that
        // is the only point a free body rotates about. The offset is reported
        // so the caller can put the body back where the art expects it.
        std::vector<Vec3> shifted;
        for (const Vec3& p : corners) shifted.push_back(p + Vec3{5, 0, 0});
        const Shape off = Shape::MakeHull(eng::geom::ConvexHull(shifted));
        CHECK(std::fabs(off.centre_offset.x - 5.0f) < 1e-3f);
        float furthest = 0.0f;
        for (const Vec3& p : off.points) furthest = std::fmax(furthest, Length(p));
        CHECK(furthest < 1.8f);  // sqrt(3), not 6 -- the vertices moved back

        // A hull body actually settling on the ground, which is the whole point.
        {
            World w;
            Body floor;
            floor.shape = Shape::MakeBox(Vec3{20, 1, 20});
            floor.position = Vec3{0, -1, 0};
            floor.SetMass(0.0f);
            w.Add(floor);

            // An octahedron, so it is genuinely not a box or a sphere.
            std::vector<Vec3> oct = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                     {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
            Body body;
            body.shape = Shape::MakeHull(eng::geom::ConvexHull(oct));
            body.position = Vec3{0, 4, 0};
            body.restitution = 0.0f;
            body.SetMass(1.0f);
            const int id = w.Add(body);

            for (int i = 0; i < 900; ++i) w.StepFixed();
            std::printf("    an octahedron dropped from y=4 settles at y=%.4f\n",
                        w[id].position.y);
            // Resting on a vertex is unstable, so it tips onto a face; either
            // way its centre ends up between the vertex height (1) and the
            // face height (1/sqrt(3) = 0.577).
            CHECK(w[id].position.y > 0.5f && w[id].position.y < 1.05f);
            CHECK(std::isfinite(w[id].position.y));
            CHECK(w[id].position.y > 0.0f);  // it did not sink through
        }
    }

    // --- distance ------------------------------------------------------------
    {
        std::printf("distance queries\n");
        Body a = Ball(Vec3{0, 0, 0}, 1.0f);
        Body b = Ball(Vec3{5, 0, 0}, 1.0f);
        Vec3 n, pa, pb;
        const float d = Distance(a, b, &n, &pa, &pb);
        // Centres 5 apart, two unit radii: 3 of clear air between the surfaces.
        CHECK(std::fabs(d - 3.0f) < 1e-3f);
        CHECK(n.x > 0.999f);                     // a -> b
        CHECK(std::fabs(pa.x - 1.0f) < 1e-3f);   // on a's surface
        CHECK(std::fabs(pb.x - 4.0f) < 1e-3f);   // on b's surface

        // Overlapping reports zero, not a negative depth. Depth is a different
        // question with a different answer and a different algorithm.
        CHECK(Distance(a, Ball(Vec3{1.0f, 0, 0}, 1.0f)) == 0.0f);
        // Exactly touching.
        CHECK(Distance(a, Ball(Vec3{2.0f, 0, 0}, 1.0f)) < 1e-3f);

        // Box to box, where the closest feature is an EDGE rather than a face
        // -- the case a naive centre-to-centre estimate gets wrong.
        Body b1, b2;
        b1.shape = Shape::MakeBox(Vec3{1, 1, 1});
        b2.shape = Shape::MakeBox(Vec3{1, 1, 1});
        b2.position = Vec3{4, 4, 0};
        // Corner (1,1,0) to corner (3,3,0): a diagonal gap of sqrt(8).
        const float dd = Distance(b1, b2);
        std::printf("    two boxes corner to corner: %.5f (exact %.5f)\n", dd,
                    std::sqrt(8.0f));
        CHECK(std::fabs(dd - std::sqrt(8.0f)) < 1e-3f);
    }

    // --- continuous collision ------------------------------------------------
    //
    // The failure this exists to stop: a small fast body and a thin wall, where
    // the discrete test looks at the start of the step and the end of the step
    // and the wall was only ever in between. It is not a rare case -- it is
    // every bullet, every thrown object, and it gets WORSE as the frame rate
    // improves the rest of the simulation.
    {
        std::printf("continuous collision\n");
        const auto fire = [](bool ccd, float speed, float* end_x, int* clamps,
                             float* end_vx = nullptr) {
            World w;
            w.gravity = Vec3{0, 0, 0};  // a flat trajectory, so only x matters
            w.ccd_enabled = ccd;
            Body wall;
            wall.shape = Shape::MakeBox(Vec3{0.05f, 5, 5});  // 10 cm thick
            wall.position = Vec3{0, 0, 0};
            wall.SetMass(0.0f);
            w.Add(wall);

            Body bullet = Ball(Vec3{-4, 0, 0}, 0.05f, 0.0f);
            bullet.bullet = true;
            bullet.velocity = Vec3{speed, 0, 0};
            bullet.SetMass(0.02f);
            const int id = w.Add(bullet);

            for (int i = 0; i < 240; ++i) w.StepFixed();
            *end_x = w[id].position.x;
            *clamps = w.Stats().toi_clamps;
            if (end_vx) *end_vx = w[id].velocity.x;
            return w[id].position.x;
        };

        // Slow enough that the discrete test cannot miss it: both must agree,
        // which is what shows CCD is not changing answers it has no business
        // changing.
        float slow_on = 0, slow_off = 0, v_on = 0, v_off = 0;
        int c1 = 0, c2 = 0;
        fire(true, 2.0f, &slow_on, &c1, &v_on);
        fire(false, 2.0f, &slow_off, &c2, &v_off);
        std::printf("    2 m/s:   ccd on x=%.3f vx=%.3f, ccd off x=%.3f vx=%.3f\n",
                    slow_on, v_on, slow_off, v_off);
        CHECK(slow_on < 0.0f && slow_off < 0.0f);  // both stopped at the wall
        CHECK(std::fabs(slow_on - slow_off) < 0.05f);
        // And STOPPED, not merely somewhere plausible. A body held in place by
        // the sweep rather than by the solver sits at the right position with
        // its full speed intact -- the position alone cannot tell the two
        // apart, and the difference is whether it leaps forward the moment the
        // wall moves.
        CHECK(std::fabs(v_on) < 1e-3f && std::fabs(v_off) < 1e-3f);

        // Fast enough to cross the wall inside one step. At 400 m/s and a
        // 1/120 s step the body moves 3.3 m per step against a 0.1 m wall: the
        // discrete test has no chance at all.
        float fast_on = 0, fast_off = 0, fv_on = 0, fv_off = 0;
        int clamps_on = 0, clamps_off = 0;
        fire(true, 400.0f, &fast_on, &clamps_on, &fv_on);
        fire(false, 400.0f, &fast_off, &clamps_off, &fv_off);
        std::printf("    400 m/s: ccd on x=%.3f (%d clamps), ccd off x=%.3f "
                    "(%d clamps)\n", fast_on, clamps_on, fast_off, clamps_off);
        // Without CCD it is somewhere far past the wall; with it, stopped short.
        CHECK(fast_off > 10.0f);
        CHECK(fast_on < 0.0f);
        CHECK(clamps_on > 0 && clamps_off == 0);
        std::printf("      final speeds: ccd on %.4f, ccd off %.4f\n", fv_on, fv_off);
        CHECK(std::fabs(fv_on) < 1e-3f);   // resolved by the solver
        CHECK(fv_off > 100.0f);            // still travelling, having missed it
        // The bullet ends up just barely inside the wall, in the same shallow
        // penetration a slow body would have. Not exactly on the surface: with
        // no overlap there is no contact and nothing ever resolves it.
        CHECK(fast_on > -0.11f && fast_on < -0.09f);

        // A sweep of speeds, because a single one can pass by luck: the step
        // size and the speed could happen to place a sample inside the wall.
        int tunnelled = 0, unresolved = 0;
        for (int i = 0; i < 40; ++i) {
            float x = 0, vx = 0;
            int clamps = 0;
            fire(true, 50.0f + float(i) * 37.5f, &x, &clamps, &vx);
            if (x > 0.1f) ++tunnelled;
            if (std::fabs(vx) > 1e-2f) ++unresolved;
        }
        std::printf("    40 speeds from 50 to 1512 m/s: %d tunnelled, %d left "
                    "with unresolved speed\n", tunnelled, unresolved);
        CHECK(tunnelled == 0);
        CHECK(unresolved == 0);

        // A BOUNCING bullet. This is what distinguishes "the solver stopped
        // it" from "the sweep froze it": with restitution the correct answer is
        // that it comes back, and a body whose velocity was zeroed by the sweep
        // sits against the wall instead. Both look identical with restitution
        // zero, which is why the tests above cannot see the difference.
        {
            World w;
            w.gravity = Vec3{0, 0, 0};
            Body wall;
            wall.shape = Shape::MakeBox(Vec3{0.05f, 5, 5});
            wall.SetMass(0.0f);
            w.Add(wall);
            Body bullet = Ball(Vec3{-4, 0, 0}, 0.05f, 0.6f);  // bouncy
            bullet.bullet = true;
            bullet.velocity = Vec3{300.0f, 0, 0};
            bullet.SetMass(0.02f);
            const int id = w.Add(bullet);
            for (int i = 0; i < 6; ++i) w.StepFixed();
            std::printf("    a bouncy bullet at 300 m/s comes back at %.2f m/s\n",
                        w[id].velocity.x);
            CHECK(w[id].velocity.x < -100.0f);   // reversed, and fast
            CHECK(w[id].velocity.x > -300.0f);   // having lost energy
        }

        // TimeOfImpact on its own, where the answer is arithmetic. A unit
        // sphere at x=0 moving +10 meets a unit sphere at x=5: their surfaces
        // touch after 3 of travel, which is 0.3 of the motion.
        Body p = Ball(Vec3{0, 0, 0}, 1.0f), q = Ball(Vec3{5, 0, 0}, 1.0f);
        const float toi = TimeOfImpact(p, q, Vec3{10, 0, 0}, Vec3{0, 0, 0});
        std::printf("    time of impact %.4f (exact 0.3)\n", toi);
        CHECK(std::fabs(toi - 0.3f) < 2e-3f);

        // Moving APART never impacts, and neither does motion that falls short.
        CHECK(TimeOfImpact(p, q, Vec3{-10, 0, 0}, Vec3{0, 0, 0}) == 1.0f);
        CHECK(TimeOfImpact(p, q, Vec3{2, 0, 0}, Vec3{0, 0, 0}) == 1.0f);
        // Both moving: closing at 10 again, so the same answer.
        CHECK(std::fabs(TimeOfImpact(p, q, Vec3{5, 0, 0}, Vec3{-5, 0, 0}) - 0.3f) <
              2e-3f);
        // Already touching: impact is now.
        CHECK(TimeOfImpact(p, Ball(Vec3{2, 0, 0}, 1.0f), Vec3{1, 0, 0},
                           Vec3{0, 0, 0}) < 1e-2f);
    }

    // --- raycasts --------------------------------------------------------------
    //
    // The engine could trace a million shadow rays on the GPU and could not
    // answer "what did the player click on". This is that.
    //
    // Every expectation here is arithmetic, because a raycast is easy to write
    // so that it ALMOST works: off by the radius, returning the far root,
    // returning the second-nearest body, or reporting the surface normal of the
    // face opposite the one it hit. None of those show up in a demo.
    {
        std::printf("raycasts\n");
        World w;
        w.gravity = Vec3{0, 0, 0};

        // A sphere of radius 1 at x = 5. A ray down +x from the origin meets
        // its NEAR surface at 4, not its centre at 5 and not its far side at 6.
        const int ball = w.Add(Ball(Vec3{5, 0, 0}, 1.0f));
        RayHit hit;
        CHECK(w.Raycast(Vec3{0, 0, 0}, Vec3{1, 0, 0}, 100.0f, &hit));
        std::printf("    sphere r=1 at x=5, ray from the origin: t = %.5f, "
                    "normal (%.2f %.2f %.2f)\n", hit.t, hit.normal.x,
                    hit.normal.y, hit.normal.z);
        CHECK(hit.body == ball);
        CHECK(std::fabs(hit.t - 4.0f) < 1e-4f);
        CHECK(std::fabs(hit.point.x - 4.0f) < 1e-4f);
        // Pointing OUT of the surface, so back toward the shooter.
        CHECK(hit.normal.x < -0.999f);

        // Misses. Past the end, beside it, and behind the origin.
        CHECK(!w.Raycast(Vec3{0, 0, 0}, Vec3{1, 0, 0}, 3.5f, &hit));
        CHECK(!w.Raycast(Vec3{0, 3, 0}, Vec3{1, 0, 0}, 100.0f, &hit));
        CHECK(!w.Raycast(Vec3{0, 0, 0}, Vec3{-1, 0, 0}, 100.0f, &hit));

        // A direction scaled to the ray's LENGTH makes t a fraction, which is
        // what a "did this segment hit anything" query wants.
        CHECK(w.Raycast(Vec3{0, 0, 0}, Vec3{8, 0, 0}, 1.0f, &hit));
        CHECK(std::fabs(hit.t - 0.5f) < 1e-4f);

        // The NEAREST of several, not the first one added or the last.
        const int near_ball = w.Add(Ball(Vec3{2, 0, 0}, 0.5f));
        CHECK(w.Raycast(Vec3{0, 0, 0}, Vec3{1, 0, 0}, 100.0f, &hit));
        CHECK(hit.body == near_ball);
        CHECK(std::fabs(hit.t - 1.5f) < 1e-4f);
        // ...and adding a FARTHER one after it must not change the answer.
        w.Add(Ball(Vec3{9, 0, 0}, 0.5f));
        CHECK(w.Raycast(Vec3{0, 0, 0}, Vec3{1, 0, 0}, 100.0f, &hit));
        CHECK(hit.body == near_ball);

        // Starting INSIDE reports a hit at zero rather than a miss. A character
        // that spawns in geometry has to be able to find out.
        CHECK(w.Raycast(Vec3{2, 0, 0}, Vec3{1, 0, 0}, 100.0f, &hit));
        CHECK(hit.body == near_ball && hit.t == 0.0f);
    }

    // --- raycasts against boxes, including rotated ones -------------------------
    {
        World w;
        Body box;
        box.shape = Shape::MakeBox(Vec3{1, 2, 3});
        box.position = Vec3{0, 0, 0};
        box.SetMass(0.0f);
        w.Add(box);

        RayHit hit;
        // Down each axis, where the answer is the half-extent.
        CHECK(w.Raycast(Vec3{-10, 0, 0}, Vec3{1, 0, 0}, 100.0f, &hit));
        CHECK(std::fabs(hit.t - 9.0f) < 1e-4f && hit.normal.x < -0.999f);
        CHECK(w.Raycast(Vec3{0, 10, 0}, Vec3{0, -1, 0}, 100.0f, &hit));
        CHECK(std::fabs(hit.t - 8.0f) < 1e-4f && hit.normal.y > 0.999f);
        CHECK(w.Raycast(Vec3{0, 0, -10}, Vec3{0, 0, 1}, 100.0f, &hit));
        CHECK(std::fabs(hit.t - 7.0f) < 1e-4f && hit.normal.z < -0.999f);

        // PARALLEL to a face. Outside the slab it must miss, inside it must
        // hit, and EXACTLY ON the face it must still decide.
        //
        // All three pass with the parallel case handled and, as it turns out,
        // without it: the signed infinities work, and the 0/0 an on-face ray
        // produces is absorbed before it reaches either bound. So these do not
        // prove the guard is needed -- they pin the ANSWERS, which is what they
        // are for. See the guard's own comment for why it stays anyway.
        CHECK(!w.Raycast(Vec3{-10, 5, 0}, Vec3{1, 0, 0}, 100.0f, &hit));
        CHECK(w.Raycast(Vec3{-10, 1.5f, 0}, Vec3{1, 0, 0}, 100.0f, &hit));
        CHECK(w.Raycast(Vec3{-10, 2.0f, 0}, Vec3{1, 0, 0}, 100.0f, &hit));
        CHECK(std::isfinite(hit.t) && std::fabs(hit.t - 9.0f) < 1e-3f);
        CHECK(w.Raycast(Vec3{-10, 0, 3.0f}, Vec3{1, 0, 0}, 100.0f, &hit));
        CHECK(std::isfinite(hit.t) && std::fabs(hit.t - 9.0f) < 1e-3f);

        // ROTATED. A cube turned 45 degrees about y presents its edge to a ray
        // down x, and the edge is half_extent * sqrt(2) from the centre.
        World r;
        Body cube;
        cube.shape = Shape::MakeBox(Vec3{1, 1, 1});
        cube.orientation = QuatFromAxisAngle(Vec3{0, 1, 0}, 0.7853981634f);
        cube.SetMass(0.0f);
        r.Add(cube);
        CHECK(r.Raycast(Vec3{-10, 0, 0}, Vec3{1, 0, 0}, 100.0f, &hit));
        const float want = 10.0f - std::sqrt(2.0f);
        std::printf("    a cube turned 45 degrees, hit at t = %.5f "
                    "(edge-on gives %.5f)\n", hit.t, want);
        CHECK(std::fabs(hit.t - want) < 1e-3f);
        // The normal is a FACE normal of the rotated cube, not the world axis.
        // At 45 degrees both x and z components are 1/sqrt(2).
        CHECK(std::fabs(std::fabs(hit.normal.x) - 0.7071f) < 1e-2f);
        CHECK(std::fabs(std::fabs(hit.normal.z) - 0.7071f) < 1e-2f);
    }

    // --- raycasts against hulls -------------------------------------------------
    //
    // A hull stores only its vertices, so the ray is marched in rather than
    // clipped against faces. The check is that it agrees with the BOX routine
    // on a shape that is both: the two share no code at all.
    {
        std::vector<Vec3> corners;
        for (int i = 0; i < 8; ++i)
            corners.push_back(Vec3{(i & 1) ? 1.0f : -1.0f, (i & 2) ? 1.0f : -1.0f,
                                   (i & 4) ? 1.0f : -1.0f});
        // First: is Distance() itself right for a point against a hull? The
        // march is built entirely on it, and a march that stops early is
        // indistinguishable from a Distance that reports zero too soon.
        {
            World probe_world;
            Body h;
            h.shape = Shape::MakeHull(corners);
            h.SetMass(0.0f);
            Body pt = Ball(Vec3{0, 0, 0}, 0.0f);
            float worst_d = 0.0f;
            for (int k = 0; k < 40; ++k) {
                const float x = 1.0f + float(k) * 0.15f;
                const float y = 0.35f * std::sin(float(k) * 0.5f);
                pt.position = Vec3{x, y, 0.2f};
                // Outside a box on the +x face only, the distance is x - 1.
                const float want = x - 1.0f;
                worst_d = std::fmax(worst_d, std::fabs(Distance(pt, h) - want));
            }
            std::printf("    point-to-hull distance, worst error over 40 "
                        "samples: %.6f\n", worst_d);
            CHECK(worst_d < 1e-3f);
        }

        float worst = 0.0f;
        int compared = 0;
        for (int k = 0; k < 24; ++k) {
            const float a = float(k) * 0.26f;
            World wb, wh;
            Body b, h;
            b.shape = Shape::MakeBox(Vec3{1, 1, 1});
            h.shape = Shape::MakeHull(corners);
            b.orientation = h.orientation =
                QuatFromAxisAngle(Normalize(Vec3{0.3f, 1.0f, 0.2f}), a);
            b.SetMass(0.0f);
            h.SetMass(0.0f);
            wb.Add(b);
            wh.Add(h);
            const Vec3 from{-8.0f, 0.4f * std::sin(a), 0.3f * std::cos(a * 1.3f)};
            RayHit hb, hh;
            const bool ok_b = wb.Raycast(from, Vec3{1, 0, 0}, 100.0f, &hb);
            const bool ok_h = wh.Raycast(from, Vec3{1, 0, 0}, 100.0f, &hh);
            CHECK(ok_b == ok_h);
            if (!ok_b) continue;
            ++compared;
            worst = std::fmax(worst, std::fabs(hb.t - hh.t));
        }
        std::printf("    %d rays at a box and the same shape as a hull: worst "
                    "disagreement %.5f\n", compared, worst);
        CHECK(compared > 20);
        // The march stops within its own tolerance of the surface, so it is
        // slightly SHORT rather than exact -- a bound, not a coincidence.
        CHECK(worst < 1e-3f);
    }

    // --- filtering ---------------------------------------------------------------
    //
    // Every game needs this on the first day: a ray from a gun must not hit the
    // gun, and a bullet must not stop at a checkpoint volume.
    {
        World w;
        Body shooter = Ball(Vec3{0, 0, 0}, 0.5f);
        shooter.layer = 0x2;
        const int me = w.Add(shooter);
        Body zone = Ball(Vec3{3, 0, 0}, 1.0f);
        zone.trigger = true;
        const int trig = w.Add(zone);
        Body wall;
        wall.shape = Shape::MakeBox(Vec3{0.2f, 5, 5});
        wall.position = Vec3{6, 0, 0};
        wall.layer = 0x4;
        wall.SetMass(0.0f);
        const int solid = w.Add(wall);

        RayHit hit;
        // Cast from inside the shooter. Without `ignore` it hits itself at t=0.
        CHECK(w.Raycast(Vec3{0, 0, 0}, Vec3{1, 0, 0}, 100.0f, &hit));
        CHECK(hit.body == me && hit.t == 0.0f);

        QueryFilter f;
        f.ignore = me;
        CHECK(w.Raycast(Vec3{0, 0, 0}, Vec3{1, 0, 0}, 100.0f, &hit, f));
        // The trigger is INVISIBLE by default, so the bullet reaches the wall.
        CHECK(hit.body == solid);
        CHECK(std::fabs(hit.t - 5.8f) < 1e-3f);

        // A query that wants triggers asks for them, and then finds the nearer.
        f.hit_triggers = true;
        CHECK(w.Raycast(Vec3{0, 0, 0}, Vec3{1, 0, 0}, 100.0f, &hit, f));
        CHECK(hit.body == trig);

        // And a mask picks a layer out.
        QueryFilter only_wall;
        only_wall.mask = 0x4;
        CHECK(w.Raycast(Vec3{0, 0, 0}, Vec3{1, 0, 0}, 100.0f, &hit, only_wall));
        CHECK(hit.body == solid);
    }

    // --- overlap queries ---------------------------------------------------------
    {
        std::printf("overlap queries\n");
        World w;
        for (int i = 0; i < 5; ++i) w.Add(Ball(Vec3{float(i) * 2.0f, 0, 0}, 0.5f));

        std::vector<int> found;
        // A sphere of radius 1 at the origin reaches x = 1, and the ball at
        // x = 2 has radius 0.5, so its surface is at 1.5. Not touching.
        CHECK(w.OverlapSphere(Vec3{0, 0, 0}, 1.0f, &found) == 1);
        CHECK(found.size() == 1 && found[0] == 0);

        found.clear();
        // Radius 2 reaches 2, and the second ball's surface is at 1.5. Two.
        CHECK(w.OverlapSphere(Vec3{0, 0, 0}, 2.0f, &found) == 2);

        found.clear();
        // A blast radius covering the lot.
        CHECK(w.OverlapSphere(Vec3{4, 0, 0}, 10.0f, &found) == 5);

        found.clear();
        CHECK(w.OverlapSphere(Vec3{0, 50, 0}, 1.0f, &found) == 0);

        // A box, and it must be tighter than the sphere that contains it: a
        // 0.9-cube at the origin does NOT reach the ball at x = 2, while a
        // sphere through its corners (radius 1.56) would.
        found.clear();
        CHECK(w.OverlapBox(Vec3{0, 0, 0}, Vec3{0.9f, 0.9f, 0.9f}, &found) == 1);
        found.clear();
        CHECK(w.OverlapBox(Vec3{0, 0, 0}, Vec3{1.6f, 0.2f, 0.2f}, &found) == 2);

        // OVERLAP APPENDS rather than replacing, so several queries can build
        // one list -- and a caller that expected replacement gets a bug that
        // only shows on the second call.
        found.clear();
        w.OverlapSphere(Vec3{0, 0, 0}, 1.0f, &found);
        w.OverlapSphere(Vec3{8, 0, 0}, 1.0f, &found);
        CHECK(found.size() == 2 && found[0] == 0 && found[1] == 4);
    }

    if (g_failures == 0) std::printf("physics_test: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
