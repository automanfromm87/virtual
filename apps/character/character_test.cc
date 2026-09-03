// A kinematic capsule character controller, against the things it exists to do.
//
// Every failure of a character controller is a thing you only notice by walking
// around: catching on a floor seam, sliding down a ramp you should stand on,
// climbing a wall by jumping at it, falling through the floor when you sprint.
// None of them show up in a still frame and all of them are reproducible in a
// test that measures where the character ends up.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/physics/character.h"
#include "engine/physics/physics.h"

namespace {

int g_failures = 0;

void Fail(const char* what, int line) {
    std::fprintf(stderr, "character_test.cc:%d  %s\n", line, what);
    ++g_failures;
}

#define CHECK(cond) \
    do { if (!(cond)) Fail(#cond, __LINE__); } while (0)

using namespace eng;
using namespace eng::physics;

// A static box, which is what a level is made of.
int AddBox(World& w, Vec3 centre, Vec3 half, Quat q = Quat{}) {
    Body b;
    b.shape = Shape::MakeBox(half);
    b.position = centre;
    b.orientation = q;
    b.SetMass(0.0f);
    return w.Add(b);
}

// Walks for `seconds` at `speed` along `dir`, with gravity, the way a game
// loop would.
void Walk(CharacterController& c, const World& w, Vec3 dir, float speed,
          float seconds, float dt = 1.0f / 60.0f) {
    float fall = 0.0f;
    for (int i = 0; i < int(seconds / dt); ++i) {
        fall = c.Grounded() ? 0.0f : fall - 9.81f * dt;
        c.Move(w, dir * (speed * dt) + Vec3{0.0f, fall * dt, 0.0f});
    }
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("character controller\n");

    // --- it stands on the floor --------------------------------------------
    {
        World w;
        AddBox(w, Vec3{0, -0.5f, 0}, Vec3{20, 0.5f, 20});
        CharacterController c;
        c.Teleport(Vec3{0, 3.0f, 0});
        CHECK(!c.Grounded());  // dropped from a height

        Walk(c, w, Vec3{0, 0, 0}, 0.0f, 2.0f);
        std::printf("    dropped from y=3, came to rest with feet at y=%.4f\n",
                    c.Feet().y);
        CHECK(c.Grounded());
        // The feet end ON the floor, within the skin. Sinking in, or hovering a
        // whole radius above it, are the two ways this is normally wrong.
        CHECK(std::fabs(c.Feet().y) < 0.05f);
        CHECK(c.GroundNormal().y > 0.99f);
        // STRICTLY CLEAR of the surface, not exactly touching it. That is the
        // skin's whole job: a capsule resolved to exactly touching overlaps
        // again next frame on floating-point noise alone, and the character
        // buzzes. Landing at exactly zero passes a "did it land" check and
        // fails this one.
        CHECK(c.Feet().y > 0.004f);

        // ...and having landed, it does not creep. A controller that resolves
        // to touching and is pushed out again every frame drifts, and the drift
        // is upward as often as not.
        const Vec3 settled = c.Feet();
        Walk(c, w, Vec3{0, 0, 0}, 0.0f, 3.0f);
        std::printf("    three more seconds standing still: moved %.6f m\n",
                    Length(c.Feet() - settled));
        CHECK(Length(c.Feet() - settled) < 1e-4f);
    }

    // --- it does not fall through the floor at speed -------------------------
    //
    // The discrete move-and-resolve failure: a move longer than the shape ends
    // up on the far side of a wall with nothing overlapping to push it back.
    {
        World w;
        // Long enough for the fastest case: 60 m/s for 1.5 s is 90 m, and a
        // 50 m floor meant the quickest runner simply reached the end of the
        // world. That reads identically to tunnelling in the final position and
        // is not the same bug at all.
        AddBox(w, Vec3{0, -0.5f, 0}, Vec3{200, 0.5f, 50});
        int fell = 0;
        for (int k = 0; k < 12; ++k) {
            const float speed = 5.0f + float(k) * 5.0f;  // up to 60 m/s
            CharacterController c;
            c.Teleport(Vec3{-20.0f, 0.0f, 0});
            Walk(c, w, Vec3{1, 0, 0}, speed, 1.5f);
            if (c.Feet().y < -0.2f) {
                ++fell;
                std::printf("      %.0f m/s ended at y=%.3f x=%.1f\n", speed,
                            c.Feet().y, c.Feet().x);
            }
        }
        std::printf("    12 speeds from 5 to 60 m/s: %d fell through the floor\n",
                    fell);
        CHECK(fell == 0);
    }

    // --- a THIN wall at speed --------------------------------------------------
    //
    // The failure substepping exists to prevent, and a thick floor cannot show
    // it: horizontal motion along a floor never tunnels however fast it is.
    // A 5 cm wall and a sprint is the case -- one move of a metre puts the
    // capsule entirely on the far side, where nothing overlaps to push it back.
    {
        World w;
        AddBox(w, Vec3{0, -0.5f, 0}, Vec3{200, 0.5f, 50});
        AddBox(w, Vec3{10.0f, 1.5f, 0}, Vec3{0.025f, 1.5f, 20.0f});  // 5 cm thick
        int through = 0;
        for (int k = 0; k < 10; ++k) {
            const float speed = 8.0f + float(k) * 6.0f;  // up to 62 m/s
            CharacterController c;
            c.Teleport(Vec3{0.0f, 0.0f, 0});
            Walk(c, w, Vec3{1, 0, 0}, speed, 2.0f);
            if (c.Feet().x > 10.0f) {
                ++through;
                std::printf("      %.0f m/s went through, ending at x=%.2f\n",
                            speed, c.Feet().x);
            }
        }
        std::printf("    10 speeds from 8 to 62 m/s at a 5 cm wall: %d passed "
                    "through it\n", through);
        CHECK(through == 0);
    }

    // --- a wall stops it, and it slides along ---------------------------------
    {
        World w;
        AddBox(w, Vec3{0, -0.5f, 0}, Vec3{20, 0.5f, 20});
        AddBox(w, Vec3{2.0f, 1.5f, 0}, Vec3{0.2f, 1.5f, 8.0f});  // a wall at x=2

        CharacterController c;
        c.Teleport(Vec3{0, 0, 0});
        Walk(c, w, Vec3{1, 0, 0}, 3.0f, 2.0f);
        std::printf("    walked into a wall at x=2.0, stopped at x=%.4f\n",
                    c.Feet().x);
        // Stopped just short: the wall's face is at 1.8 and the capsule's
        // radius is 0.35, so the centre cannot pass 1.45.
        CHECK(c.Feet().x < 1.5f);
        CHECK(c.Feet().x > 1.3f);
        // ...and did NOT lose its footing doing it.
        CHECK(c.Grounded());

        // At 45 degrees into the same wall it must SLIDE along z rather than
        // stopping dead. A controller that cancels the whole motion on contact
        // is the single most common way this is written and it feels terrible.
        CharacterController d;
        d.Teleport(Vec3{0, 0, 0});
        Walk(d, w, Normalize(Vec3{1, 0, 1}), 3.0f, 2.0f);
        std::printf("    walked into it at 45 degrees: ended at x=%.3f z=%.3f\n",
                    d.Feet().x, d.Feet().z);
        CHECK(d.Feet().x < 1.5f);
        CHECK(d.Feet().z > 2.0f);  // it kept going sideways
    }

    // --- it walks over a step and not over a wall -----------------------------
    //
    // The two must be distinguished by HEIGHT, not by slope: a kerb is vertical,
    // so any slope limit permissive enough to climb one would let a character
    // walk up a wall.
    {
        // 0.15 is contacted by the capsule's rounded underside, 0.34 is
        // contacted by its widest point where the surface is vertical and the
        // contact normal is horizontal -- the two reach the step-up by
        // different routes and only the first works by accident.
        const float heights[3] = {0.15f, 0.34f, 1.20f};
        for (int which = 0; which < 3; ++which) {
            const float h = heights[which];
            World w;
            AddBox(w, Vec3{0, -0.5f, 0}, Vec3{20, 0.5f, 20});
            // LONG, so the character ends up standing on it rather than
            // walking across and off the far side -- which is what it did, and
            // which reads in the numbers as "the step-up failed" when it had
            // in fact worked perfectly.
            AddBox(w, Vec3{12.0f, h * 0.5f, 0}, Vec3{10.0f, h * 0.5f, 8.0f});

            CharacterController c;  // step_height defaults to 0.35
            c.Teleport(Vec3{0, 0, 0});
            // Timed, not just eventual. Climbing a kerb should cost a few
            // frames, not a second: a step-up that lets the edge shove it back
            // out horizontally still gets there in the end, one retry at a
            // time, and "it arrived" cannot tell the two apart.
            int frames_to_climb = -1;
            for (int f = 0; f < 180; ++f) {
                Walk(c, w, Vec3{1, 0, 0}, 2.0f, 1.0f / 60.0f);
                if (frames_to_climb < 0 && c.Feet().y > h * 0.8f)
                    frames_to_climb = f;
            }
            if (which < 2)
                std::printf("    a %.2f m obstacle: on top after %d frames\n", h,
                            frames_to_climb);
            std::printf("    a %.2f m obstacle: ended at x=%.3f y=%.3f%s\n", h,
                        c.Feet().x, c.Feet().y,
                        which < 2 ? "  (should be on top)" : "  (should be blocked)");
            if (which < 2) {
                CHECK(c.Feet().x > 2.5f);                 // got onto it
                CHECK(std::fabs(c.Feet().y - h) < 0.06f);  // and stands on top
                CHECK(c.Grounded());
                CHECK(frames_to_climb >= 0);
                // MEASURED at 57-61 frames, which is about a second, and that
                // is slow: walking the 0.35 m needed to get the capsule's
                // underside over the edge should take a fifth of that at
                // 2 m/s. The cost is the controller being discrete -- it moves,
                // then discovers the overlap, then is pushed back, and only the
                // residue is progress. A swept controller would land on the
                // step in a handful of frames.
                //
                // The bound is set to catch a REGRESSION rather than to endorse
                // the number: anything past 90 means it has stopped climbing
                // and started retrying.
                CHECK(frames_to_climb < 90);
            } else {
                CHECK(c.Feet().x < 2.0f);        // stopped in front of it
                CHECK(c.Feet().y < 0.1f);        // and stayed on the ground
            }
        }
    }

    // --- a step under a low ceiling ---------------------------------------------
    //
    // A 0.30 kerb with a beam at 2.0 m. Standing, the 1.8 m character fits
    // under it; standing ON the kerb they would not. So the step-up must be
    // refused -- and refused by checking for HEADROOM before committing to it,
    // because the alternative is a character wedged into the ceiling.
    {
        World w;
        AddBox(w, Vec3{0, -0.5f, 0}, Vec3{20, 0.5f, 20});
        AddBox(w, Vec3{12.0f, 0.15f, 0}, Vec3{10.0f, 0.15f, 8.0f});   // the kerb
        AddBox(w, Vec3{12.0f, 2.3f, 0}, Vec3{10.0f, 0.3f, 8.0f});     // the beam

        CharacterController c;
        c.Teleport(Vec3{0, 0, 0});
        Walk(c, w, Vec3{1, 0, 0}, 2.0f, 3.0f);
        std::printf("    a 0.30 kerb under a 2.0 m beam: ended at x=%.3f "
                    "y=%.3f\n", c.Feet().x, c.Feet().y);
        // Stopped at the kerb, still on the floor and not inside the beam.
        CHECK(c.Feet().y < 0.1f);
        CHECK(c.Feet().x < 2.2f);

        // And not overlapping anything, which is the failure the headroom check
        // exists to prevent: a step-up that commits before looking leaves the
        // character standing on the kerb with its head through the beam.
        std::vector<int> hits;
        QueryFilter f;
        w.OverlapShape(c.Capsule(), c.Centre(), Quat{}, &hits, f);
        int deep = 0;
        for (int b : hits) {
            Body probe;
            probe.shape = c.Capsule();
            probe.position = c.Centre();
            Contact ct;
            if (CollideConvex(probe, w[b], &ct) && ct.depth > 0.05f) ++deep;
        }
        std::printf("    overlapping %d bodies by more than 5 cm\n", deep);
        CHECK(deep == 0);
    }

    // --- slopes ---------------------------------------------------------------
    //
    // Below the limit it walks up and stands. Above it, it must not: a
    // controller with no slope limit walks up anything short of vertical, and
    // the first thing a player does is find the steepest thing in the level and
    // walk up it.
    {
        for (int deg = 20; deg <= 70; deg += 25) {
            World w;
            // A ramp whose TOP SURFACE passes through the origin and rises
            // along +x. Placing a rotated slab by its centre instead puts its
            // lower end below the floor and its end FACE in the character's
            // way, so the test measures walking into the side of a ramp rather
            // than up it.
            // A floor to stand on at the bottom, so a ramp too steep to climb
            // leaves the character at its foot rather than falling forever --
            // "it did not climb" and "it fell out of the world" are different
            // outcomes and only one of them is the thing being tested.
            AddBox(w, Vec3{-6.0f, -0.5f, 0}, Vec3{6.0f, 0.5f, 6.0f});
            const float rad = float(deg) * 3.14159265f / 180.0f;
            const float L = 9.0f, t = 0.5f;
            AddBox(w,
                   Vec3{L * std::cos(rad) + t * std::sin(rad),
                        L * std::sin(rad) - t * std::cos(rad), 0.0f},
                   Vec3{L, t, 6.0f}, QuatFromAxisAngle(Vec3{0, 0, 1}, rad));

            CharacterController c;  // slope_limit defaults to 50 degrees
            c.Teleport(Vec3{0.1f, 0.1f, 0});
            Walk(c, w, Vec3{1, 0, 0}, 2.5f, 4.0f);
            std::printf("    a %d degree ramp: climbed to y=%.3f, grounded=%d\n",
                        deg, c.Feet().y, int(c.Grounded()));
            if (deg < 50) {
                // 4 s at 2.5 m/s is 10 m along the ground; on a ramp of this
                // angle that is 10*tan(deg) of height, less whatever the slope
                // costs. Well over half a metre either way.
                CHECK(c.Feet().y > 1.0f);
                CHECK(c.Grounded());
            } else {
                // Above the limit it is not ground. It may be touched, and the
                // step-up may borrow its budget once, but the character does
                // not climb it.
                CHECK(c.Feet().y < 0.8f);
                CHECK(c.Feet().y > -1.0f);  // and did not fall out of the world
            }
        }
    }

    // --- a floor seam ----------------------------------------------------------
    //
    // Two flush floor tiles. The join between them is a vertical face of zero
    // height, and a controller that treats any horizontal obstruction as a wall
    // stops dead on it. It is the most-reported bug a character controller has
    // and it does not exist in a level made of one big box.
    {
        World w;
        for (int i = 0; i < 8; ++i)
            AddBox(w, Vec3{float(i) * 2.0f, -0.5f, 0}, Vec3{1.0f, 0.5f, 6.0f});
        CharacterController c;
        c.Teleport(Vec3{0.0f, 0.0f, 0});
        Walk(c, w, Vec3{1, 0, 0}, 3.0f, 4.0f);
        std::printf("    across 8 flush floor tiles: reached x=%.3f of a "
                    "possible 12.0\n", c.Feet().x);
        CHECK(c.Feet().x > 10.0f);
        CHECK(std::fabs(c.Feet().y) < 0.05f);
        CHECK(c.Grounded());
    }

    // --- it never ends up inside anything ----------------------------------------
    //
    // The invariant that underpins every one of the above. Checked directly
    // rather than inferred from a position, because a character wedged inside a
    // wall still has a plausible position.
    {
        World w;
        AddBox(w, Vec3{0, -0.5f, 0}, Vec3{20, 0.5f, 20});
        for (int i = 0; i < 10; ++i)
            AddBox(w, Vec3{float(i % 5) * 2.0f - 4.0f, 0.6f,
                           float(i / 5) * 3.0f - 1.5f},
                   Vec3{0.5f, 0.6f, 0.5f});

        CharacterController c;
        c.Teleport(Vec3{-6.0f, 0.5f, -4.0f});
        int inside = 0;
        std::vector<int> hits;
        for (int i = 0; i < 600; ++i) {
            const float t = float(i) * 0.05f;
            c.Move(w, Vec3{std::cos(t) * 0.06f, -0.08f, std::sin(t * 0.7f) * 0.06f});
            hits.clear();
            // The capsule at its resolved position must not overlap anything by
            // more than the skin it deliberately keeps.
            QueryFilter f;
            w.OverlapShape(c.Capsule(), c.Centre(), Quat{}, &hits, f);
            for (int b : hits) {
                Body probe;
                probe.shape = c.Capsule();
                probe.position = c.Centre();
                Contact ct;
                if (CollideConvex(probe, w[b], &ct) && ct.depth > 0.05f) ++inside;
            }
        }
        std::printf("    600 steps through a field of blocks: %d frames "
                    "overlapping by more than 5 cm\n", inside);
        CHECK(inside == 0);
    }

    std::printf(g_failures == 0 ? "\ncharacter_test: all checks passed\n"
                                : "\ncharacter_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
