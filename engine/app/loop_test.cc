// No test framework — from scratch means from scratch.
//
// These two types decide how the whole engine advances, so they take a clock
// reading and a key bitset as ARGUMENTS rather than reading a real clock or a
// real window. That is the only reason a timing policy can be tested at all:
// the interesting cases are a two-second stall and a key held across frames,
// and neither is reachable by driving a window by hand.
#include "engine/app/actions.h"
#include "engine/app/clock.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace {

int g_failures = 0;

void Fail(const char* what, int line) {
    std::fprintf(stderr, "loop_test.cc:%d  %s\n", line, what);
    ++g_failures;
}

#define CHECK(cond) \
    do { if (!(cond)) Fail(#cond, __LINE__); } while (0)

using namespace eng::app;

}  // namespace

int main() {
    // --- the first frame has no previous frame to measure -----------------------
    {
        Clock c;
        // Starting at a large absolute time must not produce a large dt. Using
        // the raw reading here is how a demo jolts on its first frame.
        CHECK(c.Tick(1234.5) == 0.0f);
        CHECK(c.Frame() == 1);
        CHECK(c.Total() == 0.0f);
        CHECK(std::fabs(c.Tick(1234.5 + 1.0 / 60.0) - 1.0f / 60.0f) < 1e-5f);
        CHECK(c.Frame() == 2);
    }

    // --- a stall is clamped, and the clamp is not silent ------------------------
    {
        Clock c;
        c.max_dt = 0.1f;
        c.Tick(0.0);
        const float dt = c.Tick(2.0);  // two seconds: a window drag
        CHECK(std::fabs(dt - 0.1f) < 1e-6f);
        // RawDt still reports what really happened, so a caller that wants to
        // know it stalled can find out.
        CHECK(std::fabs(c.RawDt() - 2.0f) < 1e-4f);
        // Total follows the simulation, not the wall.
        CHECK(std::fabs(c.Total() - 0.1f) < 1e-6f);
    }

    // --- pausing freezes the world but not the app ------------------------------
    {
        Clock c;
        c.Tick(0.0);
        c.Tick(0.1);
        const float before = c.Total();
        c.SetPaused(true);
        CHECK(c.Tick(0.2) == 0.0f);
        CHECK(c.Tick(0.3) == 0.0f);
        CHECK(c.Total() == before);      // the world did not move
        CHECK(c.Frame() == 4);           // but frames still happened
        CHECK(c.RawDt() > 0.0f);         // and real time still passed
        c.SetPaused(false);
        CHECK(c.Tick(0.4) > 0.0f);
        CHECK(c.Total() > before);
    }

    // --- a clock that goes backwards is a reset, not negative time --------------
    {
        Clock c;
        c.Tick(100.0);
        CHECK(c.Tick(50.0) == 0.0f);  // never negative: that would run physics backwards
        CHECK(c.RawDt() == 0.0f);
    }

    // --- fps is smoothed rather than a per-frame reciprocal ---------------------
    {
        Clock c;
        c.Tick(0.0);
        double t = 0.0;
        for (int i = 0; i < 400; ++i) {
            t += 1.0 / 60.0;
            c.Tick(t);
        }
        CHECK(std::fabs(c.Fps() - 60.0f) < 1.0f);

        // The thing smoothing is FOR: ordinary frame-to-frame jitter must not
        // make the readout swing. A raw reciprocal of a +/-25% jittering frame
        // time bounces between 48 and 80 and is unreadable.
        //
        // It is deliberately NOT for hiding hitches — a real 200 ms stall
        // should show up, and asserting that it does not would be asserting
        // that the number lies.
        unsigned seed = 12345;
        float lo = 1e9f, hi = -1e9f;
        for (int i = 0; i < 600; ++i) {
            seed = seed * 1103515245u + 12345u;
            const double jitter = 1.0 + ((seed >> 16) % 500) / 1000.0 - 0.25;
            t += (1.0 / 60.0) * jitter;
            c.Tick(t);
            if (i > 100) {  // let it settle
                lo = std::fmin(lo, c.Fps());
                hi = std::fmax(hi, c.Fps());
            }
        }
        CHECK(hi - lo < 8.0f);          // steady, despite +/-25% per frame
        CHECK(lo > 50.0f && hi < 72.0f);  // and still centred on the truth

        // A sustained change does get through, eventually.
        for (int i = 0; i < 600; ++i) {
            t += 1.0 / 30.0;
            c.Tick(t);
        }
        CHECK(std::fabs(c.Fps() - 30.0f) < 1.0f);
    }

    // --- pressed is an EDGE, down is a LEVEL ------------------------------------
    {
        ActionMap m;
        m.Bind("reset", 'r');
        m.Bind("pause", ' ');

        Keys k;
        m.Update(k);
        CHECK(!m.Pressed("reset") && !m.Down("reset"));

        k.Set('r', true);
        m.Update(k);
        CHECK(m.Pressed("reset"));
        CHECK(m.Down("reset"));

        // Held for another ten frames: still down, pressed exactly once. This
        // is the whole point — a reset bound to the level would fire every
        // frame the key is held.
        int presses = 0;
        for (int i = 0; i < 10; ++i) {
            m.Update(k);
            if (m.Pressed("reset")) ++presses;
            CHECK(m.Down("reset"));
        }
        CHECK(presses == 0);

        k.Set('r', false);
        m.Update(k);
        CHECK(m.Released("reset"));
        CHECK(!m.Down("reset"));
        m.Update(k);
        CHECK(!m.Released("reset"));  // the edge is one frame wide

        // Actions are independent.
        CHECK(!m.Down("pause"));
    }

    // --- several keys can drive one action --------------------------------------
    {
        ActionMap m;
        m.Bind("fire", 'f');
        m.Bind("fire", 'x');
        Keys k;
        k.Set('x', true);
        m.Update(k);
        CHECK(m.Down("fire"));
        k.Set('x', false);
        k.Set('f', true);
        m.Update(k);
        CHECK(m.Down("fire"));
        // Still held via the other key, so this is NOT a fresh press.
        CHECK(!m.Pressed("fire"));
    }

    // --- axes read -1, 0, +1, and cancel ----------------------------------------
    {
        ActionMap m;
        m.BindAxis("forward", 's', 'w');
        Keys k;
        m.Update(k);
        CHECK(m.Axis("forward") == 0.0f);
        k.Set('w', true);
        m.Update(k);
        CHECK(m.Axis("forward") == 1.0f);
        CHECK(m.Down("forward"));  // an axis key holds the action too
        k.Set('s', true);
        m.Update(k);
        CHECK(m.Axis("forward") == 0.0f);  // both directions cancel
        k.Set('w', false);
        m.Update(k);
        CHECK(m.Axis("forward") == -1.0f);
    }

    // --- unbound names are quiet, and bindings can be listed --------------------
    {
        ActionMap m;
        m.Bind("jump", ' ');
        m.BindAxis("strafe", 'a', 'd');
        Keys k;
        m.Update(k);
        CHECK(!m.Down("nothing"));
        CHECK(!m.Pressed("nothing"));
        CHECK(m.Axis("nothing") == 0.0f);

        const std::vector<std::string> names = m.Names();
        CHECK(names.size() == 2);
        CHECK(names[0] == "jump" && names[1] == "strafe");
    }

    // --- a non-ascii key does not corrupt the bitset ----------------------------
    {
        Keys k;
        k.Set(char(0xE9), true);  // out of range: ignored rather than wrapping
        CHECK(!k.Get(char(0xE9)));
        k.Set('~', true);  // 126, in the high word
        CHECK(k.Get('~'));
        CHECK(!k.Get('a'));
    }

    if (g_failures == 0) std::printf("loop_test: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
