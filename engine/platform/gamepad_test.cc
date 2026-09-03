// The parts of gamepad handling that are arithmetic rather than framework
// calls: the radial deadzone, the rescale, and the button edges.
//
// Those three are where the bugs are. Reading GCExtendedGamepad's fields either
// works or does not and is visible in five seconds of play; a deadzone applied
// per-axis instead of radially is subtly wrong for months, and a diagonal that
// is 41% too fast is the single most common movement bug in shipped games.

#include <cmath>
#include <cstdio>

#include "engine/platform/gamepad.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

float Mag(float x, float y) { return std::sqrt(x * x + y * y); }

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    using namespace eng::platform;

    {
        std::printf("the deadzone is radial, not per-axis\n");
        // A stick pushed diagonally, each axis below the threshold but the
        // MAGNITUDE above it. Per-axis deadzoning zeroes both and the character
        // does not move; radial keeps it.
        float x = 0.15f, y = 0.15f;  // magnitude 0.212, deadzone 0.18
        ApplyDeadzone(&x, &y, 0.18f, true);
        std::printf("    (0.15, 0.15) magnitude %.3f -> (%.3f, %.3f)\n",
                    Mag(0.15f, 0.15f), x, y);
        Check(Mag(x, y) > 0.0f, "a diagonal past the threshold survives");
        // And the direction is preserved exactly -- a rescale that changed it
        // would make the stick pull toward the axes.
        Check(std::fabs(x - y) < 1e-6f, "and keeps its 45 degree direction");

        float ax = 0.15f, ay = 0.0f;
        ApplyDeadzone(&ax, &ay, 0.18f, true);
        Check(ax == 0.0f && ay == 0.0f, "a single axis below the threshold is zeroed");
    }

    {
        std::printf("\nthe rescale removes the step at the threshold\n");
        // Just past the deadzone must read as nearly zero. Without the rescale
        // it reads as 0.18 and the character lurches into motion.
        float x = 0.19f, y = 0.0f;
        ApplyDeadzone(&x, &y, 0.18f, true);
        std::printf("    0.19 in -> %.4f out\n", x);
        Check(x < 0.02f, "just past the threshold is nearly zero");

        float fx = 1.0f, fy = 0.0f;
        ApplyDeadzone(&fx, &fy, 0.18f, true);
        std::printf("    1.00 in -> %.4f out\n", fx);
        Check(std::fabs(fx - 1.0f) < 1e-5f, "and full deflection is still exactly one");

        // Monotonic, with no reversal anywhere -- a curve that dips would feel
        // like the stick sticking.
        bool monotonic = true;
        float last = -1.0f;
        for (int i = 0; i <= 100; ++i) {
            float v = float(i) / 100.0f, z = 0.0f;
            ApplyDeadzone(&v, &z, 0.18f, true);
            if (v < last - 1e-6f) monotonic = false;
            last = v;
        }
        Check(monotonic, "the response curve never goes backwards");
    }

    {
        // THE DIAGONAL SPEED BUG. A stick at full deflection on both axes
        // reports (1, 1), whose magnitude is 1.414. A rescale that divides by
        // the raw magnitude without clamping hands back a vector 41% too long,
        // and a movement system multiplies that straight into speed.
        std::printf("\nfull diagonal deflection is not 41%% too fast\n");
        float x = 1.0f, y = 1.0f;
        ApplyDeadzone(&x, &y, 0.18f, true);
        std::printf("    (1, 1) magnitude %.4f -> (%.4f, %.4f) magnitude %.4f\n",
                    Mag(1.0f, 1.0f), x, y, Mag(x, y));
        Check(Mag(x, y) <= 1.0f + 1e-5f, "the result never exceeds unit length");
        Check(std::fabs(x - y) < 1e-6f, "and is still on the diagonal");
    }

    {
        std::printf("\nno deadzone at all is a pass-through\n");
        float x = 0.03f, y = -0.02f;
        ApplyDeadzone(&x, &y, 0.0f, true);
        Check(std::fabs(x - 0.03f) < 1e-6f && std::fabs(y + 0.02f) < 1e-6f,
              "a zero deadzone changes nothing");
        float cx = 0.0f, cy = 0.0f;
        ApplyDeadzone(&cx, &cy, 0.18f, true);
        Check(cx == 0.0f && cy == 0.0f, "and a centred stick does not divide by zero");
    }

    {
        std::printf("\nbutton edges fire once\n");
        auto pads = Gamepads::Create();
        PadState raw;
        raw.connected = true;

        // Frame 1: nothing.
        pads->InjectForTest(0, raw);
        pads->ResolveForTest();
        Check(!pads->Pad(0).Pressed(PadButton::South), "an untouched button has no edge");

        // Frame 2: pressed.
        raw.down[int(PadButton::South)] = true;
        pads->InjectForTest(0, raw);
        pads->ResolveForTest();
        Check(pads->Pad(0).Pressed(PadButton::South), "pressing fires the press edge");
        Check(pads->Pad(0).Down(PadButton::South), "and the button reads as held");

        // Frame 3: still held. THE CHECK THAT MATTERS -- an edge that repeats is
        // a menu that scrolls past every entry in one frame.
        pads->InjectForTest(0, raw);
        pads->ResolveForTest();
        Check(!pads->Pad(0).Pressed(PadButton::South), "holding does not fire it again");
        Check(pads->Pad(0).Down(PadButton::South), "but it is still held");

        // Frame 4: released.
        raw.down[int(PadButton::South)] = false;
        pads->InjectForTest(0, raw);
        pads->ResolveForTest();
        Check(pads->Pad(0).Released(PadButton::South), "releasing fires the release edge");
        Check(!pads->Pad(0).Down(PadButton::South), "and it is no longer held");

        pads->InjectForTest(0, raw);
        pads->ResolveForTest();
        Check(!pads->Pad(0).Released(PadButton::South), "which also does not repeat");
    }

    {
        std::printf("\nan absent pad reads as nothing, not as garbage\n");
        auto pads = Gamepads::Create();
        pads->ResolveForTest();
        Check(!pads->First().connected, "First() on no pads is disconnected");
        Check(pads->ConnectedCount() == 0, "and the count is zero");
        Check(!pads->Pad(-1).connected && !pads->Pad(99).connected,
              "an out-of-range index is disconnected, not a crash");
        // Every accessor on a disconnected pad must be safe and false: a game
        // reading the stick before checking `connected` is the normal case.
        Check(pads->First().left_x == 0.0f && !pads->First().Down(PadButton::Start),
              "and its axes and buttons read as neutral");
    }

    {
        std::printf("\ntriggers are deadzoned but not rescaled\n");
        GamepadConfig cfg;
        cfg.trigger_deadzone = 0.06f;
        auto pads = Gamepads::Create(cfg);
        PadState raw;
        raw.connected = true;
        raw.left_trigger = 0.04f;
        raw.right_trigger = 0.50f;
        pads->InjectForTest(0, raw);
        pads->ResolveForTest();
        Check(pads->Pad(0).left_trigger == 0.0f, "resting travel reads as zero");
        // NOT rescaled: a trigger is an absolute quantity -- how far in it is --
        // and rescaling it would mean half-pressed no longer meant half.
        Check(std::fabs(pads->Pad(0).right_trigger - 0.50f) < 1e-6f,
              "and a half pull still reads as one half");
    }

    std::printf(g_failures == 0 ? "\ngamepad_test: all checks passed\n"
                                : "\ngamepad_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
