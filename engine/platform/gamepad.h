// Gamepads, through the OS's own controller framework.
//
// A SEPARATE TARGET from :platform, and separate from the keyboard and mouse in
// Window, for one reason: a controller is not attached to a window. It is a
// device the system owns, it arrives and leaves while the game runs, and a
// headless tool that wants to read one has no business linking AppKit to do it.
//
// POLLED, not evented. A gamepad's state is continuous -- a stick is somewhere
// whether or not it moved -- and the natural question is "where is it now", not
// "what happened". Buttons get edges on top, because "did they press jump" is
// genuinely a different question from "is jump held", and deriving the first
// from the second means every caller keeping its own copy of last frame.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace eng::platform {

// The layout every modern controller reports, whatever it is called on the
// hardware. Named by POSITION, not by letter: A/B/X/Y are swapped between
// Xbox and Nintendo layouts, and a call site that says `Button::A` is wrong on
// half the controllers in the world.
enum class PadButton : int {
    South = 0,   // A on Xbox, Cross on PlayStation. Jump, confirm.
    East = 1,    // B / Circle. Back, cancel.
    West = 2,    // X / Square.
    North = 3,   // Y / Triangle.
    LeftShoulder = 4,
    RightShoulder = 5,
    LeftStick = 6,   // the stick pressed in
    RightStick = 7,
    DPadUp = 8,
    DPadDown = 9,
    DPadLeft = 10,
    DPadRight = 11,
    Start = 12,
    Select = 13,
    Count = 14,
};

struct PadState {
    bool connected = false;
    // Sticks, each in -1..1, with Y POSITIVE UP -- the same convention as the
    // engine's world space, so a caller can use the value as a direction
    // without a sign flip nobody remembers to write.
    //
    // Already deadzoned and rescaled: see GamepadConfig.
    float left_x = 0.0f, left_y = 0.0f;
    float right_x = 0.0f, right_y = 0.0f;
    // Analogue triggers, 0..1.
    float left_trigger = 0.0f, right_trigger = 0.0f;

    bool down[int(PadButton::Count)] = {};
    // Edges since the last Poll. Both, because a menu wants the press and a
    // charged attack wants the release.
    bool pressed[int(PadButton::Count)] = {};
    bool released[int(PadButton::Count)] = {};

    [[nodiscard]] bool Down(PadButton b) const { return down[int(b)]; }
    [[nodiscard]] bool Pressed(PadButton b) const { return pressed[int(b)]; }
    [[nodiscard]] bool Released(PadButton b) const { return released[int(b)]; }
};

struct GamepadConfig {
    // RADIAL deadzone, applied to the stick's magnitude and not to each axis.
    //
    // Per-axis deadzoning is the common mistake and it is visible in play: it
    // carves a cross out of the stick's range, so a stick pushed diagonally at
    // low magnitude registers on one axis only, and a character asked to walk
    // slowly north-east walks slowly north. Radial keeps the circle.
    float deadzone = 0.18f;
    // What is left after the deadzone is RESCALED to the full 0..1, so the
    // first movement past the threshold is a small input rather than a jump to
    // 0.18. Without it every stick feels like it has a step in it.
    bool rescale = true;
    // Trigger travel below this reads as zero. Smaller than the stick deadzone
    // because a trigger rests against a stop and does not drift.
    float trigger_deadzone = 0.06f;
};

class Gamepads {
  public:
    // Always succeeds. A machine with no controllers, or an OS build without
    // the framework, gives a Gamepads that reports nothing connected -- because
    // "no gamepad" is the normal case and must not be an error path the caller
    // has to write.
    [[nodiscard]] static std::unique_ptr<Gamepads> Create(const GamepadConfig& = {});
    ~Gamepads();

    Gamepads(const Gamepads&) = delete;
    Gamepads& operator=(const Gamepads&) = delete;

    // Reads every connected pad and computes this frame's edges. Call once per
    // frame, before anything asks for state.
    void Poll();

    // Up to this many. Four is the number of players who fit on a sofa.
    static constexpr int kMaxPads = 4;
    [[nodiscard]] const PadState& Pad(int index) const;
    // The first connected pad, or a disconnected state if there is none. The
    // overwhelmingly common single-player call, and it saves every caller
    // writing the same scan.
    [[nodiscard]] const PadState& First() const;
    [[nodiscard]] int ConnectedCount() const;
    // What the OS calls the first connected pad -- "Xbox Wireless Controller".
    // For a settings screen, and for telling the user which device the game is
    // actually listening to when two are plugged in.
    [[nodiscard]] std::string FirstName() const;

    // --- for tests ------------------------------------------------------------
    //
    // Injects a raw state as if it had come from the OS, so the deadzone, the
    // rescaling and the edge detection can be checked without hardware. Those
    // three are the entire logic of this class; the rest is a framework call.
    void InjectForTest(int index, const PadState& raw);
    // Runs the edge detection over whatever was injected. What Poll does, minus
    // talking to the OS.
    void ResolveForTest();

  private:
    Gamepads();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// The deadzone and rescale, exposed because it is the part worth testing and
// because a caller reading a stick from somewhere else -- a network packet, a
// replay -- wants exactly the same curve applied.
void ApplyDeadzone(float* x, float* y, float deadzone, bool rescale);

}  // namespace eng::platform
