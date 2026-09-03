#include "engine/platform/gamepad.h"

#import <Foundation/Foundation.h>
#import <GameController/GameController.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace eng::platform {

void ApplyDeadzone(float* x, float* y, float deadzone, bool rescale) {
    const float mag = std::sqrt(*x * *x + *y * *y);
    if (mag <= deadzone || mag <= 1e-6f) {
        *x = 0.0f;
        *y = 0.0f;
        return;
    }
    if (!rescale) return;
    // The remaining travel stretched back over the full range. Note the
    // magnitude is CLAMPED to one first: a stick reporting 1.0 on both axes has
    // a magnitude of 1.41, and rescaling that would hand the caller a direction
    // 40% too long -- which a movement system multiplies straight into speed,
    // so the character runs faster diagonally. The classic bug.
    const float clamped = std::min(mag, 1.0f);
    const float scaled = (clamped - deadzone) / (1.0f - deadzone);
    const float k = scaled / mag;
    *x *= k;
    *y *= k;
}

namespace {

float Trigger(float v, float deadzone) { return v <= deadzone ? 0.0f : v; }

}  // namespace

struct Gamepads::Impl {
    GamepadConfig config;
    PadState pads[kMaxPads];
    // Last frame's held state, kept separately so the edges can be recomputed
    // without the raw read having to know about them.
    bool was_down[kMaxPads][int(PadButton::Count)] = {};
    PadState disconnected;
    std::string first_name;
};

Gamepads::Gamepads() : impl_(std::make_unique<Impl>()) {}
Gamepads::~Gamepads() = default;

std::unique_ptr<Gamepads> Gamepads::Create(const GamepadConfig& config) {
    std::unique_ptr<Gamepads> g(new Gamepads());
    g->impl_->config = config;
    // No error path. GCController is present on every macOS this engine builds
    // for, and a machine with nothing plugged in is not a failure -- it is
    // Tuesday. Returning null here would make every caller write a branch for
    // the normal case.
    return g;
}

void Gamepads::Poll() {
    @autoreleasepool {
        NSArray<GCController*>* all = [GCController controllers];
        const int n = std::min(int(all.count), kMaxPads);
        impl_->first_name.clear();

        for (int i = 0; i < kMaxPads; ++i) {
            PadState& p = impl_->pads[i];
            const bool was_connected = p.connected;
            p = PadState{};
            if (i >= n) {
                // A pad that has just been unplugged must not leave its buttons
                // reading as held forever -- a character would keep running.
                // Clearing the whole state is what does that, and clearing the
                // remembered edges below is what stops a phantom release the
                // next time one is plugged in.
                if (was_connected)
                    for (int b = 0; b < int(PadButton::Count); ++b)
                        impl_->was_down[i][b] = false;
                continue;
            }

            GCController* c = all[NSUInteger(i)];
            GCExtendedGamepad* g = c.extendedGamepad;
            if (!g) continue;  // a remote or a micro gamepad; not a controller
            p.connected = true;
            if (impl_->first_name.empty())
                impl_->first_name = c.vendorName ? c.vendorName.UTF8String : "gamepad";

            p.left_x = g.leftThumbstick.xAxis.value;
            p.left_y = g.leftThumbstick.yAxis.value;
            p.right_x = g.rightThumbstick.xAxis.value;
            p.right_y = g.rightThumbstick.yAxis.value;
            ApplyDeadzone(&p.left_x, &p.left_y, impl_->config.deadzone,
                          impl_->config.rescale);
            ApplyDeadzone(&p.right_x, &p.right_y, impl_->config.deadzone,
                          impl_->config.rescale);
            p.left_trigger = Trigger(g.leftTrigger.value, impl_->config.trigger_deadzone);
            p.right_trigger = Trigger(g.rightTrigger.value, impl_->config.trigger_deadzone);

            // By POSITION. GCExtendedGamepad names its faces after the Xbox
            // layout, and the framework already remaps a PlayStation or Switch
            // pad onto those names by position -- so buttonA is always the
            // bottom face whatever is printed on it.
            p.down[int(PadButton::South)] = g.buttonA.pressed;
            p.down[int(PadButton::East)] = g.buttonB.pressed;
            p.down[int(PadButton::West)] = g.buttonX.pressed;
            p.down[int(PadButton::North)] = g.buttonY.pressed;
            p.down[int(PadButton::LeftShoulder)] = g.leftShoulder.pressed;
            p.down[int(PadButton::RightShoulder)] = g.rightShoulder.pressed;
            p.down[int(PadButton::LeftStick)] = g.leftThumbstickButton.pressed;
            p.down[int(PadButton::RightStick)] = g.rightThumbstickButton.pressed;
            p.down[int(PadButton::DPadUp)] = g.dpad.up.pressed;
            p.down[int(PadButton::DPadDown)] = g.dpad.down.pressed;
            p.down[int(PadButton::DPadLeft)] = g.dpad.left.pressed;
            p.down[int(PadButton::DPadRight)] = g.dpad.right.pressed;
            p.down[int(PadButton::Start)] = g.buttonMenu.pressed;
            p.down[int(PadButton::Select)] = g.buttonOptions.pressed;
        }
    }
    ResolveForTest();
}

void Gamepads::ResolveForTest() {
    for (int i = 0; i < kMaxPads; ++i) {
        PadState& p = impl_->pads[i];
        for (int b = 0; b < int(PadButton::Count); ++b) {
            const bool now = p.down[b];
            const bool before = impl_->was_down[i][b];
            p.pressed[b] = now && !before;
            p.released[b] = !now && before;
            impl_->was_down[i][b] = now;
        }
    }
}

void Gamepads::InjectForTest(int index, const PadState& raw) {
    if (index < 0 || index >= kMaxPads) return;
    PadState p = raw;
    ApplyDeadzone(&p.left_x, &p.left_y, impl_->config.deadzone, impl_->config.rescale);
    ApplyDeadzone(&p.right_x, &p.right_y, impl_->config.deadzone, impl_->config.rescale);
    p.left_trigger = Trigger(p.left_trigger, impl_->config.trigger_deadzone);
    p.right_trigger = Trigger(p.right_trigger, impl_->config.trigger_deadzone);
    impl_->pads[index] = p;
}

const PadState& Gamepads::Pad(int index) const {
    if (index < 0 || index >= kMaxPads) return impl_->disconnected;
    return impl_->pads[index];
}

const PadState& Gamepads::First() const {
    for (int i = 0; i < kMaxPads; ++i)
        if (impl_->pads[i].connected) return impl_->pads[i];
    return impl_->disconnected;
}

int Gamepads::ConnectedCount() const {
    int n = 0;
    for (int i = 0; i < kMaxPads; ++i)
        if (impl_->pads[i].connected) ++n;
    return n;
}

std::string Gamepads::FirstName() const { return impl_->first_name; }

}  // namespace eng::platform
