// Pure C++20. Named actions instead of raw key codes.
//
// WHY: platform::Input reports characters typed and keys held. Every app then
// hard-codes `if (k == 'r') Reset()` in its event loop, which means the
// bindings are scattered through gameplay code, cannot be listed, cannot be
// rebound, and cannot be tested without a window.
//
// The distinction this exists to make is PRESSED versus DOWN. "Jump" wants the
// edge — once per press, no matter how long the key is held. "Walk forward"
// wants the level — every frame while it is held. Conflating them gives you
// either a jump that fires sixty times a second or a walk that moves one step
// per press, and both look like bugs somewhere else entirely.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace eng::app {

// Deliberately NOT platform::MouseButton, even though the values match.
//
// This library is pure C++ and is tested without a window, a GPU or AppKit --
// that is the whole reason the timing and input logic lives apart from the
// screen-sized targets next door. Including the platform header to reuse three
// enumerators would give that up. App translates between them and asserts the
// values agree, so the duplication cannot drift silently.
enum class MouseButton : int { Left = 0, Right = 1, Middle = 2, Count = 3 };

// Which ASCII keys are held right now. A plain bitset rather than a callback
// into the window, so the whole mapping can be driven from a test.
struct Keys {
    std::uint64_t low = 0, high = 0;  // 0..63 and 64..127

    void Set(char c, bool down) {
        const auto i = static_cast<unsigned char>(c);
        if (i >= 128) return;
        std::uint64_t& word = i < 64 ? low : high;
        const std::uint64_t bit = std::uint64_t{1} << (i & 63);
        if (down) word |= bit;
        else word &= ~bit;
    }
    [[nodiscard]] bool Get(char c) const {
        const auto i = static_cast<unsigned char>(c);
        if (i >= 128) return false;
        return ((i < 64 ? low : high) >> (i & 63)) & 1;
    }
};

class ActionMap {
  public:
    // Several keys may drive one action; one key may drive several actions.
    void Bind(std::string action, char key);
    // The same action names, bound to a mouse button. One namespace on purpose:
    // "fire" should not care whether it is the left button or the space bar,
    // and a caller that has to ask which is a caller that cannot be rebound.
    void BindMouse(std::string action, MouseButton);
    // A pair of keys reading as -1 and +1, which is what a movement axis is.
    void BindAxis(std::string axis, char negative, char positive);

    // Once per frame, before anything reads an action.
    //
    // `typed` is the characters the window saw this frame and `held` is the
    // keyboard state. Edges are derived from `held` rather than from `typed`:
    // key repeat makes a held key produce a stream of characters, so `typed`
    // would fire a press over and over.
    // `mouse` is a bitmask of held buttons, one bit per MouseButton, so this
    // signature does not grow a parameter every time a device is added.
    void Update(Keys held, std::uint32_t mouse = 0);

    [[nodiscard]] bool Pressed(std::string_view action) const;   // went down
    [[nodiscard]] bool Released(std::string_view action) const;  // came up
    [[nodiscard]] bool Down(std::string_view action) const;      // is held
    [[nodiscard]] float Axis(std::string_view axis) const;       // -1, 0 or +1

    // Every bound action name, sorted. So a demo can print its own controls
    // instead of a comment drifting out of date next to the loop.
    [[nodiscard]] std::vector<std::string> Names() const;

  private:
    struct Binding {
        std::vector<char> keys;         // any of these
        std::uint32_t mouse_mask = 0;   // ...or any of these buttons
        char axis_negative = 0;
        char axis_positive = 0;
        bool down = false, was_down = false;
        float axis = 0.0f;
    };
    Binding& Entry(const std::string& name);
    [[nodiscard]] const Binding* Find(std::string_view name) const;

    std::unordered_map<std::string, Binding> bindings_;
};

}  // namespace eng::app
