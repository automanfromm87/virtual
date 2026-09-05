// Pure C++20. Windows, events and the main loop — the OS side of things.
//
// Knows nothing about Metal. It hosts whatever native layer the RHI hands it
// and never looks inside, which is why the swapchain lives in engine/rhi and
// window-system integration lives here.
#pragma once

#include <memory>
#include <string>

namespace eng::platform {

// Everything that happened since the last TakeInput(). Deltas, not absolute
// state: a camera controller wants "how far did the mouse move", and making it
// difference two absolute positions itself is how frame-boundary bugs start.
// Which mouse button. Named rather than numbered: `button == 2` at a call site
// is a coin flip between middle and right, and the two conventions disagree.
enum class MouseButton : int { Left = 0, Right = 1, Middle = 2, Count = 3 };

struct Input {
    // Motion with the left button held. Kept separate from `mouse_dx` because
    // an orbit camera wants exactly this and nothing else -- moving the cursor
    // across the window must not spin the view.
    float drag_dx = 0.0f;
    float drag_dy = 0.0f;
    // ALL motion, held or not. What a first-person look needs, since a game
    // that makes you hold a button to turn your head is not a first-person
    // game.
    float mouse_dx = 0.0f;
    float mouse_dy = 0.0f;
    float scroll = 0.0f;
    std::string keys;  // characters typed this frame, lowercased — for toggles

    // The cursor, in PIXELS with the origin at the TOP LEFT -- the same
    // convention as the framebuffer, so a caller can index a rendered image
    // with it directly. AppKit's own origin is bottom left and in points; both
    // conversions happen once, here, rather than at every call site.
    float mouse_x = 0.0f;
    float mouse_y = 0.0f;
    bool mouse_inside = false;

    // Transitions this frame, so a click fires once. Held state is separate --
    // see IsMouseDown -- because "did they click" and "are they dragging" are
    // different questions and deriving one from the other needs memory.
    bool pressed[int(MouseButton::Count)] = {};
    bool released[int(MouseButton::Count)] = {};
};

class Window {
  public:
    [[nodiscard]] static std::unique_ptr<Window> Create(const char* title,
                                                        int width, int height,
                                                        std::string& error);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    // Puts the RHI's native surface (a CAMetalLayer* here) into the content
    // view. Opaque by design — this layer casts it, no one else does.
    void HostLayer(void* native_layer);

    // Drains the event queue WITHOUT blocking, then reports whether we should
    // keep going. Returns false once the window is closed or Esc is pressed.
    //
    // Non-blocking on purpose: an engine decides when a frame happens. Handing
    // control to the OS event loop is the wrong shape.
    [[nodiscard]] bool PumpEvents();

    // Size of the drawable in PIXELS, not points — already multiplied by the
    // backing scale factor.
    [[nodiscard]] int FramebufferWidth() const;
    [[nodiscard]] int FramebufferHeight() const;
    [[nodiscard]] float BackingScale() const;

    // Returns the accumulated input and CLEARS it. Call exactly once per frame.
    [[nodiscard]] Input TakeInput();

    // Held state, which TakeInput's `keys` cannot express: a toggle fires once
    // per press, but walking forward has to keep happening while the key is
    // down. Both exist because they answer different questions.
    [[nodiscard]] bool IsKeyDown(char) const;
    [[nodiscard]] bool IsMouseDown(MouseButton) const;

    // Hides the cursor and keeps it pinned to the window's centre, so that
    // motion keeps arriving after it would have reached the edge of the screen.
    // What a first-person camera needs and the only way to get unbounded
    // turning; also what stops a click landing on another application.
    void SetCursorLocked(bool);
    [[nodiscard]] bool CursorLocked() const;

  private:
    Window();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// What the display showing `native_layer` can actually do, as a multiple of
// reference white, or 1 when there is no extended range. Lives here rather
// than in the RHI because answering it needs NSScreen, and AppKit belongs to
// the platform layer: this keeps the RHI's link set to Metal + QuartzCore +
// CoreGraphics + Foundation, with no window-server framework. Read it rather
// than assuming -- the same machine reports a different headroom depending on
// the display, the brightness setting and whether it is on battery.
[[nodiscard]] float DisplayHeadroom(const void* native_layer);

}  // namespace eng::platform
