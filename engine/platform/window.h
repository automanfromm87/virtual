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
struct Input {
    float drag_dx = 0.0f;  // mouse motion with the left button held, in points
    float drag_dy = 0.0f;
    float scroll = 0.0f;
    std::string keys;  // characters typed this frame, lowercased — for toggles
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

  private:
    Window();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eng::platform
