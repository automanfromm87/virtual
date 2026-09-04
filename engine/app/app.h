// A window, a GPU, a renderer and a frame loop, wired together once.
//
// Every windowed demo in this engine used to open with the same fifty lines:
// create a device, create a swapchain, create a window, host the layer, resize,
// create a renderer, then a loop that pumps events, measures dt, checks the
// framebuffer size, resizes the swapchain, reallocates render targets, acquires
// a drawable, presents and commits. Getting any of it wrong produces a black
// window rather than an error.
//
// Still cc_library, not objc_library: everything Apple-specific lives behind
// engine/rhi and engine/platform, and this layer keeps that property.
#pragma once

#include <memory>
#include <string>

#include "engine/app/actions.h"
#include "engine/app/clock.h"
#include "engine/app/targets.h"
#include "engine/render/renderer.h"
#include "engine/rhi/rhi.h"

namespace eng::app {

struct Config {
    std::string title = "virtual";
    int width = 1100;
    int height = 760;
    rhi::Format color = rhi::Format::BGRA8Unorm;
    float max_dt = 0.1f;
    // Multisample count for the scene passes. 4 is the usual choice: it costs
    // almost nothing on a tile-based GPU because the samples never leave tile
    // memory, and it removes the single most obvious "this was drawn by a
    // computer" tell there is.
    int samples = 4;

    // NO WINDOW, NO SWAPCHAIN, and a fixed timestep.
    //
    // Everything in engine/ is already testable without one -- the offscreen
    // tests build an rhi::Device directly -- but anything built on App was not,
    // because App owns the window. So the demos, which are where whole scenes
    // are assembled, could only be measured by opening one and watching. That
    // is slow, it cannot run in CI, and it means the most integrated code in
    // the repo has the least automated coverage.
    //
    // THE FIXED TIMESTEP MATTERS AS MUCH AS THE MISSING WINDOW. A wall-clock dt
    // makes every run different: the exposure meter integrates it, the
    // character integrates it, so two identical captures differ and nothing can
    // be compared byte for byte. Headless runs at `fixed_dt` and is
    // reproducible.
    //
    // Frame::drawable is null in this mode, so a headless caller has to render
    // into a target of its own. It is not a way to run the same code unchanged;
    // it is a way to run the same SCENE.
    bool headless = false;
    float fixed_dt = 1.0f / 60.0f;
};

// What BeginFrame established about this frame.
struct Frame {
    float dt = 0.0f;    // clamped, zero while paused
    float time = 0.0f;  // accumulated dt, so it stops when paused
    std::uint64_t index = 0;
    int width = 0, height = 0;  // pixels, not points
    rhi::TextureId drawable;
    // Mouse motion since the last frame. Keys go through Actions().
    //
    // `drag_*` is motion with the left button held and `mouse_*` is all motion.
    // Both exist because an orbit camera wants the first -- dragging is the
    // gesture -- and a first-person camera wants the second, since a game that
    // makes you hold a button to turn your head is not one.
    float drag_dx = 0.0f, drag_dy = 0.0f, scroll = 0.0f;
    float mouse_dx = 0.0f, mouse_dy = 0.0f;
    // The cursor in PIXELS, origin top left, matching the framebuffer.
    float mouse_x = 0.0f, mouse_y = 0.0f;
    bool mouse_inside = false;
};

class App {
  public:
    [[nodiscard]] static std::unique_ptr<App> Create(const Config&,
                                                     std::string& error);
    ~App();

    App(const App&) = delete;
    App& operator=(const App&) = delete;

    [[nodiscard]] rhi::Device& Gpu();
    [[nodiscard]] Renderer& Draw();
    [[nodiscard]] FrameTargets& Targets();
    [[nodiscard]] ActionMap& Actions();
    [[nodiscard]] Clock& Time();

    // Hides the cursor and pins it to the window's centre, so mouse motion
    // keeps arriving after it would have run off the edge of the screen. What
    // a first-person camera needs; also what stops a click landing in another
    // application.
    void SetCursorLocked(bool);
    [[nodiscard]] bool CursorLocked() const;

    // True until the window closes. Checked BEFORE BeginFrame, so a frame that
    // could not be drawn does not look like a quit.
    [[nodiscard]] bool Running() const;

    // Pumps events, ticks the clock, feeds the action map, resizes the
    // swapchain and the targets, and acquires a drawable.
    //
    // Returns FALSE when there is no drawable this frame — minimised, occluded,
    // or the swapchain is busy. That is not an error and not a quit, so the
    // caller skips its rendering and loops. Reporting it as a quit would close
    // the app when someone minimises it; hiding it by blocking would freeze
    // the event loop.
    [[nodiscard]] bool BeginFrame();
    [[nodiscard]] const Frame& Current() const;

    // Presents and commits. Only call it after BeginFrame returned true.
    void EndFrame();

  private:
    App();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eng::app
