#include "engine/app/app.h"

#include <chrono>

#include "engine/platform/window.h"
#include "engine/render/renderer.h"

namespace eng::app {
namespace {

// A monotonic clock in seconds. steady_clock rather than system_clock: the wall
// clock can jump backwards when the machine syncs time, and a negative dt runs
// the simulation in reverse.
double Now() {
    using Clock = std::chrono::steady_clock;
    static const Clock::time_point origin = Clock::now();
    return std::chrono::duration<double>(Clock::now() - origin).count();
}

}  // namespace

struct App::Impl {
    std::unique_ptr<rhi::Device> device;
    std::unique_ptr<rhi::Swapchain> swapchain;
    std::unique_ptr<platform::Window> window;
    std::unique_ptr<Renderer> renderer;
    std::unique_ptr<FrameTargets> targets;
    ActionMap actions;
    Clock clock;
    Frame frame;
    bool running = true;
    bool headless = false;
    float fixed_dt = 1.0f / 60.0f;
    // A synthetic monotonic clock, so the fixed step goes through the SAME
    // Clock::Tick the windowed path uses -- clamping, pause and all -- rather
    // than through a second code path that would drift away from it.
    double fake_now = 0.0;
    int width = 0, height = 0;
};

App::App() : impl_(std::make_unique<Impl>()) {}
App::~App() = default;

std::unique_ptr<App> App::Create(const Config& config, std::string& error) {
    std::unique_ptr<App> app(new App());
    Impl& s = *app->impl_;

    s.device = rhi::Device::Create(error);
    if (!s.device) {
        if (error.empty()) error = "App: GPU device creation failed";
        return nullptr;
    }
    s.headless = config.headless;
    s.fixed_dt = config.fixed_dt;
    s.width = config.width;
    s.height = config.height;
    if (!s.headless) {
        s.swapchain = s.device->CreateSwapchain(config.color, error);
        if (!s.swapchain) {
            if (error.empty()) error = "App: swapchain creation failed";
            return nullptr;
        }
        s.window = platform::Window::Create(config.title.c_str(), config.width,
                                            config.height, error);
        if (!s.window) {
            if (error.empty()) error = "App: window creation failed";
            return nullptr;
        }
        s.window->HostLayer(s.swapchain->NativeLayer());
        s.swapchain->Resize(s.window->FramebufferWidth(),
                            s.window->FramebufferHeight());
        s.width = s.window->FramebufferWidth();
        s.height = s.window->FramebufferHeight();
    }

    s.renderer = Renderer::Create(*s.device, config.color, error, config.samples);
    if (!s.renderer) {
        if (error.empty()) error = "App: renderer creation failed";
        return nullptr;
    }
    s.targets = std::make_unique<FrameTargets>(*s.device, config.color,
                                               config.samples);
    s.clock.max_dt = config.max_dt;
    return app;
}

rhi::Device& App::Gpu() { return *impl_->device; }
Renderer& App::Draw() { return *impl_->renderer; }
FrameTargets& App::Targets() { return *impl_->targets; }
ActionMap& App::Actions() { return impl_->actions; }
Clock& App::Time() { return impl_->clock; }
void App::SetCursorLocked(bool locked) {
    if (impl_->window) impl_->window->SetCursorLocked(locked);
}
bool App::CursorLocked() const {
    return impl_->window && impl_->window->CursorLocked();
}

bool App::Running() const { return impl_->running; }
const Frame& App::Current() const { return impl_->frame; }

bool App::BeginFrame() {
    Impl& s = *impl_;
    if (s.headless) {
        // No events and no input: every action reads as released, which is what
        // a caller driving the scene itself wants. It sets the camera and the
        // state directly rather than pretending to press keys.
        s.actions.Update(Keys{}, 0u);
        s.fake_now += double(s.fixed_dt);
        s.clock.Tick(s.fake_now);
        s.frame.dt = s.clock.Dt();
        s.frame.time = s.clock.Total();
        s.frame.index = s.clock.Frame();
        s.frame.width = s.width;
        s.frame.height = s.height;
        s.frame.drawable = {};
        s.targets->Resize(s.width, s.height);
        s.device->BeginFrame();
        return true;
    }
    if (!s.window->PumpEvents()) {
        s.running = false;
        return false;
    }

    const platform::Input in = s.window->TakeInput();
    s.frame.drag_dx = in.drag_dx;
    s.frame.drag_dy = in.drag_dy;
    s.frame.mouse_dx = in.mouse_dx;
    s.frame.mouse_dy = in.mouse_dy;
    s.frame.mouse_x = in.mouse_x;
    s.frame.mouse_y = in.mouse_y;
    s.frame.mouse_inside = in.mouse_inside;
    s.frame.scroll = in.scroll;

    // Held state, not the typed characters: key repeat turns a held key into a
    // stream of characters, and an edge derived from that fires over and over.
    Keys held;
    for (int c = 32; c < 127; ++c) held.Set(char(c), s.window->IsKeyDown(char(c)));
    // The two enums are separate so that :loop stays free of AppKit; this is
    // what stops them drifting apart.
    static_assert(int(MouseButton::Left) == int(platform::MouseButton::Left));
    static_assert(int(MouseButton::Right) == int(platform::MouseButton::Right));
    static_assert(int(MouseButton::Middle) == int(platform::MouseButton::Middle));
    static_assert(int(MouseButton::Count) == int(platform::MouseButton::Count));
    std::uint32_t mouse = 0;
    for (int b = 0; b < int(platform::MouseButton::Count); ++b)
        if (s.window->IsMouseDown(platform::MouseButton(b))) mouse |= 1u << b;
    s.actions.Update(held, mouse);

    s.clock.Tick(Now());
    s.frame.dt = s.clock.Dt();
    s.frame.time = s.clock.Total();
    s.frame.index = s.clock.Frame();

    const int w = s.window->FramebufferWidth();
    const int h = s.window->FramebufferHeight();
    s.frame.width = w;
    s.frame.height = h;
    if (w < 1 || h < 1) {
        s.frame.drawable = {};
        return false;  // minimised: still ticking, nothing to draw into
    }
    if (w != s.swapchain->Width() || h != s.swapchain->Height())
        s.swapchain->Resize(w, h);
    s.targets->Resize(w, h);

    s.frame.drawable = s.device->AcquireDrawable(*s.swapchain);
    if (!Valid(s.frame.drawable)) return false;

    s.device->BeginFrame();
    return true;
}

void App::EndFrame() {
    Impl& s = *impl_;
    if (!s.headless) s.device->Present(*s.swapchain);
    s.device->Commit();
}

}  // namespace eng::app
