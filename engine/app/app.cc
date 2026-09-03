#include "engine/app/app.h"

#include <chrono>

#include "engine/platform/window.h"

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
};

App::App() : impl_(std::make_unique<Impl>()) {}
App::~App() = default;

std::unique_ptr<App> App::Create(const Config& config, std::string& error) {
    std::unique_ptr<App> app(new App());
    Impl& s = *app->impl_;

    s.device = rhi::Device::Create(error);
    if (!s.device) return nullptr;
    s.swapchain = s.device->CreateSwapchain(config.color, error);
    if (!s.swapchain) return nullptr;
    s.window = platform::Window::Create(config.title.c_str(), config.width,
                                        config.height, error);
    if (!s.window) return nullptr;
    s.window->HostLayer(s.swapchain->NativeLayer());
    s.swapchain->Resize(s.window->FramebufferWidth(), s.window->FramebufferHeight());

    s.renderer = Renderer::Create(*s.device, config.color, error, config.samples);
    if (!s.renderer) return nullptr;
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
bool App::Running() const { return impl_->running; }
const Frame& App::Current() const { return impl_->frame; }

bool App::BeginFrame() {
    Impl& s = *impl_;
    if (!s.window->PumpEvents()) {
        s.running = false;
        return false;
    }

    const platform::Input in = s.window->TakeInput();
    s.frame.drag_dx = in.drag_dx;
    s.frame.drag_dy = in.drag_dy;
    s.frame.scroll = in.scroll;

    // Held state, not the typed characters: key repeat turns a held key into a
    // stream of characters, and an edge derived from that fires over and over.
    Keys held;
    for (int c = 32; c < 127; ++c) held.Set(char(c), s.window->IsKeyDown(char(c)));
    s.actions.Update(held);

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
    s.device->Present(*s.swapchain);
    s.device->Commit();
}

}  // namespace eng::app
