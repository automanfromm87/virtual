// Interactive. Balls fall and bounce because physics ran; the blue arms and the
// imported panel move because their PARENT rotated and nothing else.
#include <chrono>
#include <cstdio>
#include <string>

#include "apps/world/world_scene.h"
#include "engine/platform/window.h"
#include "engine/render/rendergraph.h"
#include "engine/render/renderer.h"
#include "engine/rhi/rhi.h"

namespace {

// The TEXTURED fixture: a quad whose material points at a PNG embedded in the
// document. Loading it proves the base64 -> zlib -> PNG -> GPU path end to end.
const char* const kQuadGltf =
#include "engine/asset/testdata_textured_gltf.inc"
    ;

// The default arrangement, baked in so the binary needs nothing on disk to run.
// #embed rather than a runfile: a demo that cannot start without finding a data
// directory is a demo that stops working the moment it is moved.
constexpr char kDefaultScene[] = {
#embed "apps/world/world.scene.json"
    , 0};

int Fail(const std::string& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.c_str());
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    const auto kColor = eng::rhi::Format::BGRA8Unorm;
    std::string error;

    auto device = eng::rhi::Device::Create(error);
    if (!device) return Fail(error);
    auto swapchain = device->CreateSwapchain(kColor, error);
    if (!swapchain) return Fail(error);
    auto window = eng::platform::Window::Create("virtual — ecs + physics + gltf",
                                                1100, 760, error);
    if (!window) return Fail(error);
    window->HostLayer(swapchain->NativeLayer());
    swapchain->Resize(window->FramebufferWidth(), window->FramebufferHeight());

    auto renderer = eng::Renderer::Create(*device, kColor, error);
    if (!renderer) return Fail(error);

    // An argument overrides the built-in scene, so the arrangement can be
    // edited and reloaded without a rebuild — which is the entire reason the
    // scene is a file and not a C++ function.
    std::string scene = kDefaultScene;
    if (argc > 1) {
        std::string text;
        if (std::FILE* f = std::fopen(argv[1], "rb")) {
            char buf[4096];
            std::size_t n;
            while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
            std::fclose(f);
            scene = std::move(text);
            std::printf("scene: %s\n", argv[1]);
        } else {
            return Fail(std::string("cannot open ") + argv[1]);
        }
    }
    demo::World world = demo::Build(*device, *renderer, kQuadGltf, scene, error);
    if (!world.ok) return Fail(error.empty() ? "scene build failed" : error);

    const eng::rhi::TextureId shadow_map = device->CreateShadowMap(2048);
    if (!Valid(shadow_map)) return Fail("shadow map");

    eng::rhi::TextureId scene_color, scene_depth;
    int tex_w = 0, tex_h = 0;

    eng::OrbitController orbit;
    orbit.target = eng::Vec3{0.0f, 1.8f, 0.0f};
    orbit.distance = 17.0f;
    orbit.pitch = 0.35f;

    bool simulate = true;
    auto last = std::chrono::steady_clock::now();

    std::printf(
        "drag: orbit   scroll: zoom   space: pause physics   r: drop again\n"
        "%d bodies, %d entities   esc: quit\n",
        world.physics.Count(), world.ecs.AliveCount());

    eng::RenderGraph graph;
    while (window->PumpEvents()) {
        const eng::platform::Input in = window->TakeInput();
        orbit.Drag(in.drag_dx, in.drag_dy);
        orbit.Zoom(in.scroll);
        for (char k : in.keys) {
            if (k == ' ') simulate = !simulate;
            if (k == 'r') demo::Reset(world);
        }

        const auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;
        if (dt > 0.1f) dt = 0.1f;  // a window drag is not a physics event

        const int w = window->FramebufferWidth();
        const int h = window->FramebufferHeight();
        if (w < 1 || h < 1) continue;
        if (w != swapchain->Width() || h != swapchain->Height()) swapchain->Resize(w, h);

        if (tex_w != w || tex_h != h) {
            if (Valid(scene_color)) device->DestroyTexture(scene_color);
            if (Valid(scene_depth)) device->DestroyTexture(scene_depth);
            scene_color = device->CreateRenderTarget(w, h, kColor);
            scene_depth = device->CreateDepthTarget(w, h);
            tex_w = w;
            tex_h = h;
            if (!Valid(scene_color) || !Valid(scene_depth))
                return Fail("failed to allocate frame targets");
        }

        const eng::rhi::TextureId drawable = device->AcquireDrawable(*swapchain);
        if (!Valid(drawable)) continue;

        demo::Update(world, dt, simulate);
        eng::Scene scene = demo::ToScene(world);
        orbit.Apply(scene.camera);

        graph.Clear();
        {
            eng::RenderGraph::Pass p;
            p.name = "shadow";
            p.depth = shadow_map;
            p.clear_depth = 0.0f;
            p.keep_depth = true;
            p.execute = [&](eng::rhi::Encoder& e) { renderer->DrawShadow(e, scene); };
            graph.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "scene";
            p.color = scene_color;
            p.depth = scene_depth;
            for (int i = 0; i < 4; ++i) p.clear_color[i] = eng::kClearColor[i];
            p.clear_depth = 0.0f;
            p.reads = {shadow_map};
            p.execute = [&](eng::rhi::Encoder& e) {
                renderer->DrawScene(e, scene, w, h, shadow_map);
            };
            graph.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "composite";
            p.color = drawable;
            p.reads = {scene_color};
            p.execute = [&](eng::rhi::Encoder& e) {
                renderer->DrawComposite(e, scene_color);
            };
            graph.AddPass(std::move(p));
        }
        if (!graph.Compile(error)) return Fail(error);

        device->BeginFrame();
        graph.Execute(*device);
        device->Present(*swapchain);
        device->Commit();
    }
    return 0;
}
