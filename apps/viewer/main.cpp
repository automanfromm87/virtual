// Pure C++20. Interactive PBR material inspector.
//
// Five spheres, one material each, on a ground plane you can orbit around and
// relight. The point of being able to MOVE is that a metallic/roughness surface
// cannot be judged from a still: the highlight is the whole signal, and where
// it sits depends on where you and the light are standing. A fixed camera shows
// you one slice of a five-dimensional material and hides the rest.
//
// Four passes through the render graph: shadow -> scene -> ssao -> composite.
#include "apps/viewer/materials_scene.h"
#include "engine/platform/window.h"
#include "engine/render/rendergraph.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int Fail(const std::string& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.c_str());
    return 1;
}


}  // namespace

int main() {
    const auto kColor = eng::rhi::Format::BGRA8Unorm;
    std::string error;

    auto device = eng::rhi::Device::Create(error);
    if (!device) return Fail(error);
    auto swapchain = device->CreateSwapchain(kColor, error);
    if (!swapchain) return Fail(error);
    auto window = eng::platform::Window::Create("virtual — materials", 1100, 720, error);
    if (!window) return Fail(error);
    window->HostLayer(swapchain->NativeLayer());
    swapchain->Resize(window->FramebufferWidth(), window->FramebufferHeight());

    auto renderer = eng::Renderer::Create(*device, kColor, error);
    if (!renderer) return Fail(error);

    const demo::Assets assets = demo::Build(*renderer, *device, error);
    if (!assets.ok) return Fail(error);

    // --- targets -------------------------------------------------------------
    const eng::rhi::TextureId shadow_map = device->CreateShadowMap(2048);
    if (!Valid(shadow_map)) return Fail("shadow map");
    eng::rhi::TextureId scene_color, scene_depth, ao_target;
    int tex_w = 0, tex_h = 0;

    // --- interactive state ---------------------------------------------------
    eng::OrbitController orbit;
    orbit.target = eng::Vec3{0.0f, 0.6f, 0.0f};
    orbit.distance = 12.0f;
    orbit.yaw = 1.1f;
    orbit.pitch = 0.30f;

    // Sun azimuth starts opposite the camera. With the sun behind the viewer
    // every shadow hides behind the thing casting it, and a working shadow map
    // looks broken.
    float sun_azimuth = orbit.yaw + 3.14159f;
    float sun_elevation = 0.62f;
    bool spin = true, ssao_on = true, shadows_on = true, ortho = false;
    float spin_angle = 0.0f;

    std::printf(
        "drag: orbit    scroll: zoom    [ ]: move the sun    p: pause spin\n"
        "s: shadows     a: ambient occlusion    o: ortho    r: reset    esc: quit\n");
    for (int i = 0; i < demo::kCount; ++i)
        std::printf("  sphere %d  %s\n", i + 1, assets.names[std::size_t(i)].c_str());

    eng::RenderGraph graph;
    auto last = std::chrono::steady_clock::now();

    while (window->PumpEvents()) {
        const auto now = std::chrono::steady_clock::now();
        const float dt = std::chrono::duration<float>(now - last).count();
        last = now;

        const eng::platform::Input in = window->TakeInput();
        orbit.Drag(in.drag_dx, in.drag_dy);
        orbit.Zoom(in.scroll);
        for (char k : in.keys) {
            if (k == 'p') spin = !spin;
            if (k == 'a') ssao_on = !ssao_on;
            if (k == 's') shadows_on = !shadows_on;
            if (k == 'o') ortho = !ortho;
            if (k == 'r') {
                orbit = eng::OrbitController{};
                orbit.target = eng::Vec3{0.0f, 0.6f, 0.0f};
                orbit.distance = 12.0f;
                orbit.yaw = 1.1f;
                orbit.pitch = 0.30f;
                sun_azimuth = orbit.yaw + 3.14159f;
            }
        }
        // Held, not typed: moving the sun is a continuous action, and a key
        // repeat would make it lurch.
        if (window->IsKeyDown('[')) sun_azimuth -= dt * 1.2f;
        if (window->IsKeyDown(']')) sun_azimuth += dt * 1.2f;
        if (spin) spin_angle += dt * 0.35f;

        const int w = window->FramebufferWidth();
        const int h = window->FramebufferHeight();
        if (w < 1 || h < 1) continue;
        if (w != swapchain->Width() || h != swapchain->Height()) swapchain->Resize(w, h);

        if (tex_w != w || tex_h != h) {
            if (Valid(scene_color)) device->DestroyTexture(scene_color);
            if (Valid(scene_depth)) device->DestroyTexture(scene_depth);
            if (Valid(ao_target)) device->DestroyTexture(ao_target);
            scene_color = device->CreateRenderTarget(w, h, kColor);
            scene_depth = device->CreateDepthTarget(w, h, /*sampleable=*/true);
            ao_target = device->CreateRenderTarget(w, h, kColor);
            tex_w = w;
            tex_h = h;
            if (!Valid(scene_color) || !Valid(scene_depth) || !Valid(ao_target))
                return Fail("failed to allocate frame targets");
        }

        const eng::rhi::TextureId drawable = device->AcquireDrawable(*swapchain);
        if (!Valid(drawable)) continue;

        // --- build the frame's scene ------------------------------------------
        eng::Scene scene = demo::MakeScene(assets, spin_angle, sun_azimuth, shadows_on);
        orbit.Apply(scene.camera);
        scene.camera.projection =
            ortho ? eng::Projection::Orthographic : eng::Projection::Perspective;

        // --- passes ------------------------------------------------------------
        graph.Clear();
        {
            eng::RenderGraph::Pass p;
            p.name = "composite";
            p.color = drawable;
            p.reads = ssao_on ? std::vector<eng::rhi::TextureId>{scene_color, ao_target}
                              : std::vector<eng::rhi::TextureId>{scene_color};
            p.execute = [&](eng::rhi::Encoder& e) {
                renderer->DrawComposite(e, scene_color,
                                        ssao_on ? ao_target : eng::rhi::TextureId{});
            };
            graph.AddPass(std::move(p));
        }
        if (ssao_on) {
            eng::RenderGraph::Pass p;
            p.name = "ssao";
            p.color = ao_target;
            p.reads = {scene_depth};
            p.clear_color[0] = p.clear_color[1] = p.clear_color[2] = 1.0f;
            p.execute = [&](eng::rhi::Encoder& e) {
                renderer->DrawSsao(e, scene.camera, w, h, scene_depth);
            };
            graph.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "scene";
            p.color = scene_color;
            p.depth = scene_depth;
            for (int i = 0; i < 4; ++i) p.clear_color[i] = eng::kClearColor[i];
            p.clear_depth = 0.0f;
            p.keep_depth = ssao_on;
            if (shadows_on) p.reads = {shadow_map};
            p.execute = [&](eng::rhi::Encoder& e) {
                renderer->DrawScene(e, scene, w, h,
                                    shadows_on ? shadow_map : eng::rhi::TextureId{});
            };
            graph.AddPass(std::move(p));
        }
        if (shadows_on) {
            eng::RenderGraph::Pass p;
            p.name = "shadow";
            p.depth = shadow_map;
            p.clear_depth = 0.0f;
            p.keep_depth = true;
            p.execute = [&](eng::rhi::Encoder& e) { renderer->DrawShadow(e, scene); };
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
