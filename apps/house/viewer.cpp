// Pure C++20. The interactive floor-plan viewer.
//
// Four passes through the render graph, which is finally doing real work:
//   shadow  -> depth from the sun's point of view
//   scene   -> lit geometry, sampling the shadow map, into an offscreen target
//   ssao    -> occlusion from the scene's depth
//   composite -> scene * ao, plus a vignette, onto the drawable
// The graph derives that order from what each pass reads and writes; the passes
// below are added in a deliberately jumbled order to prove it.
#include "apps/house/scene_build.h"
#include "engine/platform/window.h"
#include "engine/render/rendergraph.h"
#include "engine/render/renderer.h"
#include "engine/rhi/rhi.h"

#include <chrono>
#include <cstdio>
#include <string>

namespace {

int Fail(const std::string& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.c_str());
    return 1;
}

constexpr float kNoCut = 1.0e9f;

}  // namespace

int main() {
    const auto kColor = eng::rhi::Format::BGRA8Unorm;
    std::string error;

    auto device = eng::rhi::Device::Create(error);
    if (!device) return Fail(error);
    auto swapchain = device->CreateSwapchain(kColor, error);
    if (!swapchain) return Fail(error);
    auto window = eng::platform::Window::Create("virtual — floor plan", 1100, 760, error);
    if (!window) return Fail(error);
    window->HostLayer(swapchain->NativeLayer());
    swapchain->Resize(window->FramebufferWidth(), window->FramebufferHeight());

    auto renderer = eng::Renderer::Create(*device, kColor, error);
    if (!renderer) return Fail(error);

    const house::Assets assets = house::Build(*renderer, error);
    if (!assets.ok) return Fail(error);

    const eng::rhi::TextureId shadow_map = device->CreateShadowMap(2048);
    if (!Valid(shadow_map)) return Fail("shadow map");

    eng::rhi::TextureId scene_color, scene_depth, ao_target;
    int tex_w = 0, tex_h = 0;

    eng::OrbitController orbit;
    orbit.target = eng::Vec3{0.0f, 1.3f, 0.0f};
    orbit.distance = 19.0f;

    bool ortho = false;
    // Whole building by default; `c` slices the roof off to look inside.
    bool cut = false;
    bool ssao_on = true;

    std::printf(
        "drag: orbit   scroll: zoom   o: ortho/perspective   c: section cut\n"
        "a: ambient occlusion   r: reset view   esc: quit\n");

    eng::RenderGraph graph;
    while (window->PumpEvents()) {
        const eng::platform::Input in = window->TakeInput();
        orbit.Drag(in.drag_dx, in.drag_dy);
        orbit.Zoom(in.scroll);
        for (char k : in.keys) {
            if (k == 'o') ortho = !ortho;
            if (k == 'c') cut = !cut;
            if (k == 'a') ssao_on = !ssao_on;
            if (k == 'r') {
                orbit = eng::OrbitController{};
                orbit.target = eng::Vec3{0.0f, 1.3f, 0.0f};
                orbit.distance = 19.0f;
            }
        }

        const int w = window->FramebufferWidth();
        const int h = window->FramebufferHeight();
        if (w < 1 || h < 1) continue;
        if (w != swapchain->Width() || h != swapchain->Height()) swapchain->Resize(w, h);

        if (tex_w != w || tex_h != h) {
            if (Valid(scene_color)) device->DestroyTexture(scene_color);
            if (Valid(scene_depth)) device->DestroyTexture(scene_depth);
            if (Valid(ao_target)) device->DestroyTexture(ao_target);
            scene_color = device->CreateRenderTarget(w, h, kColor);
            // Sampleable: SSAO reads it. That costs memoryless storage, which
            // an ordinary depth buffer would keep.
            scene_depth = device->CreateDepthTarget(w, h, /*sampleable=*/true);
            ao_target = device->CreateRenderTarget(w, h, kColor);
            tex_w = w;
            tex_h = h;
            if (!Valid(scene_color) || !Valid(scene_depth) || !Valid(ao_target))
                return Fail("failed to allocate frame targets");
        }

        const eng::rhi::TextureId drawable = device->AcquireDrawable(*swapchain);
        if (!Valid(drawable)) continue;

        eng::Scene scene = house::MakeScene(assets, cut ? 1.35f : kNoCut);
        orbit.Apply(scene.camera);
        scene.camera.projection = ortho ? eng::Projection::Orthographic
                                        : eng::Projection::Perspective;

        graph.Clear();
        // Added composite FIRST and shadow LAST, on purpose.
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
            p.keep_depth = ssao_on;  // SSAO has to be able to read it back
            p.reads = {shadow_map};
            p.execute = [&](eng::rhi::Encoder& e) {
                renderer->DrawScene(e, scene, w, h, shadow_map);
            };
            graph.AddPass(std::move(p));
        }
        {
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
