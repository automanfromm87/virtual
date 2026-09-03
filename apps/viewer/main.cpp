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
#include "engine/app/app.h"
#include "engine/render/rendergraph.h"

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
    std::string error;
    eng::app::Config config;
    config.title = "virtual — materials";
    config.width = 1100;
    config.height = 720;
    auto app = eng::app::App::Create(config, error);
    if (!app) return Fail(error);

    const demo::Assets assets = demo::Build(app->Draw(), app->Gpu(), error);
    if (!assets.ok) return Fail(error);

    const eng::rhi::TextureId shadow_map = app->Gpu().CreateShadowMap(2048);
    if (!Valid(shadow_map)) return Fail("shadow map");

    // Toggles are edges; the sun is an AXIS, because holding a key to sweep it
    // is a continuous action and key repeat would make it lurch.
    app->Actions().Bind("spin", 'p');
    app->Actions().Bind("ao", 'a');
    app->Actions().Bind("shadows", 's');
    app->Actions().Bind("ortho", 'o');
    app->Actions().Bind("reset", 'r');
    app->Actions().BindAxis("sun", '[', ']');

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
    while (app->Running()) {
        if (!app->BeginFrame()) continue;
        const eng::app::Frame& f = app->Current();
        const float dt = f.dt;

        orbit.Drag(f.drag_dx, f.drag_dy);
        orbit.Zoom(f.scroll);
        if (app->Actions().Pressed("spin")) spin = !spin;
        if (app->Actions().Pressed("ao")) ssao_on = !ssao_on;
        if (app->Actions().Pressed("shadows")) shadows_on = !shadows_on;
        if (app->Actions().Pressed("ortho")) ortho = !ortho;
        if (app->Actions().Pressed("reset")) {
            orbit = eng::OrbitController{};
            orbit.target = eng::Vec3{0.0f, 0.6f, 0.0f};
            orbit.distance = 12.0f;
            orbit.yaw = 1.1f;
            orbit.pitch = 0.30f;
            sun_azimuth = orbit.yaw + 3.14159f;
        }
        sun_azimuth += app->Actions().Axis("sun") * dt * 1.2f;
        if (spin) spin_angle += dt * 0.35f;

        const int w = f.width, h = f.height;
        const eng::rhi::TextureId scene_color = app->Targets().Color("scene");
        const eng::rhi::TextureId scene_depth =
            app->Targets().Depth("scene", /*sampleable=*/true);
        const eng::rhi::TextureId ao_target = app->Targets().Color("ao");
        const eng::rhi::TextureId drawable = f.drawable;

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
                app->Draw().DrawComposite(e, scene_color,
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
                app->Draw().DrawSsao(e, scene.camera, w, h, scene_depth);
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
                app->Draw().DrawScene(e, scene, w, h,
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
            p.execute = [&](eng::rhi::Encoder& e) { app->Draw().DrawShadow(e, scene); };
            graph.AddPass(std::move(p));
        }
        if (!graph.Compile(error)) return Fail(error);

        graph.Execute(app->Gpu());
        app->EndFrame();
    }
    return 0;
}
