// Interactive. A skinned tentacle: the mesh is one rigid buffer, and every bend
// in it comes from eight joint matrices the vertex shader blends per vertex.
#include <cstdio>
#include <string>

#include "apps/skinned/skinned_scene.h"
#include "engine/app/app.h"
#include "engine/render/rendergraph.h"

namespace {

int Fail(const std::string& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.c_str());
    return 1;
}

}  // namespace

int main() {
    std::string error;
    eng::app::Config config;
    config.title = "virtual — a flag, skinned";
    auto app = eng::app::App::Create(config, error);
    if (!app) return Fail(error);

    const demo::Assets assets = demo::Build(app->Draw(), error);
    if (!assets.ok) return Fail(error.empty() ? "scene build failed" : error);

    const eng::rhi::TextureId shadow_map = app->Gpu().CreateShadowMap(2048);
    if (!Valid(shadow_map)) return Fail("shadow map");

    app->Actions().Bind("pause", ' ');
    app->Actions().Bind("reset", 'r');
    // Held, not tapped: scrubbing is continuous, and an edge would step it.
    app->Actions().BindAxis("scrub", ',', '.');

    eng::OrbitController orbit;
    orbit.target = eng::Vec3{1.4f, 2.9f, 0.0f};
    orbit.distance = 8.5f;
    orbit.yaw = 1.32f;
    orbit.pitch = 0.20f;

    float clip_time = 0.0f;
    std::printf(
        "drag: orbit   scroll: zoom   space: pause   , .: scrub   r: reset\n"
        "%d bones, %zu vertices, one draw call   esc: quit\n",
        demo::kJoints, assets.flag.mesh.vertices.size());

    eng::RenderGraph graph;
    while (app->Running()) {
        if (!app->BeginFrame()) continue;
        const eng::app::Frame& f = app->Current();

        orbit.Drag(f.drag_dx, f.drag_dy);
        orbit.Zoom(f.scroll);
        if (app->Actions().Pressed("pause"))
            app->Time().SetPaused(!app->Time().Paused());
        if (app->Actions().Pressed("reset")) clip_time = 0.0f;

        // dt is zero while paused, so scrubbing is the only thing that moves
        // the clip then — which is what makes pause-and-step work.
        clip_time += f.dt;
        // RawDt, not dt: scrubbing has to keep working while the clock is
        // paused, and dt is zero then by design.
        clip_time += app->Actions().Axis("scrub") * app->Time().RawDt() * 1.5f;

        const eng::rhi::TextureId color = app->Targets().Color("scene");
        const eng::rhi::TextureId depth = app->Targets().Depth("scene");

        eng::Scene scene = demo::MakeScene(assets, clip_time);
        orbit.Apply(scene.camera);

        graph.Clear();
        {
            eng::RenderGraph::Pass p;
            p.name = "shadow";
            p.depth = shadow_map;
            p.clear_depth = 0.0f;
            p.keep_depth = true;
            p.execute = [&](eng::rhi::Encoder& e) { app->Draw().DrawShadow(e, scene); };
            graph.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "scene";
            p.color = color;
            p.depth = depth;
            for (int i = 0; i < 4; ++i) p.clear_color[i] = eng::kClearColor[i];
            p.clear_depth = 0.0f;
            p.reads = {shadow_map};
            p.execute = [&](eng::rhi::Encoder& e) {
                app->Draw().DrawScene(e, scene, f.width, f.height, shadow_map);
            };
            graph.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "composite";
            p.color = f.drawable;
            p.reads = {color};
            p.execute = [&](eng::rhi::Encoder& e) { app->Draw().DrawComposite(e, color); };
            graph.AddPass(std::move(p));
        }
        if (!graph.Compile(error)) return Fail(error);
        graph.Execute(app->Gpu());
        app->EndFrame();
    }
    return 0;
}
