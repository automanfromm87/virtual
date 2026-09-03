// Pure C++20. The interactive floor-plan viewer.
//
// Four passes through the render graph, which is doing real work:
//   shadow  -> depth from the sun's point of view
//   scene   -> lit geometry, sampling the shadow map, into an offscreen target
//   ssao    -> occlusion from the scene's depth
//   composite -> scene * ao, plus a vignette, onto the drawable
// The graph derives that order from what each pass reads and writes; the passes
// below are added in a deliberately jumbled order to prove it.
#include <cstdio>
#include <string>

#include "apps/house/scene_build.h"
#include "engine/app/app.h"
#include "engine/render/rendergraph.h"

namespace {

int Fail(const std::string& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.c_str());
    return 1;
}

constexpr float kNoCut = 1.0e9f;

}  // namespace

int main() {
    std::string error;
    eng::app::Config config;
    config.title = "virtual — floor plan";
    config.width = 1100;
    config.height = 760;
    auto app = eng::app::App::Create(config, error);
    if (!app) return Fail(error);

    const house::Assets assets = house::Build(app->Draw(), error);
    if (!assets.ok) return Fail(error);

    const eng::rhi::TextureId shadow_map = app->Gpu().CreateShadowMap(2048);
    if (!Valid(shadow_map)) return Fail("shadow map");

    app->Actions().Bind("ortho", 'o');
    app->Actions().Bind("cut", 'c');
    app->Actions().Bind("ao", 'a');
    app->Actions().Bind("reset", 'r');

    eng::OrbitController orbit;
    orbit.target = eng::Vec3{0.0f, 1.3f, 0.0f};
    orbit.distance = 19.0f;

    bool ortho = false;
    bool cut = false;  // whole building by default; `c` slices the roof off
    bool ssao_on = true;

    std::printf(
        "drag: orbit   scroll: zoom   o: ortho/perspective   c: section cut\n"
        "a: ambient occlusion   r: reset view   esc: quit\n");

    eng::RenderGraph graph;
    while (app->Running()) {
        if (!app->BeginFrame()) continue;
        const eng::app::Frame& f = app->Current();

        orbit.Drag(f.drag_dx, f.drag_dy);
        orbit.Zoom(f.scroll);
        if (app->Actions().Pressed("ortho")) ortho = !ortho;
        if (app->Actions().Pressed("cut")) cut = !cut;
        if (app->Actions().Pressed("ao")) ssao_on = !ssao_on;
        if (app->Actions().Pressed("reset")) {
            orbit = eng::OrbitController{};
            orbit.target = eng::Vec3{0.0f, 1.3f, 0.0f};
            orbit.distance = 19.0f;
        }

        // Multisampled, resolved into scene_color. The depth below stays
        // single-sampled and is filled by its own prepass: averaging depths
        // across a silhouette produces a value describing no surface, so a
        // multisample depth buffer cannot be resolved into one SSAO can use.
        const eng::rhi::TextureId ms_color = app->Targets().Msaa("scene");
        const eng::rhi::TextureId ms_depth = app->Targets().MsaaDepth("scene");
        const eng::rhi::TextureId scene_color = app->Targets().Hdr("scene");
        // Sampleable: SSAO reads it back. That costs memoryless storage, which
        // an ordinary depth buffer would keep.
        const eng::rhi::TextureId scene_depth =
            app->Targets().Depth("scene", /*sampleable=*/true);
        const eng::rhi::TextureId ao_target = app->Targets().Color("ao");

        eng::Scene scene = house::MakeScene(assets, cut ? 1.35f : kNoCut);
        orbit.Apply(scene.camera);
        scene.camera.projection = ortho ? eng::Projection::Orthographic
                                        : eng::Projection::Perspective;

        graph.Clear();
        // Added composite FIRST and shadow LAST, on purpose.
        {
            eng::RenderGraph::Pass p;
            p.name = "composite";
            p.color = f.drawable;
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
            p.name = "depth";
            p.depth = scene_depth;
            p.clear_depth = 0.0f;
            p.keep_depth = true;
            p.execute = [&](eng::rhi::Encoder& e) {
                app->Draw().DrawSceneDepth(e, scene, f.width, f.height);
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
                app->Draw().DrawSsao(e, scene.camera, f.width, f.height, scene_depth);
            };
            graph.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "scene";
            p.color = ms_color;
            p.resolve = scene_color;
            p.depth = ms_depth;
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
