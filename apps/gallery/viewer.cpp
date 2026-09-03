// The gallery. Six local lights plus a dim directional key, PBR metals, a
// skinned flag, and shadows.
#include <cstdio>
#include <string>

#include "apps/gallery/gallery_scene.h"
#include "engine/app/app.h"
#include "engine/render/rendergraph.h"

namespace {
int Fail(const std::string& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.c_str());
    return 1;
}
}  // namespace

int main() {
    std::string error;
    eng::app::Config config;
    config.title = "virtual — gallery";
    config.width = 1280;
    config.height = 800;
    auto app = eng::app::App::Create(config, error);
    if (!app) return Fail(error);

    const gallery::Assets assets = gallery::Build(app->Gpu(), app->Draw(), error);
    if (!assets.ok) return Fail(error.empty() ? "scene build failed" : error);

    const eng::rhi::TextureId shadow_map = app->Gpu().CreateShadowMap(2048);
    if (!Valid(shadow_map)) return Fail("shadow map");

    app->Actions().Bind("pause", ' ');
    app->Actions().Bind("ssao", 'a');
    app->Actions().Bind("lights", 'l');
    app->Actions().Bind("bloom", 'b');

    eng::OrbitController orbit;
    orbit.target = eng::Vec3{0.0f, 1.15f, 0.0f};
    orbit.distance = 9.2f;
    orbit.yaw = 1.02f;
    orbit.pitch = 0.20f;

    bool ssao_on = true, locals_on = true, bloom_on = true;
    std::printf(
        "drag: orbit   scroll: zoom   space: pause   a: ambient occlusion\n"
        "l: local lights   b: bloom   esc: quit\n");

    eng::RenderGraph graph;
    while (app->Running()) {
        if (!app->BeginFrame()) continue;
        const eng::app::Frame& f = app->Current();

        orbit.Drag(f.drag_dx, f.drag_dy);
        orbit.Zoom(f.scroll);
        if (app->Actions().Pressed("pause"))
            app->Time().SetPaused(!app->Time().Paused());
        if (app->Actions().Pressed("ssao")) ssao_on = !ssao_on;
        if (app->Actions().Pressed("lights")) locals_on = !locals_on;
        if (app->Actions().Pressed("bloom")) bloom_on = !bloom_on;

        const eng::rhi::TextureId color = app->Targets().Hdr("scene");
        const eng::rhi::TextureId depth =
            app->Targets().Depth("scene", /*sampleable=*/true);
        const eng::rhi::TextureId ao = app->Targets().Color("ao");

        eng::Scene scene = gallery::MakeScene(assets, f.time);
        if (!locals_on) scene.lights.clear();
        orbit.Apply(scene.camera);

        graph.Clear();
        {
            eng::RenderGraph::Pass p;
            p.name = "spot shadows";
            p.depth = app->Draw().ShadowAtlas();
            p.clear_depth = 0.0f;
            p.keep_depth = true;
            p.execute = [&](eng::rhi::Encoder& e) {
                app->Draw().DrawLightShadows(e, scene);
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
        {
            eng::RenderGraph::Pass p;
            p.name = "scene";
            p.color = color;
            p.depth = depth;
            p.clear_color[0] = 0.018f; p.clear_color[1] = 0.020f;
            p.clear_color[2] = 0.028f; p.clear_color[3] = 1.0f;
            p.clear_depth = 0.0f;
            p.keep_depth = ssao_on;
            p.reads = {shadow_map, app->Draw().ShadowAtlas()};
            p.execute = [&](eng::rhi::Encoder& e) {
                app->Draw().DrawScene(e, scene, f.width, f.height, shadow_map);
            };
            graph.AddPass(std::move(p));
        }
        if (ssao_on) {
            eng::RenderGraph::Pass p;
            p.name = "ssao";
            p.color = ao;
            p.reads = {depth};
            p.clear_color[0] = p.clear_color[1] = p.clear_color[2] = 1.0f;
            p.execute = [&](eng::rhi::Encoder& e) {
                app->Draw().DrawSsao(e, scene.camera, f.width, f.height, depth);
            };
            graph.AddPass(std::move(p));
        }
        // --- bloom: bright pass, then a separable blur at two octaves --------
        const eng::rhi::TextureId bA = app->Targets().Hdr("bloomA", 2);
        const eng::rhi::TextureId bB = app->Targets().Hdr("bloomB", 2);
        const eng::rhi::TextureId bC = app->Targets().Hdr("bloomC", 4);
        const eng::rhi::TextureId bD = app->Targets().Hdr("bloomD", 4);
        const eng::rhi::TextureId bOut = app->Targets().Hdr("bloomOut", 2);
        const float hx = 2.0f / float(f.width), hy = 2.0f / float(f.height);
        const float qx = 4.0f / float(f.width), qy = 4.0f / float(f.height);
        if (bloom_on) {
            struct Step {
                const char* name;
                eng::rhi::TextureId src, dst;
                float dx, dy;
            };
            const Step steps[4] = {{"blurAx", bA, bB, hx, 0.0f},
                                   {"blurAy", bB, bC, 0.0f, hy},
                                   {"blurBx", bC, bD, qx, 0.0f},
                                   {"blurBy", bD, bOut, 0.0f, qy}};
            {
                eng::RenderGraph::Pass p;
                p.name = "bright";
                p.color = bA;
                p.reads = {color};
                p.execute = [&](eng::rhi::Encoder& e) {
                    app->Draw().DrawBloomBright(e, color, 3.2f, 0.9f);
                };
                graph.AddPass(std::move(p));
            }
            for (const Step& st : steps) {
                eng::RenderGraph::Pass p;
                p.name = st.name;
                p.color = st.dst;
                p.reads = {st.src};
                p.execute = [&, st](eng::rhi::Encoder& e) {
                    app->Draw().DrawBloomBlur(e, st.src, st.dx, st.dy);
                };
                graph.AddPass(std::move(p));
            }
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "composite";
            p.color = f.drawable;
            p.reads = {color};
            if (ssao_on) p.reads.push_back(ao);
            if (bloom_on) p.reads.push_back(bOut);
            p.execute = [&](eng::rhi::Encoder& e) {
                app->Draw().DrawComposite(e, color,
                                          ssao_on ? ao : eng::rhi::TextureId{},
                                          bloom_on ? bOut : eng::rhi::TextureId{},
                                          0.34f);
            };
            graph.AddPass(std::move(p));
        }
        if (!graph.Compile(error)) return Fail(error);
        graph.Execute(app->Gpu());
        app->EndFrame();
    }
    return 0;
}
