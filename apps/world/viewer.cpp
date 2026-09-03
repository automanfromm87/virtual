// Interactive. Balls roll down a ramp because friction acts at the contact
// point; the blue arms and the imported panel move because their PARENT
// rotated and nothing else.
//
// Windows, swapchains, timing and render targets all live in engine/app now.
// What is left here is this demo's own policy and nothing else.
#include <cstdio>
#include <string>

#include "apps/world/world_scene.h"
#include "engine/app/app.h"
#include "engine/render/rendergraph.h"

namespace {

// The TEXTURED fixture: a quad whose material points at a PNG embedded in the
// document. Loading it proves the base64 -> zlib -> PNG -> GPU path end to end.
const char* const kQuadGltf =
#include "engine/asset/testdata_textured_gltf.inc"
    ;

constexpr char kDefaultScene[] = {
#embed "apps/world/world.scene.json"
    , 0};

int Fail(const std::string& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.c_str());
    return 1;
}

std::string ReadFile(const char* path, bool* ok) {
    std::string text;
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { *ok = false; return text; }
    char buf[4096];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
    std::fclose(f);
    *ok = true;
    return text;
}

}  // namespace

int main(int argc, char** argv) {
    std::string error;
    eng::app::Config config;
    config.title = "virtual — ecs + physics + gltf";
    auto app = eng::app::App::Create(config, error);
    if (!app) return Fail(error);

    // An argument overrides the built-in scene, so the arrangement can be
    // edited and rerun without a rebuild — the entire reason it is a file.
    std::string scene = kDefaultScene;
    if (argc > 1) {
        bool ok = false;
        scene = ReadFile(argv[1], &ok);
        if (!ok) return Fail(std::string("cannot open ") + argv[1]);
        std::printf("scene: %s\n", argv[1]);
    }

    demo::World world = demo::Build(app->Gpu(), app->Draw(), kQuadGltf, scene, error);
    if (!world.ok) return Fail(error.empty() ? "scene build failed" : error);

    const eng::rhi::TextureId shadow_map = app->Gpu().CreateShadowMap(2048);
    if (!Valid(shadow_map)) return Fail("shadow map");

    // Named, so the bindings sit in one place instead of scattered through the
    // event loop as character comparisons.
    app->Actions().Bind("pause", ' ');
    app->Actions().Bind("reset", 'r');

    eng::OrbitController orbit;
    orbit.target = eng::Vec3{0.0f, 1.8f, 0.0f};
    orbit.distance = 17.0f;
    orbit.pitch = 0.35f;

    std::printf(
        "drag: orbit   scroll: zoom   space: pause physics   r: drop again\n"
        "%d bodies, %d entities   esc: quit\n",
        world.physics.Count(), world.ecs.AliveCount());

    eng::RenderGraph graph;
    while (app->Running()) {
        if (!app->BeginFrame()) continue;
        const eng::app::Frame& f = app->Current();

        orbit.Drag(f.drag_dx, f.drag_dy);
        orbit.Zoom(f.scroll);
        if (app->Actions().Pressed("pause"))
            app->Time().SetPaused(!app->Time().Paused());
        if (app->Actions().Pressed("reset")) demo::Reset(world);

        const eng::rhi::TextureId color = app->Targets().Color("scene");
        const eng::rhi::TextureId depth = app->Targets().Depth("scene");

        // dt is zero while paused, so the simulation stops without this demo
        // carrying a flag of its own.
        demo::Update(world, f.dt);
        eng::Scene desc = demo::ToScene(world);
        orbit.Apply(desc.camera);

        graph.Clear();
        {
            eng::RenderGraph::Pass p;
            p.name = "shadow";
            p.depth = shadow_map;
            p.clear_depth = 0.0f;
            p.keep_depth = true;
            p.execute = [&](eng::rhi::Encoder& e) { app->Draw().DrawShadow(e, desc); };
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
                app->Draw().DrawScene(e, desc, f.width, f.height, shadow_map);
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
