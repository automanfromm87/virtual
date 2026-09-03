// Interactive. A GPU particle fountain: one compute dispatch a frame simulates
// the whole pool, one instanced draw puts it on screen, and the CPU never
// touches a particle.
#include <cstdio>
#include <string>

#include "engine/app/app.h"
#include "engine/geometry/mesh.h"
#include "engine/render/particles.h"
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
    config.title = "virtual — gpu particles";
    // ONE sample. The particle pipeline is compiled against a sample count, and
    // a fountain of soft additive sprites has no hard edges for multisampling
    // to find anyway.
    config.samples = 1;
    auto app = eng::app::App::Create(config, error);
    if (!app) return Fail(error);

    constexpr int kCapacity = 60000;
    auto ps = eng::ParticleSystem::Create(app->Gpu(), kCapacity,
                                          eng::Renderer::kSceneFormat, error, 1);
    if (!ps) return Fail(error);

    const eng::MeshHandle floor = app->Draw().UploadMesh(
        eng::MakeBox(eng::Vec3{9.0f, 0.2f, 9.0f}, eng::Vec4{1, 1, 1, 1}));
    eng::MaterialDesc md;
    md.shading = eng::Shading::Lit;
    md.base_color = eng::Vec4{0.15f, 0.16f, 0.19f, 1.0f};
    md.roughness = 0.8f;
    const eng::MaterialHandle floor_mat = app->Draw().CreateMaterial(md, error);
    if (!Valid(floor) || !Valid(floor_mat)) return Fail("scene build failed");

    app->Actions().Bind("pause", ' ');
    app->Actions().BindAxis("rate", '-', '=');
    app->Actions().BindAxis("gravity", '[', ']');

    eng::OrbitController orbit;
    orbit.target = eng::Vec3{0.0f, 1.5f, 0.0f};
    orbit.distance = 8.0f;
    orbit.yaw = 0.5f;
    orbit.pitch = 0.18f;

    eng::ParticleEmitter emitter;
    emitter.position = eng::Vec3{0.0f, 0.15f, 0.0f};
    emitter.direction = eng::Vec3{0.0f, 1.0f, 0.0f};
    emitter.spread = 0.30f;
    emitter.speed = 6.4f;
    emitter.speed_variance = 1.5f;
    emitter.lifetime = 2.2f;
    emitter.lifetime_variance = 0.7f;
    emitter.size = 0.05f;
    emitter.size_variance = 0.02f;
    emitter.rate = 14000.0f;
    emitter.color = eng::Vec4{2.8f, 1.2f, 0.35f, 1.0f};
    emitter.gravity = eng::Vec3{0.0f, -9.81f, 0.0f};
    emitter.drag = 0.25f;

    std::printf("drag: orbit   scroll: zoom   space: pause   - =: rate   "
                "[ ]: gravity\n%d particle pool, one dispatch and one draw "
                "per frame   esc: quit\n", kCapacity);

    eng::Scene scene;
    scene.lightDir = eng::Vec4{-0.35f, 0.86f, 0.37f, 0.0f};
    scene.lightColor = eng::Vec4{0.5f, 0.52f, 0.6f, 1.0f};
    scene.ambientSky = eng::Vec3{0.05f, 0.06f, 0.09f};
    scene.ambientGround = eng::Vec3{0.02f, 0.02f, 0.02f};
    {
        eng::Instance f;
        f.mesh = floor;
        f.material = floor_mat;
        f.model = eng::Mat4::Translation(eng::Vec3{0.0f, -0.2f, 0.0f});
        scene.instances.push_back(f);
    }

    eng::RenderGraph graph;
    std::uint64_t last_report = 0;
    while (app->Running()) {
        if (!app->BeginFrame()) continue;
        const eng::app::Frame& f = app->Current();

        orbit.Drag(f.drag_dx, f.drag_dy);
        orbit.Zoom(f.scroll);
        if (app->Actions().Pressed("pause"))
            app->Time().SetPaused(!app->Time().Paused());
        emitter.rate = std::clamp(
            emitter.rate * (1.0f + app->Actions().Axis("rate") * 1.5f * f.dt),
            200.0f, 60000.0f);
        emitter.gravity.y = std::clamp(
            emitter.gravity.y + app->Actions().Axis("gravity") * 12.0f * f.dt,
            -25.0f, 4.0f);
        orbit.Apply(scene.camera);

        // THE COMPUTE PASS, before any render pass opens. The two cannot
        // overlap: what a render pass writes lives in tile memory until it ends.
        {
            eng::rhi::ComputeEncoder ce = app->Gpu().BeginCompute();
            ps->Step(ce, emitter, f.dt);
            app->Gpu().EndCompute();
        }

        // A SAMPLEABLE depth target, so the particles can fade where they meet
        // the floor instead of cutting a hard line across themselves.
        const eng::rhi::TextureId color = app->Targets().Hdr("scene");
        const eng::rhi::TextureId depth = app->Targets().Depth("scene", true);

        graph.Clear();
        {
            eng::RenderGraph::Pass p;
            p.name = "scene";
            p.color = color;
            p.depth = depth;
            p.clear_color[0] = 0.014f; p.clear_color[1] = 0.016f;
            p.clear_color[2] = 0.023f; p.clear_color[3] = 1.0f;
            p.clear_depth = 0.0f;
            p.keep_depth = true;
            p.execute = [&](eng::rhi::Encoder& e) {
                app->Draw().DrawScene(e, scene, f.width, f.height);
                // AFTER the opaque geometry and in the SAME pass: the particles
                // test against the depth it just wrote.
                ps->Draw(e, scene.camera, f.width, f.height, depth);
            };
            graph.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "composite";
            p.color = f.drawable;
            p.reads = {color};
            p.execute = [&](eng::rhi::Encoder& e) {
                app->Draw().DrawComposite(e, color);
            };
            graph.AddPass(std::move(p));
        }
        if (!graph.Compile(error)) return Fail(error);
        graph.Execute(app->Gpu());
        app->EndFrame();

        if (f.index - last_report > 120) {
            last_report = f.index;
            std::printf("\r%.0f fps   %.0f particles/s   gravity %.1f      ",
                        app->Time().Fps(), emitter.rate, emitter.gravity.y);
            std::fflush(stdout);
        }
    }
    std::printf("\n");
    return 0;
}
