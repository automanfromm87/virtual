// Interactive. An SPH fluid: five compute dispatches per substep, and the whole
// simulation lives in one buffer the CPU only reads when asked to.
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/app/app.h"
#include "engine/geometry/mesh.h"
#include "engine/render/fluid.h"
#include "engine/render/rendergraph.h"

namespace {

int Fail(const std::string& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.c_str());
    return 1;
}

std::vector<eng::Vec3> Block(eng::Vec3 lo, int nx, int ny, int nz, float s) {
    std::vector<eng::Vec3> o;
    o.reserve(std::size_t(nx) * ny * nz);
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                o.push_back(lo + eng::Vec3{float(x) * s, float(y) * s, float(z) * s});
    return o;
}

eng::FluidConfig MakeConfig() {
    eng::FluidConfig c;
    c.bounds_min = eng::Vec3{-0.45f, 0.0f, -0.30f};
    c.bounds_max = eng::Vec3{0.45f, 0.80f, 0.30f};
    c.smoothing_radius = 0.055f;
    c.stiffness = 200.0f;
    c.viscosity = 0.55f;
    c.artificial_viscosity = 0.12f;
    return c;
}

}  // namespace

int main() {
    std::string error;
    eng::app::Config config;
    config.title = "virtual — sph fluid";
    config.samples = 1;
    auto app = eng::app::App::Create(config, error);
    if (!app) return Fail(error);

    const eng::FluidConfig cfg = MakeConfig();
    const float spacing = cfg.smoothing_radius * 0.48f;
    // A tall column against one wall: a dam break, which is the case that shows
    // whether it is a fluid or a pile of sand.
    const auto initial = [&] {
        return Block(eng::Vec3{-0.42f, 0.02f, -0.27f}, 11, 30, 21, spacing);
    };

    auto fluid = eng::FluidSim::Create(app->Gpu(), cfg, initial(),
                                       eng::Renderer::kSceneFormat, error, 1);
    if (!fluid) return Fail(error);

    const eng::MeshHandle floor = app->Draw().UploadMesh(
        eng::MakeBox(eng::Vec3{1.4f, 0.02f, 1.0f}, eng::Vec4{1, 1, 1, 1}));
    eng::MaterialDesc md;
    md.shading = eng::Shading::Lit;
    md.base_color = eng::Vec4{0.17f, 0.18f, 0.21f, 1.0f};
    md.roughness = 0.85f;
    const eng::MaterialHandle floor_mat = app->Draw().CreateMaterial(md, error);
    if (!Valid(floor) || !Valid(floor_mat)) return Fail("scene build failed");

    app->Actions().Bind("pause", ' ');
    app->Actions().Bind("reset", 'r');
    app->Actions().Bind("stats", 'i');

    eng::OrbitController orbit;
    orbit.target = eng::Vec3{0.0f, 0.18f, 0.0f};
    orbit.distance = 1.5f;
    orbit.yaw = 0.15f;
    orbit.pitch = 0.22f;

    std::printf("drag: orbit   scroll: zoom   space: pause   r: reset   "
                "i: density and energy\n%d particles, five dispatches a "
                "substep   esc: quit\n", fluid->Count());

    eng::Scene scene;
    scene.lightDir = eng::Vec4{-0.3f, 0.9f, 0.3f, 0.0f};
    scene.lightColor = eng::Vec4{0.9f, 0.92f, 1.0f, 1.0f};
    scene.ambientSky = eng::Vec3{0.13f, 0.15f, 0.20f};
    scene.ambientGround = eng::Vec3{0.04f, 0.04f, 0.05f};
    {
        eng::Instance f;
        f.mesh = floor;
        f.material = floor_mat;
        f.model = eng::Mat4::Translation(eng::Vec3{0.0f, -0.02f, 0.0f});
        scene.instances.push_back(f);
    }

    eng::RenderGraph graph;
    int substeps = 0;
    std::uint64_t last_report = 0;
    while (app->Running()) {
        if (!app->BeginFrame()) continue;
        const eng::app::Frame& f = app->Current();

        orbit.Drag(f.drag_dx, f.drag_dy);
        orbit.Zoom(f.scroll);
        if (app->Actions().Pressed("pause"))
            app->Time().SetPaused(!app->Time().Paused());
        if (app->Actions().Pressed("reset")) {
            std::string e2;
            auto fresh = eng::FluidSim::Create(app->Gpu(), cfg, initial(),
                                               eng::Renderer::kSceneFormat, e2, 1);
            if (fresh) fluid = std::move(fresh);
        }
        if (app->Actions().Pressed("stats")) {
            // A READBACK, on demand only. Asking every frame would give up the
            // property the whole design rests on -- that the CPU does not touch
            // the fluid -- and would stall the pipeline to do it.
            float mean = 0, lo = 0, hi = 0;
            fluid->ReadDensity(&mean, &lo, &hi);
            std::printf("\n  density %.1f (%.1f..%.1f), kinetic energy %.4f J, "
                        "%d outside the box\n", mean, lo, hi,
                        fluid->KineticEnergy(), fluid->OutsideBounds());
        }
        orbit.Apply(scene.camera);

        {
            eng::rhi::ComputeEncoder ce = app->Gpu().BeginCompute();
            substeps = fluid->Step(ce, f.dt);
            app->Gpu().EndCompute();
        }

        const eng::rhi::TextureId color = app->Targets().Hdr("scene");
        const eng::rhi::TextureId depth = app->Targets().Depth("scene");

        graph.Clear();
        {
            eng::RenderGraph::Pass p;
            p.name = "scene";
            p.color = color;
            p.depth = depth;
            p.clear_color[0] = 0.028f; p.clear_color[1] = 0.032f;
            p.clear_color[2] = 0.046f; p.clear_color[3] = 1.0f;
            p.clear_depth = 0.0f;
            p.execute = [&](eng::rhi::Encoder& e) {
                app->Draw().DrawScene(e, scene, f.width, f.height);
                fluid->Draw(e, scene.camera, f.width, f.height);
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
            std::printf("\r%.0f fps   %d substeps/frame      ",
                        app->Time().Fps(), substeps);
            std::fflush(stdout);
        }
    }
    std::printf("\n");
    return 0;
}
