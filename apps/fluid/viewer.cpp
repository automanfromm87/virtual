// A tank of water: a dam break with a CONTINUOUS surface.
//
// The fluid test renders debug billboards; this is the look -- screen-space
// surface reconstruction (sphere depth, bilateral smooth, fresnel sky
// reflection, background refraction, sun glint) over a water column that
// collapses when the run starts. Left-drag orbits, wheel zooms, R refills
// the tank, esc quits.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <cstdlib>
#include <cstring>

#include "engine/app/app.h"
#include "engine/asset/png.h"
#include "engine/geometry/mesh.h"
#include "engine/render/fluid.h"
#include "engine/render/renderer.h"
#include "engine/render/rendergraph.h"

namespace {

int Fail(const std::string& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.c_str());
    return 1;
}

// The tank interior, in metres. The sim bounds match it exactly: a particle
// outside the neighbour grid hashes nowhere, so the walls the eye sees and
// the walls the solver feels have to be the same box.
constexpr float kX0 = -0.8f, kX1 = 0.8f;
constexpr float kY1 = 0.95f;
constexpr float kZ0 = -0.4f, kZ1 = 0.4f;

// The dam: water fills the left of this x, air the right. No gate mesh --
// the column starts at rest and collapses on the first step, which is what
// a lifted gate leaves behind anyway.
constexpr float kDamX = 0.05f;
// Deep: the settled pool sits near the rim, so the surface is visible at
// every orbit angle instead of only from directly above. Shallow water in a
// tall tank reads as empty from anywhere but the top.
constexpr float kFillY = 0.68f;
constexpr float kSpacing = 0.034f;

std::vector<eng::Vec3> DamColumn() {
    std::vector<eng::Vec3> out;
    for (float z = kZ0 + 0.04f; z < kZ1 - 0.02f; z += kSpacing)
        for (float y = 0.02f; y < kFillY; y += kSpacing)
            for (float x = kX0 + 0.05f; x < kDamX; x += kSpacing)
                out.push_back(eng::Vec3{x, y, z});
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string shot_path;
    int shot_frames = 90;
    int shot_w = 1100, shot_h = 760;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shot") == 0 && i + 1 < argc)
            shot_path = argv[++i];
        else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
            shot_frames = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--size") == 0 && i + 2 < argc) {
            shot_w = std::atoi(argv[++i]);
            shot_h = std::atoi(argv[++i]);
        }
    }

    eng::app::Config config;
    config.title = "fluid";
    config.headless = !shot_path.empty();
    if (config.headless) {
        config.width = shot_w;
        config.height = shot_h;
    }
    std::string error;
    auto app = eng::app::App::Create(config, error);
    if (!app) return Fail(error);

    // Own renderer, single-sampled: the surface targets are single-sample
    // textures the shade pass reads back, and a multisampled tank would need
    // a resolve the surface could not share.
    auto renderer =
        eng::Renderer::Create(app->Gpu(), eng::rhi::Format::RGBA8Unorm, error, 1);
    if (!renderer) return Fail(error);

    // --- the tank ------------------------------------------------------------
    eng::Scene scene;
    scene.lightDir = eng::Vec4{0.4f, 0.7f, 0.6f, 0.0f};
    scene.lightColor = eng::Vec4{3.0f, 2.9f, 2.7f, 1.0f};
    scene.ambientSky = eng::Vec3{0.35f, 0.42f, 0.52f};
    scene.ambientGround = eng::Vec3{0.10f, 0.10f, 0.10f};

    // Pale sand under the water (so refraction has something to bend) and
    // dark slate walls (so the surface has something to reflect against).
    eng::MaterialDesc sand_md, slate_md;
    sand_md.shading = eng::Shading::Lit;
    sand_md.base_color = eng::Vec4{0.55f, 0.50f, 0.38f, 1.0f};
    sand_md.roughness = 0.9f;
    slate_md.shading = eng::Shading::Lit;
    slate_md.base_color = eng::Vec4{0.10f, 0.13f, 0.17f, 1.0f};
    slate_md.roughness = 0.55f;
    const eng::MaterialHandle sand_mat =
        renderer->CreateMaterial(sand_md, error);
    const eng::MaterialHandle slate_mat =
        renderer->CreateMaterial(slate_md, error);
    if (!eng::Valid(sand_mat) || !eng::Valid(slate_mat)) return Fail(error);

    // Half extents: MakeBox takes them, and the inner faces land exactly on
    // the sim bounds, so a particle touching a wall touches something the
    // eye can see.
    const auto box = [&](eng::Vec3 centre, eng::Vec3 half,
                         eng::MaterialHandle mat) {
        eng::MeshHandle mesh = renderer->UploadMesh(
            eng::MakeBox(half, eng::Vec4{1, 1, 1, 1}));
        if (!eng::Valid(mesh)) return false;
        eng::Instance inst;
        inst.mesh = mesh;
        inst.material = mat;
        inst.model = eng::Mat4::Translation(centre);
        scene.instances.push_back(inst);
        return true;
    };
    if (!box(eng::Vec3{0.0f, -0.05f, 0.0f}, eng::Vec3{0.95f, 0.05f, 0.55f},
             sand_mat))
        return Fail("floor upload");
    if (!box(eng::Vec3{0.0f, 0.5f, kZ0 - 0.03f}, eng::Vec3{0.95f, 0.55f, 0.03f},
             slate_mat))
        return Fail("wall upload");
    if (!box(eng::Vec3{0.0f, 0.5f, kZ1 + 0.03f}, eng::Vec3{0.95f, 0.55f, 0.03f},
             slate_mat))
        return Fail("wall upload");
    // End walls butt against the side walls' inner faces (z half 0.40, not
    // 0.43): overlapping boxes z-fight where their faces coincide, which
    // reads as crawling moire on the corners.
    if (!box(eng::Vec3{kX0 - 0.03f, 0.5f, 0.0f}, eng::Vec3{0.03f, 0.55f, 0.40f},
             slate_mat))
        return Fail("wall upload");
    if (!box(eng::Vec3{kX1 + 0.03f, 0.5f, 0.0f}, eng::Vec3{0.03f, 0.55f, 0.40f},
             slate_mat))
        return Fail("wall upload");

    // --- the water -----------------------------------------------------------
    eng::FluidConfig fcfg;
    fcfg.bounds_min = eng::Vec3{kX0, 0.0f, kZ0};
    fcfg.bounds_max = eng::Vec3{kX1, kY1, kZ1};
    fcfg.smoothing_radius = 0.07f;
    fcfg.stiffness = 250.0f;
    fcfg.viscosity = 0.30f;
    const std::vector<eng::Vec3> column = DamColumn();
    std::printf("  %zu particles\n", column.size());
    auto sim = eng::FluidSim::Create(app->Gpu(), fcfg, column,
                                     eng::Renderer::kSceneFormat, error, 1);
    if (!sim) return Fail(error);
    auto surface = eng::FluidSurface::Create(app->Gpu(), error);
    if (!surface) return Fail(error);
    eng::SurfaceLook look;
    look.sun_dir = eng::Vec3{0.4f, 0.7f, 0.6f};
    look.sun_intensity = 3.0f;

    // Targets. The tank renders into tank_color/tank_depth, the surface
    // shade reads both and writes water_color, the composite tone maps to
    // the drawable (or the shot file).
    const auto hdr = eng::Renderer::kSceneFormat;
    eng::rhi::TextureId tank_color, tank_depth, water_color, out_target;
    int target_w = 0, target_h = 0;
    const auto size_targets = [&](int w, int h) {
        if (w == target_w && h == target_h && eng::rhi::Valid(tank_color))
            return true;
        target_w = w;
        target_h = h;
        tank_color = app->Gpu().CreateRenderTarget(w, h, hdr);
        tank_depth = app->Gpu().CreateDepthTarget(w, h, /*sampleable=*/true);
        water_color = app->Gpu().CreateRenderTarget(w, h, hdr);
        out_target = app->Gpu().CreateRenderTarget(w, h,
                                                   eng::rhi::Format::RGBA8Unorm,
                                                   /*cpu_readable=*/true);
        if (!surface->BeginFrame(app->Gpu(), w, h, error)) return false;
        return eng::rhi::Valid(tank_color) && eng::rhi::Valid(tank_depth) &&
               eng::rhi::Valid(water_color) && eng::rhi::Valid(out_target);
    };

    app->Actions().BindMouse("orbit", eng::app::MouseButton::Left);
    app->Actions().Bind("reset", 'r');
    std::printf("drag to orbit, wheel to zoom, R to refill, esc to quit\n");

    // Pitched down steeply: the walls stand a metre tall and a shallow angle
    // sights along them instead of into the tank -- the first version of
    // this looked into a wall and concluded there was no water.
    // 0.72 looks down into the tank; 0.55 sights into the near wall and the
    // tank reads as empty (tried). Grazing angles would give more sky
    // reflection, but not through a metre of slate.
    float yaw = -0.65f, pitch = 0.72f, dist = 2.8f;
    const eng::Vec3 focus{0.0f, 0.30f, 0.0f};
    int frame = 0;
    while (app->Running()) {
        if (!app->BeginFrame()) continue;
        const eng::app::Frame& f = app->Current();
        if (!size_targets(f.width, f.height)) return Fail(error);

        if (app->Actions().Down("orbit")) {
            yaw += f.drag_dx * 0.006f;
            pitch = std::clamp(pitch + f.drag_dy * 0.004f, 0.05f, 1.35f);
        }
        dist = std::clamp(dist - f.scroll * 0.25f, 1.2f, 8.0f);
        if (app->Actions().Pressed("reset")) {
            sim = eng::FluidSim::Create(app->Gpu(), fcfg, column,
                                        eng::Renderer::kSceneFormat, error, 1);
            if (!sim) return Fail(error);
        }
        scene.camera.target = focus;
        scene.camera.eye =
            focus + eng::Vec3{std::cos(pitch) * std::cos(yaw),
                              std::sin(pitch),
                              std::cos(pitch) * std::sin(yaw)} *
                        dist;

        // The sim first, on a compute encoder: five dispatches per substep,
        // all before any pass reads the particles.
        {
            auto ce = app->Gpu().BeginCompute();
            const int sub = sim->Step(ce, config.headless ? config.fixed_dt
                                                          : std::min(f.dt, 1.0f / 30.0f));
            app->Gpu().EndCompute();
            // Every 256 frames: ReadPositions stalls the pipe, so this health
            // line costs a hitch and stays rare.
            if ((frame & 255) == 0) {
                float lo = 1e9f, hi = -1e9f;
                for (const eng::Vec3& p : sim->ReadPositions()) {
                    lo = std::min(lo, p.y);
                    hi = std::max(hi, p.y);
                }
                std::printf("frame %d: %d substeps, %d particles, y in [%.3f, %.3f], outside %d\n",
                            frame, sub, sim->Count(), lo, hi,
                            sim->OutsideBounds());
            }
        }

        // The tank, into its own color and depth. The water keeps its own
        // depth buffer (surface->Depth()): sharing one baked the walls into
        // the surface, and the shade pass resolves occlusion by reading both.
        {
            eng::rhi::PassDesc pd;
            pd.color = tank_color;
            pd.depth = tank_depth;
            pd.clear_color[0] = 0.04f;
            pd.clear_color[1] = 0.06f;
            pd.clear_color[2] = 0.09f;
            pd.clear_color[3] = 1.0f;
            pd.clear_depth = 0.0f;
            pd.keep_depth = true;
            auto e = app->Gpu().BeginPass(pd);
            renderer->DrawScene(e, scene, f.width, f.height, {});
            app->Gpu().EndPass();
        }
        // Sphere depth, water only. No color attachment: depth_only pipeline.
        {
            eng::rhi::PassDesc pd;
            pd.depth = surface->Depth();
            pd.clear_depth = 0.0f;
            pd.keep_depth = true;
            auto e = app->Gpu().BeginPass(pd);
            surface->DrawDepth(e, *sim, scene.camera, look, f.width, f.height);
            app->Gpu().EndPass();
        }
        // Bilateral smooth, one axis per pass (a target cannot be read and
        // written in the same pass).
        {
            eng::rhi::PassDesc pd;
            pd.color = surface->SmoothH();
            auto e = app->Gpu().BeginPass(pd);
            surface->SmoothH(e, scene.camera, look, f.width, f.height);
            app->Gpu().EndPass();
        }
        {
            eng::rhi::PassDesc pd;
            pd.color = surface->Smooth();
            auto e = app->Gpu().BeginPass(pd);
            surface->SmoothV(e, scene.camera, look, f.width, f.height);
            app->Gpu().EndPass();
        }
        // Water optics over the tank, then the tone map to the readable
        // target (headless) or straight to the drawable.
        {
            eng::rhi::PassDesc pd;
            pd.color = water_color;
            auto e = app->Gpu().BeginPass(pd);
            surface->DrawShade(e, scene.camera, look, f.width, f.height,
                               tank_color, tank_depth);
            app->Gpu().EndPass();
        }
        {
            eng::rhi::PassDesc pd;
            pd.color = shot_path.empty() ? f.drawable : out_target;
            auto e = app->Gpu().BeginPass(pd);
            renderer->DrawComposite(e, water_color, {}, {}, 0.0f, 1.0f);
            app->Gpu().EndPass();
        }
        app->EndFrame();
        if (!shot_path.empty() && ++frame >= shot_frames) break;
        if (shot_path.empty()) ++frame;
    }

    if (!shot_path.empty()) {
        std::vector<std::uint8_t> px(std::size_t(target_w) * target_h * 4);
        if (!app->Gpu().ReadPixels(out_target, target_w, target_h, px))
            return Fail("readback");
        if (!eng::png::EncodeFile(shot_path, px, target_w, target_h, error))
            return Fail(error);
        std::printf("wrote %s (%dx%d)\n", shot_path.c_str(), target_w, target_h);
    }
    return 0;
}
