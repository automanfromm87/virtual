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
// How much of what is behind a pane survives it. Low, because the point of a
// glass tank is the water: at 0.18 a pane reads as a faint tinted step against
// the background and dims the water behind it by about 4%.
//
// The blend is not premultiplied, so this same number scales the pane's OWN
// reflection -- a bright glint and 85% transmission cannot both come out of one
// draw. Raising it to make the glass more visible costs the water directly, and
// the right lever when the glass looks absent is its base_color, not this.
constexpr float kGlassAlpha = 0.18f;
// The dam break is the opening shot; the circulation is what keeps it alive.
// Gated, because at t=0 the whole column is at the -x end with nothing in the
// drain, and then the surge front crosses it with a fifth of the fluid under
// the drain height at once -- twenty-odd particles returned into a blob with
// room for thirteen, which is several times rest density and a fire hose.
constexpr float kFlowStartSeconds = 3.0f;
// Particles returned PER SECOND. Recirculation::per_step is per CALL and
// carries no dt, so a constant there would double the flow on a 120 Hz panel.
constexpr float kReturnsPerSecond = 440.0f;
constexpr int kHealthEvery = 60;

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

    // Pale sand under the water, so what the water REFRACTS is a tank floor
    // rather than a void. The old comment here said the dark slate walls gave
    // "the surface something to reflect against", and that was never true of
    // this renderer: the water's reflection is an analytic two-colour lerp
    // (fluid_surface.metal) and samples nothing from the scene at all. What
    // the walls actually feed is the refraction term, weighted by 1 - fresnel
    // -- which is an argument for making them see-through, not dark.
    eng::MaterialDesc sand_md, glass_md, frame_md;
    sand_md.shading = eng::Shading::Lit;
    sand_md.base_color = eng::Vec4{0.55f, 0.50f, 0.38f, 1.0f};
    sand_md.roughness = 0.9f;
    // GLASS. base_color is not near-black: with no reflection probe bound, a
    // vertical pane's whole appearance is the hemisphere ambient times its
    // albedo plus a fresnel term of about 0.08, so an albedo of 0.05 is a pane
    // that is simply not there. 0.16-0.22 puts it a visible step above the
    // background without competing with the water.
    glass_md.shading = eng::Shading::Lit;
    glass_md.base_color = eng::Vec4{0.30f, 0.34f, 0.36f, 1.0f};
    glass_md.roughness = 0.08f;
    glass_md.transparent = true;
    // A METAL FRAME, so the tank has an edge. Four glass panes with nothing
    // around them read as a smudge; the rim and the corner posts are what say
    // "tank". Not a dark metal: a dark metal is whatever the sky puts in its
    // specular lobe, which with no probe is a flat hemisphere, and it reads as
    // a hole cut in the picture.
    frame_md.shading = eng::Shading::Lit;
    frame_md.base_color = eng::Vec4{0.42f, 0.44f, 0.47f, 1.0f};
    frame_md.roughness = 0.30f;
    frame_md.metallic = 0.55f;
    const eng::MaterialHandle sand_mat = renderer->CreateMaterial(sand_md, error);
    const eng::MaterialHandle glass_mat = renderer->CreateMaterial(glass_md, error);
    const eng::MaterialHandle frame_mat = renderer->CreateMaterial(frame_md, error);
    if (!eng::Valid(sand_mat) || !eng::Valid(glass_mat) || !eng::Valid(frame_mat))
        return Fail(error);

    // Half extents: MakeBox takes them, and the inner faces land exactly on
    // the sim bounds, so a particle touching a wall touches something the
    // eye can see.
    //
    // THE PANES ARE COLLECTED, NOT PUSHED. A glass wall in front of the water
    // and a glass wall behind it have to be drawn in different passes -- see
    // the split in the frame loop -- so they are kept apart from the opaque
    // parts, each with the outward normal that decides which side it is on.
    std::vector<eng::Instance> tank_opaque;
    struct Pane {
        eng::Instance inst;
        eng::Vec3 centre, outward;
    };
    std::vector<Pane> panes;
    bool built = true;
    const auto make = [&](eng::Vec3 centre, eng::Vec3 half,
                          eng::MaterialHandle mat, float alpha) {
        eng::Instance inst;
        inst.mesh = renderer->UploadMesh(eng::MakeBox(half, eng::Vec4{1, 1, 1, 1}));
        inst.material = mat;
        inst.model = eng::Mat4::Translation(centre);
        // THE ALPHA LIVES ON THE INSTANCE TINT, not on base_color.w. The lit
        // shader ends with float4(lit + emit, in.color.a) and in.color is the
        // vertex colour times this tint; base_color contributes rgb only.
        // Setting base_color.w and expecting glass gives four fully opaque,
        // nearly black walls, which reads as a lighting bug.
        inst.tint = eng::Vec4{1.0f, 1.0f, 1.0f, alpha};
        if (!eng::Valid(inst.mesh)) built = false;
        return inst;
    };
    const auto opaque = [&](eng::Vec3 centre, eng::Vec3 half,
                            eng::MaterialHandle mat) {
        tank_opaque.push_back(make(centre, half, mat, 1.0f));
    };
    const auto pane = [&](eng::Vec3 centre, eng::Vec3 half, eng::Vec3 outward) {
        panes.push_back({make(centre, half, glass_mat, kGlassAlpha), centre, outward});
    };

    opaque(eng::Vec3{0.0f, -0.05f, 0.0f}, eng::Vec3{0.95f, 0.05f, 0.55f}, sand_mat);
    // Four panes where the slate was, same faces so the water still touches
    // something the eye can see.
    pane(eng::Vec3{0.0f, 0.5f, kZ0 - 0.03f}, eng::Vec3{0.95f, 0.55f, 0.03f},
         eng::Vec3{0.0f, 0.0f, -1.0f});
    pane(eng::Vec3{0.0f, 0.5f, kZ1 + 0.03f}, eng::Vec3{0.95f, 0.55f, 0.03f},
         eng::Vec3{0.0f, 0.0f, 1.0f});
    // End panes butt against the side panes' inner faces (z half 0.40, not
    // 0.43): overlapping boxes z-fight where their faces coincide, which
    // reads as crawling moire on the corners.
    pane(eng::Vec3{kX0 - 0.03f, 0.5f, 0.0f}, eng::Vec3{0.03f, 0.55f, 0.40f},
         eng::Vec3{-1.0f, 0.0f, 0.0f});
    pane(eng::Vec3{kX1 + 0.03f, 0.5f, 0.0f}, eng::Vec3{0.03f, 0.55f, 0.40f},
         eng::Vec3{1.0f, 0.0f, 0.0f});
    // A rim along the top of every pane and a post at every corner. The posts
    // are 0.035 rather than 0.030 so they STRADDLE the pane faces instead of
    // sitting flush -- coplanar faces z-fight, and here the glass would lose
    // the test and quietly vanish along its edge.
    opaque(eng::Vec3{0.0f, 1.03f, kZ0 - 0.03f}, eng::Vec3{0.95f, 0.03f, 0.035f}, frame_mat);
    opaque(eng::Vec3{0.0f, 1.03f, kZ1 + 0.03f}, eng::Vec3{0.95f, 0.03f, 0.035f}, frame_mat);
    opaque(eng::Vec3{kX0 - 0.03f, 1.03f, 0.0f}, eng::Vec3{0.035f, 0.03f, 0.395f}, frame_mat);
    opaque(eng::Vec3{kX1 + 0.03f, 1.03f, 0.0f}, eng::Vec3{0.035f, 0.03f, 0.395f}, frame_mat);
    for (float sx : {-1.0f, 1.0f})
        for (float sz : {-1.0f, 1.0f})
            opaque(eng::Vec3{sx * 0.83f, 0.475f, sz * 0.43f},
                   eng::Vec3{0.035f, 0.525f, 0.035f}, frame_mat);
    if (!built) return Fail("tank upload");

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
    float sim_time = 0.0f;
    std::vector<eng::Instance> behind, front;
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
            sim_time = 0.0f;
        }
        scene.camera.target = focus;
        scene.camera.eye =
            focus + eng::Vec3{std::cos(pitch) * std::cos(yaw),
                              std::sin(pitch),
                              std::cos(pitch) * std::sin(yaw)} *
                        dist;

        // WHICH PANES ARE IN FRONT OF THE WATER, recomputed every frame from
        // the live eye. Not once at setup: the orbit crosses a quadrant and the
        // near pane would start drawing underneath the water, which is an
        // intermittent angle-dependent bug and miserable to bisect.
        //
        // The panes behind go in with the opaque tank, because there the water
        // shade's refraction sample is genuinely looking through water at glass
        // and tinting it correctly. The panes in front cannot: in the tank
        // image they would be smeared by the water's own surface slope and
        // erased wherever fresnel goes to one, which is exactly the grazing
        // angle that makes glass read as glass. They are blended over the
        // finished water instead.
        //
        // The split is exact and cannot pop. The panes are axis-aligned slabs
        // strictly outside a convex interior and every particle is confined to
        // that interior, so no view ray straddles a pane.
        behind = tank_opaque;
        front.clear();
        for (const Pane& p : panes)
            (eng::Dot(scene.camera.eye - p.centre, p.outward) > 0.0f ? front : behind)
                .push_back(p.inst);

        // The sim first, on a compute encoder: five dispatches per substep,
        // all before any pass reads the particles.
        {
            const float dt = config.headless ? config.fixed_dt
                                             : std::min(f.dt, 1.0f / 30.0f);
            sim_time += dt;
            auto ce = app->Gpu().BeginCompute("fluid");
            // A NEAR-SURFACE JET AT ONE END AND A FLOOR DRAIN AT THE OTHER.
            // Without it this is a dam break, which is one collapse and then a
            // settled puddle for as long as anyone watches.
            //
            // The spout sits just UNDER the settled surface, not above it: a
            // jet with neighbours from its first step stays a jet, and one
            // dropped into air becomes a thin free-falling thread that lands
            // and disappears. It throws a mound about 12 cm proud of the pool
            // and well clear of the invisible lid at the top of the box.
            if (sim_time > kFlowStartSeconds) {
                eng::FluidSim::Recirculation flow;
                flow.spout = eng::Vec3{kX0 + 0.18f, 0.22f, 0.0f};
                flow.spread = 0.055f;
                flow.velocity = eng::Vec3{2.2f, 2.0f, 0.0f};
                flow.drain = eng::Vec3{kX1 - 0.30f, 0.0f, 0.0f};
                flow.drain_radius = 0.30f;
                flow.drain_y = 0.12f;
                flow.per_step =
                    std::clamp(int(std::lround(kReturnsPerSecond * dt)), 8, 250);
                sim->Recirculate(ce, flow);
            }
            const int sub = sim->Step(ce, dt);
            app->Gpu().EndCompute();
            // Every 256 frames: ReadPositions stalls the pipe, so this health
            // line costs a hitch and stays rare.
            // KINETIC ENERGY AND DENSITY, not just the extent. "Is it still
            // moving" and "is the solver healthy" are the two questions a
            // fluid cannot be asked from a picture, and both readbacks map the
            // same buffer ReadPositions already maps -- they ride inside a
            // stall that is being paid anyway. Every 60 frames, not 256: at
            // 256 the default capture printed exactly one line, at frame 0,
            // before a single step had run.
            if (frame % kHealthEvery == 0) {
                float lo = 1e9f, hi = -1e9f;
                for (const eng::Vec3& p : sim->ReadPositions()) {
                    lo = std::min(lo, p.y);
                    hi = std::max(hi, p.y);
                }
                float mean = 0.0f, dlo = 0.0f, dhi = 0.0f;
                sim->ReadDensity(&mean, &dlo, &dhi);
                std::printf("frame %d: %d substeps, y in [%.3f, %.3f], KE %.2f, "
                            "rho %.0f [%.0f, %.0f], outside %d\n",
                            frame, sub, lo, hi, sim->KineticEnergy(), mean, dlo,
                            dhi, sim->OutsideBounds());
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
            scene.instances = behind;
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
        // THE NEAR GLASS, over the finished water. Three things here are not
        // optional. The colour target is water_color and not the drawable,
        // because a Lit pipeline is always built for the scene's half-float
        // format and the drawable is eight-bit. `load` is on, or the pass
        // clears the water away and leaves glass on a flat background. And the
        // depth attachment has to be here: the renderer drops every instance
        // whose depth state disagrees with the pass, into a stats field, in
        // silence -- so without it the pane simply does not appear, and the
        // temptation is to keep winding the alpha up chasing it.
        if (!front.empty()) {
            scene.instances = front;
            eng::rhi::PassDesc pd;
            pd.color = water_color;
            pd.load = true;
            pd.depth = tank_depth;
            pd.load_depth = true;
            pd.keep_depth = true;
            auto e = app->Gpu().BeginPass(pd);
            renderer->DrawScene(e, scene, f.width, f.height, {});
            app->Gpu().EndPass();
        }
        {
            eng::rhi::PassDesc pd;
            pd.color = shot_path.empty() ? f.drawable : out_target;
            auto e = app->Gpu().BeginPass(pd);
            // No vignette in a capture: it darkens the corners by 38% and
            // every measurement of this frame has to special-case it.
            renderer->DrawComposite(e, water_color, {}, {}, 0.0f,
                                    shot_path.empty() ? 1.0f : 0.0f);
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
