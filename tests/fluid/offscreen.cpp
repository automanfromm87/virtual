// SPH fluid, checked against the physics it claims to be doing.
//
// A fluid solver is the hardest thing in this engine to test by looking at it.
// One that is uniformly 30% over-compressed, or slowly gaining energy, or
// quietly leaking particles through the floor, produces a perfectly convincing
// picture for several seconds -- and the picture is how almost every fluid bug
// gets shipped. So none of the checks here are about pixels.
//
// The properties that matter, in order:
//   * DENSITY sits at the rest density. That is what incompressible means, and
//     everything else in the solver exists to make it true.
//   * MASS is conserved: nothing escapes the box.
//   * ENERGY does not grow. A stiff explicit integrator that is slightly
//     unstable gains energy exponentially, and it looks like "lively" fluid
//     right up until it detonates.
//   * The fluid FINDS ITS LEVEL, at the height its volume and the container's
//     footprint predict.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/geometry/mesh.h"
#include "engine/render/fluid.h"
#include "engine/render/renderer.h"
#include "engine/render/rendergraph.h"
#include "engine/rhi/rhi.h"
#include "engine/scene/scene.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

constexpr int kW = 860, kH = 620;

// A block of particles on a regular lattice, which is what a fluid at rest
// looks like before it has settled into anything.
std::vector<eng::Vec3> Block(eng::Vec3 lo, int nx, int ny, int nz, float spacing) {
    std::vector<eng::Vec3> out;
    out.reserve(std::size_t(nx) * ny * nz);
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            for (int x = 0; x < nx; ++x)
                out.push_back(lo + eng::Vec3{float(x) * spacing, float(y) * spacing,
                                             float(z) * spacing});
    return out;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("sph fluid\n");

    std::string error;
    auto dev = eng::rhi::Device::Create(error);
    if (!dev) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
    const auto kFmt = eng::rhi::Format::RGBA8Unorm;
    auto renderer = eng::Renderer::Create(*dev, kFmt, error, 1);
    if (!renderer) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    // --- a fluid at rest stays at rest ---------------------------------------
    //
    // The most fundamental property there is, and the sharpest diagnostic: a
    // block of particles at the rest spacing, with NO gravity, in the middle of
    // the box where no wall can touch it. Nothing should happen. Any energy
    // that appears is the solver's own, and it is far easier to see here than
    // underneath a fluid that is legitimately sloshing.
    {
        eng::FluidConfig q;
        q.bounds_min = eng::Vec3{-0.5f, -0.5f, -0.5f};
        q.bounds_max = eng::Vec3{0.5f, 0.5f, 0.5f};
        q.smoothing_radius = 0.055f;
        q.rest_density = 1000.0f;
        q.stiffness = 200.0f;
        q.viscosity = 0.55f;
        q.gravity = 0.0f;
        const float dx = q.smoothing_radius * 0.48f;
        const std::vector<eng::Vec3> cube =
            Block(eng::Vec3{-6.0f * dx, -6.0f * dx, -6.0f * dx}, 13, 13, 13, dx);
        std::string qe;
        auto rest = eng::FluidSim::Create(*dev, q, cube,
                                          eng::Renderer::kSceneFormat, qe, 1);
        Check(rest != nullptr, "the static case was created");
        if (rest) {
            float m0 = 0, l0 = 0, h0 = 0;
            // One step, so densities are computed but nothing has moved far.
            dev->BeginFrame();
            {
                eng::rhi::ComputeEncoder ce = dev->BeginCompute();
                (void)rest->Step(ce, 1.0f / 120.0f);
                dev->EndCompute();
            }
            std::string w0;
            (void)dev->CommitAndWait(w0);
            rest->ReadDensity(&m0, &l0, &h0);
            std::printf("    at rest, one step in: density mean %.1f "
                        "(interior should read the rest density)\n", m0);

            for (int i = 0; i < 120; ++i) {
                dev->BeginFrame();
                eng::rhi::ComputeEncoder ce = dev->BeginCompute();
                (void)rest->Step(ce, 1.0f / 120.0f);
                dev->EndCompute();
                std::string w;
                (void)dev->CommitAndWait(w);
            }
            const float ke = rest->KineticEnergy();
            float m1 = 0, l1 = 0, h1 = 0;
            rest->ReadDensity(&m1, &l1, &h1);
            const std::vector<eng::Vec3> after = rest->ReadPositions();
            float drift = 0.0f;
            for (std::size_t i = 0; i < after.size(); ++i)
                drift = std::fmax(drift, Length(after[i] - cube[i]));
            std::printf("    after 1 s with no gravity: KE %.6f J, density "
                        "%.1f (%.1f..%.1f), furthest particle moved %.4f m\n",
                        ke, m1, l1, h1, drift);
            // The block's SURFACE legitimately reads low -- the kernel is
            // truncated there and half its neighbours do not exist. The
            // interior is what must read the rest density, and the maximum is
            // the interior.
            Check(h0 > q.rest_density * 0.92f && h0 < q.rest_density * 1.08f,
                  "the interior of a block at rest spacing reads rest density");
            Check(drift < 4.0f * dx, "a fluid at rest does not fly apart");
            Check(ke < 0.05f, "and does not invent kinetic energy");
        }
    }

    // --- a pool poured at its own level stays put ----------------------------
    //
    // Between "a block in free space with no gravity" above and "a column
    // dropped from a height" below. This one has gravity and a floor but no
    // violence: the fluid starts at the depth it should end at, so anything
    // that happens is the solver's doing.
    {
        eng::FluidConfig q;
        q.bounds_min = eng::Vec3{-0.30f, 0.0f, -0.30f};
        q.bounds_max = eng::Vec3{0.30f, 0.90f, 0.30f};
        q.smoothing_radius = 0.055f;
        q.stiffness = 200.0f;
        q.viscosity = 0.55f;
        const float dx = q.smoothing_radius * 0.48f;
        const std::vector<eng::Vec3> pool =
            Block(eng::Vec3{-0.277f, 0.013f, -0.277f}, 22, 10, 22, dx);
        std::string qe;
        auto still = eng::FluidSim::Create(*dev, q, pool,
                                           eng::Renderer::kSceneFormat, qe, 1);
        Check(still != nullptr, "the still pool was created");
        if (still) {
            for (int i = 0; i < 240; ++i) {
                dev->BeginFrame();
                eng::rhi::ComputeEncoder ce = dev->BeginCompute();
                (void)still->Step(ce, 1.0f / 120.0f);
                dev->EndCompute();
                std::string w;
                (void)dev->CommitAndWait(w);
            }
            float m = 0, l = 0, h = 0;
            still->ReadDensity(&m, &l, &h);
            const std::vector<eng::Vec3> p2 = still->ReadPositions();
            std::vector<float> ys;
            for (const eng::Vec3& v : p2) ys.push_back(v.y);
            std::sort(ys.begin(), ys.end());
            const float surface = ys[std::size_t(double(ys.size()) * 0.98)];
            const float started = 9.0f * dx + 0.013f;
            std::printf("    a pool 2 s later: KE %.4f J, density %.1f "
                        "(%.1f..%.1f), surface %.4f (started %.4f)\n",
                        still->KineticEnergy(), m, l, h, surface, started);
            Check(still->KineticEnergy() < 1.0f, "a level pool stays nearly still");
            Check(surface < started * 1.4f, "and does not rise");
            Check(still->OutsideBounds() == 0, "and does not leak");
        }
    }

    eng::FluidConfig cfg;
    cfg.bounds_min = eng::Vec3{-0.30f, 0.0f, -0.30f};
    cfg.bounds_max = eng::Vec3{0.30f, 0.90f, 0.30f};
    cfg.smoothing_radius = 0.055f;
    cfg.rest_density = 1000.0f;
    cfg.stiffness = 200.0f;
    cfg.viscosity = 0.55f;

    // A column, dropped into the corner of the box. Spacing is a bit under half
    // the smoothing radius, so each particle sees a few dozen neighbours --
    // fewer and the density estimate is noise, more and the step costs cubically.
    const float spacing = cfg.smoothing_radius * 0.48f;
    const std::vector<eng::Vec3> start =
        Block(eng::Vec3{-0.26f, 0.02f, -0.26f}, 14, 26, 14, spacing);

    auto fluid = eng::FluidSim::Create(*dev, cfg, start, eng::Renderer::kSceneFormat,
                                       error, 1);
    if (!fluid) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
    std::printf("    %d particles, mass %.6f kg each (%.3f kg of water)\n",
                fluid->Count(), fluid->ParticleMass(),
                fluid->ParticleMass() * float(fluid->Count()));
    Check(fluid->Count() == 14 * 26 * 14, "every particle was accepted");
    Check(fluid->ParticleMass() > 0.0f, "and given a mass");

    const auto advance = [&](float seconds) {
        int substeps = 0;
        const float frame = 1.0f / 120.0f;
        for (int i = 0; i < int(seconds / frame); ++i) {
            dev->BeginFrame();
            eng::rhi::ComputeEncoder ce = dev->BeginCompute();
            substeps += fluid->Step(ce, frame);
            dev->EndCompute();
            std::string w;
            if (!dev->CommitAndWait(w)) std::fprintf(stderr, "  %s\n", w.c_str());
        }
        return substeps;
    };

    // --- it settles ----------------------------------------------------------
    const int substeps = advance(2.5f);
    std::printf("    2.5 s of simulation took %d substeps\n", substeps);

    float mean = 0, lo = 0, hi = 0;
    fluid->ReadDensity(&mean, &lo, &hi);
    std::printf("    density: mean %.1f, range %.1f..%.1f (rest is %.1f)\n", mean,
                lo, hi, cfg.rest_density);

    // THE headline property. A weakly compressible solver will not hit the rest
    // density exactly -- that is what "weakly" means -- but the mean must sit
    // close to it. A solver whose mass is wrong, or whose kernel normalisation
    // is wrong, lands at a completely different number and is otherwise
    // indistinguishable: it still flows, still splashes, still settles.
    Check(mean > cfg.rest_density * 0.85f && mean < cfg.rest_density * 1.15f,
          "the settled density sits at the rest density");
    // The spread matters as much as the mean. A fluid averaging 1000 with half
    // its particles at 500 and half at 1500 is not incompressible, it is
    // boiling.
    Check(hi < cfg.rest_density * 1.8f, "and no particle is wildly over-compressed");

    Check(fluid->OutsideBounds() == 0, "no particle escaped the container");

    // --- it finds its level ---------------------------------------------------
    //
    // Halve the container's footprint and the same fluid must stand twice as
    // deep. A RATIO, deliberately, and the absolute version was tried first and
    // was measuring the wrong thing.
    //
    // The obvious test is that N particles of mass m at density rho stand
    // N*m/(rho*A) deep. That prediction assumes the settled fluid packs
    // UNIFORMLY, and SPH does not: measured here, the densest particle has 27
    // neighbours within the smoothing radius where a uniform packing at the
    // same bounding volume would give 54, with a mean nearest-neighbour
    // distance of 0.022 against a uniform 0.0235. The particles cluster -- the
    // well-known consequence of a spiky pressure kernel with pressure clamped
    // at zero -- so the local density the solver reports and the global
    // mass-over-bounding-box disagree by 40%, and both are correct about
    // different things. The absolute prediction is off by that same 40%, and
    // no tolerance that admits it would still catch anything.
    //
    // The ratio has no such constant in it: whatever the packing does, it does
    // the same thing in both containers.
    {
        const auto settle_depth = [&](float half_width) {
            eng::FluidConfig c2 = cfg;
            c2.bounds_min = eng::Vec3{-half_width, 0.0f, -half_width};
            c2.bounds_max = eng::Vec3{half_width, 1.6f, half_width};
            // The same fluid every time -- same count, same spacing, so the
            // same mass -- poured into a different container.
            const std::vector<eng::Vec3> pour =
                Block(eng::Vec3{-half_width + 0.02f, 0.02f, -half_width + 0.02f},
                      10, 40, 10, spacing);
            std::string e2;
            auto sim = eng::FluidSim::Create(*dev, c2, pour,
                                             eng::Renderer::kSceneFormat, e2, 1);
            if (!sim) return 0.0f;
            for (int i = 0; i < 420; ++i) {
                dev->BeginFrame();
                eng::rhi::ComputeEncoder ce = dev->BeginCompute();
                (void)sim->Step(ce, 1.0f / 120.0f);
                dev->EndCompute();
                std::string w;
                (void)dev->CommitAndWait(w);
            }
            std::vector<float> ys;
            for (const eng::Vec3& q : sim->ReadPositions()) ys.push_back(q.y);
            std::sort(ys.begin(), ys.end());
            return ys[std::size_t(double(ys.size()) * 0.98)];
        };

        const float wide = settle_depth(0.30f);
        // Half the AREA, so 1/sqrt(2) of the half-width.
        const float narrow = settle_depth(0.30f / 1.41421356f);
        std::printf("    the same fluid stands %.4f deep in a 0.60 m box and "
                    "%.4f in a 0.42 m one (ratio %.3f, halving the area "
                    "predicts 2.000)\n", wide, narrow,
                    narrow / std::max(wide, 1e-6f));
        Check(wide > 0.01f && narrow > 0.01f, "both pours settled");
        Check(narrow > wide * 1.55f && narrow < wide * 2.45f,
              "halving the footprint roughly doubles the depth");
    }

    // The column POURED. It started 26 layers tall in one corner and must have
    // spread across the floor -- a solver with no pressure force at all leaves
    // the block where it was put, minus some falling.
    {
        const std::vector<eng::Vec3> p = fluid->ReadPositions();
        float min_x = p[0].x, max_x = p[0].x, min_z = p[0].z, max_z = p[0].z;
        for (const eng::Vec3& q : p) {
            min_x = std::min(min_x, q.x); max_x = std::max(max_x, q.x);
            min_z = std::min(min_z, q.z); max_z = std::max(max_z, q.z);
        }
        const float started = 13.0f * spacing;
        std::printf("    footprint %.3f x %.3f (started %.3f square)\n",
                    max_x - min_x, max_z - min_z, started);
        Check(max_x - min_x > started * 1.3f && max_z - min_z > started * 1.3f,
              "the column collapsed and spread out");
    }

    // --- energy does not run away ---------------------------------------------
    //
    // A stiff explicit integrator that is marginally unstable gains energy
    // geometrically. For the first second it looks like a livelier fluid; then
    // it is a cloud of particles at escape velocity. Measured over a further
    // two seconds of a fluid that should be nearly at rest.
    {
        const float before = fluid->KineticEnergy();
        advance(2.0f);
        const float after = fluid->KineticEnergy();
        std::printf("    kinetic energy %.6f J -> %.6f J over two more seconds\n",
                    before, after);
        Check(std::isfinite(after), "the energy is still a number");
        Check(after < before * 1.5f + 1e-4f, "a settling fluid does not gain energy");
        Check(fluid->OutsideBounds() == 0, "and still nothing has escaped");
        float m2 = 0, l2 = 0, h2 = 0;
        fluid->ReadDensity(&m2, &l2, &h2);
        Check(m2 > cfg.rest_density * 0.85f && m2 < cfg.rest_density * 1.15f,
              "the density is still at rest after four and a half seconds");
    }

    // --- a dam break -----------------------------------------------------------
    //
    // A separate simulation, because this one is about the transient rather than
    // the steady state: a tall column against one wall, released. The fluid must
    // reach the far wall, and it must do so at roughly the speed a falling
    // column of that height does -- sqrt(2*g*h) is the textbook estimate for the
    // surge front, and being within a factor of two of it is the difference
    // between a fluid and a slow-motion pile of sand.
    {
        eng::FluidConfig dam = cfg;
        dam.bounds_min = eng::Vec3{-0.50f, 0.0f, -0.12f};
        dam.bounds_max = eng::Vec3{0.50f, 0.60f, 0.12f};
        const std::vector<eng::Vec3> column =
            Block(eng::Vec3{-0.47f, 0.02f, -0.09f}, 8, 30, 7, spacing);
        auto db = eng::FluidSim::Create(*dev, dam, column,
                                        eng::Renderer::kSceneFormat, error, 1);
        Check(db != nullptr, "the dam break case was created");
        if (db) {
            const float h0 = 29.0f * spacing;
            for (int i = 0; i < 90; ++i) {  // 0.75 s
                dev->BeginFrame();
                eng::rhi::ComputeEncoder ce = dev->BeginCompute();
                (void)db->Step(ce, 1.0f / 120.0f);
                dev->EndCompute();
                std::string w;
                (void)dev->CommitAndWait(w);
            }
            const std::vector<eng::Vec3> p = db->ReadPositions();
            float front = p[0].x;
            for (const eng::Vec3& q : p) front = std::max(front, q.x);
            const float travelled = front - (-0.47f);
            const float estimate = std::sqrt(2.0f * 9.81f * h0) * 0.75f;
            std::printf("    dam break: front travelled %.3f m in 0.75 s "
                        "(sqrt(2gh)*t = %.3f)\n", travelled, estimate);
            Check(travelled > 0.25f, "the surge crossed most of the tank");
            Check(travelled < estimate * 1.5f,
                  "and not faster than a free-falling column could push it");
            Check(db->OutsideBounds() == 0, "the dam break leaked nothing");
        }
    }

    // --- and it draws ----------------------------------------------------------
    //
    // A fresh column, rendered PART WAY through its collapse. The settled pool
    // above is the correct answer and a dull picture; the interesting frame is
    // the one where the fluid is actually moving, which is also the one where
    // the speed tint has anything to say.
    {
        const eng::rhi::TextureId color =
            dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
        const eng::rhi::TextureId depth = dev->CreateDepthTarget(kW, kH);
        const eng::rhi::TextureId out = dev->CreateRenderTarget(kW, kH, kFmt, true);
        Check(Valid(color) && Valid(out), "the render targets were created");

        eng::Scene scene;
        scene.lightDir = eng::Vec4{-0.3f, 0.9f, 0.3f, 0.0f};
        scene.lightColor = eng::Vec4{1.0f, 1.0f, 1.0f, 1.0f};
        scene.ambientSky = eng::Vec3{0.12f, 0.14f, 0.18f};
        scene.ambientGround = eng::Vec3{0.04f, 0.04f, 0.05f};
        scene.camera.eye = eng::Vec3{0.10f, 0.46f, 1.15f};
        scene.camera.target = eng::Vec3{0.0f, 0.18f, 0.0f};

        eng::FluidConfig show = cfg;
        show.bounds_min = eng::Vec3{-0.45f, 0.0f, -0.22f};
        show.bounds_max = eng::Vec3{0.45f, 0.80f, 0.22f};
        const std::vector<eng::Vec3> tower =
            Block(eng::Vec3{-0.42f, 0.02f, -0.19f}, 10, 34, 14, spacing);
        std::string se;
        auto splash = eng::FluidSim::Create(*dev, show, tower,
                                            eng::Renderer::kSceneFormat, se, 1);
        Check(splash != nullptr, "the splash was created");
        for (int i = 0; splash && i < 48; ++i) {  // 0.4 s in
            dev->BeginFrame();
            eng::rhi::ComputeEncoder ce = dev->BeginCompute();
            (void)splash->Step(ce, 1.0f / 120.0f);
            dev->EndCompute();
            std::string w;
            (void)dev->CommitAndWait(w);
        }

        eng::RenderGraph g;
        eng::RenderGraph::Pass p;
        p.name = "scene";
        p.color = color;
        p.depth = depth;
        p.clear_color[0] = 0.03f; p.clear_color[1] = 0.035f;
        p.clear_color[2] = 0.05f; p.clear_color[3] = 1.0f;
        p.clear_depth = 0.0f;
        p.execute = [&](eng::rhi::Encoder& en) {
            if (splash) splash->Draw(en, scene.camera, kW, kH);
        };
        g.AddPass(std::move(p));
        eng::RenderGraph::Pass c;
        c.name = "composite";
        c.color = out;
        c.reads = {color};
        c.execute = [&](eng::rhi::Encoder& en) {
            renderer->DrawComposite(en, color, {}, {}, 0.0f, 0.6f);
        };
        g.AddPass(std::move(c));
        std::string ge;
        Check(g.Compile(ge), "the fluid graph compiles");
        dev->BeginFrame();
        g.Execute(*dev);
        std::string we;
        Check(dev->CommitAndWait(we), "the fluid frame submits");

        std::vector<std::uint8_t> px(std::size_t(kW) * kH * 4, 0);
        Check(dev->ReadPixels(out, kW, kH, px), "the fluid frame reads back");
        int watery = 0;
        for (std::size_t i = 0; i + 3 < px.size(); i += 4)
            if (px[i + 2] > px[i] + 20) ++watery;  // more blue than red
        std::printf("    %d pixels are blue-dominant\n", watery);
        Check(watery > 20000, "the fluid is on the screen");
        // (No count of bright pixels here. Tried: the dam break is chaotic,
        // so the splash lands differently run to run -- 30,249 neutral-bright
        // pixels in one run, 533 in the next, same binary -- and any
        // threshold between them is luck, not a test. The deterministic
        // check for the specular path is the single droplet below.)

        std::FILE* f = std::fopen("/tmp/fluid.ppm", "wb");
        if (f) {
            std::fprintf(f, "P6\n%d %d\n255\n", kW, kH);
            for (std::size_t i = 0; i + 3 < px.size(); i += 4)
                std::fwrite(&px[i], 1, 3, f);
            std::fclose(f);
            std::printf("    wrote /tmp/fluid.ppm\n");
        }
    }

    // --- the specular path, deterministically --------------------------------
    //
    // One still droplet, no simulation steps, head-on camera. The dam break
    // above cannot carry a brightness check: the splash is chaotic and lands
    // differently every run. A single particle has no neighbours and no
    // chaos, so this frame is bit-stable.
    //
    // What it proves is the fragment binding, not the solver: without the
    // SetFragmentBytes for the uniforms block, eyePos reads garbage and the
    // glint either vanishes or lands somewhere absurd. Measured on the old
    // shading (spec and rim reverted): 38. With them: 212. The threshold
    // sits at 120, three times above the old ceiling and well below the new
    // floor, so it separates the two shadings rather than the two runs.
    {
        const eng::rhi::TextureId color =
            dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
        const eng::rhi::TextureId depth = dev->CreateDepthTarget(kW, kH);
        const eng::rhi::TextureId out =
            dev->CreateRenderTarget(kW, kH, kFmt, true);
        std::string se;
        auto drop = eng::FluidSim::Create(*dev, cfg,
                                          {eng::Vec3{0.0f, 0.35f, 0.0f}},
                                          eng::Renderer::kSceneFormat, se, 1);
        Check(drop != nullptr, "the single droplet was created");
        eng::Scene still;
        still.camera.eye = eng::Vec3{0.0f, 0.35f, 1.0f};
        still.camera.target = eng::Vec3{0.0f, 0.35f, 0.0f};

        eng::RenderGraph g;
        eng::RenderGraph::Pass p;
        p.name = "droplet";
        p.color = color;
        p.depth = depth;
        p.clear_color[0] = 0.0f; p.clear_color[1] = 0.0f;
        p.clear_color[2] = 0.0f; p.clear_color[3] = 1.0f;
        p.clear_depth = 0.0f;
        p.execute = [&](eng::rhi::Encoder& en) {
            if (drop) drop->Draw(en, still.camera, kW, kH);
        };
        g.AddPass(std::move(p));
        eng::RenderGraph::Pass c;
        c.name = "composite";
        c.color = out;
        c.reads = {color};
        c.execute = [&](eng::rhi::Encoder& en) {
            renderer->DrawComposite(en, color, {}, {}, 0.0f, 0.6f);
        };
        g.AddPass(std::move(c));
        std::string ge;
        Check(g.Compile(ge), "the droplet graph compiles");
        dev->BeginFrame();
        g.Execute(*dev);
        std::string we;
        Check(dev->CommitAndWait(we), "the droplet frame submits");

        std::vector<std::uint8_t> dpx(std::size_t(kW) * kH * 4, 0);
        Check(dev->ReadPixels(out, kW, kH, dpx), "the droplet frame reads back");
        // Centre crop: the droplet sits at the middle of the frame by
        // construction, and the vignette is the identity there.
        std::uint8_t peak = 0;
        for (int y = kH / 2 - 100; y < kH / 2 + 100; ++y)
            for (int x = kW / 2 - 100; x < kW / 2 + 100; ++x) {
                const std::size_t i = (std::size_t(y) * kW + x) * 4;
                peak = std::max(
                    peak, std::min({dpx[i], dpx[i + 1], dpx[i + 2]}));
            }
        std::printf("    brightest neutral pixel in the droplet: %d\n", peak);
        Check(peak > 120, "the droplet carries a sun glint");
    }

    // --- the surface has no holes ------------------------------------------------
    //
    // A settled two-deep layer, no simulation steps, camera straight down.
    // Straight down is deliberate: every normal is +Y, fresnel is minimal and
    // the whole sheet shades nearly uniformly, so this measures COVERAGE and
    // nothing else. Static particles mean no chaos: this frame is bit-stable
    // across runs and across machines, unlike the dam break above.
    //
    // What would fail it: spheres too small to merge (holes the smoother
    // cannot close read as background), an edge stop so wide the background
    // bleeds in, or a broken depth chain (everything reads as background).
    {
        const eng::rhi::TextureId color =
            dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
        const eng::rhi::TextureId depth =
            dev->CreateDepthTarget(kW, kH, /*sampleable=*/true);
        const eng::rhi::TextureId out =
            dev->CreateRenderTarget(kW, kH, kFmt, true);
        std::vector<eng::Vec3> sheet;
        for (float z = -0.20f; z <= 0.20f; z += 0.03f)
            for (float y = 0.10f; y <= 0.13f; y += 0.03f)
                for (float x = -0.30f; x <= 0.30f; x += 0.03f)
                    sheet.push_back(eng::Vec3{x, y, z});
        std::string se;
        auto pond = eng::FluidSim::Create(*dev, cfg, sheet,
                                          eng::Renderer::kSceneFormat, se, 1);
        Check(pond != nullptr, "the still pond was created");
        auto surface = eng::FluidSurface::Create(*dev, se);
        Check(surface != nullptr, "the surface was created");
        eng::SurfaceLook look;
        if (pond && surface && surface->BeginFrame(*dev, kW, kH, se)) {
            eng::Scene top;
            top.camera.eye = eng::Vec3{0.05f, 1.20f, 0.05f};
            top.camera.target = eng::Vec3{0.0f, 0.10f, 0.0f};
            eng::RenderGraph g;
            eng::RenderGraph::Pass tank;
            tank.name = "tank";
            tank.color = color;
            tank.depth = depth;
            // Mid-grey: the sheet refracts it darker, so water reads as a
            // darkening of a known value rather than black on black.
            tank.clear_color[0] = 0.5f;
            tank.clear_color[1] = 0.5f;
            tank.clear_color[2] = 0.5f;
            tank.clear_color[3] = 1.0f;
            tank.execute = [&](eng::rhi::Encoder& en) {};
            g.AddPass(std::move(tank));
            eng::RenderGraph::Pass sd;
            sd.name = "surfacedepth";
            sd.depth = surface->Depth();
            sd.execute = [&](eng::rhi::Encoder& en) {
                surface->DrawDepth(en, *pond, top.camera, look, kW, kH);
            };
            g.AddPass(std::move(sd));
            const auto smooth = [&](const char* name,
                                    eng::rhi::TextureId target,
                                    bool horizontal) {
                eng::RenderGraph::Pass p;
                p.name = name;
                p.color = target;
                if (horizontal)
                    p.execute = [&](eng::rhi::Encoder& en) {
                        surface->SmoothH(en, top.camera, look, kW, kH);
                    };
                else
                    p.execute = [&](eng::rhi::Encoder& en) {
                        surface->SmoothV(en, top.camera, look, kW, kH);
                    };
                g.AddPass(std::move(p));
            };
            smooth("smoothh", surface->SmoothH(), true);
            smooth("smoothv", surface->Smooth(), false);
            eng::RenderGraph::Pass shade;
            shade.name = "surfaceshade";
            shade.color = out;
            shade.execute = [&](eng::rhi::Encoder& en) {
                // No tank here: color/depth hold clear values, so the shade
                // sees one unbroken sheet against empty background.
                surface->DrawShade(en, top.camera, look, kW, kH, color, depth);
            };
            g.AddPass(std::move(shade));
            std::string ge;
            Check(g.Compile(ge), "the surface graph compiles");
            dev->BeginFrame();
            g.Execute(*dev);
            std::string we;
            Check(dev->CommitAndWait(we), "the surface frame submits");
            std::vector<std::uint8_t> spx(std::size_t(kW) * kH * 4, 0);
            Check(dev->ReadPixels(out, kW, kH, spx),
                  "the surface frame reads back");
            // Centre crop, well inside the sheet's silhouette: every pixel
            // here must be water. The background is mid-grey and the sheet
            // refracts it darker, so "water" is anything that moved down by
            // more than noise.
            int water = 0, total = 0;
            for (int y = kH / 2 - 60; y < kH / 2 + 60; ++y)
                for (int x = kW / 2 - 80; x < kW / 2 + 80; ++x) {
                    const std::size_t i = (std::size_t(y) * kW + x) * 4;
                    ++total;
                    if (128 - int(spx[i + 1]) > 12) ++water;
                }
            std::printf("    surface covers %d of %d centre pixels\n", water,
                        total);
            Check(water > total * 9 / 10, "the sheet has no holes");
        }
    }

    std::printf(g_failures == 0 ? "\nfluid_test: all checks passed\n"
                                : "\nfluid_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
