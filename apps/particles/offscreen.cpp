// GPU particles, checked against physics rather than against a picture.
//
// A particle system is easy to test badly: render it, count the bright pixels,
// declare victory. That passes for a system whose particles never move, never
// die, and all sit on top of each other. The checks here are about the
// SIMULATION -- how many are alive, where their centre of mass is, and whether
// a ballistic arc matches the closed form -- because those are the things that
// are wrong when a particle system looks nearly right.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/geometry/mesh.h"
#include "engine/render/particles.h"
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

constexpr int kW = 900, kH = 560;
constexpr float kDt = 1.0f / 120.0f;

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("gpu particles\n");

    std::string error;
    auto dev = eng::rhi::Device::Create(error);
    if (!dev) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
    const auto kFmt = eng::rhi::Format::RGBA8Unorm;
    auto renderer = eng::Renderer::Create(*dev, kFmt, error, 1);
    if (!renderer) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    constexpr int kCapacity = 20000;
    auto ps = eng::ParticleSystem::Create(*dev, kCapacity,
                                          eng::Renderer::kSceneFormat, error, 1);
    if (!ps) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
    Check(ps->Capacity() == kCapacity, "the pool was created at the size asked for");

    // Runs `n` steps of the simulation, one compute pass each. No rendering:
    // these checks are about the simulation and a render would only add ways
    // for them to be wrong for an unrelated reason.
    const auto simulate = [&](const eng::ParticleEmitter& e, int n) {
        for (int i = 0; i < n; ++i) {
            dev->BeginFrame();
            eng::rhi::ComputeEncoder ce = dev->BeginCompute();
            ps->Step(ce, e, kDt);
            dev->EndCompute();
            std::string w;
            if (!dev->CommitAndWait(w)) std::fprintf(stderr, "  %s\n", w.c_str());
        }
    };

    // --- the pool starts empty and fills at the rate asked for ---------------
    {
        Check(ps->LiveCountSlow() == 0, "the pool starts empty");

        eng::ParticleEmitter e;
        e.rate = 1200.0f;
        e.lifetime = 1.0f;
        e.lifetime_variance = 0.0f;  // so the steady state is arithmetic
        e.gravity = eng::Vec3{0, 0, 0};
        e.drag = 0.0f;

        simulate(e, 60);  // half a second
        const int half = ps->LiveCountSlow();
        // 1200/s for 0.5 s, none dead yet because they live a whole second.
        std::printf("    after 0.5 s at 1200/s: %d alive (expected ~600)\n", half);
        Check(std::abs(half - 600) < 20, "the emission rate is what was asked for");

        // A second later everything is in steady state: births equal deaths, so
        // the population settles at rate * lifetime and stays there. This is
        // the check that catches a system that emits but never expires -- which
        // looks completely correct for the first second.
        simulate(e, 240);
        const int steady = ps->LiveCountSlow();
        std::printf("    steady state: %d alive (rate x lifetime = 1200)\n", steady);
        Check(std::abs(steady - 1200) < 40, "the population settles at rate x lifetime");

        // ...and stays there rather than creeping.
        simulate(e, 240);
        const int later = ps->LiveCountSlow();
        std::printf("    two seconds later: %d alive\n", later);
        Check(std::abs(later - steady) < 60, "and does not drift");
    }

    // --- the pool is a hard ceiling, not a suggestion -------------------------
    {
        eng::ParticleEmitter e;
        // Asking for four times what the pool can hold. It must saturate, not
        // overflow: a pool that wraps corrupts live particles, and one that
        // reallocates drops every particle on the frame it does it.
        e.rate = 400000.0f;
        e.lifetime = 0.2f;
        e.lifetime_variance = 0.0f;
        e.gravity = eng::Vec3{0, 0, 0};
        simulate(e, 120);
        const int n = ps->LiveCountSlow();
        std::printf("    asking for 400000/s into a pool of %d: %d alive\n",
                    kCapacity, n);
        Check(n <= kCapacity, "the pool never exceeds its capacity");
        Check(n > kCapacity / 2, "and does fill up rather than starving");
    }

    // --- ballistics ----------------------------------------------------------
    //
    // One burst, straight up, no drag and no spread, then let it fly. Every
    // particle follows the same arc and the centre of mass is that arc, so it
    // can be compared to the closed form: y = v0*t - g*t^2/2.
    {
        auto solo = eng::ParticleSystem::Create(*dev, 4096,
                                                eng::Renderer::kSceneFormat, error, 1);
        Check(solo != nullptr, "a second system for the ballistics check");
        if (solo) {
            eng::ParticleEmitter e;
            e.position = eng::Vec3{0, 0, 0};
            e.direction = eng::Vec3{0, 1, 0};
            e.spread = 0.0f;
            e.speed = 10.0f;
            e.speed_variance = 0.0f;
            e.lifetime = 100.0f;  // nothing expires during the measurement
            e.lifetime_variance = 0.0f;
            e.size_variance = 0.0f;
            e.gravity = eng::Vec3{0.0f, -9.81f, 0.0f};
            e.drag = 0.0f;
            e.rate = 24000.0f;  // one step's worth at 1/120 s is 200

            // One step to emit, then coast.
            dev->BeginFrame();
            {
                eng::rhi::ComputeEncoder ce = dev->BeginCompute();
                solo->Step(ce, e, kDt);
                dev->EndCompute();
            }
            std::string w;
            (void)dev->CommitAndWait(w);
            const int burst = solo->LiveCountSlow();
            std::printf("    one step emitted %d particles\n", burst);
            Check(burst > 150 && burst < 260, "a single step emits rate x dt");

            e.rate = 0.0f;  // no more emission
            const int steps = 60;
            for (int i = 0; i < steps; ++i) {
                dev->BeginFrame();
                eng::rhi::ComputeEncoder ce = dev->BeginCompute();
                solo->Step(ce, e, kDt);
                dev->EndCompute();
                std::string w2;
                (void)dev->CommitAndWait(w2);
            }
            const eng::Vec3 mean = solo->MeanPositionSlow();

            // Semi-implicit Euler over n steps of h: the closed form of the
            // discrete scheme, not of the continuous one. y = v0*n*h - g*h^2 *
            // n(n+1)/2. Comparing against the CONTINUOUS solution would be
            // comparing the integrator to a different integrator and leave a
            // gap that has to be papered over with a tolerance -- and a
            // tolerance wide enough to cover that is wide enough to hide a bug.
            // n = steps, NOT steps + 1. The step that emits a particle does
            // not also integrate it: the kernel respawns the slot and returns,
            // so the particle's first move happens on the following step. The
            // difference is one step out of sixty and shows up as 3.7533
            // against 3.7951 -- close enough to look like accumulated float
            // error and be waved away with a tolerance, which is exactly how an
            // off-by-one in an integrator survives.
            const float h = kDt, g = 9.81f, v0 = 10.0f;
            const float n = float(steps);
            const float want = v0 * n * h - g * h * h * n * (n + 1.0f) * 0.5f;
            std::printf("    after %d integrating steps: y = %.5f, the scheme "
                        "says %.5f\n", steps, mean.y, want);
            Check(std::fabs(mean.y - want) < 1e-3f,
                  "the arc matches semi-implicit Euler exactly");
            Check(std::fabs(mean.x) < 1e-4f && std::fabs(mean.z) < 1e-4f,
                  "and a zero-spread emitter does not wander sideways");

            // Gravity DOWN. Sign errors here produce a fountain that falls up,
            // which is obvious in a picture and invisible in a mean position
            // unless something checks it.
            Check(want < v0 * n * h, "gravity subtracts from the climb");
        }
    }

    // --- the cone --------------------------------------------------------------
    //
    // A wide emitter must actually spread. The failure this catches is sampling
    // the cone angle linearly instead of the spherical cap, which bunches
    // everything near the axis -- a 90-degree emitter that looks like a
    // 30-degree one with a faint halo.
    {
        auto cone = eng::ParticleSystem::Create(*dev, 8192,
                                                eng::Renderer::kSceneFormat, error, 1);
        if (cone) {
            eng::ParticleEmitter e;
            e.direction = eng::Vec3{0, 1, 0};
            e.spread = 1.5707963f;  // a full hemisphere
            e.speed = 5.0f;
            e.speed_variance = 0.0f;
            e.lifetime = 100.0f;
            e.lifetime_variance = 0.0f;
            e.gravity = eng::Vec3{0, 0, 0};
            e.drag = 0.0f;
            e.rate = 120000.0f;
            dev->BeginFrame();
            {
                eng::rhi::ComputeEncoder ce = dev->BeginCompute();
                cone->Step(ce, e, kDt);
                dev->EndCompute();
            }
            std::string w;
            (void)dev->CommitAndWait(w);
            for (int i = 0; i < 60; ++i) {
                e.rate = 0.0f;
                dev->BeginFrame();
                eng::rhi::ComputeEncoder ce = dev->BeginCompute();
                cone->Step(ce, e, kDt);
                dev->EndCompute();
                std::string w2;
                (void)dev->CommitAndWait(w2);
            }
            const eng::Vec3 mean = cone->MeanPositionSlow();
            const int live = cone->LiveCountSlow();
            // Uniform over a hemisphere of radius R, the mean height is R/2.
            // Bunched at the axis it would be close to R.
            const float R = 5.0f * kDt * 61.0f;
            std::printf("    %d particles on a hemisphere of radius %.4f: mean "
                        "height %.4f (uniform gives %.4f)\n", live, R, mean.y, R * 0.5f);
            Check(live > 500, "the hemisphere emitter emitted");
            Check(std::fabs(mean.y - R * 0.5f) < R * 0.08f,
                  "the cone is sampled uniformly, not bunched at its axis");
            Check(std::fabs(mean.x) < R * 0.05f && std::fabs(mean.z) < R * 0.05f,
                  "and is symmetric about its axis");
        }
    }

    // --- the draw path ---------------------------------------------------------
    //
    // Everything above is simulation. This checks the other half: that the pool
    // actually reaches the screen, that dead particles cost nothing, and that
    // additive blending accumulates rather than replacing.
    {
        std::printf("drawing\n");
        const eng::rhi::TextureId color =
            dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
        const eng::rhi::TextureId depth = dev->CreateDepthTarget(kW, kH, true);
        const eng::rhi::TextureId out = dev->CreateRenderTarget(kW, kH, kFmt, true);
        Check(Valid(color) && Valid(depth) && Valid(out), "the targets were created");

        // A floor, so the particles have something to be in front of and the
        // soft-particle fade has something to fade against.
        const eng::MeshHandle floor = renderer->UploadMesh(
            eng::MakeBox(eng::Vec3{14.0f, 0.2f, 14.0f}, eng::Vec4{1, 1, 1, 1}));
        eng::MaterialDesc md;
        md.shading = eng::Shading::Lit;
        md.base_color = eng::Vec4{0.16f, 0.17f, 0.20f, 1.0f};
        md.roughness = 0.75f;
        const eng::MaterialHandle floor_mat = renderer->CreateMaterial(md, error);

        eng::Scene scene;
        scene.lightDir = eng::Vec4{-0.35f, 0.86f, 0.37f, 0.0f};
        scene.lightColor = eng::Vec4{0.55f, 0.57f, 0.65f, 1.0f};
        scene.ambientSky = eng::Vec3{0.05f, 0.06f, 0.09f};
        scene.ambientGround = eng::Vec3{0.02f, 0.02f, 0.02f};
        scene.camera.eye = eng::Vec3{0.0f, 2.4f, 7.5f};
        scene.camera.target = eng::Vec3{0.0f, 1.6f, 0.0f};
        {
            eng::Instance f;
            f.mesh = floor;
            f.material = floor_mat;
            f.model = eng::Mat4::Translation(eng::Vec3{0.0f, -0.2f, 0.0f});
            scene.instances.push_back(f);
        }

        auto fountain = eng::ParticleSystem::Create(*dev, 40000,
                                                    eng::Renderer::kSceneFormat,
                                                    error, 1);
        Check(fountain != nullptr, "the fountain was created");

        eng::ParticleEmitter e;
        e.position = eng::Vec3{0.0f, 0.15f, 0.0f};
        e.direction = eng::Vec3{0.0f, 1.0f, 0.0f};
        e.spread = 0.30f;
        e.speed = 6.2f;
        e.speed_variance = 1.4f;
        e.lifetime = 2.1f;
        e.lifetime_variance = 0.6f;
        e.size = 0.045f;
        e.size_variance = 0.02f;
        e.rate = 9000.0f;
        // Above 1 so it blooms: the composite tone maps, and a fountain that
        // peaks at exactly white has no highlight at all.
        e.color = eng::Vec4{2.6f, 1.15f, 0.35f, 1.0f};
        e.gravity = eng::Vec3{0.0f, -9.81f, 0.0f};
        e.drag = 0.25f;

        const auto frame = [&](bool draw_particles, std::vector<std::uint8_t>* px) {
            dev->BeginFrame();
            if (fountain) {
                eng::rhi::ComputeEncoder ce = dev->BeginCompute();
                fountain->Step(ce, e, kDt);
                dev->EndCompute();
            }
            eng::RenderGraph g;
            {
                eng::RenderGraph::Pass p;
                p.name = "scene";
                p.color = color;
                p.depth = depth;
                p.clear_color[0] = 0.015f; p.clear_color[1] = 0.017f;
                p.clear_color[2] = 0.024f; p.clear_color[3] = 1.0f;
                p.clear_depth = 0.0f;
                p.keep_depth = true;
                p.execute = [&](eng::rhi::Encoder& en) {
                    renderer->DrawScene(en, scene, kW, kH);
                    // The particles go in the SAME pass, after the opaque
                    // geometry. A separate pass would need the depth buffer
                    // attached again and would resolve nothing extra -- and
                    // they must be after, because they test against that depth.
                    if (draw_particles && fountain)
                        fountain->Draw(en, scene.camera, kW, kH);
                };
                g.AddPass(std::move(p));
            }
            {
                eng::RenderGraph::Pass p;
                p.name = "composite";
                p.color = out;
                p.reads = {color};
                p.execute = [&](eng::rhi::Encoder& en) {
                    renderer->DrawComposite(en, color, {}, {}, 0.0f, 0.85f);
                };
                g.AddPass(std::move(p));
            }
            std::string ge;
            if (!g.Compile(ge)) std::fprintf(stderr, "  graph: %s\n", ge.c_str());
            g.Execute(*dev);
            std::string we;
            if (!dev->CommitAndWait(we)) std::fprintf(stderr, "  %s\n", we.c_str());
            if (px) {
                px->assign(std::size_t(kW) * kH * 4, 0);
                (void)dev->ReadPixels(out, kW, kH, *px);
            }
        };

        // Let it reach steady state before measuring.
        for (int i = 0; i < 200; ++i) frame(true, nullptr);

        std::vector<std::uint8_t> with, without;
        frame(true, &with);
        frame(false, &without);
        if (fountain)
            std::printf("    %d particles alive at steady state\n",
                        fountain->LiveCountSlow());

        // Where the fountain is, in pixels, so the checks name a place in the
        // scene rather than a pixel someone eyeballed.
        const auto project = [&](eng::Vec3 w, int* px, int* py) {
            const eng::Vec4 clip = scene.camera.ViewProj(float(kW) / float(kH)) *
                                   eng::Vec4{w.x, w.y, w.z, 1.0f};
            *px = int((clip.x / clip.w * 0.5f + 0.5f) * float(kW));
            *py = int((1.0f - (clip.y / clip.w * 0.5f + 0.5f)) * float(kH));
        };
        const auto brightness_at = [&](const std::vector<std::uint8_t>& img,
                                       eng::Vec3 world, int r) {
            int cx = 0, cy = 0;
            project(world, &cx, &cy);
            long long sum = 0;
            int n = 0;
            for (int y = std::max(0, cy - r); y < std::min(kH, cy + r); ++y)
                for (int x = std::max(0, cx - r); x < std::min(kW, cx + r); ++x) {
                    const std::size_t i = (std::size_t(y) * kW + x) * 4;
                    sum += img[i] + img[i + 1] + img[i + 2];
                    ++n;
                }
            return n > 0 ? double(sum) / n : 0.0;
        };

        // The column of the fountain is brighter with particles than without.
        const double col_on = brightness_at(with, eng::Vec3{0, 1.6f, 0}, 40);
        const double col_off = brightness_at(without, eng::Vec3{0, 1.6f, 0}, 40);
        // ...and a patch of floor well off to the side is NOT, which is what
        // separates "the particles drew" from "something brightened the frame".
        const double side_on = brightness_at(with, eng::Vec3{5.5f, 0.0f, 2.0f}, 24);
        const double side_off = brightness_at(without, eng::Vec3{5.5f, 0.0f, 2.0f}, 24);
        std::printf("    fountain column %.1f -> %.1f, floor off to the side "
                    "%.1f -> %.1f\n", col_off, col_on, side_off, side_on);
        Check(col_on > col_off * 2.0 + 20.0, "the particles brighten the fountain");
        Check(std::fabs(side_on - side_off) < 3.0,
              "and leave the rest of the frame alone");

        // ADDITIVE, not replacing -- tested by the one comparison that needs no
        // arithmetic about the tone curve.
        //
        // Stack N identical sprites in exactly the same place. Under ALPHA
        // blending the result is the same colour as one of them: src*a +
        // dst*(1-a) with the same src converges to src and stays there. Under
        // ADDITIVE it keeps climbing. So "eight in one spot is brighter than
        // one" is true for additive and false for alpha, whatever the tone
        // mapper does afterwards.
        //
        // Two earlier versions of this check tried to measure the RATIO of
        // light against particle count and both were measuring something else:
        // the first sampled the fountain's core, which is clamped to white at
        // 708 of 765 and cannot get brighter; the second used the whole frame
        // but tripled the emission rate past what the pool can hold, so the
        // population went up 2.1x rather than 3x -- and displayed brightness is
        // a gamma-compressed function of light in any case.
        {
            eng::ParticleEmitter se;
            se.position = eng::Vec3{0.0f, 1.6f, 0.0f};
            se.spread = 0.0f;
            se.speed = 0.0f;
            se.speed_variance = 0.0f;
            // Short enough that the sprites AGE. A particle fades in over the
            // first tenth of its life -- deliberately, because appearing at
            // full brightness is the most obvious tell a particle system has --
            // so one drawn on the step it was born is exactly invisible. With a
            // lifetime of 1000 s it stays invisible for a hundred seconds, and
            // the first version of this test measured a black screen and
            // concluded that additive blending did not work.
            se.lifetime = 2.0f;
            se.lifetime_variance = 0.0f;
            se.size = 0.5f;
            se.size_variance = 0.0f;
            se.gravity = eng::Vec3{0, 0, 0};
            se.drag = 0.0f;
            // Dim, so that even eight stacked stay below the point where the
            // tone map flattens and the comparison stops being able to move.
            se.color = eng::Vec4{0.05f, 0.05f, 0.05f, 1.0f};

            const auto stacked = [&](int n) {
                // A fresh pool each time, so the count is exactly n and not n
                // plus whatever the previous call left alive.
                auto stack = eng::ParticleSystem::Create(
                    *dev, 64, eng::Renderer::kSceneFormat, error, 1);
                if (!stack) return 0.0;
                se.rate = float(n) / kDt;  // exactly n in one step
                dev->BeginFrame();
                {
                    eng::rhi::ComputeEncoder ce = dev->BeginCompute();
                    stack->Step(ce, se, kDt);
                    dev->EndCompute();
                }
                std::string w0;
                (void)dev->CommitAndWait(w0);
                // Age them past the fade-in, emitting nothing further.
                se.rate = 0.0f;
                for (int k = 0; k < 30; ++k) {
                    dev->BeginFrame();
                    eng::rhi::ComputeEncoder ce = dev->BeginCompute();
                    stack->Step(ce, se, kDt);
                    dev->EndCompute();
                    std::string wk;
                    (void)dev->CommitAndWait(wk);
                }
                dev->BeginFrame();
                eng::RenderGraph g;
                eng::RenderGraph::Pass p;
                p.name = "scene";
                p.color = color;
                p.depth = depth;
                p.clear_color[0] = 0.0f; p.clear_color[1] = 0.0f;
                p.clear_color[2] = 0.0f; p.clear_color[3] = 1.0f;
                p.clear_depth = 0.0f;
                p.keep_depth = true;
                p.execute = [&](eng::rhi::Encoder& en) {
                    stack->Draw(en, scene.camera, kW, kH);
                };
                g.AddPass(std::move(p));
                eng::RenderGraph::Pass c;
                c.name = "composite";
                c.color = out;
                c.reads = {color};
                c.execute = [&](eng::rhi::Encoder& en) {
                    renderer->DrawComposite(en, color, {}, {}, 0.0f, 0.0f);
                };
                g.AddPass(std::move(c));
                std::string ge;
                if (!g.Compile(ge)) std::fprintf(stderr, "  graph: %s\n", ge.c_str());
                g.Execute(*dev);
                std::string we;
                (void)dev->CommitAndWait(we);
                std::vector<std::uint8_t> px(std::size_t(kW) * kH * 4, 0);
                (void)dev->ReadPixels(out, kW, kH, px);
                return brightness_at(px, eng::Vec3{0.0f, 1.6f, 0.0f}, 8);
            };

            const double one = stacked(1);
            const double eight = stacked(8);
            std::printf("    one sprite %.2f, eight in the same place %.2f "
                        "(alpha blending would give the same number twice)\n",
                        one, eight);
            Check(one > 1.0, "a single sprite is visible at all");
            Check(eight > one * 2.5, "eight stacked are much brighter than one");
        }

        // More particles put more light on the screen. Monotonic rather than
        // proportional, for the reasons above.
        const auto total_light = [](const std::vector<std::uint8_t>& img) {
            long long sum = 0;
            for (std::size_t i = 0; i + 3 < img.size(); i += 4)
                sum += img[i] + img[i + 1] + img[i + 2];
            return sum;
        };
        const float saved = e.rate;
        e.rate = saved * 3.0f;
        for (int i = 0; i < 200; ++i) frame(true, nullptr);
        std::vector<std::uint8_t> denser;
        frame(true, &denser);
        const long long base = total_light(without);
        std::printf("    light added: %lld at %.0f/s, %lld at %.0f/s\n",
                    total_light(with) - base, saved,
                    total_light(denser) - base, saved * 3.0f);
        Check(total_light(with) > base, "the particles add light");
        Check(total_light(denser) > total_light(with), "and more of them add more");
        e.rate = saved;

        // A picture to look at, alongside the numbers.
        std::FILE* f = std::fopen("/tmp/particles.ppm", "wb");
        if (f) {
            std::fprintf(f, "P6\n%d %d\n255\n", kW, kH);
            for (std::size_t i = 0; i + 3 < denser.size(); i += 4)
                std::fwrite(&denser[i], 1, 3, f);
            std::fclose(f);
            std::printf("    wrote /tmp/particles.ppm\n");
        }
    }

    std::printf(g_failures == 0 ? "\nparticles_test: all checks passed\n"
                                : "\nparticles_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
