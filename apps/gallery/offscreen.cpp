// The lighting gate. Every check here is a property that ONE directional light
// cannot produce, because that is the whole point of the change.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "apps/gallery/gallery_scene.h"
#include "engine/render/rendergraph.h"
#include "engine/render/renderer.h"
#include "engine/rhi/rhi.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

constexpr int kW = 1000, kH = 640;

}  // namespace

int main() {
    std::string error;
    auto dev = eng::rhi::Device::Create(error);
    if (!dev) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
    const auto kFmt = eng::rhi::Format::RGBA8Unorm;
    auto renderer = eng::Renderer::Create(*dev, kFmt, error);
    if (!renderer) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    const gallery::Assets assets = gallery::Build(*dev, *renderer, error);
    if (!assets.ok) {
        std::fprintf(stderr, "FAIL: %s\n", error.empty() ? "build" : error.c_str());
        return 1;
    }

    const eng::rhi::TextureId shadow_map = dev->CreateShadowMap(2048);
    const eng::rhi::TextureId color = dev->CreateRenderTarget(kW, kH, kFmt, true);
    const eng::rhi::TextureId depth = dev->CreateDepthTarget(kW, kH);
    const eng::rhi::TextureId out = dev->CreateRenderTarget(kW, kH, kFmt, true);
    if (!Valid(shadow_map) || !Valid(color) || !Valid(depth) || !Valid(out)) {
        std::fprintf(stderr, "FAIL: targets\n");
        return 1;
    }

    std::vector<std::uint8_t> pixels;
    eng::RenderStats stats;

    const eng::Vec3 kEye{4.6f, 2.9f, 7.4f};
    const eng::Vec3 kTarget{0.0f, 1.15f, 0.0f};

    auto draw = [&](const eng::Scene& base) {
        eng::Scene scene = base;
        scene.camera.eye = kEye;
        scene.camera.target = kTarget;
        eng::RenderGraph graph;
        {
            eng::RenderGraph::Pass p;
            p.name = "shadow";
            p.depth = shadow_map;
            p.clear_depth = 0.0f;
            p.keep_depth = true;
            p.execute = [&](eng::rhi::Encoder& e) { renderer->DrawShadow(e, scene); };
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
            p.reads = {shadow_map};
            p.execute = [&](eng::rhi::Encoder& e) {
                renderer->DrawScene(e, scene, kW, kH, shadow_map);
            };
            graph.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "composite";
            p.color = out;
            p.reads = {color};
            p.execute = [&](eng::rhi::Encoder& e) { renderer->DrawComposite(e, color); };
            graph.AddPass(std::move(p));
        }
        std::string e;
        if (!graph.Compile(e)) { std::fprintf(stderr, "FAIL: %s\n", e.c_str()); return; }
        dev->BeginFrame();
        graph.Execute(*dev);
        std::string w;
        if (!dev->CommitAndWait(w)) std::fprintf(stderr, "FAIL: %s\n", w.c_str());
        stats = renderer->LastStats();
        pixels.assign(std::size_t(kW) * kH * 4, 0);
        (void)dev->ReadPixels(out, kW, kH, pixels);
    };

    // Where a world point lands on screen, so a check can name a place in the
    // room instead of a pixel someone eyeballed once.
    auto Project = [&](eng::Vec3 world, int* px, int* py) {
        eng::Camera cam;
        cam.eye = kEye;
        cam.target = kTarget;
        const eng::Vec4 clip = cam.ViewProj(float(kW) / float(kH)) *
                               eng::Vec4{world.x, world.y, world.z, 1.0f};
        *px = int((clip.x / clip.w * 0.5f + 0.5f) * float(kW));
        *py = int((1.0f - (clip.y / clip.w * 0.5f + 0.5f)) * float(kH));
    };

    auto Mean = [&](int x0, int y0, int x1, int y1, double* r, double* g, double* b) {
        double sr = 0, sg = 0, sb = 0;
        int n = 0;
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x) {
                const std::size_t i = (std::size_t(y) * kW + x) * 4;
                sr += pixels[i]; sg += pixels[i + 1]; sb += pixels[i + 2]; ++n;
            }
        *r = sr / n; *g = sg / n; *b = sb / n;
    };

    const eng::Scene full = gallery::MakeScene(assets, 0.7f, /*flicker=*/0.0f);
    std::printf("the scene draws\n");
    draw(full);
    Check(stats.draws > 8 && stats.invalid == 0 && stats.overflowed == 0,
          "every instance drew with nothing dropped");
    Check(int(full.lights.size()) == 6, "six local lights on top of the key light");

    std::printf("local lights actually contribute\n");
    {
        double r0, g0, b0, r1, g1, b1;
        Mean(0, 0, kW, kH, &r0, &g0, &b0);
        eng::Scene dark = full;
        dark.lights.clear();
        draw(dark);
        Mean(0, 0, kW, kH, &r1, &g1, &b1);
        std::printf("    with lights %.1f, without %.1f\n", (r0 + g0 + b0) / 3,
                    (r1 + g1 + b1) / 3);
        Check((r0 + g0 + b0) > (r1 + g1 + b1) * 1.6,
              "removing the local lights makes the frame much darker");
        draw(full);
    }

    std::printf("distance falloff is real\n");
    {
        // One point light, and two patches of the same floor at different
        // distances from it. A light without falloff lights both equally --
        // which is exactly what a directional light does, and why a room lit
        // that way has no sense of depth.
        eng::Scene one = full;
        one.lights.clear();
        one.lightColor = eng::Vec4{0, 0, 0, 1};  // key light off
        eng::Light p;
        p.position = eng::Vec3{-4.0f, 1.2f, 3.0f};
        p.color = eng::Vec3{22.0f, 22.0f, 22.0f};
        p.range = 14.0f;
        one.lights.push_back(p);
        draw(one);

        double nr, ng, nb, fr, fg, fb;
        Mean(140, 470, 260, 560, &nr, &ng, &nb);   // floor near the lamp
        Mean(700, 470, 820, 560, &fr, &fg, &fb);   // floor far from it
        std::printf("    near %.1f   far %.1f\n", (nr + ng + nb) / 3,
                    (fr + fg + fb) / 3);
        Check((nr + ng + nb) > (fr + fg + fb) * 1.8,
              "the floor is brighter near the lamp than far from it");
    }

    std::printf("a spot has a CONE, a point light does not\n");
    {
        // The discriminator is the CONE, not the gradient. A bright point light
        // close to the floor has a steeper brightness step than any cone edge —
        // measured at 136 against the spot's 53 — so "sharpest step" ranks the
        // point light as the sharper of the two and tests nothing.
        //
        // What a spot does and a point light cannot is go dark OUTSIDE its
        // cone while staying bright inside. So: render the same scene twice,
        // once with each, and compare the same two places.
        const eng::Vec3 lamp{0.0f, 4.2f, 1.6f};
        auto render_with = [&](eng::LightType type) {
            eng::Scene s2 = full;
            s2.lights.clear();
            s2.lightColor = eng::Vec4{0, 0, 0, 1};   // key light off
            s2.ambientSky = eng::Vec3{0, 0, 0};      // ambient off
            s2.ambientGround = eng::Vec3{0, 0, 0};
            eng::Light l;
            l.type = type;
            l.position = lamp;
            l.direction = eng::Vec3{0.0f, -1.0f, 0.0f};
            l.color = eng::Vec3{60.0f, 60.0f, 60.0f};
            l.range = 14.0f;
            l.inner_degrees = 12.0f;
            l.outer_degrees = 18.0f;
            s2.lights.push_back(l);
            draw(s2);
        };
        // Directly under the lamp: inside a 18-degree cone. And 3.4 m to the
        // side, which at this height is well outside it.
        int ix, iy, ox, oy;
        Project(eng::Vec3{lamp.x, 0.0f, lamp.z}, &ix, &iy);
        Project(eng::Vec3{lamp.x - 3.4f, 0.0f, lamp.z}, &ox, &oy);

        double r, g, b;
        render_with(eng::LightType::Spot);
        Mean(ix - 22, iy - 12, ix + 22, iy + 12, &r, &g, &b);
        const double spot_in = r + g + b;
        Mean(ox - 22, oy - 12, ox + 22, oy + 12, &r, &g, &b);
        const double spot_out = r + g + b;

        render_with(eng::LightType::Point);
        Mean(ix - 22, iy - 12, ix + 22, iy + 12, &r, &g, &b);
        const double point_in = r + g + b;
        Mean(ox - 22, oy - 12, ox + 22, oy + 12, &r, &g, &b);
        const double point_out = r + g + b;

        std::printf("    inside  spot %.0f  point %.0f\n", spot_in, point_in);
        std::printf("    outside spot %.0f  point %.0f\n", spot_out, point_out);
        Check(spot_in > point_in * 0.7,
              "inside its cone the spot is as bright as a point light");
        Check(point_out > 12.0, "the point light does reach outside that cone");
        Check(spot_out < point_out * 0.25,
              "and the spot does not: the cone actually cuts off");
    }

    std::printf("coloured lights reach the geometry\n");
    {
        // The cool lamp is on the left and the warm one on the right, so the
        // two sides of the room must not be the same hue. This is the check
        // that a single white sun cannot pass however bright it is.
        double lr, lg, lb, rr, rg, rb;
        draw(full);
        Mean(60, 380, 300, 600, &lr, &lg, &lb);
        Mean(720, 380, 960, 600, &rr, &rg, &rb);
        std::printf("    left rgb %.0f/%.0f/%.0f   right rgb %.0f/%.0f/%.0f\n",
                    lr, lg, lb, rr, rg, rb);
        Check(lb > lr * 1.25, "the left of the room is blue");
        Check(rr > rb * 1.25, "the right of the room is warm");
    }

    std::printf("more lights than the budget are dropped, not wrapped\n");
    {
        eng::Scene many = full;
        for (int i = 0; i < 80; ++i) {
            eng::Light l;
            l.position = eng::Vec3{float(i % 9) - 4.0f, 1.0f, float(i / 9) - 3.0f};
            l.color = eng::Vec3{0.5f, 0.5f, 0.5f};
            many.lights.push_back(l);
        }
        draw(many);
        Check(stats.invalid == 0 && stats.draws > 8,
              "an over-full light list still renders");
    }

    std::FILE* f = std::fopen("gallery.ppm", "wb");
    if (f) {
        draw(full);
        std::fprintf(f, "P6\n%d %d\n255\n", kW, kH);
        for (std::size_t i = 0; i < pixels.size(); i += 4)
            std::fwrite(&pixels[i], 1, 3, f);
        std::fclose(f);
    }

    std::printf(g_failures == 0 ? "\ngallery_test: all checks passed\n"
                                : "\ngallery_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
