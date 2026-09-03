// Pure C++20. Offscreen counterpart to the interactive material viewer.
//
// Renders the same five spheres and asserts that the materials are actually
// DISTINGUISHABLE. A demo whose five materials render identically still opens a
// window and still looks fine at a glance; the only thing that catches it is
// measuring the pixels each one produced.
#include "apps/viewer/materials_scene.h"
#include "engine/render/rendergraph.h"
#include "engine/rhi/rhi.h"

#include <cmath>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-54s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

double Luma(const std::vector<std::uint8_t>& p, std::size_t i) {
    return 0.2126 * p[i] + 0.7152 * p[i + 1] + 0.0722 * p[i + 2];
}

constexpr int kW = 1000, kH = 640;

}  // namespace

int main() {
    std::string error;
    auto dev = eng::rhi::Device::Create(error);
    if (!dev) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
    auto renderer = eng::Renderer::Create(*dev, eng::rhi::Format::RGBA8Unorm, error);
    if (!renderer) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    const demo::Assets assets = demo::Build(*renderer, *dev, error);
    if (!assets.ok) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    const auto kFmt = eng::rhi::Format::RGBA8Unorm;
    const eng::rhi::TextureId shadow_map = dev->CreateShadowMap(2048);
    const eng::rhi::TextureId scene_color = dev->CreateRenderTarget(kW, kH, kFmt, true);
    const eng::rhi::TextureId scene_depth = dev->CreateDepthTarget(kW, kH, true);
    const eng::rhi::TextureId ao_target = dev->CreateRenderTarget(kW, kH, kFmt, true);
    const eng::rhi::TextureId final_color = dev->CreateRenderTarget(kW, kH, kFmt, true);
    if (!Valid(shadow_map) || !Valid(scene_color) || !Valid(scene_depth) ||
        !Valid(ao_target) || !Valid(final_color)) {
        std::fprintf(stderr, "FAIL: targets\n");
        return 1;
    }

    eng::Camera camera;
    struct Result {
        std::vector<std::uint8_t> px;
        eng::RenderStats stats;
    };

    auto render = [&](float sun_azimuth, bool shadows) {
        Result out;
        eng::Scene scene = demo::MakeScene(assets, 0.0f, sun_azimuth, shadows);
        eng::OrbitController orbit;
        orbit.target = eng::Vec3{0.0f, 0.6f, 0.0f};
        orbit.distance = 12.0f;
        orbit.yaw = 1.1f;
        orbit.pitch = 0.30f;
        orbit.Apply(scene.camera);
        camera = scene.camera;

        eng::RenderGraph g;
        {
            eng::RenderGraph::Pass p;
            p.name = "composite";
            p.color = final_color;
            p.reads = {scene_color, ao_target};
            p.execute = [&](eng::rhi::Encoder& e) {
                renderer->DrawComposite(e, scene_color, ao_target);
            };
            g.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "ssao";
            p.color = ao_target;
            p.reads = {scene_depth};
            p.clear_color[0] = p.clear_color[1] = p.clear_color[2] = 1.0f;
            p.execute = [&](eng::rhi::Encoder& e) {
                renderer->DrawSsao(e, scene.camera, kW, kH, scene_depth);
            };
            g.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "scene";
            p.color = scene_color;
            p.depth = scene_depth;
            for (int i = 0; i < 4; ++i) p.clear_color[i] = eng::kClearColor[i];
            p.clear_depth = 0.0f;
            p.keep_depth = true;
            if (shadows) p.reads = {shadow_map};
            p.execute = [&](eng::rhi::Encoder& e) {
                renderer->DrawScene(e, scene, kW, kH,
                                    shadows ? shadow_map : eng::rhi::TextureId{});
            };
            g.AddPass(std::move(p));
        }
        if (shadows) {
            eng::RenderGraph::Pass p;
            p.name = "shadow";
            p.depth = shadow_map;
            p.clear_depth = 0.0f;
            p.keep_depth = true;
            p.execute = [&](eng::rhi::Encoder& e) { renderer->DrawShadow(e, scene); };
            g.AddPass(std::move(p));
        }
        if (!g.Compile(error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); std::exit(1); }

        dev->BeginFrame();
        g.Execute(*dev);
        out.stats = renderer->LastStats();
        if (!dev->CommitAndWait(error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); std::exit(1); }
        out.px.resize(std::size_t(kW) * kH * 4);
        if (!dev->ReadPixels(final_color, kW, kH, out.px)) {
            std::fprintf(stderr, "FAIL: readback\n");
            std::exit(1);
        }
        return out;
    };

    const float kSun = 1.1f + 3.14159f;  // opposite the camera
    const Result lit = render(kSun, true);
    const Result moved = render(kSun + 1.3f, true);   // sun swung round
    const Result flat = render(kSun, false);          // shadows off

    if (std::FILE* fp = std::fopen("materials.ppm", "wb")) {
        std::fprintf(fp, "P6\n%d %d\n255\n", kW, kH);
        for (std::size_t i = 0; i + 3 < lit.px.size(); i += 4)
            std::fwrite(&lit.px[i], 1, 3, fp);
        std::fclose(fp);
    }

    // Per-sphere statistics over a box centred on each one.
    // Mean COLOUR, not mean luminance. Two materials can share a brightness and
    // look nothing alike — a red lacquer and a blue paint sit within two 8-bit
    // codes of each other on a luma meter.
    struct Stat { double peak = 0, mean = 0; double rgb[3] = {}; int bright = 0; };
    std::vector<Stat> stats(demo::kCount);
    for (int i = 0; i < demo::kCount; ++i) {
        int cx = 0, cy = 0;
        demo::SpherePixel(i, kW, kH, camera, &cx, &cy);
        double sum = 0, r = 0, g = 0, b = 0;
        int n = 0;
        for (int y = cy - 42; y <= cy + 42; ++y) {
            for (int x = cx - 42; x <= cx + 42; ++x) {
                if (x < 0 || y < 0 || x >= kW || y >= kH) continue;
                const std::size_t k = (std::size_t(y) * kW + x) * 4;
                const double l = Luma(lit.px, k);
                if (l > stats[std::size_t(i)].peak) stats[std::size_t(i)].peak = l;
                if (l > 170.0) ++stats[std::size_t(i)].bright;
                sum += l;
                r += lit.px[k];
                g += lit.px[k + 1];
                b += lit.px[k + 2];
                ++n;
            }
        }
        if (n) {
            stats[std::size_t(i)].mean = sum / n;
            stats[std::size_t(i)].rgb[0] = r / n;
            stats[std::size_t(i)].rgb[1] = g / n;
            stats[std::size_t(i)].rgb[2] = b / n;
        }
    }

    std::printf("%dx%d  %d instances, %d draws\n", kW, kH, lit.stats.submitted,
                lit.stats.draws);
    for (int i = 0; i < demo::kCount; ++i)
        std::printf("  %-26s peak %5.1f  rgb %5.1f/%5.1f/%5.1f  highlight %4d px\n",
                    assets.names[std::size_t(i)].c_str(), stats[std::size_t(i)].peak,
                    stats[std::size_t(i)].rgb[0], stats[std::size_t(i)].rgb[1],
                    stats[std::size_t(i)].rgb[2], stats[std::size_t(i)].bright);

    Check(lit.stats.draws == demo::kCount + 1, "five spheres and a floor drew");

    // Every material must look different from every other one. Comparing the
    // (mean, highlight area) pair rather than one number: two materials can
    // easily share an average brightness while looking nothing alike.
    int identical = 0;
    for (int i = 0; i < demo::kCount; ++i) {
        for (int j = i + 1; j < demo::kCount; ++j) {
            double dc = 0;
            for (int c = 0; c < 3; ++c)
                dc += std::fabs(stats[std::size_t(i)].rgb[c] - stats[std::size_t(j)].rgb[c]);
            const int dh = std::abs(stats[std::size_t(i)].bright - stats[std::size_t(j)].bright);
            if (dc < 12.0 && dh < 30) ++identical;
        }
    }
    Check(identical == 0, "all five materials are visually distinguishable");

    // Roughness spreads the highlight: the polished sphere concentrates its
    // energy into fewer, brighter pixels than the brushed one beside it.
    Check(stats[0].peak > stats[1].peak, "polished metal peaks brighter than brushed");
    Check(stats[1].bright > stats[0].bright, "brushed metal has the wider highlight");
    // NOT asserted here: that metal is darker than dielectric. That only holds
    // at a FIXED base colour, and these five deliberately differ — gold's base
    // is far brighter than the matte blue's, so the comparison would be about
    // the colours rather than the BRDF. apps/materials tests it properly, on a
    // grid that varies metallic with everything else held constant.
    //
    // What does hold here: a surface this rough cannot concentrate a highlight
    // at all, whatever its colour.
    Check(stats[3].bright == 0, "matte paint has no concentrated highlight");
    Check(stats[0].bright > 0, "polished metal does");

    // INTERACTION. Moving the light has to change the image — if it did not,
    // the controls would be decorative.
    std::size_t changed = 0;
    for (std::size_t i = 0; i + 3 < lit.px.size(); i += 4)
        if (std::fabs(Luma(lit.px, i) - Luma(moved.px, i)) > 8.0) ++changed;
    const double moved_pct = 100.0 * double(changed) / double(kW * kH);
    std::printf("  moving the sun changes %.1f%% of the frame\n", moved_pct);
    Check(moved_pct > 5.0, "relighting visibly changes the frame");

    // ...and so does the shadow toggle.
    std::size_t shadow_diff = 0;
    for (std::size_t i = 0; i + 3 < lit.px.size(); i += 4)
        if (Luma(flat.px, i) - Luma(lit.px, i) > 12.0) ++shadow_diff;
    const double shadow_pct = 100.0 * double(shadow_diff) / double(kW * kH);
    std::printf("  shadows darken %.1f%% of the frame\n", shadow_pct);
    Check(shadow_pct > 1.0, "the spheres cast shadows on the floor");

    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
