// Pure C++20. Renders the apartment offscreen and checks the things a floor
// plan viewer has to get right. Same building and same passes as the windowed
// viewer, so this is a real gate rather than a lookalike.
//
// Writes house.ppm for eyeballing; the assertions below are what actually pass
// or fail.
#include "tests/house/scene_build.h"
#include "engine/render/rendergraph.h"
#include "engine/render/renderer.h"
#include "engine/rhi/rhi.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-56s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

double Luma(const std::vector<std::uint8_t>& p, std::size_t i) {
    return 0.2126 * p[i] + 0.7152 * p[i + 1] + 0.0722 * p[i + 2];
}

constexpr int kW = 900, kH = 700;
constexpr float kNoCut = 1.0e9f;

struct Frame {
    std::vector<std::uint8_t> px;       // final, composited
    std::vector<std::uint8_t> raw;      // pre-composite, uniform background
    eng::RenderStats stats;
    int shadow_draws = 0;
    std::vector<std::string> order;
};

}  // namespace

int main(int argc, char** argv) {
    std::string error;
    auto dev = eng::rhi::Device::Create(error);
    if (!dev) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
    auto renderer = eng::Renderer::Create(*dev, eng::rhi::Format::RGBA8Unorm, error);
    if (!renderer) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    const house::Assets assets = house::Build(*renderer, error);
    if (!assets.ok) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    const eng::rhi::TextureId shadow_map = dev->CreateShadowMap(2048);
    const auto kFmt = eng::rhi::Format::RGBA8Unorm;
    const eng::rhi::TextureId scene_color =
        dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
    const eng::rhi::TextureId scene_depth = dev->CreateDepthTarget(kW, kH, true);
    const eng::rhi::TextureId ao_target = dev->CreateRenderTarget(kW, kH, kFmt, true);
    const eng::rhi::TextureId final_color = dev->CreateRenderTarget(kW, kH, kFmt, true);
    // The scene tone mapped but otherwise untouched: no occlusion, no vignette.
    // The scene target itself is half-float and cannot be read back.
    const eng::rhi::TextureId raw_color = dev->CreateRenderTarget(kW, kH, kFmt, true);
    if (!Valid(shadow_map) || !Valid(scene_color) || !Valid(scene_depth) ||
        !Valid(ao_target) || !Valid(final_color)) {
        std::fprintf(stderr, "FAIL: targets\n");
        return 1;
    }

    // Renders one configuration through the same four-pass graph the viewer uses.
    // interior == true stands the camera inside the living room instead of
    // orbiting the building; nothing else about the frame changes.
    auto render = [&](bool cut, bool ssao, bool ortho, eng::rhi::TextureId out,
                      bool interior = false, bool shadows = true) {
        Frame f;
        eng::Scene scene = house::MakeScene(assets, cut ? 1.35f : kNoCut);
        if (!shadows) scene.shadowExtent = 0.0f;
        if (interior) {
            eng::WalkController walk;
            // In the kitchen, looking at the floor where the south window's
            // sunlight lands. The sun points from (0.42, 0.78, 0.46) TOWARD the
            // light, so its rays travel in -z: only south-facing openings let
            // it in, and a camera aimed at the north wall sees no patch at all.
            walk.position = eng::Vec3{-1.4f, 1.6f, 1.9f};
            walk.yaw = 2.85f;
            walk.pitch = -0.42f;
            walk.Apply(scene.camera);
        } else {
            eng::OrbitController orbit;
            orbit.target = eng::Vec3{0.0f, 1.3f, 0.0f};
            orbit.distance = 19.0f;
            orbit.yaw = 0.9f;
            orbit.pitch = ortho ? 1.45f : 0.62f;  // ortho defaults to looking down
            orbit.Apply(scene.camera);
            scene.camera.projection =
                ortho ? eng::Projection::Orthographic : eng::Projection::Perspective;
        }

        eng::RenderGraph g;
        {
            // A plain resolve of the scene: tone mapped, nothing else. The
            // checks that want to see the geometry against a uniform
            // background read this one.
            eng::RenderGraph::Pass p;
            p.name = "raw";
            p.color = raw_color;
            p.reads = {scene_color};
            p.execute = [&](eng::rhi::Encoder& e) {
                renderer->DrawComposite(e, scene_color, {}, {}, 0.0f,
                                        /*vignette=*/0.0f);
            };
            g.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "composite";
            p.color = out;
            p.reads = ssao ? std::vector<eng::rhi::TextureId>{scene_color, ao_target}
                           : std::vector<eng::rhi::TextureId>{scene_color};
            p.execute = [&](eng::rhi::Encoder& e) {
                renderer->DrawComposite(e, scene_color,
                                        ssao ? ao_target : eng::rhi::TextureId{});
            };
            g.AddPass(std::move(p));
        }
        if (ssao) {
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
            p.keep_depth = ssao;
            p.reads = {shadow_map};
            p.execute = [&](eng::rhi::Encoder& e) {
                renderer->DrawScene(e, scene, kW, kH, shadow_map);
            };
            g.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "shadow";
            p.depth = shadow_map;
            p.clear_depth = 0.0f;
            p.keep_depth = true;
            p.execute = [&](eng::rhi::Encoder& e) { renderer->DrawShadow(e, scene); };
            g.AddPass(std::move(p));
        }
        if (!g.Compile(error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); std::exit(1); }
        f.order = g.Order();

        dev->BeginFrame();
        g.Execute(*dev);
        f.stats = renderer->LastStats();
        f.shadow_draws = renderer->ShadowDrawCount();
        if (!dev->CommitAndWait(error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); std::exit(1); }
        f.px.resize(std::size_t(kW) * kH * 4);
        f.raw.resize(f.px.size());
        if (!dev->ReadPixels(out, kW, kH, f.px) ||
            !dev->ReadPixels(raw_color, kW, kH, f.raw)) {
            std::fprintf(stderr, "FAIL: readback\n");
            std::exit(1);
        }
        return f;
    };

    const Frame cut_on = render(true, true, false, final_color);
    const Frame cut_off = render(false, true, false, final_color);
    const Frame no_ao = render(true, false, false, final_color);
    const Frame top = render(true, true, true, final_color);
    const Frame inside = render(false, true, false, final_color, true, true);
    const Frame inside_lit = render(false, true, false, final_color, true, false);
    // Same question from OUTSIDE: does the building shadow the ground at all?
    const Frame ext_shadow = render(false, true, false, final_color, false, true);
    const Frame ext_lit = render(false, true, false, final_color, false, false);

    // Write whichever view was asked for; default is the cut perspective.
    const Frame* pick = &cut_on;
    if (argc > 1 && std::strcmp(argv[1], "top") == 0) pick = &top;
    if (argc > 1 && std::strcmp(argv[1], "inside") == 0) pick = &inside;
    const Frame& save = *pick;
    if (std::FILE* fp = std::fopen("house.ppm", "wb")) {
        std::fprintf(fp, "P6\n%d %d\n255\n", kW, kH);
        for (std::size_t i = 0; i + 3 < save.px.size(); i += 4)
            std::fwrite(&save.px[i], 1, 3, fp);
        std::fclose(fp);
    }

    // Measured on the PRE-composite image. The composite pass applies a
    // vignette, which makes the background a gradient rather than a constant —
    // after it, "is this pixel background" has no answer and every coverage
    // number comes out at ~100%.
    auto Coverage = [](const Frame& f) {
        const std::uint8_t bg[3] = {f.raw[0], f.raw[1], f.raw[2]};
        std::size_t n = 0;
        for (std::size_t i = 0; i + 3 < f.raw.size(); i += 4)
            if (f.raw[i] != bg[0] || f.raw[i + 1] != bg[1] || f.raw[i + 2] != bg[2]) ++n;
        return 100.0 * double(n) / double(kW * kH);
    };
    auto MeanLuma = [](const Frame& f) {
        double s = 0;
        for (std::size_t i = 0; i + 3 < f.px.size(); i += 4) s += Luma(f.px, i);
        return s / double(kW * kH);
    };

    std::printf("%dx%d  graph:", kW, kH);
    for (const std::string& n : cut_on.order) std::printf(" %s", n.c_str());
    std::printf("\n  instances=%d draws=%d transparent=%d culled=%d\n",
                cut_on.stats.submitted, cut_on.stats.draws,
                cut_on.stats.transparent_draws, cut_on.stats.culled);
    std::printf("  coverage cut=%.1f%% uncut=%.1f%%   mean luma ao=%.1f no-ao=%.1f\n",
                Coverage(cut_on), Coverage(cut_off), MeanLuma(cut_on), MeanLuma(no_ao));

    // The graph derived the order from resource dependencies alone; the passes
    // were added raw, composite, ssao, scene, shadow. What has to hold is that
    // the two producers come before every consumer — the relative order of
    // "ssao" and "raw", which read the same things and write different ones, is
    // genuinely free and asserting one of the two would be asserting the sort's
    // tie-break rather than its correctness.
    const auto at = [&](const char* name) {
        for (std::size_t i = 0; i < cut_on.order.size(); ++i)
            if (cut_on.order[i] == name) return int(i);
        return -1;
    };
    Check(cut_on.order.size() == 5, "all five passes ran");
    Check(at("shadow") == 0 && at("scene") == 1, "producers first, in order");
    Check(at("ssao") > at("scene") && at("raw") > at("scene"),
          "both consumers of the scene target come after it");
    Check(at("composite") > at("ssao"), "the composite comes after the occlusion");

    // Ground, floor, walls, roof, doors, frames, seven pieces of furniture, glass.
    Check(cut_on.stats.submitted == 14, "every part of the building submitted");
    Check(cut_on.stats.draws == 14, "and all of it survived culling");
    Check(cut_on.stats.transparent_draws == 1, "the glass went in the blended batch");

    // SECTION CUT, measured differentially. A silhouette count stopped working
    // once the building sat on a 30 m ground plane: the ground fills most of
    // the frame and the cut does not touch it, so total coverage barely moves.
    // Counting pixels that CHANGED asks the right question.
    std::size_t changed = 0;
    for (std::size_t i = 0; i + 3 < cut_on.px.size(); i += 4)
        if (std::fabs(Luma(cut_on.px, i) - Luma(cut_off.px, i)) > 8.0) ++changed;
    const double changed_pct = 100.0 * double(changed) / double(kW * kH);
    std::printf("  section cut changes %.1f%% of the frame\n", changed_pct);
    Check(changed_pct > 2.0, "the section cut visibly removes the roof");

    // SSAO can only darken. If it brightened anything the multiply has the
    // wrong sign or the AO buffer is not being cleared to white.
    std::size_t brighter = 0;
    for (std::size_t i = 0; i + 3 < cut_on.px.size(); i += 4)
        if (Luma(cut_on.px, i) > Luma(no_ao.px, i) + 3.0) ++brighter;
    Check(MeanLuma(cut_on) < MeanLuma(no_ao), "ambient occlusion darkens the frame");
    Check(brighter < (kW * kH) / 200, "ambient occlusion never brightens");

    // ORTHOGRAPHIC. The projection property itself is tested on the CPU in
    // engine/core:math_test, where it belongs — here we only confirm the
    // top-down view renders at all. Measuring parallelism in pixels stopped
    // being meaningful once the ground plane became the widest thing on screen.
    auto WidthAtRow = [&](const Frame& f, int y) {
        const std::uint8_t bg[3] = {f.raw[0], f.raw[1], f.raw[2]};
        int lo = kW, hi = -1;
        for (int x = 0; x < kW; ++x) {
            const std::size_t i = (std::size_t(y) * kW + x) * 4;
            if (f.raw[i] != bg[0] || f.raw[i + 1] != bg[1] || f.raw[i + 2] != bg[2]) {
                if (x < lo) lo = x;
                if (x > hi) hi = x;
            }
        }
        return hi >= lo ? hi - lo : 0;
    };
    Check(WidthAtRow(top, kH / 2) > 0, "the top-down view drew something");

    // Enough distinct colours that materials, shadows and AO are all varying.
    // The bound is empirical, not derived: a flat-shaded or single-material
    // version of this frame lands in the low hundreds, and the real render sits
    // around 1400, so 1000 separates them with room for the light to be tuned.
    std::set<std::uint32_t> hues;
    for (std::size_t i = 0; i + 3 < cut_on.px.size(); i += 4)
        hues.insert((std::uint32_t(cut_on.px[i]) << 16) |
                    (std::uint32_t(cut_on.px[i + 1]) << 8) | cut_on.px[i + 2]);
    std::printf("  distinct colours: %zu\n", hues.size());
    Check(hues.size() > 1000, "materials, shadows and AO all vary the image");

    // INTERIOR. Standing in the living room with the roof on, the room must be
    // legible — not the black box that a closed building lit by one outdoor sun
    // becomes without ambient light or sunlight through the windows.
    {
        double sum = 0;
        double brightest = 0;
        for (std::size_t i = 0; i + 3 < inside.px.size(); i += 4) {
            const double l = Luma(inside.px, i);
            sum += l;
            if (l > brightest) brightest = l;
        }
        const double mean = sum / double(kW * kH);
        double lit_sum = 0;
        for (std::size_t i = 0; i + 3 < inside_lit.px.size(); i += 4)
            lit_sum += Luma(inside_lit.px, i);
        const double lit_mean = lit_sum / double(kW * kH);
        // Count DARKENED PIXELS, not mean luma. A shadow covers a few percent of a
    // frame, so its effect on the average is smaller than one 8-bit code — the
    // mean reads identical to one decimal place whether shadows work or not,
    // which is exactly the false negative this chased for an hour.
    auto Darkened = [](const Frame& with, const Frame& without) {
        std::size_t n = 0;
        for (std::size_t i = 0; i + 3 < with.px.size(); i += 4)
            if (Luma(without.px, i) - Luma(with.px, i) > 12.0) ++n;
        return 100.0 * double(n) / double(kW * kH);
    };
    const double ext_shadowed = Darkened(ext_shadow, ext_lit);
    std::printf("  shadow casters drawn: %d; exterior shadowed area %.2f%%\n",
                ext_shadow.shadow_draws, ext_shadowed);
    Check(ext_shadow.shadow_draws > 10, "the shadow pass drew the building");
    // 0.3%, measured. The number is small because the sun and the default
    // camera are only roughly opposed, so much of the building's shadow falls
    // where the building itself hides it.
    Check(ext_shadowed > 0.3, "the building casts a shadow on the ground");
        const double shaded = Darkened(inside, inside_lit);
        std::printf("  interior: mean luma %.1f, brightest %.1f\n", mean, brightest);
        Check(mean > 25.0, "the interior is not a black box");

        // The roof shadows the interior. This was an open bug until the cause
        // turned up: fs_shadow reads the section-cut plane out of FrameUniforms,
        // but DrawShadow bound that block to the vertex stage only. An unbound
        // fragment buffer reads as ZERO rather than failing, so the cut plane
        // became y = 0 and every caster above the origin — which is the whole
        // building — was discarded from the shadow map. Measured here: 0.0%
        // before the bind was added, 64.6% after.
        std::printf("  interior shadowed area %.1f%%\n", shaded);
        Check(shaded > 10.0, "the roof shadows the interior");
    }

    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
