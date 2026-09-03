// The lighting gate. Every check here is a property that ONE directional light
// cannot produce, because that is the whole point of the change.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "apps/gallery/gallery_scene.h"
#include "engine/app/targets.h"
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
    constexpr int kSamples = 4;
    auto renderer = eng::Renderer::Create(*dev, kFmt, error, kSamples);
    if (!renderer) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    const gallery::Assets assets = gallery::Build(*dev, *renderer, error);
    if (!assets.ok) {
        std::fprintf(stderr, "FAIL: %s\n", error.empty() ? "build" : error.c_str());
        return 1;
    }

    const eng::rhi::TextureId shadow_map = dev->CreateShadowMap(2048);
    // The scene is drawn MULTISAMPLED and resolved into `color`.
    const eng::rhi::TextureId ms_color = dev->CreateRenderTarget(
        kW, kH, eng::Renderer::kSceneFormat, false, kSamples);
    const eng::rhi::TextureId ms_depth =
        dev->CreateDepthTarget(kW, kH, false, kSamples);
    const eng::rhi::TextureId color =
        dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
    const eng::rhi::TextureId depth = dev->CreateDepthTarget(kW, kH);
    const eng::rhi::TextureId out = dev->CreateRenderTarget(kW, kH, kFmt, true);
    if (!Valid(shadow_map) || !Valid(color) || !Valid(depth) || !Valid(out) ||
        !Valid(ms_color) || !Valid(ms_depth)) {
        std::fprintf(stderr, "FAIL: targets\n");
        return 1;
    }

    // The gate owns a FrameTargets so the bloom chain can ask for half- and
    // quarter-sized buffers by name instead of hand-rolling six textures.
    eng::app::FrameTargets targets(*dev, kFmt);
    targets.Resize(kW, kH);
    bool bloom_on = true;
    float bloom_strength = 0.34f;
    bool msaa_on = true;

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
            p.name = "spot shadows";
            p.depth = renderer->ShadowAtlas();
            p.clear_depth = 0.0f;
            p.keep_depth = true;
            p.execute = [&](eng::rhi::Encoder& e) {
                renderer->DrawLightShadows(e, scene);
            };
            graph.AddPass(std::move(p));
        }
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
            p.color = msaa_on ? ms_color : color;
            if (msaa_on) p.resolve = color;
            p.depth = msaa_on ? ms_depth : depth;
            p.clear_color[0] = 0.018f; p.clear_color[1] = 0.020f;
            p.clear_color[2] = 0.028f; p.clear_color[3] = 1.0f;
            p.clear_depth = 0.0f;
            p.reads = {shadow_map, renderer->ShadowAtlas()};
            p.execute = [&](eng::rhi::Encoder& e) {
                renderer->DrawScene(e, scene, kW, kH, shadow_map);
            };
            graph.AddPass(std::move(p));
        }
        // --- bloom ------------------------------------------------------------
        // Bright pass at half resolution, then a separable blur at half and
        // again at quarter. Two octaves: the tight one puts a halo right
        // against the source, the wide one the soft glow further out. One
        // radius alone gives either a hard ring or a flat wash.
        const eng::rhi::TextureId b_half = targets.Hdr("bloomA", 2);
        const eng::rhi::TextureId b_half2 = targets.Hdr("bloomB", 2);
        const eng::rhi::TextureId b_quarter = targets.Hdr("bloomC", 4);
        const eng::rhi::TextureId b_quarter2 = targets.Hdr("bloomD", 4);
        // A target of its own for the last blur. Reusing the bright pass's
        // buffer was the obvious saving and the render graph refused it: two
        // passes writing one texture have no defined order without resource
        // versioning, and it would have worked right up until the graph
        // reordered them.
        const eng::rhi::TextureId b_out = targets.Hdr("bloomOut", 2);
        const float hw = 2.0f / float(kW), hh = 2.0f / float(kH);
        const float qw = 4.0f / float(kW), qh = 4.0f / float(kH);
        {
            eng::RenderGraph::Pass p;
            p.name = "bright";
            p.color = b_half;
            p.reads = {color};
            p.execute = [&](eng::rhi::Encoder& e) {
                // Linear radiance, and the scene's lamps run into the tens.
                // 1.15 was the first guess and it bloomed off the lit floor,
                // which turned the whole frame magenta -- the pools ARE that
                // bright, so the threshold has to sit above them and not just
                // above white.
                (*renderer).DrawBloomBright(e, color, 3.2f, 0.9f);
            };
            graph.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "blurAx";
            p.color = b_half2;
            p.reads = {b_half};
            p.execute = [&](eng::rhi::Encoder& e) {
                (*renderer).DrawBloomBlur(e, b_half, hw, 0.0f);
            };
            graph.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "blurAy";
            p.color = b_quarter;
            p.reads = {b_half2};
            p.execute = [&](eng::rhi::Encoder& e) {
                (*renderer).DrawBloomBlur(e, b_half2, 0.0f, hh);
            };
            graph.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "blurBx";
            p.color = b_quarter2;
            p.reads = {b_quarter};
            p.execute = [&](eng::rhi::Encoder& e) {
                (*renderer).DrawBloomBlur(e, b_quarter, qw, 0.0f);
            };
            graph.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "blurBy";
            p.color = b_out;
            p.reads = {b_quarter2};
            p.execute = [&](eng::rhi::Encoder& e) {
                (*renderer).DrawBloomBlur(e, b_quarter2, 0.0f, qh);
            };
            graph.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "composite";
            p.color = out;
            p.reads = {color, b_out};
            p.execute = [&](eng::rhi::Encoder& e) {
                renderer->DrawComposite(e, color, {},
                                        bloom_on ? b_out : eng::rhi::TextureId{},
                                        bloom_strength);
            };
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

    std::printf("multisampling\n");
    {
        // A second renderer at ONE sample, with its own assets: mesh and
        // material handles are indices into a particular renderer's tables and
        // do not transfer. Expensive, and the only way to get a true A/B —
        // pipelines are compiled against a sample count, so one renderer cannot
        // draw both ways.
        std::string e1;
        auto plain = eng::Renderer::Create(*dev, kFmt, e1, 1);
        const gallery::Assets plain_assets =
            plain ? gallery::Build(*dev, *plain, e1) : gallery::Assets{};
        Check(plain && plain_assets.ok, "a single-sampled renderer builds too");

        // Count SILHOUETTE pixels: ones sitting partway between their
        // neighbours across a high-contrast edge. Aliasing is the absence of
        // exactly those — a hard step from one surface to the next.
        auto intermediate = [&](const std::vector<std::uint8_t>& img, int minc) {
            int n = 0;
            for (int y = 1; y < kH - 1; ++y)
                for (int x = 1; x < kW - 1; ++x) {
                    const std::size_t c = (std::size_t(y) * kW + x) * 4;
                    const std::size_t l = c - 4, r = c + 4;
                    const int lv = img[l] + img[l + 1] + img[l + 2];
                    const int cv = img[c] + img[c + 1] + img[c + 2];
                    const int rv = img[r] + img[r + 1] + img[r + 2];
                    const int lo = std::min(lv, rv), hi = std::max(lv, rv);
                    if (hi - lo < minc) continue;       // no edge here
                    if (cv > lo + 12 && cv < hi - 12) ++n;  // strictly between
                }
            return n;
        };

        const eng::Scene s4 = gallery::MakeScene(assets, 0.7f, 0.0f);
        draw(s4);
        const std::vector<std::uint8_t> aa_px = pixels;

        // The same frame through the single-sampled renderer.
        {
            eng::Scene s1 = gallery::MakeScene(plain_assets, 0.7f, 0.0f);
            s1.camera.eye = kEye;
            s1.camera.target = kTarget;
            eng::RenderGraph g;
            {
                eng::RenderGraph::Pass p;
                p.name = "scene";
                p.color = color;
                p.depth = depth;
                p.clear_color[0] = 0.018f; p.clear_color[1] = 0.020f;
                p.clear_color[2] = 0.028f; p.clear_color[3] = 1.0f;
                p.clear_depth = 0.0f;
                p.execute = [&](eng::rhi::Encoder& en) {
                    plain->DrawScene(en, s1, kW, kH);
                };
                g.AddPass(std::move(p));
            }
            {
                eng::RenderGraph::Pass p;
                p.name = "composite";
                p.color = out;
                p.reads = {color};
                p.execute = [&](eng::rhi::Encoder& en) {
                    plain->DrawComposite(en, color, {}, {}, 0.0f, 1.0f);
                };
                g.AddPass(std::move(p));
            }
            std::string ge;
            if (g.Compile(ge)) {
                dev->BeginFrame();
                g.Execute(*dev);
                std::string w;
                (void)dev->CommitAndWait(w);
                pixels.assign(std::size_t(kW) * kH * 4, 0);
                (void)dev->ReadPixels(out, kW, kH, pixels);
            }
        }
        // A high contrast bar, so only real silhouettes count. At a low one
        // the scene's own soft gradients — light pools, shadow edges — swamp
        // the measurement and the two renderers look nearly alike.
        const int aa = intermediate(aa_px, 320);
        const int no_aa = intermediate(pixels, 320);
        std::printf("    silhouette gradient pixels: 4x MSAA %d, single-sampled %d\n",
                    aa, no_aa);
        Check(aa > no_aa * 2, "multisampling puts real gradients on the silhouettes");
        draw(full);
    }

    std::printf("bloom\n");
    {
        // THE invariant. Binding the bloom texture with a strength of zero must
        // produce exactly the image you get with no bloom at all.
        //
        // This is the check that would have saved a very long hunt. The bloom
        // pipelines were compiled against the eight-bit output format while
        // rendering into half-float targets — which Metal does not reject, it
        // just leaves the contents undefined. Every pixel came back NaN, and
        // NaN times a strength of zero is still NaN, so it survived the add,
        // the tone map and saturate() to zero one colour channel of the whole
        // frame. "Turn the effect off and nothing changes" catches that at once.
        bloom_strength = 0.0f;
        draw(full);
        const std::vector<std::uint8_t> zero_strength = pixels;
        bloom_on = false;
        draw(full);
        int differ = 0;
        for (std::size_t i = 0; i < pixels.size(); i += 4)
            if (std::abs(int(pixels[i]) - int(zero_strength[i])) > 1 ||
                std::abs(int(pixels[i + 1]) - int(zero_strength[i + 1])) > 1 ||
                std::abs(int(pixels[i + 2]) - int(zero_strength[i + 2])) > 1)
                ++differ;
        std::printf("    %d pixels differ between zero-strength and no bloom\n",
                    differ);
        Check(differ == 0, "bloom at zero strength changes nothing");

        // And with it on, it brightens — but only around what is actually
        // bright. The far corner of the room has no light source in it.
        bloom_on = true;
        bloom_strength = 0.34f;
        draw(full);
        const std::vector<std::uint8_t> lit = pixels;

        auto mean_of = [&](const std::vector<std::uint8_t>& img, int x0, int y0,
                           int x1, int y1) {
            double sum = 0;
            int n = 0;
            for (int y = y0; y < y1; ++y)
                for (int x = x0; x < x1; ++x) {
                    const std::size_t i = (std::size_t(y) * kW + x) * 4;
                    sum += img[i] + img[i + 1] + img[i + 2];
                    ++n;
                }
            return sum / std::max(n, 1);
        };
        // Around the lit spheres.
        const double near_on = mean_of(lit, 380, 230, 760, 380);
        const double near_off = mean_of(zero_strength, 380, 230, 760, 380);
        // The top-left corner: dark wall, no lamp anywhere near it.
        const double far_on = mean_of(lit, 20, 20, 200, 120);
        const double far_off = mean_of(zero_strength, 20, 20, 200, 120);
        std::printf("    near the lights %.1f -> %.1f   dark corner %.1f -> %.1f\n",
                    near_off, near_on, far_off, far_on);
        Check(near_on > near_off + 4.0, "bloom brightens around the bright things");
        Check(far_on < far_off + 2.0, "and leaves the dark parts of the room alone");
        bloom_strength = 0.34f;
    }

    std::printf("spot lights cast shadows into the atlas\n");
    {
        Check(Valid(renderer->ShadowAtlas()), "the atlas exists");
        draw(full);
        Check(renderer->ShadowedLightCount() == 4,
              "three spots and one point light got shadow space");
        // Three spots at one tile each, one point light at six.
        Check(renderer->ShadowTilesUsed() == 3 + 6,
              "a point light takes six tiles and a spot one");

        // On versus off. Every darkened pixel has to be INSIDE a pool: the
        // directional key light's shadows are unchanged between these two
        // frames, so anything that moves is the spots' doing.
        std::vector<std::uint8_t> lit_with;
        draw(full);
        lit_with = pixels;

        eng::Scene noshadow = full;
        for (eng::Light& l : noshadow.lights) l.casts_shadow = false;
        draw(noshadow);
        Check(renderer->ShadowedLightCount() == 0, "and none when they ask for none");

        int darkened = 0;
        for (std::size_t i = 0; i < pixels.size(); i += 4) {
            const int a = pixels[i] + pixels[i + 1] + pixels[i + 2];
            const int b = lit_with[i] + lit_with[i + 1] + lit_with[i + 2];
            if (a - b > 45) ++darkened;
        }
        std::printf("    %d pixels darkened by the spot shadows\n", darkened);
        Check(darkened > 4000, "turning them on darkens a substantial area");

        // The shadow must be UNDER its own plinth, not somewhere else. A
        // light-space shadow map with the wrong matrix still darkens plenty of
        // pixels — just not the right ones.
        int px = 0, py = 0;
        Project(eng::Vec3{gallery::kPlinthX[2] + 0.15f, 0.02f, 1.05f}, &px, &py);
        double wr, wg, wb, nr, ng, nb;
        std::vector<std::uint8_t> off = pixels;
        pixels = lit_with;
        Mean(px - 26, py - 14, px + 26, py + 14, &wr, &wg, &wb);
        pixels = off;
        Mean(px - 26, py - 14, px + 26, py + 14, &nr, &ng, &nb);
        std::printf("    just in front of the right plinth: %.0f shadowed, "
                    "%.0f not\n", wr + wg + wb, nr + ng + nb);
        Check((wr + wg + wb) < (nr + ng + nb) * 0.75,
              "the plinth's own shadow falls in its own pool");
    }

    std::printf("each light reads its OWN tile\n");
    {
        // Reversing the light list must not change the picture. The same lights
        // are present, so each still gets a tile — a different one, holding its
        // own map. A renderer that pointed every light at tile zero renders the
        // correct image for a symmetric scene and a different one the moment
        // the order changes, which is what this catches.
        //
        // It only bites because the three plinths are different heights. With
        // identical ones every tile holds the same silhouette and no test can
        // tell which was read.
        draw(full);
        const std::vector<std::uint8_t> forward = pixels;
        eng::Scene reversed = full;
        std::reverse(reversed.lights.begin(), reversed.lights.end());
        draw(reversed);

        int differ = 0;
        for (std::size_t i = 0; i < pixels.size(); i += 4)
            if (std::abs(int(pixels[i]) - int(forward[i])) > 6) ++differ;
        std::printf("    %d pixels changed when the light order was reversed\n",
                    differ);
        Check(differ < 400, "reordering the lights does not change the frame");
    }

    std::printf("a point light shadows in every direction\n");
    {
        // The check that separates a cube shadow from a spot is not "does it
        // shadow" but "is the shadow in the right PLACE".
        //
        // "Both sides darken" was tried first and a mutation that always
        // sampled cube face zero passed it: that face's map holds plinths too,
        // so the wrong face still produced plausible shadows. So: ONE blocker,
        // off to one side, and the shadow has to land behind it and nowhere
        // else.
        eng::Scene s2;
        s2.lightColor = eng::Vec4{0, 0, 0, 1};
        s2.ambientSky = eng::Vec3{0, 0, 0};
        s2.ambientGround = eng::Vec3{0, 0, 0};

        eng::Instance ground;
        ground.mesh = assets.floor;
        ground.material = assets.floor_mat;
        ground.model = eng::Mat4::Translation(eng::Vec3{0, -0.2f, 0});
        s2.instances.push_back(ground);

        // Lamp above the origin; blocker a little to its +x, at the same
        // height. The shadow must be cast further along +x.
        const eng::Vec3 lamp_at{0.0f, 2.4f, 0.0f};
        // OFF the axis in both other directions. A blocker sitting exactly on
        // an axis maps to the centre of its cube face, where a flipped or
        // rotated `up` vector barely moves the lookup — so an on-axis test
        // cannot tell whether the shader and the renderer agree about which way
        // is up on a face. Off-centre, a disagreement rotates the shadow away.
        // Below the lamp, so the shadow reaches the floor, and off BOTH other
        // axes so the lookup lands away from the centre of its cube face.
        // X is the largest component, so face +X is the one selected — which
        // is where a disagreement about that face's `up` shows up.
        const eng::Vec3 blocker_at{1.4f, 1.6f, 0.5f};
        eng::Instance blocker;
        blocker.mesh = assets.sphere;
        blocker.material = assets.stone;
        blocker.model = eng::Mat4::Translation(blocker_at);
        s2.instances.push_back(blocker);

        eng::Light lamp;
        lamp.position = lamp_at;
        lamp.color = eng::Vec3{70.0f, 70.0f, 70.0f};
        lamp.range = 16.0f;
        lamp.casts_shadow = true;
        lamp.shadow_near = 0.25f;
        s2.lights.push_back(lamp);

        draw(s2);
        const std::vector<std::uint8_t> shadowed = pixels;
        Check(renderer->ShadowTilesUsed() == 6, "it took all six faces");
        s2.lights[0].casts_shadow = false;
        draw(s2);

        // Where the blocker's shadow must land: along the ray from the lamp
        // through the blocker, continued down to the floor.
        const eng::Vec3 dir = Normalize(blocker_at - lamp_at);
        const float t = (0.0f - lamp_at.y) / (dir.y != 0.0f ? dir.y : -1.0f);
        // The ray is horizontal here, so use a fixed distance out instead.
        // Continue the lamp->blocker ray until it reaches the floor.
        const float reach = dir.y < -1e-3f ? (0.02f - lamp_at.y) / dir.y : 3.0f;
        const eng::Vec3 expect = lamp_at + dir * reach;
        const eng::Vec3 mirror{-expect.x, expect.y, -expect.z};
        (void)t;

        auto darkened_at = [&](eng::Vec3 world) {
            int px = 0, py = 0;
            Project(world, &px, &py);
            int n = 0;
            for (int y = std::max(py - 22, 0); y < std::min(py + 22, kH); ++y)
                for (int x = std::max(px - 34, 0); x < std::min(px + 34, kW); ++x) {
                    const std::size_t i = (std::size_t(y) * kW + x) * 4;
                    const int a = pixels[i] + pixels[i + 1] + pixels[i + 2];
                    const int b = shadowed[i] + shadowed[i + 1] + shadowed[i + 2];
                    if (a - b > 25) ++n;
                }
            return n;
        };
        const int behind = darkened_at(expect);
        const int opposite = darkened_at(mirror);
        std::printf("    darkened behind the blocker %d, mirrored across the lamp %d\n",
                    behind, opposite);
        Check(behind > 300, "the blocker casts a shadow on its own far side");
        Check(opposite < behind / 4,
              "and NOT on the opposite side, which a wrong cube face would give");
    }

    std::printf("shadow space runs out gracefully\n");

    {
        eng::Scene crowd = full;
        for (int i = 0; i < 24; ++i) {
            eng::Light l;
            l.type = eng::LightType::Spot;
            l.position = eng::Vec3{float(i) - 4.0f, 3.5f, 1.0f};
            l.direction = eng::Vec3{0, -1, 0};
            l.color = eng::Vec3{20, 20, 20};
            l.casts_shadow = true;
            crowd.lights.push_back(l);
        }
        draw(crowd);
        Check(renderer->ShadowTilesUsed() == eng::Renderer::kShadowTiles,
              "tiles are handed out until they run out");
        Check(stats.invalid == 0, "the lights that missed out still light, unshadowed");

        // A point light that cannot fit ALL six faces gets none. Five faces is
        // worse than zero: the sixth direction is lit straight through walls,
        // and it is the one nobody thinks to check.
        eng::Scene tight = full;
        tight.lights.clear();
        for (int i = 0; i < 3; ++i) {
            eng::Light spot;
            spot.type = eng::LightType::Spot;
            spot.position = eng::Vec3{float(i), 4.0f, 0.0f};
            spot.direction = eng::Vec3{0, -1, 0};
            spot.color = eng::Vec3{20, 20, 20};
            spot.casts_shadow = true;
            tight.lights.push_back(spot);
        }
        // 3 spots leave 13 tiles; two point lights want 12, a third cannot fit.
        for (int i = 0; i < 3; ++i) {
            eng::Light p;
            p.position = eng::Vec3{float(i) * 2.0f - 2.0f, 1.5f, 2.0f};
            p.color = eng::Vec3{10, 10, 10};
            p.casts_shadow = true;
            tight.lights.push_back(p);
        }
        draw(tight);
        Check(renderer->ShadowTilesUsed() == 3 + 6 + 6,
              "the point light that could not fit six took none");
        Check(renderer->ShadowedLightCount() == 5, "and is simply unshadowed");
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
