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

    std::printf("cascaded shadow maps\n");
    {
        // The claim: for the same total texels, cascades put more of them near
        // the camera. Measured against a REFERENCE — a single map fitted
        // tightly around the near field, which is as good as this resolution
        // gets there but covers nothing else.
        //
        // A wide single map has to cover the same distance the cascades do, so
        // it spends most of its texels far away and comes out blocky underfoot.
        // Cascades should agree with the tight reference; the wide one should
        // not.
        //
        // Counting "shadow boundary pixels" was tried first and gave 157 for
        // both, which was a coincidence of the metric rather than a result —
        // the two images differ, they just happen to have the same number of
        // steep horizontal steps in that window.
        auto render_with = [&](int cascade_count, float extent, float distance) {
            eng::Scene s2 = full;
            s2.shadowCascades = cascade_count;
            s2.shadowExtent = extent;
            s2.shadowDistance = distance;
            draw(s2);
            return pixels;
        };
        const std::vector<std::uint8_t> tight = render_with(1, 7.0f, 7.0f);
        const std::vector<std::uint8_t> wide = render_with(1, 30.0f, 30.0f);
        const std::vector<std::uint8_t> csm = render_with(4, 30.0f, 30.0f);

        // How far each is from the reference, over the near half of the floor.
        auto distance_from_tight = [&](const std::vector<std::uint8_t>& img) {
            long long sum = 0;
            for (int y = 380; y < 620; ++y)
                for (int x = 180; x < 860; ++x) {
                    const std::size_t i = (std::size_t(y) * kW + x) * 4;
                    sum += std::abs(int(img[i]) - int(tight[i]));
                }
            return sum;
        };
        const long long wide_err = distance_from_tight(wide);
        const long long csm_err = distance_from_tight(csm);
        std::printf("    near-field difference from a tight map: wide %lld, "
                    "cascaded %lld\n", wide_err, csm_err);
        Check(csm_err < wide_err * 0.8,
              "cascades match a tight map near the camera; one wide map does not");

        // TEXEL SNAPPING, tested directly on the matrix rather than through a
        // render. Shimmer is a property of MOTION — the map's origin sliding a
        // fraction of a texel per frame, so every shadow edge re-samples and
        // the whole scene crawls — and a gate that renders from a fixed camera
        // cannot see it at all. A mutation that removed the snap passed the
        // image comparison above without a mark.
        {
            eng::Scene a = full;
            a.shadowCascades = 4;
            a.shadowDistance = 30.0f;
            a.camera.eye = kEye;
            a.camera.target = kTarget;

            const eng::Vec3 fixed_point{1.4f, 0.4f, 0.9f};
            auto texel_of = [&](float nudge) {
                eng::Scene b = a;
                // A sub-texel sideways step: far too small to change what is
                // visible, and exactly the size of step that makes an unsnapped
                // map crawl.
                b.camera.eye = eng::Vec3{kEye.x + nudge, kEye.y, kEye.z};
                b.camera.target = eng::Vec3{kTarget.x + nudge, kTarget.y, kTarget.z};
                const eng::Vec4 clip =
                    b.CascadeViewProj(0, float(kW) / float(kH)) *
                    eng::Vec4{fixed_point.x, fixed_point.y, fixed_point.z, 1.0f};
                // In texels of a 1024 tile.
                return eng::Vec3{(clip.x / clip.w * 0.5f + 0.5f) * 1024.0f,
                                 (clip.y / clip.w * 0.5f + 0.5f) * 1024.0f, 0.0f};
            };
            // The point DOES move as the camera does — that is not the
            // question, and measuring total movement was the first mistake
            // here: it comes out the same snapped or not, because it is just
            // the camera's own motion expressed in texels.
            //
            // What snapping claims is that it moves in WHOLE texels. So the
            // FRACTIONAL part of its texel coordinate should stay put; that
            // fraction changing is exactly the sub-texel slide that makes every
            // shadow edge re-sample and the scene crawl.
            float lo = 2.0f, hi = -1.0f;
            for (int k = 0; k <= 24; ++k) {
                const eng::Vec3 at = texel_of(float(k) * 0.004f);
                float frac = at.x - std::floor(at.x);
                lo = std::fmin(lo, frac);
                hi = std::fmax(hi, frac);
            }
            std::printf("    fractional texel position varies by %.3f as the "
                        "camera creeps sideways\n", hi - lo);
            Check(hi - lo < 0.25f, "the shadow map does not crawl under the camera");
        }

        // And the cascade split itself has to be sane: strictly increasing,
        // starting near the camera and ending at the shadow distance.
        eng::Scene s3 = full;
        s3.shadowCascades = 4;
        s3.shadowDistance = 40.0f;
        bool rising = true;
        for (int i = 0; i < 4; ++i)
            if (s3.CascadeSplit(i) >= s3.CascadeSplit(i + 1)) rising = false;
        std::printf("    splits: %.2f %.2f %.2f %.2f %.2f\n", s3.CascadeSplit(0),
                    s3.CascadeSplit(1), s3.CascadeSplit(2), s3.CascadeSplit(3),
                    s3.CascadeSplit(4));
        Check(rising, "the splits increase");
        Check(std::fabs(s3.CascadeSplit(4) - 40.0f) < 0.01f,
              "and the last one reaches the shadow distance");
        // A purely logarithmic split would put the first one absurdly close;
        // a purely uniform one would waste the near cascade on distance.
        Check(s3.CascadeSplit(1) > 1.0f && s3.CascadeSplit(1) < 10.0f,
              "the first split is somewhere useful");
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

    // --- deferred against forward --------------------------------------------
    //
    // The claim deferred shading makes is that it produces the SAME picture as
    // forward, more cheaply when there are many lights. That is a claim the
    // renderer can check on itself, and this scene is the right place: several
    // lights, three of them shadow-casting.
    //
    // Both paths call one ShadeSurface in the shader rather than two copies of
    // the lighting model, so agreement there is structural. The plumbing around
    // it is not shared at all: reconstructing a world position from depth,
    // round-tripping a normal through a texture, uploading the light list from
    // a second call site. Each is a chance to be a few percent wrong in a way
    // that only shows when the two are put side by side.
    {
        std::printf("\ndeferred shading\n");
        // Deferred cannot multisample -- storing and lighting every sample is
        // four times the memory and four times the lighting, which is the thing
        // deferring was for. So the comparison runs forward at ONE sample too,
        // or it would be measuring MSAA instead.
        std::string derr;
        auto plain = eng::Renderer::Create(*dev, kFmt, derr, 1);
        if (!plain) { std::fprintf(stderr, "FAIL: %s\n", derr.c_str()); return 1; }
        const gallery::Assets da = gallery::Build(*dev, *plain, derr);
        Check(da.ok, "a single-sampled renderer builds the same scene");

        const eng::rhi::TextureId dshadow = dev->CreateShadowMap(2048);
        const eng::rhi::TextureId gb0 =
            dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
        const eng::rhi::TextureId gb1 =
            dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
        // SAMPLEABLE: the lighting pass reads it back to rebuild the world
        // position. An ordinary depth target is memoryless and cannot be read.
        const eng::rhi::TextureId gdepth = dev->CreateDepthTarget(kW, kH, true);
        // The forward comparison gets its OWN depth. Sharing gdepth was the
        // obvious saving and it silently destroys the G-buffer's depth: a
        // colour pass does not keep its depth attachment, so the forward run
        // leaves it undefined and the relight below reads nothing.
        const eng::rhi::TextureId fdepth = dev->CreateDepthTarget(kW, kH);
        const eng::rhi::TextureId lit_hdr =
            dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
        const eng::rhi::TextureId dout = dev->CreateRenderTarget(kW, kH, kFmt, true);
        Check(Valid(gb0) && Valid(gb1) && Valid(gdepth) && Valid(lit_hdr) &&
                  Valid(dout),
              "the g-buffer targets were created");

        eng::Scene s = gallery::MakeScene(da, 0.0f);
        s.camera.eye = kEye;
        s.camera.target = kTarget;
        // The visible bulbs are Shading::Flat -- unlit, deliberately, so they
        // read as the light source rather than as small balls near one. A
        // G-buffer has no way to express "do not light this": every pixel in it
        // goes through the same BRDF. In a full pipeline they would be drawn
        // FORWARD after the lighting pass, alongside the transparent geometry.
        //
        // They are dropped from BOTH renders here, because leaving them in
        // would make this measure that known gap rather than the thing under
        // test -- whether the deferred plumbing reproduces the lit surfaces.
        const std::size_t all_instances = s.instances.size();
        s.instances.erase(
            std::remove_if(s.instances.begin(), s.instances.end(),
                           [&](const eng::Instance& i) {
                               return i.material.v == da.lamp_mat.v;
                           }),
            s.instances.end());
        std::printf("    %zu instances, %zu of them lit\n", all_instances,
                    s.instances.size());
        Check(s.instances.size() + 3 == all_instances,
              "the three unlit bulbs were set aside");

        std::vector<std::uint8_t> fwd_px, def_px;
        eng::RenderStats fwd_stats, def_stats;

        const auto shadow_passes = [&](eng::RenderGraph& g) {
            eng::RenderGraph::Pass a;
            a.name = "spot shadows";
            a.depth = plain->ShadowAtlas();
            a.clear_depth = 0.0f;
            a.keep_depth = true;
            a.execute = [&](eng::rhi::Encoder& e) { plain->DrawLightShadows(e, s); };
            g.AddPass(std::move(a));
            eng::RenderGraph::Pass b;
            b.name = "shadow";
            b.depth = dshadow;
            b.clear_depth = 0.0f;
            b.keep_depth = true;
            b.execute = [&](eng::rhi::Encoder& e) { plain->DrawShadow(e, s); };
            g.AddPass(std::move(b));
        };

        // Run the whole comparison TWICE, once with a single shadow box and
        // once with cascades. The cascade block is a second uniform binding
        // that the deferred lighting pass has to fill for itself, and with a
        // single box the shader never reads it -- so a version that forgot to
        // upload it would pass every check here and fail the moment a scene
        // turned cascades on. An interior room does not need cascades; this
        // turns them on to exercise the path, not because the scene wants them.
        const auto compare = [&](const char* label) {
            // DEFERRED FIRST, and the order is not arbitrary. Both paths upload
            // the light list and the cascade block into the same per-frame ring
            // buffer. Running forward first leaves that data in place, and a
            // deferred pass that forgot to upload its own would read the forward
            // pass's and look perfectly correct -- while a deferred-only
            // application rendered a black screen. Running deferred first means it
            // can only see what it uploaded itself.
            {   // deferred
                eng::RenderGraph g;
                shadow_passes(g);
                eng::RenderGraph::Pass p;
                p.name = "gbuffer";
                p.color = gb0;
                p.extra_colors = {gb1};
                p.depth = gdepth;
                // Cleared to zero, normals included. The lighting pass skips those
                // pixels on depth, so what is in them never reaches the screen --
                // but a NaN here would survive the skip and poison the bloom chain,
                // which is a bug this project has already had once.
                p.clear_color[0] = 0.0f; p.clear_color[1] = 0.0f;
                p.clear_color[2] = 0.0f; p.clear_color[3] = 0.0f;
                p.clear_depth = 0.0f;
                // KEEP the depth. A depth buffer is normally transient and thrown
                // away at end of pass, and here it is an input to the next one --
                // without this the lighting pass reads zeros, decides every pixel
                // is background, and outputs a black frame with no error anywhere.
                p.keep_depth = true;
                p.execute = [&](eng::rhi::Encoder& e) {
                    plain->DrawGBuffer(e, s, kW, kH);
                };
                g.AddPass(std::move(p));

                eng::RenderGraph::Pass l;
                l.name = "deferred light";
                l.color = lit_hdr;
                l.reads = {gb0, gb1, gdepth, dshadow, plain->ShadowAtlas()};
                l.clear_color[0] = 0.018f; l.clear_color[1] = 0.020f;
                l.clear_color[2] = 0.028f; l.clear_color[3] = 1.0f;
                l.execute = [&](eng::rhi::Encoder& e) {
                    plain->DrawDeferredLight(e, s, kW, kH, gb0, gb1, gdepth, dshadow);
                };
                g.AddPass(std::move(l));

                eng::RenderGraph::Pass c;
                c.name = "composite";
                c.color = dout;
                c.reads = {lit_hdr};
                c.execute = [&](eng::rhi::Encoder& e) {
                    plain->DrawComposite(e, lit_hdr, {}, {}, 0.0f, 0.0f);
                };
                g.AddPass(std::move(c));
                std::string e;
                Check(g.Compile(e), "the deferred graph compiles");
                dev->BeginFrame();
                g.Execute(*dev);
                std::string w;
                Check(dev->CommitAndWait(w), "the deferred frame submits");
                def_stats = plain->LastStats();
                def_px.assign(std::size_t(kW) * kH * 4, 0);
                Check(dev->ReadPixels(dout, kW, kH, def_px),
                      "the deferred frame reads back");
            }

            {   // forward, one sample
                eng::RenderGraph g;
                shadow_passes(g);
                eng::RenderGraph::Pass p;
                p.name = "scene";
                p.color = lit_hdr;
                p.depth = fdepth;
                p.clear_color[0] = 0.018f; p.clear_color[1] = 0.020f;
                p.clear_color[2] = 0.028f; p.clear_color[3] = 1.0f;
                p.clear_depth = 0.0f;
                p.reads = {dshadow, plain->ShadowAtlas()};
                p.execute = [&](eng::rhi::Encoder& e) {
                    plain->DrawScene(e, s, kW, kH, dshadow);
                };
                g.AddPass(std::move(p));
                eng::RenderGraph::Pass c;
                c.name = "composite";
                c.color = dout;
                c.reads = {lit_hdr};
                c.execute = [&](eng::rhi::Encoder& e) {
                    plain->DrawComposite(e, lit_hdr, {}, {}, 0.0f, 0.0f);
                };
                g.AddPass(std::move(c));
                std::string e;
                Check(g.Compile(e), "the forward graph compiles");
                dev->BeginFrame();
                g.Execute(*dev);
                std::string w;
                Check(dev->CommitAndWait(w), "the forward frame submits");
                fwd_stats = plain->LastStats();
                fwd_px.assign(std::size_t(kW) * kH * 4, 0);
                Check(dev->ReadPixels(dout, kW, kH, fwd_px),
                      "the forward frame reads back");
            }

            // THE LIGHTING PASS ON ITS OWN. Re-light the G-buffer already in
            // memory, in a later frame, without re-running the geometry pass.
            //
            // This is what shows DrawDeferredLight is self-sufficient. Within one
            // frame it cannot be shown: the G-buffer pass uploads the light list
            // and the cascade block to the same per-frame slot, so a lighting pass
            // that uploaded nothing would read that and look perfect. A later frame
            // is a different slot, and there is nothing there to inherit.
            //
            // It is also a real thing to want -- re-lighting a G-buffer without
            // re-rasterising is the whole basis of light-count scaling.
            {
                eng::RenderGraph g;
                // Written last frame, not this one.
                g.Import(gb0);
                g.Import(gb1);
                g.Import(gdepth);
                g.Import(dshadow);
                g.Import(plain->ShadowAtlas());
                eng::RenderGraph::Pass l;
                l.name = "relight";
                // Into the HDR target and THEN through the composite, exactly as
                // the deferred render did. Writing the lighting result straight
                // into the 8-bit output would skip the tone map, and linear
                // radiance clipped to a display range is not a dimmer version of
                // the right picture -- it is a different one.
                l.color = lit_hdr;
                l.reads = {gb0, gb1, gdepth, dshadow, plain->ShadowAtlas()};
                l.clear_color[0] = 0.018f; l.clear_color[1] = 0.020f;
                l.clear_color[2] = 0.028f; l.clear_color[3] = 1.0f;
                l.execute = [&](eng::rhi::Encoder& e) {
                    plain->DrawDeferredLight(e, s, kW, kH, gb0, gb1, gdepth, dshadow);
                };
                g.AddPass(std::move(l));
                eng::RenderGraph::Pass rc;
                rc.name = "composite";
                rc.color = dout;
                rc.reads = {lit_hdr};
                rc.execute = [&](eng::rhi::Encoder& e) {
                    plain->DrawComposite(e, lit_hdr, {}, {}, 0.0f, 0.0f);
                };
                g.AddPass(std::move(rc));
                std::string e;
                Check(g.Compile(e), "the relight graph compiles");
                // ONE frame on, and the count matters. The uniform ring has
                // kFramesInFlight slots and cycles: the deferred render took the
                // first, the forward render the second, so the next frame is the
                // third -- one this renderer has never written. Running TWO frames
                // instead wraps straight back onto the deferred render's slot and
                // the pass inherits its upload again, which is exactly the thing
                // this is here to rule out. It was two, and it proved nothing.
                static_assert(eng::rhi::kFramesInFlight == 3,
                              "the frame count below assumes a three-slot ring");
                dev->BeginFrame();
                g.Execute(*dev);
                std::string w;
                Check(dev->CommitAndWait(w), "the relight frame submits");
                std::vector<std::uint8_t> re_px(std::size_t(kW) * kH * 4, 0);
                Check(dev->ReadPixels(dout, kW, kH, re_px), "the relight frame reads back");
                long long re_sum = 0;
                int re_worst = 0;
                for (std::size_t i = 0; i + 3 < re_px.size(); i += 4) {
                    for (int c = 0; c < 3; ++c) {
                        re_sum += re_px[i + std::size_t(c)];
                        re_worst = std::max(re_worst,
                                            std::abs(int(re_px[i + std::size_t(c)]) -
                                                     int(def_px[i + std::size_t(c)])));
                    }
                }
                std::printf("    relit from the same g-buffer two frames later: "
                            "worst channel differs by %d\n", re_worst);
                Check(re_worst == 0, "the lighting pass needs nothing from the geometry pass");
                Check(re_sum > 4000000, "and the relit image is still lit");
            }

            std::printf("    forward drew %d instances, the g-buffer pass %d\n",
                        fwd_stats.draws, def_stats.draws);
            Check(def_stats.draws > 0, "the g-buffer pass drew something");
            Check(def_stats.draws == fwd_stats.draws,
                  "both paths drew the same geometry");

            // The comparison. Two paths cannot be bit-identical here and it would
            // be wrong to demand it: the normal makes a round trip through a
            // half-float texture, and the world position is REBUILT from a depth
            // buffer rather than interpolated across a triangle. Both cost a
            // fraction of a level almost everywhere.
            //
            // So the bound is the same shape as the one the jpeg tests use: the
            // MEAN must be small, and every large difference must sit on an EDGE.
            // A large difference in a flat region is not reconstruction error --
            // it is a light applied twice, a shadow sampled from the wrong place,
            // or a normal decoded wrong, and those are exactly the bugs a loose
            // worst-case bound would hide.
            const auto at_edge = [&](int x, int y) {
                const std::size_t here = (std::size_t(y) * kW + x) * 4;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nx = x + dx, ny = y + dy;
                        if (nx < 0 || ny < 0 || nx >= kW || ny >= kH) continue;
                        const std::size_t q = (std::size_t(ny) * kW + nx) * 4;
                        for (int c = 0; c < 3; ++c)
                            if (std::abs(int(fwd_px[q + std::size_t(c)]) -
                                         int(fwd_px[here + std::size_t(c)])) > 40)
                                return true;
                    }
                return false;
            };

            long long sum = 0, fwd_sum = 0, def_sum = 0;
            int worst = 0, large = 0, large_in_flat = 0, lit_pixels = 0, flat = 0;
            for (int y = 0; y < kH; ++y)
                for (int x = 0; x < kW; ++x) {
                    const std::size_t i = (std::size_t(y) * kW + x) * 4;
                    int here = 0;
                    for (int c = 0; c < 3; ++c)
                        here = std::max(here,
                                        std::abs(int(fwd_px[i + std::size_t(c)]) -
                                                 int(def_px[i + std::size_t(c)])));
                    sum += here;
                    worst = std::max(worst, here);
                    const bool edge = at_edge(x, y);
                    if (!edge) ++flat;
                    if (here > 8) {
                        ++large;
                        if (!edge) ++large_in_flat;
                    }
                    const int b = fwd_px[i] + fwd_px[i + 1] + fwd_px[i + 2];
                    if (b > 24) ++lit_pixels;
                    fwd_sum += b;
                    def_sum += def_px[i] + def_px[i + 1] + def_px[i + 2];
                }
            const double mean = double(sum) / (double(kW) * kH);
            std::printf("    [%s] forward vs deferred: mean %.3f, worst %d, "
                        "%d pixels over 8 and %d of those in flat regions\n",
                        label, mean, worst, large, large_in_flat);
            std::printf("    %d%% of the image is flat, %d pixels are lit\n",
                        100 * flat / (kW * kH), lit_pixels);
            std::printf("    total brightness: forward %lld, deferred %lld (%.2f%%)\n",
                        fwd_sum, def_sum, 100.0 * double(def_sum) / double(fwd_sum));
            Check(mean < 0.2, "deferred matches forward on average");
            Check(large_in_flat == 0, "every large difference is on an edge");
            // Without these two the whole comparison would pass on a pair of black
            // images, or on a pair where both paths lost the same light.
            Check(lit_pixels > 100000, "the comparison was made on a lit image");
            Check(flat > (kW * kH) / 2, "and most of it is flat, so the edge rule bites");
            // Without this the whole comparison would pass on two black images.
            Check(fwd_sum > 4000000 && def_sum > 4000000, "both images are lit");
        };

        compare("one shadow box");
        s.shadowCascades = 3;
        s.shadowDistance = 22.0f;
        compare("three cascades");
    }

    // --- ray-traced shadows --------------------------------------------------
    //
    // A shadow map answers "is this lit" from a grid that has nothing to do
    // with the camera's, so it has a resolution, a bias, and a box outside
    // which nothing casts. A ray asks the geometry.
    //
    // The scene here is deliberately trivial, because a trivial scene has an
    // ANALYTIC answer: a unit sphere three metres above a floor, lit from
    // straight overhead, casts a disc of radius exactly 1 centred exactly under
    // it. So the test does not compare two renderers and hope -- it checks
    // where the shadow is against where geometry says it must be.
    {
        std::printf("\nray-traced shadows\n");
        Check(renderer->RaytracingAvailable(),
              "the device reports hardware ray tracing");
        if (renderer->RaytracingAvailable()) {
            std::string rerr;
            auto rt = eng::Renderer::Create(*dev, kFmt, rerr, 1);
            if (!rt) { std::fprintf(stderr, "FAIL: %s\n", rerr.c_str()); return 1; }

            const eng::MeshHandle floor_mesh =
                rt->UploadMesh(eng::MakeBox(eng::Vec3{12.0f, 0.1f, 12.0f},
                                            eng::Vec4{0.8f, 0.8f, 0.8f, 1.0f}));
            const eng::MeshHandle ball_mesh = rt->UploadMesh(eng::MakeUVSphere(
                1.0f, 48, 64, eng::Vec4{0.9f, 0.4f, 0.3f, 1.0f},
                eng::Vec4{0.9f, 0.4f, 0.3f, 1.0f}));
            eng::MaterialDesc md;
            md.shading = eng::Shading::Lit;
            const eng::MaterialHandle mat = rt->CreateMaterial(md, rerr);
            Check(Valid(floor_mesh) && Valid(ball_mesh) && Valid(mat),
                  "the ray tracing scene's meshes uploaded");

            eng::Scene rs;
            // Straight down, so the shadow lands directly beneath the sphere
            // and its position is arithmetic rather than a projection.
            rs.lightDir = eng::Vec4{0.0f, 1.0f, 0.0f, 0.0f};
            rs.lightColor = eng::Vec4{1.0f, 1.0f, 1.0f, 1.0f};
            rs.shadowExtent = 14.0f;
            rs.shadowDistance = 40.0f;
            rs.camera.eye = eng::Vec3{6.0f, 7.0f, 9.0f};
            rs.camera.target = eng::Vec3{0.0f, 0.5f, 0.0f};
            {
                eng::Instance f;
                f.mesh = floor_mesh;
                f.material = mat;
                f.model = eng::Mat4::Translation(eng::Vec3{0.0f, -0.1f, 0.0f});
                rs.instances.push_back(f);
                eng::Instance b;
                b.mesh = ball_mesh;
                b.material = mat;
                b.model = eng::Mat4::Translation(eng::Vec3{0.0f, 3.0f, 0.0f});
                rs.instances.push_back(b);
            }

            Check(rt->BuildSceneAccel(rs, rerr),
                  "the scene's acceleration structures built");
            if (!rerr.empty()) std::fprintf(stderr, "    %s\n", rerr.c_str());

            // DEDUPLICATION, which is the reason a two-level structure exists
            // at all. Add eight more spheres and the count of bottom-level
            // structures must not move: they share one mesh, so they share one
            // BVH and differ only by a transform. Building one each would look
            // identical on screen and cost build time and memory that nothing
            // in a rendered image could ever reveal.
            {
                eng::Scene many = rs;
                for (int i = 0; i < 8; ++i) {
                    eng::Instance b;
                    b.mesh = ball_mesh;
                    b.material = mat;
                    b.model = eng::Mat4::Translation(
                        eng::Vec3{float(i) * 1.7f - 6.0f, 5.0f, -4.0f});
                    many.instances.push_back(b);
                }
                const int before = rt->BlasBuilds();
                std::string e2;
                Check(rt->BuildSceneAccel(many, e2),
                      "a scene with nine spheres builds");
                std::printf("    %zu instances of 2 distinct meshes: %d "
                            "bottom-level structures built in total\n",
                            many.instances.size(), rt->BlasBuilds());
                Check(rt->BlasBuilds() == before,
                      "eight more instances of the same mesh build no new BVH");
                Check(rt->BlasBuilds() == 2, "one per distinct mesh, ever");
            }
            // Back to the two-object scene the geometry below is about.
            Check(rt->BuildSceneAccel(rs, rerr), "and rebuilds for the test scene");

            const eng::rhi::TextureId rgb0 =
                dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
            const eng::rhi::TextureId rgb1 =
                dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
            const eng::rhi::TextureId rdepth = dev->CreateDepthTarget(kW, kH, true);
            const eng::rhi::TextureId rmask =
                dev->CreateRenderTarget(kW, kH, kFmt, true);
            Check(Valid(rgb0) && Valid(rmask), "the ray tracing targets were created");

            eng::RenderGraph g;
            {
                eng::RenderGraph::Pass p;
                p.name = "gbuffer";
                p.color = rgb0;
                p.extra_colors = {rgb1};
                p.depth = rdepth;
                p.clear_depth = 0.0f;
                p.keep_depth = true;
                p.execute = [&](eng::rhi::Encoder& e) {
                    rt->DrawGBuffer(e, rs, kW, kH);
                };
                g.AddPass(std::move(p));
            }
            {
                eng::RenderGraph::Pass p;
                p.name = "ray shadows";
                p.color = rmask;
                p.reads = {rdepth, rgb1};
                // White: a pixel the pass never touches is UNSHADOWED, which is
                // the value that makes a later multiply a no-op. Clearing to
                // black would put the background in shadow.
                p.clear_color[0] = 1.0f; p.clear_color[1] = 1.0f;
                p.clear_color[2] = 1.0f; p.clear_color[3] = 1.0f;
                p.execute = [&](eng::rhi::Encoder& e) {
                    rt->DrawRayShadows(e, rs, kW, kH, rdepth, rgb1);
                };
                g.AddPass(std::move(p));
            }
            std::string ge;
            Check(g.Compile(ge), "the ray tracing graph compiles");
            if (!ge.empty()) std::fprintf(stderr, "    %s\n", ge.c_str());
            dev->BeginFrame();
            g.Execute(*dev);
            std::string we;
            Check(dev->CommitAndWait(we), "the ray tracing frame submits");
            if (!we.empty()) std::fprintf(stderr, "    %s\n", we.c_str());

            std::vector<std::uint8_t> mask(std::size_t(kW) * kH * 4, 0);
            Check(dev->ReadPixels(rmask, kW, kH, mask), "the shadow mask reads back");

            // Where a world point lands on screen, for this camera.
            const auto project = [&](eng::Vec3 w, int* px, int* py) {
                const eng::Vec4 clip = rs.camera.ViewProj(float(kW) / float(kH)) *
                                       eng::Vec4{w.x, w.y, w.z, 1.0f};
                *px = int((clip.x / clip.w * 0.5f + 0.5f) * float(kW));
                *py = int((1.0f - (clip.y / clip.w * 0.5f + 0.5f)) * float(kH));
            };
            const auto shadowed_at = [&](eng::Vec3 w) {
                int px = 0, py = 0;
                project(w, &px, &py);
                if (px < 1 || py < 1 || px >= kW - 1 || py >= kH - 1) return -1;
                // A small average, so one stray pixel on a triangle edge does
                // not decide the answer.
                int sum = 0;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx)
                        sum += mask[(std::size_t(py + dy) * kW + (px + dx)) * 4];
                return sum / 9;
            };

            // The disc has radius 1 at the origin. Inside it is dark, outside
            // is light, and the interesting part is that the boundary is where
            // the geometry says and not where a texel grid happens to fall.
            const int centre = shadowed_at(eng::Vec3{0, 0, 0});
            const int inside = shadowed_at(eng::Vec3{0.55f, 0, 0.55f});  // r=0.78
            const int just_out = shadowed_at(eng::Vec3{1.25f, 0, 0});
            const int far_out = shadowed_at(eng::Vec3{4.0f, 0, 0});
            const int behind = shadowed_at(eng::Vec3{0, 0, 4.0f});
            std::printf("    visibility under the sphere %d, at r=0.78 %d, at "
                        "r=1.25 %d, at r=4 %d, behind %d\n",
                        centre, inside, just_out, far_out, behind);
            Check(centre < 20, "directly under the sphere is in shadow");
            Check(inside < 20, "and so is the rest of the disc");
            Check(just_out > 235, "a quarter of a metre outside it is not");
            Check(far_out > 235 && behind > 235, "and neither is the open floor");

            // The RADIUS, measured rather than assumed. Walking outward along
            // the floor, the transition must happen at 1.0 -- that number is
            // the sphere's radius and nothing in the renderer knows it.
            float edge = -1.0f;
            for (int i = 0; i <= 400; ++i) {
                const float r = float(i) * 0.005f;
                if (shadowed_at(eng::Vec3{r, 0, 0}) > 128) { edge = r; break; }
            }
            std::printf("    the shadow's edge is at r = %.3f (the sphere's "
                        "radius is 1.000)\n", edge);
            Check(edge > 0.97f && edge < 1.06f,
                  "the shadow disc is exactly as wide as the sphere");

            // And it is a real image, not a uniform one: an all-black or
            // all-white mask would satisfy some of the above by accident.
            long long lit = 0, dark = 0;
            for (std::size_t i = 0; i + 3 < mask.size(); i += 4) {
                if (mask[i] > 200) ++lit;
                else if (mask[i] < 50) ++dark;
            }
            std::printf("    %lld pixels lit, %lld in shadow\n", lit, dark);
            Check(dark > 2000, "the shadow covers a real area");
            Check(lit > dark * 3, "and most of the frame is not in it");
        }
    }

    std::printf(g_failures == 0 ? "\ngallery_test: all checks passed\n"
                                : "\ngallery_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
