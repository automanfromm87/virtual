// Order-independent transparency, checked for the one property its name claims.
//
// "Looks like glass" is not the test. Alpha blending sorted back to front also
// looks like glass, and it is what this replaces. The claim is that the result
// does not depend on the order the surfaces were drawn -- so the test draws the
// same surfaces in different orders and requires the same pixels, and then
// shows the sorted path failing the identical check.
#include "engine/geometry/mesh.h"
#include "engine/render/renderer.h"
#include "engine/rhi/rhi.h"
#include "engine/scene/scene.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-60s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

constexpr int kW = 320, kH = 320;

}  // namespace

int main() {
    using namespace eng;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string error;
    auto dev = rhi::Device::Create(error);
    if (!dev) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }
    const auto kFmt = rhi::Format::RGBA8Unorm;
    auto r = Renderer::Create(*dev, kFmt, error, 1);
    if (!r) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }

    const MeshHandle pane =
        r->UploadMesh(MakeBox(Vec3{1.4f, 1.4f, 0.02f}, Vec4{1, 1, 1, 1}));

    // Three panes, INTERSECTING. That is the case an object sort cannot get
    // right: each pane is in front along part of its area and behind along the
    // rest, so no single ordering of the three is correct for every pixel.
    struct Pane {
        Vec3 colour;
        float alpha;
        Mat4 model;
    };
    const Pane panes[3] = {
        {{0.9f, 0.15f, 0.1f}, 0.5f,
         Mat4::Translation({-0.35f, 0.0f, 0.0f}) * Mat4::RotationY(0.55f)},
        {{0.1f, 0.8f, 0.2f}, 0.5f,
         Mat4::Translation({0.0f, 0.0f, 0.0f}) * Mat4::RotationY(-0.4f)},
        {{0.15f, 0.25f, 0.95f}, 0.5f,
         Mat4::Translation({0.35f, 0.0f, 0.0f}) * Mat4::RotationY(0.9f)},
    };

    std::vector<MaterialHandle> mats;
    for (const Pane& p : panes) {
        MaterialDesc md;
        md.base_color = Vec4{p.colour.x, p.colour.y, p.colour.z, p.alpha};
        md.roughness = 0.4f;
        md.transparent = true;
        md.cull = rhi::Cull::None;  // a pane is seen from both sides
        const MaterialHandle m = r->CreateMaterial(md, error);
        if (!Valid(m)) {
            std::fprintf(stderr, "FAIL: %s\n", error.c_str());
            return 1;
        }
        mats.push_back(m);
    }

    const auto build = [&](const int order[3]) {
        Scene s;
        s.camera.eye = Vec3{0.0f, 0.4f, 4.2f};
        s.camera.target = Vec3{0.0f, 0.0f, 0.0f};
        s.lightDir = Vec4{0.3f, 0.6f, 0.7f, 0.0f};
        s.lightColor = Vec4{2.4f, 2.4f, 2.4f, 1.0f};
        s.ambientSky = Vec3{0.05f, 0.05f, 0.06f};
        s.ambientGround = Vec3{0.02f, 0.02f, 0.02f};
        for (int k = 0; k < 3; ++k) {
            Instance in;
            in.mesh = pane;
            in.material = mats[std::size_t(order[k])];
            in.model = panes[order[k]].model;
            // The alpha lives in the vertex tint as well as the base colour:
            // the fragment stage takes its alpha from the interpolated vertex
            // colour, which is vertex colour times tint.
            in.tint = Vec4{1.0f, 1.0f, 1.0f, panes[order[k]].alpha};
            s.instances.push_back(in);
        }
        return s;
    };

    const rhi::TextureId hdr =
        dev->CreateRenderTarget(kW, kH, Renderer::kSceneFormat);
    const rhi::TextureId accum =
        dev->CreateRenderTarget(kW, kH, Renderer::kSceneFormat);
    const rhi::TextureId reveal =
        dev->CreateRenderTarget(kW, kH, Renderer::kSceneFormat);
    const rhi::TextureId ldr =
        dev->CreateRenderTarget(kW, kH, kFmt, /*cpu_readable=*/true);
    const rhi::TextureId depth = dev->CreateDepthTarget(kW, kH);
    std::vector<std::uint8_t> px(std::size_t(kW) * kH * 4);

    const auto render = [&](const Scene& scene, bool use_oit) -> bool {
        r->SetOrderIndependentTransparency(use_oit);
        dev->BeginFrame();
        {
            rhi::PassDesc pd;
            pd.color = hdr;
            pd.depth = depth;
            pd.clear_depth = 0.0f;
            auto e = dev->BeginPass(pd);
            r->DrawScene(e, scene, kW, kH, {});
            dev->EndPass();
        }
        if (use_oit) {
            {
                rhi::PassDesc pd;
                pd.color = accum;
                pd.extra_colors = {reveal};
                pd.depth = depth;
                pd.keep_depth = true;
                // ACCUMULATION starts at zero and REVEALAGE at one. Revealage
                // is a running product of (1 - alpha), so starting it at zero
                // makes every pixel fully covered before anything is drawn and
                // the resolve returns the average of nothing.
                //
                // One clear colour serves both attachments, so this is done as
                // two passes: clear revealage to white first, then accumulate
                // into it with the load flag set.
                for (int i = 0; i < 4; ++i) pd.clear_color[i] = 0.0f;
                auto e = dev->BeginPass(pd);
                (void)e;
                dev->EndPass();
            }
            {
                rhi::PassDesc pd;
                pd.color = reveal;
                for (int i = 0; i < 4; ++i) pd.clear_color[i] = 1.0f;
                auto e = dev->BeginPass(pd);
                (void)e;
                dev->EndPass();
            }
            {
                rhi::PassDesc pd;
                pd.color = accum;
                pd.extra_colors = {reveal};
                pd.depth = depth;
                pd.load = true;
                pd.keep_depth = true;
                auto e = dev->BeginPass(pd);
                r->DrawTransparentOit(e, scene, kW, kH, {});
                dev->EndPass();
            }
            {
                rhi::PassDesc pd;
                pd.color = hdr;
                pd.load = true;
                auto e = dev->BeginPass(pd);
                r->DrawOitResolve(e, accum, reveal);
                dev->EndPass();
            }
        }
        {
            rhi::PassDesc pd;
            pd.color = ldr;
            auto e = dev->BeginPass(pd);
            r->DrawComposite(e, hdr, {}, {}, 0.0f, /*vignette=*/0.0f);
            dev->EndPass();
        }
        if (!dev->CommitAndWait(error)) {
            std::fprintf(stderr, "FAIL: submit: %s\n", error.c_str());
            return false;
        }
        return dev->ReadPixels(ldr, kW, kH, px);
    };

    const int forward[3] = {0, 1, 2};
    const int backward[3] = {2, 1, 0};
    const int shuffled[3] = {1, 2, 0};

    const auto compare = [&](const std::vector<std::uint8_t>& a,
                             const std::vector<std::uint8_t>& b) {
        double worst = 0.0, sum = 0.0;
        int n = 0;
        for (int i = 0; i < kW * kH; ++i)
            for (int c = 0; c < 3; ++c, ++n) {
                const double d =
                    std::fabs(double(a[std::size_t(i) * 4 + c]) -
                              double(b[std::size_t(i) * 4 + c]));
                worst = std::max(worst, d);
                sum += d;
            }
        return std::pair<double, double>{worst, sum / n};
    };

    // ---------------------------------------------------------- sorted first --
    std::printf("the object sort, for reference\n");
    r->SetOrderIndependentTransparency(false);
    if (!render(build(forward), false)) return 1;
    const std::vector<std::uint8_t> sorted_a = px;
    if (!render(build(backward), false)) return 1;
    const std::vector<std::uint8_t> sorted_b = px;
    const auto [sorted_worst, sorted_mean] = compare(sorted_a, sorted_b);
    std::printf("    submitting the same three panes in two orders differs by "
                "%.0f at worst, %.2f on average\n",
                sorted_worst, sorted_mean);
    // THE SORT IS SUBMISSION-ORDER DEPENDENT HERE, and that is the point rather
    // than a surprise. It sorts by the object's centre depth, and these three
    // panes all have their centres at z = 0 -- so the key is identical for all
    // three, the sort has nothing to separate them, and whichever order they
    // arrived in survives. Two intersecting panes at the same depth is not an
    // exotic case; it is a window frame, a plant, a puddle on glass.
    //
    // This check was originally written the other way round, asserting the sort
    // did NOT depend on submission order. It does, by 78 levels out of 255.
    Check(sorted_worst > 20.0,
          "a back-to-front sort still depends on submission order when depths tie");

    // Coverage, so the checks below are not comparing two empty images.
    int lit = 0;
    for (int i = 0; i < kW * kH; ++i)
        if (px[std::size_t(i) * 4] > 12 || px[std::size_t(i) * 4 + 1] > 12) ++lit;
    std::printf("    the panes cover %d of %d pixels\n", lit, kW * kH);
    Check(lit > 8000, "and the panes are actually on screen");

    // ------------------------------------------------------------------- OIT --
    std::printf("\norder-independent transparency\n");
    if (!render(build(forward), true)) return 1;
    const std::vector<std::uint8_t> oit_a = px;
    if (!render(build(backward), true)) return 1;
    const std::vector<std::uint8_t> oit_b = px;
    if (!render(build(shuffled), true)) return 1;
    const std::vector<std::uint8_t> oit_c = px;

    const auto [ab_worst, ab_mean] = compare(oit_a, oit_b);
    const auto [ac_worst, ac_mean] = compare(oit_a, oit_c);
    std::printf("    three submission orders differ by %.0f and %.0f at worst\n",
                ab_worst, ac_worst);
    // ONE LEVEL, not zero. The two accumulation operations commute in ALGEBRA
    // but floating-point addition is not associative, so summing three
    // half-float terms in a different order lands on a different last bit --
    // which is one level out of 255 after the tone map, on a handful of pixels.
    //
    // Against the sort's 78 on the same scene, that is the whole claim: the
    // sort's answer depends on the order, and this one depends only on how the
    // hardware rounds.
    Check(ab_worst <= 1.0 && ac_worst <= 1.0,
          "every submission order gives the same pixels to within rounding");
    Check(sorted_worst > ab_worst * 20.0,
          "which is dozens of times steadier than the sort on the same scene");

    // And it must be DIFFERENT from not drawing them: a resolve that returned
    // nothing would also be perfectly order-independent.
    const auto [vs_sorted_worst, vs_sorted_mean] = compare(oit_a, sorted_a);
    std::printf("    against the sorted image: %.0f at worst, %.2f on average\n",
                vs_sorted_worst, vs_sorted_mean);
    Check(vs_sorted_mean > 0.5,
          "and it is an approximation of the sort, not a copy of it");
    Check(vs_sorted_mean < 30.0, "but a close one, not a different picture");

    // THE CASE THE SORT CANNOT DO. Rotate the whole arrangement so the three
    // panes' depth ordering changes across the frame. An object sort picks one
    // order per pane per frame and gets it wrong over part of each; OIT has no
    // order to get wrong, so its answer changes smoothly as the camera turns
    // while the sorted one jumps when two panes swap in the sort.
    std::printf("\nturning the camera through a sort swap\n");
    double worst_sorted_jump = 0.0, worst_oit_jump = 0.0;
    std::vector<std::uint8_t> prev_sorted, prev_oit;
    for (int step = 0; step <= 12; ++step) {
        const float a = -0.35f + 0.7f * float(step) / 12.0f;
        Scene s = build(forward);
        s.camera.eye = Vec3{std::sin(a) * 4.2f, 0.4f, std::cos(a) * 4.2f};
        if (!render(s, false)) return 1;
        if (step > 0) {
            const auto [w, m] = compare(prev_sorted, px);
            worst_sorted_jump = std::max(worst_sorted_jump, m);
        }
        prev_sorted = px;
        if (!render(s, true)) return 1;
        if (step > 0) {
            const auto [w, m] = compare(prev_oit, px);
            worst_oit_jump = std::max(worst_oit_jump, m);
        }
        prev_oit = px;
    }
    std::printf("    largest frame-to-frame change: sorted %.3f, OIT %.3f\n",
                worst_sorted_jump, worst_oit_jump);
    Check(worst_oit_jump < worst_sorted_jump,
          "OIT changes more smoothly than the sort as the camera turns");

    std::printf(g_failures == 0 ? "\noit_test: all checks passed\n"
                                : "\noit_test: %d FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
