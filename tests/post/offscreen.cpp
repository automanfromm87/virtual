// The post-processing stack, measured.
//
// Post effects are the part of a renderer most often shipped on the strength of
// "it looks better now", and the part where that judgement is least reliable --
// every one of them changes the whole image, so any of them can hide any other
// being wrong. Auto-exposure that never converges just looks like a moody
// scene. A depth of field whose focal plane is at the wrong distance looks like
// an artistic choice. Fog with no height falloff looks like fog.
//
// So each one here is reduced to a number that has a known direction: exposure
// must converge on mapping the average to middle grey, fog must affect distance
// and not the near field, blur must reduce high-frequency energy and nothing
// else, and an identity grade must change nothing at all.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "engine/geometry/mesh.h"
#include "engine/render/post.h"
#include "engine/render/renderer.h"
#include "engine/rhi/rhi.h"
#include "engine/scene/scene.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

constexpr int kW = 256, kH = 256;

float Mean(const std::vector<std::uint8_t>& px, int ch, int x0, int y0, int x1,
           int y1) {
    double sum = 0.0;
    int n = 0;
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) {
            sum += px[(std::size_t(y) * kW + x) * 4 + ch];
            ++n;
        }
    return n > 0 ? float(sum / n) : 0.0f;
}

// The mean absolute difference between horizontally adjacent pixels: a direct
// measure of how much high-frequency detail an image has. A blur removes it and
// nothing else in this file does, which is what makes it the right instrument
// for depth of field and motion blur.
float Detail(const std::vector<std::uint8_t>& px, int x0, int y0, int x1, int y1) {
    double sum = 0.0;
    int n = 0;
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1 - 1; ++x) {
            const std::size_t a = (std::size_t(y) * kW + x) * 4;
            sum += std::abs(int(px[a]) - int(px[a + 4]));
            ++n;
        }
    return n > 0 ? float(sum / n) : 0.0f;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string error;
    auto dev = eng::rhi::Device::Create(error);
    if (!dev) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
    const auto kOut = eng::rhi::Format::RGBA8Unorm;
    auto r = eng::Renderer::Create(*dev, kOut, error, 1);
    if (!r) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
    auto post = eng::PostStack::Create(*dev, error);
    if (!post) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    // A CHECKERBOARD FLOOR receding to the horizon. Chosen for two properties:
    // it has detail at every distance, so a blur is measurable wherever the
    // focal plane is put, and its far end is far away, so fog has somewhere to
    // act.
    const eng::MeshHandle tile = r->UploadMesh(
        eng::MakeBox(eng::Vec3{0.5f, 0.05f, 0.5f}, eng::Vec4{1, 1, 1, 1}));
    eng::MaterialDesc light_md, dark_md;
    light_md.shading = eng::Shading::Lit;
    light_md.base_color = eng::Vec4{0.85f, 0.85f, 0.85f, 1.0f};
    dark_md.shading = eng::Shading::Lit;
    dark_md.base_color = eng::Vec4{0.08f, 0.08f, 0.08f, 1.0f};
    const eng::MaterialHandle light_mat = r->CreateMaterial(light_md, error);
    const eng::MaterialHandle dark_mat = r->CreateMaterial(dark_md, error);
    if (!eng::Valid(tile) || !eng::Valid(light_mat) || !eng::Valid(dark_mat)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }

    eng::Scene scene;
    scene.lightDir = eng::Vec4{0.3f, 0.9f, 0.3f, 0.0f};
    scene.lightColor = eng::Vec4{3.0f, 3.0f, 3.0f, 1.0f};
    scene.ambientSky = eng::Vec3{0.25f, 0.28f, 0.34f};
    scene.ambientGround = eng::Vec3{0.06f, 0.06f, 0.06f};
    for (int z = 0; z < 90; ++z)
        for (int x = -8; x <= 8; ++x) {
            eng::Instance inst;
            inst.mesh = tile;
            inst.material = ((x + z) & 1) ? light_mat : dark_mat;
            inst.model = eng::Mat4::Translation(
                eng::Vec3{float(x), 0.0f, -float(z)});
            scene.instances.push_back(inst);
        }
    scene.camera.eye = eng::Vec3{0.0f, 1.2f, 2.0f};
    scene.camera.target = eng::Vec3{0.0f, 0.6f, -40.0f};

    const eng::rhi::TextureId hdr =
        dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
    const eng::rhi::TextureId hdr2 =
        dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
    const eng::rhi::TextureId out =
        dev->CreateRenderTarget(kW, kH, kOut, /*cpu_readable=*/true);
    // SAMPLEABLE depth, which the ordinary one is not: every effect here reads
    // it, and a memoryless depth buffer stops existing at the end of its pass.
    const eng::rhi::TextureId depth =
        dev->CreateDepthTarget(kW, kH, /*sampleable=*/true);
    std::vector<std::uint8_t> px(std::size_t(kW) * kH * 4);

    // Renders the scene into `hdr`, runs whatever `effects` adds, composites,
    // and reads back. Everything is checked; a silently failed submit would
    // leave the previous frame's pixels and every assertion below would be
    // measuring the wrong image.
    const auto frame = [&](float dt, bool want_velocity,
                           const std::function<eng::rhi::TextureId()>& effects)
        -> bool {
        scene.camera.jitter = post->Jitter();
        post->BeginFrame(scene.camera, kW, kH, dt);
        scene.camera.jitter = post->Jitter();

        dev->BeginFrame();
        {
            eng::rhi::PassDesc pd;
            pd.color = hdr;
            pd.depth = depth;
            pd.keep_depth = true;
            auto e = dev->BeginPass(pd);
            r->DrawScene(e, scene, kW, kH, {});
            dev->EndPass();
        }
        if (want_velocity) {
            auto e = dev->BeginCompute();
            post->ComputeVelocity(e, depth);
            post->MeterExposure(e, hdr);
            dev->EndCompute();
        }
        const eng::rhi::TextureId final_hdr = effects ? effects() : hdr;
        {
            eng::rhi::PassDesc pd;
            pd.color = out;
            auto e = dev->BeginPass(pd);
            r->DrawComposite(e, final_hdr);
            dev->EndPass();
        }
        if (!dev->CommitAndWait(error)) {
            std::fprintf(stderr, "FAIL submit: %s\n", error.c_str());
            return false;
        }
        post->EndFrame();
        return dev->ReadPixels(out, kW, kH, px);
    };

    {
        // THE TONE MAP KEEPS THE HUE. This is the one property a per-channel
        // curve cannot have, and the reason the composite no longer uses one.
        //
        // A saturated light on a white floor makes the scene's hue exactly the
        // light's hue, so the input saturation is known to the digit: a light
        // of {0.19, 0.38, 0.15} is (0.38 - 0.15) / 0.38 = 0.605 saturated,
        // whatever intensity it is scaled to. Everything the curve does to that
        // number is the curve's own doing.
        std::printf("the tone map keeps the hue as the image brightens\n");
        const eng::Vec4 saved_light = scene.lightColor;
        const eng::Vec3 saved_sky = scene.ambientSky, saved_ground = scene.ambientGround;
        // Ambient off: a grey ambient term would dilute the hue before the
        // curve ever saw it, and this has to measure the curve.
        scene.ambientSky = eng::Vec3{0.0f, 0.0f, 0.0f};
        scene.ambientGround = eng::Vec3{0.0f, 0.0f, 0.0f};

        // Mean saturation of the pixels whose peak channel lands in a band.
        // Banding rather than sampling a fixed spot because the floor is a
        // checkerboard: this selects "upper midtones, not clipped" wherever
        // they happen to be, and a clipped pixel has no hue left to measure by
        // definition.
        //
        // MEASURED AFTER UNDOING THE DISPLAY GAMMA, which is not a detail. The
        // composite writes pow(1/2.2)-encoded bytes, and that curve lifts the
        // low channels much more than the high ones, so the same colour scores
        // a lower saturation encoded than it does linear -- 61% against 100% at
        // this brightness, when this was first written against a linear
        // reference. That number said nothing about the tone map; it was the
        // display transfer function being counted twice.
        const auto band_saturation = [&](int lo, int hi) {
            double sum = 0.0;
            int n = 0;
            for (std::size_t i = 0; i + 3 < px.size(); i += 4) {
                const int peak8 = std::max(int(px[i]),
                                           std::max(int(px[i + 1]), int(px[i + 2])));
                if (peak8 < lo || peak8 > hi) continue;
                float lin[3];
                for (int c = 0; c < 3; ++c)
                    lin[c] = std::pow(float(px[i + c]) / 255.0f, 2.2f);
                const float peak = std::max(lin[0], std::max(lin[1], lin[2]));
                const float trough = std::min(lin[0], std::min(lin[1], lin[2]));
                if (peak <= 1e-6f) continue;
                sum += double((peak - trough) / peak);
                ++n;
            }
            return n > 0 ? float(sum / double(n)) : -1.0f;
        };

        constexpr float kInputSaturation = 0.605f;
        float dim = 0.0f, bright = 0.0f;
        for (int pass = 0; pass < 2; ++pass) {
            const float k = pass == 0 ? 1.6f : 9.0f;
            scene.lightColor = eng::Vec4{0.19f * k, 0.38f * k, 0.15f * k, 1.0f};
            if (!frame(0.016f, false, nullptr)) return 1;
            const float kept = band_saturation(pass == 0 ? 60 : 190,
                                               pass == 0 ? 140 : 245);
            (pass == 0 ? dim : bright) = kept / kInputSaturation;
            std::printf("    %-14s peak in %3d..%3d, saturation kept %.0f%%\n",
                        pass == 0 ? "midtones:" : "upper mids:",
                        pass == 0 ? 60 : 190, pass == 0 ? 140 : 245,
                        100.0 * double(pass == 0 ? dim : bright));
        }
        // The midtones are the easy case -- every curve passes there.
        Check(dim > 0.85f, "a dim image comes through with its colour");
        // THE UPPER MIDTONES ARE THE CASE THAT MATTERS, and where a per-channel
        // curve falls apart: it keeps 39% of the saturation at this brightness,
        // measured, which is what turned world2's grass grey. Nothing here is
        // clipped -- the band stops at 245 - so any loss is the curve
        // dissolving the hue rather than the image running out of range.
        Check(bright > 0.80f, "and so does a bright one, which is the hard case");

        scene.lightColor = saved_light;
        scene.ambientSky = saved_sky;
        scene.ambientGround = saved_ground;
    }

    {
        std::printf("\nan identity grade changes nothing\n");
        if (!frame(0.016f, false, nullptr)) return 1;
        const std::vector<std::uint8_t> before = px;
        eng::ColorGrade g;  // all defaults
        r->SetGrade(g);
        if (!frame(0.016f, false, nullptr)) return 1;
        int diff = 0;
        for (std::size_t i = 0; i < px.size(); ++i)
            if (px[i] != before[i]) ++diff;
        Check(diff == 0, "the default ColorGrade is bit-for-bit a no-op");
    }

    {
        std::printf("\nthe grade does what it says\n");
        if (!frame(0.016f, false, nullptr)) return 1;
        const float base_mean = Mean(px, 0, 0, 0, kW, kH);
        // A "grey" test on the average of the channels would pass for a colour
        // that is green; comparing the channels to each other would not.
        eng::ColorGrade g;
        g.saturation = 0.0f;
        r->SetGrade(g);
        if (!frame(0.016f, false, nullptr)) return 1;
        float max_spread = 0.0f;
        for (int y = 0; y < kH; y += 7)
            for (int x = 0; x < kW; x += 7) {
                const std::size_t i = (std::size_t(y) * kW + x) * 4;
                const int lo = std::min({px[i], px[i + 1], px[i + 2]});
                const int hi = std::max({px[i], px[i + 1], px[i + 2]});
                max_spread = std::max(max_spread, float(hi - lo));
            }
        std::printf("    saturation 0: largest channel spread %.0f\n", max_spread);
        Check(max_spread <= 1.0f, "zero saturation makes every pixel grey");

        g = eng::ColorGrade{};
        g.gain = eng::Vec3{1.4f, 1.4f, 1.4f};
        r->SetGrade(g);
        if (!frame(0.016f, false, nullptr)) return 1;
        const float gained = Mean(px, 0, 0, 0, kW, kH);
        std::printf("    gain 1.4: mean %.1f -> %.1f\n", base_mean, gained);
        Check(gained > base_mean + 4.0f, "gain brightens");

        // CONTRAST ABOUT A PIVOT. The test that distinguishes a real contrast
        // control from a multiply: raising contrast must make darks darker AND
        // brights brighter, so the mean barely moves while the spread grows.
        // A multiply raises both and the mean with them.
        g = eng::ColorGrade{};
        g.contrast = 1.6f;
        r->SetGrade(g);
        if (!frame(0.016f, false, nullptr)) return 1;
        float lo_sum = 0.0f, hi_sum = 0.0f;
        int lo_n = 0, hi_n = 0;
        for (int y = 0; y < kH; y += 3)
            for (int x = 0; x < kW; x += 3) {
                const std::size_t i = (std::size_t(y) * kW + x) * 4;
                if (px[i] < 110) { lo_sum += px[i]; ++lo_n; }
                if (px[i] > 145) { hi_sum += px[i]; ++hi_n; }
            }
        std::printf("    contrast 1.6: dark mean %.1f, bright mean %.1f\n",
                    lo_n ? lo_sum / lo_n : 0.0f, hi_n ? hi_sum / hi_n : 0.0f);
        Check(lo_n > 0 && hi_n > 0 && lo_sum / lo_n < 90.0f && hi_sum / hi_n > 165.0f,
              "contrast pushes both ends away from the pivot");
        r->SetGrade(eng::ColorGrade{});
    }

    {
        std::printf("\nfog acts on distance, not on everything\n");
        if (!frame(0.016f, false, nullptr)) return 1;
        const float near_before = Mean(px, 2, 96, 200, 160, 250);
        const float far_before = Mean(px, 2, 96, 118, 160, 132);

        post->config.fog = true;
        post->config.fog_color = eng::Vec3{0.10f, 0.30f, 0.95f};  // unmistakable
        post->config.fog_density = 0.05f;
        post->config.fog_height_falloff = 0.0f;  // uniform, for this check
        post->config.fog_start = 2.0f;
        if (!frame(0.016f, false, [&]() -> eng::rhi::TextureId {
                eng::rhi::PassDesc pd;
                pd.color = hdr;
                // LOAD, not clear: the fog blends over the scene already there.
                // Without this the pass wipes the frame and draws fog on black,
                // which is a screen of flat fog -- and reads as "the density is
                // too high" rather than as a missing load action.
                pd.load = true;
                auto e = dev->BeginPass(pd);
                post->DrawFog(e, depth);
                dev->EndPass();
                return hdr;
            })) return 1;
        const float near_after = Mean(px, 2, 96, 200, 160, 250);
        const float far_after = Mean(px, 2, 96, 118, 160, 132);
        std::printf("    blue channel  near %.1f -> %.1f   far %.1f -> %.1f\n",
                    near_before, near_after, far_before, far_after);
        Check(far_after > far_before + 20.0f, "distant geometry takes the fog colour");
        Check(far_after - far_before > (near_after - near_before) * 2.0f,
              "and takes far more of it than the near field does");
        post->config.fog = false;
    }

    {
        // A pass that blends has to LOAD, and the failure when it does not is a
        // screen of flat fog that reads as "the density is too high". Checked
        // against the same region before the fog rather than against an
        // absolute threshold: the near field's checker squares are large, so
        // its detail number is small whether or not anything erased it.
        std::printf("\nand does not erase what it is fogging\n");
        const float fogged_detail = Detail(px, 60, 200, 200, 250);
        post->config.fog = false;
        if (!frame(0.016f, false, nullptr)) return 1;
        const float clear_detail = Detail(px, 60, 200, 200, 250);
        std::printf("    near-field detail: no fog %.2f, fogged %.2f\n",
                    clear_detail, fogged_detail);
        Check(fogged_detail > clear_detail * 0.7f,
              "the near field still has its checkerboard");
    }

    {
        // MEASURED AGAINST THE SAME REGION WITHOUT THE EFFECT, not against a
        // different part of the image.
        //
        // The first version compared detail in the near field against the far
        // field and failed with the far field at 5.02 against the near field's
        // 1.78 -- because perspective makes distant checker squares smaller, so
        // there are more edges per pixel out there whether or not anything is
        // blurred. The metric was measuring the scene, not the effect.
        std::printf("\ndepth of field blurs away from the focal plane\n");
        constexpr int kNx0 = 60, kNy0 = 200, kNx1 = 200, kNy1 = 250;  // near
        // Well below the horizon: the strip right at it is mostly the boundary
        // between the floor and the background, whose contribution to a
        // horizontal difference is nil, plus checker moiré that a blur turns
        // into different moiré rather than into smoothness.
        constexpr int kFx0 = 60, kFy0 = 138, kFx1 = 200, kFy1 = 158;   // far

        post->config.depth_of_field = false;
        if (!frame(0.016f, false, nullptr)) return 1;
        const float sharp_near = Detail(px, kNx0, kNy0, kNx1, kNy1);
        const float sharp_far = Detail(px, kFx0, kFy0, kFx1, kFy1);

        const auto dof_frame = [&](float focus) -> bool {
            post->config.depth_of_field = true;
            post->config.focus_distance = focus;
            post->config.focus_range = 6.0f;
            post->config.max_blur_radius = 26.0f;
            return frame(0.016f, false, [&]() -> eng::rhi::TextureId {
                eng::rhi::PassDesc pd;
                pd.color = hdr2;
                auto e = dev->BeginPass(pd);
                post->DrawDepthOfField(e, hdr, depth);
                dev->EndPass();
                return hdr2;
            });
        };

        // Focused NEAR: the near field keeps its detail, the far field loses it.
        if (!dof_frame(3.5f)) return 1;
        const float near_focus_near = Detail(px, kNx0, kNy0, kNx1, kNy1);
        const float near_focus_far = Detail(px, kFx0, kFy0, kFx1, kFy1);
        std::printf("    focus 3.5 m: near %.2f -> %.2f, far %.2f -> %.2f\n",
                    sharp_near, near_focus_near, sharp_far, near_focus_far);
        Check(near_focus_near > sharp_near * 0.8f, "the focal plane stays sharp");
        Check(near_focus_far < sharp_far * 0.7f, "and the distance is blurred");

        // Focused FAR: the two swap. This is what separates depth of field from
        // a plain distance blur -- which would pass every check above.
        if (!dof_frame(45.0f)) return 1;
        const float far_focus_near = Detail(px, kNx0, kNy0, kNx1, kNy1);
        const float far_focus_far = Detail(px, kFx0, kFy0, kFx1, kFy1);
        std::printf("    focus 45 m:  near %.2f -> %.2f, far %.2f -> %.2f\n",
                    sharp_near, far_focus_near, sharp_far, far_focus_far);
        Check(far_focus_near < near_focus_near * 0.8f,
              "moving the focal plane away blurs the near field instead");

        // THE RATIO, not the absolute far-field detail.
        //
        // Asking for the far field to get sharper outright does not hold with a
        // six-metre focus range: the sampled strip is about twenty metres out,
        // so it is past full blur for BOTH a 3.5 m and a 45 m focus and the two
        // differ only in noise. What must be true either way is that sharpness
        // MOVED -- the far field's share of the frame's detail has to rise when
        // the focal plane goes out, and that is a statement about the ratio.
        const float near_focus_ratio = near_focus_far / std::max(near_focus_near, 1e-3f);
        const float far_focus_ratio = far_focus_far / std::max(far_focus_near, 1e-3f);
        std::printf("    far/near detail ratio: focused near %.2f, focused far %.2f\n",
                    near_focus_ratio, far_focus_ratio);
        Check(far_focus_ratio > near_focus_ratio * 1.3f,
              "and sharpness moves from the near field to the far one");
        post->config.depth_of_field = false;
    }

    {
        // Again: the SAME region, with the effect on and off. Comparing a still
        // frame's detail against a panning frame's compares two different
        // images of two different things.
        std::printf("\nvelocity follows the camera\n");
        constexpr int kX0 = 60, kY0 = 150, kX1 = 200, kY1 = 240;
        post->config.motion_blur = true;
        post->config.shutter = 1.0f;

        const auto blur_frame = [&]() -> bool {
            return frame(0.016f, true, [&]() -> eng::rhi::TextureId {
                eng::rhi::PassDesc pd;
                pd.color = hdr2;
                auto e = dev->BeginPass(pd);
                post->DrawMotionBlur(e, hdr);
                dev->EndPass();
                return hdr2;
            });
        };

        // A STILL camera. Two frames from the same place, so the velocity is
        // zero everywhere and the blur must be a no-op. This is the check that
        // catches a reprojection which is subtly wrong: it would report motion
        // on a static camera and smear a still image, which reads as "the
        // renderer is soft".
        scene.camera.eye = eng::Vec3{0.0f, 1.2f, 2.0f};
        scene.camera.target = eng::Vec3{0.0f, 0.6f, -40.0f};
        if (!frame(0.016f, true, nullptr)) return 1;
        const float unblurred = Detail(px, kX0, kY0, kX1, kY1);
        if (!blur_frame()) return 1;
        const float still = Detail(px, kX0, kY0, kX1, kY1);
        std::printf("    still camera: detail %.2f -> %.2f\n", unblurred, still);
        Check(still > unblurred * 0.92f, "a still camera is not blurred at all");

        // A PANNING one, from the same place as the frame before it, so the
        // velocity is a real horizontal sweep.
        if (!frame(0.016f, true, nullptr)) return 1;
        scene.camera.eye = eng::Vec3{1.6f, 1.2f, 2.0f};
        scene.camera.target = eng::Vec3{1.6f, 0.6f, -40.0f};
        if (!blur_frame()) return 1;
        const float moving = Detail(px, kX0, kY0, kX1, kY1);
        std::printf("    panning:      detail %.2f -> %.2f (%.0f%% of it left)\n",
                    unblurred, moving, 100.0 * double(moving / unblurred));
        // 0.80, RE-DERIVED. It was 0.7, calibrated against the per-channel ACES
        // curve the composite used to end with; the replacement has a different
        // contrast response in the midtones and the same blur now measures
        // 0.706 instead of just under 0.7. The blur is not weaker -- a pan
        // still destroys 29% of the detail -- the number it is being compared
        // against was a property of the old curve.
        //
        // Still a real threshold: a blur that does nothing scores 1.00, which
        // is where the still-camera case above sits, so there is 20 points of
        // separation on one side and 9 on the other.
        Check(moving < unblurred * 0.80f, "a panning one is");

        post->config.motion_blur = false;
        scene.camera.eye = eng::Vec3{0.0f, 1.2f, 2.0f};
        scene.camera.target = eng::Vec3{0.0f, 0.6f, -40.0f};
    }

    {
        // THE FIRST READING SNAPS, and this has to run before anything else
        // meters: the snap fires once per PostStack, and MeterExposure returns
        // early while auto_exposure is off, so nothing above here has armed it.
        //
        // Without it a viewer OPENS AT 1.0 and crawls. Measured in world2:
        // 1.0 -> 0.785 over 40 frames and still falling, against a target of
        // 0.44 -- three seconds of white before the image appears, which reads
        // as a broken renderer rather than as an eye adjusting.
        std::printf("\nthe first exposure reading snaps rather than crawling\n");
        post->config.auto_exposure = true;
        // SLOW on purpose. At this rate a crawl from 1.0 covers 2.5% per frame,
        // so if the first reading did not snap, four frames leave it near 1.0
        // and the check below cannot pass by luck.
        post->config.adapt_brighter = 0.5f;
        post->config.adapt_darker = 0.5f;
        r->SetExposureBuffer(post->ExposureBuffer());
        scene.lightColor = eng::Vec4{14.0f, 14.0f, 14.0f, 1.0f};
        scene.ambientSky = eng::Vec3{2.0f, 2.1f, 2.4f};
        scene.ambientGround = eng::Vec3{0.5f, 0.5f, 0.5f};
        // Four, not one: LastExposure reads a buffer the GPU wrote and is one
        // to three frames stale, so one frame would measure the readback lag
        // rather than the meter.
        for (int i = 0; i < 4; ++i)
            if (!frame(0.05f, true, nullptr)) return 1;
        const float snapped = post->LastExposure();
        for (int i = 0; i < 90; ++i)
            if (!frame(0.05f, true, nullptr)) return 1;
        const float settled = post->LastExposure();
        std::printf("    after 4 frames %.3f, after 94 frames %.3f (started at 1.000)\n",
                    snapped, settled);
        Check(std::fabs(snapped - settled) < settled * 0.25f,
              "four frames in, the meter is already where it converges");
        Check(snapped < 0.6f, "and nowhere near the 1.0 it was seeded with");
    }

    {
        std::printf("\nauto-exposure converges on middle grey\n");
        post->config.auto_exposure = true;
        post->config.adapt_brighter = 12.0f;
        post->config.adapt_darker = 12.0f;
        r->SetExposureBuffer(post->ExposureBuffer());

        // A DIM scene: the meter should open up.
        scene.lightColor = eng::Vec4{0.25f, 0.25f, 0.25f, 1.0f};
        scene.ambientSky = eng::Vec3{0.02f, 0.02f, 0.03f};
        scene.ambientGround = eng::Vec3{0.005f, 0.005f, 0.005f};
        for (int i = 0; i < 40; ++i)
            if (!frame(0.05f, true, nullptr)) return 1;
        const float dim_exposure = post->LastExposure();
        const float dim_mean = Mean(px, 1, 0, 0, kW, kH);
        std::printf("    dim scene:    exposure %.2f, image mean %.1f\n",
                    dim_exposure, dim_mean);

        // A BRIGHT one: it should stop down.
        scene.lightColor = eng::Vec4{14.0f, 14.0f, 14.0f, 1.0f};
        scene.ambientSky = eng::Vec3{2.0f, 2.1f, 2.4f};
        scene.ambientGround = eng::Vec3{0.5f, 0.5f, 0.5f};
        for (int i = 0; i < 40; ++i)
            if (!frame(0.05f, true, nullptr)) return 1;
        const float bright_exposure = post->LastExposure();
        const float bright_mean = Mean(px, 1, 0, 0, kW, kH);
        std::printf("    bright scene: exposure %.2f, image mean %.1f\n",
                    bright_exposure, bright_mean);

        Check(dim_exposure > bright_exposure * 4.0f,
              "the meter opens up in the dark and stops down in the light");
        // THE POINT of exposure: the two images end up at a similar brightness
        // even though the scenes differ by fifty times. Without this the test
        // would pass for a meter that moves in the right direction and not far
        // enough.
        std::printf("    scene brightness differs 56x, image means %.1f vs %.1f\n",
                    dim_mean, bright_mean);
        Check(std::fabs(dim_mean - bright_mean) < 40.0f,
              "and the two images land at a comparable brightness");

        // ADAPTATION IS GRADUAL. An instant meter is not an exposure system, it
        // is a normalisation, and it makes a camera pan look like a fault.
        post->config.adapt_brighter = 0.5f;
        post->config.adapt_darker = 0.5f;
        scene.lightColor = eng::Vec4{0.25f, 0.25f, 0.25f, 1.0f};
        scene.ambientSky = eng::Vec3{0.02f, 0.02f, 0.03f};
        const float before = post->LastExposure();
        if (!frame(0.016f, true, nullptr)) return 1;
        if (!frame(0.016f, true, nullptr)) return 1;
        const float after_two = post->LastExposure();
        std::printf("    after a sudden blackout, 2 frames: %.3f -> %.3f\n",
                    before, after_two);
        Check(after_two < before * 4.0f,
              "two frames of darkness do not open the aperture all the way");
        post->config.auto_exposure = false;
        r->SetExposureBuffer({});
    }

    {
        // EXPOSURE COMPENSATION is a manual multiplier on the meter, in stops.
        // It used to be written into the params and never read: the resolve
        // mapped the average to middle grey whatever it said, so a viewer
        // asking for -0.5 got exactly the image it would have got at 0.
        std::printf("\nexposure compensation moves the metered exposure\n");
        post->config.auto_exposure = true;
        post->config.adapt_brighter = 12.0f;
        post->config.adapt_darker = 12.0f;
        post->config.exposure_compensation = 0.0f;
        r->SetExposureBuffer(post->ExposureBuffer());
        scene.lightColor = eng::Vec4{14.0f, 14.0f, 14.0f, 1.0f};
        scene.ambientSky = eng::Vec3{2.0f, 2.1f, 2.4f};
        scene.ambientGround = eng::Vec3{0.5f, 0.5f, 0.5f};
        for (int i = 0; i < 40; ++i)
            if (!frame(0.05f, true, nullptr)) return 1;
        const float uncompensated = post->LastExposure();
        post->config.exposure_compensation = -1.0f;
        for (int i = 0; i < 40; ++i)
            if (!frame(0.05f, true, nullptr)) return 1;
        const float stopped_down = post->LastExposure();
        std::printf("    compensation 0: exposure %.3f, at -1 stop: %.3f\n",
                    uncompensated, stopped_down);
        Check(std::fabs(stopped_down * 2.0f - uncompensated) <
                  uncompensated * 0.15f,
              "minus one stop halves the metered exposure");
        post->config.auto_exposure = false;
        post->config.exposure_compensation = 0.0f;
        r->SetExposureBuffer({});
    }

    {
        std::printf("\nthe TAA jitter is a proper low-discrepancy sequence\n");
        post->config.taa = true;
        std::vector<eng::Vec2> offsets;
        for (int i = 0; i < 8; ++i) {
            post->BeginFrame(scene.camera, kW, kH, 0.016f);
            offsets.push_back(post->Jitter());
        }
        // WITHIN A PIXEL. A jitter larger than one pixel is not antialiasing,
        // it is a shake.
        bool bounded = true;
        for (const eng::Vec2& o : offsets)
            if (std::fabs(o.x) > 1.0f / kW || std::fabs(o.y) > 1.0f / kH) bounded = false;
        Check(bounded, "every offset stays inside one pixel");

        // SPREAD OUT. The property that distinguishes Halton from random: no
        // two of eight consecutive samples land on top of each other, so every
        // frame contributes something new.
        float closest = 1e9f;
        for (std::size_t i = 0; i < offsets.size(); ++i)
            for (std::size_t j = i + 1; j < offsets.size(); ++j) {
                const float dx = (offsets[i].x - offsets[j].x) * kW;
                const float dy = (offsets[i].y - offsets[j].y) * kH;
                closest = std::min(closest, std::sqrt(dx * dx + dy * dy));
            }
        std::printf("    closest pair of 8 offsets: %.3f px apart\n", closest);
        Check(closest > 0.12f, "and none of eight consecutive ones coincide");

        // The mean is near the centre, so the accumulated image is not shifted.
        eng::Vec2 mean{0, 0};
        for (const eng::Vec2& o : offsets) mean = eng::Vec2{mean.x + o.x, mean.y + o.y};
        std::printf("    mean offset %.4f, %.4f px\n", mean.x / 8 * kW, mean.y / 8 * kH);
        Check(std::fabs(mean.x / 8 * kW) < 0.15f && std::fabs(mean.y / 8 * kH) < 0.15f,
              "and they average to the pixel centre, so nothing shifts");
        post->config.taa = false;
    }

    {
        std::printf("\nTAA converges on a still camera\n");
        post->config.taa = true;
        post->config.taa_feedback = 0.9f;
        scene.lightColor = eng::Vec4{3.0f, 3.0f, 3.0f, 1.0f};
        scene.ambientSky = eng::Vec3{0.25f, 0.28f, 0.34f};
        scene.ambientGround = eng::Vec3{0.06f, 0.06f, 0.06f};

        std::vector<std::uint8_t> previous;
        float last_change = 1e9f;
        bool settles = false;
        for (int i = 0; i < 24; ++i) {
            if (!frame(0.016f, true, [&]() -> eng::rhi::TextureId {
                    eng::rhi::PassDesc pd;
                    pd.color = post->Output();
                    auto e = dev->BeginPass(pd);
                    post->DrawTaa(e, hdr);
                    dev->EndPass();
                    return post->Output();
                })) return 1;
            if (!previous.empty()) {
                double sum = 0.0;
                for (std::size_t k = 0; k < px.size(); k += 4)
                    sum += std::abs(int(px[k]) - int(previous[k]));
                last_change = float(sum / double(px.size() / 4));
            }
            previous = px;
        }
        std::printf("    frame-to-frame change after 24 frames: %.3f of 255\n",
                    last_change);
        settles = last_change < 1.5f;
        // A resolve that never settles is one whose history is not being found
        // -- a reprojection off by a pixel, or a clamp so tight it rejects
        // everything. Both look like "TAA is a bit soft" and are not.
        Check(settles, "the image stops changing once the camera is still");
        post->config.taa = false;
        scene.camera.jitter = eng::Vec2{0.0f, 0.0f};
    }

    {
        // SCREEN-SPACE REFLECTIONS, checked by putting something in the scene
        // that a probe could not possibly reflect.
        //
        // A probe is captured from one point and knows nothing about the
        // objects in the room, so a reflection OF A SCENE OBJECT can only have
        // come from the march. A bright red box hovering over a dark metal
        // floor: if a red patch appears in the floor beneath it, the march
        // found the box.
        std::printf("\nscreen-space reflections find the scene\n");
        const eng::rhi::TextureId albedo_rough =
            dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
        const eng::rhi::TextureId normal_metal =
            dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);

        eng::Scene mirror;
        mirror.lightDir = eng::Vec4{0.2f, 0.95f, 0.2f, 0.0f};
        mirror.lightColor = eng::Vec4{2.0f, 2.0f, 2.0f, 1.0f};
        mirror.ambientSky = eng::Vec3{0.05f, 0.05f, 0.06f};
        mirror.ambientGround = eng::Vec3{0.01f, 0.01f, 0.01f};
        mirror.camera.eye = eng::Vec3{0.0f, 1.4f, 5.0f};
        mirror.camera.target = eng::Vec3{0.0f, 0.35f, 0.0f};

        eng::MaterialDesc floor_md;
        // Shading::Lit, NOT Shading::GBuffer -- the renderer derives the
        // G-buffer variant from a Lit material, and asking for the variant
        // directly is now an error rather than a silently empty frame.
        floor_md.shading = eng::Shading::Lit;
        floor_md.base_color = eng::Vec4{0.04f, 0.04f, 0.05f, 1.0f};
        floor_md.roughness = 0.05f;
        floor_md.metallic = 1.0f;
        eng::MaterialDesc box_md;
        box_md.shading = eng::Shading::Lit;
        box_md.base_color = eng::Vec4{1.0f, 0.05f, 0.05f, 1.0f};
        box_md.roughness = 0.9f;
        box_md.metallic = 0.0f;
        const eng::MaterialHandle floor_mat = r->CreateMaterial(floor_md, error);
        const eng::MaterialHandle box_mat = r->CreateMaterial(box_md, error);
        const eng::MeshHandle plate = r->UploadMesh(
            eng::MakeBox(eng::Vec3{8.0f, 0.1f, 8.0f}, eng::Vec4{1, 1, 1, 1}));
        const eng::MeshHandle box = r->UploadMesh(
            eng::MakeBox(eng::Vec3{0.7f, 0.7f, 0.7f}, eng::Vec4{1, 1, 1, 1}));
        if (!eng::Valid(floor_mat) || !eng::Valid(box_mat)) {
            std::fprintf(stderr, "FAIL: %s\n", error.c_str());
            return 1;
        }
        // The rejection above, checked: a silently empty frame is what this
        // replaced.
        eng::MaterialDesc bad;
        bad.shading = eng::Shading::GBuffer;
        std::string bad_error;
        Check(!eng::Valid(r->CreateMaterial(bad, bad_error)) && !bad_error.empty(),
              "asking for a derived shading mode is refused with a message");
        {
            eng::Instance f;
            f.mesh = plate;
            f.material = floor_mat;
            f.model = eng::Mat4::Translation(eng::Vec3{0.0f, -0.1f, 0.0f});
            mirror.instances.push_back(f);
            eng::Instance b;
            b.mesh = box;
            b.material = box_mat;
            b.model = eng::Mat4::Translation(eng::Vec3{0.0f, 1.5f, -1.0f});
            mirror.instances.push_back(b);
        }

        const auto mirror_frame = [&](bool with_ssr) -> bool {
            post->config.ssr = with_ssr;
            post->config.ssr_intensity = 1.0f;
            post->config.ssr_distance = 40.0f;
            post->BeginFrame(mirror.camera, kW, kH, 0.016f);
            dev->BeginFrame();
            {
                eng::rhi::PassDesc pd;
                pd.color = albedo_rough;
                pd.extra_colors = {normal_metal};
                pd.depth = depth;
                pd.keep_depth = true;
                auto e = dev->BeginPass(pd);
                r->DrawGBuffer(e, mirror, kW, kH);
                dev->EndPass();
            }
            {
                eng::rhi::PassDesc pd;
                pd.color = hdr;
                auto e = dev->BeginPass(pd);
                r->DrawDeferredLight(e, mirror, kW, kH, albedo_rough, normal_metal,
                                     depth, {});
                dev->EndPass();
            }
            {
                eng::rhi::PassDesc pd;
                pd.color = hdr;
                pd.load = true;  // additive over the lit scene
                auto e = dev->BeginPass(pd);
                post->DrawSsr(e, hdr, depth, normal_metal, albedo_rough);
                dev->EndPass();
            }
            {
                eng::rhi::PassDesc pd;
                pd.color = out;
                auto e = dev->BeginPass(pd);
                r->DrawComposite(e, hdr);
                dev->EndPass();
            }
            if (!dev->CommitAndWait(error)) {
                std::fprintf(stderr, "FAIL submit: %s\n", error.c_str());
                return false;
            }
            post->EndFrame();
            return dev->ReadPixels(out, kW, kH, px);
        };

        if (!mirror_frame(false)) return 1;
        // The floor below the box. The box itself sits well above the horizon;
        // this band is floor.
        const float off_r = Mean(px, 0, 100, 168, 156, 200);
        const float off_g = Mean(px, 1, 100, 168, 156, 200);
        if (!mirror_frame(true)) return 1;
        const float on_r = Mean(px, 0, 100, 168, 156, 200);
        const float on_g = Mean(px, 1, 100, 168, 156, 200);
        std::printf("    floor under the box: SSR off r %.1f g %.1f, on r %.1f g %.1f\n",
                    off_r, off_g, on_r, on_g);
        Check(on_r > off_r + 6.0f, "a red object appears in the metal floor below it");
        // RED, specifically. A march that simply brightened the floor -- a fade
        // computed wrongly, a sample taken from the wrong place -- would raise
        // every channel together.
        Check((on_r - off_r) > (on_g - off_g) * 3.0f,
              "and it arrives as red, not as a general brightening");

        // AWAY from the box the floor must be nearly unchanged: SSR that
        // brightens the whole surface is not a reflection, it is an ambient
        // term with extra steps.
        const float corner_on = Mean(px, 0, 6, 236, 46, 252);
        if (!mirror_frame(false)) return 1;
        const float corner_off = Mean(px, 0, 6, 236, 46, 252);
        std::printf("    bottom-left corner: off %.1f, on %.1f\n", corner_off,
                    corner_on);
        Check(corner_on - corner_off < (on_r - off_r) * 0.25f,
              "and the floor away from it is barely affected");
        post->config.ssr = false;
    }

    std::printf(g_failures == 0 ? "\npost_test: all checks passed\n"
                                : "\npost_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
