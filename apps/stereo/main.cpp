// Single-pass stereo rendering, checked against the two-pass version.
//
// The claim is not "it renders two images" -- two passes do that. The claim is
// that ONE pass, with the vertex stage emitting both eyes, produces the same
// two images as two passes would, and produces them for less. So the test
// renders both ways and compares, and then measures the difference.
//
// The failure mode this guards against is the one that is invisible without a
// headset: a layered pass that forgets its view mapping amplifies the vertices
// and writes every copy to slice 0, so both eyes get the left eye's image. It
// looks perfect on a monitor.
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
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

constexpr int kW = 320, kH = 320;
// 64 mm, the average human interpupillary distance.
constexpr float kIpd = 0.064f;

double Luma(const std::vector<std::uint8_t>& p, std::size_t i) {
    return 0.2126 * p[i] + 0.7152 * p[i + 1] + 0.0722 * p[i + 2];
}

// The horizontal offset, in pixels, that best aligns two images. This is what a
// stereo pair IS -- the same scene shifted by an amount that depends on depth --
// so measuring it is how you know the two eyes are actually different views of
// one scene rather than two unrelated pictures or two copies of one.
int BestShift(const std::vector<std::uint8_t>& a,
              const std::vector<std::uint8_t>& b, int y0, int y1) {
    int best = 0;
    double best_err = 1e30;
    for (int s = -20; s <= 20; ++s) {
        double err = 0.0;
        int n = 0;
        for (int y = y0; y < y1; ++y)
            for (int x = 40; x < kW - 40; ++x) {
                const int xb = x + s;
                if (xb < 0 || xb >= kW) continue;
                const double d = Luma(a, (std::size_t(y) * kW + x) * 4) -
                                 Luma(b, (std::size_t(y) * kW + xb) * 4);
                err += d * d;
                ++n;
            }
        if (n == 0) continue;
        err /= n;
        if (err < best_err) {
            best_err = err;
            best = s;
        }
    }
    return best;
}

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

    // Objects at two very different depths. Parallax is inversely proportional
    // to distance, so a near object shifts a lot between the eyes and a far one
    // barely at all -- and a test with everything at one depth cannot tell a
    // stereo pair from the same image twice.
    const MeshHandle ball =
        r->UploadMesh(MakeUVSphere(0.25f, 32, 44, Vec4{1, 1, 1, 1}, Vec4{1, 1, 1, 1}));
    MaterialDesc md;
    md.base_color = Vec4{0.85f, 0.85f, 0.85f, 1.0f};
    md.roughness = 0.5f;
    const MaterialHandle mat = r->CreateMaterial(md, error);
    if (!Valid(mat)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }

    Scene scene;
    scene.camera.eye = Vec3{-kIpd * 0.5f, 0.0f, 0.0f};  // the LEFT eye
    scene.camera.target = Vec3{-kIpd * 0.5f, 0.0f, -1.0f};
    scene.camera.fovY = 1.0f;
    scene.lightDir = Vec4{0.3f, 0.5f, 0.8f, 0.0f};
    scene.lightColor = Vec4{3.0f, 3.0f, 3.0f, 1.0f};
    scene.ambientSky = Vec3{0.03f, 0.03f, 0.04f};
    scene.ambientGround = Vec3{0.01f, 0.01f, 0.01f};
    for (int i = 0; i < 3; ++i) {
        Instance in;
        in.mesh = ball;
        in.material = mat;
        // 0.6 m, 1.8 m and 6 m away, spread across the view.
        const float z = -0.6f - float(i) * float(i) * 1.35f;
        in.model = Mat4::Translation({(float(i) - 1.0f) * 0.32f, 0.0f, z});
        scene.instances.push_back(in);
    }

    Camera right = scene.camera;
    right.eye = Vec3{kIpd * 0.5f, 0.0f, 0.0f};
    right.target = Vec3{kIpd * 0.5f, 0.0f, -1.0f};

    // A two-LAYER target, plus a 2D view of each slice so the readback can see
    // them. The array itself cannot be read back -- getBytes wants a 2D
    // texture -- which is what the slice views are for.
    const rhi::TextureId hdr_array = dev->CreateRenderTargetArray(
        kW, kH, 2, Renderer::kSceneFormat);
    const rhi::TextureId depth_array = dev->CreateDepthTargetArray(kW, kH, 2);
    const rhi::TextureId hdr_left = dev->CreateArraySlice(hdr_array, 0);
    const rhi::TextureId hdr_right = dev->CreateArraySlice(hdr_array, 1);
    const rhi::TextureId hdr_mono =
        dev->CreateRenderTarget(kW, kH, Renderer::kSceneFormat);
    const rhi::TextureId depth_mono = dev->CreateDepthTarget(kW, kH);
    const rhi::TextureId ldr =
        dev->CreateRenderTarget(kW, kH, kFmt, /*cpu_readable=*/true);
    if (!rhi::Valid(hdr_array) || !rhi::Valid(hdr_left) || !rhi::Valid(hdr_right)) {
        std::fprintf(stderr, "FAIL: could not create the layered targets\n");
        return 1;
    }
    std::vector<std::uint8_t> px(std::size_t(kW) * kH * 4);

    const auto composite = [&](rhi::TextureId src) {
        rhi::PassDesc pd;
        pd.color = ldr;
        auto e = dev->BeginPass(pd);
        r->DrawComposite(e, src, {}, {}, 0.0f, /*vignette=*/0.0f);
        dev->EndPass();
    };

    // ------------------------------------------------------- two passes, once --
    std::vector<std::uint8_t> mono_left, mono_right;
    double mono_ms = 0.0;
    {
        std::printf("two passes, one per eye\n");
        for (int eye = 0; eye < 2; ++eye) {
            Scene s = scene;
            if (eye == 1) s.camera = right;
            dev->BeginFrame();
            {
                rhi::PassDesc pd;
                pd.color = hdr_mono;
                pd.depth = depth_mono;
                pd.clear_depth = 0.0f;
                pd.timer = "eye";
                auto e = dev->BeginPass(pd);
                r->DrawScene(e, s, kW, kH, {});
                dev->EndPass();
            }
            composite(hdr_mono);
            if (!dev->CommitAndWait(error)) {
                std::fprintf(stderr, "FAIL: submit: %s\n", error.c_str());
                return 1;
            }
            if (!dev->ReadPixels(ldr, kW, kH, px)) return 1;
            if (eye == 0) mono_left = px;
            else mono_right = px;
            for (const rhi::GpuTiming& g : dev->LastFrameTimings())
                mono_ms += g.milliseconds;
        }
        int lit = 0;
        for (int i = 0; i < kW * kH; ++i)
            if (Luma(mono_left, std::size_t(i) * 4) > 10.0) ++lit;
        std::printf("    the spheres cover %d of %d pixels\n", lit, kW * kH);
        Check(lit > 3000, "the reference images are actually of something");

        // The two eyes must DIFFER, or there is no stereo to test.
        double diff = 0.0;
        for (int i = 0; i < kW * kH; ++i)
            diff += std::fabs(Luma(mono_left, std::size_t(i) * 4) -
                              Luma(mono_right, std::size_t(i) * 4));
        std::printf("    the two eyes differ by %.2f/255 on average\n",
                    diff / (kW * kH));
        Check(diff / (kW * kH) > 0.5, "and the two eyes are different views");
    }

    // -------------------------------------------------------- one pass, both --
    {
        std::printf("\none pass, both eyes, vertex amplification\n");
        double stereo_ms = 0.0;
        dev->BeginFrame();
        {
            rhi::PassDesc pd;
            pd.color = hdr_array;
            pd.depth = depth_array;
            pd.clear_depth = 0.0f;
            pd.views = 2;
            pd.timer = "stereo";
            auto e = dev->BeginPass(pd);
            r->DrawSceneStereo(e, scene, right, kW, kH, {});
            dev->EndPass();
        }
        if (!dev->CommitAndWait(error)) {
            std::fprintf(stderr, "FAIL: submit: %s\n", error.c_str());
            return 1;
        }
        for (const rhi::GpuTiming& g : dev->LastFrameTimings())
            stereo_ms += g.milliseconds;

        dev->BeginFrame();
        composite(hdr_left);
        if (!dev->CommitAndWait(error)) return 1;
        if (!dev->ReadPixels(ldr, kW, kH, px)) return 1;
        const std::vector<std::uint8_t> stereo_left = px;

        dev->BeginFrame();
        composite(hdr_right);
        if (!dev->CommitAndWait(error)) return 1;
        if (!dev->ReadPixels(ldr, kW, kH, px)) return 1;
        const std::vector<std::uint8_t> stereo_right = px;

        const auto compare = [&](const std::vector<std::uint8_t>& a,
                                 const std::vector<std::uint8_t>& b) {
            double worst = 0.0, sum = 0.0;
            for (int i = 0; i < kW * kH; ++i)
                for (int c = 0; c < 3; ++c) {
                    const double d = std::fabs(double(a[std::size_t(i) * 4 + c]) -
                                               double(b[std::size_t(i) * 4 + c]));
                    worst = std::max(worst, d);
                    sum += d;
                }
            return std::pair<double, double>{worst, sum / (kW * kH * 3)};
        };
        const auto cover = [&](const std::vector<std::uint8_t>& img) {
            int n = 0;
            for (int i = 0; i < kW * kH; ++i)
                if (Luma(img, std::size_t(i) * 4) > 10.0) ++n;
            return n;
        };
        std::printf("    coverage: mono L %d R %d, stereo L %d R %d\n",
                    cover(mono_left), cover(mono_right), cover(stereo_left),
                    cover(stereo_right));
        const auto [lw, lm] = compare(stereo_left, mono_left);
        const auto [rw, rm] = compare(stereo_right, mono_right);
        std::printf("    left slice vs its own pass:  worst %.0f, mean %.4f\n", lw,
                    lm);
        std::printf("    right slice vs its own pass: worst %.0f, mean %.4f\n", rw,
                    rm);
        // BIT-IDENTICAL. The amplified vertex stage does the same arithmetic on
        // the same inputs; only where the result is written differs. Anything
        // else means the two paths are not the same renderer.
        Check(lw == 0.0 && rw == 0.0,
              "both slices are bit-identical to their single-eye passes");

        // AND THE SLICES ARE NOT THE SAME. This is the check that catches a
        // missing view mapping: without it the vertex stage amplifies and then
        // writes every copy to slice 0, so both eyes hold the left eye's image
        // and the check above still passes for the left one.
        const auto [sw, sm] = compare(stereo_left, stereo_right);
        std::printf("    the two slices differ by %.0f at worst, %.4f on average\n",
                    sw, sm);
        Check(sm > 0.5, "and the two slices are genuinely different views");

        std::printf("    GPU time: %.4f ms for two passes, %.4f ms for one\n",
                    mono_ms, stereo_ms);
        Check(stereo_ms > 0.0, "the timer reported something for the stereo pass");
        Check(stereo_ms < mono_ms,
              "and one amplified pass costs less than two separate ones");
    }

    // ------------------------------------------------------------- parallax --
    {
        std::printf("\nparallax falls off with distance, as it must\n");
        // A near object shifts a lot between the eyes and a far one barely at
        // all -- that relationship IS depth perception, and getting it constant
        // or backwards gives a stereo pair that produces a headache rather than
        // a sense of space.
        //
        // ONE SPHERE AT A TIME, centred, and scaled so it covers the same
        // screen area at every distance. Splitting one frame into thirds does
        // not isolate them: the near sphere is large and the far one tiny, they
        // overlap on screen, and the shift measured over a band is whichever
        // sphere happens to dominate it. Measured that way the near sphere came
        // out at -40 px -- the edge of the search window -- and the far one at
        // -13 against a predicted -3.1.
        // CALIBRATED against this renderer, not derived from the projection.
        //
        // Moving the CAMERA 64 mm to the right and moving the OBJECT 64 mm to
        // the left are the same geometry, so the image has to move by the same
        // number of pixels either way -- and measuring the second gives the
        // conversion without doing any projection algebra. The algebra was
        // tried first and came out 25% low at every depth, consistently, which
        // is a mistake in the derivation rather than in the renderer: three
        // measurements agreeing with each other and disagreeing with a formula
        // by one constant factor is a formula that is wrong.
        const double kDepths[3] = {0.6, 1.95, 6.0};
        double measured[3] = {}, expected[3] = {};
        for (int i = 0; i < 3; ++i) {
            const double d = kDepths[i];
            Scene one;
            one.camera = scene.camera;
            one.lightDir = scene.lightDir;
            one.lightColor = scene.lightColor;
            one.ambientSky = scene.ambientSky;
            one.ambientGround = scene.ambientGround;
            Instance in;
            in.mesh = ball;
            in.material = mat;
            // Scale with distance so the sphere subtends the same angle: the
            // correlation search is then measuring parallax rather than being
            // biased by how many pixels there are to correlate.
            in.model = Mat4::Translation({0.0f, 0.0f, float(-d)}) *
                       Mat4::Scale(float(d / 0.6));
            one.instances.push_back(in);

            std::vector<std::uint8_t> eyes[2];
            for (int eye = 0; eye < 2; ++eye) {
                Scene e = one;
                if (eye == 1) e.camera = right;
                dev->BeginFrame();
                {
                    rhi::PassDesc pd;
                    pd.color = hdr_mono;
                    pd.depth = depth_mono;
                    pd.clear_depth = 0.0f;
                    auto enc = dev->BeginPass(pd);
                    r->DrawScene(enc, e, kW, kH, {});
                    dev->EndPass();
                }
                composite(hdr_mono);
                if (!dev->CommitAndWait(error)) return 1;
                if (!dev->ReadPixels(ldr, kW, kH, px)) return 1;
                eyes[eye] = px;
            }
            // SUB-PIXEL, by the centroid of the lit region. An integer
            // correlation search cannot resolve the 3.1 px the 6 m sphere is
            // predicted to move, and rounding it to the nearest pixel is a 30%
            // error on the number being checked.
            double cx[2] = {0, 0};
            for (int eye = 0; eye < 2; ++eye) {
                double sum = 0.0;
                for (int y = 0; y < kH; ++y)
                    for (int x = 0; x < kW; ++x) {
                        const double w = Luma(eyes[eye], (std::size_t(y) * kW + x) * 4);
                        if (w < 10.0) continue;
                        cx[eye] += w * x;
                        sum += w;
                    }
                if (sum > 0.0) cx[eye] /= sum;
            }
            // The CALIBRATION at this depth: the same sphere, one eye, moved
            // 64 mm to the left in the world.
            Scene moved = one;
            moved.instances[0].model =
                Mat4::Translation({-kIpd, 0.0f, float(-d)}) *
                Mat4::Scale(float(d / 0.6));
            dev->BeginFrame();
            {
                rhi::PassDesc pd;
                pd.color = hdr_mono;
                pd.depth = depth_mono;
                pd.clear_depth = 0.0f;
                auto enc = dev->BeginPass(pd);
                r->DrawScene(enc, moved, kW, kH, {});
                dev->EndPass();
            }
            composite(hdr_mono);
            if (!dev->CommitAndWait(error)) return 1;
            if (!dev->ReadPixels(ldr, kW, kH, px)) return 1;
            double moved_cx = 0.0, moved_sum = 0.0;
            for (int y = 0; y < kH; ++y)
                for (int x = 0; x < kW; ++x) {
                    const double w = Luma(px, (std::size_t(y) * kW + x) * 4);
                    if (w < 10.0) continue;
                    moved_cx += w * x;
                    moved_sum += w;
                }
            if (moved_sum > 0.0) moved_cx /= moved_sum;

            measured[i] = cx[1] - cx[0];
            expected[i] = moved_cx - cx[0];
            std::printf("      sphere at %.2f m: the eye offset moved it %+.2f px; "
                        "moving the object 64 mm moves it %+.2f\n",
                        d, measured[i], expected[i]);
        }

        // NEGATIVE, and the sign is not a convention. The right eye is to the
        // RIGHT, so it sees an object further to the LEFT than the left eye
        // does. The first version of this asserted a positive number.
        Check(measured[0] < 0.0, "the nearest sphere shifts, and toward the left");
        // AND IT FALLS OFF as 1/distance, which is what depth perception IS.
        // A constant shift would mean the eye offset was being applied as a
        // screen-space translation rather than as a camera position -- which
        // looks like stereo and conveys no depth at all.
        std::printf("    near %+.2f px against far %+.2f px, a ratio of %.1f "
                    "(the distances differ by 10x)\n",
                    measured[0], measured[2], measured[0] / measured[2]);
        Check(measured[0] / measured[2] > 5.0,
              "and falls off with distance, near shifting many times the far");
        // AND THE MAGNITUDE IS THE EYE SEPARATION. A 64 mm camera offset has to
        // move the image by exactly as much as a 64 mm object offset the other
        // way -- same geometry, two descriptions. A plausible 1/d gradient with
        // the wrong magnitude would be an eye separation that is not 64 mm, and
        // an eye separation that is not the viewer's is what makes a stereo
        // pair uncomfortable to look at.
        bool all_close = true;
        for (int i = 0; i < 3; ++i)
            if (std::fabs(measured[i] - expected[i]) > 0.6) all_close = false;
        Check(all_close,
              "and the eye offset moves the image exactly as far as moving the "
              "object would");
    }

    std::printf(g_failures == 0 ? "\nstereo_test: all checks passed\n"
                                : "\nstereo_test: %d FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
