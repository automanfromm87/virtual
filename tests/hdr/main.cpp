// HDR output, checked against values that are defined rather than chosen.
//
// The temptation with HDR is to check that "the image is brighter", which is
// not a property -- brighter than what, measured how? PQ has an exact answer
// for every input luminance, and extended-range linear has an exact answer too:
// it is the identity below the roll-off. Those are what get checked.
#include "engine/geometry/mesh.h"
#include "engine/platform/window.h"
#include "engine/render/renderer.h"
#include "engine/rhi/rhi.h"
#include "engine/scene/scene.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

constexpr int kW = 64, kH = 64;

// SMPTE ST 2084, on the CPU, as the reference. Written out from the standard
// rather than copied from the shader -- a test that shares its implementation
// with the thing it is testing checks that the code equals itself.
double Pq(double nits) {
    const double y = std::clamp(nits / 10000.0, 0.0, 1.0);
    const double m1 = 0.1593017578125, m2 = 78.84375;
    const double c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    const double yp = std::pow(y, m1);
    return std::pow((c1 + c2 * yp) / (1.0 + c3 * yp), m2);
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

    // The composite writes into a HALF-FLOAT target, not an 8-bit one. That is
    // the whole point: an 8-bit destination cannot hold a value above 1, so an
    // extended-range output written into one is indistinguishable from a
    // clamped one and the test would pass for the wrong reason.
    auto r = Renderer::Create(*dev, Renderer::kSceneFormat, error, 1);
    if (!r) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }
    const rhi::TextureId src =
        dev->CreateRenderTarget(kW, kH, Renderer::kSceneFormat);
    const rhi::TextureId out = dev->CreateRenderTarget(
        kW, kH, Renderer::kSceneFormat, /*cpu_readable=*/true);
    // A depth attachment, because the material depth-tests. Without one the
    // renderer counts the instance as incompatible and skips it -- correctly,
    // since Metal rejects a depth-testing pipeline in a pass with no depth --
    // and every reading comes back zero.
    const rhi::TextureId depth = dev->CreateDepthTarget(kW, kH);

    // A flat scene of one known radiance, so the number that comes out is a
    // function of the input and the encoding and nothing else.
    const MeshHandle quad =
        r->UploadMesh(MakeBox(Vec3{40.0f, 40.0f, 0.1f}, Vec4{1, 1, 1, 1}));

    std::vector<std::uint8_t> raw(std::size_t(kW) * kH * 8);
    // Returns all three channels, because a curve that is applied per channel
    // is only detectable with a colour -- and everything below used to ask for
    // grey.
    const auto composite_rgb = [&](Vec3 radiance, const ColorGrade& grade) {
        Scene s;
        s.camera.eye = Vec3{0.0f, 0.0f, 3.0f};
        s.camera.target = Vec3{0.0f, 0.0f, 0.0f};
        s.lightColor = Vec4{0, 0, 0, 1};
        s.ambientSky = Vec3{0, 0, 0};
        s.ambientGround = Vec3{0, 0, 0};
        MaterialDesc md;
        md.base_color = Vec4{0, 0, 0, 1};
        // EMISSIVE, so the value on the surface is exactly the number asked
        // for. Anything lit goes through a BRDF and a cosine first.
        md.emissive = radiance;
        Instance in;
        in.mesh = quad;
        in.material = r->CreateMaterial(md, error);
        s.instances.push_back(in);

        r->SetGrade(grade);
        dev->BeginFrame();
        {
            rhi::PassDesc pd;
            pd.color = src;
            pd.depth = depth;
            pd.clear_depth = 0.0f;
            auto e = dev->BeginPass(pd);
            r->DrawScene(e, s, kW, kH, {});
            dev->EndPass();
        }
        {
            rhi::PassDesc pd;
            pd.color = out;
            auto e = dev->BeginPass(pd);
            r->DrawComposite(e, src, {}, {}, 0.0f, /*vignette=*/0.0f);
            dev->EndPass();
        }
        if (!dev->CommitAndWait(error)) {
            std::fprintf(stderr, "FAIL: submit: %s\n", error.c_str());
            std::exit(1);
        }
        if (!dev->ReadPixels(out, kW, kH, raw)) {
            std::fprintf(stderr, "FAIL: readback\n");
            std::exit(1);
        }
        // Half-float, read back as bytes. The centre texel.
        const std::size_t i = (std::size_t(kH / 2) * kW + kW / 2) * 8;
        Vec3 out_rgb{0.0f, 0.0f, 0.0f};
        for (int c = 0; c < 3; ++c) {
            std::uint16_t bits;
            std::memcpy(&bits, raw.data() + i + std::size_t(c) * 2, 2);
            __fp16 h;
            std::memcpy(&h, &bits, 2);
            (&out_rgb.x)[c] = float(h);
        }
        return out_rgb;
    };
    const auto composite_of = [&](float radiance, const ColorGrade& grade) {
        return composite_rgb(Vec3{radiance, radiance, radiance}, grade).x;
    };

    {
        std::printf("SDR clamps, as it must\n");
        ColorGrade g;
        g.output = DisplayOutput::Sdr;
        const float at1 = composite_of(1.0f, g);
        const float at10 = composite_of(10.0f, g);
        const float at100 = composite_of(100.0f, g);
        std::printf("    radiance 1 -> %.4f, 10 -> %.4f, 100 -> %.4f\n", double(at1),
                    double(at10), double(at100));
        Check(at100 <= 1.001f, "nothing exceeds 1.0 however bright the input");
        // AND IT LOSES THE DIFFERENCE. That is not a defect of the encoding, it
        // is what an SDR display is -- and it is the reason the tone curve has
        // to know where it is sending the image, because by this point the
        // highlights are gone and nothing downstream can recover them.
        std::printf("    10 and 100 differ by %.5f after the SDR curve\n",
                    std::fabs(double(at10 - at100)));
        Check(std::fabs(at10 - at100) < 0.02f,
              "and a tenfold difference in the input survives as almost nothing");
    }

    {
        std::printf("\nextended linear keeps what SDR throws away\n");
        ColorGrade g;
        g.output = DisplayOutput::ExtendedLinear;
        g.display_headroom = 4.0f;
        g.rolloff_start = 1.0f;
        const float at05 = composite_of(0.5f, g);
        const float at1 = composite_of(1.0f, g);
        const float at2 = composite_of(2.0f, g);
        const float at10 = composite_of(10.0f, g);
        std::printf("    radiance 0.5 -> %.4f, 1 -> %.4f, 2 -> %.4f, 10 -> %.4f\n",
                    double(at05), double(at1), double(at2), double(at10));
        // BELOW the roll-off it is the IDENTITY. This is what stops an
        // SDR-looking image washing out the moment HDR is switched on, and it
        // is the part of HDR that early games got wrong.
        Check(std::fabs(at05 - 0.5f) < 0.01f,
              "below the roll-off the mapping is exactly the identity");
        Check(at2 > 1.05f, "and values above reference white stay above it");
        // The headroom is respected: nothing exceeds it, however bright.
        Check(at10 <= 4.01f && at10 > 2.0f,
              "with everything rolled into the display's headroom");
        Check(at10 > at2 && at2 > at1,
              "and a brighter input is still a brighter output");

        // AND IT KEEPS THE HUE. The shoulder used to be applied per channel,
        // which is the same fault the SDR curve was replaced for: the strongest
        // channel of a saturated colour is compressed hardest and the weakest
        // barely at all, so the ratio between them -- the hue -- dissolves in
        // proportion to how bright the pixel is.
        //
        // It survived because every reading above asks for GREY, and grey has
        // no hue to lose. A tone curve cannot be tested with an achromatic
        // input; that is not a detail of this test, it is the whole shape of
        // the mistake.
        const auto chroma = [](Vec3 c) {
            const float sum = c.x + c.y + c.z;
            return sum > 1e-6f ? Vec3{c.x / sum, c.y / sum, c.z / sum}
                               : Vec3{1.0f / 3, 1.0f / 3, 1.0f / 3};
        };
        // Well past the roll-off, where a per-channel curve does its damage,
        // and saturated enough that losing it is unmistakable.
        const Vec3 in{6.0f, 2.4f, 1.2f};
        const Vec3 out_rgb = composite_rgb(in, g);
        const Vec3 ci = chroma(in), co = chroma(out_rgb);
        const float drift = std::sqrt((ci.x - co.x) * (ci.x - co.x) +
                                      (ci.y - co.y) * (ci.y - co.y));
        std::printf("    %.1f %.1f %.1f -> %.3f %.3f %.3f, chromaticity moved %.4f\n",
                    double(in.x), double(in.y), double(in.z), double(out_rgb.x),
                    double(out_rgb.y), double(out_rgb.z), double(drift));
        // 0.01 is a hue shift nobody can see; a per-channel shoulder on this
        // input moves it by twenty times that.
        Check(drift < 0.01f, "and the colour comes out the colour that went in");
        // NOT BY DESATURATING TO WHITE either, which would also score well on
        // chromaticity drift only if the drift were measured wrong. A display
        // with four times the headroom can show a saturated highlight, and
        // throwing that away is discarding what the range was for.
        Check(out_rgb.x > out_rgb.y * 1.8f,
              "with the saturation the extra headroom exists to carry");
    }

    {
        std::printf("\nPQ against SMPTE ST 2084\n");
        ColorGrade g;
        g.output = DisplayOutput::Pq;
        g.display_headroom = 40.0f;   // no roll-off in the range being checked
        g.rolloff_start = 40.0f;
        g.reference_white_nits = 203.0f;
        // PQ is ABSOLUTE. A radiance of r is r x 203 nits, and the standard
        // says exactly what code value that is -- so every one of these has a
        // right answer that was not chosen by looking at the picture.
        bool all_ok = true;
        for (float radiance : {0.25f, 0.5f, 1.0f, 2.0f, 4.0f}) {
            const float got = composite_of(radiance, g);
            const double want = Pq(double(radiance) * 203.0);
            const double err = std::fabs(double(got) - want);
            std::printf("    %.2f x 203 = %6.1f nits -> %.4f (ST 2084 says %.4f, "
                        "error %.4f)\n",
                        double(radiance), double(radiance) * 203.0, double(got),
                        want, err);
            // 0.004 is about one code value in 10-bit, which is the precision
            // an HDR10 signal is carried at -- and half-float has plenty of
            // room, so anything larger is the curve being wrong rather than the
            // storage.
            if (err > 0.004) all_ok = false;
        }
        Check(all_ok, "every PQ code value matches the standard to a 10-bit step");

        // AND THE REFERENCE WHITE MATTERS. PQ is the one encoding here that is
        // not relative to anything, so changing what 1.0 is worth in nits has
        // to change the output. A pipeline that ignored it would look right on
        // a graph and be the wrong brightness on a television.
        g.reference_white_nits = 100.0f;
        const float at_100 = composite_of(1.0f, g);
        g.reference_white_nits = 400.0f;
        const float at_400 = composite_of(1.0f, g);
        std::printf("    reference white 100 nits -> %.4f, 400 nits -> %.4f "
                    "(ST 2084: %.4f and %.4f)\n",
                    double(at_100), double(at_400), Pq(100.0), Pq(400.0));
        Check(std::fabs(double(at_100) - Pq(100.0)) < 0.004 &&
                  std::fabs(double(at_400) - Pq(400.0)) < 0.004,
              "and the reference white in nits is honoured, not ignored");
    }

    {
        std::printf("\nthe swapchain refuses an 8-bit HDR layer\n");
        // An extended-range layer with an 8-bit format has nowhere to put a
        // value above one, whatever colour space it claims. Refusing is the
        // only honest answer: accepting it produces a picture that is exactly
        // SDR while every readback and every setting says HDR.
        std::string err;
        auto bad = dev->CreateSwapchain(rhi::Format::BGRA8Unorm, err, /*hdr=*/true);
        std::printf("    %s\n", err.c_str());
        Check(!bad, "an 8-bit HDR swapchain is refused");
        Check(err.find("RGBA16Float") != std::string::npos,
              "with a message that says what to do instead");
        err.clear();
        auto good = dev->CreateSwapchain(rhi::Format::RGBA16Float, err, /*hdr=*/true);
        Check(good != nullptr, "and a half-float one is accepted");
        if (good)
            std::printf("    this display's headroom: %.2fx reference white\n",
                        double(eng::platform::DisplayHeadroom(
                            good->NativeLayer())));
    }

    std::printf(g_failures == 0 ? "\nhdr_test: all checks passed\n"
                                : "\nhdr_test: %d FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
