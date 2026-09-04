// Transmission through a thin surface: the light that goes through a leaf.
//
// WHY THIS NEEDS ITS OWN TEST. Every other lighting check in this repo puts the
// sun in front of the thing it is looking at, because that is where a light
// belongs when you want to see a surface. This term only exists BEHIND, and it
// is exactly zero everywhere the others measure -- so nothing they do could
// have caught its absence, and for the whole life of the renderer a backlit
// leaf was black.
//
// The measurements are all against the same frame with transmission turned off,
// because "is it brighter" needs something to be brighter than.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/geometry/mesh.h"
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

}  // namespace

int main() {
    using namespace eng;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    std::string error;
    auto dev = rhi::Device::Create(error);
    if (!dev) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
    auto r = Renderer::Create(*dev, rhi::Format::RGBA8Unorm, error, 1);
    if (!r) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    // A THIN SLAB facing +z, which is what a leaf is. Thin matters: the term
    // models light crossing in one step, and a solid object is the case it is
    // explicitly not for.
    const MeshHandle leaf = r->UploadMesh(
        MakeBox(Vec3{1.0f, 1.0f, 0.01f}, Vec4{1, 1, 1, 1}));
    const rhi::TextureId hdr =
        dev->CreateRenderTarget(kW, kH, Renderer::kSceneFormat);
    const rhi::TextureId depth = dev->CreateDepthTarget(kW, kH);
    const rhi::TextureId out = dev->CreateRenderTarget(
        kW, kH, rhi::Format::RGBA8Unorm, /*cpu_readable=*/true);
    std::vector<std::uint8_t> px(std::size_t(kW) * kH * 4);

    // `eye_x` slides the camera off the sun's line, which is what tests the
    // forward bias. Everything else is fixed.
    const auto render = [&](float transmission, float eye_x) -> Vec3 {
        Scene s;
        s.camera.eye = Vec3{eye_x, 0.0f, 3.0f};
        s.camera.target = Vec3{0.0f, 0.0f, 0.0f};
        // TOWARD the light, so this puts the sun BEHIND the slab: it is at
        // -z and the camera is at +z.
        s.lightDir = Vec4{0.0f, 0.0f, -1.0f, 0.0f};
        s.lightColor = Vec4{4.0f, 4.0f, 4.0f, 1.0f};
        // No ambient at all. Any is enough to make a black surface merely dark,
        // and the whole question here is what the SUN does through the leaf.
        s.ambientSky = Vec3{0.0f, 0.0f, 0.0f};
        s.ambientGround = Vec3{0.0f, 0.0f, 0.0f};
        s.shadowExtent = 0.0f;

        MaterialDesc md;
        md.shading = Shading::Lit;
        // A dull red albedo, deliberately NOT the transmission colour, so the
        // two cannot be confused in the result.
        md.base_color = Vec4{0.35f, 0.10f, 0.08f, 1.0f};
        md.roughness = 0.85f;
        md.transmission = transmission;
        md.transmission_color = Vec3{0.20f, 0.60f, 0.10f};
        const MaterialHandle m = r->CreateMaterial(md, error);
        Instance in;
        in.mesh = leaf;
        in.material = m;
        s.instances.push_back(in);

        dev->BeginFrame();
        {
            rhi::PassDesc pd;
            pd.color = hdr;
            pd.depth = depth;
            pd.clear_depth = 0.0f;
            auto e = dev->BeginPass(pd);
            r->DrawScene(e, s, kW, kH);
            dev->EndPass();
        }
        {
            rhi::PassDesc pd;
            pd.color = out;
            auto e = dev->BeginPass(pd);
            r->DrawComposite(e, hdr, {}, {}, 0.0f, /*vignette=*/0.0f);
            dev->EndPass();
        }
        if (!dev->CommitAndWait(error)) {
            std::fprintf(stderr, "FAIL: submit: %s\n", error.c_str());
            std::exit(1);
        }
        if (!dev->ReadPixels(out, kW, kH, px)) {
            std::fprintf(stderr, "FAIL: readback\n");
            std::exit(1);
        }
        // The middle of the slab, averaged over a patch so one texel cannot
        // decide the answer.
        double c[3] = {0, 0, 0};
        int n = 0;
        for (int y = kH / 2 - 12; y < kH / 2 + 12; ++y)
            for (int x = kW / 2 - 12; x < kW / 2 + 12; ++x) {
                const std::size_t i = (std::size_t(y) * kW + x) * 4;
                for (int k = 0; k < 3; ++k) c[k] += px[i + k];
                ++n;
            }
        return Vec3{float(c[0] / n), float(c[1] / n), float(c[2] / n)};
    };

    {
        std::printf("a backlit leaf is black without it\n");
        const Vec3 off = render(0.0f, 0.0f);
        std::printf("    transmission 0, sun behind: %.1f %.1f %.1f\n", off.x, off.y,
                    off.z);
        // The reflective term is multiplied by saturate(dot(N, L)) and the sun
        // is on the far side, so it is exactly zero. With no ambient either,
        // there is nothing left. This is the state the renderer was in.
        Check(off.x + off.y + off.z < 12.0f,
              "the whole reflective model gives a backlit surface nothing");
    }

    {
        std::printf("\nand lights up with it\n");
        const Vec3 off = render(0.0f, 0.0f);
        const Vec3 on = render(0.6f, 0.0f);
        std::printf("    transmission 0.6: %.1f %.1f %.1f\n", on.x, on.y, on.z);
        Check(on.y > off.y + 40.0f, "the sun now comes through");

        // THE TRANSMITTED COLOUR, not the albedo. A leaf passes green far
        // better than red, which is why a backlit one is a more saturated
        // green than a lit one -- and using the albedo instead would give this
        // slab's dull RED, since that is what it reflects.
        std::printf("    albedo is red (0.35 0.10 0.08), transmission is green "
                    "(0.20 0.60 0.10)\n");
        Check(on.y > on.x * 1.5f,
              "and comes out the colour the leaf transmits, not the one it reflects");
    }

    {
        std::printf("\nit is forward-biased, not a flat add\n");
        // Straight through the leaf from the sun, then further and further off
        // that line. Light crossing a thin scatterer keeps most of its
        // direction, so this has to fall away -- a term that did not would be
        // an ambient light that only exists on one side.
        const Vec3 head_on = render(0.6f, 0.0f);
        const Vec3 off_axis = render(0.6f, 2.6f);
        const Vec3 side = render(0.6f, 6.0f);
        std::printf("    eye at x=0: %.1f   x=2.6: %.1f   x=6: %.1f  (green)\n",
                    head_on.y, off_axis.y, side.y);
        Check(head_on.y > off_axis.y * 1.25f, "it is brightest straight into the sun");
        Check(off_axis.y > side.y, "and keeps falling as you move off that line");
    }

    {
        std::printf("\nand does not touch a front-lit surface\n");
        // The same slab with the sun in FRONT. dot(V, -L) is negative there, so
        // the term is zero and a leaf lit from the camera's side has to look
        // exactly as it did before this existed. If transmission changed that,
        // every material in the engine would need re-tuning.
        const auto front = [&](float transmission) {
            Scene s;
            s.camera.eye = Vec3{0.0f, 0.0f, 3.0f};
            s.camera.target = Vec3{0.0f, 0.0f, 0.0f};
            s.lightDir = Vec4{0.0f, 0.0f, 1.0f, 0.0f};  // sun on the camera's side
            s.lightColor = Vec4{4.0f, 4.0f, 4.0f, 1.0f};
            s.ambientSky = Vec3{0.0f, 0.0f, 0.0f};
            s.ambientGround = Vec3{0.0f, 0.0f, 0.0f};
            s.shadowExtent = 0.0f;
            MaterialDesc md;
            md.shading = Shading::Lit;
            md.base_color = Vec4{0.35f, 0.10f, 0.08f, 1.0f};
            md.roughness = 0.85f;
            md.transmission = transmission;
            md.transmission_color = Vec3{0.20f, 0.60f, 0.10f};
            Instance in;
            in.mesh = leaf;
            in.material = r->CreateMaterial(md, error);
            s.instances.push_back(in);
            dev->BeginFrame();
            {
                rhi::PassDesc pd;
                pd.color = hdr;
                pd.depth = depth;
                pd.clear_depth = 0.0f;
                auto e = dev->BeginPass(pd);
                r->DrawScene(e, s, kW, kH);
                dev->EndPass();
            }
            {
                rhi::PassDesc pd;
                pd.color = out;
                auto e = dev->BeginPass(pd);
                r->DrawComposite(e, hdr, {}, {}, 0.0f, 0.0f);
                dev->EndPass();
            }
            std::string err;
            if (!dev->CommitAndWait(err)) std::exit(1);
            if (!dev->ReadPixels(out, kW, kH, px)) std::exit(1);
            const std::size_t i = (std::size_t(kH / 2) * kW + kW / 2) * 4;
            return float(px[i]) + float(px[i + 1]) + float(px[i + 2]);
        };
        const float plain = front(0.0f), leafy = front(0.6f);
        std::printf("    sun in front: %.1f without, %.1f with\n", plain, leafy);
        Check(std::fabs(plain - leafy) < 2.0f,
              "a front-lit surface is unchanged, so nothing else needs re-tuning");
    }

    std::printf(g_failures == 0 ? "\nfoliage_test: all checks passed\n"
                                : "\nfoliage_test: %d FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
