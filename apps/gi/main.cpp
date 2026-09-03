// The baked irradiance volume, rendered.
//
// engine/render/gi_test proves the bake is right. It cannot prove the bake
// REACHES THE SCREEN, and the gap between the two is where the mistakes live:
// a half-texel offset in the 3D lookup, the SH constants written a second time
// in MSL and differing from the C++, the coefficients uploaded per-coefficient
// instead of per-channel. Every one of those produces indirect light that is
// present and wrong, which looks exactly like indirect light that is present
// and right.
#include "engine/geometry/mesh.h"
#include "engine/render/gi.h"
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

constexpr int kW = 400, kH = 300;

// A quad's two triangles, for the bake.
void AddQuad(std::vector<eng::GiTriangle>& out, eng::Vec3 a, eng::Vec3 b,
             eng::Vec3 c, eng::Vec3 d, eng::Vec3 albedo) {
    out.push_back(eng::GiTriangle{a, b, c, albedo, eng::Vec3{0, 0, 0}});
    out.push_back(eng::GiTriangle{a, c, d, albedo, eng::Vec3{0, 0, 0}});
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

    // A white floor with a tall red wall on the left, seen from the right and
    // above. Direct light comes from over the wall, so the wall's inward face
    // is lit and the floor beside it is in the wall's own shadow -- which makes
    // the bounce the ONLY light reaching that strip.
    const float kRoom = 6.0f;
    const MeshHandle floor_mesh =
        r->UploadMesh(MakeBox(Vec3{kRoom, 0.1f, kRoom}, Vec4{1, 1, 1, 1}));
    const MeshHandle wall_mesh =
        r->UploadMesh(MakeBox(Vec3{0.1f, 3.0f, kRoom}, Vec4{1, 1, 1, 1}));

    MaterialDesc white;
    white.base_color = Vec4{0.85f, 0.85f, 0.85f, 1.0f};
    white.roughness = 0.95f;
    MaterialDesc red;
    red.base_color = Vec4{0.75f, 0.06f, 0.05f, 1.0f};
    red.roughness = 0.95f;
    const MaterialHandle white_mat = r->CreateMaterial(white, error);
    const MaterialHandle red_mat = r->CreateMaterial(red, error);
    if (!Valid(white_mat) || !Valid(red_mat)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }

    Scene scene;
    scene.camera.eye = Vec3{2.5f, 3.2f, 7.0f};
    scene.camera.target = Vec3{-2.0f, 0.3f, 0.0f};
    scene.camera.fovY = 0.9f;
    // The sun comes from the far side of the wall, so the strip of floor
    // beside it gets no direct light at all.
    scene.lightDir = Vec4{-0.62f, 0.72f, 0.31f, 0.0f};
    scene.lightColor = Vec4{3.5f, 3.5f, 3.5f, 1.0f};
    // Ambient near zero: whatever brightens the shadowed strip has to be the
    // volume, and a hemisphere ambient would drown it.
    scene.ambientSky = Vec3{0.02f, 0.02f, 0.03f};
    scene.ambientGround = Vec3{0.01f, 0.01f, 0.01f};
    scene.shadowExtent = 8.0f;

    Instance ground;
    ground.mesh = floor_mesh;
    ground.material = white_mat;
    ground.model = Mat4::Translation({0.0f, -0.1f, 0.0f});
    scene.instances.push_back(ground);
    Instance wall;
    wall.mesh = wall_mesh;
    wall.material = red_mat;
    wall.model = Mat4::Translation({-3.0f, 3.0f, 0.0f});
    scene.instances.push_back(wall);

    // The same geometry for the bake. Written out rather than derived from the
    // meshes: the bake wants world-space triangles with an albedo, and going
    // through the renderer's mesh and material tables to recover them would
    // couple the bake to the vertex format for no gain in a test.
    std::vector<GiTriangle> tris;
    AddQuad(tris, Vec3{-kRoom, 0, -kRoom}, Vec3{kRoom, 0, -kRoom},
            Vec3{kRoom, 0, kRoom}, Vec3{-kRoom, 0, kRoom},
            Vec3{0.85f, 0.85f, 0.85f});
    AddQuad(tris, Vec3{-2.9f, 0, -kRoom}, Vec3{-2.9f, 6, -kRoom},
            Vec3{-2.9f, 6, kRoom}, Vec3{-2.9f, 0, kRoom},
            Vec3{0.75f, 0.06f, 0.05f});

    GiBakeConfig cfg;
    cfg.nx = 14;
    cfg.ny = 6;
    cfg.nz = 10;
    cfg.origin = Vec3{-5.5f, 0.25f, -5.0f};
    cfg.spacing = Vec3{0.85f, 0.9f, 1.1f};
    cfg.rays = 512;
    cfg.bounces = 2;
    cfg.sun_direction = Vec3{-0.62f, 0.72f, 0.31f};
    cfg.sun_color = Vec3{3.5f, 3.5f, 3.5f};
    cfg.sky_top = Vec3{0.05f, 0.05f, 0.07f};
    cfg.sky_bottom = Vec3{0.02f, 0.02f, 0.02f};
    cfg.threads = 8;

    const IrradianceVolume vol = IrradianceVolume::Bake(tris, cfg);
    std::printf("baked %zu probes, %d of them dark\n", vol.Probes().size(),
                vol.DarkProbes());

    const rhi::TextureId hdr =
        dev->CreateRenderTarget(kW, kH, Renderer::kSceneFormat);
    const rhi::TextureId ldr =
        dev->CreateRenderTarget(kW, kH, kFmt, /*cpu_readable=*/true);
    const rhi::TextureId depth = dev->CreateDepthTarget(kW, kH);
    const rhi::TextureId shadow = dev->CreateShadowMap(2048);
    std::vector<std::uint8_t> px(std::size_t(kW) * kH * 4);

    const auto render = [&]() -> bool {
        dev->BeginFrame();
        {
            rhi::PassDesc pd;
            pd.depth = shadow;
            pd.clear_depth = 0.0f;
            pd.keep_depth = true;
            auto e = dev->BeginPass(pd);
            r->DrawShadow(e, scene);
            dev->EndPass();
        }
        {
            rhi::PassDesc pd;
            pd.color = hdr;
            pd.depth = depth;
            pd.clear_depth = 0.0f;
            auto e = dev->BeginPass(pd);
            r->DrawScene(e, scene, kW, kH, shadow);
            dev->EndPass();
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

    // Two patches of FLOOR, both in the wall's shadow: one hard against it and
    // one a long way along.
    //
    // The sun is behind the wall, so nothing the camera can see is directly
    // lit. That was not the plan -- the plan had a lit patch as a control --
    // but it makes a sharper test than the plan did: with no volume the floor
    // is a flat ambient value with no structure whatsoever, and with one it has
    // to be red AND to fall off with distance from the wall. A constant ambient
    // term, or a volume sampled at a fixed place, reproduces the brightness and
    // neither reproduces the gradient.
    //
    // Located by scanning rather than guessed -- a rectangle picked by eye is
    // the thing that has gone wrong most often in this repository.
    struct Patch {
        double r = 0, g = 0, b = 0;
    };
    const auto measure = [&](int x0, int y0, int x1, int y1) {
        Patch p;
        int n = 0;
        for (int y = y0; y < y1; ++y)
            for (int x = x0; x < x1; ++x, ++n) {
                const std::size_t i = (std::size_t(y) * kW + x) * 4;
                p.r += px[i];
                p.g += px[i + 1];
                p.b += px[i + 2];
            }
        if (n) {
            p.r /= n;
            p.g /= n;
            p.b /= n;
        }
        return p;
    };

    r->ClearIrradianceVolume();
    if (!render()) return 1;
    // Find the shadowed strip: scan a row of the floor for the darkest run.
    int shadow_x = -1;
    double darkest = 1e9;
    for (int x = 40; x < kW - 40; x += 4) {
        const Patch p = measure(x, 210, x + 12, 230);
        const double l = 0.2126 * p.r + 0.7152 * p.g + 0.0722 * p.b;
        if (l < darkest && l > 1.0) {  // above 1: still floor, not background
            darkest = l;
            shadow_x = x;
        }
    }
    std::printf("the darkest floor strip is at x = %d (luma %.1f)\n", shadow_x,
                darkest);
    if (shadow_x < 0) {
        std::fprintf(stderr, "FAIL: could not find the shadowed strip\n");
        return 1;
    }
    const int far_x = std::min(kW - 60, shadow_x + 150);

    const Patch off_near = measure(shadow_x, 210, shadow_x + 12, 230);
    const Patch off_far = measure(far_x, 210, far_x + 12, 230);

    std::string upload_error;
    Check(r->SetIrradianceVolume(vol, upload_error),
          "the volume uploads to the GPU");
    Check(r->HasIrradianceVolume(), "and the renderer reports it bound");
    if (!render()) return 1;
    const Patch on_near = measure(shadow_x, 210, shadow_x + 12, 230);
    const Patch on_far = measure(far_x, 210, far_x + 12, 230);

    std::printf("  beside the wall  without GI %.1f %.1f %.1f   with GI %.1f %.1f %.1f\n",
                off_near.r, off_near.g, off_near.b, on_near.r, on_near.g,
                on_near.b);
    std::printf("  further along    without GI %.1f %.1f %.1f   with GI %.1f %.1f %.1f\n",
                off_far.r, off_far.g, off_far.b, on_far.r, on_far.g, on_far.b);

    const double off_luma = 0.2126 * off_near.r + 0.7152 * off_near.g;
    const double on_luma = 0.2126 * on_near.r + 0.7152 * on_near.g;
    Check(on_luma > off_luma * 1.5,
          "the shadowed floor is far brighter with the volume bound");
    // AND RED. Brighter alone would be satisfied by any ambient term; the
    // colour is what says the light came off the red wall.
    const double off_ratio = off_near.r / std::max(off_near.g, 0.5);
    const double on_ratio = on_near.r / std::max(on_near.g, 0.5);
    std::printf("  r/g beside the wall: %.2f without, %.2f with\n", off_ratio,
                on_ratio);
    Check(on_ratio > off_ratio * 1.3, "and it is redder, not just brighter");
    // AND IT HAS A GRADIENT. Without the volume these two patches are the same
    // number to within a level -- there is no structure in the shadow at all.
    // With it, the strip against the wall is brighter than the one a couple of
    // metres along, which is what says the volume is being sampled at the
    // fragment's own position rather than anywhere fixed.
    std::printf("  without GI the two patches differ by %.1f/255; with GI by %.1f\n",
                std::fabs(off_near.r - off_far.r),
                std::fabs(on_near.r - on_far.r));
    Check(std::fabs(off_near.r - off_far.r) < 2.0,
          "without the volume the shadow has no structure at all");
    Check(on_near.r > on_far.r * 1.15,
          "and with it the bounce falls off with distance from the wall");

    r->ClearIrradianceVolume();
    Check(!r->HasIrradianceVolume(), "and it can be unbound again");
    if (!render()) return 1;
    const Patch back = measure(shadow_x, 210, shadow_x + 12, 230);
    Check(std::fabs(back.r - off_near.r) < 1.5,
          "which restores the original image exactly");

    std::printf(g_failures == 0 ? "\ngi_render_test: all checks passed\n"
                                : "\ngi_render_test: %d FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
