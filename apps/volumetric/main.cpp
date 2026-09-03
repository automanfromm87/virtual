// Froxel volumetric lighting, checked for the three things the analytic height
// fog it replaces cannot do.
//
// "There is fog" is not the test -- the analytic integral already puts fog in
// the distance and does it more cheaply. The claims that need checking are the
// ones that need a volume:
//
//   a SHAFT: the air is brighter where the sun reaches it than where a wall
//            shadows it, at the same distance from the camera
//   a GLOW:  the air around a lamp is lit by the lamp
//   FORWARD scattering: looking toward the sun is brighter than looking away
#include "engine/geometry/mesh.h"
#include "engine/render/renderer.h"
#include "engine/render/volumetric.h"
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

constexpr int kW = 480, kH = 270;

double Luma(const std::vector<std::uint8_t>& p, std::size_t i) {
    return 0.2126 * p[i] + 0.7152 * p[i + 1] + 0.0722 * p[i + 2];
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
    auto vol = Volumetrics::Create(*dev, error);
    if (!r || !vol) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }

    // A wall with a slot cut in it, and the sun behind. The slot lets a single
    // shaft through; everything else is in shadow. Nothing but a volume can
    // produce the bright stripe of air.
    const MeshHandle slab =
        r->UploadMesh(MakeBox(Vec3{6.0f, 5.0f, 0.3f}, Vec4{1, 1, 1, 1}));
    const MeshHandle floor_mesh =
        r->UploadMesh(MakeBox(Vec3{20.0f, 0.2f, 20.0f}, Vec4{1, 1, 1, 1}));
    // A DARK floor. Over a bright surface the fog mostly ATTENUATES -- which is
    // correct physics and a hopeless measurement, because the shaft's added
    // light is then a small perturbation on a large number. Against a nearly
    // black floor the fog's contribution is the entire signal.
    MaterialDesc md;
    md.base_color = Vec4{0.03f, 0.03f, 0.03f, 1.0f};
    md.roughness = 0.95f;
    const MaterialHandle mat = r->CreateMaterial(md, error);

    const rhi::TextureId hdr =
        dev->CreateRenderTarget(kW, kH, Renderer::kSceneFormat);
    const rhi::TextureId ldr =
        dev->CreateRenderTarget(kW, kH, kFmt, /*cpu_readable=*/true);
    const rhi::TextureId depth = dev->CreateDepthTarget(kW, kH, /*sampleable=*/true);
    const rhi::TextureId shadow = dev->CreateShadowMap(2048);
    std::vector<std::uint8_t> px(std::size_t(kW) * kH * 4);

    bool fog_shadows = true;
    const auto render = [&](const Scene& scene, bool fog) -> bool {
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
            pd.keep_depth = true;
            auto e = dev->BeginPass(pd);
            r->DrawScene(e, scene, kW, kH, shadow);
            dev->EndPass();
        }
        if (fog) {
            {
                auto e = dev->BeginCompute("froxels");
                // The renderer's own light and cascade buffers, so the shafts
                // line up with the shadows that cast them.
                vol->Build(e, scene, kW, kH, 0,
                           fog_shadows ? shadow : rhi::TextureId{},
                           r->CascadeBuffer(),
                           r->CascadeOffset(), r->LightBuffer(), r->LightOffset(),
                           int(scene.lights.size()));
                dev->EndCompute();
            }
            {
                rhi::PassDesc pd;
                pd.color = hdr;
                pd.load = true;
                auto e = dev->BeginPass(pd);
                vol->Apply(e, scene, kW, kH, depth);
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

    // A ROOF WITH A HOLE, and the sun nearly overhead. The air under the roof
    // is in shadow everywhere except the column below the hole.
    //
    // This is the third arrangement tried. A wall with a horizontal slot put
    // the shaft behind the wall from every camera that could see enough open
    // air to measure; the second put it off the top of the frame. A roof works
    // because the camera can stand inside the shadowed volume and look across
    // the shaft rather than along it.
    const auto scene_with_hole = [&]() {
        Scene s;
        // INSIDE the roofed volume, looking down at the floor. The ray from
        // every pixel then travels through shadowed air on its way to a
        // surface -- except the ones that cross the shaft.
        s.camera.eye = Vec3{0.0f, 2.2f, 8.0f};
        s.camera.target = Vec3{0.0f, 0.2f, -5.0f};
        s.camera.fovY = 1.0f;
        s.lightDir = Vec4{0.12f, 0.98f, 0.16f, 0.0f};  // nearly straight down
        s.lightColor = Vec4{6.0f, 5.8f, 5.4f, 1.0f};
        s.ambientSky = Vec3{0.01f, 0.01f, 0.015f};
        s.ambientGround = Vec3{0.005f, 0.005f, 0.005f};
        s.shadowExtent = 18.0f;

        Instance ground;
        ground.mesh = floor_mesh;
        ground.material = mat;
        ground.model = Mat4::Translation({0.0f, -0.2f, 0.0f});
        s.instances.push_back(ground);
        // Four roof panels at y = 6.5 leaving a 3 m square hole at the origin.
        const float kSpan = 9.0f, kHalfHole = 1.5f;
        const float centre = (kSpan + kHalfHole) * 0.5f;
        const float half = (kSpan - kHalfHole) * 0.5f;
        for (int i = 0; i < 4; ++i) {
            const bool along_x = i < 2;
            const float sign = (i & 1) ? 1.0f : -1.0f;
            Instance p;
            p.mesh = r->UploadMesh(MakeBox(
                along_x ? Vec3{half, 0.25f, kSpan} : Vec3{kSpan, 0.25f, half},
                Vec4{1, 1, 1, 1}));
            p.material = mat;
            p.model = Mat4::Translation(along_x ? Vec3{sign * centre, 6.5f, 0.0f}
                                                : Vec3{0.0f, 6.5f, sign * centre});
            s.instances.push_back(p);
        }
        return s;
    };

    // ------------------------------------------------------------ the shaft --
    {
        std::printf("a shaft of light through a slot\n");
        // A THIN, DARK medium for this one. The default is dense enough that
        // every ray saturates before it has crossed the frame, and a saturated
        // shaft and a saturated shadow are the same number -- the first version
        // of this measured 90.6 against 88.1 for exactly that reason. Zero
        // ambient too: the whole question is whether the air in shadow is
        // darker, and an ambient term puts a floor under it.
        VolumetricConfig cfg;
        cfg.base_density = 1.0f;
        cfg.height_density = 0.0f;
        cfg.ambient = Vec3{0.0f, 0.0f, 0.0f};
        cfg.scattering = Vec3{0.12f, 0.12f, 0.12f};
        cfg.extinction = 0.02f;
        cfg.far_distance = 40.0f;  // the roofed volume, not the horizon
        vol->SetConfig(cfg);
        const Scene s = scene_with_hole();
        if (!render(s, false)) return 1;
        const std::vector<std::uint8_t> no_fog = px;
        if (!render(s, true)) return 1;
        const std::vector<std::uint8_t> with_fog = px;

        // Scan the AIR -- the columns of the frame where nothing was drawn --
        // for the brightest and darkest rows. Found by measuring rather than
        // guessed: where a shaft lands on screen depends on the camera, the
        // slot and the sun together, and a rectangle picked by eye has been
        // wrong every time it has been tried in this repository.
        // IS THE FOG SHADOWED? Rendered twice with the identical scene and the
        // identical medium, differing only in whether the froxel pass is given
        // the shadow map.
        //
        // Three earlier versions of this looked for the shaft ON SCREEN and
        // each failed for its own reason: a wall with a slot put the shaft
        // behind the wall from every camera that could see enough air; scanning
        // rows averaged straight across a vertical stripe; and measuring air
        // pixels finds nothing at all, because a ray that hits no surface
        // integrates the whole volume including everything outside the roof.
        // Asking the question directly avoids all three -- if the shadow map
        // does not change the fog, there is no shaft to find.
        fog_shadows = false;
        if (!render(s, true)) return 1;
        const std::vector<std::uint8_t> unshadowed = px;
        fog_shadows = true;
        if (!render(s, true)) return 1;
        const std::vector<std::uint8_t> shadowed = px;

        double under_roof_lit = 0.0, under_roof_shadowed = 0.0;
        int n_roof = 0;
        for (int y = 4; y < kH - 4; ++y)
            for (int x = kW / 3; x < kW * 2 / 3; ++x, ++n_roof) {
                const std::size_t i = (std::size_t(y) * kW + x) * 4;
                under_roof_lit += Luma(unshadowed, i);
                under_roof_shadowed += Luma(shadowed, i);
            }
        under_roof_lit /= n_roof;
        under_roof_shadowed /= n_roof;
        std::printf("    fog under the roof: %.1f with the shadow map, %.1f "
                    "without it\n",
                    under_roof_shadowed, under_roof_lit);
        Check(under_roof_lit > 20.0, "unshadowed, the air under the roof is bright");
        Check(under_roof_shadowed < under_roof_lit * 0.7,
              "and the roof's shadow darkens it substantially");

        // AND THE SHAFT ITSELF. With the shadow map on, the column of air below
        // the hole must be brighter than the columns either side of it. This is
        // the same measurement as above restricted to what the hole projects
        // to, and it is what distinguishes "the fog is shadowed" from "there is
        // a shaft".
        const auto column = [&](const std::vector<std::uint8_t>& img, int x0,
                                int x1) {
            double sum = 0.0;
            int n = 0;
            for (int y = 4; y < kH - 4; ++y)
                for (int x = x0; x < x1; ++x, ++n)
                    sum += Luma(img, (std::size_t(y) * kW + x) * 4);
            return n ? sum / n : 0.0;
        };
        int best_x = kW / 2;
        double best = -1.0;
        for (int x = kW / 4; x < kW * 3 / 4 - 16; x += 4) {
            const double c = column(shadowed, x, x + 16);
            if (c > best) { best = c; best_x = x; }
        }
        const double left = column(shadowed, kW / 4, kW / 4 + 16);
        std::printf("    brightest 16-pixel column at x = %d (%.1f) against "
                    "%.1f at the edge of the roofed area\n",
                    best_x, best, left);
        Check(best > left * 1.3,
              "and there is a distinctly brighter column where the hole is");
        vol->SetConfig(VolumetricConfig{});
    }

    // -------------------------------------------------------------- the glow --
    {
        std::printf("\na lamp glowing in the air around it\n");
        Scene s;
        s.camera.eye = Vec3{0.0f, 1.5f, 8.0f};
        s.camera.target = Vec3{0.0f, 1.5f, 0.0f};
        s.lightColor = Vec4{0, 0, 0, 1};  // no sun at all
        s.ambientSky = Vec3{0.0f, 0.0f, 0.0f};
        s.ambientGround = Vec3{0.0f, 0.0f, 0.0f};
        // No geometry either. Whatever is on screen came out of the air.
        Light lamp;
        lamp.type = LightType::Point;
        lamp.position = Vec3{0.0f, 1.5f, 0.0f};
        lamp.color = Vec3{30.0f, 12.0f, 4.0f};
        lamp.range = 6.0f;
        s.lights.push_back(lamp);

        VolumetricConfig cfg = vol->Config();
        cfg.ambient = Vec3{0.0f, 0.0f, 0.0f};  // isolate the lamp
        vol->SetConfig(cfg);

        if (!render(s, false)) return 1;
        const double empty = Luma(px, (std::size_t(kH / 2) * kW + kW / 2) * 4);
        if (!render(s, true)) return 1;
        const double at_lamp = Luma(px, (std::size_t(kH / 2) * kW + kW / 2) * 4);
        const double at_edge = Luma(px, (std::size_t(kH / 2) * kW + 20) * 4);
        const std::size_t c = (std::size_t(kH / 2) * kW + kW / 2) * 4;
        std::printf("    empty frame %.1f, with fog: %.1f at the lamp, %.1f at "
                    "the edge  (rgb %d %d %d)\n",
                    empty, at_lamp, at_edge, px[c], px[c + 1], px[c + 2]);
        Check(empty < 1.0, "with no sun, no ambient and no geometry the frame is black");
        Check(at_lamp > 12.0, "and the lamp lights the air around it");
        Check(at_lamp > at_edge * 3.0, "brightest where the lamp is");
        Check(px[c] > px[c + 1] && px[c + 1] > px[c + 2],
              "and the glow is the lamp's colour");

        vol->SetConfig(VolumetricConfig{});
    }

    // ------------------------------------------------------ forward scatter --
    {
        std::printf("\nforward scattering: brighter toward the light\n");
        // Same air, same sun, camera turned to face away. A phase function of
        // zero would give the same number both ways, which is the grey wash
        // that makes fog look like a colour ramp.
        Scene s;
        s.camera.eye = Vec3{0.0f, 2.0f, 0.0f};
        s.lightDir = Vec4{0.0f, 0.25f, -0.97f, 0.0f};
        s.lightColor = Vec4{6.0f, 6.0f, 6.0f, 1.0f};
        s.ambientSky = Vec3{0.0f, 0.0f, 0.0f};
        s.ambientGround = Vec3{0.0f, 0.0f, 0.0f};

        VolumetricConfig cfg;
        cfg.ambient = Vec3{0.0f, 0.0f, 0.0f};
        vol->SetConfig(cfg);

        s.camera.target = Vec3{0.0f, 2.5f, -10.0f};  // toward the sun
        if (!render(s, true)) return 1;
        const double toward = Luma(px, (std::size_t(kH / 2) * kW + kW / 2) * 4);
        s.camera.target = Vec3{0.0f, 2.5f, 10.0f};   // away from it
        if (!render(s, true)) return 1;
        const double away = Luma(px, (std::size_t(kH / 2) * kW + kW / 2) * 4);
        std::printf("    looking toward the sun %.1f, away from it %.1f\n", toward,
                    away);
        Check(toward > away * 1.5,
              "the air is brighter looking into the light than away from it");

        cfg.anisotropy = 0.0f;
        vol->SetConfig(cfg);
        s.camera.target = Vec3{0.0f, 2.5f, -10.0f};
        if (!render(s, true)) return 1;
        const double iso_toward = Luma(px, (std::size_t(kH / 2) * kW + kW / 2) * 4);
        s.camera.target = Vec3{0.0f, 2.5f, 10.0f};
        if (!render(s, true)) return 1;
        const double iso_away = Luma(px, (std::size_t(kH / 2) * kW + kW / 2) * 4);
        std::printf("    with anisotropy 0: %.1f and %.1f\n", iso_toward, iso_away);
        Check(std::fabs(iso_toward - iso_away) < 2.0,
              "and with an isotropic phase function it is the same both ways");
    }

    std::printf(g_failures == 0 ? "\nvolumetric_test: all checks passed\n"
                                : "\nvolumetric_test: %d FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
