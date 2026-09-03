// Pure C++20. Renders one frame offscreen and writes it to a PNG.
//
// WHY this exists separately from every other offscreen app here: the others
// assert NUMBERS. That is the right way to test a BRDF -- a white furnace has
// an exact expected answer and a screenshot does not -- but it cannot answer
// "why does this look wrong", because a renderer can pass every energy check
// and still produce an image nobody would mistake for a photograph. The gap
// between "physically correct" and "looks real" is not made of bugs, and you
// cannot find it by reading means off a patch.
//
// It found two anyway, on its first run, both in code every numeric test walks
// past: the sky was discarded on every pixel by a reversed-Z depth compare, and
// the sky pipeline was built for the wrong attachment format. Neither is
// visible to a test that reads the probe instead of the frame.
//
// Two modes, and the difference between them is the answer to "why does a
// product render look like a photo and mine does not":
//
//   sky     -- the analytic atmosphere. Physically correct and featureless.
//   studio  -- a synthesised lighting rig: softboxes, a rim light, a floor.
//
// Same BRDF, same tone map, same spheres. What changes is what there is to
// reflect, and that is most of what "photographic" means for a curved surface.
#include "engine/asset/png.h"
#include "engine/geometry/mesh.h"
#include "engine/render/ibl.h"
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

constexpr int kW = 900, kH = 560;
constexpr int kCols = 6, kRows = 3;
constexpr float kRadius = 0.5f;
constexpr float kSpacing = 1.25f;
constexpr int kSamples = 4;

// ------------------------------------------------------------------ studio --
//
// A photographic studio is a small number of BIG, BRIGHT, RECTANGULAR sources
// in an otherwise dark room, and that shape is the whole trick. A softbox
// reflected in a sphere is a hard-edged bright quadrilateral that wraps with
// the curvature -- the eye reads that edge as a real object in a real room. The
// analytic sky has no edges anywhere, so a mirrored sphere under it shows a
// smooth gradient and reads as a shaded ball rather than a reflective one.
//
// Built as an equirectangular float image because that is what BakeEquirect
// takes, and float because the whole point is values far above one: the key
// light here is 40x the wall behind it, and an 8-bit source would clamp both to
// white and lose the ratio that makes the highlight a highlight.
struct Light {
    float yaw, pitch;      // centre direction, radians
    float half_w, half_h;  // angular half-size, radians
    float r, g, b;         // radiance inside
};

std::vector<float> StudioEquirect(int w, int h) {
    // KEY high and left, FILL low and right at a fifth the power, RIM behind
    // and above. The classic three-point rig: the key makes the form, the fill
    // keeps the shadow side from going black, and the rim separates the subject
    // from the background by drawing a bright line along its edge.
    const Light lights[] = {
        {-0.75f, 0.62f, 0.42f, 0.30f, 44.0f, 43.0f, 41.0f},  // key softbox
        {1.15f, 0.12f, 0.34f, 0.24f, 8.0f, 8.4f, 9.5f},      // fill, cooler
        {2.90f, 0.85f, 0.55f, 0.14f, 26.0f, 24.0f, 21.0f},   // rim, warm
    };

    std::vector<float> px(std::size_t(w) * h * 4, 0.0f);
    for (int y = 0; y < h; ++y) {
        const float v = (float(y) + 0.5f) / float(h);
        const float pitch = (0.5f - v) * 3.14159265f;  // +pi/2 up, -pi/2 down
        for (int x = 0; x < w; ++x) {
            const float u = (float(x) + 0.5f) / float(w);
            const float yaw = (u * 2.0f - 1.0f) * 3.14159265f;

            // The room. A dark cool wall above the horizon and a lighter warm
            // floor below it, with the seam softened -- a hard seam in the
            // environment shows up as a hard line across every glossy surface.
            const float t = std::clamp(pitch / 1.5708f, -1.0f, 1.0f);
            const float floor_mix = std::clamp(-t * 3.0f, 0.0f, 1.0f);
            float r = 0.055f + 0.30f * floor_mix;
            float g = 0.060f + 0.29f * floor_mix;
            float b = 0.075f + 0.27f * floor_mix;

            for (const Light& L : lights) {
                // Angular distance in each axis, with yaw wrapped: a source at
                // yaw 3.0 and a pixel at -3.0 are 0.28 radians apart, not 6.0.
                float dy = yaw - L.yaw;
                while (dy > 3.14159265f) dy -= 6.2831853f;
                while (dy < -3.14159265f) dy += 6.2831853f;
                const float dp = pitch - L.pitch;

                // A softbox is bright and flat inside its rectangle and falls
                // off fast at the edge. smoothstep over the outer tenth: a
                // perfectly hard edge aliases badly once the cube is only 128
                // texels a side, and a slow falloff stops looking like a panel.
                const float fx = 1.0f - std::clamp((std::fabs(dy) - L.half_w * 0.9f) /
                                                       (L.half_w * 0.1f + 1e-6f),
                                                   0.0f, 1.0f);
                const float fy = 1.0f - std::clamp((std::fabs(dp) - L.half_h * 0.9f) /
                                                       (L.half_h * 0.1f + 1e-6f),
                                                   0.0f, 1.0f);
                const float k = (fx * fx * (3.0f - 2.0f * fx)) *
                                (fy * fy * (3.0f - 2.0f * fy));
                r += L.r * k;
                g += L.g * k;
                b += L.b * k;
            }

            const std::size_t i = (std::size_t(y) * w + x) * 4;
            px[i + 0] = r;
            px[i + 1] = g;
            px[i + 2] = b;
            px[i + 3] = 1.0f;
        }
    }
    return px;
}

}  // namespace

int main(int argc, char** argv) {
    std::string out = "/tmp/shot.png";
    bool studio = false;
    // --ao writes the occlusion buffer instead of the frame. A debug view, and
    // the only way to tell "the AO pass is doing nothing" apart from "the AO
    // pass is doing something the composite is throwing away".
    bool dump_ao = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--studio") == 0) studio = true;
        else if (std::strcmp(argv[i], "--ao") == 0) dump_ao = true;
        else out = argv[i];
    }

    std::string error;
    auto dev = eng::rhi::Device::Create(error);
    if (!dev) {
        std::fprintf(stderr, "device: %s\n", error.c_str());
        return 1;
    }

    const auto kFmt = eng::rhi::Format::RGBA8Unorm;
    auto r = eng::Renderer::Create(*dev, kFmt, error, kSamples);
    if (!r) {
        std::fprintf(stderr, "renderer: %s\n", error.c_str());
        return 1;
    }
    // kSceneFormat, NOT kFmt. This argument is the format of the target DrawSky
    // will render into, and that is the HDR scene target, not the 8-bit one the
    // composite ends up in. Passing the wrong one does not fail: Metal builds
    // the pipeline, writes 8-bit output into a half-float attachment, and the
    // sky comes back with green and blue at exactly zero.
    auto env = eng::Environment::Create(*dev, error, 128,
                                        eng::Renderer::kSceneFormat, kSamples);
    if (!env) {
        std::fprintf(stderr, "environment: %s\n", error.c_str());
        return 1;
    }

    const eng::MeshHandle ball =
        r->UploadMesh(eng::MakeUVSphere(kRadius, 64, 96, eng::Vec4{1, 1, 1, 1},
                                        eng::Vec4{1, 1, 1, 1}));
    const eng::MeshHandle floor_mesh =
        r->UploadMesh(eng::MakeBox(eng::Vec3{14.0f, 0.15f, 14.0f},
                                   eng::Vec4{0.42f, 0.40f, 0.38f, 1.0f}));

    eng::Scene scene;
    eng::SkyConfig sky;
    // Mid-morning, off to the left, so the terminator falls across each sphere
    // rather than behind it. A sun directly behind the camera flattens
    // everything -- there is no shading gradient to see.
    sky.sun_direction = eng::Vec3{-0.45f, 0.42f, 0.35f};
    sky.turbidity = 2.4f;
    eng::Environment::ApplyTo(&scene, sky);

    if (studio) {
        // NO directional light and NO shadow map. Every photon in the studio
        // frame comes from the probe, which is the honest way to show what the
        // environment alone is worth -- and it is also how a real product shot
        // is lit, since a softbox IS the environment.
        scene.lightColor = eng::Vec4{0.0f, 0.0f, 0.0f, 1.0f};
        scene.ambientSky = eng::Vec3{0.0f, 0.0f, 0.0f};
        scene.ambientGround = eng::Vec3{0.0f, 0.0f, 0.0f};
        scene.shadowExtent = 0.0f;
    } else {
        scene.shadowExtent = 9.0f;
        scene.shadowCascades = 2;
    }

    scene.camera.eye = eng::Vec3{0.0f, 1.85f, 8.6f};
    scene.camera.target = eng::Vec3{0.0f, 1.35f, 0.0f};
    scene.camera.fovY = 0.60f;

    eng::MaterialDesc floor_desc;
    floor_desc.base_color = eng::Vec4{0.38f, 0.37f, 0.36f, 1.0f};
    floor_desc.roughness = studio ? 0.30f : 0.62f;
    floor_desc.metallic = 0.0f;
    const eng::MaterialHandle floor_mat = r->CreateMaterial(floor_desc, error);
    if (!eng::Valid(floor_mat)) {
        std::fprintf(stderr, "material: %s\n", error.c_str());
        return 1;
    }
    eng::Instance ground;
    ground.mesh = floor_mesh;
    ground.material = floor_mat;
    ground.model = eng::Mat4::Translation({0.0f, -0.15f, 0.0f});
    scene.instances.push_back(ground);

    // Roughness across, metal on the top row and a painted dielectric below.
    for (int row = 0; row < kRows; ++row)
        for (int col = 0; col < kCols; ++col) {
            eng::MaterialDesc md;
            md.roughness = 0.04f + 0.92f * float(col) / float(kCols - 1);
            md.metallic = row == 0 ? 1.0f : 0.0f;
            md.base_color = row == 0 ? eng::Vec4{0.95f, 0.93f, 0.88f, 1.0f}
                          : row == 1 ? eng::Vec4{0.55f, 0.09f, 0.07f, 1.0f}
                                     : eng::Vec4{0.14f, 0.26f, 0.42f, 1.0f};
            const eng::MaterialHandle m = r->CreateMaterial(md, error);
            if (!eng::Valid(m)) {
                std::fprintf(stderr, "material: %s\n", error.c_str());
                return 1;
            }
            eng::Instance in;
            in.mesh = ball;
            in.material = m;
            in.model = eng::Mat4::Translation(
                {(float(col) - float(kCols - 1) * 0.5f) * kSpacing,
                 kRadius + float(kRows - 1 - row) * kSpacing, 0.0f});
            scene.instances.push_back(in);
        }

    // Multisample attachments plus their single-sample resolves. SSAO reads the
    // depth, and a multisample depth buffer cannot be sampled, so the depth
    // here is single-sample and the geometry pass resolves colour into `hdr`.
    const eng::rhi::TextureId ms_color = dev->CreateRenderTarget(
        kW, kH, eng::Renderer::kSceneFormat, /*cpu_readable=*/false, kSamples);
    const eng::rhi::TextureId hdr =
        dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
    const eng::rhi::TextureId ms_depth =
        dev->CreateDepthTarget(kW, kH, /*sampleable=*/false, kSamples);
    const eng::rhi::TextureId depth = dev->CreateDepthTarget(kW, kH, true);
    const eng::rhi::TextureId ao = dev->CreateRenderTarget(kW, kH, kFmt, /*cpu_readable=*/true);
    const eng::rhi::TextureId ldr =
        dev->CreateRenderTarget(kW, kH, kFmt, /*cpu_readable=*/true);
    const eng::rhi::TextureId shadow = dev->CreateShadowMap(2048);
    const eng::rhi::TextureId bA = dev->CreateRenderTarget(
        kW / 2, kH / 2, eng::Renderer::kSceneFormat);
    const eng::rhi::TextureId bB = dev->CreateRenderTarget(
        kW / 2, kH / 2, eng::Renderer::kSceneFormat);
    const eng::rhi::TextureId bC = dev->CreateRenderTarget(
        kW / 4, kH / 4, eng::Renderer::kSceneFormat);
    const eng::rhi::TextureId bD = dev->CreateRenderTarget(
        kW / 4, kH / 4, eng::Renderer::kSceneFormat);

    // The environment source. A float texture in the studio case, because the
    // softboxes are 40x the wall and that ratio IS the lighting.
    eng::rhi::TextureId equirect;
    if (studio) {
        constexpr int kEW = 512, kEH = 256;
        const std::vector<float> px = StudioEquirect(kEW, kEH);
        equirect = dev->CreateTexture2DFloat(kEW, kEH, px.data());
        if (!eng::rhi::Valid(equirect)) {
            std::fprintf(stderr, "could not upload the studio environment\n");
            return 1;
        }
    }

    dev->BeginFrame();
    {
        auto e = dev->BeginCompute({});
        if (studio) env->BakeEquirect(e, equirect);
        else env->BakeSky(e, sky);
        dev->EndCompute();
    }
    if (!studio) {
        eng::rhi::PassDesc pd;
        pd.depth = shadow;
        pd.clear_depth = 0.0f;   // reversed-Z: 0 is the far plane
        pd.keep_depth = true;    // the lit pass samples it
        auto e = dev->BeginPass(pd);
        r->DrawShadow(e, scene);
        dev->EndPass();
    }
    // DEPTH PREPASS, single-sample, purely so SSAO has something to read.
    {
        eng::rhi::PassDesc pd;
        pd.depth = depth;
        pd.clear_depth = 0.0f;
        pd.keep_depth = true;
        auto e = dev->BeginPass(pd);
        r->DrawSceneDepth(e, scene, kW, kH);
        dev->EndPass();
    }
    {
        eng::rhi::PassDesc pd;
        pd.color = ao;
        auto e = dev->BeginPass(pd);
        r->DrawSsao(e, scene.camera, kW, kH, depth, 1.4f);
        dev->EndPass();
    }
    {
        eng::rhi::PassDesc pd;
        pd.color = ms_color;
        pd.resolve = hdr;
        pd.depth = ms_depth;
        pd.clear_depth = 0.0f;
        auto e = dev->BeginPass(pd);
        r->SetEnvironment(env->Bindings());
        r->DrawScene(e, scene, kW, kH, studio ? eng::rhi::TextureId{} : shadow);
        env->DrawSky(e, scene.camera, kW, kH);
        dev->EndPass();
    }
    // Bloom. The threshold is in LINEAR radiance and has to sit above the
    // diffuse level and below the highlights, or it either selects the whole
    // image or nothing at all.
    {
        struct Step {
            eng::rhi::TextureId src, dst;
            float dx, dy;
        };
        const float hx = 2.0f / float(kW), hy = 2.0f / float(kH);
        const float qx = 4.0f / float(kW), qy = 4.0f / float(kH);
        const Step steps[4] = {
            {bA, bB, hx, 0.0f}, {bB, bC, 0.0f, hy},
            {bC, bD, qx, 0.0f}, {bD, bA, 0.0f, qy}};
        {
            eng::rhi::PassDesc pd;
            pd.color = bA;
            auto e = dev->BeginPass(pd);
            r->DrawBloomBright(e, hdr, 2.4f, 0.7f);
            dev->EndPass();
        }
        for (const Step& s : steps) {
            eng::rhi::PassDesc pd;
            pd.color = s.dst;
            auto e = dev->BeginPass(pd);
            r->DrawBloomBlur(e, s.src, s.dx, s.dy);
            dev->EndPass();
        }
    }
    {
        eng::rhi::PassDesc pd;
        pd.color = ldr;
        auto e = dev->BeginPass(pd);
        r->DrawComposite(e, hdr, ao, bA, 0.55f, /*vignette=*/0.85f);
        dev->EndPass();
    }
    if (!dev->CommitAndWait(error)) {
        std::fprintf(stderr, "submit: %s\n", error.c_str());
        return 1;
    }

    std::vector<std::uint8_t> pixels(std::size_t(kW) * kH * 4);
    if (!dev->ReadPixels(dump_ao ? ao : ldr, kW, kH, pixels)) {
        std::fprintf(stderr, "readback failed\n");
        return 1;
    }
    if (!eng::png::EncodeFile(out, pixels, kW, kH, error)) {
        std::fprintf(stderr, "write: %s\n", error.c_str());
        return 1;
    }
    std::printf("wrote %s (%dx%d, %s)\n", out.c_str(), kW, kH,
                studio ? "studio" : "sky");
    return 0;
}
