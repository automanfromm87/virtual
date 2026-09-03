// PROJECTED DECALS, measured.
//
// A decal system fails in ways that look deliberate. It projects onto surfaces
// facing the wrong way and the smear reads as a texture choice. It draws in
// front of geometry it should be behind and reads as a z-fighting problem. It
// disappears when the camera walks into it and reads as a fade someone added.
//
// So: a red decal on a white floor with a white wall beside it. Where the red
// ends up is the whole test.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

#include "engine/geometry/mesh.h"
#include "engine/render/decals.h"
#include "engine/render/renderer.h"
#include "engine/rhi/rhi.h"
#include "engine/scene/scene.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

constexpr int kW = 320, kH = 320;

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

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string error;
    auto dev = eng::rhi::Device::Create(error);
    if (!dev) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
    const auto kOut = eng::rhi::Format::RGBA8Unorm;
    auto r = eng::Renderer::Create(*dev, kOut, error, 1);
    if (!r) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
    auto decals = eng::DecalSystem::Create(*dev, error);
    if (!decals) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    // A solid red decal texture. Solid so that "did the decal land here" is a
    // question about position rather than about which texel was sampled.
    std::vector<std::uint8_t> red(32 * 32 * 4);
    for (std::size_t i = 0; i < 32 * 32; ++i) {
        red[i * 4 + 0] = 255;
        red[i * 4 + 1] = 0;
        red[i * 4 + 2] = 0;
        red[i * 4 + 3] = 255;
    }
    const eng::rhi::TextureId red_tex = dev->CreateTexture2D(32, 32, red.data());

    eng::MaterialDesc md;
    md.shading = eng::Shading::Lit;
    md.base_color = eng::Vec4{0.9f, 0.9f, 0.9f, 1.0f};
    md.roughness = 0.8f;
    const eng::MaterialHandle mat = r->CreateMaterial(md, error);
    const eng::MeshHandle box = r->UploadMesh(
        eng::MakeBox(eng::Vec3{0.5f, 0.5f, 0.5f}, eng::Vec4{1, 1, 1, 1}));
    if (!eng::Valid(mat) || !eng::rhi::Valid(red_tex)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }

    eng::Scene scene;
    scene.lightDir = eng::Vec4{0.2f, 0.95f, 0.2f, 0.0f};
    scene.lightColor = eng::Vec4{2.5f, 2.5f, 2.5f, 1.0f};
    scene.ambientSky = eng::Vec3{0.3f, 0.3f, 0.32f};
    scene.ambientGround = eng::Vec3{0.1f, 0.1f, 0.1f};
    // A floor, and a wall standing on it at x = 2.
    {
        eng::Instance floor;
        floor.mesh = box;
        floor.material = mat;
        floor.model = eng::Mat4::Translation(eng::Vec3{0.0f, -0.5f, 0.0f}) *
                      eng::Mat4::Scale(1.0f);
        // A wide flat slab.
        floor.model = eng::Mat4{{{16.0f, 0, 0, 0}, {0, 0.4f, 0, 0}, {0, 0, 16.0f, 0},
                                 {0.0f, -0.2f, 0.0f, 1.0f}}};
        scene.instances.push_back(floor);

        eng::Instance wall;
        wall.mesh = box;
        wall.material = mat;
        wall.model = eng::Mat4{{{0.4f, 0, 0, 0}, {0, 4.0f, 0, 0}, {0, 0, 8.0f, 0},
                                {3.0f, 2.0f, 0.0f, 1.0f}}};
        scene.instances.push_back(wall);
    }
    // Looking down at the floor from above and in front.
    scene.camera.eye = eng::Vec3{0.0f, 7.0f, 7.0f};
    scene.camera.target = eng::Vec3{0.0f, 0.0f, 0.0f};

    const eng::rhi::TextureId albedo =
        dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
    const eng::rhi::TextureId normal_metal =
        dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
    const eng::rhi::TextureId hdr =
        dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
    const eng::rhi::TextureId out =
        dev->CreateRenderTarget(kW, kH, kOut, /*cpu_readable=*/true);
    const eng::rhi::TextureId depth =
        dev->CreateDepthTarget(kW, kH, /*sampleable=*/true);
    std::vector<std::uint8_t> px(std::size_t(kW) * kH * 4);

    const auto frame = [&](std::span<const eng::Decal> list) -> bool {
        dev->BeginFrame();
        {
            eng::rhi::PassDesc pd;
            pd.color = albedo;
            pd.extra_colors = {normal_metal};
            pd.depth = depth;
            pd.keep_depth = true;
            auto e = dev->BeginPass(pd);
            r->DrawGBuffer(e, scene, kW, kH);
            dev->EndPass();
        }
        if (!list.empty()) {
            eng::rhi::PassDesc pd;
            pd.color = albedo;
            // LOAD. A decal blends over the albedo the geometry pass wrote; a
            // pass that cleared first would leave the decals alone on black,
            // and the lighting pass would then light a black world.
            pd.load = true;
            auto e = dev->BeginPass(pd);
            decals->Draw(e, scene.camera, kW, kH, depth, normal_metal, list);
            dev->EndPass();
        }
        {
            eng::rhi::PassDesc pd;
            pd.color = hdr;
            auto e = dev->BeginPass(pd);
            r->DrawDeferredLight(e, scene, kW, kH, albedo, normal_metal, depth, {});
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
        return dev->ReadPixels(out, kW, kH, px);
    };

    // "Redness": how much more red than green a region is. The scene is grey,
    // so anything positive is the decal.
    const auto redness = [&](int x0, int y0, int x1, int y1) {
        return Mean(px, 0, x0, y0, x1, y1) - Mean(px, 1, x0, y0, x1, y1);
    };

    // A patch around where a WORLD POINT lands on screen.
    //
    // Hand-picked pixel rectangles were the first version and they were wrong:
    // the "wall" rectangle contained a good deal of floor, so its redness never
    // dropped when the wall stopped being painted. Projecting the point the
    // check is actually about removes the guesswork.
    const auto at_world = [&](eng::Vec3 world, int half) {
        const eng::Mat4 vp = scene.camera.ViewProj(float(kW) / float(kH));
        const eng::Vec4 clip = vp * eng::Vec4{world.x, world.y, world.z, 1.0f};
        const float sx = (clip.x / clip.w * 0.5f + 0.5f) * kW;
        const float sy = (0.5f - clip.y / clip.w * 0.5f) * kH;
        const int x = std::clamp(int(sx), half, kW - half - 1);
        const int y = std::clamp(int(sy), half, kH - half - 1);
        return redness(x - half, y - half, x + half, y + half);
    };

    {
        std::printf("a decal lands on the floor under it\n");
        if (!frame({})) return 1;
        const float before = redness(130, 150, 190, 210);

        eng::Decal d;
        // A box two metres across, centred just above the origin, projecting
        // straight down: -Z of the decal must point INTO the surface, so the
        // box is rotated to face down.
        d.model = eng::Mat4::Translation(eng::Vec3{0.0f, 0.5f, 0.0f}) *
                  eng::Mat4::RotationX(-1.5708f) *
                  eng::Mat4{{{2.0f, 0, 0, 0}, {0, 2.0f, 0, 0}, {0, 0, 2.0f, 0},
                             {0, 0, 0, 1}}};
        d.texture = red_tex;
        d.normal_fade = 0.3f;
        const eng::Decal list[1] = {d};
        if (!frame(list)) return 1;
        const float after = redness(130, 150, 190, 210);
        std::printf("    redness under the decal: %.1f -> %.1f\n", before, after);
        Check(decals->LastDecalCount() == 1, "the decal was submitted");
        Check(decals->LastDrawCalls() == 1, "in one draw call");
        Check(after > before + 40.0f, "and the floor beneath it turned red");

        // AND NOWHERE ELSE. A projection test that passed everything would
        // paint the whole screen, and a normal fade that rejected nothing would
        // paint the wall.
        const float far_corner = redness(8, 280, 60, 312);
        std::printf("    redness in the far corner: %.1f\n", far_corner);
        Check(far_corner < before + 8.0f, "and the rest of the floor did not");
    }

    {
        std::printf("\nthe normal fade keeps it off the wall\n");
        // A decal box big enough to swallow the wall as well as the floor,
        // still projecting downward. Without the normal test the wall gets the
        // texture stretched down it in vertical stripes -- the artefact that
        // makes people conclude projected decals look bad.
        eng::Decal d;
        d.model = eng::Mat4::Translation(eng::Vec3{2.4f, 1.5f, 0.0f}) *
                  eng::Mat4::RotationX(-1.5708f) *
                  eng::Mat4{{{5.0f, 0, 0, 0}, {0, 5.0f, 0, 0}, {0, 0, 6.0f, 0},
                             {0, 0, 0, 1}}};
        d.texture = red_tex;

        // A point on the wall's visible face, and one on the floor beside it.
        // Both are inside the decal box.
        const eng::Vec3 on_wall{2.79f, 2.0f, 0.0f};
        const eng::Vec3 on_floor{1.4f, 0.0f, 0.0f};

        d.normal_fade = -1.0f;  // accept anything
        eng::Decal permissive[1] = {d};
        if (!frame(permissive)) return 1;
        const float wall_permissive = at_world(on_wall, 5);
        const float floor_permissive = at_world(on_floor, 5);

        d.normal_fade = 0.6f;  // only near-horizontal surfaces
        eng::Decal strict[1] = {d};
        if (!frame(strict)) return 1;
        const float wall_strict = at_world(on_wall, 5);
        const float floor_strict = at_world(on_floor, 5);
        std::printf("    floor redness: fade off %.1f, fade on %.1f\n",
                    floor_permissive, floor_strict);
        std::printf("    wall redness: fade off %.1f, fade on %.1f (floor %.1f)\n",
                    wall_permissive, wall_strict, floor_strict);
        Check(wall_permissive > 15.0f, "without the fade the wall is painted too");
        Check(wall_strict < wall_permissive * 0.4f, "and with it the wall is spared");
        Check(floor_strict > 30.0f, "while the floor still receives it");
    }

    {
        std::printf("\nopacity and tint do what they say\n");
        eng::Decal d;
        d.model = eng::Mat4::Translation(eng::Vec3{0.0f, 0.5f, 0.0f}) *
                  eng::Mat4::RotationX(-1.5708f) *
                  eng::Mat4{{{2.0f, 0, 0, 0}, {0, 2.0f, 0, 0}, {0, 0, 2.0f, 0},
                             {0, 0, 0, 1}}};
        d.texture = red_tex;
        d.opacity = 1.0f;
        eng::Decal full[1] = {d};
        if (!frame(full)) return 1;
        const float opaque = redness(140, 160, 180, 200);

        d.opacity = 0.3f;
        eng::Decal faint[1] = {d};
        if (!frame(faint)) return 1;
        const float sheer = redness(140, 160, 180, 200);
        std::printf("    redness at opacity 1.0: %.1f, at 0.3: %.1f\n", opaque, sheer);
        Check(sheer < opaque * 0.6f, "lower opacity shows less of the decal");
        Check(sheer > 4.0f, "but not none of it");
    }

    {
        std::printf("\nmany decals sharing a texture are one draw call\n");
        std::vector<eng::Decal> many;
        for (int i = 0; i < 64; ++i) {
            eng::Decal d;
            const float x = float(i % 8) - 3.5f;
            const float z = float(i / 8) - 3.5f;
            d.model = eng::Mat4::Translation(eng::Vec3{x, 0.5f, z}) *
                      eng::Mat4::RotationX(-1.5708f) *
                      eng::Mat4{{{0.7f, 0, 0, 0}, {0, 0.7f, 0, 0}, {0, 0, 2.0f, 0},
                                 {0, 0, 0, 1}}};
            d.texture = red_tex;
            many.push_back(d);
        }
        if (!frame(many)) return 1;
        std::printf("    %d decals -> %d draw calls\n", decals->LastDecalCount(),
                    decals->LastDrawCalls());
        Check(decals->LastDecalCount() == 64, "all 64 were submitted");
        Check(decals->LastDrawCalls() == 1, "as a single instanced draw");
    }

    {
        std::printf("\nnothing is painted where there is no geometry\n");
        // A decal box floating in the air over the edge of the world. There is
        // no surface inside it, so it must produce nothing -- a decal that
        // paints the background is one that ignored the depth buffer.
        eng::Decal d;
        d.model = eng::Mat4::Translation(eng::Vec3{0.0f, 40.0f, 0.0f}) *
                  eng::Mat4{{{6.0f, 0, 0, 0}, {0, 6.0f, 0, 0}, {0, 0, 6.0f, 0},
                             {0, 0, 0, 1}}};
        d.texture = red_tex;
        eng::Decal list[1] = {d};
        if (!frame({})) return 1;
        const float before = Mean(px, 0, 0, 0, kW, kH);
        if (!frame(list)) return 1;
        const float after = Mean(px, 0, 0, 0, kW, kH);
        std::printf("    frame mean red: %.2f -> %.2f\n", before, after);
        Check(std::fabs(after - before) < 1.0f,
              "a decal with nothing inside it changes nothing");
    }

    std::printf(g_failures == 0 ? "\ndecal_test: all checks passed\n"
                                : "\ndecal_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
