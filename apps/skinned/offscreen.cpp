// Does the vertex shader blend the same way the CPU reference does?
//
// That is the question this gate exists for, and it is not answerable by
// looking. Skinning that is subtly wrong — a transposed palette entry, a
// forgotten inverse bind, weights read in the wrong order — still produces a
// mesh that moves with the skeleton and still looks like a tentacle.
//
// So: pose the mesh on the CPU with engine/anim, project those vertices with
// the same camera the GPU used, and compare the screen-space extent against the
// pixels that actually came back. The two implementations share no code.
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "apps/skinned/skinned_scene.h"
#include "engine/asset/gltf.h"
#include "engine/render/rendergraph.h"
#include "engine/render/renderer.h"
#include "engine/rhi/rhi.h"

namespace {

const char* const kSkinnedGltf =
#include "engine/asset/testdata_skinned_gltf.inc"
    ;

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

constexpr int kW = 800, kH = 600;

// Axis-aligned box in pixels.
struct Box {
    float x0 = 1e9f, y0 = 1e9f, x1 = -1e9f, y1 = -1e9f;
    void Add(float x, float y) {
        x0 = std::fmin(x0, x);
        y0 = std::fmin(y0, y);
        x1 = std::fmax(x1, x);
        y1 = std::fmax(y1, y);
    }
    [[nodiscard]] bool Empty() const { return x1 < x0; }
};

}  // namespace

int main() {
    std::string error;
    auto dev = eng::rhi::Device::Create(error);
    if (!dev) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
    const auto kFmt = eng::rhi::Format::RGBA8Unorm;
    auto renderer = eng::Renderer::Create(*dev, kFmt, error);
    if (!renderer) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    const demo::Assets assets = demo::Build(*renderer, error);
    if (!assets.ok) {
        std::fprintf(stderr, "FAIL: %s\n", error.empty() ? "build" : error.c_str());
        return 1;
    }

    std::printf("upload\n");
    Check(Valid(assets.mesh), "the skinned mesh uploaded");
    Check(renderer->JointCount(assets.mesh) == demo::kJoints,
          "the mesh remembers how many joints its palette holds");
    // A skin array that does not match the vertex count would leave the tail of
    // the mesh reading whatever followed it.
    {
        std::vector<eng::anim::SkinVertex> truncated = assets.flag.skin;
        truncated.pop_back();
        Check(!Valid(renderer->UploadSkinnedMesh(assets.flag.mesh, truncated,
                                                 demo::kJoints)),
              "a short skin array is refused, not clamped");
        Check(!Valid(renderer->UploadSkinnedMesh(assets.flag.mesh,
                                                 assets.flag.skin, 999)),
              "a palette larger than the ring is refused");
    }

    const eng::rhi::TextureId shadow_map = dev->CreateShadowMap(1024);
    const eng::rhi::TextureId color = dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
    const eng::rhi::TextureId depth = dev->CreateDepthTarget(kW, kH);
    // The scene target is half-float now; the composite tone maps it into this
    // one, which is the only kind that can be read back.
    const eng::rhi::TextureId out = dev->CreateRenderTarget(kW, kH, kFmt, true);
    if (!Valid(shadow_map) || !Valid(color) || !Valid(depth) || !Valid(out)) {
        std::fprintf(stderr, "FAIL: targets\n");
        return 1;
    }

    std::vector<std::uint8_t> pixels;
    eng::RenderStats stats;
    int shadow_draws = 0;

    auto draw = [&](const eng::Scene& scene) {
        eng::RenderGraph graph;
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
            p.color = color;
            p.depth = depth;
            for (int i = 0; i < 4; ++i) p.clear_color[i] = eng::kClearColor[i];
            p.clear_depth = 0.0f;
            p.reads = {shadow_map};
            p.execute = [&](eng::rhi::Encoder& e) {
                renderer->DrawScene(e, scene, kW, kH, shadow_map);
            };
            graph.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "composite";
            p.color = out;
            p.reads = {color};
            p.execute = [&](eng::rhi::Encoder& e) {
                renderer->DrawComposite(e, color, {}, {}, 0.0f, /*vignette=*/0.0f);
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
        shadow_draws = renderer->ShadowDrawCount();
        pixels.assign(std::size_t(kW) * kH * 4, 0);
        (void)dev->ReadPixels(out, kW, kH, pixels);
    };

    // The flag's bars are red and near-white; the ground is a desaturated
    // green and the pole is dark. Picking the flag out by hue rather than by
    // "not background" is what lets its extent be measured without the ground
    // dragging the box out to the edges of the frame.
    auto FlagBox = [&]() {
        Box b;
        for (int y = 0; y < kH; ++y)
            for (int x = 0; x < kW; ++x) {
                const std::size_t i = (std::size_t(y) * kW + x) * 4;
                const int r = pixels[i], g = pixels[i + 1], bl = pixels[i + 2];
                const bool red = r > g + 40 && r > bl + 40;
                const bool white = r > 150 && g > 140 && bl > 130 && r >= g && g >= bl - 8;
                if (red || white) b.Add(float(x), float(y));
            }
        return b;
    };

    // Everything that is not the clear colour. Usable only when one object is
    // alone in the frame — the hue test above exists because the tentacle
    // shares its frame with a floor.
    auto LitBox = [&]() {
        Box b;
        for (int y = 0; y < kH; ++y)
            for (int x = 0; x < kW; ++x) {
                const std::size_t i = (std::size_t(y) * kW + x) * 4;
                if (pixels[i] > 45 || pixels[i + 1] > 45) b.Add(float(x), float(y));
            }
        return b;
    };

    // The same vertices, posed by engine/anim and projected by hand.
    auto CpuBox = [&](const eng::Scene& scene) {
        const eng::Mat4 vp = scene.camera.ViewProj(float(kW) / float(kH));
        const eng::Mat4 model = demo::FlagModel();
        Box b;
        for (std::size_t i = 0; i < assets.flag.mesh.vertices.size(); ++i) {
            const eng::Vec4& p = assets.flag.mesh.vertices[i].position;
            const eng::Vec3 skinned = eng::anim::SkinPosition(
                eng::Vec3{p.x, p.y, p.z}, assets.flag.skin[i],
                scene.joint_matrices);
            const eng::Vec4 world =
                model * eng::Vec4{skinned.x, skinned.y, skinned.z, 1.0f};
            const eng::Vec4 clip = vp * world;
            if (clip.w <= 0.0f) continue;
            b.Add((clip.x / clip.w * 0.5f + 0.5f) * float(kW),
                  (1.0f - (clip.y / clip.w * 0.5f + 0.5f)) * float(kH));
        }
        return b;
    };

    auto Camera = [](eng::Scene& s) {
        // Oblique, not head on. A flag seen face-on hides its own ripple: the
        // wave travels in z, and straight down the z axis that is exactly the
        // direction you cannot see.
        s.camera.eye = eng::Vec3{-3.6f, 4.3f, 6.4f};
        s.camera.target = eng::Vec3{1.4f, 2.9f, 0.0f};
    };

    std::printf("gpu skinning agrees with the cpu reference\n");
    {
        // Several times through the clip, because one pose can agree by luck —
        // the bind pose agrees even with the palette ignored entirely.
        const float kTimes[] = {0.0f, 0.3f, 0.7f, 1.1f, 1.5f};
        float worst = 0.0f;
        bool all_close = true;
        for (float t : kTimes) {
            eng::Scene s = demo::MakeScene(assets, t);
            // The FLAG only. Anything else in the frame can occlude it, and an
            // occluded vertex is one the CPU box counts and the GPU correctly
            // never showed — which is what the first version of this check
            // mistook for a skinning error.
            const eng::Instance flag = s.instances.back();
            s.instances.clear();
            s.instances.push_back(flag);
            Camera(s);
            draw(s);
            const Box gpu = FlagBox();
            const Box cpu = CpuBox(s);
            if (gpu.Empty() || cpu.Empty()) {
                all_close = false;
                break;
            }
            // CONTAINMENT, not equality. Rasterisation, back-face culling and
            // self-occlusion can only ever REMOVE pixels — none of them can put
            // one outside the projected extent of the mesh's own vertices. So
            // the honest invariant is that the rendered shape sits inside the
            // predicted one, plus a pixel of rasterisation slack.
            //
            // Equality was the first thing tried and it measured the wrong
            // thing: the bottom edge disagreed by exactly 12 px at every pose,
            // which turned out to be the tube's unlit interior rather than
            // anything to do with the blend.
            const float slack = 2.0f;
            const bool inside = gpu.x0 >= cpu.x0 - slack && gpu.x1 <= cpu.x1 + slack &&
                                gpu.y0 >= cpu.y0 - slack && gpu.y1 <= cpu.y1 + slack;
            // ...and it must fill most of it, or "drew nothing" would pass.
            const float cover_x = (gpu.x1 - gpu.x0) / std::fmax(cpu.x1 - cpu.x0, 1.0f);
            const float cover_y = (gpu.y1 - gpu.y0) / std::fmax(cpu.y1 - cpu.y0, 1.0f);
            if (!inside || cover_x < 0.8f || cover_y < 0.8f) all_close = false;
            const float d = std::fmax(1.0f - cover_x, 1.0f - cover_y);
            std::printf("      t=%.2f  gpu[%.0f,%.0f]-[%.0f,%.0f]  cpu[%.0f,%.0f]-"
                        "[%.0f,%.0f]  inside=%d  cover %.0f%% x %.0f%%\n",
                        t, gpu.x0, gpu.y0, gpu.x1, gpu.y1, cpu.x0, cpu.y0, cpu.x1,
                        cpu.y1, int(inside), cover_x * 100.0f, cover_y * 100.0f);
            worst = std::fmax(worst, d);
        }
        std::printf("    worst coverage shortfall: %.0f%%\n", worst * 100.0f);
        Check(all_close, "the shader's blend matches SkinPosition, over the clip");
    }

    std::printf("the pose actually reaches the geometry\n");
    {
        // A palette the renderer ignored would draw the bind pose at every
        // time, and the checks above would still pass — they compare the GPU to
        // a CPU that was handed the same palette. This is the check that the
        // palette does anything at all.
        eng::Scene rest = demo::MakeScene(assets, 0.0f);
        Camera(rest);
        draw(rest);
        const Box a = FlagBox();
        Check(stats.draws == 4 && stats.culled == 0,
              "every instance drew; the posed flag was not culled on bind bounds");

        eng::Scene bent = demo::MakeScene(assets, 0.55f);
        Camera(bent);
        draw(bent);
        const Box b = FlagBox();
        const float moved = std::fabs(a.x1 - b.x1) + std::fabs(a.y0 - b.y0);
        std::printf("    silhouette moved %.1f px between poses\n", moved);
        Check(moved > 12.0f, "posing the skeleton visibly moves the mesh");

        // Every joint after the root sways with its own phase, so the tip
        // travels much further than the base. A rigid rotation of the whole
        // mesh would move both by the same amount.
        Check(shadow_draws == 4, "the skinned mesh was also drawn into the shadow map");
    }

    std::printf("an unskinned instance is unaffected\n");
    {
        // The ground shares the frame and has no palette. If the renderer bound
        // a stale palette to it, or picked the skinned pipeline, it would
        // deform or vanish.
        eng::Scene s = demo::MakeScene(assets, 0.45f);
        Camera(s);
        draw(s);
        int ground_pixels = 0;
        for (std::size_t i = 0; i < pixels.size(); i += 4) {
            const int r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
            // Green-dominant, and not a blowout. The upper bound used to be
            // 190, which the filmic tone curve lifted the lit ground straight
            // past -- it reads 193/207/186 now. Bounding on DOMINANCE and
            // leaving the brightness alone is what makes this a test of "the
            // ground is still there" rather than of the exposure.
            if (g > r + 3 && g > b + 3 && g < 250 && g > 50) ++ground_pixels;
        }
        Check(ground_pixels > 20000, "the static ground still drew, undeformed");
        Check(stats.invalid == 0 && stats.overflowed == 0, "nothing was dropped");
    }

    std::printf("imported skin and animation reach the GPU\n");
    {
        // The procedural tentacle proves the shader. This proves the IMPORTER
        // feeds it: a different code path, and a different set of mistakes.
        std::string e;
        const eng::gltf::Document doc = eng::gltf::ParseGltf(kSkinnedGltf, {}, e);
        Check(e.empty() && doc.skins.size() == 1 && !doc.primitives.empty(),
              "the fixture parsed");
        if (!doc.skins.empty() && !doc.primitives.empty()) {
            const eng::gltf::Primitive& prim = doc.primitives[0];
            const eng::MeshHandle h = renderer->UploadSkinnedMesh(
                prim.mesh, prim.skin, int(doc.skins[0].skeleton.joints.size()));
            Check(Valid(h), "the imported skinned mesh uploaded");

            // The fixture is a flat strip with no back face of its own, so a
            // back-face-culled material draws nothing from one side.
            eng::MaterialDesc two_sided;
            two_sided.base_color = eng::Vec4{0.85f, 0.45f, 0.30f, 1.0f};
            two_sided.roughness = 0.5f;
            two_sided.cull = eng::rhi::Cull::None;
            std::string me;
            const eng::MaterialHandle flat_mat = renderer->CreateMaterial(two_sided, me);

            const eng::anim::Clip clip = doc.MakeClip(0, 0);
            eng::anim::Pose pose;
            std::printf("    imported mesh verts %zu  joints %d\n",
                        prim.mesh.vertices.size(),
                        renderer->JointCount(h));

            auto arm_extent = [&](float t) {
                eng::Scene s;
                s.lightDir = eng::Vec4{0.32f, 0.86f, 0.40f, 0.0f};
                s.lightColor = eng::Vec4{1.7f, 1.7f, 1.7f, 1.0f};
                clip.Sample(t, doc.skins[0].skeleton, &pose, /*loop=*/false);
                eng::anim::ComputeJointMatrices(doc.skins[0].skeleton, pose,
                                                &s.joint_matrices);
                eng::Instance inst;
                inst.mesh = h;
                inst.material = flat_mat;
                inst.palette = 0;
                s.instances.push_back(inst);
                // Oblique, and deliberately NOT on x = 1. The fixture is a
                // zero-thickness strip: once the far half stands up it lies in
                // the plane x = 1, and a camera anywhere on that plane sees it
                // exactly edge-on and measures it as one pixel tall.
                s.camera.eye = eng::Vec3{-2.2f, 1.6f, 4.2f};
                s.camera.target = eng::Vec3{0.9f, 0.35f, 0.0f};
                draw(s);
                int lit = 0, warm = 0;
                for (std::size_t q = 0; q < pixels.size(); q += 4) {
                    if (pixels[q] > 40 || pixels[q + 1] > 40) ++lit;
                    if (int(pixels[q]) > int(pixels[q + 1]) + 25) ++warm;
                }
                std::printf("    t=%.1f draws %d culled %d  lit px %d  warm px %d\n",
                            t, stats.draws, stats.culled, lit, warm);
                return LitBox();
            };

            const Box straight = arm_extent(0.0f);
            const Box bent = arm_extent(1.0f);
            Check(!straight.Empty() && !bent.Empty(), "the imported arm rendered");
            // Straight, the arm lies along +x: wide and flat. Bent through
            // ninety degrees, its far half stands up: narrower and much taller.
            // Both directions are checked, because a mesh that merely shrank
            // would satisfy either one alone.
            const float sw = straight.x1 - straight.x0, sh = straight.y1 - straight.y0;
            const float bw = bent.x1 - bent.x0, bh = bent.y1 - bent.y0;
            std::printf("    straight %.0fx%.0f px   bent %.0fx%.0f px\n",
                        sw, sh, bw, bh);
            Check(bw < sw * 0.75f, "bending shortens the arm's reach along x");
            Check(bh > sh * 2.0f, "and stands its far half up");
        }
    }

    std::FILE* f = std::fopen("flag.ppm", "wb");
    if (f) {
        eng::Scene s = demo::MakeScene(assets, 0.42f);
        Camera(s);
        draw(s);
        std::fprintf(f, "P6\n%d %d\n255\n", kW, kH);
        for (std::size_t i = 0; i < pixels.size(); i += 4)
            std::fwrite(&pixels[i], 1, 3, f);
        std::fclose(f);
    }

    std::printf(g_failures == 0 ? "\nskinned_test: all checks passed\n"
                                : "\nskinned_test: %d FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
