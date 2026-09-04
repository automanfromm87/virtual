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

#include "tests/common/skinned_scene.h"
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
    // LINE buffered. A test that crashes with a full buffer prints nothing at
    // all, so the one piece of information that would locate the crash -- how
    // far it got -- is exactly what gets lost.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
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
    int shadow_culled = 0;

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
        shadow_culled = renderer->ShadowCulledCount();
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
        // Once per caster PER CASCADE: each cascade is its own pass over the
        // casters into its own tile of the map. Spelling out the product rather
        // than freezing the total is the difference between a test that
        // survives turning cascades on and one that just has to be renumbered.
        //
        // DRAWN PLUS CULLED, not drawn alone. DrawShadow frustum-tests each
        // caster against each cascade's ortho box, so a caster outside one
        // cascade is a cull rather than a draw -- the finial above the pole is
        // outside the nearest slice, and submitting it there wrote no texel
        // (verified: disabling the cull changes this count and not one
        // measured pixel). The sum is the invariant the product was reaching
        // for: every caster is CONSIDERED once per cascade, and the skinned
        // flag is drawn rather than dropped on its bind bounds.
        const int cascades = rest.shadowCascades;
        const int expect = 4 * cascades;
        std::printf("    %d shadow draws + %d culled = 4 casters x %d cascades\n",
                    shadow_draws, shadow_culled, cascades);
        Check(shadow_draws + shadow_culled == expect && shadow_draws >= cascades,
              "the skinned mesh was also drawn into every cascade");
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

    // --- compute skinning, written back to a buffer ---------------------------
    //
    // The vertex shader blends a skinned vertex and hands it straight to the
    // rasteriser, so the posed triangles exist nowhere a second consumer can
    // reach. This poses them into a buffer instead, which is what ray tracing
    // needs (an acceleration structure is built from a buffer, and one built
    // from the bind pose casts a standing character's shadow while it walks)
    // and what CPU-side collision against a posed mesh needs.
    //
    // The check is EXACT, vertex by vertex, against anim::SkinPosition. The
    // rasterised test above can only compare bounding boxes with slack, because
    // culling and occlusion legitimately remove pixels. A buffer has no such
    // excuse -- every float is either the reference value or a bug.
    {
        std::printf("compute skinning\n");
        eng::Scene s = demo::MakeScene(assets, 0.37f);
        // One instance, so the mapping from instance index to posed buffer is
        // unambiguous when a check fails.
        eng::Instance only;
        for (const eng::Instance& i : s.instances)
            if (i.mesh.v == assets.mesh.v) only = i;
        s.instances.clear();
        s.instances.push_back(only);

        dev->BeginFrame();
        int posed = 0;
        {
            eng::rhi::ComputeEncoder ce = dev->BeginCompute();
            posed = renderer->SkinToBuffers(ce, s);
            dev->EndCompute();
        }
        std::string werr;
        Check(dev->CommitAndWait(werr), "the compute pass submits");
        std::printf("    posed %d skinned instance(s)\n", posed);
        Check(posed == 1, "the skinned instance was posed");

        const eng::rhi::BufferId out = renderer->PosedVertices(0);
        Check(Valid(out), "and has a posed vertex buffer");
        const auto* gpu = static_cast<const VertexIn*>(dev->MapBuffer(out));
        Check(gpu != nullptr, "which can be read back");

        if (gpu) {
            // The same blend, on the CPU, from the same palette the compute
            // pass was given.
            const std::vector<eng::Mat4>& palette = s.joint_matrices;
            float worst_pos = 0.0f, worst_nrm = 0.0f;
            std::size_t n = assets.flag.mesh.vertices.size();
            for (std::size_t i = 0; i < n; ++i) {
                const eng::Vec4& p = assets.flag.mesh.vertices[i].position;
                const eng::Vec4& nv = assets.flag.mesh.vertices[i].normal;
                const eng::Vec3 want_p = eng::anim::SkinPosition(
                    eng::Vec3{p.x, p.y, p.z}, assets.flag.skin[i], palette);
                const eng::Vec3 want_n = eng::anim::SkinNormal(
                    eng::Vec3{nv.x, nv.y, nv.z}, assets.flag.skin[i], palette);
                const eng::Vec3 got_p{gpu[i].position.x, gpu[i].position.y,
                                      gpu[i].position.z};
                const eng::Vec3 got_n{gpu[i].normal.x, gpu[i].normal.y,
                                      gpu[i].normal.z};
                worst_pos = std::fmax(worst_pos, Length(got_p - want_p));
                // SkinNormal renormalises and the kernel does not, so compare
                // directions rather than vectors -- the length is the vertex
                // shader's business and it renormalises per fragment anyway.
                const float len = Length(got_n);
                if (len > 1e-6f)
                    worst_nrm = std::fmax(worst_nrm,
                                          Length(got_n * (1.0f / len) - want_n));
            }
            std::printf("    %zu vertices: worst position error %.7f, worst "
                        "normal error %.7f\n", n, worst_pos, worst_nrm);
            Check(worst_pos < 1e-5f, "the compute blend matches SkinPosition exactly");
            Check(worst_nrm < 1e-5f, "and the normals match SkinNormal");

            // It must actually have MOVED. Comparing against the reference
            // would pass on a kernel that copied the bind pose, if the pose
            // happened to be the rest pose -- and it is not, but nothing above
            // says so.
            float moved = 0.0f;
            for (std::size_t i = 0; i < n; ++i) {
                const eng::Vec4& p = assets.flag.mesh.vertices[i].position;
                moved = std::fmax(moved, Length(eng::Vec3{gpu[i].position.x,
                                                          gpu[i].position.y,
                                                          gpu[i].position.z} -
                                                eng::Vec3{p.x, p.y, p.z}));
            }
            std::printf("    the furthest vertex moved %.4f from the bind pose\n",
                        moved);
            Check(moved > 0.05f, "the pose is not the bind pose");

            // Colour and uv ride along unchanged, so the output is a drop-in
            // replacement for the input buffer rather than a parallel array.
            bool carried = true;
            for (std::size_t i = 0; i < n; ++i) {
                const VertexIn& src = assets.flag.mesh.vertices[i];
                if (gpu[i].uv.x != src.uv.x || gpu[i].uv.y != src.uv.y ||
                    gpu[i].color.x != src.color.x)
                    carried = false;
            }
            Check(carried, "uv and colour survive the pose unchanged");
        }

        if (!renderer->SkinError().empty())
            std::fprintf(stderr, "    skinning pipeline: %s\n",
                         renderer->SkinError().c_str());

        // A DIFFERENT pose must give a different buffer. Without this a kernel
        // that ignored the palette entirely would pass everything above on the
        // one pose it was checked at.
        if (gpu) {
            eng::Scene s2 = demo::MakeScene(assets, 0.82f);
            eng::Instance one;
            for (const eng::Instance& i : s2.instances)
                if (i.mesh.v == assets.mesh.v) one = i;
            s2.instances.clear();
            s2.instances.push_back(one);
            std::vector<VertexIn> first(
                gpu, gpu + assets.flag.mesh.vertices.size());

            dev->BeginFrame();
            {
                eng::rhi::ComputeEncoder ce = dev->BeginCompute();
                (void)renderer->SkinToBuffers(ce, s2);
                dev->EndCompute();
            }
            std::string e2;
            Check(dev->CommitAndWait(e2), "the second pose submits");
            const auto* g2 = static_cast<const VertexIn*>(
                dev->MapBuffer(renderer->PosedVertices(0)));
            float apart = 0.0f;
            if (g2)
                for (std::size_t i = 0; i < first.size(); ++i)
                    apart = std::fmax(
                        apart, Length(eng::Vec3{g2[i].position.x, g2[i].position.y,
                                                g2[i].position.z} -
                                      eng::Vec3{first[i].position.x,
                                                first[i].position.y,
                                                first[i].position.z}));
            std::printf("    a second pose moves the furthest vertex %.4f "
                        "from the first\n", apart);
            Check(apart > 0.02f, "a different pose produces a different buffer");
        }
    }

    // --- a skinned mesh casting a ray-traced shadow ---------------------------
    //
    // This is the whole reason compute skinning exists. An acceleration
    // structure is built from a BUFFER, and until now the only buffer holding
    // this mesh was the bind pose -- so a waving flag cast the shadow of a flat
    // one. The check is that the shadow MOVES when the pose does, which the
    // bind-pose version cannot do however good it looks in a still.
    {
        std::printf("ray-traced shadows from a posed mesh\n");
        if (!renderer->RaytracingAvailable()) {
            std::printf("    (no hardware ray tracing on this device)\n");
        } else {
            const eng::rhi::TextureId gb0 =
                dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
            const eng::rhi::TextureId gb1 =
                dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
            const eng::rhi::TextureId gdepth = dev->CreateDepthTarget(kW, kH, true);
            const eng::rhi::TextureId rmask =
                dev->CreateRenderTarget(kW, kH, eng::rhi::Format::RGBA8Unorm, true);
            Check(Valid(gb0) && Valid(rmask), "the ray tracing targets exist");

            // The shadow mask for one pose, as a picture.
            const auto mask_for = [&](float t, std::vector<std::uint8_t>* out,
                                      int* blas_before) {
                eng::Scene s = demo::MakeScene(assets, t);
                // Straight down, so the flag's shadow lands on the ground
                // directly beneath it and moving the flag moves the shadow.
                s.lightDir = eng::Vec4{0.0f, 1.0f, 0.0f, 0.0f};
                s.camera.eye = eng::Vec3{0.0f, 6.5f, 0.01f};
                s.camera.target = eng::Vec3{0.0f, 0.0f, 0.0f};

                dev->BeginFrame();
                {
                    eng::rhi::ComputeEncoder ce = dev->BeginCompute();
                    (void)renderer->SkinToBuffers(ce, s);
                    dev->EndCompute();
                }
                *blas_before = renderer->BlasBuilds();
                std::string e;
                if (!renderer->BuildSceneAccel(s, e))
                    std::fprintf(stderr, "    accel: %s\n", e.c_str());

                eng::RenderGraph g;
                {
                    eng::RenderGraph::Pass p;
                    p.name = "gbuffer";
                    p.color = gb0;
                    p.extra_colors = {gb1};
                    p.depth = gdepth;
                    p.clear_depth = 0.0f;
                    p.keep_depth = true;
                    p.execute = [&](eng::rhi::Encoder& e2) {
                        renderer->DrawGBuffer(e2, s, kW, kH);
                    };
                    g.AddPass(std::move(p));
                }
                {
                    eng::RenderGraph::Pass p;
                    p.name = "ray shadows";
                    p.color = rmask;
                    p.reads = {gdepth, gb1};
                    p.clear_color[0] = 1.0f; p.clear_color[1] = 1.0f;
                    p.clear_color[2] = 1.0f; p.clear_color[3] = 1.0f;
                    p.execute = [&](eng::rhi::Encoder& e2) {
                        renderer->DrawRayShadows(e2, s, kW, kH, gdepth, gb1);
                    };
                    g.AddPass(std::move(p));
                }
                std::string ce2;
                if (!g.Compile(ce2)) std::fprintf(stderr, "    graph: %s\n", ce2.c_str());
                g.Execute(*dev);
                std::string we;
                if (!dev->CommitAndWait(we)) std::fprintf(stderr, "    %s\n", we.c_str());
                out->assign(std::size_t(kW) * kH * 4, 0);
                (void)dev->ReadPixels(rmask, kW, kH, *out);
            };

            std::vector<std::uint8_t> a, b;
            int blas_a = 0, blas_b = 0;
            mask_for(0.10f, &a, &blas_a);
            mask_for(0.60f, &b, &blas_b);

            const auto shadow_pixels = [](const std::vector<std::uint8_t>& m) {
                int n = 0;
                for (std::size_t i = 0; i + 3 < m.size(); i += 4)
                    if (m[i] < 64) ++n;
                return n;
            };
            int moved = 0;
            for (std::size_t i = 0; i + 3 < a.size(); i += 4)
                if ((a[i] < 64) != (b[i] < 64)) ++moved;

            const int sa = shadow_pixels(a), sb = shadow_pixels(b);
            std::printf("    pose 0.10 shadows %d px, pose 0.60 shadows %d px, "
                        "%d px changed between them\n", sa, sb, moved);
            Check(sa > 2000 && sb > 2000, "the flag casts a shadow at all");
            // The number that matters. A structure built from the bind pose
            // gives the SAME shadow at every pose, so this is zero -- and every
            // other check here passes anyway.
            Check(moved > 800, "and the shadow moves when the pose does");

            // Confirmed by the build count: a posed mesh cannot share its
            // structure with anything, so every frame rebuilds it.
            std::printf("    bottom-level builds: %d after the first pose, %d "
                        "after the second\n", blas_b, renderer->BlasBuilds());
            Check(renderer->BlasBuilds() > blas_b,
                  "a re-posed mesh rebuilds its acceleration structure");
        }
    }

    // --- the past-the-end threads, and the zero-influence vertex --------------
    //
    // Two branches in the kernel that the flag cannot exercise, each with a
    // failure that looks like nothing at all.
    //
    // The dispatch rounds the thread count up to a whole threadgroup, so a mesh
    // whose vertex count is not a multiple of the group runs threads past its
    // end. The output buffers are POOLED and reused, so those threads do not
    // write into empty space -- they write over whatever the previous, larger
    // mesh left there. That is the observation: pose something big, then
    // something small into the same reused buffer, and the big one's tail must
    // survive.
    {
        std::printf("kernel edge cases\n");

        // A small skinned mesh: 100 vertices, which is not a multiple of the
        // 64-wide threadgroup, so 28 threads run past the end.
        constexpr int kSmall = 100;
        eng::Mesh sm;
        std::vector<eng::anim::SkinVertex> ssk;
        for (int i = 0; i < kSmall; ++i) {
            VertexIn v{};
            // NOT at the origin, and that is the point of the offset. A zero
            // blend matrix collapses a vertex to (0,0,0), so a fixture authored
            // there cannot tell "left alone" from "collapsed" -- the two give
            // the same answer and the check passes either way.
            v.position = eng::Vec4{1.0f + float(i) * 0.01f, 0.5f, -0.25f, 1.0f};
            v.normal = eng::Vec4{0, 1, 0, 0};
            v.color = eng::Vec4{1, 1, 1, 1};
            sm.vertices.push_back(v);
            eng::anim::SkinVertex sv{};
            if (i == 0) {
                // ALL-ZERO weights. Nothing influences this vertex, and the
                // kernel must leave it where the modeller put it -- a zero
                // blend matrix collapses it onto the origin instead, which on a
                // character is one vertex of the mesh stretched to the world
                // origin and very easy to blame on the exporter.
                for (int c = 0; c < 4; ++c) { sv.joints[c] = 0; sv.weights[c] = 0.0f; }
            } else {
                sv.joints[0] = std::uint32_t(i % 2);
                sv.weights[0] = 1.0f;
            }
            ssk.push_back(sv);
        }
        for (int i = 0; i + 2 < kSmall; ++i) {
            sm.indices.push_back(std::uint32_t(i));
            sm.indices.push_back(std::uint32_t(i + 1));
            sm.indices.push_back(std::uint32_t(i + 2));
        }
        const eng::MeshHandle small = renderer->UploadSkinnedMesh(sm, ssk, 2);
        Check(Valid(small), "a small skinned mesh uploaded");

        // Pose the big flag first, so the pooled buffer is sized for it and
        // filled with its data.
        eng::Scene big = demo::MakeScene(assets, 0.45f);
        eng::Instance flag_only;
        for (const eng::Instance& i : big.instances)
            if (i.mesh.v == assets.mesh.v) flag_only = i;
        big.instances.clear();
        big.instances.push_back(flag_only);
        dev->BeginFrame();
        {
            eng::rhi::ComputeEncoder ce = dev->BeginCompute();
            (void)renderer->SkinToBuffers(ce, big);
            dev->EndCompute();
        }
        std::string e1;
        Check(dev->CommitAndWait(e1), "the big pose submits");
        const eng::rhi::BufferId pooled = renderer->PosedVertices(0);
        const auto* big_out = static_cast<const VertexIn*>(dev->MapBuffer(pooled));
        std::vector<VertexIn> before;
        if (big_out)
            before.assign(big_out, big_out + assets.flag.mesh.vertices.size());

        // Now the small mesh, into the same pooled buffer.
        eng::Scene tiny;
        tiny.joint_matrices = {eng::Mat4::Translation(eng::Vec3{0.0f, 2.0f, 0.0f}),
                               eng::Mat4::Translation(eng::Vec3{0.0f, -2.0f, 0.0f})};
        {
            eng::Instance i;
            i.mesh = small;
            i.material = assets.flag_mat;
            i.palette = 0;
            tiny.instances.push_back(i);
        }
        dev->BeginFrame();
        {
            eng::rhi::ComputeEncoder ce = dev->BeginCompute();
            (void)renderer->SkinToBuffers(ce, tiny);
            dev->EndCompute();
        }
        std::string e2;
        Check(dev->CommitAndWait(e2), "the small pose submits");
        const eng::rhi::BufferId reused = renderer->PosedVertices(0);
        Check(reused.v == pooled.v, "the small mesh reused the big mesh's buffer");

        const auto* out = static_cast<const VertexIn*>(dev->MapBuffer(reused));
        if (out && !before.empty()) {
            // The zero-influence vertex stayed put.
            std::printf("    the zero-influence vertex is at (%.3f %.3f %.3f), "
                        "authored at (1.000 0.500 -0.250)\n", out[0].position.x,
                        out[0].position.y, out[0].position.z);
            Check(std::fabs(out[0].position.x - 1.0f) < 1e-5f &&
                      std::fabs(out[0].position.y - 0.5f) < 1e-5f &&
                      std::fabs(out[0].position.z + 0.25f) < 1e-5f,
                  "a vertex with no influences stays where it was authored");
            // ...and the vertices that DO have influences moved, or "nothing
            // moved at all" would pass the check above. The two joints
            // translate opposite ways, so odd and even vertices separate --
            // which also rules out a kernel that applies joint 0 to everything.
            std::printf("    vertex 1 (joint 1) y=%.3f, vertex 2 (joint 0) "
                        "y=%.3f\n", out[1].position.y, out[2].position.y);
            Check(std::fabs(out[1].position.y + 1.5f) < 1e-4f,
                  "a vertex weighted to joint 1 moved down with it");
            Check(std::fabs(out[2].position.y - 2.5f) < 1e-4f,
                  "and one weighted to joint 0 moved up with it");

            // The tail past the small mesh, which the rounded-up dispatch runs
            // threads over. 100 vertices in groups of 64 is 128 threads, so
            // indices 100..127 are the ones at risk.
            int clobbered = 0;
            for (std::size_t i = std::size_t(kSmall); i < before.size(); ++i)
                if (out[i].position.x != before[i].position.x ||
                    out[i].position.y != before[i].position.y)
                    ++clobbered;
            std::printf("    %zu vertices of the previous mesh survive past the "
                        "small one; %d were overwritten\n",
                        before.size() - kSmall, clobbered);
            Check(clobbered == 0,
                  "threads past the end of the mesh write nothing");
        }
    }

    std::printf(g_failures == 0 ? "\nskinned_test: all checks passed\n"
                                : "\nskinned_test: %d FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
