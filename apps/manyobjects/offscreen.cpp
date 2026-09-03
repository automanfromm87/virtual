// GPU-DRIVEN DRAWING, measured against the ordinary path on the same scene.
//
// Everything else in this engine is tested on scenes of a dozen objects, where
// the cost of submitting a draw is invisible. This one exists because the claim
// being made -- that culling on the GPU and drawing indirectly is cheaper --
// is only true at a scale nothing else here reaches, and an untested claim
// about performance is worth less than no claim.
//
// The two paths must also produce the SAME PICTURE. A faster renderer that
// draws something slightly different is not a faster renderer.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/geometry/mesh.h"
#include "engine/render/renderer.h"
#include "engine/render/rendergraph.h"
#include "engine/rhi/rhi.h"
#include "engine/scene/scene.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

constexpr int kW = 800, kH = 500;
// Enough that draw submission dominates, and not so many that the test is slow.
constexpr int kObjects = 6000;

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("gpu-driven drawing, %d objects\n", kObjects);

    std::string error;
    auto dev = eng::rhi::Device::Create(error);
    if (!dev) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
    const auto kFmt = eng::rhi::Format::RGBA8Unorm;
    auto r = eng::Renderer::Create(*dev, kFmt, error, 1);
    if (!r) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    // Three meshes, so there are three batches and the grouping is doing
    // something. A single mesh would collapse to one draw and prove less.
    const eng::MeshHandle meshes[3] = {
        r->UploadMesh(eng::MakeUVSphere(0.35f, 12, 16, eng::Vec4{1, 1, 1, 1},
                                        eng::Vec4{1, 1, 1, 1})),
        r->UploadMesh(eng::MakeBox(eng::Vec3{0.3f, 0.3f, 0.3f},
                                   eng::Vec4{1, 1, 1, 1})),
        r->UploadMesh(eng::MakeBox(eng::Vec3{0.2f, 0.6f, 0.2f},
                                   eng::Vec4{1, 1, 1, 1})),
    };
    eng::MaterialDesc md;
    md.shading = eng::Shading::Lit;
    md.roughness = 0.45f;
    const eng::MaterialHandle mat = r->CreateMaterial(md, error);
    Check(Valid(meshes[0]) && Valid(meshes[2]) && Valid(mat),
          "the meshes and material were created");
    if (!error.empty()) std::fprintf(stderr, "  %s\n", error.c_str());

    eng::Scene scene;
    scene.lightDir = eng::Vec4{-0.4f, 0.85f, 0.35f, 0.0f};
    scene.lightColor = eng::Vec4{1.1f, 1.05f, 0.95f, 1.0f};
    scene.ambientSky = eng::Vec3{0.10f, 0.13f, 0.18f};
    scene.ambientGround = eng::Vec3{0.05f, 0.04f, 0.03f};
    scene.camera.eye = eng::Vec3{0.0f, 9.0f, 26.0f};
    scene.camera.target = eng::Vec3{0.0f, 0.0f, 0.0f};

    // A grid that extends well BEHIND the camera as well as in front, so
    // culling has real work to do and the test can check it did it.
    for (int i = 0; i < kObjects; ++i) {
        eng::Instance inst;
        inst.mesh = meshes[i % 3];
        inst.material = mat;
        const int gx = i % 60, gz = i / 60;
        inst.model = eng::Mat4::Translation(
            eng::Vec3{float(gx) * 1.2f - 36.0f,
                      0.4f + 0.3f * std::sin(float(i) * 0.7f),
                      float(gz) * 1.2f - 60.0f});
        // SCALED, and not all by the same amount. With every instance at unit
        // scale the cull's radius scaling is unreachable -- multiplying by one
        // and not multiplying at all are the same code -- and a bug there only
        // shows on the scene that has scaled objects in it, which is every real
        // one.
        if (i % 7 == 0) inst.model = inst.model * eng::Mat4::Scale(2.6f);
        // NON-UNIFORM scale, built by hand because Mat4::Scale is uniform only
        // by design. glTF nodes carry one routinely, and it is the only thing
        // that distinguishes "the largest axis" from "the x axis" in the cull's
        // radius -- with uniform scale the two are the same number, and a cull
        // that reads one axis passes every test a uniform scene can offer.
        // Tall and thin, so reading x alone under-estimates the radius sixfold
        // and clips these at the edge of the screen.
        if (i % 11 == 0)
            inst.model = inst.model * eng::Mat4{{{0.5f, 0, 0, 0},
                                                 {0, 3.0f, 0, 0},
                                                 {0, 0, 0.5f, 0},
                                                 {0, 0, 0, 1}}};
        const float t = float(i) / float(kObjects);
        inst.tint = eng::Vec4{0.35f + 0.6f * t, 0.5f, 0.85f - 0.5f * t, 1.0f};
        scene.instances.push_back(inst);
    }

    const eng::rhi::TextureId color =
        dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
    const eng::rhi::TextureId depth = dev->CreateDepthTarget(kW, kH);
    const eng::rhi::TextureId out = dev->CreateRenderTarget(kW, kH, kFmt, true);
    Check(Valid(color) && Valid(depth) && Valid(out), "the targets were created");

    std::vector<std::uint8_t> cpu_px, gpu_px;
    eng::RenderStats cpu_stats, gpu_stats;

    // How long a frame takes, measured over several so one warm-up frame does
    // not decide it. Wall clock around a CommitAndWait, which is the honest
    // number: it includes everything the CPU does AND waits for.
    const auto time_frames = [&](int frames, const auto& body) {
        body();  // warm up: first-use pipeline compiles and buffer allocation
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < frames; ++i) body();
        const auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count() /
               double(frames);
    };

    // --- the ordinary path ---------------------------------------------------
    const auto draw_cpu = [&] {
        eng::RenderGraph g;
        eng::RenderGraph::Pass p;
        p.name = "scene";
        p.color = color;
        p.depth = depth;
        p.clear_depth = 0.0f;
        p.execute = [&](eng::rhi::Encoder& e) { r->DrawScene(e, scene, kW, kH); };
        g.AddPass(std::move(p));
        eng::RenderGraph::Pass c;
        c.name = "composite";
        c.color = out;
        c.reads = {color};
        c.execute = [&](eng::rhi::Encoder& e) {
            r->DrawComposite(e, color, {}, {}, 0.0f, 0.0f);
        };
        g.AddPass(std::move(c));
        std::string e;
        if (!g.Compile(e)) std::fprintf(stderr, "  graph: %s\n", e.c_str());
        dev->BeginFrame();
        g.Execute(*dev);
        std::string w;
        if (!dev->CommitAndWait(w)) std::fprintf(stderr, "  %s\n", w.c_str());
        cpu_stats = r->LastStats();
    };
    const double cpu_ms = time_frames(8, draw_cpu);
    cpu_px.assign(std::size_t(kW) * kH * 4, 0);
    Check(dev->ReadPixels(out, kW, kH, cpu_px), "the cpu-driven frame reads back");

    // --- GPU-driven ----------------------------------------------------------
    const auto draw_gpu = [&] {
        dev->BeginFrame();
        {
            eng::rhi::ComputeEncoder ce = dev->BeginCompute();
            (void)r->CullScene(ce, scene, kW, kH);
            dev->EndCompute();
        }
        eng::RenderGraph g;
        eng::RenderGraph::Pass p;
        p.name = "scene";
        p.color = color;
        p.depth = depth;
        p.clear_depth = 0.0f;
        p.execute = [&](eng::rhi::Encoder& e) {
            r->DrawSceneIndirect(e, scene, kW, kH);
        };
        g.AddPass(std::move(p));
        eng::RenderGraph::Pass c;
        c.name = "composite";
        c.color = out;
        c.reads = {color};
        c.execute = [&](eng::rhi::Encoder& e) {
            r->DrawComposite(e, color, {}, {}, 0.0f, 0.0f);
        };
        g.AddPass(std::move(c));
        std::string e;
        if (!g.Compile(e)) std::fprintf(stderr, "  graph: %s\n", e.c_str());
        g.Execute(*dev);
        std::string w;
        if (!dev->CommitAndWait(w)) std::fprintf(stderr, "  %s\n", w.c_str());
        gpu_stats = r->LastStats();
    };
    const double gpu_ms = time_frames(8, draw_gpu);
    gpu_px.assign(std::size_t(kW) * kH * 4, 0);
    Check(dev->ReadPixels(out, kW, kH, gpu_px), "the gpu-driven frame reads back");

    std::printf("    cpu-driven: %d draws, %d culled on the cpu, %.2f ms/frame\n",
                cpu_stats.draws, cpu_stats.culled, cpu_ms);
    std::printf("    gpu-driven: %d draws over %d batches, %d instances offered, "
                "%.2f ms/frame\n", gpu_stats.draws, r->LastBatchCount(),
                r->LastInstanceCount(), gpu_ms);
    std::printf("    %.2fx\n", cpu_ms / std::fmax(gpu_ms, 1e-6));

    Check(cpu_stats.draws > 1000, "the ordinary path really does submit one draw each");
    Check(gpu_stats.draws == 3, "the gpu-driven path submits one draw per mesh");
    Check(r->LastInstanceCount() == kObjects, "and offers every instance to the cull");
    // The CPU path culls too, so this is not a claim that culling is new -- it
    // is a check that the scene genuinely has offscreen objects, without which
    // the cull below would be measuring nothing.
    Check(cpu_stats.culled > 1000, "a good fraction of the scene is offscreen");

    // WHAT THE CULL KEPT. Not observable from the picture: a cull that keeps
    // everything draws exactly the same frame, because the instances it should
    // have dropped are off screen and rasterise nothing. Only the count says
    // whether any work was saved, and it costs a readback to ask -- which is
    // why this is a test doing it and not the renderer every frame.
    const int kept = r->VisibleAfterCull();
    const int cpu_kept = cpu_stats.draws;
    std::printf("    the gpu cull kept %d of %d; the cpu frustum kept %d\n", kept,
                kObjects, cpu_kept);
    Check(kept > 0 && kept < kObjects, "the gpu cull dropped some and kept some");
    // EXACTLY the same number, not approximately. The two are the same six
    // planes tested against the same bounding spheres, scaled by the same rule
    // -- the largest of the model matrix's three column lengths, which is
    // MaxScale on the CPU and the max of three dots in the kernel. There is no
    // source of legitimate disagreement, so a tolerance here would only hide
    // one. A tolerance of 5% was the first version and it absorbed a kernel
    // that scaled the radius by the x axis alone: only a handful of instances
    // straddle the frustum edge at any moment, and a handful fits inside 5%.
    //
    // The CPU path can also drop instances for reasons the GPU one has no
    // equivalent of, so those are checked to be absent rather than tolerated.
    Check(cpu_stats.invalid == 0 && cpu_stats.overflowed == 0 &&
              cpu_stats.incompatible == 0,
          "the cpu path dropped nothing except by the frustum");
    Check(kept == cpu_kept,
          "and the two frustums keep exactly the same instances");

    // --- and the same picture ------------------------------------------------
    //
    // The two paths share a fragment shader and differ only in where the model
    // matrix comes from, so they should agree closely. They turn out to agree
    // EXACTLY -- mean 0, worst 0, identical total brightness.
    //
    // That was not a given, and the bounds below are loose on purpose rather
    // than tightened to zero. The GPU cull emits survivors in whatever order
    // threads happened to finish, so anywhere two opaque surfaces sit at the
    // same depth the winner is order-dependent and the two paths could differ.
    // This scene has no co-planar pairs, so nothing exercises it; a scene with
    // decals or z-fighting would, and pinning this at zero would turn a
    // legitimate difference into a failure.
    long long sum = 0, cpu_lit = 0, gpu_lit = 0;
    int worst = 0, over_8 = 0;
    for (std::size_t i = 0; i + 3 < cpu_px.size(); i += 4) {
        int here = 0;
        for (int c = 0; c < 3; ++c)
            here = std::max(here, std::abs(int(cpu_px[i + std::size_t(c)]) -
                                           int(gpu_px[i + std::size_t(c)])));
        sum += here;
        worst = std::max(worst, here);
        if (here > 8) ++over_8;
        cpu_lit += cpu_px[i] + cpu_px[i + 1] + cpu_px[i + 2];
        gpu_lit += gpu_px[i] + gpu_px[i + 1] + gpu_px[i + 2];
    }
    const double mean = double(sum) / (double(kW) * kH);
    std::printf("    cpu vs gpu image: mean %.4f, worst %d, %d pixels over 8\n",
                mean, worst, over_8);
    std::printf("    total brightness: cpu %lld, gpu %lld (%.3f%%)\n", cpu_lit,
                gpu_lit, 100.0 * double(gpu_lit) / double(cpu_lit));
    Check(mean < 0.05, "the two paths draw the same picture");
    Check(over_8 < 200, "and disagree on almost no pixels at all");
    // Without this the comparison would pass on two black frames.
    Check(cpu_lit > 3000000 && gpu_lit > 3000000, "both frames drew something");

    // --- the cull actually culls ---------------------------------------------
    //
    // Point the camera away from everything. The batches are still submitted --
    // three indirect draws go out either way, because the CPU does not know
    // what survived -- but the instance count in them must collapse, and the
    // frame must come out empty.
    {
        eng::Scene away = scene;
        away.camera.target = eng::Vec3{0.0f, 400.0f, 0.0f};
        away.camera.eye = eng::Vec3{0.0f, 9.0f, 26.0f};
        const auto saved = scene;
        scene = away;
        draw_gpu();
        scene = saved;
        std::vector<std::uint8_t> empty_px(std::size_t(kW) * kH * 4, 0);
        Check(dev->ReadPixels(out, kW, kH, empty_px), "the empty frame reads back");
        long long lit = 0;
        for (std::size_t i = 0; i + 3 < empty_px.size(); i += 4)
            lit += empty_px[i] + empty_px[i + 1] + empty_px[i + 2];
        std::printf("    camera turned away: %d indirect draws, total brightness "
                    "%lld (was %lld)\n", gpu_stats.draws, lit, gpu_lit);
        Check(gpu_stats.draws == 3, "the draws are still submitted");
        Check(lit * 20 < gpu_lit, "but the cull emptied them");
    }

    std::printf(g_failures == 0 ? "\nmanyobjects_test: all checks passed\n"
                                : "\nmanyobjects_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
