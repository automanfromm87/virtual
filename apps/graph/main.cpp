// Pure C++20. Exercises the render graph with two passes and a real dependency.
//
// The passes are added in the WRONG order on purpose: composite first, scene
// second. A graph that just replays insertion order would produce a black
// screen (composite sampling a texture nothing had written yet). Getting the
// right picture out is the proof that it sorted.
#include "engine/render/rendergraph.h"
#include "engine/render/renderer.h"
#include "engine/rhi/rhi.h"
#include "engine/scene/scene.h"

#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-52s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

double Luma(const std::vector<std::uint8_t>& px, std::size_t i) {
    return 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2];
}

}  // namespace

int main() {
    constexpr int kW = 640, kH = 640;

    std::string error;
    auto dev = eng::rhi::Device::Create(error);
    if (!dev) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    auto renderer = eng::Renderer::Create(*dev, eng::rhi::Format::RGBA8Unorm, error);
    if (!renderer) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    // scene_color is written by one pass and sampled by the other; final is
    // what we read back.
    const eng::rhi::TextureId scene_color =
        dev->CreateRenderTarget(kW, kH, eng::rhi::Format::RGBA8Unorm, true);
    const eng::rhi::TextureId scene_depth = dev->CreateDepthTarget(kW, kH);
    const eng::rhi::TextureId final_color =
        dev->CreateRenderTarget(kW, kH, eng::rhi::Format::RGBA8Unorm, true);
    if (!Valid(scene_color) || !Valid(scene_depth) || !Valid(final_color)) {
        std::fprintf(stderr, "FAIL: targets\n");
        return 1;
    }

    const eng::Scene scene = eng::ShapesDemo(0.0f);
    eng::RenderGraph graph;

    // Added FIRST, must run SECOND.
    eng::RenderGraph::Pass composite;
    composite.name = "composite";
    composite.color = final_color;
    composite.reads = {scene_color};
    composite.execute = [&](eng::rhi::Encoder& e) {
        renderer->DrawComposite(e, scene_color);
    };
    graph.AddPass(std::move(composite));

    // Added SECOND, must run FIRST.
    eng::RenderGraph::Pass scene_pass;
    scene_pass.name = "scene";
    scene_pass.color = scene_color;
    scene_pass.depth = scene_depth;
    for (int i = 0; i < 4; ++i) scene_pass.clear_color[i] = eng::kClearColor[i];
    scene_pass.clear_depth = 0.0f;
    scene_pass.execute = [&](eng::rhi::Encoder& e) {
        renderer->DrawScene(e, scene, kW, kH);
    };
    graph.AddPass(std::move(scene_pass));

    if (!graph.Compile(error)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }

    dev->BeginFrame();
    graph.Execute(*dev);
    if (!dev->CommitAndWait(error)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }

    std::vector<std::uint8_t> px(std::size_t(kW) * kH * 4);
    if (!dev->ReadPixels(final_color, kW, kH, px)) {
        std::fprintf(stderr, "FAIL: readback\n");
        return 1;
    }
    if (std::FILE* fp = std::fopen("graph.ppm", "wb")) {
        std::fprintf(fp, "P6\n%d %d\n255\n", kW, kH);
        for (std::size_t i = 0; i + 3 < px.size(); i += 4)
            std::fwrite(&px[i], 1, 3, fp);
        std::fclose(fp);
    }

    auto at = [&](int x, int y) { return std::size_t(y) * kW * 4 + std::size_t(x) * 4; };

    // Vignette: the composite darkens towards the edges, so a background pixel
    // in the corner must be darker than one just off-centre. Without the
    // composite pass the background is perfectly uniform.
    const double corner = Luma(px, at(4, 4));
    const double mid = Luma(px, at(kW / 2, 40));

    std::set<std::uint32_t> distinct;
    for (std::size_t i = 0; i + 3 < px.size(); i += 4)
        distinct.insert((std::uint32_t(px[i]) << 16) |
                        (std::uint32_t(px[i + 1]) << 8) | px[i + 2]);

    // Orientation survived the fullscreen flip: the scene is still lit from the
    // upper right. Measured on the PRE-composite target, and only over pixels
    // that are actually geometry.
    //
    // Two reasons not to use the final image: the vignette makes the background
    // non-uniform so "is this a background pixel" stops being answerable, and
    // averaging whole quadrants dilutes the signal with background anyway. The
    // dilution used to be survivable; after tone mapping and gamma compressed
    // the shading range it is not, and the check would fail on a perfectly
    // correct image.
    std::vector<std::uint8_t> scene_px(std::size_t(kW) * kH * 4);
    if (!dev->ReadPixels(scene_color, kW, kH, scene_px)) {
        std::fprintf(stderr, "FAIL: scene readback\n");
        return 1;
    }
    const std::uint8_t bg[3] = {scene_px[0], scene_px[1], scene_px[2]};
    double lit = 0, dark = 0;
    std::size_t litN = 0, darkN = 0;
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const std::size_t i = at(x, y);
            if (scene_px[i] == bg[0] && scene_px[i + 1] == bg[1] &&
                scene_px[i + 2] == bg[2])
                continue;
            const double l = Luma(scene_px, i);
            if (x > kW / 2 && y < kH / 2) { lit += l; ++litN; }
            if (x < kW / 2 && y > kH / 2) { dark += l; ++darkN; }
        }
    }
    lit = litN ? lit / double(litN) : 0.0;
    dark = darkN ? dark / double(darkN) : 0.0;

    const std::vector<std::string>& order = graph.Order();
    std::printf("graph order: ");
    for (const std::string& n : order) std::printf("%s ", n.c_str());
    std::printf("\n  corner=%.1f mid=%.1f  lit=%.1f dark=%.1f  colours=%zu\n",
                corner, mid, lit, dark, distinct.size());

    Check(order.size() == 2 && order[0] == "scene" && order[1] == "composite",
          "graph reordered: scene runs before composite");
    Check(distinct.size() > 200, "the scene actually made it through both passes");
    Check(corner < mid * 0.8, "vignette applied, so the composite pass really ran");
    Check(lit > dark * 1.3, "orientation survived the fullscreen flip");

    // --- things Compile() must REFUSE ----------------------------------------
    // A read with no producer must be caught, not silently drawn black.
    {
        eng::RenderGraph bad;
        eng::RenderGraph::Pass p;
        p.name = "orphan";
        p.color = final_color;
        p.reads = {dev->CreateRenderTarget(8, 8, eng::rhi::Format::RGBA8Unorm)};
        bad.AddPass(std::move(p));
        std::string bad_error;
        Check(!bad.Compile(bad_error), "reading an unwritten texture is rejected");
    }

    // Two passes writing one colour target. Passes are declared in arbitrary
    // order, so nothing decides which write lands first — without resource
    // versioning the only correct answer is to refuse.
    {
        eng::RenderGraph bad;
        eng::RenderGraph::Pass a, b;
        a.name = "writer_a";
        a.color = final_color;
        b.name = "writer_b";
        b.color = final_color;
        bad.AddPass(std::move(a));
        bad.AddPass(std::move(b));
        std::string bad_error;
        Check(!bad.Compile(bad_error), "two writers of one colour target rejected");
    }

    // Same for a shared depth buffer: colour dependencies alone cannot make the
    // depth contents deterministic.
    {
        eng::RenderGraph bad;
        eng::RenderGraph::Pass a, b;
        a.name = "depth_a";
        a.color = final_color;
        a.depth = scene_depth;
        b.name = "depth_b";
        b.color = scene_color;
        b.depth = scene_depth;
        bad.AddPass(std::move(a));
        bad.AddPass(std::move(b));
        std::string bad_error;
        Check(!bad.Compile(bad_error), "two passes sharing a depth target rejected");
    }

    // A genuine 3-pass chain, declared in fully reversed order. Reverse-insertion
    // order alone would give c,b,a — only a real topological sort gives a,b,c.
    {
        const eng::rhi::TextureId t1 =
            dev->CreateRenderTarget(32, 32, eng::rhi::Format::RGBA8Unorm);
        const eng::rhi::TextureId t2 =
            dev->CreateRenderTarget(32, 32, eng::rhi::Format::RGBA8Unorm);
        const eng::rhi::TextureId t3 =
            dev->CreateRenderTarget(32, 32, eng::rhi::Format::RGBA8Unorm);
        eng::RenderGraph chain;
        eng::RenderGraph::Pass c, a, b;
        c.name = "c"; c.color = t3; c.reads = {t2};
        a.name = "a"; a.color = t1;
        b.name = "b"; b.color = t2; b.reads = {t1};
        chain.AddPass(std::move(c));   // added 1st, must run 3rd
        chain.AddPass(std::move(a));   // added 2nd, must run 1st
        chain.AddPass(std::move(b));   // added 3rd, must run 2nd
        std::string chain_error;
        const bool ok = chain.Compile(chain_error);
        const std::vector<std::string>& o = chain.Order();
        Check(ok && o.size() == 3 && o[0] == "a" && o[1] == "b" && o[2] == "c",
              "3-pass chain sorts a,b,c from insertion order c,a,b");
    }

    // A cycle must be reported, not silently truncated to a partial order.
    {
        const eng::rhi::TextureId x =
            dev->CreateRenderTarget(16, 16, eng::rhi::Format::RGBA8Unorm);
        const eng::rhi::TextureId y =
            dev->CreateRenderTarget(16, 16, eng::rhi::Format::RGBA8Unorm);
        eng::RenderGraph cyc;
        eng::RenderGraph::Pass p, q;
        p.name = "p"; p.color = x; p.reads = {y};
        q.name = "q"; q.color = y; q.reads = {x};
        cyc.AddPass(std::move(p));
        cyc.AddPass(std::move(q));
        std::string cyc_error;
        const bool ok = cyc.Compile(cyc_error);
        Check(!ok, "a dependency cycle is rejected");
        // On failure the order must be EMPTY, not a partial one that Execute
        // would happily run.
        Check(cyc.Order().empty(), "a failed Compile leaves no runnable order");
    }

    // Clear() must forget the passes AND the compiled order.
    {
        eng::RenderGraph g2;
        eng::RenderGraph::Pass p;
        p.name = "solo";
        p.color = final_color;
        g2.AddPass(std::move(p));
        std::string e2;
        Check(g2.Compile(e2) && g2.Order().size() == 1, "single pass compiles");
        g2.Clear();
        Check(g2.Order().empty(), "Clear() drops the compiled order");
        // Compiling an empty graph is a no-op, not an error.
        Check(g2.Compile(e2) && g2.Order().empty(), "an empty graph compiles to nothing");
    }

    // Independent passes have no edges between them, so the tie-break decides.
    // It must be deterministic (insertion order), or the frame shuffles run to
    // run and nothing about it is reproducible.
    {
        eng::rhi::TextureId t[3];
        for (int i = 0; i < 3; ++i)
            t[i] = dev->CreateRenderTarget(16, 16, eng::rhi::Format::RGBA8Unorm);
        eng::RenderGraph g3;
        const char* names[3] = {"first", "second", "third"};
        for (int i = 0; i < 3; ++i) {
            eng::RenderGraph::Pass p;
            p.name = names[i];
            p.color = t[i];
            g3.AddPass(std::move(p));
        }
        std::string e3;
        const bool ok = g3.Compile(e3);
        const std::vector<std::string>& o = g3.Order();
        Check(ok && o.size() == 3 && o[0] == "first" && o[1] == "second" &&
                  o[2] == "third",
              "independent passes keep insertion order (deterministic ties)");
    }

    // Execute() without a successful Compile() must be a no-op, not a crash and
    // not a partial frame.
    {
        eng::RenderGraph g4;
        eng::RenderGraph::Pass p;
        p.name = "never_compiled";
        p.color = final_color;
        p.execute = [](eng::rhi::Encoder&) {
            std::fprintf(stderr, "executed a pass that was never compiled\n");
            ++g_failures;
        };
        g4.AddPass(std::move(p));
        dev->BeginFrame();
        g4.Execute(*dev);  // no Compile() call at all
        if (!dev->CommitAndWait(error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
        Check(g4.Order().empty(), "Execute without Compile runs nothing");
    }

    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
