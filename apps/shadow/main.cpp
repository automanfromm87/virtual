// Pure C++20. Shadow mapping through the render graph: a depth-only pass from
// the light's point of view, then the lit pass sampling it.
//
// This is the first frame where the graph earns its keep. The shadow pass
// writes a depth target that the scene pass READS, so the ordering is a real
// dependency rather than a convention — and the passes are declared in the
// wrong order here to prove the graph derives it.
//
// The test is differential: render the same scene with shadows on and off. A
// shadow can only ever REMOVE light, so every pixel must be darker-or-equal,
// and a decent number must be strictly darker. No geometry maths in the test,
// no magic pixel coordinates.
#include "engine/geometry/mesh.h"
#include "engine/render/rendergraph.h"
#include "engine/render/renderer.h"
#include "engine/rhi/rhi.h"
#include "engine/scene/scene.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-54s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

double Luma(const std::vector<std::uint8_t>& p, std::size_t i) {
    return 0.2126 * p[i] + 0.7152 * p[i + 1] + 0.0722 * p[i + 2];
}

constexpr int kW = 640, kH = 640;
constexpr int kShadowSize = 1024;

// Ground is a big CUBE, not a squashed one: Mat4::Scale is uniform only, and
// keeping it that way is deliberate — a non-uniform scale silently breaks every
// shader that transforms a normal by the model matrix.
eng::Scene BuildScene(eng::MeshHandle ground_mesh, eng::MaterialHandle mat,
                      float shadow_extent) {
    eng::Scene s;
    s.camera.eye = eng::Vec3{4.5f, 4.0f, 7.5f};
    s.camera.target = eng::Vec3{0.0f, 0.0f, 0.0f};
    s.lightDir = eng::Vec4{0.45f, 0.80f, 0.40f, 0.0f};
    s.shadowExtent = shadow_extent;

    // MakeCube(1.4) has a half-extent of 0.7; scaled by 8 that is 5.6, so a
    // centre at -6.6 puts the top face at y = -1.
    eng::Instance ground;
    ground.mesh = ground_mesh;
    ground.material = mat;
    ground.model = eng::Mat4::Translation({0.0f, -6.6f, 0.0f}) * eng::Mat4::Scale(8.0f);
    ground.tint = eng::Vec4{0.75f, 0.75f, 0.78f, 1.0f};
    s.instances.push_back(ground);

    // RESTING on the ground, not hovering above it. The ground's top face is at
    // y = -1 and the spheres have a world radius of 0.8, so a centre at -0.2 is
    // exactly tangent. The previous values put them 0.4 to 0.8 units in the
    // air, which no amount of shading can make look grounded.
    constexpr float kRest = -0.2f;
    const eng::Vec3 at[3] = {
        {-1.6f, kRest, 0.4f}, {0.9f, kRest, -0.8f}, {1.7f, kRest, 1.6f}};
    const eng::Vec4 tints[3] = {{1.0f, 0.45f, 0.35f, 1.0f},
                                {0.35f, 1.0f, 0.5f, 1.0f},
                                {0.55f, 0.6f, 1.0f, 1.0f}};
    for (int i = 0; i < 3; ++i) {
        eng::Instance sphere;
        sphere.material = mat;
        sphere.model = eng::Mat4::Translation(at[i]) * eng::Mat4::Scale(0.8f);
        sphere.tint = tints[i];
        s.instances.push_back(sphere);
    }
    return s;
}

}  // namespace

int main() {
    std::string error;
    auto dev = eng::rhi::Device::Create(error);
    if (!dev) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
    auto renderer = eng::Renderer::Create(*dev, eng::rhi::Format::RGBA8Unorm, error);
    if (!renderer) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    // Plain cube for the ground: the built-in carries a vertex-colour checker,
    // which would make "did this pixel get darker" harder to read.
    const eng::MeshHandle ground_mesh = renderer->UploadMesh(
        eng::MakeCube(1.4f, eng::Vec4{1, 1, 1, 1}, eng::Vec4{1, 1, 1, 1}));
    eng::MaterialDesc md;
    md.roughness = 0.6f;
    const eng::MaterialHandle mat = renderer->CreateMaterial(md, error);
    if (!Valid(ground_mesh) || !Valid(mat)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }

    const eng::rhi::TextureId shadow_map = dev->CreateShadowMap(kShadowSize);
    const eng::rhi::TextureId color =
        dev->CreateRenderTarget(kW, kH, eng::rhi::Format::RGBA8Unorm, true);
    const eng::rhi::TextureId depth = dev->CreateDepthTarget(kW, kH);
    if (!Valid(shadow_map) || !Valid(color) || !Valid(depth)) {
        std::fprintf(stderr, "FAIL: targets\n");
        return 1;
    }

    std::vector<std::uint8_t> with(std::size_t(kW) * kH * 4);
    std::vector<std::uint8_t> without(with.size());
    std::vector<std::string> order;

    for (int pass_shadows = 1; pass_shadows >= 0; --pass_shadows) {
        const eng::Scene scene =
            BuildScene(ground_mesh, mat, pass_shadows ? 6.0f : 0.0f);

        eng::RenderGraph graph;
        // Declared SECOND, must run FIRST — the graph works that out from the
        // fact that the scene pass reads what this one writes.
        eng::RenderGraph::Pass lit;
        lit.name = "lit";
        lit.color = color;
        lit.depth = depth;
        for (int i = 0; i < 4; ++i) lit.clear_color[i] = eng::kClearColor[i];
        lit.clear_depth = 0.0f;
        if (pass_shadows) lit.reads = {shadow_map};
        lit.execute = [&](eng::rhi::Encoder& e) {
            renderer->DrawScene(e, scene, kW, kH,
                                pass_shadows ? shadow_map : eng::rhi::TextureId{});
        };
        graph.AddPass(std::move(lit));

        if (pass_shadows) {
            eng::RenderGraph::Pass sh;
            sh.name = "shadow";
            sh.depth = shadow_map;   // depth-only: no colour target at all
            sh.clear_depth = 0.0f;   // reversed-Z far value
            sh.keep_depth = true;    // the lit pass has to be able to read it
            sh.execute = [&](eng::rhi::Encoder& e) {
                renderer->DrawShadow(e, scene);
            };
            graph.AddPass(std::move(sh));
        }

        if (!graph.Compile(error)) {
            std::fprintf(stderr, "FAIL: %s\n", error.c_str());
            return 1;
        }
        if (pass_shadows) order = graph.Order();

        dev->BeginFrame();
        graph.Execute(*dev);
        if (!dev->CommitAndWait(error)) {
            std::fprintf(stderr, "FAIL: %s\n", error.c_str());
            return 1;
        }
        if (!dev->ReadPixels(color, kW, kH, pass_shadows ? with : without)) {
            std::fprintf(stderr, "FAIL: readback\n");
            return 1;
        }
    }

    if (std::FILE* fp = std::fopen("shadow.ppm", "wb")) {
        std::fprintf(fp, "P6\n%d %d\n255\n", kW, kH);
        for (std::size_t i = 0; i + 3 < with.size(); i += 4)
            std::fwrite(&with[i], 1, 3, fp);
        std::fclose(fp);
    }

    // --- compare -------------------------------------------------------------
    std::size_t darker = 0, brighter = 0;
    double worst_brighten = 0.0;
    for (std::size_t i = 0; i + 3 < with.size(); i += 4) {
        const double d = Luma(without, i) - Luma(with, i);
        if (d > 12.0) ++darker;
        if (d < -3.0) {
            ++brighter;
            if (-d > worst_brighten) worst_brighten = -d;
        }
    }
    const double pct = 100.0 * double(darker) / double(kW * kH);

    std::printf("graph order:");
    for (const std::string& n : order) std::printf(" %s", n.c_str());
    std::printf("\n  %zu px darkened (%.1f%%), %zu px brightened (worst %.1f)\n",
                darker, pct, brighter, worst_brighten);

    Check(order.size() == 2 && order[0] == "shadow" && order[1] == "lit",
          "graph put the shadow pass before the lit pass");
    // Enough to be a real shadow, not so much that the whole frame went dark
    // (which is what a broken lookup or a wrong bias sign produces).
    //
    // The lower bound is 0.2%, not 1%: once the spheres REST on the ground
    // instead of hovering, each one hides most of its own shadow from a camera
    // that is above and in front. A grounded object showing less shadow than a
    // floating one is the expected result, not a regression.
    Check(pct > 0.2 && pct < 25.0, "a plausible fraction of the frame is shadowed");
    // THE invariant: occlusion can only subtract light. Anything brighter means
    // the shadow term is being applied with the wrong sign somewhere.
    Check(brighter < (kW * kH) / 500, "shadows only ever darken, never brighten");

    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
