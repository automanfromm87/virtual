// Mesh shaders, checked against the vertex pipeline.
//
// meshlet_test proves the SPLIT is sound: every triangle survives, the
// two-level indices reconstruct the mesh, the bounds contain their meshlets and
// the cones contain their normals. None of that says the mesh stage draws the
// same picture, and none of it says the culling culls anything.
//
// So: the same scene both ways, compared pixel for pixel, and then a camera
// moved so that most of the geometry is off screen, with the surviving-meshlet
// count read back.
#include "engine/geometry/mesh.h"
#include "engine/geometry/meshlet.h"
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

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

constexpr int kW = 400, kH = 400;

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
    if (!r) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }

    if (!r->SupportsMeshShaders()) {
        // NOT a failure. The engine's contract is that the caller falls back to
        // the vertex path, and a machine without a mesh stage is a machine the
        // engine still has to work on. Reporting it and passing is the honest
        // outcome; failing would make the test a hardware check.
        std::printf("this GPU has no mesh shader stage; nothing to test\n");
        std::printf("\nmeshlet_render_test: skipped\n");
        return 0;
    }
    std::printf("the GPU reports a mesh shader stage\n");

    const Mesh sphere =
        MakeUVSphere(1.0f, 48, 64, Vec4{1, 1, 1, 1}, Vec4{1, 1, 1, 1});
    const MeshletBuild build = BuildMeshlets(sphere);
    std::printf("    %zu triangles in %zu meshlets\n", sphere.indices.size() / 3,
                build.meshlets.size());

    const MeshHandle vertex_mesh = r->UploadMesh(sphere);
    const MeshHandle meshlet_mesh = r->UploadMeshlets(sphere, error);
    if (!Valid(meshlet_mesh)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }
    Check(Valid(vertex_mesh) && Valid(meshlet_mesh),
          "the same mesh uploads both ways");

    MaterialDesc md;
    md.base_color = Vec4{0.8f, 0.8f, 0.82f, 1.0f};
    md.roughness = 0.45f;
    const MaterialHandle mat = r->CreateMaterial(md, error);

    const rhi::TextureId hdr =
        dev->CreateRenderTarget(kW, kH, Renderer::kSceneFormat);
    const rhi::TextureId ldr =
        dev->CreateRenderTarget(kW, kH, kFmt, /*cpu_readable=*/true);
    const rhi::TextureId depth = dev->CreateDepthTarget(kW, kH);
    const rhi::BufferId stats = dev->CreateStorageBuffer(sizeof(std::uint32_t) * 4);
    std::vector<std::uint8_t> px(std::size_t(kW) * kH * 4);

    const auto build_scene = [&](MeshHandle mesh, Vec3 eye) {
        Scene s;
        s.camera.eye = eye;
        s.camera.target = Vec3{0.0f, 0.0f, 0.0f};
        s.camera.fovY = 0.9f;
        s.lightDir = Vec4{0.4f, 0.6f, 0.7f, 0.0f};
        s.lightColor = Vec4{3.2f, 3.2f, 3.2f, 1.0f};
        s.ambientSky = Vec3{0.05f, 0.05f, 0.07f};
        s.ambientGround = Vec3{0.02f, 0.02f, 0.02f};
        Instance in;
        in.mesh = mesh;
        in.material = mat;
        s.instances.push_back(in);
        return s;
    };

    const auto render = [&](const Scene& s, bool use_meshlets) -> std::uint32_t {
        if (auto* z = static_cast<std::uint32_t*>(dev->MapBuffer(stats)))
            for (int i = 0; i < 4; ++i) z[i] = 0;
        dev->BeginFrame();
        {
            rhi::PassDesc pd;
            pd.color = hdr;
            pd.depth = depth;
            pd.clear_depth = 0.0f;
            pd.timer = "geometry";
            auto e = dev->BeginPass(pd);
            if (use_meshlets) r->DrawSceneMeshlets(e, s, kW, kH, stats, {});
            else r->DrawScene(e, s, kW, kH, {});
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
            std::exit(1);
        }
        if (!dev->ReadPixels(ldr, kW, kH, px)) {
            std::fprintf(stderr, "FAIL: readback\n");
            std::exit(1);
        }
        const auto* got = static_cast<const std::uint32_t*>(dev->MapBuffer(stats));
        return got ? got[0] : 0u;
    };

    // ------------------------------------------------------ the same picture --
    {
        std::printf("\nthe mesh stage draws what the vertex stage draws\n");
        const Vec3 eye{0.0f, 0.0f, 3.2f};
        render(build_scene(vertex_mesh, eye), false);
        const std::vector<std::uint8_t> by_vertex = px;
        const std::uint32_t drawn = render(build_scene(meshlet_mesh, eye), true);
        // The pipeline building is the one failure that produces no error at
        // the draw: DrawSceneMeshlets becomes a no-op, and a silent no-op is
        // indistinguishable from an empty scene.
        Check(r->MeshletError().empty(), "the mesh pipeline built");
        if (!r->MeshletError().empty())
            std::printf("    %s\n", r->MeshletError().c_str());
        const std::vector<std::uint8_t> by_mesh = px;

        int lit = 0;
        for (int i = 0; i < kW * kH; ++i)
            if (Luma(by_vertex, std::size_t(i) * 4) > 10.0) ++lit;
        std::printf("    the sphere covers %d px; %u of %zu meshlets survived\n",
                    lit, drawn, build.meshlets.size());
        Check(lit > 20000, "the reference image is of something");

        double worst = 0.0, sum = 0.0;
        int over2 = 0;
        for (int i = 0; i < kW * kH; ++i)
            for (int c = 0; c < 3; ++c) {
                const double d = std::fabs(double(by_vertex[std::size_t(i) * 4 + c]) -
                                           double(by_mesh[std::size_t(i) * 4 + c]));
                worst = std::max(worst, d);
                sum += d;
                if (d > 2.0) ++over2;
            }
        std::printf("    against the vertex path: worst %.0f, mean %.4f, "
                    "%d samples over 2\n",
                    worst, sum / (kW * kH * 3), over2);
        // BIT-IDENTICAL, and it comes out that way. The two paths run the same
        // fragment stage on the same vertices, and the mesh stage rasterises
        // the same triangles in the same winding -- the only difference is
        // where the indices came from. Nineteen meshlets are culled in the
        // frame this compares, so the culling is provably rejecting only
        // geometry that was invisible.
        Check(worst == 0.0, "and matches the vertex path exactly, pixel for pixel");
        Check(over2 == 0, "with not one sample differing");
    }

    // ----------------------------------------------------------- the culling --
    {
        std::printf("\nand culls what is not visible\n");
        // FACING the sphere: about half its meshlets face away and the cone
        // test should reject them. This is the number the vertex pipeline
        // cannot avoid paying, because its unit of culling is the object.
        const std::uint32_t facing =
            render(build_scene(meshlet_mesh, Vec3{0.0f, 0.0f, 3.2f}), true);
        std::printf("    from 3.2 m away: %u of %zu meshlets survive (%.0f%%)\n",
                    facing, build.meshlets.size(),
                    100.0 * facing / double(build.meshlets.size()));
        Check(facing > 0, "something survives when the sphere is on screen");
        // ABOUT HALF, which is the ceiling: a sphere shows one hemisphere, and
        // the meshlets straddling the terminator have normals spanning it and
        // can never be rejected.
        //
        // This read 81% while BuildMeshlets walked the index array -- a uv
        // sphere's index order runs along a whole latitude band, so a meshlet
        // came out as a ring with a bounding sphere larger than the model and
        // a 49-degree cone. Clustering by locality took it to 51%.
        //
        // And the frame is unchanged by any of it: the comparison above is
        // bit-identical, so every rejected meshlet was invisible.
        Check(facing < build.meshlets.size() * 3 / 5,
              "and about half of them -- the back-facing half -- are rejected");
        Check(facing > build.meshlets.size() / 3,
              "but not so many that something visible is being thrown away");

        // OFF SCREEN entirely: the camera turned away. The frustum test should
        // reject every meshlet, and the mesh stage should never run.
        Scene away = build_scene(meshlet_mesh, Vec3{0.0f, 0.0f, 3.2f});
        away.camera.target = Vec3{0.0f, 0.0f, 20.0f};  // looking the other way
        const std::uint32_t behind = render(away, true);
        int lit_behind = 0;
        for (int i = 0; i < kW * kH; ++i)
            if (Luma(px, std::size_t(i) * 4) > 10.0) ++lit_behind;
        std::printf("    looking the other way: %u meshlets survive, %d px lit\n",
                    behind, lit_behind);
        Check(behind == 0, "with the sphere behind the camera, nothing survives");
        Check(lit_behind == 0, "and nothing is drawn");

        // AND THE CULLING IS NOT JUST THE FRUSTUM. A camera very close to the
        // sphere sees only a small part of it, and both tests apply: most of
        // the sphere is outside the frustum AND most of what is inside faces
        // away.
        const std::uint32_t close_up =
            render(build_scene(meshlet_mesh, Vec3{0.0f, 0.0f, 1.15f}), true);
        std::printf("    from 1.15 m, nearly touching it: %u survive\n", close_up);
        Check(close_up < facing,
              "moving closer rejects more of it, not less");
    }

    std::printf(g_failures == 0 ? "\nmeshlet_render_test: all checks passed\n"
                                : "\nmeshlet_render_test: %d FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
