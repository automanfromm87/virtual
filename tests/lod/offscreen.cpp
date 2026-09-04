// LEVELS OF DETAIL and OCCLUSION CULLING, measured on the GPU-driven path.
//
// Both of these are invisible when they work and invisible when they do not.
// A level-of-detail system that never switches looks exactly like one that
// does, from the camera position where it was tuned. An occlusion test that
// culls nothing costs a compute pass and looks perfect. The only way to tell is
// to read back what the GPU decided.
//
// Two numbers do that. The SURVIVOR COUNT says how many instances the cull
// kept, which is what occlusion moves. The TRIANGLE COUNT says how much
// geometry the indirect draws will submit, which is what levels of detail move
// and the survivor count does not show at all -- switching every object to a
// tenth of its triangles leaves the survivor count identical.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/geometry/mesh.h"
#include "engine/geometry/simplify.h"
#include "engine/render/renderer.h"
#include "engine/rhi/rhi.h"
#include "engine/scene/scene.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

constexpr int kW = 640, kH = 400;

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string error;
    auto dev = eng::rhi::Device::Create(error);
    if (!dev) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
    const auto kFmt = eng::rhi::Format::RGBA8Unorm;
    auto r = eng::Renderer::Create(*dev, kFmt, error, 1);
    if (!r) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    const eng::Mesh sphere =
        eng::MakeUVSphere(2.0f, 40, 56, eng::Vec4{1, 1, 1, 1}, eng::Vec4{0.7f, 0.7f, 0.8f, 1});
    const std::vector<eng::Mesh> chain = eng::BuildLodChain(sphere, 4);
    const eng::MeshHandle ball = r->UploadMeshLods(chain);
    eng::MaterialDesc md;
    md.shading = eng::Shading::Lit;
    const eng::MaterialHandle mat = r->CreateMaterial(md, error);
    if (!eng::Valid(ball) || !eng::Valid(mat)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }
    std::printf("levels uploaded: %d\n", r->MeshLodCount(ball));
    for (int l = 0; l < r->MeshLodCount(ball); ++l)
        std::printf("    level %d: %d triangles\n", l, r->MeshLodIndexCount(ball, l) / 3);
    Check(r->MeshLodCount(ball) >= 3, "at least three levels made it to the GPU");

    eng::Scene scene;
    scene.lightDir = eng::Vec4{0.3f, 0.9f, 0.3f, 0.0f};
    scene.lightColor = eng::Vec4{3.0f, 3.0f, 3.0f, 1.0f};
    scene.ambientSky = eng::Vec3{0.2f, 0.22f, 0.28f};

    // A GRID, all at the same distance from the camera on each axis, so that
    // moving the camera moves every object's screen size together and the level
    // distribution is a single number rather than a spread.
    constexpr int kSide = 12;
    for (int z = 0; z < kSide; ++z)
        for (int x = 0; x < kSide; ++x) {
            eng::Instance inst;
            inst.mesh = ball;
            inst.material = mat;
            inst.model = eng::Mat4::Translation(
                eng::Vec3{(float(x) - kSide * 0.5f) * 6.0f, 0.0f,
                          (float(z) - kSide * 0.5f) * 6.0f});
            scene.instances.push_back(inst);
        }
    const int kObjects = kSide * kSide;

    const eng::rhi::TextureId color =
        dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
    const eng::rhi::TextureId depth =
        dev->CreateDepthTarget(kW, kH, /*sampleable=*/true);

    // Runs the whole GPU-driven frame: an optional depth prepass, the pyramid,
    // the cull, and the indirect draws.
    const auto frame = [&](bool occlusion) -> bool {
        r->SetOcclusionCulling(occlusion);
        dev->BeginFrame();
        if (occlusion) {
            eng::rhi::PassDesc pd;
            pd.depth = depth;
            pd.keep_depth = true;
            auto e = dev->BeginPass(pd);
            r->DrawSceneDepth(e, scene, kW, kH);
            dev->EndPass();
            auto ce = dev->BeginCompute();
            r->BuildHiZ(ce, depth, kW, kH);
            dev->EndCompute();
        }
        {
            auto ce = dev->BeginCompute();
            (void)r->CullScene(ce, scene, kW, kH);
            dev->EndCompute();
        }
        {
            eng::rhi::PassDesc pd;
            pd.color = color;
            pd.depth = depth;
            auto e = dev->BeginPass(pd);
            r->DrawSceneIndirect(e, scene, kW, kH, {});
            dev->EndPass();
        }
        if (!dev->CommitAndWait(error)) {
            std::fprintf(stderr, "FAIL submit: %s\n", error.c_str());
            return false;
        }
        return true;
    };

    {
        std::printf("\nlevels are chosen by screen size\n");
        // A CAMERA INSIDE THE GRID, so one frame contains objects at every
        // distance. The first attempt put it 12 metres from a grid of
        // half-metre spheres and asserted that some drew at full detail -- at
        // that size and distance their screen radius is eleven pixels, so the
        // correct answer was level 2 and the test was wrong about the scene
        // rather than about the code.
        scene.camera.eye = eng::Vec3{0.0f, 3.0f, 40.0f};
        scene.camera.target = eng::Vec3{0.0f, 0.0f, -20.0f};
        if (!frame(false)) return 1;
        const int near_visible = r->VisibleAfterCull();
        const long long near_tris = r->IndirectTriangles();
        const double near_per_object = double(near_tris) / std::max(near_visible, 1);
        std::printf("    inside the grid: %d visible, %lld triangles (%.0f each), "
                    "per level %d/%d/%d/%d\n",
                    near_visible, near_tris, near_per_object, r->VisibleAtLod(0),
                    r->VisibleAtLod(1), r->VisibleAtLod(2), r->VisibleAtLod(3));
        Check(r->VisibleAtLod(0) > 0, "the nearest objects draw at full detail");
        // SEVERAL LEVELS AT ONCE. A selector that picked one level for the
        // whole scene -- keyed off the camera rather than off each object --
        // would pass every other check here.
        int levels_used = 0;
        for (int l = 0; l < 4; ++l)
            if (r->VisibleAtLod(l) > 0) ++levels_used;
        Check(levels_used >= 3, "and the same frame uses three or more levels");

        // FAR: the same objects, a long way off.
        scene.camera.eye = eng::Vec3{0.0f, 120.0f, 420.0f};
        scene.camera.target = eng::Vec3{0.0f, 0.0f, 0.0f};
        if (!frame(false)) return 1;
        const int far_visible = r->VisibleAfterCull();
        const long long far_tris = r->IndirectTriangles();
        const double far_per_object = double(far_tris) / std::max(far_visible, 1);
        std::printf("    far away:        %d visible, %lld triangles (%.0f each), "
                    "per level %d/%d/%d/%d\n",
                    far_visible, far_tris, far_per_object, r->VisibleAtLod(0),
                    r->VisibleAtLod(1), r->VisibleAtLod(2), r->VisibleAtLod(3));

        Check(far_visible >= kObjects * 9 / 10,
              "nearly every object is still visible from a distance");
        // AGAINST FULL DETAIL, not against the near camera's average.
        //
        // Comparing the two averages asked for a fourfold drop and got 537 to
        // 257, and the code was right: a camera inside the grid still has most
        // of the grid in the distance, so its average is ALREADY mostly
        // level 2. Two already-reduced averages cannot differ by much, and the
        // saving that matters is against what the scene would have cost with no
        // levels at all.
        const double full_detail = double(r->MeshLodIndexCount(ball, 0)) / 3.0;
        std::printf("    full detail is %.0f triangles per object\n", full_detail);
        Check(far_per_object < full_detail / 4.0,
              "distant objects cost under a quarter of full detail");
        Check(near_per_object > far_per_object * 1.5,
              "and closer ones cost more than distant ones");
        Check(r->VisibleAtLod(0) == 0, "and none of them is still at full detail");
    }

    {
        std::printf("\nlevels respond to the thresholds\n");
        scene.camera.eye = eng::Vec3{0.0f, 30.0f, 130.0f};
        r->SetLodThresholds(eng::Vec3{1.0f, 0.5f, 0.25f});  // effectively never switch
        if (!frame(false)) return 1;
        const long long always_detailed = r->IndirectTriangles();
        r->SetLodThresholds(eng::Vec3{1e6f, 1e5f, 1e4f});  // always the coarsest
        if (!frame(false)) return 1;
        const long long always_coarse = r->IndirectTriangles();
        std::printf("    same camera: %lld triangles at full detail, %lld at the "
                    "coarsest\n", always_detailed, always_coarse);
        // The thresholds are the only thing that changed, so this isolates the
        // selection from the distance.
        Check(always_coarse < always_detailed / 4,
              "the thresholds alone change which level is drawn");
        r->SetLodThresholds(eng::Vec3{60.0f, 24.0f, 9.0f});
    }

    {
        std::printf("\nocclusion culls what is behind a wall\n");
        // A WALL across the front of the grid, close to the camera and large
        // enough to hide everything behind it.
        const eng::MeshHandle wall_mesh = r->UploadMesh(
            eng::MakeBox(eng::Vec3{140.0f, 90.0f, 0.5f}, eng::Vec4{1, 1, 1, 1}));
        eng::Instance wall;
        wall.mesh = wall_mesh;
        wall.material = mat;
        wall.model = eng::Mat4::Translation(eng::Vec3{0.0f, 0.0f, 50.0f});
        scene.instances.push_back(wall);

        scene.camera.eye = eng::Vec3{0.0f, 0.0f, 90.0f};
        scene.camera.target = eng::Vec3{0.0f, 0.0f, 0.0f};

        if (!frame(false)) return 1;
        const int without = r->VisibleAfterCull();
        if (!frame(true)) return 1;
        const int with = r->VisibleAfterCull();
        std::printf("    %d objects behind a wall: %d survive the frustum test, "
                    "%d survive occlusion too\n", kObjects, without, with);
        Check(without > kObjects / 2, "the frustum test alone keeps most of them");
        Check(with < without / 4, "and occlusion removes the ones behind the wall");
        // The wall itself must survive: an occlusion test that culls the
        // occluder has its comparison backwards, and the give-away is that the
        // screen goes empty rather than that objects are missing.
        Check(with >= 1, "while the wall itself is still drawn");

        // AND IT IS THE PYRAMID DOING IT, not the flag. Turning occlusion on
        // without building a pyramid this frame must change nothing -- the cull
        // refuses a stale one, and a test that did not check this would pass
        // for an implementation that culled on some unrelated criterion.
        r->SetOcclusionCulling(true);
        dev->BeginFrame();
        {
            auto ce = dev->BeginCompute();
            (void)r->CullScene(ce, scene, kW, kH);
            dev->EndCompute();
        }
        if (!dev->CommitAndWait(error)) return 1;
        const int no_pyramid = r->VisibleAfterCull();
        std::printf("    occlusion on but no pyramid built: %d survive\n", no_pyramid);
        Check(no_pyramid == without,
              "occlusion without a current pyramid culls nothing");
        scene.instances.pop_back();
    }

    {
        std::printf("\nan ordinary mesh still works\n");
        // Every existing caller uploads one level, and the GPU-driven path has
        // to keep drawing it whatever the camera does.
        eng::Scene plain;
        plain.lightColor = eng::Vec4{3, 3, 3, 1};
        plain.camera.eye = eng::Vec3{0.0f, 4.0f, 26.0f};
        const eng::MeshHandle single = r->UploadMesh(sphere);
        for (int i = 0; i < 20; ++i) {
            eng::Instance inst;
            inst.mesh = single;
            inst.material = mat;
            inst.model = eng::Mat4::Translation(eng::Vec3{float(i % 5) - 2.0f, 0.0f,
                                                          float(i / 5) * -6.0f});
            plain.instances.push_back(inst);
        }
        Check(r->MeshLodCount(single) == 1, "it reports one level");
        eng::Scene saved = scene;
        scene = plain;
        if (!frame(false)) return 1;
        std::printf("    %d visible, %lld triangles at level %d\n",
                    r->VisibleAfterCull(), r->IndirectTriangles(), 0);
        Check(r->VisibleAfterCull() == 20, "all twenty are drawn");
        Check(r->VisibleAtLod(0) == 20, "and all of them at level 0");
        scene = saved;
    }

    std::printf(g_failures == 0 ? "\nlod_test: all checks passed\n"
                                : "\nlod_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
