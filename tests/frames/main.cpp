// Pure C++20. Checks that the CPU actually pipelines ahead of the GPU, and
// that it is throttled to kFramesInFlight while doing so.
//
// Why this needs its own target: every other test calls CommitAndWait, which
// makes exactly one frame outstanding at a time. That path would still pass
// with a broken ring buffer, because a ring you never reuse is just a buffer.
// Here we submit without waiting, which is the only way slot reuse happens.
#include "engine/render/renderer.h"
#include "engine/rhi/rhi.h"
#include "engine/core/math.h"
#include "engine/geometry/mesh.h"
#include "engine/scene/scene.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <span>
#include <string>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-52s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

}  // namespace

int main() {
    constexpr int kW = 320, kH = 320;
    // DERIVED, not written down. This has to outrun the uniform ring: if the
    // per-frame cursor reset were deleted the allocator would starve before the
    // end and the final draws == 3 check would fail. It was 400 frames against
    // a ring of 1024, and the ring has been 8192 for some time -- so 1200
    // allocations against 8192 slices, and the guard had quietly stopped
    // guarding. Spelling out the relationship is what stops that happening
    // again the next time the ring changes size.
    constexpr int kDrawsPerFrame = 3;
    constexpr int kFrames =
        eng::Renderer::kMaxInstancesPerFrame / kDrawsPerFrame + 40;

    std::string error;
    auto dev = eng::rhi::Device::Create(error);
    if (!dev) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    auto renderer = eng::Renderer::Create(*dev, eng::rhi::Format::RGBA8Unorm, error);
    if (!renderer) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    eng::rhi::PassDesc pass;
    pass.color = dev->CreateRenderTarget(kW, kH, eng::rhi::Format::RGBA8Unorm, true);
    pass.depth = dev->CreateDepthTarget(kW, kH);
    if (!Valid(pass.color) || !Valid(pass.depth)) {
        std::fprintf(stderr, "FAIL: could not create targets\n");
        return 1;
    }
    for (int i = 0; i < 4; ++i) pass.clear_color[i] = eng::kClearColor[i];
    pass.clear_depth = 0.0f;

    // Submit without waiting. Each frame writes a DIFFERENT transform into its
    // ring slot, so a slot reused too early would show up as a stale pose.
    for (int f = 0; f < kFrames; ++f) {
        dev->BeginFrame();
        eng::rhi::Encoder enc = dev->BeginPass(pass);
        renderer->DrawScene(enc, eng::ShapesDemo(float(f) * 0.05f), kW, kH);
        dev->EndPass();
        dev->Commit();
    }

    const int peak = dev->PeakFramesInFlight();

    // One final frame we actually wait for, so we can look at the pixels.
    dev->BeginFrame();
    eng::rhi::Encoder enc = dev->BeginPass(pass);
    renderer->DrawScene(enc, eng::ShapesDemo(0.0f), kW, kH);
    dev->EndPass();
    if (!dev->CommitAndWait(error)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }

    std::vector<std::uint8_t> px(std::size_t(kW) * kH * 4);
    if (!dev->ReadPixels(pass.color, kW, kH, px)) {
        std::fprintf(stderr, "FAIL: readback\n");
        return 1;
    }
    std::set<std::uint32_t> distinct;
    std::size_t drawn = 0;
    const std::uint8_t bg[3] = {px[0], px[1], px[2]};
    for (std::size_t i = 0; i + 3 < px.size(); i += 4) {
        if (px[i] == bg[0] && px[i + 1] == bg[1] && px[i + 2] == bg[2]) continue;
        ++drawn;
        distinct.insert((std::uint32_t(px[i]) << 16) |
                        (std::uint32_t(px[i + 1]) << 8) | px[i + 2]);
    }

    std::printf("%d frames, peak in flight = %d (limit %d), final frame: %zu px, %zu colours\n",
                kFrames, peak, eng::rhi::kFramesInFlight, drawn, distinct.size());

    // --- pipeline cache ------------------------------------------------------
    // Three built-in materials: lit (Lit+depth), flat (Flat+no depth) and
    // two-sided (Lit+depth, cull None). The third differs from the first ONLY
    // in cull mode, which is encoder state rather than pipeline state — so the
    // cache must hand them the same pipeline and end up with two, not three.
    // Assert the INVARIANT, not a total. A count broke twice already — once
    // when the composite pipeline arrived and once when the shadow one did —
    // because it encoded how many shaders the engine happens to have rather
    // than what the cache actually promises.
    const eng::RenderStats& stats = renderer->LastStats();
    std::printf("  pipelines built = %d; lit=%u twoSided=%u flat=%u; %d draws, %d switches\n",
                renderer->PipelineCount(), renderer->PipelineOf(eng::kMaterialLit),
                renderer->PipelineOf(eng::kMaterialLitTwoSided),
                renderer->PipelineOf(eng::kMaterialFlat), stats.draws,
                stats.pipeline_switches);
    // Same shading and depth state, different cull mode. Cull is ENCODER state,
    // so these two must resolve to the very same pipeline object.
    Check(renderer->PipelineOf(eng::kMaterialLit) ==
              renderer->PipelineOf(eng::kMaterialLitTwoSided),
          "materials differing only in cull share one pipeline");
    // Different shading really does get its own.
    Check(renderer->PipelineOf(eng::kMaterialLit) !=
              renderer->PipelineOf(eng::kMaterialFlat),
          "materials with different shading get different pipelines");
    // There is deliberately NO assertion on PipelineCount() here. A bound on
    // the total broke three times running — once for the composite pipeline,
    // once for the shadow one, once for SSAO — because it encodes how many
    // shaders the engine happens to own rather than anything the cache
    // promises. The two checks above are the promise, and they do not care how
    // many pipelines get added later.
    // All three demo instances are lit, so after the first bind the pipeline
    // never changes again.
    Check(stats.draws == 3, "all three instances drew");
    Check(stats.pipeline_switches == 1, "redundant pipeline binds eliminated");

    // --- frustum culling -----------------------------------------------------
    // Same three visible objects plus 20 parked far off to the right, well
    // outside the 60-degree frustum. All 20 must be rejected on the CPU.
    // Every side plane in BOTH directions, plus behind the camera. The old
    // version only pushed objects out along +X, so five of the six planes were
    // never exercised in the rejecting direction and an inverted one would have
    // gone unnoticed.
    eng::Scene wide = eng::ShapesDemo(0.0f);
    const eng::Vec3 outside[6] = {
        {1000.0f, 0.0f, 0.0f},   // beyond right
        {-1000.0f, 0.0f, 0.0f},  // beyond left
        {0.0f, 1000.0f, 0.0f},   // beyond top
        {0.0f, -1000.0f, 0.0f},  // beyond bottom
        {0.0f, 0.0f, 1000.0f},   // behind the camera (eye is at +Z 4.5)
        {0.0f, 0.0f, -100000.0f}, // very far down -Z
    };
    for (const eng::Vec3& p : outside) {
        eng::Instance off;
        off.model = eng::Mat4::Translation(p);
        wide.instances.push_back(off);
    }
    dev->BeginFrame();
    eng::rhi::Encoder wenc = dev->BeginPass(pass);
    renderer->DrawScene(wenc, wide, kW, kH);
    dev->EndPass();
    const eng::RenderStats cull = renderer->LastStats();
    if (!dev->CommitAndWait(error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    std::printf("  culling: submitted=%d culled=%d draws=%d\n", cull.submitted,
                cull.culled, cull.draws);
    Check(cull.submitted == 9, "scene offered all 9 instances");
    // Note: -Z at 100000 is INSIDE the frustum — the far plane is at infinity
    // under reversed-Z, and its extracted plane is correctly degenerate. So 5
    // of the 6 outliers are culled, not 6.
    Check(cull.culled == 5, "every side plane and the near plane reject");
    Check(cull.draws == 4, "the 3 visible ones plus the infinitely-far one");

    // The inverse risk: a plane with a flipped sign would cull things it should
    // KEEP, and a test that only ever checks rejections cannot see that. Every
    // instance here is comfortably inside the frustum.
    {
        eng::Scene inside = eng::ShapesDemo(0.0f);
        for (int i = -2; i <= 2; ++i) {
            for (int j = -2; j <= 2; ++j) {
                eng::Instance in;
                in.model = eng::Mat4::Translation(
                    {float(i) * 0.35f, float(j) * 0.35f, 0.0f});
                in.model = in.model * eng::Mat4::Scale(0.12f);
                inside.instances.push_back(in);
            }
        }
        dev->BeginFrame();
        eng::rhi::Encoder ie = dev->BeginPass(pass);
        renderer->DrawScene(ie, inside, kW, kH);
        dev->EndPass();
        const eng::RenderStats st = renderer->LastStats();
        if (!dev->CommitAndWait(error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
        std::printf("  no-false-positives: submitted=%d culled=%d\n", st.submitted, st.culled);
        Check(st.culled == 0, "nothing inside the frustum is wrongly culled");
    }

    // MaxScale: a sphere whose CENTRE is outside the frustum but whose scaled
    // radius reaches in must survive. If MaxScale under-reported the scale (or
    // were ignored) this would be culled and the object would pop.
    {
        eng::Scene big;
        eng::Instance huge;
        huge.model = eng::Mat4::Translation({14.0f, 0.0f, 0.0f}) * eng::Mat4::Scale(13.0f);
        big.instances.push_back(huge);
        dev->BeginFrame();
        eng::rhi::Encoder be = dev->BeginPass(pass);
        renderer->DrawScene(be, big, kW, kH);
        dev->EndPass();
        const eng::RenderStats st = renderer->LastStats();
        if (!dev->CommitAndWait(error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
        std::printf("  scaled bounds: submitted=%d culled=%d\n", st.submitted, st.culled);
        Check(st.culled == 0, "a scaled radius reaching into view is kept (MaxScale)");
    }

    // --- allocator reset must key on the FRAME, not the ring slot -----------
    // FrameSlot() has only kFramesInFlight distinct values. An app that draws
    // on every Nth frame, for N a multiple of kFramesInFlight, sees the SAME
    // slot on every drawing call. Keying the bump-allocator reset on the slot
    // therefore never resets it, the cursor climbs to kMaxInstancesPerFrame,
    // and from then on DrawScene bails on its first instance — a scene that
    // goes permanently blank with draws == 0 and no error anywhere.
    //
    // 3 draws per call, so this needs more than kMaxInstancesPerFrame/3 drawing
    // frames to reach the cliff. The every-frame loop above cannot expose it,
    // because there the slot really does change every frame.
    {
        constexpr int kDrawingFrames =
            eng::Renderer::kMaxInstancesPerFrame / 3 + 20;
        eng::RenderStats last{};
        for (int f = 0; f < kDrawingFrames * eng::rhi::kFramesInFlight; ++f) {
            dev->BeginFrame();
            eng::rhi::Encoder e = dev->BeginPass(pass);
            if (f % eng::rhi::kFramesInFlight == 0) {
                renderer->DrawScene(e, eng::ShapesDemo(float(f) * 0.01f), kW, kH);
                last = renderer->LastStats();
            }
            dev->EndPass();
            dev->Commit();
        }
        std::printf("  every-%dth-frame cadence, %d drawing frames: last draws=%d\n",
                    eng::rhi::kFramesInFlight, kDrawingFrames, last.draws);
        Check(last.draws == 3, "uniform allocator survives a strided draw cadence");
    }

    // --- ring-slot reuse, actually observed --------------------------------
    // The 400-frame loop above never looks at a pixel, so on its own it cannot
    // tell a working ring from one where every frame writes slot 0. Here six
    // frames are submitted WITHOUT waiting into six different targets, each
    // with a different scene. Six frames over kFramesInFlight==3 slots means
    // every slot is reused exactly once. If reuse were unsafe, frames sharing a
    // slot would come out identical.
    {
        constexpr int kN = 6;
        eng::rhi::TextureId colors[kN];
        const eng::rhi::TextureId shared_depth = dev->CreateDepthTarget(kW, kH);
        for (int i = 0; i < kN; ++i)
            colors[i] = dev->CreateRenderTarget(kW, kH, eng::rhi::Format::RGBA8Unorm, true);

        for (int i = 0; i < kN; ++i) {
            dev->BeginFrame();
            eng::rhi::PassDesc pd;
            pd.color = colors[i];
            pd.depth = shared_depth;
            for (int c = 0; c < 4; ++c) pd.clear_color[c] = eng::kClearColor[c];
            pd.clear_depth = 0.0f;
            eng::rhi::Encoder e = dev->BeginPass(pd);
            renderer->DrawScene(e, eng::ShapesDemo(float(i) * 0.9f), kW, kH);
            dev->EndPass();
            if (i == kN - 1) {
                if (!dev->CommitAndWait(error)) {
                    std::fprintf(stderr, "FAIL: %s\n", error.c_str());
                    return 1;
                }
            } else {
                dev->Commit();  // pipelined on purpose
            }
        }

        std::vector<std::vector<std::uint8_t>> imgs(kN);
        for (int i = 0; i < kN; ++i) {
            imgs[i].resize(std::size_t(kW) * kH * 4);
            if (!dev->ReadPixels(colors[i], kW, kH, imgs[i])) {
                std::fprintf(stderr, "FAIL: readback %d\n", i);
                return 1;
            }
        }
        // Frames 0 and 3, 1 and 4, 2 and 5 share a ring slot. Those are the
        // pairs that would collide.
        int identical_pairs = 0;
        for (int i = 0; i + eng::rhi::kFramesInFlight < kN; ++i)
            if (imgs[i] == imgs[i + eng::rhi::kFramesInFlight]) ++identical_pairs;
        std::printf("  slot-sharing pairs that came out identical: %d of %d\n",
                    identical_pairs, kN - eng::rhi::kFramesInFlight);
        Check(identical_pairs == 0, "pipelined frames sharing a ring slot stay distinct");
    }

    // --- two DrawScene calls in ONE frame ------------------------------------
    // The ring is sub-allocated with a bump pointer that resets when the frame
    // slot changes, NOT at the top of each DrawScene. If it reset per call, the
    // second pass would rewrite the same offsets the first pass had already
    // encoded draws against, and both passes would end up showing whichever
    // scene was written last — identical images.
    //
    // (DrawScene + DrawTriangle cannot be combined here: the flat pipeline is
    // built without a depth attachment and the lit one with, so they cannot
    // share a render pass at all.)
    {
        const eng::rhi::TextureId a = dev->CreateRenderTarget(kW, kH, eng::rhi::Format::RGBA8Unorm, true);
        const eng::rhi::TextureId b = dev->CreateRenderTarget(kW, kH, eng::rhi::Format::RGBA8Unorm, true);
        const eng::rhi::TextureId d2 = dev->CreateDepthTarget(kW, kH);

        dev->BeginFrame();
        for (int which = 0; which < 2; ++which) {
            eng::rhi::PassDesc pd;
            pd.color = which ? b : a;
            pd.depth = d2;
            for (int i = 0; i < 4; ++i) pd.clear_color[i] = eng::kClearColor[i];
            pd.clear_depth = 0.0f;
            eng::rhi::Encoder e = dev->BeginPass(pd);
            // Deliberately DIFFERENT scenes, so sharing uniforms is visible.
            renderer->DrawScene(e, eng::ShapesDemo(which ? 2.0f : 0.0f), kW, kH);
            dev->EndPass();
        }
        if (!dev->CommitAndWait(error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

        std::vector<std::uint8_t> pa(std::size_t(kW) * kH * 4), pb(pa.size());
        if (!dev->ReadPixels(a, kW, kH, pa) || !dev->ReadPixels(b, kW, kH, pb)) {
            std::fprintf(stderr, "FAIL: readback\n");
            return 1;
        }
        std::size_t diff = 0;
        for (std::size_t i = 0; i < pa.size(); ++i)
            if (pa[i] != pb[i]) ++diff;
        std::printf("  two passes, one frame: %zu of %zu bytes differ\n", diff, pa.size());
        Check(diff > pa.size() / 100,
              "two DrawScene calls in one frame keep separate uniforms");
    }

    // --- texture slot recycling ----------------------------------------------
    // DestroyTexture must return the slot to a free list. Without it the handle
    // table grows by one entry per window resize, forever.
    {
        const int before = dev->TextureSlotCount();
        for (int i = 0; i < 500; ++i) {
            const eng::rhi::TextureId t = dev->CreateDepthTarget(64, 64);
            if (!Valid(t)) { std::fprintf(stderr, "FAIL: depth target\n"); return 1; }
            dev->DestroyTexture(t);
        }
        const int after = dev->TextureSlotCount();
        std::printf("  texture slots: %d -> %d after 500 create/destroy cycles\n",
                    before, after);
        Check(after <= before + 1, "destroyed texture slots are recycled");
    }

    // --- draw ordering is actually observed ----------------------------------
    // Without LastDrawOrder both sort keys could be deleted and every other
    // test would still pass: the depth buffer makes the image correct whatever
    // order the draws go in.
    //
    // Needs a SECOND depth-compatible pipeline. kMaterialFlat is depth_test =
    // false, so in a depth pass it is skipped as incompatible — hence a new
    // Flat+depth material here.
    {
        const eng::MaterialHandle flat_depth = renderer->CreateMaterial(
            {eng::Shading::Flat, true, eng::rhi::Cull::Back}, error);
        if (!Valid(flat_depth)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

        eng::Scene s;
        const eng::MaterialHandle mats[4] = {eng::kMaterialLit, flat_depth,
                                             eng::kMaterialLit, flat_depth};
        const eng::Vec3 pos[4] = {{-0.9f, 0, -2.0f}, {-0.3f, 0, 0.0f},
                                  {0.3f, 0, 1.0f},   {0.9f, 0, -1.0f}};
        for (int i = 0; i < 4; ++i) {
            eng::Instance in;
            in.material = mats[i];
            in.model = eng::Mat4::Translation(pos[i]) * eng::Mat4::Scale(0.3f);
            s.instances.push_back(in);
        }

        dev->BeginFrame();
        eng::rhi::Encoder oe = dev->BeginPass(pass);
        renderer->DrawScene(oe, s, kW, kH);
        dev->EndPass();
        const std::vector<int> order = renderer->LastDrawOrder();
        const eng::RenderStats st = renderer->LastStats();
        if (!dev->CommitAndWait(error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

        std::printf("  draw order:");
        for (int i : order) std::printf(" %d", i);
        std::printf("  (draws=%d)\n", st.draws);

        Check(order.size() == 4, "all four instances submitted");

        // Property 1: same material is contiguous — that is the pipeline key.
        bool grouped = true;
        for (std::size_t i = 2; i < order.size(); ++i)
            if (mats[order[i]] == mats[order[i - 2]] &&
                !(mats[order[i]] == mats[order[i - 1]]))
                grouped = false;
        Check(grouped, "draws are grouped by pipeline");

        // Property 2: inside each group, nearest first — the depth key.
        auto dist = [&](int i) {
            const eng::Vec3 d = pos[i] - s.camera.eye;
            return eng::Length(d);
        };
        bool front_to_back = true;
        for (std::size_t i = 1; i < order.size(); ++i)
            if (mats[order[i]] == mats[order[i - 1]] &&
                dist(order[i]) < dist(order[i - 1]))
                front_to_back = false;
        Check(front_to_back, "within a pipeline group, nearest draws first");
    }

    // --- instances the renderer refuses are counted, not silently dropped ----
    {
        eng::Scene s;
        eng::Instance bad_mesh;
        bad_mesh.mesh = eng::MeshHandle{9999};
        eng::Instance null_mat;
        null_mat.material = eng::MaterialHandle{0};
        eng::Instance wrong_depth;
        wrong_depth.material = eng::kMaterialFlat;  // depth_test = false
        eng::Instance good;
        good.model = eng::Mat4::Scale(0.3f);
        s.instances = {bad_mesh, null_mat, wrong_depth, good};

        dev->BeginFrame();
        eng::rhi::Encoder be2 = dev->BeginPass(pass);  // this pass HAS depth
        renderer->DrawScene(be2, s, kW, kH);
        dev->EndPass();
        const eng::RenderStats st = renderer->LastStats();
        if (!dev->CommitAndWait(error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

        std::printf("  rejects: submitted=%d invalid=%d incompatible=%d draws=%d\n",
                    st.submitted, st.invalid, st.incompatible, st.draws);
        Check(st.invalid == 2, "bad mesh and null material counted as invalid");
        Check(st.incompatible == 1, "depth-less material in a depth pass counted");
        Check(st.draws == 1, "only the valid instance drew");
    }

    // --- the multisample depth resolve ---------------------------------------
    //
    // PassDesc had a colour resolve and no depth one, so a multisample depth
    // attachment could not be turned into anything sampleable and apps/world
    // rendered the whole scene a SECOND time to manufacture one. The filter is
    // MAX because this engine is reversed-Z, where the larger value is the
    // nearer surface.
    //
    // DIFFERENTIAL, because max only differs from min at a silhouette: where
    // every sample of a texel is covered the two agree, and asserting a depth
    // value alone would pass with either. So the same tilted cube goes into a
    // 1x target and a 4x-plus-resolve one, and the resolve must keep MORE
    // texels -- a texel with one covered sample of four holds a real surface
    // under max and the cleared far plane under min.
    {
        constexpr int kD = 64;
        const eng::rhi::TextureId one = dev->CreateDepthTarget(
            kD, kD, /*sampleable=*/true, /*samples=*/1, /*cpu_readable=*/true);
        const eng::rhi::TextureId ms =
            dev->CreateDepthTarget(kD, kD, /*sampleable=*/false, /*samples=*/4);
        const eng::rhi::TextureId resolved = dev->CreateDepthTarget(
            kD, kD, /*sampleable=*/true, /*samples=*/1, /*cpu_readable=*/true);
        eng::MaterialDesc md;
        md.shading = eng::Shading::Flat;
        const eng::MaterialHandle mat = renderer->CreateMaterial(md, error);
        const eng::MeshHandle box = renderer->UploadMesh(
            eng::MakeCube(1.0f, eng::Vec4{1, 1, 1, 1}, eng::Vec4{1, 1, 1, 1}));

        // Orthographic down -Z, so a world z maps onto a depth by arithmetic
        // rather than by guesswork, and TILTED so its silhouette cuts across
        // texels instead of landing on their boundaries.
        eng::Scene s2;
        s2.camera.projection = eng::Projection::Orthographic;
        s2.camera.orthoHeight = 2.0f;
        s2.camera.orthoFar = 21.0f;
        s2.camera.nearZ = 1.0f;
        s2.camera.eye = eng::Vec3{0.0f, 0.0f, 10.0f};
        s2.camera.target = eng::Vec3{0.0f, 0.0f, 0.0f};
        // FAR ONE FIRST, so a depth test with the comparison backwards would
        // leave the far depth behind and fail the value check below.
        for (float z : {-4.0f, 4.0f}) {
            eng::Instance q;
            q.mesh = box;
            q.material = mat;
            q.model = eng::Mat4::Translation(eng::Vec3{0.0f, 0.0f, z}) *
                      eng::Mat4::RotationY(0.35f) * eng::Mat4::RotationX(0.35f);
            s2.instances.push_back(q);
        }

        const auto count = [&](eng::rhi::TextureId tex, float& value) {
            std::vector<std::uint8_t> raw(std::size_t(kD) * kD * 4);
            if (!dev->ReadPixels(tex, kD, kD, raw)) return -1;
            const auto* d = reinterpret_cast<const float*>(raw.data());
            int n = 0;
            value = 0.0f;
            for (int i = 0; i < kD * kD; ++i)
                if (d[i] > 0.0f) { ++n; value = std::max(value, d[i]); }
            return n;
        };

        eng::rhi::PassDesc single;
        single.depth = one;
        single.keep_depth = true;
        eng::rhi::PassDesc multi;
        multi.depth = ms;
        multi.depth_resolve = resolved;

        dev->BeginFrame();
        eng::rhi::Encoder e1 = dev->BeginPass(single);
        renderer->DrawSceneDepth(e1, s2, kD, kD);
        dev->EndPass();
        eng::rhi::Encoder e4 = dev->BeginPass(multi);
        renderer->DrawSceneDepth(e4, s2, kD, kD);
        dev->EndPass();
        if (!dev->CommitAndWait(error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

        float near1 = 0.0f, near4 = 0.0f;
        const int n1 = count(one, near1);
        const int n4 = count(resolved, near4);
        // The near cube's front face sits at world z = 4 + 0.5*sqrt(3) once the
        // two 0.35 rad rotations have turned it, so pin the value loosely and
        // let the DIFFERENCE between the two targets carry the precision.
        std::printf("  depth resolve: 1x covers %d texels (nearest %.4f), "
                    "4x+resolve covers %d (nearest %.4f)\n",
                    n1, near1, n4, near4);
        Check(n1 > 0 && n4 > 0, "both depth targets read back");
        Check(near4 > 0.7f && near4 < 0.85f,
              "the resolved depth is the NEAR cube's, not the far one's");
        Check(std::fabs(near4 - near1) < 0.01f,
              "and it agrees with the single-sampled render in the interior");
        Check(n4 > n1,
              "the resolve keeps partly-covered texels, so the filter is max");
    }

    // The throttle works: BeginFrame blocked once the ring was full.
    Check(peak <= eng::rhi::kFramesInFlight, "never exceeds kFramesInFlight");
    // And it is a real pipeline, not an accidental full stall on every frame —
    // without this the test would pass on a completely serialised renderer.
    Check(peak > 1, "CPU really does run ahead of the GPU (peak > 1)");
    Check(drawn > 0 && distinct.size() > 100, "frames still render correctly");

    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
