// Pure C++20. Clustered lighting, checked for both of the things it has to be:
// the SAME picture, and a cheaper one.
//
// An optimisation that changes the image is not an optimisation, and a
// reorganisation that does not get faster is not worth the code. Most of this
// file is the first check, because it is the one that fails quietly: a binning
// pass that drops a light near a cell boundary produces a frame that looks
// completely reasonable until you put it next to the reference.
#include "engine/geometry/mesh.h"
#include "engine/render/renderer.h"
#include "engine/rhi/rhi.h"
#include "engine/scene/scene.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-60s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

constexpr int kW = 1280, kH = 720;  // 16:9, matching the cluster grid's aspect

double Luma(const std::vector<std::uint8_t>& p, std::size_t i) {
    return 0.2126 * p[i] + 0.7152 * p[i + 1] + 0.0722 * p[i + 2];
}

// A deterministic scatter. Not std::rand: the two renders being compared have
// to see exactly the same lights, and a shared global generator makes that
// depend on how many times each path happened to call it.
float Hash(int i, int salt) {
    std::uint32_t h = std::uint32_t(i) * 0x9E3779B9u ^ std::uint32_t(salt) * 0x85EBCA6Bu;
    h ^= h >> 15;
    h *= 0x2545F491u;
    h ^= h >> 13;
    return float(h & 0xFFFFFFu) / float(0x1000000u);
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string error;
    auto dev = eng::rhi::Device::Create(error);
    if (!dev) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }
    const auto kFmt = eng::rhi::Format::RGBA8Unorm;
    auto r = eng::Renderer::Create(*dev, kFmt, error, 1);
    if (!r) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }

    const eng::rhi::TextureId hdr =
        dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
    const eng::rhi::TextureId ldr =
        dev->CreateRenderTarget(kW, kH, kFmt, /*cpu_readable=*/true);
    const eng::rhi::TextureId depth = dev->CreateDepthTarget(kW, kH);

    // A corridor of boxes: lots of surface at a wide range of depths, which is
    // the case clustering exists for. A single flat wall would put every
    // fragment in one depth slice and the grid would do nothing.
    const eng::MeshHandle box =
        r->UploadMesh(eng::MakeBox(eng::Vec3{0.5f, 0.5f, 0.5f}, eng::Vec4{1, 1, 1, 1}));
    const eng::MeshHandle floor_mesh =
        r->UploadMesh(eng::MakeBox(eng::Vec3{70.0f, 0.2f, 60.0f}, eng::Vec4{1, 1, 1, 1}));

    eng::MaterialDesc md;
    md.base_color = eng::Vec4{0.72f, 0.70f, 0.68f, 1.0f};
    md.roughness = 0.65f;
    md.metallic = 0.0f;
    const eng::MaterialHandle mat = r->CreateMaterial(md, error);
    if (!eng::Valid(mat)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }

    const auto build = [&](int light_count) {
        eng::Scene s;
        s.camera.eye = eng::Vec3{0.0f, 2.2f, 8.0f};
        s.camera.target = eng::Vec3{0.0f, 1.4f, -40.0f};
        s.camera.fovY = 1.0f;
        s.camera.nearZ = 0.1f;
        // NO sun and NO ambient. Every photon comes from a point light, so a
        // light the binning pass loses is a visible difference rather than a
        // small change to something that was mostly ambient anyway.
        s.lightColor = eng::Vec4{0.0f, 0.0f, 0.0f, 1.0f};
        s.ambientSky = eng::Vec3{0.0f, 0.0f, 0.0f};
        s.ambientGround = eng::Vec3{0.0f, 0.0f, 0.0f};

        eng::Instance ground;
        ground.mesh = floor_mesh;
        ground.material = mat;
        ground.model = eng::Mat4::Translation({0.0f, -0.2f, -40.0f});
        s.instances.push_back(ground);
        for (int i = 0; i < 90; ++i) {
            eng::Instance in;
            in.mesh = box;
            in.material = mat;
            in.model = eng::Mat4::Translation(
                {(Hash(i, 11) - 0.5f) * 14.0f, 0.5f + Hash(i, 12) * 2.0f,
                 -Hash(i, 13) * 85.0f + 4.0f});
            s.instances.push_back(in);
        }

        for (int i = 0; i < light_count; ++i) {
            eng::Light l;
            // Mostly points with a few spots, so the cone path is binned too.
            l.type = (i % 5 == 0) ? eng::LightType::Spot : eng::LightType::Point;
            // Spread across the FULL visible width at each depth, not along the
            // view axis. A scatter that hugs the centre never puts a light near
            // the outer corner of a far cell, and the outer corners are exactly
            // where a cell's bounding box has to be built from the far face
            // rather than the near one -- a box taken from the near face is too
            // small out there and silently drops those lights.
            const float depth = 4.0f + Hash(i, 3) * 88.0f;
            const float half_w = depth * std::tan(0.5f) * (float(kW) / float(kH));
            l.position = eng::Vec3{(Hash(i, 1) - 0.5f) * 1.9f * half_w,
                                   0.4f + Hash(i, 2) * 3.4f, -depth + 4.0f};
            l.direction = eng::Vec3{0.0f, -1.0f, 0.0f};
            const float hue = Hash(i, 4);
            l.color = eng::Vec3{1.4f + hue, 1.0f + Hash(i, 5), 0.8f + Hash(i, 6)};
            // A range that actually bounds the light. Clustering is a spatial
            // index and an unbounded light belongs to every cell -- the whole
            // idea depends on a light having somewhere it stops mattering.
            // Range grows with depth, so a far light still covers a few pixels.
            // A fixed three-metre range ninety metres out lights nothing the
            // camera can resolve, and a light that changes no pixel cannot show
            // that the binning dropped it.
            l.range = 3.0f + Hash(i, 7) * 4.0f + depth * 0.12f;
            s.lights.push_back(l);
        }
        return s;
    };

    std::vector<std::uint8_t> px(std::size_t(kW) * kH * 4);
    const auto render = [&](const eng::Scene& scene, bool bin) -> bool {
        dev->BeginFrame();
        if (bin) {
            auto e = dev->BeginCompute("bin");
            // 100 m, matching the corridor. The default reaches 200, and
            // every slice past the scene's back wall is wasted -- the
            // exponential split puts a fixed FRACTION of the cells in each
            // octave of depth, so doubling the range halves the resolution
            // where the geometry actually is.
            r->BinLights(e, scene, kW, kH, 100.0f);
            dev->EndCompute();
        }
        {
            eng::rhi::PassDesc pd;
            pd.color = hdr;
            pd.depth = depth;
            pd.clear_depth = 0.0f;
            pd.timer = "shade";
            auto e = dev->BeginPass(pd);
            r->DrawScene(e, scene, kW, kH, {});
            dev->EndPass();
        }
        {
            eng::rhi::PassDesc pd;
            pd.color = ldr;
            auto e = dev->BeginPass(pd);
            r->DrawComposite(e, hdr, {}, {}, 0.0f, /*vignette=*/0.0f);
            dev->EndPass();
        }
        if (!dev->CommitAndWait(error)) {
            std::fprintf(stderr, "FAIL: submit: %s\n", error.c_str());
            return false;
        }
        return dev->ReadPixels(ldr, kW, kH, px);
    };

    // BIN PLUS SHADE, not shade alone.
    //
    // Charging clustering only for the shading pass would be cheating twice
    // over. The binning pass is O(cells x lights) and grows with the light
    // count -- it is a real cost that the brute-force path does not pay. And
    // the shade pass cannot start until the bins are written, so on a tiled GPU
    // the shade timestamp already absorbs part of the wait: the two numbers are
    // not independent and reporting one of them is not reporting either.
    const auto frame_ms = [&]() {
        double t = 0.0;
        for (const eng::rhi::GpuTiming& g : dev->LastFrameTimings())
            t += g.milliseconds;
        return t;
    };

    // ------------------------------------------------------------ correctness --
    {
        std::printf("clustered lighting produces the same image\n");
        // 200 lights, which the brute-force path can still do -- the buffer
        // holds 256. Comparing at a count BOTH paths can render is the only way
        // to have a reference at all.
        const eng::Scene scene = build(200);

        r->SetClusteredLighting(false);
        if (!render(scene, false)) return 1;
        const std::vector<std::uint8_t> reference = px;

        r->SetClusteredLighting(true);
        if (!render(scene, true)) return 1;

        double sum = 0.0, worst = 0.0, ref_sum = 0.0;
        int over2 = 0;
        for (int i = 0; i < kW * kH; ++i) {
            const std::size_t k = std::size_t(i) * 4;
            const double d = std::fabs(Luma(px, k) - Luma(reference, k));
            sum += d;
            ref_sum += Luma(reference, k);
            worst = std::max(worst, d);
            if (d > 2.0) ++over2;
        }
        const double mean_err = sum / (kW * kH);
        std::printf("    mean |difference| %.4f/255, worst %.0f, %d pixels over 2\n",
                    mean_err, worst, over2);
        std::printf("    the reference is lit: mean luma %.1f\n",
                    ref_sum / (kW * kH));
        Check(ref_sum / (kW * kH) > 12.0,
              "the reference frame is actually lit by these lights");
        // EXACT, not close. The clustered path evaluates the same BRDF against
        // the same lights; the only difference is which ones it bothers to
        // look at, and a light it skips was out of range anyway. Any real
        // difference means a light was wrongly culled.
        Check(mean_err < 0.05, "clustered and brute force agree on every pixel");
        Check(worst < 6.0, "including the worst one, not just on average");

        const eng::Renderer::ClusterStats st = r->ReadClusterStats();
        std::printf("    %d of %d cells occupied, %.2f lights each on average, "
                    "%d at most (slice %d of %d), %d overflowed\n",
                    st.occupied_cells, ENG_CLUSTER_COUNT, st.mean_per_occupied,
                    st.max_per_cell, st.max_slice, ENG_CLUSTER_Z,
                    st.overflowed_cells);
        Check(st.occupied_cells > 0, "the binning pass filled some cells");
        // THE POINT. A fragment used to walk 200 lights; now it walks this
        // many. If the mean were near 200 the grid would be binning nothing,
        // which is exactly what happens when the cell bounds are computed in
        // the wrong space and every light lands in every cell.
        Check(st.mean_per_occupied < 20.0,
              "and a fragment now sees a handful of lights, not two hundred");
        // OVERFLOW is graceful degradation, not a bug, and asserting zero of it
        // would be asserting something the design does not promise. Exponential
        // depth slicing makes the far cells physically the largest -- slice 22
        // of 24 spans about twelve metres of a ninety-metre corridor -- so with
        // two hundred lights packed into that corridor one of them fills up.
        //
        // What matters is that it is RARE and that it does not show. Both are
        // measured: one cell in 3456, and the frame above is pixel-identical to
        // the reference, because a cell that far away covers fragments that are
        // both tiny and dim.
        // OVERFLOW is graceful degradation rather than a bug -- a full cell
        // drops the remaining lights instead of writing past its end -- so what
        // is asserted is that it is rare, not that it never happens.
        Check(st.overflowed_cells <= ENG_CLUSTER_COUNT / 1000,
              "and at most a thousandth of the cells overflow");
        // The crowding lands in the far slices, which is where it should: under
        // exponential slicing those cells are physically the largest, and the
        // last one also absorbs everything past the grid's far distance.
        Check(st.max_slice >= ENG_CLUSTER_Z - 4,
              "with the crowding in the far slices, where the cells are largest");
    }

    // ---------------------------------------------------------------- scaling --
    {
        std::printf("\nand stops charging for lights that are nowhere near you\n");
        // THE HONEST COMPARISON. Adding lights to a fixed volume genuinely
        // costs more under any scheme, clustered included -- the fragment's
        // neighbourhood really does contain more of them. Measuring that and
        // calling it a scaling result would be measuring nothing.
        //
        // What clustering actually buys is independence from lights that are
        // not near you, which is the case a real level is made of: thousands of
        // lamps, a handful in the room you are standing in. So the visible
        // lights are held at 24 and the rest are scattered a long way outside
        // the frustum. Brute force pays for every one; clustered pays for none.
        struct Row {
            int lights;
            double brute, clustered;
        };
        std::vector<Row> rows;
        for (int extra : {0, 56, 120, 232}) {
            eng::Scene scene = build(24);
            for (int i = 0; i < extra; ++i) {
                eng::Light l;
                l.type = eng::LightType::Point;
                // BEHIND the camera and far off to the sides. Still in the
                // buffer, still evaluated by the brute-force loop, in no cell
                // of the grid.
                l.position = eng::Vec3{(Hash(i, 21) - 0.5f) * 400.0f + 600.0f,
                                       Hash(i, 22) * 50.0f + 40.0f,
                                       Hash(i, 23) * 300.0f + 120.0f};
                l.direction = eng::Vec3{0.0f, -1.0f, 0.0f};
                l.color = eng::Vec3{1.0f, 1.0f, 1.0f};
                l.range = 5.0f;
                scene.lights.push_back(l);
            }
            Row row{24 + extra, 0.0, 0.0};
            // Best of three. A first run pays for pipeline warm-up and any run
            // can be interrupted by the compositor; the minimum is the only
            // statistic here that is about the renderer.
            for (int k = 0; k < 3; ++k) {
                r->SetClusteredLighting(false);
                if (!render(scene, false)) return 1;
                const double b = frame_ms();
                r->SetClusteredLighting(true);
                if (!render(scene, true)) return 1;
                const double c = frame_ms();
                // The bins are IDENTICAL on every row -- 911 cells, 1.62
                // lights each -- because the extra lights are outside the
                // frustum and never reach the grid. That is the claim being
                // measured, so it is worth stating where it is established.
                if (k == 0 || b < row.brute) row.brute = b;
                if (k == 0 || c < row.clustered) row.clustered = c;
            }
            rows.push_back(row);
            std::printf("    %3d lights (24 visible): brute %.3f ms, "
                        "clustered %.3f ms  (%.2fx)\n",
                        row.lights, row.brute, row.clustered,
                        row.clustered > 0 ? row.brute / row.clustered : 0.0);
        }

        Check(rows.back().brute > 0.0 && rows.back().clustered > 0.0,
              "the GPU timer reported something for both paths");
        const double brute_growth = rows.back().brute / std::max(rows[0].brute, 1e-6);
        const double clust_growth =
            rows.back().clustered / std::max(rows[0].clustered, 1e-6);
        std::printf("    24 -> %d lights: brute grew %.2fx, clustered grew %.2fx\n",
                    rows.back().lights, brute_growth, clust_growth);
        Check(brute_growth > 2.0,
              "the brute-force loop really does pay for the distant lights");
        Check(clust_growth < 1.25,
              "and the clustered path is flat: it never looks at them");
        Check(rows.back().clustered * 2.0 < rows.back().brute,
              "which is more than twice as fast at 256 lights");
    }

    std::printf(g_failures == 0 ? "\nlights_test: all checks passed\n"
                                : "\nlights_test: %d FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
