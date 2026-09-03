// CLUSTERED LIGHTING, with the light count on a key and the cost on screen.
//
// The point of this one is the comparison, so everything is arranged to make it
// immediate: `c` switches clustering on and off, `[` and `]` change how many
// lights there are, and the HUD shows the frame time either way. Turn the
// lights up to two hundred with clustering off and the frame time climbs into
// the tens of milliseconds; press `c` and it drops back.
//
// The lights are DRAWN, as small emissive spheres, and that matters for
// believing what you are looking at. A light was an invisible point in this
// engine until emissive materials existed, so a scene of two hundred of them
// looked like a room with unexplained bright patches on the floor.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "engine/app/app.h"
#include "engine/geometry/mesh.h"
#include "engine/platform/font.h"
#include "engine/render/rendergraph.h"
#include "engine/shaders/shader_types.h"
#include "engine/text/ui.h"

namespace {

int Fail(const std::string& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.c_str());
    return 1;
}

// A deterministic scatter, so the scene is the same every run and two
// measurements of it are comparable.
float Hash(int i, int salt) {
    std::uint32_t h = std::uint32_t(i) * 0x9E3779B9u ^ std::uint32_t(salt) * 0x85EBCA6Bu;
    h ^= h >> 15;
    h *= 0x2545F491u;
    h ^= h >> 13;
    return float(h & 0xFFFFFFu) / float(0x1000000u);
}

// Eight light colours, cycled. One MATERIAL per colour rather than per light:
// the emissive value lives in the material, and two hundred materials that
// differ only in a uniform would be two hundred table entries for eight
// distinct looks.
constexpr int kColours = 8;
eng::Vec3 ColourOf(int i) {
    const eng::Vec3 table[kColours] = {
        {1.00f, 0.42f, 0.22f}, {0.30f, 0.70f, 1.00f}, {1.00f, 0.85f, 0.35f},
        {0.45f, 1.00f, 0.55f}, {1.00f, 0.35f, 0.62f}, {0.60f, 0.45f, 1.00f},
        {0.95f, 0.95f, 0.90f}, {0.25f, 1.00f, 0.85f}};
    return table[i % kColours];
}

constexpr int kMaxLights = ENG_MAX_LIGHTS;  // 256, the light buffer's size
constexpr float kRoom = 26.0f;

}  // namespace

int main(int argc, char** argv) {
    // Starting state on the command line, so a run can be compared against
    // another run without anyone having to press a key at the right moment.
    int start_lights = 128;
    bool start_clustered = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--brute") == 0) start_clustered = false;
        else if (std::strcmp(argv[i], "--lights") == 0 && i + 1 < argc)
            start_lights = std::atoi(argv[++i]);
    }

    std::string error;
    eng::app::Config config;
    config.title = "virtual — clustered lighting";
    auto app = eng::app::App::Create(config, error);
    if (!app) return Fail(error);

    const eng::platform::FontAtlas font =
        eng::platform::RasterizeFont("Menlo", 24.0f, error);
    if (!font.Valid()) return Fail(error.empty() ? "font" : error);
    auto ui = eng::ui::Canvas::Create(app->Gpu(), font, config.color, error, 1);
    if (!ui) return Fail(error);

    eng::Renderer& draw = app->Draw();

    // --- geometry ------------------------------------------------------------
    const eng::MeshHandle floor_mesh = draw.UploadMesh(
        eng::MakeBox(eng::Vec3{kRoom, 0.3f, kRoom}, eng::Vec4{1, 1, 1, 1}));
    const eng::MeshHandle pillar = draw.UploadMesh(
        eng::MakeBox(eng::Vec3{0.55f, 3.2f, 0.55f}, eng::Vec4{1, 1, 1, 1}));
    const eng::MeshHandle bulb = draw.UploadMesh(eng::MakeUVSphere(
        0.13f, 12, 16, eng::Vec4{1, 1, 1, 1}, eng::Vec4{1, 1, 1, 1}));

    eng::MaterialDesc surface;
    surface.base_color = eng::Vec4{0.62f, 0.60f, 0.58f, 1.0f};
    surface.roughness = 0.55f;
    const eng::MaterialHandle floor_mat = draw.CreateMaterial(surface, error);
    surface.base_color = eng::Vec4{0.48f, 0.50f, 0.54f, 1.0f};
    surface.roughness = 0.35f;
    surface.metallic = 0.25f;
    const eng::MaterialHandle pillar_mat = draw.CreateMaterial(surface, error);
    if (!eng::Valid(floor_mat) || !eng::Valid(pillar_mat)) return Fail(error);

    std::vector<eng::MaterialHandle> bulb_mats;
    for (int i = 0; i < kColours; ++i) {
        eng::MaterialDesc md;
        md.base_color = eng::Vec4{0, 0, 0, 1};
        // EMISSIVE, well above 1 so it also blooms. Below 1 a surface cannot
        // glow, because the bloom threshold is in the same linear units.
        const eng::Vec3 c = ColourOf(i);
        md.emissive = eng::Vec3{c.x * 14.0f, c.y * 14.0f, c.z * 14.0f};
        const eng::MaterialHandle m = draw.CreateMaterial(md, error);
        if (!eng::Valid(m)) return Fail(error);
        bulb_mats.push_back(m);
    }

    // --- the scene -----------------------------------------------------------
    eng::Scene scene;
    scene.camera.fovY = 1.0f;
    // NO SUN. Every photon in the room comes from a point light, so the effect
    // of adding and removing them is the whole picture rather than a detail on
    // top of a lit scene.
    scene.lightColor = eng::Vec4{0.0f, 0.0f, 0.0f, 1.0f};
    scene.ambientSky = eng::Vec3{0.012f, 0.013f, 0.018f};
    scene.ambientGround = eng::Vec3{0.006f, 0.006f, 0.008f};

    eng::Instance ground;
    ground.mesh = floor_mesh;
    ground.material = floor_mat;
    ground.model = eng::Mat4::Translation({0.0f, -0.3f, 0.0f});
    scene.instances.push_back(ground);

    // A forest of pillars: they give the room depth to cluster over, and they
    // occlude, so the lights behind them read as being somewhere.
    constexpr int kPillars = 120;
    for (int i = 0; i < kPillars; ++i) {
        eng::Instance in;
        in.mesh = pillar;
        in.material = pillar_mat;
        in.model = eng::Mat4::Translation(
            {(Hash(i, 3) - 0.5f) * 2.0f * (kRoom - 2.0f), 3.2f,
             (Hash(i, 4) - 0.5f) * 2.0f * (kRoom - 2.0f)});
        scene.instances.push_back(in);
    }
    const std::size_t static_instances = scene.instances.size();

    eng::OrbitController orbit;
    orbit.target = eng::Vec3{0.0f, 2.0f, 0.0f};
    orbit.distance = 34.0f;
    orbit.pitch = 0.42f;

    app->Actions().Bind("cluster", 'c');
    app->Actions().Bind("fewer", '[');
    app->Actions().Bind("more", ']');
    app->Actions().Bind("pause", ' ');
    app->Actions().Bind("bench", 'b');

    int light_count = std::clamp(start_lights, 1, kMaxLights);
    bool clustered = start_clustered;
    bool animate = true;
    draw.SetClusteredLighting(clustered);

    // A rolling average, because a single frame's GPU time jumps around by
    // tens of percent and the whole point here is comparing two numbers.
    constexpr int kWindow = 30;
    double history[kWindow] = {};
    int history_at = 0, history_filled = 0;
    // The last steady reading in each mode, kept so both can be on screen at
    // once -- flipping back and forth to remember the other number is a poor
    // way to compare them.
    double last_clustered_ms = 0.0, last_brute_ms = 0.0;
    int settle = 0;

    eng::RenderGraph graph;
    const eng::rhi::TextureId shadow_map = app->Gpu().CreateShadowMap(1024);
    float clock = 0.0f;

    while (app->Running()) {
        if (!app->BeginFrame()) continue;
        const eng::app::Frame& f = app->Current();

        // --- input -----------------------------------------------------------
        if (app->Actions().Pressed("cluster")) {
            clustered = !clustered;
            draw.SetClusteredLighting(clustered);
            settle = 0;
            history_filled = 0;
        }
        if (app->Actions().Pressed("pause")) animate = !animate;
        if (app->Actions().Pressed("fewer")) {
            light_count = std::max(1, light_count / 2);
            history_filled = 0;
        }
        if (app->Actions().Pressed("more")) {
            light_count = std::min(kMaxLights, light_count * 2);
            history_filled = 0;
        }
        // A one-key A/B: sit still and let both numbers settle.
        if (app->Actions().Pressed("bench")) {
            clustered = !clustered;
            draw.SetClusteredLighting(clustered);
            settle = 0;
            history_filled = 0;
        }
        orbit.Drag(f.drag_dx, f.drag_dy);
        orbit.Zoom(f.scroll * 1.2f);
        orbit.Apply(scene.camera);
        if (animate) clock += f.dt;

        // --- the lights ------------------------------------------------------
        scene.lights.clear();
        scene.instances.resize(static_instances);
        for (int i = 0; i < light_count; ++i) {
            // Slow orbits at different radii and speeds, so the set is always
            // moving and the clustering is never measured on a static bin.
            const float radius = 3.0f + Hash(i, 1) * (kRoom - 5.0f);
            const float speed = 0.08f + Hash(i, 2) * 0.22f;
            const float phase = Hash(i, 5) * 6.2831853f;
            const float a = phase + clock * speed;
            const eng::Vec3 p{std::cos(a) * radius, 0.6f + Hash(i, 6) * 3.4f,
                              std::sin(a) * radius};
            const eng::Vec3 colour = ColourOf(i);

            eng::Light l;
            l.type = eng::LightType::Point;
            l.position = p;
            l.color = eng::Vec3{colour.x * 6.0f, colour.y * 6.0f, colour.z * 6.0f};
            // A RANGE, and it is not decoration: clustering is a spatial index
            // and a light with no end belongs to every cell, so the whole idea
            // depends on a light having somewhere it stops mattering.
            l.range = 4.5f;
            scene.lights.push_back(l);

            eng::Instance in;
            in.mesh = bulb;
            in.material = bulb_mats[std::size_t(i % kColours)];
            in.model = eng::Mat4::Translation(p);
            scene.instances.push_back(in);
        }

        // --- the frame -------------------------------------------------------
        const eng::rhi::TextureId color = app->Targets().Hdr("color");
        const eng::rhi::TextureId ms_color = app->Targets().Msaa("ms");
        const eng::rhi::TextureId ms_depth = app->Targets().MsaaDepth("ms_depth");
        const eng::rhi::TextureId bloom_a = app->Targets().Hdr("bloomA", 2);
        const eng::rhi::TextureId bloom_b = app->Targets().Hdr("bloomB", 2);

        graph.Clear();
        {
            eng::RenderGraph::Pass p;
            p.name = "shadow";
            p.depth = shadow_map;
            p.keep_depth = true;
            p.execute = [&](eng::rhi::Encoder& e) { draw.DrawShadow(e, scene); };
            graph.AddPass(p);
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "scene";
            p.color = ms_color;
            p.resolve = color;
            p.depth = ms_depth;
            p.reads = {shadow_map};
            p.timer = "scene";
            p.execute = [&](eng::rhi::Encoder& e) {
                draw.DrawScene(e, scene, f.width, f.height, shadow_map);
            };
            graph.AddPass(p);
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "bright";
            p.color = bloom_a;
            p.reads = {color};
            p.execute = [&](eng::rhi::Encoder& e) {
                draw.DrawBloomBright(e, color, 2.6f, 0.8f);
            };
            graph.AddPass(p);
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "blur";
            p.color = bloom_b;
            p.reads = {bloom_a};
            p.execute = [&](eng::rhi::Encoder& e) {
                draw.DrawBloomBlur(e, bloom_a, 2.0f / float(f.width), 0.0f);
            };
            graph.AddPass(p);
        }

        // --- HUD ---------------------------------------------------------------
        const eng::Renderer::ClusterStats cs =
            clustered ? draw.ReadClusterStats() : eng::Renderer::ClusterStats{};
        const double gpu_ms = app->Gpu().LastFrameGpuMilliseconds();
        if (++settle > 8) {
            history[history_at] = gpu_ms;
            history_at = (history_at + 1) % kWindow;
            history_filled = std::min(history_filled + 1, kWindow);
        }
        double average = 0.0;
        for (int i = 0; i < history_filled; ++i) average += history[i];
        if (history_filled > 0) {
            average /= history_filled;
            if (history_filled >= kWindow) {
                if (clustered) last_clustered_ms = average;
                else last_brute_ms = average;
            }
        }

        ui->Begin(f.width, f.height);
        ui->Rect(12, 12, 560, 208, eng::Vec4{0.05f, 0.06f, 0.09f, 0.78f});
        ui->Outline(12, 12, 560, 208, 1.0f, eng::Vec4{0.34f, 0.40f, 0.50f, 0.9f});
        char line[224];
        const eng::Vec4 bright{0.93f, 0.95f, 0.99f, 1.0f};
        const eng::Vec4 dim{0.62f, 0.68f, 0.80f, 1.0f};

        std::snprintf(line, sizeof(line), "%d point lights   %s", light_count,
                      clustered ? "CLUSTERED" : "brute force (every light, every pixel)");
        ui->Text(26, 26, line,
                 clustered ? eng::Vec4{0.55f, 0.95f, 0.65f, 1.0f}
                           : eng::Vec4{1.0f, 0.65f, 0.45f, 1.0f});

        std::snprintf(line, sizeof(line), "gpu %.2f ms   (%.1f fps)", average,
                      average > 0.0 ? 1000.0 / average : 0.0);
        ui->Text(26, 52, line, bright);

        if (last_clustered_ms > 0.0 && last_brute_ms > 0.0)
            std::snprintf(line, sizeof(line),
                          "settled: clustered %.2f ms   brute %.2f ms   %.1fx",
                          last_clustered_ms, last_brute_ms,
                          last_brute_ms / last_clustered_ms);
        else
            std::snprintf(line, sizeof(line),
                          "press c and wait a second to fill in both numbers");
        ui->Text(26, 76, line, bright);

        if (clustered)
            std::snprintf(line, sizeof(line),
                          "bins: %d of %d cells, %.1f lights each, %d at most",
                          cs.occupied_cells, ENG_CLUSTER_COUNT,
                          cs.mean_per_occupied, cs.max_per_cell);
        else
            std::snprintf(line, sizeof(line),
                          "every fragment evaluates all %d lights", light_count);
        ui->Text(26, 100, line, dim);

        const eng::RenderStats& rs = draw.LastStats();
        std::snprintf(line, sizeof(line), "%d draws, %d culled, %d instances",
                      rs.draws, rs.culled, rs.submitted);
        ui->Text(26, 124, line, dim);

        ui->Text(26, 152, "c: clustering on/off      [ ]: halve / double lights",
                 dim);
        ui->Text(26, 176, "drag: orbit   scroll: zoom   space: pause the lights",
                 dim);

        // The same numbers to stdout once a second. A demo whose only output is
        // a window cannot be checked from a terminal, and this one exists to
        // report a measurement.
        static float report = 0.0f;
        report += f.dt;
        if (report > 1.0f) {
            report = 0.0f;
            std::printf("%3d lights  %-9s  gpu %6.2f ms  bins %4d cells %5.1f each"
                        "  %d draws\n",
                        light_count, clustered ? "clustered" : "brute",
                        average, cs.occupied_cells, cs.mean_per_occupied, rs.draws);
            std::fflush(stdout);
        }

        {
            eng::RenderGraph::Pass p;
            p.name = "composite";
            p.color = f.drawable;
            p.reads = {color, bloom_b};
            p.execute = [&](eng::rhi::Encoder& e) {
                draw.DrawComposite(e, color, {}, bloom_b, 0.6f, 0.9f);
                ui->Draw(e);
            };
            graph.AddPass(p);
        }

        if (!graph.Compile(error)) return Fail(error);
        // THE BINNING PASS, outside the graph and before it.
        //
        // The graph orders passes by the TEXTURES they read and write, and this
        // one writes buffers -- so it has nothing to hang a dependency on and
        // the graph would reject it for writing nothing. Encoding it first in
        // the command buffer is the ordering, which is what the graph would
        // have produced anyway.
        if (clustered) {
            auto e = app->Gpu().BeginCompute("bin");
            draw.BinLights(e, scene, f.width, f.height, 90.0f);
            app->Gpu().EndCompute();
        }
        graph.Execute(app->Gpu());
        app->EndFrame();
    }
    return 0;
}
