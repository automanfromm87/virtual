// IMAGE-BASED LIGHTING, with a sun you can move.
//
// The grid is the point. Roughness runs left to right and metalness runs front
// to back, so every sphere differs from its neighbours in exactly one property
// and the effect of each is separable by eye. Before there was an environment
// this grid was almost featureless: the metal row went black, because a metal
// has no diffuse lobe and there was nothing for it to reflect, and roughness
// changed only the size of a single highlight.
//
// Dragging the sun is what shows the second half. The sky, the key light's
// colour, the ambient and every reflection all come from ONE parameter, so they
// cannot disagree -- which is the failure this replaces, where a scene had a
// warm sunset sky and a white key light and read as two times of day at once.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/app/app.h"
#include "engine/geometry/mesh.h"
#include "engine/platform/font.h"
#include "engine/render/ibl.h"
#include "engine/render/rendergraph.h"
#include "engine/text/ui.h"

namespace {

int Fail(const std::string& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.c_str());
    return 1;
}

constexpr int kCols = 7;  // roughness
constexpr int kRows = 3;  // metalness

}  // namespace

int main() {
    std::string error;
    eng::app::Config config;
    config.title = "virtual — sky and image-based lighting";
    auto app = eng::app::App::Create(config, error);
    if (!app) return Fail(error);

    const eng::platform::FontAtlas font =
        eng::platform::RasterizeFont("Menlo", 24.0f, error);
    if (!font.Valid()) return Fail(error.empty() ? "font" : error);
    auto ui = eng::ui::Canvas::Create(app->Gpu(), font, config.color, error, 1);
    if (!ui) return Fail(error);

    auto env = eng::Environment::Create(app->Gpu(), error, 256, config.color,
                                        config.samples);
    if (!env) return Fail(error);

    const eng::MeshHandle ball = app->Draw().UploadMesh(
        eng::MakeUVSphere(0.5f, 32, 48, eng::Vec4{1, 1, 1, 1}, eng::Vec4{1, 1, 1, 1}));
    const eng::MeshHandle slab = app->Draw().UploadMesh(
        eng::MakeBox(eng::Vec3{9.0f, 0.25f, 5.0f}, eng::Vec4{1, 1, 1, 1}));

    // ONE MATERIAL PER CELL, because roughness and metalness live in the
    // material rather than per instance. Twenty-one pipelines sounds wasteful
    // and is not: they differ only in their uniform values, so the renderer's
    // pipeline cache hands back the same compiled program for all of them.
    std::vector<eng::MaterialHandle> cells;
    for (int r = 0; r < kRows; ++r)
        for (int c = 0; c < kCols; ++c) {
            eng::MaterialDesc md;
            md.shading = eng::Shading::Lit;
            md.roughness = 0.04f + 0.96f * float(c) / float(kCols - 1);
            md.metallic = float(r) / float(kRows - 1);
            // A copper-ish base, so the metal rows are recognisably metal and
            // not just grey mirrors. A dielectric ignores base colour in its
            // specular and shows it in the diffuse; a metal does the opposite,
            // and having one colour makes that contrast visible down a column.
            md.base_color = eng::Vec4{0.94f, 0.72f, 0.52f, 1.0f};
            cells.push_back(app->Draw().CreateMaterial(md, error));
        }
    eng::MaterialDesc floor_md;
    floor_md.shading = eng::Shading::Lit;
    floor_md.roughness = 0.35f;
    floor_md.metallic = 0.0f;
    floor_md.base_color = eng::Vec4{0.22f, 0.22f, 0.24f, 1.0f};
    const eng::MaterialHandle floor_mat = app->Draw().CreateMaterial(floor_md, error);
    const eng::rhi::TextureId shadow_map = app->Gpu().CreateShadowMap(2048);
    if (!Valid(ball) || !Valid(floor_mat) || !Valid(shadow_map))
        return Fail("scene build failed");

    eng::Scene scene;
    scene.shadowExtent = 12.0f;
    scene.shadowCascades = 3;
    scene.shadowDistance = 40.0f;

    for (int r = 0; r < kRows; ++r)
        for (int c = 0; c < kCols; ++c) {
            eng::Instance inst;
            inst.mesh = ball;
            inst.material = cells[std::size_t(r * kCols + c)];
            inst.model = eng::Mat4::Translation(
                eng::Vec3{(float(c) - float(kCols - 1) * 0.5f) * 1.35f, 0.62f,
                          (float(r) - float(kRows - 1) * 0.5f) * 1.35f});
            scene.instances.push_back(inst);
        }
    {
        eng::Instance inst;
        inst.mesh = slab;
        inst.material = floor_mat;
        inst.model = eng::Mat4::Translation(eng::Vec3{0.0f, 0.12f, 0.0f});
        scene.instances.push_back(inst);
    }

    eng::SkyConfig sky;
    // Mid-morning: high enough for a blue sky, low enough that the spheres cast
    // shadows you can see.
    float sun_azimuth = 0.9f, sun_elevation = 0.62f;
    float turbidity = 2.6f;
    bool auto_sun = false;
    bool ibl_on = true;

    eng::OrbitController orbit;
    orbit.distance = 11.0f;
    orbit.yaw = 0.5f;
    orbit.pitch = 0.28f;
    orbit.target = eng::Vec3{0.0f, 0.7f, 0.0f};

    app->Actions().BindMouse("sun", eng::app::MouseButton::Right);
    app->Actions().Bind("auto", 'g');
    app->Actions().Bind("ibl", 'i');
    app->Actions().Bind("hazy", 'h');
    app->Actions().Bind("clear", 'c');

    // A bake is only needed when the sun has actually moved. Tracking the last
    // baked state rather than baking every frame is the whole reason this is
    // affordable: the chain is a few milliseconds, which is a third of a frame.
    float baked_az = 1e9f, baked_el = 1e9f, baked_turb = 1e9f;
    int bakes = 0;

    eng::RenderGraph graph;
    while (app->Running()) {
        if (!app->BeginFrame()) continue;
        const eng::app::Frame& f = app->Current();

        if (app->Actions().Pressed("auto")) auto_sun = !auto_sun;
        if (app->Actions().Pressed("ibl")) ibl_on = !ibl_on;
        if (app->Actions().Pressed("hazy")) turbidity = std::min(turbidity + 1.5f, 14.0f);
        if (app->Actions().Pressed("clear")) turbidity = std::max(turbidity - 1.5f, 1.2f);
        if (auto_sun) sun_elevation = 0.55f + 0.75f * std::sin(f.time * 0.25f);

        // Right-drag moves the sun; left-drag orbits the camera. Two gestures
        // rather than a modifier, because the whole demo is about comparing the
        // two and swapping between them must not need a keyboard.
        if (app->Actions().Down("sun")) {
            sun_azimuth += f.mouse_dx * 0.006f;
            sun_elevation = std::clamp(sun_elevation - f.mouse_dy * 0.004f, -0.12f, 1.55f);
        } else {
            orbit.Drag(f.drag_dx, f.drag_dy);
        }
        orbit.Zoom(f.scroll * 0.5f);
        orbit.Apply(scene.camera);

        sky.sun_direction = eng::Vec3{std::cos(sun_azimuth) * std::cos(sun_elevation),
                                      std::sin(sun_elevation),
                                      std::sin(sun_azimuth) * std::cos(sun_elevation)};
        sky.turbidity = turbidity;
        eng::Environment::ApplyTo(&scene, sky);

        const bool needs_bake = std::fabs(sun_azimuth - baked_az) > 0.004f ||
                                std::fabs(sun_elevation - baked_el) > 0.004f ||
                                std::fabs(turbidity - baked_turb) > 0.01f;
        if (needs_bake) {
            baked_az = sun_azimuth;
            baked_el = sun_elevation;
            baked_turb = turbidity;
            ++bakes;
        }
        if (ibl_on) {
            app->Draw().SetEnvironment(env->Bindings());
        } else {
            app->Draw().ClearEnvironment();
        }

        // --- HUD ---------------------------------------------------------------
        ui->Begin(f.width, f.height);
        ui->Rect(12, 12, 430, 132, eng::Vec4{0.05f, 0.06f, 0.08f, 0.72f});
        ui->Outline(12, 12, 430, 132, 1.0f, eng::Vec4{0.35f, 0.40f, 0.48f, 0.9f});
        char line[192];
        std::snprintf(line, sizeof(line), "sun  %+.0f deg   turbidity %.1f",
                      sun_elevation * 57.2958f, turbidity);
        ui->Text(26, 26, line, eng::Vec4{0.92f, 0.94f, 0.98f, 1.0f});
        const eng::Vec3 sc = eng::Environment::SunColor(sky);
        std::snprintf(line, sizeof(line), "sun colour %.2f %.2f %.2f", sc.x, sc.y, sc.z);
        ui->Text(26, 50, line, eng::Vec4{sc.x / 22.0f + 0.3f, sc.y / 22.0f + 0.3f,
                                         sc.z / 22.0f + 0.3f, 1.0f});
        std::snprintf(line, sizeof(line), "IBL %s   bakes %d   gpu %.2f ms",
                      ibl_on ? "on" : "OFF (hemisphere fallback)", bakes,
                      app->Gpu().LastFrameGpuMilliseconds());
        ui->Text(26, 74, line, eng::Vec4{0.80f, 0.86f, 0.94f, 1.0f});
        ui->Text(26, 98, "drag: orbit   right-drag: sun   i: IBL   g: animate",
                 eng::Vec4{0.62f, 0.68f, 0.78f, 1.0f});
        ui->Text(26, 120, "rows: metalness 0 -> 1     columns: roughness 0 -> 1",
                 eng::Vec4{0.62f, 0.68f, 0.78f, 1.0f});

        // --- passes ------------------------------------------------------------
        const eng::rhi::TextureId color = app->Targets().Hdr("color");
        const eng::rhi::TextureId ms_color = app->Targets().Msaa("ms");
        const eng::rhi::TextureId ms_depth = app->Targets().MsaaDepth("ms_depth");
        graph.Clear();

        // The bake is a COMPUTE pass and it goes first, because everything
        // after it samples what it writes.
        if (needs_bake) {
            graph.AddCompute("sky", {env->Radiance(), env->Bindings().irradiance,
                                     env->Bindings().specular,
                                     env->Bindings().brdf_lut},
                             [&](eng::rhi::ComputeEncoder& e) { env->BakeSky(e, sky); },
                             "sky bake");
        }

        {
            eng::RenderGraph::Pass p;
            p.name = "shadow";
            p.depth = shadow_map;
            p.keep_depth = true;
            p.timer = "shadow";
            p.execute = [&](eng::rhi::Encoder& e) { app->Draw().DrawShadow(e, scene); };
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
                app->Draw().DrawScene(e, scene, f.width, f.height, shadow_map);
                // The sky LAST among the opaque work, and depth-tested. It
                // fills only the pixels nothing covered, so the cost is the
                // background and not the whole screen -- and drawing it first
                // would shade every pixel twice.
                env->DrawSky(e, scene.camera, f.width, f.height);
            };
            graph.AddPass(p);
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "composite";
            p.color = f.drawable;
            p.reads = {color};
            p.timer = "composite";
            p.execute = [&](eng::rhi::Encoder& e) {
                app->Draw().DrawComposite(e, color);
                ui->Draw(e);
            };
            graph.AddPass(p);
        }
        if (!graph.Compile(error)) return Fail(error);
        graph.Execute(app->Gpu());
        app->EndFrame();
    }
    return 0;
}
