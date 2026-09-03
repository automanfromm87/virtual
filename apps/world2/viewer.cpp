// EVERYTHING AT ONCE: terrain, a navmesh, a character that paths across it, an
// image-based sky, and the post-processing stack.
//
// The other demos each show one thing. This one exists because the interesting
// failures are between systems, not inside them -- the navmesh disagreeing with
// the collider about where the ground is, the terrain's chunks popping as they
// change level, the fog and the sky disagreeing about what colour the distance
// should be. None of those is visible in a demo that only has one of the two
// systems involved.
//
// Click anywhere to send the character there. It walks the path the navmesh
// found, over terrain the physics collider agrees with, lit by a sky you can
// drag around.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/app/app.h"
#include "engine/geometry/mesh.h"
#include "engine/geometry/simplify.h"
#include "engine/geometry/terrain.h"
#include "engine/nav/navmesh.h"
#include "engine/physics/character.h"
#include "engine/physics/physics.h"
#include "engine/platform/font.h"
#include "engine/render/ibl.h"
#include "engine/render/post.h"
#include "engine/render/rendergraph.h"
#include "engine/text/ui.h"

namespace {

int Fail(const std::string& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.c_str());
    return 1;
}

// Ridged noise, summed over a few octaves. Value noise rather than Perlin --
// the difference is invisible at this scale and this is twenty lines.
float Hash(int x, int z) {
    std::uint32_t h = std::uint32_t(x) * 374761393u + std::uint32_t(z) * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return float((h ^ (h >> 16)) & 0xFFFFFFu) / 16777215.0f;
}

float ValueNoise(float x, float z) {
    const int ix = int(std::floor(x)), iz = int(std::floor(z));
    const float fx = x - float(ix), fz = z - float(iz);
    // Smoothstep on the interpolant, so the derivative is continuous across
    // cell boundaries. Linear interpolation leaves a visible crease on every
    // grid line, which at four octaves becomes a plaid pattern.
    const float sx = fx * fx * (3.0f - 2.0f * fx);
    const float sz = fz * fz * (3.0f - 2.0f * fz);
    const float a = Hash(ix, iz), b = Hash(ix + 1, iz);
    const float c = Hash(ix, iz + 1), d = Hash(ix + 1, iz + 1);
    return (a + (b - a) * sx) + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sz;
}

float Landscape(float x, float z) {
    float height = 0.0f, amplitude = 6.0f, frequency = 0.012f;
    for (int octave = 0; octave < 4; ++octave) {
        height += (ValueNoise(x * frequency, z * frequency) - 0.5f) * amplitude;
        amplitude *= 0.5f;
        frequency *= 2.1f;
    }
    // A bowl, so the playable area is enclosed and the character cannot walk
    // off the edge of the world -- which is a level design job that a demo has
    // to do for itself.
    const float r = std::sqrt(x * x + z * z);
    height += std::max(0.0f, (r - 46.0f) * 0.55f);
    return height;
}

struct Chunk {
    std::size_t instance = 0;
    int cx = 0, cz = 0;
    eng::MeshHandle lods[4];
    int lod_count = 0;
};

}  // namespace

int main() {
    // LINE BUFFERED. Redirected to a file, stdout is fully buffered, and a
    // windowed app that is quit rather than returning never flushes -- so the
    // load-time diagnostics below simply never appear.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string error;
    eng::app::Config config;
    config.title = "virtual — terrain, navigation, sky";
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
    auto post = eng::PostStack::Create(app->Gpu(), error);
    if (!post) return Fail(error);

    // --- the terrain ---------------------------------------------------------
    std::printf("generating terrain...\n");
    eng::TerrainConfig tc;
    tc.resolution = 257;
    tc.world_size = 128.0f;
    tc.origin = eng::Vec3{-64.0f, 0.0f, -64.0f};
    tc.chunk_resolution = 33;
    tc.skirt_depth = 3.0f;
    const eng::Terrain terrain = eng::Terrain::Generate(tc, Landscape);
    if (!terrain.Valid()) return Fail("terrain");

    eng::MaterialDesc ground_md;
    ground_md.shading = eng::Shading::Lit;
    ground_md.base_color = eng::Vec4{0.34f, 0.40f, 0.26f, 1.0f};
    ground_md.roughness = 0.92f;
    const eng::MaterialHandle ground_mat = app->Draw().CreateMaterial(ground_md, error);
    if (!eng::Valid(ground_mat)) return Fail(error);

    eng::Scene scene;
    std::vector<Chunk> chunks;
    for (int cz = 0; cz < terrain.ChunksZ(); ++cz)
        for (int cx = 0; cx < terrain.ChunksX(); ++cx) {
            Chunk chunk;
            chunk.cx = cx;
            chunk.cz = cz;
            // A LEVEL CHAIN PER CHUNK, built from the terrain's own coarser
            // samplings rather than by simplifying the finest one -- the
            // terrain already knows how to produce a coarse version, and it
            // produces one that lines up with its neighbours.
            for (int lod = 0; lod < 3; ++lod) {
                const eng::Mesh m = terrain.BuildChunk(cx, cz, lod);
                if (m.vertices.empty()) break;
                chunk.lods[lod] = app->Draw().UploadMesh(m);
                if (!eng::Valid(chunk.lods[lod])) break;
                ++chunk.lod_count;
            }
            if (chunk.lod_count == 0) continue;
            eng::Instance inst;
            inst.mesh = chunk.lods[0];
            inst.material = ground_mat;
            chunk.instance = scene.instances.size();
            scene.instances.push_back(inst);
            chunks.push_back(chunk);
        }
    std::printf("  %zu chunks\n", chunks.size());

    // --- physics -------------------------------------------------------------
    auto field = std::make_shared<eng::physics::HeightfieldData>();
    field->resolution = tc.resolution;
    field->spacing = tc.world_size / float(tc.resolution - 1);
    field->origin = tc.origin;
    field->heights.assign(terrain.Heights().begin(), terrain.Heights().end());

    eng::physics::World world;
    eng::physics::Body ground_body;
    ground_body.shape = eng::physics::Shape::MakeHeightfield(field);
    ground_body.inverse_mass = 0.0f;
    world.Add(ground_body);

    // --- the navmesh ---------------------------------------------------------
    //
    // Built from the terrain's COARSEST level. A navmesh voxelises whatever it
    // is given, so feeding it the finest mesh costs eight times the rasterising
    // for a walkable surface that is identical at the agent's scale.
    std::printf("building navmesh...\n");
    std::vector<eng::Vec3> nav_vertices;
    std::vector<std::uint32_t> nav_indices;
    for (int cz = 0; cz < terrain.ChunksZ(); ++cz)
        for (int cx = 0; cx < terrain.ChunksX(); ++cx) {
            const eng::Mesh m = terrain.BuildChunk(cx, cz, 2);
            const std::uint32_t base = std::uint32_t(nav_vertices.size());
            for (const VertexIn& v : m.vertices)
                nav_vertices.push_back(eng::Vec3{v.position.x, v.position.y, v.position.z});
            for (std::uint16_t i : m.indices) nav_indices.push_back(base + i);
        }
    eng::nav::BuildConfig nav_config;
    nav_config.cell_size = 0.6f;
    nav_config.cell_height = 0.2f;
    nav_config.agent_radius = 0.45f;
    nav_config.agent_height = 1.8f;
    nav_config.agent_max_climb = 0.5f;
    nav_config.agent_max_slope_degrees = 42.0f;
    nav_config.min_region_cells = 6;
    const eng::nav::NavMesh navmesh =
        eng::nav::NavMesh::Build(nav_vertices, nav_indices, nav_config);
    std::printf("  %d polygons, %d portals, %.0f ms\n", navmesh.PolyCount(),
                navmesh.Stats().portals, navmesh.Stats().build_seconds * 1000.0);

    // --- the character -------------------------------------------------------
    const eng::MeshHandle capsule_mesh = app->Draw().UploadMesh(
        eng::MakeUVSphere(0.45f, 16, 20, eng::Vec4{1, 1, 1, 1}, eng::Vec4{1, 1, 1, 1}));
    eng::MaterialDesc body_md;
    body_md.shading = eng::Shading::Lit;
    body_md.base_color = eng::Vec4{0.92f, 0.35f, 0.22f, 1.0f};
    body_md.roughness = 0.4f;
    const eng::MaterialHandle body_mat = app->Draw().CreateMaterial(body_md, error);
    eng::Instance body;
    body.mesh = capsule_mesh;
    body.material = body_mat;
    const std::size_t body_instance = scene.instances.size();
    scene.instances.push_back(body);

    // A marker at the destination, so a click has visible feedback even before
    // the character starts moving.
    eng::MaterialDesc marker_md;
    marker_md.shading = eng::Shading::Lit;
    marker_md.base_color = eng::Vec4{0.2f, 0.9f, 0.4f, 1.0f};
    marker_md.roughness = 0.3f;
    const eng::MaterialHandle marker_mat = app->Draw().CreateMaterial(marker_md, error);
    eng::Instance marker;
    marker.mesh = capsule_mesh;
    marker.material = marker_mat;
    const std::size_t marker_instance = scene.instances.size();
    scene.instances.push_back(marker);

    eng::physics::CharacterConfig cc;
    cc.radius = 0.4f;
    cc.height = 1.8f;
    cc.slope_limit_degrees = 45.0f;
    cc.step_height = 0.4f;
    eng::physics::CharacterController player(cc);
    player.Teleport(eng::Vec3{0.0f, terrain.HeightAt(0.0f, 0.0f) + 0.5f, 0.0f});

    const eng::rhi::TextureId shadow_map = app->Gpu().CreateShadowMap(2048);
    scene.shadowExtent = 30.0f;
    scene.shadowCascades = 3;
    scene.shadowDistance = 90.0f;

    eng::SkyConfig sky;
    float sun_azimuth = 1.1f, sun_elevation = 0.55f;
    post->config.fog = true;
    post->config.fog_density = 0.0055f;
    post->config.fog_height_falloff = 0.035f;
    post->config.fog_start = 6.0f;

    std::vector<eng::Vec3> path;
    std::size_t path_at = 0;
    float camera_yaw = 0.7f, camera_pitch = 0.55f, camera_distance = 22.0f;
    float baked_az = 1e9f, baked_el = 1e9f;

    app->Actions().BindMouse("click", eng::app::MouseButton::Left);
    app->Actions().BindMouse("orbit", eng::app::MouseButton::Right);
    app->Actions().Bind("fog", 'f');
    app->Actions().Bind("reset", 'r');

    eng::RenderGraph graph;
    while (app->Running()) {
        if (!app->BeginFrame()) continue;
        const eng::app::Frame& f = app->Current();

        // --- camera ----------------------------------------------------------
        if (app->Actions().Down("orbit")) {
            camera_yaw += f.mouse_dx * 0.006f;
            camera_pitch = std::clamp(camera_pitch - f.mouse_dy * 0.004f, 0.15f, 1.35f);
        }
        camera_distance = std::clamp(camera_distance - f.scroll * 1.2f, 6.0f, 60.0f);
        if (app->Actions().Pressed("fog")) post->config.fog = !post->config.fog;
        if (app->Actions().Pressed("reset")) {
            player.Teleport(eng::Vec3{0.0f, terrain.HeightAt(0.0f, 0.0f) + 0.5f, 0.0f});
            path.clear();
        }

        const eng::Vec3 focus = player.Feet() + eng::Vec3{0.0f, 1.0f, 0.0f};
        const eng::Vec3 offset{std::cos(camera_yaw) * std::cos(camera_pitch),
                               std::sin(camera_pitch),
                               std::sin(camera_yaw) * std::cos(camera_pitch)};
        scene.camera.eye = focus + offset * camera_distance;
        scene.camera.target = focus;
        // The camera must not go underground, which on a bowl-shaped terrain it
        // otherwise does every time it swings downhill.
        const float ground_at_camera =
            terrain.HeightAt(scene.camera.eye.x, scene.camera.eye.z);
        scene.camera.eye.y = std::max(scene.camera.eye.y, ground_at_camera + 1.5f);

        // --- picking and pathing ---------------------------------------------
        if (app->Actions().Pressed("click") && f.mouse_inside && navmesh.Valid()) {
            const float ndc_x = f.mouse_x / float(f.width) * 2.0f - 1.0f;
            const float ndc_y = 1.0f - f.mouse_y / float(f.height) * 2.0f;
            const eng::Mat4 inv =
                eng::Inverse(scene.camera.ViewProj(float(f.width) / float(f.height)));
            const eng::Vec4 near_h = inv * eng::Vec4{ndc_x, ndc_y, 1.0f, 1.0f};
            const eng::Vec4 far_h = inv * eng::Vec4{ndc_x, ndc_y, 0.0f, 1.0f};
            const eng::Vec3 a{near_h.x / near_h.w, near_h.y / near_h.w,
                              near_h.z / near_h.w};
            const eng::Vec3 b{far_h.x / far_h.w, far_h.y / far_h.w, far_h.z / far_h.w};

            float t = 0.0f;
            if (terrain.Raycast(a, eng::Normalize(b - a), 400.0f, &t, nullptr)) {
                const eng::Vec3 hit = a + eng::Normalize(b - a) * t;
                std::vector<eng::Vec3> found;
                if (navmesh.FindPath(player.Feet(), hit, &found) && found.size() > 1) {
                    path = found;
                    path_at = 1;  // 0 is where we already are
                }
            }
        }

        // --- movement ---------------------------------------------------------
        eng::Vec3 wish{0.0f, 0.0f, 0.0f};
        if (path_at < path.size()) {
            const eng::Vec3 target = path[path_at];
            eng::Vec3 to{target.x - player.Feet().x, 0.0f, target.z - player.Feet().z};
            const float distance = eng::Length(to);
            // The waypoint is reached when the character is within its own
            // radius, not at zero: aiming for an exact point makes the
            // character orbit it forever, because it overshoots every frame.
            if (distance < cc.radius + 0.15f) {
                ++path_at;
            } else {
                wish = to * (1.0f / distance) * 4.5f;
            }
        }
        const float dt = std::min(f.dt, 0.05f);
        player.Move(world, eng::Vec3{wish.x * dt, -9.81f * dt * dt * 6.0f, wish.z * dt});

        scene.instances[body_instance].model =
            eng::Mat4::Translation(player.Feet() + eng::Vec3{0.0f, 0.9f, 0.0f});
        if (path_at < path.size()) {
            scene.instances[marker_instance].model =
                eng::Mat4::Translation(path.back() + eng::Vec3{0.0f, 0.45f, 0.0f}) *
                eng::Mat4::Scale(0.6f);
        } else {
            // Parked underground rather than removed: the instance list's
            // indices are held by everything above, and compacting it would
            // invalidate them.
            scene.instances[marker_instance].model =
                eng::Mat4::Translation(eng::Vec3{0.0f, -1000.0f, 0.0f});
        }

        // --- levels of detail --------------------------------------------------
        int lod_counts[4] = {0, 0, 0, 0};
        for (Chunk& chunk : chunks) {
            eng::Vec3 lo, hi;
            terrain.ChunkBounds(chunk.cx, chunk.cz, &lo, &hi);
            const eng::Vec3 centre = (lo + hi) * 0.5f;
            const float distance = eng::Length(centre - scene.camera.eye);
            int lod = 0;
            if (distance > 40.0f) lod = 1;
            if (distance > 90.0f) lod = 2;
            lod = std::min(lod, chunk.lod_count - 1);
            scene.instances[chunk.instance].mesh = chunk.lods[lod];
            ++lod_counts[lod];
        }

        // --- sky ---------------------------------------------------------------
        sky.sun_direction =
            eng::Vec3{std::cos(sun_azimuth) * std::cos(sun_elevation),
                      std::sin(sun_elevation),
                      std::sin(sun_azimuth) * std::cos(sun_elevation)};
        eng::Environment::ApplyTo(&scene, sky);
        app->Draw().SetEnvironment(env->Bindings());
        const bool needs_bake = std::fabs(sun_azimuth - baked_az) > 0.004f ||
                                std::fabs(sun_elevation - baked_el) > 0.004f;
        if (needs_bake) {
            baked_az = sun_azimuth;
            baked_el = sun_elevation;
        }
        post->BeginFrame(scene.camera, f.width, f.height, dt);

        // --- HUD ----------------------------------------------------------------
        ui->Begin(f.width, f.height);
        ui->Rect(12, 12, 470, 156, eng::Vec4{0.05f, 0.06f, 0.08f, 0.72f});
        ui->Outline(12, 12, 470, 156, 1.0f, eng::Vec4{0.35f, 0.40f, 0.48f, 0.9f});
        char line[192];
        std::snprintf(line, sizeof(line), "%zu chunks   lod %d/%d/%d   %d nav polys",
                      chunks.size(), lod_counts[0], lod_counts[1], lod_counts[2],
                      navmesh.PolyCount());
        ui->Text(26, 26, line, eng::Vec4{0.92f, 0.94f, 0.98f, 1.0f});
        std::snprintf(line, sizeof(line), "path: %zu points, at %zu   ground %.2f m",
                      path.size(), path_at, terrain.HeightAt(player.Feet().x,
                                                             player.Feet().z));
        ui->Text(26, 50, line, eng::Vec4{0.80f, 0.86f, 0.94f, 1.0f});
        std::snprintf(line, sizeof(line), "gpu %.2f ms   fog %s   %s",
                      app->Gpu().LastFrameGpuMilliseconds(),
                      post->config.fog ? "on" : "off",
                      player.Grounded() ? "grounded" : "airborne");
        ui->Text(26, 74, line, eng::Vec4{0.80f, 0.86f, 0.94f, 1.0f});
        ui->Text(26, 104, "click: walk there    right-drag: orbit    scroll: zoom",
                 eng::Vec4{0.62f, 0.68f, 0.78f, 1.0f});
        ui->Text(26, 128, "f: fog    r: reset",
                 eng::Vec4{0.62f, 0.68f, 0.78f, 1.0f});

        // --- passes --------------------------------------------------------------
        const eng::rhi::TextureId color = app->Targets().Hdr("color");
        const eng::rhi::TextureId ms_color = app->Targets().Msaa("ms");
        const eng::rhi::TextureId ms_depth = app->Targets().MsaaDepth("ms_depth");
        const eng::rhi::TextureId depth = app->Targets().Depth("depth", true);
        graph.Clear();

        if (needs_bake)
            graph.AddCompute("sky", {env->Radiance(), env->Bindings().irradiance,
                                     env->Bindings().specular,
                                     env->Bindings().brdf_lut},
                             [&](eng::rhi::ComputeEncoder& e) { env->BakeSky(e, sky); },
                             "sky bake");
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
            // A DEPTH PREPASS, so the fog has a depth buffer it can sample. The
            // scene pass's own depth is multisampled and cannot be read.
            eng::RenderGraph::Pass p;
            p.name = "depth";
            p.depth = depth;
            p.keep_depth = true;
            p.timer = "depth";
            p.execute = [&](eng::rhi::Encoder& e) {
                app->Draw().DrawSceneDepth(e, scene, f.width, f.height);
            };
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
                env->DrawSky(e, scene.camera, f.width, f.height);
            };
            graph.AddPass(p);
        }
        if (post->config.fog) {
            eng::RenderGraph::Pass p;
            p.name = "fog";
            // MODIFIES, not writes. The scene pass already produced `color`,
            // and the graph's single-writer rule -- correctly -- refuses two
            // passes writing one target. A modification is ordered after the
            // producer and before the next reader, which is exactly what
            // blending over the scene means.
            p.modifies = color;
            p.reads = {depth};
            p.timer = "fog";
            p.execute = [&](eng::rhi::Encoder& e) { post->DrawFog(e, depth); };
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
        post->EndFrame();
        app->EndFrame();
    }
    return 0;
}
