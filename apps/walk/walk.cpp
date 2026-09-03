// Interactive. Walk around a level with the capsule controller, and point at
// things with a raycast.
//
// The two halves of this are the two halves of what a game is built on:
// SIMULATION says where everything ends up, and QUERIES say what is under the
// crosshair. The engine could draw and simulate anything long before it could
// answer the second question.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/app/app.h"
#include "engine/geometry/mesh.h"
#include "engine/physics/character.h"
#include "engine/physics/physics.h"
#include "engine/render/rendergraph.h"

namespace {

int Fail(const std::string& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.c_str());
    return 1;
}

// A block of level geometry: a static body and the instance that draws it, kept
// together so the raycast's body index maps straight back to something visible.
struct Block {
    int body = -1;
    std::size_t instance = 0;
    eng::Vec4 tint{1, 1, 1, 1};
};

}  // namespace

int main() {
    std::string error;
    eng::app::Config config;
    config.title = "virtual — walk";
    auto app = eng::app::App::Create(config, error);
    if (!app) return Fail(error);

    // --- meshes and materials ---------------------------------------------
    const eng::MeshHandle cube =
        app->Draw().UploadMesh(eng::MakeBox(eng::Vec3{0.5f, 0.5f, 0.5f},
                                            eng::Vec4{1, 1, 1, 1}));
    eng::MaterialDesc md;
    md.shading = eng::Shading::Lit;
    md.roughness = 0.65f;
    const eng::MaterialHandle mat = app->Draw().CreateMaterial(md, error);
    if (!Valid(cube) || !Valid(mat)) return Fail("scene build failed");

    const eng::rhi::TextureId shadow_map = app->Gpu().CreateShadowMap(2048);
    if (!Valid(shadow_map)) return Fail("shadow map");

    eng::Scene scene;
    scene.lightDir = eng::Vec4{-0.35f, 0.82f, 0.45f, 0.0f};
    scene.lightColor = eng::Vec4{1.05f, 1.02f, 0.95f, 1.0f};
    scene.ambientSky = eng::Vec3{0.16f, 0.19f, 0.26f};
    scene.ambientGround = eng::Vec3{0.06f, 0.05f, 0.05f};
    scene.shadowExtent = 26.0f;
    scene.shadowCascades = 3;
    scene.shadowDistance = 45.0f;

    eng::physics::World world;
    std::vector<Block> blocks;

    // Everything in the level is one call: a static box in the physics world
    // and a scaled cube in the scene, remembered together.
    const auto add = [&](eng::Vec3 centre, eng::Vec3 half, eng::Vec4 tint) {
        eng::physics::Body b;
        b.shape = eng::physics::Shape::MakeBox(half);
        b.position = centre;
        b.SetMass(0.0f);
        Block blk;
        blk.body = world.Add(b);
        blk.tint = tint;
        eng::Instance inst;
        inst.mesh = cube;
        inst.material = mat;
        inst.model = eng::Mat4::Translation(centre) *
                     eng::Mat4{{{half.x * 2, 0, 0, 0},
                                {0, half.y * 2, 0, 0},
                                {0, 0, half.z * 2, 0},
                                {0, 0, 0, 1}}};
        inst.tint = tint;
        blk.instance = scene.instances.size();
        scene.instances.push_back(inst);
        blocks.push_back(blk);
    };

    const eng::Vec4 kFloor{0.42f, 0.44f, 0.48f, 1};
    const eng::Vec4 kWall{0.55f, 0.50f, 0.46f, 1};
    const eng::Vec4 kStep{0.38f, 0.52f, 0.60f, 1};
    const eng::Vec4 kRamp{0.60f, 0.48f, 0.38f, 1};

    add(eng::Vec3{0, -0.5f, 0}, eng::Vec3{24, 0.5f, 24}, kFloor);

    // A flight of steps, each inside the controller's step height, so they are
    // walked up rather than jumped.
    for (int i = 0; i < 6; ++i)
        add(eng::Vec3{-6.0f, 0.15f + float(i) * 0.30f, -6.0f + float(i) * 1.2f},
            eng::Vec3{2.0f, 0.15f + float(i) * 0.30f, 0.6f}, kStep);

    // A ramp at 30 degrees, and one at 65 which is too steep to climb.
    for (int k = 0; k < 2; ++k) {
        const float deg = k == 0 ? 30.0f : 65.0f;
        const float rad = deg * 3.14159265f / 180.0f;
        const float L = 5.0f, t = 0.4f;
        eng::physics::Body b;
        b.shape = eng::physics::Shape::MakeBox(eng::Vec3{L, t, 2.5f});
        b.position = eng::Vec3{L * std::cos(rad) + t * std::sin(rad) + 4.0f,
                               L * std::sin(rad) - t * std::cos(rad),
                               k == 0 ? 6.0f : -6.0f};
        b.orientation = QuatFromAxisAngle(eng::Vec3{0, 0, 1}, rad);
        b.SetMass(0.0f);
        Block blk;
        blk.body = world.Add(b);
        blk.tint = kRamp;
        eng::Instance inst;
        inst.mesh = cube;
        inst.material = mat;
        inst.model = eng::Mat4::Translation(b.position) * QuatToMat4(b.orientation) *
                     eng::Mat4{{{L * 2, 0, 0, 0}, {0, t * 2, 0, 0},
                                {0, 0, 5.0f, 0}, {0, 0, 0, 1}}};
        inst.tint = kRamp;
        blk.instance = scene.instances.size();
        scene.instances.push_back(inst);
        blocks.push_back(blk);
    }

    // Pillars to point at and walk around.
    for (int i = 0; i < 7; ++i) {
        const float a = float(i) * 0.897f;
        add(eng::Vec3{std::cos(a) * 9.0f, 1.2f, std::sin(a) * 9.0f},
            eng::Vec3{0.5f, 1.2f, 0.5f}, kWall);
    }
    // A low beam to duck under -- or rather, not to be able to step under.
    add(eng::Vec3{0, 2.1f, -11.0f}, eng::Vec3{4.0f, 0.25f, 0.4f}, kWall);
    add(eng::Vec3{0, 0.15f, -10.0f}, eng::Vec3{4.0f, 0.15f, 0.6f}, kStep);

    // --- input --------------------------------------------------------------
    app->Actions().BindAxis("forward", 's', 'w');
    app->Actions().BindAxis("strafe", 'a', 'd');
    app->Actions().Bind("jump", ' ');
    app->Actions().Bind("sprint", 'q');
    app->Actions().Bind("reset", 'r');

    eng::physics::CharacterController player;
    player.Teleport(eng::Vec3{0, 0.2f, 2.0f});
    // Aimed at the steps, and tilted down enough that the crosshair meets the
    // floor within the ray's 40 m rather than leaving the level's 48 m square
    // first -- which it did at a near-level pitch, and reported "nothing"
    // entirely correctly.
    float yaw = 3.6f, pitch = -0.18f;
    float vertical = 0.0f;

    std::printf(
        "hold LEFT MOUSE and drag to look   w a s d: walk   q: sprint\n"
        "space: jump   r: respawn   esc: quit\n"
        "the crosshair raycasts the world: whatever it hits lights up\n");

    eng::RenderGraph graph;
    std::uint64_t last_report = 0;
    int looking_at = -1;
    float looking_dist = 0.0f;

    while (app->Running()) {
        if (!app->BeginFrame()) continue;
        const eng::app::Frame& f = app->Current();

        // MOUSE LOOK, on the drag the platform layer reports. There is no
        // cursor-lock or raw-delta path in the window layer yet, so it is
        // hold-to-look rather than always-look.
        yaw += f.drag_dx * 0.005f;
        pitch = std::clamp(pitch - f.drag_dy * 0.005f, -1.4f, 1.4f);

        const eng::Vec3 forward{std::cos(pitch) * std::cos(yaw), std::sin(pitch),
                                std::cos(pitch) * std::sin(yaw)};
        // Movement is on the GROUND PLANE, so looking up does not slow you down
        // and looking down does not drive you into the floor.
        const eng::Vec3 flat = Normalize(eng::Vec3{forward.x, 0.0f, forward.z});
        const eng::Vec3 right = Normalize(Cross(flat, eng::Vec3{0, 1, 0}));

        if (app->Actions().Pressed("reset"))
            player.Teleport(eng::Vec3{0, 0.2f, 2.0f});

        const float speed = app->Actions().Down("sprint") ? 8.5f : 3.6f;
        eng::Vec3 wish = flat * app->Actions().Axis("forward") +
                         right * app->Actions().Axis("strafe");
        if (Dot(wish, wish) > 1e-6f) wish = Normalize(wish) * speed;

        // GRAVITY, and a jump only from the ground. The controller reports
        // whether it is standing on something walkable; it does not integrate
        // anything itself, which is what makes it a controller rather than a
        // body.
        if (player.Grounded() && vertical <= 0.0f) {
            vertical = 0.0f;
            if (app->Actions().Pressed("jump")) vertical = 5.2f;
        } else {
            vertical -= 18.0f * f.dt;
        }
        player.Move(world, wish * f.dt + eng::Vec3{0.0f, vertical * f.dt, 0.0f});
        // Having hit a ceiling or landed, stop accumulating fall speed -- the
        // controller absorbed the motion and the caller has to notice.
        if (player.Grounded() && vertical < 0.0f) vertical = 0.0f;

        const eng::Vec3 eye = player.Feet() + eng::Vec3{0.0f, 1.65f, 0.0f};
        scene.camera.eye = eye;
        scene.camera.target = eye + forward;

        // THE CROSSHAIR. One ray, ignoring the player's own body -- except the
        // player has no body, being kinematic, so there is nothing to ignore
        // and the ray starts inside nothing.
        eng::physics::RayHit hit;
        looking_at = -1;
        if (world.Raycast(eye, forward, 40.0f, &hit)) {
            looking_at = hit.body;
            looking_dist = hit.t;
        }

        // Tint whatever is being looked at, which is the whole demonstration:
        // a body index came back from a query and turned into something on
        // screen.
        for (const Block& b : blocks) {
            eng::Vec4 t = b.tint;
            if (b.body == looking_at) t = eng::Vec4{t.x * 2.2f + 0.3f, t.y * 2.2f,
                                                    t.z * 1.4f, 1.0f};
            scene.instances[b.instance].tint = t;
        }

        const eng::rhi::TextureId ms_color = app->Targets().Msaa("scene");
        const eng::rhi::TextureId ms_depth = app->Targets().MsaaDepth("scene");
        const eng::rhi::TextureId color = app->Targets().Hdr("scene");

        graph.Clear();
        {
            eng::RenderGraph::Pass p;
            p.name = "shadow";
            p.depth = shadow_map;
            p.clear_depth = 0.0f;
            p.keep_depth = true;
            p.execute = [&](eng::rhi::Encoder& e) {
                app->Draw().DrawShadow(e, scene);
            };
            graph.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "scene";
            p.color = ms_color;
            p.resolve = color;
            p.depth = ms_depth;
            p.clear_color[0] = 0.05f; p.clear_color[1] = 0.06f;
            p.clear_color[2] = 0.09f; p.clear_color[3] = 1.0f;
            p.clear_depth = 0.0f;
            p.reads = {shadow_map};
            p.execute = [&](eng::rhi::Encoder& e) {
                app->Draw().DrawScene(e, scene, f.width, f.height, shadow_map);
            };
            graph.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "composite";
            p.color = f.drawable;
            p.reads = {color};
            p.execute = [&](eng::rhi::Encoder& e) {
                app->Draw().DrawComposite(e, color);
            };
            graph.AddPass(std::move(p));
        }
        if (!graph.Compile(error)) return Fail(error);
        graph.Execute(app->Gpu());
        app->EndFrame();

        if (f.index - last_report > 30) {
            last_report = f.index;
            // No text rendering yet, so the terminal is the HUD.
            std::printf("\r%.0f fps  feet (%.2f %.2f %.2f)  %s  looking at %s",
                        app->Time().Fps(), player.Feet().x, player.Feet().y,
                        player.Feet().z,
                        player.Grounded() ? "grounded"
                                          : (player.Supported() ? "on a slope"
                                                                : "in the air "),
                        looking_at < 0 ? "nothing        " : "");
            if (looking_at >= 0) std::printf("body %-3d at %.1f m ", looking_at,
                                             looking_dist);
            std::fflush(stdout);
        }
    }
    std::printf("\n");
    return 0;
}
