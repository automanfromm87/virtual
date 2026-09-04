// The integration gate. Each check below fails if ONE of the three subsystems
// is disconnected, even though everything still compiles and still renders.
//
// The distinction that matters: a check like "the frame is not empty" passes
// with physics deleted. These check the specific thing each layer contributes —
// that geometry came from a FILE, that children move because a PARENT moved,
// and that positions changed because the SOLVER ran.
//
// Writes world.ppm for eyeballing; the assertions are what pass or fail.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tests/world/world_scene.h"
#include "engine/render/rendergraph.h"
#include "engine/render/renderer.h"
#include "engine/rhi/rhi.h"

namespace {

// The TEXTURED fixture: a quad whose material points at a PNG embedded in the
// document. Loading it proves the base64 -> zlib -> PNG -> GPU path end to end.
const char* const kQuadGltf =
#include "engine/asset/testdata_textured_gltf.inc"
    ;

// The default arrangement, baked in so the binary needs nothing on disk to run.
// #embed rather than a runfile: a demo that cannot start without finding a data
// directory is a demo that stops working the moment it is moved.
constexpr char kDefaultScene[] = {
#embed "tests/world/world.scene.json"
    , 0};

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

constexpr int kW = 900, kH = 700;

double Luma(const std::vector<std::uint8_t>& p, std::size_t i) {
    return 0.2126 * p[i] + 0.7152 * p[i + 1] + 0.0722 * p[i + 2];
}

eng::Vec3 Origin(const eng::Mat4& m) {
    const eng::Vec4 o = m * eng::Vec4{0, 0, 0, 1};
    return eng::Vec3{o.x, o.y, o.z};
}

}  // namespace

// An optional scene path, same as the viewer: the point of a scene file is
// that changing it does not mean rebuilding, and a gate that can only ever
// check the baked-in copy would not be testing that.
std::string ReadOr(const char* path, const char* fallback) {
    if (!path) return fallback;
    std::FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "FAIL: cannot open %s\n", path); std::exit(1); }
    std::string text;
    char buf[4096];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) text.append(buf, n);
    std::fclose(f);
    return text;
}

int main(int argc, char** argv) {
    const std::string scene_text = ReadOr(argc > 1 ? argv[1] : nullptr, kDefaultScene);
    std::string error;
    auto dev = eng::rhi::Device::Create(error);
    if (!dev) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
    const auto kFmt = eng::rhi::Format::RGBA8Unorm;
    auto renderer = eng::Renderer::Create(*dev, kFmt, error);
    if (!renderer) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    demo::World world = demo::Build(*dev, *renderer, kQuadGltf, scene_text, error);
    if (!world.ok) {
        std::fprintf(stderr, "FAIL: %s\n", error.empty() ? "build" : error.c_str());
        return 1;
    }

    // ---------------------------------------------------------------- import
    std::printf("asset import\n");
    {
        std::string e;
        const eng::gltf::Document doc = eng::gltf::ParseGltf(kQuadGltf, {}, e);
        Check(e.empty() && !doc.primitives.empty(), "glTF parsed into primitives");
        const eng::Mesh& m = doc.primitives[0].mesh;
        Check(!m.vertices.empty() && !m.indices.empty(), "imported mesh has geometry");
        // Normals came out of the file's NORMAL accessor, not from a default.
        // A reader that dropped the attribute would leave these zero and the
        // surface would be unlit — visible, but wrong.
        bool normalized = !m.vertices.empty();
        for (const auto& v : m.vertices) {
            const float len = std::sqrt(v.normal.x * v.normal.x +
                                        v.normal.y * v.normal.y +
                                        v.normal.z * v.normal.z);
            if (std::fabs(len - 1.0f) > 1e-3f) normalized = false;
        }
        Check(normalized, "imported normals are unit length");
        Check(!doc.materials.empty(), "imported a material definition");
        Check(Valid(world.gltf_mesh) && world.gltf_mesh != world.sphere_mesh,
              "imported mesh uploaded as its own handle");
    }

    // ------------------------------------------------------------- hierarchy
    std::printf("transform hierarchy\n");
    {
        demo::Update(world, 0.0f, /*simulate=*/false);
        const eng::Vec3 arm_before = Origin(world.ecs.WorldOf(world.arms[0]));
        const eng::Vec3 sign_before = Origin(world.ecs.WorldOf(world.sign));
        const eng::Vec3 local_before = world.ecs.transforms.Get(world.arms[0])->position;

        // Rotate ONLY the parent. Nothing touches the children.
        const float half = 1.5707963f * 0.5f;  // quarter turn, half-angle
        world.ecs.transforms.Get(world.carousel)->rotation =
            eng::Quat{0.0f, std::sin(half), 0.0f, std::cos(half)};
        demo::Update(world, 0.0f, /*simulate=*/false);

        const eng::Vec3 arm_after = Origin(world.ecs.WorldOf(world.arms[0]));
        // Direction, not point: the .w = 0 makes this ignore translation.
        const eng::Vec4 sa = world.ecs.WorldOf(world.sign) * eng::Vec4{1, 0, 0, 0};
        const eng::Vec3 sign_axis_after = Normalize(eng::Vec3{sa.x, sa.y, sa.z});
        const eng::Vec3 local_after = world.ecs.transforms.Get(world.arms[0])->position;

        Check(Length(arm_after - arm_before) > 1.0f,
              "rotating the parent moved the child in world space");
        Check(Length(local_after - local_before) < 1e-6f,
              "the child's LOCAL transform was not touched");
        // A quarter turn about Y sends +x to -z. Direction, not just distance:
        // a hierarchy that applied the rotation on the wrong side would still
        // move the child, just to the wrong place.
        const eng::Vec3 pivot = Origin(world.ecs.WorldOf(world.carousel));
        const eng::Vec3 r0 = arm_before - pivot;
        const eng::Vec3 r1 = arm_after - pivot;
        Check(std::fabs(r1.x - r0.z) < 1e-3f && std::fabs(r1.z + r0.x) < 1e-3f,
              "child rotated the right way (+x -> -z about +y)");
        Check(std::fabs(Length(r1) - Length(r0)) < 1e-4f,
              "the child kept its distance from the pivot");
        // The imported mesh hangs on the carousel too, but directly on the
        // rotation axis — so its ORIGIN provably cannot move, and checking
        // position here would assert something false. What must change is its
        // ORIENTATION: a hierarchy that propagated only translation would move
        // the arms correctly and leave this one facing the wrong way.
        Check(Length(Origin(world.ecs.WorldOf(world.sign)) - sign_before) < 1e-4f,
              "a child on the rotation axis stayed put");
        Check(std::fabs(sign_axis_after.x - 0.0f) < 1e-3f &&
                  std::fabs(sign_axis_after.z + 1.0f) < 1e-3f,
              "the imported mesh inherited the parent's ROTATION");
        Check(world.ecs.Depth(world.arms[0]) == 1 && world.ecs.Depth(world.carousel) == 0,
              "hierarchy depth is what the parenting says");

        world.ecs.transforms.Get(world.carousel)->rotation = eng::Quat{};
        demo::Update(world, 0.0f, /*simulate=*/false);
    }

    // --------------------------------------------------------------- physics
    std::printf("physics\n");
    const float start_energy = world.physics.Energy();
    float peak_energy = start_energy;
    float highest_y = -1e9f;
    {
        for (const eng::ecs::Entity e : world.balls)
            highest_y = std::max(highest_y, world.ecs.transforms.Get(e)->position.y);

        // The top ball starts 16 m up, so it is still falling at four seconds
        // and the pile below it is still absorbing bounces. Ten seconds is
        // what it actually takes to settle; a shorter window would make this
        // check measure the clock rather than the solver.
        for (int i = 0; i < 1200; ++i) {
            world.physics.StepFixed();
            peak_energy = std::max(peak_energy, world.physics.Energy());
        }
        Check(peak_energy <= start_energy * 1.02f + 1.0f,
              "the solver never injected energy");

        float lowest = 1e9f, max_speed = 0.0f;
        for (const eng::ecs::Entity e : world.balls) {
            const eng::physics::Body& b = world.physics[world.ecs.bodies.Get(e)->body];
            lowest = std::min(lowest, b.position.y);
            max_speed = std::max(max_speed, Length(b.velocity));
        }
        // Nothing tunnelled through a static box.
        Check(lowest > demo::kFloorTop + demo::kBallRadius - 0.05f,
              "every ball rests ON the floor, not through it");
        Check(max_speed < 0.5f, "the pile came to rest");
        Check(world.physics.Stats().contacts > 0, "contacts were generated");

        // The ECS has not seen any of this yet — Update() is what copies it.
        float ecs_highest = -1e9f;
        for (const eng::ecs::Entity e : world.balls)
            ecs_highest = std::max(ecs_highest, world.ecs.transforms.Get(e)->position.y);
        Check(std::fabs(ecs_highest - highest_y) < 1e-6f,
              "physics did not reach through and write the ECS itself");
        demo::Update(world, 0.0f, /*simulate=*/true);
        float after = -1e9f;
        for (const eng::ecs::Entity e : world.balls)
            after = std::max(after, world.ecs.transforms.Get(e)->position.y);
        Check(after < highest_y - 1.0f, "Update() copied the simulation into the ECS");

        // ROLLING. The balls are dropped on a tilted ramp with no initial spin,
        // so every degree of rotation they have came from friction acting at
        // the contact point. A solver that applies impulses at the centre of
        // mass moves them exactly as far and leaves them facing forward.
        int turned = 0;
        float mean_x = 0.0f, start_x = 0.0f;
        for (std::size_t i = 0; i < world.balls.size(); ++i) {
            const eng::physics::Body& b =
                world.physics[world.ecs.bodies.Get(world.balls[i])->body];
            if (std::fabs(b.orientation.w) < 0.98f) ++turned;  // > ~23 degrees
            mean_x += b.position.x;
            start_x += world.ball_start[i].x;
        }
        mean_x /= float(world.balls.size());
        start_x /= float(world.balls.size());
        Check(turned == int(world.balls.size()), "every ball turned, not just slid");
        Check(mean_x < start_x - 2.0f, "they ran downhill off the ramp");
        // And the ECS carries the orientation, not just the position — the
        // renderer reads its transform, so a ball spinning only inside the
        // physics world would still look frozen on screen.
        const eng::ecs::Transform* t = world.ecs.transforms.Get(world.balls[0]);
        Check(std::fabs(t->rotation.w) < 0.98f, "the ECS transform carries the spin");
    }

    // ---------------------------------------------------------------- render
    std::printf("render\n");
    const eng::rhi::TextureId shadow_map = dev->CreateShadowMap(2048);
    const eng::rhi::TextureId color = dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
    const eng::rhi::TextureId depth = dev->CreateDepthTarget(kW, kH);
    const eng::rhi::TextureId out = dev->CreateRenderTarget(kW, kH, kFmt, true);
    if (!Valid(shadow_map) || !Valid(color) || !Valid(depth) || !Valid(out)) {
        std::fprintf(stderr, "FAIL: targets\n");
        return 1;
    }

    eng::OrbitController orbit;
    orbit.target = eng::Vec3{0.0f, 1.8f, 0.0f};
    orbit.distance = 17.0f;
    orbit.pitch = 0.35f;

    std::vector<std::uint8_t> pixels;
    eng::RenderStats stats;
    int shadow_draws = 0;

    auto draw = [&](const eng::Scene& base) {
        eng::Scene scene = base;
        orbit.Apply(scene.camera);
        eng::RenderGraph graph;
        {
            eng::RenderGraph::Pass p;
            p.name = "shadow";
            p.depth = shadow_map;
            p.clear_depth = 0.0f;
            p.keep_depth = true;
            p.execute = [&](eng::rhi::Encoder& e) { renderer->DrawShadow(e, scene); };
            graph.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "scene";
            p.color = color;
            p.depth = depth;
            for (int i = 0; i < 4; ++i) p.clear_color[i] = eng::kClearColor[i];
            p.clear_depth = 0.0f;
            p.reads = {shadow_map};
            p.execute = [&](eng::rhi::Encoder& e) {
                renderer->DrawScene(e, scene, kW, kH, shadow_map);
            };
            graph.AddPass(std::move(p));
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "composite";
            p.color = out;
            p.reads = {color};
            p.execute = [&](eng::rhi::Encoder& e) { renderer->DrawComposite(e, color); };
            graph.AddPass(std::move(p));
        }
        std::string e;
        if (!graph.Compile(e)) { std::fprintf(stderr, "FAIL: %s\n", e.c_str()); return; }
        dev->BeginFrame();
        graph.Execute(*dev);
        std::string werr;
        if (!dev->CommitAndWait(werr))
            std::fprintf(stderr, "FAIL: %s\n", werr.c_str());
        stats = renderer->LastStats();
        shadow_draws = renderer->ShadowDrawCount();
        pixels.assign(std::size_t(kW) * kH * 4, 0);
        (void)dev->ReadPixels(out, kW, kH, pixels);
    };

    // A caster's HEIGHT above the receiver must not decide whether it casts.
    //
    // Regression test for a real bug: fs_shadow reads the section-cut plane out
    // of FrameUniforms, but DrawShadow bound that block to the vertex stage
    // only. An unbound fragment buffer reads as zero rather than failing, so
    // the cut plane silently became y = 0 and every caster above the origin was
    // discarded from the shadow map. Objects resting on the ground still cast,
    // which is why it survived: the reference scene's spheres straddle y = 0.
    //
    // The sweep is the test. A single height cannot catch this — pick the wrong
    // one and it passes against a completely broken shadow pass.
    std::printf("shadow, by caster height\n");
    {
        auto shadowed_pixels = [&](float gap) {
            eng::Scene s;
            s.shadowExtent = 6.0f;
            s.lightDir = eng::Vec4{0.45f, 0.80f, 0.40f, 0.0f};
            s.lightColor = eng::Vec4{3.0f, 3.0f, 3.0f, 1.0f};
            s.instances.push_back(
                {world.box_mesh, world.floor_mat,
                 eng::Mat4::Translation(eng::Vec3{0, -0.5f, 0}), eng::Vec4{1, 1, 1, 1}});
            s.instances.push_back(
                {eng::kMeshSphere, world.ball_mat,
                 eng::Mat4::Translation(eng::Vec3{0, 0.8f + gap, 0}) *
                     eng::Mat4::Scale(0.8f),
                 eng::Vec4{1, 1, 1, 1}});
            draw(s);
            const std::vector<std::uint8_t> lit_with = pixels;
            s.shadowExtent = 0.0f;
            draw(s);
            int n = 0;
            for (std::size_t i = 0; i < pixels.size(); i += 4)
                if (Luma(pixels, i) - Luma(lit_with, i) > 8.0) ++n;
            return n;
        };
        for (float gap : {0.0f, 0.6f, 1.5f, 3.0f}) {
            const int n = shadowed_pixels(gap);
            char what[64];
            std::snprintf(what, sizeof what, "a caster %.1f m up still casts", gap);
            Check(n > 150, what);
        }
    }

    const eng::Scene scene = demo::ToScene(world);
    // Not AliveCount(): the carousel pivot is an entity with a Transform and
    // deliberately no Renderable, which is the whole reason a component pool is
    // not just a field on a game object.
    Check(int(scene.instances.size()) == int(world.ecs.renderables.Size()) &&
              world.ecs.AliveCount() == int(world.ecs.renderables.Size()) + 1,
          "every visible entity became an instance, the pivot did not");
    draw(scene);
    Check(stats.draws > 0 && stats.invalid == 0 && stats.overflowed == 0,
          "the whole ECS drew with nothing dropped");
    Check(shadow_draws > 0, "the ECS scene also fed the shadow pass");

    // The IMPORTED TEXTURE reaches the screen. The panel's pixels come from a
    // PNG that was base64'd into the glTF document — no image file exists in
    // the source tree, and nothing else in the scene is textured.
    //
    // Checking that the panel is "not black" would pass with the texture
    // dropped, because an untextured material binds a 1x1 white stand-in. The
    // fixture is a checker of a WARM and a COOL quadrant, so the falsifiable
    // claim is that both appear.
    {
        std::vector<bool> was;
        for (std::size_t i = 0; i < world.ecs.renderables.Size(); ++i) {
            was.push_back(world.ecs.renderables.At(i).visible);
            world.ecs.renderables.At(i).visible =
                world.ecs.renderables.Owner(i) == world.sign;
        }
        world.ecs.transforms.Get(world.carousel)->rotation = eng::Quat{};
        world.ecs.UpdateTransforms();

        const eng::OrbitController saved = orbit;
        orbit.target = Origin(world.ecs.WorldOf(world.sign));
        // yaw = pi/2 puts the eye on +z. The quad faces +z, and at yaw 0 the
        // camera sits on +x and sees it edge-on — a zero-pixel "pass".
        orbit.yaw = 1.5707963f;
        orbit.pitch = 0.0f;
        orbit.distance = 7.0f;
        eng::Scene only_panel = demo::ToScene(world);
        only_panel.shadowExtent = 0.0f;
        draw(only_panel);

        int warm = 0, cool = 0;
        for (std::size_t i = 0; i < pixels.size(); i += 4) {
            const int r = pixels[i], b = pixels[i + 2];
            if (r > b + 30) ++warm;
            if (b > r + 30) ++cool;
        }
        Check(stats.draws == 1, "only the imported panel drew");
        // Measured: 29480 warm and 52608 cool. Untextured, one of the two
        // collapses to zero because a flat material has a single hue.
        Check(warm > 5000 && cool > 5000,
              "the imported PNG's two colours both reached the frame");

        orbit = saved;
        for (std::size_t i = 0; i < world.ecs.renderables.Size(); ++i)
            world.ecs.renderables.At(i).visible = was[i];
        world.ecs.UpdateTransforms();
        // Put the full scene back in `stats`: the checks below measure a delta
        // against the last render, and leaving a one-draw frame there would
        // make the next one compare against the wrong baseline.
        draw(demo::ToScene(world));
    }

    // A renderable that is switched off stops drawing. This is the one line of
    // the Renderable component that the frame can actually falsify.
    {
        world.ecs.renderables.Get(world.sign)->visible = false;
        const int before = stats.draws;
        draw(demo::ToScene(world));
        Check(stats.draws == before - 1, "hiding a renderable removes exactly one draw");
        world.ecs.renderables.Get(world.sign)->visible = true;
    }

    // Destroying an entity removes it from the world AND from the frame, with
    // no stale handle left pointing at it.
    {
        const eng::ecs::Entity victim = world.arms.back();
        const int before = world.ecs.AliveCount();
        const int drawn_before = int(world.ecs.renderables.Size());
        world.ecs.Destroy(victim);
        Check(!world.ecs.Alive(victim) && world.ecs.AliveCount() == before - 1,
              "Destroy() retired the entity");
        const eng::Scene s = demo::ToScene(world);
        Check(int(s.instances.size()) == drawn_before - 1,
              "the destroyed entity left the draw list");
        draw(s);
        Check(stats.invalid == 0, "no stale handle survived the destroy");
    }

    // The image itself: geometry over background, and more than one material.
    {
        draw(demo::ToScene(world));
        int lit = 0;
        for (std::size_t i = 0; i < pixels.size(); i += 4)
            if (Luma(pixels, i) > 30.0) ++lit;
        const double frac = double(lit) / double(kW * kH);
        Check(frac > 0.15 && frac < 0.95, "the frame is geometry, not a flat fill");
        // The balls are warm, the arms are cold. One material for everything
        // would leave these equal.
        // A mean over the whole frame is useless here: the background is most
        // of it and is itself slightly blue, so warm geometry disappears into
        // the average. Counting warm and cool pixels separately is the thing
        // that actually goes to zero if every material collapses into one.
        int warm = 0, cool = 0;
        for (std::size_t i = 0; i < pixels.size(); i += 4) {
            if (int(pixels[i]) > int(pixels[i + 2]) + 20) ++warm;
            if (int(pixels[i + 2]) > int(pixels[i]) + 20) ++cool;
        }
        Check(warm > 2000 && cool > 300,
              "warm and cool materials are both on screen");

        std::FILE* f = std::fopen("world.ppm", "wb");
        if (f) {
            std::fprintf(f, "P6\n%d %d\n255\n", kW, kH);
            for (std::size_t i = 0; i < pixels.size(); i += 4)
                std::fwrite(&pixels[i], 1, 3, f);
            std::fclose(f);
        }
    }

    std::printf(
        "\nenergy: start %.1f peak %.1f end %.1f   entities %d   bodies %d   draws %d\n",
        start_energy, peak_energy, world.physics.Energy(), world.ecs.AliveCount(),
        world.physics.Count(), stats.draws);
    std::printf(g_failures == 0 ? "world_test: all checks passed\n"
                                : "world_test: %d FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
