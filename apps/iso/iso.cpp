// A 2.5D isometric game view: 3D world, fixed camera angle, click to move.
//
// "2.5D" here means the world is fully three-dimensional -- real geometry, real
// shadows, a real character controller walking on it -- while the CAMERA is
// pinned to one angle and one orthographic projection. That is what makes it
// read as a board rather than a space: with no perspective, a crate is the same
// size wherever it stands, so distance is judged by position on the board and
// never by how large something looks.
//
// It is also the view that needs mouse picking most. There is no crosshair to
// aim: the player points at a place on the ground, which means turning a pixel
// back into a world position -- a ray through the camera and a raycast against
// the world.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/app/app.h"
#include "engine/geometry/mesh.h"
#include "engine/physics/character.h"
#include "engine/physics/physics.h"
#include "engine/audio/system.h"
#include "engine/platform/audio_file.h"
#include "engine/platform/font.h"
#include "engine/render/rendergraph.h"
#include "engine/text/ui.h"

namespace {

int Fail(const std::string& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.c_str());
    return 1;
}

struct Prop {
    int body = -1;
    std::size_t instance = 0;
    eng::Vec4 tint{1, 1, 1, 1};
    bool collected = false;
};

}  // namespace

int main() {
    std::string error;
    eng::app::Config config;
    config.title = "virtual — isometric";
    auto app = eng::app::App::Create(config, error);
    if (!app) return Fail(error);

    // --- the HUD's font -----------------------------------------------------
    const eng::platform::FontAtlas font =
        eng::platform::RasterizeFont("Menlo", 26.0f, error);
    if (!font.Valid()) return Fail(error.empty() ? "font" : error);
    auto ui = eng::ui::Canvas::Create(app->Gpu(), font, config.color,
                                      error, 1);
    if (!ui) return Fail(error);

    // --- audio ---------------------------------------------------------------
    //
    // Optional, deliberately: a machine with no output device, or a missing
    // music file, should cost the sound and not the game.
    auto audio = eng::audio::AudioSystem::Create(error);
    if (!audio) std::fprintf(stderr, "no audio: %s\n", error.c_str());

    eng::audio::Clip music;
    if (audio) {
        std::string e;
        music = eng::platform::DecodeAudioFile(
            "music.flac", e);
        if (!music.Valid()) std::fprintf(stderr, "no music: %s\n", e.c_str());
    }

    // Synthesised rather than loaded, so the demo needs no asset it did not
    // ship with. A coin is a two-note arpeggio and a footstep is filtered
    // noise -- both are a few lines and neither needs an artist.
    const int rate = audio ? audio->SampleRate() : 48000;
    eng::audio::Clip coin_sfx, step_sfx;
    {
        coin_sfx.rate = rate;
        coin_sfx.channels = 1;
        const int n = rate / 5;
        coin_sfx.samples.resize(std::size_t(n));
        for (int i = 0; i < n; ++i) {
            const float t = float(i) / float(rate);
            // Up a fifth part way through, which is what makes it read as a
            // pickup rather than a beep.
            const float hz = t < 0.06f ? 880.0f : 1318.5f;
            const float env = std::min(t * 90.0f, 1.0f) * std::exp(-t * 9.0f);
            coin_sfx.samples[std::size_t(i)] =
                0.5f * env * std::sin(2.0f * 3.14159265f * hz * t);
        }

        step_sfx.rate = rate;
        step_sfx.channels = 1;
        const int m = rate / 12;
        step_sfx.samples.resize(std::size_t(m));
        std::uint32_t seed = 12345;
        float low = 0.0f;
        for (int i = 0; i < m; ++i) {
            seed = seed * 1664525u + 1013904223u;
            const float white = float(seed >> 8) / 8388608.0f - 1.0f;
            // One-pole low pass: unfiltered noise is a hiss, and a footstep is
            // a thud.
            low += (white - low) * 0.12f;
            const float t = float(i) / float(rate);
            step_sfx.samples[std::size_t(i)] =
                0.7f * low * std::exp(-t * 34.0f);
        }
    }

    const eng::MeshHandle cube = app->Draw().UploadMesh(
        eng::MakeBox(eng::Vec3{0.5f, 0.5f, 0.5f}, eng::Vec4{1, 1, 1, 1}));
    const eng::MeshHandle ball = app->Draw().UploadMesh(
        eng::MakeUVSphere(0.5f, 20, 28, eng::Vec4{1, 1, 1, 1},
                          eng::Vec4{1, 1, 1, 1}));
    eng::MaterialDesc md;
    md.shading = eng::Shading::Lit;
    md.roughness = 0.6f;
    const eng::MaterialHandle mat = app->Draw().CreateMaterial(md, error);
    const eng::rhi::TextureId shadow_map = app->Gpu().CreateShadowMap(2048);
    if (!Valid(cube) || !Valid(mat) || !Valid(shadow_map))
        return Fail("scene build failed");

    eng::Scene scene;
    scene.lightDir = eng::Vec4{-0.42f, 0.80f, 0.42f, 0.0f};
    scene.lightColor = eng::Vec4{1.15f, 1.10f, 0.98f, 1.0f};
    scene.ambientSky = eng::Vec3{0.20f, 0.24f, 0.32f};
    scene.ambientGround = eng::Vec3{0.07f, 0.06f, 0.06f};
    scene.shadowExtent = 18.0f;

    eng::physics::World world;
    std::vector<Prop> props;
    std::size_t player_instance = 0;

    const auto add = [&](eng::MeshHandle mesh, eng::Vec3 centre, eng::Vec3 half,
                         eng::Vec4 tint, bool solid) {
        Prop p;
        if (solid) {
            eng::physics::Body b;
            b.shape = eng::physics::Shape::MakeBox(half);
            b.position = centre;
            b.SetMass(0.0f);
            p.body = world.Add(b);
        }
        eng::Instance inst;
        inst.mesh = mesh;
        inst.material = mat;
        inst.model = eng::Mat4::Translation(centre) *
                     eng::Mat4{{{half.x * 2, 0, 0, 0}, {0, half.y * 2, 0, 0},
                                {0, 0, half.z * 2, 0}, {0, 0, 0, 1}}};
        inst.tint = tint;
        p.tint = tint;
        p.instance = scene.instances.size();
        scene.instances.push_back(inst);
        props.push_back(p);
        return props.size() - 1;
    };

    // A board: a floor, a low wall around it, some crates, and coins to collect.
    add(cube, eng::Vec3{0, -0.25f, 0}, eng::Vec3{11, 0.25f, 11},
        eng::Vec4{0.34f, 0.37f, 0.40f, 1}, true);
    for (int i = -1; i <= 1; i += 2) {
        add(cube, eng::Vec3{float(i) * 11.0f, 0.6f, 0}, eng::Vec3{0.5f, 0.6f, 11.0f},
            eng::Vec4{0.46f, 0.42f, 0.38f, 1}, true);
        add(cube, eng::Vec3{0, 0.6f, float(i) * 11.0f}, eng::Vec3{11.0f, 0.6f, 0.5f},
            eng::Vec4{0.46f, 0.42f, 0.38f, 1}, true);
    }
    for (int i = 0; i < 9; ++i) {
        const float a = float(i) * 0.7f;
        add(cube, eng::Vec3{std::cos(a) * (3.0f + float(i) * 0.6f), 0.5f,
                            std::sin(a) * (3.0f + float(i) * 0.6f)},
            eng::Vec3{0.5f, 0.5f, 0.5f}, eng::Vec4{0.55f, 0.40f, 0.30f, 1}, true);
    }
    std::vector<std::size_t> coins;
    for (int i = 0; i < 8; ++i) {
        const float a = float(i) * 0.785f + 0.4f;
        coins.push_back(add(ball, eng::Vec3{std::cos(a) * 7.5f, 0.45f,
                                            std::sin(a) * 7.5f},
                            eng::Vec3{0.22f, 0.22f, 0.22f},
                            eng::Vec4{2.4f, 1.7f, 0.4f, 1}, false));
    }
    // The player: drawn, not collided -- the controller is the collision.
    {
        eng::Instance inst;
        inst.mesh = ball;
        inst.material = mat;
        inst.tint = eng::Vec4{0.35f, 0.75f, 1.05f, 1};
        player_instance = scene.instances.size();
        scene.instances.push_back(inst);
    }

    eng::physics::CharacterConfig cc;
    cc.radius = 0.35f;
    cc.height = 1.1f;
    eng::physics::CharacterController player(cc);
    player.Teleport(eng::Vec3{0, 0.2f, 0});

    app->Actions().BindMouse("move_to", eng::app::MouseButton::Left);
    app->Actions().BindAxis("x", 'a', 'd');
    app->Actions().BindAxis("z", 'w', 's');
    app->Actions().Bind("reset", 'r');

    // THE ISOMETRIC FRAME. A fixed direction, a fixed distance, and an
    // orthographic projection -- the camera follows the player by translating,
    // never by rotating, which is what keeps the board's geometry readable.
    const eng::Vec3 kViewDir = Normalize(eng::Vec3{-1.0f, -1.15f, -1.0f});
    float zoom = 9.0f;
    int score = 0;
    eng::Vec3 destination{0, 0, 0};
    bool has_destination = false;
    eng::Vec3 pick{0, 0, 0};
    bool has_pick = false;

    std::printf("left click the ground to walk there   w a s d also works\n"
                "scroll: zoom   r: reset   esc: quit\n");

    if (audio && music.Valid()) {
        eng::audio::PlayDesc bg;
        bg.clip = &music;
        bg.gain = 0.9f;
        bg.loop = true;
        (void)audio->Play(bg);
    }
    float step_timer = 0.0f;

    eng::RenderGraph graph;
    while (app->Running()) {
        if (!app->BeginFrame()) continue;
        const eng::app::Frame& f = app->Current();

        zoom = std::clamp(zoom - f.scroll * 0.4f, 4.0f, 20.0f);
        if (app->Actions().Pressed("reset")) {
            player.Teleport(eng::Vec3{0, 0.2f, 0});
            score = 0;
            has_destination = false;
            for (std::size_t c : coins) props[c].collected = false;
        }

        // The camera, placed relative to the player along the fixed direction.
        const eng::Vec3 focus = player.Feet() + eng::Vec3{0.0f, 0.5f, 0.0f};
        scene.camera.projection = eng::Projection::Orthographic;
        scene.camera.orthoHeight = zoom;
        scene.camera.eye = focus - kViewDir * 40.0f;
        scene.camera.target = focus;
        const float aspect = float(f.width) / float(f.height);

        // --- PICKING: a pixel back into the world -----------------------------
        //
        // Under an orthographic projection every ray shares the view direction
        // and they differ only in origin, so the pixel picks the START of the
        // ray rather than its direction. Under perspective it is the other way
        // round -- which is exactly the sort of thing that makes a picking
        // routine written for one projection silently wrong under the other.
        has_pick = false;
        if (f.mouse_inside) {
            const float ndc_x = f.mouse_x / float(f.width) * 2.0f - 1.0f;
            const float ndc_y = 1.0f - f.mouse_y / float(f.height) * 2.0f;
            const eng::Vec3 right =
                Normalize(Cross(kViewDir, eng::Vec3{0, 1, 0}));
            const eng::Vec3 up = Cross(right, kViewDir);
            const eng::Vec3 origin = scene.camera.eye +
                                     right * (ndc_x * zoom * aspect) +
                                     up * (ndc_y * zoom);
            eng::physics::RayHit hit;
            if (world.Raycast(origin, kViewDir, 200.0f, &hit)) {
                pick = hit.point;
                has_pick = true;
                if (app->Actions().Pressed("move_to") && hit.normal.y > 0.5f) {
                    destination = hit.point;
                    has_destination = true;
                }
            }
        }

        // --- movement ---------------------------------------------------------
        eng::Vec3 wish{0, 0, 0};
        const float kx = app->Actions().Axis("x"), kz = app->Actions().Axis("z");
        if (std::fabs(kx) + std::fabs(kz) > 0.0f) {
            // Keys move along the SCREEN's axes, not the world's. On an
            // isometric board the world axes run diagonally, so "press right,
            // go right" is the only mapping that is not baffling.
            const eng::Vec3 right = Normalize(Cross(kViewDir, eng::Vec3{0, 1, 0}));
            const eng::Vec3 fwd = Normalize(eng::Vec3{kViewDir.x, 0, kViewDir.z});
            wish = Normalize(right * kx + fwd * kz) * 4.2f;
            has_destination = false;  // taking manual control cancels the order
        } else if (has_destination) {
            const eng::Vec3 to = destination - player.Feet();
            const eng::Vec3 flat{to.x, 0.0f, to.z};
            if (Length(flat) > 0.25f) wish = Normalize(flat) * 4.2f;
            else has_destination = false;
        }
        player.Move(world, wish * f.dt + eng::Vec3{0.0f, -9.0f * f.dt, 0.0f});

        // THE LISTENER rides the player, not the camera. In an isometric view
        // the camera is forty metres back and fixed; putting the listener there
        // would make everything equally distant and everything centred, which
        // is every spatial cue there is.
        //
        // Its forward is the camera's, though, because left and right on screen
        // must be left and right in the ears -- the player has no facing of
        // their own to use.
        if (audio) {
            audio->SetListener(player.Feet() + eng::Vec3{0, 0.8f, 0}, kViewDir,
                               eng::Vec3{0, 1, 0});
        }

        // FOOTSTEPS, paced by distance travelled rather than by time. Timed
        // steps keep playing while the character is pressed against a wall and
        // going nowhere.
        if (audio && player.Grounded()) {
            step_timer += Length(eng::Vec3{player.LastMotion().x, 0.0f,
                                           player.LastMotion().z});
            if (step_timer > 0.9f) {
                step_timer = 0.0f;
                eng::audio::PlayDesc d;
                d.clip = &step_sfx;
                d.spatial = true;
                d.position = player.Feet();
                d.gain = 0.55f;
                // A little detune each time, because a footstep sample played
                // identically twice in a row stops sounding like a footstep and
                // starts sounding like a sample.
                d.pitch = 0.9f + 0.2f * std::fabs(std::sin(f.time * 12.3f));
                (void)audio->Play(d);
            }
        }

        scene.instances[player_instance].model =
            eng::Mat4::Translation(player.Feet() + eng::Vec3{0, 0.5f, 0}) *
            eng::Mat4::Scale(1.0f);

        // Coins, collected by walking into them.
        for (std::size_t c : coins) {
            Prop& p = props[c];
            eng::Instance& inst = scene.instances[p.instance];
            if (p.collected) { inst.tint = eng::Vec4{0, 0, 0, 0}; continue; }
            const eng::Vec4 pos = inst.model.col[3];
            if (Length(eng::Vec3{pos.x, 0, pos.z} -
                       eng::Vec3{player.Feet().x, 0, player.Feet().z}) < 0.8f) {
                p.collected = true;
                ++score;
                if (audio) {
                    eng::audio::PlayDesc d;
                    d.clip = &coin_sfx;
                    d.spatial = true;
                    d.position = eng::Vec3{pos.x, pos.y, pos.z};
                    d.min_distance = 2.0f;
                    d.gain = 0.8f;
                    // Rising with the count, so the eighth coin is a payoff
                    // rather than the same sound an eighth time.
                    d.pitch = 1.0f + float(score) * 0.06f;
                    (void)audio->Play(d);
                }
            }
            const float pulse = 1.0f + 0.35f * std::sin(f.time * 4.0f);
            inst.tint = eng::Vec4{p.tint.x * pulse, p.tint.y * pulse,
                                  p.tint.z * pulse, 1.0f};
        }

        // --- the HUD ----------------------------------------------------------
        ui->Begin(f.width, f.height);
        const float pad = 14.0f;
        ui->Rect(pad, pad, 260.0f, 96.0f, eng::Vec4{0.04f, 0.05f, 0.08f, 0.72f});
        ui->Outline(pad, pad, 260.0f, 96.0f, 1.0f,
                    eng::Vec4{0.45f, 0.55f, 0.70f, 0.8f});
        char line[128];
        std::snprintf(line, sizeof(line), "coins  %d / %zu", score, coins.size());
        ui->Text(pad + 14.0f, pad + 10.0f, line, eng::Vec4{1, 0.92f, 0.55f, 1});
        std::snprintf(line, sizeof(line), "%.0f fps", app->Time().Fps());
        ui->Text(pad + 14.0f, pad + 38.0f, line, eng::Vec4{0.75f, 0.82f, 0.9f, 1});
        if (audio)
            std::snprintf(line, sizeof(line), "%d voices   peak %.2f",
                          audio->ActiveVoices(), audio->LastPeak());
        else
            std::snprintf(line, sizeof(line), "no audio device");
        ui->Text(pad + 14.0f, pad + 64.0f, line, eng::Vec4{0.6f, 0.7f, 0.8f, 1});

        // A marker under the cursor, drawn in SCREEN space over the world.
        if (has_pick) {
            std::snprintf(line, sizeof(line), "%.1f, %.1f", pick.x, pick.z);
            ui->Text(f.mouse_x + 16.0f, f.mouse_y - 6.0f, line,
                     eng::Vec4{0.85f, 0.9f, 1.0f, 0.85f}, eng::ui::Align::Left,
                     0.7f);
            ui->Outline(f.mouse_x - 7.0f, f.mouse_y - 7.0f, 14.0f, 14.0f, 1.0f,
                        eng::Vec4{0.9f, 0.95f, 1.0f, 0.9f});
        }
        if (score == int(coins.size()))
            ui->Text(float(f.width) * 0.5f, float(f.height) * 0.42f,
                     "all collected  -  r to play again",
                     eng::Vec4{1.0f, 0.95f, 0.7f, 1.0f}, eng::ui::Align::Centre,
                     1.3f);

        // --- draw --------------------------------------------------------------
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
            p.execute = [&](eng::rhi::Encoder& e) { app->Draw().DrawShadow(e, scene); };
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
                // The HUD goes in the SAME pass, after the tone map, straight
                // onto the drawable: it is authored in display colours and
                // would be tone mapped into something paler if it went through
                // the scene target.
                ui->Draw(e);
            };
            graph.AddPass(std::move(p));
        }
        if (!graph.Compile(error)) return Fail(error);
        graph.Execute(app->Gpu());
        app->EndFrame();
    }
    return 0;
}
