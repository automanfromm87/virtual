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
#include <chrono>
#include <cmath>
#include <random>
#include <thread>
#include <cstdio>
#include <string>
#include <vector>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>

#include "engine/app/app.h"
#include "engine/app/targets.h"
#include "engine/asset/gltf.h"
#include "engine/asset/png.h"
#include "engine/geometry/mesh.h"
#include "engine/geometry/simplify.h"
#include "apps/world/districts.h"
#include "engine/asset/texgen.h"
#include "engine/render/gi.h"
#include "engine/render/fluid.h"
#include "engine/render/particles.h"
#include "engine/render/volumetric.h"
#include "engine/geometry/terrain.h"
#include "engine/geometry/tree.h"
#include "apps/world/humanoid.h"
#include "engine/asset/soundgen.h"
#include "engine/audio/system.h"
#include "engine/anim/anim.h"
#include "engine/anim/blend.h"
#include "engine/anim/ik.h"
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

// The raw ground: noise plus the bowl that keeps the character in the world.
float Wilderness(float x, float z) {
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

// The ground the world is actually built on: the wilderness, with a level pad
// cut under each district. See world::kPads for why this is not optional.
//
// The terrain mesh, the physics heightfield and every ground_y the builders use
// all come through here, so they cannot disagree about where the ground is --
// which is the failure this replaced, in the other direction.
float Landscape(float x, float z) {
    float height = Wilderness(x, z);
    for (const world::Pad& pad : world::kPads) {
        const float dx = x - pad.centre.x, dz = z - pad.centre.z;
        const float d = std::sqrt(dx * dx + dz * dz);
        const float level = Wilderness(pad.centre.x, pad.centre.z);
        // Inside the pad it is flat; outside, it may pull away from pad level
        // at kPadSlope and no faster. Clamping rather than blending is what
        // bounds the join's steepness by construction -- see world::kPadSlope.
        const float room = std::max(0.0f, d - pad.radius) * world::kPadSlope;
        height = std::clamp(height, level - room, level + room);
    }
    return height;
}

struct Chunk {
    std::size_t instance = 0;
    int cx = 0, cz = 0;
    eng::MeshHandle lods[4];
    int lod_count = 0;
};

}  // namespace

// The asset lives in the source tree, but neither `bazel run` nor a test
// harness starts with the workspace root as the working directory. Walk up
// from the cwd and from the binary itself; the first tree containing the
// asset wins. A harness with no assets finds nothing and keeps the
// procedural fallback.
std::string ResolveAsset(const std::string& rel, int argc, char** argv) {
    const std::string argv0 = argc > 0 ? argv[0] : "";
    const std::string bases[2] = {std::filesystem::current_path().string(),
                                  std::filesystem::path(argv0).parent_path().string()};
    for (const std::string& base : bases) {
        std::error_code ec;
        std::filesystem::path dir = std::filesystem::canonical(base, ec);
        if (ec) dir = std::filesystem::path(base);
        for (int up = 0; up < 8; ++up) {
            const std::string cand = (dir / rel).string();
            std::ifstream probe(cand, std::ios::binary);
            if (probe.good()) return cand;
            dir = dir.parent_path();
        }
    }
    return rel;
}

// Reads a .gltf (siblings resolved next to it) or a .glb into a Document.
// A missing file is an error string, not a crash: callers fall back.
eng::gltf::Document LoadAssetDoc(const std::string& path, std::string& error) {
    std::ifstream f(path, std::ios::binary);
    std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(f),
                                    std::istreambuf_iterator<char>()};
    if (!f) {
        error = "gltf: cannot open " + path;
        return {};
    }
    if (eng::gltf::IsGlb(bytes)) return eng::gltf::ParseGlb(bytes, error);
    return eng::gltf::LoadGltfFile(path, error);
}

int main(int argc, char** argv) {
    // CAPTURE MODE. Renders a fixed number of frames into a readable target and
    // writes a PNG instead of opening on the drawable.
    //
    // The drawable cannot be read back, so a bug that only shows on screen
    // cannot be looked at from a terminal -- which is how this demo came to be
    // washed out without anyone noticing.
    std::string shot_path;
    int shot_frames = 12;
    // The player's look: a real glTF asset when it resolves (a .gltf with
    // sibling .bin/.png, or a .glb), the generated humanoid otherwise --
    // notably under `bazel test`, where no assets ship. Missing file is a
    // fallback, not a failure: the valley gate runs the same binary with no
    // assets beside it and must see exactly the old figure.
    std::string fox_path = "assets/fox/Fox.gltf";
    // Crowd boulders: Kenney rocks (CC0, OBJ converted to .glb once --
    // flat facet normals kept as authored), alternating for variety.
    // Missing files keep spheres.
    std::string rock_path = "assets/rock/rock_tallA.glb";
    std::string rock_path2 = "assets/rock/rock_largeA.glb";
    bool start_fog = true;
    // For measuring temporal aliasing. Two captures a fraction of a pixel apart
    // should differ by a fraction of a pixel's worth of colour; anything that
    // flips hard under a sub-pixel move is aliasing, and that is the only way
    // to separate shimmer from honest motion.
    float yaw_offset = 0.0f;
    float lod_near = 45.0f;
    int shot_w = 1100, shot_h = 760;
    float shadow_dist = 90.0f;
    float ssao_radius = 1.1f;
    bool ssao_start = true;
    bool shafts_start = true;
    bool sparks_start = true;
    // 0.004. At 0.014 the far treeline vanished entirely and the meter dropped
    // most of a stop trying to save the sun glow, which crushed the ground.
    // Looking into a low sun through trees IS hazy -- that is the effect -- but
    // the haze has to leave the trees in it.
    float shaft_ext = 0.004f;
    int tour_start = 0;
    float headless_walk = 0.0f, headless_heading = 0.0f;
    bool no_foot_ik = false;
    float pitch_start = 0.26f;
    float focus_start = 1.0f;
    float dist_start = 0.0f;  // 0 = whatever the stop chooses
    float leaf_transmission = 0.55f;
    float exposure_comp = -1.0f;
    // RENDER SCALE: the scene renders at this fraction of the display, and
    // the TAA resolve rebuilds full resolution out of history. 1.0 is
    // native; 0.5 quarters the scene's pixels. The UI stays at display.
    float render_scale = 1.0f;
    bool decals_on = true;
    // GPU-DRIVEN (#3): the opaque statics go through CullScene (a compute
    // frustum cull) and one indirect draw per (mesh, material) batch, instead
    // of one CPU draw per object. Whatever the indirect path excludes --
    // the skinned figure, transparents -- is drawn the ordinary way after it.
    bool indirect_on = false;
    // GPU-DRIVEN showcase: this many free-standing boulders (independent
    // instances of one mesh and material) scattered around the clearing. The
    // ordinary path submits one draw each; the indirect path batches them
    // into one. 0 is off.
    int crowd_count = 0;
    // FORCES THE DEBOUNCED GI RE-BAKE every N frames, with no input. It exists
    // so tests/valley can exercise the re-bake path from a headless capture:
    // that path opened a nested device frame, leaked a frames-in-flight permit
    // per bake and deadlocked the app outright on the third, and a defect that
    // needs three keypresses to reproduce is a defect no test will ever see.
    // NO HUD. The debug panel is 500x276 points plus five text lines that
    // overflow it onto the sky -- 16.5% of a 1100x760 capture -- and there was
    // no way to turn it off, which is why this repository has never had a
    // screenshot of its own renderer that was not mostly a screenshot of its
    // own diagnostics.
    //
    // It also makes a capture REPRODUCIBLE. Two --shot runs of the same frame
    // differ in 0.18-0.30% of pixels and every one of those pixels is inside
    // the HUD: the frame counter, the exposure reading, the pass timings. The
    // scene under it is bit-identical.
    bool hud_on = true;
    // The screen-space surface's two shape knobs, on the command line because
    // they are the pair that has to be re-tuned for every particle scale and
    // the only way to settle them is to look.
    //
    // BOTH WELL PAST apps/fluid's, and past what fluid.h's own guidance
    // suggests: the tank uses 0.06 and 0.5, and the header warns that a sphere
    // radius over 0.7 floats the surface a visible radius above the particles.
    // It does, and it is still the right trade here. That tank is a quarter of
    // this basin's volume at half the particle spacing, so its spheres already
    // overlap in the depth buffer; these do not, and the bilateral filter
    // refuses to merge across the gaps between them -- it reads the several
    // centimetres of empty space behind each sphere as a different surface and
    // stops. Swept it and looked: at 0.11/0.58 the water is a heap of discrete
    // beads with bright rims, at 0.25/0.85 the outlines are still there, and at
    // 0.45/1.10 it is a sheet with a clean waterline against the stone. The
    // cost is 0.15 m of float on a 0.45 m basin, which is invisible next to
    // beads.
    float water_edge_stop = 0.45f, water_sphere = 1.10f;
    int rebake_every = 0;
    // BLOOM, in LINEAR radiance before the tone map. The threshold has to sit
    // above the brightest thing that is merely lit and below the things that
    // are sources: sunlit grass in this valley comes in under one, the lantern
    // glass is at 7 and the sparks at 14.
    // MEASURED on this valley, not carried over from tests/gallery, which uses
    // 3.2 -- that scene's lamps run into the tens of units and this one's
    // radiance tops out near 1.2, so a threshold of 3.2 here selects nothing at
    // all and the pass is a no-op that costs 0.9 ms. At 1.0 the mean rises by
    // 3.10 at the lantern hall, 0.70 in the clearing and 0.33 at the fire pit:
    // strongest where the emissive sources are, which is the shape it should
    // have.
    // LOCAL-LIGHT SHADOWS, off by default and the reason is arithmetic, not
    // taste: the atlas is 16 tiles and this valley has 60 lanterns, so at best
    // a quarter of them cast and DrawLightShadows takes the first sixteen it
    // finds. Shadowing an arbitrary subset of a colonnade of identical lamps
    // looks worse than shadowing none of them. The flag exists so the cost can
    // be measured and so the tile budget has something to be measured against.
    // MEASURED once it was wired: the valley's 61 lanterns request no shadow at
    // all -- Light::ShadowFaces() is zero for every one of them -- so the pass
    // records nothing and reports 0 tiles of 61 lights. Making them cast is
    // content work in districts.h, and even then 16 tiles over 61 lamps means
    // an arbitrary sixteen of a colonnade of identical lights cast and the rest
    // do not, which reads worse than none of them casting. The flag is here so
    // the cost has something to be measured against when the tile budget grows.
    bool local_shadows_start = false;
    bool bloom_start = true;
    float bloom_threshold = 1.0f;
    float bloom_strength = 0.6f;
    // 4096, MEASURED against an 8192 render of the same frame. The fraction of
    // pixels more than 20 of 255 away from that reference:
    //
    //     2048, 3 cascades   6.01%   2.06 ms   16 MB
    //     2048, 4 cascades   5.24%   2.35 ms   16 MB
    //     4096, 3 cascades   2.01%   2.16 ms   64 MB
    //     4096, 4 cascades   1.62%   2.53 ms   64 MB
    //
    // Three times closer for four percent more time. A FOURTH CASCADE is the
    // cheaper-looking option -- the 2x2 tile layout already has an empty tile
    // waiting -- and it is not worth it: it buys 6.01 to 5.24 and costs a whole
    // extra pass over every caster, 14% of the frame.
    //
    // Cutting shadowDistance was tried and makes it WORSE, 6.01 to 10.23. It
    // does not just tighten the cascades, it stops anything past it casting at
    // all, and the far half of the scene loses its shadows.
    int shadow_px = 4096;
    int shadow_cascades = 3;
    // Sun elevation in degrees. Exposed so a capture can be taken at a stated
    // time of day rather than only at the one the demo starts with.
    float sun_start = 31.5f;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shot") == 0 && i + 1 < argc)
            shot_path = argv[++i];
        else if (std::strcmp(argv[i], "--fox") == 0 && i + 1 < argc)
            fox_path = argv[++i];
        else if (std::strcmp(argv[i], "--rock") == 0 && i + 1 < argc)
            rock_path = argv[++i];
        else if (std::strcmp(argv[i], "--rock2") == 0 && i + 1 < argc)
            rock_path2 = argv[++i];
        else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc)
            shot_frames = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--nofog") == 0) start_fog = false;
        else if (std::strcmp(argv[i], "--yaw") == 0 && i + 1 < argc)
            yaw_offset = float(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--sun") == 0 && i + 1 < argc)
            sun_start = float(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--lodnear") == 0 && i + 1 < argc)
            lod_near = float(std::atof(argv[++i]));
        // Headless renders at exactly this size -- there is no window and so no
        // backing-scale factor to double it.
        else if (std::strcmp(argv[i], "--shadowdist") == 0 && i + 1 < argc)
            shadow_dist = float(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--shadowpx") == 0 && i + 1 < argc)
            shadow_px = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--cascades") == 0 && i + 1 < argc)
            shadow_cascades = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ssao") == 0 && i + 1 < argc)
            ssao_radius = float(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--nossao") == 0) ssao_start = false;
        else if (std::strcmp(argv[i], "--noshafts") == 0) shafts_start = false;
        else if (std::strcmp(argv[i], "--nosparks") == 0) sparks_start = false;
        else if (std::strcmp(argv[i], "--shaft") == 0 && i + 1 < argc)
            shaft_ext = float(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--noik") == 0)
            no_foot_ik = true;
        else if (std::strcmp(argv[i], "--walk") == 0)
            headless_walk = world::kWalkSpeed;
        else if (std::strcmp(argv[i], "--run") == 0)
            headless_walk = world::kRunSpeed;
        else if (std::strcmp(argv[i], "--heading") == 0 && i + 1 < argc)
            headless_heading = float(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--look") == 0 && i + 1 < argc)
            tour_start = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--pitch") == 0 && i + 1 < argc)
            pitch_start = float(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--focush") == 0 && i + 1 < argc)
            focus_start = float(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--dist") == 0 && i + 1 < argc)
            dist_start = float(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--noleaf") == 0) leaf_transmission = 0.0f;
        else if (std::strcmp(argv[i], "--nodecal") == 0) decals_on = false;
        else if (std::strcmp(argv[i], "--ec") == 0 && i + 1 < argc)
            exposure_comp = float(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--scale") == 0 && i + 1 < argc)
            render_scale = float(std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--localshadows") == 0)
            local_shadows_start = true;
        else if (std::strcmp(argv[i], "--water") == 0 && i + 2 < argc) {
            water_edge_stop = float(std::atof(argv[++i]));
            water_sphere = float(std::atof(argv[++i]));
        }
        else if (std::strcmp(argv[i], "--nohud") == 0) hud_on = false;
        else if (std::strcmp(argv[i], "--nobloom") == 0) bloom_start = false;
        else if (std::strcmp(argv[i], "--bloom") == 0 && i + 2 < argc) {
            bloom_threshold = float(std::atof(argv[++i]));
            bloom_strength = float(std::atof(argv[++i]));
        }
        else if (std::strcmp(argv[i], "--rebake") == 0 && i + 1 < argc)
            rebake_every = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--indirect") == 0) indirect_on = true;
        else if (std::strcmp(argv[i], "--crowd") == 0 && i + 1 < argc)
            crowd_count = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--size") == 0 && i + 2 < argc) {
            shot_w = std::atoi(argv[++i]);
            shot_h = std::atoi(argv[++i]);
        }
    }

    // LINE BUFFERED. Redirected to a file, stdout is fully buffered, and a
    // windowed app that is quit rather than returning never flushes -- so the
    // load-time diagnostics below simply never appear.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string error;
    eng::app::Config config;
    config.title = "virtual — terrain, navigation, sky";
    // A CAPTURE OPENS NO WINDOW. --shot already renders into a target of its
    // own rather than the drawable, so the window was doing nothing but existing
    // -- and it meant every measurement of this scene needed a window server.
    // Headless also fixes the timestep, which is what makes two captures of the
    // same frame byte-identical; with a wall-clock dt the exposure meter and the
    // character both integrate a different number each run.
    config.headless = !shot_path.empty();
    if (config.headless) {
        config.width = shot_w;
        config.height = shot_h;
    }
    auto app = eng::app::App::Create(config, error);
    if (!app) return Fail(error);

    const eng::platform::FontAtlas font =
        eng::platform::RasterizeFont("Menlo", 24.0f, error);
    if (!font.Valid()) return Fail(error.empty() ? "font" : error);
    auto ui = eng::ui::Canvas::Create(app->Gpu(), font, config.color, error, 1);
    if (!ui) return Fail(error);
    // RGBA16Float, NOT config.color. This argument is the format of the target
    // the SKY is drawn into, and that is the scene's HDR buffer -- the swapchain
    // format was passed here, so the pipeline was built for BGRA8Unorm and
    // bound to an RGBA16Float attachment. A mismatched attachment format does
    // not fail, it produces undefined pixels: 1.3% of the frame came back with
    // green and blue annihilated and red stuck at 0 or 255, speckled through
    // the sky and along the treeline. The default is RGBA16Float for exactly
    // this reason and naming a format here was the mistake.
    auto env = eng::Environment::Create(app->Gpu(), error, 256,
                                        eng::rhi::Format::RGBA16Float,
                                        config.samples);
    if (!env) return Fail(error);
    auto post = eng::PostStack::Create(app->Gpu(), error);
    if (!post) return Fail(error);
    post->config.render_scale = render_scale;
    // The scene's own targets, at render size: everything up to the TAA
    // resolve runs small, and the history/output the post stack owns stay at
    // display. A second FrameTargets rather than resized app ones -- the app
    // sizes its own to the window every frame, and nothing display-sized
    // needs app targets anyway (history/output are post-owned, the composite
    // writes the drawable, the UI draws into the composite pass).
    //
    // Same format and samples as the app's: BGRA8 colour, 4x MSAA, which are
    // the Config defaults this viewer never overrides.
    eng::app::FrameTargets scene_targets(app->Gpu(), eng::rhi::Format::BGRA8Unorm, 4);
    // LIGHT SHAFTS. The one effect a forest is actually about: the canopy cuts
    // the sunlight into beams and you see them in the air, not on a surface.
    // Everything else here lights things; this lights the space between them.
    auto vol = eng::Volumetrics::Create(app->Gpu(), error);
    if (!vol) return Fail(error);
    // The fire. Drawn into the scene target with the scene's depth, so a spark
    // behind a stone is behind it and one in front fades softly into the ground
    // instead of cutting a hard edge into it.
    // IN THE SCENE PASS, so config.samples and not 1.
    //
    // ONE SAMPLE AND NO DEPTH, matching the pass they are actually drawn into.
    //
    // They used to be drawn inside the multisampled scene pass, and the comment
    // here said so; when the depth resolve let that pass produce a sampleable
    // depth, the sparks moved out to their own pass over the RESOLVED colour --
    // which is single-sampled and has no depth attachment -- and this line was
    // not moved with them. Metal rejects the bind and drops every draw after
    // it, silently unless MTL_DEBUG_LAYER is set. Measured: 1,576 live
    // particles and 0 differing pixels of 279,000 at the fire pit.
    //
    // Losing the hardware depth test costs nothing here. particles.metal
    // already fades each sprite against the sampled scene depth -- that is what
    // stops a spark cutting a hard edge into the ground -- and a soft fade is a
    // better occlusion edge than a binary test.
    auto sparks = eng::ParticleSystem::Create(app->Gpu(), 6000,
                                              eng::Renderer::kSceneFormat, error,
                                              /*samples=*/1, /*depth_test=*/false);
    if (!sparks) return Fail(error);

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

    // --- procedural surfaces ---------------------------------------------------
    //
    // Generated, not loaded, because this engine ships no assets -- and an
    // untextured world is the biggest single thing between a correct render and
    // a photograph. Flat albedo has nothing for the shading to act on, so every
    // surface reads as the same plastic whatever its roughness says.
    //
    // UPLOADED AS LINEAR, not sRGB. These maps MULTIPLY the material's base
    // colour, so they are ratios rather than colours, and a ratio has no gamma
    // -- decoding one as sRGB would turn a 0.87 modulation into 0.73 and darken
    // everything they touch.
    std::printf("generating textures...\n");
    const eng::texgen::Surface grass_tex = eng::texgen::Ground(512, 7);
    const eng::texgen::Surface bark_tex = eng::texgen::Bark(512, 23);
    const eng::texgen::Image leaf_tex = eng::texgen::Foliage(256, 41);
    const auto upload = [&](const eng::texgen::Image& im) {
        return app->Gpu().CreateTexture2D(im.width, im.height, im.rgba.data(),
                                          /*mips=*/true, /*srgb=*/false);
    };
    const eng::rhi::TextureId grass_albedo = upload(grass_tex.albedo);
    const eng::rhi::TextureId grass_normal = upload(grass_tex.normal);
    const eng::rhi::TextureId bark_albedo = upload(bark_tex.albedo);
    const eng::rhi::TextureId bark_normal = upload(bark_tex.normal);
    const eng::rhi::TextureId leaf_albedo = upload(leaf_tex);
    // sRGB and NOT the linear upload the others get: this one is a COLOUR laid
    // over a surface, not a ratio multiplying one, so it goes through the same
    // transfer function any painted texture would.
    const eng::texgen::Image soot_img = eng::texgen::Soot(256, 12);
    const eng::rhi::TextureId soot_tex = app->Gpu().CreateTexture2D(
        soot_img.width, soot_img.height, soot_img.rgba.data(), true, true);
    if (!eng::rhi::Valid(grass_albedo) || !eng::rhi::Valid(bark_normal))
        return Fail("texture upload");

    eng::MaterialDesc ground_md;
    ground_md.shading = eng::Shading::Lit;
    // A PHYSICALLY PLAUSIBLE ALBEDO, which this was not. It was
    // {0.34, 0.40, 0.26} -- 34 to 40 percent reflectance, which is concrete or
    // dry sand, not grass. Living vegetation reflects 10 to 20 percent in the
    // visible band, and the terrain's vertex colour is white so nothing else
    // was bringing it down.
    //
    // This is why the sunlit ground read as straw no matter what the tone map
    // did: a surface three times too bright and half as saturated as grass is
    // not grass, and no amount of exposure or curve fixes the albedo. The meter
    // simply opens up to compensate, which lifts the shadows too -- so the
    // scene gets MORE contrast range out of the correction, not less.
    ground_md.base_color = eng::Vec4{0.11f, 0.17f, 0.07f, 1.0f};
    ground_md.roughness = 0.92f;
    ground_md.albedo = grass_albedo;
    ground_md.normal_map = grass_normal;
    ground_md.normal_strength = 0.9f;
    // A terrain chunk lays uv 0..1 across the WHOLE 128 m terrain, so without a
    // scale one blade of grass would be four metres wide. 40 repeats puts the
    // tile at about three metres, which is the scale a clump of grass actually
    // is and is fine enough that the repeat is not the first thing seen.
    ground_md.uv_scale = eng::Vec2{40.0f, 40.0f};
    const eng::MaterialHandle ground_mat = app->Draw().CreateMaterial(ground_md, error);
    if (!eng::Valid(ground_mat)) return Fail(error);

    // --- the forest ------------------------------------------------------------
    //
    // ONE MESH FOR EVERY TRUNK and one for every canopy. A hundred trees as a
    // hundred instances is two hundred draw calls for scenery that never moves;
    // merged, it is two, and the whole forest costs less to submit than the
    // character does. The cost is that it cannot be culled per tree -- which is
    // the right trade here, because at 128 m across the whole forest is on
    // screen most of the time anyway.
    //
    // SIX SEEDS, not two hundred. Every tree from one seed is identical, and a
    // row of identical trees is the most obvious tell there is; but two hundred
    // distinct skeletons cost two hundred generations and nobody can tell six
    // apart once they are rotated and scaled differently.
    std::printf("growing trees...\n");
    // A GRID OF CELLS, not one mesh for the whole forest.
    //
    // Merging every tree into a single pair of meshes made the forest two draw
    // calls, which was the right trade when the alternative was four hundred --
    // but one mesh is one bounding volume, so a forest that spans the world can
    // never be culled and never has a distance. 2.5M triangles were submitted
    // every frame regardless of where the camera pointed, and again into each
    // of three shadow cascades.
    //
    // Six by six over 128 metres is a 21 m cell. Small enough that most of them
    // are off screen at any moment and the far ones can drop detail, large
    // enough that a full view is 72 draws rather than 400.
    constexpr int kForestCells = 6;
    // THREE LEVELS, REGENERATED rather than simplified.
    //
    // BuildLodChain was tried first and it destroys this content. It clusters
    // vertices in cells scaled to the MESH, and a cell here is 21 metres across
    // -- so at a thirty-second that is a 0.66 m cell, which is larger than a
    // leaf blob. Every blob collapsed to a handful of vertices and the canopy
    // became dark angular shards. It was not the simplifier being bad; a blob
    // is 56 triangles and there is nothing in it to remove.
    //
    // The reduction a canopy actually admits is FEWER, BIGGER BLOBS -- which
    // keeps the volume and the silhouette, the only things that survive to a
    // distant pixel. That is a generation parameter, not a mesh operation, so
    // the cheap versions are grown rather than crushed.
    constexpr int kForestLods = 3;
    struct ForestCell { eng::Mesh trunk, leaves, ground; };
    std::vector<ForestCell> cells(kForestLods * kForestCells * kForestCells);
    // ONE SPHERE PER CANOPY, kept for the light bake below. The forest is 2.5M
    // triangles and tracing rays against that is hopeless, but GI does not need
    // the shape of a leaf -- it needs to know that the sky is blocked here and
    // that what bounces off it is green. A coarse sphere per tree says both, in
    // a hundredth of the geometry.
    struct CanopyProxy { eng::Vec3 centre; float radius; };
    std::vector<CanopyProxy> canopies;
    // A CAPSULE PER TRUNK. Two hundred trees and not one of them was solid --
    // the physics world held the terrain and nothing else, so the forest was a
    // painting you walked through. Recorded here and added once the world
    // exists, because planting happens first.
    struct TrunkCollider { eng::Vec3 base; float radius, height; };
    std::vector<TrunkCollider> trunks;
    {
        // sides and segments take no random draws, and the leaf stream is
        // separate from the skeleton's, so all four of these can move without
        // moving a single branch. tree_test asserts exactly that -- a cheap
        // tree that stood somewhere else would pop sideways when it swapped.
        static const int kSides[kForestLods] = {7, 6, 4};
        static const int kSegments[kForestLods] = {4, 3, 1};
        static const int kClusters[kForestLods] = {5, 3, 1};
        // Bigger as they get fewer, so the canopy keeps roughly its volume:
        // five at 0.52 and one at 0.95 are within a fifth of each other cubed.
        static const float kLeafSize[kForestLods] = {0.52f, 0.63f, 0.95f};
        eng::Tree variants[kForestLods][6];
        for (int lod = 0; lod < kForestLods; ++lod)
            for (int i = 0; i < 6; ++i) {
                eng::TreeParams tp;
                tp.seed = 7919u + std::uint32_t(i) * 104729u;
                tp.height = 4.4f + float(i) * 0.5f;
                tp.trunk_radius = 0.17f;
                tp.levels = 5;
                tp.splits = 2;
                tp.spread = 0.52f;
                tp.leaf_scatter = 0.42f;
                tp.sides = kSides[lod];
                tp.segments = kSegments[lod];
                tp.leaf_clusters = kClusters[lod];
                tp.leaf_size = kLeafSize[lod];
                variants[lod][i] = eng::MakeTree(tp);
            }

        // The same xorshift the generator uses, so the forest is identical on
        // every machine and every run -- a screenshot comparison is worthless
        // otherwise.
        std::uint32_t rs = 20260904u;
        const auto rnd = [&] {
            rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5;
            return float(rs & 0xFFFFFFu) / float(0x1000000u);
        };

        int planted = 0, rejected_slope = 0, rejected_clearing = 0;
        for (int attempt = 0; attempt < 2000 && planted < 200; ++attempt) {
            const float x = (rnd() - 0.5f) * 118.0f;
            const float z = (rnd() - 0.5f) * 118.0f;
            // A CLEARING around the origin, because that is where the
            // character spawns and where the camera starts. It has to be wider
            // than the camera's orbit distance or the camera itself stands
            // inside the treeline -- which at 12 m it did, and the whole first
            // shot was one trunk filling the frame.
            if (x * x + z * z < 17.0f * 17.0f) { ++rejected_clearing; continue; }
            // AND A CLEARING ROUND EVERY DISTRICT. Trees are planted anywhere
            // the slope allows, which without this means through the middle of
            // a colonnade and over the gallery. Two effects, both bad: the
            // thing cannot be seen, and it sits in permanent canopy shade so
            // the meter opens two stops for it and blows out every sunlit patch
            // in the same frame.
            {
                bool in_district = false;
                for (const auto& d : {world::kGallery, world::kLanternHall,
                                      world::kGlassPavilion, world::kFirePit,
                                      world::kFlag}) {
                    const float dx = x - d.x, dz = z - d.z;
                    // 20 m, not 15. A stop leaves the character 9 m from the
                    // middle of a district and the camera 9 m behind that, so
                    // the CAMERA sits 18 m out -- outside a 15 m clearing, with
                    // a trunk in front of it as often as not.
                    if (dx * dx + dz * dz < 20.0f * 20.0f) in_district = true;
                }
                if (in_district) { ++rejected_clearing; continue; }
            }
            // NOT ON A CLIFF. A tree planted on a steep face floats with its
            // roots in the air on the downhill side, which is the single most
            // obvious scattering artefact and the cheapest one to avoid.
            if (terrain.NormalAt(x, z).y < 0.86f) { ++rejected_slope; continue; }

            const int which = int(rnd() * 6.0f) % 6;
            const eng::Tree& t = variants[0][which];
            const float scale = 0.75f + rnd() * 0.7f;
            // Sunk slightly, so the trunk's flat base is never visible above a
            // terrain triangle that slopes away from it.
            const eng::Vec3 at{x, terrain.HeightAt(x, z) - 0.15f * scale, z};
            // A LEAN, per tree, in the model matrix.
            //
            // Every trunk was exactly vertical, which is one of the clearest
            // marks of a generated forest -- real trees lean, toward light and
            // away from weather and downhill on a slope, and a stand of perfect
            // verticals reads as a diagram of a wood.
            //
            // Here rather than in TreeParams because the six skeletons are
            // generated once and shared by two hundred trees: putting it in the
            // generator would mean a generation per tree. Rotating about the
            // base leaves the trunk planted where it was and tilts everything
            // above it, which is what leaning is.
            const float lean = rnd() * 0.13f;
            const float lean_dir = rnd() * 6.2831853f;
            const eng::Mat4 model = eng::Mat4::Translation(at) *
                                    eng::Mat4::RotationY(lean_dir) *
                                    eng::Mat4::RotationX(lean) *
                                    eng::Mat4::RotationY(rnd() * 6.2831853f) *
                                    eng::Mat4::Scale(scale);
            // A tint per tree, so the canopy is not one flat green across the
            // whole valley. Warmer on some, cooler on others.
            const float warm = 0.88f + rnd() * 0.30f;
            const int gx = std::clamp(int((x + 64.0f) / 128.0f * kForestCells), 0,
                                      kForestCells - 1);
            const int gz = std::clamp(int((z + 64.0f) / 128.0f * kForestCells), 0,
                                      kForestCells - 1);
            for (int lod = 0; lod < kForestLods; ++lod) {
                ForestCell& cell = cells[std::size_t(
                    (lod * kForestCells + gz) * kForestCells + gx)];
                eng::AppendTransformed(cell.trunk, variants[lod][which].trunk, model,
                                       eng::Vec4{1.0f, 1.0f, 1.0f, 1.0f});
                eng::AppendTransformed(cell.leaves, variants[lod][which].foliage, model,
                                       eng::Vec4{warm, 1.0f, 2.0f - warm, 1.0f});
            }
            const eng::Vec4 c = model * eng::Vec4{t.foliage.bounds.center.x,
                                                  t.foliage.bounds.center.y,
                                                  t.foliage.bounds.center.z, 1.0f};
            // 0.45 OF THE BOUNDING RADIUS, and that is not a fudge. The
            // bounding sphere of a crown is 10 metres on these trees -- they
            // are 15 to 20 metres tall -- and a crown is not a solid ball, it
            // is a scattered cloud of leaf blobs that passes most of the light
            // reaching it. Filling the bound solid put 200 opaque 10 m spheres
            // over a 128 m field and blocked the sky almost everywhere:
            // irradiance in an open clearing came out at 0.009 against the
            // 0.183 an unoccluded bake gives.
            //
            // A sphere at 0.45 covers a fifth of the bound's projected area,
            // which is about what a crown's leaves actually subtend. The honest
            // fix is a transmittance on GiTriangle so a canopy can pass light
            // rather than either blocking it or not; this is the cheap version
            // of the same idea and it is the radius, not the model, that was
            // wrong.
            canopies.push_back(
                {eng::Vec3{c.x, c.y, c.z}, t.foliage.bounds.radius * scale * 0.45f});
            // Only the trunk, and only the part a person can walk into. The
            // branches above head height are not worth a collider each, and the
            // canopy is not something you bump into on the ground.
            trunks.push_back({at, 0.17f * scale * 1.25f, 4.4f * scale});
            ++planted;
        }
        // --- undergrowth ---------------------------------------------------
        //
        // Into the SAME cells as the trees, so it culls and drops detail with
        // them for free rather than needing a second system that does the same
        // thing.
        //
        // LEVEL 0 ONLY. A tuft is a few centimetres across; past thirty metres
        // it is smaller than a pixel and all it can do is alias. Carrying it
        // into the far levels would be tens of thousands of triangles a cell
        // producing nothing but shimmer.
        {
            std::uint32_t gs = 918273u;
            const auto grnd = [&] {
                gs ^= gs << 13; gs ^= gs >> 17; gs ^= gs << 5;
                return float(gs & 0xFFFFFFu) / float(0x1000000u);
            };
            world::Rng rng(5573u);
            int tufts = 0, rocks = 0;
            for (int attempt = 0; attempt < 52000; ++attempt) {
                const float x = (grnd() - 0.5f) * 124.0f;
                const float z = (grnd() - 0.5f) * 124.0f;
                const eng::Vec3 nrm = terrain.NormalAt(x, z);
                const float y = terrain.HeightAt(x, z);
                const int gx = std::clamp(int((x + 64.0f) / 128.0f * kForestCells), 0,
                                          kForestCells - 1);
                const int gz = std::clamp(int((z + 64.0f) / 128.0f * kForestCells), 0,
                                          kForestCells - 1);
                eng::Mesh& into = cells[std::size_t(gz * kForestCells + gx)].ground;
                if (grnd() < 0.88f) {
                    // Grass thins out on anything steep, for the same reason
                    // the ground goes bare there.
                    if (nrm.y < 0.90f) continue;
                    // A little below the surface, so a tuft on a slope has its
                    // base buried rather than floating at one corner.
                    world::AppendTuft(into, rng, eng::Vec3{x, y - 0.03f, z},
                                      0.26f + rng.Unit() * 0.22f,
                                      eng::Vec4{0.62f, 1.05f, 0.42f, 1.0f});
                    ++tufts;
                } else {
                    if (nrm.y < 0.55f) continue;
                    const float size = 0.10f + rng.Unit() * 0.30f;
                    // Sunk by a third: a boulder resting exactly on the surface
                    // looks dropped, one partly in the ground looks weathered.
                    // The vertex colour has to UNDO the material's green. One
                    // material serves grass and stone, and its base colour is
                    // tuned for grass -- multiplying a grey through it gave
                    // dark green pebbles. These numbers put stone back at a
                    // neutral 0.17-ish whatever the base is.
                    const float grey = 0.85f + rng.Unit() * 0.5f;
                    world::AppendRock(into, rng, eng::Vec3{x, y - size * 0.33f, z}, size,
                                      eng::Vec4{grey * 1.38f, grey * 1.13f,
                                                grey * 1.78f, 1.0f});
                    ++rocks;
                }
            }
            std::size_t gtris = 0;
            for (const ForestCell& c : cells) gtris += c.ground.indices.size() / 3;
            std::printf("  %d tufts and %d rocks, %zu tris\n", tufts, rocks, gtris);
        }

        std::size_t per_lod[kForestLods] = {0, 0, 0}, used = 0;
        for (int lod = 0; lod < kForestLods; ++lod)
            for (int c = 0; c < kForestCells * kForestCells; ++c) {
                const ForestCell& cell =
                    cells[std::size_t(lod * kForestCells * kForestCells + c)];
                per_lod[lod] +=
                    (cell.trunk.indices.size() + cell.leaves.indices.size()) / 3;
                if (lod == 0 && !cell.trunk.vertices.empty()) ++used;
            }
        std::printf("  %d trees over %zu of %d cells, %zu / %zu / %zu tris per "
                    "level, rejected %d steep, %d in the clearing\n",
                    planted, used, kForestCells * kForestCells, per_lod[0],
                    per_lod[1], per_lod[2], rejected_slope, rejected_clearing);
    }

    eng::MaterialDesc bark_md;
    bark_md.shading = eng::Shading::Lit;
    bark_md.base_color = eng::Vec4{0.30f, 0.22f, 0.16f, 1.0f};
    bark_md.roughness = 0.88f;
    bark_md.albedo = bark_albedo;
    bark_md.normal_map = bark_normal;
    // Deep, because bark is: the grooves are the whole reason a trunk does not
    // read as a brown cylinder.
    bark_md.normal_strength = 1.6f;
    // u runs around the trunk and v along it. One trunk is roughly a metre
    // around and several long, so v repeats more often than u or the grain
    // comes out stretched into smears.
    bark_md.uv_scale = eng::Vec2{1.5f, 3.0f};
    const eng::MaterialHandle bark_mat = app->Draw().CreateMaterial(bark_md, error);
    if (!eng::Valid(bark_mat)) return Fail(error);

    // ONE MATERIAL for grass and stone both. They differ in vertex colour, and
    // the shader multiplies that into the albedo -- two materials would be two
    // draws per cell for two things that shade identically.
    eng::MaterialDesc scatter_md;
    scatter_md.shading = eng::Shading::Lit;
    scatter_md.base_color = eng::Vec4{0.13f, 0.15f, 0.09f, 1.0f};
    scatter_md.roughness = 0.93f;
    const eng::MaterialHandle scatter_mat = app->Draw().CreateMaterial(scatter_md, error);
    if (!eng::Valid(scatter_mat)) return Fail(error);

    eng::MaterialDesc leaf_md;
    leaf_md.shading = eng::Shading::Lit;
    // Tried 0.30 green against this 0.38 and reverted: the frontlit tops sit
    // at the tone-map ceiling either way (canopy green p90 224 vs 221,
    // measured on the clearing camera), so albedo is not the lever up there
    // -- and the darker leaves pushed 30 more GI probes under the dark
    // threshold (96 to 126 buried) for those three codes. A bad trade.
    leaf_md.base_color = eng::Vec4{0.19f, 0.38f, 0.15f, 1.0f};
    // Leaves are rougher than almost anything else outdoors and not at all
    // metallic; a smooth canopy picks up a sheen that reads as wet plastic.
    leaf_md.roughness = 0.95f;
    leaf_md.albedo = leaf_albedo;
    // No normal map: a leaf blob already stands in for a thousand leaves, and
    // bump detail on it would be detail at the wrong scale pretending to be
    // detail at the right one.
    leaf_md.uv_scale = eng::Vec2{2.0f, 2.0f};
    // THE THING THAT MAKES FOLIAGE FOLIAGE. A leaf with the sun behind it is
    // the brightest object in a forest; a purely reflective BRDF renders it
    // black, because dot(N, L) is negative there. Until this existed the whole
    // canopy was flat green from every angle and read as plastic.
    leaf_md.transmission = leaf_transmission;
    // Green, and much more saturated than the albedo. A leaf passes green far
    // better than red -- which is why a backlit leaf is a more vivid green than
    // a lit one, and using the albedo here would lose exactly that.
    leaf_md.transmission_color = eng::Vec3{0.22f, 0.62f, 0.10f};
    const eng::MaterialHandle leaf_mat = app->Draw().CreateMaterial(leaf_md, error);
    if (!eng::Valid(leaf_mat)) return Fail(error);

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
                eng::Mesh m = terrain.BuildChunk(cx, cz, lod);
                // THE GROUND IS NOT ONE THING. It was: a single albedo across
                // 128 metres, so every slope, every hollow and every ridge was
                // the same green. That is the clearest tell that a landscape
                // was generated -- real ground changes with the shape it is on,
                // because what grows and what washes away depend on the slope.
                //
                // Done in the VERTEX COLOUR because the shader already
                // multiplies it into the albedo, so it costs nothing and needs
                // no engine change. The cost is that the blend is only as sharp
                // as the mesh, which for a 0.5 m grid is finer than the
                // transition wants to be anyway.
                for (VertexIn& v : m.vertices) {
                    // Steepness, 0 flat to 1 vertical.
                    const float steep = std::clamp(1.0f - v.normal.y, 0.0f, 1.0f);
                    // Grass gives way to bare earth on anything that sheds
                    // water, and to pale rock where it is steep enough that
                    // nothing holds at all.
                    const float earth = std::clamp((steep - 0.06f) / 0.16f, 0.0f, 1.0f);
                    const float rock = std::clamp((steep - 0.26f) / 0.22f, 0.0f, 1.0f);
                    // And a slow drift with height, so the valley floor is
                    // lusher than the rim: cold and thin soil up top.
                    const float high = std::clamp((v.position.y + 2.0f) / 18.0f,
                                                  0.0f, 1.0f);
                    const auto mix3 = [](eng::Vec3 a, eng::Vec3 b, float t) {
                        return eng::Vec3{a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t,
                                         a.z + (b.z - a.z) * t};
                    };
                    eng::Vec3 c{1.0f, 1.0f, 1.0f};
                    c = mix3(c, eng::Vec3{2.05f, 1.55f, 0.95f}, earth);
                    c = mix3(c, eng::Vec3{2.60f, 2.45f, 2.25f}, rock);
                    c = mix3(c, eng::Vec3{1.15f, 1.05f, 0.80f}, high * 0.5f);
                    // PATCHINESS, independent of shape. Slope and height vary
                    // the ground with the land, but a flat clearing is neither
                    // steep nor high and still read as one green -- a real
                    // meadow is blotchy at every scale. Two slow octaves, 20 m
                    // and 7 m, far above the 3 m texture tile so they add a
                    // scale the texture cannot, and non-repeating so there is
                    // no tiling to spot. Dry patches run yellow: red is kept
                    // while blue falls, which is what dry grass does.
                    const float blotch =
                        (ValueNoise(v.position.x * 0.045f, v.position.z * 0.045f) - 0.5f) * 0.55f +
                        (ValueNoise(v.position.x * 0.13f + 7.3f, v.position.z * 0.13f + 3.1f) - 0.5f) *
                            0.30f;
                    c.x *= 1.0f + blotch;
                    c.y *= 1.0f + blotch * 0.55f;
                    c.z *= 1.0f - blotch * 0.45f;
                    v.color = eng::Vec4{c.x, c.y, c.z, 1.0f};
                }
                // The terrain builder does not make tangents, and the ground
                // now has a normal map. Without a frame the shader falls back
                // to the geometric normal and the map does nothing at all --
                // silently, which is the worst way for it to not work.
                eng::GenerateTangents(m);
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

    // The trunks, now that there is a world to put them in.
    for (const TrunkCollider& t : trunks) {
        eng::physics::Body b;
        // A capsule and not a box: a character brushing a round trunk should
        // slide round it, and a box catches on its corners.
        b.shape = eng::physics::Shape::MakeCapsule(t.radius, t.height * 0.5f);
        b.position = t.base + eng::Vec3{0.0f, t.height * 0.5f, 0.0f};
        b.SetMass(0.0f);
        world.Add(b);
    }
    std::printf("  %zu trunk colliders\n", trunks.size());

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
            for (std::uint32_t i : m.indices) nav_indices.push_back(base + i);
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

    // Declared here rather than with the rest of the frame state below: the
    // light bake needs the sun before the first frame runs, and a bake done
    // against a different sun than the frame uses lights the scene as though
    // it were two times of day at once.
    eng::SkyConfig sky;
    float sun_azimuth = 1.1f;
    float sun_elevation = sun_start / 57.2958f;

    // --- baked indirect light --------------------------------------------------
    //
    // The shadows were lit by a constant hemisphere term, which is why they were
    // flat: every shaded pixel in the scene got the same ambient regardless of
    // whether it was in the open, under a tree, or in a hollow. That reads as a
    // renderer, not as a forest.
    //
    // What a volume adds here is two things the constant cannot: the sky is
    // OCCLUDED under the canopy, so it gets darker where a tree is over it, and
    // the ground BOUNCES, so shaded grass is lit by green light from the sunlit
    // grass beside it rather than by grey.
    //
    // THE PROXY GEOMETRY is the coarse terrain plus one sphere per canopy. The
    // real forest is 2.5M triangles and the bake traces hundreds of thousands
    // of rays; the sphere says "sky blocked here, and green" which is the whole
    // of a tree's contribution to indirect light.
    // A LAMBDA, not a block that runs once, because the volume goes STALE the
    // moment the sun moves: it holds where the light was, shadowed by geometry
    // that has not moved, and nothing about it decays or warns. Baking once at
    // startup is only correct for a scene whose sun is nailed down, which this
    // one was until it got a key to move it.
    double last_bake_seconds = 0.0;
    int last_bake_dark = 0;
    // Mean radiance of the sky's four horizontal faces, refreshed by every
    // bake, so the water's reflection follows the sun. See the read below.
    eng::Vec3 sky_horizon{0.55f, 0.62f, 0.72f};
    const auto bake_indirect = [&]() -> bool {
        // The sun and sky the bake assumes have to be the ones the frame uses,
        // or the indirect light disagrees with the direct light and the scene
        // looks lit by two different times of day.
        sky.sun_direction = eng::Vec3{std::cos(sun_azimuth) * std::cos(sun_elevation),
                                      std::sin(sun_elevation),
                                      std::sin(sun_azimuth) * std::cos(sun_elevation)};
        eng::Environment::ApplyTo(&scene, sky);

        // THE SKY THE BAKE USES HAS TO BE THE SKY THE FRAME USES, and
        // scene.ambientSky is not it: ApplyTo says in its own comment that the
        // hemisphere term is a FALLBACK for paths without image-based lighting,
        // a hand-fitted constant "scaled to roughly what the sky contributes".
        // The forward path here reads the irradiance cube instead, so feeding
        // the bake the fallback made the indirect light disagree with the
        // direct -- it was four times too dim.
        //
        // So: bake the sky and read the irradiance cube back. The +Y and -Y
        // faces are the irradiance on a level surface facing up and down, which
        // is exactly what the bake wants for sky_top and sky_bottom.
        //
        // NOT corrected for the solar disc, and that needs saying because the
        // cube does contain one. kSunDiscGain is deliberately set two hundred
        // times below the true radiance-per-irradiance ratio so that the sun is
        // NOT counted twice -- once by the directional light and once by the
        // environment. At that gain the disc puts about 0.05 units into this
        // face against the sky's 1.7, and subtracting the directional light
        // instead, which is what this did first, takes off 8.3 and leaves zero.
        //
        // THIS OPENS A DEVICE FRAME OF ITS OWN, so it must not be called from
        // inside one. BeginFrame takes one of the kFramesInFlight permits and
        // starts a command buffer; CommitAndWait hands the permit back through
        // the completion handler and clears the buffer. Called from inside the
        // app's frame, the second BeginFrame takes a SECOND permit and replaces
        // the app's command buffer, which is then never committed -- so that
        // permit is never returned, and the rest of that frame records into a
        // nil buffer and is thrown away. Three sun moves leak three permits and
        // the next BeginFrame blocks forever: the app freezes with no error.
        // The caller in the frame loop defers to the top of the next iteration
        // for exactly this reason.
        eng::Vec3 sky_up{0.0f, 0.0f, 0.0f}, sky_down{0.0f, 0.0f, 0.0f};
        {
            constexpr int kFace = 4;
            app->Gpu().BeginFrame();
            {
                auto e = app->Gpu().BeginCompute("skybake");
                env->BakeSky(e, sky);
                env->ReadCube(e, eng::Environment::Probe::Irradiance, kFace, 0.0f);
                app->Gpu().EndCompute();
            }
            if (!app->Gpu().CommitAndWait(error)) return false;
            const std::vector<eng::Vec4> cube = env->TakeCube();
            const auto face_mean = [&](int face) {
                eng::Vec3 sum{0.0f, 0.0f, 0.0f};
                const int n = kFace * kFace;
                for (int i = 0; i < n; ++i) {
                    const eng::Vec4& v = cube[std::size_t(face * n + i)];
                    sum = sum + eng::Vec3{v.x, v.y, v.z};
                }
                return sum * (1.0f / float(n));
            };
            if (cube.size() >= std::size_t(6 * kFace * kFace)) {
                sky_up = face_mean(2);    // +Y
                sky_down = face_mean(3);  // -Y
                // THE HORIZON, for the water to reflect. The four side faces,
                // averaged, because a basin reflects whichever way it is looked
                // at from. Read here rather than invented as a constant: the
                // spring's water has to reflect the sky the atmosphere model is
                // actually producing, or it goes on looking like noon at dusk.
                //
                // The first attempt used scene.ambientSky, which is the wrong
                // quantity entirely -- ApplyTo's own comment calls it a
                // hand-fitted FALLBACK for paths without image-based lighting.
                // At (0.215, 0.266, 0.367) against this face's value it is
                // about a third as bright, and water reflecting something
                // nearly black is a dark jelly rather than water.
                sky_horizon = (face_mean(0) + face_mean(1) + face_mean(4) +
                               face_mean(5)) * 0.25f;
            }
        }
        // TIMES PI for the printed sky figure. The cube holds cosine-weighted
        // MEAN RADIANCE, which is what the shader wants to multiply an albedo
        // by; irradiance is pi times it. The GI bake below is fed the raw value
        // on purpose -- its own convention is that a constant environment
        // integrates to itself -- but comparing it against the SUN, which is an
        // irradiance, needs the conversion or the ratio comes out three times
        // too small.
        {
            const float sun_e = scene.lightColor.y * std::max(sky.sun_direction.y, 0.0f);
            const float sky_e = sky_up.y * 3.14159265f;
            std::printf("    sun %.2f, sky %.2f (%.1f:1) at %.0f degrees\n", sun_e,
                        sky_e, double(sun_e / std::max(sky_e, 1e-4f)),
                        sun_elevation * 57.2958f);
        }
        std::vector<eng::GiTriangle> soup;
        const eng::Vec3 grass{ground_md.base_color.x, ground_md.base_color.y,
                              ground_md.base_color.z};
        for (std::size_t i = 0; i + 2 < nav_indices.size(); i += 3)
            soup.push_back({nav_vertices[nav_indices[i]], nav_vertices[nav_indices[i + 1]],
                            nav_vertices[nav_indices[i + 2]], grass});
        const std::size_t ground_tris = soup.size();

        const eng::Mesh proxy = eng::MakeUVSphere(1.0f, 6, 8, eng::Vec4{1, 1, 1, 1},
                                                  eng::Vec4{1, 1, 1, 1});
        const eng::Vec3 leafy{leaf_md.base_color.x, leaf_md.base_color.y,
                              leaf_md.base_color.z};
        for (const CanopyProxy& c : canopies)
            for (std::size_t i = 0; i + 2 < proxy.indices.size(); i += 3) {
                const auto at = [&](std::size_t k) {
                    const VertexIn& v = proxy.vertices[proxy.indices[i + k]];
                    return eng::Vec3{c.centre.x + v.position.x * c.radius,
                                     c.centre.y + v.position.y * c.radius,
                                     c.centre.z + v.position.z * c.radius};
                };
                soup.push_back({at(0), at(1), at(2), leafy});
            }

        eng::GiBakeConfig gi;
        // 8 metres between probes horizontally and 5 vertically, over the
        // playable bowl. Finer would resolve the shade under individual trees,
        // and at 8 m it resolves the shade under a STAND of them -- which is
        // the scale the light actually varies on outdoors, and a sixteenth of
        // the bake time.
        gi.nx = 17;
        gi.ny = 7;
        gi.nz = 17;
        gi.origin = eng::Vec3{-64.0f, -10.0f, -64.0f};
        gi.spacing = eng::Vec3{8.0f, 5.0f, 8.0f};
        gi.rays = 96;
        gi.bounces = 2;
        gi.sun_direction = sky.sun_direction;
        gi.sun_color = eng::Vec3{scene.lightColor.x, scene.lightColor.y,
                                 scene.lightColor.z};
        gi.sky_top = sky_up;
        gi.sky_bottom = sky_down;
        gi.threads = int(std::thread::hardware_concurrency());

        std::printf("baking indirect light (%d probes, %zu ground + %zu canopy tris)...\n",
                    gi.nx * gi.ny * gi.nz, ground_tris, soup.size() - ground_tris);
        const auto t0 = std::chrono::steady_clock::now();
        eng::IrradianceVolume volume = eng::IrradianceVolume::Bake(soup, gi);
        const int dark = volume.DarkProbes();
        // A probe grid is a box and terrain is not, so a large fraction of this
        // one is underground and bakes to zero. Left alone they drag every
        // shaded surface interpolating against them toward black -- which is
        // exactly what happened the first time this ran: the forest floor went
        // from flat to nearly unlit and the meter opened two more stops trying
        // to compensate.
        const int filled = volume.FillDark();
        const double secs =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (!app->Draw().SetIrradianceVolume(volume, error)) return false;
        last_bake_seconds = secs;
        last_bake_dark = dark;
        std::printf("  %.2f s, %d of %d probes were buried, %d filled from neighbours\n",
                    secs, dark, gi.nx * gi.ny * gi.nz, filled);
        {
            // The two numbers worth printing: what the volume gives in the open
            // and what it gives under the trees. If the first is far below an
            // unoccluded bake the proxy geometry is wrong, and if the second is
            // not clearly below the first the trees are not occluding anything.
            const eng::Vec3 up{0.0f, 1.0f, 0.0f};
            const eng::Vec3 open = volume.Sample(
                eng::Vec3{0.0f, terrain.HeightAt(0.0f, 0.0f) + 0.2f, 0.0f}, up);
            const eng::Vec3 under = volume.Sample(
                eng::Vec3{34.0f, terrain.HeightAt(34.0f, 34.0f) + 0.2f, 34.0f}, up);
            std::printf("  irradiance in the open (%.3f %.3f %.3f), "
                        "under canopy (%.3f %.3f %.3f)\n",
                        open.x, open.y, open.z, under.x, under.y, under.z);
        }
        return true;
    };
    if (!bake_indirect()) return Fail(error);

    // One instance per cell per material, with a LEVEL CHAIN behind each.
    //
    // The model matrix is identity throughout -- AppendTransformed baked the
    // trees into world space -- so a cell's mesh bounds ARE its world bounds
    // and the renderer's frustum cull works on them with nothing else to set
    // up. That is the whole reason for splitting the mesh.
    struct ForestChunk {
        std::vector<eng::MeshHandle> trunk_lods, leaf_lods;
        eng::MeshHandle ground;  // undergrowth, level 0 only
        eng::Vec3 centre;
        std::size_t trunk_instance = 0, leaf_instance = 0, ground_instance = 0;
    };
    std::vector<ForestChunk> forest;
    {
        std::printf("building forest levels...\n");
        const auto t0 = std::chrono::steady_clock::now();
        std::size_t coarse_tris = 0, fine_tris = 0;
        const int kCellCount = kForestCells * kForestCells;
        for (int c = 0; c < kCellCount; ++c) {
            const ForestCell& base = cells[std::size_t(c)];
            if (base.trunk.vertices.empty() && base.leaves.vertices.empty()) continue;
            ForestChunk chunk;
            chunk.centre = base.leaves.vertices.empty() ? base.trunk.bounds.center
                                                        : base.leaves.bounds.center;
            for (int lod = 0; lod < kForestLods; ++lod) {
                const ForestCell& cell = cells[std::size_t(lod * kCellCount + c)];
                chunk.trunk_lods.push_back(app->Draw().UploadMesh(cell.trunk));
                chunk.leaf_lods.push_back(app->Draw().UploadMesh(cell.leaves));
                if (lod == 0) fine_tris += cell.leaves.indices.size() / 3;
                if (lod == kForestLods - 1) coarse_tris += cell.leaves.indices.size() / 3;
            }
            eng::Instance ti;
            ti.mesh = chunk.trunk_lods.front();
            ti.material = bark_mat;
            chunk.trunk_instance = scene.instances.size();
            scene.instances.push_back(ti);
            eng::Instance li;
            li.mesh = chunk.leaf_lods.front();
            li.material = leaf_mat;
            chunk.leaf_instance = scene.instances.size();
            scene.instances.push_back(li);
            if (!base.ground.vertices.empty()) {
                chunk.ground = app->Draw().UploadMesh(base.ground);
                eng::Instance gi;
                gi.mesh = chunk.ground;
                gi.material = scatter_mat;
                chunk.ground_instance = scene.instances.size();
                scene.instances.push_back(gi);
            }
            forest.push_back(std::move(chunk));
        }
        std::printf("  %zu cells, %zu draws, canopy %zu tris at level 0 and %zu "
                    "at level 2 (%.2f s)\n",
                    forest.size(), forest.size() * 2, fine_tris, coarse_tris,
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count());
    }

    // --- the districts ---------------------------------------------------------
    //
    // One valley with a place in it for each thing the engine does, instead of
    // a demo apiece. See districts.h for why that is the arrangement rather
    // than a convenience.
    const eng::MeshHandle unit_sphere = app->Draw().UploadMesh(
        eng::MakeUVSphere(0.5f, 24, 32, eng::Vec4{1, 1, 1, 1}, eng::Vec4{1, 1, 1, 1}));
    world::BuildGallery(app->Draw(), scene, world, unit_sphere,
                        terrain.HeightAt(world::kGallery.x, world::kGallery.z), error);
    const world::LanternHall hall = world::BuildLanternHall(
        app->Draw(), scene, world, unit_sphere,
        terrain.HeightAt(world::kLanternHall.x, world::kLanternHall.z), error);
    world::BuildGlassPavilion(
        app->Draw(), scene, world,
        terrain.HeightAt(world::kGlassPavilion.x, world::kGlassPavilion.z), error);
    world::BuildFirePit(app->Draw(), scene, world, unit_sphere,
                        decals_on ? soot_tex : eng::rhi::TextureId{},
                        terrain.HeightAt(world::kFirePit.x, world::kFirePit.z),
                        [&](float x, float z) { return terrain.HeightAt(x, z); }, error);
    const world::Banner banner = world::BuildBanner(
        app->Draw(), scene, world, terrain.HeightAt(world::kFlag.x, world::kFlag.z), error);
    const world::Spring spring = world::BuildSpring(
        app->Draw(), scene, world,
        terrain.HeightAt(world::kSpring.x, world::kSpring.z), error);
    if (!error.empty()) return Fail(error);

    // --- the water -----------------------------------------------------------
    //
    // Real smoothed-particle hydrodynamics, in the basin BuildSpring just laid
    // out. The solver's box is the basin's inner faces; the seeding fills it to
    // just under the rim on a lattice at the rest spacing, which is what lets
    // the solver derive each particle's mass from the rest density.
    //
    // The spacing is twice apps/fluid's and so is the smoothing radius, because
    // this basin is twice that tank in every dimension and SPH costs volume:
    // keeping the tank's numbers here would be eight times the particles for a
    // fluid nobody would look at any closer.
    eng::FluidConfig fcfg;
    fcfg.bounds_min = spring.bounds_min;
    fcfg.bounds_max = spring.bounds_max;
    fcfg.smoothing_radius = 0.14f;
    fcfg.viscosity = 0.30f;
    std::vector<eng::Vec3> seed;
    {
        constexpr float kSpacing = 0.07f;
        for (float x = spring.bounds_min.x + kSpacing; x < spring.bounds_max.x - kSpacing * 0.5f; x += kSpacing)
            for (float y = spring.bounds_min.y + kSpacing * 0.5f; y < spring.fill_y; y += kSpacing)
                for (float z = spring.bounds_min.z + kSpacing; z < spring.bounds_max.z - kSpacing * 0.5f; z += kSpacing)
                    seed.push_back(eng::Vec3{x, y, z});
    }
    auto water = eng::FluidSim::Create(app->Gpu(), fcfg, seed,
                                       eng::Renderer::kSceneFormat, error, 1);
    if (!water) return Fail(error);
    auto water_surface = eng::FluidSurface::Create(app->Gpu(), error);
    if (!water_surface) return Fail(error);
    std::printf("spring: %zu water particles in a %.1f x %.1f x %.1f m basin\n",
                seed.size(), spring.bounds_max.x - spring.bounds_min.x,
                spring.bounds_max.y - spring.bounds_min.y,
                spring.bounds_max.z - spring.bounds_min.z);

    // CLUSTERED LIGHTING ON, because 60 lanterns is well past the point where
    // a forward pass can loop over every light for every fragment -- and
    // because without it the lights are in the scene and contribute nothing at
    // all. They were, for the first build of this: sixty lights, no light.
    app->Draw().SetClusteredLighting(true);
    std::printf("districts: %zu instances, %zu lights\n", scene.instances.size(),
                scene.lights.size());

    // --- the crowd -------------------------------------------------------------
    //
    // The GPU-driven showcase (#3): free-standing boulders as INDEPENDENT
    // instances of the one sphere mesh and one stone material, scattered in
    // rings around the clearing. The ordinary path submits one draw per
    // boulder; CullScene puts the whole lot into one batch and the scene pass
    // draws it with one indirect draw. Fixed seed, so --crowd 600 looks the
    // same every run and screenshots compare.
    if (crowd_count > 0) {
        eng::MaterialDesc crowd_md;
        crowd_md.shading = eng::Shading::Lit;
        crowd_md.base_color = eng::Vec4{0.42f, 0.41f, 0.38f, 1.0f};
        crowd_md.roughness = 0.92f;
        const eng::MaterialHandle crowd_mat =
            app->Draw().CreateMaterial(crowd_md, error);
        if (!eng::Valid(crowd_mat)) return Fail(error);
        eng::MeshHandle crowd_mesh[2] = {unit_sphere, unit_sphere};
        {
            const std::string paths[2] = {rock_path, rock_path2};
            for (int r = 0; r < 2; ++r) {
                std::string rock_error;
                const eng::gltf::Document rock_doc = LoadAssetDoc(
                    ResolveAsset(paths[r], argc, argv), rock_error);
                const eng::MeshHandle rock_up =
                    rock_error.empty() && !rock_doc.Empty()
                        ? app->Draw().UploadMesh(rock_doc.primitives[0].mesh)
                        : eng::MeshHandle{};
                if (eng::Valid(rock_up)) {
                    crowd_mesh[r] = rock_up;
                    std::printf("crowd: loaded rock (%s)\n", paths[r].c_str());
                } else {
                    std::printf("crowd: unit sphere (%s)\n",
                                rock_error.empty() ? "rock unusable" : rock_error.c_str());
                }
            }
        }
        std::mt19937 rng(1234);
        std::uniform_real_distribution<float> unit(0.0f, 1.0f);
        const float kPi = 3.14159265f;
        for (int i = 0; i < crowd_count; ++i) {
            // Area-uniform rings: the clearing centre stays walkable, the
            // boulders thicken toward the treeline.
            const float angle = unit(rng) * 2.0f * kPi;
            const float radius =
                8.0f + 26.0f * std::sqrt(unit(rng));
            const float x = std::cos(angle) * radius;
            const float z = std::sin(angle) * radius;
            const float s = 0.5f + 1.6f * unit(rng);
            eng::Instance b;
            // Alternate the two rocks; either missing file falls back to its
            // sphere slot, so a half-present assets/ degrades gracefully.
            b.mesh = crowd_mesh[i % 2];
            b.material = crowd_mat;
            const float ground = terrain.HeightAt(x, z);
            // The rocks sit base-down (their files start at y = 0); the
            // sphere floated on its centre, hence the two offsets.
            const float yoff = (b.mesh == unit_sphere)
                                   ? 0.5f * s * 0.55f
                                   : -0.05f * s;
            b.model = eng::Mat4::Translation(eng::Vec3{x, ground + yoff, z}) *
                      eng::Mat4::RotationY(unit(rng) * 2.0f * kPi) *
                      eng::Mat4::Scale(s);
            const float shade = 0.82f + 0.30f * unit(rng);
            b.tint = eng::Vec4{shade, shade, shade * 0.99f, 1.0f};
            scene.instances.push_back(b);
        }
        std::printf("crowd: %d boulders\n", crowd_count);
    }

    // --- the character -------------------------------------------------------
    //
    // A skinned figure, generated the way everything else here is. It replaced
    // an orange sphere, and the sphere was hiding more than it looked like: a
    // sphere has no feet, so nothing in this demo had ever had to agree with
    // the ground at a point rather than on average, and it has no gait, so the
    // character's speed had never had to be a speed a body could move at.
    // The player is a fox when the asset resolves, the generated humanoid
    // otherwise. Either way the frame below only sees one skeleton, one
    // skinned mesh and three clips -- idle, walk, run.
    eng::anim::Skeleton skeleton = world::BuildHumanSkeleton();
    world::HumanBody figure = world::BuildHumanBody(skeleton);
    eng::anim::Clip clip_idle = world::MakeIdle();
    eng::anim::Clip clip_walk = world::MakeGait("walk", world::kWalkGait);
    eng::anim::Clip clip_run = world::MakeGait("run", world::kRunGait);
    // Humanoid values; the fox overrides both. Its file is in centimetres
    // against an engine in metres, and which way it faces is settled below.
    float figure_scale = 1.0f;
    float figure_yaw = 0.0f;
    bool figure_fox = false;
    float body_roughness = 0.72f;
    float body_metallic = 0.0f;
    eng::rhi::TextureId figure_tex{};
    {
        std::string fox_error;
        eng::gltf::Document fox_doc =
            LoadAssetDoc(ResolveAsset(fox_path, argc, argv), fox_error);
        const bool fox_usable =
            fox_error.empty() && !fox_doc.Empty() && !fox_doc.skins.empty() &&
            fox_doc.animations.size() >= 3 && !fox_doc.images.empty() &&
            !fox_doc.images[0].Empty() &&
            int(fox_doc.skins[0].skeleton.joints.size()) <=
                eng::Renderer::kMaxJoints;
        if (!fox_usable) {
            std::printf("figure: generated humanoid (%s)\n",
                        fox_error.empty() ? "fox unusable" : fox_error.c_str());
        } else {
            skeleton = fox_doc.skins[0].skeleton;
            figure.mesh = fox_doc.primitives[0].mesh;
            figure.skin = fox_doc.primitives[0].skin;
            // Survey, Walk, Run in the sample's order.
            clip_idle = fox_doc.MakeClip(0, 0);
            clip_walk = fox_doc.MakeClip(1, 0);
            clip_run = fox_doc.MakeClip(2, 0);
            figure_fox = !clip_walk.channels.empty();
            if (figure_fox) {
                figure_scale = 0.012f;
                // sRGB: a painted albedo is a COLOUR, like soot, not a linear
                // ratio like the terrain modulators.
                const eng::Texture2D& img = fox_doc.images[0];
                figure_tex = app->Gpu().CreateTexture2D(
                    img.width, img.height, img.rgba.data(), true, true);
                figure_fox = eng::rhi::Valid(figure_tex);
                if (!fox_doc.materials.empty()) {
                    // Applied to body_md below; kept here so the fallback
                    // path reads exactly as it did before the fox existed.
                    body_roughness = fox_doc.materials[0].roughness;
                    body_metallic = fox_doc.materials[0].metallic;
                }
            }
            std::printf("figure: %s (%s)\n", figure_fox ? "loaded fox" : "generated humanoid",
                        fox_path.c_str());
        }
    }
    const eng::MeshHandle capsule_mesh = app->Draw().UploadMesh(
        eng::MakeUVSphere(0.45f, 16, 20, eng::Vec4{1, 1, 1, 1}, eng::Vec4{1, 1, 1, 1}));
    const eng::MeshHandle figure_mesh = app->Draw().UploadSkinnedMesh(
        figure.mesh, figure.skin, int(skeleton.joints.size()));
    if (!eng::Valid(figure_mesh)) return Fail("skinned upload");
    eng::MaterialDesc body_md;
    body_md.shading = eng::Shading::Lit;
    // WHITE, because the mesh's vertex colours already carry the cloth and the
    // skin. A base colour here would multiply into both and turn the face the
    // colour of the coat.
    body_md.base_color = eng::Vec4{1.0f, 1.0f, 1.0f, 1.0f};
    body_md.roughness = body_roughness;
    body_md.metallic = body_metallic;
    if (figure_fox) {
        // No vertex colours on the fox; its painted texture is the albedo.
        body_md.albedo = figure_tex;
    }
    const eng::MaterialHandle body_mat = app->Draw().CreateMaterial(body_md, error);
    eng::Instance body;
    body.mesh = figure_mesh;
    body.material = body_mat;
    const std::size_t body_instance = scene.instances.size();
    scene.instances.push_back(body);

    // --- sound ---------------------------------------------------------------
    //
    // Generated, like everything else. A forest that makes no sound is not
    // quiet, it is switched off -- silence is the one thing a real place never
    // is, and the absence reads as the world not being finished rather than as
    // a missing feature.
    //
    // SILENT MODE FOR CAPTURES, rather than skipping audio entirely. The mixer
    // still runs and the voices are still counted, so a headless shot can
    // report that the fire is audible and the footsteps are triggering -- which
    // is the only way to check any of this without sitting and listening.
    std::string audio_error;
    std::unique_ptr<eng::audio::AudioSystem> audio =
        shot_path.empty() ? eng::audio::AudioSystem::Create(audio_error)
                          : eng::audio::AudioSystem::CreateSilent(48000);
    const int audio_rate = audio ? audio->SampleRate() : 48000;
    // FOUR footsteps, not one. A single step replayed at a constant interval is
    // heard as a machine however good the sample is, and the ear picks it out
    // from under everything else in the mix.
    std::vector<eng::audio::Clip> steps;
    for (std::uint32_t i = 0; i < 4; ++i)
        steps.push_back(eng::soundgen::Footstep(audio_rate, 101u + i * 37u, 0.8f));
    const eng::audio::Clip fire_clip = eng::soundgen::Fire(audio_rate, 13);
    const eng::audio::Clip wind_clip = eng::soundgen::Wind(audio_rate, 21, 6.0f, 0.30f);
    if (audio) {
        eng::audio::PlayDesc d;
        d.clip = &fire_clip;
        d.loop = true;
        d.spatial = true;
        d.gain = 2.6f;
        d.position = eng::Vec3{world::kFirePit.x,
                               terrain.HeightAt(world::kFirePit.x, world::kFirePit.z) + 0.4f,
                               world::kFirePit.z};
        // A fire is audible from further away than it is bright, and the near
        // radius is what stops the inverse square going to infinity when you
        // walk into it.
        d.min_distance = 2.5f;
        d.max_distance = 34.0f;
        audio->Play(d);

        // The wind is NOT spatial. It has no source -- it is the whole canopy
        // at once -- and giving it a position puts the entire forest's rustle
        // at one point that you can walk away from.
        eng::audio::PlayDesc w;
        w.clip = &wind_clip;
        w.loop = true;
        w.spatial = false;
        w.gain = 1.05f;
        audio->Play(w);
    }
    // SAY SO WHEN THERE IS NO SOUND. A failed device and a silent mix are the
    // same experience, and both are the same experience as a trigger that never
    // fires -- so the one thing that must not happen here is failing quietly.
    // Not fatal: a demo with no speaker should still run.
    if (!audio)
        std::printf("audio: unavailable (%s) -- running silent\n",
                    audio_error.empty() ? "no device" : audio_error.c_str());
    else
        std::printf("audio: %d Hz, %.2f s of fire, %.2f s of wind, %zu footsteps\n",
                    audio_rate, fire_clip.Seconds(), wind_clip.Seconds(), steps.size());

    int step_rotation = 0;
    bool was_planted[2] = {false, false};
    std::vector<float> audio_scratch;
    // COUNTED, not sampled. A footstep is a fifth of a second long, so whether
    // one happens to be sounding at the instant a screenshot is taken says
    // almost nothing -- and "no voices" would read as "the trigger is broken"
    // when it means "between steps".
    int steps_taken = 0;
    float audio_peak = 0.0f;

    eng::anim::Animator animator(skeleton);
    eng::anim::StateDesc idle_state;
    idle_state.name = "idle";
    idle_state.clip = &clip_idle;
    idle_state.blend_in = 0.25f;
    const int state_idle = animator.AddState(idle_state);
    // A BLEND SPACE RATHER THAN TWO STATES. A character at 2 m/s is genuinely
    // between a walk and a run, and crossfading between them only looks right
    // at the instant of the switch. The samples sit at the speeds the clips
    // actually travel at, measured in tests/humanoid -- put them anywhere else
    // and the legs cycle at one speed while the body moves at another.
    eng::anim::BlendSpaceDesc loco;
    loco.name = "locomotion";
    loco.samples = {{&clip_walk, world::kWalkSpeed, 1.0f},
                    {&clip_run, world::kRunSpeed, 1.0f}};
    loco.synchronise = true;
    loco.blend_in = 0.2f;
    const int state_loco = animator.AddBlendSpace(loco);
    animator.PlayImmediate(state_idle);
    int animator_state = state_idle;
    float facing = 0.0f;  // radians about +Y; the figure is modelled facing +Z

    // Foot IK, one config per leg. The pole points FORWARD, which is the way a
    // knee bends; without it the solver has a whole circle of solutions and the
    // usual fallback -- keep the current bend -- flips every time the leg
    // passes through straight, snapping the knee backwards once per step.
    // The fox keeps its authored cycle: its joints are not the humanoid's,
    // so the humanoid's IK chains would pose something that is not a leg.
    const bool foot_ik = !no_foot_ik && !figure_fox;
    eng::anim::FootIkConfig ik_left, ik_right;
    ik_left.limb.root = world::kThighL;
    ik_left.limb.mid = world::kShinL;
    ik_left.limb.end = world::kFootL;
    ik_left.limb.pole = eng::Vec3{0.0f, 0.0f, 1.0f};
    ik_left.limb.max_extension = 0.985f;
    ik_left.ankle_to_sole = eng::Vec3{0.0f, -0.09f, 0.0f};
    ik_right = ik_left;
    ik_right.limb.root = world::kThighR;
    ik_right.limb.mid = world::kShinR;
    ik_right.limb.end = world::kFootR;
    const std::initializer_list<const eng::anim::FootIkConfig*> kBothFeet{&ik_left, &ik_right};
    const std::initializer_list<const eng::anim::FootIkConfig*> kNoFeet{};

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

    // A COLLISION SELF-CHECK, on a scratch controller so nothing real moves.
    //
    // IT WALKS INTO THE BODIES THEMSELVES, sampled out of the physics world,
    // rather than toward the districts. Two earlier versions were wrong in ways
    // worth keeping written down:
    //
    //   - the first probed only the banner pole and reported "blocked". It was
    //     right; the pole did have a collider. The GALLERY'S SPHERES did not,
    //     so the character stood in the middle of them overlapping four at
    //     once, and a check that passes while that is true checks nothing.
    //
    //   - the second probed each district's CENTRE and reported four of five
    //     walked through. Also wrong, in the other direction: the gallery has
    //     no sphere on its centre line, the pillars are at +/-1.7 and +/-5.1,
    //     and every glass pane is on the far side. It was walking down the
    //     gaps. A district is not an object and cannot be collided with.
    //
    // A count of colliders settles none of it either -- a capsule at the wrong
    // height or a sphere of the wrong radius adds to the count and stops
    // nothing. Walking into the actual body is the only thing that does.
    //   - the third WALKED the character at each body from a few metres out.
    //     Wrong for a reason specific to this scene: the approach runs through
    //     two hundred trees, so the probe stops on a DIFFERENT trunk and either
    //     reports a pass it did not earn or, having been deflected sideways,
    //     slides past the target and reports a failure that is not real. A
    //     shape test on flat ground (engine/physics/heightfield_test.cc) shows
    //     box, sphere and capsule all stop a character, so those failures were
    //     the probe's, not the engine's.
    //
    // What the user actually reported was standing INSIDE the gallery spheres,
    // and that is a property with no path in it: put the character where the
    // body is, take one step, and it must no longer be inside. That is the
    // depenetration path, it needs no clear approach lane, and it costs one
    // step per body instead of several hundred -- so this checks every body
    // rather than a sample of two dozen.
    {
        const eng::physics::Shape body_shape = eng::physics::Shape::MakeCapsule(
            cc.radius, std::max(0.01f, (cc.height - 2.0f * cc.radius) * 0.5f));
        const eng::Quat upright{0.0f, 0.0f, 0.0f, 1.0f};
        std::vector<int> hits;
        int tested = 0, stuck = 0, phantom = 0;
        const int n = world.Count();
        for (int i = 1; i < n; ++i) {
            const eng::physics::Body& body = world[i];
            const float ground = terrain.HeightAt(body.position.x, body.position.z);
            // Out of a walking character's reach -- a lantern four metres up is
            // not something you can stand inside, and counting it would make
            // the check impossible to satisfy.
            if (body.position.y - ground > cc.height) continue;
            // Feet on the ground, centred on the body: the pose a player ends
            // up in by walking at something that turns out not to be there.
            const eng::Vec3 centre{body.position.x, ground + cc.height * 0.5f,
                                   body.position.z};
            hits.clear();
            world.OverlapShape(body_shape, centre, upright, &hits, {});
            // A body at head height whose own centre the character can stand
            // on WITHOUT touching it is a collider that is too small or in the
            // wrong place -- which is the other half of the bug, and skipping
            // it quietly is how a check like this passes while the scene is
            // broken. Counted, not ignored.
            if (std::find(hits.begin(), hits.end(), i) == hits.end()) {
                ++phantom;
                if (phantom <= 6)
                    std::printf("    body %d: r=%.2f at (%.1f, %.1f), nothing to "
                                "touch at its own centre (y=%.2f, ground=%.2f)\n", i, body.shape.bounds_radius,
                                body.position.x, body.position.z, body.position.y, ground);
                continue;
            }
            ++tested;
            eng::physics::CharacterController probe(cc);
            probe.Teleport(eng::Vec3{body.position.x, ground, body.position.z});
            // SEVERAL steps, not one. Depenetration moves a bounded distance
            // per call on purpose -- teleporting a character out of a deep
            // overlap launches it through whatever is on the other side -- so
            // one step is not enough to clear a two-metre trunk and asking for
            // it would fail a controller that is behaving correctly. Half a
            // second is the honest bar: you may clip in, you may not stay in.
            for (int k = 0; k < 30; ++k)
                probe.Move(world, eng::Vec3{0.0f, -cc.skin * (2.0f / 3.0f), 0.0f});
            const eng::Vec3 f = probe.Feet();
            hits.clear();
            world.OverlapShape(body_shape, eng::Vec3{f.x, f.y + cc.height * 0.5f, f.z},
                               upright, &hits, {});
            if (std::find(hits.begin(), hits.end(), i) != hits.end()) {
                ++stuck;
                if (stuck <= 6)
                    std::printf("    body %d: r=%.2f, still inside after a step, "
                                "moved %.2f m\n", i, body.shape.bounds_radius,
                                Length(f - eng::Vec3{body.position.x, ground,
                                                     body.position.z}));
            }
        }
        // AND YOU CAN STILL WALK IN. Cutting a level pad into a hillside
        // steepens the ground at the join -- smoothstep's slope peaks at 1.5x
        // the linear ramp, and the bowl wall is already 0.55 m per metre -- so
        // the fix for "the district is buried" has an obvious way to become
        // "the district is a plateau with a cliff round it".
        //
        // ASKING THE NAVMESH DOES NOT ANSWER THIS. That was the first version
        // here, and squeezing the blend from 9 m to 0.5 m -- which turns the
        // rim into a wall -- still reported all five reachable, because the
        // navmesh is voxelised from the terrain's COARSEST lod and a half-metre
        // cliff falls between its samples. A check that cannot fail is worse
        // than no check: it reads as evidence. The slope itself is one
        // subtraction and it is exact at any width.
        {
            const float limit = std::tan(cc.slope_limit_degrees * world::kPi / 180.0f);
            float worst = 0.0f, raw_worst = 0.0f;
            int blocked = 0;
            for (const world::Pad& pad : world::kPads) {
                float pad_worst = 0.0f, pad_raw = 0.0f;
                // Through the whole blend annulus, finely enough to land inside
                // a narrow one -- the step is what sets the smallest cliff this
                // can see, so it is a fraction of a metre and not of the blend.
                // Out well past where the clamp can still be biting: it
                // releases where the terrain comes back within kPadSlope of pad
                // level, which for a several-metre drop is tens of metres.
                for (float d = pad.radius - 1.0f; d < pad.radius + 30.0f; d += 0.25f)
                    for (int a = 0; a < 48; ++a) {
                        const float th = float(a) / 48.0f * 2.0f * world::kPi;
                        const float x = pad.centre.x + std::cos(th) * d;
                        const float z = pad.centre.z + std::sin(th) * d;
                        // Sampled radially, the direction the pad's own slope
                        // runs; a gradient in x and z would average it away.
                        const float h0 = terrain.HeightAt(x, z);
                        const float h1 = terrain.HeightAt(pad.centre.x + std::cos(th) * (d + 0.25f),
                                                          pad.centre.z + std::sin(th) * (d + 0.25f));
                        pad_worst = std::max(pad_worst, std::fabs(h1 - h0) / 0.25f);
                        // THE CONTROL. Terrain this size has steep ground in it
                        // anyway -- the forest already rejects 31 degrees for
                        // planting -- so an absolute number says nothing about
                        // whether the PAD did it. This is the same ring on the
                        // ground that would be there without one.
                        pad_raw = std::max(
                            pad_raw,
                            std::fabs(Wilderness(pad.centre.x + std::cos(th) * (d + 0.25f),
                                                 pad.centre.z + std::sin(th) * (d + 0.25f)) -
                                      Wilderness(x, z)) / 0.25f);
                    }
                worst = std::max(worst, pad_worst);
                raw_worst = std::max(raw_worst, pad_raw);
                if (pad_worst > limit) ++blocked;
                std::printf("    pad at (%4.0f, %4.0f): rim %2.0f deg, bare ground %2.0f deg%s\n",
                            pad.centre.x, pad.centre.z,
                            std::atan(pad_worst) * 180.0f / world::kPi,
                            std::atan(pad_raw) * 180.0f / world::kPi,
                            pad_worst > limit ? "   WALLED OFF" : "");
            }
            std::printf("  %zu district pads, steepest rim %.0f degrees (bare ground "
                        "there is %.0f) against a %.0f degree limit, %d walled off\n",
                        std::size(world::kPads), std::atan(worst) * 180.0f / world::kPi,
                        std::atan(raw_worst) * 180.0f / world::kPi,
                        cc.slope_limit_degrees, blocked);
        }
        std::printf("  stood inside %d of the world's %d bodies: %d did not push "
                    "back, %d were not where they claim to be\n", tested, n, stuck,
                    phantom);
    }



    const eng::rhi::TextureId shadow_map = app->Gpu().CreateShadowMap(shadow_px);
    app->Draw().SetShadowMapSize(shadow_px);
    scene.shadowExtent = 30.0f;
    scene.shadowCascades = shadow_cascades;
    scene.shadowDistance = shadow_dist;

    // AUTO EXPOSURE. Without it the composite applies a fixed exposure of 1,
    // and this scene is far too bright for that: the ground's albedo is
    // {0.34, 0.40, 0.26}, a green-grey, and it was rendering at 209,202,189 --
    // not just too light but DESATURATED, because all three channels had been
    // pushed past the tone-map's knee where they compress toward white
    // together. That is what "白茫茫" is. The meter puts the scene's middle
    // fifth at 0.18, which is what a light meter does and what makes the green
    // come back.
    post->config.auto_exposure = true;
    // A stop under the meter. The meter puts the middle fifth of the histogram
    // at middle grey, which is right, but this scene is bimodal -- hard sun
    // and deep shade, about three and a half stops apart -- so the middle
    // fifth falls between the two modes and the sunlit ground sits hot enough
    // to wash out its own texture: at -0.5 the sunlit mask averages 240 of 255
    // with a local deviation of 11.9, measured, and the blotches the texture
    // paints are compressed to invisibility. At -1.0 the mask averages 233
    // with a deviation of 14.0 -- the detail is back -- while the shade still
    // averages 52 and stays readable. (These numbers date from the fix that
    // made compensation do anything at all; before it the setting was written
    // into the params and never read, so every earlier measurement of it was
    // really a measurement of 0.)
    post->config.exposure_compensation = exposure_comp;
    app->Draw().SetExposureBuffer(post->ExposureBuffer());
    // TEMPORAL ANTIALIASING. 4x MSAA resolves EDGES, and this forest is not
    // made of edges -- it is two hundred trees of branches thinner than a pixel
    // at any distance. MSAA gives a pixel four coverage samples, so a branch
    // covering an eighth of it either lands on one or does not, and which way
    // that goes changes as the camera moves. Measured: a QUARTER PIXEL of
    // camera yaw flips 0.014% of pixels by more than a third of the range, and
    // they are in the canopy 103 to 1 over the ground. That is the sparkle.
    //
    // TAA jitters the sample point within the pixel each frame and averages
    // over the history, so a branch that covers an eighth of a pixel
    // contributes an eighth of the time and converges to an eighth of the
    // colour. It is the only thing that fixes geometry finer than a sample.
    post->config.taa = true;
    post->config.fog = start_fog;
    post->config.fog_density = 0.0055f;
    post->config.fog_height_falloff = 0.035f;
    post->config.fog_start = 6.0f;

    std::vector<eng::Vec3> path;
    std::size_t path_at = 0;
    float fall_speed = 0.0f;
        // PITCHED DOWN 17 DEGREES, NOT 31. At 0.55 rad the camera looks at the
    // ground: the frame was all terrain, no horizon, no canopy and no sky, and
    // a scene photographed straight down has nothing in it to judge the light
    // by. Looking ACROSS the clearing puts the treeline and the sky in frame,
    // which is what the trees were added for.
    float focus_height = focus_start;
    float camera_yaw = 0.7f + yaw_offset, camera_pitch = pitch_start,
          camera_distance = 13.0f;
    float baked_az = 1e9f, baked_el = 1e9f;
    bool ssao_on = ssao_start;
    bool shafts_on = shafts_start;
    bool bloom_on = bloom_start;
    // The valley's water, not the tank's. The tank shades against a slate wall
    // in a dark room; this sits outdoors under the same sky the atmosphere
    // model is producing, so the horizon tint and the sun direction have to
    // follow the scene rather than be constants.
    eng::SurfaceLook water_look;
    // SCALED WITH THE PARTICLES. The defaults are apps/fluid's, and that tank
    // is half this basin in every dimension: its spacing is 0.034 m and this
    // one's is 0.07. edge_stop is the bilateral smooth's "these two taps are
    // different surfaces" threshold in metres -- below the particle spacing the
    // sheet falls back apart into the individual spheres, which is exactly what
    // the default did here. Above the spacing, and the basin's own stone starts
    // bleeding into the water.
    water_look.edge_stop = water_edge_stop;
    water_look.sphere_radius = water_sphere;
    const bool local_shadows_on = local_shadows_start;
    int lamp_draws = 0, lamp_culled = 0;
    bool sparks_on = sparks_start;
    int here = tour_start;      // which stop was last jumped to, for the HUD
    int goto_stop = tour_start; // -1 when there is nothing to jump to
    bool gi_stale = false;
    int gi_still_frames = 0;

    app->Actions().BindMouse("click", eng::app::MouseButton::Left);
    app->Actions().BindMouse("orbit", eng::app::MouseButton::Right);
    app->Actions().Bind("fog", 'f');
    app->Actions().Bind("taa", 't');
    app->Actions().Bind("sun_back", '[');
    app->Actions().Bind("sun_fwd", ']');
    app->Actions().Bind("ssao", 'o');
    app->Actions().Bind("shafts", 'v');
    app->Actions().Bind("stop0", '0');
    app->Actions().Bind("stop1", '1');
    app->Actions().Bind("stop2", '2');
    app->Actions().Bind("stop3", '3');
    app->Actions().Bind("stop4", '4');
    app->Actions().Bind("stop5", '5');
    app->Actions().Bind("stop6", '6');
    app->Actions().Bind("reset", 'r');
    app->Actions().Bind("passes", 'p');
    app->Actions().Bind("bloom", 'b');
    app->Actions().Bind("forward", 'w');
    app->Actions().Bind("back", 's');
    app->Actions().Bind("left", 'a');
    app->Actions().Bind("right", 'd');
    app->Actions().Bind("run", ' ');
    app->Actions().Bind("raise", 'e');
    app->Actions().Bind("lower", 'q');

    eng::RenderGraph graph;
    // Allocated lazily on the first frame, when the framebuffer size is known.
    eng::rhi::TextureId shot_target;

    // GPU-DRIVEN snapshots, outside the frame loop on purpose: the HUD draws
    // before the passes run, so it reads LAST frame's numbers -- same as the
    // LastStats it used to read. Declared here, assigned in the scene pass.
    eng::Scene rem_scene;
    int indirect_batches = 0, indirect_draws = 0, rem_draws = 0;
    // The debounce below decides to re-bake in the middle of a frame; the bake
    // itself has to happen BETWEEN frames. See bake_indirect: it opens a device
    // frame of its own and waits on it, which from inside App::BeginFrame's
    // frame leaks a frames-in-flight permit and discards the frame in progress.
    bool gi_rebake_pending = false;
    // --- the frame, accounted for ---------------------------------------------
    //
    // The app named a GPU timer on eight render passes and four compute
    // encoders and then printed one aggregate number, so a frame that was too
    // slow gave no hint as to which of its fourteen encoders to look at. These
    // three carry last frame's answer, because the HUD is built before this
    // frame's passes run -- every number on it is one frame old, and pretending
    // otherwise would be worse than the lag.
    //
    // cpu_ms is the loop body: BeginFrame to EndFrame, so it excludes the wait
    // for a drawable and for a frame slot. Against Clock::RawDt, which is the
    // whole wall period including those waits, it says which side the frame is
    // bound on -- cpu near period means the CPU is the wall, cpu far below it
    // means we are waiting on the GPU or on vsync.
    //
    // period_ms is measured here rather than taken from Clock::RawDt because
    // headless FIXES the timestep on purpose (app.h: "two identical captures
    // differ and nothing can be compared byte for byte"), so RawDt reports the
    // nominal 1/60 in a capture no matter how long the frame really took. The
    // wall clock is the same number in both modes.
    double cpu_ms = 0.0, period_ms = 0.0;
    auto period_prev = std::chrono::steady_clock::now();
    std::vector<eng::rhi::GpuTiming> pass_ms;
    bool profile_panel = false;
    while (app->Running()) {
        if (gi_rebake_pending) {
            gi_rebake_pending = false;
            if (!bake_indirect()) return Fail(error);
            gi_stale = false;
        }
        if (!app->BeginFrame()) continue;
        const auto cpu_begin = std::chrono::steady_clock::now();
        period_ms = std::chrono::duration<double, std::milli>(cpu_begin - period_prev).count();
        period_prev = cpu_begin;
        const eng::app::Frame& f = app->Current();

        // --- camera ----------------------------------------------------------
        if (app->Actions().Down("orbit")) {
            camera_yaw += f.mouse_dx * 0.006f;
            // DOWN TO -0.55, so the camera can get below the focus and look
            // UP. It was clamped at 0.15 and could only ever look down -- in a
            // forest, where the canopy is the most interesting thing in the
            // scene and the one place backlit leaves are visible, there was no
            // way to point at it at all.
            camera_pitch = std::clamp(camera_pitch - f.mouse_dy * 0.004f, -1.20f, 1.35f);
        }
        // 2.5 m, not 6. Six metres is a reasonable floor for following a
        // character and it makes everything in the valley impossible to look
        // AT: the spring's basin is three metres across, so at the old minimum
        // it never filled more than a tenth of the frame, and --dist was
        // silently clamped up every frame without saying so.
        camera_distance = std::clamp(camera_distance - f.scroll * 1.2f, 2.5f, 60.0f);
        // 0 goes back to the character, 1-5 to the districts.
        static const char* kStopAction[7] = {"stop0", "stop1", "stop2", "stop3",
                                            "stop4", "stop5", "stop6"};
        for (int k = 0; k < 7; ++k)
            if (app->Actions().Pressed(kStopAction[k])) goto_stop = k;
        if (app->Actions().Pressed("fog")) post->config.fog = !post->config.fog;
        if (app->Actions().Pressed("taa")) post->config.taa = !post->config.taa;
        if (app->Actions().Down("raise"))
            focus_height = std::min(focus_height + f.dt * 9.0f, 22.0f);
        if (app->Actions().Down("lower"))
            focus_height = std::max(focus_height - f.dt * 9.0f, 0.4f);
        if (app->Actions().Pressed("ssao")) ssao_on = !ssao_on;
        if (app->Actions().Pressed("shafts")) shafts_on = !shafts_on;
        if (app->Actions().Pressed("passes")) profile_panel = !profile_panel;
        if (app->Actions().Pressed("bloom")) bloom_on = !bloom_on;

        // --- the sun, and the indirect light that has to follow it -------------
        //
        // The volume holds where the light WAS. Move the sun and it keeps
        // reporting the old bounce, shadowed by the old geometry, and nothing
        // about it decays or complains -- the scene just quietly disagrees with
        // itself. That was invisible here only because there was no way to move
        // the sun.
        //
        // REBAKED ON A DEBOUNCE, not per frame. It costs 0.09 s, which is six
        // frames dropped, so doing it while a key is held would make the sun
        // unusable. Waiting until it has been still for a fifth of a second
        // means one hitch at the end of a movement instead of a hitch per
        // frame, and the stale volume in between is a far better wrong answer
        // than a constant ambient.
        float sun_input = 0.0f;
        if (app->Actions().Down("sun_back")) sun_input -= 1.0f;
        if (app->Actions().Down("sun_fwd")) sun_input += 1.0f;
        if (sun_input != 0.0f) {
            sun_elevation = std::clamp(sun_elevation + sun_input * f.dt * 0.30f,
                                       0.08f, 1.45f);
            sun_azimuth += sun_input * f.dt * 0.45f;
            gi_stale = true;
            gi_still_frames = 0;
        } else if (rebake_every > 0 && f.index > 0 &&
                   int(f.index) % rebake_every == 0) {
            gi_rebake_pending = true;
        } else if (gi_stale && ++gi_still_frames > 12) {
            // QUEUED, not run. bake_indirect opens a device frame and blocks on
            // it, and this is the middle of one -- see the top of the loop.
            gi_rebake_pending = true;
        }
        if (app->Actions().Pressed("reset")) {
            player.Teleport(eng::Vec3{0.0f, terrain.HeightAt(0.0f, 0.0f) + 0.5f, 0.0f});
            path.clear();
        }

        // WHERE TO GO. Fast travel, not a camera mode.
        //
        // These used to move the CAMERA and pin its yaw and distance every
        // frame, which froze the orbit and the zoom the moment you arrived
        // anywhere -- you could look at a district and not look around it. A
        // stop teleports the CHARACTER instead, once, and then everything works
        // the way it does everywhere else: the camera follows, WASD walks,
        // right-drag orbits, scroll zooms.
        struct Stop { const char* name; eng::Vec3 at; };
        static const Stop kStopList[] = {
            {"the clearing", eng::Vec3{0.0f, 0.0f, 0.0f}},
            {"material gallery", world::kGallery},
            {"lantern hall", world::kLanternHall},
            {"glass pavilion", world::kGlassPavilion},
            {"fire pit", world::kFirePit},
            {"the banner", world::kFlag},
            {"the spring", world::kSpring},
        };
        constexpr int kStops = int(sizeof(kStopList) / sizeof(kStopList[0]));
        if (goto_stop >= 0 && goto_stop < kStops) {
            const Stop& to = kStopList[goto_stop];
            // ARRIVE FROM THE INSIDE. The districts sit near the rim of a bowl
            // -- the lantern hall is 46 m out, which is exactly where the
            // terrain starts climbing -- so stepping back from one along a
            // hand-picked angle put the character up the slope at y = 3.3,
            // looking down on the tops of the pillars. Backing toward the
            // MIDDLE of the valley instead lands on the flat every time, and
            // needs no angle chosen per district.
            const float r = std::sqrt(to.at.x * to.at.x + to.at.z * to.at.z);
            const eng::Vec3 inward =
                r > 1e-3f ? eng::Vec3{-to.at.x / r, 0.0f, -to.at.z / r}
                          : eng::Vec3{0.0f, 0.0f, 1.0f};
            const float x = to.at.x + inward.x * 9.0f;
            const float z = to.at.z + inward.z * 9.0f;
            player.Teleport(eng::Vec3{x, terrain.HeightAt(x, z) + 0.5f, z});
            // Looking back OUT at the district. The camera sits at focus +
            // offset and looks along -offset, so the offset is the inward
            // direction and the yaw is its angle.
            // Plus whatever --yaw asked for. It is an offset on the arrival
            // angle, and without adding it here the teleport silently discards
            // it -- which made every capture face wherever the district
            // happened to be rather than where it was asked to face.
            camera_yaw = std::atan2(inward.z, inward.x) + yaw_offset;
            camera_distance = dist_start > 0.0f ? dist_start : 9.0f;
            path.clear();
            path_at = 0;
            here = goto_stop;
            goto_stop = -1;
        }
        // LOOK-AT HEIGHT, and it is not cosmetic. The camera always looks AT
        // the focus, so with the focus pinned a metre above the character's
        // feet there is no combination of pitch and distance that points it at
        // the canopy -- tilting only moves the camera, it does not aim it. In a
        // forest that means the most interesting thing in the scene, and the
        // only place a backlit leaf can be seen, was unreachable.
        const eng::Vec3 focus = player.Feet() + eng::Vec3{0.0f, focus_height, 0.0f};

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

            // CROSS-MULTIPLIED, not two perspective divides.
            //
            // The far plane is at INFINITY under this projection and depth 0 is
            // where it lives, so far_h.w is EXACTLY zero for every pixel of
            // every frame. Dividing by it gives an infinity, subtracting the
            // near point gives an infinity, and normalising that gives NaN --
            // so the raycast was handed a NaN direction, returned false, and
            // click-to-walk quietly did nothing at all. It has never worked.
            //
            // The direction is (far/far.w - near/near.w) scaled by far.w *
            // near.w, which is the same vector times a positive number and has
            // no divide in it. This is the third place in the engine to hit
            // this: the sky shader's ray reconstruction had it, and its fix is
            // the same three lines.
            const eng::Vec3 dir = eng::Normalize(eng::Vec3{
                far_h.x * near_h.w - near_h.x * far_h.w,
                far_h.y * near_h.w - near_h.y * far_h.w,
                far_h.z * near_h.w - near_h.z * far_h.w});

            float t = 0.0f;
            if (terrain.Raycast(a, dir, 400.0f, &t, nullptr)) {
                const eng::Vec3 hit = a + dir * t;
                std::vector<eng::Vec3> found;
                if (navmesh.FindPath(player.Feet(), hit, &found) && found.size() > 1) {
                    path = found;
                    path_at = 1;  // 0 is where we already are
                }
            }
        }

        // --- movement ---------------------------------------------------------
        //
        // WASD FIRST, and it takes priority over the path. A world with five
        // districts sixty metres apart had exactly one way to cross it: click a
        // patch of ground and wait. That needs the ground you want to reach to
        // be visible and on the navmesh, which rules out walking to anything
        // you are currently looking at from above -- and there was no way at
        // all to make a small adjustment.
        //
        // RELATIVE TO THE CAMERA, not to the world. Forward means "away from
        // the camera" because that is what the key means to the person pressing
        // it; world-relative movement in an orbiting camera sends you sideways
        // the moment you turn.
        eng::Vec3 wish{0.0f, 0.0f, 0.0f};
        {
            const float fx = std::cos(camera_yaw), fz = std::sin(camera_yaw);
            // The camera sits at focus + offset, so it LOOKS along -offset.
            eng::Vec3 walk{0.0f, 0.0f, 0.0f};
            if (app->Actions().Down("forward")) walk = walk + eng::Vec3{-fx, 0.0f, -fz};
            if (app->Actions().Down("back")) walk = walk + eng::Vec3{fx, 0.0f, fz};
            // Right of the view direction: the view is (-fx, -fz), so its right
            // in a y-up left-handed sense is (-fz, fx).
            if (app->Actions().Down("left")) walk = walk + eng::Vec3{fz, 0.0f, -fx};
            if (app->Actions().Down("right")) walk = walk + eng::Vec3{-fz, 0.0f, fx};
            const float len = eng::Length(walk);
            if (len > 1e-4f) {
                // Any key cancels a path in progress, or the two fight and the
                // character crabs sideways toward a waypoint it was told to
                // forget about.
                path.clear();
                path_at = 0;
                // THE GAITS' OWN SPEEDS, not a number that felt right. These
                // were 4.5 and 11 m/s, which are a sphere's speeds -- 11 m/s is
                // faster than a person has ever run -- and they were invisible
                // because the character was a sphere. With legs on it, any
                // speed that is not the one the clip travels at shows up
                // immediately as the feet skating.
                const float speed =
                    app->Actions().Down("run") ? world::kRunSpeed : world::kWalkSpeed;
                wish = walk * (speed / len);
            }
        }
        // HEADLESS LOCOMOTION. Capture mode has no keyboard, so without this
        // the only pose that can ever be photographed is the idle -- and a walk
        // cycle is exactly the thing that has to be looked at, not just
        // measured. Drives the same `wish` the keys do, so it exercises the
        // blend space, the turn rate and the foot IK rather than a shortcut
        // past them.
        if (headless_walk != 0.0f)
            wish = eng::Vec3{std::sin(headless_heading), 0.0f, std::cos(headless_heading)} *
                   headless_walk;
        if (wish.x == 0.0f && wish.z == 0.0f && path_at < path.size()) {
            const eng::Vec3 target = path[path_at];
            eng::Vec3 to{target.x - player.Feet().x, 0.0f, target.z - player.Feet().z};
            const float distance = eng::Length(to);
            // The waypoint is reached when the character is within its own
            // radius, not at zero: aiming for an exact point makes the
            // character orbit it forever, because it overshoots every frame.
            if (distance < cc.radius + 0.15f) {
                ++path_at;
            } else {
                wish = to * (1.0f / distance) * world::kRunSpeed;
            }
        }
        const float dt = std::min(f.dt, 0.05f);

        // A CONSTANT GROUND STICK, sized into the middle of a narrow window.
        //
        // This used to push down by 9.81 * dt * dt * 6 every frame -- 16.3 mm
        // at 60 Hz -- and the character never came to rest: feet.y alternating
        // between -5.609918 and -5.623887 for as long as the demo ran, while x
        // and z crept. The camera is rigidly attached to it, so that went
        // straight into the view matrix, and against branches a pixel wide it
        // read as the whole PICTURE flickering. 0.68% of pixels changed between
        // two consecutive frames of a completely still shot; it is 0.011% now,
        // all of it the HUD's exposure readout.
        //
        // heightfield_test has the mechanism, measured on this exact terrain.
        // There are TWO failures either side of a good window: below about half
        // the controller's skin the character is never caught and simply falls,
        // and above about four fifths of it the step penetrates, depenetration
        // pushes back out along the surface NORMAL, and on a slope that has a
        // horizontal component -- so it slides, lands at a different height and
        // does it again. Which is exactly the sideways creep in the trace.
        //
        // The window is 7.5 to 12 mm for a 15 mm skin. Two thirds of the skin
        // is the middle of it, and the test asserts the MARGIN rather than the
        // value: the first version of this fix used half the skin, which is
        // exactly the lower edge, one step away from a 65 mm failure. It worked
        // by luck.
        if (player.Grounded()) {
            fall_speed = -cc.skin * (2.0f / 3.0f) / std::max(dt, 1e-4f);
        } else {
            fall_speed -= 9.81f * dt;
            // Terminal velocity, so a long fall cannot produce a step longer
            // than the capsule and tunnel through the ground.
            fall_speed = std::max(fall_speed, -55.0f);
        }
        player.Move(world, eng::Vec3{wish.x * dt, fall_speed * dt, wish.z * dt});

        // --- posing the character ---------------------------------------------
        {
            const float ground_speed = eng::Length(eng::Vec3{wish.x, 0.0f, wish.z});
            // Face the way you are going, turning at a rate rather than
            // snapping: a character who changes facing in one frame reads as a
            // sprite being flipped.
            if (ground_speed > 0.05f) {
                const float want = std::atan2(wish.x, wish.z);
                float delta = want - facing;
                // Through the short way round. Without this a turn from +179 to
                // -179 degrees spins the character all the way about.
                while (delta > world::kPi) delta -= 2.0f * world::kPi;
                while (delta < -world::kPi) delta += 2.0f * world::kPi;
                facing += delta * std::min(1.0f, dt * 9.0f);
            }
            const int want_state = ground_speed > 0.12f ? state_loco : state_idle;
            // Only on a CHANGE. Play() starts a fade, so calling it every frame
            // restarts the fade every frame and the blend never completes --
            // the character is permanently 20% into a transition.
            if (want_state != animator_state) {
                animator.Play(want_state);
                animator_state = want_state;
            }
            animator.SetParameter(ground_speed);
            animator.Update(dt);

            const eng::Mat4 to_world = eng::Mat4::Translation(player.Feet()) *
                                       eng::Mat4::RotationY(facing) *
                                       eng::Mat4::RotationY(figure_yaw) *
                                       eng::Mat4::Scale(figure_scale);
            eng::anim::Pose pose = animator.CurrentPose();

            // FOOT IK, which is the reason this demo is the right place for a
            // character: the ground here is never flat. Without it both feet
            // sit at the height the clip drew them, so on a slope one hovers
            // and the other is buried, and the error is exactly the thing the
            // eye uses to judge whether someone is standing on the ground.
            //
            // The ground query is the terrain function rather than a raycast:
            // it is exact, it costs nothing, and the collider is built from the
            // same heights, so the foot cannot disagree with what the character
            // is standing on.
            if (foot_ik) {
                // GATED ON CONTACT, and iterated with the hip drop.
                //
                // Full-weight IK on both feet every frame does not work. At the
                // extreme of a stride the swing leg is nearly straight and its
                // foot is at its farthest forward, so dragging it down to the
                // ground asks for more leg than there is; the solver refuses to
                // stretch a bone, correctly, and both feet end up reaching for
                // the floor and never arriving. It measured 118 mm off. On
                // screen it read as the legs having collapsed into sticks,
                // which is a different fault and would have been hunted
                // somewhere else entirely -- see tests/humanoid, which says
                // which of the three possible causes it is in three numbers.
                //
                // The weights come from the ANIMATED pose, before any solving,
                // so that a foot's contact does not depend on what the IK did
                // to the other one.
                std::vector<eng::Mat4> jw;
                eng::anim::ComputeJointWorld(skeleton, pose, &jw);
                eng::anim::FootIkConfig cfg[2] = {ik_left, ik_right};
                eng::anim::GroundHit hit[2];
                for (int i = 0; i < 2; ++i) {
                    const eng::Mat4 ankle = to_world * jw[std::size_t(cfg[i].limb.end)];
                    const eng::Vec4 sole = ankle * eng::Vec4{world::kAnkleToSole.x,
                                                             world::kAnkleToSole.y,
                                                             world::kAnkleToSole.z, 1.0f};
                    const float ground = terrain.HeightAt(sole.x, sole.z);
                    hit[i].hit = true;
                    hit[i].point = eng::Vec3{sole.x, ground, sole.z};
                    hit[i].normal = terrain.NormalAt(sole.x, sole.z);
                    cfg[i].limb.weight = world::FootPlantWeight(sole.y - ground);

                    // A FOOTSTEP IS A FOOT LANDING, and the plant weight is
                    // already the answer to "is this foot on the ground". Using
                    // it means the sound cannot disagree with what the legs are
                    // doing -- triggering off the clip's phase instead would be
                    // a second copy of the same fact, and the two drift the
                    // moment the blend space is between clips.
                    const bool planted = cfg[i].limb.weight > 0.5f;
                    if (planted && !was_planted[i] && audio && ground_speed > 0.2f) {
                        eng::audio::PlayDesc d;
                        d.clip = &steps[std::size_t(step_rotation++ % steps.size())];
                        d.spatial = true;
                        d.position = eng::Vec3{sole.x, ground, sole.z};
                        d.min_distance = 1.2f;
                        d.max_distance = 22.0f;
                        // Louder and a touch lower when running, because a run
                        // lands harder. Pitch, not a second set of clips: the
                        // difference between a walk and a run landing is mostly
                        // that it is bigger, and generating four more clips to
                        // say so would be four more things to keep in step.
                        const float effort =
                            std::clamp((ground_speed - world::kWalkSpeed) /
                                           (world::kRunSpeed - world::kWalkSpeed),
                                       0.0f, 1.0f);
                        d.gain = 1.5f + effort * 1.4f;
                        d.pitch = 1.06f - effort * 0.16f;
                        audio->Play(d);
                        ++steps_taken;
                    }
                    was_planted[i] = planted;
                }
                // Three passes. Lowering the pelvis by the shortfall changes
                // the geometry, so each solve asks for a little less and the
                // residual falls off geometrically: 32 mm to 4 mm by the third,
                // where one pass alone leaves 12. The drop has to be applied
                // BEFORE the solve that needs the reach -- doing it afterwards,
                // which is the obvious reading of "the caller applies it",
                // moves the foot further from the ground it just failed to
                // touch.
                for (int pass = 0; pass < 3; ++pass) {
                    float drop = 0.0f;
                    for (int i = 0; i < 2; ++i)
                        drop = std::min(drop, eng::anim::SolveFootIk(skeleton, cfg[i],
                                                                     to_world, hit[i], &pose));
                    // BOTH FEET DECIDE THE HIPS: with one foot on a rock and
                    // the other on the ground, the pelvis has to come down to
                    // the lower of the two or the higher leg locks straight.
                    if (drop >= -1e-4f) break;
                    pose.local[world::kPelvis].translation.y += drop;
                }
            }

            if (audio) {
                // The listener rides the CAMERA, not the character. The camera
                // is what the player is looking through, so a sound to the
                // right of the character but behind the camera has to come from
                // behind -- attaching the listener to the body puts the whole
                // mix a couple of metres out whenever the camera orbits.
                const eng::Vec3 fwd = scene.camera.target - scene.camera.eye;
                audio->SetListener(scene.camera.eye, fwd, eng::Vec3{0.0f, 1.0f, 0.0f});
            }
            // DRIVE THE MIXER IN CAPTURE MODE. With no device nothing pulls
            // samples, so the command queue never drains and every Play above
            // is a no-op -- which would look exactly like working audio from
            // the outside. Rendering the block the frame would have consumed
            // makes the voice count and the peak below mean something.
            if (audio && !shot_path.empty()) {
                const int frames = std::max(1, int(dt * float(audio_rate)));
                audio_scratch.resize(std::size_t(frames) * 2);
                audio->RenderForTest(audio_scratch.data(), frames);
            }
            // THE WHOLE RUN'S PEAK, not the last block's. Clipping is a thing
            // that happens once, when two loud things coincide, and a reading
            // taken at the moment a screenshot is written will almost never be
            // the moment it happened.
            if (audio) audio_peak = std::max(audio_peak, audio->LastPeak());
            eng::anim::ComputeJointMatrices(skeleton, pose, &scene.joint_matrices);
            scene.instances[body_instance].palette = 0;
            scene.instances[body_instance].model = to_world;
        }
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

        // The same distance thresholds as the terrain, for the same reason: a
        // cell 90 m away is a few hundred pixels tall and cannot show the
        // difference. The renderer culls whatever is off screen on its own,
        // from the bounds of whichever level is selected here.
        int forest_lod_counts[3] = {0, 0, 0};
        const float kForestLodNear = lod_near, kForestLodFar = lod_near * 2.1f;
        for (const ForestChunk& chunk : forest) {
            const float distance = eng::Length(chunk.centre - scene.camera.eye);
            int lod = 0;
            if (distance > kForestLodNear) lod = 1;
            if (distance > kForestLodFar) lod = 2;
            const int t_lod = std::min(lod, int(chunk.trunk_lods.size()) - 1);
            const int l_lod = std::min(lod, int(chunk.leaf_lods.size()) - 1);
            scene.instances[chunk.trunk_instance].mesh = chunk.trunk_lods[std::size_t(t_lod)];
            scene.instances[chunk.leaf_instance].mesh = chunk.leaf_lods[std::size_t(l_lod)];
            // The undergrowth is level 0 or nothing: parked underground rather
            // than removed, because every index into the instance list is held
            // by something and compacting it would invalidate them all.
            if (eng::Valid(chunk.ground))
                scene.instances[chunk.ground_instance].model =
                    lod == 0 ? eng::Mat4::Identity()
                             : eng::Mat4::Translation(eng::Vec3{0.0f, -1000.0f, 0.0f});
            ++forest_lod_counts[lod];
        }

        // A SYNTHETIC PICK, once, during a capture. Clicking cannot be tested
        // without a window, and the bug this checks for produced no error and
        // no visible effect -- the raycast simply returned false forever.
        if (!shot_path.empty() && f.index == 20) {
            const eng::Mat4 inv =
                eng::Inverse(scene.camera.ViewProj(float(f.width) / float(f.height)));
            const eng::Vec4 n4 = inv * eng::Vec4{0.0f, -0.4f, 1.0f, 1.0f};
            const eng::Vec4 f4 = inv * eng::Vec4{0.0f, -0.4f, 0.0f, 1.0f};
            const eng::Vec3 from{n4.x / n4.w, n4.y / n4.w, n4.z / n4.w};
            const eng::Vec3 d = eng::Normalize(eng::Vec3{f4.x * n4.w - n4.x * f4.w,
                                                         f4.y * n4.w - n4.y * f4.w,
                                                         f4.z * n4.w - n4.z * f4.w});
            float t = 0.0f;
            const bool hit = terrain.Raycast(from, d, 400.0f, &t, nullptr);
            std::vector<eng::Vec3> found;
            const bool routed =
                hit && navmesh.FindPath(player.Feet(), from + d * t, &found);
            std::printf("  pick check: far.w %.6f, dir (%.3f %.3f %.3f), hit %d at "
                        "%.1f m, path %d with %zu points\n",
                        f4.w, d.x, d.y, d.z, int(hit), t, int(routed), found.size());
        }

        world::PoseBanner(banner, scene, f.time);

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
        // JITTER, TWICE. BeginFrame advances the sequence and also builds the
        // previous-frame matrices from the camera it is handed, so the camera
        // needs this frame's offset going in and the next one coming out.
        // Without it TAA is a pure blur -- every frame samples the same points
        // and averaging them recovers nothing.
        scene.camera.jitter = post->Jitter();
        post->BeginFrame(scene.camera, f.width, f.height, dt);
        scene.camera.jitter = post->Jitter();
        // RENDER SIZE, from the post stack's render_scale: every pass below
        // up to the TAA resolve runs at rw/rh, and the resolve rebuilds the
        // display image. At the default scale of 1.0 these are f.width and
        // f.height, and nothing below differs from before.
        const int rw = post->RenderWidth(), rh = post->RenderHeight();
        scene_targets.Resize(rw, rh);

        // GPU-DRIVEN remainder: everything CullScene did NOT batch (skinned,
        // transparent), collected ahead of Execute and drawn the ordinary way
        // in the same pass. A whole-scene copy is cheap -- the vectors hold
        // handles, not geometry -- and it keeps lights, camera and the joint
        // palette identical, so the remainder draws exactly what DrawScene
        // would have drawn for it. rem_scene and the snapshot ints live
        // outside the frame loop, so the HUD below reads last frame's.

        // --- HUD ----------------------------------------------------------------
        //
        // Begin runs even with the HUD off: it is what clears last frame's
        // quads, and skipping it would leave the composite drawing a stale
        // panel forever rather than none.
        ui->Begin(f.width, f.height);
        if (hud_on) {
        ui->Rect(12, 12, 500, 276, eng::Vec4{0.05f, 0.06f, 0.08f, 0.72f});
        ui->Outline(12, 12, 500, 276, 1.0f, eng::Vec4{0.35f, 0.40f, 0.48f, 0.9f});
        char line[192];
        const eng::RenderStats& rs = app->Draw().LastStats();
        // In indirect mode LastStats is the REMAINDER's (the forward pass runs
        // last), so the HUD reads the snapshots the scene pass took instead.
        // They are one frame old -- everything on this HUD is, it draws before
        // the passes run -- same as rs.
        if (indirect_on)
            std::snprintf(line, sizeof(line),
                          "%zu chunks lod %d/%d/%d   forest %d/%d/%d   %d draws (%d gpu + %d rem)",
                          chunks.size(), lod_counts[0], lod_counts[1], lod_counts[2],
                          forest_lod_counts[0], forest_lod_counts[1],
                          forest_lod_counts[2],
                          indirect_draws + rem_draws, indirect_draws, rem_draws);
        else
            std::snprintf(line, sizeof(line),
                          "%zu chunks lod %d/%d/%d   forest %d/%d/%d   %d draws, %d culled",
                          chunks.size(), lod_counts[0], lod_counts[1], lod_counts[2],
                          forest_lod_counts[0], forest_lod_counts[1],
                          forest_lod_counts[2], rs.draws, rs.culled);
        ui->Text(26, 26, line, eng::Vec4{0.92f, 0.94f, 0.98f, 1.0f});
        // THE SCENE PASS IS NOT THE FRAME. The line above counts one of the
        // three passes that walk the whole scene, so with --crowd 600 it read
        // 403 while the frame submitted three thousand, and the shadow pass --
        // the biggest of the three -- was invisible in the one place anyone
        // looks. DROPPED is the number that was shown nowhere at all: the
        // uniform ring is shared by every pass, and when it runs dry the passes
        // AFTER the one that drained it draw nothing. A black frame, no error.
        const int dropped = app->Draw().DroppedThisFrame();
        const int faults = app->Gpu().GpuFaultCount();
        char tail[80] = "";
        if (dropped)
            std::snprintf(tail, sizeof(tail), "   %d DRAWS DROPPED: RING FULL", dropped);
        else if (faults)
            std::snprintf(tail, sizeof(tail), "   %d GPU FAULTS (see stderr)", faults);
        std::snprintf(line, sizeof(line), "shadow %d draws, %d culled%s",
                      app->Draw().ShadowDrawCount(),
                      app->Draw().ShadowCulledCount(), tail);
        ui->Text(26, 50, line,
                 (dropped || faults) ? eng::Vec4{1.0f, 0.45f, 0.35f, 1.0f}
                                     : eng::Vec4{0.62f, 0.68f, 0.78f, 1.0f});
        std::snprintf(line, sizeof(line),
                      "sun %.0f deg   gi %s (%.0f ms, %d buried)",
                      sun_elevation * 57.2958f,
                      gi_stale ? "STALE" : "baked", last_bake_seconds * 1000.0,
                      last_bake_dark);
        ui->Text(26, 74, line,
                 gi_stale ? eng::Vec4{1.0f, 0.72f, 0.35f, 1.0f}
                          : eng::Vec4{0.80f, 0.86f, 0.94f, 1.0f});
        // FEET ABOVE GROUND, because a character floating a hand's width over
        // the terrain is nearly invisible in a screenshot and unmistakable as a
        // number. It should be within a millimetre or two of the contact skin.
        std::snprintf(line, sizeof(line),
                      "path: %zu points, at %zu   ground %.2f m   feet %+.3f m",
                      path.size(), path_at,
                      terrain.HeightAt(player.Feet().x, player.Feet().z),
                      player.Feet().y - terrain.HeightAt(player.Feet().x, player.Feet().z));
        ui->Text(26, 98, line, eng::Vec4{0.80f, 0.86f, 0.94f, 1.0f});
        // THE HEADLINE IS THREE NUMBERS, not one. gpu alone cannot say whether
        // a slow frame is the GPU's fault: cpu is the loop body and period is
        // the wall including every wait, so cpu ~= period means the CPU is the
        // wall and cpu << period means it is not. `slowest` is the one pass
        // worth looking at first; `p` opens the rest.
        const char* slowest = "-";
        double slowest_ms = 0.0;
        for (const eng::rhi::GpuTiming& t : pass_ms)
            if (t.milliseconds > slowest_ms) {
                slowest_ms = t.milliseconds;
                slowest = t.label;
            }
        std::snprintf(line, sizeof(line),
                      "cpu %.2f  gpu %.2f  period %.2f ms  %.0f fps   slowest: %s %.2f",
                      cpu_ms, app->Gpu().LastFrameGpuMilliseconds(), period_ms,
                      app->Time().Fps(), slowest, slowest_ms);
        ui->Text(26, 122, line, eng::Vec4{0.80f, 0.86f, 0.94f, 1.0f});
        std::snprintf(line, sizeof(line),
                      "fog %s  taa %s  ao %s  bloom %s  exposure %.2f (%+.1f EV)  %s",
                      post->config.fog ? "on" : "off",
                      post->config.taa ? "on" : "off", ssao_on ? "on" : "off",
                      bloom_on ? "on" : "off",
                      post->LastExposure(),
                      std::log2(std::max(post->LastExposure(), 1e-6f)),
                      player.Grounded() ? "grounded" : "airborne");
        ui->Text(26, 146, line, eng::Vec4{0.80f, 0.86f, 0.94f, 1.0f});
        ui->Text(26, 176, "click: walk there   right-drag: orbit   scroll: zoom",
                 eng::Vec4{0.62f, 0.68f, 0.78f, 1.0f});
        const eng::Renderer::ClusterStats cs =
            app->Draw().ClusteredLighting() ? app->Draw().ReadClusterStats()
                                            : eng::Renderer::ClusterStats{};
        // AUDIO ON THE HUD, because there is no other way to see it. A sound
        // that never triggered and a sound that triggered into a full voice
        // table are indistinguishable from the listener's chair, and both are
        // indistinguishable from a bug in the trigger. Peak above 1 is the mix
        // clipping, which is audible as distortion and shows up nowhere else.
        std::snprintf(line, sizeof(line),
                      "%zu lights, %d cells lit, %d max per cell%s | %d voices, "
                      "%d steps, peak %.2f%s%s",
                      scene.lights.size(), cs.occupied_cells, cs.max_per_cell,
                      cs.overflowed_cells ? "  OVERFLOW" : "",
                      audio ? audio->ActiveVoices() : 0, steps_taken,
                      audio_peak,
                      audio && audio->StarvedVoices() ? "  STARVED" : "",
                      audio_peak > 1.0f ? "  CLIPPING" : "");
        ui->Text(26, 224, line,
                 cs.overflowed_cells ? eng::Vec4{1.0f, 0.72f, 0.35f, 1.0f}
                                     : eng::Vec4{0.80f, 0.86f, 0.94f, 1.0f});
        std::snprintf(line, sizeof(line), "at: %s   (0-5 to travel)",
                      kStopList[std::clamp(here, 0, kStops - 1)].name);
        ui->Text(26, 200, line, eng::Vec4{0.92f, 0.94f, 0.98f, 1.0f});
        ui->Text(26, 248, "wasd: walk  space: run  qe: look down/up  f t o v b: effects  [ ]: sun  r: reset  p: passes",
                 eng::Vec4{0.62f, 0.68f, 0.78f, 1.0f});

        // --- the pass table -------------------------------------------------
        //
        // A SEPARATE PANEL, off by default, so the default frame a screenshot
        // captures is unchanged and the pixel comparisons in the commit
        // messages stay comparable. The headline row above is always on, which
        // is what keeps this from rotting the way the timing feature already
        // did once when nothing read it.
        //
        // These do NOT sum to the gpu figure and they are not meant to: a tiler
        // overlaps one pass's vertex work with the last one's fragment work, so
        // the sum exceeds the frame. The sum is printed anyway, because the
        // RATIO of the two is how much overlap the GPU found.
        if (profile_panel && !pass_ms.empty()) {
            // A TIMELINE, not a column of durations. Drawn as a bar per pass
            // positioned by begin_ms and end_ms, so the overlap the tiler finds
            // is a picture instead of a sum that does not add up.
            const int rows = int(pass_ms.size()) + 1;
            const int top = 300, w = 460, h = rows * 20 + 20;
            const int bar_x = 130, bar_w = w - bar_x - 14;
            double span = 0.0;
            for (const eng::rhi::GpuTiming& t : pass_ms)
                span = std::max(span, t.end_ms);
            span = std::max(span, 1e-3);
            ui->Rect(12, top, w, h, eng::Vec4{0.05f, 0.06f, 0.08f, 0.72f});
            ui->Outline(12, top, w, h, 1.0f, eng::Vec4{0.35f, 0.40f, 0.48f, 0.9f});
            int row = 0;
            for (const eng::rhi::GpuTiming& t : pass_ms) {
                const int y = top + 12 + row * 20;
                std::snprintf(line, sizeof(line), "%-9s %5.2f", t.label,
                              t.milliseconds);
                ui->Text(24, y + 6, line,
                         t.milliseconds >= slowest_ms
                             ? eng::Vec4{1.0f, 0.82f, 0.45f, 1.0f}
                             : eng::Vec4{0.80f, 0.86f, 0.94f, 1.0f});
                const float x0 = float(t.begin_ms / span) * float(bar_w);
                const float x1 = float(t.end_ms / span) * float(bar_w);
                ui->Rect(float(12 + bar_x) + x0, float(y) + 2.0f,
                         std::max(x1 - x0, 1.0f), 11.0f,
                         t.milliseconds >= slowest_ms
                             ? eng::Vec4{1.0f, 0.72f, 0.30f, 0.95f}
                             : eng::Vec4{0.42f, 0.62f, 0.86f, 0.95f});
                ++row;
            }
            std::snprintf(line, sizeof(line),
                          "gpu %.2f ms   span %.2f ms   %d passes",
                          app->Gpu().LastFrameGpuMilliseconds(), span,
                          int(pass_ms.size()));
            ui->Text(24, top + 12 + row * 20 + 6, line,
                     eng::Vec4{0.92f, 0.94f, 0.98f, 1.0f});
        }
        }  // hud_on

        // --- passes --------------------------------------------------------------
        // config.color, NOT a format of its own. The composite and the UI
        // pipelines are built for config.color (BGRA8Unorm); a capture target
        // in RGBA8Unorm is a pipeline/attachment mismatch that Metal rejects at
        // bind time, so the whole composite pass is dropped. This hardware
        // tolerated it -- the channel order came back right and every
        // screenshot in every commit message so far was taken through it --
        // which is exactly the kind of undefined behaviour that works here and
        // faults somewhere else.
        if (!shot_path.empty() && !eng::rhi::Valid(shot_target))
            shot_target = app->Gpu().CreateRenderTarget(f.width, f.height,
                                                        config.color,
                                                        /*cpu_readable=*/true);
        const eng::rhi::TextureId color = scene_targets.Hdr("color");
        const eng::rhi::TextureId ao = scene_targets.Color("ao");
        const eng::rhi::TextureId ms_color = scene_targets.Msaa("ms");
        const eng::rhi::TextureId ms_depth = scene_targets.MsaaDepth("ms_depth");
        const eng::rhi::TextureId depth = scene_targets.Depth("depth", true);
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
        if (local_shadows_on) {
            eng::RenderGraph::Pass p;
            p.name = "lampshadow";
            p.depth = app->Draw().ShadowAtlas();
            p.keep_depth = true;
            p.timer = "lampshadow";
            p.execute = [&](eng::rhi::Encoder& e) {
                // The counters accumulate across both shadow passes, so the
                // lamps' own contribution is the difference.
                const int before = app->Draw().ShadowDrawCount();
                const int before_culled = app->Draw().ShadowCulledCount();
                app->Draw().DrawLightShadows(e, scene);
                lamp_draws = app->Draw().ShadowDrawCount() - before;
                lamp_culled = app->Draw().ShadowCulledCount() - before_culled;
            };
            graph.AddPass(p);
        }
        // THE DEPTH PREPASS IS GONE. It existed only to manufacture a
        // single-sample depth texture for the passes that sample one -- fog,
        // SSAO, shafts, motion vectors -- because a multisample depth
        // attachment cannot be read, and it bought no early-Z at all: the scene
        // pass attaches its own multisample depth and clears it on entry. It
        // was a second full pass over the scene, 1.27 ms of a 5.05 ms frame at
        // 2200x1520. rhi::PassDesc::depth_resolve makes the hardware produce
        // the same texture for free at the end of the pass that was drawing
        // anyway.
        {
            eng::RenderGraph::Pass p;
            p.name = "scene";
            p.color = ms_color;
            p.resolve = color;
            p.depth = ms_depth;
            p.depth_resolve = depth;
            p.reads = {shadow_map};
            p.timer = "scene";
            p.execute = [&](eng::rhi::Encoder& e) {
                if (indirect_on) {
                    app->Draw().DrawSceneIndirect(e, scene, rw, rh, shadow_map);
                    indirect_batches = app->Draw().LastBatchCount();
                    indirect_draws = app->Draw().LastStats().draws;
                    rem_draws = 0;
                    if (!rem_scene.instances.empty()) {
                        app->Draw().DrawScene(e, rem_scene, rw, rh, shadow_map);
                        rem_draws = app->Draw().LastStats().draws;
                    }
                } else {
                    app->Draw().DrawScene(e, scene, rw, rh, shadow_map);
                }
                env->DrawSky(e, scene.camera, rw, rh);
            };
            graph.AddPass(p);
        }
        if (sparks_on) {
            // THE SPARKS MOVED OUT of the scene pass, and had to: they sample
            // the frame's depth to fade where they meet a surface, and the
            // resolve that produces a sampleable depth only happens when the
            // pass ENDS. So they blend over the resolved colour instead of the
            // multisample one -- a soft additive billboard loses nothing worth
            // measuring by being rasterised once rather than four times, and
            // what it gains is that the scene is no longer drawn twice.
            eng::RenderGraph::Pass p;
            // Not "sparks": the compute encoder that steps them already has
            // that name, and two rows with one label in a profile is worse than
            // no profile.
            p.name = "sparkdraw";
            p.modifies = color;
            p.reads = {depth};
            p.timer = "sparkdraw";
            p.execute = [&](eng::rhi::Encoder& e) {
                sparks->Draw(e, scene.camera, rw, rh, depth);
            };
            graph.AddPass(p);
        }
        if (shafts_on) {
            eng::VolumetricConfig vc;
            vc.far_distance = 110.0f;
            // Thin. This is air with dust in it, not smoke -- the whole effect
            // has to be something you notice without being able to point at.
            vc.extinction = shaft_ext;
            vc.base_density = 0.22f;
            // Denser low down, because that is where the dust and the moisture
            // are, and it is what makes a shaft visible near the ground and
            // invisible against the sky.
            vc.height_falloff = 0.055f;
            vc.height_reference = terrain.HeightAt(0.0f, 0.0f);
            // Strongly forward-scattering, which is why a shaft is bright when
            // you look toward the sun and nearly gone when you look away. An
            // isotropic phase gives uniform haze and no beams at all.
            // 0.60, not 0.72. The peak is what makes a beam a beam, but too
            // sharp a one turns the whole sky around the sun into a single
            // blown highlight and there is no beam left to see inside it.
            vc.anisotropy = 0.60f;
            vol->SetConfig(vc);
            graph.AddCompute("froxels", {vol->Volume()},
                             [&](eng::rhi::ComputeEncoder& e) {
                                 // The renderer's OWN cascade and light
                                 // buffers. A shaft computed against a second
                                 // copy of the shadow setup drifts away from
                                 // the shadow it is supposed to be cast by,
                                 // and the beam lands beside the gap.
                                 vol->Build(e, scene, rw, rh, f.index,
                                            shadow_map, app->Draw().CascadeBuffer(),
                                            app->Draw().CascadeOffset(),
                                            app->Draw().LightBuffer(),
                                            app->Draw().LightOffset(),
                                            int(scene.lights.size()));
                             },
                             "froxels");
            eng::RenderGraph::Pass p;
            p.name = "shafts";
            p.modifies = color;
            p.reads = {depth, vol->Volume()};
            p.timer = "shafts";
            p.execute = [&](eng::rhi::Encoder& e) {
                vol->Apply(e, scene, rw, rh, depth);
            };
            graph.AddPass(p);
        }
        if (ssao_on) {
            // CONTACT OCCLUSION, from the depth prepass the fog already needs.
            //
            // Nothing in this scene was darkened where it MEETS something else.
            // Two hundred trunks stood on the ground with the same brightness
            // at the root as at head height, which is the single clearest tell
            // that an object is pasted onto a scene rather than standing in it
            // -- the eye reads the missing contact darkening as "not touching".
            //
            // Cleared to WHITE, because this is a multiplier and an unwritten
            // texel has to mean "not occluded". Cleared to black it would
            // switch the lights off wherever the pass did not reach.
            eng::RenderGraph::Pass p;
            p.name = "ssao";
            p.color = ao;
            p.reads = {depth};
            p.clear_color[0] = p.clear_color[1] = p.clear_color[2] = 1.0f;
            p.timer = "ssao";
            p.execute = [&](eng::rhi::Encoder& e) {
                app->Draw().DrawSsao(e, scene.camera, rw, rh, depth,
                                     ssao_radius);
            };
            graph.AddPass(p);
        }
        // NOT BOTH. DrawFog is an analytic height fog and the froxel volume is
        // a marched one: they model the same air, and running both counts every
        // scattering event twice. With both on, looking into a low sun blew the
        // whole upper half of the frame to white and the meter dropped a stop
        // trying to save it, which crushed the ground instead.
        //
        // The froxel version is strictly the better one where it applies -- it
        // is shadowed, so it produces beams rather than a wash -- so it wins and
        // the analytic one is the fallback for when shafts are off.
        if (post->config.fog && !shafts_on) {
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
        eng::rhi::TextureId composited = color;
        if (post->config.taa) {
            // Velocity first: the resolve needs to know where each pixel was
            // last frame, or a moving camera smears everything it passes.
            graph.AddCompute("velocity", {post->Velocity()},
                             [&](eng::rhi::ComputeEncoder& e) {
                                 post->ComputeVelocity(e, depth);
                             },
                             "velocity");
            eng::RenderGraph::Pass p;
            p.name = "taa";
            p.color = post->Output();
            p.reads = {color, post->Velocity()};
            p.timer = "taa";
            p.execute = [&](eng::rhi::Encoder& e) { post->DrawTaa(e, color); };
            graph.AddPass(p);
            composited = post->Output();
        }
        if (sparks_on) {
            eng::VolumetricConfig unused;  // keeps the emitter block self-contained
            (void)unused;
            eng::ParticleEmitter em;
            em.position = eng::Vec3{world::kFirePit.x,
                                    terrain.HeightAt(world::kFirePit.x, world::kFirePit.z) + 0.25f,
                                    world::kFirePit.z};
            em.direction = eng::Vec3{0.0f, 1.0f, 0.0f};
            em.spread = 0.30f;
            em.speed = 2.6f;
            em.speed_variance = 1.1f;
            em.lifetime = 1.9f;
            em.lifetime_variance = 0.7f;
            em.size = 0.055f;
            em.size_variance = 0.03f;
            em.rate = 900.0f;
            // Hot at the source. The colour is a radiance, not a tint, so this
            // is what puts the sparks above the tone curve's knee and makes
            // them read as embers rather than as orange dots.
            em.color = eng::Vec4{14.0f, 4.2f, 0.9f, 1.0f};
            // Buoyant, not falling: hot air goes up, and gravity pointing down
            // here gives a fountain of sand.
            em.gravity = eng::Vec3{0.0f, 1.7f, 0.0f};
            em.drag = 1.4f;
            // STEPPED OUTSIDE THE GRAPH, for the third time and the same
            // reason: the simulation writes particle BUFFERS and the graph
            // orders passes by the textures they write. The skinning and the
            // light binning are out here for the same reason. Three separate
            // systems have now hit this, which is the graph's boundary showing
            // rather than three oversights.
            {
                auto e = app->Gpu().BeginCompute("sparks");
                sparks->Step(e, em, dt);
                app->Gpu().EndCompute();
            }
        }
        water_look.sun_dir = eng::Vec3{sky.sun_direction.x, sky.sun_direction.y,
                                       sky.sun_direction.z};
        water_look.sky_horizon = sky_horizon;

        // --- the water -------------------------------------------------------
        //
        // SCREEN-SPACE, in four passes, and the order is forced by what each
        // one reads: instanced spheres into a depth buffer of the water ALONE,
        // a separable bilateral smooth that turns those beads into one sheet,
        // then the optics -- fresnel sky reflection, background refraction and
        // a sun glint -- shaded from the smoothed sheet's screen-space normals.
        //
        // The water gets its OWN depth buffer and not the scene's. Sharing one
        // was tried in apps/fluid and it bakes the world into the surface: the
        // smoother cannot tell a sphere from a kerb, so the whole basin shades
        // as water. What the shade pass DOES need from the scene is its depth,
        // to know when a sphere is behind something -- and that is the resolved
        // depth the scene pass now produces, which did not exist until the
        // prepass was deleted. This district is only affordable because of it.
        if (!water_surface->BeginFrame(app->Gpu(), rw, rh, error))
            return Fail(error);
        const eng::rhi::TextureId watered = scene_targets.Hdr("watered");
        // BOUND BY VALUE, before the reassignment below. The pass lambdas
        // capture by reference and run inside graph.Execute, long after
        // `composited` has been pointed at `watered` -- so a shade pass that
        // read `composited` would read the target it is writing, and the frame
        // comes out black.
        const eng::rhi::TextureId dry = composited;
        {
            eng::RenderGraph::Pass p;
            p.name = "waterdepth";
            p.depth = water_surface->Depth();
            p.clear_depth = 0.0f;
            p.keep_depth = true;
            // Not "water": the compute encoder that steps the solver has that
            // name, and two rows with one label in a profile is worse than none.
            p.timer = "waterdepth";
            p.execute = [&](eng::rhi::Encoder& e) {
                water_surface->DrawDepth(e, *water, scene.camera, water_look, rw, rh);
            };
            graph.AddPass(p);
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "watersmoothh";
            p.color = water_surface->SmoothH();
            p.reads = {water_surface->Depth()};
            p.execute = [&](eng::rhi::Encoder& e) {
                water_surface->SmoothH(e, scene.camera, water_look, rw, rh);
            };
            graph.AddPass(p);
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "watersmoothv";
            p.color = water_surface->Smooth();
            p.reads = {water_surface->SmoothH()};
            p.execute = [&](eng::rhi::Encoder& e) {
                water_surface->SmoothV(e, scene.camera, water_look, rw, rh);
            };
            graph.AddPass(p);
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "watershade";
            p.color = watered;
            p.reads = {dry, depth, water_surface->Smooth()};
            p.timer = "watershade";
            p.execute = [&, dry](eng::rhi::Encoder& e) {
                water_surface->DrawShade(e, scene.camera, water_look, rw, rh,
                                         dry, depth);
            };
            graph.AddPass(p);
        }
        // Everything downstream reads the watered image, not the dry one.
        composited = watered;

        // --- bloom ---------------------------------------------------------
        //
        // THE ENGINE HAD IT AND THE VALLEY NEVER RAN IT. DrawComposite was
        // called with its bloom texture and strength left at their defaults, in
        // a scene whose lantern glass is emissive at {7.0, 4.6, 2.2}, whose fire
        // pit throws embers, and whose sparks sit at a radiance of {14, 4.2,
        // 0.9} with a comment saying that value "is what puts the sparks above
        // the tone curve's knee" -- above a knee that nothing was looking at, so
        // they clipped to white instead of glowing.
        //
        // Bright pass at half, two separable blurs, then two more at quarter.
        // Separable because a flat kernel costs n squared taps where two
        // one-dimensional passes cost 2n, and two scales because one blur wide
        // enough to read as a halo is also wide enough to be expensive.
        const eng::rhi::TextureId b_half = scene_targets.Hdr("bloomHalf", 2);
        const eng::rhi::TextureId b_half2 = scene_targets.Hdr("bloomHalf2", 2);
        const eng::rhi::TextureId b_quarter = scene_targets.Hdr("bloomQuarter", 4);
        const eng::rhi::TextureId b_quarter2 = scene_targets.Hdr("bloomQuarter2", 4);
        const eng::rhi::TextureId b_out = scene_targets.Hdr("bloomOut", 4);
        if (bloom_on) {
            const float hw = 2.0f / float(rw), hh = 2.0f / float(rh);
            const float qw = 4.0f / float(rw), qh = 4.0f / float(rh);
            const struct { const char* name; eng::rhi::TextureId out, in; float x, y; }
                kBlurs[] = {{"blurAx", b_half2, b_half, hw, 0.0f},
                            {"blurAy", b_quarter, b_half2, 0.0f, hh},
                            {"blurBx", b_quarter2, b_quarter, qw, 0.0f},
                            {"blurBy", b_out, b_quarter2, 0.0f, qh}};
            {
                eng::RenderGraph::Pass p;
                p.name = "bright";
                p.color = b_half;
                p.reads = {composited};
                p.timer = "bright";
                p.execute = [&](eng::rhi::Encoder& e) {
                    app->Draw().DrawBloomBright(e, composited, bloom_threshold, 0.9f);
                };
                graph.AddPass(p);
            }
            for (const auto& b : kBlurs) {
                eng::RenderGraph::Pass p;
                p.name = b.name;
                p.color = b.out;
                p.reads = {b.in};
                p.execute = [&, b](eng::rhi::Encoder& e) {
                    app->Draw().DrawBloomBlur(e, b.in, b.x, b.y);
                };
                graph.AddPass(p);
            }
        }
        {
            eng::RenderGraph::Pass p;
            p.name = "composite";
            p.color = shot_path.empty() ? f.drawable : shot_target;
            p.reads = ssao_on ? std::vector<eng::rhi::TextureId>{composited, ao}
                              : std::vector<eng::rhi::TextureId>{composited};
            if (bloom_on) p.reads.push_back(b_out);
            p.timer = "composite";
            p.execute = [&](eng::rhi::Encoder& e) {
                app->Draw().DrawComposite(e, composited,
                                          ssao_on ? ao : eng::rhi::TextureId{},
                                          bloom_on ? b_out : eng::rhi::TextureId{},
                                          bloom_on ? bloom_strength : 0.0f);
                ui->Draw(e);
            };
            graph.AddPass(p);
        }
        if (!graph.Compile(error)) return Fail(error);
        // NO SKINNING DISPATCH. It was here, and it produced a buffer this app
        // never read: SkinToBuffers' output reaches the world only through
        // Renderer::PosedVertices, whose only consumer is BuildSceneAccel,
        // which apps/world does not call. The forward path poses the mesh in
        // its own vertex stage. One compute encoder and one palette upload a
        // frame, for nothing. The engine keeps the entry point -- ray-traced
        // shadows from a posed mesh need it, and tests/skinned covers that.
        // THE FLUID, outside the graph for the fourth time and the same reason:
        // it writes particle BUFFERS and the graph orders passes by textures.
        {
            auto e = app->Gpu().BeginCompute("water");
            // A SPOUT AT ONE END AND A DRAIN AT THE OTHER, so the basin never
            // settles. Without it this is a bowl of water that finished moving
            // two seconds after the app started and cannot be disturbed again
            // -- the solver has no forces, no moving walls and no coupling to
            // anything, so the only motion it will ever produce is whatever the
            // seeding gave it. Measured: kinetic energy 63.5 at frame 30 and
            // 0.19 at frame 210.
            //
            // The jet points along +x and slightly up, so the water crosses the
            // basin, and the drain is the far bottom corner. What comes out is
            // a standing current rather than a fountain: at this scale a plume
            // that clears the rim just throws water on the grass, where the
            // solver's box wall stops it dead in mid-air.
            eng::FluidSim::Recirculation flow;
            flow.spout = eng::Vec3{spring.bounds_min.x + 0.18f,
                                   spring.bounds_min.y + 0.30f,
                                   (spring.bounds_min.z + spring.bounds_max.z) * 0.5f};
            flow.velocity = eng::Vec3{1.5f, 0.9f, 0.0f};
            flow.spread = 0.07f;
            flow.drain = eng::Vec3{spring.bounds_max.x - 0.25f, 0.0f,
                                   (spring.bounds_min.z + spring.bounds_max.z) * 0.5f};
            flow.drain_radius = 0.45f;
            flow.drain_y = spring.bounds_min.y + 0.12f;
            flow.per_step = 70;
            water->Recirculate(e, flow);
            (void)water->Step(e, dt);
            app->Gpu().EndCompute();
        }
        // BINNING, outside the graph for the same reason the exposure meter is:
        // it writes BUFFERS, and the graph orders passes by the textures they
        // write. Before the scene pass, not after -- unlike the meter, this
        // frame's shading needs this frame's bins.
        if (app->Draw().ClusteredLighting() && !scene.lights.empty()) {
            auto e = app->Gpu().BeginCompute("bin");
            app->Draw().BinLights(e, scene, rw, rh, 90.0f);
            app->Gpu().EndCompute();
        }
        // GPU-DRIVEN cull, outside the graph for the same reason as skinning:
        // the scene pass lambda above reads rem_scene, which is only known
        // after this runs. The cull itself is a compute dispatch (frustum on
        // the GPU); the remainder collection is plain CPU.
        if (indirect_on) {
            {
                auto e = app->Gpu().BeginCompute("cull");
                (void)app->Draw().CullScene(e, scene, rw, rh);
                app->Gpu().EndCompute();
            }
            rem_scene = scene;
            rem_scene.instances.clear();
            for (std::size_t i = 0; i < scene.instances.size(); ++i)
                if (!app->Draw().WasBatched(int(i)))
                    rem_scene.instances.push_back(scene.instances[i]);
        }
        graph.Execute(app->Gpu());

        // METERING RUNS OUTSIDE THE GRAPH, and the graph is what insisted:
        // what this produces is a BUFFER, the graph orders passes by the
        // textures they write, and a pass writing nothing it tracks has no
        // defined place in the order. It said so and refused to compile.
        //
        // So it goes here, after the composite, reading the colour this frame
        // produced and leaving the exposure for the NEXT frame to apply. That
        // is not a workaround, it is how metering is normally done -- exposing
        // the frame you are about to composite means waiting for it to finish
        // first, and one frame of lag in a value already smoothed over half a
        // second is invisible.
        {
            auto e = app->Gpu().BeginCompute("meter");
            post->MeterExposure(e, color);
            app->Gpu().EndCompute();
        }
        post->EndFrame();
        app->EndFrame();
        // AFTER EndFrame, so cpu_ms covers the whole loop body including the
        // commit, and so the timings are the newest the GPU has finished. Both
        // are read by the HUD one frame later.
        cpu_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - cpu_begin)
                     .count();
        pass_ms = app->Gpu().LastFrameTimings();
        // A GPU FAULT IS NOT A BLACK FRAME ANY MORE. The description names the
        // encoders that faulted; printed once each because the read clears it,
        // and the count stays on the HUD after the text is gone.
        if (const std::string fault = app->Gpu().TakeGpuFault(); !fault.empty())
            std::fprintf(stderr, "GPU FAULT %s\n", fault.c_str());
        // Every frame during a capture, because the two things most likely to
        // be wrong in a screenshot are an exposure that has not settled and a
        // camera that is not where it was asked to be.
        if (!shot_path.empty()) {
            std::printf("  frame %3d  cpu %6.3f  gpu %6.3f  period %6.3f ms |",
                        int(f.index), cpu_ms,
                        app->Gpu().LastFrameGpuMilliseconds(), period_ms);
            // begin-end, not just a duration: the passes overlap, and a log of
            // durations that sum past the frame time has fooled every reader.
            for (const eng::rhi::GpuTiming& t : pass_ms)
                std::printf(" %s %.2f-%.2f", t.label, t.begin_ms, t.end_ms);
            std::printf("\n");
            if (indirect_on)
                std::printf("             indirect %d draws over %d batches + %d remainder, "
                            "shadow %d/%d culled, %d dropped, %d sparks\n",
                            indirect_draws, indirect_batches, rem_draws,
                            app->Draw().ShadowDrawCount(), app->Draw().ShadowCulledCount(),
                            app->Draw().DroppedThisFrame(), sparks->LiveCountSlow());
            else
                std::printf("             %d draws (%d transparent), %d culled, "
                            "shadow %d/%d culled, %d dropped, %d sparks\n",
                            app->Draw().LastStats().draws, app->Draw().LastStats().transparent_draws, app->Draw().LastStats().culled,
                            app->Draw().ShadowDrawCount(), app->Draw().ShadowCulledCount(),
                            app->Draw().DroppedThisFrame(), sparks->LiveCountSlow());
        }

        if (!shot_path.empty() && int(f.index) >= shot_frames) {
            std::vector<std::uint8_t> px(std::size_t(f.width) * f.height * 4);
            if (!app->Gpu().ReadPixels(shot_target, f.width, f.height, px))
                return Fail("readback");
            // BGRA OUT, RGBA IN. ReadPixels returns the texture's own bytes in
            // the texture's own order, and the capture target is config.color
            // -- BGRA8Unorm -- because the composite and UI pipelines are built
            // for that and a mismatch is a pipeline Metal refuses to bind.
            // png::EncodeFile documents RGBA. Two channels, swapped once.
            //
            // Worth spelling out because the previous arrangement was wrong in
            // a way that LOOKED right: the target was RGBA8Unorm against BGRA
            // pipelines, which Metal rejects under validation and this hardware
            // tolerated, and the tolerated behaviour happened to store the
            // channels in the order the encoder wanted. Fixing the format
            // without fixing the encode turned every fire in the demo blue.
            if (config.color == eng::rhi::Format::BGRA8Unorm)
                for (std::size_t i = 0; i + 3 < px.size(); i += 4)
                    std::swap(px[i], px[i + 2]);
            if (!eng::png::EncodeFile(shot_path, px, f.width, f.height, error))
                return Fail(error);
            std::printf("wrote %s (%dx%d)\n", shot_path.c_str(), f.width, f.height);
            // ONE LINE A TEST CAN READ. peak in flight is the one that matters:
            // it must never exceed kFramesInFlight, and the GI re-bake used to
            // leak a permit per bake until the third one blocked forever.
            std::printf("summary: peak in flight %d, gpu faults %d, dropped %d, "
                        "instances %zu, draws %d, shadow %d/%d culled, slices %d\n",
                        app->Gpu().PeakFramesInFlight(), app->Gpu().GpuFaultCount(),
                        app->Draw().DroppedThisFrame(), scene.instances.size(),
                        app->Draw().LastStats().draws, app->Draw().ShadowDrawCount(),
                        app->Draw().ShadowCulledCount(),
                        app->Draw().UniformSlicesThisFrame());
            if (local_shadows_on)
                std::printf("lamps: %d casters submitted, %d rejected, %d tiles of %d lights\n",
                            lamp_draws, lamp_culled, app->Draw().ShadowTilesUsed(),
                            int(scene.lights.size()));
            return 0;
        }
    }
    return 0;
}
