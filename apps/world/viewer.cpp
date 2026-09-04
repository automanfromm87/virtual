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
#include <thread>
#include <cstdio>
#include <string>
#include <vector>

#include <cstdlib>
#include <cstring>

#include "engine/app/app.h"
#include "engine/asset/png.h"
#include "engine/geometry/mesh.h"
#include "engine/geometry/simplify.h"
#include "engine/asset/texgen.h"
#include "engine/render/gi.h"
#include "engine/render/volumetric.h"
#include "engine/geometry/terrain.h"
#include "engine/geometry/tree.h"
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

int main(int argc, char** argv) {
    // CAPTURE MODE. Renders a fixed number of frames into a readable target and
    // writes a PNG instead of opening on the drawable.
    //
    // The drawable cannot be read back, so a bug that only shows on screen
    // cannot be looked at from a terminal -- which is how this demo came to be
    // washed out without anyone noticing.
    std::string shot_path;
    int shot_frames = 12;
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
    // 0.004. At 0.014 the far treeline vanished entirely and the meter dropped
    // most of a stop trying to save the sun glow, which crushed the ground.
    // Looking into a low sun through trees IS hazy -- that is the effect -- but
    // the haze has to leave the trees in it.
    float shaft_ext = 0.004f;
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
        else if (std::strcmp(argv[i], "--shaft") == 0 && i + 1 < argc)
            shaft_ext = float(std::atof(argv[++i]));
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
    // LIGHT SHAFTS. The one effect a forest is actually about: the canopy cuts
    // the sunlight into beams and you see them in the air, not on a surface.
    // Everything else here lights things; this lights the space between them.
    auto vol = eng::Volumetrics::Create(app->Gpu(), error);
    if (!vol) return Fail(error);

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
    struct ForestCell { eng::Mesh trunk, leaves; };
    std::vector<ForestCell> cells(kForestLods * kForestCells * kForestCells);
    // ONE SPHERE PER CANOPY, kept for the light bake below. The forest is 2.5M
    // triangles and tracing rays against that is hopeless, but GI does not need
    // the shape of a leaf -- it needs to know that the sky is blocked here and
    // that what bounces off it is green. A coarse sphere per tree says both, in
    // a hundredth of the geometry.
    struct CanopyProxy { eng::Vec3 centre; float radius; };
    std::vector<CanopyProxy> canopies;
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
            const eng::Mat4 model = eng::Mat4::Translation(at) *
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
            ++planted;
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

    eng::MaterialDesc leaf_md;
    leaf_md.shading = eng::Shading::Lit;
    leaf_md.base_color = eng::Vec4{0.19f, 0.38f, 0.15f, 1.0f};
    // Leaves are rougher than almost anything else outdoors and not at all
    // metallic; a smooth canopy picks up a sheen that reads as wet plastic.
    leaf_md.roughness = 0.95f;
    leaf_md.albedo = leaf_albedo;
    // No normal map: a leaf blob already stands in for a thousand leaves, and
    // bump detail on it would be detail at the wrong scale pretending to be
    // detail at the right one.
    leaf_md.uv_scale = eng::Vec2{2.0f, 2.0f};
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
        eng::Vec3 sky_up{0.0f, 0.0f, 0.0f}, sky_down{0.0f, 0.0f, 0.0f};
        {
            constexpr int kFace = 4;
            app->Gpu().BeginFrame();
            {
                auto e = app->Gpu().BeginCompute();
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
            }
        }
        std::printf("    sun %.2f, sky %.2f (%.1f:1) at %.0f degrees\n",
                    scene.lightColor.y * std::max(sky.sun_direction.y, 0.0f), sky_up.y,
                    double(scene.lightColor.y * std::max(sky.sun_direction.y, 0.0f) /
                           std::max(sky_up.y, 1e-4f)),
                    sun_elevation * 57.2958f);
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
        eng::Vec3 centre;
        std::size_t trunk_instance = 0, leaf_instance = 0;
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
            forest.push_back(std::move(chunk));
        }
        std::printf("  %zu cells, %zu draws, canopy %zu tris at level 0 and %zu "
                    "at level 2 (%.2f s)\n",
                    forest.size(), forest.size() * 2, fine_tris, coarse_tris,
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count());
    }

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
    // Half a stop under the meter. The meter puts the middle fifth of the
    // histogram at middle grey, which is right, but this scene is bimodal --
    // hard sun and deep shade, about three and a half stops apart -- so the
    // middle fifth falls between the two modes and the sunlit ground sits at
    // 214 of 255. Nothing clips, measured, but it is a high-key image and half
    // a stop down puts the sun side where it reads as grass.
    post->config.exposure_compensation = -0.5f;
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
    float camera_yaw = 0.7f + yaw_offset, camera_pitch = 0.26f,
          camera_distance = 13.0f;
    float baked_az = 1e9f, baked_el = 1e9f;
    bool ssao_on = ssao_start;
    bool shafts_on = shafts_start;
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
    app->Actions().Bind("reset", 'r');

    eng::RenderGraph graph;
    // Allocated lazily on the first frame, when the framebuffer size is known.
    eng::rhi::TextureId shot_target;

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
        if (app->Actions().Pressed("taa")) post->config.taa = !post->config.taa;
        if (app->Actions().Pressed("ssao")) ssao_on = !ssao_on;
        if (app->Actions().Pressed("shafts")) shafts_on = !shafts_on;

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
        } else if (gi_stale && ++gi_still_frames > 12) {
            if (!bake_indirect()) return Fail(error);
            gi_stale = false;
        }
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
            ++forest_lod_counts[lod];
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
        // JITTER, TWICE. BeginFrame advances the sequence and also builds the
        // previous-frame matrices from the camera it is handed, so the camera
        // needs this frame's offset going in and the next one coming out.
        // Without it TAA is a pure blur -- every frame samples the same points
        // and averaging them recovers nothing.
        scene.camera.jitter = post->Jitter();
        post->BeginFrame(scene.camera, f.width, f.height, dt);
        scene.camera.jitter = post->Jitter();

        // --- HUD ----------------------------------------------------------------
        ui->Begin(f.width, f.height);
        ui->Rect(12, 12, 470, 180, eng::Vec4{0.05f, 0.06f, 0.08f, 0.72f});
        ui->Outline(12, 12, 470, 180, 1.0f, eng::Vec4{0.35f, 0.40f, 0.48f, 0.9f});
        char line[192];
        const eng::RenderStats& rs = app->Draw().LastStats();
        std::snprintf(line, sizeof(line),
                      "%zu chunks lod %d/%d/%d   forest %d/%d/%d   %d draws, %d culled",
                      chunks.size(), lod_counts[0], lod_counts[1], lod_counts[2],
                      forest_lod_counts[0], forest_lod_counts[1],
                      forest_lod_counts[2], rs.draws, rs.culled);
        ui->Text(26, 26, line, eng::Vec4{0.92f, 0.94f, 0.98f, 1.0f});
        std::snprintf(line, sizeof(line),
                      "sun %.0f deg   gi %s (%.0f ms, %d buried)",
                      sun_elevation * 57.2958f,
                      gi_stale ? "STALE" : "baked", last_bake_seconds * 1000.0,
                      last_bake_dark);
        ui->Text(26, 50, line,
                 gi_stale ? eng::Vec4{1.0f, 0.72f, 0.35f, 1.0f}
                          : eng::Vec4{0.80f, 0.86f, 0.94f, 1.0f});
        std::snprintf(line, sizeof(line), "path: %zu points, at %zu   ground %.2f m",
                      path.size(), path_at, terrain.HeightAt(player.Feet().x,
                                                             player.Feet().z));
        ui->Text(26, 74, line, eng::Vec4{0.80f, 0.86f, 0.94f, 1.0f});
        std::snprintf(line, sizeof(line),
                      "gpu %.2f ms  fog %s  taa %s  ao %s  exposure %.2f (%+.1f EV)  %s",
                      app->Gpu().LastFrameGpuMilliseconds(),
                      post->config.fog ? "on" : "off",
                      post->config.taa ? "on" : "off", ssao_on ? "on" : "off",
                      post->LastExposure(),
                      std::log2(std::max(post->LastExposure(), 1e-6f)),
                      player.Grounded() ? "grounded" : "airborne");
        ui->Text(26, 98, line, eng::Vec4{0.80f, 0.86f, 0.94f, 1.0f});
        ui->Text(26, 128, "click: walk there    right-drag: orbit    scroll: zoom",
                 eng::Vec4{0.62f, 0.68f, 0.78f, 1.0f});
        ui->Text(26, 152, "f: fog   t: taa   o: ao   v: shafts   [ ]: sun   r: reset",
                 eng::Vec4{0.62f, 0.68f, 0.78f, 1.0f});

        // --- passes --------------------------------------------------------------
        if (!shot_path.empty() && !eng::rhi::Valid(shot_target))
            shot_target = app->Gpu().CreateRenderTarget(
                f.width, f.height, eng::rhi::Format::RGBA8Unorm,
                /*cpu_readable=*/true);
        const eng::rhi::TextureId color = app->Targets().Hdr("color");
        const eng::rhi::TextureId ao = app->Targets().Color("ao");
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
                                 vol->Build(e, scene, f.width, f.height, f.index,
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
                vol->Apply(e, scene, f.width, f.height, depth);
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
                app->Draw().DrawSsao(e, scene.camera, f.width, f.height, depth,
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
        {
            eng::RenderGraph::Pass p;
            p.name = "composite";
            p.color = shot_path.empty() ? f.drawable : shot_target;
            p.reads = ssao_on ? std::vector<eng::rhi::TextureId>{composited, ao}
                              : std::vector<eng::rhi::TextureId>{composited};
            p.timer = "composite";
            p.execute = [&](eng::rhi::Encoder& e) {
                app->Draw().DrawComposite(e, composited, ssao_on ? ao : eng::rhi::TextureId{});
                ui->Draw(e);
            };
            graph.AddPass(p);
        }
        if (!graph.Compile(error)) return Fail(error);
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
            auto e = app->Gpu().BeginCompute();
            post->MeterExposure(e, color);
            app->Gpu().EndCompute();
        }
        post->EndFrame();
        app->EndFrame();
        // Every frame during a capture, because the two things most likely to
        // be wrong in a screenshot are an exposure that has not settled and a
        // camera that is not where it was asked to be.
        if (!shot_path.empty())
            std::printf("  frame %3d  gpu %.3f ms  draws %d  culled %d\n",
                        int(f.index), app->Gpu().LastFrameGpuMilliseconds(),
                        app->Draw().LastStats().draws, app->Draw().LastStats().culled);

        if (!shot_path.empty() && int(f.index) >= shot_frames) {
            std::vector<std::uint8_t> px(std::size_t(f.width) * f.height * 4);
            if (!app->Gpu().ReadPixels(shot_target, f.width, f.height, px))
                return Fail("readback");
            if (!eng::png::EncodeFile(shot_path, px, f.width, f.height, error))
                return Fail(error);
            std::printf("wrote %s (%dx%d)\n", shot_path.c_str(), f.width, f.height);
            return 0;
        }
    }
    return 0;
}
