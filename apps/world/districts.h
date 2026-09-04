// The districts: one valley with a place in it for each thing the engine does.
//
// WHY ONE SCENE AND NOT THIRTY. Every feature used to have a demo of its own,
// each with a camera pointed at exactly the thing it wanted to show and nothing
// else in frame. That is the right way to TEST something and a bad way to see
// whether it works: a material gallery on a black background says nothing about
// whether those materials sit in a landscape, and a hall of two hundred lights
// with no daylight never has to agree with a sun.
//
// Putting them in one world makes them argue with each other, which is the
// point. The lanterns have to look like lanterns at noon and at dusk. The glass
// has to be transparent against foliage rather than against grey. The exposure
// meter has to find a setting that works for a lit hall and an open field in
// the same frame -- and if it cannot, that is a real thing to know and thirty
// separate demos would each have hidden it.
//
// BOXES ARE BUILT AT SIZE, NOT SCALED. Mat4 has only a uniform Scale, and that
// is deliberate: the renderer transforms normals by the model matrix and
// renormalises, which is right for rotation and uniform scale and wrong for a
// non-uniform one -- that needs the inverse transpose. A pillar squashed into
// shape by a matrix would be lit as though it were still a cube. So each
// distinct box shape is its own mesh, which is a handful of meshes and no
// special case anywhere else.
//
// AND EACH ONE ADDS ITS COLLIDERS. This was missed entirely on the first pass:
// the physics world had exactly one body in it, the terrain, so the character
// walked through every pillar, stone, pane and tree in the valley. Nothing
// reported it because nothing was wrong -- there was simply nothing there to
// hit. A district that draws something solid and does not collide it is half
// built, so the world goes in beside the scene rather than being somebody
// else's job to remember.
//
// EACH DISTRICT APPENDS to a scene it does not own. They return the handles the
// frame loop needs to animate them and nothing else; placement, materials and
// geometry are decided here so the loop stays a loop.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "engine/geometry/mesh.h"
#include "engine/physics/physics.h"
#include "engine/render/renderer.h"
#include "engine/scene/scene.h"

namespace world {

constexpr float kPi = 3.14159265358979f;

// EVERY PIECE OF GEOMETRY GOES IN THROUGH ONE OF THESE TWO, and
// scene.instances.push_back is not called anywhere else in this file.
//
// The first version added colliders "beside the geometry", which sounds like
// enough and is not: eleven things were drawn and five were collided, and the
// six that were missed included the gallery's display spheres -- so the
// character walked into the middle of them and stood there overlapping four at
// once. Nothing reported it, because a missing collider is an absence.
//
// Naming the two cases makes the choice explicit at every call. A decoration is
// a deliberate statement that this thing has no substance -- a mark painted on
// the ground, a hanging cloth -- and it reads as one. Forgetting is no longer
// available.
inline void AddSolid(eng::Scene& scene, eng::physics::World& world,
                     const eng::Instance& in, const eng::physics::Shape& shape,
                     eng::Vec3 at, eng::Quat orient = eng::Quat{0, 0, 0, 1}) {
    scene.instances.push_back(in);
    eng::physics::Body b;
    b.shape = shape;
    b.position = at;
    b.orientation = orient;
    b.SetMass(0.0f);
    world.Add(b);
}

// Drawn and not solid, on purpose.
inline void AddDecoration(eng::Scene& scene, const eng::Instance& in) {
    scene.instances.push_back(in);
}

// Where each district sits, in metres from the middle of the valley.
//
// All of them are outside the 26 m spawn clearing and inside the 46 m rim where
// the bowl starts climbing. They are far enough apart that no two are ever the
// only thing in frame, which is the arrangement that makes them useful.
// PLACED INSIDE THE BOWL, not on its rim.
//
// These were at radius 42 to 46, and the bowl wall in Landscape() starts at 46
// and climbs 0.55 m per metre: the lantern hall's far corner was 4.6 m up it,
// with its pillars buried to the capital. Flattening a pad under each district
// fixes the burying wherever they are, but on the rim the pad has several
// metres of hillside to cut away and reads as a crater. Ten metres further in
// costs nothing -- the playable area is 46 m of radius and the character walks
// it in twenty seconds -- and the pads become a levelling touch rather than
// earthworks.
inline constexpr eng::Vec3 kGallery{27.0f, 0.0f, -21.0f};
inline constexpr eng::Vec3 kLanternHall{-28.0f, 0.0f, -22.0f};
inline constexpr eng::Vec3 kGlassPavilion{24.0f, 0.0f, 24.0f};
inline constexpr eng::Vec3 kFirePit{-22.0f, 0.0f, 23.0f};
inline constexpr eng::Vec3 kFlag{14.0f, 0.0f, 6.0f};

// A DISTRICT IS BUILT ON A LEVEL PAD, and the terrain function has to cut it.
//
// Each builder below takes one `ground_y`, sampled once at the district's
// centre, and places every pillar and plinth at that height. That is right for
// architecture -- a colonnade whose bases follow a hillside is not a colonnade
// -- but it is only right if the ground under it is actually level. The lantern
// hall is 17 by 20 metres and its centre sits at r=45.6, one metre inside the
// bowl rim at r=46 where the ground climbs 0.55 m per metre: its far corner was
// 4.6 m up the wall, so the pillars there were buried to the capital and their
// lanterns were underground. Nothing complained, because a collider inside a
// hill is still a collider -- you simply cannot reach it, and it never stops
// anyone. The self-check in the viewer counts exactly that case.
//
// So the flattening lives here, next to the coordinates it depends on, rather
// than in the terrain function where it would be five magic circles that
// silently stop matching when a district moves.
struct Pad {
    eng::Vec3 centre;
    float radius;  // level out to here, then blend back over kPadBlend metres
};
inline constexpr Pad kPads[] = {
    {kGallery, 8.0f},        // plinth is 6.1 x 6.1
    {kLanternHall, 13.5f},   // 5x6 pillars at 3.4 m spacing
    {kGlassPavilion, 6.5f},  // a fan of panes at radius 4.2
    {kFirePit, 7.0f},        // stones at radius 3.4, embers out to 5
    {kFlag, 4.0f},           // one pole and its shadow
};
// HOW FAST THE GROUND MAY CLIMB AWAY from a pad, as a gradient.
//
// The first version blended the pad back into the terrain over a fixed nine
// metres with a smoothstep, and it walled off four of the five districts: the
// pad has to absorb the whole height change across its radius into the blend,
// so the join came out at about twice the natural slope -- 67 degrees where the
// bare ground was 34 -- against a character that can climb 45. Widening the
// blend enough to fix that made the pads overlap each other.
//
// So the join is a CONE CLAMP instead: past the pad's edge the ground may
// differ from pad level by this gradient times the distance, and is otherwise
// left alone. The slope of the result is then either the terrain's own or
// exactly this, never a product of the two, and a deep drop simply takes
// further to reach instead of turning into a wall. There is nothing to tune
// against the pad radius, which is what made the blend width fragile.
//
// 0.55 is 29 degrees, comfortably inside the character's 45 and inside the 31
// the forest requires to plant a tree, so a pad skirt stays walkable and
// plantable rather than becoming a ring of bare steep ground.
inline constexpr float kPadSlope = 0.55f;

// A deterministic little generator, so a district is the same every run and two
// screenshots can be compared. Same xorshift the tree generator uses.
struct Rng {
    std::uint32_t s;
    explicit Rng(std::uint32_t seed) : s(seed ? seed : 1u) {}
    float Unit() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return float(s & 0xFFFFFFu) / float(0x1000000u);
    }
    float Signed() { return Unit() * 2.0f - 1.0f; }
};

// --- the material gallery ---------------------------------------------------
//
// A grid of spheres over roughness and metallic, on a plinth, OUTDOORS. The
// point of putting it in the valley rather than in a box is that these are the
// two parameters everything else in the renderer feeds: image-based lighting,
// the multi-scatter compensation, the tone curve's treatment of a highlight.
// Against a black background a metal sphere is a silhouette with a dot on it.
// Against a sky and a treeline it either reflects the world or it does not.
struct Gallery {
    static constexpr int kCols = 6;  // roughness, smooth to rough
    static constexpr int kRows = 4;  // metallic, dielectric to metal
};

inline void BuildGallery(eng::Renderer& r, eng::Scene& scene,
                         eng::physics::World& world, eng::MeshHandle sphere,
                         float ground_y, std::string& error) {
    // The plinth first, so the spheres have something to sit on that is
    // obviously man-made -- a sphere floating over grass reads as a bug.
    eng::MaterialDesc stone;
    stone.shading = eng::Shading::Lit;
    stone.base_color = eng::Vec4{0.20f, 0.19f, 0.18f, 1.0f};
    stone.roughness = 0.75f;
    const eng::MaterialHandle stone_mat = r.CreateMaterial(stone, error);

    constexpr float kStep = 1.7f;
    constexpr float kTop = 0.55f;
    const float w = kStep * float(Gallery::kCols) * 0.5f + 1.0f;
    const float d = kStep * float(Gallery::kRows) * 0.5f + 1.0f;
    eng::Instance plinth;
    plinth.mesh = r.UploadMesh(
        eng::MakeBox(eng::Vec3{w, kTop * 0.5f, d}, eng::Vec4{1, 1, 1, 1}));
    plinth.material = stone_mat;
    const eng::Vec3 plinth_at{kGallery.x, ground_y + kTop * 0.5f, kGallery.z};
    plinth.model = eng::Mat4::Translation(plinth_at);
    AddSolid(scene, world, plinth,
             eng::physics::Shape::MakeBox(eng::Vec3{w, kTop * 0.5f, d}), plinth_at);

    for (int row = 0; row < Gallery::kRows; ++row)
        for (int col = 0; col < Gallery::kCols; ++col) {
            eng::MaterialDesc md;
            md.shading = eng::Shading::Lit;
            // A copper-ish base, because a grey one hides the single most
            // visible difference between a metal and a dielectric: a metal
            // tints its reflection and a dielectric does not.
            md.base_color = eng::Vec4{0.94f, 0.62f, 0.42f, 1.0f};
            // NOT down to zero roughness. A perfect mirror shows the
            // environment cube's resolution and nothing about the material,
            // and 0.04 is about as smooth as anything real gets.
            md.roughness = 0.04f + float(col) / float(Gallery::kCols - 1) * 0.92f;
            md.metallic = float(row) / float(Gallery::kRows - 1);
            const eng::MaterialHandle m = r.CreateMaterial(md, error);
            if (!eng::Valid(m)) return;

            const eng::Vec3 at{
                kGallery.x + (float(col) - float(Gallery::kCols - 1) * 0.5f) * kStep,
                ground_y + kTop + 0.6f,
                kGallery.z + (float(row) - float(Gallery::kRows - 1) * 0.5f) * kStep};
            eng::Instance in;
            in.mesh = sphere;
            in.material = m;
            in.model = eng::Mat4::Translation(at);
            // The mesh is a unit sphere of radius 0.5 drawn unscaled, so the
            // collider is that and not the grid spacing -- getting this wrong
            // is how a collider ends up the right count and the wrong size.
            AddSolid(scene, world, in, eng::physics::Shape::MakeSphere(0.5f), at);
        }
}

// --- the lantern hall -------------------------------------------------------
//
// A colonnade with a lantern on every pillar: the clustered lighting path, in
// daylight, next to a forest. Two hundred and forty lights is well past what a
// forward renderer does one at a time, and the whole reason for cutting the
// frustum into froxels.
//
// OPEN, not enclosed. A roofed hall would make the lanterns the only light and
// the district would be a night scene sitting inside a day one -- which is the
// separate-demo failure this file exists to avoid. Open, they have to hold
// their own against the sun, which is the harder and more honest case.
struct LanternHall {
    std::vector<std::size_t> lights;  // indices into Scene::lights
    std::vector<eng::Vec3> at;        // where each one hangs
};

inline LanternHall BuildLanternHall(eng::Renderer& r, eng::Scene& scene,
                                    eng::physics::World& world,
                                    eng::MeshHandle sphere, float ground_y,
                                    std::string& error) {
    LanternHall hall;
    eng::MaterialDesc pillar_md;
    pillar_md.shading = eng::Shading::Lit;
    pillar_md.base_color = eng::Vec4{0.26f, 0.24f, 0.22f, 1.0f};
    pillar_md.roughness = 0.68f;
    const eng::MaterialHandle pillar_mat = r.CreateMaterial(pillar_md, error);

    // The lantern glass itself is EMISSIVE, so the source is visible as well as
    // its effect. A point light with nothing where it is supposed to be reads
    // as the scene glowing for no reason, which is the commonest tell that a
    // renderer has lights but no lamps.
    eng::MaterialDesc glass_md;
    glass_md.shading = eng::Shading::Lit;
    glass_md.base_color = eng::Vec4{0.9f, 0.7f, 0.4f, 1.0f};
    glass_md.roughness = 0.3f;
    glass_md.emissive = eng::Vec3{7.0f, 4.6f, 2.2f};
    const eng::MaterialHandle glass_mat = r.CreateMaterial(glass_md, error);
    if (!eng::Valid(pillar_mat) || !eng::Valid(glass_mat)) return hall;

    constexpr int kRows = 6, kCols = 5;
    constexpr float kSpan = 3.4f, kHeight = 4.2f;
    const eng::MeshHandle pillar_mesh = r.UploadMesh(
        eng::MakeBox(eng::Vec3{0.22f, kHeight * 0.5f, 0.22f}, eng::Vec4{1, 1, 1, 1}));
    Rng rng(4919);
    for (int row = 0; row < kRows; ++row)
        for (int col = 0; col < kCols; ++col) {
            const eng::Vec3 base{
                kLanternHall.x + (float(col) - float(kCols - 1) * 0.5f) * kSpan,
                ground_y,
                kLanternHall.z + (float(row) - float(kRows - 1) * 0.5f) * kSpan};

            eng::Instance p;
            p.mesh = pillar_mesh;
            p.material = pillar_mat;
            const eng::Vec3 pillar_at = base + eng::Vec3{0, kHeight * 0.5f, 0};
            p.model = eng::Mat4::Translation(pillar_at);
            AddSolid(scene, world, p,
                     eng::physics::Shape::MakeBox(eng::Vec3{0.22f, kHeight * 0.5f, 0.22f}),
                     pillar_at);

            // Two lanterns per pillar at different heights, which is what takes
            // this past a hundred lights and into the range where clustering is
            // the difference between working and not.
            for (int k = 0; k < 2; ++k) {
                const eng::Vec3 at = base + eng::Vec3{0.0f, kHeight - 0.5f - float(k) * 1.6f, 0.0f};
                eng::Instance bulb;
                bulb.mesh = sphere;
                bulb.material = glass_mat;
                bulb.model = eng::Mat4::Translation(at) * eng::Mat4::Scale(0.34f);
                // The lower lantern hangs at head height, so it is solid. The
                // unit sphere is radius 0.5 and the scale is 0.34.
                AddSolid(scene, world, bulb,
                         eng::physics::Shape::MakeSphere(0.5f * 0.34f), at);

                eng::Light light;
                light.type = eng::LightType::Point;
                light.position = at;
                // Warm, and varied. Every lamp the same colour is the other
                // half of the tell -- real ones differ, and the variation is
                // what lets the eye read them as separate sources rather than
                // as one glow.
                const float warm = 0.85f + rng.Unit() * 0.3f;
                // Bright enough to be a light and not a decoration. At 2.6 the
                // bulbs glowed and the ground under them did not change, which
                // is the same failure as having no lights at all only harder to
                // notice.
                light.color = eng::Vec3{9.0f * warm, 5.8f * warm, 2.8f};
                light.range = 9.0f;
                hall.lights.push_back(scene.lights.size());
                hall.at.push_back(at);
                scene.lights.push_back(light);
            }
        }
    return hall;
}

// --- the glass pavilion -----------------------------------------------------
//
// Order-independent transparency, against foliage. Glass over a grey background
// only has to be dimmer; glass over a treeline has to let a recognisable
// treeline through, and overlapping panes have to accumulate rather than pick a
// winner -- which is the whole difference between weighted-blended OIT and a
// back-to-front sort that gets the order wrong when depths tie.
inline void BuildGlassPavilion(eng::Renderer& r, eng::Scene& scene,
                               eng::physics::World& world, float ground_y,
                               std::string& error) {
    eng::MaterialDesc frame_md;
    frame_md.shading = eng::Shading::Lit;
    // NOT nearly-black metal. A metal has no diffuse at all, so a dark one
    // outdoors is whatever the sky happens to put in its specular lobe and
    // reads as a hole in the picture. Lighter, and less than fully metallic.
    frame_md.base_color = eng::Vec4{0.42f, 0.44f, 0.47f, 1.0f};
    frame_md.roughness = 0.30f;
    frame_md.metallic = 0.55f;
    const eng::MaterialHandle frame_mat = r.CreateMaterial(frame_md, error);
    if (!eng::Valid(frame_mat)) return;

    // ALPHA COMES FROM THE VERTEX COLOUR, not from base_color.w.
    //
    // The lit shader ends with float4(lit + emit, in.color.a), and in.color is
    // the mesh's vertex colour times the instance tint -- the material's
    // base_color contributes only rgb. Setting base_color.w and expecting glass
    // is the obvious mistake and it does not look like a transparency bug: the
    // panes come out fully opaque and, being smooth dielectrics facing away
    // from the sun, nearly black. Which reads as "the lighting is broken",
    // which is where the first hour went.
    //
    // Per INSTANCE rather than baked into the mesh, so one pane mesh serves all
    // three tints.
    //
    // Three tints, so overlapping panes are visibly accumulating and not just
    // getting darker. Two greens over each other look like one green.
    const eng::Vec4 tints[3] = {{0.55f, 0.80f, 0.95f, 0.34f},
                                {0.95f, 0.72f, 0.45f, 0.30f},
                                {0.62f, 0.95f, 0.70f, 0.28f}};
    eng::MaterialHandle glass[3];
    for (int i = 0; i < 3; ++i) {
        eng::MaterialDesc md;
        md.shading = eng::Shading::Lit;
        md.base_color = eng::Vec4{tints[i].x, tints[i].y, tints[i].z, 1.0f};
        md.roughness = 0.08f;
        md.transparent = true;
        glass[i] = r.CreateMaterial(md, error);
        if (!eng::Valid(glass[i])) return;
    }

    constexpr int kPanes = 7;
    constexpr float kW = 1.5f, kH = 2.6f;
    // A POST, not a slab. The first version made this 0.52 of the pane's width
    // against the pane's 0.5 -- a solid box very slightly LARGER than the glass
    // it was framing, sitting in front of it. The panes were transparent and
    // completely hidden, which looks exactly like transparency not working.
    const eng::MeshHandle frame_mesh = r.UploadMesh(
        eng::MakeBox(eng::Vec3{0.05f, kH * 0.55f, 0.05f}, eng::Vec4{1, 1, 1, 1}));
    const eng::MeshHandle pane_mesh = r.UploadMesh(
        eng::MakeBox(eng::Vec3{kW * 0.5f, kH * 0.5f, 0.02f}, eng::Vec4{1, 1, 1, 1}));
    for (int i = 0; i < kPanes; ++i) {
        const float t = float(i) / float(kPanes - 1) * 2.0f - 1.0f;
        // A fan, so that from most angles several panes overlap -- one pane
        // proves only that alpha works.
        const float angle = t * 1.15f;
        const eng::Vec3 at{kGallery.x * 0.0f + kGlassPavilion.x + std::sin(angle) * 4.2f,
                           ground_y + kH * 0.5f + 0.1f,
                           kGlassPavilion.z + std::cos(angle) * 4.2f};

        // One at each vertical edge, so the pane is held rather than covered.
        for (int side = -1; side <= 1; side += 2) {
            eng::Instance post;
            post.mesh = frame_mesh;
            post.material = frame_mat;
            const float c2 = std::cos(-angle), s2 = std::sin(-angle);
            const eng::Vec3 off{float(side) * kW * 0.5f, 0.0f, 0.0f};
            const eng::Vec3 post_at =
                at + eng::Vec3{off.x * c2 + off.z * s2, -0.05f, -off.x * s2 + off.z * c2};
            post.model = eng::Mat4::Translation(at + eng::Vec3{0, -0.05f, 0}) *
                         eng::Mat4::RotationY(-angle) *
                         eng::Mat4::Translation(off);
            AddSolid(scene, world, post,
                     eng::physics::Shape::MakeBox(eng::Vec3{0.05f, kH * 0.55f, 0.05f}),
                     post_at);
        }

        eng::Instance pane;
        pane.mesh = pane_mesh;
        pane.material = glass[std::size_t(i % 3)];
        pane.tint = eng::Vec4{1.0f, 1.0f, 1.0f, tints[std::size_t(i % 3)].w};
        pane.model = eng::Mat4::Translation(at) * eng::Mat4::RotationY(-angle);
        // Glass is SOLID. Being able to see through a thing is not the same as
        // being able to walk through it, and a transparent material is exactly
        // where that gets forgotten.
        AddSolid(scene, world, pane,
                 eng::physics::Shape::MakeBox(eng::Vec3{kW * 0.5f, kH * 0.5f, 0.06f}), at,
                 eng::Quat{0.0f, std::sin(-angle * 0.5f), 0.0f, std::cos(-angle * 0.5f)});
    }
}

// --- the fire pit -----------------------------------------------------------
//
// A ring of stones and a fire: the particle system, over terrain, with the
// scene's own depth so the sparks fade against the ground instead of cutting
// into it.
//
// SCORCH MARKS, and they are GROUND DECALS rather than projected ones.
//
// The projected kind writes into the G-buffer's albedo and cannot work on a
// forward path -- decals.h says so in its first paragraph, and this world is
// forward. What a forward path can do is build geometry that follows the
// receiving surface and draw it as an ordinary transparent object, which is
// what MakeGroundDecal is. It has to know what it lands on and be rebuilt if
// that changes; in exchange it works here, sorts like anything else, and is lit
// by the same sun as the ground under it.
inline void BuildFirePit(eng::Renderer& r, eng::Scene& scene,
                         eng::physics::World& world, eng::MeshHandle sphere,
                         eng::rhi::TextureId soot, float ground_y,
                         const std::function<float(float, float)>& height,
                         std::string& error) {
    eng::MaterialDesc stone_md;
    stone_md.shading = eng::Shading::Lit;
    stone_md.base_color = eng::Vec4{0.16f, 0.15f, 0.145f, 1.0f};
    stone_md.roughness = 0.85f;
    const eng::MaterialHandle stone_mat = r.CreateMaterial(stone_md, error);

    eng::MaterialDesc ember_md;
    ember_md.shading = eng::Shading::Lit;
    ember_md.base_color = eng::Vec4{0.25f, 0.10f, 0.05f, 1.0f};
    ember_md.roughness = 0.9f;
    ember_md.emissive = eng::Vec3{5.0f, 1.4f, 0.25f};
    const eng::MaterialHandle ember_mat = r.CreateMaterial(ember_md, error);
    if (!eng::Valid(stone_mat) || !eng::Valid(ember_mat)) return;

    // A ring of boulders, each a squashed sphere at its own angle -- uniform
    // scale only, so the variation has to come from the radius and the
    // placement rather than from flattening them.
    Rng rng(1301);
    constexpr int kStones = 11;
    for (int i = 0; i < kStones; ++i) {
        const float a = float(i) / float(kStones) * 6.2831853f + rng.Signed() * 0.12f;
        const float radius = 1.35f + rng.Signed() * 0.12f;
        const float size = 0.30f + rng.Unit() * 0.22f;
        eng::Instance in;
        in.mesh = sphere;
        in.material = stone_mat;
        in.model = eng::Mat4::Translation(
                       eng::Vec3{kFirePit.x + std::cos(a) * radius,
                                 ground_y + size * 0.35f,
                                 kFirePit.z + std::sin(a) * radius}) *
                   eng::Mat4::Scale(size * 2.0f);
        AddSolid(scene, world, in, eng::physics::Shape::MakeSphere(size),
                 eng::Vec3{kFirePit.x + std::cos(a) * radius, ground_y + size * 0.35f,
                           kFirePit.z + std::sin(a) * radius});
    }
    // Embers in the middle, so the fire has a source when the particles are off.
    for (int i = 0; i < 5; ++i) {
        const float a = rng.Unit() * 6.2831853f, d = rng.Unit() * 0.55f;
        eng::Instance in;
        in.mesh = sphere;
        in.material = ember_mat;
        in.model = eng::Mat4::Translation(
                       eng::Vec3{kFirePit.x + std::cos(a) * d, ground_y + 0.10f,
                                 kFirePit.z + std::sin(a) * d}) *
                   eng::Mat4::Scale(0.20f + rng.Unit() * 0.12f);
        // Embers are ash and charcoal in the fire's middle. Nothing to walk
        // into that the stone ring does not already stop.
        AddDecoration(scene, in);
    }

    // A light in the fire. Without one the pit glows and lights nothing, which
    // is the same failure the lanterns had -- and here it is worse, because a
    // fire that does not light the stones around it reads as a decal.
    // The burn, under the stones and spreading past them. Drawn before the
    // stones are added to the scene would put it in front of them in the
    // transparent sort; after is where it belongs.
    eng::MaterialDesc soot_md;
    soot_md.shading = eng::Shading::Lit;
    soot_md.base_color = eng::Vec4{1.0f, 1.0f, 1.0f, 1.0f};
    soot_md.roughness = 0.95f;
    soot_md.albedo = soot;
    soot_md.transparent = true;
    const eng::MaterialHandle soot_mat = r.CreateMaterial(soot_md, error);
    if (eng::Valid(soot_mat) && eng::rhi::Valid(soot)) {
        // Three overlapping marks at different sizes and angles rather than one
        // big one: a single circular scorch is the most obvious way to look
        // procedural, and three at odd rotations read as a fire that has been
        // lit more than once.
        const float kR[3] = {3.1f, 2.2f, 1.6f};
        const float kAng[3] = {0.0f, 1.9f, 4.1f};
        const float kOff[3][2] = {{0.0f, 0.0f}, {0.8f, -0.5f}, {-0.6f, 0.7f}};
        for (int i = 0; i < 3; ++i) {
            eng::GroundDecalDesc gd;
            gd.centre = eng::Vec3{kFirePit.x + kOff[i][0], 0.0f, kFirePit.z + kOff[i][1]};
            gd.radius = kR[i];
            gd.rotation = kAng[i];
            gd.lift = 0.04f;
            gd.segments = 22;
            gd.tint = eng::Vec4{1.0f, 1.0f, 1.0f, 1.0f};
            eng::Instance in;
            in.mesh = r.UploadMesh(eng::MakeGroundDecal(gd, height));
            in.material = soot_mat;
            // A mark painted on the ground has no substance.
            AddDecoration(scene, in);
        }
    }

    eng::Light fire;
    fire.type = eng::LightType::Point;
    fire.position = eng::Vec3{kFirePit.x, ground_y + 0.6f, kFirePit.z};
    fire.color = eng::Vec3{11.0f, 4.4f, 1.3f};
    fire.range = 8.0f;
    scene.lights.push_back(fire);
}

// --- the banner -------------------------------------------------------------
//
// A cloth on a pole, skinned to a row of joints and driven by a travelling
// wave. Generated rather than imported for the same reason the test's version
// is: every weight comes from a formula, so what it is supposed to look like is
// known rather than asserted.
struct Banner {
    eng::MeshHandle mesh;
    std::size_t instance = 0;
    int joint_offset = 0;
    static constexpr int kJoints = 9;
    static constexpr float kWidth = 3.2f;
    static constexpr float kHeight = 1.9f;
    static constexpr float kTop = 4.3f;
    eng::Vec3 root{0.0f, 0.0f, 0.0f};
};

inline Banner BuildBanner(eng::Renderer& r, eng::Scene& scene,
                          eng::physics::World& world, float ground_y,
                          std::string& error) {
    Banner b;
    b.root = eng::Vec3{kFlag.x, ground_y, kFlag.z};

    eng::MaterialDesc pole_md;
    pole_md.shading = eng::Shading::Lit;
    pole_md.base_color = eng::Vec4{0.30f, 0.28f, 0.26f, 1.0f};
    pole_md.roughness = 0.55f;
    const eng::MaterialHandle pole_mat = r.CreateMaterial(pole_md, error);
    eng::Instance pole;
    pole.mesh = r.UploadMesh(eng::MakeBox(eng::Vec3{0.07f, (b.kTop + 0.4f) * 0.5f, 0.07f},
                                          eng::Vec4{1, 1, 1, 1}));
    pole.material = pole_mat;
    const eng::Vec3 pole_at = b.root + eng::Vec3{0, (b.kTop + 0.4f) * 0.5f, 0};
    pole.model = eng::Mat4::Translation(pole_at);
    AddSolid(scene, world, pole,
             eng::physics::Shape::MakeBox(eng::Vec3{0.07f, (b.kTop + 0.4f) * 0.5f, 0.07f}),
             pole_at);

    // The cloth: a grid, finer along the wave than across it, because that is
    // the axis it bends on and a coarse grid there shows the wave as facets.
    constexpr int kCols = 40, kRows = 12;
    eng::Mesh cloth;
    std::vector<eng::anim::SkinVertex> skin;
    for (int row = 0; row <= kRows; ++row)
        for (int col = 0; col <= kCols; ++col) {
            const float u = float(col) / float(kCols), v = float(row) / float(kRows);
            VertexIn vtx{};
            // Built in the POLE's space with x running out along the cloth, so
            // the joints below are a straight row and the palette is readable.
            vtx.position = eng::Vec4{u * Banner::kWidth,
                                     b.kTop - v * Banner::kHeight, 0.0f, 0.0f};
            vtx.normal = eng::Vec4{0.0f, 0.0f, 1.0f, 0.0f};
            vtx.color = eng::Vec4{0.72f, 0.20f, 0.16f, 1.0f};
            vtx.uv = eng::Vec4{u, v, 0.0f, 0.0f};
            cloth.vertices.push_back(vtx);

            // TWO INFLUENCES, linearly blended between the joints either side.
            // One influence per vertex makes the cloth a row of rigid strips
            // that scissor at every joint, which is the classic look of a mesh
            // that was skinned by rounding.
            const float j = u * float(Banner::kJoints - 1);
            const int j0 = std::min(int(j), Banner::kJoints - 2);
            const float t = j - float(j0);
            eng::anim::SkinVertex sv{};
            sv.joints[0] = std::uint16_t(j0);
            sv.joints[1] = std::uint16_t(j0 + 1);
            sv.weights[0] = 1.0f - t;
            sv.weights[1] = t;
            skin.push_back(sv);
        }
    for (int row = 0; row < kRows; ++row)
        for (int col = 0; col < kCols; ++col) {
            const std::uint32_t a = std::uint32_t(row * (kCols + 1) + col);
            const std::uint32_t bq = a + 1, c = a + std::uint32_t(kCols + 1), d = c + 1;
            cloth.indices.insert(cloth.indices.end(), {a, c, bq, bq, c, d});
        }
    // A cloth is seen from both sides, and back-face culling is on for
    // everything -- so it is two-sided by having two sets of triangles rather
    // than by a material flag the renderer does not have.
    const std::size_t half = cloth.indices.size();
    for (std::size_t i = 0; i < half; i += 3)
        cloth.indices.insert(cloth.indices.end(),
                             {cloth.indices[i], cloth.indices[i + 2], cloth.indices[i + 1]});
    eng::GenerateTangents(cloth);

    eng::MaterialDesc cloth_md;
    cloth_md.shading = eng::Shading::Lit;
    cloth_md.base_color = eng::Vec4{1.0f, 1.0f, 1.0f, 1.0f};
    cloth_md.roughness = 0.88f;
    const eng::MaterialHandle cloth_mat = r.CreateMaterial(cloth_md, error);
    if (!eng::Valid(pole_mat) || !eng::Valid(cloth_mat)) return b;

    b.mesh = r.UploadSkinnedMesh(cloth, skin, Banner::kJoints);
    eng::Instance in;
    in.mesh = b.mesh;
    in.material = cloth_mat;
    in.model = eng::Mat4::Translation(b.root);
    b.joint_offset = int(scene.joint_matrices.size());
    in.palette = b.joint_offset;
    b.instance = scene.instances.size();
    // Cloth. It moves every frame and nothing should stop on it.
    AddDecoration(scene, in);
    scene.joint_matrices.resize(scene.joint_matrices.size() + Banner::kJoints,
                                eng::Mat4::Identity());
    return b;
}

// The pose at `time`: a wave running out from the pole, growing as it goes.
//
// The palette is joint-world times inverse-bind. The bind pose here is the
// identity translation of each joint along x, so the inverse bind is a
// translation back -- written out rather than stored, because for a straight
// row of joints it is one subtraction and a stored array would be a second
// thing to keep in step with the first.
inline void PoseBanner(const Banner& b, eng::Scene& scene, float time) {
    if (!eng::Valid(b.mesh)) return;
    const float spacing = Banner::kWidth / float(Banner::kJoints - 1);
    for (int j = 0; j < Banner::kJoints; ++j) {
        const float along = float(j) * spacing;
        // Amplitude grows with distance from the pole: the fixed end cannot
        // move and the free end moves most, which is most of what makes cloth
        // read as attached to something.
        const float grip = float(j) / float(Banner::kJoints - 1);
        const float phase = time * 3.1f - along * 1.9f;
        const eng::Vec3 offset{0.0f,
                               std::sin(phase * 0.7f) * 0.30f * grip * grip,
                               std::sin(phase) * 0.85f * grip * grip};
        scene.joint_matrices[std::size_t(b.joint_offset + j)] =
            eng::Mat4::Translation(offset);
    }
}

// --- undergrowth ------------------------------------------------------------
//
// The nought-to-two-metre band was empty. That is the distance the eye looks
// hardest at, and a landscape with detailed ground texture, trees at head
// height and NOTHING in between reads as a diorama -- the texture says the
// ground is grass and the silhouette says it is a painted floor.
//
// SOLID GEOMETRY, not alpha cards. A card needs an alpha cutout, a cutout needs
// the same test in the shadow pass or every tuft casts a rectangle, and edge-on
// a card disappears. Actual little blades cost more triangles and none of that:
// they are opaque, they shadow correctly, and they look the same from every
// direction because they are the same from every direction.

// One blade: a tapered strip that bends over as it rises.
inline void AppendBlade(eng::Mesh& m, eng::Vec3 base, float angle, float height,
                        float width, float bend, eng::Vec4 colour) {
    const eng::Vec3 out{std::cos(angle), 0.0f, std::sin(angle)};
    // Perpendicular to the blade's lean, so the strip faces outward rather than
    // edge-on to whoever is looking along it.
    const eng::Vec3 side{-out.z * width, 0.0f, out.x * width};

    // Three levels: full width at the base, half at the middle, a point at the
    // tip. A single triangle from base to tip is cheaper and reads as a spike.
    const eng::Vec3 mid = base + eng::Vec3{out.x * bend * 0.35f, height * 0.55f,
                                           out.z * bend * 0.35f};
    const eng::Vec3 tip = base + eng::Vec3{out.x * bend, height, out.z * bend};
    const eng::Vec3 nrm = Normalize(Cross(eng::Vec3{0.0f, height, 0.0f}, side));

    const auto push = [&](eng::Vec3 p, float u, float v) {
        VertexIn out_v{};
        out_v.position = eng::Vec4{p.x, p.y, p.z, 0.0f};
        out_v.normal = eng::Vec4{nrm.x, nrm.y, nrm.z, 0.0f};
        out_v.color = colour;
        out_v.uv = eng::Vec4{u, v, 0.0f, 0.0f};
        m.vertices.push_back(out_v);
    };
    const auto base_i = std::uint32_t(m.vertices.size());
    push(base - side, 0.0f, 0.0f);
    push(base + side, 1.0f, 0.0f);
    push(mid - side * 0.55f, 0.15f, 0.55f);
    push(mid + side * 0.55f, 0.85f, 0.55f);
    push(tip, 0.5f, 1.0f);
    const std::uint32_t f[9] = {0, 2, 1, 1, 2, 3, 2, 4, 3};
    for (int i = 0; i < 9; i += 3) {
        // Both windings. A blade is a surface with no inside, and back-face
        // culling would make half of every tuft vanish depending on where you
        // stood.
        m.indices.insert(m.indices.end(),
                         {base_i + f[i], base_i + f[i + 1], base_i + f[i + 2]});
        m.indices.insert(m.indices.end(),
                         {base_i + f[i], base_i + f[i + 2], base_i + f[i + 1]});
    }
}

inline void AppendTuft(eng::Mesh& m, Rng& rng, eng::Vec3 base, float scale,
                       eng::Vec4 colour) {
    const int blades = 4 + int(rng.Unit() * 3.0f);
    const float roll = rng.Unit() * 6.2831853f;
    for (int i = 0; i < blades; ++i) {
        const float a = roll + float(i) / float(blades) * 6.2831853f +
                        rng.Signed() * 0.5f;
        // Darker toward the base, which is what a clump of anything looks like
        // and what stops a field of tufts reading as one flat colour.
        const float shade = 0.72f + rng.Unit() * 0.5f;
        AppendBlade(m, base + eng::Vec3{rng.Signed() * 0.04f, 0.0f, rng.Signed() * 0.04f},
                    a, scale * (0.7f + rng.Unit() * 0.6f), scale * 0.055f,
                    scale * (0.25f + rng.Unit() * 0.4f),
                    eng::Vec4{colour.x * shade, colour.y * shade, colour.z * shade, 1.0f});
    }
}

// A boulder: a low sphere pushed about, so no two are the same and none of them
// is a ball.
inline void AppendRock(eng::Mesh& m, Rng& rng, eng::Vec3 centre, float size,
                       eng::Vec4 colour) {
    // 5 by 8, not 4 by 6. Four stacks is a 45 degree step, and with the poles
    // as fans that makes the top a six-sided cone -- the rocks came out looking
    // like little pyramids rather than boulders.
    constexpr int kStacks = 5, kSlices = 8;
    const auto base = std::uint32_t(m.vertices.size());
    const float wob_a = rng.Unit() * 6.2831853f, wob_b = rng.Unit() * 6.2831853f;
    const float squash = 0.55f + rng.Unit() * 0.3f;
    const auto point = [&](float phi, float theta) {
        const float sp = std::sin(phi);
        const eng::Vec3 u{sp * std::cos(theta), std::cos(phi), sp * std::sin(theta)};
        const float lump = 1.0f + 0.15f * std::sin(theta * 2.0f + wob_a) +
                           0.11f * std::sin(phi * 3.0f + wob_b);
        return eng::Vec3{u.x * size * lump, u.y * size * lump * squash,
                         u.z * size * lump};
    };
    for (int i = 0; i <= kStacks; ++i) {
        const float phi = float(i) / float(kStacks) * kPi;
        for (int j = 0; j <= kSlices; ++j) {
            const float theta = float(j) / float(kSlices) * 2.0f * kPi;
            const eng::Vec3 p = point(phi, theta);
            constexpr float h = 1e-3f;
            eng::Vec3 n = Cross(point(phi, theta + h) - point(phi, theta - h),
                                point(phi + h, theta) - point(phi - h, theta));
            if (Length(n) < 1e-9f) n = p;
            n = Normalize(n);
            VertexIn v{};
            v.position = eng::Vec4{centre.x + p.x, centre.y + p.y, centre.z + p.z, 0.0f};
            v.normal = eng::Vec4{n.x, n.y, n.z, 0.0f};
            v.color = colour;
            v.uv = eng::Vec4{float(j) / float(kSlices), float(i) / float(kStacks), 0, 0};
            m.vertices.push_back(v);
        }
    }
    for (int i = 0; i < kStacks; ++i)
        for (int j = 0; j < kSlices; ++j) {
            const auto a = std::uint32_t(base + i * (kSlices + 1) + j);
            const auto b = std::uint32_t(a + 1);
            const auto c = std::uint32_t(a + kSlices + 1);
            const auto dd = std::uint32_t(c + 1);
            // Pole rows as fans, for the same reason the leaf blobs are: half of
            // each quad there has zero area and a face normal made of rounding
            // noise.
            if (i > 0) m.indices.insert(m.indices.end(), {a, b, c});
            if (i < kStacks - 1) m.indices.insert(m.indices.end(), {b, dd, c});
        }
}

}  // namespace world
