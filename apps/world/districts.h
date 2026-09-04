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
// EACH DISTRICT APPENDS to a scene it does not own. They return the handles the
// frame loop needs to animate them and nothing else; placement, materials and
// geometry are decided here so the loop stays a loop.
#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "engine/geometry/mesh.h"
#include "engine/render/renderer.h"
#include "engine/scene/scene.h"

namespace world {

// Where each district sits, in metres from the middle of the valley.
//
// All of them are outside the 26 m spawn clearing and inside the 46 m rim where
// the bowl starts climbing. They are far enough apart that no two are ever the
// only thing in frame, which is the arrangement that makes them useful.
inline constexpr eng::Vec3 kGallery{34.0f, 0.0f, -26.0f};
inline constexpr eng::Vec3 kLanternHall{-36.0f, 0.0f, -28.0f};
inline constexpr eng::Vec3 kGlassPavilion{30.0f, 0.0f, 30.0f};
inline constexpr eng::Vec3 kFirePit{-26.0f, 0.0f, 28.0f};
inline constexpr eng::Vec3 kFlag{14.0f, 0.0f, 6.0f};

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

inline void BuildGallery(eng::Renderer& r, eng::Scene& scene, eng::MeshHandle sphere,
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
    plinth.model = eng::Mat4::Translation(
        eng::Vec3{kGallery.x, ground_y + kTop * 0.5f, kGallery.z});
    scene.instances.push_back(plinth);

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

            eng::Instance in;
            in.mesh = sphere;
            in.material = m;
            in.model = eng::Mat4::Translation(eng::Vec3{
                kGallery.x + (float(col) - float(Gallery::kCols - 1) * 0.5f) * kStep,
                ground_y + kTop + 0.6f,
                kGallery.z + (float(row) - float(Gallery::kRows - 1) * 0.5f) * kStep});
            scene.instances.push_back(in);
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
            p.model = eng::Mat4::Translation(base + eng::Vec3{0, kHeight * 0.5f, 0});
            scene.instances.push_back(p);

            // Two lanterns per pillar at different heights, which is what takes
            // this past a hundred lights and into the range where clustering is
            // the difference between working and not.
            for (int k = 0; k < 2; ++k) {
                const eng::Vec3 at = base + eng::Vec3{0.0f, kHeight - 0.5f - float(k) * 1.6f, 0.0f};
                eng::Instance bulb;
                bulb.mesh = sphere;
                bulb.material = glass_mat;
                bulb.model = eng::Mat4::Translation(at) * eng::Mat4::Scale(0.34f);
                scene.instances.push_back(bulb);

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
inline void BuildGlassPavilion(eng::Renderer& r, eng::Scene& scene, float ground_y,
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
            post.model = eng::Mat4::Translation(at + eng::Vec3{0, -0.05f, 0}) *
                         eng::Mat4::RotationY(-angle) *
                         eng::Mat4::Translation(eng::Vec3{float(side) * kW * 0.5f, 0, 0});
            scene.instances.push_back(post);
        }

        eng::Instance pane;
        pane.mesh = pane_mesh;
        pane.material = glass[std::size_t(i % 3)];
        pane.tint = eng::Vec4{1.0f, 1.0f, 1.0f, tints[std::size_t(i % 3)].w};
        pane.model = eng::Mat4::Translation(at) * eng::Mat4::RotationY(-angle);
        scene.instances.push_back(pane);
    }
}

// --- the fire pit -----------------------------------------------------------
//
// A ring of stones and a fire: the particle system, over terrain, with the
// scene's own depth so the sparks fade against the ground instead of cutting
// into it.
//
// NO DECALS HERE, and that is not an omission. decals.h says why in its first
// paragraph: a projected decal writes into the G-buffer's albedo, and a
// forward renderer shades each surface once as it draws it, so there is no
// moment between "the surface exists" and "the surface is lit" at which to
// intervene. This world is forward. The decal system works and has a test; it
// belongs to the other path, and pretending otherwise by painting a dark quad
// on the ground would be a picture of a decal rather than a decal.
inline void BuildFirePit(eng::Renderer& r, eng::Scene& scene, eng::MeshHandle sphere,
                         float ground_y, std::string& error) {
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
        scene.instances.push_back(in);
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
        scene.instances.push_back(in);
    }

    // A light in the fire. Without one the pit glows and lights nothing, which
    // is the same failure the lanterns had -- and here it is worse, because a
    // fire that does not light the stones around it reads as a decal.
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

inline Banner BuildBanner(eng::Renderer& r, eng::Scene& scene, float ground_y,
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
    pole.model = eng::Mat4::Translation(b.root + eng::Vec3{0, (b.kTop + 0.4f) * 0.5f, 0});
    scene.instances.push_back(pole);

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
    scene.instances.push_back(in);
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

}  // namespace world
