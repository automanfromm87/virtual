// No test framework — from scratch means from scratch.
//
// The fixture is a real glTF 2.0 document with a base64-embedded buffer, so the
// test exercises the whole path — JSON, base64, bufferViews, accessors, node
// hierarchy — without needing a file on disk or a copy of Blender.
#include "engine/asset/gltf.h"

#include <cmath>
#include <cstdio>
#include <span>

namespace {

int g_failures = 0;

void Fail(const char* what, int line) {
    std::fprintf(stderr, "gltf_test.cc:%d  %s\n", line, what);
    ++g_failures;
}

#define CHECK(cond) \
    do { if (!(cond)) Fail(#cond, __LINE__); } while (0)

constexpr const char* kQuad =
#include "engine/asset/testdata_quad_gltf.inc"
    ;

// A quad with a real PNG baked into the document, so this exercises the whole
// chain: base64 -> zlib -> PNG -> RGBA, then glTF's material -> texture ->
// image indirection on top of it.
const char* const kSkinnedGltf =
#include "engine/asset/testdata_skinned_gltf.inc"
    ;

const char* const kMorphGltf =
#include "engine/asset/testdata_morph_gltf.inc"
    ;

const char* const kTexturedGltf =
#include "engine/asset/testdata_textured_gltf.inc"
    ;

#include "engine/asset/testdata_glb.inc"
#include "engine/asset/testdata_fox.inc"

bool Near(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }

}  // namespace

int main() {
    using namespace eng;
    std::string err;

    const gltf::Document doc = gltf::ParseGltf(kQuad, {}, err);
    if (!err.empty()) {
        std::fprintf(stderr, "parse failed: %s\n", err.c_str());
        return 1;
    }

    // --- geometry ------------------------------------------------------------
    CHECK(doc.primitives.size() == 1);
    const Mesh& m = doc.primitives[0].mesh;
    CHECK(m.vertices.size() == 4);
    CHECK(m.indices.size() == 6);

    // Positions came through in order and un-transposed.
    CHECK(Near(m.vertices[0].position.x, 0) && Near(m.vertices[0].position.y, 0));
    CHECK(Near(m.vertices[2].position.x, 1) && Near(m.vertices[2].position.y, 1));
    // Normals, which live in a SEPARATE bufferView — if the accessor offsets
    // were wrong these would come back as positions.
    for (const VertexIn& v : m.vertices) {
        CHECK(Near(v.normal.x, 0) && Near(v.normal.y, 0) && Near(v.normal.z, 1));
    }
    // uv is a VEC2 accessor, so .zw must be left alone rather than filled with
    // whatever follows in the buffer.
    CHECK(Near(m.vertices[1].uv.x, 1) && Near(m.vertices[1].uv.y, 0));
    CHECK(Near(m.vertices[3].uv.x, 0) && Near(m.vertices[3].uv.y, 1));
    // Bounds were fitted on import.
    CHECK(m.bounds.radius > 0.5f && m.bounds.radius < 1.0f);

    // --- material ------------------------------------------------------------
    CHECK(doc.materials.size() == 1);
    CHECK(doc.materials[0].name == "brass");
    CHECK(Near(doc.materials[0].base_color.x, 0.8f));
    CHECK(Near(doc.materials[0].metallic, 1.0f));
    CHECK(Near(doc.materials[0].roughness, 0.3f));
    CHECK(doc.primitives[0].material == 0);

    // --- hierarchy -----------------------------------------------------------
    CHECK(doc.nodes.size() == 2);
    CHECK(doc.roots.size() == 1 && doc.roots[0] == 0);
    CHECK(doc.nodes[0].name == "parent");
    CHECK(doc.nodes[1].parent == 0);   // derived; glTF only stores children
    CHECK(doc.nodes[1].primitives.size() == 1);

    // THE hierarchy check: the child carries a 90-degree turn about +Y and the
    // parent a translation, so the child's WORLD transform has to be both,
    // composed in that order. A reader that ignored parents would put the quad
    // at the origin, unrotated, and every other assertion here would still pass.
    const std::vector<Mat4> world = doc.WorldTransforms();
    CHECK(world.size() == 2);
    const Vec4 origin = world[1] * Vec4{0, 0, 0, 1};
    CHECK(Near(origin.x, 10.0f) && Near(origin.y, 0) && Near(origin.z, 0));
    // +X on the child points down -Z in the world after a +90 turn about +Y.
    const Vec4 dir = world[1] * Vec4{1, 0, 0, 0};
    CHECK(Near(dir.x, 0) && Near(dir.z, -1.0f));

    // --- failure paths -------------------------------------------------------
    // Version gating: a glTF 1.0 file has an incompatible buffer layout and
    // must be rejected rather than half-read.
    (void)gltf::ParseGltf(R"({"asset":{"version":"1.0"}})", {}, err);
    CHECK(!err.empty());
    (void)gltf::ParseGltf("not json at all", {}, err);
    CHECK(!err.empty());
    (void)gltf::ParseGltf(R"({"asset":{"version":"2.0"},"meshes":[{"primitives":[{}]}]})",
                          {}, err);
    CHECK(!err.empty());  // a primitive with no POSITION

    // --- an embedded PNG comes back as pixels ---------------------------------
    {
        std::string error;
        const gltf::Document doc = gltf::ParseGltf(kTexturedGltf, {}, error);
        CHECK(error.empty());
        CHECK(doc.images.size() == 1);
        CHECK(doc.images[0].width == 4 && doc.images[0].height == 4);
        CHECK(doc.materials.size() == 1);

        // The material points at the IMAGE, with glTF's texture/sampler
        // indirection already collapsed.
        CHECK(doc.materials[0].base_color_image == 0);
        CHECK(doc.materials[0].metallic_roughness_image == -1);

        // Specific texels, not just "not empty". The fixture is a 2x2 checker
        // of warm and cool quadrants; a decoder that transposed rows or swapped
        // channels would still produce a 4x4 image full of plausible colour.
        const std::vector<std::uint8_t>& p = doc.images[0].rgba;
        auto at = [&](int x, int y, int c) { return int(p[(std::size_t(y) * 4 + x) * 4 + c]); };
        CHECK(at(0, 0, 0) == 255 && at(0, 0, 2) == 32);   // top-left warm
        CHECK(at(3, 0, 0) == 32 && at(3, 0, 2) == 255);   // top-right cool
        CHECK(at(0, 3, 0) == 32 && at(0, 3, 2) == 255);   // bottom-left cool
        CHECK(at(3, 3, 0) == 255 && at(3, 3, 2) == 32);   // bottom-right warm
        CHECK(at(0, 0, 3) == 255);                        // opaque

        // The geometry still loaded alongside it, with uvs.
        CHECK(doc.primitives.size() == 1);
        CHECK(doc.primitives[0].mesh.vertices.size() == 4);
        CHECK(doc.primitives[0].mesh.indices.size() == 6);
        bool uv_varies = false;
        for (const auto& v : doc.primitives[0].mesh.vertices)
            if (v.uv.x > 0.5f) uv_varies = true;
        CHECK(uv_varies);
    }

    // --- a file with no images still loads ------------------------------------
    {
        std::string error;
        const gltf::Document plain = gltf::ParseGltf(kQuad, {}, error);
        CHECK(error.empty());
        CHECK(plain.images.empty());
        CHECK(!plain.materials.empty());
        CHECK(plain.materials[0].base_color_image == -1);
    }

    // --- skins and animations -------------------------------------------------
    {
        std::string error;
        const gltf::Document doc = gltf::ParseGltf(kSkinnedGltf, {}, error);
        CHECK(error.empty());
        if (!error.empty()) std::fprintf(stderr, "  skinned: %s\n", error.c_str());

        // The skin became a posable skeleton, with parents resolved from the
        // node hierarchy rather than declared.
        CHECK(doc.skins.size() == 1);
        const gltf::SkinDef& sk = doc.skins[0];
        CHECK(sk.skeleton.joints.size() == 2);
        CHECK(sk.skeleton.joints[0].name == "joint_root");
        CHECK(sk.skeleton.joints[1].name == "joint_elbow");
        CHECK(sk.skeleton.joints[0].parent == -1);
        CHECK(sk.skeleton.joints[1].parent == 0);
        CHECK(Near(sk.skeleton.joints[1].rest.translation.x, 1.0f));

        // inverseBindMatrices came across without a transpose. glTF and this
        // engine are both column-major, so the elbow's is a -1 on x; a
        // transposed read would put it in the bottom row and the mesh would
        // shear instead of translate.
        const Vec4 ib = sk.skeleton.joints[1].inverse_bind * Vec4{0, 0, 0, 1};
        CHECK(Near(ib.x, -1.0f) && Near(ib.y, 0.0f));

        // JOINTS_0 and WEIGHTS_0 came in, and the middle pair is the 50/50
        // split that a wrong palette order would get wrong.
        CHECK(doc.primitives.size() == 1);
        const gltf::Primitive& prim = doc.primitives[0];
        CHECK(prim.Skinned());
        CHECK(prim.skin.size() == prim.mesh.vertices.size());
        CHECK(prim.skin[0].joints[0] == 0 && Near(prim.skin[0].weights[0], 1.0f));
        CHECK(Near(prim.skin[2].weights[0], 0.5f));
        CHECK(Near(prim.skin[2].weights[1], 0.5f));
        CHECK(prim.skin[4].joints[1] == 1 && Near(prim.skin[4].weights[1], 1.0f));

        // The node knows it is skinned.
        CHECK(doc.nodes.size() == 3);
        CHECK(doc.nodes[0].skin == 0);
        CHECK(doc.nodes[1].skin == -1);

        // The animation parsed, and retargets onto the skin.
        CHECK(doc.animations.size() == 1);
        CHECK(doc.animations[0].name == "bend");
        CHECK(Near(doc.animations[0].duration, 1.0f));
        const anim::Clip clip = doc.MakeClip(0, 0);
        CHECK(clip.channels.size() == 1);
        CHECK(clip.channels[0].joint == 1);  // node 2 -> joint 1
        CHECK(clip.channels[0].path == anim::Path::Rotation);
        CHECK(clip.channels[0].Valid(sk.skeleton.joints.size()));

        // AND IT ACTUALLY POSES. At t=0 the mesh is at rest; at t=1 the elbow
        // has turned a quarter turn, so the tip swings from x=2 to (1,1).
        anim::Pose pose;
        std::vector<Mat4> palette;

        clip.Sample(0.0f, sk.skeleton, &pose);
        anim::ComputeJointMatrices(sk.skeleton, pose, &palette);
        Vec3 tip = anim::SkinPosition(Vec3{2, 0, 0}, prim.skin[4], palette);
        CHECK(Near(tip.x, 2.0f) && Near(tip.y, 0.0f));

        clip.Sample(1.0f, sk.skeleton, &pose, /*loop=*/false);
        anim::ComputeJointMatrices(sk.skeleton, pose, &palette);
        tip = anim::SkinPosition(Vec3{2, 0, 0}, prim.skin[4], palette);
        CHECK(Near(tip.x, 1.0f, 1e-3f) && Near(tip.y, 1.0f, 1e-3f));

        // Halfway is 45 degrees, so the tip is on the unit circle about the
        // elbow. A step or a hold would leave it at one of the ends.
        clip.Sample(0.5f, sk.skeleton, &pose);
        anim::ComputeJointMatrices(sk.skeleton, pose, &palette);
        tip = anim::SkinPosition(Vec3{2, 0, 0}, prim.skin[4], palette);
        CHECK(Near(tip.x, 1.0f + 0.70710678f, 1e-3f));
        CHECK(Near(tip.y, 0.70710678f, 1e-3f));

        // The root-weighted vertex never moves, whatever the elbow does.
        const Vec3 base = anim::SkinPosition(Vec3{0, 0, 0}, prim.skin[0], palette);
        CHECK(Near(base.x, 0.0f) && Near(base.y, 0.0f));

        // Retargeting onto a skin the channel does not belong to yields
        // nothing rather than posing the wrong character.
        CHECK(doc.MakeClip(0, 5).channels.empty());
        CHECK(doc.MakeClip(9, 0).channels.empty());
    }

    // --- an unskinned file has no skin data ------------------------------------
    {
        std::string error;
        const gltf::Document plain = gltf::ParseGltf(kQuad, {}, error);
        CHECK(error.empty());
        CHECK(plain.skins.empty());
        CHECK(plain.animations.empty());
        CHECK(!plain.primitives[0].Skinned());
    }

    // --- .glb container, with a SPARSE accessor inside it ----------------------
    {
        const std::span<const std::uint8_t> bytes(
            reinterpret_cast<const std::uint8_t*>(kSparseGlb), sizeof(kSparseGlb));
        CHECK(gltf::IsGlb(bytes));
        CHECK(!gltf::IsGlb(std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t*>(kQuad), 8)));

        std::string error;
        const gltf::Document doc = gltf::ParseGlb(bytes, error);
        CHECK(error.empty());
        if (!error.empty()) std::fprintf(stderr, "  glb: %s\n", error.c_str());
        CHECK(doc.primitives.size() == 1);
        if (doc.primitives.empty()) return 1;

        const eng::Mesh& m = doc.primitives[0].mesh;
        CHECK(m.vertices.size() == 4);
        CHECK(m.indices.size() == 6);

        // The buffer had NO uri: its bytes are the glb's BIN chunk. If that
        // were not wired up the positions would all be zero.
        CHECK(Near(m.vertices[0].position.x, 0.0f));
        CHECK(Near(m.vertices[2].position.x, 2.0f));

        // SPARSE: the base is a flat line at y = 0, and the override list
        // lifts vertices 1 and 3 to y = 5. Both halves have to work — reading
        // only the base gives a flat line, reading only the overrides gives
        // two vertices and two holes.
        CHECK(Near(m.vertices[0].position.y, 0.0f));
        CHECK(Near(m.vertices[1].position.y, 5.0f));
        CHECK(Near(m.vertices[2].position.y, 0.0f));
        CHECK(Near(m.vertices[3].position.y, 5.0f));
        // ...and the overridden elements kept their x, so the whole vec3 was
        // replaced rather than just the component that changed.
        CHECK(Near(m.vertices[1].position.x, 1.0f));
        CHECK(Near(m.vertices[3].position.x, 3.0f));
    }

    // --- a real exporter asset: the Fox sample ---------------------------------
    // None of the fixtures above combines node transforms, an embedded PNG,
    // a skin and multi-channel clips; third-party exporters do all four at
    // once, and the demo loads these exact bytes behind --fox.
    {
        const std::span<const std::uint8_t> bytes(
            reinterpret_cast<const std::uint8_t*>(kFoxGlb), sizeof(kFoxGlb));
        CHECK(gltf::IsGlb(bytes));
        std::string error;
        const gltf::Document fox = gltf::ParseGlb(bytes, error);
        CHECK(error.empty());
        if (!error.empty()) std::fprintf(stderr, "  fox: %s\n", error.c_str());
        CHECK(fox.primitives.size() == 1);
        if (fox.primitives.empty()) return 1;
        CHECK(fox.primitives[0].Skinned());
        CHECK(fox.images.size() == 1);
        CHECK(!fox.images[0].Empty());
        CHECK(fox.images[0].width == 1024 && fox.images[0].height == 1024);
        CHECK(fox.skins.size() == 1);
        CHECK(fox.animations.size() == 3);
        // Walk is animation 1; retargeted onto the skin it must keep
        // channels, or the demo plays a fox that never moves.
        CHECK(!fox.MakeClip(1, 0).channels.empty());
    }

    // --- a corrupt container is refused ----------------------------------------
    {
        std::string error;
        std::vector<std::uint8_t> bad(kSparseGlb, kSparseGlb + sizeof(kSparseGlb));
        bad[4] = 3;  // version 3
        CHECK(gltf::ParseGlb(bad, error).primitives.empty());
        CHECK(error.find("version") != std::string::npos);

        error.clear();
        std::vector<std::uint8_t> truncated(kSparseGlb, kSparseGlb + 40);
        CHECK(gltf::ParseGlb(truncated, error).primitives.empty());
        CHECK(!error.empty());

        error.clear();
        const std::uint8_t junk[12] = {'x', 'x', 'x', 'x'};
        CHECK(gltf::ParseGlb(junk, error).primitives.empty());
        CHECK(error.find("magic") != std::string::npos);
    }

    // --- a sparse accessor cannot read past its own buffer ---------------------
    {
        std::printf("  sparse bounds\n");
        // A HAND-WRITTEN MALICIOUS DOCUMENT, because the point is a count that
        // does not match the bytes behind it and that is easier to state than
        // to patch into a binary fixture.
        //
        // Four positions, and a sparse override list that CLAIMS nine entries
        // over bufferViews holding two. Everything else is well formed. The
        // reader used to check only that the override list STARTED inside the
        // buffer and then read nine elements from it -- seven of them past the
        // end, at offsets taken straight out of the file.
        const char* kJson = R"({
          "asset": {"version": "2.0"},
          "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
          "accessors": [{
            "bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
            "sparse": {
              "count": COUNT,
              "indices": {"bufferView": 1, "componentType": 5125},
              "values": {"bufferView": 2}
            }
          }],
          "bufferViews": [
            {"buffer": 0, "byteOffset": 0,  "byteLength": 48},
            {"buffer": 0, "byteOffset": 48, "byteLength": 8},
            {"buffer": 0, "byteOffset": 56, "byteLength": 24}
          ],
          "buffers": [{"byteLength": 80}]
        })";
        // 4 vec3 positions, 2 uint32 indices, 2 vec3 values = 80 bytes exactly.
        std::vector<std::uint8_t> bin(80, 0);

        const auto with_count = [&](const char* n) {
            std::string j(kJson);
            j.replace(j.find("COUNT"), 5, n);
            return j;
        };

        // The honest version loads.
        std::string error;
        const gltf::Document ok = gltf::ParseGltf(with_count("2"), bin, error);
        std::printf("    count 2 (the truth): %zu primitives, error \"%s\"\n",
                    ok.primitives.size(), error.c_str());
        CHECK(error.empty());
        CHECK(ok.primitives.size() == 1);

        // The lie is REFUSED rather than read.
        error.clear();
        const gltf::Document bad = gltf::ParseGltf(with_count("9"), bin, error);
        std::printf("    count 9 over a 2-element list: error \"%s\"\n", error.c_str());
        CHECK(bad.primitives.empty());
        // THE SPECIFIC ERROR, not merely some error. With the old
        // start-offset-only check this still failed -- but on the NEXT check
        // along, "overrides an element that is not there", because the value it
        // read from past the end of the buffer happened to be a large index.
        // That is the out-of-bounds read succeeding and then being caught by
        // luck downstream, which is exactly what this is supposed to prevent,
        // and a test that accepts any error cannot tell the two apart.
        CHECK(error.find("runs past its buffer") != std::string::npos);

        // AND SO IS A NEGATIVE ONE. A signed count that goes negative turns
        // into an enormous size_t the moment it is used as a length, which is
        // the same bug wearing a different sign.
        error.clear();
        const gltf::Document neg = gltf::ParseGltf(with_count("-1"), bin, error);
        std::printf("    count -1: error \"%s\"\n", error.c_str());
        CHECK(neg.primitives.empty());
        CHECK(!error.empty());
    }

    // --- morph targets --------------------------------------------------
    {
        std::printf("morph targets\n");
        std::string error;
        const gltf::Document doc = gltf::ParseGltf(kMorphGltf, {}, error);
        CHECK(error.empty());
        CHECK(doc.primitives.size() == 1);
        const gltf::Primitive& prim = doc.primitives[0];
        CHECK(prim.Morphed());
        CHECK(prim.morph_targets.size() == 2);
        CHECK(prim.morph_targets[0].name == "rise");
        CHECK(prim.morph_targets[1].name == "lean");

        // A target that touches only positions keeps an EMPTY normal array
        // rather than a zeroed one -- a face rig has dozens of targets and the
        // difference is 12 bytes a vertex each.
        CHECK(prim.morph_targets[0].normals.size() == 4);
        CHECK(prim.morph_targets[1].normals.empty());

        // Deltas, not absolute positions. The base quad's top-right vertex is
        // at (1,1,0); the target stores the (0,0.5,0) it MOVES BY.
        CHECK(Near(prim.morph_targets[0].positions[3].y, 0.5f));
        CHECK(Near(prim.morph_targets[0].positions[0].y, 0.0f));
        CHECK(Near(prim.morph_targets[1].positions[3].x, 0.75f));

        // The mesh's defaults, and a node that overrides them.
        CHECK(prim.morph_weights.size() == 2);
        CHECK(Near(prim.morph_weights[0], 0.25f));
        CHECK(Near(prim.morph_weights[1], 0.0f));
        CHECK(doc.nodes[0].morph_weights.empty());  // no override: mesh wins
        CHECK(doc.nodes[1].morph_weights.size() == 2);
        CHECK(Near(doc.nodes[1].morph_weights[1], 0.5f));

        std::vector<Vec3> base;
        for (const VertexIn& v : prim.mesh.vertices)
            base.push_back(Vec3{v.position.x, v.position.y, v.position.z});

        // Both targets at once, and they ADD: the top-right vertex is the only
        // one both move, and it must pick up both deltas. Anything that treats
        // targets as alternatives rather than a sum leaves it short.
        std::vector<Vec3> out;
        anim::ApplyMorph(base, prim.morph_targets, {1.0f, 1.0f}, &out);
        CHECK(out.size() == 4);
        CHECK(Near(out[3].x, 1.75f) && Near(out[3].y, 1.5f));
        CHECK(Near(out[2].x, 0.0f) && Near(out[2].y, 1.5f));   // rise only
        CHECK(Near(out[1].x, 1.75f) && Near(out[1].y, 0.0f));  // lean only
        CHECK(Near(out[0].x, 0.0f) && Near(out[0].y, 0.0f));   // neither

        // Fractional and OVERSHOOTING weights. glTF permits weights outside
        // [0,1] deliberately, and clamping them silently caps an exaggerated
        // expression at its neutral extreme.
        anim::ApplyMorph(base, prim.morph_targets, {0.5f, -1.0f}, &out);
        CHECK(Near(out[3].y, 1.25f));
        CHECK(Near(out[3].x, 1.0f - 0.75f));

        // All weights zero must give back the base mesh exactly.
        anim::ApplyMorph(base, prim.morph_targets, {0.0f, 0.0f}, &out);
        for (std::size_t i = 0; i < out.size(); ++i)
            CHECK(Near(out[i].x, base[i].x) && Near(out[i].y, base[i].y));

        // Normals: only "rise" has any, so a pure "lean" leaves them alone.
        std::vector<Vec3> nbase;
        for (const VertexIn& v : prim.mesh.vertices)
            nbase.push_back(Vec3{v.normal.x, v.normal.y, v.normal.z});
        std::vector<Vec3> nout;
        anim::ApplyMorphNormals(nbase, prim.morph_targets, {0.0f, 1.0f}, &nout);
        CHECK(Near(nout[3].z, 1.0f) && Near(nout[3].y, 0.0f));
        anim::ApplyMorphNormals(nbase, prim.morph_targets, {1.0f, 0.0f}, &nout);
        CHECK(nout[3].y > 0.2f);
        // Renormalised: the weighted sum of unit vectors is not a unit vector,
        // and leaving it long shades as a lighting bug rather than a morph one.
        CHECK(Near(Length(nout[3]), 1.0f));

        // --- the weights animation channel ---
        CHECK(doc.animations.size() == 1);
        const anim::MorphTrack track = doc.MakeMorphTrack(0, 1);
        CHECK(track.Valid());
        CHECK(track.targets == 2);   // from the MESH; the channel does not say
        CHECK(track.times.size() == 3);
        CHECK(track.values.size() == 6);

        std::vector<float> w;
        track.Sample(0.0f, &w);
        CHECK(w.size() == 2 && Near(w[0], 0.0f) && Near(w[1], 0.0f));
        track.Sample(0.5f, &w);
        CHECK(Near(w[0], 1.0f) && Near(w[1], 0.5f));
        // Between keys, and the two targets interpolate INDEPENDENTLY -- a
        // stride bug that reads one float per key instead of two gives both
        // the same value here.
        track.Sample(0.25f, &w);
        CHECK(Near(w[0], 0.5f) && Near(w[1], 0.25f));
        track.Sample(0.75f, &w);
        CHECK(Near(w[0], 0.5f) && Near(w[1], 0.75f));

        // A weights channel is not a joint channel, and must be rejected as
        // one on PURPOSE rather than by accident. A mesh with THREE targets and
        // two keys stores six floats -- exactly what a two-key translation
        // channel stores -- so nothing about its shape gives it away. Read as a
        // translation it would throw the joint across the scene.
        anim::Channel as_joint;
        as_joint.joint = 0;
        as_joint.path = anim::Path::Weights;
        as_joint.times = {0.0f, 1.0f};
        as_joint.values = {0, 0, 0, 1, 1, 1};
        CHECK(as_joint.values.size() ==
              as_joint.times.size() * std::size_t(as_joint.Components()));
        CHECK(!as_joint.Valid(4));

        // And it does not survive into a clip either.
        for (const anim::Channel& c : doc.MakeClip(0, 0).channels)
            CHECK(c.path != anim::Path::Weights);

        // A node with no morph targets has no track, rather than an empty one
        // that would sample zeros over a mesh that has no weights at all.
        CHECK(!doc.MakeMorphTrack(0, 0).Valid());  // node 0 is not animated
        CHECK(!doc.MakeMorphTrack(9, 1).Valid());
        CHECK(!doc.MakeMorphTrack(0, 99).Valid());
    }

    // A morph target must cover every vertex. One that is short would morph
    // the front of the mesh and leave the rest behind, which reads as the
    // model tearing rather than as a bad file.
    {
        std::string doc(kMorphGltf);
        const std::string from = R"("bufferView":3,"componentType":5126,"count":4)";
        const std::string to = R"("bufferView":3,"componentType":5126,"count":3)";
        const std::size_t at = doc.find(from);
        CHECK(at != std::string::npos);
        doc.replace(at, from.size(), to);
        std::string error;
        CHECK(gltf::ParseGltf(doc, {}, error).primitives.empty());
        CHECK(error.find("morph target") != std::string::npos);
    }

    if (g_failures == 0) std::printf("gltf_test: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
