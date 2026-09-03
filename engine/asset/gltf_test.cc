// No test framework — from scratch means from scratch.
//
// The fixture is a real glTF 2.0 document with a base64-embedded buffer, so the
// test exercises the whole path — JSON, base64, bufferViews, accessors, node
// hierarchy — without needing a file on disk or a copy of Blender.
#include "engine/asset/gltf.h"

#include <cmath>
#include <cstdio>

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

const char* const kTexturedGltf =
#include "engine/asset/testdata_textured_gltf.inc"
    ;

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

    if (g_failures == 0) std::printf("gltf_test: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
