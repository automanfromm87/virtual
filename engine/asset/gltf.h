// Pure C++20. A glTF 2.0 reader.
//
// glTF because it is the only interchange format whose material model maps onto
// this renderer without translation: base colour, metallic, roughness, one set
// of texture coordinates. An OBJ importer would be a third the size and then
// need a second, invented convention for everything a PBR material needs.
//
// SUPPORTED: .gltf with embedded base64 buffers or a sibling .bin; meshes
// (POSITION, NORMAL, TEXCOORD_0, JOINTS_0, WEIGHTS_0, indices); the node
// hierarchy with TRS or matrix transforms; pbrMetallicRoughness factors; PNG
// images, whether embedded as a data uri, stored in a bufferView, or sitting
// next to the file; skins; animations with LINEAR, STEP and CUBICSPLINE
// sampling.
//
// NOT SUPPORTED, and each is a real piece of work rather than an oversight:
// .glb containers, JPEG images, sparse accessors, morph targets, cameras.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "engine/anim/anim.h"
#include "engine/core/math.h"
#include "engine/geometry/mesh.h"
#include "engine/texture/texture.h"

namespace eng::gltf {

struct MaterialDef {
    std::string name;
    Vec4 base_color{1.0f, 1.0f, 1.0f, 1.0f};
    float metallic = 1.0f;   // glTF's default, not this engine's
    float roughness = 1.0f;
    // Indices into Document::images, or -1. Already resolved through glTF's
    // material -> texture -> image indirection, because nothing above this
    // layer has a use for the sampler objects in between.
    int base_color_image = -1;
    int metallic_roughness_image = -1;
};

// One glTF mesh primitive, already converted into the engine's vertex format.
struct Primitive {
    Mesh mesh;
    int material = -1;  // index into Document::materials, or -1
    // Parallel to mesh.vertices, and EMPTY for an unskinned primitive. Kept
    // separate so a static mesh does not carry 32 bytes a vertex of zeroes.
    // The joint indices are as the file wrote them, which is what the skin's
    // joint array is indexed by.
    std::vector<anim::SkinVertex> skin;

    [[nodiscard]] bool Skinned() const { return !skin.empty(); }
};

struct Node {
    std::string name;
    // Index into Document::skins, or -1. A node with both a mesh and a skin is
    // a skinned mesh instance.
    int skin = -1;
    Mat4 local = Mat4::Identity();
    // Indices into Document::primitives. A glTF mesh may hold several
    // primitives, and each becomes its own draw.
    std::vector<int> primitives;
    std::vector<int> children;
    int parent = -1;
};

// A glTF skin, already turned into something the animation layer can pose.
struct SkinDef {
    std::string name;
    anim::Skeleton skeleton;
    // Joint index -> the glTF node it came from. Animation channels target
    // nodes, so this is what retargets a clip onto this skeleton.
    std::vector<int> joint_nodes;
};

// A channel as the file stores it: aimed at a NODE, not at a joint. Which joint
// that is depends on which skin you are posing, and a file may have several.
struct NodeChannel {
    int node = -1;
    anim::Path path = anim::Path::Translation;
    anim::Interp interp = anim::Interp::Linear;
    std::vector<float> times;
    std::vector<float> values;
};

struct AnimationDef {
    std::string name;
    float duration = 0.0f;
    std::vector<NodeChannel> channels;
};

struct Document {
    std::vector<Primitive> primitives;
    std::vector<MaterialDef> materials;
    // Decoded to RGBA8. An image the reader could not decode is left EMPTY
    // rather than dropped, so material indices stay valid and the caller can
    // substitute a placeholder for exactly the one that failed.
    std::vector<Texture2D> images;
    // Parallel to `images`. Non-empty only where the image lives in a separate
    // file that ParseGltf could not reach; LoadGltfFile resolves those.
    std::vector<std::string> image_uris;
    std::vector<Node> nodes;
    std::vector<int> roots;

    std::vector<SkinDef> skins;
    std::vector<AnimationDef> animations;

    [[nodiscard]] bool Empty() const { return primitives.empty(); }

    // Retargets animation `animation` onto skin `skin`'s skeleton, dropping any
    // channel aimed at a node that skin does not use. Returns an empty clip for
    // an out-of-range index.
    //
    // Separate from parsing because the two are genuinely independent: one file
    // can hold several skins and several clips, and every pairing is legal.
    [[nodiscard]] anim::Clip MakeClip(int animation, int skin) const;
    // Accumulated parent transforms, in the same order as `nodes`.
    [[nodiscard]] std::vector<Mat4> WorldTransforms() const;
};

// `bin` supplies any buffer whose uri is absent or external; buffers with a
// base64 data uri are decoded from the JSON itself.
[[nodiscard]] Document ParseGltf(std::string_view json,
                                 const std::vector<std::uint8_t>& bin,
                                 std::string& error);

// Reads a .gltf from disk, resolving a sibling .bin if the document names one.
[[nodiscard]] Document LoadGltfFile(const std::string& path, std::string& error);

}  // namespace eng::gltf
