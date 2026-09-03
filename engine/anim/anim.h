// Pure C++20. Skeletons, animation clips and vertex skinning.
//
// TWO separate things that are usually said in one breath:
//
//   ANIMATION is keyframes driving transforms over time. It needs no mesh and
//   no skeleton — the same machinery moves a camera or a door.
//   SKINNING is one vertex being influenced by several joints at once. It needs
//   no keyframes — a skinned mesh in a fixed pose is still skinned.
//
// They are separate types here for that reason. A clip samples into a Pose; a
// skeleton turns a Pose into a matrix palette; skinning consumes the palette.
// Nothing in this file touches a GPU, so the whole of it is exactly testable —
// which matters, because the failure mode of skinning is a mesh that looks
// almost right.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "engine/core/math.h"

namespace eng::anim {

// A joint's local transform. Stored as components, not a matrix: a rotation
// cannot be interpolated in matrix form without leaving the rotation group, and
// interpolation is the entire job here.
//
// Scale is a VECTOR. glTF allows non-uniform scale on a node and exporters
// emit it, so refusing it would mean refusing real files. Positions come out
// exact; normals are rotated but not inverse-transposed, which is correct for
// uniform scale and an approximation otherwise.
struct Transform {
    Vec3 translation{0.0f, 0.0f, 0.0f};
    Quat rotation;
    Vec3 scale{1.0f, 1.0f, 1.0f};

    [[nodiscard]] Mat4 Matrix() const;
};

struct Joint {
    std::string name;
    // Index into Skeleton::joints, or -1 for a root. A joint's parent is always
    // at a LOWER index — see Skeleton::ParentsFirst.
    int parent = -1;
    // Mesh space -> this joint's space, in the bind pose. glTF ships these
    // rather than deriving them, because the bind pose need not be the rest
    // pose stored in the node hierarchy.
    Mat4 inverse_bind = Mat4::Identity();
    // Local transform in the rest pose. Used for any joint a clip does not
    // animate, which is most of them in most clips.
    Transform rest;
};

struct Skeleton {
    std::vector<Joint> joints;

    // Evaluation order: indices into `joints`, every parent before its
    // children. Empty means "just walk joints in order", which is correct when
    // ParentsFirst() holds.
    //
    // The joints THEMSELVES cannot simply be sorted, because glTF's JOINTS_0
    // attribute indexes into the skin's joint array as the file wrote it.
    // Reordering the joints would mean rewriting every vertex; reordering the
    // evaluation is free.
    std::vector<int> order;

    // Builds `order`. Returns false if the parent links contain a cycle, in
    // which case `order` is left empty rather than partially built — a partial
    // order silently poses half a skeleton.
    bool Finalize();

    // True when every joint's parent index is smaller than its own, so one
    // forward pass computes every world matrix. Checked rather than assumed:
    // a file that violates it produces a skeleton where children are posed
    // from their parents' PREVIOUS frame, which reads as a one-frame lag that
    // only shows up when something moves fast.
    [[nodiscard]] bool ParentsFirst() const;
    [[nodiscard]] int Find(std::string_view name) const;
    [[nodiscard]] std::size_t Size() const { return joints.size(); }
};

// Local transforms for every joint at one instant.
struct Pose {
    std::vector<Transform> local;
};

enum class Interp : std::uint8_t {
    Linear,       // slerp for rotations, lerp for the rest
    Step,         // hold the previous key
    CubicSpline,  // Hermite, with tangents stored either side of each value
};

enum class Path : std::uint8_t { Translation, Rotation, Scale };

// One animated property of one joint.
struct Channel {
    int joint = -1;
    Path path = Path::Translation;
    Interp interp = Interp::Linear;
    // Ascending, in seconds.
    std::vector<float> times;
    // Components per key: 3 for translation and scale, 4 for rotation. Under
    // CubicSpline there are THREE times that many — in-tangent, value,
    // out-tangent — which is why this is a flat float array rather than a
    // vector of Vec3.
    std::vector<float> values;

    [[nodiscard]] int Components() const { return path == Path::Rotation ? 4 : 3; }
    [[nodiscard]] std::size_t KeyCount() const { return times.size(); }
    // Cheap structural check: the right number of values, times ascending, a
    // joint index in range.
    [[nodiscard]] bool Valid(std::size_t joint_count) const;
};

struct Clip {
    std::string name;
    float duration = 0.0f;
    std::vector<Channel> channels;

    // Fills `out` with the skeleton's rest pose, then applies every channel.
    // Starting from rest rather than from identity is not a detail: a clip
    // that animates only an arm would otherwise collapse the rest of the
    // skeleton into a heap at the origin.
    void Sample(float time, const Skeleton&, Pose* out, bool loop = true) const;
};

// Local transforms -> the matrices a skinned vertex is blended with.
//
// palette[j] = world(j) * inverse_bind(j). The inverse bind undoes the bind
// pose first, so a joint sitting exactly at its rest transform contributes the
// identity and the mesh does not move. Forget it and the mesh explodes away
// from the origin the moment it is skinned — a spectacular failure, at least.
void ComputeJointMatrices(const Skeleton&, const Pose&, std::vector<Mat4>* palette);

// World matrix per joint, before the inverse bind. For attaching things to
// bones, and for drawing the skeleton itself.
void ComputeJointWorld(const Skeleton&, const Pose&, std::vector<Mat4>* world);

// How many joints may influence one vertex. Four is the near-universal choice
// and what glTF's JOINTS_0/WEIGHTS_0 pair carries.
inline constexpr int kMaxInfluences = 4;

// Per-vertex skinning attributes, kept in their own array rather than added to
// VertexIn: a static mesh would otherwise pay 32 bytes a vertex for four zero
// weights it never uses.
struct SkinVertex {
    std::uint16_t joints[kMaxInfluences] = {0, 0, 0, 0};
    float weights[kMaxInfluences] = {0.0f, 0.0f, 0.0f, 0.0f};
};

// The reference blend. The GPU does the same arithmetic in a vertex shader;
// this exists so the arithmetic itself can be checked against known input, and
// so anything CPU-side that needs a posed vertex — a collider, a raycast — has
// one answer rather than two.
[[nodiscard]] Vec3 SkinPosition(Vec3 position, const SkinVertex&,
                                const std::vector<Mat4>& palette);
[[nodiscard]] Vec3 SkinNormal(Vec3 normal, const SkinVertex&,
                              const std::vector<Mat4>& palette);

// Weights that do not sum to 1 scale the vertex, which reads as a limb
// shrinking near a joint. Exporters round, so this fixes rather than rejects;
// an all-zero set falls back to joint 0 at full weight because dropping the
// vertex to the origin is worse than putting it somewhere.
void NormalizeWeights(SkinVertex*);

}  // namespace eng::anim
