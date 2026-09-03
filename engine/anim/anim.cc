#include "engine/anim/anim.h"

#include <algorithm>
#include <cmath>

namespace eng::anim {
namespace {

// Non-uniform scale, which Mat4::Scale deliberately does not offer. Confined to
// this file: it is correct for the positions skinning cares about, and the
// normal path below uses only the rotation, so no shader ever sees a basis it
// would need the inverse transpose of.
Mat4 ScaleMatrix(Vec3 s) {
    return Mat4{{{s.x, 0, 0, 0}, {0, s.y, 0, 0}, {0, 0, s.z, 0}, {0, 0, 0, 1}}};
}

Vec3 Lerp(Vec3 a, Vec3 b, float t) { return a + (b - a) * t; }

Vec3 ReadVec3(const std::vector<float>& v, std::size_t at) {
    return Vec3{v[at], v[at + 1], v[at + 2]};
}

Quat ReadQuat(const std::vector<float>& v, std::size_t at) {
    return Quat{v[at], v[at + 1], v[at + 2], v[at + 3]};
}

// Index of the last key at or before `t`. `times` is ascending and non-empty.
std::size_t KeyBefore(const std::vector<float>& times, float t) {
    // upper_bound gives the first key strictly after t; the one before it is
    // the segment start. Binary search because a long clip has thousands of
    // keys and this runs per channel per frame.
    const auto it = std::upper_bound(times.begin(), times.end(), t);
    if (it == times.begin()) return 0;
    return std::size_t(it - times.begin()) - 1;
}

// Hermite, with glTF's tangent scaling: the stored tangents are per unit of
// clip time, so they are multiplied by the segment length.
float Hermite(float v0, float out0, float v1, float in1, float t, float dt) {
    const float t2 = t * t;
    const float t3 = t2 * t;
    return (2 * t3 - 3 * t2 + 1) * v0 + dt * (t3 - 2 * t2 + t) * out0 +
           (-2 * t3 + 3 * t2) * v1 + dt * (t3 - t2) * in1;
}

}  // namespace

Mat4 Transform::Matrix() const {
    return Mat4::Translation(translation) * QuatToMat4(rotation) * ScaleMatrix(scale);
}

bool Skeleton::ParentsFirst() const {
    for (std::size_t i = 0; i < joints.size(); ++i) {
        const int p = joints[i].parent;
        if (p >= int(i)) return false;   // forward reference, or self
        if (p < -1) return false;
    }
    return true;
}

bool Skeleton::Finalize() {
    order.clear();
    const int n = int(joints.size());
    // Depth of each joint, computed by walking up. A cycle shows up as a walk
    // that never reaches a root within n steps.
    std::vector<int> depth(std::size_t(n), 0);
    for (int i = 0; i < n; ++i) {
        int cursor = i, steps = 0;
        while (cursor >= 0 && steps <= n) {
            const int p = joints[std::size_t(cursor)].parent;
            if (p < 0 || p >= n) break;
            cursor = p;
            ++steps;
        }
        if (steps > n) return false;  // cyclic
        depth[std::size_t(i)] = steps;
    }
    order.resize(std::size_t(n));
    for (int i = 0; i < n; ++i) order[std::size_t(i)] = i;
    // Stable, so joints at the same depth keep the file's order and the result
    // is reproducible between runs.
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return depth[std::size_t(a)] < depth[std::size_t(b)];
    });
    return true;
}

int Skeleton::Find(std::string_view name) const {
    for (std::size_t i = 0; i < joints.size(); ++i)
        if (joints[i].name == name) return int(i);
    return -1;
}

bool Channel::Valid(std::size_t joint_count) const {
    // Morph weights are not a joint property and have no fixed component count;
    // they belong in a MorphTrack. Refusing one here is what stops a weights
    // channel that happens to target a joint node from being decoded as a
    // translation and flinging the joint across the scene.
    if (path == Path::Weights) return false;
    if (joint < 0 || std::size_t(joint) >= joint_count) return false;
    if (times.empty()) return false;
    const std::size_t per_key =
        std::size_t(Components()) * (interp == Interp::CubicSpline ? 3u : 1u);
    if (values.size() != times.size() * per_key) return false;
    for (std::size_t i = 1; i < times.size(); ++i)
        if (times[i] < times[i - 1]) return false;
    return true;
}

void Clip::Sample(float time, const Skeleton& skeleton, Pose* out, bool loop) const {
    out->local.resize(skeleton.joints.size());
    for (std::size_t i = 0; i < skeleton.joints.size(); ++i)
        out->local[i] = skeleton.joints[i].rest;

    if (duration > 0.0f) {
        if (loop) {
            time = std::fmod(time, duration);
            if (time < 0.0f) time += duration;  // fmod keeps the sign
        } else {
            time = std::clamp(time, 0.0f, duration);
        }
    } else {
        time = 0.0f;
    }

    for (const Channel& ch : channels) {
        if (!ch.Valid(skeleton.joints.size())) continue;
        Transform& t = out->local[std::size_t(ch.joint)];
        const int n = ch.Components();
        const bool cubic = ch.interp == Interp::CubicSpline;
        const int per_key = n * (cubic ? 3 : 1);

        const std::size_t k0 = KeyBefore(ch.times, time);
        const std::size_t last = ch.times.size() - 1;
        // Off either end, or a single key: hold. Extrapolating a rotation past
        // the end of a clip is how a limb ends up somewhere anatomically
        // impossible on the frame after the animation stops.
        const bool at_end = k0 >= last || time <= ch.times.front();
        const std::size_t k1 = at_end ? k0 : k0 + 1;

        float u = 0.0f;
        if (!at_end) {
            const float span = ch.times[k1] - ch.times[k0];
            u = span > 0.0f ? (time - ch.times[k0]) / span : 0.0f;
        }
        if (ch.interp == Interp::Step) u = 0.0f;

        // Under CubicSpline each key is [in-tangent, value, out-tangent], so
        // the value sits one component-group into the key.
        const std::size_t v0 = std::size_t(k0) * std::size_t(per_key) +
                               (cubic ? std::size_t(n) : 0u);
        const std::size_t v1 = std::size_t(k1) * std::size_t(per_key) +
                               (cubic ? std::size_t(n) : 0u);

        if (ch.path == Path::Rotation) {
            const Quat a = ReadQuat(ch.values, v0);
            const Quat b = ReadQuat(ch.values, v1);
            if (cubic && !at_end) {
                const float dt = ch.times[k1] - ch.times[k0];
                float q[4];
                for (int c = 0; c < 4; ++c)
                    q[c] = Hermite(ch.values[v0 + std::size_t(c)],
                                   ch.values[v0 + std::size_t(n + c)],
                                   ch.values[v1 + std::size_t(c)],
                                   ch.values[v1 - std::size_t(n) + std::size_t(c)],
                                   u, dt);
                // Hermite leaves the result off the unit sphere; a rotation
                // matrix built from it would scale the joint.
                t.rotation = Normalize(Quat{q[0], q[1], q[2], q[3]});
            } else {
                // SLERP, not lerp. glTF says spherical, and a lerp of two
                // widely separated rotations speeds up in the middle — the
                // classic "robot arm snaps" artefact.
                t.rotation = Slerp(a, b, u);
            }
            continue;
        }

        Vec3 value;
        if (cubic && !at_end) {
            const float dt = ch.times[k1] - ch.times[k0];
            float c3[3];
            for (int c = 0; c < 3; ++c)
                c3[c] = Hermite(ch.values[v0 + std::size_t(c)],
                                ch.values[v0 + std::size_t(n + c)],
                                ch.values[v1 + std::size_t(c)],
                                ch.values[v1 - std::size_t(n) + std::size_t(c)],
                                u, dt);
            value = Vec3{c3[0], c3[1], c3[2]};
        } else {
            value = Lerp(ReadVec3(ch.values, v0), ReadVec3(ch.values, v1), u);
        }
        if (ch.path == Path::Translation) t.translation = value;
        else t.scale = value;
    }
}

void ComputeJointWorld(const Skeleton& skeleton, const Pose& pose,
                       std::vector<Mat4>* world) {
    const std::size_t n = skeleton.joints.size();
    world->assign(n, Mat4::Identity());
    for (std::size_t k = 0; k < n; ++k) {
        // Walk the evaluation order when there is one; otherwise index order,
        // which is the same thing whenever ParentsFirst() holds.
        const std::size_t i =
            skeleton.order.size() == n ? std::size_t(skeleton.order[k]) : k;
        const Transform& local =
            i < pose.local.size() ? pose.local[i] : skeleton.joints[i].rest;
        const Mat4 m = local.Matrix();
        const int p = skeleton.joints[i].parent;
        (*world)[i] = (p >= 0 && std::size_t(p) < n) ? (*world)[std::size_t(p)] * m : m;
    }
}

void ComputeJointMatrices(const Skeleton& skeleton, const Pose& pose,
                          std::vector<Mat4>* palette) {
    ComputeJointWorld(skeleton, pose, palette);
    for (std::size_t i = 0; i < palette->size(); ++i)
        (*palette)[i] = (*palette)[i] * skeleton.joints[i].inverse_bind;
}

void NormalizeWeights(SkinVertex* v) {
    float sum = 0.0f;
    for (float w : v->weights) sum += w;
    if (sum <= 1e-8f) {
        v->weights[0] = 1.0f;
        for (int i = 1; i < kMaxInfluences; ++i) v->weights[i] = 0.0f;
        return;
    }
    const float inv = 1.0f / sum;
    for (float& w : v->weights) w *= inv;
}

Vec3 SkinPosition(Vec3 position, const SkinVertex& s,
                  const std::vector<Mat4>& palette) {
    Vec3 out{0.0f, 0.0f, 0.0f};
    float used = 0.0f;
    for (int i = 0; i < kMaxInfluences; ++i) {
        const float w = s.weights[i];
        if (w == 0.0f) continue;
        const std::size_t j = s.joints[i];
        if (j >= palette.size()) continue;  // a bad index must not read memory
        const Vec4 p = palette[j] * Vec4{position.x, position.y, position.z, 1.0f};
        out = out + Vec3{p.x, p.y, p.z} * w;
        used += w;
    }
    // Nothing influenced it: leave it where it was rather than at the origin.
    return used > 0.0f ? out : position;
}

Vec3 SkinNormal(Vec3 normal, const SkinVertex& s, const std::vector<Mat4>& palette) {
    Vec3 out{0.0f, 0.0f, 0.0f};
    float used = 0.0f;
    for (int i = 0; i < kMaxInfluences; ++i) {
        const float w = s.weights[i];
        if (w == 0.0f) continue;
        const std::size_t j = s.joints[i];
        if (j >= palette.size()) continue;
        // w = 0 drops the translation: a normal is a direction.
        const Vec4 nn = palette[j] * Vec4{normal.x, normal.y, normal.z, 0.0f};
        out = out + Vec3{nn.x, nn.y, nn.z} * w;
        used += w;
    }
    if (used <= 0.0f) return normal;
    const float len = Length(out);
    // Blending two opposed rotations can cancel to nothing; keeping the
    // original beats handing the shader a zero vector to normalise.
    return len > 1e-6f ? out * (1.0f / len) : normal;
}

bool MorphTrack::Valid() const {
    if (targets <= 0 || times.empty()) return false;
    const std::size_t per_key =
        std::size_t(targets) * (interp == Interp::CubicSpline ? 3u : 1u);
    if (values.size() != times.size() * per_key) return false;
    for (std::size_t i = 1; i < times.size(); ++i)
        if (times[i] < times[i - 1]) return false;
    return true;
}

void MorphTrack::Sample(float time, std::vector<float>* out, bool loop) const {
    if (!out || !Valid()) return;
    out->assign(std::size_t(targets), 0.0f);

    if (duration > 0.0f) {
        if (loop) {
            time = std::fmod(time, duration);
            if (time < 0.0f) time += duration;
        } else {
            time = std::clamp(time, 0.0f, duration);
        }
    } else {
        time = 0.0f;
    }

    const bool cubic = interp == Interp::CubicSpline;
    const int per_key = targets * (cubic ? 3 : 1);
    const std::size_t k0 = KeyBefore(times, time);
    const std::size_t last = times.size() - 1;
    const bool at_end = k0 >= last || time <= times.front();
    const std::size_t k1 = at_end ? k0 : k0 + 1;

    float u = 0.0f;
    if (!at_end) {
        const float span = times[k1] - times[k0];
        u = span > 0.0f ? (time - times[k0]) / span : 0.0f;
    }
    if (interp == Interp::Step) u = 0.0f;

    const std::size_t n = std::size_t(targets);
    const std::size_t v0 =
        std::size_t(k0) * std::size_t(per_key) + (cubic ? n : 0u);
    const std::size_t v1 =
        std::size_t(k1) * std::size_t(per_key) + (cubic ? n : 0u);

    for (std::size_t c = 0; c < n; ++c) {
        if (cubic && !at_end) {
            (*out)[c] = Hermite(values[v0 + c], values[v0 + n + c],
                                values[v1 + c], values[v1 - n + c], u,
                                times[k1] - times[k0]);
        } else {
            (*out)[c] = values[v0 + c] + (values[v1 + c] - values[v0 + c]) * u;
        }
    }
}

void ApplyMorph(const std::vector<Vec3>& base,
                const std::vector<MorphTarget>& targets,
                const std::vector<float>& weights, std::vector<Vec3>* out) {
    if (!out) return;
    *out = base;
    const std::size_t n =
        std::min(targets.size(), weights.size());  // a short list is not fatal
    for (std::size_t t = 0; t < n; ++t) {
        const float w = weights[t];
        if (w == 0.0f) continue;
        const std::vector<Vec3>& d = targets[t].positions;
        const std::size_t m = std::min(d.size(), out->size());
        for (std::size_t i = 0; i < m; ++i) (*out)[i] = (*out)[i] + d[i] * w;
    }
}

void ApplyMorphNormals(const std::vector<Vec3>& base,
                       const std::vector<MorphTarget>& targets,
                       const std::vector<float>& weights,
                       std::vector<Vec3>* out) {
    if (!out) return;
    *out = base;
    const std::size_t n = std::min(targets.size(), weights.size());
    bool moved = false;
    for (std::size_t t = 0; t < n; ++t) {
        const float w = weights[t];
        if (w == 0.0f || targets[t].normals.empty()) continue;
        moved = true;
        const std::vector<Vec3>& d = targets[t].normals;
        const std::size_t m = std::min(d.size(), out->size());
        for (std::size_t i = 0; i < m; ++i) (*out)[i] = (*out)[i] + d[i] * w;
    }
    // Renormalised, because the weighted sum of unit vectors is not one. Left
    // alone, the shading darkens wherever two targets pull a normal apart --
    // which looks like a lighting bug rather than a morph bug.
    if (!moved) return;
    for (Vec3& v : *out) {
        const float len = Length(v);
        if (len > 1e-6f) v = v * (1.0f / len);
    }
}

}  // namespace eng::anim
