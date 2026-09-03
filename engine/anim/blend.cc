#include "engine/anim/blend.h"

#include <algorithm>
#include <cmath>

namespace eng::anim {
namespace {

// Slerp already negates one end when the two point into opposite hemispheres,
// which is the shortest-arc fix -- q and -q are the same rotation, and without
// it two keys ten degrees apart with opposite signs interpolate the long way
// and spin the joint 350 degrees in one frame. Named here so the call sites
// below say what they rely on rather than assuming it.
Quat ShortestSlerp(const Quat& a, const Quat& b, float t) {
    return Normalize(Slerp(a, b, t));
}

Transform BlendTransform(const Transform& a, const Transform& b, float t) {
    Transform out;
    out.translation = a.translation + (b.translation - a.translation) * t;
    out.rotation = ShortestSlerp(a.rotation, b.rotation, t);
    out.scale = a.scale + (b.scale - a.scale) * t;
    return out;
}

}  // namespace

void BlendPoses(const Pose& a, const Pose& b, float t, Pose* out) {
    if (!out) return;
    const std::size_t n = std::min(a.local.size(), b.local.size());
    out->local.resize(std::max(a.local.size(), b.local.size()));
    t = std::clamp(t, 0.0f, 1.0f);
    for (std::size_t i = 0; i < n; ++i)
        out->local[i] = BlendTransform(a.local[i], b.local[i], t);
    // Joints only one side has keep that side's value rather than being left
    // uninitialised. Mismatched sizes mean a caller mixed two skeletons, which
    // is a bug -- but producing garbage transforms makes it look like a
    // skinning bug instead.
    for (std::size_t i = n; i < a.local.size(); ++i) out->local[i] = a.local[i];
    for (std::size_t i = n; i < b.local.size(); ++i) out->local[i] = b.local[i];
}

void BlendPosesMasked(const Pose& a, const Pose& b, float t,
                      const std::vector<float>& mask, Pose* out) {
    if (!out) return;
    const std::size_t n = std::min(a.local.size(), b.local.size());
    out->local.resize(a.local.size());
    for (std::size_t i = 0; i < a.local.size(); ++i) out->local[i] = a.local[i];
    for (std::size_t i = 0; i < n; ++i) {
        const float w = i < mask.size() ? mask[i] : 0.0f;
        const float k = std::clamp(t * w, 0.0f, 1.0f);
        if (k <= 0.0f) continue;
        out->local[i] = BlendTransform(a.local[i], b.local[i], k);
    }
}

void AddPose(const Pose& base, const Pose& additive, const Pose& reference,
             float weight, Pose* out) {
    if (!out) return;
    out->local.resize(base.local.size());
    weight = std::clamp(weight, 0.0f, 1.0f);
    for (std::size_t i = 0; i < base.local.size(); ++i) {
        out->local[i] = base.local[i];
        if (i >= additive.local.size() || i >= reference.local.size()) continue;
        const Transform& a = additive.local[i];
        const Transform& r = reference.local[i];

        // TRANSLATION AND SCALE add; ROTATION composes. That asymmetry is not a
        // choice -- rotations do not add, and treating a quaternion difference
        // as a sum gives a value that is not a unit quaternion and a joint that
        // stretches as it turns.
        out->local[i].translation =
            base.local[i].translation + (a.translation - r.translation) * weight;
        out->local[i].scale =
            base.local[i].scale + (a.scale - r.scale) * weight;

        // The delta rotation, then applied to the base by the weight. Conjugate
        // rather than inverse: these are unit quaternions, so they are the same
        // thing and the conjugate is three sign flips instead of a division.
        const Quat inv_ref = Quat{-r.rotation.x, -r.rotation.y, -r.rotation.z,
                                  r.rotation.w};
        Quat delta = Normalize(a.rotation * inv_ref);
        // Scaled by slerping from identity, which is the only way to take a
        // fraction of a rotation.
        const Quat identity;
        delta = ShortestSlerp(identity, delta, weight);
        out->local[i].rotation = Normalize(delta * base.local[i].rotation);
    }
}

void SetSubtreeWeight(const Skeleton& skeleton, int root_joint, float weight,
                      std::vector<float>* mask) {
    if (!mask) return;
    mask->resize(skeleton.joints.size(), 0.0f);
    if (root_joint < 0 || std::size_t(root_joint) >= skeleton.joints.size()) return;
    (*mask)[std::size_t(root_joint)] = weight;
    // ONE FORWARD PASS, which works because a parent's index is always lower
    // than its children's -- the property Skeleton::ParentsFirst exists to
    // assert. A skeleton that violated it would need a real traversal, and the
    // symptom of using this anyway is a mask that covers some of a limb.
    for (std::size_t i = 0; i < skeleton.joints.size(); ++i) {
        const int parent = skeleton.joints[i].parent;
        if (parent < 0 || std::size_t(parent) >= i) continue;
        if ((*mask)[std::size_t(parent)] > 0.0f) (*mask)[i] = weight;
    }
}

// -------------------------------------------------------------------- Animator

namespace {

struct State {
    StateDesc desc;
    BlendSpaceDesc space;
    bool is_space = false;
    float duration = 0.0f;
};

float ClipDuration(const Clip* c) { return c ? c->duration : 0.0f; }

}  // namespace

struct Animator::Impl {
    const Skeleton* skeleton = nullptr;
    std::vector<State> states;

    int current = -1;
    int previous = -1;
    float time = 0.0f;
    float previous_time = 0.0f;
    float blend = 1.0f;         // 0 at the start of a fade, 1 when done
    float blend_rate = 0.0f;    // per second
    float parameter = 0.0f;

    // The blended pose at the moment a fade was interrupted. See Play().
    Pose frozen;
    bool use_frozen = false;

    int layer_state = -1;
    std::vector<float> layer_mask;
    float layer_weight = 0.0f;
    float layer_time = 0.0f;

    RootMotionConfig root_config;
    RootMotion pending;
    Transform last_root;
    bool have_last_root = false;

    Pose pose;          // the answer
    Pose scratch_a, scratch_b, scratch_c, scratch_layer;

    // Samples one state into `out`, and returns where in its cycle it is.
    float SampleState(int index, float at, Pose* out) const;
};

float Animator::Impl::SampleState(int index, float at, Pose* out) const {
    if (index < 0 || std::size_t(index) >= states.size() || !skeleton) return 0.0f;
    const State& s = states[std::size_t(index)];
    if (!s.is_space) {
        if (!s.desc.clip) return 0.0f;
        s.desc.clip->Sample(at, *skeleton, out, s.desc.loop);
        return s.duration > 0.0f ? at / s.duration : 0.0f;
    }

    // --- a blend space -------------------------------------------------------
    const auto& samples = s.space.samples;
    if (samples.empty()) return 0.0f;
    if (samples.size() == 1) {
        if (samples[0].clip) samples[0].clip->Sample(at, *skeleton, out, s.space.loop);
        return 0.0f;
    }

    // The two samples the parameter sits between. A linear scan: a blend space
    // has a handful of samples and a binary search would be more code than the
    // loop it replaced.
    std::size_t hi = 0;
    while (hi < samples.size() && samples[hi].position < parameter) ++hi;
    const std::size_t b = std::min(hi, samples.size() - 1);
    const std::size_t a = b == 0 ? 0 : b - 1;
    float t = 0.0f;
    if (a != b) {
        const float span = samples[b].position - samples[a].position;
        t = span > 1e-6f ? (parameter - samples[a].position) / span : 0.0f;
        t = std::clamp(t, 0.0f, 1.0f);
    }

    if (!samples[a].clip || !samples[b].clip) return 0.0f;

    if (s.space.synchronise) {
        // PHASE, not time. Each clip is sampled at the same FRACTION of its own
        // length, so a walk and a run both have their left foot down at the
        // same moment and the blend between them is two similar poses rather
        // than two opposite ones.
        //
        // Without this the average of "left foot down" and "right foot down" is
        // a character standing still with both feet together, and the legs
        // visibly stop moving through the transition. It is the single thing
        // that makes a locomotion blend space work.
        const float phase = at;  // already normalised, see Update
        const float da = ClipDuration(samples[a].clip);
        const float db = ClipDuration(samples[b].clip);
        Pose& pa = const_cast<Impl*>(this)->scratch_a;
        Pose& pb = const_cast<Impl*>(this)->scratch_b;
        samples[a].clip->Sample(phase * da, *skeleton, &pa, s.space.loop);
        samples[b].clip->Sample(phase * db, *skeleton, &pb, s.space.loop);
        BlendPoses(pa, pb, t, out);
        return phase;
    }

    Pose& pa = const_cast<Impl*>(this)->scratch_a;
    Pose& pb = const_cast<Impl*>(this)->scratch_b;
    samples[a].clip->Sample(at, *skeleton, &pa, s.space.loop);
    samples[b].clip->Sample(at, *skeleton, &pb, s.space.loop);
    BlendPoses(pa, pb, t, out);
    return 0.0f;
}

Animator::Animator(const Skeleton& skeleton) : impl_(std::make_unique<Impl>()) {
    impl_->skeleton = &skeleton;
    impl_->pose.local.resize(skeleton.joints.size());
    for (std::size_t i = 0; i < skeleton.joints.size(); ++i)
        impl_->pose.local[i] = skeleton.joints[i].rest;
}
Animator::~Animator() = default;

int Animator::AddState(const StateDesc& desc) {
    State s;
    s.desc = desc;
    s.duration = ClipDuration(desc.clip);
    impl_->states.push_back(std::move(s));
    return int(impl_->states.size()) - 1;
}

int Animator::AddBlendSpace(const BlendSpaceDesc& desc) {
    State s;
    s.space = desc;
    s.is_space = true;
    s.desc.name = desc.name;
    s.desc.loop = desc.loop;
    s.desc.blend_in = desc.blend_in;
    // SORTED by position, so the scan in SampleState can stop at the first
    // sample past the parameter. An unsorted list gives a blend between two
    // arbitrary clips, which looks like the parameter doing nothing.
    std::sort(s.space.samples.begin(), s.space.samples.end(),
              [](const BlendSpaceDesc::Sample& a, const BlendSpaceDesc::Sample& b) {
                  return a.position < b.position;
              });
    for (const auto& sample : s.space.samples)
        s.duration = std::max(s.duration, ClipDuration(sample.clip));
    impl_->states.push_back(std::move(s));
    return int(impl_->states.size()) - 1;
}

int Animator::FindState(std::string_view name) const {
    for (std::size_t i = 0; i < impl_->states.size(); ++i)
        if (impl_->states[i].desc.name == name) return int(i);
    return -1;
}

void Animator::Play(int state, float blend_seconds) {
    Impl& im = *impl_;
    if (state < 0 || std::size_t(state) >= im.states.size()) return;
    if (state == im.current && im.blend >= 1.0f) return;

    if (blend_seconds < 0.0f) blend_seconds = im.states[std::size_t(state)].desc.blend_in;
    if (blend_seconds <= 1e-4f) {
        PlayImmediate(state);
        return;
    }

    // INTERRUPTING A FADE needs a different source from starting one.
    //
    // Normally the pose being faded FROM is another state, and sampling it live
    // is what keeps its own animation running through the transition -- a
    // character changing direction mid-stride keeps striding. That is worth
    // having and it is why `previous` is a state index rather than a snapshot.
    //
    // But when a fade is ALREADY running, the pose on screen is a blend of two
    // states and is not any state. Naming either of them as the source makes
    // the pose jump to it: measured at 35.65 degrees in one frame for a
    // reversal a third of the way through a half-second fade. So an interrupt
    // freezes the blended pose and fades from that instead. The frozen source
    // is slightly stiffer than a live one, and an interrupt is exactly the
    // moment when nobody can tell.
    if (im.blend < 1.0f && im.previous >= 0) {
        im.frozen = im.pose;
        im.use_frozen = true;
        im.previous = -1;
    } else {
        im.use_frozen = false;
        im.previous = im.current;
        im.previous_time = im.time;
    }
    im.current = state;
    im.time = 0.0f;
    im.blend = 0.0f;
    im.blend_rate = 1.0f / blend_seconds;
}

void Animator::PlayImmediate(int state) {
    Impl& im = *impl_;
    if (state < 0 || std::size_t(state) >= im.states.size()) return;
    im.previous = -1;
    im.use_frozen = false;
    im.current = state;
    im.time = 0.0f;
    im.blend = 1.0f;
    im.blend_rate = 0.0f;
    im.have_last_root = false;
}

int Animator::CurrentState() const { return impl_->current; }
float Animator::BlendProgress() const { return impl_->blend; }
bool Animator::Transitioning() const {
    return impl_->blend < 1.0f && impl_->previous >= 0;
}
void Animator::SetParameter(float v) { impl_->parameter = v; }
float Animator::Parameter() const { return impl_->parameter; }

void Animator::SetLayer(int state, const std::vector<float>& mask, float weight) {
    impl_->layer_state = state;
    impl_->layer_mask = mask;
    impl_->layer_weight = std::clamp(weight, 0.0f, 1.0f);
}
void Animator::ClearLayer() {
    impl_->layer_state = -1;
    impl_->layer_weight = 0.0f;
}

void Animator::SetRootMotion(const RootMotionConfig& c) { impl_->root_config = c; }
const RootMotionConfig& Animator::RootMotionSettings() const {
    return impl_->root_config;
}

void Animator::Update(float dt) {
    Impl& im = *impl_;
    if (!im.skeleton || im.current < 0) return;
    const State& cur = im.states[std::size_t(im.current)];

    // A blend space advances in PHASE -- a fraction of a cycle -- rather than
    // in seconds, because its samples have different lengths and a shared
    // second would put them at different points in their cycles. An ordinary
    // state advances in seconds, which is what Clip::Sample expects.
    if (cur.is_space) {
        float rate = 1.0f;
        if (!cur.space.samples.empty()) {
            // The cycle rate at the current parameter, interpolated the same
            // way the poses are. A walk cycle is 1.2 s and a run is 0.7 s, so
            // playing the blend at either one makes the other's feet slide.
            const auto& s = cur.space.samples;
            std::size_t hi = 0;
            while (hi < s.size() && s[hi].position < im.parameter) ++hi;
            const std::size_t b = std::min(hi, s.size() - 1);
            const std::size_t a = b == 0 ? 0 : b - 1;
            float t = 0.0f;
            if (a != b) {
                const float span = s[b].position - s[a].position;
                t = span > 1e-6f ? std::clamp((im.parameter - s[a].position) / span,
                                              0.0f, 1.0f)
                                 : 0.0f;
            }
            const float da = std::max(ClipDuration(s[a].clip), 1e-4f);
            const float db = std::max(ClipDuration(s[b].clip), 1e-4f);
            rate = 1.0f / (da + (db - da) * t);
        }
        im.time += dt * cur.desc.speed * rate;
        if (cur.space.loop) im.time = im.time - std::floor(im.time);
        else im.time = std::clamp(im.time, 0.0f, 1.0f);
    } else {
        im.time += dt * cur.desc.speed;
        if (cur.desc.loop && cur.duration > 0.0f)
            im.time = std::fmod(im.time, cur.duration);
        else if (cur.duration > 0.0f)
            im.time = std::min(im.time, cur.duration);
    }
    im.previous_time += dt;
    im.layer_time += dt;

    if (im.blend < 1.0f) {
        im.blend = std::min(1.0f, im.blend + dt * im.blend_rate);
        if (im.blend >= 1.0f) {
            im.previous = -1;
            im.use_frozen = false;
        }
    }

    // --- sample --------------------------------------------------------------
    im.SampleState(im.current, im.time, &im.scratch_c);
    if (im.blend < 1.0f && (im.previous >= 0 || im.use_frozen)) {
        // SMOOTHSTEP rather than a straight ramp. A linear crossfade has a
        // velocity discontinuity at both ends -- the pose starts moving at full
        // speed and stops dead -- and on a fast transition that reads as a
        // flinch at each end.
        const float t = im.blend * im.blend * (3.0f - 2.0f * im.blend);
        if (im.use_frozen) {
            BlendPoses(im.frozen, im.scratch_c, t, &im.pose);
        } else {
            Pose from;
            im.SampleState(im.previous, im.previous_time, &from);
            BlendPoses(from, im.scratch_c, t, &im.pose);
        }
    } else {
        im.pose = im.scratch_c;
    }

    if (im.layer_state >= 0 && im.layer_weight > 0.0f) {
        im.SampleState(im.layer_state, im.layer_time, &im.scratch_layer);
        Pose base = im.pose;
        BlendPosesMasked(base, im.scratch_layer, im.layer_weight, im.layer_mask,
                         &im.pose);
    }

    // --- root motion ---------------------------------------------------------
    const RootMotionConfig& rc = im.root_config;
    if (rc.joint >= 0 && std::size_t(rc.joint) < im.pose.local.size() &&
        (rc.horizontal || rc.vertical || rc.rotation)) {
        Transform& root = im.pose.local[std::size_t(rc.joint)];
        const Transform now = root;
        if (im.have_last_root) {
            // The DELTA since last frame, in the root's own frame. A looping
            // clip wraps, and the wrap produces a delta the size of the whole
            // cycle backwards -- which would teleport the character to the
            // start of the loop every cycle. Rejecting a delta larger than a
            // plausible step is the standard guard, and it is why this cannot
            // simply subtract.
            Vec3 d = now.translation - im.last_root.translation;
            const float span = Length(d);
            // Half a metre in one frame at 60 Hz is 30 m/s. Anything larger is
            // the loop point, not motion.
            if (span < 0.5f) {
                if (!rc.horizontal) { d.x = 0.0f; d.z = 0.0f; }
                if (!rc.vertical) d.y = 0.0f;
                im.pending.translation = im.pending.translation + d;
            }
            if (rc.rotation) {
                const Quat inv = Quat{-im.last_root.rotation.x, -im.last_root.rotation.y,
                                      -im.last_root.rotation.z, im.last_root.rotation.w};
                const Quat delta = Normalize(now.rotation * inv);
                if (std::fabs(delta.w) > 0.9f)  // under ~50 degrees; not a wrap
                    im.pending.rotation = Normalize(delta * im.pending.rotation);
            }
        }
        im.last_root = now;
        im.have_last_root = true;

        // REMOVED from the pose, so the skeleton does not also travel. Leaving
        // it in and applying the delta to the character as well moves everything
        // at double speed, and the character drifts away from its own collision
        // capsule.
        if (rc.horizontal) { root.translation.x = 0.0f; root.translation.z = 0.0f; }
        if (rc.vertical) root.translation.y = 0.0f;
        if (rc.rotation) root.rotation = Quat{};
    }
}

const Pose& Animator::CurrentPose() const { return impl_->pose; }

RootMotion Animator::TakeRootMotion() {
    const RootMotion m = impl_->pending;
    impl_->pending = RootMotion{};
    return m;
}

float Animator::Phase() const {
    const Impl& im = *impl_;
    if (im.current < 0) return 0.0f;
    const State& s = im.states[std::size_t(im.current)];
    if (s.is_space) return im.time;  // already normalised
    return s.duration > 0.0f ? std::clamp(im.time / s.duration, 0.0f, 1.0f) : 0.0f;
}

bool Animator::Finished() const {
    const Impl& im = *impl_;
    if (im.current < 0) return true;
    const State& s = im.states[std::size_t(im.current)];
    if (s.is_space ? s.space.loop : s.desc.loop) return false;
    return s.duration <= 0.0f || im.time >= s.duration - 1e-5f;
}

}  // namespace eng::anim
