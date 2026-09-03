// Pose blending, crossfades, a state machine, and root motion.
//
// WHAT WAS MISSING. Sampling a clip gives one pose. A character needs to be
// walking and turning at once, to change from walking to running without the
// legs teleporting, and to have the animation MOVE it rather than sliding its
// feet along the ground while a separate controller drives the position. None
// of those is a property of a clip; all of them are properties of the layer
// between the clips and the skeleton, and that layer is what this is.
//
// WHY THIS IS NOT PART OF Clip. A clip is data -- keyframes, sampled at a time.
// Everything here is STATE: which clip is playing, how far into it, what it is
// fading from, how much of the previous fade is left. Putting state on the data
// means one clip cannot be played by two characters at once, which is the first
// thing anyone tries.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "engine/anim/anim.h"
#include "engine/core/math.h"

namespace eng::anim {

// Blends two poses component by component. `t` of 0 is `a`, 1 is `b`.
//
// COMPONENT-WISE, not matrix-wise, and this is the whole reason a Pose holds
// translation, rotation and scale separately. Blending two joint MATRICES
// halfway gives a matrix that is not a rigid transform: the rotation part
// shrinks toward the average of two rotation matrices, which is not a rotation,
// and a limb halfway through a 180-degree turn collapses to zero length.
//
// Rotations use the SHORTEST arc: two quaternions describing rotations 10
// degrees apart can have opposite signs, and interpolating those the long way
// spins the joint 350 degrees the wrong direction in one frame.
void BlendPoses(const Pose& a, const Pose& b, float t, Pose* out);

// The same, but only for `mask`ed joints -- the rest keep `a`'s values.
//
// What a LAYER is: an upper-body reload played over whatever the legs are
// doing. Without a mask the only way to combine two clips is to author a third
// containing both, which multiplies the animation budget by the number of
// combinations.
void BlendPosesMasked(const Pose& a, const Pose& b, float t,
                      const std::vector<float>& mask, Pose* out);

// ADDITIVE: adds (b - reference) on top of a.
//
// The case masks cannot cover. A breathing motion, a limp, a recoil -- these
// are deltas from a base pose that should apply whatever the base is doing,
// and blending toward them by weight would replace the base rather than modify
// it. The reference is the pose the additive clip was authored against, which
// is almost always its own first frame.
void AddPose(const Pose& base, const Pose& additive, const Pose& reference,
             float weight, Pose* out);

// A per-joint weight, built by naming a joint and taking everything under it.
// Values outside the subtree stay at whatever they were, so several calls
// compose.
void SetSubtreeWeight(const Skeleton&, int root_joint, float weight,
                      std::vector<float>* mask);

// --- root motion --------------------------------------------------------------
//
// THE PROBLEM IT SOLVES. A walk cycle authored in place has the feet sliding
// unless the character is moved at exactly the speed the animator drew. Get it
// wrong by 10% and the feet skate, which is the single most recognisable sign
// of a cheap animation system. Root motion inverts the relationship: the clip
// says how far the character travelled, and the controller applies that.
//
// The delta is EXTRACTED and REMOVED. Extracted so the caller can move the
// character by it; removed so the skeleton does not also move, which would
// double it.
struct RootMotion {
    Vec3 translation{0.0f, 0.0f, 0.0f};
    Quat rotation;
};

struct RootMotionConfig {
    // Which joint carries the motion. Usually the skeleton's root, or a
    // dedicated joint an exporter writes for exactly this.
    int joint = 0;
    // Take the horizontal motion. Off means the clip's forward travel is
    // ignored and the controller drives it.
    bool horizontal = true;
    // Take the VERTICAL motion too. Usually off: a walk cycle's bob should
    // stay in the animation, and letting it drive the character makes them
    // hop. On for a jump, where the arc IS the animation.
    bool vertical = false;
    bool rotation = true;
};

// --- the state machine --------------------------------------------------------

struct StateDesc {
    std::string name;
    const Clip* clip = nullptr;
    bool loop = true;
    float speed = 1.0f;
    // Seconds to fade INTO this state from whatever was playing.
    float blend_in = 0.2f;
};

// A one-dimensional blend space: several clips laid out along a parameter, with
// the two nearest blended.
//
// The case a state machine handles badly. Walk and run are not two states with
// a transition -- a character moving at 2.5 m/s is genuinely between them, and
// crossfading only looks right at the moment of the switch. Laying them out
// against speed and blending continuously is what makes acceleration read.
struct BlendSpaceDesc {
    std::string name;
    struct Sample {
        const Clip* clip = nullptr;
        // Where on the axis this clip sits: the speed, angle or lean it was
        // authored for.
        float position = 0.0f;
        // How fast this clip plays at its own position. Used for the phase
        // synchronisation below.
        float rate = 1.0f;
    };
    std::vector<Sample> samples;
    bool loop = true;
    float blend_in = 0.2f;

    // SYNCHRONISE THE PHASE across the samples, so that both clips have their
    // left foot down at the same moment.
    //
    // Without it, blending a walk and a run means blending a pose with its
    // left foot down against one with its right foot down, and the average is
    // a character standing with both feet in the middle -- the legs visibly
    // stop moving through the transition. This is the single thing that makes a
    // locomotion blend space work.
    bool synchronise = true;
};

class Animator {
  public:
    explicit Animator(const Skeleton&);
    ~Animator();

    Animator(const Animator&) = delete;
    Animator& operator=(const Animator&) = delete;

    // Returns an index used to refer to the state later.
    int AddState(const StateDesc&);
    int AddBlendSpace(const BlendSpaceDesc&);
    [[nodiscard]] int FindState(std::string_view name) const;

    // Starts a fade to `state`. Interrupting a fade already in progress is
    // allowed and is the normal case -- the current BLENDED pose becomes the
    // thing the new fade starts from, so a character interrupted mid-transition
    // does not snap back.
    void Play(int state, float blend_seconds = -1.0f);
    void PlayImmediate(int state);
    [[nodiscard]] int CurrentState() const;
    // 0 while a fade is running, 1 once it has finished.
    [[nodiscard]] float BlendProgress() const;
    [[nodiscard]] bool Transitioning() const;

    // The blend space's axis value. Ignored by ordinary states.
    void SetParameter(float);
    [[nodiscard]] float Parameter() const;

    // A layer played on top, masked to the joints whose weight is non-zero.
    // Weight 0 disables it without discarding its time, so an upper-body action
    // can be faded out and back in.
    void SetLayer(int state, const std::vector<float>& mask, float weight);
    void ClearLayer();

    void SetRootMotion(const RootMotionConfig&);
    [[nodiscard]] const RootMotionConfig& RootMotionSettings() const;

    // Advances by `dt` and produces the pose. The root motion accumulated over
    // this step is available from TakeRootMotion afterwards.
    void Update(float dt);
    [[nodiscard]] const Pose& CurrentPose() const;

    // The motion the clip travelled this step, in the CHARACTER's space, and
    // clears it. Taking rather than peeking, because applying the same delta
    // twice moves the character at double speed and the bug looks like a tuning
    // problem.
    RootMotion TakeRootMotion();

    // Normalised time within the active state, 0 to 1. For a footstep event, or
    // for deciding when a one-shot has finished.
    [[nodiscard]] float Phase() const;
    [[nodiscard]] bool Finished() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eng::anim
