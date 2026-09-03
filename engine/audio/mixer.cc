#include "engine/audio/mixer.h"

#include <algorithm>
#include <cmath>

namespace eng::audio {
namespace {

constexpr float kPi = 3.14159265358979f;

// How long a gain change takes to arrive, in seconds.
//
// Not zero. Applying a new gain instantly steps the waveform, and a step is a
// click -- which is why a sound that is merely stopped clicks, and why a
// volume slider dragged quickly crackles. Five milliseconds is short enough to
// be imperceptible as a fade and long enough to be inaudible as an edge.
constexpr float kRampSeconds = 0.005f;

}  // namespace

struct Mixer::Voice {
    const Clip* clip = nullptr;
    std::uint32_t generation = 0;  // 0 = free
    bool active = false;
    bool loop = false;
    bool stopping = false;

    // Read position in the CLIP's own frames, fractional because the clip's
    // rate and the device's rarely match.
    double cursor = 0.0;
    double step = 1.0;

    float gain = 1.0f;
    bool spatial = false;
    Vec3 position{0, 0, 0};
    float min_distance = 1.0f;
    float max_distance = 40.0f;
    float pan = 0.0f;

    // The gains actually applied, which chase the targets. See kRampSeconds.
    float left = 0.0f, right = 0.0f;
    float target_left = 0.0f, target_right = 0.0f;
    bool primed = false;
};

Mixer::Mixer(int sample_rate, int max_voices)
    : voices_(std::size_t(std::max(1, max_voices))),
      rate_(std::max(1, sample_rate)) {}
Mixer::~Mixer() = default;

int Mixer::ActiveVoices() const {
    int n = 0;
    for (const Voice& v : voices_)
        if (v.active) ++n;
    return n;
}

Mixer::Voice* Mixer::Find(VoiceId id) {
    if (!id.Valid()) return nullptr;
    const std::uint32_t index = id.bits & 0xFFFFu;
    const std::uint32_t generation = id.bits >> 16;
    if (index >= voices_.size()) return nullptr;
    Voice& v = voices_[index];
    // The generation is what stops a stale handle controlling whatever took
    // over the slot. Without it, a caller holding the id of a footstep that
    // finished can silence the music that reused its voice.
    return v.generation == generation && v.active ? &v : nullptr;
}
const Mixer::Voice* Mixer::Find(VoiceId id) const {
    return const_cast<Mixer*>(this)->Find(id);
}

bool Mixer::Playing(VoiceId id) const { return Find(id) != nullptr; }

void Mixer::SetListener(Vec3 position, Vec3 forward, Vec3 up) {
    listener_ = position;
    const Vec3 f = Normalize(forward);
    Vec3 r = Cross(f, up);
    // A listener looking straight up has a forward parallel to `up`, so the
    // cross product collapses and panning becomes a division by zero. Any
    // perpendicular will do when that happens; the alternative is silence in
    // one ear.
    if (Dot(r, r) < 1e-8f) r = Cross(f, Vec3{1.0f, 0.0f, 0.0f});
    if (Dot(r, r) < 1e-8f) r = Vec3{1.0f, 0.0f, 0.0f};
    listener_right_ = Normalize(r);
    listener_forward_ = f;
}

VoiceId Mixer::Play(const PlayDesc& d) {
    if (!d.clip || !d.clip->Valid()) return {};
    std::size_t slot = voices_.size();
    for (std::size_t i = 0; i < voices_.size(); ++i)
        if (!voices_[i].active) { slot = i; break; }
    if (slot == voices_.size()) {
        // Every voice busy. REFUSED rather than stealing one: a stolen voice
        // cuts a sound that is still playing, and the sound most likely to be
        // cut is the longest one -- the music.
        ++starved_;
        return {};
    }

    Voice& v = voices_[slot];
    v = Voice{};
    v.clip = d.clip;
    v.active = true;
    v.loop = d.loop;
    v.gain = d.gain;
    v.spatial = d.spatial;
    v.position = d.position;
    v.min_distance = std::max(1e-3f, d.min_distance);
    v.max_distance = std::max(v.min_distance + 1e-3f, d.max_distance);
    v.pan = std::clamp(d.pan, -1.0f, 1.0f);
    // The clip's rate against the device's IS the resampling. A 44.1 kHz clip
    // played on a 48 kHz device without this is a semitone sharp and 9% short.
    v.step = double(d.clip->rate) / double(rate_) * double(std::max(0.01f, d.pitch));
    v.generation = next_generation_++;
    if (next_generation_ > 0xFFFFu) next_generation_ = 1;
    return VoiceId{(v.generation << 16) | std::uint32_t(slot)};
}

void Mixer::Stop(VoiceId id) {
    if (Voice* v = Find(id)) v->stopping = true;
}

void Mixer::StopAll() {
    for (Voice& v : voices_)
        if (v.active) v.stopping = true;
}

void Mixer::SetGain(VoiceId id, float g) {
    if (Voice* v = Find(id)) v->gain = std::max(0.0f, g);
}

void Mixer::SetPosition(VoiceId id, Vec3 p) {
    if (Voice* v = Find(id)) v->position = p;
}

void Mixer::Render(float* out, int frames) {
    if (!out || frames <= 0) return;
    std::fill(out, out + std::size_t(frames) * 2, 0.0f);

    const float ramp = 1.0f / (kRampSeconds * float(rate_));

    for (Voice& v : voices_) {
        if (!v.active) continue;
        const Clip& c = *v.clip;
        const std::size_t clip_frames = c.Frames();

        // --- where this voice sits in the stereo field ------------------------
        float amplitude = v.gain;
        float pan = v.pan;
        if (v.spatial) {
            const Vec3 to = v.position - listener_;
            const float distance = Length(to);
            // Inverse distance, rolled off to zero at max_distance. Pure
            // inverse-square never reaches zero, so a sound would have to be
            // mixed for as long as it existed however far away it was -- the
            // same reason the renderer's lights are windowed.
            if (distance >= v.max_distance) {
                amplitude = 0.0f;
            } else if (distance > v.min_distance) {
                const float raw = v.min_distance / distance;
                const float window =
                    1.0f - (distance - v.min_distance) /
                               (v.max_distance - v.min_distance);
                amplitude *= raw * window * window;
            }
            // Panned by the component along the listener's RIGHT, which is why
            // the listener needs an orientation and not just a position: a
            // sound to the west is on your left or your right depending on
            // which way you are facing.
            pan = distance > 1e-4f ? std::clamp(Dot(to, listener_right_) / distance,
                                                -1.0f, 1.0f)
                                   : 0.0f;
        }

        // EQUAL POWER panning: the two gains are a cosine and a sine of the
        // same angle, so their squares sum to one everywhere. Linear panning --
        // (1-p)/2 and (1+p)/2 -- sums to one in AMPLITUDE instead, which is
        // three decibels quieter in the middle, and a sound panning across the
        // field audibly sags as it passes the centre.
        const float angle = (pan + 1.0f) * 0.25f * kPi;
        v.target_left = amplitude * std::cos(angle) * master_;
        v.target_right = amplitude * std::sin(angle) * master_;
        if (v.stopping) { v.target_left = 0.0f; v.target_right = 0.0f; }
        // A voice starting is ramped up from silence like any other change; a
        // voice that began at full gain would click at its own first sample.
        if (!v.primed) { v.primed = true; }

        for (int i = 0; i < frames; ++i) {
            if (v.cursor >= double(clip_frames)) {
                if (!v.loop) { v.active = false; break; }
                // Wrapped by SUBTRACTION, not reset to zero: the cursor is
                // fractional, and dropping the fraction at every loop point
                // shortens the clip by up to a sample each time round and
                // makes the seam audible.
                v.cursor -= double(clip_frames);
            }

            const std::size_t i0 = std::size_t(v.cursor);
            const std::size_t i1 = (i0 + 1 < clip_frames)
                                       ? i0 + 1
                                       : (v.loop ? 0 : clip_frames - 1);
            const float t = float(v.cursor - double(i0));

            float l = 0.0f, r = 0.0f;
            if (c.channels == 1) {
                const float a = c.samples[i0], b = c.samples[i1];
                l = r = a + (b - a) * t;
            } else {
                const std::size_t s0 = i0 * std::size_t(c.channels);
                const std::size_t s1 = i1 * std::size_t(c.channels);
                l = c.samples[s0] + (c.samples[s1] - c.samples[s0]) * t;
                r = c.samples[s0 + 1] + (c.samples[s1 + 1] - c.samples[s0 + 1]) * t;
            }

            v.left += std::clamp(v.target_left - v.left, -ramp, ramp);
            v.right += std::clamp(v.target_right - v.right, -ramp, ramp);
            out[i * 2] += l * v.left;
            out[i * 2 + 1] += r * v.right;
            v.cursor += v.step;
        }

        // A stopping voice is freed only once it has actually faded, or the
        // fade never happens.
        if (v.stopping && v.left < 1e-4f && v.right < 1e-4f) v.active = false;
        if (!v.active) v.generation = 0;
    }

    // CLAMPED, and the peak reported. Letting samples past 1 through wraps or
    // saturates in whatever the device does with them, and either way the
    // symptom is distortion with no clue where it came from.
    peak_ = 0.0f;
    for (int i = 0; i < frames * 2; ++i) {
        peak_ = std::max(peak_, std::fabs(out[i]));
        out[i] = std::clamp(out[i], -1.0f, 1.0f);
    }
}

}  // namespace eng::audio
