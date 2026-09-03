// The mixer: many voices in, one stereo stream out.
//
// Pure C++ and completely free of CoreAudio, which is the whole point. The
// arithmetic here -- panning, distance, resampling, summing -- is what can be
// wrong, and every bit of it is checkable by rendering into an array and
// measuring it. An audio bug found by listening is a bug found late.
//
// SINGLE THREADED by design. Render() runs on the audio callback's thread and
// nothing else may touch the mixer while it does. Making it thread-safe with a
// lock would put a lock in a real-time callback, which is the classic way to
// produce dropouts; engine/audio/engine.h owns the queue that avoids that.
#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "engine/audio/clip.h"
#include "engine/core/math.h"

namespace eng::audio {

// A handle to a playing sound. Generational, so a handle to a voice that has
// finished and had its slot reused does not silently control the new sound.
struct VoiceId {
    std::uint32_t bits = 0;
    [[nodiscard]] bool Valid() const { return bits != 0; }
};

struct PlayDesc {
    const Clip* clip = nullptr;
    float gain = 1.0f;
    // Playback rate multiplier. 2 is an octave up and half as long.
    float pitch = 1.0f;
    bool loop = false;

    // --- 3D ------------------------------------------------------------------
    // When false the sound is placed by `pan` alone: music, narration, UI.
    bool spatial = false;
    Vec3 position{0.0f, 0.0f, 0.0f};
    // Full volume within `min_distance`, silent beyond `max_distance`. The
    // near radius exists because pure inverse-square goes to infinity at zero,
    // and a sound the listener walks through would blow the mix apart.
    float min_distance = 1.0f;
    float max_distance = 40.0f;
    // -1 hard left, +1 hard right. Ignored when spatial.
    float pan = 0.0f;
};

class Mixer {
  public:
    explicit Mixer(int sample_rate, int max_voices = 64);
    ~Mixer();

    Mixer(const Mixer&) = delete;
    Mixer& operator=(const Mixer&) = delete;

    [[nodiscard]] VoiceId Play(const PlayDesc&);
    // Fades out over a few milliseconds rather than cutting. Stopping a voice
    // mid-waveform steps the output to zero, and a step is a click.
    void Stop(VoiceId);
    void StopAll();
    [[nodiscard]] bool Playing(VoiceId) const;
    void SetGain(VoiceId, float);
    void SetPosition(VoiceId, Vec3);

    // Where the listener is and which way it faces. `forward` and `up` need not
    // be normalised or exactly perpendicular.
    void SetListener(Vec3 position, Vec3 forward, Vec3 up);
    // Clamped to >= 0, like SetGain. A negative master is not a quieter mix, it
    // is an inverted one -- and it defeats the fade-out test in Render, which
    // asks whether a gain has reached zero.
    void SetMasterGain(float g) { master_ = std::max(0.0f, g); }

    // Renders `frames` of INTERLEAVED STEREO into `out`, which must hold
    // frames * 2 floats. Adds nothing: the buffer is overwritten.
    void Render(float* out, int frames);

    [[nodiscard]] int ActiveVoices() const;
    [[nodiscard]] int SampleRate() const { return rate_; }
    // How many Play calls were refused because every voice was busy. A silently
    // dropped sound is indistinguishable from one that was never triggered.
    [[nodiscard]] int Starved() const { return starved_; }
    // The largest absolute sample the last Render produced, before clamping.
    // Above 1 means the mix is clipping, which is audible as distortion and
    // invisible in every other measurement.
    [[nodiscard]] float LastPeak() const { return peak_; }

  private:
    struct Voice;
    [[nodiscard]] Voice* Find(VoiceId);
    [[nodiscard]] const Voice* Find(VoiceId) const;

    std::vector<Voice> voices_;
    int rate_ = 48000;
    std::uint32_t next_generation_ = 1;
    int starved_ = 0;
    float peak_ = 0.0f;
    float master_ = 1.0f;
    Vec3 listener_{0.0f, 0.0f, 0.0f};
    // The listener's RIGHT is the only part of its orientation the pan needs --
    // the forward it was derived from is not stored, because storing a vector
    // nothing reads implies a front/back attenuation that does not exist.
    Vec3 listener_right_{1.0f, 0.0f, 0.0f};
};

}  // namespace eng::audio
