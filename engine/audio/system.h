// The audio system: a mixer, a speaker, and the one queue between them.
//
// This is the only audio type a game touches. Play a sound, move the listener,
// stop the music -- all from the game thread, none of it blocking, none of it
// aware that a real-time thread is pulling samples out the other side.
#pragma once

#include <memory>
#include <string>

#include "engine/audio/clip.h"
#include "engine/audio/mixer.h"
#include "engine/core/math.h"

namespace eng::audio {

// A handle to something playing. Minted on the game thread, which is why it is
// not a Mixer::VoiceId -- that one names a slot the audio thread owns and the
// game thread has no business knowing about.
struct Sound {
    std::uint32_t id = 0;
    [[nodiscard]] bool Valid() const { return id != 0; }
};

class AudioSystem {
  public:
    // `sample_rate` is a request. The device may pick another; SampleRate()
    // reports what it settled on and clips are resampled to it.
    [[nodiscard]] static std::unique_ptr<AudioSystem> Create(
        std::string& error, int sample_rate = 48000, int max_voices = 64);
    ~AudioSystem();

    AudioSystem(const AudioSystem&) = delete;
    AudioSystem& operator=(const AudioSystem&) = delete;

    // The clip must OUTLIVE the sound. Nothing is copied: a Clip is megabytes
    // and copying one per footstep would allocate on the game thread at exactly
    // the moment the frame is busiest.
    Sound Play(const PlayDesc&);
    void Stop(Sound);
    void StopAll();
    void SetGain(Sound, float);
    void SetPosition(Sound, Vec3);
    void SetListener(Vec3 position, Vec3 forward, Vec3 up);
    void SetMasterGain(float);

    [[nodiscard]] int SampleRate() const;
    [[nodiscard]] int ActiveVoices() const;
    // Commands lost because the queue was full. Non-zero means sounds are being
    // silently dropped, which is indistinguishable from never triggering them.
    [[nodiscard]] int DroppedCommands() const;
    // The loudest sample the last block produced BEFORE clamping. Above 1 the
    // mix is clipping, which is audible as distortion and shows up nowhere else.
    [[nodiscard]] float LastPeak() const;

    // Runs the mixer directly instead of on a device, for tests and for
    // rendering audio offline. Only valid on a system created with no device.
    void RenderForTest(float* out, int frames);
    [[nodiscard]] static std::unique_ptr<AudioSystem> CreateSilent(
        int sample_rate, int max_voices = 64);

    // Public because the audio thread's C callback needs it, and a function
    // with C linkage cannot be a friend. Opaque regardless.
    struct Impl;

  private:
    AudioSystem();
    std::unique_ptr<Impl> impl_;
};

}  // namespace eng::audio
