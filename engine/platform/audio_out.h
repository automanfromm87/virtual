// The speaker.
//
// Everything above this is pure C++ and testable without hardware; this is the
// part that genuinely is the OS's. It owns one audio unit, asks a callback for
// samples, and does nothing else.
//
// The callback runs on a REAL-TIME THREAD that is not the game's. It has a hard
// deadline measured in milliseconds and misses it audibly, so it must not
// allocate, must not take a lock, and must not block on anything. That
// constraint is why the mixer above is lock-free and single-threaded rather
// than merely thread-safe: making it safe with a mutex would put a mutex here.
#pragma once

#include <memory>
#include <string>

namespace eng::platform {

class AudioOut {
  public:
    // Fills `out` with `frames` of INTERLEAVED STEREO. Called from the audio
    // thread. See the warning above.
    using RenderFn = void (*)(void* user, float* out, int frames);

    // `sample_rate` is a request; the device may choose another and SampleRate()
    // reports what it settled on. Resampling a mix to a rate the device did not
    // ask for is the device's job and it is better at it.
    [[nodiscard]] static std::unique_ptr<AudioOut> Create(int sample_rate,
                                                          RenderFn, void* user,
                                                          std::string& error);
    ~AudioOut();

    AudioOut(const AudioOut&) = delete;
    AudioOut& operator=(const AudioOut&) = delete;

    [[nodiscard]] int SampleRate() const;
    void Start();
    void Stop();
    [[nodiscard]] bool Running() const;

    // Public because the C render callback needs it, and a callback with C
    // linkage cannot be a friend. Opaque either way: nothing outside the
    // implementation can do anything with an incomplete type.
    struct Impl;

  private:
    AudioOut();
    std::unique_ptr<Impl> impl_;
};

}  // namespace eng::platform
