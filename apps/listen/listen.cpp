// Plays a file, and moves a sound around the listener so the spatialisation is
// audible rather than merely tested.
//
// Headless: it runs the mixer offline as well as through the speaker, so the
// same code path can be measured and heard.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "engine/audio/system.h"
#include "engine/audio/wav.h"
#include "engine/platform/audio_file.h"

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: listen <audio file>\n"
                     "  Any format the OS decodes: flac, mp3, m4a, aac, wav.\n");
        return 2;
    }
    const std::string path = argv[1];

    std::string error;
    // Through the OS, so FLAC, mp3, m4a and wav all work. engine/audio's own
    // wav decoder is for short sounds, where paying a system decode at the
    // moment a footstep is triggered is a spike in the frame that caused it.
    const eng::audio::Clip music = eng::platform::DecodeAudioFile(path, error);
    if (!music.Valid()) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }
    std::printf("%s\n  %d channels, %d Hz, %.2f seconds\n", path.c_str(),
                music.channels, music.rate, music.Seconds());

    // A short blip, synthesised rather than loaded: it needs no asset and its
    // frequency is known, which makes it the thing to move around.
    //
    // DECLARED BEFORE THE SYSTEM, like `music` above. Locals destruct in
    // reverse order, so this is what puts the device's teardown ahead of the
    // samples it was reading; the other order frees them under a live audio
    // callback. The blips are the ones that matter -- they are never stopped,
    // so one triggered near the end is still playing at the return.
    eng::audio::Clip blip;

    auto audio = eng::audio::AudioSystem::Create(error);
    if (!audio) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }
    std::printf("  device running at %d Hz\n", audio->SampleRate());

    blip.rate = audio->SampleRate();
    blip.channels = 1;
    {
        const int frames = blip.rate / 4;
        blip.samples.resize(std::size_t(frames));
        for (int i = 0; i < frames; ++i) {
            const float t = float(i) / float(blip.rate);
            // An envelope, because a tone that starts and stops instantly
            // clicks at both ends and the click is louder than the tone.
            const float env = std::min(t * 40.0f, 1.0f) * std::exp(-t * 6.0f);
            blip.samples[std::size_t(i)] =
                0.6f * env * std::sin(2.0f * 3.14159265f * 660.0f * t);
        }
    }

    audio->SetListener(eng::Vec3{0, 0, 0}, eng::Vec3{0, 0, -1}, eng::Vec3{0, 1, 0});

    eng::audio::PlayDesc m;
    m.clip = &music;
    m.gain = 1.4f;  // the track's rms is 0.06; it is a quiet master
    m.loop = true;
    (void)audio->Play(m);
    std::printf("  playing, and a blip orbits the listener once every 4 s\n"
                "  (ctrl-c to stop)\n");

    const auto start = std::chrono::steady_clock::now();
    eng::audio::Sound orbit;
    double last_blip = -1.0;
    for (;;) {
        const double t = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - start).count();
        if (t > 40.0) break;

        // A new blip twice a second, each placed on a circle around the
        // listener: the pan should sweep left to right and back.
        if (t - last_blip > 0.35) {
            last_blip = t;
            const float a = float(t) * 1.57f;
            eng::audio::PlayDesc d;
            d.clip = &blip;
            d.spatial = true;
            d.position = eng::Vec3{std::sin(a) * 6.0f, 0.0f, -std::cos(a) * 6.0f};
            d.min_distance = 1.0f;
            d.max_distance = 30.0f;
            d.gain = 0.9f;
            orbit = audio->Play(d);
        }
        // Reported on its own cadence, NOT at the moment a blip is triggered.
        // Printing there showed one voice for a scene that plainly had two: the
        // command has not been drained yet and the previous blip, being shorter
        // than the gap, has already ended.
        static double last_print = -1.0;
        if (t - last_print > 0.25) {
            last_print = t;
            const float a = float(t) * 1.57f;
            std::printf("\r  %5.1fs  %d voices  peak %.3f  blip bearing "
                        "%+.2f (left to right)      ", t, audio->ActiveVoices(),
                        audio->LastPeak(), std::sin(a));
            std::fflush(stdout);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
    // EVERYTHING, not just the song: the orbiting blips were never stopped and
    // one of them is always mid-flight here. The sleep is what lets the 5 ms
    // fade actually happen before the device goes away.
    audio->StopAll();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::printf("\n  dropped commands: %d\n", audio->DroppedCommands());
    return 0;
}
