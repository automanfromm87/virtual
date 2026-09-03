// The layer between the game and the mixer.
//
// mixer_test covers the arithmetic. Nothing covered the part in front of it:
// the command queue, the id-to-voice table, and the ORDER Drain does things in.
// That layer has no sound of its own, which is precisely why its bugs are the
// quiet kind -- a handle that controls nothing, a Play that is refused with no
// way to find out, a Stop that arrives a block late.
//
// CreateSilent plus RenderForTest is the seam: no device, no real-time thread,
// and one audio block per call, so a test can say "now the audio thread runs"
// and mean it.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/audio/system.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

constexpr int kRate = 48000;
constexpr int kBlock = 256;

eng::audio::Clip Tone(float seconds, float amplitude = 1.0f) {
    eng::audio::Clip c;
    c.rate = kRate;
    c.channels = 1;
    const int frames = int(seconds * float(kRate));
    c.samples.resize(std::size_t(frames));
    for (int i = 0; i < frames; ++i)
        c.samples[std::size_t(i)] =
            amplitude * std::sin(2.0f * 3.14159265f * 440.0f * float(i) / kRate);
    return c;
}

// Runs one audio block and reports the loudest sample in it. Every command
// posted before this call is drained by it, and none posted after are.
float Block(eng::audio::AudioSystem& a) {
    std::vector<float> buf(std::size_t(kBlock) * 2, 0.0f);
    a.RenderForTest(buf.data(), kBlock);
    float peak = 0.0f;
    for (float v : buf) peak = std::fmax(peak, std::fabs(v));
    return peak;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const eng::audio::Clip tone = Tone(10.0f);
    const eng::audio::Clip blip = Tone(0.01f);  // 480 frames, under two blocks

    {
        std::printf("a handle controls the sound it was returned for\n");
        auto a = eng::audio::AudioSystem::CreateSilent(kRate, 8);
        eng::audio::PlayDesc d;
        d.clip = &tone;
        d.loop = true;
        const eng::audio::Sound s = a->Play(d);
        Check(s.Valid(), "Play returns a handle");
        const float before = Block(*a);
        Check(before > 0.1f, "and the sound is audible after one block");

        a->Stop(s);
        // Two blocks: the first drains the Stop and runs the 5 ms fade, the
        // second is the silence after it. One block would still be fading.
        Block(*a);
        Check(Block(*a) < 1e-3f, "and Stop through that handle silences it");
    }

    {
        std::printf("\nPlaying() tracks a one-shot from trigger to finish\n");
        auto a = eng::audio::AudioSystem::CreateSilent(kRate, 8);
        eng::audio::PlayDesc d;
        d.clip = &blip;
        const eng::audio::Sound s = a->Play(d);
        // BEFORE any block has run. The command is still in the queue and the
        // mixer has never heard of it -- but the game asked for a sound and one
        // is coming, so the honest answer is yes. Reporting false here is the
        // trap: a caller polling "has it finished" would see finished
        // immediately and retrigger every frame.
        Check(a->Playing(s), "a sound still in the queue reads as playing");
        Block(*a);
        Check(a->Playing(s), "and while it is actually sounding");
        // 480 frames at 256 a block: gone by the third.
        Block(*a);
        Block(*a);
        Check(!a->Playing(s), "and stops reading as playing once it runs out");
    }

    {
        std::printf("\nvoice starvation is reportable, not silent\n");
        auto a = eng::audio::AudioSystem::CreateSilent(kRate, 4);
        eng::audio::PlayDesc d;
        d.clip = &tone;
        d.loop = true;
        d.gain = 0.1f;
        eng::audio::Sound refused;
        for (int i = 0; i < 6; ++i) refused = a->Play(d);
        Block(*a);
        std::printf("    four voices, six asked for: %d active, %d starved, "
                    "%d dropped commands\n",
                    a->ActiveVoices(), a->StarvedVoices(), a->DroppedCommands());
        Check(a->ActiveVoices() == 4, "four voices are playing");
        Check(a->StarvedVoices() == 2, "and the two refusals are counted");
        // The distinction that matters: the QUEUE was never full, so the
        // dropped-command counter -- the only signal that existed before -- is
        // zero while two sounds went missing.
        Check(a->DroppedCommands() == 0, "with no commands dropped, which is a "
                                         "different failure");
        Check(!a->Playing(refused), "a refused sound does not read as playing");
    }

    {
        // THE REGRESSION. Drain used to process commands first and retire
        // finished handles afterwards, so a Play arriving while the table was
        // full took a mixer voice but no table slot -- and the handle it
        // returned controlled nothing for the life of the sound.
        //
        // Reproduced by filling the table with one-shots, letting them all
        // finish inside one block, and playing again in the next.
        std::printf("\na handle minted while the table is full still controls "
                    "its sound\n");
        auto a = eng::audio::AudioSystem::CreateSilent(kRate, 4);
        eng::audio::PlayDesc d;
        d.clip = &blip;
        d.gain = 0.5f;
        for (int i = 0; i < 4; ++i) (void)a->Play(d);
        // 480 frames of blip, 512 frames of block: every one of the four ends
        // inside this call, and its Live entry is still occupied at the top of
        // the next Drain.
        std::vector<float> buf(std::size_t(512) * 2, 0.0f);
        a->RenderForTest(buf.data(), 512);
        Check(a->ActiveVoices() == 0, "the table's four one-shots have finished");

        eng::audio::PlayDesc loop;
        loop.clip = &tone;
        loop.loop = true;
        const eng::audio::Sound s = a->Play(loop);
        Check(Block(*a) > 0.1f, "the next sound plays");
        a->Stop(s);
        Block(*a);
        const float after = Block(*a);
        std::printf("    peak two blocks after Stop: %.5f\n", after);
        Check(after < 1e-3f, "and its handle can still stop it");
    }

    {
        std::printf("\nStopAll clears the table, not just the voices\n");
        auto a = eng::audio::AudioSystem::CreateSilent(kRate, 8);
        eng::audio::PlayDesc d;
        d.clip = &tone;
        d.loop = true;
        d.gain = 0.2f;
        std::vector<eng::audio::Sound> all;
        for (int i = 0; i < 5; ++i) all.push_back(a->Play(d));
        Block(*a);
        a->StopAll();
        Block(*a);
        Block(*a);
        Check(a->ActiveVoices() == 0, "every voice is freed");
        bool any = false;
        for (eng::audio::Sound s : all) any = any || a->Playing(s);
        Check(!any, "and no handle claims to be playing");
        // The table is genuinely empty, not merely marked: five more sounds fit.
        for (int i = 0; i < 5; ++i) (void)a->Play(d);
        Block(*a);
        Check(a->ActiveVoices() == 5, "and five more sounds fit afterwards");
    }

    {
        std::printf("\nthe queue reports what it drops\n");
        auto a = eng::audio::AudioSystem::CreateSilent(kRate, 8);
        eng::audio::PlayDesc d;
        d.clip = &tone;
        // The ring holds 511. Posting past that without ever running a block
        // means nothing drains, and the overflow must be counted rather than
        // overwriting a command the audio thread has not read.
        for (int i = 0; i < 600; ++i) (void)a->Play(d);
        std::printf("    600 posted with no block run: %d dropped\n",
                    a->DroppedCommands());
        Check(a->DroppedCommands() == 600 - 511,
              "the overflow is counted exactly");
        // And a full queue refuses rather than corrupting: the 511 that fit are
        // still there and still play.
        Block(*a);
        Check(a->ActiveVoices() == 8, "the commands that fit still arrive");
    }

    {
        std::printf("\nSetPosition and SetGain reach the voice they name\n");
        auto a = eng::audio::AudioSystem::CreateSilent(kRate, 8);
        a->SetListener(eng::Vec3{0, 0, 0}, eng::Vec3{0, 0, -1}, eng::Vec3{0, 1, 0});
        eng::audio::PlayDesc d;
        d.clip = &tone;
        d.loop = true;
        d.spatial = true;
        d.position = eng::Vec3{0.0f, 0.0f, -2.0f};
        d.min_distance = 1.0f;
        d.max_distance = 40.0f;
        const eng::audio::Sound s = a->Play(d);
        Block(*a);
        Block(*a);  // past the 5 ms ramp
        const float near_peak = Block(*a);

        a->SetPosition(s, eng::Vec3{0.0f, 0.0f, -30.0f});
        Block(*a);
        Block(*a);
        const float far_peak = Block(*a);
        std::printf("    2 m away: %.4f, 30 m away: %.4f\n", near_peak, far_peak);
        Check(far_peak < near_peak * 0.25f, "moving a sound away quietens it");

        a->SetGain(s, 0.0f);
        Block(*a);
        Block(*a);
        Check(Block(*a) < 1e-3f, "and a gain of zero silences it");
    }

    std::printf(g_failures == 0 ? "\nsystem_test: all checks passed\n"
                                : "\nsystem_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
