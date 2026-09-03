// The mixer, measured rather than listened to.
//
// Every audio bug sounds like "something is a bit off" and none of them are
// visible in a waveform at a glance. A pan that dips in the middle, a clip
// resampled to the wrong pitch, a loop that clicks once a second, a voice that
// never frees its slot: all of them are arithmetic, and arithmetic can be
// checked against a number.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/audio/mixer.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

constexpr int kRate = 48000;

eng::audio::Clip Sine(float hz, float seconds, int rate = kRate, int channels = 1,
                      float amplitude = 1.0f) {
    eng::audio::Clip c;
    c.rate = rate;
    c.channels = channels;
    const int frames = int(seconds * float(rate));
    c.samples.resize(std::size_t(frames) * channels);
    for (int i = 0; i < frames; ++i) {
        const float v = amplitude * std::sin(2.0f * 3.14159265f * hz *
                                             float(i) / float(rate));
        for (int ch = 0; ch < channels; ++ch)
            c.samples[std::size_t(i) * channels + ch] = v;
    }
    return c;
}

// Root mean square of one channel of an interleaved stereo buffer. The honest
// measure of loudness -- a peak says what the loudest sample was and nothing
// about how loud it sounds.
float Rms(const std::vector<float>& buf, int channel) {
    double sum = 0.0;
    std::size_t n = 0;
    for (std::size_t i = std::size_t(channel); i < buf.size(); i += 2) {
        sum += double(buf[i]) * buf[i];
        ++n;
    }
    return n ? float(std::sqrt(sum / double(n))) : 0.0f;
}

// Zero crossings, which give the frequency without a Fourier transform.
int Crossings(const std::vector<float>& buf, int channel) {
    int n = 0;
    float prev = 0.0f;
    bool first = true;
    for (std::size_t i = std::size_t(channel); i < buf.size(); i += 2) {
        if (!first && ((prev < 0.0f) != (buf[i] < 0.0f))) ++n;
        prev = buf[i];
        first = false;
    }
    return n;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("mixer\n");

    const eng::audio::Clip tone = Sine(440.0f, 1.0f);
    const int kFrames = kRate / 4;  // a quarter second
    std::vector<float> buf(std::size_t(kFrames) * 2);

    // --- silence when nothing is playing --------------------------------------
    {
        eng::audio::Mixer m(kRate);
        // Deliberately dirty, because Render must OVERWRITE rather than add:
        // an audio callback hands the same buffer back every time, and a mixer
        // that accumulates into it turns one frame of sound into a growing
        // roar within a second.
        std::fill(buf.begin(), buf.end(), 0.5f);
        m.Render(buf.data(), kFrames);
        Check(Rms(buf, 0) == 0.0f && Rms(buf, 1) == 0.0f,
              "an idle mixer writes silence over whatever was there");
        Check(m.ActiveVoices() == 0, "and has no voices");
    }

    // --- one voice, centred ------------------------------------------------------
    {
        eng::audio::Mixer m(kRate);
        eng::audio::PlayDesc d;
        d.clip = &tone;
        const auto id = m.Play(d);
        Check(id.Valid() && m.Playing(id), "a voice starts");
        m.Render(buf.data(), kFrames);
        const float l = Rms(buf, 0), r = Rms(buf, 1);
        std::printf("    centred: left rms %.4f, right rms %.4f\n", l, r);
        Check(l > 0.1f, "and is audible");
        Check(std::fabs(l - r) < 1e-4f, "equally in both ears when centred");

        // A quarter second of 440 Hz crosses zero 220 times, twice per cycle.
        const int x = Crossings(buf, 0);
        std::printf("    %d zero crossings in 0.25 s (440 Hz gives 220)\n", x);
        Check(std::abs(x - 220) <= 2, "at the frequency the clip was written at");
    }

    // --- EQUAL POWER panning ------------------------------------------------------
    //
    // The check that separates equal-power from linear panning. Both put a
    // centred sound in the middle and a hard-panned one in one ear; they differ
    // only in the middle's LOUDNESS, and linear panning is 3 dB quiet there --
    // audible as a sound sagging as it crosses the centre.
    {
        std::printf("    panning:\n");
        float centre_power = 0.0f, left_power = 0.0f;
        for (int k = -2; k <= 2; ++k) {
            const float pan = float(k) * 0.5f;
            eng::audio::Mixer m(kRate);
            eng::audio::PlayDesc d;
            d.clip = &tone;
            d.pan = pan;
            (void)m.Play(d);
            m.Render(buf.data(), kFrames);
            const float l = Rms(buf, 0), r = Rms(buf, 1);
            const float power = l * l + r * r;
            std::printf("      pan %+.1f: l %.4f r %.4f, total power %.5f\n", pan,
                        l, r, power);
            if (k == 0) centre_power = power;
            if (k == -2) left_power = power;
        }
        Check(std::fabs(centre_power - left_power) < left_power * 0.05f,
              "total power is constant across the pan, not just amplitude");
    }
    {
        // ...and the ears are the right way round. A swapped pan is the single
        // easiest audio bug to ship, because it sounds completely normal.
        eng::audio::Mixer m(kRate);
        eng::audio::PlayDesc d;
        d.clip = &tone;
        d.pan = -1.0f;
        (void)m.Play(d);
        m.Render(buf.data(), kFrames);
        std::printf("    hard left: l %.4f r %.4f\n", Rms(buf, 0), Rms(buf, 1));
        Check(Rms(buf, 0) > Rms(buf, 1) * 10.0f, "pan -1 is the LEFT ear");
    }

    // --- resampling ----------------------------------------------------------------
    //
    // A 44.1 kHz clip on a 48 kHz device must come out at the frequency it was
    // recorded at. Playing it sample-for-sample instead is a semitone sharp and
    // nine percent short, which sounds fine until it is next to something that
    // was resampled properly.
    {
        const eng::audio::Clip at_44k = Sine(440.0f, 1.0f, 44100);
        eng::audio::Mixer m(kRate);
        eng::audio::PlayDesc d;
        d.clip = &at_44k;
        (void)m.Play(d);
        m.Render(buf.data(), kFrames);
        const int x = Crossings(buf, 0);
        std::printf("    a 44.1 kHz clip on a 48 kHz device: %d crossings "
                    "(440 Hz gives 220, unresampled would give %d)\n",
                    x, int(220.0 * 48000.0 / 44100.0));
        Check(std::abs(x - 220) <= 2, "is resampled to its recorded pitch");
    }
    {
        // ...and PITCH is a deliberate multiplier on top of that.
        eng::audio::Mixer m(kRate);
        eng::audio::PlayDesc d;
        d.clip = &tone;
        d.pitch = 2.0f;
        (void)m.Play(d);
        m.Render(buf.data(), kFrames);
        const int x = Crossings(buf, 0);
        std::printf("    pitch 2.0: %d crossings (an octave up gives 440)\n", x);
        Check(std::abs(x - 440) <= 3, "doubling the pitch doubles the frequency");
    }

    // --- distance ------------------------------------------------------------------
    {
        eng::audio::Mixer m(kRate);
        m.SetListener(eng::Vec3{0, 0, 0}, eng::Vec3{0, 0, -1}, eng::Vec3{0, 1, 0});
        std::printf("    distance:\n");
        float previous = 1e9f;
        bool falls = true;
        for (float dist : {1.0f, 2.0f, 5.0f, 10.0f, 20.0f, 39.0f, 41.0f}) {
            eng::audio::Mixer mm(kRate);
            mm.SetListener(eng::Vec3{0, 0, 0}, eng::Vec3{0, 0, -1},
                           eng::Vec3{0, 1, 0});
            eng::audio::PlayDesc d;
            d.clip = &tone;
            d.spatial = true;
            d.position = eng::Vec3{0.0f, 0.0f, -dist};
            (void)mm.Play(d);
            mm.Render(buf.data(), kFrames);
            const float power = Rms(buf, 0) + Rms(buf, 1);
            std::printf("      %5.1f m: %.5f\n", dist, power);
            if (power > previous + 1e-6f) falls = false;
            previous = power;
        }
        Check(falls, "loudness falls monotonically with distance");
        Check(previous < 1e-5f, "and reaches silence past max_distance");
    }
    {
        // A sound to the listener's right is in the RIGHT ear -- and that
        // depends on which way the listener faces, which is why the listener
        // has an orientation and not merely a position.
        const eng::Vec3 east{5.0f, 0.0f, 0.0f};
        eng::audio::Mixer facing_north(kRate);
        facing_north.SetListener(eng::Vec3{0, 0, 0}, eng::Vec3{0, 0, -1},
                                 eng::Vec3{0, 1, 0});
        eng::audio::Mixer facing_south(kRate);
        facing_south.SetListener(eng::Vec3{0, 0, 0}, eng::Vec3{0, 0, 1},
                                 eng::Vec3{0, 1, 0});
        eng::audio::PlayDesc d;
        d.clip = &tone;
        d.spatial = true;
        d.position = east;
        (void)facing_north.Play(d);
        (void)facing_south.Play(d);
        std::vector<float> a(buf.size()), b(buf.size());
        facing_north.Render(a.data(), kFrames);
        facing_south.Render(b.data(), kFrames);
        std::printf("    a sound to the east: facing north l %.4f r %.4f, "
                    "facing south l %.4f r %.4f\n", Rms(a, 0), Rms(a, 1),
                    Rms(b, 0), Rms(b, 1));
        Check(Rms(a, 1) > Rms(a, 0), "is on the right when facing north");
        Check(Rms(b, 0) > Rms(b, 1), "and on the left when facing south");
    }

    // --- looping and finishing --------------------------------------------------
    {
        const eng::audio::Clip blip = Sine(1000.0f, 0.01f);  // 10 ms
        eng::audio::Mixer m(kRate);
        eng::audio::PlayDesc d;
        d.clip = &blip;
        const auto id = m.Play(d);
        // Render far longer than the clip. A one-shot must END, and free its
        // slot: a mixer that leaves finished voices active runs out of them and
        // then everything else is silent.
        m.Render(buf.data(), kFrames);
        std::printf("    a 10 ms one-shot after 250 ms: %d active voices\n",
                    m.ActiveVoices());
        Check(!m.Playing(id), "a one-shot finishes");
        Check(m.ActiveVoices() == 0, "and frees its voice");
    }
    {
        const eng::audio::Clip blip = Sine(1000.0f, 0.01f);
        eng::audio::Mixer m(kRate);
        eng::audio::PlayDesc d;
        d.clip = &blip;
        d.loop = true;
        const auto id = m.Play(d);
        m.Render(buf.data(), kFrames);
        Check(m.Playing(id), "a looping voice keeps going");
        // No CLICKS at the seam. A discontinuity between one sample and the
        // next is a click, and the loop point is where one is created by
        // resetting the cursor carelessly. 25 loops in this buffer, so any
        // seam artefact is in it.
        float worst = 0.0f;
        for (std::size_t i = 2; i < buf.size(); i += 2)
            worst = std::fmax(worst, std::fabs(buf[i] - buf[i - 2]));
        // A 1 kHz sine at 48 kHz steps at most 0.131 between samples.
        std::printf("    largest step between samples across 25 loops: %.4f "
                    "(a clean 1 kHz sine steps 0.131)\n", worst);
        Check(worst < 0.2f, "and does not click at the seam");
    }

    // --- voices ---------------------------------------------------------------------
    {
        eng::audio::Mixer m(kRate, 4);
        eng::audio::PlayDesc d;
        d.clip = &tone;
        d.loop = true;
        std::vector<eng::audio::VoiceId> ids;
        for (int i = 0; i < 6; ++i) ids.push_back(m.Play(d));
        std::printf("    six plays into four voices: %d active, %d starved\n",
                    m.ActiveVoices(), m.Starved());
        Check(m.ActiveVoices() == 4, "the voice count is a hard limit");
        Check(!ids[4].Valid() && !ids[5].Valid(), "and the excess is refused");
        // REFUSED, not stolen, and it says so. A silently dropped sound is
        // indistinguishable from one that was never triggered.
        Check(m.Starved() == 2, "and counted");

        // A STALE handle must not control whatever took its slot. Stop the
        // first voice, start another, and the old handle must not touch it.
        m.Stop(ids[0]);
        m.Render(buf.data(), 2048);  // let the fade finish
        const auto fresh = m.Play(d);
        Check(fresh.Valid(), "a freed slot is reused");
        m.Stop(ids[0]);  // the stale handle again
        m.Render(buf.data(), 2048);
        Check(m.Playing(fresh), "and a stale handle does not stop the new voice");
    }

    // --- stopping does not click ------------------------------------------------
    {
        eng::audio::Mixer m(kRate);
        eng::audio::PlayDesc d;
        d.clip = &tone;
        d.loop = true;
        const auto id = m.Play(d);
        m.Render(buf.data(), 4096);  // reach full gain
        m.Stop(id);
        std::vector<float> tail(std::size_t(4096) * 2);
        m.Render(tail.data(), 4096);
        float worst = 0.0f;
        for (std::size_t i = 2; i < tail.size(); i += 2)
            worst = std::fmax(worst, std::fabs(tail[i] - tail[i - 2]));
        std::printf("    stopping: largest step %.4f, ends at %.6f\n", worst,
                    std::fabs(tail[tail.size() - 2]));
        Check(worst < 0.2f, "a stopped voice fades rather than cutting");
        Check(std::fabs(tail[tail.size() - 2]) < 1e-4f, "and reaches silence");
        Check(!m.Playing(id), "and then frees itself");
    }

    // --- summing and clipping -----------------------------------------------------
    {
        eng::audio::Mixer m(kRate);
        eng::audio::PlayDesc d;
        d.clip = &tone;
        d.loop = true;
        for (int i = 0; i < 8; ++i) (void)m.Play(d);
        m.Render(buf.data(), 8192);
        std::vector<float> big(std::size_t(8192) * 2);
        m.Render(big.data(), 8192);
        float peak = 0.0f;
        for (float v : big) peak = std::fmax(peak, std::fabs(v));
        std::printf("    eight full-scale voices: output peak %.4f, reported "
                    "peak before clamping %.3f\n", peak, m.LastPeak());
        Check(peak <= 1.0f + 1e-5f, "the output never leaves -1..1");
        // ...and the mixer SAYS it clipped, which is the only way to find out
        // short of hearing the distortion.
        Check(m.LastPeak() > 1.5f, "and reports the peak it had to clamp");
    }

    std::printf(g_failures == 0 ? "\nmixer_test: all checks passed\n"
                                : "\nmixer_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
