#include "engine/asset/soundgen.h"

#include <algorithm>
#include <cmath>

namespace eng::soundgen {
namespace {

constexpr float kTau = 6.28318530718f;

// The same integer hash the texture generator uses, for the same reason: a
// generator that is not deterministic cannot be regression-tested, and one that
// depends on <random>'s engine state cannot be sampled out of order.
struct Rng {
    std::uint32_t state;
    explicit Rng(std::uint32_t seed) : state(seed * 747796405u + 2891336453u) {}
    std::uint32_t Next() {
        state = state * 747796405u + 2891336453u;
        std::uint32_t w = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (w >> 22u) ^ w;
    }
    // Uniform in [-1, 1): white noise.
    float Noise() { return float(Next()) * (2.0f / 4294967296.0f) - 1.0f; }
    float Unit() { return float(Next()) * (1.0f / 4294967296.0f); }
};

// A one-pole low pass. The coefficient is derived from the cutoff rather than
// picked, so a generator reads in hertz and stays right when the device
// resamples: a hard-coded coefficient is a different filter at 48 kHz than at
// 44.1, and the sound gets brighter on hardware that happens to run faster.
struct LowPass {
    float y = 0.0f, a = 0.0f;
    LowPass(float cutoff_hz, int rate)
        : a(1.0f - std::exp(-kTau * cutoff_hz / float(rate))) {}
    float operator()(float x) {
        y += a * (x - y);
        return y;
    }
};

// Two poles, for a steeper slope. One pole is 6 dB per octave, which leaves so
// much of the noise above the cutoff that the filter is barely audible.
struct LowPass2 {
    LowPass a, b;
    LowPass2(float cutoff_hz, int rate) : a(cutoff_hz, rate), b(cutoff_hz, rate) {}
    float operator()(float x) { return b(a(x)); }
};

// A high pass built as "everything the low pass rejected". Cheap and exact.
struct HighPass {
    LowPass lp;
    HighPass(float cutoff_hz, int rate) : lp(cutoff_hz, rate) {}
    float operator()(float x) { return x - lp(x); }
};

// Peak-normalise to `target`, then leave it alone.
//
// NOT to 1. These get mixed several at a time with distance attenuation on top,
// and a clip peaking at 1 has no room for a second one -- two footsteps at once
// clip the master, which is heard as a crack and looks like nothing at all.
void Normalise(std::vector<float>& s, float target) {
    float peak = 0.0f;
    for (float v : s) peak = std::max(peak, std::fabs(v));
    if (peak < 1e-6f) return;
    const float g = target / peak;
    for (float& v : s) v *= g;
}

// Remove any constant offset.
//
// A DC offset is inaudible on its own and costs headroom on everything: a clip
// sitting at +0.2 has 20% less room before clipping, and it thumps when the
// voice starts and stops because the signal steps from zero and back.
void RemoveDc(std::vector<float>& s) {
    if (s.empty()) return;
    double sum = 0.0;
    for (float v : s) sum += v;
    const float mean = float(sum / double(s.size()));
    for (float& v : s) v -= mean;
}

}  // namespace

void MakeLoopable(audio::Clip* clip, float fade_seconds) {
    if (clip == nullptr || clip->rate <= 0) return;
    const int n = int(clip->samples.size());
    int fade = int(fade_seconds * float(clip->rate));
    // The fade eats the tail, so it cannot be longer than half the clip -- at
    // exactly half, every sample is a blend of two and the material is audibly
    // doubled.
    fade = std::clamp(fade, 1, std::max(1, n / 3));
    if (n <= fade * 2) return;
    const int kept = n - fade;
    for (int i = 0; i < fade; ++i) {
        // EQUAL POWER, not linear. Two uncorrelated noise signals crossfaded
        // linearly dip by 3 dB in the middle, and a dip once per loop is as
        // audible as the click it was meant to remove.
        const float t = float(i) / float(fade);
        const float a = std::cos(t * kTau * 0.25f), b = std::sin(t * kTau * 0.25f);
        clip->samples[std::size_t(i)] =
            clip->samples[std::size_t(i)] * b + clip->samples[std::size_t(kept + i)] * a;
    }
    clip->samples.resize(std::size_t(kept));
}

audio::Clip Footstep(int rate, std::uint32_t seed, float softness) {
    audio::Clip c;
    c.channels = 1;
    c.rate = rate;
    softness = std::clamp(softness, 0.0f, 1.0f);
    // Soft ground rings for longer -- grass keeps rustling after the weight has
    // landed, gravel does not.
    const float seconds = 0.09f + softness * 0.20f;
    const int n = int(seconds * float(rate));
    c.samples.resize(std::size_t(n));

    Rng rng(seed ^ 0x5bd1u);
    // The rustle: mid-band noise, brighter on hard ground.
    LowPass2 body(1400.0f + (1.0f - softness) * 3400.0f, rate);
    HighPass thin(180.0f, rate);
    // The impact: a short low thump, which is what makes it a footstep rather
    // than a brush. Almost gone on soft ground.
    LowPass2 thump(110.0f, rate);
    // A little variation per step, so a rotation of a few clips does not sound
    // like a rotation of a few clips.
    const float rate_jitter = 0.85f + rng.Unit() * 0.3f;

    for (int i = 0; i < n; ++i) {
        const float t = float(i) / float(rate);
        // A FAST ATTACK AND AN EXPONENTIAL DECAY. A linear decay sounds like a
        // sound being turned down; everything that is struck decays
        // exponentially, and the ear knows the difference immediately.
        const float attack = 1.0f - std::exp(-t * 900.0f);
        const float decay = std::exp(-t * (34.0f - softness * 21.0f) * rate_jitter);
        const float rustle = thin(body(rng.Noise())) * (0.55f + softness * 0.45f);
        const float hit = thump(rng.Noise()) * std::exp(-t * 90.0f) * (1.0f - softness * 0.65f);
        c.samples[std::size_t(i)] = (rustle + hit * 2.2f) * attack * decay;
    }
    RemoveDc(c.samples);
    Normalise(c.samples, 0.62f);
    return c;
}

audio::Clip Fire(int rate, std::uint32_t seed, float seconds, float pops_per_second) {
    audio::Clip c;
    c.channels = 1;
    c.rate = rate;
    const int n = int(std::max(0.5f, seconds) * float(rate));
    c.samples.assign(std::size_t(n), 0.0f);

    Rng rng(seed ^ 0x9e37u);
    // The bed: brown-ish noise, well below anything you could call a pitch.
    LowPass2 rumble(95.0f, rate);
    LowPass breath(2.4f, rate);  // slow swell, so the bed is not a constant
    for (int i = 0; i < n; ++i) {
        const float swell = 0.6f + 0.5f * breath(rng.Noise());
        c.samples[std::size_t(i)] = rumble(rng.Noise()) * 3.2f * swell;
    }

    // THE POPS ARE THE FIRE. A rumble on its own is a washing machine; what
    // makes it read as burning is that it is irregular at a rate slow enough
    // for individual events to be heard. Scheduled as a Poisson process --
    // gaps drawn from an exponential -- because evenly spaced pops sound
    // mechanical no matter how they are shaped.
    const float mean_gap = float(rate) / std::max(0.5f, pops_per_second);
    float at = 0.0f;
    while (at < float(n)) {
        const int start = int(at);
        const float loud = 0.25f + rng.Unit() * rng.Unit() * 1.75f;  // mostly small
        const float pitch = 700.0f + rng.Unit() * 2600.0f;
        const float decay = 60.0f + rng.Unit() * 340.0f;
        LowPass2 crack(pitch, rate);
        HighPass edge(320.0f, rate);
        const int len = std::min(n - start, int(0.09f * float(rate)));
        for (int i = 0; i < len; ++i) {
            const float t = float(i) / float(rate);
            c.samples[std::size_t(start + i)] +=
                edge(crack(rng.Noise())) * std::exp(-t * decay) * loud * 3.0f;
        }
        // Exponential gap: -ln(U) has mean 1.
        at += mean_gap * -std::log(std::max(1e-6f, rng.Unit()));
    }

    RemoveDc(c.samples);
    Normalise(c.samples, 0.55f);
    MakeLoopable(&c, 0.35f);
    return c;
}

audio::Clip Wind(int rate, std::uint32_t seed, float seconds, float brightness) {
    audio::Clip c;
    c.channels = 1;
    c.rate = rate;
    brightness = std::clamp(brightness, 0.0f, 1.0f);
    const int n = int(std::max(1.0f, seconds) * float(rate));
    c.samples.resize(std::size_t(n));

    Rng rng(seed ^ 0x2c1au);
    LowPass2 leaves(280.0f + brightness * 2600.0f, rate);
    HighPass body(70.0f, rate);
    for (int i = 0; i < n; ++i) {
        const float t = float(i) / float(rate);
        // GUSTS, from two incommensurate rates. Wind at a constant level is a
        // fan, and a single sine is a siren -- two periods that do not divide
        // each other never repeat within the loop, so the drift does not
        // announce the clip's length.
        const float gust = 0.42f + 0.34f * std::sin(kTau * 0.083f * t) +
                           0.24f * std::sin(kTau * 0.191f * t + 1.7f);
        c.samples[std::size_t(i)] = body(leaves(rng.Noise())) * gust;
    }
    RemoveDc(c.samples);
    Normalise(c.samples, 0.45f);
    MakeLoopable(&c, 0.6f);
    return c;
}

}  // namespace eng::soundgen
