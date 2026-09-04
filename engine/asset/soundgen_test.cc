// Procedural sound, checked for the things that make a clip unusable.
//
// "Does it sound like fire" is not testable and is not what breaks. What breaks
// is: a loop whose end does not meet its beginning, so there is a click every
// four seconds; a clip with a DC offset, which costs headroom on everything it
// is mixed with and thumps when the voice starts; a clip normalised to 1, which
// clips the master the moment a second one plays; a clip that is silent, which
// is indistinguishable from the sound never being triggered; and a generator
// that is not deterministic, so nothing about it can ever be compared. All five
// are arithmetic.
//
// The loop test is texgen's seam test in one dimension, and for the same
// reason: the wrap should be no more of a discontinuity than an ordinary step
// between neighbouring samples, so the measure is a RATIO against the typical
// step rather than a threshold on the gap.
#include "engine/asset/soundgen.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
constexpr int kRate = 44100;

void Check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

float Peak(const eng::audio::Clip& c) {
    float p = 0.0f;
    for (float v : c.samples) p = std::max(p, std::fabs(v));
    return p;
}

float Rms(const eng::audio::Clip& c) {
    if (c.samples.empty()) return 0.0f;
    double sq = 0.0;
    for (float v : c.samples) sq += double(v) * double(v);
    return float(std::sqrt(sq / double(c.samples.size())));
}

float Dc(const eng::audio::Clip& c) {
    if (c.samples.empty()) return 0.0f;
    double sum = 0.0;
    for (float v : c.samples) sum += v;
    return float(sum / double(c.samples.size()));
}

// THE LOOP TEST. The step from the last sample back to the first, against the
// typical step in the middle. 1 means the wrap is indistinguishable from any
// other sample boundary; a generator that does not close its loop scores in the
// tens, because the two ends are unrelated noise and differ by the full
// amplitude rather than by one step.
float WrapRatio(const eng::audio::Clip& c) {
    if (c.samples.size() < 64) return 0.0f;
    const std::size_t n = c.samples.size();
    const float wrap = std::fabs(c.samples[0] - c.samples[n - 1]);
    // The MEAN interior step, not the largest: the largest step in a noise
    // signal is itself nearly the full amplitude, so comparing against it would
    // pass any wrap at all.
    double sum = 0.0;
    for (std::size_t i = 1; i < n; ++i) sum += std::fabs(double(c.samples[i]) - c.samples[i - 1]);
    const float typical = float(sum / double(n - 1));
    return typical > 1e-9f ? wrap / typical : 0.0f;
}

// Fraction of the signal's energy below `hz`, by counting zero crossings
// against what white noise would give.
//
// A cheap spectral centroid. A proper FFT would say more, but the question
// here is only "is the filter doing anything", and a rate of zero crossings
// answers that: unfiltered noise crosses on about half of all sample
// boundaries, and every octave of low-pass roughly halves it.
float CrossingRate(const eng::audio::Clip& c) {
    if (c.samples.size() < 2) return 0.0f;
    int crossings = 0;
    for (std::size_t i = 1; i < c.samples.size(); ++i)
        if ((c.samples[i] >= 0.0f) != (c.samples[i - 1] >= 0.0f)) ++crossings;
    return float(crossings) / float(c.samples.size() - 1);
}

// The loudest short window, as a fraction of the whole clip's rms. A transient
// -- a footstep -- is loud briefly and quiet after; an ambience is not.
float CrestOverTime(const eng::audio::Clip& c, float window_seconds) {
    const int w = std::max(1, int(window_seconds * float(c.rate)));
    if (int(c.samples.size()) < w * 2) return 1.0f;
    float loudest = 0.0f;
    for (std::size_t start = 0; start + std::size_t(w) <= c.samples.size();
         start += std::size_t(w)) {
        double sq = 0.0;
        for (int i = 0; i < w; ++i) sq += double(c.samples[start + std::size_t(i)]) *
                                          double(c.samples[start + std::size_t(i)]);
        loudest = std::max(loudest, float(std::sqrt(sq / double(w))));
    }
    const float overall = Rms(c);
    return overall > 1e-9f ? loudest / overall : 1.0f;
}

}  // namespace

int main() {
    using namespace eng::soundgen;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    const eng::audio::Clip step = Footstep(kRate, 7);
    const eng::audio::Clip fire = Fire(kRate, 13);
    const eng::audio::Clip wind = Wind(kRate, 21);

    {
        std::printf("every clip is structurally valid\n");
        for (const auto& e : {std::pair{"footstep", &step}, std::pair{"fire", &fire},
                              std::pair{"wind", &wind}}) {
            std::printf("    %-9s %.2f s, peak %.3f, rms %.4f, dc %+.2e, crossings %.3f\n",
                        e.first, e.second->Seconds(), Peak(*e.second), Rms(*e.second),
                        Dc(*e.second), CrossingRate(*e.second));
            Check(e.second->Valid() && e.second->Frames() > 100, "it has samples and a rate");
            // A clip that is silent is indistinguishable from a sound that was
            // never triggered, which is the single most confusing audio bug
            // there is -- everything looks correct and nothing is heard.
            Check(Rms(*e.second) > 0.01f, "and it is not silent");
        }
    }

    {
        std::printf("\nnothing clips, and nothing wastes the headroom it leaves\n");
        for (const auto& e : {std::pair{"footstep", &step}, std::pair{"fire", &fire},
                              std::pair{"wind", &wind}}) {
            // These are MIXED, several at once, with distance attenuation over
            // the top. A clip peaking at 1 has no room for a second one and the
            // master clips the moment two things happen together -- audible as
            // a crack, and visible in no test that looks at one clip at a time.
            Check(Peak(*e.second) <= 0.8f,
                  (std::string(e.first) + " leaves headroom for the rest of the mix").c_str());
            // A DC offset costs that headroom back, silently, and thumps at the
            // start and end of every voice.
            Check(std::fabs(Dc(*e.second)) < 1e-3f,
                  (std::string("and ") + e.first + " has no DC offset").c_str());
        }
    }

    {
        std::printf("\nthe looping clips actually loop\n");
        // OVER SEVERAL SEEDS, because the ratio is computed from ONE pair of
        // samples and one pair can be lucky. Removing the crossfade entirely
        // made the fire score 26.6x -- caught -- and the wind 0.15x, passing,
        // because with that seed the clip's last sample happened to land near
        // zero all by itself. A check that a broken generator passes five times
        // out of six is not a check; six seeds makes the coincidence cost
        // nothing to rule out.
        for (int which = 0; which < 2; ++which) {
            float worst = 0.0f;
            std::uint32_t worst_seed = 0;
            for (std::uint32_t seed = 1; seed <= 6; ++seed) {
                const eng::audio::Clip c = which == 0 ? Fire(kRate, seed) : Wind(kRate, seed);
                const float ratio = WrapRatio(c);
                if (ratio > worst) {
                    worst = ratio;
                    worst_seed = seed;
                }
            }
            std::printf("    %-5s worst wrap over six seeds is %.2fx a normal sample "
                        "step (seed %u)\n", which == 0 ? "fire" : "wind", worst, worst_seed);
            // 4x. A clip that closes its loop scores near 1; one that does not
            // scores in the tens, because the two ends are unrelated noise and
            // the gap is the full amplitude rather than one step.
            Check(worst < 4.0f, which == 0 ? "fire meets its own beginning"
                                           : "wind meets its own beginning");
        }
        // The footstep is a ONE-SHOT and must not be forced to loop: a
        // cross-fade would eat its tail, which is the decay, and a footstep
        // without a decay is a click.
        Check(std::fabs(step.samples.back()) < 0.02f,
              "and the footstep ends in silence rather than being faded");
    }

    {
        std::printf("\nthe filters are doing something\n");
        // Unfiltered white noise crosses zero on about half of all sample
        // boundaries. Every octave of low-pass roughly halves that, so this is
        // the cheapest possible check that the cutoffs are connected to
        // anything -- a generator whose filter coefficient is accidentally 1
        // passes every other test here and outputs hiss.
        const float fire_x = CrossingRate(fire), wind_x = CrossingRate(wind);
        Check(fire_x < 0.06f, "the fire is low-frequency, as a fire is");
        Check(wind_x < 0.25f && wind_x > fire_x,
              "and the wind is broader than the fire but still not hiss");
        const eng::audio::Clip bright = Wind(kRate, 21, 6.0f, 1.0f);
        const eng::audio::Clip dark = Wind(kRate, 21, 6.0f, 0.0f);
        std::printf("    wind crossings: dark %.3f, default %.3f, bright %.3f\n",
                    CrossingRate(dark), wind_x, CrossingRate(bright));
        // A parameter that is accepted and ignored is worse than one that does
        // not exist, because the caller tunes against it and nothing changes.
        Check(CrossingRate(bright) > CrossingRate(dark) * 1.5f,
              "and `brightness` moves the filter rather than being ignored");
    }

    {
        std::printf("\na footstep is a transient and an ambience is not\n");
        // The shape over time, which is what separates the two kinds of clip.
        // A footstep whose energy is spread evenly is a burst of noise; an
        // ambience with all its energy in one window has a bang in it.
        const float step_crest = CrestOverTime(step, 0.02f);
        const float wind_crest = CrestOverTime(wind, 0.02f);
        const float fire_crest = CrestOverTime(fire, 0.02f);
        std::printf("    loudest 20 ms against the whole clip: footstep %.2fx, "
                    "fire %.2fx, wind %.2fx\n", step_crest, fire_crest, wind_crest);
        Check(step_crest > 1.4f, "the footstep's energy is concentrated at its start");
        Check(wind_crest < 2.0f, "and the wind has no bang in it");
        // The fire is BETWEEN the two on purpose: pops are transients, so a
        // fire with a perfectly even envelope has no pops and is a rumble.
        Check(fire_crest > 1.15f, "and the fire has audible individual pops");
    }

    {
        std::printf("\nseeds are honoured\n");
        Check(Footstep(kRate, 7).samples == step.samples,
              "the same seed gives a byte-identical clip");
        Check(Footstep(kRate, 8).samples != step.samples,
              "and a different seed gives a different one");
        // The reason there is a seed at all: one footstep replayed at a
        // constant interval is heard as a machine, however good the sample is.
        const eng::audio::Clip other = Footstep(kRate, 8);
        double diff = 0.0;
        const std::size_t n = std::min(step.samples.size(), other.samples.size());
        for (std::size_t i = 0; i < n; ++i)
            diff += std::fabs(double(step.samples[i]) - other.samples[i]);
        std::printf("    two steps differ by %.4f per sample on average\n",
                    diff / double(std::max<std::size_t>(1, n)));
        Check(diff / double(std::max<std::size_t>(1, n)) > 0.01,
              "and they differ audibly, not in the last bit");
    }

    {
        std::printf("\nedge cases do not crash or produce a broken clip\n");
        Check(Fire(kRate, 1, 0.05f).Valid(), "a fire shorter than its own crossfade is valid");
        Check(Wind(kRate, 1, 0.0f).Valid(), "and so is a wind of zero seconds");
        eng::audio::Clip tiny;
        tiny.rate = kRate;
        tiny.samples.assign(8, 0.5f);
        MakeLoopable(&tiny, 10.0f);
        Check(tiny.Valid() && !tiny.samples.empty(),
              "and a crossfade longer than the clip leaves something behind");
        MakeLoopable(nullptr, 0.1f);
        Check(true, "and a null clip is ignored rather than dereferenced");
        // Rates other than 44100 are not hypothetical: the device picks, and
        // the filters are derived from the rate precisely so this holds.
        const eng::audio::Clip at48 = Wind(48000, 21);
        std::printf("    wind crossings at 44.1 kHz %.3f, at 48 kHz %.3f\n",
                    CrossingRate(wind), CrossingRate(at48));
        // A filter with a hard-coded coefficient is a DIFFERENT filter at a
        // different rate, and the sound gets brighter on faster hardware --
        // which nobody would think to look for.
        Check(std::fabs(CrossingRate(at48) - CrossingRate(wind)) < 0.03f,
              "and the same wind at 48 kHz has the same spectrum");
    }

    std::printf(g_failures == 0 ? "\nsoundgen_test: all checks passed\n"
                                : "\nsoundgen_test: %d FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
