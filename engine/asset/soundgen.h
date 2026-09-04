// Pure C++20. Procedural sound: footsteps, fire, wind, computed rather than
// recorded.
//
// WHY. The same argument as texgen, and it lands harder here. A forest that
// makes no sound is not quiet, it is broken -- silence is the one thing a real
// place never is, and the absence reads as the world not being switched on
// rather than as a missing feature. But shipping audio means shipping a decoder
// and a few megabytes of somebody's field recording, and neither of those is
// this engine.
//
// A footstep is a filtered noise burst with a fast decay. Wind is filtered
// noise with a slow one. Fire is a low rumble with pops on top. None of them is
// more than a hundred lines of arithmetic, and on a surface -- which is to say
// coming out of a speaker, behind everything else -- the generated version is
// not the part anyone notices.
//
// EVERYTHING THAT LOOPS, LOOPS SEAMLESSLY. This is the audio version of
// texgen's tiling rule and it fails the same way: an ambience whose end does
// not meet its beginning produces a click once per loop, and a click every four
// seconds is far more objectionable than no ambience at all. The generators
// below cross-fade their own tail over their head, and soundgen_test measures
// the discontinuity at the wrap against the discontinuity between neighbouring
// samples in the middle -- a ratio, for the same reason the texture seam test
// is a ratio.
//
// AMPLITUDES ARE CONSERVATIVE. These are mixed together, several at a time,
// with distance attenuation on top. A clip normalised to peak at 1 leaves no
// headroom for a second one, and the result is a mix that clips whenever two
// things happen at once -- which is audible as distortion and shows up in no
// visual test anywhere.
#pragma once

#include <cstdint>
#include <vector>

#include "engine/audio/clip.h"

namespace eng::soundgen {

// A footstep on soft ground. `softness` runs from 0, a hard surface with a
// sharp transient, to 1, deep grass that is nearly all rustle.
//
// SEEDED SO THAT NO TWO STEPS ARE THE SAME. A single footstep sample replayed
// at a constant interval is instantly recognisable as a machine, and the ear
// picks it out even under everything else in the mix -- so a caller generates a
// handful with different seeds and rotates through them.
[[nodiscard]] audio::Clip Footstep(int rate, std::uint32_t seed, float softness = 0.7f);

// A fire, looping. A low rumble with pops scattered over it.
//
// The pops are the whole thing. A rumble alone is a washing machine; what makes
// a fire is that it is irregular, and irregular at a rate slow enough to hear
// individual events. `pops_per_second` is that rate.
[[nodiscard]] audio::Clip Fire(int rate, std::uint32_t seed, float seconds = 4.0f,
                               float pops_per_second = 11.0f);

// Wind through a canopy, looping. Filtered noise whose loudness drifts, because
// wind that does not gust is a fan.
//
// `brightness` moves the filter: low is a distant roar through heavy leaves,
// high is a hiss through bare branches.
[[nodiscard]] audio::Clip Wind(int rate, std::uint32_t seed, float seconds = 6.0f,
                               float brightness = 0.35f);

// Cross-fades a clip's tail over its head so it can be looped.
//
// Exposed rather than kept private because it is the one operation every
// looping generator needs and the one most likely to be got wrong: the fade
// must be long enough to hide a discontinuity but short enough not to audibly
// double the material, and the samples consumed by it come OFF THE END, so the
// returned clip is shorter than what went in. A caller that forgets that gets a
// loop whose period is not what it asked for.
void MakeLoopable(audio::Clip*, float fade_seconds = 0.25f);

}  // namespace eng::soundgen
