// Decoding compressed audio, through whatever the OS already knows.
//
// The same call as CoreText for fonts and Metal for pixels: FLAC, mp3, m4a,
// aac, caf and wav all decode through AudioToolbox, which is present, correct
// and better tested than anything that could be written here. A FLAC decoder
// alone is Rice coding and LPC prediction -- a week of work to be worse at a
// solved problem.
//
// engine/audio's own wav decoder stays, and is not redundant: it is pure C++,
// testable without a machine that has audio, and does not pay a system decode
// for a twenty-millisecond footstep.
#pragma once

#include <string>

#include "engine/audio/clip.h"

namespace eng::platform {

// Decodes an entire file into memory as float samples at its native rate.
//
// ENTIRELY, which is the limitation to know about: twenty seconds of stereo is
// about eight megabytes, and an hour is not something to hold. Streaming is a
// different feature with a different shape and is not here.
[[nodiscard]] audio::Clip DecodeAudioFile(const std::string& path,
                                          std::string& error);

}  // namespace eng::platform
