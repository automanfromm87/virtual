// RIFF/WAVE decoding.
//
// The half of audio file support that every game needs and no game needs more
// of: a wav is uncompressed, so a decoder is a header parse and a format
// conversion, and short sounds -- footsteps, clicks, impacts -- are kept
// uncompressed anyway because decoding one at the moment it is triggered is a
// spike in the frame that produced it.
//
// Music wants a compressed format and streaming, which is a different feature
// with a different shape, and is not here.
#pragma once

#include <cstdint>
#include <span>
#include <string>

#include "engine/audio/clip.h"

namespace eng::audio {

// True if the bytes begin "RIFF....WAVE".
[[nodiscard]] bool IsWav(std::span<const std::uint8_t> bytes);

// Decodes PCM (8, 16, 24 and 32-bit integer) and IEEE float. Anything else --
// ADPCM, a-law, mp3 in a wav container -- is refused BY NAME rather than
// producing noise.
[[nodiscard]] Clip DecodeWav(std::span<const std::uint8_t> bytes,
                             std::string& error);
[[nodiscard]] Clip LoadWavFile(const std::string& path, std::string& error);

}  // namespace eng::audio
