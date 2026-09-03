// Decoded audio, in the one format everything downstream works in.
#pragma once

#include <cstdint>
#include <vector>

namespace eng::audio {

// FLOAT samples, interleaved, in the range -1 to 1.
//
// Not the 16-bit integers a wav file holds. Every mix, every gain change and
// every resample is a multiply, and doing those in fixed point means deciding
// where to put the binary point and rounding at each step. The memory cost is
// two bytes a sample; a minute of stereo is ten megabytes either way and
// nobody keeps a minute of stereo in memory.
struct Clip {
    int channels = 1;
    int rate = 44100;  // samples per second, per channel
    std::vector<float> samples;

    [[nodiscard]] bool Valid() const {
        return channels > 0 && rate > 0 && !samples.empty() &&
               samples.size() % std::size_t(channels) == 0;
    }
    [[nodiscard]] std::size_t Frames() const {
        return channels > 0 ? samples.size() / std::size_t(channels) : 0;
    }
    [[nodiscard]] double Seconds() const {
        return rate > 0 ? double(Frames()) / double(rate) : 0.0;
    }
};

}  // namespace eng::audio
