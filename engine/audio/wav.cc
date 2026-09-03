#include "engine/audio/wav.h"

#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

namespace eng::audio {
namespace {

std::uint32_t U32(std::span<const std::uint8_t> b, std::size_t at) {
    return std::uint32_t(b[at]) | (std::uint32_t(b[at + 1]) << 8) |
           (std::uint32_t(b[at + 2]) << 16) | (std::uint32_t(b[at + 3]) << 24);
}
std::uint16_t U16(std::span<const std::uint8_t> b, std::size_t at) {
    return std::uint16_t(std::uint32_t(b[at]) | (std::uint32_t(b[at + 1]) << 8));
}
bool Tag(std::span<const std::uint8_t> b, std::size_t at, const char* what) {
    return at + 4 <= b.size() && std::memcmp(b.data() + at, what, 4) == 0;
}

}  // namespace

bool IsWav(std::span<const std::uint8_t> b) {
    return b.size() >= 12 && Tag(b, 0, "RIFF") && Tag(b, 8, "WAVE");
}

Clip DecodeWav(std::span<const std::uint8_t> bytes, std::string& error) {
    Clip clip;
    error.clear();
    if (!IsWav(bytes)) {
        error = "wav: not a RIFF/WAVE file";
        return clip;
    }

    int format = 0, channels = 0, rate = 0, bits = 0;
    std::span<const std::uint8_t> data;
    bool have_fmt = false;

    // Chunk walk. A wav is a sequence of tagged chunks and the interesting two
    // are not required to be adjacent or in order -- editors leave LIST, fact
    // and cue chunks between them, and a decoder that assumes fmt is followed
    // by data reads a comment as audio.
    std::size_t p = 12;
    while (p + 8 <= bytes.size()) {
        const std::uint32_t size = U32(bytes, p + 4);
        const std::size_t body = p + 8;
        if (body + size > bytes.size()) {
            // A truncated final chunk is common in files cut short by a crash;
            // take what is there rather than refusing the whole file.
            if (Tag(bytes, p, "data") && body < bytes.size()) {
                data = bytes.subspan(body, bytes.size() - body);
                break;
            }
            error = "wav: a chunk runs past the end of the file";
            return Clip{};
        }
        if (Tag(bytes, p, "fmt ")) {
            if (size < 16) {
                error = "wav: the format chunk is too short";
                return Clip{};
            }
            format = U16(bytes, body);
            channels = U16(bytes, body + 2);
            rate = int(U32(bytes, body + 4));
            bits = U16(bytes, body + 14);
            // WAVE_FORMAT_EXTENSIBLE hides the real format in a GUID whose
            // first two bytes are the format tag it is standing in for. Files
            // from a modern recorder are almost all this, and a decoder that
            // only knows tag 1 rejects all of them.
            if (format == 0xFFFE && size >= 40) format = U16(bytes, body + 24);
            have_fmt = true;
        } else if (Tag(bytes, p, "data")) {
            data = bytes.subspan(body, size);
        }
        // Chunks are PADDED to an even length, and the pad byte is not counted
        // in the size. Ignoring it desynchronises the walk on the first
        // odd-sized chunk, and every chunk after it is read as garbage.
        p = body + size + (size & 1);
    }

    if (!have_fmt) {
        error = "wav: no format chunk";
        return Clip{};
    }
    if (data.empty()) {
        error = "wav: no data chunk";
        return Clip{};
    }
    if (channels < 1 || channels > 8) {
        error = "wav: " + std::to_string(channels) + " channels is out of range";
        return Clip{};
    }
    if (rate < 1000 || rate > 384000) {
        error = "wav: a sample rate of " + std::to_string(rate) + " is implausible";
        return Clip{};
    }

    constexpr int kPcm = 1, kFloat = 3;
    if (format != kPcm && format != kFloat) {
        error = "wav: format " + std::to_string(format) +
                " is not PCM or float (compressed wavs are not supported)";
        return Clip{};
    }

    const int bytes_per = bits / 8;
    if (bytes_per < 1 || bits % 8 != 0) {
        error = "wav: " + std::to_string(bits) + " bits per sample";
        return Clip{};
    }
    const std::size_t count = data.size() / std::size_t(bytes_per);

    clip.channels = channels;
    clip.rate = rate;
    clip.samples.resize(count - count % std::size_t(channels));

    for (std::size_t i = 0; i < clip.samples.size(); ++i) {
        const std::size_t at = i * std::size_t(bytes_per);
        if (format == kFloat) {
            if (bits == 32) {
                float v;
                std::memcpy(&v, data.data() + at, 4);
                clip.samples[i] = v;
            } else {
                double v;
                std::memcpy(&v, data.data() + at, 8);
                clip.samples[i] = float(v);
            }
            continue;
        }
        switch (bits) {
            case 8:
                // 8-bit wav is UNSIGNED, alone among the integer widths. Read
                // as signed it comes out as a loud buzz with the waveform
                // folded around zero.
                clip.samples[i] = (float(data[at]) - 128.0f) / 128.0f;
                break;
            case 16: {
                const std::int16_t v =
                    std::int16_t(std::uint16_t(data[at]) |
                                 (std::uint16_t(data[at + 1]) << 8));
                clip.samples[i] = float(v) / 32768.0f;
                break;
            }
            case 24: {
                // Sign-extended by hand: there is no 24-bit integer to load it
                // into, and shifting an unsigned value left into the top byte
                // and back down arithmetically is the shortest correct way.
                const std::int32_t raw =
                    std::int32_t((std::uint32_t(data[at]) << 8) |
                                 (std::uint32_t(data[at + 1]) << 16) |
                                 (std::uint32_t(data[at + 2]) << 24));
                clip.samples[i] = float(raw >> 8) / 8388608.0f;
                break;
            }
            case 32: {
                const std::int32_t v = std::int32_t(U32(data, at));
                clip.samples[i] = float(v) / 2147483648.0f;
                break;
            }
            default:
                error = "wav: " + std::to_string(bits) + "-bit PCM";
                return Clip{};
        }
    }
    return clip;
}

Clip LoadWavFile(const std::string& path, std::string& error) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        error = "wav: cannot open " + path;
        return Clip{};
    }
    const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(f),
                                          std::istreambuf_iterator<char>()};
    return DecodeWav(bytes, error);
}

}  // namespace eng::audio
