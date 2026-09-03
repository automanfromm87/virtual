// WAV decoding, against files built byte by byte.
//
// The fixtures are assembled here rather than checked in, because the whole
// point is the BYTES: a decoder is a header parse, and the failures are an
// unsigned format read as signed, a chunk walk that loses a pad byte, or a
// bit depth converted with the wrong divisor. Each of those produces audio --
// just the wrong audio -- so every check below is on sample values.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "engine/audio/wav.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

void Put32(std::vector<std::uint8_t>& b, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(std::uint8_t(v >> (i * 8)));
}
void Put16(std::vector<std::uint8_t>& b, std::uint16_t v) {
    for (int i = 0; i < 2; ++i) b.push_back(std::uint8_t(v >> (i * 8)));
}
void Tag(std::vector<std::uint8_t>& b, const char* t) {
    for (int i = 0; i < 4; ++i) b.push_back(std::uint8_t(t[i]));
}

// A whole wav around a block of sample bytes.
std::vector<std::uint8_t> Wrap(int format, int channels, int rate, int bits,
                               const std::vector<std::uint8_t>& data,
                               bool extra_chunk = false, bool odd_chunk = false) {
    std::vector<std::uint8_t> b;
    Tag(b, "RIFF");
    Put32(b, 0);  // patched below
    Tag(b, "WAVE");
    if (extra_chunk) {
        // A chunk between fmt and data, which real editors write and a decoder
        // that assumes they are adjacent reads as audio.
        Tag(b, "LIST");
        const std::uint32_t n = odd_chunk ? 5u : 4u;
        Put32(b, n);
        for (std::uint32_t i = 0; i < n; ++i) b.push_back('x');
        if (odd_chunk) b.push_back(0);  // the pad byte, not counted in the size
    }
    Tag(b, "fmt ");
    Put32(b, 16);
    Put16(b, std::uint16_t(format));
    Put16(b, std::uint16_t(channels));
    Put32(b, std::uint32_t(rate));
    Put32(b, std::uint32_t(rate * channels * bits / 8));  // byte rate
    Put16(b, std::uint16_t(channels * bits / 8));         // block align
    Put16(b, std::uint16_t(bits));
    Tag(b, "data");
    Put32(b, std::uint32_t(data.size()));
    b.insert(b.end(), data.begin(), data.end());
    const std::uint32_t riff = std::uint32_t(b.size() - 8);
    for (int i = 0; i < 4; ++i) b[4 + i] = std::uint8_t(riff >> (i * 8));
    return b;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("wav decoding\n");

    // --- 16-bit PCM, the common case -----------------------------------------
    {
        std::vector<std::uint8_t> d;
        // Full scale positive, full scale negative, silence, half scale.
        const std::int16_t vals[4] = {32767, -32768, 0, 16384};
        for (std::int16_t v : vals) Put16(d, std::uint16_t(v));
        std::string e;
        const auto c = eng::audio::DecodeWav(Wrap(1, 1, 44100, 16, d), e);
        Check(e.empty() && c.Valid(), "a 16-bit mono wav decodes");
        Check(c.channels == 1 && c.rate == 44100, "with the right rate and channels");
        Check(c.Frames() == 4, "and the right number of frames");
        std::printf("    samples %.5f %.5f %.5f %.5f\n", c.samples[0], c.samples[1],
                    c.samples[2], c.samples[3]);
        // Divided by 32768, not 32767: the range is asymmetric, and dividing by
        // the positive maximum makes the most negative sample exceed -1 and
        // clip on the way out.
        Check(std::fabs(c.samples[0] - 32767.0f / 32768.0f) < 1e-6f, "+full scale");
        Check(std::fabs(c.samples[1] + 1.0f) < 1e-6f, "-full scale is exactly -1");
        Check(c.samples[2] == 0.0f, "silence is exactly zero");
        Check(std::fabs(c.samples[3] - 0.5f) < 1e-6f, "half scale is exactly 0.5");
    }

    // --- 8-bit is UNSIGNED, alone among the integer widths --------------------
    {
        const std::vector<std::uint8_t> d = {128, 255, 0, 192};
        std::string e;
        const auto c = eng::audio::DecodeWav(Wrap(1, 1, 22050, 8, d), e);
        Check(c.Valid(), "an 8-bit wav decodes");
        std::printf("    8-bit: %.4f %.4f %.4f %.4f\n", c.samples[0], c.samples[1],
                    c.samples[2], c.samples[3]);
        // 128 is SILENCE. Read as signed it is -1, and the whole file comes out
        // as a loud buzz with the waveform folded around zero.
        Check(std::fabs(c.samples[0]) < 1e-6f, "128 is silence, not full scale");
        Check(c.samples[1] > 0.99f, "255 is the positive peak");
        Check(std::fabs(c.samples[2] + 1.0f) < 1e-6f, "0 is the negative peak");
        Check(c.samples[3] > 0.4f && c.samples[3] < 0.6f, "192 is half way up");
    }

    // --- 24-bit, where sign extension has to be done by hand ------------------
    {
        std::vector<std::uint8_t> d;
        const std::int32_t vals[3] = {0x7FFFFF, -0x800000, -1};
        for (std::int32_t v : vals) {
            const std::uint32_t u = std::uint32_t(v);
            d.push_back(std::uint8_t(u));
            d.push_back(std::uint8_t(u >> 8));
            d.push_back(std::uint8_t(u >> 16));
        }
        std::string e;
        const auto c = eng::audio::DecodeWav(Wrap(1, 1, 48000, 24, d), e);
        Check(c.Valid() && c.Frames() == 3, "a 24-bit wav decodes");
        std::printf("    24-bit: %.6f %.6f %.6f\n", c.samples[0], c.samples[1],
                    c.samples[2]);
        Check(c.samples[0] > 0.999f, "+full scale");
        Check(std::fabs(c.samples[1] + 1.0f) < 1e-6f, "-full scale");
        // Minus one is the value that catches a missing sign extension: read
        // unsigned it becomes almost +1 rather than almost 0.
        Check(std::fabs(c.samples[2]) < 1e-5f, "and -1 is very nearly silence");
    }

    // --- 32-bit float ----------------------------------------------------------
    {
        std::vector<std::uint8_t> d;
        for (float v : {1.0f, -1.0f, 0.25f}) {
            std::uint32_t u;
            std::memcpy(&u, &v, 4);
            Put32(d, u);
        }
        std::string e;
        const auto c = eng::audio::DecodeWav(Wrap(3, 1, 48000, 32, d), e);
        Check(c.Valid(), "a float wav decodes");
        Check(c.samples[0] == 1.0f && c.samples[1] == -1.0f &&
                  c.samples[2] == 0.25f,
              "and its samples pass through untouched");
    }

    // --- stereo interleaving ----------------------------------------------------
    {
        std::vector<std::uint8_t> d;
        // L R L R, distinct so a swapped or dropped channel is visible.
        for (std::int16_t v : {std::int16_t(16384), std::int16_t(-16384),
                               std::int16_t(8192), std::int16_t(-8192)})
            Put16(d, std::uint16_t(v));
        std::string e;
        const auto c = eng::audio::DecodeWav(Wrap(1, 2, 44100, 16, d), e);
        Check(c.Valid() && c.channels == 2, "a stereo wav decodes");
        Check(c.Frames() == 2, "two frames, not four");
        Check(c.samples[0] > 0 && c.samples[1] < 0 && c.samples[2] > 0 &&
                  c.samples[3] < 0,
              "and stays interleaved in the order it was written");
        Check(std::fabs(c.Seconds() - 2.0 / 44100.0) < 1e-9, "the duration is right");
    }

    // --- the chunk walk -----------------------------------------------------------
    //
    // Two ways it goes wrong, both silent. A decoder that assumes data follows
    // fmt reads a LIST chunk as audio; one that forgets chunks are padded to an
    // even length desynchronises on the first odd-sized chunk and reads
    // everything after it as garbage.
    {
        std::vector<std::uint8_t> d;
        for (std::int16_t v : {std::int16_t(16384), std::int16_t(-16384)})
            Put16(d, std::uint16_t(v));
        std::string e1, e2;
        const auto even = eng::audio::DecodeWav(Wrap(1, 1, 44100, 16, d, true), e1);
        const auto odd = eng::audio::DecodeWav(Wrap(1, 1, 44100, 16, d, true, true), e2);
        Check(even.Valid() && even.Frames() == 2,
              "a chunk between fmt and data is skipped");
        Check(std::fabs(even.samples[0] - 0.5f) < 1e-4f, "and the audio is the audio");
        Check(odd.Valid() && odd.Frames() == 2,
              "an ODD-sized chunk is skipped with its pad byte");
        Check(std::fabs(odd.samples[0] - 0.5f) < 1e-4f,
              "and does not shift everything after it");
    }

    // --- refusals -------------------------------------------------------------
    //
    // Each names what it was. A decoder that returns silence for a compressed
    // file sends whoever hits it looking for a broken speaker.
    {
        std::string e;
        const std::uint8_t junk[16] = {'N', 'O', 'P', 'E'};
        Check(eng::audio::DecodeWav(junk, e).samples.empty() &&
                  e.find("RIFF") != std::string::npos,
              "a non-wav is refused by name");

        e.clear();
        std::vector<std::uint8_t> d(64, 0);
        // Format 17 is IMA ADPCM: a real format, compressed, and decoding its
        // bytes as PCM produces a convincing hiss.
        Check(eng::audio::DecodeWav(Wrap(17, 1, 44100, 4, d), e).samples.empty() &&
                  e.find("not PCM") != std::string::npos,
              "a compressed wav is refused, and says so");

        e.clear();
        Check(eng::audio::DecodeWav(Wrap(1, 0, 44100, 16, d), e).samples.empty() &&
                  e.find("channels") != std::string::npos,
              "zero channels is refused");

        e.clear();
        Check(eng::audio::DecodeWav(Wrap(1, 1, 3, 16, d), e).samples.empty() &&
                  e.find("rate") != std::string::npos,
              "an implausible sample rate is refused");

        e.clear();
        std::vector<std::uint8_t> no_data;
        Tag(no_data, "RIFF");
        Put32(no_data, 4);
        Tag(no_data, "WAVE");
        Check(eng::audio::DecodeWav(no_data, e).samples.empty() && !e.empty(),
              "a wav with no chunks at all is refused");
    }

    // --- WAVE_FORMAT_EXTENSIBLE -------------------------------------------------
    //
    // Nearly everything a modern recorder writes. The real format tag hides in
    // the first two bytes of a GUID, and a decoder that only knows tag 1
    // rejects the lot.
    {
        std::vector<std::uint8_t> b;
        Tag(b, "RIFF");
        Put32(b, 0);
        Tag(b, "WAVE");
        Tag(b, "fmt ");
        Put32(b, 40);
        Put16(b, 0xFFFE);       // extensible
        Put16(b, 1);            // channels
        Put32(b, 44100);
        Put32(b, 88200);
        Put16(b, 2);
        Put16(b, 16);           // bits
        Put16(b, 22);           // cbSize
        Put16(b, 16);           // valid bits
        Put32(b, 4);            // channel mask
        Put16(b, 1);            // the GUID's first two bytes: PCM
        for (int i = 0; i < 14; ++i) b.push_back(0);
        Tag(b, "data");
        Put32(b, 4);
        Put16(b, std::uint16_t(std::int16_t(16384)));
        Put16(b, std::uint16_t(std::int16_t(-16384)));
        const std::uint32_t riff = std::uint32_t(b.size() - 8);
        for (int i = 0; i < 4; ++i) b[4 + i] = std::uint8_t(riff >> (i * 8));

        std::string e;
        const auto c = eng::audio::DecodeWav(b, e);
        Check(c.Valid() && c.Frames() == 2, "an extensible-format wav decodes");
        Check(std::fabs(c.samples[0] - 0.5f) < 1e-4f, "as the PCM it says it is");
    }

    std::printf(g_failures == 0 ? "\nwav_test: all checks passed\n"
                                : "\nwav_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
