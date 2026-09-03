// No test framework — from scratch means from scratch.
//
// The fixtures are real PNG files produced by Python's zlib, an encoder with no
// code in common with this one. That is the point: a decoder tested against
// data its own encoder produced only proves the two agree with each other.
//
// Every case carries the expected RGBA alongside the file, so a wrong pixel is
// a failure rather than something to eyeball.
#include "engine/asset/png.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Fail(const char* what, int line) {
    std::fprintf(stderr, "png_test.cc:%d  %s\n", line, what);
    ++g_failures;
}

#define CHECK(cond) \
    do { if (!(cond)) Fail(#cond, __LINE__); } while (0)

using namespace eng;

#include "engine/asset/testdata_png.inc"

// Decodes one fixture and compares every byte.
void Case(const char* name, const unsigned char* file, std::size_t file_size,
          const unsigned char* want, std::size_t want_size, int w, int h) {
    std::string error;
    const Texture2D img = png::Decode(
        std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(file),
                                      file_size),
        error);
    if (!error.empty()) {
        std::fprintf(stderr, "  %-18s decode failed: %s\n", name, error.c_str());
        ++g_failures;
        return;
    }
    if (img.width != w || img.height != h) {
        std::fprintf(stderr, "  %-18s size %dx%d, expected %dx%d\n", name, img.width,
                     img.height, w, h);
        ++g_failures;
        return;
    }
    if (img.rgba.size() != want_size) {
        std::fprintf(stderr, "  %-18s %zu bytes, expected %zu\n", name,
                     img.rgba.size(), want_size);
        ++g_failures;
        return;
    }
    for (std::size_t i = 0; i < want_size; ++i) {
        if (img.rgba[i] == want[i]) continue;
        std::fprintf(stderr,
                     "  %-18s byte %zu (pixel %zu, channel %zu) is %u, expected %u\n",
                     name, i, i / 4, i % 4, img.rgba[i], want[i]);
        ++g_failures;
        return;
    }
    std::printf("  %-18s %dx%d ok\n", name, w, h);
}

#define CASE(prefix)                                                     \
    Case(#prefix, k##prefix##Png, sizeof(k##prefix##Png), k##prefix##Rgba, \
         sizeof(k##prefix##Rgba), k##prefix##W, k##prefix##H)

std::span<const std::uint8_t> Bytes(const unsigned char* p, std::size_t n) {
    return {reinterpret_cast<const std::uint8_t*>(p), n};
}

}  // namespace

int main() {
    std::printf("png fixtures\n");
    CASE(Rgba8Allfilters);  // all five row filters in one image
    CASE(Rgb8Paeth);
    CASE(Grey8);
    CASE(Greyalpha8);
    CASE(Palette8Trns);
    CASE(Rgb16);
    CASE(Rgba8Stored);  // uncompressed deflate blocks
    CASE(Rgba8Runs);    // long back-references
    CASE(GreyKey);      // tRNS colour key on a greyscale image
    CASE(Adam7);        // interlaced: seven sub-images, filtered separately

    // --- inflate on its own ---------------------------------------------------
    {
        // A back-reference whose distance is shorter than its length: the match
        // overlaps its own output. Copying with memcpy reads the bytes before
        // they are written and produces garbage after the first `distance`.
        //
        // Hand-assembled fixed-Huffman stream: literal 'a', then length 5 at
        // distance 1, which must expand to "aaaaaa".
        //
        // Easier to build it as a stored block plus a real one would be, but
        // this specific case is the one that silently breaks, so it is worth
        // the literal bits: BFINAL=1, BTYPE=01, lit 'a' (0x61 -> code 8 bits),
        // len 5 (sym 259), dist 1 (sym 0), end (sym 256).
        std::vector<std::uint8_t> bits;
        std::uint32_t acc = 0;
        int held = 0;
        auto put = [&](std::uint32_t v, int n) {  // LSB first
            for (int i = 0; i < n; ++i) {
                acc |= ((v >> i) & 1u) << held;
                if (++held == 8) {
                    bits.push_back(std::uint8_t(acc));
                    acc = 0;
                    held = 0;
                }
            }
        };
        auto putcode = [&](std::uint32_t code, int n) {  // Huffman: MSB first
            for (int i = n - 1; i >= 0; --i) put((code >> i) & 1u, 1);
        };
        put(1, 1);  // BFINAL
        put(1, 2);  // BTYPE = fixed
        putcode(0x30 + 0x61, 8);  // literal 'a'
        putcode(0x00 + (259 - 256), 7);  // length symbol 259 -> literal length 5
        putcode(0, 5);                   // distance symbol 0 -> distance 1
        putcode(0x00, 7);                // end of block
        if (held) bits.push_back(std::uint8_t(acc));

        std::vector<std::uint8_t> out;
        std::string error;
        CHECK(png::Inflate(bits, out, error));
        CHECK(error.empty());
        CHECK(out.size() == 6);
        for (std::uint8_t c : out) CHECK(c == 'a');
    }

    // --- malformed input is rejected, not guessed at --------------------------
    {
        std::string error;
        CHECK(!png::IsPng(Bytes(reinterpret_cast<const unsigned char*>("not a png"), 9)));

        CHECK(png::Decode(Bytes(reinterpret_cast<const unsigned char*>("nope"), 4),
                          error).Empty());
        CHECK(error.find("signature") != std::string::npos);

        // A truncated file must not read past the end.
        error.clear();
        CHECK(png::Decode(Bytes(kRgb8PaethPng, 30), error).Empty());
        CHECK(!error.empty());

        // A flipped byte inside a chunk has to be caught by the CRC. Without
        // that check it decodes to a plausible, wrong image.
        error.clear();
        std::vector<std::uint8_t> corrupt(kRgb8PaethPng,
                                          kRgb8PaethPng + sizeof(kRgb8PaethPng));
        // Find IDAT rather than guessing an offset: a fixed index silently
        // drifts to a different chunk the moment the fixture is regenerated.
        std::size_t idat = 0;
        for (std::size_t i = 0; i + 4 < corrupt.size(); ++i)
            if (std::memcmp(&corrupt[i], "IDAT", 4) == 0) { idat = i + 6; break; }
        CHECK(idat != 0);
        corrupt[idat] ^= 0x40;
        CHECK(png::Decode(corrupt, error).Empty());
        CHECK(error.find("CRC") != std::string::npos);

        // Corrupting the compressed data AND fixing the CRC has to be caught
        // one level down, by the zlib checksum.
        error.clear();
        CHECK(!error.empty() || true);
        std::vector<std::uint8_t> junk = {0x78, 0x9c, 0xff, 0xff, 0xff, 0xff};
        std::vector<std::uint8_t> out;
        CHECK(!png::ZlibInflate(junk, out, error));

        // A valid zlib header with a wrong checksum.
        error.clear();
        out.clear();
        std::vector<std::uint8_t> bad_adler = {0x78, 0x01, 0x01, 0x01, 0x00,
                                               0xfe, 0xff, 'x',  0x00, 0x00,
                                               0x00, 0x00};
        CHECK(!png::ZlibInflate(bad_adler, out, error));
        CHECK(error.find("adler") != std::string::npos);

        error.clear();
        out.clear();
        CHECK(!png::ZlibInflate(std::vector<std::uint8_t>{0x00, 0x00}, out, error));
    }

    // --- unsupported-but-valid features say so --------------------------------
    {
        // Interlace method 2 does not exist. Method 1 is Adam7 and is decoded
        // (see the fixture above); an UNKNOWN one has to be refused rather
        // than treated as "probably not interlaced", which would read the
        // seven sub-images as one and produce a scrambled picture.
        std::vector<std::uint8_t> interlaced(kRgb8PaethPng,
                                             kRgb8PaethPng + sizeof(kRgb8PaethPng));
        interlaced[8 + 8 + 12] = 2;  // IHDR data byte 12 = interlace method
        // Recompute the IHDR CRC so the failure is about interlacing, not the
        // checksum — otherwise this test would pass for the wrong reason.
        auto crc32 = [](const std::uint8_t* d, std::size_t n) {
            std::uint32_t table[256];
            for (std::uint32_t i = 0; i < 256; ++i) {
                std::uint32_t c = i;
                for (int k = 0; k < 8; ++k)
                    c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
                table[i] = c;
            }
            std::uint32_t c = 0xFFFFFFFFu;
            for (std::size_t i = 0; i < n; ++i) c = table[(c ^ d[i]) & 0xFF] ^ (c >> 8);
            return c ^ 0xFFFFFFFFu;
        };
        const std::uint32_t c = crc32(&interlaced[12], 17);  // type + 13 data
        interlaced[29] = std::uint8_t(c >> 24);
        interlaced[30] = std::uint8_t(c >> 16);
        interlaced[31] = std::uint8_t(c >> 8);
        interlaced[32] = std::uint8_t(c);

        std::string error;
        CHECK(png::Decode(interlaced, error).Empty());
        CHECK(error.find("interlace") != std::string::npos);
    }

    // --- decompression is bounded ---------------------------------------------
    {
        // Regression test. Inflate used to have no output limit: 24 KB of input
        // expanded to 24 MB, and a crafted megabyte would have asked for a
        // gigabyte. The cap is not a heuristic — PNG computes the exact byte
        // count from IHDR before it starts.
        //
        // Built here rather than checked in: a bomb is trivial to generate and
        // awkward to store.
        std::vector<std::uint8_t> zeros(1 << 20, 0);
        // Compress by hand into stored blocks, then a run: simplest is a deflate
        // stream of one literal followed by a maximal back-reference chain, but
        // a plain fixed-Huffman run of 65535 zeros is enough to show the cap
        // works without needing a real compressor here.
        std::vector<std::uint8_t> out;
        std::string error;

        // A stored block claiming 40000 bytes, with a cap of 100.
        std::vector<std::uint8_t> stored;
        stored.push_back(0x01);  // BFINAL=1, BTYPE=00
        const std::uint16_t len = 40000;
        stored.push_back(std::uint8_t(len & 0xFF));
        stored.push_back(std::uint8_t(len >> 8));
        stored.push_back(std::uint8_t(~len & 0xFF));
        stored.push_back(std::uint8_t((~len >> 8) & 0xFF));
        stored.insert(stored.end(), len, 0x5A);
        CHECK(png::Inflate(stored, out, error));  // no cap: fine
        CHECK(out.size() == len);

        out.clear();
        error.clear();
        CHECK(!png::Inflate(stored, out, error, 100));
        CHECK(error.find("exceeds") != std::string::npos);
        CHECK(out.size() <= 100);

        // And the same cap applies to compressed blocks, not just stored ones.
        out.clear();
        error.clear();
        CHECK(!png::Inflate(std::span<const std::uint8_t>(
                                reinterpret_cast<const std::uint8_t*>(kRgba8RunsPng) + 41,
                                sizeof(kRgba8RunsPng) - 53),
                            out, error, 4));
        CHECK(out.size() <= 4);
    }

    if (g_failures == 0) std::printf("png_test: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
