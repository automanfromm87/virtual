// Baseline JPEG, against files decoded by Apple's ImageIO.
//
// JPEG is lossy and the IDCT is not specified to the last level, so this cannot
// be an exact comparison the way the PNG tests are. It is a bounded one: the
// tolerances below were MEASURED, not guessed, and are printed on every run so
// that a change which quietly doubles the error is visible rather than merely
// still-passing.

#include "engine/asset/jpeg.h"

#include <cmath>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Fail(const char* what) {
    std::fprintf(stderr, "  %-58s FAIL\n", what);
    ++g_failures;
}

void Check(bool ok, const char* what) {
    if (ok) {
        std::printf("  %-58s ok\n", what);
    } else {
        Fail(what);
    }
}

#include "engine/asset/testdata_jpeg.inc"

std::span<const std::uint8_t> Bytes(const unsigned char* p, std::size_t n) {
    return {reinterpret_cast<const std::uint8_t*>(p), n};
}

// Decodes and compares to the reference, reporting the worst and the mean
// per-channel deviation.
struct Deviation {
    int worst = 0;
    double mean = 0.0;
    int large = 0;          // channels off by more than kLarge
    int large_in_flat = 0;  // ...of those, ones NOT at an edge
    double flat_fraction = 0.0;  // how much of the image the rule constrains
};

// Where two decoders are allowed to disagree.
//
// They cannot be compared exactly. JPEG does not specify the IDCT to the last
// level, and chroma upsampling kernels differ between implementations -- at a
// hard chroma edge, two reasonable kernels can land ninety levels apart, and
// the fixtures here are deliberately adversarial about that (a saturated
// magenta rectangle on a checkerboard is about the worst case a photograph
// never contains).
//
// So the bound is not a single number. It is: the MEAN must be small, and every
// large deviation must sit at an EDGE. A large deviation in a flat region is
// not a kernel difference -- it is DC drift, a bad quantisation table, or a
// missed restart, and those are exactly the bugs a loose worst-case bound hides.
constexpr int kLarge = 12;
constexpr int kEdge = 40;    // reference gradient that counts as an edge
// Radius 2, not 1, and for a reason on each side: ringing from an 8x8 DCT
// spreads across the block, and a 4:2:0 upsampling kernel reaches two output
// pixels either side of a chroma sample. Measured, too -- the deviations this
// misses at radius 1 are all exactly two pixels from a checkerboard edge.
constexpr int kReach = 2;

// An edge is a property of the PIXEL, not of one channel. A chroma
// discontinuity reaches all three RGB channels through the colour transform,
// including ones whose own value barely moves: at the fixture's magenta
// boundary red shifts 15 levels while green shifts 71, and judging the red
// channel on its own gradient calls that spot flat when it is the sharpest
// colour edge in the image.
bool NearEdge(const unsigned char* ref, int w, int h, int x, int y) {
    const unsigned char* here = &ref[(std::size_t(y) * w + x) * 3];
    for (int dy = -kReach; dy <= kReach; ++dy)
        for (int dx = -kReach; dx <= kReach; ++dx) {
            const int nx = x + dx, ny = y + dy;
            if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
            const unsigned char* there = &ref[(std::size_t(ny) * w + nx) * 3];
            for (int c = 0; c < 3; ++c)
                if (std::abs(int(there[c]) - int(here[c])) > kEdge) return true;
        }
    return false;
}

bool Compare(const char* name, const unsigned char* jpg, std::size_t jpg_n,
             const unsigned char* ref, std::size_t ref_n, int w, int h,
             Deviation* dev) {
    std::string error;
    const eng::Texture2D img = eng::jpeg::Decode(Bytes(jpg, jpg_n), error);
    if (!error.empty()) {
        std::fprintf(stderr, "  %-14s decode failed: %s\n", name, error.c_str());
        ++g_failures;
        return false;
    }
    if (img.width != w || img.height != h) {
        std::fprintf(stderr, "  %-14s decoded %dx%d, expected %dx%d\n", name,
                     img.width, img.height, w, h);
        ++g_failures;
        return false;
    }
    if (ref_n != std::size_t(w) * std::size_t(h) * 3) {
        std::fprintf(stderr, "  %-14s reference is the wrong size\n", name);
        ++g_failures;
        return false;
    }

    long long sum = 0;
    for (std::size_t i = 0; i < std::size_t(w) * std::size_t(h); ++i) {
        const int x = int(i % std::size_t(w)), y = int(i / std::size_t(w));
        for (int c = 0; c < 3; ++c) {
            const int got = img.rgba[i * 4 + std::size_t(c)];
            const int want = ref[i * 3 + std::size_t(c)];
            const int d = std::abs(got - want);
            sum += d;
            if (d > dev->worst) dev->worst = d;
            if (d > kLarge) {
                ++dev->large;
                if (!NearEdge(ref, w, h, x, y)) ++dev->large_in_flat;
            }
        }
        // JPEG has no alpha, and inventing one from the luma is a guess.
        if (img.rgba[i * 4 + 3] != 255) {
            std::fprintf(stderr, "  %-14s alpha is not opaque\n", name);
            ++g_failures;
            return false;
        }
    }
    dev->mean = double(sum) / (double(w) * h * 3);

    // How much of the image is flat, and therefore how much the
    // "large deviations only at edges" rule actually forbids. Printed because a
    // rule that constrains 4% of an image is not a test, and the number is the
    // only way to tell the difference from the outside.
    long long flat = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            if (!NearEdge(ref, w, h, x, y)) ++flat;
    dev->flat_fraction = double(flat) / (double(w) * h);
    return true;
}

#define CASE(prefix, mean_max)                                                \
    do {                                                                      \
        Deviation dev;                                                        \
        if (Compare(#prefix, k##prefix##Jpg, sizeof(k##prefix##Jpg),          \
                    k##prefix##Rgb, sizeof(k##prefix##Rgb), k##prefix##W,     \
                    k##prefix##H, &dev)) {                                    \
            std::printf("  %-10s %2dx%-2d  mean %.3f  worst %2d  (%3d over "  \
                        "%d, %d in flat; %.0f%% of the image is flat)\n",      \
                        #prefix, k##prefix##W, k##prefix##H, dev.mean,        \
                        dev.worst, dev.large, kLarge, dev.large_in_flat,      \
                        dev.flat_fraction * 100.0);                           \
            if (dev.mean > (mean_max))                                        \
                Fail(#prefix ": mean deviation from ImageIO is small");       \
            if (dev.large_in_flat != 0)                                       \
                Fail(#prefix ": every large deviation is at an edge");        \
        }                                                                     \
    } while (0)

}  // namespace

int main() {
    std::printf("baseline jpeg, against Apple's ImageIO\n");

    // Mean bounds are per case because they are not equally hard: a full
    // resolution image needs no chroma reconstruction at all and agrees to
    // within half a level, while 4:2:0 has to invent three quarters of the
    // colour samples and no two decoders invent quite the same ones.
    CASE(E444, 0.5);
    CASE(E422, 0.7);
    CASE(E420, 1.6);
    CASE(E420Rst, 1.6);
    CASE(E444Low, 1.1);
    CASE(EGrey, 0.2);
    CASE(Apple444, 0.4);
    CASE(Apple420, 3.5);
    // These two are mostly smooth, so the edge rule below constrains nearly the
    // whole image rather than a few percent of it.
    CASE(EFlat, 0.6);
    CASE(EFlat444, 0.4);

    // Restart markers must actually be USED, not merely tolerated. The same
    // image encoded with and without them has to decode to the same picture:
    // a decoder that ignores the DC reset at each restart drifts, and the drift
    // is cumulative, so the error grows down the image rather than appearing as
    // noise.
    {
        std::string e1, e2;
        const eng::Texture2D plain =
            eng::jpeg::Decode(Bytes(kE420Jpg, sizeof(kE420Jpg)), e1);
        const eng::Texture2D rst =
            eng::jpeg::Decode(Bytes(kE420RstJpg, sizeof(kE420RstJpg)), e2);
        int worst = 0;
        if (plain.rgba.size() == rst.rgba.size() && !plain.rgba.empty()) {
            for (std::size_t i = 0; i < plain.rgba.size(); ++i)
                worst = std::max(worst, std::abs(int(plain.rgba[i]) - int(rst.rgba[i])));
        } else {
            worst = 999;
        }
        std::printf("    same image with and without restart markers: worst "
                    "channel differs by %d\n", worst);
        Check(worst <= 2, "restart intervals decode to the same picture");
    }

    // --- refusals ----------------------------------------------------------
    //
    // Each of these has to name what it was. A decoder that returns a green
    // rectangle for a progressive file sends whoever hits it looking for a bug
    // in their texture pipeline.
    {
        std::string error;
        const std::uint8_t not_jpeg[8] = {0x89, 'P', 'N', 'G'};
        Check(eng::jpeg::Decode(not_jpeg, error).rgba.empty() &&
                  error.find("SOI") != std::string::npos,
              "a non-jpeg is refused by name");

        Check(!eng::jpeg::IsJpeg(not_jpeg) &&
                  eng::jpeg::IsJpeg(Bytes(kE444Jpg, sizeof(kE444Jpg))),
              "IsJpeg tells the two apart");

        // SOF0 -> SOF2 makes it progressive. Nothing else about the file
        // changes, so a decoder that does not check the marker will happily
        // read the first scan and produce a plausible, wrong image.
        std::vector<std::uint8_t> prog(kE444Jpg, kE444Jpg + sizeof(kE444Jpg));
        int patched = 0;
        for (std::size_t i = 0; i + 1 < prog.size(); ++i)
            if (prog[i] == 0xFF && prog[i + 1] == 0xC0) {
                prog[i + 1] = 0xC2;
                ++patched;
                break;
            }
        error.clear();
        Check(patched == 1 && eng::jpeg::Decode(prog, error).rgba.empty() &&
                  error.find("progressive") != std::string::npos,
              "a progressive file is refused, and says so");

        // Truncated mid-scan. The entropy decoder must run off the end into
        // zeroes rather than past the buffer.
        error.clear();
        const eng::Texture2D cut = eng::jpeg::Decode(
            Bytes(kE444Jpg, sizeof(kE444Jpg) / 2), error);
        Check(cut.rgba.empty() || cut.width == kE444W,
              "a truncated file does not read past its own bytes");

        // A segment claiming to be longer than the file.
        std::vector<std::uint8_t> lying(kE444Jpg, kE444Jpg + sizeof(kE444Jpg));
        for (std::size_t i = 2; i + 3 < lying.size(); ++i)
            if (lying[i] == 0xFF && lying[i + 1] == 0xDB) {
                lying[i + 2] = 0x7F;
                lying[i + 3] = 0xFF;
                break;
            }
        error.clear();
        Check(eng::jpeg::Decode(lying, error).rgba.empty() &&
                  error.find("past the end") != std::string::npos,
              "a segment longer than the file is refused");

        // A stream that ends at SOI: no frame, no scan.
        error.clear();
        const std::uint8_t empty[2] = {0xFF, 0xD8};
        Check(eng::jpeg::Decode(empty, error).rgba.empty() && !error.empty(),
              "a file with no frame header is refused");
    }

    if (g_failures == 0) std::printf("jpeg_test: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
