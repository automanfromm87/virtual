// Block compression, checked as error and as size.
//
// A lossy codec cannot be tested for equality, so the checks are of two kinds:
// the SIZE is exact arithmetic and must be exact, and the ERROR is measured
// against what the format can represent rather than against zero. The second
// needs a reference: BC1 quantises colour to 5-6-5, so an image that is already
// 5-6-5 must survive it almost exactly, while a smooth gradient must not.
#include "engine/texture/compress.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

// Root-mean-square error per channel, in 0..255 units.
double Rms(const eng::Texture2D& a, const eng::Texture2D& b, int channels) {
    if (a.width != b.width || a.height != b.height) return 1e30;
    double sum = 0.0;
    long long n = 0;
    for (int i = 0; i < a.width * a.height; ++i)
        for (int c = 0; c < channels; ++c) {
            const double d = double(a.rgba[std::size_t(i) * 4 + c]) -
                             double(b.rgba[std::size_t(i) * 4 + c]);
            sum += d * d;
            ++n;
        }
    return std::sqrt(sum / double(n));
}

double WorstOf(const eng::Texture2D& a, const eng::Texture2D& b, int channels) {
    double worst = 0.0;
    for (int i = 0; i < a.width * a.height; ++i)
        for (int c = 0; c < channels; ++c)
            worst = std::max(worst,
                             std::fabs(double(a.rgba[std::size_t(i) * 4 + c]) -
                                       double(b.rgba[std::size_t(i) * 4 + c])));
    return worst;
}

eng::Texture2D Gradient(int w, int h) {
    eng::Texture2D t;
    t.width = w;
    t.height = h;
    t.rgba.resize(std::size_t(w) * h * 4);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            const std::size_t i = (std::size_t(y) * w + x) * 4;
            t.rgba[i + 0] = std::uint8_t(x * 255 / std::max(w - 1, 1));
            t.rgba[i + 1] = std::uint8_t(y * 255 / std::max(h - 1, 1));
            t.rgba[i + 2] = std::uint8_t((x + y) * 255 / std::max(w + h - 2, 1));
            t.rgba[i + 3] = std::uint8_t(255 - y * 255 / std::max(h - 1, 1));
        }
    return t;
}

// A tangent-space normal map of round bumps: unit vectors, which is what BC5 is
// for and what BC1 is worst at.
eng::Texture2D Bumps(int size) {
    eng::Texture2D t;
    t.width = size;
    t.height = size;
    t.rgba.resize(std::size_t(size) * size * 4);
    const float period = float(size) / 8.0f;
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x) {
            const float u = std::fmod(float(x), period) / period * 2.0f - 1.0f;
            const float v = std::fmod(float(y), period) / period * 2.0f - 1.0f;
            const float r2 = std::min(u * u + v * v, 1.0f);
            const float nx = u, ny = v;
            const std::size_t i = (std::size_t(y) * size + x) * 4;
            t.rgba[i + 0] = std::uint8_t(std::lround((nx * 0.5f + 0.5f) * 255.0f));
            t.rgba[i + 1] = std::uint8_t(std::lround((ny * 0.5f + 0.5f) * 255.0f));
            t.rgba[i + 2] = std::uint8_t(
                std::lround((std::sqrt(1.0f - r2) * 0.5f + 0.5f) * 255.0f));
            t.rgba[i + 3] = 255;
        }
    return t;
}

}  // namespace

int main() {
    using namespace eng;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    {
        std::printf("sizes are exact\n");
        const Texture2D src = Gradient(256, 128);
        const CompressedTexture bc1 = Compress(src, BlockFormat::BC1, true, false);
        const CompressedTexture bc3 = Compress(src, BlockFormat::BC3, true, false);

        // 256x128 is 64x32 blocks at the top level. The chain halves both
        // dimensions until 1x1, and every level below 4 texels is still one
        // block -- which is why the total is not simply 4/3 of the top.
        std::size_t expect1 = 0;
        for (int w = 256, h = 128;; w = std::max(1, w / 2), h = std::max(1, h / 2)) {
            expect1 += std::size_t((w + 3) / 4) * std::size_t((h + 3) / 4) * 8;
            if (w == 1 && h == 1) break;
        }
        std::printf("    256x128 RGBA8 is %zu bytes; BC1 with mips is %zu (%.1f:1)\n",
                    std::size_t(256 * 128 * 4), bc1.data.size(),
                    double(256 * 128 * 4) / double(bc1.data.size()));
        Check(bc1.data.size() == expect1, "BC1 is exactly 8 bytes per 4x4 block");
        Check(bc3.data.size() == expect1 * 2, "and BC3 is exactly twice that");
        Check(bc1.Levels() == 9, "the chain runs 256x128 down to 1x1: 9 levels");
        Check(bc1.level_offsets[0] == 0 && bc1.level_offsets[1] == 64 * 32 * 8,
              "with each level starting where the last one ended");
        // 4 bits a texel against RGBA8's 32 is 8:1 at the top level. The
        // figure usually quoted for BC1 is 6:1, which is against RGB8 -- and
        // 6:1 is also what comes out here against RGBA8 once the mip chain is
        // counted, because a chain adds a third to the compressed side and
        // nothing to the uncompressed figure it is being compared with.
        const double top_ratio = double(256 * 128 * 4) / double(64 * 32 * 8);
        Check(std::fabs(top_ratio - 8.0) < 0.01,
              "8:1 against RGBA8 at the top level, exactly");
        Check(std::fabs(double(256 * 128 * 4) / double(bc1.data.size()) - 6.0) < 0.2,
              "and 6:1 with the whole chain against the unmipped original");
    }

    {
        std::printf("\nBC1 loses what 5-6-5 cannot hold, and no more\n");
        const Texture2D src = Gradient(128, 128);
        const CompressedTexture c = Compress(src, BlockFormat::BC1, false, false);
        const Texture2D back = DecodeBlocks(c.data, 128, 128, BlockFormat::BC1);
        const double rms = Rms(src, back, 3);
        const double worst = WorstOf(src, back, 3);
        std::printf("    smooth gradient: rms %.2f/255, worst %.0f\n", rms, worst);
        // The floor is set by the format. A 5-bit channel steps by 8, so an
        // encoder that did nothing but quantise would sit near 8/sqrt(12) = 2.3;
        // the four-point palette does better than that on a smooth block.
        Check(rms < 3.0, "the error is at the quantisation floor, not above it");
        Check(worst < 24.0, "and no single texel is wildly wrong");

        // A RAMP WITH OUTLIERS is the pattern that separates endpoint
        // strategies. Insetting the endpoints -- the standard advice -- was
        // tried here and measured worse on exactly this: 6.27 against 4.95,
        // because the outliers it gives up on are the texels whose squared
        // error dominates. The threshold below is set from the measurement, so
        // putting the inset back fails it.
        Texture2D speckle;
        speckle.width = speckle.height = 64;
        speckle.rgba.assign(64u * 64u * 4u, 255);
        for (int y = 0; y < 64; ++y)
            for (int x = 0; x < 64; ++x) {
                const std::size_t i = (std::size_t(y) * 64 + x) * 4;
                // A smooth ramp with a few outliers scattered through it, which
                // is what a real texture looks like inside one block.
                const bool spike = ((x * 7 + y * 13) % 17) == 0;
                const int v = spike ? 250 : 40 + (x * 3) % 60;
                speckle.rgba[i] = speckle.rgba[i + 1] = speckle.rgba[i + 2] =
                    std::uint8_t(v);
            }
        const CompressedTexture sc2 =
            Compress(speckle, BlockFormat::BC1, false, false);
        const double srms =
            Rms(speckle, DecodeBlocks(sc2.data, 64, 64, BlockFormat::BC1), 3);
        std::printf("    a ramp with outliers: rms %.2f/255\n", srms);
        Check(srms < 5.5,
              "endpoints at the block's extremes keep its outliers");

        // A FLAT block is exactly representable and must come back exactly.
        // This is the check that catches an endpoint ordering bug: get c0 and
        // c1 the wrong way round and a flat block still decodes flat, but at
        // the wrong one of the two.
        Texture2D flat;
        flat.width = flat.height = 16;
        flat.rgba.assign(16 * 16 * 4, 0);
        for (int i = 0; i < 16 * 16; ++i) {
            flat.rgba[std::size_t(i) * 4 + 0] = 132;  // exactly representable
            flat.rgba[std::size_t(i) * 4 + 1] = 130;  // in 5-6-5 after the
            flat.rgba[std::size_t(i) * 4 + 2] = 74;   // low-bit replication
            flat.rgba[std::size_t(i) * 4 + 3] = 255;
        }
        const CompressedTexture fc = Compress(flat, BlockFormat::BC1, false, false);
        const Texture2D fb = DecodeBlocks(fc.data, 16, 16, BlockFormat::BC1);
        std::printf("    flat block (132,130,74) came back (%d,%d,%d)\n",
                    fb.rgba[0], fb.rgba[1], fb.rgba[2]);
        Check(WorstOf(flat, fb, 3) < 1.5,
              "a flat colour survives a round trip exactly");
    }

    {
        std::printf("\nBC3 keeps an alpha channel BC1 cannot\n");
        const Texture2D src = Gradient(128, 128);
        const CompressedTexture c = Compress(src, BlockFormat::BC3, false, false);
        const Texture2D back = DecodeBlocks(c.data, 128, 128, BlockFormat::BC3);
        double a_rms = 0.0;
        for (int i = 0; i < 128 * 128; ++i) {
            const double d = double(src.rgba[std::size_t(i) * 4 + 3]) -
                             double(back.rgba[std::size_t(i) * 4 + 3]);
            a_rms += d * d;
        }
        a_rms = std::sqrt(a_rms / (128.0 * 128.0));
        std::printf("    alpha rms %.3f/255, colour rms %.2f/255\n", a_rms,
                    Rms(src, back, 3));
        // Alpha gets a whole BC4 block to itself: eight levels between two
        // 8-bit endpoints, per 4x4 tile. On a gradient that is nearly lossless.
        Check(a_rms < 1.0, "alpha survives almost exactly");
        Check(Rms(src, back, 3) < 3.0, "and the colour is no worse than BC1's");
    }

    {
        std::printf("\nBC5 is the normal map format and BC1 is not\n");
        const Texture2D src = Bumps(128);
        const CompressedTexture bc5 = Compress(src, BlockFormat::BC5, false, false);
        const CompressedTexture bc1 = Compress(src, BlockFormat::BC1, false, false);
        const Texture2D from5 = DecodeBlocks(bc5.data, 128, 128, BlockFormat::BC5);
        const Texture2D from1 = DecodeBlocks(bc1.data, 128, 128, BlockFormat::BC1);

        // Compare the x and y channels only: those are what a normal map
        // carries and what the shader reads. BC5 stores exactly them; BC1
        // stores all three as one line through RGB, which is the mistake.
        const double e5 = Rms(src, from5, 2);
        const double e1 = Rms(src, from1, 2);
        std::printf("    xy error: BC5 %.2f/255 (16 bytes), BC1 %.2f/255 (8 bytes)\n",
                    e5, e1);
        Check(e5 < 2.0, "BC5 keeps a normal map almost exactly");
        Check(e1 > e5 * 2.5,
              "and BC1 is several times worse on the same data");

        // The angular error is what actually matters, because that is what the
        // lighting sees. A degree is roughly the point where banding stops
        // being visible on a smooth surface.
        const auto worst_angle = [&](const Texture2D& t) {
            double worst = 0.0;
            for (int i = 0; i < 128 * 128; ++i) {
                const auto unpack = [&](const Texture2D& s, int c) {
                    return double(s.rgba[std::size_t(i) * 4 + c]) / 127.5 - 1.0;
                };
                const double ax = unpack(src, 0), ay = unpack(src, 1);
                const double az = std::sqrt(std::max(1.0 - ax * ax - ay * ay, 0.0));
                const double bx = unpack(t, 0), by = unpack(t, 1);
                const double bz = std::sqrt(std::max(1.0 - bx * bx - by * by, 0.0));
                const double d = std::clamp(ax * bx + ay * by + az * bz, -1.0, 1.0);
                worst = std::max(worst, std::acos(d) * 180.0 / 3.14159265);
            }
            return worst;
        };
        // The WORST angle is dominated by the handful of blocks that straddle
        // a bump's rim, where the normal swings through ninety degrees inside
        // one 4x4 tile and no two endpoints can span it. That number says
        // nothing about the format's quality on real data, so what is asserted
        // is the fraction of texels that are visibly off -- a degree is roughly
        // where banding stops being perceptible on a smooth surface.
        const auto angle_stats = [&](const Texture2D& t) {
            double worst = 0.0;
            int over_one = 0;
            for (int i = 0; i < 128 * 128; ++i) {
                const auto unpack = [&](const Texture2D& s, int c) {
                    return double(s.rgba[std::size_t(i) * 4 + c]) / 127.5 - 1.0;
                };
                const double ax = unpack(src, 0), ay = unpack(src, 1);
                const double az = std::sqrt(std::max(1.0 - ax * ax - ay * ay, 0.0));
                const double bx = unpack(t, 0), by = unpack(t, 1);
                const double bz = std::sqrt(std::max(1.0 - bx * bx - by * by, 0.0));
                const double d = std::clamp(ax * bx + ay * by + az * bz, -1.0, 1.0);
                const double deg = std::acos(d) * 180.0 / 3.14159265;
                worst = std::max(worst, deg);
                if (deg > 1.0) ++over_one;
            }
            return std::pair<double, double>{worst,
                                             100.0 * over_one / (128.0 * 128.0)};
        };
        const auto [w5, p5] = angle_stats(from5);
        const auto [w1, p1] = angle_stats(from1);
        std::printf("    angular error: BC5 worst %.2f deg, %.2f%% over 1 deg; "
                    "BC1 worst %.2f deg, %.2f%% over 1 deg\n",
                    w5, p5, w1, p1);
        // Half the texels being off by more than a degree is not a defect, it
        // is arithmetic: 3-bit indices give eight steps per block, this pattern
        // sweeps x by half its range inside one 4x4 tile, and one step of that
        // is already about 1.8 degrees. No 16-byte format does better on a
        // gradient that steep. What the comparison shows is the ratio.
        Check(w5 * 4.0 < w1, "BC5's worst normal is far closer than BC1's");
        Check(p5 < p1 * 0.7, "and fewer of them are visibly off");
    }

    {
        std::printf("\nmips, and the sRGB average that is easy to get wrong\n");
        // A checkerboard of black and white. Its correct average in LINEAR
        // light is 0.5, which is 188 in sRGB -- not 128. Averaging the bytes
        // directly gives 128, and that is the classic mip-chain darkening: by
        // the time a black-and-white pattern has been reduced to one texel it
        // should be a light grey and instead it is a mid grey.
        Texture2D checks;
        checks.width = checks.height = 64;
        checks.rgba.assign(64 * 64 * 4, 255);
        for (int y = 0; y < 64; ++y)
            for (int x = 0; x < 64; ++x) {
                const std::uint8_t v = ((x + y) & 1) ? 255 : 0;
                const std::size_t i = (std::size_t(y) * 64 + x) * 4;
                checks.rgba[i] = checks.rgba[i + 1] = checks.rgba[i + 2] = v;
            }

        const CompressedTexture lin = Compress(checks, BlockFormat::BC1, true, false);
        const CompressedTexture srgb = Compress(checks, BlockFormat::BC1, true, true);
        const int last = lin.Levels() - 1;
        const Texture2D lin_top = DecodeBlocks(
            {lin.data.data() + lin.level_offsets[std::size_t(last)],
             lin.data.size() - lin.level_offsets[std::size_t(last)]},
            1, 1, BlockFormat::BC1);
        const Texture2D srgb_top = DecodeBlocks(
            {srgb.data.data() + srgb.level_offsets[std::size_t(last)],
             srgb.data.size() - srgb.level_offsets[std::size_t(last)]},
            1, 1, BlockFormat::BC1);
        std::printf("    a black/white checker's 1x1 mip: byte average %d, "
                    "linear-light average %d (188 is correct)\n",
                    lin_top.rgba[0], srgb_top.rgba[0]);
        Check(std::abs(int(lin_top.rgba[0]) - 128) < 12,
              "averaging the bytes gives the wrong, darker answer");
        Check(std::abs(int(srgb_top.rgba[0]) - 188) < 12,
              "and decoding to linear first gives the right one");
    }

    {
        std::printf("\nedges and degenerate input\n");
        // 13x7 is not a multiple of four in either direction, so both the last
        // block column and the last block row are partial.
        //
        // A FLAT colour, not a gradient. What this section is testing is the
        // PADDING -- whether the three missing columns of the last block are
        // filled by replicating the edge or by something that drags the block's
        // endpoints somewhere else. On a flat image the correct answer is exact
        // and any error at all is the padding leaking; on a gradient the answer
        // would be buried under the codec's ordinary error and the check would
        // have measured the wrong thing.
        Texture2D odd;
        odd.width = 13;
        odd.height = 7;
        odd.rgba.assign(13u * 7u * 4u, 255);
        for (int i = 0; i < 13 * 7; ++i) {
            odd.rgba[std::size_t(i) * 4 + 0] = 132;
            odd.rgba[std::size_t(i) * 4 + 1] = 130;
            odd.rgba[std::size_t(i) * 4 + 2] = 74;
        }
        const CompressedTexture c = Compress(odd, BlockFormat::BC1, false, false);
        Check(c.data.size() == 4u * 2u * 8u, "a 13x7 image is 4x2 blocks");
        const Texture2D back = DecodeBlocks(c.data, 13, 7, BlockFormat::BC1);
        Check(back.width == 13 && back.height == 7,
              "and decodes back to 13x7, not to 16x8");
        std::printf("    13x7 flat colour: worst error %.0f/255\n",
                    WorstOf(odd, back, 3));
        Check(WorstOf(odd, back, 3) < 1.5,
              "the partial blocks pad by replication, so a flat image is exact");

        // AND a case that can actually distinguish one padding strategy from
        // another, which neither of the two above can.
        //
        // BC1 stores both ENDPOINTS exactly, so a block whose real texels are
        // its own min and max comes back right however the filler is chosen --
        // which is why a flat image, and an image with one bright edge column,
        // both pass under zero padding, wrap-around padding and replication
        // alike. Only an INTERIOR value can tell them apart.
        //
        // 14 wide: the last block holds two real columns and two filler ones.
        // Column 12 is 150 and column 13 is 200, so 150 is interior. Replicate
        // the edge and the block spans 150..200 and 150 is an endpoint, exact.
        // Pad with black and it spans 0..200, where the nearest palette point
        // to 150 is 137.
        Texture2D interior;
        interior.width = 14;
        interior.height = 8;
        interior.rgba.assign(14u * 8u * 4u, 255);
        for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 14; ++x) {
                const std::uint8_t v = x == 13 ? 200 : (x == 12 ? 150 : 100);
                const std::size_t i = (std::size_t(y) * 14 + x) * 4;
                interior.rgba[i] = interior.rgba[i + 1] = interior.rgba[i + 2] = v;
            }
        const CompressedTexture ic =
            Compress(interior, BlockFormat::BC1, false, false);
        const Texture2D ib = DecodeBlocks(ic.data, 14, 8, BlockFormat::BC1);
        const int interior_col = ib.rgba[(std::size_t(4) * 14 + 12) * 4];
        std::printf("    an interior value beside the edge (150) came back as %d "
                    "(148 = replicated, ~137 = filler)\n",
                    interior_col);
        Check(interior_col > 143,
              "and the filler never enters the block's colour range");

        // And a gradient the same size, for the record. The error is large and
        // it is the format, not the padding: a 4x4 tile of a 13-wide 2D
        // gradient spans a PATCH of colours, and BC1 can only represent a line
        // through them.
        const Texture2D odd_grad = Gradient(13, 7);
        const CompressedTexture gc =
            Compress(odd_grad, BlockFormat::BC1, false, false);
        std::printf("    13x7 2D gradient: rms %.2f/255 (a 4x4 tile spans a "
                    "patch, and BC1 stores a line)\n",
                    Rms(odd_grad, DecodeBlocks(gc.data, 13, 7, BlockFormat::BC1), 3));

        Check(Compress(Texture2D{}, BlockFormat::BC1).Empty(),
              "an empty image compresses to nothing rather than crashing");
        Check(DecodeBlocks({}, 4, 4, BlockFormat::BC1).Empty(),
              "and too few bytes decode to nothing rather than overreading");
    }

    std::printf(g_failures == 0 ? "\ncompress_test: all checks passed\n"
                                : "\ncompress_test: %d FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
