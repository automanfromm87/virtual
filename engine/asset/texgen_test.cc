// Procedural textures, checked for the things that make one unusable.
//
// "Does it look like grass" is not testable. What breaks is: a texture that
// does not tile, so the ground shows a grid of seams; an albedo map whose mean
// is not one, so it silently darkens every material that uses it; a normal map
// that is not unit length or is biased off the surface, so every lit surface
// tilts; and a generator that is not deterministic, so no screenshot can ever
// be compared. All four are arithmetic.
#include "engine/asset/texgen.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "engine/asset/png.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-64s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

float Chan(const eng::texgen::Image& im, int x, int y, int c) {
    const std::size_t i =
        (std::size_t(y) * std::size_t(im.width) + std::size_t(x)) * 4 + std::size_t(c);
    return float(im.rgba[i]) / 255.0f;
}

// Mean absolute difference between two columns, over all three channels.
float ColumnGap(const eng::texgen::Image& im, int xa, int xb) {
    double sum = 0.0;
    for (int y = 0; y < im.height; ++y)
        for (int c = 0; c < 3; ++c)
            sum += std::fabs(double(Chan(im, xa, y, c)) - double(Chan(im, xb, y, c)));
    return float(sum / (double(im.height) * 3.0));
}

float RowGap(const eng::texgen::Image& im, int ya, int yb) {
    double sum = 0.0;
    for (int x = 0; x < im.width; ++x)
        for (int c = 0; c < 3; ++c)
            sum += std::fabs(double(Chan(im, x, ya, c)) - double(Chan(im, x, yb, c)));
    return float(sum / (double(im.width) * 3.0));
}

// THE TILING TEST, and the reason it is a ratio rather than a threshold.
//
// The wrap is seamless when column 0 continues from column size-1 the way any
// interior column continues from its neighbour. It is NOT that the two are
// equal -- they are different texels and should differ. So this compares the
// gap across the wrap against the typical gap between neighbours: 1.0 means the
// seam is indistinguishable from the rest of the texture, and a generator that
// does not wrap scores several times that because two unrelated hash lattices
// meet there.
float SeamRatio(const eng::texgen::Image& im) {
    const float wrap_x = ColumnGap(im, im.width - 1, 0);
    const float wrap_y = RowGap(im, im.height - 1, 0);
    double interior = 0.0;
    int n = 0;
    for (int x = 1; x < im.width - 1; x += 7) {
        interior += ColumnGap(im, x, x + 1);
        ++n;
    }
    for (int y = 1; y < im.height - 1; y += 7) {
        interior += RowGap(im, y, y + 1);
        ++n;
    }
    const float typical = n > 0 ? float(interior / n) : 1.0f;
    return typical > 1e-6f ? std::max(wrap_x, wrap_y) / typical : 0.0f;
}

float MeanChannel(const eng::texgen::Image& im, int c) {
    double sum = 0.0;
    for (int y = 0; y < im.height; ++y)
        for (int x = 0; x < im.width; ++x) sum += double(Chan(im, x, y, c));
    return float(sum / (double(im.width) * double(im.height)));
}

// Standard deviation over all three channels: how much detail there actually is.
float Variation(const eng::texgen::Image& im) {
    double sum = 0.0, sq = 0.0;
    int n = 0;
    for (int y = 0; y < im.height; ++y)
        for (int x = 0; x < im.width; ++x)
            for (int c = 0; c < 3; ++c) {
                const double v = double(Chan(im, x, y, c));
                sum += v;
                sq += v * v;
                ++n;
            }
    const double mean = sum / n;
    return float(std::sqrt(std::max(0.0, sq / n - mean * mean)));
}

void WriteIfAsked(const char* name, const eng::texgen::Image& im) {
    const char* dir = std::getenv("TEXGEN_DUMP");
    if (dir == nullptr) return;
    std::string path = std::string(dir) + "/" + name + ".png";
    std::string error;
    if (eng::png::EncodeFile(path, im.rgba, im.width, im.height, error))
        std::printf("    wrote %s\n", path.c_str());
}

}  // namespace

int main() {
    using namespace eng::texgen;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    constexpr int kSize = 256;

    const Surface ground = Ground(kSize, 3);
    const Surface bark = Bark(kSize, 11);
    const Image foliage = Foliage(kSize, 29);
    WriteIfAsked("ground_albedo", ground.albedo);
    WriteIfAsked("ground_normal", ground.normal);
    WriteIfAsked("bark_albedo", bark.albedo);
    WriteIfAsked("bark_normal", bark.normal);
    WriteIfAsked("foliage_albedo", foliage);

    {
        std::printf("every texture tiles\n");
        struct { const char* name; const Image* im; } all[] = {
            {"ground albedo", &ground.albedo}, {"ground normal", &ground.normal},
            {"bark albedo", &bark.albedo},     {"bark normal", &bark.normal},
            {"foliage", &foliage},
        };
        float worst = 0.0f;
        for (const auto& e : all) {
            const float ratio = SeamRatio(*e.im);
            std::printf("    %-14s seam is %.2fx a normal texel step\n", e.name, ratio);
            worst = std::max(worst, ratio);
        }
        // 3x, measured against a generator that wraps correctly and scores
        // between 0.9 and 1.6. A generator that does NOT wrap scores 20x or
        // more, because the two edges come from unrelated lattice cells and
        // differ by the full amplitude of the noise rather than by one step.
        Check(worst < 3.0f, "no edge is more than 3x a normal step from its wrap");
    }

    {
        std::printf("\nalbedo maps modulate rather than darken\n");
        // These MULTIPLY the material's base colour. A map with a mean of 0.7
        // would darken every material using it by a third, and the mistake is
        // invisible -- it looks like the lighting is wrong.
        for (const auto& e : {std::pair{"ground", &ground.albedo},
                              std::pair{"bark", &bark.albedo},
                              std::pair{"foliage", &foliage}}) {
            const float r = MeanChannel(*e.second, 0), g = MeanChannel(*e.second, 1),
                        b = MeanChannel(*e.second, 2);
            std::printf("    %-8s mean %.3f, %.3f, %.3f   variation %.3f\n", e.first, r,
                        g, b, Variation(*e.second));
            Check(std::fabs((r + g + b) / 3.0f - 1.0f) < 0.12f,
                  "its mean is within 12% of one");
            // A FLAT texture would pass every other check here and be worse
            // than no texture at all: it costs a megabyte and a sample to
            // multiply by one.
            Check(Variation(*e.second) > 0.04f, "and it actually carries detail");
        }
    }

    {
        std::printf("\nnormal maps decode to unit vectors facing out\n");
        for (const auto& e : {std::pair{"ground", &ground.normal},
                              std::pair{"bark", &bark.normal}}) {
            const Image& im = *e.second;
            float worst_len = 0.0f;
            double zsum = 0.0, xsum = 0.0, ysum = 0.0;
            int n = 0, backwards = 0;
            for (int y = 0; y < im.height; ++y)
                for (int x = 0; x < im.width; ++x) {
                    const float nx = Chan(im, x, y, 0) * 2.0f - 1.0f;
                    const float ny = Chan(im, x, y, 1) * 2.0f - 1.0f;
                    const float nz = Chan(im, x, y, 2) * 2.0f - 1.0f;
                    worst_len = std::max(worst_len,
                                         std::fabs(std::sqrt(nx * nx + ny * ny + nz * nz) - 1.0f));
                    if (nz <= 0.0f) ++backwards;
                    xsum += nx;
                    ysum += ny;
                    zsum += nz;
                    ++n;
                }
            std::printf("    %-8s worst length error %.4f, mean (%.3f, %.3f, %.3f), "
                        "%d facing into the surface\n",
                        e.first, worst_len, xsum / n, ysum / n, zsum / n, backwards);
            // 8-bit quantisation alone costs about 1/255 per channel, so the
            // tolerance is what the storage format permits and not a guess.
            Check(worst_len < 0.02f, "every texel is unit length to within 8-bit rounding");
            Check(backwards == 0, "and none of them points into the surface");
            // A BIASED map tilts every surface it is applied to, uniformly,
            // which reads as the lighting direction being wrong rather than as
            // a texture bug.
            Check(std::fabs(xsum / n) < 0.02f && std::fabs(ysum / n) < 0.02f,
                  "and the map has no net tilt");
        }
    }

    {
        std::printf("\nbark grain runs along the branch, not around it\n");
        // The tube generator lays u around the circumference and v along the
        // length. A groove is therefore something that varies in u and holds in
        // v -- if this is the wrong way round the trunk gets tree rings, which
        // is a convincing texture of the wrong thing.
        double across = 0.0, along = 0.0;
        for (int y = 0; y < kSize; y += 3) {
            for (int x = 0; x < kSize - 1; ++x)
                across += std::fabs(double(Chan(bark.albedo, x, y, 0)) -
                                    double(Chan(bark.albedo, x + 1, y, 0)));
            for (int x = 0; x < kSize; x += 3)
                if (y + 1 < kSize)
                    along += std::fabs(double(Chan(bark.albedo, x, y, 0)) -
                                       double(Chan(bark.albedo, x, y + 1, 0))) * 3.0;
        }
        std::printf("    variation across u %.1f, along v %.1f\n", across, along);
        Check(across > along * 2.0, "the grain varies far more across u than along v");
    }

    {
        std::printf("\nthe same seed gives the same texture\n");
        const Surface again = Ground(kSize, 3);
        const Surface other = Ground(kSize, 4);
        Check(again.albedo.rgba == ground.albedo.rgba,
              "seed 3 twice is byte-for-byte identical");
        Check(other.albedo.rgba != ground.albedo.rgba, "and seed 4 is a different texture");
    }

    {
        std::printf("\nedge cases do not crash or produce a broken image\n");
        const Surface tiny = Ground(4, 1);
        Check(tiny.albedo.Valid() && tiny.normal.Valid(), "a 4x4 texture is still valid");
        const std::vector<float> empty;
        const Image from_nothing = NormalFromHeight(empty, 16, 1.0f);
        Check(from_nothing.Valid(),
              "a height field that is too small returns a valid image rather than reading past it");
        // OCTAVES ARE CAPPED at the point where the lattice is finer than the
        // image, because past that the noise is pure hash: white noise, which
        // aliases into a shimmering mess the moment it is minified.
        const std::vector<float> over = Fbm(64, 1, 4, 4, 12, 0.5f);
        Check(over.size() == 64u * 64u, "asking for more octaves than fit is clamped");
    }

    std::printf(g_failures == 0 ? "\ntexgen_test: all checks passed\n"
                                : "\ntexgen_test: %d FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
