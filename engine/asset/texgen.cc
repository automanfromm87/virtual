#include "engine/asset/texgen.h"

#include <algorithm>
#include <cmath>

namespace eng::texgen {
namespace {

// A hash of two integers and a seed, uniform in [0, 1).
//
// Written out rather than taken from <random>: the same seed has to give the
// same texture on every machine and every standard library, and only arithmetic
// spelled here does.
float Hash(int x, int y, std::uint32_t seed) {
    std::uint32_t h = seed;
    h ^= std::uint32_t(x) * 0x9E3779B9u;
    h ^= std::uint32_t(y) * 0x85EBCA6Bu;
    h ^= h >> 15;
    h *= 0x2545F491u;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return float(h & 0xFFFFFFu) / float(0x1000000u);
}

// Positive modulo. The built-in % is negative for negative inputs, which on a
// wrapping lattice means the cell to the left of zero hashes differently from
// the cell it is supposed to equal -- one seam, on one edge, every time.
int Wrap(int v, int n) { return ((v % n) + n) % n; }

// Value noise on a lattice that repeats every `px` cells across and `py` down.
float TileNoise(float x, float y, int px, int py, std::uint32_t seed) {
    const int ix = int(std::floor(x)), iy = int(std::floor(y));
    const float fx = x - float(ix), fy = y - float(iy);
    // Smoothstep on the interpolant, so the derivative is continuous across
    // cell boundaries. Linear interpolation leaves a crease on every grid line,
    // which at four octaves becomes a plaid.
    const float sx = fx * fx * (3.0f - 2.0f * fx);
    const float sy = fy * fy * (3.0f - 2.0f * fy);
    const int x0 = Wrap(ix, px), x1 = Wrap(ix + 1, px);
    const int y0 = Wrap(iy, py), y1 = Wrap(iy + 1, py);
    const float a = Hash(x0, y0, seed), b = Hash(x1, y0, seed);
    const float c = Hash(x0, y1, seed), d = Hash(x1, y1, seed);
    const float top = a + (b - a) * sx;
    const float bottom = c + (d - c) * sx;
    return top + (bottom - top) * sy;
}

std::uint8_t Byte(float v) {
    return std::uint8_t(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
}

// Writes a linear-space colour to an RGBA8 image.
//
// NO GAMMA ENCODING HERE. These are uploaded to sRGB texture formats, so the
// hardware does the decode on sample and expects encoded bytes -- but an albedo
// MODULATION is a ratio, not a colour, and a ratio has no gamma. Encoding one
// would make a 0.5 multiplier arrive as 0.21.
//
// This is the one thing about these textures that is easy to get backwards, so:
// the caller decides the format, and Ground/Bark/Foliage below are all built to
// be uploaded as LINEAR (non-sRGB) because they multiply rather than replace.
void Put(Image& im, int x, int y, Vec3 c, float a = 1.0f) {
    const std::size_t i = (std::size_t(y) * std::size_t(im.width) + std::size_t(x)) * 4;
    im.rgba[i] = Byte(c.x);
    im.rgba[i + 1] = Byte(c.y);
    im.rgba[i + 2] = Byte(c.z);
    im.rgba[i + 3] = Byte(a);
}

Image Blank(int size) {
    Image im;
    im.width = im.height = size;
    im.rgba.assign(std::size_t(size) * std::size_t(size) * 4, 0);
    return im;
}

float Sample(const std::vector<float>& f, int size, int x, int y) {
    return f[std::size_t(Wrap(y, size)) * std::size_t(size) + std::size_t(Wrap(x, size))];
}

}  // namespace

std::vector<float> Fbm(int size, std::uint32_t seed, int period_x, int period_y,
                       int octaves, float gain) {
    size = std::max(size, 1);
    period_x = std::max(period_x, 1);
    period_y = std::max(period_y, 1);
    octaves = std::clamp(octaves, 1, 12);

    std::vector<float> out(std::size_t(size) * std::size_t(size), 0.0f);
    float total = 0.0f, amplitude = 1.0f;
    int px = period_x, py = period_y;
    for (int o = 0; o < octaves; ++o) {
        const float sx = float(px) / float(size), sy = float(py) / float(size);
        for (int y = 0; y < size; ++y)
            for (int x = 0; x < size; ++x)
                out[std::size_t(y) * std::size_t(size) + std::size_t(x)] +=
                    TileNoise(float(x) * sx, float(y) * sy, px, py,
                              seed + std::uint32_t(o) * 7919u) *
                    amplitude;
        total += amplitude;
        amplitude *= gain;
        // Each octave doubles the lattice. Capped at the image size: a lattice
        // finer than one cell per texel is pure hash with no interpolation,
        // which is white noise and aliases the moment it is minified.
        if (px * 2 > size || py * 2 > size) break;
        px *= 2;
        py *= 2;
    }
    if (total > 0.0f)
        for (float& v : out) v /= total;
    return out;
}

Image NormalFromHeight(const std::vector<float>& height, int size, float strength) {
    Image im = Blank(std::max(size, 1));
    if (height.size() < std::size_t(size) * std::size_t(size)) return im;

    // The slope is per TEXEL, and a texel is 1/size of the tile, so the
    // difference has to be scaled by `size` to become a gradient in tile units.
    // Leaving it out makes the map's depth depend on its resolution -- the same
    // surface at 1024 comes out four times flatter than at 256, which reads as
    // the higher-resolution texture being wrong.
    const float scale = strength * float(size) * 0.5f;
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x) {
            const float dx =
                (Sample(height, size, x + 1, y) - Sample(height, size, x - 1, y)) * scale;
            const float dy =
                (Sample(height, size, x, y + 1) - Sample(height, size, x, y - 1)) * scale;
            // The surface is z = h(x, y), so its tangents are (1, 0, dx) and
            // (0, 1, dy) and the normal is their cross product: (-dx, -dy, 1).
            Vec3 n = Normalize(Vec3{-dx, -dy, 1.0f});
            Put(im, x, y, Vec3{n.x * 0.5f + 0.5f, n.y * 0.5f + 0.5f, n.z * 0.5f + 0.5f});
        }
    return im;
}

Surface Ground(int size, std::uint32_t seed, Vec3 dry, float blade) {
    Surface s;
    size = std::max(size, 4);

    // THREE SCALES, because that is what makes ground read as ground: broad
    // patches of dry and damp that the eye picks up from across the field, a
    // clump structure at arm's length, and blades at the texel level. One scale
    // of noise looks like a cloud whatever you do to its colour.
    const std::vector<float> patch = Fbm(size, seed, 8, 8, 3, 0.5f);
    const std::vector<float> clump = Fbm(size, seed + 101u, 16, 16, 3, 0.55f);
    const std::vector<float> fine = Fbm(size, seed + 202u, 64, 64, 3, 0.6f);

    s.albedo = Blank(size);
    std::vector<float> height(std::size_t(size) * std::size_t(size));
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x) {
            const std::size_t i = std::size_t(y) * std::size_t(size) + std::size_t(x);
            // Centred on zero so the product below has a mean of one: this map
            // multiplies the material's base colour, and a map with a mean of
            // 0.7 would silently darken every material that used it by a third.
            const float p = patch[i] - 0.5f;
            const float c = clump[i] - 0.5f;
            const float f = fine[i] - 0.5f;

            const float value = 1.0f + c * 0.26f + f * blade;
            // Dryness follows the BROAD scale only. Tying hue to the fine scale
            // makes every blade a different colour, which is confetti.
            const float dryness = std::clamp(p * 1.1f + 0.5f, 0.0f, 1.0f);
            const Vec3 tint{1.0f + (dry.x - 1.0f) * dryness,
                            1.0f + (dry.y - 1.0f) * dryness,
                            1.0f + (dry.z - 1.0f) * dryness};
            Put(s.albedo, x, y, Vec3{value * tint.x, value * tint.y, value * tint.z});

            // The height the normal map is built from is the clump and blade
            // structure, NOT the dry patches: a dry patch is a change of colour,
            // not of shape, and giving it relief makes the ground look quilted.
            height[i] = clump[i] * 0.45f + fine[i] * 0.55f;
        }
    s.normal = NormalFromHeight(height, size, 0.012f);
    return s;
}

Surface Bark(int size, std::uint32_t seed) {
    Surface s;
    size = std::max(size, 4);
    s.albedo = Blank(size);

    // 8 across and 2 down, so features are FOUR TIMES LONGER along v than
    // across u. This is the stretch the ridge formula below assumes; without it
    // both noise fields are isotropic and drag the grain back toward round --
    // measured at 1.53x directional against the 2x a groove needs, when this
    // said 8 and 8.
    const std::vector<float> grain = Fbm(size, seed, 8, 2, 4, 0.55f);
    const std::vector<float> rough = Fbm(size, seed + 313u, 32, 8, 3, 0.5f);

    std::vector<float> height(std::size_t(size) * std::size_t(size));
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x) {
            const std::size_t i = std::size_t(y) * std::size_t(size) + std::size_t(x);
            const float u = float(x) / float(size);

            // RIDGES ACROSS u, held along v. u runs around the trunk and v runs
            // up it, so a feature that varies in u and not in v is a vertical
            // groove -- which is what bark is. The noise fields are stretched
            // 6:1 along v rather than dropped, because a perfectly straight
            // groove reads as a machined column.
            const float wander = (grain[i] - 0.5f) * 0.22f;
            // Sharpened with a power so the grooves are narrow and the flats
            // between them are wide, which is the profile of real bark; a plain
            // sinusoid gives corduroy.
            const float ridge = std::pow(
                std::fabs(std::sin((u + wander) * 3.14159265f * 9.0f)), 0.45f);

            const float detail = (rough[i] - 0.5f) * 0.30f;
            // Grooves are DARKER, because they are in shadow and hold dirt.
            const float value = 0.72f + ridge * 0.42f + detail;
            Put(s.albedo, x, y, Vec3{value, value * 0.98f, value * 0.94f});
            height[i] = ridge * 0.7f + rough[i] * 0.3f;
        }
    s.normal = NormalFromHeight(height, size, 0.020f);
    return s;
}

Image Foliage(int size, std::uint32_t seed) {
    size = std::max(size, 4);
    Image im = Blank(size);

    const std::vector<float> mass = Fbm(size, seed, 6, 6, 3, 0.5f);
    const std::vector<float> leaf = Fbm(size, seed + 77u, 40, 40, 3, 0.6f);

    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x) {
            const std::size_t i = std::size_t(y) * std::size_t(size) + std::size_t(x);
            const float m = mass[i] - 0.5f, l = leaf[i] - 0.5f;
            const float value = 1.0f + m * 0.34f + l * 0.46f;
            // Lighter leaves are YELLOWER, which is what new growth on the
            // sunlit outside of a canopy looks like. Only the red channel is
            // lifted: the first version pushed blue DOWN by the same amount to
            // make the shaded leaves cooler, and shaded leaves are not blue --
            // they are darker green. It read as mould.
            const float warm = std::max(value - 1.0f, 0.0f) * 0.45f;
            Put(im, x, y, Vec3{value * (1.0f + warm), value * (1.0f + warm * 0.35f),
                               value});
        }
    return im;
}

}  // namespace eng::texgen
