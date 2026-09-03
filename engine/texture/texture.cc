#include "engine/texture/texture.h"

#include <algorithm>
#include <cmath>

namespace eng {
namespace {

std::uint8_t ToByte(float v) {
    const float c = std::clamp(v, 0.0f, 1.0f);
    return std::uint8_t(std::lround(c * 255.0f));
}

void Put(Texture2D& t, int x, int y, Vec4 c) {
    const std::size_t i = (std::size_t(y) * std::size_t(t.width) + std::size_t(x)) * 4;
    t.rgba[i + 0] = ToByte(c.x);
    t.rgba[i + 1] = ToByte(c.y);
    t.rgba[i + 2] = ToByte(c.z);
    t.rgba[i + 3] = ToByte(c.w);
}

Texture2D Alloc(int size) {
    Texture2D t;
    if (size <= 0) return t;
    t.width = size;
    t.height = size;
    t.rgba.resize(std::size_t(size) * std::size_t(size) * 4);
    return t;
}

}  // namespace

Texture2D MakeChecker(int size, int squares, Vec4 a, Vec4 b) {
    Texture2D t = Alloc(size);
    if (t.Empty() || squares <= 0) return {};
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            const int cx = (x * squares) / size;
            const int cy = (y * squares) / size;
            Put(t, x, y, ((cx + cy) & 1) ? b : a);
        }
    }
    return t;
}

Texture2D MakeRoughnessRamp(int size, int bands) {
    Texture2D t = Alloc(size);
    if (t.Empty() || bands <= 0) return {};
    for (int y = 0; y < size; ++y) {
        // Quantised into bands rather than a smooth gradient: discrete steps
        // make it obvious where each roughness value actually lands, where a
        // continuous ramp just looks like a soft blur.
        const int band = (y * bands) / size;
        const float roughness = bands > 1 ? float(band) / float(bands - 1) : 0.5f;
        for (int x = 0; x < size; ++x)
            Put(t, x, y, Vec4{roughness, 0.0f, 0.0f, 1.0f});
    }
    return t;
}

Texture2D MakeUVDebug(int size) {
    Texture2D t = Alloc(size);
    if (t.Empty()) return {};
    const float inv = size > 1 ? 1.0f / float(size - 1) : 0.0f;
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x)
            Put(t, x, y, Vec4{float(x) * inv, float(y) * inv, 0.0f, 1.0f});
    return t;
}

}  // namespace eng
