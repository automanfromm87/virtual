// Pure C++20. CPU-side pixel buffers and the procedural generators that fill
// them. No Metal, no file IO — there is no asset pipeline yet, so every texture
// in the engine is computed.
#pragma once

#include <cstdint>
#include <vector>

#include "engine/core/math.h"

namespace eng {

// Tightly packed RGBA8, row-major, first row is the TOP row — matching Metal's
// texture origin so an upload needs no flip.
struct Texture2D {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgba;  // width * height * 4

    [[nodiscard]] bool Empty() const { return rgba.empty(); }
};

// `squares` alternating tiles per axis. The classic albedo test pattern: any
// uv error — wrong scale, swapped axes, a flipped v — is instantly readable.
[[nodiscard]] Texture2D MakeChecker(int size, int squares, Vec4 a, Vec4 b);

// Horizontal bands running from roughness 0 at the top to 1 at the bottom,
// written into the RED channel. Point a PBR shader at this and the whole
// mirror-to-matte range shows up on a single object.
[[nodiscard]] Texture2D MakeRoughnessRamp(int size, int bands);

// u in red, v in green. Not decorative: it makes uv orientation and continuity
// directly visible, which is otherwise guesswork.
[[nodiscard]] Texture2D MakeUVDebug(int size);

}  // namespace eng
