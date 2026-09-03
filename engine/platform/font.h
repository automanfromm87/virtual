// Rasterising a system font into an atlas.
//
// Here rather than in a text module because it is the one part of drawing text
// that is genuinely the OS's business: hinting, kerning and the outlines
// themselves live in CoreText, and reimplementing them to avoid an Apple header
// would mean shipping a worse font renderer for no gain. Everything ABOVE this
// -- layout, batching, the shader -- is pure C++ and does not know CoreText
// exists.
//
// The alternative was embedding a bitmap font. It needs no OS at all and looks
// like 1985 at any size but the one it was drawn for.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace eng::platform {

// One glyph's place in the atlas and how to position it.
struct Glyph {
    // Pixel rectangle within the atlas.
    int x = 0, y = 0, w = 0, h = 0;
    // Where to put that rectangle relative to the pen, in pixels, with y DOWN.
    // `bearing_y` is usually negative: most of a glyph sits above the baseline.
    float bearing_x = 0.0f, bearing_y = 0.0f;
    // How far the pen moves afterwards.
    float advance = 0.0f;
};

// A rasterised font: one 8-bit coverage atlas and the glyphs in it.
//
// COVERAGE, not colour. Text is drawn by multiplying a colour by this, so one
// atlas serves every colour anything is ever drawn in -- and it is a quarter of
// the memory of RGBA for exactly the same result.
struct FontAtlas {
    int width = 0, height = 0;
    std::vector<std::uint8_t> coverage;  // width * height, one byte per pixel
    // Indexed by character code. ASCII only: everything above 127 needs a map
    // and a shaping pass, and a HUD needs neither.
    static constexpr int kFirst = 32, kLast = 126;
    Glyph glyphs[kLast - kFirst + 1];
    // Distance from one baseline to the next, and how far the tallest glyph
    // rises above the baseline. Both in pixels.
    float line_height = 0.0f;
    float ascent = 0.0f;

    [[nodiscard]] bool Valid() const { return width > 0 && !coverage.empty(); }
    [[nodiscard]] const Glyph* Find(char c) const {
        const int i = int(static_cast<unsigned char>(c));
        if (i < kFirst || i > kLast) return nullptr;
        return &glyphs[i - kFirst];
    }
};

// `name` is a PostScript or family name -- "Menlo", "Helvetica". An unknown one
// falls back to the system font rather than failing: a missing font should cost
// the wrong typeface, not the whole HUD.
[[nodiscard]] FontAtlas RasterizeFont(const std::string& name, float pixel_size,
                                      std::string& error);

}  // namespace eng::platform
