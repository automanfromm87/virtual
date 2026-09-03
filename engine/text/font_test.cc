// The font atlas, checked against what a rasteriser must produce.
//
// Text rendering fails in ways that all look like "the text is slightly wrong":
// glyphs packed on top of each other, a baseline off by the ascent, an atlas
// flipped vertically, a space with no advance so every word runs together. None
// of them are visible in a single letter and all of them are measurable.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/platform/font.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-60s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

// How much ink is inside a glyph's rectangle.
long Ink(const eng::platform::FontAtlas& a, const eng::platform::Glyph& g) {
    long sum = 0;
    for (int y = g.y; y < g.y + g.h && y < a.height; ++y)
        for (int x = g.x; x < g.x + g.w && x < a.width; ++x)
            sum += a.coverage[std::size_t(y) * a.width + x];
    return sum;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("font atlas\n");

    std::string error;
    const eng::platform::FontAtlas a =
        eng::platform::RasterizeFont("Menlo", 32.0f, error);
    if (!error.empty()) std::fprintf(stderr, "  %s\n", error.c_str());
    std::printf("    %dx%d atlas, line height %.2f, ascent %.2f\n", a.width,
                a.height, a.line_height, a.ascent);
    Check(a.Valid(), "the atlas was rasterised");
    Check(a.line_height > 32.0f && a.line_height < 64.0f,
          "the line height is about the pixel size");
    Check(a.ascent > 0.0f && a.ascent < a.line_height,
          "the ascent is positive and less than a line");

    // Every printable glyph fits INSIDE the atlas. A packer that runs off the
    // end produces glyphs that sample whatever is at the wrap, which reads as
    // random letters appearing inside other letters.
    bool inside = true;
    for (int c = eng::platform::FontAtlas::kFirst;
         c <= eng::platform::FontAtlas::kLast; ++c) {
        const auto* g = a.Find(char(c));
        if (!g) { inside = false; continue; }
        if (g->x < 0 || g->y < 0 || g->x + g->w > a.width ||
            g->y + g->h > a.height)
            inside = false;
    }
    Check(inside, "every glyph's rectangle is inside the atlas");

    // No two glyphs OVERLAP. The failure is subtle and permanent: two letters
    // share pixels, so each is drawn with a fragment of the other stuck to it.
    int overlaps = 0;
    for (int i = eng::platform::FontAtlas::kFirst;
         i <= eng::platform::FontAtlas::kLast; ++i) {
        const auto* p = a.Find(char(i));
        for (int j = i + 1; j <= eng::platform::FontAtlas::kLast; ++j) {
            const auto* q = a.Find(char(j));
            if (!p || !q || p->w == 0 || q->w == 0) continue;
            const bool apart = p->x + p->w <= q->x || q->x + q->w <= p->x ||
                               p->y + p->h <= q->y || q->y + q->h <= p->y;
            if (!apart) ++overlaps;
        }
    }
    std::printf("    %d overlapping glyph pairs\n", overlaps);
    Check(overlaps == 0, "no two glyphs share atlas pixels");

    // INK where there should be ink, and none where there should not.
    const auto* space = a.Find(' ');
    const auto* W = a.Find('W');
    const auto* period = a.Find('.');
    Check(space && W && period, "space, W and full stop all exist");
    if (space && W && period) {
        std::printf("    ink: 'W' %ld, '.' %ld, ' ' %ld\n", Ink(a, *W),
                    Ink(a, *period), Ink(a, *space));
        Check(Ink(a, *W) > 0, "'W' has ink");
        // A space is blank but still ADVANCES. A rasteriser that gives it no
        // advance runs every word together, and one that gives it ink draws a
        // block.
        Check(Ink(a, *space) == 0, "' ' has none");
        Check(space->advance > 1.0f, "and still moves the pen");
        // W is far more ink than a full stop. This catches an atlas that
        // rasterised every glyph into the same cell, which passes every check
        // above.
        // Measured at 49240 against 8845. The factor was 8 when it was
        // written, which passed only because the full stop was rendering as
        // nothing at all -- the bug this file went on to find.
        Check(Ink(a, *W) > Ink(a, *period) * 3, "'W' is much more ink than '.'");
    }

    // NOT MIRRORED. The atlas is drawn through two coordinate conventions --
    // CoreGraphics' origin is bottom left while its buffer starts at the top --
    // and getting them wrong produces a perfectly packed atlas of upside-down
    // letters. A mirrored 'o' is an 'o', so it reads as a font that renders
    // slightly badly rather than as an axis error, and no check on ink totals,
    // rectangles or overlaps notices it.
    //
    // 'T' is nearly all crossbar at the top; 'L' is nearly all foot at the
    // bottom. Their ink distributions are opposite, so between them they pin
    // the orientation.
    {
        const auto* T = a.Find('T');
        const auto* L = a.Find('L');
        const auto half_ink = [&](const eng::platform::Glyph& g, bool top) {
            long sum = 0;
            const int mid = g.y + g.h / 2;
            const int lo = top ? g.y : mid, hi = top ? mid : g.y + g.h;
            for (int y = lo; y < hi && y < a.height; ++y)
                for (int x = g.x; x < g.x + g.w && x < a.width; ++x)
                    sum += a.coverage[std::size_t(y) * a.width + x];
            return sum;
        };
        if (T && L) {
            const long t_top = half_ink(*T, true), t_bot = half_ink(*T, false);
            const long l_top = half_ink(*L, true), l_bot = half_ink(*L, false);
            std::printf("    'T' ink %ld above / %ld below its middle; "
                        "'L' %ld / %ld\n", t_top, t_bot, l_top, l_bot);
            // Measured at 1.96 and 1.89. A flipped atlas inverts both to about
            // 0.5, so 1.5 separates them by a wide margin without pretending
            // the letters are more lopsided than they are.
            Check(t_top > t_bot * 1.5, "'T' is top-heavy, so the atlas is not flipped");
            Check(l_bot > l_top * 1.5, "and 'L' is bottom-heavy");
        }
    }

    // Which glyphs came out blank?
    {
        std::string blank;
        for (int c = 33; c <= 126; ++c) {
            const auto* gg = a.Find(char(c));
            if (gg && Ink(a, *gg) == 0) blank += char(c);
        }
        std::printf("    glyphs with no ink: [%s]\n", blank.c_str());
        const auto* dot = a.Find('.');
        if (dot) std::printf("    '.' rect %dx%d at (%d,%d)\n", dot->w, dot->h,
                             dot->x, dot->y);
    }

    // MONOSPACE: Menlo is, so every printable glyph must advance the same
    // amount. A rasteriser that read the advances from the wrong array, or
    // reused one glyph's metrics for all, fails here and nowhere else.
    if (a.Valid()) {
        const float adv = a.Find('m')->advance;
        bool uniform = true;
        for (int c = 33; c <= 126; ++c)
            if (std::fabs(a.Find(char(c))->advance - adv) > 0.01f) uniform = false;
        std::printf("    Menlo advance %.4f px at 32 px\n", adv);
        Check(uniform, "a monospaced font advances every glyph equally");
        Check(adv > 8.0f && adv < 32.0f, "and by a sensible fraction of the size");
    }

    // The BASELINE. 'x' has no descender and sits on it; 'g' hangs below. Both
    // are expressed through bearing_y, which is measured DOWN from the pen --
    // so 'x' must be entirely above the baseline and 'g' must reach below it.
    const auto* x = a.Find('x');
    const auto* g = a.Find('g');
    if (x && g) {
        const float x_bottom = x->bearing_y + float(x->h);
        const float g_bottom = g->bearing_y + float(g->h);
        std::printf("    'x' spans %.1f..%.1f, 'g' spans %.1f..%.1f "
                    "(baseline at 0)\n", x->bearing_y, x_bottom, g->bearing_y,
                    g_bottom);
        Check(x_bottom <= 1.5f, "'x' sits on the baseline");
        Check(g_bottom > 2.0f, "'g' descends below it");
        Check(g->bearing_y > x->bearing_y - 2.0f,
              "and does not also start higher");
    }

    // A SCALED atlas is bigger in proportion. Catches a rasteriser that ignores
    // the size it was asked for, which is invisible at whatever size it does
    // use.
    std::string e2;
    const eng::platform::FontAtlas big =
        eng::platform::RasterizeFont("Menlo", 64.0f, e2);
    if (big.Valid() && a.Valid()) {
        const float ratio = big.Find('m')->advance / a.Find('m')->advance;
        std::printf("    doubling the pixel size scales the advance by %.3f\n",
                    ratio);
        Check(std::fabs(ratio - 2.0f) < 0.05f, "twice the size is twice the advance");
    }

    // An unknown font falls back rather than failing: a missing typeface should
    // cost the wrong letters, not the whole HUD.
    std::string e3;
    const eng::platform::FontAtlas fallback =
        eng::platform::RasterizeFont("NoSuchFontExistsAnywhere", 24.0f, e3);
    Check(fallback.Valid(), "an unknown font name falls back to the system font");

    // The atlas itself, so a failure can be looked at rather than deduced.
    if (a.Valid()) {
        std::FILE* f = std::fopen("/tmp/atlas.pgm", "wb");
        if (f) {
            std::fprintf(f, "P5\n%d %d\n255\n", a.width, a.height);
            std::fwrite(a.coverage.data(), 1, a.coverage.size(), f);
            std::fclose(f);
            std::printf("    wrote /tmp/atlas.pgm\n");
        }
    }

    std::printf(g_failures == 0 ? "\nfont_test: all checks passed\n"
                                : "\nfont_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
