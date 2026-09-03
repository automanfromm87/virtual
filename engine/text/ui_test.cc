// The UI canvas, rendered offscreen and measured.
//
// A HUD is the one part of a renderer where being a few pixels wrong is the
// whole bug: text half a line too high, a panel that does not line up with its
// border, a right-aligned number that drifts as it changes width. None of it is
// visible in a screenshot of a working build and all of it is measurable.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/platform/font.h"
#include "engine/rhi/rhi.h"
#include "engine/text/ui.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-60s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

constexpr int kW = 640, kH = 360;

struct Image {
    std::vector<std::uint8_t> px;
    int Ink(int x0, int y0, int x1, int y1) const {
        int n = 0;
        for (int y = std::max(0, y0); y < std::min(kH, y1); ++y)
            for (int x = std::max(0, x0); x < std::min(kW, x1); ++x)
                if (px[(std::size_t(y) * kW + x) * 4] > 40) ++n;
        return n;
    }
};

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("ui canvas\n");

    std::string error;
    auto dev = eng::rhi::Device::Create(error);
    if (!dev) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
    const auto kFmt = eng::rhi::Format::RGBA8Unorm;

    const eng::platform::FontAtlas font =
        eng::platform::RasterizeFont("Menlo", 24.0f, error);
    Check(font.Valid(), "the font rasterised");

    auto ui = eng::ui::Canvas::Create(*dev, font, kFmt, error, 1);
    if (!ui) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
    Check(true, "the canvas was created");

    const eng::rhi::TextureId out = dev->CreateRenderTarget(kW, kH, kFmt, true);
    Check(Valid(out), "the target was created");

    Image img;
    const auto render = [&](const auto& build) {
        ui->Begin(kW, kH);
        build();
        eng::rhi::PassDesc pd;
        pd.color = out;
        pd.clear_color[0] = 0.0f; pd.clear_color[1] = 0.0f;
        pd.clear_color[2] = 0.0f; pd.clear_color[3] = 1.0f;
        dev->BeginFrame();
        eng::rhi::Encoder e = dev->BeginPass(pd);
        ui->Draw(e);
        dev->EndPass();
        std::string w;
        if (!dev->CommitAndWait(w)) std::fprintf(stderr, "  %s\n", w.c_str());
        img.px.assign(std::size_t(kW) * kH * 4, 0);
        (void)dev->ReadPixels(out, kW, kH, img.px);
    };

    // --- a solid rectangle ---------------------------------------------------
    //
    // The simplest thing the canvas draws, and it pins the coordinate system:
    // pixels, origin at the TOP LEFT. A y-flip here would put the rectangle at
    // the bottom and look entirely reasonable on its own.
    render([&] { ui->Rect(100, 40, 200, 60, eng::Vec4{1, 0, 0, 1}); });
    {
        const auto at = [&](int x, int y) {
            const std::size_t i = (std::size_t(y) * kW + x) * 4;
            return std::array<int, 3>{img.px[i], img.px[i + 1], img.px[i + 2]};
        };
        const auto in = at(200, 70), above = at(200, 20), below = at(200, 200);
        std::printf("    inside (%d,%d,%d), above (%d,%d,%d), below (%d,%d,%d)\n",
                    in[0], in[1], in[2], above[0], above[1], above[2], below[0],
                    below[1], below[2]);
        Check(in[0] > 240 && in[1] < 15 && in[2] < 15,
              "a solid rect is exactly the colour asked for");
        Check(above[0] < 10 && below[0] < 10, "and does not bleed outside itself");
        // Its EDGES are where they were asked for, to the pixel.
        Check(img.Ink(99, 39, 100, 100) == 0, "its left edge is at x = 100");
        Check(img.Ink(100, 39, 101, 100) > 0, "and starts there");
        Check(img.Ink(299, 39, 300, 100) > 0, "its right edge is at x = 300");
        Check(img.Ink(300, 39, 301, 100) == 0, "and stops there");
    }

    // --- text ------------------------------------------------------------------
    render([&] {
        ui->Text(40, 40, "Hamburgefonstiv", eng::Vec4{1, 1, 1, 1});
    });
    {
        const float w = ui->Measure("Hamburgefonstiv");
        // The drawn extent must match what Measure promised. A caller lays a
        // panel out against Measure and then draws the text; if the two
        // disagree the text runs off its own background.
        int first = kW, last = 0;
        for (int x = 0; x < kW; ++x)
            if (img.Ink(x, 0, x + 1, kH) > 0) { first = std::min(first, x); last = x; }
        std::printf("    Measure says %.1f px, drawn ink spans %d..%d (%d px)\n", w,
                    first, last, last - first);
        Check(first >= 39 && first <= 44, "text starts where it was placed");
        // Slack at the right for the final glyph's side bearing: the advance
        // includes it and the ink does not.
        Check(std::fabs(float(last - 40) - w) < 6.0f,
              "and its drawn width matches Measure");

        Check(img.Ink(40, 40, kW, 40 + int(ui->LineHeight() * 1.4f)) > 200,
              "the text drew something");
    }

    // --- NOT MIRRORED, through the whole path -----------------------------------
    //
    // The atlas being the right way up is tested next door. This is the second
    // chance to flip it: the uv rectangle handed to the shader. Getting that
    // wrong produces upside-down text from a perfectly correct atlas.
    //
    // Comparing a line of mixed case against its own midpoint does NOT work and
    // was tried: "Hamburgefonstiv" is mostly lowercase, whose x-height sits
    // below the middle of the line, so the correct rendering is bottom-heavy
    // and the check failed on working code. 'T' against 'L' has the asymmetry
    // in the letters themselves rather than in the choice of word.
    {
        const auto ink_split = [&](const char* text) {
            render([&] { ui->Text(60, 60, text, eng::Vec4{1, 1, 1, 1}, 
                                  eng::ui::Align::Left, 3.0f); });
            int top = kH, bottom = 0;
            for (int y = 0; y < kH; ++y)
                if (img.Ink(0, y, kW, y + 1) > 0) { top = std::min(top, y); bottom = y; }
            const int mid = (top + bottom) / 2;
            return std::pair<int, int>{img.Ink(0, top, kW, mid),
                                       img.Ink(0, mid, kW, bottom + 1)};
        };
        const auto t = ink_split("TTT");
        const auto l = ink_split("LLL");
        std::printf("    'TTT' ink %d above / %d below its own middle; "
                    "'LLL' %d / %d\n", t.first, t.second, l.first, l.second);
        Check(t.first > t.second, "'T' renders top-heavy");
        Check(l.second > l.first, "'L' renders bottom-heavy");
    }

    // --- alignment ---------------------------------------------------------------
    //
    // Right alignment is the one a HUD actually needs -- a score that grows a
    // digit must not shove its own label sideways -- and it is the one that is
    // silently wrong if Measure and Text disagree.
    for (int digits = 1; digits <= 4; ++digits) {
        const std::string s(std::size_t(digits), '8');
        render([&] {
            ui->Text(500, 100, s, eng::Vec4{1, 1, 1, 1}, eng::ui::Align::Right);
        });
        int last = 0;
        for (int x = 0; x < kW; ++x)
            if (img.Ink(x, 90, x + 1, 130) > 0) last = x;
        std::printf("    %d digits right-aligned at 500: ink ends at %d\n", digits,
                    last);
        Check(last >= 494 && last <= 500, "right-aligned text ends where it was put");
    }
    {
        render([&] {
            ui->Text(320, 100, "centred", eng::Vec4{1, 1, 1, 1},
                     eng::ui::Align::Centre);
        });
        int first = kW, last = 0;
        for (int x = 0; x < kW; ++x)
            if (img.Ink(x, 90, x + 1, 130) > 0) { first = std::min(first, x); last = x; }
        const int centre = (first + last) / 2;
        std::printf("    centred text spans %d..%d, midpoint %d (asked for 320)\n",
                    first, last, centre);
        Check(std::abs(centre - 320) < 6, "centred text is centred");
    }

    // --- overflow ----------------------------------------------------------------
    //
    // The buffer is finite. Running out must be REPORTED, because a HUD that
    // silently loses its last few labels looks like a HUD that was never asked
    // to draw them.
    {
        ui->Begin(kW, kH);
        for (int i = 0; i < 20000; ++i) ui->Rect(0, 0, 1, 1, eng::Vec4{1, 1, 1, 1});
        std::printf("    20000 quads into an 8192 buffer: %d queued, %d dropped\n",
                    ui->QuadCount(), ui->Overflowed());
        Check(ui->QuadCount() <= 8192, "the queue does not exceed its capacity");
        Check(ui->Overflowed() > 0, "and says how much it dropped");
    }

    // --- a whole HUD, to look at ---------------------------------------------
    render([&] {
        ui->Rect(20, 20, 260, 100, eng::Vec4{0.05f, 0.06f, 0.10f, 0.85f});
        ui->Outline(20, 20, 260, 100, 1.0f, eng::Vec4{0.45f, 0.6f, 0.8f, 1});
        ui->Text(34, 30, "coins  6 / 8", eng::Vec4{1, 0.92f, 0.55f, 1});
        ui->Text(34, 58, "120 fps", eng::Vec4{0.75f, 0.82f, 0.9f, 1});
        ui->Text(34, 84, "grounded", eng::Vec4{0.6f, 0.7f, 0.8f, 1});
        ui->Text(320, 300, "all collected  -  r to play again",
                 eng::Vec4{1, 0.95f, 0.7f, 1}, eng::ui::Align::Centre, 1.2f);
    });
    {
        // The panel is translucent over black, so its interior is dim but not
        // zero, while the text on it is bright. Both must be true: a canvas
        // that ignored alpha would make the panel opaque and one that ignored
        // the atlas would make the text solid blocks.
        const std::size_t inner = (std::size_t(115) * kW + 250) * 4;
        std::printf("    panel interior (%d,%d,%d)\n", img.px[inner],
                    img.px[inner + 1], img.px[inner + 2]);
        Check(img.px[inner + 2] > 8 && img.px[inner + 2] < 60,
              "a translucent panel is dim rather than opaque or absent");
        Check(img.Ink(34, 30, 280, 110) > 200, "and the text on it is bright");

        std::FILE* f = std::fopen("/tmp/ui.ppm", "wb");
        if (f) {
            std::fprintf(f, "P6\n%d %d\n255\n", kW, kH);
            for (std::size_t i = 0; i + 3 < img.px.size(); i += 4)
                std::fwrite(&img.px[i], 1, 3, f);
            std::fclose(f);
            std::printf("    wrote /tmp/ui.ppm\n");
        }
    }

    std::printf(g_failures == 0 ? "\nui_test: all checks passed\n"
                                : "\nui_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
