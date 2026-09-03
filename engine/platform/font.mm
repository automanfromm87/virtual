#include "engine/platform/font.h"

#import <CoreGraphics/CoreGraphics.h>
#import <CoreText/CoreText.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>

namespace eng::platform {

FontAtlas RasterizeFont(const std::string& name, float pixel_size,
                        std::string& error) {
    @autoreleasepool {
        FontAtlas atlas;
        error.clear();
        if (pixel_size < 1.0f) {
            error = "font size must be at least one pixel";
            return atlas;
        }

        CTFontRef font = nullptr;
        if (!name.empty()) {
            CFStringRef cf = CFStringCreateWithCString(nullptr, name.c_str(),
                                                       kCFStringEncodingUTF8);
            font = CTFontCreateWithName(cf, pixel_size, nullptr);
            CFRelease(cf);
        }
        // A missing font costs the wrong typeface, not the whole HUD.
        if (!font) font = CTFontCreateUIFontForLanguage(kCTFontUIFontUser,
                                                        pixel_size, nullptr);
        if (!font) {
            error = "could not create a font";
            return atlas;
        }

        atlas.ascent = float(CTFontGetAscent(font));
        atlas.line_height = float(CTFontGetAscent(font) + CTFontGetDescent(font) +
                                  CTFontGetLeading(font));

        constexpr int kCount = FontAtlas::kLast - FontAtlas::kFirst + 1;
        UniChar chars[kCount];
        CGGlyph glyphs[kCount];
        for (int i = 0; i < kCount; ++i) chars[i] = UniChar(FontAtlas::kFirst + i);
        CTFontGetGlyphsForCharacters(font, chars, glyphs, kCount);

        CGRect bounds[kCount];
        CGSize advances[kCount];
        CTFontGetBoundingRectsForGlyphs(font, kCTFontOrientationHorizontal, glyphs,
                                        bounds, kCount);
        CTFontGetAdvancesForGlyphs(font, kCTFontOrientationHorizontal, glyphs,
                                   advances, kCount);

        // A shelf pack: fill a row until it is full, then start the next. Not
        // the tightest possible arrangement, and for one row of similarly sized
        // glyphs the tightest arrangement IS a shelf -- a real packer earns its
        // complexity on sprite sheets, not on ASCII.
        constexpr int kPad = 1;  // so bilinear sampling cannot bleed neighbours
        int pen_x = kPad, pen_y = kPad, row_h = 0;
        const int width = 512;
        for (int i = 0; i < kCount; ++i) {
            const int w = int(std::ceil(bounds[i].size.width)) + 2;
            const int h = int(std::ceil(bounds[i].size.height)) + 2;
            if (pen_x + w + kPad > width) {
                pen_x = kPad;
                pen_y += row_h + kPad;
                row_h = 0;
            }
            atlas.glyphs[i].x = pen_x;
            atlas.glyphs[i].y = pen_y;
            atlas.glyphs[i].w = w;
            atlas.glyphs[i].h = h;
            // CoreText's y grows UP from the baseline; the atlas and every
            // consumer use y DOWN from the top left. The flip happens here so
            // that no drawing code has to remember it.
            atlas.glyphs[i].bearing_x = float(bounds[i].origin.x) - 1.0f;
            atlas.glyphs[i].bearing_y =
                -float(bounds[i].origin.y + bounds[i].size.height) - 1.0f;
            atlas.glyphs[i].advance = float(advances[i].width);
            pen_x += w + kPad;
            row_h = std::max(row_h, h);
        }
        const int height = pen_y + row_h + kPad;

        atlas.width = width;
        atlas.height = height;
        atlas.coverage.assign(std::size_t(width) * height, 0);

        CGColorSpaceRef grey = CGColorSpaceCreateDeviceGray();
        CGContextRef ctx = CGBitmapContextCreate(
            atlas.coverage.data(), std::size_t(width), std::size_t(height), 8,
            std::size_t(width), grey, kCGImageAlphaNone);
        CGColorSpaceRelease(grey);
        if (!ctx) {
            error = "could not create the rasterisation context";
            CFRelease(font);
            return FontAtlas{};
        }
        CGContextSetShouldAntialias(ctx, true);
        CGContextSetShouldSmoothFonts(ctx, false);  // grayscale, not subpixel
        CGContextSetGrayFillColor(ctx, 1.0, 1.0);
        // No text-matrix mirror and no flip of the finished bitmap. Both were
        // tried and each fixes half of the problem the other creates.
        //
        // CGBitmapContext has its coordinate origin at the BOTTOM left and
        // stores its rows starting from the TOP, so buffer_row = height - 1 -
        // device_y. That single relationship does everything needed: a glyph
        // drawn the right way up in device space lands the right way up in the
        // buffer, and a pen placed by the arithmetic below lands in the row it
        // was packed into.
        //
        // Flipping the finished bitmap "to convert bottom-up rows to top-down"
        // is the intuitive move and it is wrong twice over: it reverses the row
        // order out of correspondence with the packed rectangles AND mirrors
        // every letter. A mirrored 'o' is an 'o', so it reads as a font that
        // renders slightly badly rather than as an axis error.

        for (int i = 0; i < kCount; ++i) {
            if (atlas.glyphs[i].w <= 2) continue;  // space and friends
            // The pen in DEVICE space, placed so that buffer_row =
            // height - 1 - device_y puts the ink one pixel inside the rect
            // this glyph was packed into.
            const CGFloat baseline_y =
                CGFloat(height - atlas.glyphs[i].y - atlas.glyphs[i].h) -
                bounds[i].origin.y + 1.0;
            const CGPoint at = CGPointMake(
                CGFloat(atlas.glyphs[i].x) - bounds[i].origin.x + 1.0, baseline_y);
            CTFontDrawGlyphs(font, &glyphs[i], &at, 1, ctx);
        }
        CGContextRelease(ctx);
        CFRelease(font);

        // NO FLIP. CGBitmapContext's coordinate origin is bottom left, but the
        // buffer it writes stores the TOP row first -- so the data is already
        // in the y-down order everything downstream wants, and flipping it
        // reverses the rows out of correspondence with the rectangles recorded
        // during packing.
        //
        // The two conventions cancelling is also why glyphs are drawn mirrored
        // above: the buffer's implicit flip un-mirrors them.
        return atlas;
    }
}

}  // namespace eng::platform
