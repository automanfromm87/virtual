// Immediate-mode 2D drawing: text, rectangles, lines, on top of the frame.
//
// IMMEDIATE MODE means there is no retained widget tree. A caller says what it
// wants on screen this frame and says it again next frame, and the layer's only
// job is to turn that into one vertex buffer and one draw. Nothing is allocated
// per widget, nothing has to be destroyed, and a HUD that shows a different
// number every frame costs nothing extra -- which is the case a retained tree
// is worst at and a HUD is entirely made of.
//
// Coordinates are PIXELS with the origin at the TOP LEFT, matching the
// framebuffer and the cursor position, so a caller can hit-test a button
// against the mouse without converting anything.
#pragma once

#include <memory>
#include <string>
#include <string_view>

#include "engine/core/math.h"
#include "engine/platform/font.h"
#include "engine/rhi/rhi.h"

namespace eng::ui {

// Where a run of text sits relative to the position given.
enum class Align : std::uint8_t { Left, Centre, Right };

class Canvas {
  public:
    [[nodiscard]] static std::unique_ptr<Canvas> Create(
        rhi::Device&, const platform::FontAtlas&, rhi::Format color,
        std::string& error, int samples = 1);
    ~Canvas();

    Canvas(const Canvas&) = delete;
    Canvas& operator=(const Canvas&) = delete;

    // Once per frame, before anything is drawn. The size is the framebuffer's,
    // in pixels.
    void Begin(int width, int height);

    void Rect(float x, float y, float w, float h, Vec4 colour);
    // A one-pixel outline, drawn as four rectangles. Thin enough that a
    // dedicated line primitive would be a second pipeline for no gain.
    void Outline(float x, float y, float w, float h, float thickness, Vec4 colour);
    void Text(float x, float y, std::string_view text, Vec4 colour,
              Align align = Align::Left, float scale = 1.0f);

    // How wide a string would be, for laying out against it. Advances only --
    // no kerning pairs, because the atlas has none.
    [[nodiscard]] float Measure(std::string_view text, float scale = 1.0f) const;
    [[nodiscard]] float LineHeight(float scale = 1.0f) const;

    // Everything queued since Begin, in one draw. Must run in a render pass
    // with the same colour format the canvas was created for.
    void Draw(rhi::Encoder&);

    [[nodiscard]] int QuadCount() const;
    // Quads dropped because the frame's buffer filled. Non-zero means the HUD
    // is silently incomplete, which is exactly the failure a caller cannot see.
    [[nodiscard]] int Overflowed() const;

  private:
    Canvas();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eng::ui
