#include "engine/text/ui.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace eng::ui {
namespace {

constexpr char kUiSrc[] = {
#embed "engine/shaders/ui.metal"
    , 0};

struct UiVertex {
    float x = 0.0f, y = 0.0f;
    float u = 0.0f, v = 0.0f;
    Vec4 colour{1, 1, 1, 1};
};
static_assert(sizeof(UiVertex) == 32, "UiVertex layout must match the shader");

// Six vertices a quad, not four plus an index buffer.
//
// An index buffer saves a third of the vertex memory and costs a second buffer
// to keep in step, an extra binding, and index arithmetic at every append. At
// these counts -- a HUD is hundreds of quads, not hundreds of thousands -- the
// memory is irrelevant and the simplicity is not.
constexpr int kVertsPerQuad = 6;
constexpr int kMaxQuads = 8192;

}  // namespace

struct Canvas::Impl {
    rhi::Device* dev = nullptr;
    rhi::PipelineId pipeline;
    rhi::TextureId atlas;
    rhi::SamplerId sampler;
    rhi::BufferId vertices;
    std::uint8_t* mapped = nullptr;
    std::size_t slot_bytes = 0;

    platform::FontAtlas font;
    // Where the one fully-opaque texel is, for solid quads. See Create.
    float white_u = 0.0f, white_v = 0.0f;

    std::vector<UiVertex> queued;
    int width = 0, height = 0;
    int overflowed = 0;
};

Canvas::Canvas() : impl_(std::make_unique<Impl>()) {}
Canvas::~Canvas() = default;

int Canvas::QuadCount() const {
    return int(impl_->queued.size()) / kVertsPerQuad;
}
int Canvas::Overflowed() const { return impl_->overflowed; }

std::unique_ptr<Canvas> Canvas::Create(rhi::Device& dev,
                                       const platform::FontAtlas& font,
                                       rhi::Format color, std::string& error,
                                       int samples) {
    if (!font.Valid()) {
        error = "the font atlas is empty";
        return nullptr;
    }
    std::unique_ptr<Canvas> c(new Canvas());
    Impl& im = *c->impl_;
    im.dev = &dev;
    im.font = font;

    // A WHITE TEXEL, added to the atlas so solid rectangles can use the text
    // pipeline. One shader, one binding, no branch: a solid quad points every
    // one of its corners at this texel and the multiply is by one.
    //
    // In the bottom-right corner, which the shelf packer leaves empty because
    // it starts a new row rather than filling the last one.
    std::vector<std::uint8_t> pixels(font.coverage);
    const int wx = font.width - 2, wy = font.height - 2;
    if (wx < 0 || wy < 0) {
        error = "the font atlas is too small to hold a white texel";
        return nullptr;
    }
    for (int y = wy; y < font.height; ++y)
        for (int x = wx; x < font.width; ++x)
            pixels[std::size_t(y) * font.width + x] = 255;
    im.white_u = (float(wx) + 0.5f) / float(font.width);
    im.white_v = (float(wy) + 0.5f) / float(font.height);

    // Widened to RGBA, because the RHI uploads RGBA8 and a one-channel texture
    // format is a feature the renderer has never needed. Four times the memory
    // for a 512x140 atlas is a quarter of a megabyte.
    std::vector<std::uint8_t> rgba(std::size_t(font.width) * font.height * 4);
    for (std::size_t i = 0; i < pixels.size(); ++i) {
        rgba[i * 4 + 0] = pixels[i];
        rgba[i * 4 + 1] = pixels[i];
        rgba[i * 4 + 2] = pixels[i];
        rgba[i * 4 + 3] = pixels[i];
    }
    im.atlas = dev.CreateTexture2D(font.width, font.height, rgba.data());
    if (!Valid(im.atlas)) {
        error = "could not upload the font atlas";
        return nullptr;
    }
    im.sampler = dev.CreateSampler(rhi::Filter::Linear, rhi::Wrap::Clamp);

    im.slot_bytes = sizeof(UiVertex) * kMaxQuads * kVertsPerQuad;
    im.vertices = dev.CreateDynamicBuffer(im.slot_bytes * rhi::kFramesInFlight);
    im.mapped = static_cast<std::uint8_t*>(dev.MapBuffer(im.vertices));
    if (!Valid(im.vertices) || !im.mapped) {
        error = "could not allocate the ui vertex buffer";
        return nullptr;
    }

    rhi::PipelineDesc pd;
    pd.source = kUiSrc;
    pd.vertex_fn = "vs_ui";
    pd.fragment_fn = "fs_ui";
    pd.color = color;
    pd.samples = samples;
    // No depth at all. A HUD is drawn last, over everything, in the order it
    // was queued -- testing it against the scene's depth would hide it behind
    // whatever the player is standing in front of.
    pd.depth = false;
    pd.depth_write = false;
    pd.blend = rhi::Blend::Alpha;
    im.pipeline = dev.CreatePipeline(pd, error);
    if (!Valid(im.pipeline)) return nullptr;
    return c;
}

void Canvas::Begin(int width, int height) {
    impl_->width = width;
    impl_->height = height;
    impl_->queued.clear();
    impl_->overflowed = 0;
}

void Canvas::Rect(float x, float y, float w, float h, Vec4 colour) {
    Impl& im = *impl_;
    if (int(im.queued.size()) / kVertsPerQuad >= kMaxQuads) {
        ++im.overflowed;
        return;
    }
    const float u = im.white_u, v = im.white_v;
    const UiVertex a{x, y, u, v, colour};
    const UiVertex b{x + w, y, u, v, colour};
    const UiVertex c{x, y + h, u, v, colour};
    const UiVertex d{x + w, y + h, u, v, colour};
    im.queued.insert(im.queued.end(), {a, b, c, c, b, d});
}

void Canvas::Outline(float x, float y, float w, float h, float t, Vec4 colour) {
    Rect(x, y, w, t, colour);
    Rect(x, y + h - t, w, t, colour);
    // The verticals are inset by the thickness so the corners are not drawn
    // twice: with a translucent colour, a doubled corner is visibly darker.
    Rect(x, y + t, t, h - 2 * t, colour);
    Rect(x + w - t, y + t, t, h - 2 * t, colour);
}

float Canvas::Measure(std::string_view text, float scale) const {
    float w = 0.0f;
    for (char ch : text) {
        const platform::Glyph* g = impl_->font.Find(ch);
        if (g) w += g->advance * scale;
    }
    return w;
}

float Canvas::LineHeight(float scale) const {
    return impl_->font.line_height * scale;
}

void Canvas::Text(float x, float y, std::string_view text, Vec4 colour,
                  Align align, float scale) {
    Impl& im = *impl_;
    if (align == Align::Centre) x -= Measure(text, scale) * 0.5f;
    if (align == Align::Right) x -= Measure(text, scale);

    // `y` is the TOP of the line, not the baseline. A caller laying out a HUD
    // thinks in boxes and would otherwise have to add the ascent at every call
    // -- and would forget at one of them.
    const float baseline = y + im.font.ascent * scale;

    float pen = x;
    for (char ch : text) {
        const platform::Glyph* g = im.font.Find(ch);
        if (!g) continue;
        if (g->w > 0 && g->h > 0) {
            if (int(im.queued.size()) / kVertsPerQuad >= kMaxQuads) {
                ++im.overflowed;
                return;
            }
            const float gx = pen + g->bearing_x * scale;
            const float gy = baseline + g->bearing_y * scale;
            const float gw = float(g->w) * scale, gh = float(g->h) * scale;
            const float u0 = float(g->x) / float(im.font.width);
            const float v0 = float(g->y) / float(im.font.height);
            const float u1 = float(g->x + g->w) / float(im.font.width);
            const float v1 = float(g->y + g->h) / float(im.font.height);
            const UiVertex a{gx, gy, u0, v0, colour};
            const UiVertex b{gx + gw, gy, u1, v0, colour};
            const UiVertex c{gx, gy + gh, u0, v1, colour};
            const UiVertex d{gx + gw, gy + gh, u1, v1, colour};
            im.queued.insert(im.queued.end(), {a, b, c, c, b, d});
        }
        pen += g->advance * scale;
    }
}

void Canvas::Draw(rhi::Encoder& enc) {
    Impl& im = *impl_;
    if (im.queued.empty() || im.width <= 0 || im.height <= 0) return;

    const std::size_t offset =
        std::size_t(im.dev->FrameSlot()) * im.slot_bytes;
    std::memcpy(im.mapped + offset, im.queued.data(),
                im.queued.size() * sizeof(UiVertex));

    const float viewport[2] = {float(im.width), float(im.height)};
    enc.SetPipeline(im.pipeline);
    enc.SetCull(rhi::Cull::None, rhi::Winding::CounterClockwise);
    enc.SetVertexBuffer(im.vertices, offset, 0);
    enc.SetVertexBytes(viewport, sizeof(viewport), 1);
    enc.SetFragmentTexture(im.atlas, 0);
    enc.SetFragmentSampler(im.sampler, 0);
    enc.Draw(im.queued.size());
}

}  // namespace eng::ui
