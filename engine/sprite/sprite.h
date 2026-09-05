// 2D sprites: a CPU batch turning textured quads into one vertex buffer.
//
// Phase 1 of the 2D gameplay set (batch, then flipbook, tilemap, 2D
// physics, demo). This layer answers one question: given N quads with a
// texture id, a uv subrect, a tint and a layer, what are the vertices, in
// what order. It does not upload, draw, or know what a texture id means --
// the caller maps ids to GPU textures and flushes one batch per id, which
// is why the bake sorts by (texture, layer).
//
// Coordinates are world metres on the z = 0 plane, +Y up, counter-clockwise
// winding facing +Z. UVs share VertexIn's top-left origin: (u0, v0) is the
// subrect's top-left. Rotation is counter-clockwise about the center.
#pragma once

#include <cstdint>
#include <vector>

#include "engine/core/math.h"
#include "engine/geometry/mesh.h"
#include "engine/shaders/shader_types.h"

namespace eng::sprite {

struct Sprite {
    Vec2 center{0.0f, 0.0f};
    Vec2 size{1.0f, 1.0f};
    // Radians, counter-clockwise, about the center. Zero is axis-aligned.
    float rotation = 0.0f;
    // (u0, v0, u1, v1): the atlas subrect, v0 its top edge.
    Vec4 uv{0.0f, 0.0f, 1.0f, 1.0f};
    Vec4 tint{1.0f, 1.0f, 1.0f, 1.0f};
    // Caller-defined atlas id. The bake sorts by it, so ids that share a GPU
    // texture draw in one run with no rebinds in between.
    int texture = 0;
    // Draw order within one texture, low first. Higher layers overdraw lower
    // ones once the caller enables alpha blending.
    int layer = 0;
};

class SpriteBatch {
  public:
    void Push(const Sprite& s);
    void Clear();
    // Stable-sorts pushes by (texture, layer) and bakes 4 vertices and 6
    // indices per sprite. Stable, so ties keep push order -- two sprites on
    // one layer draw in the order they were pushed, which is the only sane
    // rule for that case.
    void Bake();

    [[nodiscard]] std::size_t Count() const;
    [[nodiscard]] const std::vector<VertexIn>& vertices() const;
    [[nodiscard]] const std::vector<std::uint32_t>& indices() const;
    // The baked quads as a Mesh, with bounds fitted. An empty batch yields
    // an empty mesh with a zero-radius bound.
    [[nodiscard]] Mesh BuildMesh() const;

  private:
    std::vector<Sprite> pending_;
    std::vector<VertexIn> vertices_;
    std::vector<std::uint32_t> indices_;
};

}  // namespace eng::sprite
