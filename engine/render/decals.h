// Projected decals: bullet holes, scorch marks, puddles, posters.
//
// WHY THEY NEED A SYSTEM AT ALL. A decal is not an object -- it has no geometry
// of its own and it is not lit separately. It is a modification to the surface
// that is already there, applied after the G-buffer is filled and before
// anything is lit. That is a specific moment in the frame with nothing else in
// it, which is why this is its own class rather than a kind of Instance.
//
// DEFERRED ONLY, and that is a property of the technique rather than a gap. A
// projected decal writes into the G-buffer's albedo; a forward renderer has no
// G-buffer and shades each surface once as it draws it, so there is no point
// between "the surface exists" and "the surface is lit" at which to intervene.
// The alternative for a forward path is to re-draw the receiving geometry with
// the decal's texture and a depth-equal test, which needs to know which
// geometry receives -- exactly the knowledge projection exists to avoid needing.
#pragma once

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "engine/core/math.h"
#include "engine/rhi/rhi.h"
#include "engine/scene/scene.h"

namespace eng {

struct Decal {
    // Where the projection box is. The decal covers the box's XY extent and
    // projects along its local -Z, so `Mat4::Translation(hit) * orientation *
    // Mat4::Scale(...)` with -Z pointing INTO the surface is the usual
    // construction.
    Mat4 model = Mat4::Identity();
    Vec4 tint{1.0f, 1.0f, 1.0f, 1.0f};
    float opacity = 1.0f;

    // The steepest surface, relative to the projection direction, that still
    // receives this decal. Expressed as a cosine: 0.3 is about 72 degrees.
    //
    // NOT OPTIONAL. A decal box intersecting a floor and a wall projects onto
    // both, and the wall gets the texture stretched into vertical stripes --
    // which is the single most recognisable decal artefact and the reason
    // people conclude projected decals look bad.
    float normal_fade = 0.35f;

    // Which texture. Decals share one draw call, so they must share a texture;
    // an atlas is how several different marks coexist.
    rhi::TextureId texture;
};

class DecalSystem {
  public:
    [[nodiscard]] static std::unique_ptr<DecalSystem> Create(
        rhi::Device&, std::string& error,
        rhi::Format gbuffer = rhi::Format::RGBA16Float, int capacity = 512);
    ~DecalSystem();

    DecalSystem(const DecalSystem&) = delete;
    DecalSystem& operator=(const DecalSystem&) = delete;

    // Draws every decal into the pass's colour attachment, which must be the
    // G-buffer's ALBEDO target with `load` set -- a decal blends over what the
    // geometry pass wrote, so a pass that cleared first would leave only decals.
    //
    // `depth` and `normal_metal` are the G-buffer's other two attachments:
    // depth to reconstruct where the surface is, normals to reject the ones
    // facing the wrong way.
    void Draw(rhi::Encoder&, const Camera&, int width, int height,
              rhi::TextureId depth, rhi::TextureId normal_metal,
              std::span<const Decal>);

    // How many were submitted by the last Draw. Decals sharing a texture are
    // ONE draw call however many there are; a number here that equals the decal
    // count means they are not being batched.
    [[nodiscard]] int LastDrawCalls() const;
    [[nodiscard]] int LastDecalCount() const;
    [[nodiscard]] int Capacity() const;

  private:
    DecalSystem();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eng
