// Pure C++20. Froxel volumetric lighting: light shafts, fog that a shadow
// darkens, and the glow in the air around a lamp.
//
// WHAT THE EXISTING FOG CANNOT DO. engine/render/post integrates height fog
// analytically, which assumes the air is uniformly lit -- so it can tint the
// distance a colour and nothing else. It cannot put a shaft of light through a
// window, cannot darken the fog inside a shadow, and cannot glow around a lamp.
// Those three are the reason to have fog that is not a colour ramp.
//
// The grid is a set of "frustum voxels": a screen-space tile crossed with an
// exponential depth slice, exactly like the light clustering's cells. Three
// passes fill it -- scatter, integrate, apply -- and the middle one is separate
// because it is a prefix sum along each column.
#pragma once

#include <memory>
#include <string>

#include "engine/rhi/rhi.h"
#include "engine/scene/scene.h"

namespace eng {

struct VolumetricConfig {
    // How far the volume reaches. Beyond it a pixel gets the last slice's
    // integrated value, so distant geometry is fogged by the whole volume
    // rather than by nothing.
    float near_distance = 0.3f;
    float far_distance = 120.0f;

    // Per metre. Scattering says how much light the medium redirects toward the
    // eye; extinction says how much it removes from whatever is behind.
    //
    // TWO numbers and not one. A medium that scatters as much as it absorbs is
    // a very particular medium: smoke absorbs far more than it scatters, clean
    // air the reverse, and tying them together makes dense fog necessarily
    // bright.
    Vec3 scattering{0.02f, 0.022f, 0.028f};
    float extinction = 0.03f;

    // Density = constant + exponential falling off above `height_reference`.
    float base_density = 0.35f;
    float height_density = 1.0f;
    float height_reference = 0.0f;
    float height_falloff = 0.08f;

    // Henyey-Greenstein anisotropy, -1 to 1. Positive throws light forward,
    // which is why a shaft is bright when you look toward the lamp and faint
    // when you look away. Zero is a grey wash with no directionality at all.
    float anisotropy = 0.55f;

    // Light that has already bounced, standing in for the multiple scattering
    // this does not simulate. Without it the inside of a shadow is perfectly
    // black fog, which no real medium is.
    Vec3 ambient{0.02f, 0.024f, 0.03f};
};

class Volumetrics {
  public:
    [[nodiscard]] static std::unique_ptr<Volumetrics> Create(
        rhi::Device&, std::string& error,
        rhi::Format scene_color = rhi::Format::RGBA16Float);
    ~Volumetrics();
    Volumetrics(const Volumetrics&) = delete;
    Volumetrics& operator=(const Volumetrics&) = delete;

    void SetConfig(const VolumetricConfig&);
    [[nodiscard]] const VolumetricConfig& Config() const;

    // Passes one and two, both compute. `frame` advances the slice jitter;
    // pass the frame index. `shadow_map` and `cascades_buffer` are the same
    // ones the surface shading uses -- sharing them is what makes a shaft line
    // up with the shadow that casts it.
    void Build(rhi::ComputeEncoder&, const Scene&, int width, int height,
               std::uint64_t frame, rhi::TextureId shadow_map,
               rhi::BufferId cascades, std::size_t cascade_offset,
               rhi::BufferId lights, std::size_t light_offset, int light_count);

    // Pass three: samples the integrated volume at each pixel's depth and
    // writes premultiplied fog. Blend it over the scene with
    // (ONE, ONE_MINUS_SRC_ALPHA).
    void Apply(rhi::Encoder&, const Scene&, int width, int height,
               rhi::TextureId scene_depth);

    // The integrated volume, for anything that wants to read it directly.
    [[nodiscard]] rhi::TextureId Volume() const;

  private:
    Volumetrics();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eng
