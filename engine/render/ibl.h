// Image-based lighting, and the sky that feeds it.
//
// WHAT WAS MISSING WITHOUT THIS. The shading model here is Cook-Torrance, which
// takes a roughness and a metalness -- but a metal has no diffuse colour at
// all. Its entire appearance IS what it reflects. With only a directional light
// and a two-colour hemisphere ambient, a metal sphere reflects a single
// highlight and a gradient, so roughness could only change the size of the
// highlight and metalness could only make the object darker. Every material in
// the engine was being evaluated against an environment that did not exist.
//
// THE SKY IS ANALYTIC, and that is a deliberate constraint rather than a
// limitation to apologise for: this engine ships no assets, so an environment
// that has to be loaded from an HDRI would mean IBL is off by default and every
// demo keeps looking the way it did. A scattering model costs a few milliseconds
// once, when the sun moves, and gives a physically consistent sky, sun colour
// and ambient light from a single parameter -- the sun's direction. An
// equirectangular image can be supplied instead, and BakeEquirect is that path.
//
// WHEN TO REBAKE. Only when the environment changes. A fixed sun means baking
// once at startup; a day-night cycle means baking when the sun has moved enough
// to matter, which is far less often than every frame.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "engine/core/math.h"
#include "engine/rhi/rhi.h"
#include "engine/scene/scene.h"

namespace eng {

struct SkyConfig {
    // TOWARD the sun, world space. Need not be normalised. The same convention
    // as Scene::lightDir, so one value drives the sky and the key light and
    // they cannot disagree -- which they will the moment they are two fields.
    Vec3 sun_direction{0.35f, 0.55f, 0.45f};

    // How much haze. 1 is an impossibly clean sky, 2 to 3 is a clear day, 6 is
    // muggy, 20 is fog. It scales the AEROSOLS only: scaling the air as well
    // would be modelling a planet with a different atmosphere, not weather.
    float turbidity = 2.6f;

    // What the ground bounces back up. Not decoration -- it is most of the
    // light on the underside of everything in the scene, and leaving it at zero
    // is why untuned renders have models that look like they are floating.
    Vec3 ground_albedo{0.16f, 0.15f, 0.13f};

    // Radiance of the solar disc before atmospheric attenuation, in the
    // engine's arbitrary-but-consistent units. The one dial for "how bright is
    // it"; everything else follows from the physics.
    float sun_intensity = 22.0f;
    // The real sun subtends about half a degree. Widening it softens every
    // shadow and every specular highlight together, which is the physically
    // honest way to soften them.
    float sun_angular_radius = 0.00465f;
    // A final multiplier on the whole sky, for matching it to a scene's
    // exposure without changing the physics above.
    float exposure = 1.0f;
    // A floor under the darkest sky direction. Pure black in an environment map
    // makes a mirror read as a hole in the world.
    float night_lift = 0.0015f;
};

// What a shader needs to consume the environment. Passed as a group because
// the four are useless apart -- a specular probe without its BRDF table cannot
// be evaluated, and a caller that binds three of the four gets a picture that
// is wrong rather than one that fails.
struct EnvironmentBindings {
    rhi::TextureId irradiance;   // cube, diffuse
    rhi::TextureId specular;     // cube, prefiltered mip chain
    rhi::TextureId brdf_lut;     // 2D, RG
    rhi::SamplerId cube_sampler; // trilinear, clamped
    rhi::SamplerId lut_sampler;  // bilinear, clamped
    int specular_mips = 1;
    [[nodiscard]] bool Valid() const {
        return rhi::Valid(irradiance) && rhi::Valid(specular) &&
               rhi::Valid(brdf_lut);
    }
};

class Environment {
  public:
    // `cube_size` is the radiance cube's face size. 256 is ample: the sky is
    // low frequency apart from the sun, and the sun is added analytically at
    // full precision whatever the resolution.
    [[nodiscard]] static std::unique_ptr<Environment> Create(
        rhi::Device&, std::string& error, int cube_size = 256,
        rhi::Format scene_color = rhi::Format::RGBA16Float, int samples = 1);
    ~Environment();

    Environment(const Environment&) = delete;
    Environment& operator=(const Environment&) = delete;

    // Bakes the whole chain: sky -> radiance cube -> mips -> irradiance ->
    // prefiltered specular. Must be called inside a compute pass.
    //
    // The BRDF table is baked only ONCE, on the first call, because it does not
    // depend on the environment -- it is a property of the BRDF. Rebaking it
    // per sky change would be a thousand samples per texel for an identical
    // answer.
    void BakeSky(rhi::ComputeEncoder&, const SkyConfig&);

    // The same chain from a supplied equirectangular image. `equirect` must be
    // a float texture -- the values above one are the entire point, and an
    // 8-bit source has none of them.
    void BakeEquirect(rhi::ComputeEncoder&, rhi::TextureId equirect);

    // Draws the environment behind everything already in the target. Depth
    // TESTED and not written, so it fills only where nothing was drawn.
    void DrawSky(rhi::Encoder&, const Camera&, int width, int height,
                 float intensity = 1.0f);

    [[nodiscard]] rhi::TextureId Radiance() const;   // the full-resolution cube
    [[nodiscard]] EnvironmentBindings Bindings() const;
    [[nodiscard]] int CubeSize() const;
    [[nodiscard]] int SpecularMips() const;
    // How many times the environment has been baked. A HUD number: a scene
    // rebaking every frame is spending several milliseconds on an answer that
    // did not change, and nothing else makes that visible.
    [[nodiscard]] int BakeCount() const;

    // The sun's colour AFTER the atmosphere has taken its cut, and the ambient
    // the sky contributes. Computed on the CPU from the same model the shader
    // uses, so the key light and the sky agree.
    //
    // This is what makes a sunset work: at a low sun angle the direct light is
    // orange because the blue has been scattered out of it, and a scene whose
    // key light stays white while its sky goes red looks like two different
    // times of day at once.
    [[nodiscard]] static Vec3 SunColor(const SkyConfig&);
    // Applies the model's sun colour and a matching hemisphere ambient to a
    // scene, so one call keeps the light, the sky and the fallback ambient
    // consistent.
    static void ApplyTo(Scene*, const SkyConfig&);

    // --- readbacks, for tests ---------------------------------------------------
    //
    // Every property that matters here is a number, and none of them can be
    // settled by looking: an irradiance map that is 15% too bright, or a
    // prefilter that loses energy at high roughness, looks entirely convincing.
    //
    // Each schedules a compute copy; the values are available through the
    // matching Take... after the command buffer has completed.
    enum class Probe { Radiance, Irradiance, Specular };
    void ReadCube(rhi::ComputeEncoder&, Probe, int face_size, float lod);
    [[nodiscard]] std::vector<Vec4> TakeCube() const;
    void ReadLut(rhi::ComputeEncoder&, int size);
    [[nodiscard]] std::vector<Vec4> TakeLut() const;

  private:
    Environment();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eng
