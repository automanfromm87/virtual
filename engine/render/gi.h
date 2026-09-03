// Pure C++20. An irradiance volume: baked indirect light on a grid, sampled
// trilinearly, with real multiple bounces.
//
// WHAT IS MISSING WITHOUT IT. The engine has a directional light, local lights,
// and one image-based probe for the whole scene. That is enough outdoors, where
// the sky genuinely is the same in every direction from everywhere. Indoors it
// is not even close: a room lit through one window has a bright wall opposite
// the window, a dim one beside it, and a ceiling lit entirely by bounce -- and
// a single global probe gives all three the same ambient. The room comes out
// flat, and no amount of tuning the ambient colour fixes it, because the
// problem is that ambient is being treated as a constant when it is a field.
//
// WHAT THIS STORES. Irradiance at each grid point as spherical harmonics, order
// 1: four coefficients per colour channel. L1 is the deliberate choice and not a
// budget cut -- irradiance is the cosine-weighted integral of incoming light,
// and that convolution destroys everything above order 2 almost completely.
// L1 keeps the direction the light comes from, which is the entire visual
// payoff; L2 costs twice the memory for a difference you have to measure.
//
// HOW IT IS BAKED. Path tracing on the CPU against a BVH of the scene's
// triangles, one pass per bounce. Bounce N samples the volume produced by
// bounce N-1 at each hit point, which is what makes colour bleed: a white wall
// beside a red one is lit by a red wall.
//
// CPU and not the GPU's ray tracing hardware, even though the engine has it,
// because a bake is not a frame. It runs once, it wants to be debuggable, and
// putting it on the CPU means the whole thing is testable without a device --
// which matters more here than anywhere else in the renderer, since every
// failure mode of a GI bake produces a picture that looks plausible.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "engine/core/math.h"

namespace eng {

// One surface for the bake. A triangle soup with material, deliberately not a
// Mesh: the bake needs world-space positions and an albedo and nothing else,
// and taking a Mesh would drag the vertex format and the renderer's material
// table into a file that has no business knowing about either.
struct GiTriangle {
    Vec3 a, b, c;
    Vec3 albedo{0.5f, 0.5f, 0.5f};
    // Radiance the surface emits on its own. This is how a glowing panel
    // becomes a light source in the bake -- and it is the only way, because the
    // tracer knows about the sun and the sky and nothing else.
    Vec3 emissive{0.0f, 0.0f, 0.0f};
};

struct GiBakeConfig {
    // Grid resolution. The cost is probes x rays x bounces, and probes is the
    // product of these three, so it is the one to be careful with.
    int nx = 8, ny = 4, nz = 8;
    // World-space corner and the size of one cell. A probe sits at
    // origin + (i, j, k) * spacing.
    Vec3 origin{0.0f, 0.0f, 0.0f};
    Vec3 spacing{1.0f, 1.0f, 1.0f};

    int rays = 128;     // per probe per bounce, on a Fibonacci sphere
    int bounces = 2;    // 0 is direct light only, 1 adds one bounce, and so on

    Vec3 sun_direction{0.4f, 0.8f, 0.4f};  // TOWARD the sun
    Vec3 sun_color{3.0f, 2.9f, 2.7f};
    // Sky radiance above and ground bounce below, blended by the ray's
    // elevation. What a ray that escapes the geometry sees.
    Vec3 sky_top{0.35f, 0.45f, 0.7f};
    Vec3 sky_bottom{0.16f, 0.14f, 0.12f};

    // Threads for the bake. 0 means the calling thread only, which is what a
    // test wants: the bake is deterministic either way, and proving that is one
    // of the checks.
    int threads = 0;
};

// Four SH-L1 coefficients per channel, laid out so that one RGBA texel per
// channel holds a whole probe: (L00, L1-1, L10, L11).
struct ShProbe {
    Vec4 r{0, 0, 0, 0};
    Vec4 g{0, 0, 0, 0};
    Vec4 b{0, 0, 0, 0};
};

class IrradianceVolume {
  public:
    // Bakes and returns the volume. An empty triangle list is not an error --
    // every probe then sees only the sky, which is exactly right for an
    // outdoor scene and is also the sanity case a test starts from.
    [[nodiscard]] static IrradianceVolume Bake(std::span<const GiTriangle>,
                                               const GiBakeConfig&);

    // Irradiance arriving at `world` on a surface facing `normal`, in the same
    // radiance units the rest of the renderer uses. Trilinear between the eight
    // surrounding probes; clamped at the volume's edges rather than falling to
    // zero, because a surface just outside the grid should look like the
    // nearest place inside it and not like a hole.
    [[nodiscard]] Vec3 Sample(Vec3 world, Vec3 normal) const;

    // The raw coefficients, x fastest then y then z. For upload.
    [[nodiscard]] const std::vector<ShProbe>& Probes() const { return probes_; }
    [[nodiscard]] const GiBakeConfig& Config() const { return cfg_; }
    [[nodiscard]] bool Empty() const { return probes_.empty(); }

    // Probes that baked far darker than the rest of the volume: below a
    // seventh of its median. Almost always probes inside geometry -- one buried
    // in a wall sees nothing but the wall's unlit inner faces -- and they are
    // the ones that drag down every surface interpolating against them, which
    // is the classic dark blotch along a skirting board.
    //
    // Detected as a DARK OUTLIER rather than by geometry, and that is a
    // deliberate retreat. The textbook test is "most rays hit a back face",
    // which needs consistently outward winding that nothing in this engine
    // guarantees; the obvious fallback, "most rays hit something very close",
    // fails on a thin slab because most rays escape sideways. Both were tried.
    // Darkness is the symptom that actually matters and it needs neither.
    //
    // Reported rather than corrected: the fix is to move the grid or supply a
    // fallback, and both are the caller's decision.
    [[nodiscard]] int DarkProbes() const { return dark_; }

  private:
    // Shared by Sample and by the bake's own bounce lookup. The bake cannot
    // call Sample: it reads the PREVIOUS pass's probes while writing this
    // pass's, and reading and writing one buffer would make the result depend
    // on the order the probes happened to be visited.
    [[nodiscard]] Vec3 SampleFrom(const std::vector<ShProbe>&, Vec3 world,
                                  Vec3 normal) const;

    std::vector<ShProbe> probes_;
    GiBakeConfig cfg_;
    int dark_ = 0;
};

// Projects a single radiance sample arriving from `dir` into SH-L1 and adds it
// to `into`, weighted by `weight`. Exposed because the projection and the
// evaluation have to be exact inverses up to the convolution, and testing that
// directly is far sharper than testing it through a bake.
void ShAccumulate(ShProbe& into, Vec3 dir, Vec3 radiance, float weight);
// Irradiance from a set of coefficients, for a surface facing `normal`.
// Includes the cosine-lobe convolution and the 1/pi that turns irradiance into
// the outgoing radiance of a Lambertian surface of albedo 1.
[[nodiscard]] Vec3 ShIrradiance(const ShProbe&, Vec3 normal);

}  // namespace eng
