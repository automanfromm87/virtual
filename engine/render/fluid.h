// A smoothed-particle-hydrodynamics fluid, simulated entirely on the GPU.
//
// The fluid IS the particles. There is no grid holding it and no surface mesh;
// density at a point is the weighted sum of nearby particles' mass, pressure is
// a function of density, and the pressure gradient is the force that stops the
// fluid collapsing into itself. The grid inside exists only to find neighbours
// in constant time.
//
// This is WEAKLY compressible: pressure comes from an explicit state equation
// rather than from solving a Poisson equation for incompressibility. That is a
// large simplification and it is paid for in timestep -- the stiffness that
// keeps the density near its rest value is also what makes the system stiff, so
// a step that is too long does not degrade gracefully, it detonates. Step()
// substeps internally for exactly that reason.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "engine/core/math.h"
#include "engine/rhi/rhi.h"
#include "engine/scene/scene.h"

namespace eng {

struct FluidConfig {
    // The box the fluid lives in. Walls, and also the extent of the neighbour
    // grid: a particle outside it would hash to a cell that does not exist.
    Vec3 bounds_min{-0.5f, 0.0f, -0.5f};
    Vec3 bounds_max{0.5f, 1.2f, 0.5f};

    // The smoothing radius, and the grid's cell size with it. Everything scales
    // off this: too small and a particle has no neighbours and no pressure, too
    // large and every particle sees hundreds and the step costs cubically more.
    // Around four particle spacings is the usual choice.
    float smoothing_radius = 0.10f;
    float rest_density = 1000.0f;
    // The state equation's constant. Higher is less compressible and less
    // stable; this is the dial that trades one for the other.
    float stiffness = 250.0f;
    // The PHYSICAL viscosity: a look, not a stability measure. Honey against
    // water.
    float viscosity = 0.30f;
    // MONAGHAN'S ARTIFICIAL VISCOSITY, which is the stability measure, and a
    // different thing despite the name. It acts only on pairs that are
    // APPROACHING each other, so it damps the compression-rebound that makes
    // weakly compressible SPH unstable and leaves shear and separation alone.
    // The usual range is 0.02 to 0.5; below that the fluid boils and above it
    // everything is treacle.
    float artificial_viscosity = 0.12f;
    float gravity = -9.81f;
    // How much of the normal velocity survives a wall. Low, because a fluid
    // does not bounce off a bucket.
    float wall_restitution = 0.15f;

    // Particles per grid cell. A cell is one smoothing radius across, so at the
    // rest spacing it holds around 30; the excess is headroom for compression.
    // Overfull cells drop particles from their neighbour lists rather than
    // overrunning the bucket, which softens locally instead of corrupting.
    int bucket_capacity = 48;

    // A CEILING on the substep, not the substep itself. The solver derives its
    // own from the CFL conditions below and takes whichever is smaller.
    //
    // Deriving it is not a convenience. Weakly compressible SPH is stiff, and
    // the stability limit depends on the stiffness, the smoothing radius and
    // the viscosity all at once -- so a hand-picked number is only right for
    // one set of parameters and silently wrong for every other. The failure is
    // not a slightly worse fluid either: it churns, never settles, reads well
    // below the rest density because its particles spend their time in flight,
    // and looks like a lively fluid the whole time.
    float max_substep = 1.0f / 240.0f;
    // A hard ceiling on substeps per Step(), so a long frame cannot turn into
    // an unbounded amount of work and make the next frame longer still. When
    // this bites, the fluid runs in SLOW MOTION rather than becoming unstable,
    // which is the right way to fail: a viewer notices sluggish water, and does
    // not notice water that is quietly integrating past its stability limit.
    int max_substeps = 40;

    // The CFL safety factor. 0.25 is the usual choice for SPH; lower is slower
    // and steadier.
    float cfl = 0.25f;
};

class FluidSim {
  public:
    // `positions` is the initial state, at rest. Particle mass is derived from
    // the rest density and the volume each particle is given, so the caller
    // places particles and the solver works out what they weigh.
    [[nodiscard]] static std::unique_ptr<FluidSim> Create(
        rhi::Device&, const FluidConfig&, const std::vector<Vec3>& positions,
        rhi::Format color, std::string& error, int samples = 1);
    ~FluidSim();

    FluidSim(const FluidSim&) = delete;
    FluidSim& operator=(const FluidSim&) = delete;

    // Advances by `dt`, in as many internal substeps as the stability limit
    // requires. Returns how many it took.
    int Step(rhi::ComputeEncoder&, float dt);

    // A SOURCE AND A SINK, and the only way to make this solver keep moving.
    //
    // Everything else here is a closed box under gravity. There are no forces
    // a caller can apply, no way to move a boundary and no coupling to anything
    // outside -- so water in it settles and then stays settled. Measured on the
    // valley's basin: kinetic energy 63.5 at frame 30, 0.19 at frame 210, a
    // factor of 340 in under two seconds. apps/fluid never noticed because a
    // dam break is a collapse, and a collapse is the only motion it has.
    //
    // Call it once per Step, before it. Particles that reach the drain are put
    // back at the spout with `velocity`, so the same water falls through the
    // basin forever; the count never changes, so mass is conserved exactly.
    struct Recirculation {
        Vec3 spout{0.0f, 0.0f, 0.0f};     // where a returned particle reappears
        float spread = 0.06f;             // scattered over a disc this wide
        Vec3 velocity{0.0f, 0.0f, 0.0f};  // and leaves at this, in m/s
        Vec3 drain{0.0f, 0.0f, 0.0f};     // the sink's centre
        float drain_radius = 0.35f;       // and its radius in the horizontal
        float drain_y = 0.0f;             // below this and inside it, caught
        // AT MOST this many particles return per call. The rate limit is not a
        // performance knob: returning a whole slab of the basin floor at once
        // stacks it inside one smoothing radius, and the pressure term answers
        // a density several times rest with thousands of m/s^2.
        int per_step = 90;
    };
    void Recirculate(rhi::ComputeEncoder&, const Recirculation&);

    void Draw(rhi::Encoder&, const Camera&, int width, int height);

    [[nodiscard]] int Count() const;
    [[nodiscard]] float ParticleMass() const;
    // GPU-side particle buffer and smoothing radius, for the surface renderer
    // below. The surface draws the same particles the solver integrates; it
    // does not copy them.
    [[nodiscard]] rhi::BufferId ParticleBuffer() const;
    [[nodiscard]] float SmoothingRadius() const;

    // --- readbacks, for tests ------------------------------------------------
    //
    // Every one of these breaks the rule that the CPU does not touch the fluid,
    // and they exist because a fluid solver cannot be checked from a picture. A
    // fluid that is 30% over-compressed, or slowly gaining energy, or leaking
    // particles through a wall, looks entirely convincing for several seconds.
    [[nodiscard]] std::vector<Vec3> ReadPositions() const;
    // Mean, minimum and maximum density. The headline number for a fluid: the
    // mean should sit at the rest density and the spread should be small.
    void ReadDensity(float* mean, float* lo, float* hi) const;
    [[nodiscard]] float KineticEnergy() const;
    // How many particles are outside the box. Should always be zero -- a solver
    // that leaks is losing mass, and losing mass changes the answer.
    [[nodiscard]] int OutsideBounds() const;

  private:
    FluidSim();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// A continuous water surface out of a FluidSim's particles, in screen space.
//
// The billboard Draw above is a debug visualisation; this is the look. Depth
// from instanced spheres, a bilateral smooth into one sheet, then water
// optics (fresnel sky reflection, background refraction, sun glint) shaded
// from screen-space normals. See fluid_surface.metal for the passes.
//
// Owns its intermediate targets (sized in BeginFrame) but not the tank: the
// app draws its scene into its own depth, and DrawShade compares the two.
// Sharing one depth buffer was tried first and bakes the walls into the
// surface -- the whole tank shades as water.
struct SurfaceLook {
    Vec3 sun_dir{0.4f, 0.7f, 0.6f};
    float sun_intensity = 3.0f;
    Vec3 water_tint{0.55f, 0.72f, 0.78f};
    // Refraction pull in uv units per unit of surface slope. Real refraction
    // bends by the index ratio over the thickness; without a thickness pass
    // this linear pull is the approximation, and past ~0.1 the background
    // visibly detaches from the water's edge.
    float refract = 0.05f;
    Vec3 sky_horizon{0.55f, 0.62f, 0.72f};
    // Bilateral edge stop in metres: taps disagreeing by more are another
    // surface and must not merge. Below the particle spacing the sheet falls
    // apart into balls again; far above it, walls bleed into the water.
    float edge_stop = 0.06f;
    // Sphere radius in units of the smoothing radius. Below ~0.4 the depth
    // has holes the smoother cannot close; above ~0.7 the surface floats a
    // visible radius above where the particles are.
    float sphere_radius = 0.5f;
};

class FluidSurface {
  public:
    [[nodiscard]] static std::unique_ptr<FluidSurface> Create(
        rhi::Device&, std::string& error);
    ~FluidSurface();

    FluidSurface(const FluidSurface&) = delete;
    FluidSurface& operator=(const FluidSurface&) = delete;

    // (Re)allocates the depth and smooth targets for `width` x `height`.
    // A no-op when the size is unchanged.
    bool BeginFrame(rhi::Device&, int width, int height, std::string& error);

    // Instanced spheres into Depth(), in a pass whose depth attachment is
    // Depth(). Water only -- the tank has its own buffer (see DrawShade).
    void DrawDepth(rhi::Encoder&, const FluidSim&, const Camera&,
                   const SurfaceLook&, int width, int height);
    // Bilateral H over Depth() into SmoothH(), then V over that into
    // Smooth(). Two fullscreen passes -- the app issues one render pass per
    // call, because a render target cannot be written and re-read in the
    // same pass.
    void SmoothH(rhi::Encoder&, const Camera&, const SurfaceLook&, int width,
                 int height);
    void SmoothV(rhi::Encoder&, const Camera&, const SurfaceLook&, int width,
                 int height);
    // Fullscreen shade of Smooth() over `background` into the current color
    // target. Pass-through where there is no water, so no blending.
    // `tank_depth` is the tank's own (sampleable) depth: Depth() holds water
    // only and cannot tell a submerged-behind-wall sphere from a visible one.
    void DrawShade(rhi::Encoder&, const Camera&, const SurfaceLook&, int width,
                   int height, rhi::TextureId background,
                   rhi::TextureId tank_depth);

    [[nodiscard]] rhi::TextureId Depth() const;
    [[nodiscard]] rhi::TextureId SmoothH() const;
    [[nodiscard]] rhi::TextureId Smooth() const;

  private:
    FluidSurface();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eng
