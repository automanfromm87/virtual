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

    void Draw(rhi::Encoder&, const Camera&, int width, int height);

    [[nodiscard]] int Count() const;
    [[nodiscard]] float ParticleMass() const;

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

}  // namespace eng
