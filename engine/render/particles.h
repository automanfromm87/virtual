// A GPU particle system: simulated in a compute pass, drawn as one instanced
// call, and never read by the CPU.
//
// The design constraint that decides everything else is that the CPU must not
// touch a particle. A system where it does costs the same whether ten particles
// are alive or a million -- walking the pool, integrating, building a vertex
// buffer -- and that cost lands on the frame budget, not the GPU's. Here a step
// is one dispatch and a draw is one call, both with a size fixed at creation.
//
// The consequence to be aware of: the CPU does not know how many particles are
// alive, or where any of them are. Anything that wanted that -- gameplay
// reacting to a spark landing, sorting by depth -- has to move to the GPU or be
// given up. Additive blending is what makes giving up the sort acceptable:
// addition commutes, so order does not matter.
#pragma once

#include <memory>
#include <string>

#include "engine/core/math.h"
#include "engine/rhi/rhi.h"
#include "engine/scene/scene.h"

namespace eng {

// What to emit and how it moves. One emitter per system: several would need
// either several systems or a per-particle emitter index, and the second is
// only worth it once there are enough emitters for the dispatch overhead to
// matter.
struct ParticleEmitter {
    Vec3 position{0.0f, 0.0f, 0.0f};
    // Mean emit direction, and the half-angle of the cone around it. Zero is a
    // perfectly straight beam; pi is a sphere.
    Vec3 direction{0.0f, 1.0f, 0.0f};
    float spread = 0.35f;

    float speed = 6.0f;
    float speed_variance = 1.5f;
    float lifetime = 2.0f;
    float lifetime_variance = 0.5f;
    float size = 0.06f;
    float size_variance = 0.02f;

    // Particles per second. The pool is finite, so an emitter that asks for
    // more than lifetime * rate particles simply runs out of slots and emits
    // fewer -- which looks like the effect thinning out rather than failing,
    // and is the right behaviour for something with a hard memory budget.
    float rate = 800.0f;

    Vec4 color{1.0f, 0.6f, 0.2f, 1.0f};
    Vec3 gravity{0.0f, -9.81f, 0.0f};
    // Per-second velocity loss, applied as a ratio. Zero is vacuum.
    float drag = 0.4f;
};

class ParticleSystem {
  public:
    // `capacity` is the pool size and the hard ceiling on live particles. It
    // cannot grow: growing would mean reallocating a buffer the GPU may be
    // reading, and the frame that did it would be the one that dropped every
    // particle at once.
    [[nodiscard]] static std::unique_ptr<ParticleSystem> Create(
        rhi::Device&, int capacity, rhi::Format color, std::string& error,
        int samples = 1);
    ~ParticleSystem();

    ParticleSystem(const ParticleSystem&) = delete;
    ParticleSystem& operator=(const ParticleSystem&) = delete;

    // One simulation step. Must run in a compute pass, before the render pass
    // that draws it.
    //
    // `dt` is clamped: a step longer than a particle's whole life would move it
    // its entire trajectory in one go, which turns a fountain into a starburst
    // on the frame after a stall.
    void Step(rhi::ComputeEncoder&, const ParticleEmitter&, float dt);

    // One instanced draw of the whole pool. Dead particles collapse to a point
    // in the vertex shader and rasterise nothing.
    //
    // `scene_depth` enables SOFT particles -- fading the sprite out as it
    // approaches solid geometry, so a billboard intersecting the floor does not
    // cut a hard straight line across itself. It must be a SAMPLEABLE depth
    // target from the same frame. A null handle turns the fade off.
    void Draw(rhi::Encoder&, const Camera&, int width, int height,
              rhi::TextureId scene_depth = {}, float soft_distance = 0.35f);

    [[nodiscard]] int Capacity() const;
    // How many particles were emitted by the last Step. Known without a
    // readback because the CPU chose the number.
    [[nodiscard]] int LastEmitted() const;

    // How many are alive, by READING BACK the pool. This is the one operation
    // that breaks the CPU-never-touches-a-particle rule, and it is here for
    // tests: whether the pool fills, whether particles expire, and whether the
    // rate is what was asked for are all invisible otherwise. Calling it per
    // frame gives up the property the whole design is built on.
    [[nodiscard]] int LiveCountSlow() const;
    // The mean position of the live particles, likewise for tests. A fountain
    // that is not moving, or is moving the wrong way, shows up here and in
    // nothing else without eyeballing a picture.
    [[nodiscard]] Vec3 MeanPositionSlow() const;

  private:
    ParticleSystem();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eng
