#include "engine/render/particles.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "engine/shaders/shader_types.h"

namespace eng {
namespace {

constexpr char kShaderTypesSrc[] = {
#embed "engine/shaders/shader_types.h"
    , 0};
constexpr char kParticlesSrc[] = {
#embed "engine/shaders/particles.metal"
    , 0};

static_assert(sizeof(GpuParticle) == 64, "GpuParticle layout drifted");
static_assert(sizeof(GpuParticleParams) == 112, "GpuParticleParams layout drifted");

std::string Source() {
    return std::string(kShaderTypesSrc) + "\n" + kParticlesSrc;
}

// A step longer than this is not slow motion, it is a discontinuity. Clamping
// keeps a stall from teleporting every particle to the end of its arc.
constexpr float kMaxStep = 1.0f / 30.0f;

}  // namespace

struct ParticleSystem::Impl {
    rhi::Device* dev = nullptr;
    int capacity = 0;
    rhi::BufferId pool;      // GpuParticle[capacity]
    rhi::BufferId spawned;   // one atomic uint, reset each step
    rhi::BufferId quad_ib;   // six indices, shared by every particle
    rhi::ComputePipelineId step;
    rhi::PipelineId draw_additive;
    rhi::SamplerId sampler;
    int last_emitted = 0;
    // Carried between steps, because a rate of 800/s and a step of 1/120 s is
    // 6.67 particles -- and truncating to 6 every frame emits 720/s instead of
    // 800. The fraction has to survive the step that could not use it.
    float emit_debt = 0.0f;
    std::uint64_t frame = 0;
};

ParticleSystem::ParticleSystem() : impl_(std::make_unique<Impl>()) {}
ParticleSystem::~ParticleSystem() = default;

std::unique_ptr<ParticleSystem> ParticleSystem::Create(rhi::Device& dev,
                                                       int capacity,
                                                       rhi::Format color,
                                                       std::string& error,
                                                       int samples) {
    if (capacity <= 0) {
        error = "particle capacity must be positive";
        return nullptr;
    }
    std::unique_ptr<ParticleSystem> ps(new ParticleSystem());
    ps->impl_->dev = &dev;
    ps->impl_->capacity = capacity;

    // Zero-filled, so every slot starts DEAD and available. A pool of
    // uninitialised memory would spend its first second emitting particles
    // with garbage lifetimes from garbage positions.
    // Braces: with parens this is the most vexing parse and declares a function.
    const std::vector<GpuParticle> zeros{std::size_t(capacity), GpuParticle{}};
    ps->impl_->pool =
        dev.CreateBuffer(zeros.data(), sizeof(GpuParticle) * zeros.size());
    const std::uint32_t zero[4] = {0, 0, 0, 0};
    ps->impl_->spawned = dev.CreateBuffer(zero, sizeof(zero));

    // Two triangles, shared by every particle. The four corners come from the
    // vertex id, so there is no vertex buffer at all.
    const std::uint16_t quad[6] = {0, 1, 2, 2, 1, 3};
    ps->impl_->quad_ib = dev.CreateBuffer(quad, sizeof(quad));
    if (!Valid(ps->impl_->pool) || !Valid(ps->impl_->spawned) ||
        !Valid(ps->impl_->quad_ib)) {
        error = "could not allocate the particle buffers";
        return nullptr;
    }

    ps->impl_->step = dev.CreateComputePipeline(Source(), "cs_particle_step", error);
    if (!Valid(ps->impl_->step)) return nullptr;

    rhi::PipelineDesc pd;
    pd.source = Source();
    pd.vertex_fn = "vs_particle";
    pd.fragment_fn = "fs_particle";
    pd.color = color;
    pd.samples = samples;
    pd.depth = true;
    // Depth TESTED but not WRITTEN. Tested so a particle behind a wall is
    // hidden; not written because a translucent sprite that writes depth
    // occludes every particle behind it, and a fountain becomes a single layer
    // of sprites with holes punched through it.
    pd.depth_write = false;
    pd.blend = rhi::Blend::Additive;
    ps->impl_->draw_additive = dev.CreatePipeline(pd, error);
    if (!Valid(ps->impl_->draw_additive)) return nullptr;

    ps->impl_->sampler = dev.CreateSampler(rhi::Filter::Linear, rhi::Wrap::Clamp);
    return ps;
}

int ParticleSystem::Capacity() const { return impl_->capacity; }
int ParticleSystem::LastEmitted() const { return impl_->last_emitted; }

void ParticleSystem::Step(rhi::ComputeEncoder& enc, const ParticleEmitter& e,
                          float dt) {
    if (!Valid(impl_->step)) return;
    dt = std::clamp(dt, 0.0f, kMaxStep);

    // How many to emit, carrying the fraction. See Impl::emit_debt.
    impl_->emit_debt += e.rate * dt;
    int spawn = int(impl_->emit_debt);
    impl_->emit_debt -= float(spawn);
    // Never more than the pool in one step: the kernel would reject the excess
    // anyway, and asking for it makes the atomic contended for nothing.
    spawn = std::clamp(spawn, 0, impl_->capacity);
    impl_->last_emitted = spawn;

    // The spawn counter starts at zero EVERY step. It is a GPU-written buffer,
    // so it holds the last step's total otherwise, and every slot would find
    // the budget already spent.
    if (auto* c = static_cast<std::uint32_t*>(impl_->dev->MapBuffer(impl_->spawned)))
        c[0] = 0;

    const Vec3 dir = Normalize(e.direction);
    GpuParticleParams p{};
    p.origin = Vec4{e.position.x, e.position.y, e.position.z, e.spread};
    p.direction = Vec4{dir.x, dir.y, dir.z, e.speed};
    p.gravity = Vec4{e.gravity.x, e.gravity.y, e.gravity.z, e.drag};
    p.color = e.color;
    p.motion = Vec4{dt, e.speed_variance, e.lifetime, e.lifetime_variance};
    p.emit = Vec4{e.size, e.size_variance, float(spawn), float(impl_->frame)};
    p.limits = Vec4{float(impl_->capacity), 0.0f, 0.0f, 0.0f};
    ++impl_->frame;

    enc.SetPipeline(impl_->step);
    enc.SetBuffer(impl_->pool, 0, 0);
    enc.SetBytes(&p, sizeof(p), 1);
    enc.SetBuffer(impl_->spawned, 0, 2);
    enc.Dispatch(impl_->capacity);
}

void ParticleSystem::Draw(rhi::Encoder& enc, const Camera& camera, int width,
                          int height, rhi::TextureId scene_depth,
                          float soft_distance) {
    if (!Valid(impl_->draw_additive) || width <= 0 || height <= 0) return;

    FrameUniforms u{};
    u.viewProj = camera.ViewProj(float(width) / float(height));
    u.eyePos = Vec4{camera.eye.x, camera.eye.y, camera.eye.z, 1.0f};
    // The soft-particle block. x is the RECIPROCAL of the fade distance so the
    // shader multiplies instead of dividing; zero disables it, which is what a
    // caller with no depth target gets.
    if (Valid(scene_depth) && soft_distance > 0.0f) {
        u.ssao = Vec4{1.0f / soft_distance, 1.0f / float(width),
                      1.0f / float(height), camera.nearZ};
    } else {
        u.ssao = Vec4{0.0f, 0.0f, 0.0f, 0.0f};
    }

    enc.SetPipeline(impl_->draw_additive);
    // No culling. A billboard faces the camera, so there is no back face to
    // cull -- and a winding that comes out backwards for half the screen is
    // exactly the sort of thing that makes half a fountain disappear.
    enc.SetCull(rhi::Cull::None, rhi::Winding::CounterClockwise);
    enc.SetVertexBuffer(impl_->pool, 0, 0);
    enc.SetVertexBytes(&u, sizeof(u), 1);
    enc.SetFragmentBytes(&u, sizeof(u), 1);
    if (Valid(scene_depth)) enc.SetFragmentTexture(scene_depth, 0);
    enc.SetFragmentSampler(impl_->sampler, 0);
    enc.DrawIndexedInstancedU16(impl_->quad_ib, 6, std::size_t(impl_->capacity));
}

int ParticleSystem::LiveCountSlow() const {
    const auto* p = static_cast<const GpuParticle*>(impl_->dev->MapBuffer(impl_->pool));
    if (!p) return 0;
    int n = 0;
    for (int i = 0; i < impl_->capacity; ++i)
        if (p[i].position.w > 0.0f) ++n;
    return n;
}

Vec3 ParticleSystem::MeanPositionSlow() const {
    const auto* p = static_cast<const GpuParticle*>(impl_->dev->MapBuffer(impl_->pool));
    if (!p) return Vec3{0, 0, 0};
    Vec3 sum{0, 0, 0};
    int n = 0;
    for (int i = 0; i < impl_->capacity; ++i) {
        if (p[i].position.w <= 0.0f) continue;
        sum = sum + Vec3{p[i].position.x, p[i].position.y, p[i].position.z};
        ++n;
    }
    return n > 0 ? sum * (1.0f / float(n)) : Vec3{0, 0, 0};
}

}  // namespace eng
