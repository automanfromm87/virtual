#include "engine/render/fluid.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "engine/geometry/mesh.h"
#include "engine/shaders/shader_types.h"

namespace eng {
namespace {

constexpr char kShaderTypesSrc[] = {
#embed "engine/shaders/shader_types.h"
    , 0};
constexpr char kFluidSrc[] = {
#embed "engine/shaders/fluid.metal"
    , 0};

static_assert(sizeof(GpuFluidParticle) == 32, "GpuFluidParticle layout drifted");
static_assert(sizeof(GpuFluidParams) == 112, "GpuFluidParams layout drifted");

std::string Source() { return std::string(kShaderTypesSrc) + "\n" + kFluidSrc; }

constexpr float kPi = 3.14159265358979f;

}  // namespace

struct FluidSim::Impl {
    rhi::Device* dev = nullptr;
    FluidConfig cfg;
    int count = 0;
    float mass = 0.0f;
    int cells_x = 0, cells_y = 0, cells_z = 0;

    rhi::BufferId particles;  // GpuFluidParticle[count]
    rhi::BufferId accel;      // float4[count]
    rhi::BufferId counts;     // uint per cell
    rhi::BufferId buckets;    // uint[cells * bucket_capacity]

    rhi::ComputePipelineId clear, bin, density, forces, integrate, recycle;
    // Which index class the recirculation considers next. Advancing it every
    // call is what turns a rate limit into a steady jet rather than the same
    // ninety particles being offered the drain forever.
    int recycle_phase = 0;
    rhi::PipelineId draw;
    rhi::BufferId quad_ib;

    [[nodiscard]] int CellCount() const { return cells_x * cells_y * cells_z; }

    GpuFluidParams Params(float dt) const {
        const float h = cfg.smoothing_radius;
        GpuFluidParams p{};
        p.bounds_min = Vec4{cfg.bounds_min.x, cfg.bounds_min.y, cfg.bounds_min.z, h};
        p.bounds_max = Vec4{cfg.bounds_max.x, cfg.bounds_max.y, cfg.bounds_max.z,
                            mass};
        p.grid = Vec4{float(cells_x), float(cells_y), float(cells_z), h};
        p.physics = Vec4{cfg.rest_density, cfg.stiffness, cfg.viscosity, dt};
        p.misc = Vec4{float(count), cfg.wall_restitution,
                      float(cfg.bucket_capacity), cfg.gravity};
        // The kernel normalisations, computed once. Each is a constant times a
        // power of h between six and nine; recomputing them per neighbour would
        // put three pow() calls in the innermost loop of the simulation.
        const float h3 = h * h * h;
        const float h6 = h3 * h3;
        const float h9 = h6 * h3;
        p.kernels = Vec4{315.0f / (64.0f * kPi * h9),  // poly6
                         -45.0f / (kPi * h6),          // spiky gradient
                         45.0f / (kPi * h6),           // viscosity laplacian
                         0.0f};
        p.artificial = Vec4{cfg.artificial_viscosity,
                            std::sqrt(std::max(cfg.stiffness, 1e-6f)), 0.0f, 0.0f};
        return p;
    }
};

FluidSim::FluidSim() : impl_(std::make_unique<Impl>()) {}
FluidSim::~FluidSim() = default;

int FluidSim::Count() const { return impl_->count; }
float FluidSim::ParticleMass() const { return impl_->mass; }

std::unique_ptr<FluidSim> FluidSim::Create(rhi::Device& dev,
                                           const FluidConfig& cfg,
                                           const std::vector<Vec3>& positions,
                                           rhi::Format color, std::string& error,
                                           int samples) {
    if (positions.empty()) {
        error = "a fluid needs particles";
        return nullptr;
    }
    if (cfg.smoothing_radius <= 0.0f) {
        error = "the smoothing radius must be positive";
        return nullptr;
    }

    std::unique_ptr<FluidSim> fs(new FluidSim());
    Impl& im = *fs->impl_;
    im.dev = &dev;
    im.cfg = cfg;
    im.count = int(positions.size());

    const Vec3 span = cfg.bounds_max - cfg.bounds_min;
    if (span.x <= 0.0f || span.y <= 0.0f || span.z <= 0.0f) {
        error = "the fluid's bounds are empty or inverted";
        return nullptr;
    }
    im.cells_x = std::max(1, int(std::ceil(span.x / cfg.smoothing_radius)));
    im.cells_y = std::max(1, int(std::ceil(span.y / cfg.smoothing_radius)));
    im.cells_z = std::max(1, int(std::ceil(span.z / cfg.smoothing_radius)));

    // PARTICLE MASS, from the rest density and the volume each particle is
    // being asked to represent. Setting it by hand is the classic way to get a
    // fluid that is uniformly 40% too dense and never settles: the solver then
    // spends forever pushing against a rest density it can never reach.
    //
    // The volume per particle is the box's volume divided between them only if
    // they fill the box, which they generally do not -- so it is derived from
    // the actual spacing instead, taken as the cube root of the fluid's own
    // bounding volume per particle.
    Vec3 lo = positions[0], hi = positions[0];
    for (const Vec3& p : positions) {
        lo = Vec3{std::min(lo.x, p.x), std::min(lo.y, p.y), std::min(lo.z, p.z)};
        hi = Vec3{std::max(hi.x, p.x), std::max(hi.y, p.y), std::max(hi.z, p.z)};
    }
    const Vec3 fluid_span = hi - lo;
    // Each particle sits at the centre of its own cell of fluid, so the
    // occupied volume is one spacing wider than the extent of their centres in
    // each axis. Ignoring that under-counts the volume, and the error is worst
    // for the small blocks a test uses.
    const float n = float(im.count);
    const float guess = std::cbrt(std::max(1e-12f,
                                           (fluid_span.x + 1e-6f) *
                                           (fluid_span.y + 1e-6f) *
                                           (fluid_span.z + 1e-6f)) / n);
    const float volume = (fluid_span.x + guess) * (fluid_span.y + guess) *
                         (fluid_span.z + guess);
    im.mass = cfg.rest_density * volume / n;

    std::vector<GpuFluidParticle> init(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i) {
        init[i].position = Vec4{positions[i].x, positions[i].y, positions[i].z,
                                cfg.rest_density};
        init[i].velocity = Vec4{0.0f, 0.0f, 0.0f, 0.0f};
    }
    im.particles = dev.CreateBuffer(init.data(),
                                    sizeof(GpuFluidParticle) * init.size());
    im.accel = dev.CreateStorageBuffer(sizeof(Vec4) * init.size());
    im.counts = dev.CreateStorageBuffer(sizeof(std::uint32_t) *
                                        std::size_t(im.CellCount()));
    im.buckets = dev.CreateStorageBuffer(
        sizeof(std::uint32_t) * std::size_t(im.CellCount()) *
        std::size_t(cfg.bucket_capacity));
    const std::uint32_t quad[6] = {0, 1, 2, 2, 1, 3};
    im.quad_ib = dev.CreateBuffer(quad, sizeof(quad));
    if (!Valid(im.particles) || !Valid(im.accel) || !Valid(im.counts) ||
        !Valid(im.buckets) || !Valid(im.quad_ib)) {
        error = "could not allocate the fluid's buffers";
        return nullptr;
    }

    const std::string src = Source();
    im.clear = dev.CreateComputePipeline(src, "cs_fluid_clear", error);
    im.bin = dev.CreateComputePipeline(src, "cs_fluid_bin", error);
    im.density = dev.CreateComputePipeline(src, "cs_fluid_density", error);
    im.forces = dev.CreateComputePipeline(src, "cs_fluid_forces", error);
    im.integrate = dev.CreateComputePipeline(src, "cs_fluid_integrate", error);
    im.recycle = dev.CreateComputePipeline(src, "cs_fluid_recycle", error);
    if (!Valid(im.clear) || !Valid(im.bin) || !Valid(im.density) ||
        !Valid(im.forces) || !Valid(im.integrate) || !Valid(im.recycle))
        return nullptr;

    rhi::PipelineDesc pd;
    pd.source = src;
    pd.vertex_fn = "vs_fluid";
    pd.fragment_fn = "fs_fluid";
    pd.color = color;
    pd.samples = samples;
    pd.depth = true;
    // Opaque droplets: depth WRITTEN, unlike the additive particles. A fluid
    // occludes what is behind it, and without depth writes the far side of the
    // body draws over the near side.
    pd.depth_write = true;
    pd.blend = rhi::Blend::None;
    im.draw = dev.CreatePipeline(pd, error);
    if (!Valid(im.draw)) return nullptr;

    return fs;
}

int FluidSim::Step(rhi::ComputeEncoder& enc, float dt) {
    if (dt <= 0.0f) return 0;
    Impl& im = *impl_;

    // THE SUBSTEP, from the physics rather than from a guess. Three limits,
    // and the smallest wins:
    //
    //   Sound speed. The state equation p = k(rho - rho0) has dp/drho = k, so
    //   the speed of sound is sqrt(k). A pressure wave must not cross a
    //   smoothing radius in one step, or the particle that should have been
    //   pushed has already moved past the one pushing it.
    //
    //   Viscosity, which is a diffusion and has its own tighter limit -- the
    //   explicit scheme for it goes unstable at dt > h^2 / (8 nu) and this is
    //   the usual conservative form of that.
    //
    //   Gravity, via the force limit: a particle must not fall a smoothing
    //   radius in one step either.
    const float sr = im.cfg.smoothing_radius;
    const float c = std::sqrt(std::max(im.cfg.stiffness, 1e-6f));
    const float dt_sound = im.cfg.cfl * sr / c;
    const float dt_visc = im.cfg.viscosity > 1e-6f
                              ? 0.125f * sr * sr / im.cfg.viscosity
                              : 1e9f;
    const float g = std::fabs(im.cfg.gravity);
    const float dt_force = g > 1e-6f ? im.cfg.cfl * std::sqrt(sr / g) : 1e9f;
    const float limit =
        std::min(std::min(dt_sound, dt_visc),
                 std::min(dt_force, std::max(im.cfg.max_substep, 1e-6f)));

    const int substeps =
        std::clamp(int(std::ceil(dt / limit)), 1, std::max(1, im.cfg.max_substeps));
    const float h = dt / float(substeps);
    const GpuFluidParams params = im.Params(h);

    for (int s = 0; s < substeps; ++s) {
        // Five dispatches, and the ORDER is the algorithm. Density must be
        // complete before any force reads it -- a force computed against a
        // half-updated density field is not a smaller error, it is a different
        // simulation. Separate dispatches are what guarantees that: within one
        // there is no ordering between threadgroups at all.
        enc.SetPipeline(im.clear);
        enc.SetBuffer(im.counts, 0, 0);
        enc.SetBytes(&params, sizeof(params), 1);
        enc.Dispatch(im.CellCount());

        enc.SetPipeline(im.bin);
        enc.SetBuffer(im.particles, 0, 0);
        enc.SetBytes(&params, sizeof(params), 1);
        enc.SetBuffer(im.counts, 0, 2);
        enc.SetBuffer(im.buckets, 0, 3);
        enc.Dispatch(im.count);

        enc.SetPipeline(im.density);
        enc.SetBuffer(im.particles, 0, 0);
        enc.SetBytes(&params, sizeof(params), 1);
        enc.SetBuffer(im.counts, 0, 2);
        enc.SetBuffer(im.buckets, 0, 3);
        enc.Dispatch(im.count);

        enc.SetPipeline(im.forces);
        enc.SetBuffer(im.particles, 0, 0);
        enc.SetBytes(&params, sizeof(params), 1);
        enc.SetBuffer(im.counts, 0, 2);
        enc.SetBuffer(im.buckets, 0, 3);
        enc.SetBuffer(im.accel, 0, 4);
        enc.Dispatch(im.count);

        enc.SetPipeline(im.integrate);
        enc.SetBuffer(im.particles, 0, 0);
        enc.SetBytes(&params, sizeof(params), 1);
        enc.SetBuffer(im.accel, 0, 2);
        enc.Dispatch(im.count);
    }
    return substeps;
}

void FluidSim::Recirculate(rhi::ComputeEncoder& enc, const Recirculation& r) {
    Impl& im = *impl_;
    if (!Valid(im.recycle) || im.count <= 0 || r.per_step <= 0) return;

    GpuFluidRecycle g{};
    g.spout = Vec4{r.spout.x, r.spout.y, r.spout.z, r.spread};
    g.velocity = Vec4{r.velocity.x, r.velocity.y, r.velocity.z, 0.0f};
    g.drain = Vec4{r.drain.x, r.drain.y, r.drain.z, r.drain_radius};
    // THE STRIDE IS DERIVED FROM THE RATE, not asked for. A caller thinking
    // about a fountain thinks in particles per step; the kernel needs the index
    // class that yields it, and getting the relationship backwards is how a
    // rate limit silently stops limiting when the particle count changes.
    const int stride = std::max(1, im.count / std::max(r.per_step, 1));
    g.catchment = Vec4{r.drain_y, float(im.count), float(stride),
                       float(im.recycle_phase % stride)};
    im.recycle_phase = (im.recycle_phase + 1) % stride;

    enc.SetPipeline(im.recycle);
    enc.SetBuffer(im.particles, 0, 0);
    enc.SetBytes(&g, sizeof(g), 1);
    enc.Dispatch(im.count);
}

void FluidSim::Draw(rhi::Encoder& enc, const Camera& camera, int width,
                    int height) {
    if (!Valid(impl_->draw) || width <= 0 || height <= 0) return;
    FrameUniforms u{};
    u.viewProj = camera.ViewProj(float(width) / float(height));
    u.eyePos = Vec4{camera.eye.x, camera.eye.y, camera.eye.z, 1.0f};
    const GpuFluidParams params = impl_->Params(0.0f);

    enc.SetPipeline(impl_->draw);
    enc.SetCull(rhi::Cull::None, rhi::Winding::CounterClockwise);
    enc.SetVertexBuffer(impl_->particles, 0, 0);
    enc.SetVertexBytes(&u, sizeof(u), 1);
    enc.SetVertexBytes(&params, sizeof(params), 2);
    // The fragment stage reads eyePos out of the same block, and vertex and
    // fragment bindings are separate tables -- setting it for the vertex stage
    // alone leaves the fragment reading an unbound buffer.
    enc.SetFragmentBytes(&u, sizeof(u), 1);
    enc.DrawIndexedInstancedU32(impl_->quad_ib, 6, std::size_t(impl_->count));
}

std::vector<Vec3> FluidSim::ReadPositions() const {
    std::vector<Vec3> out;
    const auto* p = static_cast<const GpuFluidParticle*>(
        impl_->dev->MapBuffer(impl_->particles));
    if (!p) return out;
    out.reserve(std::size_t(impl_->count));
    for (int i = 0; i < impl_->count; ++i)
        out.push_back(Vec3{p[i].position.x, p[i].position.y, p[i].position.z});
    return out;
}

void FluidSim::ReadDensity(float* mean, float* lo, float* hi) const {
    const auto* p = static_cast<const GpuFluidParticle*>(
        impl_->dev->MapBuffer(impl_->particles));
    if (!p || impl_->count == 0) {
        if (mean) *mean = 0.0f;
        if (lo) *lo = 0.0f;
        if (hi) *hi = 0.0f;
        return;
    }
    double sum = 0.0;
    float mn = p[0].position.w, mx = p[0].position.w;
    for (int i = 0; i < impl_->count; ++i) {
        const float d = p[i].position.w;
        sum += d;
        mn = std::min(mn, d);
        mx = std::max(mx, d);
    }
    if (mean) *mean = float(sum / impl_->count);
    if (lo) *lo = mn;
    if (hi) *hi = mx;
}

float FluidSim::KineticEnergy() const {
    const auto* p = static_cast<const GpuFluidParticle*>(
        impl_->dev->MapBuffer(impl_->particles));
    if (!p) return 0.0f;
    double e = 0.0;
    for (int i = 0; i < impl_->count; ++i) {
        const Vec3 v{p[i].velocity.x, p[i].velocity.y, p[i].velocity.z};
        e += 0.5 * double(impl_->mass) * double(Dot(v, v));
    }
    return float(e);
}

int FluidSim::OutsideBounds() const {
    const auto* p = static_cast<const GpuFluidParticle*>(
        impl_->dev->MapBuffer(impl_->particles));
    if (!p) return 0;
    const Vec3 lo = impl_->cfg.bounds_min, hi = impl_->cfg.bounds_max;
    const float eps = 1e-4f;
    int n = 0;
    for (int i = 0; i < impl_->count; ++i) {
        const Vec4& q = p[i].position;
        if (q.x < lo.x - eps || q.y < lo.y - eps || q.z < lo.z - eps ||
            q.x > hi.x + eps || q.y > hi.y + eps || q.z > hi.z + eps ||
            !std::isfinite(q.x) || !std::isfinite(q.y) || !std::isfinite(q.z))
            ++n;
    }
    return n;
}

rhi::BufferId FluidSim::ParticleBuffer() const { return impl_->particles; }

float FluidSim::SmoothingRadius() const { return impl_->cfg.smoothing_radius; }

// --- the surface -------------------------------------------------------------

constexpr char kSurfaceSrc[] = {
#embed "engine/shaders/fluid_surface.metal"
    , 0};

static_assert(sizeof(GpuSurfaceParams) == 80,
              "GpuSurfaceParams layout drifted");

namespace {

std::string SurfaceSource() {
    return std::string(kShaderTypesSrc) + "\n" + kSurfaceSrc;
}

}  // namespace

struct FluidSurface::Impl {
    rhi::Device* dev = nullptr;
    int width = 0, height = 0;

    rhi::BufferId sphere_vb;
    rhi::BufferId sphere_ib;
    int sphere_indices = 0;

    rhi::PipelineId depth;
    rhi::PipelineId smooth_h;
    rhi::PipelineId smooth_v;
    rhi::PipelineId shade;

    rhi::TextureId depth_tex;
    rhi::TextureId smooth_a;
    rhi::TextureId smooth_b;
};

FluidSurface::FluidSurface() : impl_(std::make_unique<Impl>()) {}
FluidSurface::~FluidSurface() = default;

std::unique_ptr<FluidSurface> FluidSurface::Create(rhi::Device& dev,
                                                  std::string& error) {
    std::unique_ptr<FluidSurface> fs(new FluidSurface());
    Impl& im = *fs->impl_;
    im.dev = &dev;

    // A unit sphere the vertex stage scales per particle. 8x12 is 216
    // triangles: round enough that the smoother never sees a facet, cheap
    // enough that ten thousand particles stay under three million of them.
    const Mesh sphere =
        MakeUVSphere(1.0f, 8, 12, Vec4{1, 1, 1, 1}, Vec4{1, 1, 1, 1});
    im.sphere_vb = dev.CreateBuffer(sphere.vertices.data(),
                                    sizeof(VertexIn) * sphere.vertices.size());
    im.sphere_ib = dev.CreateBuffer(sphere.indices.data(),
                                    sizeof(std::uint32_t) * sphere.indices.size());
    im.sphere_indices = int(sphere.indices.size());
    if (!Valid(im.sphere_vb) || !Valid(im.sphere_ib) || im.sphere_indices <= 0) {
        error = "could not allocate the surface sphere";
        return nullptr;
    }

    const std::string src = SurfaceSource();
    auto full = [&](const char* frag, rhi::Format color) {
        rhi::PipelineDesc pd;
        pd.source = src;
        pd.vertex_fn = "vs_surface_full";
        pd.fragment_fn = frag;
        pd.color = color;
        pd.samples = 1;
        pd.depth = false;
        pd.blend = rhi::Blend::None;
        return dev.CreatePipeline(pd, error);
    };
    rhi::PipelineDesc dd;
    dd.source = src;
    dd.vertex_fn = "vs_surface_depth";
    dd.fragment_fn = "";  // depth-only: no fragment stage (see the shader)
    dd.color = rhi::Format::RGBA8Unorm;  // unused; depth_only ignores it
    dd.samples = 1;
    dd.depth = true;
    dd.depth_only = true;
    dd.depth_write = true;
    im.depth = dev.CreatePipeline(dd, error);
    im.smooth_h = full("fs_surface_smooth_h", rhi::Format::RGBA16Float);
    im.smooth_v = full("fs_surface_smooth_v", rhi::Format::RGBA16Float);
    im.shade = full("fs_surface_shade", rhi::Format::RGBA16Float);
    if (!Valid(im.depth) || !Valid(im.smooth_h) || !Valid(im.smooth_v) ||
        !Valid(im.shade))
        return nullptr;
    return fs;
}

bool FluidSurface::BeginFrame(rhi::Device& dev, int width, int height,
                              std::string& error) {
    Impl& im = *impl_;
    if (width <= 0 || height <= 0) {
        error = "the surface needs a positive size";
        return false;
    }
    if (width == im.width && height == im.height && Valid(im.depth_tex)) return true;
    im.width = width;
    im.height = height;
    im.depth_tex = dev.CreateDepthTarget(width, height, /*sampleable=*/true);
    im.smooth_a = dev.CreateRenderTarget(width, height, rhi::Format::RGBA16Float);
    im.smooth_b = dev.CreateRenderTarget(width, height, rhi::Format::RGBA16Float);
    if (!Valid(im.depth_tex) || !Valid(im.smooth_a) || !Valid(im.smooth_b)) {
        error = "could not allocate the surface targets";
        return false;
    }
    return true;
}

namespace {

FrameUniforms SurfaceUniforms(const Camera& camera, int width, int height) {
    FrameUniforms u{};
    u.viewProj = camera.ViewProj(float(width) / float(height));
    u.invViewProj = Inverse(u.viewProj);
    u.eyePos = Vec4{camera.eye.x, camera.eye.y, camera.eye.z, 1.0f};
    return u;
}

GpuSurfaceParams SurfaceParams(const SurfaceLook& look, const Camera& camera,
                               int width, int height, float h) {
    GpuSurfaceParams s{};
    const float radius =
        look.sphere_radius > 0.0f ? look.sphere_radius * h : 0.5f * h;
    s.misc = Vec4{radius, camera.nearZ, look.edge_stop, 0.0f};
    s.screen = Vec4{float(width), float(height), 1.0f / float(width),
                    1.0f / float(height)};
    s.sun = Vec4{look.sun_dir.x, look.sun_dir.y, look.sun_dir.z,
                 look.sun_intensity};
    s.water = Vec4{look.water_tint.x, look.water_tint.y, look.water_tint.z,
                   look.refract};
    s.sky = Vec4{look.sky_horizon.x, look.sky_horizon.y, look.sky_horizon.z, 0.0f};
    return s;
}

}  // namespace

void FluidSurface::DrawDepth(rhi::Encoder& enc, const FluidSim& sim,
                             const Camera& camera, const SurfaceLook& look,
                             int width, int height) {
    Impl& im = *impl_;
    if (!Valid(im.depth) || sim.Count() <= 0) return;
    const FrameUniforms u = SurfaceUniforms(camera, width, height);
    const GpuSurfaceParams s =
        SurfaceParams(look, camera, width, height, sim.SmoothingRadius());
    enc.SetPipeline(im.depth);
    enc.SetCull(rhi::Cull::Back, rhi::Winding::CounterClockwise);
    enc.SetVertexBuffer(im.sphere_vb, 0, 0);
    enc.SetVertexBuffer(sim.ParticleBuffer(), 0, 1);
    enc.SetVertexBytes(&u, sizeof(u), 2);
    enc.SetVertexBytes(&s, sizeof(s), 3);
    enc.DrawIndexedInstancedU32(im.sphere_ib, std::size_t(im.sphere_indices),
                                std::size_t(sim.Count()));
}

void FluidSurface::SmoothH(rhi::Encoder& enc, const Camera& camera,
                             const SurfaceLook& look, int width, int height) {
    Impl& im = *impl_;
    if (!Valid(im.smooth_h) || !Valid(im.depth_tex)) return;
    const GpuSurfaceParams s =
        SurfaceParams(look, camera, width, height, /*h=*/0.0f);
    enc.SetPipeline(im.smooth_h);
    enc.SetFragmentTexture(im.depth_tex, 0);
    enc.SetFragmentBytes(&s, sizeof(s), 2);
    enc.Draw(3);
}

void FluidSurface::SmoothV(rhi::Encoder& enc, const Camera& camera,
                             const SurfaceLook& look, int width, int height) {
    Impl& im = *impl_;
    if (!Valid(im.smooth_v) || !Valid(im.smooth_a)) return;
    const GpuSurfaceParams s =
        SurfaceParams(look, camera, width, height, /*h=*/0.0f);
    enc.SetPipeline(im.smooth_v);
    enc.SetFragmentTexture(im.smooth_a, 0);
    enc.SetFragmentBytes(&s, sizeof(s), 2);
    enc.Draw(3);
}

void FluidSurface::DrawShade(rhi::Encoder& enc, const Camera& camera,
                             const SurfaceLook& look, int width, int height,
                             rhi::TextureId background,
                             rhi::TextureId tank_depth) {
    Impl& im = *impl_;
    if (!Valid(im.shade)) return;
    const FrameUniforms u = SurfaceUniforms(camera, width, height);
    const GpuSurfaceParams s =
        SurfaceParams(look, camera, width, height, /*h=*/0.0f);
    enc.SetPipeline(im.shade);
    enc.SetFragmentTexture(im.smooth_b, 0);
    enc.SetFragmentTexture(background, 1);
    enc.SetFragmentTexture(tank_depth, 2);
    enc.SetFragmentBytes(&u, sizeof(u), 1);
    enc.SetFragmentBytes(&s, sizeof(s), 2);
    enc.Draw(3);
}

rhi::TextureId FluidSurface::Depth() const { return impl_->depth_tex; }
rhi::TextureId FluidSurface::SmoothH() const { return impl_->smooth_a; }
rhi::TextureId FluidSurface::Smooth() const { return impl_->smooth_b; }

}  // namespace eng
