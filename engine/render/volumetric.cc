#include "engine/render/volumetric.h"

#include <cmath>
#include <cstring>

#include "engine/shaders/shader_types.h"

namespace eng {
namespace {

constexpr char kShaderTypesSrc[] = {
#embed "engine/shaders/shader_types.h"
    , 0};
constexpr char kVolumetricSrc[] = {
#embed "engine/shaders/volumetric.metal"
    , 0};

std::string Source() {
    return std::string(kShaderTypesSrc) + "\n" + kVolumetricSrc;
}

static_assert(sizeof(GpuVolumetrics) == 96, "GpuVolumetrics layout drifted");

// R2, the low-discrepancy sequence. The slice jitter has to differ every frame
// and has to spread evenly over 0..1 -- a random value clusters and a simple
// counter marches, and both show as a pattern crawling through the fog.
float JitterFor(std::uint64_t frame) {
    constexpr double kPhi = 1.32471795724474602596;  // the plastic number
    const double a = 1.0 / kPhi;
    const double x = a * double(frame);
    return float(x - std::floor(x));
}

}  // namespace

struct Volumetrics::Impl {
    rhi::Device* dev = nullptr;
    VolumetricConfig cfg;
    rhi::ComputePipelineId scatter;
    rhi::ComputePipelineId integrate;
    rhi::PipelineId apply;
    // TWO volumes, not one. The integration pass reads froxel k-1's result
    // while writing froxel k's, and a single texture would have it reading
    // values it had already overwritten -- which is not a race here, because
    // one thread owns a whole column, but it would become one the moment the
    // integration were parallelised across z.
    rhi::TextureId raw;
    rhi::TextureId integrated;
    rhi::SamplerId sampler;
    // A depth texture to bind when the caller has no shadow map. Sampling an
    // unbound depth texture is undefined rather than merely dark.
    rhi::TextureId dummy_shadow;
};

Volumetrics::Volumetrics() : impl_(std::make_unique<Impl>()) {}
Volumetrics::~Volumetrics() = default;

std::unique_ptr<Volumetrics> Volumetrics::Create(rhi::Device& dev,
                                                 std::string& error,
                                                 rhi::Format scene_color) {
    std::unique_ptr<Volumetrics> v(new Volumetrics());
    Impl& im = *v->impl_;
    im.dev = &dev;
    const std::string src = Source();
    im.scatter = dev.CreateComputePipeline(src, "cs_volumetric_scatter", error);
    if (!Valid(im.scatter)) return nullptr;
    im.integrate = dev.CreateComputePipeline(src, "cs_volumetric_integrate", error);
    if (!Valid(im.integrate)) return nullptr;

    rhi::PipelineDesc pd;
    pd.source = src;
    pd.vertex_fn = "vs_volumetric_apply";
    pd.fragment_fn = "fs_volumetric_apply";
    pd.color = scene_color;
    pd.depth = false;
    // PREMULTIPLIED alpha: the fragment writes what the fog adds in rgb and how
    // much of the scene it hides in a, so one blend does both jobs. Ordinary
    // alpha blending would multiply the added light by its own coverage twice.
    pd.blend = rhi::Blend::Alpha;
    pd.depth_write = false;
    im.apply = dev.CreatePipeline(pd, error);
    if (!Valid(im.apply)) return nullptr;

    im.raw = dev.CreateStorageTexture3D(ENG_FROXEL_X, ENG_FROXEL_Y, ENG_FROXEL_Z,
                                        rhi::Format::RGBA16Float);
    im.integrated = dev.CreateStorageTexture3D(
        ENG_FROXEL_X, ENG_FROXEL_Y, ENG_FROXEL_Z, rhi::Format::RGBA16Float);
    im.sampler = dev.CreateSampler(rhi::Filter::Linear, rhi::Wrap::Clamp);
    im.dummy_shadow = dev.CreateShadowMap(4);
    if (!Valid(im.raw) || !Valid(im.integrated) || !Valid(im.sampler) ||
        !Valid(im.dummy_shadow)) {
        error = "could not allocate the froxel volume";
        return nullptr;
    }
    return v;
}

void Volumetrics::SetConfig(const VolumetricConfig& c) { impl_->cfg = c; }
const VolumetricConfig& Volumetrics::Config() const { return impl_->cfg; }
rhi::TextureId Volumetrics::Volume() const { return impl_->integrated; }

namespace {

GpuVolumetrics BlockFor(const VolumetricConfig& cfg, std::uint64_t frame) {
    GpuVolumetrics g{};
    g.grid = Vec4{float(ENG_FROXEL_X), float(ENG_FROXEL_Y), float(ENG_FROXEL_Z),
                  0.0f};
    const float nearZ = std::max(cfg.near_distance, 1e-3f);
    const float farZ = std::max(cfg.far_distance, nearZ * 2.0f);
    g.range = Vec4{nearZ, farZ, std::log(farZ / nearZ), JitterFor(frame)};
    g.scatter = Vec4{cfg.scattering.x, cfg.scattering.y, cfg.scattering.z,
                     cfg.extinction};
    g.ambient = Vec4{cfg.ambient.x, cfg.ambient.y, cfg.ambient.z, 0.0f};
    g.medium = Vec4{cfg.base_density, cfg.height_density, cfg.height_reference,
                    cfg.height_falloff};
    g.medium2 = Vec4{std::clamp(cfg.anisotropy, -0.95f, 0.95f), 0.0f, 0.0f, 0.0f};
    return g;
}

// The froxel passes need the same FrameUniforms the surfaces get, but nothing
// about a material. Filled here rather than reusing the renderer's per-draw
// slice: that slice is allocated inside a draw loop and recycled, and a compute
// pass that ran before any draw would be reading whatever the last frame left.
FrameUniforms UniformsFor(const Scene& scene, int width, int height) {
    FrameUniforms u{};
    const float aspect = float(width) / float(std::max(height, 1));
    u.viewProj = scene.camera.ViewProj(aspect);
    u.lightDir = scene.lightDir;
    u.lightColor = scene.lightColor;
    u.eyePos = Vec4{scene.camera.eye.x, scene.camera.eye.y, scene.camera.eye.z,
                    1.0f};
    const Vec3 vd = Normalize(scene.camera.target - scene.camera.eye);
    u.viewDir = Vec4{vd.x, vd.y, vd.z, 0.0f};
    u.ssao = Vec4{std::max(scene.camera.nearZ, 1e-4f),
                  1.0f / std::tan(scene.camera.fovY * 0.5f), aspect, 0.0f};
    // surface.z carries "are shadows on", which is what the scatter pass tests
    // before doing a cascade lookup.
    u.surface = Vec4{1.0f, 0.0f, scene.shadowExtent > 0.0f ? 1.0f : 0.0f, 1e30f};
    return u;
}

}  // namespace

void Volumetrics::Build(rhi::ComputeEncoder& enc, const Scene& scene, int width,
                        int height, std::uint64_t frame,
                        rhi::TextureId shadow_map, rhi::BufferId cascades,
                        std::size_t cascade_offset, rhi::BufferId lights,
                        std::size_t light_offset, int light_count) {
    const GpuVolumetrics g = BlockFor(impl_->cfg, frame);
    FrameUniforms u = UniformsFor(scene, width, height);
    u.lighting = Vec4{float(light_count), 0.0f, 1.0f, 0.0f};
    if (!Valid(shadow_map)) u.surface.z = 0.0f;

    enc.SetPipeline(impl_->scatter);
    enc.SetBytes(&g, sizeof(g), 0);
    enc.SetBytes(&u, sizeof(u), 1);
    if (Valid(lights)) enc.SetBuffer(lights, light_offset, 2);
    if (Valid(cascades)) enc.SetBuffer(cascades, cascade_offset, 3);
    enc.SetTexture(Valid(shadow_map) ? shadow_map : impl_->dummy_shadow, 0);
    enc.SetTexture(impl_->dummy_shadow, 1);
    enc.SetSampler(impl_->sampler, 0);
    enc.SetTexture(impl_->raw, 2);
    enc.Dispatch3D(ENG_FROXEL_X, ENG_FROXEL_Y, ENG_FROXEL_Z, 4, 4, 4);

    enc.SetPipeline(impl_->integrate);
    enc.SetBytes(&g, sizeof(g), 0);
    enc.SetTexture(impl_->raw, 0);
    enc.SetTexture(impl_->integrated, 1);
    // 2D: one thread per COLUMN. The prefix sum walks z inside the thread, so
    // dispatching over z as well would have every slice recomputing the whole
    // column in front of it.
    enc.Dispatch2D(ENG_FROXEL_X, ENG_FROXEL_Y, 8, 8);
}

void Volumetrics::Apply(rhi::Encoder& enc, const Scene& scene, int width,
                        int height, rhi::TextureId scene_depth) {
    const GpuVolumetrics g = BlockFor(impl_->cfg, 0);
    const FrameUniforms u = UniformsFor(scene, width, height);
    enc.SetPipeline(impl_->apply);
    enc.SetCull(rhi::Cull::None, rhi::Winding::CounterClockwise);
    enc.SetFragmentBytes(&g, sizeof(g), 0);
    enc.SetFragmentBytes(&u, sizeof(u), 1);
    enc.SetFragmentTexture(scene_depth, 0);
    enc.SetFragmentTexture(impl_->integrated, 1);
    enc.SetFragmentSampler(impl_->sampler, 0);
    enc.Draw(3);
}

}  // namespace eng
