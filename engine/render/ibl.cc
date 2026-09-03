#include "engine/render/ibl.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "engine/shaders/shader_types.h"

namespace eng {
namespace {

constexpr char kShaderTypesSrc[] = {
#embed "engine/shaders/shader_types.h"
    , 0};
constexpr char kSkySrc[] = {
#embed "engine/shaders/sky.metal"
    , 0};
constexpr char kIblSrc[] = {
#embed "engine/shaders/ibl.metal"
    , 0};

std::string SkySource() { return std::string(kShaderTypesSrc) + "\n" + kSkySrc; }
std::string IblSource() { return std::string(kShaderTypesSrc) + "\n" + kIblSrc; }

// Mirrors sky.metal's SkyParams. Checked by the static_assert below rather than
// trusted: a layout drift here does not crash, it produces a sky lit from the
// wrong direction, and nothing about that says the struct is misaligned.
struct SkyParams {
    Vec4 sun_dir;
    Vec4 ground;
    Vec4 tune;
    EngUVec4 size;
};
static_assert(sizeof(SkyParams) == 64, "SkyParams layout drifted");

struct CubeParams {
    EngUVec4 size;
    Vec4 tune;
};
static_assert(sizeof(CubeParams) == 32, "CubeParams layout drifted");

struct SkyDrawParams {
    Mat4 invViewProj;
    Vec4 eye;
};
static_assert(sizeof(SkyDrawParams) == 80, "SkyDrawParams layout drifted");

// The diffuse probe. 32 is not a compromise: irradiance is a cosine-weighted
// average over a hemisphere, which is a second-order spherical function, and no
// amount of resolution adds detail that the convolution has not already
// removed. Larger costs the full integral per texel for an identical picture.
constexpr int kIrradianceSize = 32;
// The specular chain's base. Roughness 0 samples this level directly, so it
// bounds how sharp a mirror can be -- 128 is a visibly soft mirror on a large
// surface, and the compromise is deliberate: this is a per-scene probe, not a
// planar reflection, and a sharp one would need the full radiance cube's memory
// at every roughness.
constexpr int kSpecularSize = 128;
constexpr int kSpecularMips = 6;
constexpr int kBrdfSize = 256;
// Samples per texel in the prefilter. 256 is where the noise stops being
// visible given the mip-selection trick; without that trick even 4096 sparkles.
constexpr int kPrefilterSamples = 256;

int MipCount(int size) {
    int n = 1;
    while (size > 1) {
        size /= 2;
        ++n;
    }
    return n;
}

// The CPU's copy of the atmosphere, used only for the sun's colour. Far coarser
// than the shader's -- it integrates one ray rather than a cube -- and that is
// fine, because all it has to produce is the transmittance along the single
// straight line from the sun to the viewer.
Vec3 SunTransmittance(Vec3 sun, float turbidity) {
    constexpr float kGround = 6360000.0f;
    constexpr float kTop = 6420000.0f;
    const Vec3 rayleigh{5.8e-6f, 13.5e-6f, 33.1e-6f};
    const float mie = 21e-6f * std::max(turbidity, 0.0f);

    const Vec3 eye{0.0f, kGround + 2.0f, 0.0f};
    const Vec3 dir = Normalize(sun);
    // Where the ray to the sun leaves the atmosphere.
    const float b = Dot(eye, dir);
    const float c = Dot(eye, eye) - kTop * kTop;
    const float disc = b * b - c;
    if (disc < 0.0f) return Vec3{0.0f, 0.0f, 0.0f};
    const float far = -b + std::sqrt(disc);

    constexpr int kSteps = 32;
    const float step = far / float(kSteps);
    float od_r = 0.0f, od_m = 0.0f;
    for (int i = 0; i < kSteps; ++i) {
        const Vec3 p = eye + dir * (step * (float(i) + 0.5f));
        const float h = std::max(Length(p) - kGround, 0.0f);
        od_r += std::exp(-h / 8000.0f) * step;
        od_m += std::exp(-h / 1200.0f) * step;
    }
    return Vec3{std::exp(-(rayleigh.x * od_r + mie * 1.1f * od_m)),
                std::exp(-(rayleigh.y * od_r + mie * 1.1f * od_m)),
                std::exp(-(rayleigh.z * od_r + mie * 1.1f * od_m))};
}

}  // namespace

struct Environment::Impl {
    rhi::Device* dev = nullptr;
    int cube_size = 256;
    int radiance_mips = 1;

    rhi::TextureId radiance;
    rhi::TextureId irradiance;
    rhi::TextureId specular;
    rhi::TextureId brdf_lut;
    // One write view per mip of each cube, made once at creation. Making them
    // per bake would leak a texture handle per mip per bake -- the handle table
    // has no generation counters, so a view created and dropped every frame
    // grows it without bound.
    std::vector<rhi::TextureId> radiance_views;
    std::vector<rhi::TextureId> specular_views;
    rhi::TextureId irradiance_view;
    rhi::TextureId brdf_view;

    rhi::ComputePipelineId sky_cube;
    rhi::ComputePipelineId equirect;
    rhi::ComputePipelineId irradiance_pass;
    rhi::ComputePipelineId prefilter;
    rhi::ComputePipelineId brdf;
    rhi::ComputePipelineId downsample;
    rhi::ComputePipelineId cube_readback;
    rhi::ComputePipelineId lut_readback;
    rhi::PipelineId sky_draw;

    rhi::SamplerId cube_sampler;
    rhi::SamplerId lut_sampler;

    rhi::BufferId readback;
    int readback_count = 0;
    bool brdf_baked = false;
    int bakes = 0;

    // Everything downstream of the radiance cube. Shared by BakeSky and
    // BakeEquirect, because the only difference between them is where the top
    // mip came from.
    void BakeChain(rhi::ComputeEncoder& enc);
};

Environment::Environment() : impl_(std::make_unique<Impl>()) {}
Environment::~Environment() = default;

std::unique_ptr<Environment> Environment::Create(rhi::Device& dev,
                                                 std::string& error,
                                                 int cube_size,
                                                 rhi::Format scene_color,
                                                 int samples) {
    if (cube_size < 8) {
        error = "environment cube must be at least 8 texels a side";
        return nullptr;
    }
    std::unique_ptr<Environment> e(new Environment());
    Impl& im = *e->impl_;
    im.dev = &dev;
    im.cube_size = cube_size;
    im.radiance_mips = MipCount(cube_size);

    // RGBA16Float, not 32. The radiance cube is sampled hundreds of times per
    // prefilter texel, so its bandwidth is the prefilter's cost, and half
    // floats reach 65504 -- above the sun's radiance here by a wide margin.
    im.radiance = dev.CreateCubemap(cube_size, rhi::Format::RGBA16Float, im.radiance_mips);
    im.irradiance = dev.CreateCubemap(kIrradianceSize, rhi::Format::RGBA16Float, 1);
    im.specular = dev.CreateCubemap(kSpecularSize, rhi::Format::RGBA16Float, kSpecularMips);
    im.brdf_lut = dev.CreateStorageTexture2D(kBrdfSize, kBrdfSize, rhi::Format::RG16Float, 1);
    if (!Valid(im.radiance) || !Valid(im.irradiance) || !Valid(im.specular) ||
        !Valid(im.brdf_lut)) {
        error = "could not allocate the environment textures";
        return nullptr;
    }

    for (int m = 0; m < im.radiance_mips; ++m)
        im.radiance_views.push_back(dev.CreateMipView(im.radiance, m));
    for (int m = 0; m < kSpecularMips; ++m)
        im.specular_views.push_back(dev.CreateMipView(im.specular, m));
    im.irradiance_view = dev.CreateMipView(im.irradiance, 0);
    im.brdf_view = dev.CreateMipView(im.brdf_lut, 0);

    const std::string ibl = IblSource();
    im.sky_cube = dev.CreateComputePipeline(SkySource(), "cs_sky_cube", error);
    if (!Valid(im.sky_cube)) return nullptr;
    im.equirect = dev.CreateComputePipeline(ibl, "cs_equirect_to_cube", error);
    if (!Valid(im.equirect)) return nullptr;
    im.irradiance_pass = dev.CreateComputePipeline(ibl, "cs_irradiance", error);
    if (!Valid(im.irradiance_pass)) return nullptr;
    im.prefilter = dev.CreateComputePipeline(ibl, "cs_prefilter", error);
    if (!Valid(im.prefilter)) return nullptr;
    im.brdf = dev.CreateComputePipeline(ibl, "cs_brdf_lut", error);
    if (!Valid(im.brdf)) return nullptr;
    im.downsample = dev.CreateComputePipeline(ibl, "cs_cube_downsample", error);
    if (!Valid(im.downsample)) return nullptr;
    im.cube_readback = dev.CreateComputePipeline(ibl, "cs_cube_readback", error);
    if (!Valid(im.cube_readback)) return nullptr;
    im.lut_readback = dev.CreateComputePipeline(ibl, "cs_lut_readback", error);
    if (!Valid(im.lut_readback)) return nullptr;

    rhi::PipelineDesc pd;
    pd.source = SkySource();
    pd.vertex_fn = "vs_sky";
    pd.fragment_fn = "fs_sky";
    pd.color = scene_color;
    pd.samples = samples;
    pd.depth = true;
    // TESTED, NOT WRITTEN, and drawn last among the opaque work. The sky sits
    // at the far plane, so with a reversed-Z Greater test it survives only
    // where nothing else was drawn -- which means it shades exactly the pixels
    // that need it and none of the ones already covered by geometry.
    pd.depth_write = false;
    pd.depth_compare = rhi::Compare::Greater;
    im.sky_draw = dev.CreatePipeline(pd, error);
    if (!Valid(im.sky_draw)) return nullptr;

    im.cube_sampler = dev.CreateMipSampler(rhi::Wrap::Clamp);
    im.lut_sampler = dev.CreateSampler(rhi::Filter::Linear, rhi::Wrap::Clamp);
    return e;
}

void Environment::Impl::BakeChain(rhi::ComputeEncoder& enc) {
    // 1. MIPS of the radiance cube. Not decoration: the prefilter picks a mip
    //    per sample to avoid fireflies, so without a chain every sample reads
    //    level zero and the sun sparkles across the whole probe.
    for (int m = 1; m < radiance_mips; ++m) {
        const int size = std::max(1, cube_size >> m);
        CubeParams p{};
        p.size.x = std::uint32_t(size);
        p.tune = Vec4{float(m - 1), 0.0f, 0.0f, 0.0f};
        enc.SetPipeline(downsample);
        enc.SetTexture(radiance_views[std::size_t(m)], 0);
        enc.SetTexture(radiance, 1);
        enc.SetSampler(cube_sampler, 0);
        enc.SetBytes(&p, sizeof(p), 0);
        enc.Dispatch3D(size, size, 6);
    }

    // 2. DIFFUSE irradiance.
    {
        CubeParams p{};
        p.size.x = std::uint32_t(kIrradianceSize);
        enc.SetPipeline(irradiance_pass);
        enc.SetTexture(irradiance_view, 0);
        enc.SetTexture(radiance, 1);
        enc.SetSampler(cube_sampler, 0);
        enc.SetBytes(&p, sizeof(p), 0);
        enc.Dispatch3D(kIrradianceSize, kIrradianceSize, 6);
    }

    // 3. SPECULAR, one dispatch per mip, roughness rising with the level.
    for (int m = 0; m < kSpecularMips; ++m) {
        const int size = std::max(1, kSpecularSize >> m);
        CubeParams p{};
        p.size.x = std::uint32_t(size);
        // Linear in roughness across the chain, which is the convention the
        // shader's lookup has to match: it maps roughness to a mip with the
        // same straight line. A perceptual curve here would look better and
        // would have to be inverted exactly in the shader, and a mismatch shows
        // up as reflections that are sharp or blurry at the wrong roughness.
        p.tune = Vec4{float(m) / float(kSpecularMips - 1), float(cube_size),
                      float(kPrefilterSamples), 0.0f};
        enc.SetPipeline(prefilter);
        enc.SetTexture(specular_views[std::size_t(m)], 0);
        enc.SetTexture(radiance, 1);
        enc.SetSampler(cube_sampler, 0);
        enc.SetBytes(&p, sizeof(p), 0);
        enc.Dispatch3D(size, size, 6);
    }

    // 4. The BRDF table, ONCE. It is a property of the BRDF, not of the sky.
    if (!brdf_baked) {
        CubeParams p{};
        p.size.x = std::uint32_t(kBrdfSize);
        enc.SetPipeline(brdf);
        enc.SetTexture(brdf_view, 0);
        enc.SetBytes(&p, sizeof(p), 0);
        enc.Dispatch2D(kBrdfSize, kBrdfSize);
        brdf_baked = true;
    }
    ++bakes;
}

void Environment::BakeSky(rhi::ComputeEncoder& enc, const SkyConfig& sky) {
    const Vec3 sun = Normalize(sky.sun_direction);
    SkyParams p{};
    p.sun_dir = Vec4{sun.x, sun.y, sun.z, sky.sun_angular_radius};
    p.ground = Vec4{sky.ground_albedo.x, sky.ground_albedo.y, sky.ground_albedo.z,
                    sky.turbidity};
    p.tune = Vec4{sky.sun_intensity, sky.exposure, sky.night_lift, 0.0f};
    p.size.x = std::uint32_t(impl_->cube_size);

    enc.SetPipeline(impl_->sky_cube);
    enc.SetTexture(impl_->radiance_views[0], 0);
    enc.SetBytes(&p, sizeof(p), 0);
    enc.Dispatch3D(impl_->cube_size, impl_->cube_size, 6);
    impl_->BakeChain(enc);
}

void Environment::BakeEquirect(rhi::ComputeEncoder& enc, rhi::TextureId equirect) {
    if (!Valid(equirect)) return;
    CubeParams p{};
    p.size.x = std::uint32_t(impl_->cube_size);
    enc.SetPipeline(impl_->equirect);
    enc.SetTexture(impl_->radiance_views[0], 0);
    enc.SetTexture(equirect, 1);
    enc.SetSampler(impl_->lut_sampler, 0);
    enc.SetBytes(&p, sizeof(p), 0);
    enc.Dispatch3D(impl_->cube_size, impl_->cube_size, 6);
    impl_->BakeChain(enc);
}

void Environment::DrawSky(rhi::Encoder& enc, const Camera& camera, int width,
                          int height, float intensity) {
    if (!Valid(impl_->sky_draw) || width <= 0 || height <= 0) return;
    SkyDrawParams p{};
    p.invViewProj = Inverse(camera.ViewProj(float(width) / float(height)));
    p.eye = Vec4{camera.eye.x, camera.eye.y, camera.eye.z, intensity};

    enc.SetPipeline(impl_->sky_draw);
    enc.SetCull(rhi::Cull::None, rhi::Winding::CounterClockwise);
    enc.SetVertexBytes(&p, sizeof(p), 0);
    enc.SetFragmentBytes(&p, sizeof(p), 0);
    enc.SetFragmentTexture(impl_->radiance, 0);
    enc.SetFragmentSampler(impl_->cube_sampler, 0);
    enc.Draw(3);
}

rhi::TextureId Environment::Radiance() const { return impl_->radiance; }
int Environment::CubeSize() const { return impl_->cube_size; }
int Environment::SpecularMips() const { return kSpecularMips; }
int Environment::BakeCount() const { return impl_->bakes; }

EnvironmentBindings Environment::Bindings() const {
    EnvironmentBindings b;
    b.irradiance = impl_->irradiance;
    b.specular = impl_->specular;
    b.brdf_lut = impl_->brdf_lut;
    b.cube_sampler = impl_->cube_sampler;
    b.lut_sampler = impl_->lut_sampler;
    b.specular_mips = kSpecularMips;
    return b;
}

Vec3 Environment::SunColor(const SkyConfig& sky) {
    const Vec3 t = SunTransmittance(Normalize(sky.sun_direction), sky.turbidity);
    const float k = sky.sun_intensity * sky.exposure;
    return Vec3{t.x * k, t.y * k, t.z * k};
}

void Environment::ApplyTo(Scene* scene, const SkyConfig& sky) {
    if (!scene) return;
    const Vec3 sun = Normalize(sky.sun_direction);
    scene->lightDir = Vec4{sun.x, sun.y, sun.z, 0.0f};
    const Vec3 c = SunColor(sky);
    scene->lightColor = Vec4{c.x, c.y, c.z, 1.0f};
    // The hemisphere ambient is left as a FALLBACK, scaled to roughly what the
    // sky contributes, and it is not dead code: the deferred path and any
    // material without the IBL bindings still read it, and a scene that looks
    // completely different depending on which path drew it is worse than one
    // that is slightly flatter on the fallback.
    const float up = std::max(sun.y, 0.0f);
    scene->ambientSky = Vec3{0.10f + 0.22f * up, 0.13f + 0.26f * up,
                             0.20f + 0.32f * up} * sky.exposure;
    scene->ambientGround =
        Vec3{sky.ground_albedo.x, sky.ground_albedo.y, sky.ground_albedo.z} *
        (0.25f * up * sky.exposure);
}

void Environment::ReadCube(rhi::ComputeEncoder& enc, Probe which, int face_size,
                           float lod) {
    face_size = std::clamp(face_size, 1, 256);
    const int count = face_size * face_size * 6;
    if (!Valid(impl_->readback) || impl_->readback_count < count) {
        impl_->readback = impl_->dev->CreateStorageBuffer(sizeof(Vec4) * std::size_t(count));
        impl_->readback_count = count;
    }
    rhi::TextureId src = impl_->radiance;
    if (which == Probe::Irradiance) src = impl_->irradiance;
    if (which == Probe::Specular) src = impl_->specular;

    CubeParams p{};
    p.size.x = std::uint32_t(face_size);
    p.tune = Vec4{lod, 0.0f, 0.0f, 0.0f};
    enc.SetPipeline(impl_->cube_readback);
    enc.SetTexture(src, 0);
    enc.SetSampler(impl_->cube_sampler, 0);
    enc.SetBuffer(impl_->readback, 0, 0);
    enc.SetBytes(&p, sizeof(p), 1);
    enc.Dispatch3D(face_size, face_size, 6);
}

void Environment::ReadLut(rhi::ComputeEncoder& enc, int size) {
    size = std::clamp(size, 1, 256);
    const int count = size * size;
    if (!Valid(impl_->readback) || impl_->readback_count < count) {
        impl_->readback = impl_->dev->CreateStorageBuffer(sizeof(Vec4) * std::size_t(count));
        impl_->readback_count = count;
    }
    CubeParams p{};
    p.size.x = std::uint32_t(size);
    enc.SetPipeline(impl_->lut_readback);
    enc.SetTexture(impl_->brdf_lut, 0);
    enc.SetSampler(impl_->lut_sampler, 0);
    enc.SetBuffer(impl_->readback, 0, 0);
    enc.SetBytes(&p, sizeof(p), 1);
    enc.Dispatch2D(size, size);
}

std::vector<Vec4> Environment::TakeCube() const {
    std::vector<Vec4> out;
    const auto* src = static_cast<const Vec4*>(impl_->dev->MapBuffer(impl_->readback));
    if (!src) return out;
    out.assign(src, src + impl_->readback_count);
    return out;
}

std::vector<Vec4> Environment::TakeLut() const { return TakeCube(); }

}  // namespace eng
