#include "engine/render/post.h"

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
constexpr char kPostSrc[] = {
#embed "engine/shaders/post.metal"
    , 0};

std::string Source() { return std::string(kShaderTypesSrc) + "\n" + kPostSrc; }

struct PostParams {
    Mat4 invViewProj;
    Mat4 prevViewProj;
    Vec4 eye;
    Vec4 fog;
    Vec4 fog2;
    Vec4 dof;
    Vec4 tune;
    Vec4 screen;
    EngUVec4 bins;
    Vec4 lum;
};
static_assert(sizeof(PostParams) == 256, "PostParams layout drifted");

constexpr int kBins = 256;

// A HALTON sequence, base 2 and base 3.
//
// Not random and not a regular grid. A regular grid of N offsets repeats every
// N frames, and any object moving at the wrong speed beats against the period
// and shimmers. Random offsets clump -- some frames land almost on top of each
// other and contribute nothing. Halton is low-discrepancy: every prefix of it
// is spread out, so the accumulation is even however many frames have gone by.
float Halton(int index, int base) {
    float result = 0.0f, f = 1.0f / float(base);
    int i = index;
    while (i > 0) {
        result += f * float(i % base);
        i /= base;
        f /= float(base);
    }
    return result;
}

}  // namespace

struct PostStack::Impl {
    rhi::Device* dev = nullptr;
    rhi::Format hdr_format = rhi::Format::RGBA16Float;
    int width = 0, height = 0;

    rhi::ComputePipelineId histogram, resolve, velocity_cs;
    rhi::PipelineId fog, dof, motion_blur, taa;
    rhi::SamplerId sampler;

    rhi::BufferId histogram_buf, exposure_buf;
    rhi::TextureId velocity;
    rhi::TextureId scratch, output;
    // TWO history buffers, swapped each frame. One would mean the resolve reads
    // and writes the same texture, which is undefined everywhere.
    rhi::TextureId history[2];
    int history_index = 0;

    PostParams params{};
    Mat4 prev_view_proj = Mat4::Identity();
    bool have_prev = false;
    std::uint64_t frame = 0;
    Vec2 jitter{0.0f, 0.0f};
    float dt = 0.0f;

    void Resize(int w, int h);
};

PostStack::PostStack() : impl_(std::make_unique<Impl>()) {}
PostStack::~PostStack() = default;

std::unique_ptr<PostStack> PostStack::Create(rhi::Device& dev, std::string& error,
                                             rhi::Format hdr) {
    std::unique_ptr<PostStack> p(new PostStack());
    Impl& im = *p->impl_;
    im.dev = &dev;
    im.hdr_format = hdr;

    const std::string src = Source();
    im.histogram = dev.CreateComputePipeline(src, "cs_luminance_histogram", error);
    if (!Valid(im.histogram)) return nullptr;
    im.resolve = dev.CreateComputePipeline(src, "cs_luminance_resolve", error);
    if (!Valid(im.resolve)) return nullptr;
    im.velocity_cs = dev.CreateComputePipeline(src, "cs_velocity", error);
    if (!Valid(im.velocity_cs)) return nullptr;

    const auto make = [&](const char* fragment, rhi::Blend blend) {
        rhi::PipelineDesc pd;
        pd.source = src;
        pd.vertex_fn = "vs_post";
        pd.fragment_fn = fragment;
        pd.color = hdr;
        pd.blend = blend;
        pd.depth = false;
        pd.depth_write = false;
        return dev.CreatePipeline(pd, error);
    };
    // Fog ALPHA BLENDS over the scene; the others replace it. Fog is the only
    // one that can, because its output is a colour and a coverage rather than a
    // filtered version of what was there -- which is also why it needs no
    // scratch target and the others do.
    im.fog = make("fs_fog", rhi::Blend::Alpha);
    if (!Valid(im.fog)) return nullptr;
    im.dof = make("fs_dof", rhi::Blend::None);
    if (!Valid(im.dof)) return nullptr;
    im.motion_blur = make("fs_motion_blur", rhi::Blend::None);
    if (!Valid(im.motion_blur)) return nullptr;
    im.taa = make("fs_taa", rhi::Blend::None);
    if (!Valid(im.taa)) return nullptr;

    im.sampler = dev.CreateSampler(rhi::Filter::Linear, rhi::Wrap::Clamp);
    const std::vector<std::uint32_t> zeros(kBins, 0u);
    im.histogram_buf = dev.CreateBuffer(zeros.data(), sizeof(std::uint32_t) * kBins);
    const float one = 1.0f;
    im.exposure_buf = dev.CreateBuffer(&one, sizeof(float));
    if (!Valid(im.histogram_buf) || !Valid(im.exposure_buf)) {
        error = "could not allocate the exposure buffers";
        return nullptr;
    }
    return p;
}

void PostStack::Impl::Resize(int w, int h) {
    if (w == width && h == height) return;
    for (rhi::TextureId t : {velocity, scratch, output, history[0], history[1]})
        if (Valid(t)) dev->DestroyTexture(t);
    width = w;
    height = h;
    if (w <= 0 || h <= 0) {
        velocity = scratch = output = history[0] = history[1] = rhi::TextureId{};
        return;
    }
    // RG16Float: a velocity is two numbers, both small and both needing sign
    // and sub-pixel precision. Eight-bit would quantise a one-pixel motion to
    // nothing at 1080p, and RGBA16 would waste half the bandwidth of a texture
    // that is read three times a frame.
    velocity = dev->CreateStorageTexture2D(w, h, rhi::Format::RG16Float, 1);
    scratch = dev->CreateRenderTarget(w, h, hdr_format);
    output = dev->CreateRenderTarget(w, h, hdr_format);
    history[0] = dev->CreateRenderTarget(w, h, hdr_format);
    history[1] = dev->CreateRenderTarget(w, h, hdr_format);
}

void PostStack::BeginFrame(const Camera& camera, int width, int height, float dt) {
    Impl& im = *impl_;
    im.Resize(width, height);
    im.dt = dt;
    ++im.frame;

    const float aspect = height > 0 ? float(width) / float(height) : 1.0f;
    // THE UNJITTERED matrix, for both the reprojection and the world
    // reconstruction. Velocity is where a surface moved; a sub-pixel wobble
    // applied to one of the two frames and not the other would appear in the
    // answer as motion, and TAA would chase its own jitter.
    const Mat4 vp = camera.ViewProjNoJitter(aspect);
    im.params.invViewProj = Inverse(vp);
    im.params.prevViewProj = im.have_prev ? im.prev_view_proj : vp;
    im.prev_view_proj = vp;
    im.have_prev = true;

    im.params.eye = Vec4{camera.eye.x, camera.eye.y, camera.eye.z, camera.nearZ};
    im.params.fog = Vec4{config.fog_color.x, config.fog_color.y, config.fog_color.z,
                         config.fog ? config.fog_density : 0.0f};
    im.params.fog2 = Vec4{config.fog_height_falloff, config.fog_ground_height,
                          config.fog_start, config.fog_max_distance};
    im.params.dof = Vec4{config.focus_distance, config.focus_range,
                         config.max_blur_radius, 0.0f};
    im.params.tune = Vec4{config.fixed_exposure, dt, config.shutter,
                          config.taa_feedback};
    im.params.screen = Vec4{float(width), float(height),
                            width > 0 ? 1.0f / float(width) : 0.0f,
                            height > 0 ? 1.0f / float(height) : 0.0f};
    im.params.bins.x = std::uint32_t(kBins);
    im.params.lum = Vec4{config.min_log_luminance,
                         config.max_log_luminance - config.min_log_luminance,
                         config.adapt_brighter, config.adapt_darker};

    if (config.taa && width > 0 && height > 0) {
        // 8 is enough to converge and short enough that a slowly moving edge
        // does not take a second to settle. The offsets are centred on zero --
        // Halton runs 0..1 -- and scaled to one pixel in NDC, where the whole
        // screen is 2 units across.
        const int index = int(im.frame % 8) + 1;
        im.jitter = Vec2{(Halton(index, 2) - 0.5f) * 2.0f / float(std::max(width, 1)),
                         (Halton(index, 3) - 0.5f) * 2.0f / float(std::max(height, 1))};
    } else {
        im.jitter = Vec2{0.0f, 0.0f};
    }
}

Vec2 PostStack::Jitter() const { return impl_->jitter; }
rhi::TextureId PostStack::Velocity() const { return impl_->velocity; }
rhi::TextureId PostStack::Scratch() const { return impl_->scratch; }
rhi::TextureId PostStack::Output() const {
    return impl_->history[impl_->history_index];
}
rhi::BufferId PostStack::ExposureBuffer() const { return impl_->exposure_buf; }

float PostStack::LastExposure() const {
    const auto* e = static_cast<const float*>(impl_->dev->MapBuffer(impl_->exposure_buf));
    return e ? *e : 1.0f;
}

void PostStack::EndFrame() {
    // Swap so that this frame's output becomes next frame's history. Both are
    // real textures; nothing is copied.
    impl_->history_index ^= 1;
}

void PostStack::ComputeVelocity(rhi::ComputeEncoder& enc, rhi::TextureId depth) {
    Impl& im = *impl_;
    if (!Valid(im.velocity) || !Valid(depth)) return;
    enc.SetPipeline(im.velocity_cs);
    enc.SetTexture(im.velocity, 0);
    enc.SetTexture(depth, 1);
    enc.SetBytes(&im.params, sizeof(im.params), 0);
    enc.Dispatch2D(im.width, im.height);
}

void PostStack::DrawFog(rhi::Encoder& enc, rhi::TextureId depth) {
    Impl& im = *impl_;
    if (!config.fog || !Valid(im.fog) || !Valid(depth)) return;
    enc.SetPipeline(im.fog);
    enc.SetCull(rhi::Cull::None, rhi::Winding::CounterClockwise);
    enc.SetFragmentTexture(depth, 0);
    enc.SetFragmentBytes(&im.params, sizeof(im.params), 0);
    enc.SetFragmentSampler(im.sampler, 0);
    enc.Draw(3);
}

void PostStack::DrawDepthOfField(rhi::Encoder& enc, rhi::TextureId src,
                                 rhi::TextureId depth) {
    Impl& im = *impl_;
    if (!Valid(im.dof) || !Valid(src) || !Valid(depth)) return;
    enc.SetPipeline(im.dof);
    enc.SetCull(rhi::Cull::None, rhi::Winding::CounterClockwise);
    enc.SetFragmentTexture(src, 0);
    enc.SetFragmentTexture(depth, 1);
    enc.SetFragmentBytes(&im.params, sizeof(im.params), 0);
    enc.SetFragmentSampler(im.sampler, 0);
    enc.Draw(3);
}

void PostStack::DrawMotionBlur(rhi::Encoder& enc, rhi::TextureId src) {
    Impl& im = *impl_;
    if (!Valid(im.motion_blur) || !Valid(src) || !Valid(im.velocity)) return;
    enc.SetPipeline(im.motion_blur);
    enc.SetCull(rhi::Cull::None, rhi::Winding::CounterClockwise);
    enc.SetFragmentTexture(src, 0);
    enc.SetFragmentTexture(im.velocity, 1);
    enc.SetFragmentBytes(&im.params, sizeof(im.params), 0);
    enc.SetFragmentSampler(im.sampler, 0);
    enc.Draw(3);
}

void PostStack::DrawTaa(rhi::Encoder& enc, rhi::TextureId src) {
    Impl& im = *impl_;
    if (!Valid(im.taa) || !Valid(src) || !Valid(im.velocity)) return;
    enc.SetPipeline(im.taa);
    enc.SetCull(rhi::Cull::None, rhi::Winding::CounterClockwise);
    enc.SetFragmentTexture(src, 0);
    // The OTHER buffer -- the one written last frame. Reading the one being
    // written is undefined, and the swap in EndFrame is what keeps them apart.
    enc.SetFragmentTexture(im.history[im.history_index ^ 1], 1);
    enc.SetFragmentTexture(im.velocity, 2);
    enc.SetFragmentBytes(&im.params, sizeof(im.params), 0);
    enc.SetFragmentSampler(im.sampler, 0);
    enc.Draw(3);
}

void PostStack::MeterExposure(rhi::ComputeEncoder& enc, rhi::TextureId hdr) {
    Impl& im = *impl_;
    if (!config.auto_exposure || !Valid(im.histogram) || !Valid(hdr)) return;
    im.params.tune.x = std::exp2(config.exposure_compensation);
    enc.SetPipeline(im.histogram);
    enc.SetTexture(hdr, 0);
    enc.SetBuffer(im.histogram_buf, 0, 0);
    enc.SetBytes(&im.params, sizeof(im.params), 1);
    // 16x16, which is 256 threads -- exactly the number of bins, so the
    // threadgroup-local clear and merge are one bin per thread with no loop.
    enc.Dispatch2D(im.width, im.height, 16, 16);

    enc.SetPipeline(im.resolve);
    enc.SetBuffer(im.histogram_buf, 0, 0);
    enc.SetBuffer(im.exposure_buf, 0, 1);
    enc.SetBytes(&im.params, sizeof(im.params), 2);
    enc.Dispatch(1, 1);
}

}  // namespace eng
