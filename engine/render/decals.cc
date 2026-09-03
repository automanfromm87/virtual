#include "engine/render/decals.h"

#include <algorithm>
#include <cstring>

#include "engine/shaders/shader_types.h"

namespace eng {
namespace {

constexpr char kShaderTypesSrc[] = {
#embed "engine/shaders/shader_types.h"
    , 0};
constexpr char kDecalSrc[] = {
#embed "engine/shaders/decal.metal"
    , 0};

std::string Source() { return std::string(kShaderTypesSrc) + "\n" + kDecalSrc; }

struct DecalParams {
    Mat4 invViewProj;
    Mat4 viewProj;
    Vec4 screen;
};
static_assert(sizeof(DecalParams) == 144, "DecalParams layout drifted");

struct GpuDecal {
    Mat4 invModel;
    Mat4 model;
    Vec4 tint;
    Vec4 params;
};
static_assert(sizeof(GpuDecal) == 160, "GpuDecal layout drifted");

}  // namespace

struct DecalSystem::Impl {
    rhi::Device* dev = nullptr;
    rhi::PipelineId pipeline;
    rhi::SamplerId sampler;
    rhi::BufferId instances;
    int capacity = 0;
    int last_draws = 0;
    int last_decals = 0;
    std::vector<const Decal*> sorted;
};

DecalSystem::DecalSystem() : impl_(std::make_unique<Impl>()) {}
DecalSystem::~DecalSystem() = default;

std::unique_ptr<DecalSystem> DecalSystem::Create(rhi::Device& dev,
                                                 std::string& error,
                                                 rhi::Format gbuffer,
                                                 int capacity) {
    std::unique_ptr<DecalSystem> d(new DecalSystem());
    Impl& im = *d->impl_;
    im.dev = &dev;
    im.capacity = std::max(1, capacity);

    rhi::PipelineDesc pd;
    pd.source = Source();
    pd.vertex_fn = "vs_decal";
    pd.fragment_fn = "fs_decal";
    pd.color = gbuffer;
    pd.blend = rhi::Blend::Alpha;
    // NO DEPTH AT ALL, neither tested nor written.
    //
    // Not an optimisation. The box's own depth is meaningless -- it is a volume,
    // not a surface, and the fragment being shaded belongs to whatever is
    // inside it. Testing the box against the depth buffer would clip the decal
    // wherever the box is behind the geometry it is projecting onto, which is
    // most of the box.
    //
    // The consequence is that the camera can enter the box and the decal keeps
    // drawing, because the far faces are still ahead. That is why the cull is
    // front-and-back below.
    pd.depth = false;
    pd.depth_write = false;
    im.pipeline = dev.CreatePipeline(pd, error);
    if (!Valid(im.pipeline)) return nullptr;

    im.sampler = dev.CreateSampler(rhi::Filter::Linear, rhi::Wrap::Clamp);
    im.instances = dev.CreateDynamicBuffer(sizeof(GpuDecal) * std::size_t(im.capacity));
    if (!Valid(im.instances)) {
        error = "could not allocate the decal instance buffer";
        return nullptr;
    }
    return d;
}

int DecalSystem::LastDrawCalls() const { return impl_->last_draws; }
int DecalSystem::LastDecalCount() const { return impl_->last_decals; }
int DecalSystem::Capacity() const { return impl_->capacity; }

void DecalSystem::Draw(rhi::Encoder& enc, const Camera& camera, int width,
                       int height, rhi::TextureId depth,
                       rhi::TextureId normal_metal, std::span<const Decal> decals) {
    Impl& im = *impl_;
    im.last_draws = 0;
    im.last_decals = 0;
    if (!Valid(im.pipeline) || decals.empty() || width <= 0 || height <= 0) return;
    if (!Valid(depth) || !Valid(normal_metal)) return;

    // GROUPED BY TEXTURE, because that is the only thing that forces a
    // separate draw. Everything else about a decal -- its transform, its tint,
    // its fade -- travels in the instance buffer.
    im.sorted.clear();
    im.sorted.reserve(decals.size());
    for (const Decal& d : decals)
        if (Valid(d.texture)) im.sorted.push_back(&d);
    if (im.sorted.empty()) return;
    std::sort(im.sorted.begin(), im.sorted.end(),
              [](const Decal* a, const Decal* b) { return a->texture.v < b->texture.v; });

    DecalParams p{};
    const float aspect = float(width) / float(height);
    p.viewProj = camera.ViewProj(aspect);
    p.invViewProj = Inverse(p.viewProj);
    p.screen = Vec4{float(width), float(height), 1.0f / float(width),
                    1.0f / float(height)};

    auto* dst = static_cast<GpuDecal*>(im.dev->MapBuffer(im.instances));
    if (!dst) return;

    enc.SetPipeline(im.pipeline);
    // BOTH FACES. A decal box the camera is inside shows only its back faces,
    // and culling them makes the decal vanish exactly when the player walks up
    // to it -- which reads as the decal fading out on approach.
    enc.SetCull(rhi::Cull::None, rhi::Winding::CounterClockwise);
    enc.SetVertexBytes(&p, sizeof(p), 0);
    enc.SetFragmentBytes(&p, sizeof(p), 0);
    enc.SetFragmentTexture(depth, 0);
    enc.SetFragmentTexture(normal_metal, 1);
    enc.SetFragmentSampler(im.sampler, 0);

    std::size_t written = 0;
    std::size_t run_start = 0;
    for (std::size_t i = 0; i <= im.sorted.size(); ++i) {
        const bool flush =
            i == im.sorted.size() ||
            (i > run_start && im.sorted[i]->texture.v != im.sorted[run_start]->texture.v) ||
            written >= std::size_t(im.capacity);
        if (flush && i > run_start) {
            enc.SetFragmentTexture(im.sorted[run_start]->texture, 2);
            // BOTH STAGES. The vertex shader reads the transform to place the
            // box and the fragment shader reads it again to project -- binding
            // only the vertex side leaves the fragment side reading an unbound
            // buffer, which is zeros: an identity-free matrix, a zero tint, and
            // every fragment discarded for having no alpha. The decal simply
            // does not appear, with nothing anywhere saying why.
            enc.SetVertexBuffer(im.instances, sizeof(GpuDecal) * run_start, 1);
            enc.SetFragmentBuffer(im.instances, sizeof(GpuDecal) * run_start, 1);
            // 36 vertices of a unit cube, generated from the vertex id, times
            // the number of decals in this run.
            enc.DrawInstanced(36, i - run_start);
            ++im.last_draws;
            run_start = i;
        }
        if (i == im.sorted.size()) break;
        if (written >= std::size_t(im.capacity)) break;

        const Decal& d = *im.sorted[i];
        GpuDecal g{};
        g.model = d.model;
        // INVERTED ONCE HERE, on the CPU. The shader needs world -> local for
        // every pixel the box covers, and a per-pixel matrix inverse is not an
        // option -- but nor is asking the caller for both, because two matrices
        // that are supposed to be inverses of each other are two things that
        // can disagree.
        g.invModel = Inverse(d.model);
        g.tint = d.tint;
        g.params = Vec4{std::clamp(d.normal_fade, -1.0f, 0.99f),
                        std::clamp(d.opacity, 0.0f, 1.0f), 0.0f, 0.0f};
        dst[written++] = g;
    }
    im.last_decals = int(written);
}

}  // namespace eng
