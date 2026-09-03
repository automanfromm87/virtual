// PURE C++. No Metal, no Objective-C, no Apple frameworks — everything GPU
// goes through engine/rhi.
#include "engine/render/renderer.h"

#include <cstddef>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "engine/core/math.h"
#include "engine/geometry/mesh.h"
#include "engine/resource/handles.h"
#include "engine/shaders/shader_types.h"

namespace eng {
namespace {

// Shader source is baked into the binary at compile time by #embed (a clang C23
// extension that also works in C++20). No genrule, no xxd, no runfiles, no
// filesystem access at runtime.
constexpr char kShaderTypesSrc[] = {
#embed "engine/shaders/shader_types.h"
    , 0};
constexpr char kTriangleSrc[] = {
#embed "engine/shaders/triangle.metal"
    , 0};
constexpr char kLitSrc[] = {
#embed "engine/shaders/lit.metal"
    , 0};
constexpr char kCompositeSrc[] = {
#embed "engine/shaders/composite.metal"
    , 0};
constexpr char kShadowSrc[] = {
#embed "engine/shaders/shadow.metal"
    , 0};
constexpr char kSsaoSrc[] = {
#embed "engine/shaders/ssao.metal"
    , 0};

// CPU and GPU must agree byte-for-byte. Assert it — do not hope for it.
static_assert(sizeof(FrameUniforms) == 304, "FrameUniforms layout drifted");
static_assert(sizeof(VertexIn) == 64, "VertexIn layout drifted");
static_assert(offsetof(VertexIn, normal) == 16, "GPU float3 occupies 16 bytes");
static_assert(offsetof(VertexIn, color) == 32, "GPU float3 occupies 16 bytes");
static_assert(offsetof(VertexIn, uv) == 48, "uv must follow color");

// The triangle predates the Camera type and bakes its vertices straight into
// world space, so it carries its own projection constants.
constexpr float kFovY = 1.0472f;  // 60 degrees
constexpr float kNearZ = 0.1f;
constexpr Vec4 kLightDir{0.4082f, 0.8165f, 0.4082f, 0.0f};
constexpr Vec3 kSphereEye{0.0f, 0.0f, 3.0f};

// Uniform slot 1 in both stages; vertex buffer in slot 0.
constexpr int kVertexSlot = 0;
constexpr int kUniformSlot = 1;

// Ring capacity per frame slot. Exceeding it drops draws rather than
// corrupting the frame — see the clamp in DrawScene.
constexpr std::size_t kMaxInstancesPerFrame = 1024;

std::size_t AlignUp(std::size_t v, std::size_t a) { return ((v + a - 1) / a) * a; }

std::string ShaderSource(const char* body) {
    return std::string(kShaderTypesSrc) + "\n" + body;
}

}  // namespace

// One entry of the mesh registry: the GPU buffers plus the CPU-side bounds
// that frustum culling needs.
struct GpuMesh {
    rhi::BufferId vb;
    rhi::BufferId ib;
    std::size_t index_count = 0;
    Bounds bounds;
};

struct GpuMaterial {
    rhi::PipelineId pipeline;
    rhi::Cull cull = rhi::Cull::Back;
    bool depth_test = true;
    bool transparent = false;
    Vec4 base_color{1.0f, 1.0f, 1.0f, 1.0f};
    float roughness = 0.5f;
    float metallic = 0.0f;
    rhi::TextureId albedo;
    rhi::TextureId roughness_map;
};

// A survivor of culling, with the keys it gets sorted on.
struct DrawItem {
    const Instance* inst = nullptr;
    int index = 0;  // position in Scene::instances, for LastDrawOrder()
    std::uint32_t pipeline = 0;
    float depth = 0.0f;  // distance from the eye
    bool transparent = false;
};

struct Renderer::Impl {
    rhi::Device* dev = nullptr;
    rhi::Format color = rhi::Format::BGRA8Unorm;

    // Index 0 is the null handle in both tables, so handles are 1-based and
    // match the kMesh* / kMaterial* constants in engine/resource/handles.h.
    std::vector<GpuMesh> meshes{GpuMesh{}};
    std::vector<GpuMaterial> materials{GpuMaterial{}};

    // Keyed on the state that actually goes INTO a pipeline object. Cull mode
    // is absent on purpose: it is encoder state, so two materials that differ
    // only in culling collapse to one entry here.
    std::unordered_map<std::uint64_t, rhi::PipelineId> pipeline_cache;

    rhi::BufferId triangle_vb;
    rhi::PipelineId composite;
    rhi::PipelineId shadow;
    rhi::PipelineId ssao;
    rhi::TextureId dummy_shadow;  // bound when shadows are off
    // 1x1 opaque white. Standing in for an absent map keeps the shader
    // branch-free: every material multiplies, some just multiply by one.
    rhi::TextureId white;
    rhi::SamplerId sampler;

    // One uniform ring covering every frame slot. Writing slot N is safe only
    // because Device::BeginFrame blocks until the frame that last used slot N
    // has finished on the GPU.
    rhi::BufferId uniforms;
    std::uint8_t* uniform_map = nullptr;
    std::size_t uniform_stride = 0;  // per instance, alignment-padded
    std::size_t slot_bytes = 0;      // per frame slot

    RenderStats stats;
    int shadow_draws = 0;
    // Reused across frames so a steady-state frame allocates nothing.
    std::vector<DrawItem> visible;
    std::vector<int> draw_order;

    // Bump allocator inside the current frame's ring slot. Multiple Draw* calls
    // in one frame must each get their own slice, or the second would overwrite
    // uniforms the first had already encoded a draw against.
    //
    // Keyed on the MONOTONIC frame index, not on FrameSlot(). FrameSlot only has
    // kFramesInFlight distinct values, so an app that draws on, say, every third
    // frame would see the same slot every time, never reset, and silently
    // starve once the cursor passed kMaxInstancesPerFrame — a scene that goes
    // blank forever with draws == 0 and no error anywhere.
    std::uint64_t last_frame = ~std::uint64_t{0};
    std::size_t cursor = 0;

    // Returns the byte offset for one FrameUniforms, or kNoSpace when this
    // frame's slot is full.
    static constexpr std::size_t kNoSpace = ~std::size_t{0};
    std::size_t AllocUniform() {
        const std::uint64_t frame = dev->FrameIndex();
        if (frame != last_frame) {
            last_frame = frame;
            cursor = 0;
        }
        if (cursor >= kMaxInstancesPerFrame) return kNoSpace;
        const std::size_t off = std::size_t(dev->FrameSlot()) * slot_bytes +
                                cursor * uniform_stride;
        ++cursor;
        return off;
    }

    [[nodiscard]] rhi::PipelineId GetOrCreatePipeline(Shading shading,
                                                      bool depth_test,
                                                      bool blend,
                                                      std::string& error);
};

rhi::PipelineId Renderer::Impl::GetOrCreatePipeline(Shading shading,
                                                    bool depth_test,
                                                    bool blend,
                                                    std::string& error) {
    // Blend and depth-write ARE pipeline state, unlike cull mode, so they have
    // to be in the key. Leaving them out would hand a transparent material the
    // opaque pipeline and quietly turn glass solid.
    const std::uint64_t key = (std::uint64_t(shading) << 12) |
                              (std::uint64_t(blend) << 8) |
                              (std::uint64_t(depth_test) << 4) |
                              std::uint64_t(color);
    if (auto it = pipeline_cache.find(key); it != pipeline_cache.end())
        return it->second;

    rhi::PipelineDesc desc;
    if (shading == Shading::Lit) {
        desc.source = ShaderSource(kLitSrc);
        desc.vertex_fn = "vs_lit";
        desc.fragment_fn = "fs_lit";
    } else if (shading == Shading::Composite) {
        desc.source = ShaderSource(kCompositeSrc);
        desc.vertex_fn = "vs_composite";
        desc.fragment_fn = "fs_composite";
    } else if (shading == Shading::Ssao) {
        desc.source = ShaderSource(kSsaoSrc);
        desc.vertex_fn = "vs_ssao";
        desc.fragment_fn = "fs_ssao";
    } else if (shading == Shading::ShadowDepth) {
        desc.source = ShaderSource(kShadowSrc);
        desc.vertex_fn = "vs_shadow";
        desc.fragment_fn = "fs_shadow";  // void, but it discards for the cut
        desc.depth_only = true;
    } else {
        desc.source = ShaderSource(kTriangleSrc);
        desc.vertex_fn = "vs_main";
        desc.fragment_fn = "fs_main";
    }
    desc.color = color;
    desc.depth = depth_test;
    desc.blend = blend;
    // A blended surface must not write depth, or the transparent objects behind
    // it get rejected before they are ever shaded.
    desc.depth_write = !blend;
    // Reversed-Z: near maps to 1 and far to 0, so "closer" means GREATER. Pair
    // with clear_depth = 0 or nothing draws at all.
    desc.depth_compare = rhi::Compare::Greater;

    const rhi::PipelineId id = dev->CreatePipeline(desc, error);
    if (Valid(id)) pipeline_cache.emplace(key, id);
    return id;
}

Renderer::Renderer() : impl_(std::make_unique<Impl>()) {}
Renderer::~Renderer() = default;

const RenderStats& Renderer::LastStats() const { return impl_->stats; }
const std::vector<int>& Renderer::LastDrawOrder() const { return impl_->draw_order; }
int Renderer::PipelineCount() const { return int(impl_->pipeline_cache.size()); }
int Renderer::ShadowDrawCount() const { return impl_->shadow_draws; }

std::uint32_t Renderer::PipelineOf(MaterialHandle m) const {
    if (!Valid(m) || m.v >= impl_->materials.size()) return 0;
    return impl_->materials[m.v].pipeline.v;
}

MeshHandle Renderer::UploadMesh(const Mesh& mesh) {
    if (mesh.vertices.empty() || mesh.indices.empty()) return {};
    GpuMesh gm;
    gm.vb = impl_->dev->CreateBuffer(mesh.vertices.data(),
                                     mesh.vertices.size() * sizeof(VertexIn));
    gm.ib = impl_->dev->CreateBuffer(mesh.indices.data(),
                                     mesh.indices.size() * sizeof(std::uint16_t));
    if (!Valid(gm.vb) || !Valid(gm.ib)) return {};
    gm.index_count = mesh.indices.size();
    gm.bounds = mesh.bounds;
    impl_->meshes.push_back(gm);
    return MeshHandle{std::uint32_t(impl_->meshes.size() - 1)};
}

MaterialHandle Renderer::CreateMaterial(const MaterialDesc& desc,
                                        std::string& error) {
    GpuMaterial m;
    m.pipeline = impl_->GetOrCreatePipeline(desc.shading, desc.depth_test,
                                            desc.transparent, error);
    if (!Valid(m.pipeline)) return {};
    m.cull = desc.cull;
    m.depth_test = desc.depth_test;
    m.transparent = desc.transparent;
    m.base_color = desc.base_color;
    m.roughness = desc.roughness;
    m.metallic = desc.metallic;
    m.albedo = Valid(desc.albedo) ? desc.albedo : impl_->white;
    m.roughness_map =
        Valid(desc.roughness_map) ? desc.roughness_map : impl_->white;
    impl_->materials.push_back(m);
    return MaterialHandle{std::uint32_t(impl_->materials.size() - 1)};
}

std::unique_ptr<Renderer> Renderer::Create(rhi::Device& dev, rhi::Format color,
                                           std::string& error) {
    auto r = std::unique_ptr<Renderer>(new Renderer());
    r->impl_->dev = &dev;
    r->impl_->color = color;

    // Defaults first: CreateMaterial substitutes these for absent maps, so they
    // have to exist before any material does.
    const std::uint8_t kWhitePixel[4] = {255, 255, 255, 255};
    r->impl_->white = dev.CreateTexture2D(1, 1, kWhitePixel);
    r->impl_->sampler = dev.CreateSampler(rhi::Filter::Linear, rhi::Wrap::Repeat);
    if (!Valid(r->impl_->white) || !Valid(r->impl_->sampler)) {
        error = "failed to create the default texture or sampler";
        return nullptr;
    }

    // Built-ins go in FIRST, in this exact order, so their handles match the
    // kMesh* / kMaterial* constants that engine/scene uses.
    //
    // NEUTRAL checker on purpose: the mesh supplies a luminance pattern and the
    // per-instance tint supplies the hue. Baking a colour into the mesh would
    // make two differently-tinted instances share a hue wherever the checker is
    // strongly coloured.
    const Vec4 kLightSq{1.0f, 1.0f, 1.0f, 1.0f};
    const Vec4 kDarkSq{0.42f, 0.42f, 0.42f, 1.0f};
    const MeshHandle sphere =
        r->UploadMesh(MakeUVSphere(1.0f, 32, 64, kLightSq, kDarkSq));
    const MeshHandle cube = r->UploadMesh(MakeCube(1.4f, kLightSq, kDarkSq));
    if (!(sphere == kMeshSphere) || !(cube == kMeshCube)) {
        error = "built-in mesh handles did not come out as expected";
        return nullptr;
    }

    const MaterialHandle lit =
        r->CreateMaterial({Shading::Lit, true, rhi::Cull::Back}, error);
    const MaterialHandle flat =
        r->CreateMaterial({Shading::Flat, false, rhi::Cull::None}, error);
    // Same shading and depth state as `lit`, so the cache must hand it the very
    // same pipeline object — only the encoder-side cull mode differs.
    const MaterialHandle two_sided =
        r->CreateMaterial({Shading::Lit, true, rhi::Cull::None}, error);
    if (!(lit == kMaterialLit) || !(flat == kMaterialFlat) ||
        !(two_sided == kMaterialLitTwoSided)) {
        if (error.empty()) error = "built-in material handles came out wrong";
        return nullptr;
    }

    // Flat-shaded, so the normals are unused filler — but the vertex format is
    // shared with the meshes and must still be filled in.
    const VertexIn tri[3] = {
        {{0.0f, 0.6f, -2.0f, 0.0f}, {0, 0, 1, 0}, {1, 0, 0, 1}},
        {{-0.6f, -0.4f, -2.0f, 0.0f}, {0, 0, 1, 0}, {0, 1, 0, 1}},
        {{0.6f, -0.4f, -2.0f, 0.0f}, {0, 0, 1, 0}, {0, 0, 1, 1}},
    };
    r->impl_->triangle_vb = dev.CreateBuffer(tri, sizeof(tri));
    if (!Valid(r->impl_->triangle_vb)) {
        error = "failed to upload the triangle vertex buffer";
        return nullptr;
    }

    r->impl_->composite =
        r->impl_->GetOrCreatePipeline(Shading::Composite, false, false, error);
    if (!Valid(r->impl_->composite)) return nullptr;

    r->impl_->shadow =
        r->impl_->GetOrCreatePipeline(Shading::ShadowDepth, true, false, error);
    if (!Valid(r->impl_->shadow)) return nullptr;

    r->impl_->ssao = r->impl_->GetOrCreatePipeline(Shading::Ssao, false, false, error);
    if (!Valid(r->impl_->ssao)) return nullptr;

    // Something has to be bound at the shadow slot even when shadows are off;
    // the shader guards on a flag rather than reading it, so 1x1 is plenty.
    r->impl_->dummy_shadow = dev.CreateShadowMap(1);
    if (!Valid(r->impl_->dummy_shadow)) {
        error = "failed to create the placeholder shadow map";
        return nullptr;
    }

    // Uniform ring: kFramesInFlight slots, each holding kMaxInstancesPerFrame
    // alignment-padded FrameUniforms.
    r->impl_->uniform_stride =
        AlignUp(sizeof(FrameUniforms), dev.UniformAlignment());
    r->impl_->slot_bytes = r->impl_->uniform_stride * kMaxInstancesPerFrame;
    r->impl_->uniforms =
        dev.CreateDynamicBuffer(r->impl_->slot_bytes * rhi::kFramesInFlight);
    r->impl_->uniform_map =
        static_cast<std::uint8_t*>(dev.MapBuffer(r->impl_->uniforms));
    if (!Valid(r->impl_->uniforms) || !r->impl_->uniform_map) {
        error = "failed to allocate the uniform ring buffer";
        return nullptr;
    }
    return r;
}

void Renderer::DrawShadow(rhi::Encoder& enc, const Scene& scene) {
    impl_->shadow_draws = 0;
    if (scene.shadowExtent <= 0.0f) return;
    const Mat4 lightViewProj = scene.LightViewProj();

    enc.SetPipeline(impl_->shadow);
    // Front faces culled instead of back: shifting the recorded depth to the
    // BACK of each caster is the cheapest peter-panning-free way to keep a
    // surface from shadowing itself, and it costs nothing here because every
    // mesh in this engine is closed.
    enc.SetCull(rhi::Cull::Front, rhi::Winding::CounterClockwise);

    for (const Instance& inst : scene.instances) {
        if (!Valid(inst.mesh) || inst.mesh.v >= impl_->meshes.size()) continue;
        // TRANSPARENT SURFACES DO NOT CAST. A window pane written into the
        // shadow map is opaque there, so it blocks the very sunlight it is
        // supposed to let through and the room stays dark — a lighting symptom
        // with a shadow-pass cause.
        //
        // Skipping outright is the crude answer; the correct one is a coloured
        // or attenuating shadow, which needs more than a depth buffer.
        if (Valid(inst.material) && inst.material.v < impl_->materials.size() &&
            impl_->materials[inst.material.v].transparent)
            continue;

        const GpuMesh& gm = impl_->meshes[inst.mesh.v];
        const std::size_t offset = impl_->AllocUniform();
        if (offset == Impl::kNoSpace) break;

        FrameUniforms u{};
        u.lightViewProj = lightViewProj;
        u.model = inst.model;
        // The cut has to reach the shadow map too, or a cutaway renders its
        // interior in the shadow of the roof it just removed.
        u.surface = Vec4{0.0f, 0.0f, 0.0f, scene.clipY};
        std::memcpy(impl_->uniform_map + offset, &u, sizeof(u));

        enc.SetVertexBuffer(gm.vb, 0, kVertexSlot);
        enc.SetVertexBuffer(impl_->uniforms, offset, kUniformSlot);
        // FRAGMENT too. fs_shadow reads clipY out of the same block, and an
        // unbound fragment buffer reads as zero rather than failing — so the
        // cut plane silently becomes y = 0 and every caster above the origin
        // is discarded from the shadow map. The symptom is that objects high
        // off the ground stop casting, which looks like a lighting bug and is
        // a missing bind.
        enc.SetFragmentBuffer(impl_->uniforms, offset, kUniformSlot);
        enc.DrawIndexedU16(gm.ib, gm.index_count);
        ++impl_->shadow_draws;
    }
}

void Renderer::DrawScene(rhi::Encoder& enc, const Scene& scene, int width,
                         int height, rhi::TextureId shadow_map) {
    impl_->stats = RenderStats{};
    impl_->stats.submitted = int(scene.instances.size());
    impl_->draw_order.clear();
    if (width <= 0 || height <= 0 || scene.instances.empty()) return;

    // A pipeline built without depth cannot run in a pass that has one, and
    // vice versa — Metal rejects the draw outright. Rather than let the caller
    // find out as a validation abort, skip and count.
    const bool pass_depth = impl_->dev->CurrentPassHasDepth();

    const Mat4 viewProj = scene.camera.ViewProj(float(width) / float(height));
    const bool shadows = scene.shadowExtent > 0.0f && Valid(shadow_map);
    const Mat4 lightViewProj = shadows ? scene.LightViewProj() : Mat4::Identity();
    const rhi::TextureId shadow_tex = shadows ? shadow_map : impl_->dummy_shadow;

    // --- cull ----------------------------------------------------------------
    // Reject before touching the GPU at all. A culled object costs one plane
    // test; a submitted one costs a draw call plus whatever it rasterises.
    const Frustum frustum = Frustum::FromViewProj(viewProj);
    const Vec3 eye = scene.camera.eye;

    impl_->visible.clear();
    for (int idx = 0; idx < int(scene.instances.size()); ++idx) {
        const Instance& inst = scene.instances[idx];
        if (!Valid(inst.mesh) || inst.mesh.v >= impl_->meshes.size() ||
            !Valid(inst.material) || inst.material.v >= impl_->materials.size()) {
            ++impl_->stats.invalid;
            continue;
        }
        if (impl_->materials[inst.material.v].depth_test != pass_depth) {
            ++impl_->stats.incompatible;
            continue;
        }

        const GpuMesh& gm = impl_->meshes[inst.mesh.v];
        // Object-space bounds pushed out to world space. A sphere only needs
        // its centre moved and its radius scaled — no re-fitting, which is
        // exactly why the bounds are a sphere and not a box.
        const Vec4 c = inst.model * Vec4{gm.bounds.center.x, gm.bounds.center.y,
                                         gm.bounds.center.z, 1.0f};
        const Vec3 world_center{c.x, c.y, c.z};
        const float world_radius = gm.bounds.radius * MaxScale(inst.model);

        if (!frustum.IntersectsSphere(world_center, world_radius)) {
            ++impl_->stats.culled;
            continue;
        }

        DrawItem item;
        item.inst = &inst;
        item.index = idx;
        item.pipeline = impl_->materials[inst.material.v].pipeline.v;
        item.depth = Length(world_center - eye);
        item.transparent = impl_->materials[inst.material.v].transparent;
        impl_->visible.push_back(item);
    }

    // --- sort ----------------------------------------------------------------
    // TWO batches with opposite rules, which is not a style choice:
    //
    //  * Opaque: pipeline first, then front-to-back. The depth buffer resolves
    //    visibility whatever the order, so grouping by pipeline to cut state
    //    changes is free, and nearest-first inside a group lets early-Z reject
    //    the overdraw behind it.
    //  * Transparent: strictly BACK-TO-FRONT, and only after every opaque
    //    object. Blending is not commutative, so here the order IS the result —
    //    and with depth writes off there is nothing else to sort it out.
    std::sort(impl_->visible.begin(), impl_->visible.end(),
              [](const DrawItem& a, const DrawItem& b) {
                  if (a.transparent != b.transparent) return b.transparent;
                  if (a.transparent) return a.depth > b.depth;  // far first
                  if (a.pipeline != b.pipeline) return a.pipeline < b.pipeline;
                  return a.depth < b.depth;
              });

    // --- submit --------------------------------------------------------------
    // Redundant state is the cheapest thing to eliminate, so track what is
    // actually bound rather than setting it on every draw.
    rhi::PipelineId bound_pipeline;
    rhi::Cull bound_cull = rhi::Cull::None;
    bool cull_set = false;

    for (const DrawItem& item : impl_->visible) {
        const Instance& inst = *item.inst;
        const std::size_t offset = impl_->AllocUniform();
        if (offset == Impl::kNoSpace) {
            // Ring slot full. Drop the rest rather than overrun, and SAY so —
            // a silently short frame is the worst possible failure here.
            impl_->stats.overflowed =
                int(impl_->visible.size()) - impl_->stats.draws;
            break;
        }

        const GpuMesh& gm = impl_->meshes[inst.mesh.v];
        const GpuMaterial& mat = impl_->materials[inst.material.v];

        if (mat.pipeline.v != bound_pipeline.v) {
            enc.SetPipeline(mat.pipeline);
            bound_pipeline = mat.pipeline;
            ++impl_->stats.pipeline_switches;
        }
        if (!cull_set || mat.cull != bound_cull) {
            // The mesh generators guarantee CCW-from-outside.
            enc.SetCull(mat.cull, rhi::Winding::CounterClockwise);
            bound_cull = mat.cull;
            cull_set = true;
        }

        FrameUniforms u{};
        u.viewProj = viewProj;
        u.model = inst.model;
        u.tint = inst.tint;
        u.lightDir = scene.lightDir;
        u.lightColor = scene.lightColor;
        u.baseColor = mat.base_color;
        u.lightViewProj = lightViewProj;
        u.surface = Vec4{mat.roughness, mat.metallic, shadows ? 1.0f : 0.0f,
                         scene.clipY};
        u.eyePos = Vec4{scene.camera.eye.x, scene.camera.eye.y,
                        scene.camera.eye.z, 1.0f};

        // Sub-allocate out of this frame's slot instead of a per-draw setBytes.
        std::memcpy(impl_->uniform_map + offset, &u, sizeof(u));

        enc.SetVertexBuffer(gm.vb, 0, kVertexSlot);
        enc.SetVertexBuffer(impl_->uniforms, offset, kUniformSlot);
        enc.SetFragmentBuffer(impl_->uniforms, offset, kUniformSlot);
        enc.SetFragmentTexture(mat.albedo, 0);
        enc.SetFragmentTexture(mat.roughness_map, 1);
        enc.SetFragmentTexture(shadow_tex, 2);
        enc.SetFragmentSampler(impl_->sampler, 0);
        enc.DrawIndexedU16(gm.ib, gm.index_count);
        impl_->draw_order.push_back(item.index);
        ++impl_->stats.draws;
        if (item.transparent) ++impl_->stats.transparent_draws;
    }
}

void Renderer::DrawTriangle(rhi::Encoder& enc, int width, int height) {
    if (width <= 0 || height <= 0) return;

    FrameUniforms u{};
    u.viewProj =
        Mat4::PerspectiveReverseZ(kFovY, float(width) / float(height), kNearZ);
    u.model = Mat4::Identity();  // vertices are already in world space
    u.tint = Vec4{1, 1, 1, 1};
    u.lightDir = kLightDir;
    u.lightColor = Vec4{1, 1, 1, 1};

    // Its own slice of the ring, NOT a hardcoded offset 0 — otherwise calling
    // DrawScene and DrawTriangle in the same frame would make the triangle
    // stomp the uniforms the first instance had already encoded a draw against.
    const std::size_t offset = impl_->AllocUniform();
    if (offset == Impl::kNoSpace) return;
    std::memcpy(impl_->uniform_map + offset, &u, sizeof(u));

    const GpuMaterial& mat = impl_->materials[kMaterialFlat.v];
    enc.SetPipeline(mat.pipeline);
    enc.SetCull(mat.cull, rhi::Winding::CounterClockwise);
    enc.SetVertexBuffer(impl_->triangle_vb, 0, kVertexSlot);
    enc.SetVertexBuffer(impl_->uniforms, offset, kUniformSlot);
    enc.Draw(3);
}

void Renderer::DrawComposite(rhi::Encoder& enc, rhi::TextureId src,
                             rhi::TextureId ao) {
    enc.SetPipeline(impl_->composite);
    enc.SetCull(rhi::Cull::None, rhi::Winding::CounterClockwise);
    enc.SetFragmentTexture(src, 0);
    enc.SetFragmentTexture(Valid(ao) ? ao : impl_->white, 1);
    enc.SetFragmentSampler(impl_->sampler, 0);
    enc.Draw(3);  // one oversized triangle, generated from the vertex id
}

void Renderer::DrawSsao(rhi::Encoder& enc, const Camera& cam, int width,
                        int height, rhi::TextureId depth, float radius) {
    if (width <= 0 || height <= 0 || !Valid(depth)) return;

    FrameUniforms u{};
    // Only the reconstruction parameters matter here; the pass shades nothing.
    u.ssao = Vec4{cam.nearZ, 1.0f / std::tan(cam.fovY * 0.5f),
                  float(width) / float(height), radius};

    const std::size_t offset = impl_->AllocUniform();
    if (offset == Impl::kNoSpace) return;
    std::memcpy(impl_->uniform_map + offset, &u, sizeof(u));

    enc.SetPipeline(impl_->ssao);
    enc.SetCull(rhi::Cull::None, rhi::Winding::CounterClockwise);
    enc.SetFragmentBuffer(impl_->uniforms, offset, kUniformSlot);
    enc.SetFragmentTexture(depth, 0);
    enc.SetFragmentSampler(impl_->sampler, 0);
    enc.Draw(3);
}

// --- one-shot offscreen helpers ---------------------------------------------

namespace {

// Shared body of every offscreen path: device, renderer, targets, one pass,
// readback. `draw` records the actual geometry.
template <class DrawFn>
Image RenderOffscreen(int width, int height, bool want_depth, std::string& error,
                      DrawFn&& draw) {
    if (width <= 0 || height <= 0) {
        error = "width and height must both be positive";
        return {};
    }

    auto dev = rhi::Device::Create(error);
    if (!dev) return {};

    auto renderer = Renderer::Create(*dev, rhi::Format::RGBA8Unorm, error);
    if (!renderer) return {};

    rhi::PassDesc pass;
    pass.color = dev->CreateRenderTarget(width, height, rhi::Format::RGBA8Unorm, /*cpu_readable=*/true);
    if (want_depth) pass.depth = dev->CreateDepthTarget(width, height);
    if (!Valid(pass.color) || (want_depth && !Valid(pass.depth))) {
        error = "failed to create offscreen render targets";
        return {};
    }
    for (int i = 0; i < 4; ++i) pass.clear_color[i] = kClearColor[i];
    pass.clear_depth = 0.0f;  // reversed-Z: 0 is the far plane

    dev->BeginFrame();
    rhi::Encoder enc = dev->BeginPass(pass);
    draw(*renderer, enc);
    dev->EndPass();
    if (!dev->CommitAndWait(error)) return {};

    Image img;
    img.width = width;
    img.height = height;
    img.rgba.resize(std::size_t(width) * std::size_t(height) * 4);
    if (!dev->ReadPixels(pass.color, width, height, img.rgba)) {
        error = "pixel readback failed";
        return {};
    }
    return img;
}

}  // namespace

Image RenderSceneOffscreen(const Scene& scene, int width, int height,
                           std::string& error) {
    return RenderOffscreen(width, height, /*want_depth=*/true, error,
                           [&](Renderer& r, rhi::Encoder& e) {
                               r.DrawScene(e, scene, width, height);
                           });
}

Image RenderSphereOffscreen(int width, int height, float angle_radians,
                            std::string& error) {
    // A single sphere is just a one-instance scene; no reason for two paths.
    Scene s;
    s.camera.eye = kSphereEye;
    s.instances.push_back({kMeshSphere, kMaterialLit,
                           Mat4::RotationY(angle_radians),
                           Vec4{0.35f, 0.62f, 1.00f, 1.0f}});
    return RenderSceneOffscreen(s, width, height, error);
}

Image RenderTriangleOffscreen(int width, int height, std::string& error) {
    return RenderOffscreen(width, height, /*want_depth=*/false, error,
                           [&](Renderer& r, rhi::Encoder& e) {
                               r.DrawTriangle(e, width, height);
                           });
}

}  // namespace eng
