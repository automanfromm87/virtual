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

// Mirrors composite.metal's GradeParams. Checked rather than trusted: a drift
// here silently shifts every colour in the frame, which reads as an art
// decision rather than as a layout bug.
struct GradeParams {
    Vec4 tone;
    Vec4 lift;
    Vec4 gamma;
    Vec4 gain;
    Vec4 look;
};
static_assert(sizeof(GradeParams) == 80, "GradeParams layout drifted");


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
constexpr char kBloomSrc[] = {
#embed "engine/shaders/bloom.metal"
    , 0};
// Prepended to every shader that lights a surface, so the forward and deferred
// paths run the same code rather than two copies of it.
constexpr char kShadingSrc[] = {
#embed "engine/shaders/shading.metal"
    , 0};
constexpr char kDeferredSrc[] = {
#embed "engine/shaders/deferred.metal"
    , 0};
constexpr char kRaytraceSrc[] = {
#embed "engine/shaders/raytrace.metal"
    , 0};
constexpr char kSkinningSrc[] = {
#embed "engine/shaders/skinning.metal"
    , 0};
constexpr char kCullSrc[] = {
#embed "engine/shaders/cull.metal"
    , 0};
constexpr char kClusterSrc[] = {
#embed "engine/shaders/cluster.metal"
    , 0};

// CPU and GPU must agree byte-for-byte. Assert it — do not hope for it.
static_assert(sizeof(FrameUniforms) == 448, "FrameUniforms layout drifted");
static_assert(sizeof(GpuClusters) == 64, "GpuClusters layout drifted");
static_assert(sizeof(GpuLight) == 144, "GpuLight layout drifted");
static_assert(sizeof(GpuCascades) == 288, "GpuCascades layout drifted");
static_assert(sizeof(GpuInstance) == 96, "GpuInstance layout drifted");
// Metal reads this as MTLDrawIndexedPrimitivesIndirectArguments. A mismatch is
// not a compile error anywhere -- it is a draw of the wrong thing.
static_assert(sizeof(GpuDrawArgs) == 20, "GpuDrawArgs must match Metal's layout");
static_assert(sizeof(VertexIn) == 80, "VertexIn layout drifted");
static_assert(offsetof(VertexIn, normal) == 16, "GPU float3 occupies 16 bytes");
static_assert(offsetof(VertexIn, color) == 32, "GPU float3 occupies 16 bytes");
static_assert(offsetof(VertexIn, uv) == 48, "uv must follow color");
static_assert(offsetof(VertexIn, tangent) == 64, "tangent must follow uv");

// The triangle predates the Camera type and bakes its vertices straight into
// world space, so it carries its own projection constants.
constexpr float kFovY = 1.0472f;  // 60 degrees
constexpr float kNearZ = 0.1f;
constexpr Vec4 kLightDir{0.4082f, 0.8165f, 0.4082f, 0.0f};
constexpr Vec3 kSphereEye{0.0f, 0.0f, 3.0f};

// Uniform slot 1 in both stages; vertex buffer in slot 0.
constexpr int kVertexSlot = 0;
constexpr int kUniformSlot = 1;
constexpr int kSkinSlot = 2;     // per-vertex joints and weights
constexpr int kPaletteSlot = 3;  // this instance's joint matrices
constexpr int kLightSlot = 2;    // FRAGMENT stage: the scene's local lights
constexpr int kCascadeSlot = 3;  // FRAGMENT stage: the sun's cascades
constexpr int kClusterSlot = 5;         // FRAGMENT: the cluster grid's placement
constexpr int kClusterCountSlot = 6;    // FRAGMENT: lights per cell
constexpr int kClusterIndexSlot = 7;    // FRAGMENT: the per-cell light indices
// VERTEX stage: the per-instance model and tint for an instanced draw. 4 and
// not 2, even though the skin slot is free on this path -- reusing a slot whose
// name says "skin" is how a future skinned-and-instanced draw gets one of them
// silently overwritten.
constexpr int kInstanceSlot = 4;

// Ring capacity per frame slot. Exceeding it drops draws rather than corrupting
// the frame -- see the clamp in DrawScene.
//
// It was 1024, and a scene of 6000 objects showed why that is not merely
// "some objects go missing". The ring is shared by EVERY pass: when the scene
// pass exhausts it, the composite that follows asks for a slice, gets none, and
// silently draws nothing. The frame comes out BLACK -- not partially drawn --
// and the only clue is a stats field nobody reads on a frame that rendered
// nothing at all.
//
// 8192 at 512 bytes a slice is 4 MB per frame slot, 12 MB in flight. That is a
// real cost and it is the right trade: past 8192 draws in one frame the answer
// is CullScene and DrawSceneIndirect, which need three slices for six thousand
// objects instead of six thousand.
constexpr std::size_t kMaxInstancesPerFrame = 8192;

std::size_t AlignUp(std::size_t v, std::size_t a) { return ((v + a - 1) / a) * a; }

std::string ShaderSource(const char* body) {
    return std::string(kShaderTypesSrc) + "\n" + body;
}

// For anything that lights a surface. The shading library goes in between, so
// the forward and deferred fragment shaders call the SAME ShadeSurface rather
// than each carrying a copy that drifts.
std::string ShadingSource(const char* body) {
    return std::string(kShaderTypesSrc) + "\n" + kShadingSrc + "\n" + body;
}

}  // namespace

// One entry of the mesh registry: the GPU buffers plus the CPU-side bounds
// that frustum culling needs.
// The most levels of detail one mesh may carry. Four is not a compromise: each
// level covers roughly a quarter of the screen area of the one before, so four
// spans a factor of sixteen in distance, and past that the object is a handful
// of pixels and the next thing to do is not draw it.
constexpr int kMaxLods = 4;

struct GpuMesh {
    rhi::BufferId vb;
    rhi::BufferId ib;
    std::size_t index_count = 0;
    // ADDITIONAL levels, coarsest last. Level 0 is `vb`/`ib` above rather than
    // lods[0], so every existing path -- the forward draw, the shadow pass,
    // skinning, the ray tracing build -- keeps working untouched and only the
    // GPU-driven path knows levels exist.
    struct Lod {
        rhi::BufferId vb;
        rhi::BufferId ib;
        std::size_t index_count = 0;
        std::size_t vertex_count = 0;
    };
    Lod lods[kMaxLods];
    int lod_count = 1;
    // Needed by compute skinning, which dispatches one thread per VERTEX while
    // every draw path counts indices.
    std::size_t vertex_count = 0;
    Bounds bounds;
    // Null for a static mesh. Skinning is a property of the GEOMETRY, not of
    // the material: the same material may be worn by a prop and a character.
    rhi::BufferId skin_vb;
    int joint_count = 0;
};

struct GpuMaterial {
    rhi::PipelineId pipeline;
    // The same state compiled against the skinned vertex stage. Built up front
    // rather than on first use, so a skinned draw never fails at frame time for
    // a reason a material could have reported at creation.
    rhi::PipelineId skinned_pipeline;
    // The same material compiled for the DEFERRED geometry pass. Built for
    // every opaque lit material whether or not the deferred path is used --
    // the alternative is a pipeline compile in the middle of the first frame
    // that switches to it, which is a visible hitch at exactly the moment
    // someone is looking for a difference between the two paths.
    rhi::PipelineId gbuffer_pipeline;
    rhi::PipelineId gbuffer_skinned_pipeline;
    // Compiled against vs_lit_instanced, which reads model and tint from a
    // buffer instead of the per-draw uniform block.
    rhi::PipelineId instanced_pipeline;
    rhi::Cull cull = rhi::Cull::Back;
    bool depth_test = true;
    bool transparent = false;
    Vec4 base_color{1.0f, 1.0f, 1.0f, 1.0f};
    float roughness = 0.5f;
    float metallic = 0.0f;
    rhi::TextureId albedo;
    rhi::TextureId roughness_map;
    rhi::TextureId metallic_map;
    rhi::TextureId emissive_map;
    rhi::TextureId occlusion_map;
    rhi::TextureId normal_map;
    Vec3 emissive{0.0f, 0.0f, 0.0f};
    float normal_strength = 1.0f;
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
    // --- occlusion and levels of detail ---------------------------------------
    rhi::TextureId hiz;
    int hiz_width = 0, hiz_height = 0, hiz_mips = 1;
    std::vector<rhi::TextureId> hiz_views;
    rhi::ComputePipelineId hiz_copy, hiz_reduce;
    bool hiz_tried = false;
    // Which frame the pyramid was last filled on. The cull refuses to use one
    // from an earlier frame: it was built for a different camera.
    std::uint64_t hiz_built_frame = ~0ull;
    bool occlusion_enabled = true;
    // Screen radius in pixels at which each level takes over. Level 0 above
    // 60 px, then 24, then 9 -- roughly halving the linear size each step,
    // which is quartering the area.
    Vec3 lod_thresholds{60.0f, 24.0f, 9.0f};
    ColorGrade grade;
    rhi::BufferId exposure;
    // A one-float buffer holding 1.0, bound when the caller supplied no
    // exposure. Cheaper than a shader branch and impossible to get out of step
    // with what was actually bound.
    rhi::BufferId unit_exposure;
    // The environment probe, or an all-null set when there is none. Binding a
    // null texture is legal and is what the shader's is_null_texture check
    // reads, so there is no separate "enabled" flag to fall out of step.
    EnvironmentBindings env;
    rhi::Device* dev = nullptr;
    // The format of whatever the LAST pass writes into: a drawable, or an
    // offscreen target a test reads back.
    rhi::Format color = rhi::Format::BGRA8Unorm;
    // Multisample count for the scene passes. The fullscreen ones stay at 1:
    // they run after the resolve.
    int samples = 1;

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
    rhi::PipelineId deferred_light;
    // Built lazily: compiling a ray tracing shader on a device that has none
    // fails, and most scenes never ask for one.
    rhi::PipelineId ray_shadow;
    bool ray_shadow_tried = false;
    rhi::AccelId scene_tlas;
    // Compute skinning. Built on first use, like the ray tracing pipeline:
    // most scenes have no skinned geometry and would pay a shader compile for
    // nothing.
    rhi::ComputePipelineId skin_pipeline;
    bool skin_pipeline_tried = false;
    std::string skin_error;
    // Posed output per scene instance index, and the buffer pool behind it.
    // Keyed by instance rather than by mesh: two instances of one mesh at
    // different poses are different geometry and cannot share a buffer.
    std::vector<rhi::BufferId> posed_of_instance;
    std::vector<rhi::BufferId> posed_pool;
    std::vector<std::size_t> posed_pool_bytes;
    std::size_t posed_used = 0;

    // --- GPU-driven drawing ---
    rhi::ComputePipelineId cull_pipeline, cull_finish_pipeline;
    bool cull_tried = false;
    std::string cull_error;
    // One batch per (mesh, material). Each owns four buffers: the instances the
    // CPU uploaded, the compacted survivors, an atomic counter, and the draw
    // arguments. Pooled, because allocating them per frame would leak.
    struct Batch {
        MeshHandle mesh;
        MaterialHandle material;
        int count = 0;
        rhi::BufferId instances;   // dynamic, CPU-written
        rhi::BufferId visible;     // storage, GPU-written
        rhi::BufferId counter;     // storage, one uint
        rhi::BufferId args;        // storage, one GpuDrawArgs
        std::size_t capacity = 0;  // instances the buffers can hold
    };
    std::vector<Batch> batches;
    int live_batches = 0;
    int batched_instances = 0;
    // Bottom-level structure per registered mesh, or a null handle for one
    // that has not needed one yet. Indexed by MeshHandle.
    std::vector<rhi::AccelId> mesh_blas;
    int blas_builds = 0;
    rhi::PipelineId shadow;
    rhi::PipelineId shadow_skinned;
    rhi::TextureId shadow_atlas;
    // First atlas tile each scene light owns, or -1. A spot owns one, a point
    // light six consecutive ones. Kept so DrawScene can tell the shader where
    // to look.
    std::vector<int> shadow_tile_of_light;
    int shadow_tiles_used = 0;
    // Aspect of the last DrawScene, so DrawShadow can fit its cascades to the
    // frustum the camera will actually use. The shadow pass runs first and has
    // no width or height of its own.
    float last_aspect = 0.0f;
    Mat4 inv_view_proj = Mat4::Identity();
    rhi::PipelineId ssao;
    rhi::TextureId dummy_shadow;  // bound when shadows are off
    // 1x1 opaque white. Standing in for an absent map keeps the shader
    // branch-free: every material multiplies, some just multiply by one.
    rhi::TextureId white;
    // 1x1 opaque BLACK, for slots that are ADDED rather than multiplied. White
    // would make an absent bloom brighten the whole frame.
    rhi::TextureId black;
    rhi::PipelineId bloom_bright;
    rhi::PipelineId bloom_blur;
    // TWO samplers, and the split is not cosmetic. `sampler` wraps, which is
    // what a tiling albedo map needs. Every FULLSCREEN pass wants the opposite:
    // a separable blur, an SSAO kernel and a depth lookup all tap at an offset,
    // and near the border that offset leaves the image -- with Repeat it comes
    // back on the far side. A bright object at the top of the frame then bleeds
    // a blurred copy of itself along the bottom, which is what this cost to
    // find: it is invisible in a dark scene and unmistakable in a bright one.
    rhi::SamplerId sampler;
    rhi::SamplerId clamp_sampler;
    // The SHADOW sampler: linear, clamped, isotropic. Its own object because
    // the other two are wrong for a depth comparison in different ways -- the
    // material sampler wraps and is 16x anisotropic, the fullscreen one is
    // shared with passes that may want something else later.
    rhi::SamplerId shadow_sampler;
    int anisotropy = 16;

    // --- clustered lighting ---------------------------------------------------
    bool clustered = false;
    // Set by BinLights and cleared by nothing: a frame that shades without
    // having binned falls back to the whole light buffer, which is correct and
    // merely slow. Tracked so the bindings are not made from stale bins on the
    // very first frame, when the buffers hold uninitialised memory -- THAT is
    // not merely slow, it reads arbitrary light indices.
    bool bins_valid = false;
    rhi::ComputePipelineId cluster_bin;
    rhi::BufferId cluster_counts;
    rhi::BufferId cluster_indices;
    rhi::BufferId cluster_lights;
    rhi::BufferId cluster_candidates;
    GpuClusters cluster_params{};
    // Shared by UploadLights (per-draw ring) and BinLights (its own buffer).
    // One function, because two would drift and the symptom would be the
    // binning pass culling against ranges the shading pass does not use.
    void FillLights(const Scene&, GpuLight* dst, int count) const;

    // One uniform ring covering every frame slot. Writing slot N is safe only
    // because Device::BeginFrame blocks until the frame that last used slot N
    // has finished on the GPU.
    rhi::BufferId uniforms;
    std::uint8_t* uniform_map = nullptr;
    std::size_t uniform_stride = 0;  // per instance, alignment-padded
    std::size_t slot_bytes = 0;      // per frame slot

    // A SECOND ring, for joint matrices. Not folded into FrameUniforms: sixty-
    // four matrices is four kilobytes, and every unskinned draw in the scene
    // would pay for it.
    // The scene's local lights. One buffer per frame slot, not per draw: the
    // list is a property of the scene and every fragment of every object reads
    // the same one.
    rhi::BufferId lights;
    std::uint8_t* light_map = nullptr;
    std::size_t light_slot_bytes = 0;

    // The sun's cascades, one copy per frame slot for the same reason.
    rhi::BufferId cascades;
    std::uint8_t* cascade_map = nullptr;
    std::size_t cascade_slot_bytes = 0;

    rhi::BufferId palettes;
    std::uint8_t* palette_map = nullptr;
    std::size_t palette_stride = 0;
    std::size_t palette_slot_bytes = 0;
    std::uint64_t palette_frame = ~std::uint64_t{0};
    std::size_t palette_cursor = 0;

    // Byte offset for one palette, or kNoSpace when this frame's ring is full.
    std::size_t AllocPalette() {
        const std::uint64_t frame = dev->FrameIndex();
        if (frame != palette_frame) {
            palette_frame = frame;
            palette_cursor = 0;
        }
        if (palette_cursor >= std::size_t(Renderer::kMaxSkinnedPerFrame))
            return kNoSpace;
        const std::size_t off = std::size_t(dev->FrameSlot()) * palette_slot_bytes +
                                palette_cursor * palette_stride;
        ++palette_cursor;
        return off;
    }

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

    void DrawGeometry(rhi::Encoder&, const Scene&, int width, int height,
                      rhi::TextureId shadow_map, bool gbuffer);
    void UploadLights(const Scene&, std::size_t light_offset, int light_count);
    void UploadCascades(const Scene&, std::size_t cascade_offset, float aspect);
    [[nodiscard]] rhi::Format FormatFor(Shading) const;
    // Fullscreen passes read a resolved texture and write a single-sampled one.
    [[nodiscard]] int SamplesFor(Shading shading) const {
        switch (shading) {
            case Shading::Lit:
            case Shading::Flat:
            case Shading::LitInstanced:
                return samples;
            // A shadow map is never multisampled: it stores a distance, and
            // averaging two distances across a silhouette gives a value that
            // describes no surface. Single-sampling it also lets the same
            // pipeline serve the camera-space depth prepass.
            default:
                return 1;
        }
    }
    [[nodiscard]] rhi::PipelineId GetOrCreatePipeline(Shading shading,
                                                      bool depth_test,
                                                      bool blend,
                                                      std::string& error,
                                                      bool skinned = false);
};

// Which target a given shading writes into, and therefore which format its
// pipeline has to be compiled against.
//
// A pipeline is only valid for the format it was built against, and getting it
// wrong does NOT produce an error here — the draw lands in the attachment and
// what comes back out is undefined. It cost a long hunt: the bloom pipelines
// were compiled for the eight-bit output format while rendering into half-float
// targets, and every pixel of every bloom buffer sampled back as NaN. A NaN
// multiplied by a bloom strength of zero is still a NaN, so it survived to the
// framebuffer and zeroed one colour channel of the entire image.
//
// The rule is simply: does this shading write into an HDR target?
rhi::Format Renderer::Impl::FormatFor(Shading shading) const {
    switch (shading) {
        // The scene, and the bloom chain that reads it. All half-float, so the
        // passes downstream can tell a lamp from a sheet of white paper.
        case Shading::Lit:
        case Shading::Flat:
        case Shading::BloomBright:
        case Shading::BloomBlur:
        // The G-buffer is half-float for the same reason the scene is: an
        // emissive albedo can be far above 1, and an 8-bit target clips it to
        // white and takes the colour with it. The normal needs the precision
        // too -- an 8-bit normal quantises to about a degree, which shows as
        // banding across every smooth curve.
        case Shading::GBuffer:
        case Shading::DeferredLight:
        case Shading::LitInstanced:
            return Renderer::kSceneFormat;
        // The composite is the pass that brings it down to a display.
        default:
            return color;
    }
}

rhi::PipelineId Renderer::Impl::GetOrCreatePipeline(Shading shading,
                                                    bool depth_test,
                                                    bool blend,
                                                    std::string& error,
                                                    bool skinned) {
    const rhi::Format fmt = FormatFor(shading);
    const int msaa = SamplesFor(shading);
    // Blend and depth-write ARE pipeline state, unlike cull mode, so they have
    // to be in the key. Leaving them out would hand a transparent material the
    // opaque pipeline and quietly turn glass solid.
    // Skinning is in the KEY because it is a different vertex stage, not a
    // different uniform. Leaving it out would hand a skinned mesh the static
    // pipeline, which reads no palette and draws the bind pose forever.
    const std::uint64_t key = (std::uint64_t(msaa) << 20) |
                              (std::uint64_t(skinned) << 16) |
                              (std::uint64_t(shading) << 12) |
                              (std::uint64_t(blend) << 8) |
                              (std::uint64_t(depth_test) << 4) |
                              std::uint64_t(fmt);
    if (auto it = pipeline_cache.find(key); it != pipeline_cache.end())
        return it->second;

    rhi::PipelineDesc desc;
    if (shading == Shading::Lit) {
        desc.source = ShadingSource(kLitSrc);
        desc.vertex_fn = skinned ? "vs_skinned" : "vs_lit";
        desc.fragment_fn = "fs_lit";
    } else if (shading == Shading::LitInstanced) {
        // Same fragment stage as Lit, so the two agree about shading by
        // construction; only where the model matrix comes from differs.
        desc.source = ShadingSource(kLitSrc);
        desc.vertex_fn = "vs_lit_instanced";
        desc.fragment_fn = "fs_lit";
    } else if (shading == Shading::GBuffer) {
        // The G-buffer pass reuses the LIT vertex shader verbatim: the geometry
        // side of forward and deferred is identical, and only what the fragment
        // stage does with it differs.
        desc.source = ShadingSource(std::string(kLitSrc).append(kDeferredSrc).c_str());
        desc.vertex_fn = skinned ? "vs_skinned" : "vs_lit";
        desc.fragment_fn = "fs_gbuffer";
        desc.extra_colors = {kSceneFormat};  // attachment 1: normal + metallic
    } else if (shading == Shading::RayShadow) {
        desc.source = ShaderSource(kRaytraceSrc);
        desc.vertex_fn = "vs_rt_shadow";
        desc.fragment_fn = "fs_rt_shadow";
    } else if (shading == Shading::DeferredLight) {
        desc.source = ShadingSource(std::string(kLitSrc).append(kDeferredSrc).c_str());
        desc.vertex_fn = "vs_deferred";
        desc.fragment_fn = "fs_deferred";
    } else if (shading == Shading::Composite) {
        desc.source = ShaderSource(kCompositeSrc);
        desc.vertex_fn = "vs_composite";
        desc.fragment_fn = "fs_composite";
    } else if (shading == Shading::BloomBright) {
        desc.source = ShaderSource(kBloomSrc);
        desc.vertex_fn = "vs_bloom";
        desc.fragment_fn = "fs_bloom_bright";
    } else if (shading == Shading::BloomBlur) {
        desc.source = ShaderSource(kBloomSrc);
        desc.vertex_fn = "vs_bloom";
        desc.fragment_fn = "fs_bloom_blur";
    } else if (shading == Shading::Ssao) {
        desc.source = ShaderSource(kSsaoSrc);
        desc.vertex_fn = "vs_ssao";
        desc.fragment_fn = "fs_ssao";
    } else if (shading == Shading::ShadowDepth) {
        desc.source = ShaderSource(kShadowSrc);
        desc.vertex_fn = skinned ? "vs_shadow_skinned" : "vs_shadow";
        desc.fragment_fn = "fs_shadow";  // void, but it discards for the cut
        desc.depth_only = true;
    } else {
        desc.source = ShaderSource(kTriangleSrc);
        desc.vertex_fn = "vs_main";
        desc.fragment_fn = "fs_main";
    }
    desc.color = fmt;
    desc.samples = msaa;
    desc.depth = depth_test;
    desc.blend = blend ? rhi::Blend::Alpha : rhi::Blend::None;
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

namespace {

// True when this instance should be drawn through the skinned vertex stage.
bool IsSkinned(const GpuMesh& gm, const Instance& inst, const Scene& scene) {
    if (!Valid(gm.skin_vb) || gm.joint_count <= 0) return false;
    if (inst.palette < 0) return false;
    // The scene has to actually carry the matrices it points at. A short array
    // would otherwise be read past its end for the tail of the palette.
    return std::size_t(inst.palette) + std::size_t(gm.joint_count) <=
           scene.joint_matrices.size();
}

}  // namespace

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

// A mesh with no tangent frame, fixed up. Returns a reference to `scratch` when
// it had to build one, and to the original otherwise -- so a mesh that already
// carries tangents is uploaded with no copy at all.
//
// The check is on the FIRST vertex only. GenerateTangents always writes a unit
// vector, so a zero tangent means nothing ever ran; a mesh with one zeroed
// vertex and the rest filled cannot be produced by any path here.
static const Mesh& WithTangents(const Mesh& mesh, Mesh& scratch) {
    if (!mesh.vertices.empty() &&
        (mesh.vertices[0].tangent.x != 0.0f || mesh.vertices[0].tangent.y != 0.0f ||
         mesh.vertices[0].tangent.z != 0.0f))
        return mesh;
    // Here rather than in each generator on purpose. There are a dozen places
    // that build a Mesh -- boxes, terrain chunks, floor plans, hulls, the glTF
    // importer, the LOD simplifier -- and a normal map on a mesh whose tangent
    // is the zero vector does not look wrong, it produces NaN and a black
    // surface. One choke point that every mesh passes through cannot be
    // forgotten by the next generator someone writes.
    scratch = mesh;
    GenerateTangents(scratch);
    return scratch;
}

MeshHandle Renderer::UploadMesh(const Mesh& original) {
    if (original.vertices.empty() || original.indices.empty()) return {};
    Mesh scratch;
    const Mesh& mesh = WithTangents(original, scratch);
    GpuMesh gm;
    gm.vb = impl_->dev->CreateBuffer(mesh.vertices.data(),
                                     mesh.vertices.size() * sizeof(VertexIn));
    gm.ib = impl_->dev->CreateBuffer(mesh.indices.data(),
                                     mesh.indices.size() * sizeof(std::uint32_t));
    if (!Valid(gm.vb) || !Valid(gm.ib)) return {};
    gm.index_count = mesh.indices.size();
    gm.vertex_count = mesh.vertices.size();
    gm.bounds = mesh.bounds;
    gm.lods[0].vb = gm.vb;
    gm.lods[0].ib = gm.ib;
    gm.lods[0].index_count = gm.index_count;
    gm.lods[0].vertex_count = gm.vertex_count;
    gm.lod_count = 1;
    impl_->meshes.push_back(gm);
    return MeshHandle{std::uint32_t(impl_->meshes.size() - 1)};
}

MeshHandle Renderer::UploadMeshLods(std::span<const Mesh> levels) {
    if (levels.empty()) return {};
    const MeshHandle h = UploadMesh(levels[0]);
    if (!Valid(h)) return {};
    GpuMesh& gm = impl_->meshes[h.v];
    const int count = std::min(int(levels.size()), kMaxLods);
    for (int i = 1; i < count; ++i) {
        Mesh lod_scratch;
        const Mesh& m = WithTangents(levels[std::size_t(i)], lod_scratch);
        if (m.vertices.empty() || m.indices.empty()) break;
        GpuMesh::Lod lod;
        lod.vb = impl_->dev->CreateBuffer(m.vertices.data(),
                                          m.vertices.size() * sizeof(VertexIn));
        lod.ib = impl_->dev->CreateBuffer(m.indices.data(),
                                          m.indices.size() * sizeof(std::uint32_t));
        // A level that fails to allocate TRUNCATES the chain rather than
        // leaving a hole. A null buffer in the middle would be selected by the
        // cull pass and drawn as nothing, so the object would vanish at one
        // distance and come back at the next.
        if (!Valid(lod.vb) || !Valid(lod.ib)) break;
        lod.index_count = m.indices.size();
        lod.vertex_count = m.vertices.size();
        gm.lods[i] = lod;
        gm.lod_count = i + 1;
    }
    return h;
}

int Renderer::MeshLodCount(MeshHandle m) const {
    if (!Valid(m) || m.v >= impl_->meshes.size()) return 0;
    return impl_->meshes[m.v].lod_count;
}

int Renderer::MeshLodIndexCount(MeshHandle m, int lod) const {
    if (!Valid(m) || m.v >= impl_->meshes.size()) return 0;
    const GpuMesh& gm = impl_->meshes[m.v];
    if (lod < 0 || lod >= gm.lod_count) return 0;
    return int(gm.lods[lod].index_count);
}

MeshHandle Renderer::UploadSkinnedMesh(const Mesh& mesh,
                                       const std::vector<anim::SkinVertex>& skin,
                                       int joint_count) {
    // A short skin array would leave the tail of the mesh reading whatever came
    // after it in memory, so this is a refusal rather than a clamp.
    if (skin.size() != mesh.vertices.size()) return {};
    if (joint_count <= 0 || joint_count > kMaxJoints) return {};

    const MeshHandle h = UploadMesh(mesh);
    if (!Valid(h)) return {};

    // Widened to 32-bit indices to match the shader's uint4. See SkinIn.
    std::vector<SkinIn> gpu(skin.size());
    for (std::size_t i = 0; i < skin.size(); ++i) {
        for (int c = 0; c < anim::kMaxInfluences; ++c) {
            const std::uint16_t j = skin[i].joints[c];
            // An index past the palette would read another instance's matrices.
            // Clamping to 0 with the weight left alone is wrong-looking but
            // bounded; the alternative is undefined.
            const std::uint32_t safe = j < joint_count ? j : 0u;
            (&gpu[i].joints.x)[c] = safe;
            (&gpu[i].weights.x)[c] = skin[i].weights[c];
        }
    }
    GpuMesh& gm = impl_->meshes[h.v];
    gm.skin_vb = impl_->dev->CreateBuffer(gpu.data(), gpu.size() * sizeof(SkinIn));
    gm.joint_count = joint_count;
    if (!Valid(gm.skin_vb)) return {};
    return h;
}

int Renderer::JointCount(MeshHandle m) const {
    if (!Valid(m) || m.v >= impl_->meshes.size()) return 0;
    return impl_->meshes[m.v].joint_count;
}

MaterialHandle Renderer::CreateMaterial(const MaterialDesc& desc,
                                        std::string& error) {
    // DERIVED SHADING MODES ARE NOT ASKABLE-FOR. GBuffer and LitInstanced are
    // variants the renderer builds from a Lit material -- DrawGBuffer and
    // DrawSceneIndirect reach for `gbuffer_pipeline` and `instanced_pipeline`,
    // which are only filled in for Lit.
    //
    // Asking for one directly used to succeed and produce a material that drew
    // NOTHING: its `pipeline` was the G-buffer program, its `gbuffer_pipeline`
    // was null, and DrawGBuffer skipped it as incompatible. An empty frame with
    // no error is the worst possible answer, and it cost a long debugging
    // session to find. Rejecting it says so at the call site instead.
    if (desc.shading == Shading::GBuffer || desc.shading == Shading::LitInstanced) {
        error =
            "Shading::GBuffer and Shading::LitInstanced are variants the "
            "renderer derives from Shading::Lit, not modes to ask for: create "
            "the material as Lit and call DrawGBuffer or DrawSceneIndirect";
        return {};
    }
    GpuMaterial m;
    m.pipeline = impl_->GetOrCreatePipeline(desc.shading, desc.depth_test,
                                            desc.transparent, error);
    if (!Valid(m.pipeline)) return {};
    // Only Lit has a skinned counterpart. A fullscreen or depth-only pass has
    // no vertices of its own to blend.
    if (desc.shading == Shading::Lit) {
        m.skinned_pipeline = impl_->GetOrCreatePipeline(
            desc.shading, desc.depth_test, desc.transparent, error, /*skinned=*/true);
        if (!Valid(m.skinned_pipeline)) return {};
        // A TRANSPARENT material has no G-buffer form, and that is not an
        // oversight: a G-buffer holds one surface per pixel and glass needs
        // the one behind it too. Transparent geometry is drawn forward, after
        // the deferred lighting pass.
        if (!desc.transparent && desc.depth_test) {
            m.instanced_pipeline = impl_->GetOrCreatePipeline(
                Shading::LitInstanced, desc.depth_test, false, error);
            if (!Valid(m.instanced_pipeline)) return {};
            m.gbuffer_pipeline = impl_->GetOrCreatePipeline(
                Shading::GBuffer, desc.depth_test, false, error);
            if (!Valid(m.gbuffer_pipeline)) return {};
            m.gbuffer_skinned_pipeline = impl_->GetOrCreatePipeline(
                Shading::GBuffer, desc.depth_test, false, error, /*skinned=*/true);
            if (!Valid(m.gbuffer_skinned_pipeline)) return {};
        }
    }
    m.cull = desc.cull;
    m.depth_test = desc.depth_test;
    m.transparent = desc.transparent;
    m.base_color = desc.base_color;
    m.roughness = desc.roughness;
    m.metallic = desc.metallic;
    m.emissive = desc.emissive;
    m.normal_strength = desc.normal_strength;
    m.albedo = Valid(desc.albedo) ? desc.albedo : impl_->white;
    m.roughness_map =
        Valid(desc.roughness_map) ? desc.roughness_map : impl_->white;
    m.metallic_map =
        Valid(desc.metallic_map) ? desc.metallic_map : impl_->white;
    m.emissive_map =
        Valid(desc.emissive_map) ? desc.emissive_map : impl_->white;
    m.occlusion_map =
        Valid(desc.occlusion_map) ? desc.occlusion_map : impl_->white;
    // NOT substituted with white. Every other map defaults to a 1x1 white
    // texture so the shader can multiply unconditionally, but a normal map has
    // no such identity value -- white unpacks to (1, 1) which is off the unit
    // circle entirely. The shader tests is_null_texture instead, so this stays
    // a null handle and binds nothing.
    m.normal_map = desc.normal_map;
    impl_->materials.push_back(m);
    return MaterialHandle{std::uint32_t(impl_->materials.size() - 1)};
}

int Renderer::Samples() const { return impl_->samples; }

void Renderer::SetClusteredLighting(bool on) {
    if (on == impl_->clustered) return;
    if (on) {
        std::string error;
        if (!Valid(impl_->cluster_bin)) {
            const std::string src = std::string(kShaderTypesSrc) + "\n" + kClusterSrc;
            impl_->cluster_bin =
                impl_->dev->CreateComputePipeline(src, "cs_cluster_lights", error);
        }
        if (!Valid(impl_->cluster_counts))
            impl_->cluster_counts = impl_->dev->CreateStorageBuffer(
                sizeof(std::uint32_t) * ENG_CLUSTER_COUNT);
        if (!Valid(impl_->cluster_indices))
            impl_->cluster_indices = impl_->dev->CreateStorageBuffer(
                sizeof(std::uint32_t) * ENG_CLUSTER_COUNT * ENG_CLUSTER_CAPACITY);
        // Any allocation failure leaves clustering OFF rather than half on. A
        // half-configured clustered path binds a null index buffer, and the
        // shader's null check then silently falls back -- which is the right
        // behaviour but hides the failure, so refuse here where it is visible.
        if (!Valid(impl_->cluster_bin) || !Valid(impl_->cluster_counts) ||
            !Valid(impl_->cluster_indices))
            return;
    }
    impl_->clustered = on;
    impl_->bins_valid = false;
}

bool Renderer::ClusteredLighting() const { return impl_->clustered; }

void Renderer::BinLights(rhi::ComputeEncoder& enc, const Scene& scene, int width,
                         int height, float far_distance) {
    if (!impl_->clustered || width <= 0 || height <= 0) return;
    const int light_count =
        std::min(int(scene.lights.size()), int(ENG_MAX_LIGHTS));

    const float aspect = float(width) / float(height);
    const float near_z = std::max(scene.camera.nearZ, 1e-3f);
    const float far_z = std::max(far_distance, near_z * 2.0f);

    GpuClusters& c = impl_->cluster_params;
    c.grid = Vec4{float(ENG_CLUSTER_X), float(ENG_CLUSTER_Y), float(ENG_CLUSTER_Z),
                  float(ENG_CLUSTER_CAPACITY)};
    c.depth = Vec4{near_z, far_z, std::log(far_z / near_z), float(width)};
    c.screen = Vec4{float(height), float(width) / float(ENG_CLUSTER_X),
                    float(height) / float(ENG_CLUSTER_Y), 0.0f};
    c.slope = Vec4{1.0f / std::tan(scene.camera.fovY * 0.5f), aspect, 0.0f, 0.0f};

    // The lights go in their own buffer for the binning pass. The per-draw
    // uniform ring cannot serve: this runs in a COMPUTE pass, before any draw
    // has allocated a slice, and the slice would be recycled by the time the
    // fragment stage read it.
    if (!Valid(impl_->cluster_lights))
        impl_->cluster_lights =
            impl_->dev->CreateStorageBuffer(sizeof(GpuLight) * ENG_MAX_LIGHTS);
    if (!Valid(impl_->cluster_candidates))
        impl_->cluster_candidates =
            impl_->dev->CreateStorageBuffer(sizeof(std::uint32_t) * ENG_MAX_LIGHTS);
    if (!Valid(impl_->cluster_lights) || !Valid(impl_->cluster_candidates)) return;
    if (auto* dst = static_cast<GpuLight*>(impl_->dev->MapBuffer(impl_->cluster_lights)))
        impl_->FillLights(scene, dst, light_count);

    // FRUSTUM PRE-CULL, on the CPU, before the grid ever sees a light.
    //
    // Binning is O(cells x lights) -- 3456 sphere-box tests per light -- and
    // without this it is O(cells x EVERY light in the scene), so a level with a
    // thousand lamps pays for all thousand however few are on screen. That
    // turns the one cost clustering was supposed to remove into a smaller
    // version of itself.
    //
    // On the CPU because there are at most a few hundred lights and each test
    // is six dot products: a kernel to do it would cost more in dispatch than
    // in arithmetic, and it would need a compaction pass to produce the list.
    int candidate_count = 0;
    if (auto* cand = static_cast<std::uint32_t*>(
            impl_->dev->MapBuffer(impl_->cluster_candidates))) {
        const Frustum frustum = Frustum::FromViewProj(
            scene.camera.ViewProj(aspect));
        for (int i = 0; i < light_count; ++i) {
            const Light& l = scene.lights[std::size_t(i)];
            if (!frustum.IntersectsSphere(l.position, l.range)) continue;
            cand[candidate_count++] = std::uint32_t(i);
        }
    }

    // WORLD -> VIEW. The binning shader works entirely in view space, where a
    // cell is an axis-aligned box; in world space every cell would be an
    // arbitrarily oriented one and the sphere test would need eight planes.
    const Mat4 view = Mat4::LookAt(scene.camera.eye, scene.camera.target,
                                   scene.camera.up);

    // The candidate count goes in AFTER the cull, which is why this is not set
    // with the rest of the block above.
    c.screen.w = float(candidate_count);

    enc.SetPipeline(impl_->cluster_bin);
    enc.SetBytes(&c, sizeof(c), 0);
    enc.SetBytes(&view, sizeof(view), 1);
    enc.SetBuffer(impl_->cluster_lights, 0, 2);
    enc.SetBuffer(impl_->cluster_counts, 0, 3);
    enc.SetBuffer(impl_->cluster_indices, 0, 4);
    enc.SetBuffer(impl_->cluster_candidates, 0, 5);
    enc.Dispatch3D(ENG_CLUSTER_X, ENG_CLUSTER_Y, ENG_CLUSTER_Z, 4, 3, 4);
    impl_->bins_valid = true;
}

Renderer::ClusterStats Renderer::ReadClusterStats() {
    ClusterStats st;
    if (!impl_->bins_valid || !Valid(impl_->cluster_counts)) return st;
    const auto* counts =
        static_cast<const std::uint32_t*>(impl_->dev->MapBuffer(impl_->cluster_counts));
    if (!counts) return st;
    long long total = 0;
    for (int i = 0; i < ENG_CLUSTER_COUNT; ++i) {
        const int n = int(counts[i]);
        if (n <= 0) continue;
        ++st.occupied_cells;
        total += n;
        if (n > st.max_per_cell) {
            st.max_per_cell = n;
            st.max_slice = i / (ENG_CLUSTER_X * ENG_CLUSTER_Y);
        }
        if (n >= ENG_CLUSTER_CAPACITY) ++st.overflowed_cells;
    }
    if (st.occupied_cells > 0)
        st.mean_per_occupied = double(total) / st.occupied_cells;
    return st;
}

void Renderer::SetAnisotropy(int max_anisotropy) {
    const int n = max_anisotropy < 1 ? 1 : (max_anisotropy > 16 ? 16 : max_anisotropy);
    if (n == impl_->anisotropy) return;
    const rhi::SamplerId s =
        impl_->dev->CreateSampler(rhi::Filter::Linear, rhi::Wrap::Repeat, n);
    // Keep the old one on failure. A null sampler bound to slot 0 is not a
    // degraded picture, it is undefined sampling in every material.
    if (!Valid(s)) return;
    impl_->sampler = s;
    impl_->anisotropy = n;
}

int Renderer::Anisotropy() const { return impl_->anisotropy; }

std::unique_ptr<Renderer> Renderer::Create(rhi::Device& dev, rhi::Format color,
                                           std::string& error, int samples) {
    auto r = std::unique_ptr<Renderer>(new Renderer());
    r->impl_->dev = &dev;
    r->impl_->color = color;
    // 1, 2, 4 or 8. Anything else is silently a typo, and a pipeline built for
    // a sample count no attachment can have fails at draw time rather than here.
    r->impl_->samples = (samples == 2 || samples == 4 || samples == 8) ? samples : 1;

    // Defaults first: CreateMaterial substitutes these for absent maps, so they
    // have to exist before any material does.
    const std::uint8_t kWhitePixel[4] = {255, 255, 255, 255};
    const std::uint8_t kBlackPixel[4] = {0, 0, 0, 255};
    r->impl_->white = dev.CreateTexture2D(1, 1, kWhitePixel);
    r->impl_->black = dev.CreateTexture2D(1, 1, kBlackPixel);
    const float kUnitExposure = 1.0f;
    r->impl_->unit_exposure = dev.CreateBuffer(&kUnitExposure, sizeof(float));
    // 16x anisotropic on the MATERIAL sampler. Every textured surface in a
    // scene is eventually seen at a grazing angle -- a floor always is -- and
    // that is exactly the case a mip chain alone handles badly.
    r->impl_->sampler =
        dev.CreateSampler(rhi::Filter::Linear, rhi::Wrap::Repeat, 16);
    r->impl_->clamp_sampler = dev.CreateSampler(rhi::Filter::Linear, rhi::Wrap::Clamp);
    r->impl_->shadow_sampler = dev.CreateSampler(rhi::Filter::Linear, rhi::Wrap::Clamp);
    if (!Valid(r->impl_->white) || !Valid(r->impl_->black) ||
        !Valid(r->impl_->sampler) || !Valid(r->impl_->clamp_sampler) ||
        !Valid(r->impl_->shadow_sampler)) {
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
    r->impl_->bloom_bright = r->impl_->GetOrCreatePipeline(
        Shading::BloomBright, false, false, error);
    if (!Valid(r->impl_->bloom_bright)) return nullptr;
    r->impl_->bloom_blur = r->impl_->GetOrCreatePipeline(
        Shading::BloomBlur, false, false, error);
    if (!Valid(r->impl_->bloom_blur)) return nullptr;
    r->impl_->shadow_skinned = r->impl_->GetOrCreatePipeline(
        Shading::ShadowDepth, true, false, error, /*skinned=*/true);
    if (!Valid(r->impl_->shadow_skinned)) return nullptr;
    // No depth test: the lighting pass covers the screen and decides what is
    // background by reading the depth buffer as a TEXTURE, which it could not
    // do if the same buffer were attached for testing.
    r->impl_->deferred_light = r->impl_->GetOrCreatePipeline(
        Shading::DeferredLight, false, false, error);
    if (!Valid(r->impl_->deferred_light)) return nullptr;

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
    r->impl_->cascade_slot_bytes =
        AlignUp(sizeof(GpuCascades), dev.UniformAlignment());
    r->impl_->cascades =
        dev.CreateDynamicBuffer(r->impl_->cascade_slot_bytes * rhi::kFramesInFlight);
    r->impl_->cascade_map =
        static_cast<std::uint8_t*>(dev.MapBuffer(r->impl_->cascades));
    if (!Valid(r->impl_->cascades) || !r->impl_->cascade_map) {
        error = "failed to allocate the cascade buffer";
        return nullptr;
    }

    r->impl_->light_slot_bytes =
        AlignUp(sizeof(GpuLight) * ENG_MAX_LIGHTS, dev.UniformAlignment());
    r->impl_->lights =
        dev.CreateDynamicBuffer(r->impl_->light_slot_bytes * rhi::kFramesInFlight);
    r->impl_->light_map = static_cast<std::uint8_t*>(dev.MapBuffer(r->impl_->lights));
    if (!Valid(r->impl_->lights) || !r->impl_->light_map) {
        error = "failed to allocate the light buffer";
        return nullptr;
    }

    // Joint palettes get their own ring, sized for kMaxSkinnedPerFrame draws.
    r->impl_->palette_stride =
        AlignUp(sizeof(Mat4) * kMaxJoints, dev.UniformAlignment());
    r->impl_->palette_slot_bytes =
        r->impl_->palette_stride * std::size_t(kMaxSkinnedPerFrame);
    r->impl_->palettes =
        dev.CreateDynamicBuffer(r->impl_->palette_slot_bytes * rhi::kFramesInFlight);
    r->impl_->palette_map =
        static_cast<std::uint8_t*>(dev.MapBuffer(r->impl_->palettes));
    if (!Valid(r->impl_->palettes) || !r->impl_->palette_map) {
        error = "failed to allocate the joint palette ring";
        return nullptr;
    }

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
    // Cascades tile the map. One cascade uses the whole thing, which is the
    // behaviour this had before they existed.
    const int cascades = std::clamp(scene.shadowCascades, 1, 4);
    const int per_side = cascades > 1 ? 2 : 1;
    const int tile_px = kDirectionalShadowSize / per_side;
    const float aspect = impl_->last_aspect > 0.0f ? impl_->last_aspect : 1.7778f;

    // The pipeline is set PER DRAW now rather than once, because a skinned
    // caster needs the skinned vertex stage. Tracked so an all-static scene
    // still binds it exactly once.
    rhi::PipelineId bound_shadow;
    // Front faces culled instead of back: shifting the recorded depth to the
    // BACK of each caster is the cheapest peter-panning-free way to keep a
    // surface from shadowing itself, and it costs nothing here because every
    // mesh in this engine is closed.
    enc.SetCull(rhi::Cull::Front, rhi::Winding::CounterClockwise);

    for (int cascade = 0; cascade < cascades; ++cascade) {
    const Mat4 lightViewProj =
        cascades > 1 ? scene.CascadeViewProj(cascade, aspect) : scene.LightViewProj();
    if (cascades > 1) {
        enc.SetViewport((cascade % per_side) * tile_px, (cascade / per_side) * tile_px,
                        tile_px, tile_px);
        enc.SetScissor((cascade % per_side) * tile_px, (cascade / per_side) * tile_px,
                       tile_px, tile_px);
    }
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

        const bool skinned = IsSkinned(gm, inst, scene);
        std::size_t palette_offset = Impl::kNoSpace;
        if (skinned) {
            palette_offset = impl_->AllocPalette();
            if (palette_offset == Impl::kNoSpace) continue;  // ring full: skip
            std::memcpy(impl_->palette_map + palette_offset,
                        scene.joint_matrices.data() + inst.palette,
                        sizeof(Mat4) * std::size_t(gm.joint_count));
        }
        const rhi::PipelineId want =
            skinned ? impl_->shadow_skinned : impl_->shadow;
        if (want.v != bound_shadow.v) {
            enc.SetPipeline(want);
            bound_shadow = want;
        }

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
        if (skinned) {
            enc.SetVertexBuffer(gm.skin_vb, 0, kSkinSlot);
            enc.SetVertexBuffer(impl_->palettes, palette_offset, kPaletteSlot);
        }
        enc.DrawIndexedU32(gm.ib, gm.index_count);
        ++impl_->shadow_draws;
    }
    }
    if (cascades > 1) {
        enc.SetViewport(0, 0, kDirectionalShadowSize, kDirectionalShadowSize);
        enc.SetScissor(0, 0, kDirectionalShadowSize, kDirectionalShadowSize);
    }
}

rhi::TextureId Renderer::ShadowAtlas() {
    if (!Valid(impl_->shadow_atlas))
        impl_->shadow_atlas = impl_->dev->CreateShadowMap(kShadowAtlasSize);
    return impl_->shadow_atlas;
}

int Renderer::ShadowedLightCount() const {
    int n = 0;
    for (int t : impl_->shadow_tile_of_light)
        if (t >= 0) ++n;
    return n;
}

int Renderer::ShadowTilesUsed() const { return impl_->shadow_tiles_used; }

void Renderer::DrawLightShadows(rhi::Encoder& enc, const Scene& scene) {
    impl_->shadow_tile_of_light.assign(scene.lights.size(), -1);
    impl_->shadow_tiles_used = 0;
    if (!Valid(ShadowAtlas())) return;

    constexpr int kTile = kShadowAtlasSize / kShadowTilesPerSide;
    enc.SetPipeline(impl_->shadow);
    // Same reasoning as the directional pass: recording the BACK of each caster
    // keeps a surface from shadowing itself without a peter-panning offset.
    enc.SetCull(rhi::Cull::Front, rhi::Winding::CounterClockwise);
    rhi::PipelineId bound = impl_->shadow;

    int tile = 0;
    for (std::size_t li = 0; li < scene.lights.size(); ++li) {
        const Light& light = scene.lights[li];
        const int faces = light.ShadowFaces();
        if (faces == 0) continue;
        // All the faces or none. Five sides of a cube shadow is worse than no
        // cube shadow: the sixth direction is lit straight through walls, and
        // it is the one direction nobody thinks to check.
        if (tile + faces > kShadowTiles) continue;

        for (int f = 0; f < faces; ++f) {
            const int index = tile + f;
            const int tx = (index % kShadowTilesPerSide) * kTile;
            const int ty = (index / kShadowTilesPerSide) * kTile;
            enc.SetViewport(tx, ty, kTile, kTile);
            // The scissor as well. The viewport only remaps clip space; without
            // this a triangle that lands outside the tile still rasterises,
            // into whichever neighbour is there.
            enc.SetScissor(tx, ty, kTile, kTile);

            const Mat4 light_vp = faces == 1 ? light.ViewProj()
                                             : light.CubeFaceViewProj(f);
            for (const Instance& inst : scene.instances) {
                if (!Valid(inst.mesh) || inst.mesh.v >= impl_->meshes.size()) continue;
                if (Valid(inst.material) && inst.material.v < impl_->materials.size() &&
                    impl_->materials[inst.material.v].transparent)
                    continue;
                const GpuMesh& gm = impl_->meshes[inst.mesh.v];
                const std::size_t offset = impl_->AllocUniform();
                if (offset == Impl::kNoSpace) break;

                const bool skinned = IsSkinned(gm, inst, scene);
                std::size_t palette_offset = Impl::kNoSpace;
                if (skinned) {
                    palette_offset = impl_->AllocPalette();
                    if (palette_offset == Impl::kNoSpace) continue;
                    std::memcpy(impl_->palette_map + palette_offset,
                                scene.joint_matrices.data() + inst.palette,
                                sizeof(Mat4) * std::size_t(gm.joint_count));
                }
                const rhi::PipelineId want =
                    skinned ? impl_->shadow_skinned : impl_->shadow;
                if (want.v != bound.v) {
                    enc.SetPipeline(want);
                    bound = want;
                }

                FrameUniforms u{};
                u.lightViewProj = light_vp;
                u.model = inst.model;
                u.surface = Vec4{0.0f, 0.0f, 0.0f, scene.clipY};
                std::memcpy(impl_->uniform_map + offset, &u, sizeof(u));

                enc.SetVertexBuffer(gm.vb, 0, kVertexSlot);
                enc.SetVertexBuffer(impl_->uniforms, offset, kUniformSlot);
                enc.SetFragmentBuffer(impl_->uniforms, offset, kUniformSlot);
                if (skinned) {
                    enc.SetVertexBuffer(gm.skin_vb, 0, kSkinSlot);
                    enc.SetVertexBuffer(impl_->palettes, palette_offset, kPaletteSlot);
                }
                enc.DrawIndexedU32(gm.ib, gm.index_count);
            }
        }
        impl_->shadow_tile_of_light[li] = tile;
        tile += faces;
    }
    impl_->shadow_tiles_used = tile;
    // Hand the whole target back, or every later pass inherits the last tile.
    enc.SetViewport(0, 0, kShadowAtlasSize, kShadowAtlasSize);
    enc.SetScissor(0, 0, kShadowAtlasSize, kShadowAtlasSize);
}

void Renderer::DrawSceneDepth(rhi::Encoder& enc, const Scene& scene, int width,
                              int height) {
    if (width <= 0 || height <= 0) return;
    // The shadow vertex stage with the CAMERA's matrix instead of a light's.
    // Depth from the eye is the same operation as depth from a lamp, and the
    // pipeline is single-sampled, which is what makes this readable by SSAO
    // when the colour pass is running multisampled.
    const Mat4 viewProj = scene.camera.ViewProj(float(width) / float(height));

    rhi::PipelineId bound;
    enc.SetCull(rhi::Cull::Back, rhi::Winding::CounterClockwise);
    for (const Instance& inst : scene.instances) {
        if (!Valid(inst.mesh) || inst.mesh.v >= impl_->meshes.size()) continue;
        if (Valid(inst.material) && inst.material.v < impl_->materials.size() &&
            impl_->materials[inst.material.v].transparent)
            continue;
        const GpuMesh& gm = impl_->meshes[inst.mesh.v];
        const std::size_t offset = impl_->AllocUniform();
        if (offset == Impl::kNoSpace) break;

        const bool skinned = IsSkinned(gm, inst, scene);
        std::size_t palette_offset = Impl::kNoSpace;
        if (skinned) {
            palette_offset = impl_->AllocPalette();
            if (palette_offset == Impl::kNoSpace) continue;
            std::memcpy(impl_->palette_map + palette_offset,
                        scene.joint_matrices.data() + inst.palette,
                        sizeof(Mat4) * std::size_t(gm.joint_count));
        }
        const rhi::PipelineId want = skinned ? impl_->shadow_skinned : impl_->shadow;
        if (want.v != bound.v) {
            enc.SetPipeline(want);
            bound = want;
        }

        FrameUniforms u{};
        u.lightViewProj = viewProj;
        u.model = inst.model;
        u.surface = Vec4{0.0f, 0.0f, 0.0f, scene.clipY};
        std::memcpy(impl_->uniform_map + offset, &u, sizeof(u));

        enc.SetVertexBuffer(gm.vb, 0, kVertexSlot);
        enc.SetVertexBuffer(impl_->uniforms, offset, kUniformSlot);
        enc.SetFragmentBuffer(impl_->uniforms, offset, kUniformSlot);
        if (skinned) {
            enc.SetVertexBuffer(gm.skin_vb, 0, kSkinSlot);
            enc.SetVertexBuffer(impl_->palettes, palette_offset, kPaletteSlot);
        }
        enc.DrawIndexedU32(gm.ib, gm.index_count);
    }
}

void Renderer::DrawScene(rhi::Encoder& enc, const Scene& scene, int width,
                         int height, rhi::TextureId shadow_map) {
    impl_->DrawGeometry(enc, scene, width, height, shadow_map, /*gbuffer=*/false);
}

void Renderer::DrawGBuffer(rhi::Encoder& enc, const Scene& scene, int width,
                           int height) {
    // No shadow map: the G-buffer pass does not light anything, so it has no
    // use for one. Shadows are sampled by the lighting pass that follows.
    impl_->DrawGeometry(enc, scene, width, height, {}, /*gbuffer=*/true);
}

// The light list and the cascade block, uploaded once per pass.
//
// Extracted so the forward and deferred paths share them rather than each
// filling the buffer its own way. They are read by the same shader function, so
// a difference here would be a difference in the picture -- and one that only
// appears when the renderer is switched between paths, which is the hardest
// kind to notice.
void Renderer::Impl::UploadLights(const Scene& scene, std::size_t light_offset,
                                  int light_count) {
    FillLights(scene, reinterpret_cast<GpuLight*>(light_map + light_offset),
               light_count);
}

void Renderer::Impl::FillLights(const Scene& scene, GpuLight* dst,
                                int light_count) const {
        for (int i = 0; i < light_count; ++i) {
            const Light& l = scene.lights[std::size_t(i)];
            GpuLight g{};
            g.position = Vec4{l.position.x, l.position.y, l.position.z,
                              l.type == LightType::Spot ? 1.0f : 0.0f};
            const Vec3 dir = Normalize(l.direction);
            g.direction = Vec4{dir.x, dir.y, dir.z, l.range};
            g.color = Vec4{l.color.x, l.color.y, l.color.z, 0.0f};
            // Cosines, not degrees: the shader compares against a dot product,
            // and converting per fragment would be an acos in the inner loop.
            const float inner = std::cos(l.inner_degrees * 3.14159265f / 180.0f);
            const float outer = std::cos(l.outer_degrees * 3.14159265f / 180.0f);
            // Guarded: an inner cone wider than the outer one divides by a
            // negative and inverts the falloff.
            g.cone = Vec4{std::max(inner, outer + 1e-3f), outer, 0.0f, 0.0f};

            // Where its depth map sits, if DrawLightShadows gave it one.
            const int tile = std::size_t(i) < shadow_tile_of_light.size()
                                 ? shadow_tile_of_light[std::size_t(i)]
                                 : -1;
            if (tile >= 0) {
                // A spot carries its projection; a point light does not need
                // one, because its lookup is a cube-face pick from a direction.
                if (l.type == LightType::Spot) g.viewProj = l.ViewProj();
                g.shadow = Vec4{float(tile), float(kShadowTilesPerSide),
                                l.shadow_near,
                                l.type == LightType::Spot ? 1.0f : 2.0f};
            }
            dst[i] = g;
        }
    }

void Renderer::Impl::UploadCascades(const Scene& scene,
                                    std::size_t cascade_offset, float aspect) {
        const int n = std::clamp(scene.shadowCascades, 1, 4);
        auto* dst = reinterpret_cast<GpuCascades*>(cascade_map + cascade_offset);
        GpuCascades g{};
        for (int i = 0; i < n; ++i) {
            g.viewProj[i] = n > 1 ? scene.CascadeViewProj(i, aspect)
                                  : scene.LightViewProj();
            (&g.splits.x)[i] = scene.CascadeSplit(i + 1);
        }
        // Anything past the last cascade falls back to it rather than going
        // unshadowed at a hard line across the floor.
        for (int i = n; i < 4; ++i) (&g.splits.x)[i] = 1e9f;
        g.info = Vec4{float(n), float(n > 1 ? 2 : 1), n > 1 ? 0.5f : 1.0f, 0.0f};
        *dst = g;
    }

void Renderer::Impl::DrawGeometry(rhi::Encoder& enc, const Scene& scene,
                                  int width, int height,
                                  rhi::TextureId shadow_map, bool gbuffer) {
    stats = RenderStats{};
    stats.submitted = int(scene.instances.size());
    draw_order.clear();
    if (width <= 0 || height <= 0 || scene.instances.empty()) return;

    // A pipeline built without depth cannot run in a pass that has one, and
    // vice versa — Metal rejects the draw outright. Rather than let the caller
    // find out as a validation abort, skip and count.
    const bool pass_depth = dev->CurrentPassHasDepth();

    const Mat4 viewProj = scene.camera.ViewProj(float(width) / float(height));
    // The light list, uploaded once for the whole pass. Anything past the
    // budget is DROPPED rather than wrapped: silently lighting a scene with a
    // different subset every frame would flicker.
    const std::size_t light_offset =
        std::size_t(dev->FrameSlot()) * light_slot_bytes;
    const int light_count =
        std::min(int(scene.lights.size()), int(ENG_MAX_LIGHTS));
    UploadLights(scene, light_offset, light_count);

    last_aspect = float(width) / float(height);
    // Kept for the deferred lighting pass, which reconstructs a world position
    // from the depth buffer and has no vertex to have carried one.
    inv_view_proj = Inverse(viewProj);
    const bool shadows = scene.shadowExtent > 0.0f && Valid(shadow_map);

    const std::size_t cascade_offset =
        std::size_t(dev->FrameSlot()) * cascade_slot_bytes;
    UploadCascades(scene, cascade_offset, last_aspect);

    const Mat4 lightViewProj = shadows ? scene.LightViewProj() : Mat4::Identity();
    const rhi::TextureId shadow_tex = shadows ? shadow_map : dummy_shadow;

    // --- cull ----------------------------------------------------------------
    // Reject before touching the GPU at all. A culled object costs one plane
    // test; a submitted one costs a draw call plus whatever it rasterises.
    const Frustum frustum = Frustum::FromViewProj(viewProj);
    const Vec3 eye = scene.camera.eye;

    visible.clear();
    for (int idx = 0; idx < int(scene.instances.size()); ++idx) {
        const Instance& inst = scene.instances[idx];
        if (!Valid(inst.mesh) || inst.mesh.v >= meshes.size() ||
            !Valid(inst.material) || inst.material.v >= materials.size()) {
            ++stats.invalid;
            continue;
        }
        if (materials[inst.material.v].depth_test != pass_depth) {
            ++stats.incompatible;
            continue;
        }

        const GpuMesh& gm = meshes[inst.mesh.v];
        // Object-space bounds pushed out to world space. A sphere only needs
        // its centre moved and its radius scaled — no re-fitting, which is
        // exactly why the bounds are a sphere and not a box.
        const Vec4 c = inst.model * Vec4{gm.bounds.center.x, gm.bounds.center.y,
                                         gm.bounds.center.z, 1.0f};
        const Vec3 world_center{c.x, c.y, c.z};
        const float world_radius = gm.bounds.radius * MaxScale(inst.model);

        if (!frustum.IntersectsSphere(world_center, world_radius)) {
            ++stats.culled;
            continue;
        }

        DrawItem item;
        item.inst = &inst;
        item.index = idx;
        item.pipeline = materials[inst.material.v].pipeline.v;
        item.depth = Length(world_center - eye);
        item.transparent = materials[inst.material.v].transparent;
        visible.push_back(item);
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
    std::sort(visible.begin(), visible.end(),
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

    for (const DrawItem& item : visible) {
        const Instance& inst = *item.inst;
        const std::size_t offset = AllocUniform();
        if (offset == Impl::kNoSpace) {
            // Ring slot full. Drop the rest rather than overrun, and SAY so —
            // a silently short frame is the worst possible failure here.
            stats.overflowed =
                int(visible.size()) - stats.draws;
            break;
        }

        const GpuMesh& gm = meshes[inst.mesh.v];
        const GpuMaterial& mat = materials[inst.material.v];

        const bool skinned = IsSkinned(gm, *item.inst, scene);
        std::size_t palette_offset = Impl::kNoSpace;
        if (skinned) {
            palette_offset = AllocPalette();
            if (palette_offset == Impl::kNoSpace) {
                ++stats.overflowed;
                continue;
            }
            std::memcpy(palette_map + palette_offset,
                        scene.joint_matrices.data() + item.inst->palette,
                        sizeof(Mat4) * std::size_t(gm.joint_count));
        }
        // In the deferred geometry pass a material without a G-buffer form is
        // SKIPPED, not drawn forward: it is transparent, and drawing it here
        // would write its surface into the buffer as if it were opaque and
        // erase whatever is behind it.
        const rhi::PipelineId want =
            gbuffer ? (skinned ? mat.gbuffer_skinned_pipeline : mat.gbuffer_pipeline)
                    : (skinned ? mat.skinned_pipeline : mat.pipeline);
        if (!Valid(want)) {
            if (gbuffer) ++stats.incompatible;
            else ++stats.invalid;
            continue;
        }
        if (want.v != bound_pipeline.v) {
            enc.SetPipeline(want);
            bound_pipeline = want;
            ++stats.pipeline_switches;
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
        {
        const Vec3 vd = Normalize(scene.camera.target - scene.camera.eye);
        u.viewDir = Vec4{vd.x, vd.y, vd.z, 0.0f};
    }
        u.invViewProj = inv_view_proj;
        u.lighting = Vec4{float(light_count),
                          float(env.specular_mips), 0.0f, 0.0f};
        u.ambientSky = Vec4{scene.ambientSky.x, scene.ambientSky.y,
                            scene.ambientSky.z, 0.0f};
        u.ambientGround = Vec4{scene.ambientGround.x, scene.ambientGround.y,
                               scene.ambientGround.z, 0.0f};
        u.emissive = Vec4{mat.emissive.x, mat.emissive.y, mat.emissive.z,
                          mat.normal_strength};

        // Sub-allocate out of this frame's slot instead of a per-draw setBytes.
        std::memcpy(uniform_map + offset, &u, sizeof(u));

        enc.SetVertexBuffer(gm.vb, 0, kVertexSlot);
        enc.SetVertexBuffer(uniforms, offset, kUniformSlot);
        enc.SetFragmentBuffer(uniforms, offset, kUniformSlot);
        enc.SetFragmentBuffer(lights, light_offset, kLightSlot);
        enc.SetFragmentBuffer(cascades, cascade_offset, kCascadeSlot);
        enc.SetFragmentTexture(mat.albedo, 0);
        enc.SetFragmentTexture(mat.roughness_map, 1);
        enc.SetFragmentTexture(mat.normal_map, 4);
        enc.SetFragmentTexture(mat.metallic_map, 8);
        enc.SetFragmentTexture(mat.emissive_map, 9);
        enc.SetFragmentTexture(mat.occlusion_map, 10);
        enc.SetFragmentSampler(shadow_sampler, 2);
        // CLUSTERS, or nothing. Binding stale bins is worse than binding none:
        // the shader's null check falls back to the full buffer, while a stale
        // index list points at lights that may no longer exist.
        if (bins_valid) {
            enc.SetFragmentBytes(&cluster_params, sizeof(GpuClusters),
                                 kClusterSlot);
            enc.SetFragmentBuffer(cluster_counts, 0, kClusterCountSlot);
            enc.SetFragmentBuffer(cluster_indices, 0, kClusterIndexSlot);
        }
        enc.SetFragmentTexture(shadow_tex, 2);
        // The local lights' atlas. Something has to be bound even when no light
        // has a tile, so the placeholder stands in — the shader guards on the
        // per-light flag rather than on the texture.
        enc.SetFragmentTexture(
            Valid(shadow_atlas) ? shadow_atlas : dummy_shadow, 3);
        enc.SetFragmentSampler(sampler, 0);
        // The environment probe. All three may be null handles, which binds
        // nothing -- and nothing is exactly what the shader's is_null_texture
        // check is looking for.
        enc.SetFragmentTexture(env.irradiance, 5);
        enc.SetFragmentTexture(env.specular, 6);
        enc.SetFragmentTexture(env.brdf_lut, 7);
        enc.SetFragmentSampler(env.cube_sampler, 1);
        if (skinned) {
            enc.SetVertexBuffer(gm.skin_vb, 0, kSkinSlot);
            enc.SetVertexBuffer(palettes, palette_offset, kPaletteSlot);
        }
        enc.DrawIndexedU32(gm.ib, gm.index_count);
        draw_order.push_back(item.index);
        ++stats.draws;
        if (item.transparent) ++stats.transparent_draws;
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

void Renderer::BuildHiZ(rhi::ComputeEncoder& enc, rhi::TextureId depth, int width,
                        int height) {
    Impl& im = *impl_;
    if (!Valid(depth) || width <= 0 || height <= 0) return;
    if (!im.hiz_tried) {
        im.hiz_tried = true;
        std::string err;
        im.hiz_copy = im.dev->CreateComputePipeline(ShaderSource(kCullSrc),
                                                    "cs_hiz_copy", err);
        im.hiz_reduce = im.dev->CreateComputePipeline(ShaderSource(kCullSrc),
                                                      "cs_hiz_reduce", err);
        if (!im.cull_error.empty() || (!Valid(im.hiz_copy) && im.cull_error.empty()))
            im.cull_error = err;
    }
    if (!Valid(im.hiz_copy) || !Valid(im.hiz_reduce)) return;

    if (width != im.hiz_width || height != im.hiz_height || !Valid(im.hiz)) {
        for (rhi::TextureId v : im.hiz_views)
            if (Valid(v)) im.dev->DestroyTexture(v);
        im.hiz_views.clear();
        if (Valid(im.hiz)) im.dev->DestroyTexture(im.hiz);
        int mips = 1;
        for (int w = width, h = height; w > 1 || h > 1; ++mips) {
            w = std::max(w / 2, 1);
            h = std::max(h / 2, 1);
        }
        // R32Float, not the depth format: a depth texture cannot be bound for
        // shader writes, and the pyramid is written by compute at every level.
        im.hiz = im.dev->CreateStorageTexture2D(width, height,
                                                rhi::Format::RGBA16Float, mips);
        if (!Valid(im.hiz)) return;
        im.hiz_width = width;
        im.hiz_height = height;
        im.hiz_mips = mips;
        for (int m = 0; m < mips; ++m)
            im.hiz_views.push_back(im.dev->CreateMipView(im.hiz, m));
    }

    EngUVec4 size{};
    size.x = std::uint32_t(width);
    size.y = std::uint32_t(height);
    enc.SetPipeline(im.hiz_copy);
    enc.SetTexture(depth, 0);
    enc.SetTexture(im.hiz_views[0], 1);
    enc.SetBytes(&size, sizeof(size), 0);
    enc.Dispatch2D(width, height);

    int sw = width, sh = height;
    for (int m = 1; m < im.hiz_mips; ++m) {
        const int dw = std::max(sw / 2, 1), dh = std::max(sh / 2, 1);
        EngUVec4 s2{};
        s2.x = std::uint32_t(dw);
        s2.y = std::uint32_t(dh);
        s2.z = std::uint32_t(sw);
        s2.w = std::uint32_t(sh);
        enc.SetPipeline(im.hiz_reduce);
        enc.SetTexture(im.hiz_views[std::size_t(m - 1)], 0);
        enc.SetTexture(im.hiz_views[std::size_t(m)], 1);
        enc.SetBytes(&s2, sizeof(s2), 0);
        enc.Dispatch2D(dw, dh);
        sw = dw;
        sh = dh;
    }
    im.hiz_built_frame = im.dev->FrameIndex();
}

void Renderer::SetOcclusionCulling(bool on) { impl_->occlusion_enabled = on; }
bool Renderer::OcclusionCulling() const { return impl_->occlusion_enabled; }
void Renderer::SetLodThresholds(Vec3 pixels) { impl_->lod_thresholds = pixels; }
Vec3 Renderer::LodThresholds() const { return impl_->lod_thresholds; }
rhi::TextureId Renderer::HiZ() const { return impl_->hiz; }

void Renderer::SetGrade(const ColorGrade& g) { impl_->grade = g; }
const ColorGrade& Renderer::Grade() const { return impl_->grade; }
void Renderer::SetExposureBuffer(rhi::BufferId b) { impl_->exposure = b; }

void Renderer::DrawComposite(rhi::Encoder& enc, rhi::TextureId src,
                             rhi::TextureId ao, rhi::TextureId bloom,
                             float bloom_strength, float vignette) {
    FrameUniforms u{};
    u.lighting = Vec4{0.0f, Valid(bloom) ? bloom_strength : 0.0f, vignette, 0.0f};
    const std::size_t offset = impl_->AllocUniform();
    if (offset == Impl::kNoSpace) return;
    std::memcpy(impl_->uniform_map + offset, &u, sizeof(u));

    const ColorGrade& g = impl_->grade;
    GradeParams gp{};
    gp.tone = Vec4{g.lift_all, g.gamma_all, g.gain_all, g.contrast_pivot};
    gp.lift = Vec4{g.lift.x, g.lift.y, g.lift.z, 0.0f};
    gp.gamma = Vec4{g.gamma.x, g.gamma.y, g.gamma.z, 0.0f};
    gp.gain = Vec4{g.gain.x, g.gain.y, g.gain.z, 0.0f};
    gp.look = Vec4{g.contrast, g.saturation, g.temperature, g.tint};

    enc.SetPipeline(impl_->composite);
    enc.SetCull(rhi::Cull::None, rhi::Winding::CounterClockwise);
    enc.SetFragmentBuffer(impl_->uniforms, offset, kUniformSlot);
    enc.SetFragmentBytes(&gp, sizeof(gp), 2);
    // A FIXED exposure of one when no buffer is bound. The shader reads a
    // buffer either way, so there is one code path rather than a branch on a
    // flag that could disagree with what was bound.
    enc.SetFragmentBuffer(Valid(impl_->exposure) ? impl_->exposure : impl_->unit_exposure,
                          0, 3);
    enc.SetFragmentTexture(src, 0);
    enc.SetFragmentTexture(Valid(ao) ? ao : impl_->white, 1);
    // The bloom slot needs SOMETHING bound. Black, not white: an absent bloom
    // has to add nothing, and the strength above is already zero.
    enc.SetFragmentTexture(Valid(bloom) ? bloom : impl_->black, 2);
    enc.SetFragmentSampler(impl_->clamp_sampler, 0);
    enc.Draw(3);  // one oversized triangle, generated from the vertex id
}

void Renderer::DrawDeferredLight(rhi::Encoder& enc, const Scene& scene,
                                 int width, int height,
                                 rhi::TextureId albedo_rough,
                                 rhi::TextureId normal_metal,
                                 rhi::TextureId depth,
                                 rhi::TextureId shadow_map) {
    if (width <= 0 || height <= 0) return;
    if (!Valid(albedo_rough) || !Valid(normal_metal) || !Valid(depth)) return;

    // The light list and the cascade block, exactly as the forward path builds
    // them -- the same buffers, the same layout, because the shader reading
    // them is the same shader.
    const std::size_t light_offset =
        std::size_t(impl_->dev->FrameSlot()) * impl_->light_slot_bytes;
    const int light_count =
        std::min(int(scene.lights.size()), int(ENG_MAX_LIGHTS));
    impl_->UploadLights(scene, light_offset, light_count);

    const std::size_t cascade_offset =
        std::size_t(impl_->dev->FrameSlot()) * impl_->cascade_slot_bytes;
    const float aspect = float(width) / float(height);
    impl_->UploadCascades(scene, cascade_offset, aspect);

    const bool shadows = scene.shadowExtent > 0.0f && Valid(shadow_map);
    const Mat4 viewProj = scene.camera.ViewProj(aspect);

    FrameUniforms u{};
    u.viewProj = viewProj;
    u.invViewProj = Inverse(viewProj);
    u.model = Mat4::Identity();
    u.tint = Vec4{1, 1, 1, 1};
    u.lightDir = scene.lightDir;
    u.lightColor = scene.lightColor;
    u.baseColor = Vec4{1, 1, 1, 1};
    u.lightViewProj = shadows ? scene.LightViewProj() : Mat4::Identity();
    // Roughness and metallic come from the G-buffer, not from here. The two
    // slots that still matter are the shadow flag and the section cut -- and
    // the cut has ALREADY been applied, by the G-buffer pass discarding those
    // fragments, so applying it again here would do nothing but cost a branch.
    u.surface = Vec4{1.0f, 0.0f, shadows ? 1.0f : 0.0f, 1e30f};
    u.eyePos = Vec4{scene.camera.eye.x, scene.camera.eye.y, scene.camera.eye.z,
                    1.0f};
    {
        const Vec3 vd = Normalize(scene.camera.target - scene.camera.eye);
        u.viewDir = Vec4{vd.x, vd.y, vd.z, 0.0f};
    }
    u.lighting = Vec4{float(light_count),
                      float(impl_->env.specular_mips), 0.0f, 0.0f};
    u.ambientSky = Vec4{scene.ambientSky.x, scene.ambientSky.y,
                        scene.ambientSky.z, 0.0f};
    u.ambientGround = Vec4{scene.ambientGround.x, scene.ambientGround.y,
                           scene.ambientGround.z, 0.0f};
    // No material here: a fullscreen lighting pass reads the G-buffer,
    // and emission was already added by the pass that wrote it.
    u.emissive = Vec4{0.0f, 0.0f, 0.0f, 1.0f};

    const std::size_t offset = impl_->AllocUniform();
    if (offset == Impl::kNoSpace) return;
    std::memcpy(impl_->uniform_map + offset, &u, sizeof(u));

    enc.SetPipeline(impl_->deferred_light);
    enc.SetCull(rhi::Cull::None, rhi::Winding::CounterClockwise);
    enc.SetFragmentBuffer(impl_->uniforms, offset, kUniformSlot);
    enc.SetFragmentBuffer(impl_->lights, light_offset, kLightSlot);
    enc.SetFragmentBuffer(impl_->cascades, cascade_offset, kCascadeSlot);
    enc.SetFragmentTexture(albedo_rough, 0);
    enc.SetFragmentTexture(normal_metal, 1);
    enc.SetFragmentTexture(shadows ? shadow_map : impl_->dummy_shadow, 2);
    enc.SetFragmentTexture(
        Valid(impl_->shadow_atlas) ? impl_->shadow_atlas : impl_->dummy_shadow, 3);
    enc.SetFragmentTexture(depth, 4);
    // Slot 0 reads the G-BUFFER here, one texel per pixel, so it clamps and is
    // isotropic like every other fullscreen pass. It used to be the material
    // sampler, which wraps -- shared with the shadow lookup, which is now on
    // slot 2 with a sampler of its own.
    enc.SetFragmentSampler(impl_->clamp_sampler, 0);
    enc.SetFragmentSampler(impl_->shadow_sampler, 2);
    // CLUSTERS, or nothing. Binding stale bins is worse than binding
    // none: the shader's null check falls back to the full buffer, while a
    // stale index list points at lights that may no longer exist.
    if (impl_->bins_valid) {
        enc.SetFragmentBytes(&impl_->cluster_params, sizeof(GpuClusters),
                             kClusterSlot);
        enc.SetFragmentBuffer(impl_->cluster_counts, 0, kClusterCountSlot);
        enc.SetFragmentBuffer(impl_->cluster_indices, 0, kClusterIndexSlot);
    }
    // The environment probe. All three may be null handles, which binds
    // nothing -- and nothing is exactly what the shader's is_null_texture
    // check is looking for.
    enc.SetFragmentTexture(impl_->env.irradiance, 5);
    enc.SetFragmentTexture(impl_->env.specular, 6);
    enc.SetFragmentTexture(impl_->env.brdf_lut, 7);
    enc.SetFragmentSampler(impl_->env.cube_sampler, 1);
    enc.Draw(3);
}

int Renderer::LastBatchCount() const { return impl_->live_batches; }
int Renderer::LastInstanceCount() const { return impl_->batched_instances; }

int Renderer::VisibleAfterCull() const {
    int total = 0;
    for (int i = 0; i < impl_->live_batches; ++i) {
        const auto* args = static_cast<const GpuDrawArgs*>(
            impl_->dev->MapBuffer(impl_->batches[std::size_t(i)].args));
        if (!args) continue;
        // EVERY LEVEL. Reading only the first would report a scene that had all
        // gone to level 2 as entirely culled, which is a very convincing way to
        // conclude the frustum test is broken.
        for (int l = 0; l < kMaxLods; ++l) total += int(args[l].instance_count);
    }
    return total;
}

int Renderer::VisibleAtLod(int lod) const {
    if (lod < 0 || lod >= kMaxLods) return 0;
    int total = 0;
    for (int i = 0; i < impl_->live_batches; ++i) {
        const auto* args = static_cast<const GpuDrawArgs*>(
            impl_->dev->MapBuffer(impl_->batches[std::size_t(i)].args));
        if (args) total += int(args[lod].instance_count);
    }
    return total;
}

long long Renderer::IndirectTriangles() const {
    long long total = 0;
    for (int i = 0; i < impl_->live_batches; ++i) {
        const auto* args = static_cast<const GpuDrawArgs*>(
            impl_->dev->MapBuffer(impl_->batches[std::size_t(i)].args));
        if (!args) continue;
        for (int l = 0; l < kMaxLods; ++l)
            total += static_cast<long long>(args[l].instance_count) *
                     static_cast<long long>(args[l].index_count) / 3;
    }
    return total;
}

int Renderer::CullScene(rhi::ComputeEncoder& enc, const Scene& scene, int width,
                        int height) {
    impl_->live_batches = 0;
    impl_->batched_instances = 0;
    if (width <= 0 || height <= 0 || scene.instances.empty()) return 0;

    if (!impl_->cull_tried) {
        impl_->cull_tried = true;
        std::string err;
        impl_->cull_pipeline = impl_->dev->CreateComputePipeline(
            ShaderSource(kCullSrc), "cs_cull", err);
        if (!Valid(impl_->cull_pipeline)) impl_->cull_error = err;
        impl_->cull_finish_pipeline = impl_->dev->CreateComputePipeline(
            ShaderSource(kCullSrc), "cs_cull_finish", err);
        if (!Valid(impl_->cull_finish_pipeline) && impl_->cull_error.empty())
            impl_->cull_error = err;
    }
    if (!Valid(impl_->cull_pipeline) || !Valid(impl_->cull_finish_pipeline))
        return 0;

    // --- group by mesh and material ------------------------------------------
    // A batch is one draw, so everything in it must share the geometry AND the
    // pipeline. Skinned and transparent instances are excluded here rather than
    // filtered later: a skinned mesh needs a palette per instance, and a
    // transparent one needs an order this path cannot provide.
    std::vector<std::vector<const Instance*>> groups;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> keys;
    for (const Instance& inst : scene.instances) {
        if (!Valid(inst.mesh) || inst.mesh.v >= impl_->meshes.size()) continue;
        if (!Valid(inst.material) || inst.material.v >= impl_->materials.size())
            continue;
        const GpuMesh& gm = impl_->meshes[inst.mesh.v];
        const GpuMaterial& mat = impl_->materials[inst.material.v];
        if (gm.joint_count > 0 || mat.transparent || !mat.depth_test) continue;

        const auto key = std::make_pair(inst.mesh.v, inst.material.v);
        std::size_t at = keys.size();
        for (std::size_t i = 0; i < keys.size(); ++i)
            if (keys[i] == key) { at = i; break; }
        if (at == keys.size()) {
            keys.push_back(key);
            groups.emplace_back();
        }
        groups[at].push_back(&inst);
    }
    if (groups.empty()) return 0;

    const Mat4 viewProj = scene.camera.ViewProj(float(width) / float(height));
    const Frustum frustum = Frustum::FromViewProj(viewProj);
    const Vec3 lod_thresholds = impl_->lod_thresholds;
    // Only when a pyramid was actually built this frame. A stale one from a
    // different camera would cull things that are plainly visible, and the
    // symptom -- objects missing near the edges of occluders -- looks like a
    // frustum bug rather than a stale texture.
    const bool use_occlusion = impl_->occlusion_enabled && Valid(impl_->hiz) &&
                               impl_->hiz_built_frame == impl_->dev->FrameIndex();
    impl_->last_aspect = float(width) / float(height);
    impl_->inv_view_proj = Inverse(viewProj);

    // Mirrors cull.metal's CullParams.
    struct CullParams {
        Mat4 viewProj;
        Vec4 planes[6];
        Vec4 eye;
        Vec4 screen;
        Vec4 lod_px;
        EngUVec4 counts;
        EngUVec4 index_counts;
    };
    static_assert(sizeof(CullParams) == 240, "CullParams layout drifted");

    // PIXELS PER WORLD UNIT AT ONE METRE. The projected radius of a sphere of
    // radius r at distance d is r * this / d, which is what the level choice
    // and the occlusion footprint both need.
    const float half_height =
        scene.camera.projection == Projection::Orthographic
            ? float(height) * 0.5f / std::max(scene.camera.orthoHeight, 1e-4f)
            : float(height) * 0.5f / std::tan(scene.camera.fovY * 0.5f);

    for (std::size_t g = 0; g < groups.size(); ++g) {
        if (impl_->batches.size() <= g) impl_->batches.emplace_back();
        Impl::Batch& b = impl_->batches[g];
        b.mesh = MeshHandle{keys[g].first};
        b.material = MaterialHandle{keys[g].second};
        b.count = int(groups[g].size());

        const std::size_t need = groups[g].size();
        if (b.capacity < need || !Valid(b.instances)) {
            // Grown, not resized to fit. A scene whose object count creeps up
            // by one a frame would otherwise reallocate every frame forever.
            const std::size_t cap = std::max<std::size_t>(need * 2, 64);
            b.instances = impl_->dev->CreateDynamicBuffer(sizeof(GpuInstance) * cap);
            // ONE REGION PER LEVEL. An instance can land in any of them, so
            // each has to be sized for the whole group -- there is no
            // distribution to exploit, because the camera decides it.
            b.visible =
                impl_->dev->CreateStorageBuffer(sizeof(GpuInstance) * cap * kMaxLods);
            b.counter = impl_->dev->CreateStorageBuffer(sizeof(std::uint32_t) * kMaxLods);
            b.args = impl_->dev->CreateStorageBuffer(sizeof(GpuDrawArgs) * kMaxLods);
            b.capacity = cap;
            if (!Valid(b.instances) || !Valid(b.visible) || !Valid(b.counter) ||
                !Valid(b.args))
                return 0;
        }

        auto* dst = static_cast<GpuInstance*>(impl_->dev->MapBuffer(b.instances));
        if (!dst) return 0;
        const GpuMesh& gm = impl_->meshes[keys[g].first];
        for (std::size_t i = 0; i < need; ++i) {
            const Instance& inst = *groups[g][i];
            dst[i].model = inst.model;
            dst[i].tint = inst.tint;
            // OBJECT-space bounds. The kernel transforms them, because the
            // transform is already there and doing it here would mean the CPU
            // touching every object for the one number the GPU could derive.
            dst[i].bounds = Vec4{gm.bounds.center.x, gm.bounds.center.y,
                                 gm.bounds.center.z, gm.bounds.radius};
        }

        // EVERY counter must start at zero. They are storage buffers, so they
        // hold last frame's totals otherwise, and the draw would ask for twice
        // as many instances as the buffer has.
        auto* zero = static_cast<std::uint32_t*>(impl_->dev->MapBuffer(b.counter));
        if (zero)
            for (int l = 0; l < kMaxLods; ++l) zero[l] = 0;

        CullParams params{};
        params.viewProj = viewProj;
        for (int i = 0; i < 6; ++i)
            params.planes[i] = Vec4{frustum.planes[i].n.x, frustum.planes[i].n.y,
                                    frustum.planes[i].n.z, frustum.planes[i].d};
        params.eye = Vec4{scene.camera.eye.x, scene.camera.eye.y,
                          scene.camera.eye.z, scene.camera.nearZ};
        params.screen = Vec4{float(width), float(height), half_height,
                             float(impl_->hiz_mips)};
        params.lod_px = Vec4{lod_thresholds.x, lod_thresholds.y, lod_thresholds.z,
                             0.0f};
        params.counts.x = std::uint32_t(need);
        params.counts.y = std::uint32_t(gm.lod_count);
        params.counts.z = std::uint32_t(b.capacity);
        params.counts.w = use_occlusion ? 1u : 0u;
        std::uint32_t* index_counts = &params.index_counts.x;
        for (int l = 0; l < kMaxLods; ++l)
            index_counts[l] = std::uint32_t(
                l < gm.lod_count ? gm.lods[l].index_count : 0);

        enc.SetPipeline(impl_->cull_pipeline);
        enc.SetBuffer(b.instances, 0, 0);
        enc.SetBytes(&params, sizeof(params), 1);
        enc.SetBuffer(b.visible, 0, 2);
        enc.SetBuffer(b.counter, 0, 3);
        enc.SetBuffer(b.args, 0, 4);
        if (use_occlusion) enc.SetTexture(impl_->hiz, 0);
        enc.Dispatch(int(need));

        // A SECOND dispatch to publish the counts. Writing them from inside the
        // cull kernel would race with the threads still counting -- the last
        // thread to increment is not the last thread to run.
        enc.SetPipeline(impl_->cull_finish_pipeline);
        enc.SetBuffer(b.counter, 0, 0);
        enc.SetBuffer(b.args, 0, 1);
        enc.SetBytes(&params, sizeof(params), 2);
        enc.Dispatch(1);

        impl_->batched_instances += int(need);
    }
    impl_->live_batches = int(groups.size());
    return impl_->live_batches;
}

void Renderer::DrawSceneIndirect(rhi::Encoder& enc, const Scene& scene, int width,
                                 int height, rhi::TextureId shadow_map) {
    impl_->stats = RenderStats{};
    impl_->stats.submitted = impl_->batched_instances;
    impl_->draw_order.clear();
    if (impl_->live_batches == 0) return;

    const std::size_t light_offset =
        std::size_t(impl_->dev->FrameSlot()) * impl_->light_slot_bytes;
    const int light_count =
        std::min(int(scene.lights.size()), int(ENG_MAX_LIGHTS));
    impl_->UploadLights(scene, light_offset, light_count);
    const std::size_t cascade_offset =
        std::size_t(impl_->dev->FrameSlot()) * impl_->cascade_slot_bytes;
    impl_->UploadCascades(scene, cascade_offset, impl_->last_aspect);

    const bool shadows = scene.shadowExtent > 0.0f && Valid(shadow_map);
    const rhi::TextureId shadow_tex = shadows ? shadow_map : impl_->dummy_shadow;
    const Mat4 viewProj = scene.camera.ViewProj(float(width) / float(height));

    for (int i = 0; i < impl_->live_batches; ++i) {
        const Impl::Batch& b = impl_->batches[std::size_t(i)];
        const GpuMesh& gm = impl_->meshes[b.mesh.v];
        const GpuMaterial& mat = impl_->materials[b.material.v];
        if (!Valid(mat.instanced_pipeline)) { ++impl_->stats.invalid; continue; }

        const std::size_t offset = impl_->AllocUniform();
        if (offset == Impl::kNoSpace) { ++impl_->stats.overflowed; break; }

        FrameUniforms u{};
        u.viewProj = viewProj;
        // model and tint come from the instance buffer; leaving them here would
        // be two values that look authoritative and are never read.
        u.model = Mat4::Identity();
        u.tint = Vec4{1, 1, 1, 1};
        u.invViewProj = impl_->inv_view_proj;
        u.lightDir = scene.lightDir;
        u.lightColor = scene.lightColor;
        u.baseColor = mat.base_color;
        u.lightViewProj = shadows ? scene.LightViewProj() : Mat4::Identity();
        u.surface = Vec4{mat.roughness, mat.metallic, shadows ? 1.0f : 0.0f,
                         scene.clipY};
        u.eyePos = Vec4{scene.camera.eye.x, scene.camera.eye.y, scene.camera.eye.z,
                        1.0f};
        {
        const Vec3 vd = Normalize(scene.camera.target - scene.camera.eye);
        u.viewDir = Vec4{vd.x, vd.y, vd.z, 0.0f};
    }
        u.lighting = Vec4{float(light_count),
                          float(impl_->env.specular_mips), 0.0f, 0.0f};
        u.ambientSky = Vec4{scene.ambientSky.x, scene.ambientSky.y,
                            scene.ambientSky.z, 0.0f};
        u.ambientGround = Vec4{scene.ambientGround.x, scene.ambientGround.y,
                               scene.ambientGround.z, 0.0f};
        u.emissive = Vec4{mat.emissive.x, mat.emissive.y, mat.emissive.z,
                          mat.normal_strength};
        std::memcpy(impl_->uniform_map + offset, &u, sizeof(u));

        enc.SetPipeline(mat.instanced_pipeline);
        enc.SetCull(mat.cull, rhi::Winding::CounterClockwise);
        enc.SetVertexBuffer(impl_->uniforms, offset, kUniformSlot);
        enc.SetFragmentBuffer(impl_->uniforms, offset, kUniformSlot);
        enc.SetFragmentBuffer(impl_->lights, light_offset, kLightSlot);
        enc.SetFragmentBuffer(impl_->cascades, cascade_offset, kCascadeSlot);
        enc.SetFragmentTexture(mat.albedo, 0);
        enc.SetFragmentTexture(mat.roughness_map, 1);
        enc.SetFragmentTexture(mat.normal_map, 4);
        enc.SetFragmentTexture(mat.metallic_map, 8);
        enc.SetFragmentTexture(mat.emissive_map, 9);
        enc.SetFragmentTexture(mat.occlusion_map, 10);
        enc.SetFragmentSampler(impl_->shadow_sampler, 2);
        // CLUSTERS, or nothing. Binding stale bins is worse than binding
        // none: the shader's null check falls back to the full buffer, while a
        // stale index list points at lights that may no longer exist.
        if (impl_->bins_valid) {
            enc.SetFragmentBytes(&impl_->cluster_params, sizeof(GpuClusters),
                                 kClusterSlot);
            enc.SetFragmentBuffer(impl_->cluster_counts, 0, kClusterCountSlot);
            enc.SetFragmentBuffer(impl_->cluster_indices, 0, kClusterIndexSlot);
        }
        enc.SetFragmentTexture(shadow_tex, 2);
        enc.SetFragmentTexture(
            Valid(impl_->shadow_atlas) ? impl_->shadow_atlas : impl_->dummy_shadow, 3);
        enc.SetFragmentSampler(impl_->sampler, 0);
        // The environment probe. All three may be null handles, which binds
        // nothing -- and nothing is exactly what the shader's is_null_texture
        // check is looking for.
        enc.SetFragmentTexture(impl_->env.irradiance, 5);
        enc.SetFragmentTexture(impl_->env.specular, 6);
        enc.SetFragmentTexture(impl_->env.brdf_lut, 7);
        enc.SetFragmentSampler(impl_->env.cube_sampler, 1);

        // ONE INDIRECT DRAW PER LEVEL, and the instance count of each comes out
        // of the buffer the cull wrote -- so the CPU still never learns how the
        // instances were distributed between them.
        //
        // A level with no survivors costs a draw call with a zero instance
        // count, which the hardware discards immediately. Skipping it would
        // need the count, and asking for the count is the readback this whole
        // path exists to avoid.
        for (int l = 0; l < gm.lod_count; ++l) {
            const GpuMesh::Lod& lod = gm.lods[l];
            if (!Valid(lod.vb) || !Valid(lod.ib)) continue;
            enc.SetVertexBuffer(lod.vb, 0, kVertexSlot);
            // The SURVIVORS of this level, not the instances that were offered.
            // Each level owns a fixed-stride region of one buffer.
            enc.SetVertexBuffer(b.visible, sizeof(GpuInstance) * b.capacity * std::size_t(l),
                                kInstanceSlot);
            enc.DrawIndexedIndirectU32(lod.ib, b.args, sizeof(GpuDrawArgs) * std::size_t(l));
            ++impl_->stats.draws;
        }
        ++impl_->stats.pipeline_switches;
    }
}

int Renderer::BlasBuilds() const { return impl_->blas_builds; }

rhi::BufferId Renderer::PosedVertices(int instance_index) const {
    if (instance_index < 0 ||
        std::size_t(instance_index) >= impl_->posed_of_instance.size())
        return {};
    return impl_->posed_of_instance[std::size_t(instance_index)];
}

const std::string& Renderer::SkinError() const { return impl_->skin_error; }

int Renderer::SkinToBuffers(rhi::ComputeEncoder& enc, const Scene& scene) {
    impl_->posed_of_instance.assign(scene.instances.size(), rhi::BufferId{});
    impl_->posed_used = 0;

    if (!impl_->skin_pipeline_tried) {
        impl_->skin_pipeline_tried = true;
        std::string err;
        impl_->skin_pipeline = impl_->dev->CreateComputePipeline(
            ShaderSource(kSkinningSrc), "cs_skin", err);
        // KEPT, not discarded. This pipeline is built lazily, so its failure
        // arrives in the middle of a frame with nowhere to return an error --
        // and "SkinToBuffers posed nothing" is indistinguishable from "the
        // scene has no skinned meshes". It cost an hour once already.
        if (!Valid(impl_->skin_pipeline)) impl_->skin_error = err;
    }
    if (!Valid(impl_->skin_pipeline)) return 0;

    int posed = 0;
    bool pipeline_set = false;
    for (std::size_t idx = 0; idx < scene.instances.size(); ++idx) {
        const Instance& inst = scene.instances[idx];
        if (!Valid(inst.mesh) || inst.mesh.v >= impl_->meshes.size()) continue;
        const GpuMesh& gm = impl_->meshes[inst.mesh.v];
        if (gm.joint_count <= 0 || !Valid(gm.skin_vb)) continue;
        if (inst.palette < 0 ||
            std::size_t(inst.palette) + std::size_t(gm.joint_count) >
                scene.joint_matrices.size())
            continue;

        // A pooled output buffer, reused frame to frame. Allocating one per
        // skinned instance per frame would work and would also leak a vertex
        // buffer's worth of memory every frame until the driver gave up.
        const std::size_t bytes = sizeof(VertexIn) * gm.vertex_count;
        rhi::BufferId out;
        if (impl_->posed_used < impl_->posed_pool.size() &&
            impl_->posed_pool_bytes[impl_->posed_used] >= bytes) {
            out = impl_->posed_pool[impl_->posed_used];
        } else {
            out = impl_->dev->CreateStorageBuffer(bytes);
            if (!Valid(out)) continue;
            if (impl_->posed_used < impl_->posed_pool.size()) {
                impl_->posed_pool[impl_->posed_used] = out;
                impl_->posed_pool_bytes[impl_->posed_used] = bytes;
            } else {
                impl_->posed_pool.push_back(out);
                impl_->posed_pool_bytes.push_back(bytes);
            }
        }
        ++impl_->posed_used;

        // The palette, into the same ring the draw path uses.
        const std::size_t palette_offset = impl_->AllocPalette();
        if (palette_offset == Impl::kNoSpace) continue;
        std::memcpy(impl_->palette_map + palette_offset,
                    scene.joint_matrices.data() + inst.palette,
                    sizeof(Mat4) * std::size_t(gm.joint_count));

        if (!pipeline_set) {
            enc.SetPipeline(impl_->skin_pipeline);
            pipeline_set = true;
        }
        struct SkinParams {
            std::uint32_t vertex_count;
            std::uint32_t joint_count;
            std::uint32_t pad0 = 0, pad1 = 0;
        } params{std::uint32_t(gm.vertex_count), std::uint32_t(gm.joint_count)};

        enc.SetBuffer(gm.vb, 0, 0);
        enc.SetBuffer(gm.skin_vb, 0, 1);
        enc.SetBuffer(impl_->palettes, palette_offset, 2);
        enc.SetBytes(&params, sizeof(params), 3);
        enc.SetBuffer(out, 0, 4);
        enc.Dispatch(int(gm.vertex_count));

        impl_->posed_of_instance[idx] = out;
        ++posed;
    }
    return posed;
}

bool Renderer::RaytracingAvailable() const {
    return impl_->dev->SupportsRaytracing();
}

bool Renderer::BuildSceneAccel(const Scene& scene, std::string& error) {
    if (!impl_->dev->SupportsRaytracing()) {
        error = "this device has no hardware ray tracing";
        return false;
    }

    impl_->mesh_blas.resize(impl_->meshes.size());
    std::vector<rhi::Device::AccelInstance> instances;
    instances.reserve(scene.instances.size());

    for (const Instance& inst : scene.instances) {
        if (!Valid(inst.mesh) || inst.mesh.v >= impl_->meshes.size()) continue;
        const GpuMesh& gm = impl_->meshes[inst.mesh.v];
        // A SKINNED mesh uses the buffer SkinToBuffers posed, not the one on
        // the mesh -- that one is the bind pose, and a structure built from it
        // casts the shadow of a character standing still while the character
        // walks.
        //
        // It also cannot be shared between instances the way a static mesh's
        // is: two characters of the same mesh in different poses are different
        // geometry. So a posed instance gets its own structure, rebuilt each
        // time the pose changes, and the deduplication below does not apply to
        // it. That is the real cost of animated ray tracing and there is no
        // version of it that is cheaper.
        const rhi::BufferId posed =
            gm.joint_count > 0 ? PosedVertices(int(&inst - scene.instances.data()))
                               : rhi::BufferId{};
        if (gm.joint_count > 0 && !Valid(posed)) {
            // Skinned, but nothing posed it this frame. Skipping is right --
            // the alternative is silently using the bind pose, which is the
            // wrong picture rather than a missing one.
            continue;
        }

        if (Valid(posed)) {
            const rhi::AccelId blas = impl_->dev->CreateBlas(
                posed, int(sizeof(VertexIn)), gm.ib, int(gm.index_count), error);
            if (!Valid(blas)) return false;
            ++impl_->blas_builds;
            rhi::Device::AccelInstance ai;
            ai.blas = blas;
            std::memcpy(ai.transform, &inst.model.col[0].x, sizeof(float) * 16);
            instances.push_back(ai);
            continue;
        }

        rhi::AccelId& blas = impl_->mesh_blas[inst.mesh.v];
        if (!Valid(blas)) {
            // Built ONCE per mesh and reused by every instance of it. That is
            // the entire reason for a two-level structure: a thousand crates
            // build one BVH and a thousand transforms, not a thousand BVHs.
            blas = impl_->dev->CreateBlas(gm.vb, int(sizeof(VertexIn)), gm.ib,
                                          int(gm.index_count), error);
            if (!Valid(blas)) return false;
            ++impl_->blas_builds;
        }
        rhi::Device::AccelInstance ai;
        ai.blas = blas;
        std::memcpy(ai.transform, &inst.model.col[0].x, sizeof(float) * 16);
        instances.push_back(ai);
    }

    if (instances.empty()) {
        error = "no ray-traceable geometry in the scene";
        return false;
    }
    const rhi::AccelId tlas = impl_->dev->CreateTlas(instances, error);
    if (!Valid(tlas)) return false;
    impl_->scene_tlas = tlas;
    return true;
}

void Renderer::DrawRayShadows(rhi::Encoder& enc, const Scene& scene, int width,
                              int height, rhi::TextureId depth,
                              rhi::TextureId normals) {
    if (width <= 0 || height <= 0) return;
    if (!Valid(impl_->scene_tlas) || !Valid(depth) || !Valid(normals)) return;

    if (!impl_->ray_shadow_tried) {
        impl_->ray_shadow_tried = true;
        std::string err;
        impl_->ray_shadow =
            impl_->GetOrCreatePipeline(Shading::RayShadow, false, false, err);
    }
    if (!Valid(impl_->ray_shadow)) return;

    const float aspect = float(width) / float(height);
    const Mat4 viewProj = scene.camera.ViewProj(aspect);
    FrameUniforms u{};
    u.viewProj = viewProj;
    u.invViewProj = Inverse(viewProj);
    u.lightDir = scene.lightDir;
    // The ray's maximum length. A directional light has no position, so without
    // a bound the ray runs to infinity and every pixel pays for traversing the
    // empty half of the BVH.
    u.ssao = Vec4{0.0f, 0.0f, 0.0f, scene.shadowDistance > 0.0f
                                        ? scene.shadowDistance * 2.0f
                                        : 200.0f};

    const std::size_t offset = impl_->AllocUniform();
    if (offset == Impl::kNoSpace) return;
    std::memcpy(impl_->uniform_map + offset, &u, sizeof(u));

    enc.SetPipeline(impl_->ray_shadow);
    enc.SetCull(rhi::Cull::None, rhi::Winding::CounterClockwise);
    enc.SetFragmentBuffer(impl_->uniforms, offset, kUniformSlot);
    enc.SetFragmentTexture(depth, 0);
    enc.SetFragmentTexture(normals, 1);
    enc.SetFragmentSampler(impl_->clamp_sampler, 0);
    enc.SetFragmentAccel(impl_->scene_tlas, 4);
    enc.Draw(3);
}

void Renderer::DrawBloomBright(rhi::Encoder& enc, rhi::TextureId src,
                               float threshold, float knee) {
    if (!Valid(src)) return;
    FrameUniforms u{};
    u.ssao = Vec4{threshold, knee, 0.0f, 0.0f};
    const std::size_t offset = impl_->AllocUniform();
    if (offset == Impl::kNoSpace) return;
    std::memcpy(impl_->uniform_map + offset, &u, sizeof(u));

    enc.SetPipeline(impl_->bloom_bright);
    enc.SetCull(rhi::Cull::None, rhi::Winding::CounterClockwise);
    enc.SetFragmentBuffer(impl_->uniforms, offset, kUniformSlot);
    enc.SetFragmentTexture(src, 0);
    enc.SetFragmentSampler(impl_->clamp_sampler, 0);
    enc.Draw(3);
}

void Renderer::DrawBloomBlur(rhi::Encoder& enc, rhi::TextureId src,
                             float texel_x, float texel_y) {
    if (!Valid(src)) return;
    FrameUniforms u{};
    u.ssao = Vec4{0.0f, 0.0f, texel_x, texel_y};
    const std::size_t offset = impl_->AllocUniform();
    if (offset == Impl::kNoSpace) return;
    std::memcpy(impl_->uniform_map + offset, &u, sizeof(u));

    enc.SetPipeline(impl_->bloom_blur);
    enc.SetCull(rhi::Cull::None, rhi::Winding::CounterClockwise);
    enc.SetFragmentBuffer(impl_->uniforms, offset, kUniformSlot);
    enc.SetFragmentTexture(src, 0);
    enc.SetFragmentSampler(impl_->clamp_sampler, 0);
    enc.Draw(3);
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
    enc.SetFragmentSampler(impl_->clamp_sampler, 0);
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

    // TWO passes, because the tone map now lives in the composite. The scene
    // goes into a half-float target and is mapped down on the way out — the
    // same shape as every windowed app, rather than a shortcut that would need
    // its own version of the shading.
    rhi::PassDesc pass;
    pass.color = dev->CreateRenderTarget(width, height, Renderer::kSceneFormat);
    if (want_depth) pass.depth = dev->CreateDepthTarget(width, height);
    const rhi::TextureId readable = dev->CreateRenderTarget(
        width, height, rhi::Format::RGBA8Unorm, /*cpu_readable=*/true);
    if (!Valid(pass.color) || !Valid(readable) ||
        (want_depth && !Valid(pass.depth))) {
        error = "failed to create offscreen render targets";
        return {};
    }
    for (int i = 0; i < 4; ++i) pass.clear_color[i] = kClearColor[i];
    pass.clear_depth = 0.0f;  // reversed-Z: 0 is the far plane

    dev->BeginFrame();
    {
        rhi::Encoder enc = dev->BeginPass(pass);
        draw(*renderer, enc);
        dev->EndPass();
    }
    {
        rhi::PassDesc resolve;
        resolve.color = readable;
        rhi::Encoder enc = dev->BeginPass(resolve);
        // No vignette. This path exists to measure what the renderer drew, and
        // darkened corners are a look rather than a result.
        renderer->DrawComposite(enc, pass.color, {}, {}, 0.0f, /*vignette=*/0.0f);
        dev->EndPass();
    }
    if (!dev->CommitAndWait(error)) return {};

    Image img;
    img.width = width;
    img.height = height;
    img.rgba.resize(std::size_t(width) * std::size_t(height) * 4);
    if (!dev->ReadPixels(readable, width, height, img.rgba)) {
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

void Renderer::SetEnvironment(const EnvironmentBindings& e) { impl_->env = e; }
void Renderer::ClearEnvironment() { impl_->env = EnvironmentBindings{}; }
bool Renderer::HasEnvironment() const { return impl_->env.Valid(); }

}  // namespace eng
