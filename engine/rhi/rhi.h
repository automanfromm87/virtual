// Pure C++20. The Render Hardware Interface.
//
// This is the ONLY layer that knows which graphics API we are on. Nothing in
// this header names Metal, and nothing above it should ever have to. The test
// of whether the boundary is real: engine/render is a cc_library, not an
// objc_library — it compiles as plain C++ with no Apple frameworks at all.
//
// Deliberately thin. An RHI abstracts DEVICES, RESOURCES and SUBMISSION. It
// must not know what a material, a light or a shadow is; those are the
// renderer's vocabulary, and putting them here is how an RHI rots.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace eng::rhi {

enum class Format : std::uint8_t {
    RGBA8Unorm,
    BGRA8Unorm,
    // Half-float HDR. What a scene target has to be if anything downstream
    // wants to know how bright a pixel REALLY was: an 8-bit target clamps at
    // one, so a lamp and a sheet of white paper arrive at the next pass
    // indistinguishable, and a bloom built on that glows off the paper.
    RGBA16Float,
    Depth32Float,
    // Two half-floats. The BRDF integration lookup table is exactly this: a
    // scale and a bias, both in 0..1, and nothing else. RGBA16Float would work
    // and would waste half the bandwidth of a texture sampled once per pixel.
    RG16Float,
    // sRGB. The SAME bytes as RGBA8Unorm, decoded to linear by the sampler and
    // encoded back on write. Not a convenience: an 8-bit albedo texture is
    // authored in sRGB -- that is what every paint program and every camera
    // produces -- and reading it as linear makes midtones roughly twice as
    // bright as they should be. It also makes mip generation correct, because
    // the hardware averages after decoding rather than averaging gamma-encoded
    // bytes, which darkens every mip level.
    RGBA8Srgb,
    // BLOCK-COMPRESSED, decoded by the sampler at no cost. A 4x4 tile in 8
    // bytes (BC1) or 16 (BC3, BC5) against the 64 an RGBA8 tile costs.
    //
    // These cannot be render targets and their mip chains cannot be generated
    // on the GPU -- generateMipmaps has nothing to write through. Every level
    // is compressed on the CPU and uploaded, which is what
    // CreateTexture2DCompressed exists for.
    BC1, BC1Srgb,   // RGB
    BC3, BC3Srgb,   // RGBA
    BC5,            // two channels; a normal map's x and y
    // Full float, for the places where half is measurably not enough. An
    // equirectangular sky holds a sun at tens of thousands of nits, and half
    // saturates at 65504 -- close enough to matter, and the failure is a sun
    // that is a flat disc instead of a gradient.
    RGBA32Float,
};
enum class Cull : std::uint8_t { None, Back, Front };
enum class Winding : std::uint8_t { Clockwise, CounterClockwise };
// GreaterEqual is not a rounding-out of the enum -- it is the only compare a
// reversed-Z SKY can use. The sky is drawn at the far plane, which is depth 0,
// against a buffer cleared to exactly 0, and `0 > 0` is false: with Greater the
// sky is discarded on precisely the pixels it exists to fill.
enum class Compare : std::uint8_t { Never, Less, Greater, GreaterEqual, Always };
enum class Filter : std::uint8_t { Nearest, Linear };
enum class Wrap : std::uint8_t { Clamp, Repeat };

// Opaque handles. 0 is always the null handle — never a valid resource.
struct BufferId {
    std::uint32_t v = 0;
};
struct TextureId {
    std::uint32_t v = 0;
};
// Comparable, because "is this the same target as the one that pass wrote"
// is a question the render graph and its tests keep needing to ask.
inline bool operator==(TextureId a, TextureId b) { return a.v == b.v; }
inline bool operator!=(TextureId a, TextureId b) { return a.v != b.v; }
struct PipelineId {
    std::uint32_t v = 0;
};
struct SamplerId {
    std::uint32_t v = 0;
};
// A ray tracing acceleration structure -- either the per-mesh BVH over
// triangles, or the top-level one over instances of those.
struct AccelId {
    std::uint32_t v = 0;
};
// A compute pipeline. Separate from PipelineId because the two cannot be
// substituted for one another anywhere -- a compute pipeline has no
// attachments, no blend state and no vertex stage, and a single handle type
// would make binding the wrong one a runtime surprise rather than a compile
// error.
struct ComputePipelineId {
    std::uint32_t v = 0;
};
inline bool Valid(BufferId h) { return h.v != 0; }
inline bool Valid(TextureId h) { return h.v != 0; }
inline bool Valid(PipelineId h) { return h.v != 0; }
inline bool Valid(SamplerId h) { return h.v != 0; }
inline bool Valid(AccelId h) { return h.v != 0; }
inline bool Valid(ComputePipelineId h) { return h.v != 0; }

enum class Blend : std::uint8_t {
    None,
    Alpha,
    Additive,
    // WEIGHTED-BLENDED ORDER-INDEPENDENT TRANSPARENCY, the accumulation half.
    //
    // Two attachments with different blend functions: attachment 0 sums
    // weighted premultiplied colour (src + dst), attachment 1 multiplies down
    // the remaining visibility (dst * (1 - src)). Both operations commute, so
    // the result does not depend on the order the surfaces were drawn -- which
    // is the whole point, because sorting by object breaks the moment two
    // transparent objects intersect and there is no order that is right for
    // every pixel of them.
    OitAccumulate,
};

struct PipelineDesc {
    // Shader source in the backend's own language. The RHI compiles it but
    // does not interpret it — choosing what the shader SAYS is the renderer's
    // job, which is why this is a string and not a struct of lighting options.
    std::string source;
    std::string vertex_fn;
    std::string fragment_fn;
    Format color = Format::BGRA8Unorm;
    // MULTIPLE RENDER TARGETS. `color` is attachment 0; these are 1..n, and an
    // empty list is the ordinary single-target case. A deferred G-buffer is the
    // reason they exist: writing albedo, normal and material in one pass over
    // the geometry rather than three.
    //
    // The formats must match the pass's attachments exactly. Metal validates
    // that -- one of the few format errors it does not let through silently.
    std::vector<Format> extra_colors;
    bool depth = false;  // attach + test depth
    // Depth-only: no fragment shader and no colour attachment. That is the
    // whole shadow pass — position out, depth written, nothing shaded.
    bool depth_only = false;
    // How the fragment combines with what is already there. Normally paired
    // with depth_write = false: a translucent surface must not stop what is
    // behind it from drawing.
    //
    //   Alpha     src*a + dst*(1-a). Glass, smoke, anything that OCCLUDES.
    //             Order-dependent, so it needs sorting back to front.
    //   Additive  src*a + dst. Fire, sparks, magic -- anything that only ever
    //             adds light. Order-INDEPENDENT, because addition commutes,
    //             which is why a particle system that can use it should:
    //             ten thousand sparks need no sort at all.
    Blend blend = Blend::None;
    // Per-attachment blending for OitAccumulate: attachment 0 adds and
    // attachment 1 multiplies. Metal takes a blend state per attachment, so
    // this is a property of the pipeline rather than of the encoder.
    // VERTEX AMPLIFICATION: how many views one draw produces. 1 is an ordinary
    // pipeline; 2 is stereo.
    //
    // The vertex stage runs ONCE and emits `amplification` copies of its
    // output, each tagged with a render-target array index. That is the whole
    // saving: the vertex work, the culling, the state changes and the draw call
    // are all paid once for both eyes, where two passes pay for everything
    // twice and the second eye's geometry is identical to the first's.
    //
    // Needs a layered render target -- a 2D array with one slice per view.
    int amplification = 1;
    // Multisample count. Must match the attachments the pipeline will be used
    // with — a pipeline built for one sample cannot draw into a four-sample
    // target, and Metal rejects that outright rather than silently aliasing.
    int samples = 1;
    Compare depth_compare = Compare::Greater;  // reversed-Z default
    bool depth_write = true;
};

// A MESH-SHADER pipeline: an object stage, a mesh stage and a fragment stage.
//
// The object stage runs one threadgroup per meshlet, decides whether the
// meshlet is visible, and launches zero or one mesh threadgroups accordingly.
// That is the whole point: a culled meshlet costs one object threadgroup and
// nothing else -- no vertex fetch, no transform, no rasterisation, and no
// indirect-draw buffer to write and read back.
struct MeshPipelineDesc {
    std::string source;
    std::string object_fn;
    std::string mesh_fn;
    std::string fragment_fn;
    Format color = Format::BGRA8Unorm;
    std::vector<Format> extra_colors;
    bool depth = false;
    bool depth_write = true;
    Compare depth_compare = Compare::Greater;
    int samples = 1;
    // Bytes the object stage may pass to the mesh stage. It is threadgroup
    // memory, so it is small and it has to be declared.
    int payload_bytes = 16;
    // Threads per group in each stage. The mesh stage's has to be at least the
    // larger of the meshlet's vertex and primitive counts, because one thread
    // writes one of each.
    int object_threads = 1;
    int mesh_threads = 128;
};

// How long one pass took ON THE GPU.
//
// Not the CPU time spent recording it, which is what a CPU profiler measures
// and which is unrelated: a pass that costs 40 microseconds to encode can cost
// eight milliseconds to run. Without GPU timings, optimising a renderer is
// guesswork -- the frame is slow, and the only way to find out which pass is
// responsible is to comment passes out one at a time.
struct GpuTiming {
    // Points at the caller's own string literal. Not copied: these are set from
    // a PassDesc every frame and allocating a string per pass per frame to
    // report a timing would be its own performance problem.
    const char* label = "";
    double milliseconds = 0.0;
    // WHEN, not just how long, relative to the earliest sample in the frame.
    //
    // A duration on its own is misleading on a tile-based GPU and measurably
    // so: sampling at stage boundaries times a pass from its VERTEX start to
    // its FRAGMENT end, and the hardware overlaps one pass's fragment work with
    // the next one's vertex work. The eleven passes of the world app's frame
    // summed to 19.5 ms of a 4.3 ms frame -- not because any figure was wrong,
    // but because the last four all ended within 0.11 ms of each other and each
    // reported the whole span back to its own start.
    //
    // With begin and end the overlap is visible instead of confusing, and the
    // question a profile is actually asked -- what is on the critical path --
    // has an answer. Both are milliseconds from the first sample the frame
    // took, so they are comparable across passes and meaningless across frames.
    double begin_ms = 0.0;
    double end_ms = 0.0;
};

struct PassDesc {
    TextureId color;
    // VIEWS. Above 1 makes this a layered pass writing `views` slices of an
    // array attachment, and the pipeline's `amplification` must match.
    //
    // Metal wants the mapping from amplification index to array slice supplied
    // per encoder rather than per pipeline, so the encoder sets the identity
    // mapping: view i goes to slice i. Anything else would be a foveated or
    // multi-viewport arrangement this does not do.
    int views = 1;
    // Names this pass in the GPU timing report. Null means untimed, which
    // costs nothing at all -- a sample-buffer attachment is only added to
    // passes that asked for one.
    const char* timer = nullptr;
    // Names this pass's ENCODER in a frame capture. Falls back to `timer`, so a
    // timed pass is named without asking. Separate from `timer` because naming
    // is free and timing is not: a pass can be worth finding in a capture
    // without being worth two timestamp slots.
    const char* label = nullptr;
    // Attachments 1..n, paired with PipelineDesc::extra_colors. All are
    // cleared to `clear_color` -- a G-buffer wants every channel cleared to a
    // known value anyway, and a per-attachment clear colour would be four more
    // fields for one call site.
    std::vector<TextureId> extra_colors;
    // Where a multisample colour attachment is averaged down to. Ignored when
    // `color` has one sample.
    //
    // Resolving is part of ENDING the pass, not a separate step: the samples
    // only exist inside tile memory, and a pass that stores them instead would
    // pay four times the bandwidth to write out data nothing can read.
    TextureId resolve;
    float clear_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    // KEEP what is already in the colour attachment instead of clearing it.
    //
    // For a pass that BLENDS over the previous one -- fog, decals, a UI layer,
    // anything with Blend::Alpha -- and it is not optional there: a pass that
    // clears first has nothing to blend over, so the result is the effect alone
    // on black. That failure looks plausible, which is the problem. Fog drawn
    // over a cleared target is a screen of flat fog, and "the fog is too
    // strong" is the wrong diagnosis.
    //
    // Off by default because clearing is free on a tile-based GPU and loading
    // is not: a load pulls the previous contents from memory into tile memory
    // at the start of every tile, which is bandwidth a pass that overwrites
    // everything has no reason to spend.
    bool load = false;
    TextureId depth;             // null handle = no depth attachment
    float clear_depth = 0.0f;    // reversed-Z: 0 is the FAR plane
    // KEEP the existing depth instead of clearing it, the depth twin of `load`.
    // A second pass drawing into the geometry a first pass already recorded
    // needs it; without it BeginPass clears unconditionally, which is why no
    // pass in this engine could ever inherit another's depth.
    bool load_depth = false;
    // WHERE THE MULTISAMPLE DEPTH IS RESOLVED TO, the depth twin of `resolve`,
    // and its absence cost this engine a whole extra pass over the scene.
    //
    // A multisample depth attachment cannot be sampled, so anything wanting the
    // frame's depth as a texture -- fog, SSAO, light shafts, motion vectors --
    // had no way to get it, and apps/world rendered the ENTIRE SCENE a second
    // time into a single-sample target to manufacture one. That prepass bought
    // no early-Z either, because the colour pass attaches its own multisample
    // depth which is cleared on entry. It measured 1.27 ms of a 5.05 ms frame
    // at 2200x1520.
    //
    // The filter is MAX, and under reversed-Z max is NEAREST. The comment this
    // replaces argued a depth resolve was meaningless because "averaging two
    // distances across a silhouette gives a value describing no surface" --
    // true of an average, and Metal never offers one: the filters are sample
    // zero, min and max.
    TextureId depth_resolve;
    // Keep the depth results after the pass. Off by default because a depth
    // buffer is normally transient; a shadow map is the exception.
    bool keep_depth = false;
};

// How many frames the CPU may run ahead of the GPU. 2 is the minimum that
// overlaps anything; 3 hides a hitchy frame without adding visible latency.
inline constexpr int kFramesInFlight = 3;

class Device;

// Records draws into the pass currently open on its Device. Cheap to copy; it
// owns nothing. Invalid once EndPass() is called.
class Encoder {
  public:
    void SetPipeline(PipelineId);
    void SetCull(Cull, Winding);

    // Restricts drawing to a rectangle of the current target, in pixels.
    //
    // What this is FOR: a shadow atlas. Several lights each need their own
    // depth map, and one texture divided into tiles costs one pass and one
    // texture binding instead of N of each. The viewport remaps clip space onto
    // the tile; the scissor is what stops a triangle that falls outside it from
    // scribbling over a neighbour, and Metal will not do the second because you
    // asked for the first.
    void SetViewport(int x, int y, int width, int height);
    void SetScissor(int x, int y, int width, int height);
    void SetVertexBuffer(BufferId, std::size_t offset, int slot);
    void SetFragmentBuffer(BufferId, std::size_t offset, int slot);
    // Binds a render target from an earlier pass as a sampled input.
    void SetFragmentTexture(TextureId, int slot);
    void SetFragmentSampler(SamplerId, int slot);
    // Binds a top-level acceleration structure for the fragment stage, and
    // makes it and everything it references resident. Residency is the part
    // that is easy to miss: an acceleration structure reached indirectly, and
    // the geometry buffers under it, are invisible to the driver's automatic
    // tracking.
    void SetFragmentAccel(AccelId, int slot);
    // Small per-draw constants. The backend copies immediately, so the pointer
    // does not have to outlive the call.
    void SetVertexBytes(const void* data, std::size_t bytes, int slot);
    void SetFragmentBytes(const void* data, std::size_t bytes, int slot);
    void Draw(std::size_t vertex_count);
    // `instance_count` copies of a NON-INDEXED draw. The indexed version below
    // is the usual one; this exists for geometry generated entirely from the
    // vertex id, where there is no index buffer to point at -- a decal's
    // projection box, a batch of billboards.
    void DrawInstanced(std::size_t vertex_count, std::size_t instance_count);
    // 32-BIT indices, always. 16 bits caps a mesh at 65535 vertices, which any
    // scanned or sculpted asset clears without trying -- and the failure is not
    // an error, it is triangles wrapping back to the start of the buffer. The
    // extra two bytes per index is the cheapest correctness there is.
    // The OBJECT and MESH stages have their own argument tables. A mesh-shader
    // pipeline has three producer stages and SetVertexBuffer reaches none of
    // them -- binding through it compiles, runs, and delivers nothing, which
    // reads as a scene with no geometry in it.
    void SetObjectBuffer(BufferId, std::size_t offset, int slot);
    void SetMeshBuffer(BufferId, std::size_t offset, int slot);
    // One object threadgroup per meshlet. Each one decides for itself whether
    // to launch a mesh threadgroup, so `count` is the number of CANDIDATES and
    // not the number that will be drawn.
    void DrawMeshThreadgroups(int count, int object_threads, int mesh_threads);
    void DrawIndexedU32(BufferId indices, std::size_t index_count);
    // One draw, `instance_count` copies. The shader reads instance_id and
    // looks up whatever differs per copy; nothing else changes.
    void DrawIndexedInstancedU32(BufferId indices, std::size_t index_count,
                                 std::size_t instance_count);
    // The instance count and the index range come from a BUFFER the GPU wrote,
    // so the CPU never learns them. That is the point: a culling pass on the
    // GPU can decide how much to draw without a readback, and a readback would
    // cost a full pipeline stall -- which is more than the culling saves.
    //
    // `args` must hold a GpuDrawArgs at `offset`.
    void DrawIndexedIndirectU32(BufferId indices, BufferId args,
                                std::size_t offset);

  private:
    friend class Device;
    explicit Encoder(Device* d) : device_(d) {}
    Device* device_ = nullptr;
};

// Records work into the compute pass currently open on its Device.
//
// WHY compute is a separate encoder and not a mode of the other one: a render
// pass and a compute pass cannot be open at the same time. Everything a render
// pass writes is in tile memory until the pass ends, so a compute dispatch
// cannot read it, and Metal enforces that by making them different encoder
// objects. Modelling them as one type would let a caller write code the API
// forbids and find out at runtime.
class ComputeEncoder {
  public:
    void SetPipeline(ComputePipelineId);
    void SetBuffer(BufferId, std::size_t offset, int slot);
    void SetTexture(TextureId, int slot);
    // A sampler, for kernels that READ a texture rather than index it.
    //
    // Not redundant with SetTexture. A compute kernel writing to a storage
    // texture addresses texels directly and needs none, but one that filters --
    // a mip reduction, a cubemap convolution -- needs the hardware's
    // interpolation and, for a cube, its cross-face blending, and there is no
    // way to ask for either without a sampler.
    void SetSampler(SamplerId, int slot);
    // Small constants, copied immediately.
    void SetBytes(const void* data, std::size_t bytes, int slot);

    // `count` items, in groups of `group`. Not a threadgroup COUNT: this takes
    // the number of items and lets the backend round up, because the alternative
    // is every call site doing the same ceiling division and one of them getting
    // it wrong in a way that silently skips the last partial group.
    //
    // The shader still has to guard on the item index -- the rounding means the
    // last group runs threads past the end, and nothing else stops them writing
    // there.
    void Dispatch(int count, int group = 64);

    // A 2D or 3D grid, for kernels whose work IS 2D or 3D -- an image filter, a
    // cubemap's six faces. Flattening those onto Dispatch() and unpacking the
    // index in the shader works and costs a divide and a modulo per thread, and
    // more importantly it destroys the 2D locality the texture cache is built
    // around: a linear index walks a whole row before touching the next, while
    // an 8x8 group stays inside one tile.
    void Dispatch2D(int width, int height, int gx = 8, int gy = 8);
    void Dispatch3D(int width, int height, int depth, int gx = 8, int gy = 8,
                    int gz = 1);

  private:
    friend class Device;
    explicit ComputeEncoder(Device* d) : device_(d) {}
    Device* device_ = nullptr;
};

// A presentable surface. The RHI creates the native layer; the platform layer
// is responsible for putting it on screen, because window systems are not the
// graphics API's business.
class Swapchain {
  public:
    ~Swapchain();
    // CAMetalLayer* on this backend. Opaque on purpose — engine/platform casts
    // it, nothing else may.
    [[nodiscard]] void* NativeLayer() const;
    void Resize(int width, int height);
    [[nodiscard]] int Width() const;
    [[nodiscard]] int Height() const;

  private:
    friend class Device;
    Swapchain();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Device {
  public:
    [[nodiscard]] static std::unique_ptr<Device> Create(std::string& error);
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    // --- resources -----------------------------------------------------------
    [[nodiscard]] BufferId CreateBuffer(const void* data, std::size_t bytes);

    // CPU-writable buffer with a persistent mapping, for per-frame data. The
    // caller must sub-allocate it per frame slot and never write a region the
    // GPU might still be reading — that is what kFramesInFlight is for.
    [[nodiscard]] BufferId CreateDynamicBuffer(std::size_t bytes);
    [[nodiscard]] void* MapBuffer(BufferId);
    // Required alignment for a buffer binding offset on this device.
    [[nodiscard]] std::size_t UniformAlignment() const;
    // cpu_readable picks Shared storage so ReadPixels works. Leave it false for
    // targets that only ever feed another pass — Private lets the driver keep
    // them in whatever memory suits the GPU.
    // `samples` above 1 makes a MULTISAMPLE target. It cannot be sampled or
    // read back — it exists to be resolved into an ordinary one at end of pass.
    [[nodiscard]] TextureId CreateRenderTarget(int width, int height, Format,
                                               bool cpu_readable = false,
                                               int samples = 1);
    // `sampleable` gives up memoryless storage in exchange for being readable
    // by a later pass — which SSAO needs and an ordinary depth buffer does not.
    // `samples` must match the colour attachment it is paired with.
    // `cpu_readable` picks Shared storage so ReadPixels works, exactly as it
    // does for a colour target. Off by default and it should stay off outside a
    // test: a depth buffer another pass samples wants Private, and one nothing
    // samples wants Memoryless and never touches DRAM at all.
    [[nodiscard]] TextureId CreateDepthTarget(int width, int height,
                                              bool sampleable = false,
                                              int samples = 1,
                                              bool cpu_readable = false);
    // Like CreateDepthTarget, but SAMPLEABLE and backed by real memory. The
    // ordinary one is memoryless: perfect for a depth buffer you throw away at
    // end of pass, useless for one another pass has to read.
    [[nodiscard]] TextureId CreateShadowMap(int size);

    // Sampled texture uploaded from CPU pixels. `rgba8` must hold
    // width*height*4 bytes, first row at the top.
    //
    // `mips` builds the full chain down to 1x1 and fills it on the GPU. On by
    // default because the alternative is not "slightly softer at distance", it
    // is severe aliasing: a textured floor sampled at one texel per several
    // pixels shimmers violently as the camera moves, and temporal
    // antialiasing cannot fix it -- TAA averages samples of a signal that is
    // already undersampled, so it converges on the wrong answer smoothly.
    //
    // `srgb` decodes in the sampler. Any 8-bit ALBEDO or emissive map wants it
    // on; a roughness, metallic, occlusion or NORMAL map wants it off, because
    // those store numbers rather than colours and an sRGB decode bends them.
    [[nodiscard]] TextureId CreateTexture2D(int width, int height,
                                            const void* rgba8, bool mips = true,
                                            bool srgb = false);
    // The same, from FLOAT pixels: `rgba32f` holds width*height*4 floats.
    // What an HDR environment arrives as, and the reason it cannot go through
    // the 8-bit path -- the whole point of an environment map is the values
    // above one, and an 8-bit texture has none.
    [[nodiscard]] TextureId CreateTexture2DFloat(int width, int height,
                                                 const float* rgba32f);
    // A LAYERED render target: `layers` slices of a 2D array, all rendered in
    // one pass. Stereo is two of them.
    [[nodiscard]] TextureId CreateRenderTargetArray(int width, int height, int layers,
                                                    Format, bool cpu_readable = false);
    [[nodiscard]] TextureId CreateDepthTargetArray(int width, int height, int layers);
    // One slice of an array, as a handle that can be sampled or read back on
    // its own. The array itself cannot be: a readback wants a 2D texture.
    [[nodiscard]] TextureId CreateArraySlice(TextureId array, int slice);
    // A 3D texture from float pixels: width*height*depth*4 floats, x fastest.
    // Stored as RGBA16Float, which halves the bandwidth of a lookup that
    // happens once per fragment and holds far more precision than an irradiance
    // coefficient needs.
    //
    // Sampled with an ordinary linear sampler, which on a 3D texture is
    // TRILINEAR in hardware -- eight texels and seven lerps for free. Doing the
    // same interpolation by hand from a buffer is eight scattered reads and a
    // dozen instructions, per fragment.
    [[nodiscard]] TextureId CreateTexture3DFloat(int width, int height, int depth,
                                                 const float* rgba32f);
    // A block-compressed texture, every mip level supplied.
    //
    // `level_offsets` gives each level's start in `data`, and its size is the
    // level count. Levels halve from `width` x `height` and are laid out as
    // ceil(w/4) x ceil(h/4) blocks -- so a 5-texel-wide level is two blocks
    // across, and the last block's right-hand column is never sampled.
    //
    // All levels in one call because a compressed texture has no other way to
    // get them: the GPU cannot generate them, so the alternative is a call per
    // level and a chance to forget one, and a missing level does not fail --
    // it samples whatever was in that memory.
    [[nodiscard]] TextureId CreateTexture2DCompressed(
        int width, int height, Format, std::span<const std::uint8_t> data,
        std::span<const std::size_t> level_offsets);
    // `max_anisotropy` above 1 turns on anisotropic filtering: up to that many
    // samples along the axis of greatest compression.
    //
    // It is the fix for the OTHER half of minification aliasing. A mip chain
    // picks one level for both axes, so a floor seen at a grazing angle -- very
    // compressed along one axis and barely along the other -- has to choose
    // between aliasing across and blurring along. 16 is the hardware maximum
    // and costs nothing on a surface facing the camera, because the sample
    // count scales with the actual anisotropy rather than being fixed.
    [[nodiscard]] SamplerId CreateSampler(Filter, Wrap, int max_anisotropy = 1);
    // A sampler that filters ACROSS MIP LEVELS as well as within one. Separate
    // from the ordinary one because it is wrong nearly everywhere else: a UI
    // atlas or a shadow map has no mip chain, and asking for trilinear on one
    // samples level zero anyway while paying for the decision.
    //
    // The prefiltered specular probe is the case that needs it. Roughness
    // selects a mip continuously, and without interpolation between levels a
    // surface whose roughness varies smoothly shows hard bands where the mip
    // index steps.
    [[nodiscard]] SamplerId CreateMipSampler(Wrap);

    // --- cubemaps and storage textures -----------------------------------------
    //
    // A CUBEMAP is six square faces sharing one texture. It exists rather than
    // six separate textures because the hardware can filter ACROSS the seams:
    // sampling near an edge blends texels from the neighbouring face, and doing
    // that by hand needs to know the adjacency and the winding of all twelve
    // edges. A prefiltered environment map with visible seams is the classic
    // symptom of having tried.
    //
    // `mip_levels` of 0 means the full chain down to 1x1.
    [[nodiscard]] TextureId CreateCubemap(int size, Format, int mip_levels = 1);
    // A 2D texture a COMPUTE kernel writes into. Distinct from a render target
    // because the usage flags differ and Metal validates them: a texture
    // created for rendering cannot be bound for shader writes.
    // A 3D texture a compute pass WRITES and a later pass samples. The froxel
    // volume is exactly this: filled by one kernel, prefix-summed by another
    // and read trilinearly by a fullscreen pass.
    [[nodiscard]] TextureId CreateStorageTexture3D(int width, int height, int depth,
                                                   Format);
    [[nodiscard]] TextureId CreateStorageTexture2D(int width, int height, Format,
                                                   int mip_levels = 1);

    // A VIEW of one mip level of a texture, as a handle that can be bound for
    // writing.
    //
    // Needed because a compute kernel writes to a texture at ONE size, and a
    // mip chain is a different size at every level. Metal expresses that as a
    // view over a level range; without it the only way to fill mip 3 of a probe
    // would be to allocate a separate texture per level and blit them together.
    //
    // For a cubemap the view is a 2D ARRAY of six slices, because that is what
    // a compute kernel can write: `texture2d_array<float, access::write>` with
    // the face in gid.z. Cube adjacency is a sampling-time property, not a
    // storage one, so nothing is lost.
    [[nodiscard]] TextureId CreateMipView(TextureId, int mip);

    [[nodiscard]] int TextureWidth(TextureId) const;
    [[nodiscard]] int TextureHeight(TextureId) const;
    [[nodiscard]] int TextureMipLevels(TextureId) const;

    // --- ray tracing ---------------------------------------------------------
    //
    // Two levels, and the split is the whole reason hardware ray tracing is
    // affordable. A BOTTOM-level structure is a BVH over one mesh's triangles,
    // built once and reused; a TOP-level structure is a much smaller BVH over
    // instances of those, each with its own transform, rebuilt when things
    // move. A scene of a thousand crates builds one crate BVH and a thousand
    // cheap instance records, not a thousand BVHs.
    //
    // False on hardware without ray tracing. Everything below returns a null
    // handle in that case rather than failing at build time, so a renderer can
    // fall back to shadow maps at runtime.
    [[nodiscard]] bool SupportsRaytracing() const;

    // A BVH over `index_count` indices into `vb`. The vertex buffer's positions
    // must be the FIRST three floats of each `vertex_stride` bytes, which is
    // what the engine's VertexIn already is.
    //
    // No vertex COUNT: the triangles are named entirely by the index buffer, so
    // Metal never needs one, and a parameter that is only validated and then
    // ignored is worse than no parameter at all.
    [[nodiscard]] AccelId CreateBlas(BufferId vb, int vertex_stride, BufferId ib,
                                     int index_count, std::string& error);

    struct AccelInstance {
        AccelId blas;
        // Object -> world, COLUMN-MAJOR, 16 floats. A plain array and not the
        // engine's Mat4: this header depends on nothing inside the engine, and
        // a matrix type is exactly the sort of thing that quietly makes an RHI
        // reach upward.
        //
        // Only the upper 3x4 is used. An instance cannot carry a projection,
        // and while a non-uniform scale is allowed, a shear is where
        // intersection precision starts to suffer.
        float transform[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    };
    [[nodiscard]] AccelId CreateTlas(std::span<const AccelInstance> instances,
                                     std::string& error);
    [[nodiscard]] PipelineId CreatePipeline(const PipelineDesc&, std::string& error);
    // A mesh-shader pipeline. Returns a null handle with a message when the
    // hardware has no mesh stage; the caller falls back to the vertex path.
    [[nodiscard]] PipelineId CreateMeshPipeline(const MeshPipelineDesc&,
                                                std::string& error);
    [[nodiscard]] bool SupportsMeshShaders() const;
    // `source` is backend shader source, `fn` the kernel's name.
    [[nodiscard]] ComputePipelineId CreateComputePipeline(const std::string& source,
                                                          const std::string& fn,
                                                          std::string& error);
    // A buffer the GPU writes and nothing uploads: `bytes` of undefined
    // contents, readable afterwards through MapBuffer.
    //
    // SHARED storage, not private. Private is the reflex -- "the CPU never
    // touches it, so do not pay for coherency" -- and on unified memory there
    // is nothing to pay: there is one copy either way. What private would cost
    // is the ability to read the result back at all without a blit, and reading
    // it back is half the point of writing it out. A posed mesh is wanted on
    // the CPU for collision as much as on the GPU for ray tracing.
    [[nodiscard]] BufferId CreateStorageBuffer(std::size_t bytes);
    // Releases the texture. The handle is dangling afterwards — the caller is
    // responsible for not reusing it. No generation counters yet.
    void DestroyTexture(TextureId);

    // `hdr` asks the window server for extended dynamic range: the layer keeps
    // values above 1 instead of clamping them, so a highlight the tone map
    // decided is four times reference white reaches the panel as four times
    // reference white.
    //
    // It needs a half-float format to mean anything -- an 8-bit layer has
    // nowhere to put a value above one whatever the colour space says -- so
    // asking for it with an 8-bit format is refused rather than silently
    // ignored.
    [[nodiscard]] std::unique_ptr<Swapchain> CreateSwapchain(Format,
                                                             std::string& error,
                                                             bool hdr = false);
    // What the display can actually do, as a multiple of reference white, or 1
    // when there is no extended range. Read it rather than assuming: the same
    // machine reports a different headroom depending on the display, the
    // brightness setting and whether it is on battery.
    [[nodiscard]] float DisplayHeadroom(const Swapchain&) const;

    // --- submission ----------------------------------------------------------
    // Returns a null handle when no drawable is available; skip the frame.
    [[nodiscard]] TextureId AcquireDrawable(Swapchain&);

    // Blocks until fewer than kFramesInFlight frames are outstanding. Without
    // this the CPU laps the GPU and overwrites uniforms mid-read.
    void BeginFrame();
    // Which of the kFramesInFlight ring slots this frame owns.
    [[nodiscard]] int FrameSlot() const;
    // Monotonic frame counter. Unlike FrameSlot() this never repeats, so it is
    // the correct thing to key per-frame allocator resets on.
    [[nodiscard]] std::uint64_t FrameIndex() const;
    // Diagnostics: the largest number of frames ever simultaneously in flight.
    // Must never exceed kFramesInFlight — the occlusion test asserts it.
    [[nodiscard]] int PeakFramesInFlight() const;
    // Diagnostics: entries in the texture handle table. Should stay bounded
    // when textures are destroyed and recreated (e.g. on window resize).
    [[nodiscard]] int TextureSlotCount() const;

    // --- GPU timing ------------------------------------------------------------
    //
    // Whether the hardware can timestamp at pass boundaries. False on older
    // Macs, and everything below then reports zero rather than failing --
    // a profiler that cannot be built into the engine because it might not be
    // supported is a profiler nobody has when they need it.
    [[nodiscard]] bool SupportsGpuTiming() const;
    // Per-pass times from the most recently COMPLETED frame -- which is two or
    // three frames behind the one being recorded, because reading them any
    // sooner would mean waiting for the GPU. That lag is why these are for a
    // HUD and a log, not for anything that feeds back into the frame.
    //
    // BY VALUE, and that is not an oversight. These are published from a Metal
    // completion handler on a driver thread; handing back a view into the
    // engine's own vector meant the caller held a pointer that the next
    // completion freed. A dozen structs copied once a frame is not a cost worth
    // a lifetime hazard.
    [[nodiscard]] std::vector<GpuTiming> LastFrameTimings() const;
    // Wall-clock GPU time for the whole of the last completed command buffer.
    // Available even without stage-boundary sampling, and the honest headline
    // number: the per-pass timings do not sum to it, because passes overlap.
    [[nodiscard]] double LastFrameGpuMilliseconds() const;

    // --- GPU faults ------------------------------------------------------------
    //
    // A command buffer that fails on the GPU produces no image and no error
    // return: Commit is fire-and-forget, and the failure arrives later on a
    // driver thread. Nothing looked, so every GPU fault this engine has ever
    // had presented as a black or stale frame with nothing to point at.
    //
    // The most recent fault's description, INCLUDING the labels of the encoders
    // that faulted or were affected, and cleared by the read -- so a caller
    // that prints it once prints it once. Empty when there has been none.
    [[nodiscard]] std::string TakeGpuFault();
    // How many faults since the device was created. Unlike TakeGpuFault this
    // does not clear, so a HUD can keep saying that something went wrong after
    // the description has been logged.
    [[nodiscard]] int GpuFaultCount() const;

    [[nodiscard]] Encoder BeginPass(const PassDesc&);
    void EndPass();

    // A compute pass. Must not overlap a render pass on the same device -- see
    // ComputeEncoder for why that is a hardware fact and not a rule this layer
    // invented. `timer` and `label` behave as PassDesc's do.
    [[nodiscard]] ComputeEncoder BeginCompute(const char* timer = nullptr,
                                              const char* label = nullptr);
    void EndCompute();
    // Whether the pass currently open has a depth attachment. A pipeline built
    // without depth cannot be used in a pass that has one, and vice versa —
    // Metal rejects the draw — so the renderer has to be able to ask.
    [[nodiscard]] bool CurrentPassHasDepth() const;
    void Present(Swapchain&);  // schedule before Commit
    void Commit();
    [[nodiscard]] bool CommitAndWait(std::string& error);

    // Blocking readback of a render target. Offscreen paths only.
    // Blocks until the most recently submitted work has finished.
    //
    // ANY CPU READ OF SOMETHING THE GPU WROTE NEEDS THIS FIRST. Neither
    // getBytes nor a mapped pointer synchronises -- they hand back whatever is
    // in memory at the instant they are called, which for work still in flight
    // is a partial result with no indication that it is one. ReadPixels calls
    // it for you; a caller reading a storage buffer through MapBuffer has to
    // call it itself, because MapBuffer is also how per-frame uploads are
    // written and waiting there would stall every one of them.
    //
    // A no-op after CommitAndWait, which has already waited.
    void WaitForGpu();

    [[nodiscard]] bool ReadPixels(TextureId, int width, int height,
                                  std::span<std::uint8_t> out_rgba);

  private:
    friend class Encoder;
    friend class ComputeEncoder;
    Device();
    // Attaches the frame's completion handler. MUST run after the last pass is
    // encoded and before the command buffer is committed: it captures the list
    // of timed passes, and that list does not exist until the passes have been
    // encoded.
    void InstallFrameCompletion();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eng::rhi
