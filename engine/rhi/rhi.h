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

namespace eng::rhi {

enum class Format : std::uint8_t { RGBA8Unorm, BGRA8Unorm, Depth32Float };
enum class Cull : std::uint8_t { None, Back, Front };
enum class Winding : std::uint8_t { Clockwise, CounterClockwise };
enum class Compare : std::uint8_t { Never, Less, Greater, Always };
enum class Filter : std::uint8_t { Nearest, Linear };
enum class Wrap : std::uint8_t { Clamp, Repeat };

// Opaque handles. 0 is always the null handle — never a valid resource.
struct BufferId {
    std::uint32_t v = 0;
};
struct TextureId {
    std::uint32_t v = 0;
};
struct PipelineId {
    std::uint32_t v = 0;
};
struct SamplerId {
    std::uint32_t v = 0;
};
inline bool Valid(BufferId h) { return h.v != 0; }
inline bool Valid(TextureId h) { return h.v != 0; }
inline bool Valid(PipelineId h) { return h.v != 0; }
inline bool Valid(SamplerId h) { return h.v != 0; }

struct PipelineDesc {
    // Shader source in the backend's own language. The RHI compiles it but
    // does not interpret it — choosing what the shader SAYS is the renderer's
    // job, which is why this is a string and not a struct of lighting options.
    std::string source;
    std::string vertex_fn;
    std::string fragment_fn;
    Format color = Format::BGRA8Unorm;
    bool depth = false;  // attach + test depth
    // Depth-only: no fragment shader and no colour attachment. That is the
    // whole shadow pass — position out, depth written, nothing shaded.
    bool depth_only = false;
    // Straight alpha blending, and normally paired with depth_write = false:
    // a transparent surface must not stop what is behind it from drawing.
    bool blend = false;
    Compare depth_compare = Compare::Greater;  // reversed-Z default
    bool depth_write = true;
};

struct PassDesc {
    TextureId color;
    float clear_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    TextureId depth;             // null handle = no depth attachment
    float clear_depth = 0.0f;    // reversed-Z: 0 is the FAR plane
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
    void SetVertexBuffer(BufferId, std::size_t offset, int slot);
    void SetFragmentBuffer(BufferId, std::size_t offset, int slot);
    // Binds a render target from an earlier pass as a sampled input.
    void SetFragmentTexture(TextureId, int slot);
    void SetFragmentSampler(SamplerId, int slot);
    // Small per-draw constants. The backend copies immediately, so the pointer
    // does not have to outlive the call.
    void SetVertexBytes(const void* data, std::size_t bytes, int slot);
    void SetFragmentBytes(const void* data, std::size_t bytes, int slot);
    void Draw(std::size_t vertex_count);
    void DrawIndexedU16(BufferId indices, std::size_t index_count);

  private:
    friend class Device;
    explicit Encoder(Device* d) : device_(d) {}
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
    [[nodiscard]] TextureId CreateRenderTarget(int width, int height, Format,
                                               bool cpu_readable = false);
    // `sampleable` gives up memoryless storage in exchange for being readable
    // by a later pass — which SSAO needs and an ordinary depth buffer does not.
    [[nodiscard]] TextureId CreateDepthTarget(int width, int height,
                                              bool sampleable = false);
    // Like CreateDepthTarget, but SAMPLEABLE and backed by real memory. The
    // ordinary one is memoryless: perfect for a depth buffer you throw away at
    // end of pass, useless for one another pass has to read.
    [[nodiscard]] TextureId CreateShadowMap(int size);

    // Sampled texture uploaded from CPU pixels. `rgba8` must hold
    // width*height*4 bytes, first row at the top.
    [[nodiscard]] TextureId CreateTexture2D(int width, int height,
                                            const void* rgba8);
    [[nodiscard]] SamplerId CreateSampler(Filter, Wrap);
    [[nodiscard]] PipelineId CreatePipeline(const PipelineDesc&, std::string& error);
    // Releases the texture. The handle is dangling afterwards — the caller is
    // responsible for not reusing it. No generation counters yet.
    void DestroyTexture(TextureId);

    [[nodiscard]] std::unique_ptr<Swapchain> CreateSwapchain(Format, std::string& error);

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
    [[nodiscard]] Encoder BeginPass(const PassDesc&);
    void EndPass();
    // Whether the pass currently open has a depth attachment. A pipeline built
    // without depth cannot be used in a pass that has one, and vice versa —
    // Metal rejects the draw — so the renderer has to be able to ask.
    [[nodiscard]] bool CurrentPassHasDepth() const;
    void Present(Swapchain&);  // schedule before Commit
    void Commit();
    [[nodiscard]] bool CommitAndWait(std::string& error);

    // Blocking readback of a render target. Offscreen paths only.
    [[nodiscard]] bool ReadPixels(TextureId, int width, int height,
                                  std::span<std::uint8_t> out_rgba);

  private:
    friend class Encoder;
    Device();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eng::rhi
