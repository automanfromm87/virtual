// Pure C++20. Decides WHAT to draw and in what order; engine/rhi decides how to
// talk to the GPU.
//
// Note the BUILD rule: cc_library, not objc_library. This whole layer compiles
// with no Apple frameworks. That is the test of whether the RHI boundary is
// real — if a Metal type ever leaks up here, this file stops building and you
// find out immediately instead of at porting time.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "engine/anim/anim.h"
#include "engine/geometry/mesh.h"
#include "engine/resource/handles.h"
#include "engine/rhi/rhi.h"
#include "engine/scene/scene.h"

namespace eng {

struct Image {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgba;  // width * height * 4
};

inline constexpr float kClearColor[4] = {0.1f, 0.1f, 0.12f, 1.0f};

enum class Shading : std::uint8_t {
    Lit,        // Lambert + ambient, needs normals
    Flat,       // straight vertex colour
    Composite,   // fullscreen, samples a previous pass's colour target
    ShadowDepth, // depth only, from the light's point of view
    Ssao,        // fullscreen, depth in and an occlusion factor out
};

// What a surface looks like and how it is rasterised.
//
// Only `shading` and `depth_test` are baked into a pipeline object; `cull` is
// encoder state that can change between draws. Two materials that differ ONLY
// in cull mode therefore share one pipeline — which is the point of the cache.
struct MaterialDesc {
    Shading shading = Shading::Lit;
    bool depth_test = true;
    rhi::Cull cull = rhi::Cull::Back;

    // --- surface (metallic/roughness workflow) -------------------------------
    // A metal has no diffuse lobe and its specular colour IS its base colour;
    // a dielectric has a diffuse lobe and a colourless 4% specular. `metallic`
    // blends between those two physical facts rather than being a look dial.
    Vec4 base_color{1.0f, 1.0f, 1.0f, 1.0f};
    float roughness = 0.5f;
    float metallic = 0.0f;

    // Maps MULTIPLY the scalars above. A null handle binds a 1x1 white texture,
    // so an untextured material needs no branch in the shader.
    rhi::TextureId albedo;
    rhi::TextureId roughness_map;  // read from the RED channel

    // Alpha blended, drawn after every opaque object, back to front, and NOT
    // writing depth. Glass needs all four of those or it stops looking like
    // glass — or worse, stops the room behind it from drawing at all.
    bool transparent = false;
};

struct RenderStats {
    int submitted = 0;          // instances the scene offered
    int invalid = 0;            // dropped: null or out-of-range mesh/material
    int incompatible = 0;       // dropped: material depth state vs pass mismatch
    int culled = 0;             // rejected by the frustum
    int draws = 0;              // draw calls actually issued
    int transparent_draws = 0;  // of which, in the back-to-front batch
    int pipeline_switches = 0;  // times the bound pipeline really changed
    int overflowed = 0;         // dropped: this frame's uniform ring slot filled
};

// Owns the GPU-side mesh and material tables, and the pipeline cache. Build one
// per colour format: a pipeline state is only valid for the format it was
// compiled against, and a drawable (BGRA) differs from an offscreen target
// (RGBA).
class Renderer {
  public:
    [[nodiscard]] static std::unique_ptr<Renderer> Create(rhi::Device&,
                                                          rhi::Format color,
                                                          std::string& error);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    // Uploads vertex + index data and returns a handle the scene can name.
    // Returns the null handle if the mesh is empty or the upload fails.
    [[nodiscard]] MeshHandle UploadMesh(const Mesh&);

    // The same, plus per-vertex joint indices and weights. `skin` must be
    // parallel to mesh.vertices; a mismatch returns the null handle rather than
    // reading past the end of the shorter one.
    //
    // `joint_count` is how many matrices a palette for this mesh holds, and is
    // capped at kMaxJoints. Storing it on the MESH rather than the material is
    // deliberate: skinning is a property of the geometry, and the same material
    // may be worn by a static prop and a character.
    [[nodiscard]] MeshHandle UploadSkinnedMesh(const Mesh&,
                                               const std::vector<anim::SkinVertex>&,
                                               int joint_count);

    // Largest palette a single mesh may use. A frame may hold several.
    static constexpr int kMaxJoints = 64;
    // How many skinned instances one frame can draw before the palette ring is
    // full. Overflowing drops the draw, and shows up in RenderStats::overflowed.
    static constexpr int kMaxSkinnedPerFrame = 16;

    // Joints the mesh was uploaded with, or 0 if it is not skinned.
    [[nodiscard]] int JointCount(MeshHandle) const;

    // Registers a material, reusing an existing pipeline when the state matches.
    [[nodiscard]] MaterialHandle CreateMaterial(const MaterialDesc&,
                                                std::string& error);

    // Both record into an already-open pass. width/height set the projection's
    // aspect ratio only.
    // Depth-only pass from the light's point of view. Must run BEFORE
    // DrawScene, into a pass whose depth target is `shadow_map`.
    //
    // Does NOT frustum-cull: an object outside the CAMERA's view can still cast
    // a shadow into it, so culling here with the camera's frustum would make
    // shadows pop as you turn around.
    void DrawShadow(rhi::Encoder&, const Scene&);

    // `shadow_map` is the depth target DrawShadow wrote, or a null handle for
    // no shadows.
    void DrawScene(rhi::Encoder&, const Scene&, int width, int height,
                   rhi::TextureId shadow_map = {});
    void DrawTriangle(rhi::Encoder&, int width, int height);

    // Fullscreen pass that samples `src` (a colour target written earlier this
    // frame) and applies a vignette. Needs no vertex buffer — the triangle is
    // generated from the vertex id.
    // `ao` may be null, in which case a 1x1 white texture is bound and the
    // multiply is a no-op.
    void DrawComposite(rhi::Encoder&, rhi::TextureId src, rhi::TextureId ao = {});

    // Fullscreen occlusion pass. `depth` must be a sampleable depth target
    // written by an earlier pass with the SAME camera.
    void DrawSsao(rhi::Encoder&, const Camera&, int width, int height,
                  rhi::TextureId depth, float radius = 1.1f);

    [[nodiscard]] const RenderStats& LastStats() const;
    // Indices into the last DrawScene's Scene::instances, in the order they
    // were actually submitted. Exposed so the sort keys can be tested — without
    // it, both of them could be deleted and every test would still pass.
    [[nodiscard]] const std::vector<int>& LastDrawOrder() const;
    // How many distinct pipeline objects the cache actually built.
    [[nodiscard]] int PipelineCount() const;
    // Casters written by the most recent DrawShadow. Separate from RenderStats
    // because DrawScene resets that, and the shadow pass runs first.
    [[nodiscard]] int ShadowDrawCount() const;
    // Which pipeline a material resolved to. Exposed so a test can assert the
    // cache's actual INVARIANT — that two materials differing only in encoder
    // state share one pipeline — instead of a total count, which goes stale
    // every time the engine grows a new shader.
    [[nodiscard]] std::uint32_t PipelineOf(MaterialHandle) const;

  private:
    Renderer();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// --- one-shot offscreen helpers ---------------------------------------------
// Each spins up its own device and renderer, so they are convenient rather than
// fast. On failure they return an empty Image and fill `error`.
[[nodiscard]] Image RenderSceneOffscreen(const Scene&, int width, int height,
                                         std::string& error);
[[nodiscard]] Image RenderSphereOffscreen(int width, int height,
                                          float angle_radians,
                                          std::string& error);
[[nodiscard]] Image RenderTriangleOffscreen(int width, int height,
                                            std::string& error);

}  // namespace eng
