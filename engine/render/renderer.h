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
#include <span>

#include "engine/render/gi.h"
#include "engine/render/ibl.h"
#include "engine/rhi/rhi.h"
#include "engine/scene/scene.h"

namespace eng {

struct Image {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgba;  // width * height * 4
};

// LINEAR, not display-referred. Surface shading writes linear HDR and the
// composite gamma-corrects at the end, so a value that used to land straight in
// the framebuffer as a dark grey now gets brightened on the way out. 0.1 became
// an 8-bit 92 rather than the 26 it had always been.
inline constexpr float kClearColor[4] = {0.014f, 0.014f, 0.018f, 1.0f};

enum class Shading : std::uint8_t {
    Lit,        // Lambert + ambient, needs normals
    Flat,       // straight vertex colour
    Composite,   // fullscreen, samples a previous pass's colour target
    ShadowDepth, // depth only, from the light's point of view
    Ssao,        // fullscreen, depth in and an occlusion factor out
    BloomBright, // fullscreen, keeps what is above a threshold
    BloomBlur,   // fullscreen, one axis of a separable gaussian
    // DEFERRED. GBuffer writes the surface into two colour targets plus depth;
    // DeferredLight is the fullscreen pass that reads them back and lights each
    // pixel once, however many lights there are.
    // Lit, with model and tint read per instance from a buffer rather than
    // from the per-draw uniform block. One draw for any number of copies.
    LitInstanced,
    GBuffer,
    DeferredLight,
    // Ray-traced shadows: fullscreen, traces one ray per pixel at the sun and
    // writes visibility. Needs hardware ray tracing and a built scene BVH.
    RayShadow,
    // STEREO. The same fragment stage as Lit, with a vertex stage that emits
    // both eyes from one invocation.
    LitStereo,
    // ORDER-INDEPENDENT TRANSPARENCY. OitAccumulate writes the two commutative
    // buffers; OitResolve is the fullscreen pass that turns them into one
    // premultiplied colour to blend over the opaque frame.
    OitAccumulate,
    OitResolve,
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

    // EMISSIVE radiance, added after everything else and lit by nothing. Not a
    // colour in 0..1: a lamp filament is thousands of times a wall, the scene
    // target is half-float precisely so it can hold that, and the bloom
    // threshold is in linear radiance. A value of 1 is "as bright as a white
    // surface in full light" and will not glow.
    //
    // This is the only way to make a light SOURCE visible. Scene::lights put
    // radiance into the world; nothing before this made the lamp itself bright,
    // so every light in every scene here was an invisible point that other
    // things reflected.
    Vec3 emissive{0.0f, 0.0f, 0.0f};

    // Maps MULTIPLY the scalars above. A null handle binds a 1x1 white texture,
    // so an untextured material needs no branch in the shader.
    rhi::TextureId albedo;
    rhi::TextureId roughness_map;  // read from the RED channel
    rhi::TextureId metallic_map;   // read from the RED channel
    rhi::TextureId emissive_map;   // multiplies `emissive`, RGB
    // Baked ambient occlusion, RED channel. Multiplies the AMBIENT and image-
    // based terms only, never the direct light: a crevice that the sky cannot
    // reach is still lit by a lamp pointed into it, and darkening direct light
    // with a static map is how baked AO turns into visible dirt.
    rhi::TextureId occlusion_map;

    // TANGENT-SPACE normal map. The single most valuable texture a material can
    // have: it is what puts brick mortar, fabric weave, brushed metal and skin
    // pores on a surface that has none of them in its triangles.
    //
    // Only xy are read; z is rebuilt as sqrt(1 - x^2 - y^2). That makes a
    // two-channel map work unchanged, and on an 8-bit three-channel one it is
    // more accurate than the stored blue.
    rhi::TextureId normal_map;
    // Scales the tangent-space slope before z is rebuilt, so 0 is flat, 1 is
    // the map as authored and 2 is twice as deep. Scaling the unpacked vector
    // instead would rotate it toward the surface normal and saturate.
    float normal_strength = 1.0f;

    // Alpha blended, drawn after every opaque object, back to front, and NOT
    // writing depth. Glass needs all four of those or it stops looking like
    // glass — or worse, stops the room behind it from drawing at all.
    //
    // The back-to-front sort is by OBJECT, and that is exactly as far as it
    // goes: two transparent objects that intersect have no single correct
    // order, so one pops in front of the other along the intersection line.
    // Renderer::SetOrderIndependentTransparency replaces the sort entirely.
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
// A colourist's controls, applied after the tone map.
//
// The defaults are a NO-OP: unit gain, unit gamma, no lift, contrast and
// saturation at one. That matters more than it sounds -- a grade whose defaults
// change the picture makes every existing test's expected colour wrong, and
// makes "is this the grade or the lighting" unanswerable.
// Where the composite's output is going. The tone curve has to know: an SDR
// frame throws its highlights away as it is written, and nothing downstream can
// put them back.
enum class DisplayOutput : std::uint8_t {
    // The ACES curve clamped to 0..1 and gamma-encoded. A conventional display.
    Sdr,
    // Extended-range linear, which is what scRGB and Apple's EDR are: no clamp
    // and no encoding, 1.0 IS reference white, and a highlight at 6.0 stays at
    // six times reference white all the way to the panel.
    ExtendedLinear,
    // Rec.2100 PQ -- SMPTE ST 2084 -- which is what an HDR10 signal carries.
    // Its input is ABSOLUTE luminance, so `reference_white_nits` is not a
    // preference here, it decides how bright the picture actually is.
    Pq,
};

struct ColorGrade {
    // Per-channel three-way control. Lift moves the blacks, gain the whites,
    // gamma the middle without moving either end.
    Vec3 lift{0.0f, 0.0f, 0.0f};
    Vec3 gamma{1.0f, 1.0f, 1.0f};
    Vec3 gain{1.0f, 1.0f, 1.0f};
    // The same three applied to every channel at once, for the common case of
    // "a bit brighter" without touching the balance.
    float lift_all = 0.0f;
    float gamma_all = 1.0f;
    float gain_all = 1.0f;

    float contrast = 1.0f;
    // The value contrast pivots about, in display-referred terms. 0.435 is 18%
    // grey after the ACES curve; pivoting about zero would make raising the
    // contrast darken the whole image.
    float contrast_pivot = 0.435f;
    float saturation = 1.0f;
    // Warm above zero, cool below. Not a colour temperature in kelvin: this is
    // a relative shift, and a kelvin value would imply a white point the rest
    // of the pipeline does not track.
    float temperature = 0.0f;
    float tint = 0.0f;  // green against magenta

    // --- display ---------------------------------------------------------------
    DisplayOutput output = DisplayOutput::Sdr;
    // How much brighter than reference white the display can go. 1 is an SDR
    // panel; a typical HDR laptop screen manages 3 or 4 and a good television
    // 10. The highlight roll-off is built from this, so setting it higher than
    // the panel can do puts the highlights where the panel clips them itself --
    // abruptly, and in one channel at a time, which is how HDR content ends up
    // with magenta suns.
    float display_headroom = 4.0f;
    // Where the roll-off starts, in multiples of reference white. Below it the
    // mapping is the IDENTITY, so an SDR-looking image on an HDR display looks
    // the same rather than washed out -- which is the failure everyone
    // remembers from the first generation of HDR games.
    float rolloff_start = 1.0f;
    // What "1.0" is worth, in nits. Only PQ uses it, and only because PQ is an
    // absolute encoding: 203 is the value the ITU recommends for graphics white
    // in an HDR10 signal.
    float reference_white_nits = 203.0f;
};

class Renderer {
  public:
    // The environment probe every lit pass will sample. Set once after baking;
    // an unset environment leaves the probe textures unbound, and the shader
    // falls back to the two-colour hemisphere rather than sampling nothing.
    //
    // Held here rather than passed per draw because it is a property of the
    // SCENE's surroundings, not of any one object, and threading it through
    // every draw call would put an IBL parameter in the signature of a shadow
    // pass that has no use for one.
    void SetEnvironment(const EnvironmentBindings&);
    void ClearEnvironment();
    [[nodiscard]] bool HasEnvironment() const;

    // `samples` is the multisample count the SCENE passes will run at. It has
    // to be known here rather than per-pass: a pipeline is compiled against a
    // sample count and Metal rejects a mismatch, so the renderer builds its
    // lit and shadow pipelines for one value and keeps to it.
    //
    // The fullscreen passes are always single-sampled — they run after the
    // resolve, on an ordinary texture.
    [[nodiscard]] static std::unique_ptr<Renderer> Create(rhi::Device&,
                                                          rhi::Format color,
                                                          std::string& error,
                                                          int samples = 1);
    // What Create was given. Apps need it to size their scene targets to match.
    [[nodiscard]] int Samples() const;
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
    // Depth maps for the scene's local lights, all into one atlas.
    //
    // ONE pass, tiled, rather than a pass per light: a shadowed light otherwise
    // costs a render target, a pass and a texture binding each, and the shader
    // would need an array of samplers whose length is fixed at compile time.
    //
    // Must run before DrawScene, into a pass whose depth target is
    // ShadowAtlas(). Assigns tiles in scene order and stops when they run out —
    // a light that cannot fit is drawn unshadowed rather than dropped, and a
    // point light needs all six or none: five faces of a cube shadow is worse
    // than no cube shadow, because the sixth direction is lit through walls.
    void DrawLightShadows(rhi::Encoder&, const Scene&);

    // The atlas texture, created lazily on the first call. Null before that.
    [[nodiscard]] rhi::TextureId ShadowAtlas();
    // Lights that were given tiles by the most recent DrawLightShadows.
    [[nodiscard]] int ShadowedLightCount() const;
    // Tiles those lights consumed: one per spot, six per point light.
    [[nodiscard]] int ShadowTilesUsed() const;

    // What a scene target must be. HDR, because the tone map now happens in
    // the composite rather than in every surface shader — which is what lets
    // the passes in between see how bright a pixel really was.
    static constexpr rhi::Format kSceneFormat = rhi::Format::RGBA16Float;

    // 4x4 tiles of 1024. Sixteen because a POINT light needs six of them —
    // one per cube face — so a 2x2 atlas could not hold a single one.
    // What a directional shadow map should be created at. Cascades tile it, so
    // four of them get 1024 each.
    static constexpr int kDirectionalShadowSize = 2048;

    static constexpr int kShadowAtlasSize = 4096;
    static constexpr int kShadowTilesPerSide = 4;
    static constexpr int kShadowTiles = kShadowTilesPerSide * kShadowTilesPerSide;

    // Depth-only pass from the light's point of view. Must run BEFORE
    // DrawScene, into a pass whose depth target is `shadow_map`.
    //
    // Does NOT frustum-cull: an object outside the CAMERA's view can still cast
    // a shadow into it, so culling here with the camera's frustum would make
    // shadows pop as you turn around.
    void DrawShadow(rhi::Encoder&, const Scene&);

    // Depth only, from the CAMERA. A depth prepass, and the reason it exists
    // here: SSAO reads depth back, and a multisample depth buffer cannot be
    // resolved into one that means anything — averaging two distances across a
    // silhouette gives a value describing no surface. So when the colour pass
    // runs multisampled, the occlusion pass gets its own single-sampled depth.
    void DrawSceneDepth(rhi::Encoder&, const Scene&, int width, int height);

    // `shadow_map` is the depth target DrawShadow wrote, or a null handle for
    // no shadows.
    void DrawScene(rhi::Encoder&, const Scene&, int width, int height,
                   rhi::TextureId shadow_map = {});

    // --- deferred ------------------------------------------------------------
    //
    // Two passes instead of one. DrawGBuffer rasterises the geometry and writes
    // what each surface IS -- albedo and roughness into attachment 0, world
    // normal and metallic into attachment 1 -- and DrawDeferredLight reads that
    // back with the depth buffer and lights every pixel exactly once.
    //
    // The point is that the cost of lighting stops depending on the geometry.
    // Forward shades every fragment it rasterises, including the ones a later
    // triangle covers, and runs the whole light loop for each; deferred runs it
    // once per pixel that survived.
    //
    // Both paths call the same ShadeSurface in the shader, so the two produce
    // the same picture by construction rather than by agreement.
    //
    // It is not free. There is no transparency here (a G-buffer holds one
    // surface per pixel; glass needs two, so transparent geometry is drawn
    // forward afterwards) and no MSAA (storing and lighting every sample costs
    // four times the memory and four times the lighting, which is the thing
    // deferring was for).
    void DrawGBuffer(rhi::Encoder&, const Scene&, int width, int height);

    // `depth` must be the SAMPLEABLE depth target the G-buffer pass wrote.
    void DrawDeferredLight(rhi::Encoder&, const Scene&, int width, int height,
                           rhi::TextureId albedo_rough, rhi::TextureId normal_metal,
                           rhi::TextureId depth, rhi::TextureId shadow_map = {});

    // --- GPU-driven drawing ---------------------------------------------------
    //
    // The ordinary path submits one draw per object and tests six planes per
    // object on the CPU. Both scale with the object count and neither has
    // anything to do with how many triangles reach the screen. This groups the
    // scene's opaque instances by mesh and material, culls each group on the
    // GPU, and emits ONE indirect draw per group -- whose instance count comes
    // from the buffer the cull wrote, so the CPU never learns it and never
    // stalls to find out.
    //
    // CullScene runs in a compute pass and DrawSceneIndirect in the render pass
    // after it. They are separate calls because a compute pass and a render
    // pass cannot be open at once.
    //
    // Limits, and they are the reason this does not simply replace DrawScene:
    //   * OPAQUE only. Blending needs back-to-front order and there is none
    //     here -- the survivors come out in whatever order threads finished.
    //   * No skinning. A skinned mesh has a joint palette per instance.
    //   * The CPU cannot know what was drawn, so LastDrawOrder() is empty.
    [[nodiscard]] int CullScene(rhi::ComputeEncoder&, const Scene&, int width,
                                int height);
    void DrawSceneIndirect(rhi::Encoder&, const Scene&, int width, int height,
                           rhi::TextureId shadow_map = {});
    // How many instances each batch was given, and how many draws it took.
    // The pair is the whole point: 5000 and 3.
    [[nodiscard]] int LastBatchCount() const;
    [[nodiscard]] int LastInstanceCount() const;

    // How many instances the GPU cull actually kept, summed over the batches.
    //
    // This READS BACK a buffer the GPU wrote, so it is only meaningful after
    // the frame has completed, and calling it every frame gives up the thing
    // indirect drawing was for -- the CPU not having to know. It exists so a
    // test can check that the cull culls, which is otherwise unobservable:
    // a cull that keeps everything draws the same picture, because the extra
    // instances are off screen and rasterise nothing.
    [[nodiscard]] int VisibleAfterCull() const;
    // How many survivors landed on one level of detail. A DIAGNOSTIC and not a
    // number to render from: it reads the buffer the GPU wrote, so it is only
    // meaningful after the frame has completed.
    [[nodiscard]] int VisibleAtLod(int lod) const;
    // Triangles the indirect draws will actually submit, summed over every
    // level. The number LOD exists to move, and the only one that shows whether
    // it did: the survivor count is unchanged by levels of detail.
    [[nodiscard]] long long IndirectTriangles() const;

    // --- ray tracing ---------------------------------------------------------
    //
    // Builds a bottom-level structure per registered mesh and a top-level one
    // over the scene's instances. Call it after the scene's transforms change;
    // the per-mesh structures are reused, which is the whole reason the split
    // exists.
    //
    // Returns false and leaves the previous structure in place if the device
    // has no ray tracing, so a caller can ask and fall back.
    // --- compute skinning ----------------------------------------------------
    //
    // Poses a skinned mesh into a buffer, so that the posed triangles exist
    // somewhere other than inside the rasteriser. Ray tracing needs that (an
    // acceleration structure is built from a buffer, and one built from the
    // bind pose casts a standing character's shadow while it walks), and so
    // does anything that wants to collide against the posed mesh.
    //
    // Must be called between BeginFrame and the first render pass: a compute
    // pass and a render pass cannot be open at the same time.
    //
    // Returns the number of meshes posed. Instances sharing a mesh AND a pose
    // are done once; instances of the same mesh at different poses each get
    // their own output, because they are different geometry.
    int SkinToBuffers(rhi::ComputeEncoder&, const Scene&);

    // The posed vertex buffer for a scene instance, or a null handle if that
    // instance is not skinned or has not been posed this frame.
    [[nodiscard]] rhi::BufferId PosedVertices(int instance_index) const;

    // Why SkinToBuffers posed nothing, if it was the pipeline's fault. Empty
    // otherwise. The compute pipeline is built lazily, so its compile error
    // arrives mid-frame from a function with no error parameter -- and without
    // this, a shader that does not compile looks exactly like a scene with no
    // skinned meshes in it.
    [[nodiscard]] const std::string& SkinError() const;

    [[nodiscard]] bool BuildSceneAccel(const Scene&, std::string& error);
    [[nodiscard]] bool RaytracingAvailable() const;

    // How many bottom-level structures BuildSceneAccel has actually BUILT,
    // cumulatively. Builds, not live structures: a two-level structure exists
    // so that a thousand instances of one mesh cost one BVH and a thousand
    // transforms, and a version that rebuilt per instance would still end up
    // with one per mesh in the table. Counting the table cannot see the
    // difference; counting the builds can, and nothing else would ever notice,
    // because the picture is identical either way.
    [[nodiscard]] int BlasBuilds() const;

    // One shadow ray per pixel, at the directional light. Writes visibility in
    // 0..1 -- the same thing a shadow-map lookup produces, so it goes in the
    // same place. `normals` is the G-buffer's normal target; the ray needs it
    // to offset its origin off the surface it starts on.
    void DrawRayShadows(rhi::Encoder&, const Scene&, int width, int height,
                        rhi::TextureId depth, rhi::TextureId normals);
    void DrawTriangle(rhi::Encoder&, int width, int height);

    // Fullscreen pass that samples `src` (a colour target written earlier this
    // frame) and applies a vignette. Needs no vertex buffer — the triangle is
    // generated from the vertex id.
    // `ao` may be null, in which case a 1x1 white texture is bound and the
    // multiply is a no-op.
    // Tone maps, and optionally adds ambient occlusion, bloom and a vignette.
    //
    // The tone map is not optional and is the reason this pass exists: surface
    // shading now writes linear HDR, so SOMETHING has to bring it down to a
    // display. `vignette` is: it is a look, and a measurement path wants the
    // image the renderer produced rather than a stylised one.
    void DrawComposite(rhi::Encoder&, rhi::TextureId src, rhi::TextureId ao = {},
                       rhi::TextureId bloom = {}, float bloom_strength = 0.0f,
                       float vignette = 1.0f);

    // The colour grade the composite applies, and where its exposure comes
    // from. Set once, or animated; the composite reads whatever is here.
    //
    // Held on the renderer rather than passed to DrawComposite because a grade
    // is a property of the LOOK, not of one call -- and threading eight more
    // parameters through a function that already takes five would make the
    // common call unreadable to configure something that changes once a scene.
    // A mesh with LEVELS OF DETAIL, coarsest last. Level 0 is the one every
    // non-GPU-driven path uses, so a mesh uploaded this way behaves exactly
    // like an ordinary one everywhere except in DrawSceneIndirect.
    //
    // At most four levels; extras are ignored. geom::BuildLodChain produces a
    // suitable list from one mesh.
    [[nodiscard]] MeshHandle UploadMeshLods(std::span<const Mesh> levels);
    [[nodiscard]] int MeshLodCount(MeshHandle) const;
    [[nodiscard]] int MeshLodIndexCount(MeshHandle, int lod) const;

    // --- occlusion culling ----------------------------------------------------
    //
    // Builds the depth pyramid the GPU cull reads. `depth` must be a SAMPLEABLE
    // depth target that already holds this frame's scene -- a depth prepass, or
    // a previous pass with keep_depth.
    //
    // THIS FRAME'S DEPTH, not last frame's, which is the other common design.
    // Reusing last frame's needs no prepass and costs a frame of latency, and
    // that latency is visible: rotate quickly and objects that should have
    // appeared at the edge of an occluder are culled for a frame. Requiring a
    // prepass costs a geometry pass with no fragment shader, and buys an exact
    // answer plus zero overdraw in the pass that follows.
    //
    // The pyramid holds the FARTHEST depth in each region, which is the minimum
    // value under reversed-Z. An averaged pyramid -- what a blit-generated mip
    // chain gives -- is not conservative and culls visible geometry.
    void BuildHiZ(rhi::ComputeEncoder&, rhi::TextureId depth, int width,
                  int height);
    // On by default, but inert until BuildHiZ has run this frame: the cull
    // checks that the pyramid is current rather than trusting a stale one.
    void SetOcclusionCulling(bool);
    [[nodiscard]] bool OcclusionCulling() const;
    [[nodiscard]] rhi::TextureId HiZ() const;

    // Screen radius in PIXELS at which each level of detail takes over: x is
    // where level 1 starts, y level 2, z level 3.
    //
    // In pixels rather than in metres because that is the quantity that decides
    // whether the detail can be seen. A distance threshold has to be re-tuned
    // for every object size and again whenever the field of view changes; one
    // set of screen sizes works for a pebble and a mountain at once.
    void SetLodThresholds(Vec3 pixels);
    [[nodiscard]] Vec3 LodThresholds() const;

    void SetGrade(const ColorGrade&);
    [[nodiscard]] const ColorGrade& Grade() const;
    // A buffer holding one float, the linear exposure multiplier. Null means a
    // fixed exposure of 1. PostStack::ExposureBuffer() is what usually goes
    // here, so an automatic exposure never round-trips through the CPU.
    void SetExposureBuffer(rhi::BufferId);

    // BLOOM, in three stages the caller sequences through the render graph.
    //
    // `threshold` and `knee` are in LINEAR radiance, which is why the scene
    // target has to be half-float: on an eight-bit one every bright thing has
    // already been clamped to one and a threshold above that selects nothing.
    void DrawBloomBright(rhi::Encoder&, rhi::TextureId src, float threshold,
                         float knee);
    // One axis. `texel` is the step between taps in uv — (1/width, 0) for the
    // horizontal pass, (0, 1/height) for the vertical. Two one-dimensional
    // passes, because a separable kernel costs 2n taps where the flat one costs
    // n squared.
    void DrawBloomBlur(rhi::Encoder&, rhi::TextureId src, float texel_x,
                       float texel_y);

    // Fullscreen occlusion pass. `depth` must be a sampleable depth target
    // written by an earlier pass with the SAME camera.
    void DrawSsao(rhi::Encoder&, const Camera&, int width, int height,
                  rhi::TextureId depth, float radius = 1.1f);

    // ORDER-INDEPENDENT TRANSPARENCY for every transparent material.
    //
    // Off by default. It is an approximation and it shows where alphas are
    // high -- a stack of nearly opaque sheets comes out too uniform, because
    // the depth weight cannot fully express "the front one hides the rest".
    // A scene whose transparency is a few well-separated panes is better served
    // by the sort; a scene with intersecting glass, foliage or smoke is not
    // served by the sort at all.
    void SetOrderIndependentTransparency(bool on);
    [[nodiscard]] bool OrderIndependentTransparency() const;
    // Draws only the TRANSPARENT instances, into the two accumulation targets.
    // Both must be cleared first: accumulation to (0,0,0,0) and revealage to
    // (1,1,1,1), because revealage is a running product and starting it at zero
    // makes every pixel fully covered before anything is drawn.
    void DrawTransparentOit(rhi::Encoder&, const Scene&, int width, int height,
                            rhi::TextureId shadow_map = {});
    // Resolves them into one premultiplied colour. Blend it over the opaque
    // frame with (ONE, ONE_MINUS_SRC_ALPHA).
    void DrawOitResolve(rhi::Encoder&, rhi::TextureId accum,
                        rhi::TextureId revealage);

    // A BAKED IRRADIANCE VOLUME, replacing the hemisphere-ambient guess with
    // measured indirect light. Uploads three 3D textures; pass an empty volume
    // to go back to the guess.
    //
    // It replaces the DIFFUSE ambient rather than adding to it -- the bake
    // already integrated the sky, so adding would count it twice and the room
    // would come out about twice as bright as the bake says. The specular half
    // stays, because SH-L1 cannot represent a reflection and dropping it would
    // make every metal surface black indoors.
    [[nodiscard]] bool SetIrradianceVolume(const IrradianceVolume&,
                                           std::string& error);
    void ClearIrradianceVolume();
    [[nodiscard]] bool HasIrradianceVolume() const;

    // CLUSTERED LIGHTING. Cuts the view frustum into a 16x9x24 grid, works out
    // on the GPU which lights touch which cell, and lets each fragment read
    // only its own cell's list.
    //
    // Off by default, and that is not timidity: below a few dozen lights the
    // binning pass costs more than the loop it saves, because a fragment
    // already rejects an out-of-range light with one distance test. It earns
    // its place when the light count climbs past the point where that test
    // itself is the cost.
    //
    // Must be followed by BinLights() inside a compute pass each frame, before
    // any pass that shades. Draws silently fall back to the whole light buffer
    // if the bins were never built -- wrong, but only in being slow.
    void SetClusteredLighting(bool on);
    [[nodiscard]] bool ClusteredLighting() const;
    // `far_distance` is how deep the grid reaches. Beyond it, fragments use the
    // last slice, so they see the lights of a cell that may be much larger than
    // they are -- over-lit rather than unlit, which is the failure to prefer.
    void BinLights(rhi::ComputeEncoder&, const Scene&, int width, int height,
                   float far_distance = 200.0f);
    // How many lights the busiest cell holds, and how many cells overflowed,
    // after the most recent BinLights. A readback, so it costs a stall -- for
    // a HUD or a test, not for a frame.
    struct ClusterStats {
        int max_per_cell = 0;
        int occupied_cells = 0;
        int overflowed_cells = 0;
        double mean_per_occupied = 0.0;
        // Which depth slice the busiest cell is in. Almost always the LAST one,
        // and that is by design rather than by accident: fragments past the
        // grid's far distance are clamped into the final slice, so it stands in
        // for everything from there to the horizon and collects the lights of a
        // volume with no end. Worth reporting, because "the busiest cell is
        // slice 23 of 24" and "the busiest cell is slice 4" mean completely
        // different things about how the grid is configured.
        int max_slice = -1;
    };
    [[nodiscard]] ClusterStats ReadClusterStats();

    // STEREO RENDERING, into a two-layer target, in ONE pass.
    //
    // `right` is the second eye's camera. The scene's own camera is the left
    // eye, so a monoscopic caller changes nothing.
    //
    // One pass and not two. The vertex stage runs once and emits both views, so
    // the draw calls, the state changes, the index fetches and the frustum
    // culling are all paid once -- and the geometry submitted for the two eyes
    // is identical anyway, which is why paying twice for it is pure waste.
    void DrawSceneStereo(rhi::Encoder&, const Scene&, const Camera& right,
                         int width, int height, rhi::TextureId shadow_map = {});

    // The light list and the cascade block this frame, for a pass outside the
    // renderer that has to agree with it.
    //
    // Froxel volumetrics is the caller: a shaft of light has to line up exactly
    // with the shadow that casts it, and reading the same cascade matrices at
    // the same offsets is the only way to guarantee that. Filling a second copy
    // would put the shaft's edge somewhere the geometry's shadow is not, by a
    // fraction of a texel, on every frame.
    //
    // VALID ONLY after a DrawScene this frame -- both are slices of a ring
    // buffer that DrawScene allocates. Before then they point at the previous
    // frame's contents.
    [[nodiscard]] rhi::BufferId LightBuffer() const;
    [[nodiscard]] std::size_t LightOffset() const;
    [[nodiscard]] rhi::BufferId CascadeBuffer() const;
    [[nodiscard]] std::size_t CascadeOffset() const;

    // Anisotropic filtering for MATERIAL textures, 1 to 16. Rebuilds the
    // sampler, so it is a settings change and not a per-frame one.
    //
    // A knob rather than a constant because it is a quality/bandwidth trade
    // like any other -- and because a hard-coded value cannot be tested: the
    // only way to show anisotropy is doing something is to render the same
    // frame without it.
    void SetAnisotropy(int max_anisotropy);
    [[nodiscard]] int Anisotropy() const;

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
