// The post-processing stack: exposure, fog, depth of field, motion blur and
// temporal antialiasing.
//
// WHY ONE CLASS. These share three things -- the HDR scene target, the depth
// buffer, and a strict order -- and separating them would mean each one
// rediscovering the others' resources and each caller getting the order right
// by hand. The order is the part that is easy to get wrong and hard to notice:
//
//   1. VELOCITY, from depth against the previous frame's view projection.
//   2. FOG, while everything is still linear and before bloom, so that a
//      distant lamp fogs and then blooms rather than blooming and then fogging.
//   3. DEPTH OF FIELD, after fog -- a blurred background must blur the fog that
//      is in front of it too.
//   4. MOTION BLUR, after depth of field, because a moving out-of-focus object
//      is both, and smearing a sharp image and then blurring it gives a
//      different and wrong answer.
//   5. EXPOSURE metering, on the finished HDR image.
//   6. TEMPORAL RESOLVE, last among the HDR passes, so that everything
//      accumulated is deterministic per frame. Bloom and the tone map come
//      after, in the composite.
//
// WHAT IS NOT HERE: bloom, ambient occlusion, the tone map and colour grading.
// Those live in the Renderer's composite, because they were there first and
// because the composite is where display-referred colour begins.
#pragma once

#include <memory>
#include <string>

#include "engine/core/math.h"
#include "engine/rhi/rhi.h"
#include "engine/scene/scene.h"

namespace eng {

struct PostConfig {
    // --- exposure --------------------------------------------------------------
    //
    // A camera that adapts. Off by default, because a fixed exposure is what
    // every test in this engine assumes and an automatic one would make each of
    // them depend on the average brightness of its own scene.
    bool auto_exposure = false;
    // The metered range, in log2 luminance. Outside it everything piles into the
    // end bins and the meter stops responding; -8 to +6 covers moonlight to
    // direct sun in the units this engine's lights use.
    float min_log_luminance = -8.0f;
    float max_log_luminance = 6.0f;
    // How fast the eye opens and closes, as an exponential rate per second.
    // Different in each direction on purpose: adapting to a brighter scene
    // takes seconds and to a darker one takes minutes, and a single rate is
    // visibly wrong one way or the other.
    float adapt_brighter = 3.0f;
    float adapt_darker = 0.8f;
    // A manual multiplier on whatever the meter decides, in stops.
    float exposure_compensation = 0.0f;
    // Used when auto_exposure is off.
    float fixed_exposure = 1.0f;

    // --- fog -------------------------------------------------------------------
    bool fog = false;
    Vec3 fog_color{0.55f, 0.62f, 0.72f};
    // Extinction per metre at the reference height. 0.01 is a light haze at a
    // hundred metres; 0.1 is thick.
    float fog_density = 0.012f;
    // How fast the density falls off with height, per metre. Zero is uniform
    // fog, which is the flat wash that reads as a slider rather than as air.
    float fog_height_falloff = 0.08f;
    float fog_ground_height = 0.0f;
    // Nothing closer than this is fogged, so the character in front of the
    // camera stays clear.
    float fog_start = 2.0f;
    // How far the sky counts as being, for fogging the background.
    float fog_max_distance = 400.0f;

    // --- depth of field --------------------------------------------------------
    bool depth_of_field = false;
    float focus_distance = 8.0f;
    // How far from the focal plane it takes to reach maximum blur.
    float focus_range = 14.0f;
    // The largest circle of confusion, in PIXELS -- so the look does not change
    // with the window size, which it would if this were in world units.
    float max_blur_radius = 12.0f;

    // --- motion blur -----------------------------------------------------------
    bool motion_blur = false;
    // The shutter, as a fraction of the frame. 0.5 is a 180-degree shutter,
    // which is the film convention and what most footage looks like.
    float shutter = 0.5f;

    // --- screen-space reflections ----------------------------------------------
    //
    // A LAYER over the environment probe, not a replacement. A screen-space
    // march can only reflect what is on the screen, so where it fails -- off
    // the edge of the frame, behind the reflector, behind the camera -- it
    // fades out and the probe shows through. Running it without a probe leaves
    // holes wherever the march misses.
    //
    // Needs the deferred path's normal buffer. There is no reconstruct-from-
    // depth fallback on purpose: depth derivatives give a normal that is wrong
    // along every silhouette, and a reflection is at its most visible exactly
    // there.
    bool ssr = false;
    // How far a ray travels, in world metres. The cost is linear in this and in
    // the step count together.
    float ssr_distance = 30.0f;
    int ssr_steps = 40;
    // How many halvings after the crossing step. Four takes the error to a
    // sixteenth of a step, below which the banding is gone.
    int ssr_refine_steps = 4;
    // How thick the depth buffer's surfaces are assumed to be. The depth buffer
    // says how far away a surface is and nothing about its depth, so this is
    // the guess that decides whether a ray passed BEHIND something or hit it.
    // Too small and reflections drop out behind thin geometry; too large and
    // rays attach to things they passed by.
    float ssr_thickness = 0.6f;
    float ssr_intensity = 1.0f;
    // Above this roughness a surface is left entirely to the probe: one mirror
    // ray cannot represent a wide lobe, and the probe's prefiltered chain
    // already can.
    float ssr_max_roughness = 0.4f;
    // How far from the edge of the frame the fade starts, in uv. A reflection
    // that stops dead at the viewport boundary is the most recognisable SSR
    // artefact there is.
    float ssr_edge_fade = 0.12f;

    // --- temporal antialiasing -------------------------------------------------
    //
    // Needs Camera::jitter to be driven from Jitter() below. Without the jitter
    // TAA is a pure blur: it averages frames that all sampled the same points.
    bool taa = false;
    // How much history survives each frame. 0.9 means a tenth of the image is
    // new, which converges in about twenty frames and trails visibly above 0.95.
    float taa_feedback = 0.9f;
    // Renders the scene below display resolution; the TAA resolve rebuilds
    // the display image out of sub-pixel-jittered history, which is temporal
    // upsampling rather than mere antialiasing. 1.0 is native. The history
    // and output stay at display size -- only the scene chain (and the
    // velocity, which is derived from its depth) runs small -- so at 1.0
    // every size below is what it always was, and this setting changes
    // nothing.
    float render_scale = 1.0f;
};

class PostStack {
  public:
    [[nodiscard]] static std::unique_ptr<PostStack> Create(
        rhi::Device&, std::string& error,
        rhi::Format hdr = rhi::Format::RGBA16Float);
    ~PostStack();

    PostStack(const PostStack&) = delete;
    PostStack& operator=(const PostStack&) = delete;

    // Call once per frame BEFORE building the graph. Sizes the internal targets,
    // advances the jitter sequence, and rolls the previous frame's matrices.
    void BeginFrame(const Camera&, int width, int height, float dt);

    // The sub-pixel offset this frame's camera should use, in NDC. Zero when TAA
    // is off. Assign it to Camera::jitter before anything computes a matrix from
    // the camera -- including BeginFrame, which is why this is separate.
    [[nodiscard]] Vec2 Jitter() const;

    // --- passes ---------------------------------------------------------------
    // Each returns the texture the next stage should read, so a caller chains
    // them and does not have to know which ones were enabled.
    void ComputeVelocity(rhi::ComputeEncoder&, rhi::TextureId depth);
    [[nodiscard]] rhi::TextureId Velocity() const;
    // The scene size BeginFrame derived from render_scale: the width and
    // height every scene-chain pass (and every Draw* width/height argument)
    // must use. At the default scale of 1.0 these are the BeginFrame size.
    [[nodiscard]] int RenderWidth() const;
    [[nodiscard]] int RenderHeight() const;

    // Blended over `target` in place. Needs a pass whose colour attachment is
    // the HDR image and which reads `depth`.
    void DrawFog(rhi::Encoder&, rhi::TextureId depth);
    void DrawDepthOfField(rhi::Encoder&, rhi::TextureId src, rhi::TextureId depth);
    void DrawMotionBlur(rhi::Encoder&, rhi::TextureId src);
    // ADDITIVE over the pass's existing colour, so it needs a pass with
    // `load` set. `normal_metal` is the deferred G-buffer's second attachment.
    void DrawSsr(rhi::Encoder&, rhi::TextureId scene, rhi::TextureId depth,
                 rhi::TextureId normal_metal, rhi::TextureId albedo_rough);
    // Resolves `src` against the history and writes the result. The caller must
    // target Output() so the history stays owned here.
    void DrawTaa(rhi::Encoder&, rhi::TextureId src);

    void MeterExposure(rhi::ComputeEncoder&, rhi::TextureId hdr);
    // The buffer holding one float: the exposure the composite should apply. A
    // BUFFER and not a returned value, because reading it on the CPU would mean
    // waiting for the GPU -- and the number is only ever consumed by a shader.
    [[nodiscard]] rhi::BufferId ExposureBuffer() const;
    // The last metered exposure, one to three frames stale. For a HUD only.
    [[nodiscard]] float LastExposure() const;

    // Scratch targets, sized to the window. Owned here so a caller chaining
    // passes does not have to allocate ping-pong pairs itself.
    [[nodiscard]] rhi::TextureId Scratch() const;
    [[nodiscard]] rhi::TextureId Output() const;
    // Rotates the history. Call once after the temporal resolve.
    void EndFrame();

    PostConfig config;

  private:
    PostStack();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eng
