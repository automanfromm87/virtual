// Pure C++20. What to draw, described independently of how it gets drawn —
// no Metal, no vertex buffers, no GPU handles. engine/render consumes this.
#pragma once

#include <cstdint>
#include <vector>

#include "engine/core/math.h"
#include "engine/resource/handles.h"

namespace eng {

enum class Projection : std::uint8_t {
    Perspective,   // walkthrough
    Orthographic,  // the plan view an architect actually wants
};

struct Camera {
    Vec3 eye{0.0f, 0.0f, 4.5f};
    Vec3 target{0.0f, 0.0f, 0.0f};
    Vec3 up{0.0f, 1.0f, 0.0f};
    float fovY = 1.0472f;  // 60 degrees
    float nearZ = 0.1f;    // reversed-Z, infinite far plane

    Projection projection = Projection::Perspective;
    // Half-height of the orthographic view volume, in world metres. Ignored
    // under perspective.
    float orthoHeight = 8.0f;
    float orthoFar = 200.0f;

    // SUB-PIXEL JITTER, in NDC units (so a whole pixel is 2/width). Added to
    // the projection's translation, which shifts every sample by a fraction of
    // a pixel without changing anything else about the view.
    //
    // This is the input to temporal antialiasing and it is useless without it:
    // jitter alone just makes the image shimmer. TAA is what turns a different
    // sample position each frame into a higher effective resolution, by
    // accumulating them. Zero disables it, which is the right default -- a
    // renderer that jitters with nothing to resolve the jitter looks broken.
    Vec2 jitter{0.0f, 0.0f};

    // With and without the jitter. The unjittered one is what a REPROJECTION
    // must use: velocity is where a surface moved between frames, and a
    // sub-pixel wobble applied to only one of the two matrices would show up in
    // the answer as motion that did not happen.
    [[nodiscard]] Mat4 ViewProj(float aspect) const;
    [[nodiscard]] Mat4 ViewProjNoJitter(float aspect) const;
};

// First-person walkthrough. An orbit camera cannot show you a room from the
// inside: it always looks AT a point, so the moment you are within arm's reach
// of that point the geometry is behind you. Standing somewhere and turning your
// head is a different motion entirely.
struct WalkController {
    Vec3 position{0.0f, 1.6f, 0.0f};  // eye height, roughly
    float yaw = 0.0f;
    float pitch = 0.0f;

    void Look(float dx, float dy);
    // `forward` and `right` are -1, 0 or +1; movement stays in the horizontal
    // plane, so looking down does not walk you into the floor.
    void Move(float forward, float right, float dt);
    void Apply(Camera& cam) const;
};

// Turntable camera: drag to orbit, scroll to dolly. Enough for inspecting a
// building; a walkthrough would want WASD and a free look instead.
struct OrbitController {
    Vec3 target{0.0f, 0.0f, 0.0f};
    float yaw = 0.9f;
    float pitch = 0.55f;
    float distance = 16.0f;

    void Drag(float dx, float dy);
    void Zoom(float amount);
    // Writes eye/target into `cam`, leaving its projection settings alone.
    void Apply(Camera& cam) const;
};

enum class LightType : std::uint8_t { Point, Spot };

// A local light: one that has a POSITION, and therefore falls off with distance.
struct Light {
    LightType type = LightType::Point;
    Vec3 position{0.0f, 0.0f, 0.0f};
    // The direction the cone points, from the lamp INTO the scene. Only a spot
    // uses it.
    //
    // Deliberately the opposite sense to Scene::lightDir. A directional light
    // has no position, so the only thing worth recording is where it IS; a spot
    // has a position, so the only thing worth recording is where it AIMS. One
    // convention that means two things would be worse than two that are named.
    Vec3 direction{0.0f, -1.0f, 0.0f};
    // Radiance at one metre. Not a 0..1 colour: the shader tone maps, and a
    // lamp you can stand next to has to be able to blow out.
    Vec3 color{1.0f, 1.0f, 1.0f};
    // Distance at which it reaches exactly zero. Physical falloff is 1/d² and
    // never gets there, so a cutoff is what makes a light cullable at all.
    float range = 10.0f;
    // Full brightness inside the inner cone, nothing outside the outer one, a
    // smooth edge between. Ignored by a point light.
    float inner_degrees = 22.0f;
    float outer_degrees = 34.0f;

    // Asks for shadow map space. A spot needs one tile; a POINT light needs
    // six, because it casts in every direction and a single frustum covers at
    // most one of them.
    //
    // Not free: a shadowed spot adds one pass over the casters, a shadowed
    // point light six.
    bool casts_shadow = false;
    // How close to the lamp the shadow map starts. Too small wastes precision
    // on space nothing occupies; too large clips geometry near the bulb.
    float shadow_near = 0.20f;

    // World -> this light's clip space. For a spot. Meaningless for a point
    // light, which has six of them.
    [[nodiscard]] Mat4 ViewProj() const;

    // One face of a point light's cube, 0..5 in the order +X -X +Y -Y +Z -Z —
    // the order Metal and every other API number cube faces in, so a lookup
    // can pick a face from a direction without a translation table.
    [[nodiscard]] Mat4 CubeFaceViewProj(int face) const;
    // How many shadow tiles this light wants: 0, 1 for a spot, or 6.
    [[nodiscard]] int ShadowFaces() const {
        if (!casts_shadow) return 0;
        return type == LightType::Spot ? 1 : 6;
    }
};

// One drawable placement: which mesh, where, what colour. The scene refers to
// the mesh by HANDLE and never sees a vertex buffer — that is what keeps this
// layer independent of the renderer.
struct Instance {
    MeshHandle mesh = kMeshSphere;
    MaterialHandle material = kMaterialLit;
    Mat4 model = Mat4::Identity();
    Vec4 tint = Vec4{1.0f, 1.0f, 1.0f, 1.0f};
    // LOD CROSSFADE. The magnitude is this instance's share of the pixels, from
    // 0 to 1; the SIGN selects which half of the dither pattern it takes.
    //
    // A crossfade is two draws: the incoming level at +f and the outgoing one
    // at -(1 - f). Both numbers read as "my share", and the opposite signs are
    // what makes the two patterns exact complements -- every pixel covered once,
    // no holes and no doubling.
    //
    // The sign is not decoration. Giving both levels a positive fade makes the
    // patterns NESTED rather than complementary: the same "keep where fade is
    // above this pixel's threshold" test means the larger fade keeps a superset
    // of the smaller one, so the overlap is drawn twice and the rest not at
    // all. Measured on the first attempt at this: 3331 pixels doubled and 3324
    // holes out of 8880.
    //
    // What it is for: swapping one level of detail for another is a POP, and
    // the eye is extremely good at catching a silhouette that changes in one
    // frame. Drawing both levels for a moment with complementary fades -- one
    // at f, the other at 1-f -- turns the pop into a dissolve, and because the
    // two dither patterns are complements every pixel is covered exactly once.
    //
    // A DITHER and not alpha blending, which would need the two levels sorted
    // against each other and against everything else, would not write depth,
    // and would double-shade the overlap. Discarding is order-independent and
    // costs nothing but the discard.
    float lod_fade = 1.0f;
    // Offset into Scene::joint_matrices, or -1 for an unskinned instance. The
    // mesh knows how many joints to read.
    //
    // An offset into one flat array rather than a palette per instance: two
    // characters in the same pose still need two palettes, but the renderer
    // wants one contiguous upload, not a vector of vectors.
    int palette = -1;
};

struct Scene {
    Camera camera;
    // Unit vector pointing FROM the surface TOWARD the light, world space.
    Vec4 lightDir{0.4082f, 0.8165f, 0.4082f, 0.0f};
    // Radiance, not a 0..1 colour. A physically-scaled BRDF needs more than
    // unit energy to produce a visible highlight; the shader tone maps.
    Vec4 lightColor{3.0f, 3.0f, 3.0f, 1.0f};
    // Half-extent, in world units, of the orthographic box the shadow map
    // covers, centred on the origin. 0 disables shadows entirely.
    //
    // A single fixed box is the simplest thing that works, and it is what you
    // get with one cascade. Used as the extent of the LAST cascade when there
    // are several.
    float shadowExtent = 0.0f;

    // CASCADES. One shadow map has to choose between covering the distance and
    // having enough texels near the camera; a fixed box does the first and the
    // shadows come out blocky underfoot. Splitting the view into ranges and
    // giving each its own map spends the resolution where it is looked at.
    //
    // 1 is the old behaviour exactly: one box of `shadowExtent`, centred on the
    // origin.
    int shadowCascades = 1;
    // How far from the camera the cascades reach. Beyond it, unshadowed.
    float shadowDistance = 40.0f;
    // Blend between a purely logarithmic split and a uniform one. Logarithmic
    // is right for perspective — each cascade should cover a constant fraction
    // of screen space, and depth is not linear — but it puts the first split
    // absurdly close, so everyone mixes the two.
    float cascadeBlend = 0.75f;

    // Bounds of cascade `i`, in view distance from the camera.
    [[nodiscard]] float CascadeSplit(int i) const;
    // World -> cascade `i`'s clip space.
    [[nodiscard]] Mat4 CascadeViewProj(int i, float aspect) const;

    // Hemisphere ambient: cool from the sky, dim warm bounce from the ground,
    // blended by which way a normal points. The cheapest useful stand-in for
    // image-based lighting, and the thing that stops the underside of an object
    // being as bright as its top.
    //
    // A DIAL, not a constant. These defaults suit an outdoor scene; a room lit
    // by lamps needs them an order of magnitude lower, and leaving them high
    // there means the ambient does the lighting and the lamps are decoration.
    Vec3 ambientSky{0.13f, 0.16f, 0.24f};
    Vec3 ambientGround{0.055f, 0.045f, 0.035f};

    // SECTION CUT. Fragments above this world Y are discarded, which is how you
    // see into a building without deleting its roof. Storeys stack, so cutting
    // by height is also how you show one floor at a time.
    float clipY = 1.0e9f;

    // World -> the light's clip space, for the shadow pass and the lookup.
    [[nodiscard]] Mat4 LightViewProj() const;
    std::vector<Instance> instances;

    // Local lights, on top of the directional key light above.
    //
    // The split is a real limitation, not a taxonomy: the directional light is
    // the one with a shadow map, and everything in here is unshadowed. Giving a
    // point light shadows means a cube map per light and a pass per face, which
    // is the next thing this wants.
    std::vector<Light> lights;

    // Joint matrices for every skinned instance this frame, concatenated.
    // Already world-relative: palette[j] = jointWorld * inverseBind, so the
    // instance's model matrix still applies on top.
    std::vector<Mat4> joint_matrices;
};

// Two spheres that OVERLAP ON SCREEN at different depths, plus a cube above.
//
// The overlap is the entire point. A single convex object with back-face
// culling never needs a depth buffer — the near hemisphere is all that survives
// culling, so a completely broken depth test still looks perfect. Only mutual
// occlusion between separate objects exercises it.
//
// The cube is there to prove the mesh registry holds more than one thing.
[[nodiscard]] Scene ShapesDemo(float time_seconds);

// The same scene with instance `drop` removed, for differential testing: render
// with and without an occluder and compare how much of the others survives.
[[nodiscard]] Scene ShapesDemoWithout(float time_seconds, int drop);

// Spheres orbiting above a ground plane, with shadows enabled. The ground is a
// big CUBE rather than a squashed one because Mat4::Scale is uniform only —
// deliberately, since a non-uniform scale silently breaks normal transforms.
[[nodiscard]] Scene ShadowDemo(float time_seconds);

}  // namespace eng
