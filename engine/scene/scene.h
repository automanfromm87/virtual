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

    [[nodiscard]] Mat4 ViewProj(float aspect) const;
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

// One drawable placement: which mesh, where, what colour. The scene refers to
// the mesh by HANDLE and never sees a vertex buffer — that is what keeps this
// layer independent of the renderer.
struct Instance {
    MeshHandle mesh = kMeshSphere;
    MaterialHandle material = kMaterialLit;
    Mat4 model = Mat4::Identity();
    Vec4 tint = Vec4{1.0f, 1.0f, 1.0f, 1.0f};
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
    // A single fixed box is the simplest thing that works. It is also the thing
    // that stops working the moment a scene is bigger than the box — which is
    // what cascaded shadow maps exist to fix.
    float shadowExtent = 0.0f;

    // SECTION CUT. Fragments above this world Y are discarded, which is how you
    // see into a building without deleting its roof. Storeys stack, so cutting
    // by height is also how you show one floor at a time.
    float clipY = 1.0e9f;

    // World -> the light's clip space, for the shadow pass and the lookup.
    [[nodiscard]] Mat4 LightViewProj() const;
    std::vector<Instance> instances;
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
