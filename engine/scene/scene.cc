#include "engine/scene/scene.h"

#include <algorithm>
#include <cmath>

namespace eng {

Mat4 Camera::ViewProj(float aspect) const {
    const Mat4 view = Mat4::LookAt(eye, target, up);
    if (projection == Projection::Orthographic) {
        // Same reversed-Z convention as the perspective path, so the depth
        // state does not have to change when the projection does.
        return Mat4::OrthographicReverseZ(orthoHeight * aspect, orthoHeight,
                                          nearZ, orthoFar) *
               view;
    }
    return Mat4::PerspectiveReverseZ(fovY, aspect, nearZ) * view;
}

void WalkController::Look(float dx, float dy) {
    yaw -= dx * 0.006f;
    pitch = std::clamp(pitch - dy * 0.006f, -1.4f, 1.4f);
}

void WalkController::Move(float forward, float right, float dt) {
    // Horizontal only. Using the full view direction would send you through the
    // floor whenever you looked down while walking.
    const Vec3 fwd{std::cos(yaw), 0.0f, std::sin(yaw)};
    const Vec3 side{-std::sin(yaw), 0.0f, std::cos(yaw)};
    const float speed = 3.0f;  // metres per second, an unhurried walk
    position = position + fwd * (forward * speed * dt) + side * (right * speed * dt);
}

void WalkController::Apply(Camera& cam) const {
    const float cp = std::cos(pitch);
    cam.eye = position;
    cam.target = position + Vec3{std::cos(yaw) * cp, std::sin(pitch), std::sin(yaw) * cp};
    cam.projection = Projection::Perspective;
}

void OrbitController::Drag(float dx, float dy) {
    yaw -= dx * 0.008f;
    pitch += dy * 0.008f;
    // Stop just short of the poles: at exactly vertical the view direction is
    // parallel to `up` and LookAt degenerates.
    pitch = std::clamp(pitch, -1.5f, 1.5f);
}

void OrbitController::Zoom(float amount) {
    // Multiplicative, so a scroll notch moves the same PROPORTION of the way in
    // whether you are across the room or across the street.
    distance = std::clamp(distance * std::exp(-amount * 0.12f), 1.5f, 200.0f);
}

void OrbitController::Apply(Camera& cam) const {
    const float cp = std::cos(pitch);
    cam.eye = target + Vec3{std::cos(yaw) * cp, std::sin(pitch), std::sin(yaw) * cp} *
                           distance;
    cam.target = target;
    cam.orthoHeight = distance * 0.45f;  // zoom means the same thing in both modes
}

Mat4 Scene::LightViewProj() const {
    const Vec3 dir = Normalize(Vec3{lightDir.x, lightDir.y, lightDir.z});
    // lightDir points TOWARD the light, so the virtual camera sits out along it
    // and looks back at the origin.
    const float distance = shadowExtent * 2.0f;
    const Vec3 eye = dir * distance;

    // LookAt degenerates when forward is parallel to up. A light straight
    // overhead is the single most common case, so this is not a corner case.
    const Vec3 up = (std::fabs(dir.y) > 0.99f) ? Vec3{0.0f, 0.0f, 1.0f}
                                               : Vec3{0.0f, 1.0f, 0.0f};
    const Mat4 view = Mat4::LookAt(eye, Vec3{0.0f, 0.0f, 0.0f}, up);
    const Mat4 proj = Mat4::OrthographicReverseZ(shadowExtent, shadowExtent,
                                                 0.05f, distance * 2.0f);
    return proj * view;
}

namespace {

// T * R * S: scale, then spin, then place. Written right-to-left because that
// is the order the vertex actually travels through.
Mat4 Place(Vec3 at, float scale, float spin) {
    return Mat4::Translation(at) * Mat4::RotationY(spin) * Mat4::Scale(scale);
}

}  // namespace

Scene ShapesDemo(float time_seconds) {
    Scene s;
    const float spin = time_seconds * 0.6f;  // radians per second

    // index 0 — NEAR sphere. Big on screen, and never occluded by anything.
    s.instances.push_back({kMeshSphere, kMaterialLit,
                           Place({-0.35f, -0.15f, 1.2f}, 0.60f, spin),
                           Vec4{1.00f, 0.45f, 0.35f, 1.0f}});

    // index 1 — FAR sphere. Placed so its screen disc overlaps the near one;
    // the near sphere must eat a bite out of it. Centres are 2.5 apart in 3D,
    // so they do not actually intersect — this is pure depth ordering, not
    // interpenetration.
    s.instances.push_back({kMeshSphere, kMaterialLit,
                           Place({0.35f, 0.15f, -1.2f}, 0.60f, -spin),
                           Vec4{0.35f, 1.00f, 0.50f, 1.0f}});

    // index 2 — a CUBE above, mostly clear of the other two. Two-sided so it
    // exercises a second material that must still share the lit pipeline.
    s.instances.push_back({kMeshCube, kMaterialLitTwoSided,
                           Place({0.05f, 1.05f, -0.2f}, 0.42f, spin * 1.7f),
                           Vec4{0.70f, 0.55f, 1.00f, 1.0f}});

    return s;
}

Scene ShadowDemo(float time_seconds) {
    Scene s;
    const float t = time_seconds * 0.5f;
    s.camera.eye = Vec3{4.5f, 4.0f, 7.5f};
    s.lightDir = Vec4{0.45f, 0.80f, 0.40f, 0.0f};
    s.shadowExtent = 6.0f;

    // MakeCube(1.4) has a half-extent of 0.7; scaled by 8 that is 5.6, so a
    // centre at -6.6 puts the top face at y = -1.
    Instance ground;
    ground.mesh = kMeshCube;
    ground.model = Mat4::Translation({0.0f, -6.6f, 0.0f}) * Mat4::Scale(8.0f);
    ground.tint = Vec4{0.78f, 0.78f, 0.80f, 1.0f};
    s.instances.push_back(ground);

    const Vec4 tints[3] = {{1.00f, 0.45f, 0.35f, 1.0f},
                           {0.35f, 1.00f, 0.50f, 1.0f},
                           {0.55f, 0.60f, 1.00f, 1.0f}};
    for (int i = 0; i < 3; ++i) {
        const float phase = t + float(i) * 2.0944f;  // 120 degrees apart
        Instance sphere;
        // Bobbing, but the bottom of the arc TOUCHES the ground: the top face
        // is at y = -1 and the world radius is 0.8, so -0.2 is tangent.
        // Animation that never makes contact reads as floating no matter how
        // good the shading is.
        const float bob = std::fabs(std::sin(t * 1.7f + float(i))) * 0.55f;
        sphere.model = Mat4::Translation({std::cos(phase) * 2.0f, -0.2f + bob,
                                          std::sin(phase) * 2.0f}) *
                       Mat4::RotationY(t * 1.3f) * Mat4::Scale(0.8f);
        sphere.tint = tints[i];
        s.instances.push_back(sphere);
    }
    return s;
}

Scene ShapesDemoWithout(float time_seconds, int drop) {
    Scene s = ShapesDemo(time_seconds);
    if (drop >= 0 && drop < int(s.instances.size()))
        s.instances.erase(s.instances.begin() + drop);
    return s;
}

}  // namespace eng
