#include "engine/scene/scene.h"

#include <algorithm>
#include <cmath>

namespace eng {

Mat4 Camera::ViewProjNoJitter(float aspect) const {
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

Mat4 Camera::ViewProj(float aspect) const {
    Mat4 vp = ViewProjNoJitter(aspect);
    if (jitter.x == 0.0f && jitter.y == 0.0f) return vp;
    // ROW 0 += jitter.x * ROW 3, and likewise for y.
    //
    // What that does is clip.xy += jitter * clip.w, which is a constant shift
    // in NDC AFTER the perspective divide -- so every fragment moves by the
    // same fraction of a pixel whatever its depth. That is the property TAA
    // needs: a jitter that varied with distance would be a shear, and
    // accumulating sheared frames converges on a blur rather than on detail.
    //
    // Under an orthographic projection w is 1 and this reduces to adding the
    // offset to the last column, which is the same shift by a different route.
    // Perturbing the projection's SCALE instead -- the tempting one-liner --
    // would change the field of view by a fraction of a pixel each frame, which
    // is a zoom wobble, not a sample offset.
    for (int i = 0; i < 4; ++i) {
        vp.col[i].x += jitter.x * vp.col[i].w;
        vp.col[i].y += jitter.y * vp.col[i].w;
    }
    return vp;
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

Mat4 Light::ViewProj() const {
    const Vec3 forward = Normalize(direction);
    // LookAt degenerates when forward is parallel to up, and a lamp aimed
    // straight down is the single most common case in a room.
    const Vec3 up = (std::fabs(forward.y) > 0.99f) ? Vec3{0.0f, 0.0f, 1.0f}
                                                   : Vec3{0.0f, 1.0f, 0.0f};
    const Mat4 view = Mat4::LookAt(position, position + forward, up);
    // The full cone, plus a margin: a fragment exactly on the cone boundary
    // must still land inside the map, or the pool's edge samples nothing and
    // the rim of every spot goes unshadowed.
    const float fov = std::min(outer_degrees * 2.2f, 170.0f) * 3.14159265f / 180.0f;
    // PERSPECTIVE, not orthographic: a spot's rays diverge from a point. The
    // directional light gets an ortho box for exactly the opposite reason.
    return Mat4::PerspectiveReverseZ(fov, 1.0f, shadow_near) * view;
}

Mat4 Light::CubeFaceViewProj(int face) const {
    // The six axis directions, in the standard cube-face order. The `up`
    // vectors are the conventional ones: what matters is only that each is
    // perpendicular to its forward, and that the SAME choice is used here and
    // in the shader — disagree and the lookup samples a rotated copy of the
    // right face, which reads as a shadow that slides as the light turns.
    static const Vec3 kForward[6] = {{1, 0, 0},  {-1, 0, 0}, {0, 1, 0},
                                     {0, -1, 0}, {0, 0, 1},  {0, 0, -1}};
    static const Vec3 kUp[6] = {{0, -1, 0}, {0, -1, 0}, {0, 0, 1},
                                {0, 0, -1}, {0, -1, 0}, {0, -1, 0}};
    const int f = (face < 0 || face > 5) ? 0 : face;
    const Mat4 view = Mat4::LookAt(position, position + kForward[f], kUp[f]);
    // Exactly ninety degrees, so the six frusta tile the sphere with no gap and
    // no overlap. Anything else leaves seams or wastes resolution.
    return Mat4::PerspectiveReverseZ(1.5707963f, 1.0f, shadow_near) * view;
}

float Scene::CascadeSplit(int i) const {
    const int n = std::clamp(shadowCascades, 1, 4);
    if (i <= 0) return camera.nearZ;
    if (i >= n) return shadowDistance;
    const float near = std::max(camera.nearZ, 0.01f);
    const float far = std::max(shadowDistance, near * 2.0f);
    const float f = float(i) / float(n);
    // Logarithmic wants each cascade to cover a constant ratio of depth;
    // uniform wants a constant slice. Neither alone is usable: the first puts
    // split zero at a few centimetres, the second wastes the near cascade on
    // distance nothing is looking at.
    const float log_split = near * std::pow(far / near, f);
    const float uniform = near + (far - near) * f;
    return cascadeBlend * log_split + (1.0f - cascadeBlend) * uniform;
}

Mat4 Scene::CascadeViewProj(int i, float aspect) const {
    const int n = std::clamp(shadowCascades, 1, 4);
    if (n == 1) return LightViewProj();

    const float split_near = CascadeSplit(i);
    const float split_far = CascadeSplit(i + 1);

    // The eight corners of this slice of the camera frustum, in world space.
    const Vec3 forward = Normalize(camera.target - camera.eye);
    const Vec3 right = Normalize(Cross(forward, camera.up));
    const Vec3 up = Cross(right, forward);
    const float tan_half = std::tan(camera.fovY * 0.5f);

    Vec3 corners[8];
    int c = 0;
    for (int side = 0; side < 2; ++side) {
        const float d = side ? split_far : split_near;
        const float h = tan_half * d;
        const float w = h * aspect;
        const Vec3 centre = camera.eye + forward * d;
        for (int sy = -1; sy <= 1; sy += 2)
            for (int sx = -1; sx <= 1; sx += 2)
                corners[c++] = centre + right * (w * float(sx)) + up * (h * float(sy));
    }

    // A SPHERE around the slice, not a box. A box fitted to the corners changes
    // size as the camera turns, and a shadow map whose extent changes every
    // frame shimmers along every edge. A sphere is rotation-invariant.
    Vec3 centre{0.0f, 0.0f, 0.0f};
    for (const Vec3& p : corners) centre = centre + p;
    centre = centre * 0.125f;
    float radius = 0.0f;
    for (const Vec3& p : corners) radius = std::max(radius, Length(p - centre));
    radius = std::ceil(radius * 16.0f) / 16.0f;  // quantised, for the same reason

    const Vec3 dir = Normalize(Vec3{lightDir.x, lightDir.y, lightDir.z});
    const Vec3 light_up = (std::fabs(dir.y) > 0.99f) ? Vec3{0.0f, 0.0f, 1.0f}
                                                     : Vec3{0.0f, 1.0f, 0.0f};
    // Pulled back far enough that casters BEHIND the slice still make it in —
    // a shadow is cast by something outside the view as often as not.
    const float pull = radius * 4.0f;

    // TEXEL SNAPPING, in LIGHT SPACE and BEFORE the view matrix is built.
    //
    // Without it the map's origin slides a fraction of a texel as the camera
    // walks, every shadow edge re-samples slightly differently, and the whole
    // scene crawls. Quantising the centre locks the map's grid to the world, so
    // a fixed point stays on its texel until the camera has moved a whole one.
    //
    // The obvious version of this — build the view, then round the centre's
    // position in it — does nothing at all: the view is built LOOKING AT the
    // centre, so the centre is at the origin by construction and rounding zero
    // gives zero. The rounding has to happen in the light's basis while the
    // centre is still a real position.
    const Vec3 forward_l = dir * -1.0f;
    const Vec3 right_l = Normalize(Cross(forward_l, light_up));
    const Vec3 up_l = Cross(right_l, forward_l);
    const float texels = 1024.0f;  // matches the tile size the renderer uses
    const float units_per_texel = radius * 2.0f / texels;
    const float cx = Dot(right_l, centre), cy = Dot(up_l, centre);
    const float snapped_x = std::floor(cx / units_per_texel) * units_per_texel;
    const float snapped_y = std::floor(cy / units_per_texel) * units_per_texel;
    centre = centre + right_l * (snapped_x - cx) + up_l * (snapped_y - cy);

    const Mat4 view = Mat4::LookAt(centre + dir * pull, centre, light_up);
    const Mat4 proj =
        Mat4::OrthographicReverseZ(radius, radius, 0.05f, pull + radius * 2.0f);
    return proj * view;
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
