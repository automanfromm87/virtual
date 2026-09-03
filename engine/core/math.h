// Hand-rolled linear algebra. No <simd/simd.h>, no glm — the layout of these
// types is a contract with the GPU, so we own it.
//
// CONVENTIONS — decided once, never revisited:
//   * Right-handed. +Y is up. The camera looks down -Z.
//   * Column-major storage, column-vector math:  v' = M * v.
//     M.col[i] is the i-th COLUMN, which is also how Metal's float4x4 is laid out.
//   * Metal clip space: x,y in [-1,1], z in [0,1]  (NOT [-1,1] like OpenGL).
//   * Reversed-Z: near maps to 1, far maps to 0. Pair with depthCompare=greater
//     and clearDepth=0. Costs nothing now, fixes depth precision forever.
//   * Units: meters, seconds, radians.
#pragma once

#include <cmath>
#include <concepts>
#include <cstddef>

namespace eng {

template <class T>
concept Real = std::floating_point<T>;

// ---------------------------------------------------------------- Vec2 ----
// Plan-space: a floor plan is inherently 2D, and carrying a dead z through it
// invites someone to put something in it.
struct Vec2 {
    float x = 0.0f, y = 0.0f;

    constexpr Vec2 operator+(Vec2 b) const { return {x + b.x, y + b.y}; }
    constexpr Vec2 operator-(Vec2 b) const { return {x - b.x, y - b.y}; }
    constexpr Vec2 operator*(float s) const { return {x * s, y * s}; }
};

constexpr float Dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
// 2D cross product: signed area of the parallelogram. Positive means b turns
// left from a — which is how the triangulator tells convex from reflex.
constexpr float Cross2(Vec2 a, Vec2 b) { return a.x * b.y - a.y * b.x; }
inline float Length(Vec2 v) { return std::sqrt(Dot(v, v)); }
inline Vec2 Normalize(Vec2 v) {
    const float len = Length(v);
    return len > 0.0f ? v * (1.0f / len) : Vec2{};
}
// Rotated 90 degrees left. Wall thickness runs along this.
constexpr Vec2 Perp(Vec2 v) { return {-v.y, v.x}; }

// ---------------------------------------------------------------- Vec3 ----
// 12 bytes. This is the CPU type. Do NOT put it in a GPU struct — Metal's
// float3 is padded to 16. See engine/shaders/shader_types.h.
struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;

    constexpr Vec3 operator+(Vec3 b) const { return {x + b.x, y + b.y, z + b.z}; }
    constexpr Vec3 operator-(Vec3 b) const { return {x - b.x, y - b.y, z - b.z}; }
    constexpr Vec3 operator-() const { return {-x, -y, -z}; }
    constexpr Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
};

constexpr float Dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

constexpr Vec3 Cross(Vec3 a, Vec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

inline float Length(Vec3 v) { return std::sqrt(Dot(v, v)); }

inline Vec3 Normalize(Vec3 v) {
    const float len = Length(v);
    return len > 0.0f ? v * (1.0f / len) : Vec3{};
}

// ---------------------------------------------------------------- Vec4 ----
// 16 bytes, 16-byte aligned — byte-identical to MSL's float4.
struct alignas(16) Vec4 {
    float x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f;

    constexpr Vec4 operator+(Vec4 b) const {
        return {x + b.x, y + b.y, z + b.z, w + b.w};
    }
    constexpr Vec4 operator-(Vec4 b) const {
        return {x - b.x, y - b.y, z - b.z, w - b.w};
    }
    constexpr Vec4 operator*(float s) const { return {x * s, y * s, z * s, w * s}; }
};

static_assert(sizeof(Vec4) == 16 && alignof(Vec4) == 16);

// ---------------------------------------------------------------- Mat4 ----
// 64 bytes — byte-identical to MSL's float4x4. col[i] is column i.
struct alignas(16) Mat4 {
    Vec4 col[4];

    static constexpr Mat4 Identity() {
        return Mat4{{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}};
    }

    static constexpr Mat4 Translation(Vec3 t) {
        return Mat4{{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {t.x, t.y, t.z, 1}}};
    }

    // Uniform scale only. Non-uniform scale would break every shader that
    // transforms normals by this matrix — those need the inverse transpose.
    static constexpr Mat4 Scale(float s) {
        return Mat4{{{s, 0, 0, 0}, {0, s, 0, 0}, {0, 0, s, 0}, {0, 0, 0, 1}}};
    }

    // Right-handed rotations: positive angle turns counter-clockwise when the
    // axis points AT you. Remember col[i] is column i, so these look transposed
    // compared to the row-major form you find in most textbooks.
    static Mat4 RotationX(Real auto radians) {
        const float c = std::cos(static_cast<float>(radians));
        const float s = std::sin(static_cast<float>(radians));
        return Mat4{{{1, 0, 0, 0}, {0, c, s, 0}, {0, -s, c, 0}, {0, 0, 0, 1}}};
    }

    static Mat4 RotationY(Real auto radians) {
        const float c = std::cos(static_cast<float>(radians));
        const float s = std::sin(static_cast<float>(radians));
        return Mat4{{{c, 0, -s, 0}, {0, 1, 0, 0}, {s, 0, c, 0}, {0, 0, 0, 1}}};
    }

    // Reversed-Z perspective with an infinite far plane.
    //   z_ndc = near / -z_view   ->   1 at the near plane, 0 at infinity.
    static Mat4 PerspectiveReverseZ(Real auto fovYRadians, Real auto aspect,
                                    Real auto nearZ) {
        const float f = 1.0f / std::tan(static_cast<float>(fovYRadians) * 0.5f);
        return Mat4{{{f / static_cast<float>(aspect), 0, 0, 0},
                     {0, f, 0, 0},
                     {0, 0, 0, -1},
                     {0, 0, static_cast<float>(nearZ), 0}}};
    }

    // Reversed-Z orthographic, for a DIRECTIONAL light's shadow map. Parallel
    // rays have no vanishing point, so a perspective frustum is simply the
    // wrong shape — and unlike PerspectiveReverseZ this one needs a finite far
    // plane, because a shadow map has to quantise a bounded range.
    //   z_view = -nearZ -> 1, z_view = -farZ -> 0.
    static Mat4 OrthographicReverseZ(Real auto halfWidth, Real auto halfHeight,
                                     Real auto nearZ, Real auto farZ) {
        const float n = static_cast<float>(nearZ);
        const float f = static_cast<float>(farZ);
        const float invRange = 1.0f / (f - n);
        return Mat4{{{1.0f / static_cast<float>(halfWidth), 0, 0, 0},
                     {0, 1.0f / static_cast<float>(halfHeight), 0, 0},
                     {0, 0, invRange, 0},
                     {0, 0, f * invRange, 1}}};
    }

    // Right-handed view matrix. The camera sits at `eye` looking at `center`.
    static Mat4 LookAt(Vec3 eye, Vec3 center, Vec3 up) {
        const Vec3 f = Normalize(center - eye);  // forward
        const Vec3 s = Normalize(Cross(f, up));  // right
        const Vec3 u = Cross(s, f);              // true up
        return Mat4{{{s.x, u.x, -f.x, 0},
                     {s.y, u.y, -f.y, 0},
                     {s.z, u.z, -f.z, 0},
                     {-Dot(s, eye), -Dot(u, eye), Dot(f, eye), 1}}};
    }
};

static_assert(sizeof(Mat4) == 64 && alignof(Mat4) == 16);

// v' = m * v
constexpr Vec4 operator*(const Mat4& m, Vec4 v) {
    return m.col[0] * v.x + m.col[1] * v.y + m.col[2] * v.z + m.col[3] * v.w;
}

// c = a * b  (apply b first, then a)
constexpr Mat4 operator*(const Mat4& a, const Mat4& b) {
    return Mat4{{a * b.col[0], a * b.col[1], a * b.col[2], a * b.col[3]}};
}

// ---------------------------------------------------------------- Quat ----
// Unit quaternion, (x,y,z) vector part and w scalar. Stored in glTF's order.
//
// The reason to have one at all: Euler angles gimbal-lock and do not
// interpolate, and a rotation MATRIX cannot be blended without drifting off the
// rotation group. Any animation or camera smoothing needs this.
struct Quat {
    float x = 0.0f, y = 0.0f, z = 0.0f, w = 1.0f;
};

inline Quat Normalize(Quat q) {
    const float len = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (len <= 0.0f) return Quat{};
    const float inv = 1.0f / len;
    return Quat{q.x * inv, q.y * inv, q.z * inv, q.w * inv};
}

// Column-major rotation matrix. col[i] is column i, so this is the transpose of
// the form most references print.
inline Mat4 QuatToMat4(Quat q) {
    q = Normalize(q);
    const float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    const float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    const float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
    return Mat4{{{1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy), 0},
                 {2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx), 0},
                 {2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy), 0},
                 {0, 0, 0, 1}}};
}

// Composition: applying `b` and then `a`, the same order as matrix product.
inline Quat operator*(Quat a, Quat b) {
    return Quat{a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
                a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

// For a UNIT quaternion the conjugate is the inverse, which is the only case
// this engine produces — everything is normalised on the way in.
inline Quat Conjugate(Quat q) { return Quat{-q.x, -q.y, -q.z, q.w}; }

// v rotated by q. The expanded form rather than q*v*q⁻¹: same result, about
// half the multiplies, and no temporary quaternion.
inline Vec3 Rotate(Quat q, Vec3 v) {
    const Vec3 u{q.x, q.y, q.z};
    const Vec3 t = Cross(u, v) * 2.0f;
    return v + t * q.w + Cross(u, t);
}

// Rotates a vector by q INVERSE, i.e. world space back into the body's frame.
inline Vec3 RotateInverse(Quat q, Vec3 v) { return Rotate(Conjugate(q), v); }

inline Quat QuatFromAxisAngle(Vec3 axis, float radians) {
    const float len = Length(axis);
    if (len <= 0.0f) return Quat{};
    const Vec3 n = axis * (1.0f / len);
    const float h = radians * 0.5f;
    const float s = std::sin(h);
    return Quat{n.x * s, n.y * s, n.z * s, std::cos(h)};
}

// Shortest-arc spherical interpolation. Falls back to a lerp when the two are
// nearly parallel, where the sin() denominator loses all its precision.
inline Quat Slerp(Quat a, Quat b, float t) {
    a = Normalize(a);
    b = Normalize(b);
    float d = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    // Negate one end if they point into opposite hemispheres: q and -q are the
    // same rotation, and without this the interpolation takes the long way.
    if (d < 0.0f) {
        b = Quat{-b.x, -b.y, -b.z, -b.w};
        d = -d;
    }
    if (d > 0.9995f) {
        return Normalize(Quat{a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t,
                              a.z + (b.z - a.z) * t, a.w + (b.w - a.w) * t});
    }
    const float theta = std::acos(d);
    const float st = std::sin(theta);
    const float wa = std::sin((1.0f - t) * theta) / st;
    const float wb = std::sin(t * theta) / st;
    return Quat{a.x * wa + b.x * wb, a.y * wa + b.y * wb, a.z * wa + b.z * wb,
                a.w * wa + b.w * wb};
}

// --------------------------------------------------------------- Frustum ----
// Rows of a column-major matrix. clip.x is row 0 dotted with the vertex, and
// the frustum planes are built out of those rows.
constexpr Vec4 Row(const Mat4& m, int i) {
    switch (i) {
        case 0: return Vec4{m.col[0].x, m.col[1].x, m.col[2].x, m.col[3].x};
        case 1: return Vec4{m.col[0].y, m.col[1].y, m.col[2].y, m.col[3].y};
        case 2: return Vec4{m.col[0].z, m.col[1].z, m.col[2].z, m.col[3].z};
        default: return Vec4{m.col[0].w, m.col[1].w, m.col[2].w, m.col[3].w};
    }
}

// Dot(n, p) + d >= 0 means p is on the inside.
struct Plane {
    Vec3 n{0.0f, 0.0f, 0.0f};
    float d = 1.0f;  // a zero normal with positive d never culls anything
};

// Six planes in world space, extracted straight from viewProj (Gribb-Hartmann).
//
// Metal clip space is x,y in [-1,1] but z in [0,1] — NOT [-1,1] like OpenGL —
// so the near/far pair comes from `z >= 0` and `z <= w`, not `z >= -w`. Getting
// this wrong culls half the world or nothing at all.
struct Frustum {
    Plane planes[6];

    static Frustum FromViewProj(const Mat4& vp) {
        const Vec4 r0 = Row(vp, 0), r1 = Row(vp, 1);
        const Vec4 r2 = Row(vp, 2), r3 = Row(vp, 3);
        const Vec4 raw[6] = {
            r3 + r0,  // left    x >= -w
            r3 - r0,  // right   x <=  w
            r3 + r1,  // bottom  y >= -w
            r3 - r1,  // top     y <=  w
            r2,       // z >= 0. Degenerate under an infinite far plane: with
                      // reversed-Z, row 2 is (0,0,0,nearZ), so this plane has a
                      // zero normal and correctly never culls.
            r3 - r2,  // z <= w  (the near plane)
        };
        Frustum f;
        for (int i = 0; i < 6; ++i) {
            Vec3 n{raw[i].x, raw[i].y, raw[i].z};
            const float len = Length(n);
            if (len > 1e-6f) {
                f.planes[i] = Plane{n * (1.0f / len), raw[i].w / len};
            } else {
                f.planes[i] = Plane{};  // never culls
            }
        }
        return f;
    }

    // Conservative: a sphere straddling a plane counts as visible.
    [[nodiscard]] bool IntersectsSphere(Vec3 center, float radius) const {
        for (const Plane& p : planes)
            if (Dot(p.n, center) + p.d < -radius) return false;
        return true;
    }
};

// Largest axis scale in the upper 3x3, for growing an object-space bounding
// radius into world space.
inline float MaxScale(const Mat4& m) {
    const float a = Length(Vec3{m.col[0].x, m.col[0].y, m.col[0].z});
    const float b = Length(Vec3{m.col[1].x, m.col[1].y, m.col[1].z});
    const float c = Length(Vec3{m.col[2].x, m.col[2].y, m.col[2].z});
    return a > b ? (a > c ? a : c) : (b > c ? b : c);
}

}  // namespace eng
