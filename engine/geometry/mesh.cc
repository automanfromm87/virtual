#include "engine/geometry/mesh.h"

#include <cmath>
#include <numbers>

namespace eng {

Mesh MakeUVSphere(float radius, int stacks, int slices, Vec4 a, Vec4 b) {
    Mesh m;
    if (stacks < 2 || slices < 3 || radius <= 0.0f) return m;

    constexpr float kPi = std::numbers::pi_v<float>;
    const int stride = slices + 1;  // +1: the seam vertex is duplicated

    // --- vertices ------------------------------------------------------------
    // The seam (theta = 0 and theta = 2pi) and the poles are emitted more than
    // once on purpose. They share a position but will not share a texcoord once
    // this mesh grows UVs, and a vertex can only carry one of each.
    m.vertices.reserve(std::size_t(stacks + 1) * std::size_t(stride));
    for (int i = 0; i <= stacks; ++i) {
        const float phi = float(i) / float(stacks) * kPi;  // 0 = +Y pole
        const float y = std::cos(phi);
        const float r = std::sin(phi);
        for (int j = 0; j <= slices; ++j) {
            const float theta = float(j) / float(slices) * 2.0f * kPi;
            const float x = r * std::cos(theta);
            const float z = r * std::sin(theta);

            VertexIn v{};
            v.position = Vec4{x * radius, y * radius, z * radius, 0.0f};
            // Equirectangular: u wraps once around, v runs pole to pole. The
            // duplicated seam column is exactly why u can reach 1.0 — a shared
            // seam vertex would have to be both 0 and 1 at once.
            v.uv = Vec4{float(j) / float(slices), float(i) / float(stacks), 0.0f,
                        0.0f};
            // Unit sphere at the origin: the outward normal IS the direction of
            // the position, so no normalize() and no cross products needed.
            v.normal = Vec4{x, y, z, 0.0f};
            // 8x8 checker in lat/long. Quantise by band index rather than by
            // i/j parity, or the squares come out one triangle wide.
            const int band = (i * 8) / stacks + (j * 8) / slices;
            v.color = (band & 1) ? b : a;
            m.vertices.push_back(v);
        }
    }

    // --- indices -------------------------------------------------------------
    // Counter-clockwise seen from outside, so back-face culling keeps the near
    // hemisphere. Each quad is (a, a+1, b) + (b, a+1, b+1).
    m.indices.reserve(std::size_t(stacks) * std::size_t(slices) * 6);
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            const auto v0 = std::uint16_t(i * stride + j);
            const auto v1 = std::uint16_t(v0 + stride);

            // At the poles the whole top (or bottom) row collapses to one
            // point, so one triangle of each quad there is zero-area.
            if (i != 0) {
                m.indices.push_back(v0);
                m.indices.push_back(std::uint16_t(v0 + 1));
                m.indices.push_back(v1);
            }
            if (i != stacks - 1) {
                m.indices.push_back(v1);
                m.indices.push_back(std::uint16_t(v0 + 1));
                m.indices.push_back(std::uint16_t(v1 + 1));
            }
        }
    }

    m.bounds.center = Vec3{0.0f, 0.0f, 0.0f};
    m.bounds.radius = radius;
    return m;
}

Mesh MakeBox(Vec3 half_extents, Vec4 color) {
    Mesh m;
    if (half_extents.x <= 0.0f || half_extents.y <= 0.0f || half_extents.z <= 0.0f)
        return m;

    struct Face {
        Vec3 n, u, v;
    };
    const Face faces[6] = {
        {{1, 0, 0}, {0, 0, -1}, {0, 1, 0}},   // +X
        {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},   // -X
        {{0, 1, 0}, {1, 0, 0}, {0, 0, -1}},   // +Y
        {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},   // -Y
        {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},    // +Z
        {{0, 0, -1}, {-1, 0, 0}, {0, 1, 0}},  // -Z
    };
    auto scale = [&](Vec3 axis) {
        return Vec3{axis.x * half_extents.x, axis.y * half_extents.y,
                    axis.z * half_extents.z};
    };

    const float su[4] = {-1, 1, 1, -1};
    const float sv[4] = {-1, -1, 1, 1};
    for (int f = 0; f < 6; ++f) {
        const Vec3 n = scale(faces[f].n);
        const Vec3 u = scale(faces[f].u);
        const Vec3 v = scale(faces[f].v);
        const auto base = std::uint16_t(m.vertices.size());
        for (int c = 0; c < 4; ++c) {
            const Vec3 p = n + u * su[c] + v * sv[c];
            VertexIn vtx{};
            vtx.position = Vec4{p.x, p.y, p.z, 0.0f};
            vtx.normal = Vec4{faces[f].n.x, faces[f].n.y, faces[f].n.z, 0.0f};
            vtx.color = color;
            vtx.uv = Vec4{(su[c] + 1.0f) * 0.5f, (1.0f - sv[c]) * 0.5f, 0, 0};
            m.vertices.push_back(vtx);
        }
        m.indices.push_back(base);
        m.indices.push_back(std::uint16_t(base + 1));
        m.indices.push_back(std::uint16_t(base + 2));
        m.indices.push_back(base);
        m.indices.push_back(std::uint16_t(base + 2));
        m.indices.push_back(std::uint16_t(base + 3));
    }
    m.bounds.center = Vec3{0, 0, 0};
    m.bounds.radius = Length(half_extents);
    return m;
}

Mesh MakeCube(float size, Vec4 a, Vec4 b) {
    Mesh m;
    if (size <= 0.0f) return m;
    const float h = size * 0.5f;

    // Per face: an outward normal plus two tangent axes with u x v == n. That
    // identity is what makes the corner order below come out counter-clockwise
    // when seen from outside; get it wrong on one face and that face vanishes
    // under back-face culling.
    struct Face {
        Vec3 n, u, v;
    };
    const Face faces[6] = {
        {{1, 0, 0}, {0, 0, -1}, {0, 1, 0}},   // +X
        {{-1, 0, 0}, {0, 0, 1}, {0, 1, 0}},   // -X
        {{0, 1, 0}, {1, 0, 0}, {0, 0, -1}},   // +Y
        {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}},   // -Y
        {{0, 0, 1}, {1, 0, 0}, {0, 1, 0}},    // +Z
        {{0, 0, -1}, {-1, 0, 0}, {0, 1, 0}},  // -Z
    };

    m.vertices.reserve(24);
    m.indices.reserve(36);
    for (int f = 0; f < 6; ++f) {
        const Face& face = faces[f];
        const Vec4 color = (f & 1) ? b : a;
        const auto base = std::uint16_t(m.vertices.size());

        // bottom-left, bottom-right, top-right, top-left in face space.
        const float su[4] = {-1, 1, 1, -1};
        const float sv[4] = {-1, -1, 1, 1};
        for (int c = 0; c < 4; ++c) {
            const Vec3 p = face.n * h + face.u * (su[c] * h) + face.v * (sv[c] * h);
            VertexIn vtx{};
            vtx.position = Vec4{p.x, p.y, p.z, 0.0f};
            // Each face gets the full 0..1 square.
            vtx.uv = Vec4{(su[c] + 1.0f) * 0.5f, (1.0f - sv[c]) * 0.5f, 0.0f, 0.0f};
            vtx.normal = Vec4{face.n.x, face.n.y, face.n.z, 0.0f};
            vtx.color = color;
            m.vertices.push_back(vtx);
        }
        m.indices.push_back(base);
        m.indices.push_back(std::uint16_t(base + 1));
        m.indices.push_back(std::uint16_t(base + 2));
        m.indices.push_back(base);
        m.indices.push_back(std::uint16_t(base + 2));
        m.indices.push_back(std::uint16_t(base + 3));
    }

    m.bounds.center = Vec3{0.0f, 0.0f, 0.0f};
    m.bounds.radius = h * std::sqrt(3.0f);  // half the space diagonal
    return m;
}

}  // namespace eng
