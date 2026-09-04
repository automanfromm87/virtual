#include "engine/geometry/mesh.h"

#include <cmath>
#include <numbers>
#include <vector>

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
            const auto v0 = std::uint32_t(i * stride + j);
            const auto v1 = std::uint32_t(v0 + stride);

            // At the poles the whole top (or bottom) row collapses to one
            // point, so one triangle of each quad there is zero-area.
            if (i != 0) {
                m.indices.push_back(v0);
                m.indices.push_back(std::uint32_t(v0 + 1));
                m.indices.push_back(v1);
            }
            if (i != stacks - 1) {
                m.indices.push_back(v1);
                m.indices.push_back(std::uint32_t(v0 + 1));
                m.indices.push_back(std::uint32_t(v1 + 1));
            }
        }
    }

    m.bounds.center = Vec3{0.0f, 0.0f, 0.0f};
    m.bounds.radius = radius;
    GenerateTangents(m);
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
        const auto base = std::uint32_t(m.vertices.size());
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
        m.indices.push_back(std::uint32_t(base + 1));
        m.indices.push_back(std::uint32_t(base + 2));
        m.indices.push_back(base);
        m.indices.push_back(std::uint32_t(base + 2));
        m.indices.push_back(std::uint32_t(base + 3));
    }
    m.bounds.center = Vec3{0, 0, 0};
    m.bounds.radius = Length(half_extents);
    GenerateTangents(m);
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
        const auto base = std::uint32_t(m.vertices.size());

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
        m.indices.push_back(std::uint32_t(base + 1));
        m.indices.push_back(std::uint32_t(base + 2));
        m.indices.push_back(base);
        m.indices.push_back(std::uint32_t(base + 2));
        m.indices.push_back(std::uint32_t(base + 3));
    }

    m.bounds.center = Vec3{0.0f, 0.0f, 0.0f};
    m.bounds.radius = h * std::sqrt(3.0f);  // half the space diagonal
    GenerateTangents(m);
    return m;
}

void GenerateTangents(Mesh& m) {
    const std::size_t n = m.vertices.size();
    if (n == 0) return;

    // Two accumulators, not one. The tangent alone cannot tell you the
    // handedness: a mirrored uv shell produces the same tangent direction and
    // the opposite bitangent, and only comparing the two reveals it.
    std::vector<Vec3> tan(n, Vec3{0.0f, 0.0f, 0.0f});
    std::vector<Vec3> bit(n, Vec3{0.0f, 0.0f, 0.0f});

    for (std::size_t i = 0; i + 2 < m.indices.size(); i += 3) {
        const std::uint32_t i0 = m.indices[i], i1 = m.indices[i + 1],
                            i2 = m.indices[i + 2];
        if (i0 >= n || i1 >= n || i2 >= n) continue;
        const VertexIn& v0 = m.vertices[i0];
        const VertexIn& v1 = m.vertices[i1];
        const VertexIn& v2 = m.vertices[i2];

        const Vec3 p0{v0.position.x, v0.position.y, v0.position.z};
        const Vec3 e1{v1.position.x - p0.x, v1.position.y - p0.y,
                      v1.position.z - p0.z};
        const Vec3 e2{v2.position.x - p0.x, v2.position.y - p0.y,
                      v2.position.z - p0.z};

        const float du1 = v1.uv.x - v0.uv.x, dv1 = v1.uv.y - v0.uv.y;
        const float du2 = v2.uv.x - v0.uv.x, dv2 = v2.uv.y - v0.uv.y;
        const float det = du1 * dv2 - du2 * dv1;
        // A DEGENERATE uv triangle -- a seam collapsed to a point, or a face
        // with no unwrap at all. 1/det is then enormous or infinite and would
        // poison the accumulation of every vertex it touches, including the
        // ones whose other triangles are perfectly fine. Skipping leaves those
        // vertices to their good triangles and leaves the rest to the fallback
        // below.
        if (std::fabs(det) < 1e-12f) continue;
        const float r = 1.0f / det;

        const Vec3 t{(e1.x * dv2 - e2.x * dv1) * r, (e1.y * dv2 - e2.y * dv1) * r,
                     (e1.z * dv2 - e2.z * dv1) * r};
        const Vec3 b{(e2.x * du1 - e1.x * du2) * r, (e2.y * du1 - e1.y * du2) * r,
                     (e2.z * du1 - e1.z * du2) * r};

        for (std::uint32_t k : {i0, i1, i2}) {
            tan[k] = tan[k] + t;
            bit[k] = bit[k] + b;
        }
    }

    for (std::size_t i = 0; i < n; ++i) {
        VertexIn& v = m.vertices[i];
        const Vec3 nrm = Normalize(Vec3{v.normal.x, v.normal.y, v.normal.z});
        Vec3 t = tan[i];

        // GRAM-SCHMIDT. The accumulated tangent is an average over triangles
        // that do not all share this vertex's normal, so it is generally not
        // perpendicular to it. Projecting out the normal component is what
        // makes the frame orthonormal, which the shader relies on to invert it
        // by transpose rather than by a real inverse.
        const float along = Dot(nrm, t);
        t = Vec3{t.x - nrm.x * along, t.y - nrm.y * along, t.z - nrm.z * along};

        const float len = Length(t);
        if (len < 1e-8f) {
            // No usable uv gradient here: every triangle touching this vertex
            // was degenerate, or the tangent came out parallel to the normal.
            // ANY perpendicular direction is a valid frame for a surface with
            // no unwrap -- what matters is that it is finite and unit length,
            // because a zero tangent makes the whole TBN singular and the
            // fragment's normal becomes NaN.
            //
            // Cross with whichever axis the normal is least aligned to, so the
            // cross product never approaches zero.
            const Vec3 axis = std::fabs(nrm.x) < 0.9f ? Vec3{1.0f, 0.0f, 0.0f}
                                                      : Vec3{0.0f, 1.0f, 0.0f};
            t = Normalize(Cross(axis, nrm));
        } else {
            t = Vec3{t.x / len, t.y / len, t.z / len};
        }

        // HANDEDNESS. cross(N, T) is one of the two possible bitangents; the
        // accumulated one says which. A mirrored shell -- half of every
        // character model -- has the opposite sign, and getting it wrong turns
        // every dent on that half into a bump.
        const float sign = Dot(Cross(nrm, t), bit[i]) < 0.0f ? -1.0f : 1.0f;
        v.tangent = Vec4{t.x, t.y, t.z, sign};
    }
}

Mesh MakeGroundDecal(const GroundDecalDesc& d,
                     const std::function<float(float, float)>& height) {
    Mesh m;
    const int n = std::max(d.segments, 2);
    const float c = std::cos(d.rotation), s = std::sin(d.rotation);

    for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i) {
            const float u = float(i) / float(n), v = float(j) / float(n);
            // Local, in [-1, 1], then rotated. The rotation is applied to the
            // PLACEMENT and not to the uv, so a rotated decal samples its
            // texture the same way -- rotating the uv instead would spin the
            // texture inside a patch that stayed square to the world.
            const float lx = (u * 2.0f - 1.0f) * d.radius;
            const float lz = (v * 2.0f - 1.0f) * d.radius;
            const float wx = d.centre.x + lx * c - lz * s;
            const float wz = d.centre.z + lx * s + lz * c;

            VertexIn out{};
            out.position = Vec4{wx, height(wx, wz) + d.lift, wz, 0.0f};
            // The surface normal by central differences, so the decal is lit
            // like the ground it lies on. Taking +y instead makes a decal on a
            // slope brighter than the slope, which reads as it glowing.
            const float e = d.radius / float(n);
            const float dx = height(wx + e, wz) - height(wx - e, wz);
            const float dz = height(wx, wz + e) - height(wx, wz - e);
            const Vec3 nrm = Normalize(Vec3{-dx, 2.0f * e, -dz});
            out.normal = Vec4{nrm.x, nrm.y, nrm.z, 0.0f};

            // RADIAL falloff, in the local square. Squared distance so the
            // fade is gentle near the middle and quick at the rim, which is
            // what a scorch or a stain looks like.
            const float r = std::sqrt(lx * lx + lz * lz) / std::max(d.radius, 1e-6f);
            const float t =
                std::clamp((1.0f - r) / std::max(1.0f - d.fade_from, 1e-6f), 0.0f, 1.0f);
            out.color = Vec4{d.tint.x, d.tint.y, d.tint.z, d.tint.w * t * t};
            out.uv = Vec4{u, v, 0.0f, 0.0f};
            m.vertices.push_back(out);
        }

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const auto a = std::uint32_t(j * (n + 1) + i);
            const auto b = std::uint32_t(a + 1);
            const auto cc = std::uint32_t(a + n + 1);
            const auto dd = std::uint32_t(cc + 1);
            m.indices.insert(m.indices.end(), {a, cc, b, b, cc, dd});
        }
    GenerateTangents(m);
    // The bounds have to be computed, not assumed: the patch follows the ground,
    // so its vertical extent is whatever the terrain does across it and there is
    // no closed form. Without this the renderer culls it by a zero-radius sphere
    // at the origin and it vanishes everywhere but the middle of the world.
    Vec3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
    for (const VertexIn& v : m.vertices) {
        lo = Vec3{std::min(lo.x, v.position.x), std::min(lo.y, v.position.y),
                  std::min(lo.z, v.position.z)};
        hi = Vec3{std::max(hi.x, v.position.x), std::max(hi.y, v.position.y),
                  std::max(hi.z, v.position.z)};
    }
    m.bounds.center = (lo + hi) * 0.5f;
    float far2 = 0.0f;
    for (const VertexIn& v : m.vertices)
        far2 = std::max(far2, Length(Vec3{v.position.x, v.position.y, v.position.z} -
                                     m.bounds.center));
    m.bounds.radius = far2;
    return m;
}
}  // namespace eng
