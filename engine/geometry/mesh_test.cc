// No test framework — from scratch means from scratch.
//
// The point of this file: winding and orientation are exactly what a
// pixel-coverage check CANNOT see (permuting the framebuffer does not change a
// colour histogram). On the CPU they are cheap and exact, so test them here and
// let the GPU harness worry about whether anything got drawn at all.
#include "engine/geometry/mesh.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <tuple>
#include <utility>

namespace {

int g_failures = 0;

void Fail(const char* what, const char* which, int line) {
    std::fprintf(stderr, "mesh_test.cc:%d  [%s] %s\n", line, which, what);
    ++g_failures;
}

#define CHECK(cond) \
    do { if (!(cond)) Fail(#cond, which, __LINE__); } while (0)

// Every closed mesh in this engine must satisfy these, whatever its shape.
void CheckClosedMesh(const eng::Mesh& m, const char* which) {
    using namespace eng;

    CHECK(!m.vertices.empty());
    CHECK(!m.indices.empty());
    CHECK(m.indices.size() % 3 == 0);
    CHECK(m.bounds.radius > 0.0f);

    for (std::uint16_t i : m.indices) CHECK(i < m.vertices.size());

    for (const VertexIn& v : m.vertices) {
        const Vec3 n{v.normal.x, v.normal.y, v.normal.z};
        CHECK(std::fabs(Length(n) - 1.0f) < 1e-4f);  // unit normals

        // Every vertex fits inside the declared bounding sphere, or frustum
        // culling will pop geometry off the edge of the screen.
        const Vec3 p{v.position.x, v.position.y, v.position.z};
        CHECK(Length(p - m.bounds.center) <= m.bounds.radius + 1e-4f);

        // uv inside the unit square. Out-of-range coordinates are invisible
        // with clamp addressing and catastrophic with repeat.
        CHECK(v.uv.x >= -1e-6f && v.uv.x <= 1.0f + 1e-6f);
        CHECK(v.uv.y >= -1e-6f && v.uv.y <= 1.0f + 1e-6f);
    }

    // THE important one: every triangle must wind counter-clockwise as seen
    // from outside, or back-face culling turns the object inside out. The
    // right-hand rule normal of a CCW triangle points away from the centre.
    int backwards = 0, degenerate = 0;
    for (std::size_t t = 0; t + 2 < m.indices.size(); t += 3) {
        const VertexIn& a = m.vertices[m.indices[t + 0]];
        const VertexIn& b = m.vertices[m.indices[t + 1]];
        const VertexIn& c = m.vertices[m.indices[t + 2]];
        const Vec3 pa{a.position.x, a.position.y, a.position.z};
        const Vec3 pb{b.position.x, b.position.y, b.position.z};
        const Vec3 pc{c.position.x, c.position.y, c.position.z};

        const Vec3 face = Cross(pb - pa, pc - pa);
        if (Length(face) < 1e-9f) { ++degenerate; continue; }
        // Outward direction at this triangle = its centroid, since both shapes
        // are centred on the origin.
        if (Dot(face, pa + pb + pc) <= 0.0f) ++backwards;
    }
    CHECK(backwards == 0);
    CHECK(degenerate == 0);  // pole quads should be skipped, not emitted

    // CLOSURE. A hole in the mesh passes every check above — normals, winding
    // and bounds are all per-triangle properties. On a closed orientable
    // surface each directed edge appears exactly once and its reverse exactly
    // once; a hole leaves an edge unmatched.
    //
    // Keyed on POSITION, not index: the UV sphere duplicates its seam and pole
    // vertices, so index-based edges would look unmatched even when the surface
    // is perfectly closed.
    {
        using Key = std::tuple<long long, long long, long long>;
        auto key = [](const Vec4& p) {
            auto q = [](float v) { return (long long)std::llround(v * 8192.0); };
            return Key{q(p.x), q(p.y), q(p.z)};
        };
        std::map<std::pair<Key, Key>, int> directed;
        for (std::size_t t = 0; t + 2 < m.indices.size(); t += 3) {
            const Key k[3] = {key(m.vertices[m.indices[t + 0]].position),
                              key(m.vertices[m.indices[t + 1]].position),
                              key(m.vertices[m.indices[t + 2]].position)};
            for (int e = 0; e < 3; ++e)
                ++directed[{k[e], k[(e + 1) % 3]}];
        }
        int unmatched = 0, duplicated = 0;
        for (const auto& [edge, count] : directed) {
            if (count != 1) ++duplicated;  // same edge, same direction, twice
            auto rev = directed.find({edge.second, edge.first});
            if (rev == directed.end()) ++unmatched;
        }
        CHECK(unmatched == 0);   // no holes
        CHECK(duplicated == 0);  // consistently oriented, no folded-over faces
    }

    // The face normal and the interpolated vertex normals must agree, or
    // lighting fights the silhouette.
    int flipped = 0;
    for (std::size_t t = 0; t + 2 < m.indices.size(); t += 3) {
        const VertexIn& a = m.vertices[m.indices[t + 0]];
        const VertexIn& b = m.vertices[m.indices[t + 1]];
        const VertexIn& c = m.vertices[m.indices[t + 2]];
        const Vec3 pa{a.position.x, a.position.y, a.position.z};
        const Vec3 pb{b.position.x, b.position.y, b.position.z};
        const Vec3 pc{c.position.x, c.position.y, c.position.z};
        const Vec3 face = Cross(pb - pa, pc - pa);
        const Vec3 avg{(a.normal.x + b.normal.x + c.normal.x),
                       (a.normal.y + b.normal.y + c.normal.y),
                       (a.normal.z + b.normal.z + c.normal.z)};
        if (Length(face) > 1e-9f && Dot(face, avg) <= 0.0f) ++flipped;
    }
    CHECK(flipped == 0);
}

}  // namespace

int main() {
    using namespace eng;

    {
        const char* which = "sphere";
        constexpr float kR = 2.0f;
        constexpr int kStacks = 16, kSlices = 24;
        const Mesh m =
            MakeUVSphere(kR, kStacks, kSlices, Vec4{1, 1, 1, 1}, Vec4{0, 0, 0, 1});
        CheckClosedMesh(m, which);

        CHECK(m.vertices.size() == std::size_t(kStacks + 1) * (kSlices + 1));
        // Poles contribute one degenerate triangle per quad, so two rows lose one.
        CHECK(m.indices.size() ==
              std::size_t(kStacks * kSlices * 2 - kSlices * 2) * 3);
        CHECK(std::fabs(m.bounds.radius - kR) < 1e-5f);

        // On a sphere the normal IS the normalised position. Checking only the
        // radius, as this used to, would pass for ANY unit normal — including
        // one pointing the wrong way.
        for (const VertexIn& v : m.vertices) {
            const Vec3 p{v.position.x, v.position.y, v.position.z};
            const Vec3 n{v.normal.x, v.normal.y, v.normal.z};
            CHECK(std::fabs(Length(p) - kR) < 1e-4f);
            CHECK(Dot(Normalize(p), n) > 1.0f - 1e-4f);
        }
    }

    {
        const char* which = "cube";
        constexpr float kSize = 2.0f;
        const Mesh m = MakeCube(kSize, Vec4{1, 1, 1, 1}, Vec4{0, 0, 0, 1});
        CheckClosedMesh(m, which);

        // 24, not 8: a hard edge needs a different normal per face.
        CHECK(m.vertices.size() == 24);
        CHECK(m.indices.size() == 36);
        // Bounding sphere is the half space-diagonal, not the half side.
        CHECK(std::fabs(m.bounds.radius - kSize * 0.5f * std::sqrt(3.0f)) < 1e-5f);

        // Every face covers the whole 0..1 uv square: 4 corners x 6 faces means
        // each of the four corner values appears exactly six times.
        int corners[4] = {0, 0, 0, 0};
        for (const VertexIn& v : m.vertices) {
            const int cu = v.uv.x > 0.5f ? 1 : 0;
            const int cv = v.uv.y > 0.5f ? 1 : 0;
            ++corners[cv * 2 + cu];
        }
        for (int i = 0; i < 4; ++i) CHECK(corners[i] == 6);

        // Exactly six distinct normals, one per face.
        int axis_hits = 0;
        for (const VertexIn& v : m.vertices) {
            const float ax = std::fabs(v.normal.x) + std::fabs(v.normal.y) +
                             std::fabs(v.normal.z);
            if (std::fabs(ax - 1.0f) < 1e-5f) ++axis_hits;  // axis-aligned
        }
        CHECK(axis_hits == 24);
    }

    {
        const char* which = "box";
        const Mesh m = MakeBox(Vec3{2.0f, 0.25f, 1.0f}, Vec4{1, 1, 1, 1});
        CheckClosedMesh(m, which);
        CHECK(m.vertices.size() == 24);
        // Bounding radius is the half-diagonal, not the largest half-extent.
        CHECK(std::fabs(m.bounds.radius - Length(Vec3{2.0f, 0.25f, 1.0f})) < 1e-5f);
        // Extents come out exactly as asked, on every axis independently.
        float mx = 0, my = 0, mz = 0;
        for (const VertexIn& v : m.vertices) {
            mx = std::fmax(mx, std::fabs(v.position.x));
            my = std::fmax(my, std::fabs(v.position.y));
            mz = std::fmax(mz, std::fabs(v.position.z));
        }
        CHECK(std::fabs(mx - 2.0f) < 1e-5f);
        CHECK(std::fabs(my - 0.25f) < 1e-5f);
        CHECK(std::fabs(mz - 1.0f) < 1e-5f);
    }

    {
        const char* which = "degenerate inputs";
        CHECK(MakeUVSphere(1.0f, 1, 8, Vec4{}, Vec4{}).vertices.empty());
        CHECK(MakeUVSphere(1.0f, 8, 2, Vec4{}, Vec4{}).vertices.empty());
        CHECK(MakeUVSphere(0.0f, 8, 8, Vec4{}, Vec4{}).vertices.empty());
        CHECK(MakeCube(0.0f, Vec4{}, Vec4{}).vertices.empty());
        CHECK(MakeCube(-1.0f, Vec4{}, Vec4{}).vertices.empty());
        CHECK(MakeBox(Vec3{0, 1, 1}, Vec4{}).vertices.empty());
        CHECK(MakeBox(Vec3{1, -1, 1}, Vec4{}).vertices.empty());
    }

    if (g_failures == 0) std::printf("mesh_test: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
