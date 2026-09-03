#include "engine/geometry/simplify.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

namespace eng {
namespace {

// A quadric: the sum over a set of planes of the outer product of the plane
// with itself. Evaluating it at a point gives the sum of squared distances to
// those planes, which is exactly the error a merged vertex introduces.
//
// Symmetric, so ten coefficients rather than sixteen. Stored in doubles: the
// entries are products of coordinates, so they scale with the square of the
// model's size, and a model a kilometre across overflows a float's precision
// long before it overflows its range -- the symptom is a solve that returns a
// point nowhere near the surface.
struct Quadric {
    double a00 = 0, a01 = 0, a02 = 0, a03 = 0;
    double a11 = 0, a12 = 0, a13 = 0;
    double a22 = 0, a23 = 0;
    double a33 = 0;

    void AddPlane(double x, double y, double z, double w, double weight) {
        a00 += x * x * weight; a01 += x * y * weight; a02 += x * z * weight;
        a03 += x * w * weight; a11 += y * y * weight; a12 += y * z * weight;
        a13 += y * w * weight; a22 += z * z * weight; a23 += z * w * weight;
        a33 += w * w * weight;
    }
    void Add(const Quadric& q) {
        a00 += q.a00; a01 += q.a01; a02 += q.a02; a03 += q.a03;
        a11 += q.a11; a12 += q.a12; a13 += q.a13;
        a22 += q.a22; a23 += q.a23; a33 += q.a33;
    }
};

// The point minimising the quadric, by solving the 3x3 system of its gradient.
// Returns false when the system is singular -- which happens whenever the
// cluster's planes do not pin the point down: a flat sheet leaves it free to
// slide in two directions, a straight edge in one. The caller falls back to the
// centroid there, which is the right answer for an under-determined cluster.
bool SolveQuadric(const Quadric& q, Vec3* out) {
    const double m[9] = {q.a00, q.a01, q.a02,
                         q.a01, q.a11, q.a12,
                         q.a02, q.a12, q.a22};
    const double det =
        m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
        m[2] * (m[3] * m[7] - m[4] * m[6]);
    // Scaled against the matrix's own magnitude rather than an absolute
    // epsilon: the entries grow with the square of the model's size, so a fixed
    // threshold would call every large model singular and every small one
    // solvable.
    const double scale = std::abs(m[0]) + std::abs(m[4]) + std::abs(m[8]) + 1e-30;
    if (std::abs(det) < 1e-10 * scale * scale * scale) return false;

    const double b[3] = {-q.a03, -q.a13, -q.a23};
    const double inv_det = 1.0 / det;
    const double x = inv_det * (b[0] * (m[4] * m[8] - m[5] * m[7]) -
                                m[1] * (b[1] * m[8] - m[5] * b[2]) +
                                m[2] * (b[1] * m[7] - m[4] * b[2]));
    const double y = inv_det * (m[0] * (b[1] * m[8] - m[5] * b[2]) -
                                b[0] * (m[3] * m[8] - m[5] * m[6]) +
                                m[2] * (m[3] * b[2] - b[1] * m[6]));
    const double z = inv_det * (m[0] * (m[4] * b[2] - b[1] * m[7]) -
                                m[1] * (m[3] * b[2] - b[1] * m[6]) +
                                b[0] * (m[3] * m[7] - m[4] * m[6]));
    *out = Vec3{float(x), float(y), float(z)};
    return true;
}

struct Cluster {
    Quadric quadric;
    Vec3 position_sum{0, 0, 0};
    Vec3 normal_sum{0, 0, 0};
    Vec4 color_sum{0, 0, 0, 0};
    Vec4 uv_sum{0, 0, 0, 0};
    Vec3 seed_normal{0, 0, 0};
    Vec4 seed_color{0, 0, 0, 0};
    int count = 0;
    int output = -1;
};

struct CellKey {
    std::int32_t x, y, z;
    // A BUCKET INDEX as well as a cell, so two vertices in the same cell with
    // very different normals land in different clusters. See
    // SimplifyOptions::normal_weld_degrees.
    std::int32_t bucket;
    bool operator==(const CellKey& o) const {
        return x == o.x && y == o.y && z == o.z && bucket == o.bucket;
    }
};

struct CellHash {
    std::size_t operator()(const CellKey& k) const {
        // Three large odd primes. A hash that simply concatenates the
        // coordinates collides catastrophically on axis-aligned geometry, which
        // is most of what a game contains.
        std::size_t h = std::size_t(k.x) * 73856093u;
        h ^= std::size_t(k.y) * 19349663u;
        h ^= std::size_t(k.z) * 83492791u;
        h ^= std::size_t(k.bucket) * 2654435761u;
        return h;
    }
};

Vec3 Xyz(const Vec4& v) { return Vec3{v.x, v.y, v.z}; }

}  // namespace

Mesh Simplify(const Mesh& in, const SimplifyOptions& options) {
    if (in.vertices.empty() || in.indices.size() < 3) return in;

    // --- the grid ------------------------------------------------------------
    Vec3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
    for (const VertexIn& v : in.vertices) {
        lo = Vec3{std::min(lo.x, v.position.x), std::min(lo.y, v.position.y),
                  std::min(lo.z, v.position.z)};
        hi = Vec3{std::max(hi.x, v.position.x), std::max(hi.y, v.position.y),
                  std::max(hi.z, v.position.z)};
    }
    const Vec3 extent = hi - lo;
    const float diagonal = Length(extent);
    if (diagonal <= 1e-9f) return in;
    const float cell = std::max(diagonal * options.cell_fraction, 1e-6f);

    // --- per-vertex quadrics from the face planes ----------------------------
    //
    // WEIGHTED BY AREA. A quadric weighted equally per face lets a fan of tiny
    // slivers outvote the one large triangle they sit on, and the merged vertex
    // is pulled off the surface. Area weighting is the standard fix and it is
    // free: the cross product's length is twice the area and it is already
    // being computed for the normal.
    std::vector<Quadric> quadrics(in.vertices.size());
    const std::size_t triangles = in.indices.size() / 3;
    for (std::size_t t = 0; t < triangles; ++t) {
        const std::uint16_t i0 = in.indices[t * 3 + 0];
        const std::uint16_t i1 = in.indices[t * 3 + 1];
        const std::uint16_t i2 = in.indices[t * 3 + 2];
        if (i0 >= in.vertices.size() || i1 >= in.vertices.size() ||
            i2 >= in.vertices.size())
            continue;
        const Vec3 p0 = Xyz(in.vertices[i0].position);
        const Vec3 p1 = Xyz(in.vertices[i1].position);
        const Vec3 p2 = Xyz(in.vertices[i2].position);
        const Vec3 cross = Cross(p1 - p0, p2 - p0);
        const float area2 = Length(cross);
        if (area2 <= 1e-20f) continue;
        const Vec3 n = cross * (1.0f / area2);
        const double d = -double(Dot(n, p0));
        Quadric q;
        q.AddPlane(n.x, n.y, n.z, d, double(area2));
        quadrics[i0].Add(q);
        quadrics[i1].Add(q);
        quadrics[i2].Add(q);
    }

    // --- cluster ------------------------------------------------------------
    const float cos_weld = std::cos(options.normal_weld_degrees * 3.14159265f / 180.0f);
    std::unordered_map<CellKey, int, CellHash> lookup;
    std::vector<Cluster> clusters;
    std::vector<int> vertex_cluster(in.vertices.size(), -1);

    for (std::size_t i = 0; i < in.vertices.size(); ++i) {
        const VertexIn& v = in.vertices[i];
        const Vec3 p = Xyz(v.position);
        const Vec3 n = Xyz(v.normal);
        CellKey key{std::int32_t(std::floor((p.x - lo.x) / cell)),
                    std::int32_t(std::floor((p.y - lo.y) / cell)),
                    std::int32_t(std::floor((p.z - lo.z) / cell)), 0};

        // Find a bucket in this cell whose seed this vertex is allowed to join.
        // Linear, and that is fine: a cell holds a handful of buckets, because
        // a surface passing through a cell has a handful of distinct normals.
        int found = -1;
        for (std::int32_t bucket = 0; bucket < 8; ++bucket) {
            key.bucket = bucket;
            auto it = lookup.find(key);
            if (it == lookup.end()) break;
            const Cluster& c = clusters[std::size_t(it->second)];
            const bool normal_ok = Dot(n, c.seed_normal) >= cos_weld;
            const bool color_ok =
                std::fabs(v.color.x - c.seed_color.x) <= options.color_weld &&
                std::fabs(v.color.y - c.seed_color.y) <= options.color_weld &&
                std::fabs(v.color.z - c.seed_color.z) <= options.color_weld;
            if (normal_ok && color_ok) { found = it->second; break; }
        }
        if (found < 0) {
            // A new bucket, unless the cell is already full of them. Past eight
            // the cell is not a cluster any more and merging further would only
            // damage the shape, so the ninth and later vertices join the last
            // bucket -- which caps the memory and, on a pathological input,
            // simply means less simplification rather than a wrong mesh.
            std::int32_t bucket = 0;
            for (; bucket < 8; ++bucket) {
                key.bucket = bucket;
                if (lookup.find(key) == lookup.end()) break;
            }
            if (bucket >= 8) {
                key.bucket = 7;
                found = lookup[key];
            } else {
                key.bucket = bucket;
                found = int(clusters.size());
                clusters.emplace_back();
                clusters.back().seed_normal = n;
                clusters.back().seed_color = v.color;
                lookup[key] = found;
            }
        }

        Cluster& c = clusters[std::size_t(found)];
        c.quadric.Add(quadrics[i]);
        c.position_sum = c.position_sum + p;
        c.normal_sum = c.normal_sum + n;
        c.color_sum = Vec4{c.color_sum.x + v.color.x, c.color_sum.y + v.color.y,
                           c.color_sum.z + v.color.z, c.color_sum.w + v.color.w};
        c.uv_sum = Vec4{c.uv_sum.x + v.uv.x, c.uv_sum.y + v.uv.y, 0.0f, 0.0f};
        ++c.count;
        vertex_cluster[i] = found;
    }

    // Nothing merged: hand back the input rather than a rebuilt copy of it.
    if (clusters.size() >= in.vertices.size()) return in;

    // --- emit ----------------------------------------------------------------
    Mesh out;
    out.vertices.reserve(clusters.size());
    for (Cluster& c : clusters) {
        if (c.count == 0) continue;
        const float inv = 1.0f / float(c.count);
        const Vec3 centroid = c.position_sum * inv;

        Vec3 position = centroid;
        Vec3 solved;
        if (SolveQuadric(c.quadric, &solved)) {
            // BOUNDED to the cluster's own neighbourhood. The solve is exact
            // for the planes it was given and says nothing about how far away
            // the answer is allowed to be: a nearly-flat cluster is nearly
            // singular rather than singular, so the determinant test passes and
            // the intersection point comes back hundreds of metres away. One
            // spike like that ruins a whole model, and it is the failure mode
            // that makes people give up on quadrics.
            if (Length(solved - centroid) <= cell * 1.5f) position = solved;
        }

        VertexIn v{};
        v.position = Vec4{position.x, position.y, position.z, 0.0f};
        const Vec3 n = c.normal_sum;
        const Vec3 unit = Dot(n, n) > 1e-12f ? Normalize(n) : Vec3{0.0f, 1.0f, 0.0f};
        v.normal = Vec4{unit.x, unit.y, unit.z, 0.0f};
        v.color = Vec4{c.color_sum.x * inv, c.color_sum.y * inv,
                       c.color_sum.z * inv, c.color_sum.w * inv};
        v.uv = Vec4{c.uv_sum.x * inv, c.uv_sum.y * inv, 0.0f, 0.0f};
        c.output = int(out.vertices.size());
        out.vertices.push_back(v);
    }

    out.indices.reserve(in.indices.size());
    for (std::size_t t = 0; t < triangles; ++t) {
        const int c0 = vertex_cluster[in.indices[t * 3 + 0]];
        const int c1 = vertex_cluster[in.indices[t * 3 + 1]];
        const int c2 = vertex_cluster[in.indices[t * 3 + 2]];
        if (c0 < 0 || c1 < 0 || c2 < 0) continue;
        // DEGENERATE triangles dropped, not emitted with zero area. Keeping
        // them would mean the index count never falls, which is the whole
        // point -- and a zero-area triangle still costs a primitive setup and
        // still shades a quad on most hardware.
        if (c0 == c1 || c1 == c2 || c0 == c2) continue;
        const int o0 = clusters[std::size_t(c0)].output;
        const int o1 = clusters[std::size_t(c1)].output;
        const int o2 = clusters[std::size_t(c2)].output;
        if (o0 < 0 || o1 < 0 || o2 < 0) continue;
        out.indices.push_back(std::uint16_t(o0));
        out.indices.push_back(std::uint16_t(o1));
        out.indices.push_back(std::uint16_t(o2));
    }

    // THE BOUNDS ARE THE ORIGINAL'S, not the simplified mesh's.
    //
    // A level of detail must occupy the same space as the object it stands in
    // for: culling uses these, and a smaller sphere on a coarser level means an
    // object vanishes when it switches -- at a distance, which is exactly where
    // nobody is looking closely enough to see why.
    out.bounds = in.bounds;
    if (out.indices.empty()) return in;
    return out;
}

std::vector<Mesh> BuildLodChain(const Mesh& in, int levels,
                                const SimplifyOptions& options) {
    std::vector<Mesh> chain;
    levels = std::max(1, levels);
    chain.push_back(in);
    SimplifyOptions o = options;
    for (int i = 1; i < levels; ++i) {
        // Each level is simplified from the ORIGINAL, not from the level above.
        // Simplifying a simplified mesh compounds the error -- and worse, the
        // second pass's quadrics are built from planes that are already wrong,
        // so it optimises toward the previous level's mistakes.
        o.cell_fraction = options.cell_fraction * float(1 << i);
        Mesh next = Simplify(in, o);
        // Stop when a level stops being cheaper than the one before it. Emitting
        // it anyway would cost a draw call and an indirect argument slot to
        // render the same triangles.
        if (next.indices.size() >= chain.back().indices.size()) break;
        chain.push_back(std::move(next));
    }
    return chain;
}

}  // namespace eng
