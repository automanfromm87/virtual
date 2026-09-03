#include "engine/geometry/hull.h"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace eng::geom {
namespace {

struct Face {
    int a = 0, b = 0, c = 0;
    Vec3 normal;
    float offset = 0.0f;  // plane is Dot(normal, p) = offset
    bool dead = false;
};

Face MakeFace(const std::vector<Vec3>& v, int a, int b, int c) {
    Face f{a, b, c, {}, 0.0f, false};
    const Vec3 n = Cross(v[std::size_t(b)] - v[std::size_t(a)],
                         v[std::size_t(c)] - v[std::size_t(a)]);
    const float len = Length(n);
    f.normal = len > 0.0f ? n * (1.0f / len) : Vec3{0, 1, 0};
    f.offset = Dot(f.normal, v[std::size_t(a)]);
    return f;
}

// An undirected edge, so that the two faces sharing it cancel out when both are
// removed. Ordering the pair is what makes "appears twice" detectable.
struct Edge {
    int u, v;
    bool operator==(const Edge& o) const { return u == o.u && v == o.v; }
};
struct EdgeHash {
    std::size_t operator()(const Edge& e) const {
        return std::size_t(e.u) * 73856093u ^ std::size_t(e.v) * 19349663u;
    }
};

}  // namespace

Vec3 Hull::Normal(std::size_t face) const {
    if (face * 3 + 2 >= indices.size()) return Vec3{0, 1, 0};
    const Vec3& a = vertices[indices[face * 3]];
    const Vec3& b = vertices[indices[face * 3 + 1]];
    const Vec3& c = vertices[indices[face * 3 + 2]];
    return Normalize(Cross(b - a, c - a));
}

float Hull::Volume() const {
    // Divergence theorem: the volume is the sum of the signed volumes of the
    // tetrahedra from the origin to each face. Points outside contribute
    // negative volumes that cancel exactly, so the origin need not be inside.
    float v = 0.0f;
    for (std::size_t f = 0; f + 2 < indices.size(); f += 3) {
        const Vec3& a = vertices[indices[f]];
        const Vec3& b = vertices[indices[f + 1]];
        const Vec3& c = vertices[indices[f + 2]];
        v += Dot(a, Cross(b, c));
    }
    return v / 6.0f;
}

Vec3 Hull::Centroid() const {
    float vol = 0.0f;
    Vec3 acc{0, 0, 0};
    for (std::size_t f = 0; f + 2 < indices.size(); f += 3) {
        const Vec3& a = vertices[indices[f]];
        const Vec3& b = vertices[indices[f + 1]];
        const Vec3& c = vertices[indices[f + 2]];
        const float dv = Dot(a, Cross(b, c)) / 6.0f;
        vol += dv;
        // A tetrahedron's centroid is the mean of its four corners, one of
        // which is the origin.
        acc = acc + (a + b + c) * (dv * 0.25f);
    }
    if (std::fabs(vol) < 1e-12f) {
        // Degenerate: fall back to the mean of the vertices, which at least
        // lands inside the point set rather than at the origin.
        if (vertices.empty()) return Vec3{0, 0, 0};
        Vec3 m{0, 0, 0};
        for (const Vec3& p : vertices) m = m + p;
        return m * (1.0f / float(vertices.size()));
    }
    return acc * (1.0f / vol);
}

Mat3 Hull::Inertia() const {
    // Tetrahedron decomposition, each one from the ORIGIN to a face, with the
    // same sign cancellation the volume uses. The canonical unit-tetrahedron
    // covariance is scaled by the determinant of the corner matrix and
    // accumulated, then shifted to the centroid at the end.
    //
    // Doing it per tetrahedron rather than by sampling is not fussiness: a
    // voxel or point-sample approximation of a long thin shape gets the two
    // large moments roughly right and the small one badly wrong, and the small
    // one is the axis it spins fastest about.
    float xx = 0, yy = 0, zz = 0, xy = 0, xz = 0, yz = 0, vol = 0;
    for (std::size_t f = 0; f + 2 < indices.size(); f += 3) {
        const Vec3& a = vertices[indices[f]];
        const Vec3& b = vertices[indices[f + 1]];
        const Vec3& c = vertices[indices[f + 2]];
        // SIX times the signed volume: that is what the scalar triple product
        // of three corners is, and using it directly as the weight makes every
        // moment six times too large -- which looks like a plausible tensor and
        // makes everything spin as if it were six times as reluctant.
        const float dv = Dot(a, Cross(b, c)) / 6.0f;
        vol += dv;
        const float px[4] = {0, a.x, b.x, c.x};
        const float py[4] = {0, a.y, b.y, c.y};
        const float pz[4] = {0, a.z, b.z, c.z};
        // Integral of x*x over a tetrahedron with corners p0..p3, divided by
        // volume, is (sum_i sum_j p_i.x p_j.x + sum_i p_i.x^2) / 20.
        const auto pair = [&](const float* u, const float* v) {
            float s = 0.0f;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) s += u[i] * v[j];
            for (int i = 0; i < 4; ++i) s += u[i] * v[i];
            return s / 20.0f;
        };
        xx += dv * pair(px, px);
        yy += dv * pair(py, py);
        zz += dv * pair(pz, pz);
        xy += dv * pair(px, py);
        xz += dv * pair(px, pz);
        yz += dv * pair(py, pz);
    }
    if (std::fabs(vol) < 1e-12f) return Mat3::Diagonal(Vec3{0, 0, 0});

    // Parallel axis, back to the centroid.
    const Vec3 c = Centroid();
    xx -= vol * c.x * c.x;
    yy -= vol * c.y * c.y;
    zz -= vol * c.z * c.z;
    xy -= vol * c.x * c.y;
    xz -= vol * c.x * c.z;
    yz -= vol * c.y * c.z;

    // The tensor itself, not the raw second moments: the diagonal is the sum of
    // the OTHER two, and the off-diagonals are negated. Getting that wrong
    // produces a plausible symmetric matrix that spins everything backwards.
    return Mat3{{{yy + zz, -xy, -xz}, {-xy, xx + zz, -yz}, {-xz, -yz, xx + yy}}};
}

bool Hull::Contains(Vec3 p, float tolerance) const {
    if (Empty()) return false;
    for (std::size_t f = 0; f < FaceCount(); ++f) {
        const Vec3& a = vertices[indices[f * 3]];
        if (Dot(Normal(f), p - a) > tolerance) return false;
    }
    return true;
}

Hull ConvexHull(std::span<const Vec3> points, float tolerance) {
    Hull hull;
    if (points.size() < 4) return hull;

    // --- an initial tetrahedron ---------------------------------------------
    //
    // Four points that are not coplanar. Found by extremes rather than by
    // trying combinations: the two furthest apart along a coordinate axis, then
    // the point furthest from that LINE, then the point furthest from that
    // PLANE. Each step maximises the thing the next step needs, which is what
    // keeps the seed tetrahedron from being a sliver on a nearly flat cloud.
    int i0 = 0, i1 = 0;
    for (std::size_t i = 0; i < points.size(); ++i) {
        if (points[i].x < points[std::size_t(i0)].x) i0 = int(i);
        if (points[i].x > points[std::size_t(i1)].x) i1 = int(i);
    }
    if (i0 == i1) return hull;  // every point has the same x AND is the same
    const Vec3 p0 = points[std::size_t(i0)], p1 = points[std::size_t(i1)];
    const Vec3 axis = p1 - p0;
    if (Dot(axis, axis) < tolerance * tolerance) return hull;

    int i2 = -1;
    float best = tolerance;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const float d = Length(Cross(points[i] - p0, axis));
        if (d > best) { best = d; i2 = int(i); }
    }
    if (i2 < 0) return hull;  // collinear
    const Vec3 p2 = points[std::size_t(i2)];

    const Vec3 n = Normalize(Cross(p1 - p0, p2 - p0));
    int i3 = -1;
    best = tolerance;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const float d = std::fabs(Dot(points[i] - p0, n));
        if (d > best) { best = d; i3 = int(i); }
    }
    if (i3 < 0) return hull;  // coplanar

    std::vector<Vec3> v = {p0, p1, p2, points[std::size_t(i3)]};
    std::vector<Face> faces;
    // Wind every seed face outward: if the fourth point is in front of a face,
    // that face is inward and its winding is flipped.
    const auto seed = [&](int a, int b, int c, int opposite) {
        Face f = MakeFace(v, a, b, c);
        if (Dot(f.normal, v[std::size_t(opposite)]) - f.offset > 0.0f)
            f = MakeFace(v, a, c, b);
        faces.push_back(f);
    };
    seed(0, 1, 2, 3);
    seed(0, 1, 3, 2);
    seed(0, 2, 3, 1);
    seed(1, 2, 3, 0);

    // --- add the remaining points -------------------------------------------
    std::vector<char> used(points.size(), 0);
    used[std::size_t(i0)] = used[std::size_t(i1)] = 1;
    used[std::size_t(i2)] = used[std::size_t(i3)] = 1;

    std::unordered_set<Edge, EdgeHash> horizon;
    for (std::size_t pi = 0; pi < points.size(); ++pi) {
        if (used[pi]) continue;
        const Vec3 p = points[pi];

        // Which faces can "see" the point. Their union is a connected patch,
        // and its boundary is the horizon.
        bool any = false;
        for (Face& f : faces)
            if (!f.dead && Dot(f.normal, p) - f.offset > tolerance) any = true;
        if (!any) continue;  // inside: contributes nothing

        horizon.clear();
        for (Face& f : faces) {
            if (f.dead || Dot(f.normal, p) - f.offset <= tolerance) continue;
            f.dead = true;
            // An edge shared by two removed faces appears twice and is
            // INTERIOR to the patch, not on the horizon. Cancelling in pairs
            // is what distinguishes the two, and skipping it leaves interior
            // edges in the rim -- which builds a non-convex fan and, a few
            // points later, a hull with faces inside itself.
            const int tri[3][2] = {{f.a, f.b}, {f.b, f.c}, {f.c, f.a}};
            for (const auto& e : tri) {
                const Edge fwd{e[0], e[1]}, rev{e[1], e[0]};
                if (horizon.erase(rev) == 0) horizon.insert(fwd);
            }
        }

        const int added = int(v.size());
        v.push_back(p);
        for (const Edge& e : horizon) {
            // The horizon edges are already wound so that the new faces come
            // out consistently oriented; no per-face flip is needed.
            faces.push_back(MakeFace(v, e.u, e.v, added));
        }
        faces.erase(std::remove_if(faces.begin(), faces.end(),
                                   [](const Face& f) { return f.dead; }),
                    faces.end());
    }

    // --- compact -------------------------------------------------------------
    // Only the vertices that survived on a face. The interior points are the
    // whole reason to build a hull rather than collide against the raw cloud.
    std::vector<int> remap(v.size(), -1);
    for (const Face& f : faces)
        for (int idx : {f.a, f.b, f.c})
            if (remap[std::size_t(idx)] < 0) {
                remap[std::size_t(idx)] = int(hull.vertices.size());
                hull.vertices.push_back(v[std::size_t(idx)]);
            }
    if (hull.vertices.size() > 65535) return Hull{};  // 16-bit indices
    for (const Face& f : faces)
        for (int idx : {f.a, f.b, f.c})
            hull.indices.push_back(std::uint16_t(remap[std::size_t(idx)]));

    // A hull with no volume is degenerate however it was arrived at, and saying
    // so here is much cheaper than a division by zero in the inertia tensor.
    if (hull.Volume() < 1e-9f) return Hull{};
    return hull;
}

}  // namespace eng::geom
