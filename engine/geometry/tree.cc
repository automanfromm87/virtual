#include "engine/geometry/tree.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace eng {
namespace {

constexpr float kPi = std::numbers::pi_v<float>;

// A seeded generator, because a tree has to be the same every run.
//
// Not std::mt19937: the sequence has to be reproducible across standard library
// versions as well as across runs, and only a generator written out here is.
// A tree that differs between two machines makes every screenshot comparison
// meaningless.
struct Rng {
    std::uint32_t s;
    explicit Rng(std::uint32_t seed) : s(seed ? seed : 1u) {}
    std::uint32_t Next() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return s;
    }
    float Unit() { return float(Next() & 0xFFFFFFu) / float(0x1000000u); }
    // Symmetric about zero, which is what every jitter below wants.
    float Signed() { return Unit() * 2.0f - 1.0f; }
};

// Any unit vector perpendicular to `n`.
Vec3 Perpendicular(Vec3 n) {
    // Cross with whichever axis `n` is least aligned to, so the cross product
    // never approaches zero and the result is never a normalised nothing.
    const Vec3 axis = std::fabs(n.x) < 0.9f ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    return Normalize(Cross(axis, n));
}

Vec3 RotateAround(Vec3 v, Vec3 axis, float angle) {
    // Rodrigues. Cheaper than building a matrix for one vector, and the axis is
    // already unit here.
    const float c = std::cos(angle), s = std::sin(angle);
    return v * c + Cross(axis, v) * s + axis * (Dot(axis, v) * (1.0f - c));
}

void PushVertex(Mesh& m, Vec3 p, Vec3 n, Vec4 colour, float u, float v) {
    VertexIn out{};
    out.position = Vec4{p.x, p.y, p.z, 0.0f};
    const Vec3 nn = Normalize(n);
    out.normal = Vec4{nn.x, nn.y, nn.z, 0.0f};
    out.color = colour;
    out.uv = Vec4{u, v, 0.0f, 0.0f};
    m.vertices.push_back(out);
}

// One tapered, curved tube from `start` heading along `dir`.
//
// Returns where it ended and which way it was pointing there, so the caller can
// hang children off the tip without recomputing the curve.
struct TubeEnd {
    Vec3 position;
    Vec3 direction;
};

TubeEnd SweepTube(Mesh& m, Vec3 start, Vec3 dir, Vec3 bend, float length,
                  float radius, float tip_radius, int sides, int segments,
                  Vec4 colour) {
    sides = std::max(sides, 3);
    segments = std::max(segments, 1);

    Vec3 p = start;
    Vec3 d = Normalize(dir);
    const auto base = std::uint32_t(m.vertices.size());

    for (int seg = 0; seg <= segments; ++seg) {
        const float t = float(seg) / float(segments);
        const float r = radius + (tip_radius - radius) * t;
        // The ring's own frame. Rebuilt per ring from the CURRENT direction
        // rather than carried along, so a curving branch's rings stay
        // perpendicular to it -- a carried frame twists, and the tube develops
        // a visible spiral crease.
        const Vec3 side = Perpendicular(d);
        const Vec3 up = Cross(d, side);
        for (int i = 0; i < sides; ++i) {
            const float a = float(i) / float(sides) * 2.0f * kPi;
            const Vec3 n = side * std::cos(a) + up * std::sin(a);
            PushVertex(m, p + n * r, n, colour, float(i) / float(sides), t);
        }
        if (seg == segments) break;
        // Advance, then bend. Bending after the step means the ring just
        // written is square to the segment it starts, which is what keeps the
        // tube's silhouette smooth.
        p = p + d * (length / float(segments));
        d = Normalize(d + bend * (1.0f / float(segments)));
    }

    for (int seg = 0; seg < segments; ++seg)
        for (int i = 0; i < sides; ++i) {
            const auto a = std::uint32_t(base + seg * sides + i);
            const auto b = std::uint32_t(base + seg * sides + (i + 1) % sides);
            const auto c = std::uint32_t(a + sides);
            const auto e = std::uint32_t(b + sides);
            // Counter-clockwise seen from OUTSIDE, so back-face culling keeps
            // the near half.
            //
            // The order matters and the first version had it backwards. With
            // (side, up, d) right-handed, the first triangle's edges are d and
            // the ring tangent, and cross(d, tangent) points INWARD -- so every
            // tube was culled from outside and what showed was the inside of
            // the far half, lit by normals facing away from the camera. The
            // trunks came out uniformly dark and read as being in shadow.
            m.indices.push_back(a);
            m.indices.push_back(b);
            m.indices.push_back(c);
            m.indices.push_back(b);
            m.indices.push_back(e);
            m.indices.push_back(c);
        }
    return TubeEnd{p, d};
}

// A low-polygon blob for a leaf cluster, built directly rather than by scaling
// a sphere -- the flattening is non-uniform, and a normal transformed by a
// non-uniform scale is wrong unless you use the inverse transpose. Building it
// in place lets the normal be computed from the shape it actually has.
void AppendBlob(Mesh& m, Vec3 centre, float radius, float flatten, Vec4 colour,
                Rng& rng) {
    constexpr int kStacks = 5, kSlices = 7;
    const auto base = std::uint32_t(m.vertices.size());
    // A per-blob wobble, so a canopy of blobs does not read as a pile of
    // identical balls.
    const float wobble_a = rng.Unit() * 6.2831853f;
    const float wobble_b = 0.82f + rng.Unit() * 0.36f;

    // The blob's surface as a function of its two parameters, so the normal
    // can be taken from the SHAPE rather than from the ellipsoid it started as.
    //
    // The lump below deforms it non-uniformly, and an analytic ellipsoid normal
    // ignores that: measured at 1045 of 17010 triangles whose winding
    // disagreed with their own normals, in exactly the places where the lump's
    // gradient is steepest. Central differences cost four extra evaluations per
    // vertex at bake time and are right whatever the deformation does.
    const auto point = [&](float phi, float theta) {
        const float sp = std::sin(phi);
        const Vec3 unit{sp * std::cos(theta), std::cos(phi), sp * std::sin(theta)};
        const float lump = 1.0f + 0.18f * std::sin(theta * 3.0f + wobble_a) * sp;
        return Vec3{unit.x * radius * lump * wobble_b,
                    unit.y * radius * flatten * lump,
                    unit.z * radius * lump / wobble_b};
    };

    for (int i = 0; i <= kStacks; ++i) {
        const float phi = float(i) / float(kStacks) * kPi;
        for (int j = 0; j <= kSlices; ++j) {
            const float theta = float(j) / float(kSlices) * 2.0f * kPi;
            const Vec3 p = point(phi, theta);
            constexpr float h = 1e-3f;
            const Vec3 dphi = point(phi + h, theta) - point(phi - h, theta);
            const Vec3 dtheta = point(phi, theta + h) - point(phi, theta - h);
            // theta THEN phi. phi runs from the top down and theta around, so
            // cross(dphi, dtheta) points into the blob and this is the order
            // that points out.
            Vec3 n = Cross(dtheta, dphi);
            // At the poles dphi and dtheta are parallel and the cross product
            // vanishes. The position is a perfectly good normal there -- a pole
            // is the one place an ellipsoid's normal and its position agree.
            if (Length(n) < 1e-9f) n = p;
            PushVertex(m, centre + p, Normalize(n), colour,
                       float(j) / float(kSlices), float(i) / float(kStacks));
        }
    }

    for (int i = 0; i < kStacks; ++i)
        for (int j = 0; j < kSlices; ++j) {
            const auto a = std::uint32_t(base + i * (kSlices + 1) + j);
            const auto b = std::uint32_t(a + 1);
            const auto c = std::uint32_t(a + kSlices + 1);
            const auto d = std::uint32_t(c + 1);
            // Same correction as the tube above: phi runs from the top down and
            // theta around, so cross(d/dphi, d/dtheta) points into the blob.
            //
            // The pole rows are FANS, not quads. At a pole every vertex of the
            // row is the same point, so half of each quad has zero area -- and
            // not cleanly zero either: sin(pi) in float is -8.7e-8, so the
            // sliver has an area around 3e-8 and a face normal that is pure
            // rounding noise pointing in an arbitrary direction. Emitting them
            // put 14 of every 70 triangles, a fifth of the blob, through the
            // whole vertex pipeline to cover no pixels.
            if (i > 0) {
                m.indices.push_back(a);
                m.indices.push_back(b);
                m.indices.push_back(c);
            }
            if (i < kStacks - 1) {
                m.indices.push_back(b);
                m.indices.push_back(d);
                m.indices.push_back(c);
            }
        }
}

void Recurse(Tree& tree, const TreeParams& p, Rng& rng, Rng& leaf_rng, Vec3 start,
             Vec3 dir, float length, float radius, int level) {
    const bool last = level >= p.levels;
    const float tip = radius * p.radius_falloff;

    // Branches DROOP: gravity bends them, and the bend is stronger the thinner
    // and longer they are. A tree of straight sticks reads as a diagram.
    Vec3 bend{0.0f, -p.droop * float(level + 1) * 0.35f, 0.0f};
    bend = bend + Vec3{rng.Signed(), 0.0f, rng.Signed()} * (p.droop * 0.25f);
    // AND BACK UP. The droop above is gravity; this is the branch growing
    // toward the light, and it is stronger the more horizontal the branch
    // already is -- a vertical one has nowhere to turn. The two together give
    // the S-curve that reads as growth rather than as a diagram.
    const float horizontal = 1.0f - std::fabs(Normalize(dir).y);
    bend = bend + Vec3{0.0f, p.upward * horizontal, 0.0f};

    const TubeEnd end = SweepTube(tree.trunk, start, dir, bend, length, radius,
                                  tip, p.sides, p.segments, p.bark);

    if (last) {
        for (int i = 0; i < std::max(p.leaf_clusters, 1); ++i) {
            // A SEPARATE STREAM for the leaves, and it is not tidiness.
            //
            // The recursion is depth first, so every draw taken here shifts
            // every branch generated afterwards. With one stream, changing
            // leaf_clusters changes the SKELETON -- which makes a level of
            // detail impossible, because the cheap version would be a different
            // tree standing in a different place rather than the same tree with
            // fewer leaves. Splitting it is what lets the leaf count be a
            // detail setting.
            const Vec3 offset{leaf_rng.Signed(), leaf_rng.Signed() * 0.6f,
                              leaf_rng.Signed()};
            const float shade = 1.0f + leaf_rng.Signed() * p.leaf_variation;
            const Vec4 colour{p.leaf.x * shade, p.leaf.y * shade,
                              p.leaf.z * shade, 1.0f};
            AppendBlob(tree.foliage, end.position + offset * (length * p.leaf_scatter),
                       length * p.leaf_size, p.leaf_flatten, colour, leaf_rng);
        }
        return;
    }

    const int splits = std::max(p.splits, 1);
    // The children are spread around the parent by rotating a leaning direction
    // about it. Starting at a random angle stops every fork in the tree from
    // pointing the same way, which is the single most obvious tell that a tree
    // was generated.
    const float roll = rng.Unit() * 2.0f * kPi;
    const Vec3 axis = Perpendicular(end.direction);
    for (int i = 0; i < splits; ++i) {
        const float around = roll + float(i) / float(splits) * 2.0f * kPi +
                             rng.Signed() * 0.35f;
        const float lean = p.spread + rng.Signed() * p.spread_jitter;
        Vec3 child = RotateAround(end.direction, axis, lean);
        child = Normalize(RotateAround(child, end.direction, around));
        Recurse(tree, p, rng, leaf_rng, end.position, child,
                length * p.length_falloff * (0.85f + rng.Unit() * 0.3f),
                tip, level + 1);
    }
}

void FitBounds(Mesh& m) {
    if (m.vertices.empty()) {
        m.bounds = Bounds{};
        return;
    }
    Vec3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
    for (const VertexIn& v : m.vertices) {
        lo = Vec3{std::min(lo.x, v.position.x), std::min(lo.y, v.position.y),
                  std::min(lo.z, v.position.z)};
        hi = Vec3{std::max(hi.x, v.position.x), std::max(hi.y, v.position.y),
                  std::max(hi.z, v.position.z)};
    }
    m.bounds.center = (lo + hi) * 0.5f;
    float r = 0.0f;
    for (const VertexIn& v : m.vertices)
        r = std::max(r, Length(Vec3{v.position.x, v.position.y, v.position.z} -
                               m.bounds.center));
    m.bounds.radius = r;
}

}  // namespace

Tree MakeTree(const TreeParams& params) {
    Tree tree;
    TreeParams p = params;
    p.levels = std::clamp(p.levels, 1, 6);
    p.splits = std::clamp(p.splits, 1, 5);
    // SIX LEVELS AND FIVE SPLITS is 5^6 = 15625 branches, each a tube of
    // 7 x 5 vertices -- half a million vertices for one tree. The clamp is not
    // taste, it is the difference between a tree and running out of memory.

    Rng rng(p.seed);
    // Derived from the same seed, so a tree is still reproducible from one
    // number, but advanced independently -- see Recurse.
    Rng leaf_rng(p.seed ^ 0x9E3779B9u);
    Recurse(tree, p, rng, leaf_rng, Vec3{0, 0, 0}, Vec3{0, 1, 0}, p.height,
            p.trunk_radius, 0);

    // TANGENT FRAMES, so bark can carry a normal map. Without them the shader's
    // degenerate-frame guard returns the geometric normal and the map silently
    // does nothing -- which looks exactly like a texture that failed to load.
    GenerateTangents(tree.trunk);
    GenerateTangents(tree.foliage);

    FitBounds(tree.trunk);
    FitBounds(tree.foliage);

    // The union of the two, which is what a caller placing the tree needs.
    if (tree.trunk.vertices.empty()) {
        tree.bounds = tree.foliage.bounds;
    } else if (tree.foliage.vertices.empty()) {
        tree.bounds = tree.trunk.bounds;
    } else {
        const Vec3 c = (tree.trunk.bounds.center + tree.foliage.bounds.center) * 0.5f;
        const float d = Length(tree.trunk.bounds.center - tree.foliage.bounds.center);
        tree.bounds.center = c;
        tree.bounds.radius =
            std::max(tree.trunk.bounds.radius, tree.foliage.bounds.radius) + d * 0.5f;
    }
    return tree;
}

void AppendTransformed(Mesh& dst, const Mesh& src, const Mat4& model, Vec4 tint) {
    const auto base = std::uint32_t(dst.vertices.size());
    dst.vertices.reserve(dst.vertices.size() + src.vertices.size());
    for (const VertexIn& v : src.vertices) {
        VertexIn out = v;
        const Vec4 p = model * Vec4{v.position.x, v.position.y, v.position.z, 1.0f};
        out.position = Vec4{p.x, p.y, p.z, 0.0f};
        // w = 0 drops the translation: a normal is a direction.
        const Vec4 n = model * Vec4{v.normal.x, v.normal.y, v.normal.z, 0.0f};
        const Vec3 nn = Normalize(Vec3{n.x, n.y, n.z});
        out.normal = Vec4{nn.x, nn.y, nn.z, 0.0f};
        const Vec4 t = model * Vec4{v.tangent.x, v.tangent.y, v.tangent.z, 0.0f};
        const Vec3 tt = Normalize(Vec3{t.x, t.y, t.z});
        out.tangent = Vec4{tt.x, tt.y, tt.z, v.tangent.w};
        out.color = Vec4{v.color.x * tint.x, v.color.y * tint.y,
                         v.color.z * tint.z, v.color.w * tint.w};
        dst.vertices.push_back(out);
    }
    dst.indices.reserve(dst.indices.size() + src.indices.size());
    for (std::uint32_t i : src.indices) dst.indices.push_back(base + i);
    FitBounds(dst);
}

}  // namespace eng
