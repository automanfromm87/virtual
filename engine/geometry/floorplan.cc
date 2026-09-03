#include "engine/geometry/floorplan.h"

#include <algorithm>
#include <cmath>

namespace eng {
namespace {

// Appends an axis-aligned box given in WALL-LOCAL coordinates:
//   u along the wall, v up, n across the thickness.
// The caller supplies the basis so this stays one function instead of six.
void AppendBox(Mesh& m, Vec3 origin, Vec3 u_axis, Vec3 v_axis, Vec3 n_axis,
               Vec2 u_range, Vec2 v_range, float half_thickness, Vec4 color) {
    if (u_range.y - u_range.x <= 1e-5f || v_range.y - v_range.x <= 1e-5f) return;

    // Eight corners, then six quads. Sharing corners would mean sharing
    // normals, which on a box means rounding every edge.
    const Vec3 c[8] = {
        origin + u_axis * u_range.x + v_axis * v_range.x + n_axis * -half_thickness,
        origin + u_axis * u_range.y + v_axis * v_range.x + n_axis * -half_thickness,
        origin + u_axis * u_range.y + v_axis * v_range.y + n_axis * -half_thickness,
        origin + u_axis * u_range.x + v_axis * v_range.y + n_axis * -half_thickness,
        origin + u_axis * u_range.x + v_axis * v_range.x + n_axis * half_thickness,
        origin + u_axis * u_range.y + v_axis * v_range.x + n_axis * half_thickness,
        origin + u_axis * u_range.y + v_axis * v_range.y + n_axis * half_thickness,
        origin + u_axis * u_range.x + v_axis * v_range.y + n_axis * half_thickness,
    };
    // Each face: four corner indices wound CCW seen from outside, plus its
    // outward normal.
    const int face[6][4] = {
        {4, 5, 6, 7},  // +n
        {1, 0, 3, 2},  // -n
        {5, 1, 2, 6},  // +u
        {0, 4, 7, 3},  // -u
        {3, 7, 6, 2},  // +v
        {0, 1, 5, 4},  // -v
    };
    const Vec3 normal[6] = {n_axis,  n_axis * -1.0f, u_axis,
                            u_axis * -1.0f, v_axis, v_axis * -1.0f};
    // Texture the box in world metres so a tiled material keeps a constant
    // scale no matter how long the wall segment is.
    const float du = u_range.y - u_range.x;
    const float dv = v_range.y - v_range.x;
    const Vec2 extent[6] = {{du, dv}, {du, dv}, {2 * half_thickness, dv},
                            {2 * half_thickness, dv}, {du, 2 * half_thickness},
                            {du, 2 * half_thickness}};

    for (int f = 0; f < 6; ++f) {
        const auto base = std::uint16_t(m.vertices.size());
        const float uvs[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
        for (int k = 0; k < 4; ++k) {
            VertexIn v{};
            const Vec3 p = c[face[f][k]];
            v.position = Vec4{p.x, p.y, p.z, 0.0f};
            v.normal = Vec4{normal[f].x, normal[f].y, normal[f].z, 0.0f};
            v.color = color;
            v.uv = Vec4{uvs[k][0] * extent[f].x, uvs[k][1] * extent[f].y, 0, 0};
            m.vertices.push_back(v);
        }
        m.indices.push_back(base);
        m.indices.push_back(std::uint16_t(base + 1));
        m.indices.push_back(std::uint16_t(base + 2));
        m.indices.push_back(base);
        m.indices.push_back(std::uint16_t(base + 2));
        m.indices.push_back(std::uint16_t(base + 3));
    }
}

void FitBounds(Mesh& m) {
    if (m.vertices.empty()) return;
    Vec3 lo{m.vertices[0].position.x, m.vertices[0].position.y,
            m.vertices[0].position.z};
    Vec3 hi = lo;
    for (const VertexIn& v : m.vertices) {
        lo.x = std::min(lo.x, v.position.x);
        lo.y = std::min(lo.y, v.position.y);
        lo.z = std::min(lo.z, v.position.z);
        hi.x = std::max(hi.x, v.position.x);
        hi.y = std::max(hi.y, v.position.y);
        hi.z = std::max(hi.z, v.position.z);
    }
    m.bounds.center = (lo + hi) * 0.5f;
    m.bounds.radius = Length(hi - m.bounds.center);
}

// A triangular prism: the triangle a-b-c extruded by +/- half_thickness along
// its own normal. Used for the gable ends, which are the one part of a building
// that is genuinely triangular.
void AppendPrism(Mesh& m, Vec3 a, Vec3 b, Vec3 c, float half_thickness,
                 Vec4 color) {
    Vec3 n = Cross(b - a, c - a);
    const float len = Length(n);
    if (len < 1e-9f) return;
    n = n * (1.0f / len);
    const Vec3 off = n * half_thickness;

    const Vec3 front[3] = {a + off, b + off, c + off};
    const Vec3 back[3] = {a - off, b - off, c - off};

    auto push = [&](Vec3 p, Vec3 nrm, float u, float v) {
        VertexIn vt{};
        vt.position = Vec4{p.x, p.y, p.z, 0.0f};
        vt.normal = Vec4{nrm.x, nrm.y, nrm.z, 0.0f};
        vt.color = color;
        vt.uv = Vec4{u, v, 0, 0};
        m.vertices.push_back(vt);
        return std::uint16_t(m.vertices.size() - 1);
    };

    // Two caps. The back one is wound the other way so it faces -n.
    const std::uint16_t f0 = push(front[0], n, 0, 0);
    const std::uint16_t f1 = push(front[1], n, 1, 0);
    const std::uint16_t f2 = push(front[2], n, 0, 1);
    m.indices.push_back(f0); m.indices.push_back(f1); m.indices.push_back(f2);

    const Vec3 bn = n * -1.0f;
    const std::uint16_t b0 = push(back[0], bn, 0, 0);
    const std::uint16_t b1 = push(back[1], bn, 1, 0);
    const std::uint16_t b2 = push(back[2], bn, 0, 1);
    m.indices.push_back(b0); m.indices.push_back(b2); m.indices.push_back(b1);

    // Three side quads, closing the prism.
    for (int e = 0; e < 3; ++e) {
        const int e2 = (e + 1) % 3;
        Vec3 edge_n = Cross(front[e2] - front[e], back[e] - front[e]);
        const float el = Length(edge_n);
        if (el < 1e-9f) continue;
        edge_n = edge_n * (1.0f / el);
        const std::uint16_t q0 = push(front[e], edge_n, 0, 0);
        const std::uint16_t q1 = push(front[e2], edge_n, 1, 0);
        const std::uint16_t q2 = push(back[e2], edge_n, 1, 1);
        const std::uint16_t q3 = push(back[e], edge_n, 0, 1);
        m.indices.push_back(q0); m.indices.push_back(q1); m.indices.push_back(q2);
        m.indices.push_back(q0); m.indices.push_back(q2); m.indices.push_back(q3);
    }
}

float PolygonArea(const std::vector<Vec2>& p) {
    float a = 0.0f;
    for (std::size_t i = 0, n = p.size(); i < n; ++i)
        a += Cross2(p[i], p[(i + 1) % n]);
    return a * 0.5f;
}

bool PointInTriangle(Vec2 p, Vec2 a, Vec2 b, Vec2 c) {
    const float d1 = Cross2(b - a, p - a);
    const float d2 = Cross2(c - b, p - b);
    const float d3 = Cross2(a - c, p - c);
    const bool neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
    const bool pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
    return !(neg && pos);
}

}  // namespace

std::vector<std::uint16_t> TriangulatePolygon(const std::vector<Vec2>& outline) {
    std::vector<std::uint16_t> out;
    const std::size_t n = outline.size();
    if (n < 3) return out;

    // Work on a CCW copy so "convex" always means a left turn.
    std::vector<int> idx(n);
    for (std::size_t i = 0; i < n; ++i) idx[i] = int(i);
    if (PolygonArea(outline) < 0.0f) std::reverse(idx.begin(), idx.end());

    // Ear clipping. O(n^2), which for a floor plan's few dozen corners is
    // instant and much easier to get right than anything faster.
    int guard = int(n) * int(n) + 16;
    while (idx.size() > 3 && guard-- > 0) {
        bool clipped = false;
        for (std::size_t i = 0; i < idx.size(); ++i) {
            const int ia = idx[(i + idx.size() - 1) % idx.size()];
            const int ib = idx[i];
            const int ic = idx[(i + 1) % idx.size()];
            const Vec2 a = outline[ia], b = outline[ib], c = outline[ic];

            // Reflex corners are not ears.
            if (Cross2(b - a, c - b) <= 0.0f) continue;

            // An ear must also be empty: no other vertex inside it.
            bool empty = true;
            for (int other : idx) {
                if (other == ia || other == ib || other == ic) continue;
                if (PointInTriangle(outline[other], a, b, c)) { empty = false; break; }
            }
            if (!empty) continue;

            out.push_back(std::uint16_t(ia));
            out.push_back(std::uint16_t(ib));
            out.push_back(std::uint16_t(ic));
            idx.erase(idx.begin() + long(i));
            clipped = true;
            break;
        }
        // No ear found: the polygon is self-intersecting or degenerate. Bail
        // rather than spin, and let the caller's test catch the short output.
        if (!clipped) break;
    }
    if (idx.size() == 3) {
        out.push_back(std::uint16_t(idx[0]));
        out.push_back(std::uint16_t(idx[1]));
        out.push_back(std::uint16_t(idx[2]));
    }
    return out;
}

Mesh MakeWalls(const FloorPlan& plan, Vec4 color) {
    Mesh m;
    const float h = plan.wall_height;
    const float ht = plan.wall_thickness * 0.5f;

    for (const WallSpec& w : plan.walls) {
        const Vec2 delta = w.to - w.from;
        const float len = Length(delta);
        if (len < 1e-4f) continue;
        const Vec2 dir = delta * (1.0f / len);
        const Vec2 nrm = Perp(dir);

        const Vec3 origin{w.from.x, 0.0f, w.from.y};
        const Vec3 u_axis{dir.x, 0.0f, dir.y};
        const Vec3 v_axis{0.0f, 1.0f, 0.0f};
        const Vec3 n_axis{nrm.x, 0.0f, nrm.y};

        // Sort and clamp the openings so the sweep below can assume order.
        std::vector<Opening> holes = w.openings;
        std::sort(holes.begin(), holes.end(),
                  [](const Opening& a, const Opening& b) { return a.start < b.start; });

        float cursor = 0.0f;
        for (const Opening& o : holes) {
            const float s = std::clamp(o.start, 0.0f, len);
            const float e = std::clamp(o.end, s, len);
            if (e <= cursor) continue;

            // Full-height jamb between the previous hole and this one.
            AppendBox(m, origin, u_axis, v_axis, n_axis, {cursor, s}, {0.0f, h}, ht, color);
            // Sill below the hole, lintel above it. A door has no sill.
            const float b = std::clamp(o.bottom, 0.0f, h);
            const float t = std::clamp(o.top, b, h);
            AppendBox(m, origin, u_axis, v_axis, n_axis, {s, e}, {0.0f, b}, ht, color);
            AppendBox(m, origin, u_axis, v_axis, n_axis, {s, e}, {t, h}, ht, color);
            cursor = e;
        }
        AppendBox(m, origin, u_axis, v_axis, n_axis, {cursor, len}, {0.0f, h}, ht, color);
    }
    FitBounds(m);
    return m;
}

Mesh MakeFloorSlab(const FloorPlan& plan, Vec4 color) {
    Mesh m;
    const std::vector<Vec2>& poly = plan.floor_outline;
    const std::vector<std::uint16_t> tris = TriangulatePolygon(poly);
    if (tris.empty()) return m;

    const float top = 0.0f;
    const float bottom = -plan.floor_thickness;

    // Top face, then bottom face wound the other way, then a skirt around the
    // edge so the slab is a closed solid rather than two loose sheets.
    for (int face = 0; face < 2; ++face) {
        const float y = face ? bottom : top;
        const Vec4 n = face ? Vec4{0, -1, 0, 0} : Vec4{0, 1, 0, 0};
        const auto base = std::uint16_t(m.vertices.size());
        for (const Vec2& p : poly) {
            VertexIn v{};
            v.position = Vec4{p.x, y, p.y, 0.0f};
            v.normal = n;
            v.color = color;
            v.uv = Vec4{p.x, p.y, 0, 0};  // world metres, so tiling stays even
            m.vertices.push_back(v);
        }
        // The triangulator returns counter-clockwise in PLAN space (x, z). Seen
        // from +Y — which is where you look at a floor from — that same order
        // is CLOCKWISE, because plan space is a left-handed view of the XZ
        // plane. So the TOP face is the one that needs reversing.
        //
        // Getting this backwards culls the floor's upper surface and leaves you
        // looking at the underside of the slab from inside the room: the floor
        // reads as nearly black and the bug looks like a lighting problem.
        for (std::size_t i = 0; i + 2 < tris.size(); i += 3) {
            if (face) {  // underside keeps the triangulator's order
                m.indices.push_back(std::uint16_t(base + tris[i + 0]));
                m.indices.push_back(std::uint16_t(base + tris[i + 1]));
                m.indices.push_back(std::uint16_t(base + tris[i + 2]));
            } else {
                m.indices.push_back(std::uint16_t(base + tris[i + 2]));
                m.indices.push_back(std::uint16_t(base + tris[i + 1]));
                m.indices.push_back(std::uint16_t(base + tris[i + 0]));
            }
        }
    }

    const bool ccw = PolygonArea(poly) > 0.0f;
    for (std::size_t i = 0, n = poly.size(); i < n; ++i) {
        const Vec2 a = poly[i];
        const Vec2 b = poly[(i + 1) % n];
        Vec2 outward = Normalize(Perp(b - a));
        if (ccw) outward = outward * -1.0f;

        const auto base = std::uint16_t(m.vertices.size());
        const Vec3 quad[4] = {{a.x, bottom, a.y}, {b.x, bottom, b.y},
                              {b.x, top, b.y},    {a.x, top, a.y}};
        for (int k = 0; k < 4; ++k) {
            VertexIn v{};
            v.position = Vec4{quad[k].x, quad[k].y, quad[k].z, 0.0f};
            v.normal = Vec4{outward.x, 0.0f, outward.y, 0.0f};
            v.color = color;
            v.uv = Vec4{float(k & 1), float((k >> 1) & 1), 0, 0};
            m.vertices.push_back(v);
        }
        // Perp() of an edge points INWARD for a counter-clockwise outline, so
        // `outward` above was negated — and the winding has to be negated with
        // it. Flipping only the stored normal leaves the geometric normal
        // pointing the other way, and back-face culling follows the geometry.
        if (ccw) {
            m.indices.push_back(base);
            m.indices.push_back(std::uint16_t(base + 2));
            m.indices.push_back(std::uint16_t(base + 1));
            m.indices.push_back(base);
            m.indices.push_back(std::uint16_t(base + 3));
            m.indices.push_back(std::uint16_t(base + 2));
        } else {
            m.indices.push_back(base);
            m.indices.push_back(std::uint16_t(base + 1));
            m.indices.push_back(std::uint16_t(base + 2));
            m.indices.push_back(base);
            m.indices.push_back(std::uint16_t(base + 2));
            m.indices.push_back(std::uint16_t(base + 3));
        }
    }
    FitBounds(m);
    return m;
}

Mesh MakeGableRoof(const FloorPlan& plan, float rise, float overhang,
                   Vec4 color) {
    Mesh m;
    if (plan.floor_outline.size() < 3 || rise <= 0.0f) return m;

    float x0 = plan.floor_outline[0].x, x1 = x0;
    float z0 = plan.floor_outline[0].y, z1 = z0;
    for (const Vec2& p : plan.floor_outline) {
        x0 = std::min(x0, p.x); x1 = std::max(x1, p.x);
        z0 = std::min(z0, p.y); z1 = std::max(z1, p.y);
    }

    const float eave = plan.wall_height;
    const float ridge = eave + rise;
    const float ht = 0.06f;  // roof slab half-thickness

    // Ridge along the LONGER axis, which is what a roof does: the slopes shed
    // water down the short direction.
    const bool ridge_along_x = (x1 - x0) >= (z1 - z0);
    const float ex0 = x0 - overhang, ex1 = x1 + overhang;
    const float ez0 = z0 - overhang, ez1 = z1 + overhang;

    // Each slope is a thin box in a rotated basis: u along the ridge, v up the
    // slope, n out of the roof plane. Reusing AppendBox keeps the eaves and the
    // ridge closed solids rather than infinitely thin sheets.
    auto slope = [&](Vec3 eave_start, Vec3 ridge_start, Vec3 along) {
        const Vec3 up_slope = ridge_start - eave_start;
        const float slope_len = Length(up_slope);
        if (slope_len < 1e-5f) return;
        const Vec3 v_axis = up_slope * (1.0f / slope_len);
        const Vec3 u_axis = Normalize(along);
        Vec3 n_axis = Normalize(Cross(u_axis, v_axis));
        if (n_axis.y < 0.0f) n_axis = n_axis * -1.0f;  // face the sky
        AppendBox(m, eave_start, u_axis, v_axis, n_axis, {0.0f, Length(along)},
                  {0.0f, slope_len}, ht, color);
    };

    if (ridge_along_x) {
        const float zc = (z0 + z1) * 0.5f;
        slope({ex0, eave, ez0}, {ex0, ridge, zc}, {ex1 - ex0, 0, 0});
        slope({ex0, eave, ez1}, {ex0, ridge, zc}, {ex1 - ex0, 0, 0});
        // Gable walls sit on the WALL line, not the overhang.
        AppendPrism(m, {x0, eave, z0}, {x0, eave, z1}, {x0, ridge, zc},
                    plan.wall_thickness * 0.5f, color);
        AppendPrism(m, {x1, eave, z0}, {x1, eave, z1}, {x1, ridge, zc},
                    plan.wall_thickness * 0.5f, color);
    } else {
        const float xc = (x0 + x1) * 0.5f;
        slope({ex0, eave, ez0}, {xc, ridge, ez0}, {0, 0, ez1 - ez0});
        slope({ex1, eave, ez0}, {xc, ridge, ez0}, {0, 0, ez1 - ez0});
        AppendPrism(m, {x0, eave, z0}, {x1, eave, z0}, {xc, ridge, z0},
                    plan.wall_thickness * 0.5f, color);
        AppendPrism(m, {x0, eave, z1}, {x1, eave, z1}, {xc, ridge, z1},
                    plan.wall_thickness * 0.5f, color);
    }
    FitBounds(m);
    return m;
}

Mesh MakeDoorLeaves(const FloorPlan& plan, float ajar, Vec4 color) {
    Mesh m;
    for (const WallSpec& w : plan.walls) {
        const Vec2 delta = w.to - w.from;
        const float len = Length(delta);
        if (len < 1e-4f) continue;
        const Vec2 dir = delta * (1.0f / len);
        const Vec2 nrm = Perp(dir);

        for (const Opening& o : w.openings) {
            if (o.bottom > 1e-4f) continue;  // a window, not a door
            const float width = o.end - o.start;
            if (width <= 1e-4f) continue;

            // Hinged at `start` and swung open by `ajar`, so the leaf reads as
            // a door rather than as a panel filling the hole.
            const float c = std::cos(ajar), s = std::sin(ajar);
            const Vec2 swing{dir.x * c + nrm.x * s, dir.y * c + nrm.y * s};
            const Vec2 face = Perp(swing);
            const Vec3 hinge{w.from.x + dir.x * o.start, 0.0f,
                             w.from.y + dir.y * o.start};
            AppendBox(m, hinge, Vec3{swing.x, 0.0f, swing.y}, Vec3{0, 1, 0},
                      Vec3{face.x, 0.0f, face.y}, {0.0f, width},
                      {0.02f, o.top - 0.02f}, 0.022f, color);
        }
    }
    FitBounds(m);
    return m;
}

Mesh MakeWindowFrames(const FloorPlan& plan, float depth, Vec4 color) {
    Mesh m;
    const float b = 0.055f;  // frame bar half-width
    for (const WallSpec& w : plan.walls) {
        const Vec2 delta = w.to - w.from;
        const float len = Length(delta);
        if (len < 1e-4f) continue;
        const Vec2 dir = delta * (1.0f / len);
        const Vec2 nrm = Perp(dir);
        const Vec3 origin{w.from.x, 0.0f, w.from.y};
        const Vec3 u{dir.x, 0.0f, dir.y}, v{0, 1, 0}, n{nrm.x, 0.0f, nrm.y};

        for (const Opening& o : w.openings) {
            if (o.bottom <= 1e-4f) continue;  // doors get a leaf instead
            // Four bars around the opening plus one mullion down the middle.
            AppendBox(m, origin, u, v, n, {o.start, o.start + b * 2}, {o.bottom, o.top}, depth, color);
            AppendBox(m, origin, u, v, n, {o.end - b * 2, o.end}, {o.bottom, o.top}, depth, color);
            AppendBox(m, origin, u, v, n, {o.start, o.end}, {o.bottom, o.bottom + b * 2}, depth, color);
            AppendBox(m, origin, u, v, n, {o.start, o.end}, {o.top - b * 2, o.top}, depth, color);
            const float mid = (o.start + o.end) * 0.5f;
            AppendBox(m, origin, u, v, n, {mid - b, mid + b}, {o.bottom, o.top}, depth * 0.7f, color);
        }
    }
    FitBounds(m);
    return m;
}

Mesh MakeGlass(const FloorPlan& plan, Vec4 color) {
    Mesh m;
    for (const WallSpec& w : plan.walls) {
        const Vec2 delta = w.to - w.from;
        const float len = Length(delta);
        if (len < 1e-4f) continue;
        const Vec2 dir = delta * (1.0f / len);
        const Vec2 nrm = Perp(dir);

        for (const Opening& o : w.openings) {
            // A door (sill at the floor) gets no glass.
            if (o.bottom <= 1e-4f) continue;
            const Vec3 origin{w.from.x, 0.0f, w.from.y};
            // Paper-thin, and double sided by way of the material's cull mode:
            // you have to be able to see a window from indoors too.
            AppendBox(m, origin, Vec3{dir.x, 0.0f, dir.y}, Vec3{0, 1, 0},
                      Vec3{nrm.x, 0.0f, nrm.y}, {o.start, o.end}, {o.bottom, o.top},
                      0.008f, color);
        }
    }
    FitBounds(m);
    return m;
}

}  // namespace eng
