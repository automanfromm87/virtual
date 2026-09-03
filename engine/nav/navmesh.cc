#include "engine/nav/navmesh.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <queue>
#include <unordered_map>

namespace eng::nav {
namespace {

// A SOLID SPAN in one column: the world is voxelised into vertical intervals of
// solid, not into a set of surfaces.
//
// The distinction is the whole difference between a navmesh that knows about
// walls and one that does not. A surface-based field records "there is floor at
// y = 0 and something at y = 2" and cannot tell a solid two-metre wall from a
// two-metre archway -- both have the same two surfaces, and the headroom test
// says the agent fits under either. A span records that the wall is solid from
// 0 to 2, and the floor inside it stops existing.
//
// It is also why a triangle has to be CLIPPED to each cell rather than sampled
// at its centre. A wall's side faces are vertical: edge-on in plan, zero area
// in XZ, and a centre sample skips them entirely -- which is exactly how a
// first attempt here produced a mesh with no walls in it at all.
struct Span {
    float lo = 0.0f;
    float hi = 0.0f;
    // Whether the surface at `hi` came from a triangle shallow enough to stand
    // on. Carried through the merge below: when two spans join, the flag of
    // whichever contributes the higher top wins, because that is the surface.
    bool walkable = false;
    // Filled during filtering.
    float clearance = 0.0f;
    int poly = -1;
};

struct Column {
    std::vector<Span> spans;  // ascending, non-overlapping
};

// Sutherland-Hodgman against one axis-aligned plane, in the XZ plane.
// `axis` is 0 for x and 2 for z; `keep_greater` selects which side survives.
int ClipPoly(const Vec3* in, int count, float at, int axis, bool keep_greater,
             Vec3* out) {
    int n = 0;
    for (int i = 0; i < count; ++i) {
        const Vec3& a = in[i];
        const Vec3& b = in[(i + 1) % count];
        const float da = (axis == 0 ? a.x : a.z) - at;
        const float db = (axis == 0 ? b.x : b.z) - at;
        const bool ina = keep_greater ? da >= 0.0f : da <= 0.0f;
        const bool inb = keep_greater ? db >= 0.0f : db <= 0.0f;
        if (ina) out[n++] = a;
        if (ina != inb) {
            const float t = da / (da - db);
            out[n++] = a + (b - a) * t;
        }
    }
    return n;
}

// The y range of a triangle over one cell, or false when it does not cover it.
bool TriangleSpanInCell(const Vec3& a, const Vec3& b, const Vec3& c, float x0,
                        float z0, float x1, float z1, float* out_lo,
                        float* out_hi) {
    Vec3 buf[2][12];
    buf[0][0] = a;
    buf[0][1] = b;
    buf[0][2] = c;
    int count = 3, which = 0;
    const float planes[4] = {x0, x1, z0, z1};
    const int axes[4] = {0, 0, 2, 2};
    const bool greater[4] = {true, false, true, false};
    for (int p = 0; p < 4; ++p) {
        count = ClipPoly(buf[which], count, planes[p], axes[p], greater[p],
                         buf[1 - which]);
        which = 1 - which;
        if (count < 3) return false;
    }
    float lo = 1e30f, hi = -1e30f;
    for (int i = 0; i < count; ++i) {
        lo = std::min(lo, buf[which][i].y);
        hi = std::max(hi, buf[which][i].y);
    }
    *out_lo = lo;
    *out_hi = hi;
    return true;
}

// Inserts a span, merging with anything it touches. `merge_gap` is how close two
// spans have to be to count as one surface -- a step the agent can climb is not
// a gap it can fall into.
void AddSpan(Column& col, float lo, float hi, bool walkable, float merge_gap) {
    Span s{lo, hi, walkable, 0.0f, -1};
    std::vector<Span> merged;
    merged.reserve(col.spans.size() + 1);
    for (Span& existing : col.spans) {
        if (existing.hi + merge_gap < s.lo) {
            merged.push_back(existing);
            continue;
        }
        if (s.hi + merge_gap < existing.lo) {
            merged.push_back(s);
            s = existing;
            continue;
        }
        // Overlapping or touching: fuse. The WALKABLE FLAG follows whichever
        // span reaches higher, because that is the surface an agent would stand
        // on -- a walkable floor fused with the solid wall standing on it must
        // come out unwalkable, or the mesh has floor inside the wall.
        if (std::fabs(existing.hi - s.hi) <= merge_gap)
            s.walkable = s.walkable || existing.walkable;
        else if (existing.hi > s.hi)
            s.walkable = existing.walkable;
        s.lo = std::min(s.lo, existing.lo);
        s.hi = std::max(s.hi, existing.hi);
    }
    merged.push_back(s);
    std::sort(merged.begin(), merged.end(),
              [](const Span& a, const Span& b) { return a.lo < b.lo; });
    col.spans = std::move(merged);
}

int Ceil(float v) { return int(std::ceil(v)); }
int Floor(float v) { return int(std::floor(v)); }

}  // namespace

struct NavMesh::Impl {
    BuildConfig config;
    BuildStats stats;
    Vec3 origin{0.0f, 0.0f, 0.0f};
    int width = 0, depth = 0;  // cells along x and z
    std::vector<Column> columns;

    std::vector<Poly> polys;
    std::vector<Portal> portals;
    mutable int last_visited = 0;

    [[nodiscard]] int Index(int x, int z) const { return z * width + x; }
    [[nodiscard]] bool InBounds(int x, int z) const {
        return x >= 0 && z >= 0 && x < width && z < depth;
    }
    [[nodiscard]] Vec3 CellCentre(int x, int z, float y) const {
        return Vec3{origin.x + (float(x) + 0.5f) * config.cell_size, y,
                    origin.z + (float(z) + 0.5f) * config.cell_size};
    }
    // The walkable span in this column whose TOP is nearest `y`, or -1. The
    // top is the surface; the bottom is inside the solid.
    [[nodiscard]] int SurfaceNear(int x, int z, float y, float tolerance) const {
        if (!InBounds(x, z)) return -1;
        const Column& c = columns[std::size_t(Index(x, z))];
        int best = -1;
        float best_d = tolerance;
        for (std::size_t i = 0; i < c.spans.size(); ++i) {
            if (!c.spans[i].walkable) continue;
            const float d = std::fabs(c.spans[i].hi - y);
            if (d <= best_d) {
                best_d = d;
                best = int(i);
            }
        }
        return best;
    }
    [[nodiscard]] float SurfaceHeight(int x, int z, int span) const {
        return columns[std::size_t(Index(x, z))].spans[std::size_t(span)].hi;
    }
};

NavMesh::NavMesh() : impl_(std::make_unique<Impl>()) {}
NavMesh::~NavMesh() = default;
NavMesh::NavMesh(NavMesh&&) noexcept = default;
NavMesh& NavMesh::operator=(NavMesh&&) noexcept = default;

bool NavMesh::Valid() const { return !impl_->polys.empty(); }
int NavMesh::PolyCount() const { return int(impl_->polys.size()); }
const Poly& NavMesh::GetPoly(int i) const { return impl_->polys[std::size_t(i)]; }
std::span<const Portal> NavMesh::Portals() const {
    return {impl_->portals.data(), impl_->portals.size()};
}
const BuildStats& NavMesh::Stats() const { return impl_->stats; }
int NavMesh::LastSearchVisited() const { return impl_->last_visited; }

NavMesh NavMesh::Build(std::span<const Vec3> vertices,
                       std::span<const std::uint32_t> indices,
                       const BuildConfig& config) {
    const auto t0 = std::chrono::steady_clock::now();
    NavMesh mesh;
    Impl& im = *mesh.impl_;
    im.config = config;
    if (vertices.empty() || indices.size() < 3) return mesh;

    // --- 1. bounds ------------------------------------------------------------
    Vec3 lo = config.bounds_min, hi = config.bounds_max;
    if (!(hi.x > lo.x && hi.y > lo.y && hi.z > lo.z)) {
        lo = Vec3{1e30f, 1e30f, 1e30f};
        hi = Vec3{-1e30f, -1e30f, -1e30f};
        for (const Vec3& v : vertices) {
            lo = Vec3{std::min(lo.x, v.x), std::min(lo.y, v.y), std::min(lo.z, v.z)};
            hi = Vec3{std::max(hi.x, v.x), std::max(hi.y, v.y), std::max(hi.z, v.z)};
        }
        // A margin, so a floor whose edge lands exactly on the boundary still
        // gets its last row of cells rasterised.
        const float m = config.cell_size * 2.0f;
        lo = lo - Vec3{m, m, m};
        hi = hi + Vec3{m, m, m};
    }
    const float cs = std::max(config.cell_size, 1e-3f);
    im.origin = lo;
    im.width = std::max(1, Ceil((hi.x - lo.x) / cs));
    im.depth = std::max(1, Ceil((hi.z - lo.z) / cs));
    // A guard against a cell size that would allocate the world. Fifty million
    // columns is four gigabytes of spans; a caller who asked for that made a
    // units mistake, and failing is better than swapping for a minute.
    if (static_cast<long long>(im.width) * im.depth > 50'000'000LL) return mesh;
    im.columns.resize(std::size_t(im.width) * std::size_t(im.depth));
    im.stats.cells_total = im.width * im.depth;

    // --- 2. rasterise ---------------------------------------------------------
    //
    // Each triangle contributes a surface to every column its footprint covers,
    // at the height of the triangle over that column's centre. Sampling at the
    // centre rather than clipping the triangle to the cell is an approximation
    // and the right one here: the grid is finer than the agent, so a surface
    // that only clips a corner of a cell is one the agent could not have used.
    const float cos_slope =
        std::cos(config.agent_max_slope_degrees * 3.14159265f / 180.0f);
    const std::size_t triangles = indices.size() / 3;
    for (std::size_t t = 0; t < triangles; ++t) {
        const std::uint32_t i0 = indices[t * 3 + 0];
        const std::uint32_t i1 = indices[t * 3 + 1];
        const std::uint32_t i2 = indices[t * 3 + 2];
        if (i0 >= vertices.size() || i1 >= vertices.size() || i2 >= vertices.size())
            continue;
        const Vec3 a = vertices[i0], b = vertices[i1], c = vertices[i2];
        const Vec3 n_raw = Cross(b - a, c - a);
        const float area2 = Length(n_raw);
        if (area2 < 1e-12f) continue;
        const Vec3 n = n_raw * (1.0f / area2);
        // ABSOLUTE, so winding does not decide walkability. A level with
        // inconsistently wound floors is normal, and refusing half of them
        // would look like the navmesh randomly having holes.
        const bool walkable = std::fabs(n.y) >= cos_slope;

        const float tri_lo_x = std::min({a.x, b.x, c.x});
        const float tri_hi_x = std::max({a.x, b.x, c.x});
        const float tri_lo_z = std::min({a.z, b.z, c.z});
        const float tri_hi_z = std::max({a.z, b.z, c.z});
        const int x0 = std::max(0, Floor((tri_lo_x - lo.x) / cs));
        const int x1 = std::min(im.width - 1, Floor((tri_hi_x - lo.x) / cs));
        const int z0 = std::max(0, Floor((tri_lo_z - lo.z) / cs));
        const int z1 = std::min(im.depth - 1, Floor((tri_hi_z - lo.z) / cs));

        for (int z = z0; z <= z1; ++z)
            for (int x = x0; x <= x1; ++x) {
                const float cx0 = lo.x + float(x) * cs;
                const float cz0 = lo.z + float(z) * cs;
                float span_lo = 0.0f, span_hi = 0.0f;
                if (!TriangleSpanInCell(a, b, c, cx0, cz0, cx0 + cs, cz0 + cs,
                                        &span_lo, &span_hi))
                    continue;
                // AT LEAST ONE VOXEL TALL. A perfectly flat floor clips to a
                // zero-height span, and two zero-height spans a hair apart do
                // not merge -- so a floor made of two coplanar triangles grows
                // a phantom ceiling over itself.
                if (span_hi - span_lo < config.cell_height)
                    span_hi = span_lo + config.cell_height;
                AddSpan(im.columns[std::size_t(im.Index(x, z))], span_lo, span_hi,
                        walkable, config.cell_height);
            }
    }

    // --- 3. filter ------------------------------------------------------------
    for (Column& col : im.columns) {
        for (std::size_t i = 0; i < col.spans.size(); ++i) {
            // HEADROOM: from this span's TOP to the next span's BOTTOM. Using
            // the next span's top instead would measure through the solid and
            // let an agent stand inside a thick ceiling.
            col.spans[i].clearance = (i + 1 < col.spans.size())
                                         ? col.spans[i + 1].lo - col.spans[i].hi
                                         : 1e30f;
            if (col.spans[i].clearance < config.agent_height)
                col.spans[i].walkable = false;
            if (col.spans[i].walkable) ++im.stats.cells_walkable;
        }
    }

    // LEDGES. A cell whose neighbour drops further than the agent can climb is
    // the edge of a drop, and standing on the very lip is where a character
    // gets pushed off by its own collision radius. Removing the lip is cheaper
    // and more robust than trying to keep the agent on it.
    {
        std::vector<std::pair<int, int>> to_clear;  // column index, surface index
        for (int z = 0; z < im.depth; ++z)
            for (int x = 0; x < im.width; ++x) {
                const int ci = im.Index(x, z);
                Column& col = im.columns[std::size_t(ci)];
                for (std::size_t s = 0; s < col.spans.size(); ++s) {
                    if (!col.spans[s].walkable) continue;
                    const float y = col.spans[s].hi;
                    const int dx[4] = {1, -1, 0, 0};
                    const int dz[4] = {0, 0, 1, -1};
                    for (int d = 0; d < 4; ++d) {
                        const int nx = x + dx[d], nz = z + dz[d];
                        if (!im.InBounds(nx, nz)) continue;
                        const int ns = im.SurfaceNear(nx, nz, y, config.agent_max_climb);
                        if (ns < 0) {
                            to_clear.emplace_back(ci, int(s));
                            break;
                        }
                    }
                }
            }
        for (const auto& [ci, si] : to_clear)
            im.columns[std::size_t(ci)].spans[std::size_t(si)].walkable = false;
    }

    // --- 4. erode by the agent's radius ---------------------------------------
    //
    // A brushfire distance transform, then a threshold. Doing this here rather
    // than shrinking the finished path is not a preference: a path is routed
    // through the middle of a polygon, and if the polygon reaches into a gap
    // narrower than the agent then the route already goes somewhere it cannot.
    {
        const int radius_cells = std::max(0, Ceil(config.agent_radius / cs) - 1);
        for (int pass = 0; pass < radius_cells; ++pass) {
            std::vector<std::pair<int, int>> to_clear;
            for (int z = 0; z < im.depth; ++z)
                for (int x = 0; x < im.width; ++x) {
                    const int ci = im.Index(x, z);
                    Column& col = im.columns[std::size_t(ci)];
                    for (std::size_t s = 0; s < col.spans.size(); ++s) {
                        if (!col.spans[s].walkable) continue;
                        const float y = col.spans[s].hi;
                        const int dx[4] = {1, -1, 0, 0};
                        const int dz[4] = {0, 0, 1, -1};
                        for (int d = 0; d < 4; ++d) {
                            const int nx = x + dx[d], nz = z + dz[d];
                            if (im.InBounds(nx, nz) &&
                                im.SurfaceNear(nx, nz, y, config.agent_max_climb) >= 0)
                                continue;
                            to_clear.emplace_back(ci, int(s));
                            break;
                        }
                    }
                }
            // CLEARED AFTER THE WHOLE PASS. Clearing as we go would let one
            // pass erode by several cells, because a cell already removed this
            // pass would count as an obstacle for the next one along.
            for (const auto& [ci, si] : to_clear)
                im.columns[std::size_t(ci)].spans[std::size_t(si)].walkable = false;
        }
        for (const Column& col : im.columns)
            for (const Span& s : col.spans)
                if (s.walkable) ++im.stats.cells_after_erosion;
    }

    // --- 5. merge into rectangles ---------------------------------------------
    //
    // Greedy: take the first unclaimed cell, extend as far right as the row
    // allows, then extend downward as long as every cell of the next row is
    // available at a compatible height. Not optimal -- finding the fewest
    // rectangles covering a region is NP-hard -- and the greedy answer is a
    // small constant factor off, which for a graph searched a few times a
    // second does not matter.
    {
        const auto available = [&](int x, int z, float y) -> int {
            if (!im.InBounds(x, z)) return -1;
            const Column& col = im.columns[std::size_t(im.Index(x, z))];
            const int s = im.SurfaceNear(x, z, y, config.agent_max_climb);
            if (s < 0) return -1;
            if (col.spans[std::size_t(s)].poly >= 0) return -1;
            return s;
        };

        for (int z = 0; z < im.depth; ++z)
            for (int x = 0; x < im.width; ++x) {
                Column& col = im.columns[std::size_t(im.Index(x, z))];
                for (std::size_t s0 = 0; s0 < col.spans.size(); ++s0) {
                    if (!col.spans[s0].walkable || col.spans[s0].poly >= 0)
                        continue;
                    const float seed_y = col.spans[s0].hi;

                    int x1 = x;
                    while (x1 + 1 < im.width && available(x1 + 1, z, seed_y) >= 0) ++x1;

                    int z1 = z;
                    for (;;) {
                        if (z1 + 1 >= im.depth) break;
                        bool row_ok = true;
                        for (int cx = x; cx <= x1; ++cx)
                            if (available(cx, z1 + 1, seed_y) < 0) { row_ok = false; break; }
                        if (!row_ok) break;
                        ++z1;
                    }

                    const int cells = (x1 - x + 1) * (z1 - z + 1);
                    Poly p;
                    p.min = Vec3{lo.x + float(x) * cs, 0.0f, lo.z + float(z) * cs};
                    p.max = Vec3{lo.x + float(x1 + 1) * cs, 0.0f,
                                 lo.z + float(z1 + 1) * cs};
                    p.height_min = 1e30f;
                    p.height_max = -1e30f;
                    const int index = int(im.polys.size());
                    for (int cz = z; cz <= z1; ++cz)
                        for (int cx = x; cx <= x1; ++cx) {
                            Column& c2 = im.columns[std::size_t(im.Index(cx, cz))];
                            const int si = im.SurfaceNear(cx, cz, seed_y,
                                                          config.agent_max_climb);
                            if (si < 0) continue;
                            c2.spans[std::size_t(si)].poly = index;
                            p.height_min =
                                std::min(p.height_min, c2.spans[std::size_t(si)].hi);
                            p.height_max =
                                std::max(p.height_max, c2.spans[std::size_t(si)].hi);
                        }
                    p.min.y = p.height_min;
                    p.max.y = p.height_max;
                    if (cells < config.min_region_cells) {
                        // SLIVERS DROPPED, and the cells released rather than
                        // left claimed -- a claimed cell with no polygon is a
                        // hole in the mesh that nothing can path through and
                        // nothing reports.
                        for (int cz = z; cz <= z1; ++cz)
                            for (int cx = x; cx <= x1; ++cx) {
                                Column& c2 = im.columns[std::size_t(im.Index(cx, cz))];
                                for (Span& sf : c2.spans)
                                    if (sf.poly == index) {
                                        sf.poly = -1;
                                        sf.walkable = false;
                                    }
                            }
                        continue;
                    }
                    im.polys.push_back(p);
                }
            }
    }

    // --- 6. portals -----------------------------------------------------------
    //
    // Two polygons are neighbours where their cells touch at a compatible
    // height. The shared segment is the portal, and its ENDPOINTS must be
    // ordered so that the left one is on the left going from `from` to `to` --
    // the funnel algorithm has no way to work that out and produces a path that
    // crosses itself when it is wrong.
    {
        // Accumulate each pair's shared edge, then emit one portal per pair.
        // Named EdgeSpan and not Span: the file already has a Span, which is a
        // vertical interval of solid, and two different things called Span in
        // one function is how a reader spends five minutes on a type error.
        struct EdgeSpan {
            float lo = 1e30f, hi = -1e30f;
            bool along_x = false;  // the shared edge runs along X
            float at = 0.0f;       // the edge's fixed coordinate
            float height = 0.0f;
        };
        std::unordered_map<std::uint64_t, EdgeSpan> shared;
        const auto key = [](int a, int b) {
            return (std::uint64_t(std::min(a, b)) << 32) | std::uint32_t(std::max(a, b));
        };

        for (int z = 0; z < im.depth; ++z)
            for (int x = 0; x < im.width; ++x) {
                const Column& col = im.columns[std::size_t(im.Index(x, z))];
                for (const Span& s : col.spans) {
                    if (s.poly < 0) continue;
                    const int dx[2] = {1, 0};
                    const int dz[2] = {0, 1};
                    for (int d = 0; d < 2; ++d) {
                        const int nx = x + dx[d], nz = z + dz[d];
                        if (!im.InBounds(nx, nz)) continue;
                        const int ns =
                            im.SurfaceNear(nx, nz, s.hi, config.agent_max_climb);
                        if (ns < 0) continue;
                        const Span& other =
                            im.columns[std::size_t(im.Index(nx, nz))].spans[std::size_t(ns)];
                        if (other.poly < 0 || other.poly == s.poly) continue;

                        EdgeSpan& sp = shared[key(s.poly, other.poly)];
                        sp.height = (s.hi + other.hi) * 0.5f;
                        if (d == 0) {
                            // Neighbour along +X: the shared edge runs along Z.
                            sp.along_x = false;
                            sp.at = lo.x + float(x + 1) * cs;
                            sp.lo = std::min(sp.lo, lo.z + float(z) * cs);
                            sp.hi = std::max(sp.hi, lo.z + float(z + 1) * cs);
                        } else {
                            sp.along_x = true;
                            sp.at = lo.z + float(z + 1) * cs;
                            sp.lo = std::min(sp.lo, lo.x + float(x) * cs);
                            sp.hi = std::max(sp.hi, lo.x + float(x + 1) * cs);
                        }
                    }
                }
            }

        // One directed portal each way, grouped by source so a polygon's
        // portals are contiguous.
        std::vector<std::vector<Portal>> by_poly(im.polys.size());
        for (const auto& [k, sp] : shared) {
            const int a = int(k >> 32);
            const int b = int(k & 0xFFFFFFFFu);
            const auto make = [&](int from, int to) {
                Portal p;
                p.from = from;
                p.to = to;
                Vec3 e0, e1;
                if (sp.along_x) {
                    e0 = Vec3{sp.lo, sp.height, sp.at};
                    e1 = Vec3{sp.hi, sp.height, sp.at};
                } else {
                    e0 = Vec3{sp.at, sp.height, sp.lo};
                    e1 = Vec3{sp.at, sp.height, sp.hi};
                }
                // ORDERED by the direction of travel. The cross product of the
                // travel direction with the edge says which endpoint is on the
                // left; getting it backwards makes the funnel swap its sides
                // and the path zig-zag across every portal.
                const Vec3 fc = (im.polys[std::size_t(from)].min +
                                 im.polys[std::size_t(from)].max) * 0.5f;
                const Vec3 tc = (im.polys[std::size_t(to)].min +
                                 im.polys[std::size_t(to)].max) * 0.5f;
                const Vec3 travel = Vec3{tc.x - fc.x, 0.0f, tc.z - fc.z};
                const Vec3 edge = Vec3{e1.x - e0.x, 0.0f, e1.z - e0.z};
                // In a right-handed system with Y up, travel x edge points up
                // when e1 is to the LEFT of travel.
                const float side = travel.z * edge.x - travel.x * edge.z;
                p.left = side > 0.0f ? e1 : e0;
                p.right = side > 0.0f ? e0 : e1;
                return p;
            };
            by_poly[std::size_t(a)].push_back(make(a, b));
            by_poly[std::size_t(b)].push_back(make(b, a));
        }
        for (std::size_t i = 0; i < im.polys.size(); ++i) {
            im.polys[i].first_portal = int(im.portals.size());
            im.polys[i].portal_count = int(by_poly[i].size());
            for (const Portal& p : by_poly[i]) im.portals.push_back(p);
        }
    }

    im.stats.polys = int(im.polys.size());
    im.stats.portals = int(im.portals.size());
    im.stats.build_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return mesh;
}

int NavMesh::FindPoly(Vec3 point, float search_radius) const {
    const Impl& im = *impl_;
    if (im.polys.empty()) return -1;
    const float cs = im.config.cell_size;
    const int cx = Floor((point.x - im.origin.x) / cs);
    const int cz = Floor((point.z - im.origin.z) / cs);

    // The column the point is in, first. Exact and O(1) for the common case.
    if (im.InBounds(cx, cz)) {
        const int s = im.SurfaceNear(cx, cz, point.y, im.config.agent_height);
        if (s >= 0) {
            const int p = im.columns[std::size_t(im.Index(cx, cz))]
                              .spans[std::size_t(s)].poly;
            if (p >= 0) return p;
        }
    }

    // A widening ring search. Bounded by the radius, because "the nearest
    // polygon however far away" would silently teleport an agent standing
    // inside a wall to the other side of the level.
    const int max_ring = std::max(1, Ceil(search_radius / cs));
    int best = -1;
    float best_d2 = search_radius * search_radius;
    for (int ring = 0; ring <= max_ring; ++ring) {
        bool any = false;
        for (int dz = -ring; dz <= ring; ++dz)
            for (int dx = -ring; dx <= ring; ++dx) {
                if (std::max(std::abs(dx), std::abs(dz)) != ring) continue;
                const int x = cx + dx, z = cz + dz;
                if (!im.InBounds(x, z)) continue;
                const Column& col = im.columns[std::size_t(im.Index(x, z))];
                for (const Span& s : col.spans) {
                    if (s.poly < 0) continue;
                    const Vec3 c = im.CellCentre(x, z, s.hi);
                    const Vec3 d = c - point;
                    const float d2 = Dot(d, d);
                    if (d2 < best_d2) {
                        best_d2 = d2;
                        best = s.poly;
                    }
                    any = true;
                }
            }
        // Stop one ring after the first hit: a closer polygon cannot be more
        // than one ring further out than the nearest cell found.
        if (best >= 0 && any) break;
    }
    return best;
}

float NavMesh::HeightAt(int poly, Vec3 point) const {
    const Impl& im = *impl_;
    if (poly < 0 || std::size_t(poly) >= im.polys.size()) return point.y;
    const float cs = im.config.cell_size;
    const int cx = std::clamp(Floor((point.x - im.origin.x) / cs), 0, im.width - 1);
    const int cz = std::clamp(Floor((point.z - im.origin.z) / cs), 0, im.depth - 1);
    const Column& col = im.columns[std::size_t(im.Index(cx, cz))];
    for (const Span& s : col.spans)
        if (s.poly == poly) return s.hi;
    // The point is outside the polygon's own cells. Its own rectangle's mean
    // is the best available answer and is what a caller sampling just off the
    // edge -- which happens constantly at portals -- should get.
    return (im.polys[std::size_t(poly)].height_min +
            im.polys[std::size_t(poly)].height_max) * 0.5f;
}

Vec3 NavMesh::ClosestPoint(Vec3 point, float search_radius) const {
    const int p = FindPoly(point, search_radius);
    if (p < 0) return point;
    const Poly& poly = impl_->polys[std::size_t(p)];
    const Vec3 clamped{std::clamp(point.x, poly.min.x, poly.max.x), 0.0f,
                       std::clamp(point.z, poly.min.z, poly.max.z)};
    return Vec3{clamped.x, HeightAt(p, clamped), clamped.z};
}

bool NavMesh::FindCorridor(Vec3 start, Vec3 end, std::vector<int>* out) const {
    const Impl& im = *impl_;
    impl_->last_visited = 0;
    if (!out) return false;
    out->clear();
    const int from = FindPoly(start);
    const int to = FindPoly(end);
    if (from < 0 || to < 0) return false;
    if (from == to) {
        out->push_back(from);
        return true;
    }

    const auto centre = [&](int p) {
        const Poly& q = im.polys[std::size_t(p)];
        return Vec3{(q.min.x + q.max.x) * 0.5f, (q.height_min + q.height_max) * 0.5f,
                    (q.min.z + q.max.z) * 0.5f};
    };
    const Vec3 goal = centre(to);

    const std::size_t n = im.polys.size();
    std::vector<float> g(n, 1e30f);
    std::vector<int> came(n, -1);
    std::vector<char> closed(n, 0);
    // (f, poly). A pair rather than a struct: the default comparison orders by
    // f first, which is exactly what a priority queue of A* nodes needs.
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
                        std::greater<>> open;
    g[std::size_t(from)] = 0.0f;
    open.emplace(Length(goal - centre(from)), from);

    while (!open.empty()) {
        const int current = open.top().second;
        open.pop();
        if (closed[std::size_t(current)]) continue;
        closed[std::size_t(current)] = 1;
        ++impl_->last_visited;
        if (current == to) break;

        const Poly& p = im.polys[std::size_t(current)];
        for (int k = 0; k < p.portal_count; ++k) {
            const Portal& portal = im.portals[std::size_t(p.first_portal + k)];
            const int next = portal.to;
            if (next < 0 || closed[std::size_t(next)]) continue;
            // Through the PORTAL's midpoint rather than centre to centre. Two
            // large polygons meeting at a narrow doorway are close by their
            // centres and far apart by the route actually walked, and a cost
            // that ignores the doorway sends the search the wrong way.
            const Vec3 mid = (portal.left + portal.right) * 0.5f;
            const float step = Length(mid - centre(current)) +
                               Length(centre(next) - mid);
            const float tentative = g[std::size_t(current)] + step;
            if (tentative >= g[std::size_t(next)]) continue;
            g[std::size_t(next)] = tentative;
            came[std::size_t(next)] = current;
            // STRAIGHT-LINE distance as the heuristic, which never overestimates
            // -- no route can be shorter than the direct line -- so A* is
            // admissible and the first path it finds is the shortest.
            open.emplace(tentative + Length(goal - centre(next)), next);
        }
    }

    if (came[std::size_t(to)] < 0 && from != to) return false;
    for (int at = to; at >= 0; at = came[std::size_t(at)]) {
        out->push_back(at);
        if (at == from) break;
    }
    std::reverse(out->begin(), out->end());
    return !out->empty() && out->front() == from;
}

namespace {

// Twice the signed area of the triangle, in XZ. Positive when c is to the LEFT
// of the line a->b.
float Cross2(const Vec3& a, const Vec3& b, const Vec3& c) {
    return (b.x - a.x) * (c.z - a.z) - (b.z - a.z) * (c.x - a.x);
}

bool NearlyEqual(const Vec3& a, const Vec3& b) {
    return std::fabs(a.x - b.x) < 1e-4f && std::fabs(a.z - b.z) < 1e-4f;
}

}  // namespace

void NavMesh::StringPull(const NavMesh& mesh, Vec3 start, Vec3 end,
                         std::span<const int> corridor, std::vector<Vec3>* out) {
    if (!out) return;
    out->clear();
    if (corridor.empty()) return;

    // THE FUNNEL. Walk the portals keeping a wedge between the current apex and
    // the tightest left and right bounds seen so far. When a new portal's left
    // edge crosses the right bound, the right bound was a corner the path has
    // to go round -- so it becomes the new apex and the walk restarts from
    // there.
    //
    // The result is the shortest path THROUGH THE CORRIDOR, which is what makes
    // it worth doing: the alternative -- joining the portal midpoints -- is a
    // path that visibly zig-zags along a corridor with nothing in it.
    std::vector<Portal> gates;
    const Impl& im = *mesh.impl_;
    for (std::size_t i = 0; i + 1 < corridor.size(); ++i) {
        const Poly& p = im.polys[std::size_t(corridor[i])];
        bool found = false;
        for (int k = 0; k < p.portal_count; ++k) {
            const Portal& portal = im.portals[std::size_t(p.first_portal + k)];
            if (portal.to == corridor[i + 1]) {
                gates.push_back(portal);
                found = true;
                break;
            }
        }
        if (!found) {
            // The corridor is not actually connected. Falling back to the
            // polygon centres is wrong but visible; silently truncating the
            // path is wrong and looks like the agent changed its mind.
            Portal p2;
            const Poly& q = im.polys[std::size_t(corridor[i + 1])];
            const Vec3 c{(q.min.x + q.max.x) * 0.5f, q.height_min,
                         (q.min.z + q.max.z) * 0.5f};
            p2.left = c;
            p2.right = c;
            gates.push_back(p2);
        }
    }
    // A closing gate at the destination, so the loop below terminates on it
    // rather than needing a separate final step.
    Portal last;
    last.left = end;
    last.right = end;
    gates.push_back(last);

    Vec3 apex = start;
    Vec3 left = start, right = start;
    std::size_t left_index = 0, right_index = 0;
    out->push_back(start);

    for (std::size_t i = 0; i < gates.size(); ++i) {
        const Vec3 gl = gates[i].left;
        const Vec3 gr = gates[i].right;

        // Tighten the RIGHT side.
        if (Cross2(apex, right, gr) <= 0.0f) {
            if (NearlyEqual(apex, right) || Cross2(apex, left, gr) > 0.0f) {
                right = gr;
                right_index = i;
            } else {
                // The right side crossed the left: the LEFT bound is a corner.
                if (!NearlyEqual(out->back(), left)) out->push_back(left);
                apex = left;
                right = apex;
                left = apex;
                i = left_index;
                right_index = left_index;
                continue;
            }
        }
        // And the LEFT side.
        if (Cross2(apex, left, gl) >= 0.0f) {
            if (NearlyEqual(apex, left) || Cross2(apex, right, gl) < 0.0f) {
                left = gl;
                left_index = i;
            } else {
                if (!NearlyEqual(out->back(), right)) out->push_back(right);
                apex = right;
                left = apex;
                right = apex;
                i = right_index;
                left_index = right_index;
                continue;
            }
        }
    }
    if (out->empty() || !NearlyEqual(out->back(), end)) out->push_back(end);

    // The funnel works in XZ, so every corner came out at the height of the
    // portal it belongs to. Sampling the mesh puts them back on the floor.
    for (std::size_t i = 1; i + 1 < out->size(); ++i) {
        const int p = mesh.FindPoly((*out)[i]);
        if (p >= 0) (*out)[i].y = mesh.HeightAt(p, (*out)[i]);
    }
}

bool NavMesh::FindPath(Vec3 start, Vec3 end, std::vector<Vec3>* out) const {
    if (!out) return false;
    out->clear();
    std::vector<int> corridor;
    if (!FindCorridor(start, end, &corridor)) return false;
    // SNAPPED onto the mesh first. A start or end a little off the walkable
    // surface -- a click on a wall, a character whose capsule centre is above
    // the floor -- would otherwise put the funnel's apex outside the corridor,
    // and the first segment of the path would cut a corner through geometry.
    const Vec3 s = ClosestPoint(start);
    const Vec3 e = ClosestPoint(end);
    StringPull(*this, s, e, corridor, out);
    return !out->empty();
}

}  // namespace eng::nav
