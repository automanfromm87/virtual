// No test framework — from scratch means from scratch.
//
// The floor plan generators are pure geometry, so everything here runs on the
// CPU with no GPU in sight. That matters: a hole in a floor or a wall wound
// inside out is a geometry bug, and hunting it in pixels is far slower.
#include "engine/geometry/floorplan.h"

#include <cmath>
#include <cstdio>

namespace {

int g_failures = 0;

void Fail(const char* what, const char* which, int line) {
    std::fprintf(stderr, "floorplan_test.cc:%d  [%s] %s\n", line, which, what);
    ++g_failures;
}

#define CHECK(cond) \
    do { if (!(cond)) Fail(#cond, which, __LINE__); } while (0)

// Is the point (px, py) covered by any triangle, seen face-on?
//
// Neither vertices nor centroids work for this. A box's vertices only sit on
// its corners, so "no vertex in the doorway" is true of ANY box decomposition
// including one that never cut the hole. And a face this coarse is two
// triangles, whose centroids land at the 1/3 and 2/3 marks and can straddle a
// doorway entirely. Asking whether the wall is SOLID at a point is the actual
// question.
//
// The walls here run along X, so projecting to XY is face-on for them. The end
// caps and the top/bottom faces project to zero-area slivers and drop out.
// `zmax` restricts the test to one wall: a rectangular room has two walls
// running along X, and both project onto the same XY region, so without this
// the far wall answers for the near one.
bool CoversXY(const eng::Mesh& m, float px, float py, float zmax = 0.5f) {
    auto side = [](float ax, float ay, float bx, float by, float cx, float cy) {
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    };
    for (std::size_t t = 0; t + 2 < m.indices.size(); t += 3) {
        const eng::Vec4& a = m.vertices[m.indices[t + 0]].position;
        const eng::Vec4& b = m.vertices[m.indices[t + 1]].position;
        const eng::Vec4& c = m.vertices[m.indices[t + 2]].position;
        if (std::fabs((a.z + b.z + c.z) / 3.0f) > zmax) continue;
        const float d1 = side(a.x, a.y, b.x, b.y, px, py);
        const float d2 = side(b.x, b.y, c.x, c.y, px, py);
        const float d3 = side(c.x, c.y, a.x, a.y, px, py);
        const bool neg = d1 < 0 || d2 < 0 || d3 < 0;
        const bool pos = d1 > 0 || d2 > 0 || d3 > 0;
        if (!(neg && pos)) return true;
    }
    return false;
}

// Every triangle's RIGHT-HAND-RULE normal must agree with the normal its
// vertices carry.
//
// Checking the stored normal alone proves nothing: it is assigned by hand in
// the generator, so it is right by construction even when the winding is
// backwards. The winding is what back-face culling actually looks at, and a
// reversed floor slab renders as a black room.
int WindingMismatches(const eng::Mesh& m) {
    int bad = 0;
    for (std::size_t t = 0; t + 2 < m.indices.size(); t += 3) {
        const VertexIn& a = m.vertices[m.indices[t + 0]];
        const VertexIn& b = m.vertices[m.indices[t + 1]];
        const VertexIn& c = m.vertices[m.indices[t + 2]];
        const eng::Vec3 pa{a.position.x, a.position.y, a.position.z};
        const eng::Vec3 pb{b.position.x, b.position.y, b.position.z};
        const eng::Vec3 pc{c.position.x, c.position.y, c.position.z};
        const eng::Vec3 face = eng::Cross(pb - pa, pc - pa);
        if (eng::Length(face) < 1e-9f) continue;
        const eng::Vec3 stored{a.normal.x, a.normal.y, a.normal.z};
        if (eng::Dot(face, stored) <= 0.0f) ++bad;
    }
    return bad;
}

// Total surface area of a mesh, by summing triangle areas.
float SurfaceArea(const eng::Mesh& m) {
    float a = 0.0f;
    for (std::size_t t = 0; t + 2 < m.indices.size(); t += 3) {
        const eng::Vec4& p0 = m.vertices[m.indices[t + 0]].position;
        const eng::Vec4& p1 = m.vertices[m.indices[t + 1]].position;
        const eng::Vec4& p2 = m.vertices[m.indices[t + 2]].position;
        const eng::Vec3 u{p1.x - p0.x, p1.y - p0.y, p1.z - p0.z};
        const eng::Vec3 v{p2.x - p0.x, p2.y - p0.y, p2.z - p0.z};
        a += eng::Length(eng::Cross(u, v)) * 0.5f;
    }
    return a;
}

eng::FloorPlan OneRoom() {
    eng::FloorPlan p;
    p.wall_height = 2.7f;
    p.wall_thickness = 0.1f;
    // 4 x 3 metre room, counter-clockwise in plan space (x, z).
    p.floor_outline = {{0, 0}, {4, 0}, {4, 3}, {0, 3}};
    p.walls = {{{0, 0}, {4, 0}, {}},
               {{4, 0}, {4, 3}, {}},
               {{4, 3}, {0, 3}, {}},
               {{0, 3}, {0, 0}, {}}};
    return p;
}

}  // namespace

int main() {
    using namespace eng;

    {
        const char* which = "triangulate square";
        const std::vector<Vec2> square = {{0, 0}, {2, 0}, {2, 2}, {0, 2}};
        const std::vector<std::uint16_t> t = TriangulatePolygon(square);
        // n-gon -> n-2 triangles, always. A triangulator that gives up early
        // leaves an invisible hole in the floor.
        CHECK(t.size() == 2 * 3);
        for (std::uint16_t i : t) CHECK(i < square.size());
    }

    {
        const char* which = "triangulate L-shape";
        // Concave: an ear clipper that ignores reflex corners produces
        // triangles outside the polygon.
        const std::vector<Vec2> l = {{0, 0}, {4, 0}, {4, 2}, {2, 2}, {2, 4}, {0, 4}};
        const std::vector<std::uint16_t> t = TriangulatePolygon(l);
        CHECK(t.size() == 4 * 3);

        // The triangles must cover exactly the polygon's area — 12 for this L.
        // Too much means a triangle strayed outside the reflex corner.
        float area = 0.0f;
        for (std::size_t i = 0; i + 2 < t.size(); i += 3) {
            const Vec2 a = l[t[i]], b = l[t[i + 1]], c = l[t[i + 2]];
            area += std::fabs(Cross2(b - a, c - a)) * 0.5f;
        }
        CHECK(std::fabs(area - 12.0f) < 1e-3f);
    }

    {
        const char* which = "clockwise input";
        // Same square wound the other way. Plan data comes from wherever it
        // comes from; the triangulator has to cope.
        const std::vector<Vec2> cw = {{0, 2}, {2, 2}, {2, 0}, {0, 0}};
        CHECK(TriangulatePolygon(cw).size() == 2 * 3);
    }

    {
        const char* which = "degenerate";
        CHECK(TriangulatePolygon({}).empty());
        CHECK(TriangulatePolygon({{0, 0}, {1, 1}}).empty());
    }

    {
        const char* which = "solid walls";
        const FloorPlan p = OneRoom();
        const Mesh m = MakeWalls(p, Vec4{1, 1, 1, 1});
        CHECK(!m.vertices.empty());
        CHECK(m.indices.size() % 3 == 0);
        // Four walls, one box each, six faces of two triangles.
        CHECK(m.indices.size() == 4 * 6 * 2 * 3);
        // Nothing pokes below the floor or above the ceiling.
        for (const VertexIn& v : m.vertices) {
            CHECK(v.position.y >= -1e-4f);
            CHECK(v.position.y <= p.wall_height + 1e-4f);
        }
        CHECK(m.bounds.radius > 0.0f);
        CHECK(WindingMismatches(m) == 0);
    }

    {
        const char* which = "a door removes material";
        FloorPlan solid = OneRoom();
        FloorPlan holed = OneRoom();
        // A 0.9 m doorway, floor to 2.05 m, in the first wall.
        holed.walls[0].openings = {{1.5f, 2.4f, 0.0f, 2.05f}};

        const Mesh a = MakeWalls(solid, Vec4{1, 1, 1, 1});
        const Mesh b = MakeWalls(holed, Vec4{1, 1, 1, 1});
        // A door leaves a lintel above it, so the wall becomes THREE boxes
        // where it was one: more geometry, less material.
        CHECK(b.indices.size() > a.indices.size());
        CHECK(SurfaceArea(b) < SurfaceArea(a));

        // The solid wall IS solid where the doorway will be; the holed one is
        // not. This is the check that actually distinguishes them.
        CHECK(CoversXY(a, 1.95f, 1.0f));
        CHECK(!CoversXY(b, 1.95f, 1.0f));
        // ...but the lintel above the door survives, and so do both jambs.
        CHECK(CoversXY(b, 1.95f, 2.4f));
        CHECK(CoversXY(b, 0.8f, 1.0f));
        CHECK(CoversXY(b, 3.2f, 1.0f));
    }

    {
        const char* which = "a window leaves a sill";
        FloorPlan p = OneRoom();
        p.walls[0].openings = {{1.0f, 2.0f, 0.9f, 2.1f}};
        const Mesh m = MakeWalls(p, Vec4{1, 1, 1, 1});
        // Jamb, sill, lintel, jamb on the windowed wall plus one box each for
        // the other three: seven boxes in total.
        CHECK(m.indices.size() == 7 * 6 * 2 * 3);

        // A window differs from a door by having material BELOW it. Both have a
        // lintel above; only the window has a sill.
        CHECK(CoversXY(m, 1.5f, 0.45f));   // sill
        CHECK(CoversXY(m, 1.5f, 2.4f));    // lintel
        CHECK(!CoversXY(m, 1.5f, 1.5f));   // the opening itself
    }

    {
        const char* which = "glass fills windows only";
        FloorPlan p = OneRoom();
        p.walls[0].openings = {{1.0f, 2.0f, 0.9f, 2.1f}};   // window
        p.walls[1].openings = {{0.5f, 1.4f, 0.0f, 2.05f}};  // door
        const Mesh g = MakeGlass(p, Vec4{1, 1, 1, 0.3f});
        // One pane, one box. A doorway gets no glass.
        CHECK(g.indices.size() == 6 * 2 * 3);
        CHECK(WindingMismatches(g) == 0);
        for (const VertexIn& v : g.vertices) {
            CHECK(v.position.y > 0.85f);
            CHECK(v.position.y < 2.15f);
        }
    }

    {
        const char* which = "floor slab";
        const FloorPlan p = OneRoom();
        const Mesh m = MakeFloorSlab(p, Vec4{1, 1, 1, 1});
        // Top + bottom (2 triangles each) plus a four-sided skirt.
        CHECK(m.indices.size() == (2 + 2) * 3 + 4 * 2 * 3);

        // Top face area must equal the room's 4x3.
        float top_area = 0.0f;
        for (std::size_t t = 0; t + 2 < m.indices.size(); t += 3) {
            const VertexIn& a = m.vertices[m.indices[t]];
            if (a.normal.y < 0.99f) continue;
            const Vec4& p0 = m.vertices[m.indices[t + 0]].position;
            const Vec4& p1 = m.vertices[m.indices[t + 1]].position;
            const Vec4& p2 = m.vertices[m.indices[t + 2]].position;
            top_area += std::fabs(Cross2(Vec2{p1.x - p0.x, p1.z - p0.z},
                                         Vec2{p2.x - p0.x, p2.z - p0.z})) * 0.5f;
        }
        CHECK(std::fabs(top_area - 12.0f) < 1e-3f);

        // THE one that was wrong: the slab's top face has to actually face up.
        CHECK(WindingMismatches(m) == 0);

        // Slab hangs below y = 0 and never rises above it.
        for (const VertexIn& v : m.vertices) {
            CHECK(v.position.y <= 1e-4f);
            CHECK(v.position.y >= -p.floor_thickness - 1e-4f);
        }
    }

    if (g_failures == 0) std::printf("floorplan_test: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
