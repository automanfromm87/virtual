// The navmesh, and the two questions it has to answer correctly.
//
// CAN I GET THERE. A pathfinder that returns a route through a wall is
// obviously broken; one that fails to find a route that exists is not, because
// the character simply stands still and it looks like a decision. Both are
// checked, and so is the case that matters most: a wall with a doorway, where
// the direct line is blocked and the route is long.
//
// IS THE PATH STRAIGHT. This is the one that separates a navmesh from a grid.
// A* over cells produces a staircase through cell centres; the funnel pulls it
// taut. A character following the staircase visibly zig-zags down an empty
// corridor, and no amount of steering smoothing hides it, because the waypoints
// really are in the wrong places.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "engine/nav/navmesh.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

using eng::Vec3;

struct Soup {
    std::vector<Vec3> vertices;
    std::vector<std::uint32_t> indices;

    // An axis-aligned box, all six faces.
    void Box(Vec3 centre, Vec3 half) {
        const std::uint32_t base = std::uint32_t(vertices.size());
        for (int i = 0; i < 8; ++i)
            vertices.push_back(Vec3{centre.x + ((i & 1) ? half.x : -half.x),
                                    centre.y + ((i & 2) ? half.y : -half.y),
                                    centre.z + ((i & 4) ? half.z : -half.z)});
        const int faces[6][4] = {{0, 2, 3, 1}, {4, 5, 7, 6}, {0, 1, 5, 4},
                                 {2, 6, 7, 3}, {0, 4, 6, 2}, {1, 3, 7, 5}};
        for (const auto& f : faces) {
            indices.insert(indices.end(), {base + f[0], base + f[1], base + f[2]});
            indices.insert(indices.end(), {base + f[0], base + f[2], base + f[3]});
        }
    }
    // A flat quad in XZ at height y.
    void Floor(float x0, float z0, float x1, float z1, float y) {
        const std::uint32_t b = std::uint32_t(vertices.size());
        vertices.push_back(Vec3{x0, y, z0});
        vertices.push_back(Vec3{x1, y, z0});
        vertices.push_back(Vec3{x1, y, z1});
        vertices.push_back(Vec3{x0, y, z1});
        indices.insert(indices.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
    }
};

float PathLength(const std::vector<Vec3>& p) {
    float total = 0.0f;
    for (std::size_t i = 1; i < p.size(); ++i) {
        const Vec3 d = p[i] - p[i - 1];
        total += std::sqrt(d.x * d.x + d.z * d.z);
    }
    return total;
}

// The largest turn between consecutive segments, in degrees. A taut path has a
// handful of real corners; a staircase has one at every waypoint.
float WorstTurn(const std::vector<Vec3>& p) {
    float worst = 0.0f;
    for (std::size_t i = 2; i < p.size(); ++i) {
        const Vec3 a{p[i - 1].x - p[i - 2].x, 0.0f, p[i - 1].z - p[i - 2].z};
        const Vec3 b{p[i].x - p[i - 1].x, 0.0f, p[i].z - p[i - 1].z};
        const float la = std::sqrt(a.x * a.x + a.z * a.z);
        const float lb = std::sqrt(b.x * b.x + b.z * b.z);
        if (la < 1e-4f || lb < 1e-4f) continue;
        const float c = std::clamp((a.x * b.x + a.z * b.z) / (la * lb), -1.0f, 1.0f);
        worst = std::max(worst, std::acos(c) * 57.2958f);
    }
    return worst;
}

eng::nav::BuildConfig Config() {
    eng::nav::BuildConfig c;
    c.cell_size = 0.2f;
    c.agent_radius = 0.3f;
    c.agent_height = 1.8f;
    c.agent_max_climb = 0.4f;
    return c;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    {
        std::printf("an empty floor becomes one big region\n");
        Soup soup;
        soup.Floor(-10.0f, -10.0f, 10.0f, 10.0f, 0.0f);
        const eng::nav::NavMesh mesh =
            eng::nav::NavMesh::Build(soup.vertices, soup.indices, Config());
        const eng::nav::BuildStats& s = mesh.Stats();
        std::printf("    %d cells, %d walkable, %d after erosion, %d polys, "
                    "%d portals, %.1f ms\n", s.cells_total, s.cells_walkable,
                    s.cells_after_erosion, s.polys, s.portals,
                    s.build_seconds * 1000.0);
        Check(mesh.Valid() && s.polys > 0, "it builds");
        // EROSION took cells away. Without it a path can be routed along the
        // very edge of the floor, where the agent's body is over the drop.
        Check(s.cells_after_erosion < s.cells_walkable,
              "and the agent's radius eroded the edges");

        // A straight path across open ground must be exactly straight: two
        // points, no corners. This is the simplest thing the funnel has to get
        // right and the one a midpoint-joining implementation fails.
        std::vector<Vec3> path;
        Check(mesh.FindPath(Vec3{-8.0f, 0.0f, -8.0f}, Vec3{8.0f, 0.0f, 8.0f}, &path),
              "and a path across it is found");
        const float direct = std::sqrt(16.0f * 16.0f * 2.0f);
        std::printf("    open-ground path: %zu points, %.2f m (direct is %.2f m)\n",
                    path.size(), PathLength(path), direct);
        Check(path.size() == 2, "an unobstructed path has no intermediate corners");
        Check(std::fabs(PathLength(path) - direct) < 0.4f,
              "and its length is the straight-line distance");
    }

    {
        std::printf("\na wall with a doorway forces the long way round\n");
        Soup soup;
        soup.Floor(-10.0f, -10.0f, 10.0f, 10.0f, 0.0f);
        // A wall across the middle with a three-metre doorway at one end.
        // The left piece runs x = -10 to 0, the right x = 3 to 10.
        soup.Box(Vec3{-5.0f, 1.0f, 0.0f}, Vec3{5.0f, 1.0f, 0.3f});
        soup.Box(Vec3{6.5f, 1.0f, 0.0f}, Vec3{3.5f, 1.0f, 0.3f});
        const eng::nav::NavMesh mesh =
            eng::nav::NavMesh::Build(soup.vertices, soup.indices, Config());
        std::printf("    %d polys, %d portals\n", mesh.Stats().polys,
                    mesh.Stats().portals);

        std::vector<Vec3> path;
        const Vec3 from{-6.0f, 0.0f, -5.0f};
        const Vec3 to{-6.0f, 0.0f, 5.0f};
        const bool found = mesh.FindPath(from, to, &path);
        std::printf("    %zu points, %.2f m (the direct line is 10.00 m)\n",
                    path.size(), PathLength(path));
        Check(found, "a route around the wall is found");
        // MUCH LONGER than the direct line, because it has to go through the
        // doorway at the far end and come back. A path close to 10 m went
        // through the wall.
        Check(PathLength(path) > 16.0f, "and it is far longer than the direct line");

        // AND IT DOES NOT CROSS THE WALL. Checked by sampling: every point
        // along the path within the wall's z band must be inside the doorway.
        bool crossed = false;
        for (std::size_t i = 1; i < path.size(); ++i) {
            for (int k = 0; k <= 200; ++k) {
                const float t = float(k) / 200.0f;
                const Vec3 p = path[i - 1] + (path[i] - path[i - 1]) * t;
                // Inside the wall's z band but not in the doorway.
                if (std::fabs(p.z) < 0.3f && (p.x < -0.1f || p.x > 3.1f))
                    crossed = true;
            }
        }
        Check(!crossed, "and no point of it passes through the wall");
    }

    {
        std::printf("\nthe funnel pulls the path taut\n");
        // A long diagonal across open ground. A corridor of rectangles running
        // diagonally is a staircase, and joining their portal midpoints gives a
        // path with a corner at every step.
        Soup soup;
        soup.Floor(-14.0f, -14.0f, 14.0f, 14.0f, 0.0f);
        const eng::nav::NavMesh mesh =
            eng::nav::NavMesh::Build(soup.vertices, soup.indices, Config());

        std::vector<int> corridor;
        Check(mesh.FindCorridor(Vec3{-12.0f, 0.0f, -12.0f}, Vec3{12.0f, 0.0f, 12.0f},
                                &corridor),
              "a diagonal corridor is found");
        std::vector<Vec3> pulled;
        eng::nav::NavMesh::StringPull(mesh, Vec3{-12.0f, 0.0f, -12.0f},
                                      Vec3{12.0f, 0.0f, 12.0f}, corridor, &pulled);
        const float direct = std::sqrt(24.0f * 24.0f * 2.0f);
        std::printf("    corridor of %zu polygons -> %zu path points, %.2f m "
                    "(direct %.2f m)\n", corridor.size(), pulled.size(),
                    PathLength(pulled), direct);
        // THE MEASUREMENT. Joining portal midpoints across a staircase corridor
        // gives a path several percent longer than the straight line and a
        // corner at every portal. Taut means neither.
        Check(PathLength(pulled) < direct * 1.02f,
              "the pulled path is within 2% of the straight line");
        // Corridor plus one: a path always has a start and an end, so a
        // corridor of one polygon is two points. Asking for no more points than
        // polygons fails on the simplest possible case, which is what the first
        // version of this did.
        Check(pulled.size() <= corridor.size() + 1,
              "and has no more corners than the corridor has portals");
        std::printf("    worst turn along it: %.1f degrees\n", WorstTurn(pulled));
        Check(WorstTurn(pulled) < 20.0f, "with no sharp zig-zags");
    }

    {
        std::printf("\nA* is guided rather than exhaustive\n");
        // A FIELD OF PILLARS, so the mesh has hundreds of polygons.
        //
        // The first version of this used an empty floor, which merges into ONE
        // rectangle -- so the search returned immediately having visited
        // nothing, and "not every polygon is expanded" was true of a search
        // that never ran. A heuristic check needs a graph.
        Soup soup;
        soup.Floor(-20.0f, -20.0f, 20.0f, 20.0f, 0.0f);
        for (int z = -4; z <= 4; ++z)
            for (int x = -4; x <= 4; ++x)
                soup.Box(Vec3{float(x) * 4.0f, 1.2f, float(z) * 4.0f},
                         Vec3{0.9f, 1.2f, 0.9f});
        eng::nav::BuildConfig c = Config();
        c.cell_size = 0.3f;
        const eng::nav::NavMesh mesh =
            eng::nav::NavMesh::Build(soup.vertices, soup.indices, c);
        std::vector<int> corridor;
        const bool found = mesh.FindCorridor(Vec3{-18.0f, 0.0f, -18.0f},
                                             Vec3{18.0f, 0.0f, 18.0f}, &corridor);
        const int visited = mesh.LastSearchVisited();
        std::printf("    %d polygons in the mesh, %d visited, corridor of %zu\n",
                    mesh.PolyCount(), visited, corridor.size());
        Check(found, "a route through the pillars is found");
        Check(mesh.PolyCount() > 100, "the pillar field makes a real graph");
        // An admissible heuristic that is actually wired up visits a fraction
        // of the graph. Dijkstra -- the same search with the heuristic set to
        // zero -- still finds the right path and visits nearly all of it, so
        // this is the check that says the heuristic is doing anything at all.
        Check(visited < mesh.PolyCount() * 3 / 4,
              "and the search expands well under three quarters of the graph");
    }

    {
        std::printf("\ntwo floors connected by a ramp\n");
        Soup soup;
        soup.Floor(-8.0f, -8.0f, 8.0f, -2.0f, 0.0f);   // lower
        soup.Floor(-8.0f, 2.0f, 8.0f, 8.0f, 2.0f);     // upper
        // The ramp between them: a quad rising from z = -2 to z = 2.
        {
            const std::uint32_t b = std::uint32_t(soup.vertices.size());
            soup.vertices.push_back(Vec3{-3.0f, 0.0f, -2.0f});
            soup.vertices.push_back(Vec3{3.0f, 0.0f, -2.0f});
            soup.vertices.push_back(Vec3{3.0f, 2.0f, 2.0f});
            soup.vertices.push_back(Vec3{-3.0f, 2.0f, 2.0f});
            soup.indices.insert(soup.indices.end(), {b, b + 1, b + 2, b, b + 2, b + 3});
        }
        const eng::nav::NavMesh mesh =
            eng::nav::NavMesh::Build(soup.vertices, soup.indices, Config());
        std::printf("    %d polys, %d portals\n", mesh.Stats().polys,
                    mesh.Stats().portals);

        std::vector<Vec3> path;
        const bool found =
            mesh.FindPath(Vec3{-6.0f, 0.0f, -6.0f}, Vec3{6.0f, 2.0f, 6.0f}, &path);
        Check(found, "a path from the lower floor to the upper is found");
        if (found) {
            // IT GOES VIA THE RAMP. The two floors do not touch, so a path that
            // did not pass through the ramp's x range went straight up a wall.
            // SAMPLED ALONG THE SEGMENTS, not at the waypoints. A taut path
            // across the ramp has its corners at the ramp's EDGES or no corners
            // at all, so checking only the waypoints asks whether the funnel
            // happened to put one in the middle -- which it should not.
            bool used_ramp = false;
            for (std::size_t i = 1; i < path.size(); ++i)
                for (int k = 0; k <= 100; ++k) {
                    const float t = float(k) / 100.0f;
                    const Vec3 p = path[i - 1] + (path[i] - path[i - 1]) * t;
                    if (std::fabs(p.z) < 2.2f && std::fabs(p.x) < 3.2f)
                        used_ramp = true;
                }
            Check(used_ramp, "and it goes via the ramp rather than up the wall");
            // The heights follow the ground rather than staying at the start's.
            std::printf("    start y %.2f, end y %.2f\n", path.front().y,
                        path.back().y);
            Check(path.back().y > 1.5f, "and it ends on the upper floor");
        }

        // A DISCONNECTED island: no route, and that must be reported rather
        // than answered with a path that teleports.
        Soup split;
        split.Floor(-8.0f, -8.0f, -2.0f, 8.0f, 0.0f);
        split.Floor(2.0f, -8.0f, 8.0f, 8.0f, 0.0f);
        const eng::nav::NavMesh two =
            eng::nav::NavMesh::Build(split.vertices, split.indices, Config());
        std::vector<Vec3> nowhere;
        Check(!two.FindPath(Vec3{-5.0f, 0.0f, 0.0f}, Vec3{5.0f, 0.0f, 0.0f}, &nowhere),
              "two unconnected islands report no route");
        Check(nowhere.empty(), "and produce no path");
    }

    {
        std::printf("\nqueries off the mesh answer honestly\n");
        Soup soup;
        soup.Floor(-5.0f, -5.0f, 5.0f, 5.0f, 0.0f);
        const eng::nav::NavMesh mesh =
            eng::nav::NavMesh::Build(soup.vertices, soup.indices, Config());
        Check(mesh.FindPoly(Vec3{0.0f, 0.0f, 0.0f}) >= 0, "a point on the mesh is found");
        // A point a hundred metres away must NOT snap to the nearest polygon:
        // an agent standing inside a wall would be teleported across the level.
        Check(mesh.FindPoly(Vec3{100.0f, 0.0f, 100.0f}, 2.0f) < 0,
              "and one far outside it is not");
        std::vector<Vec3> path;
        Check(!mesh.FindPath(Vec3{0.0f, 0.0f, 0.0f}, Vec3{100.0f, 0.0f, 100.0f}, &path),
              "and a path to nowhere is refused");

        // A point just off the edge SHOULD snap -- a click a few centimetres
        // past the floor is a click on the floor.
        const Vec3 snapped = mesh.ClosestPoint(Vec3{5.4f, 0.0f, 0.0f});
        std::printf("    (5.40, 0) snapped to (%.2f, %.2f)\n", snapped.x, snapped.z);
        Check(snapped.x < 5.0f, "and one just off the edge snaps back on");
    }

    {
        std::printf("\ndegenerate inputs\n");
        const eng::nav::NavMesh empty = eng::nav::NavMesh::Build({}, {}, Config());
        Check(!empty.Valid() && empty.PolyCount() == 0, "no geometry builds nothing");
        std::vector<Vec3> path;
        Check(!empty.FindPath(Vec3{0, 0, 0}, Vec3{1, 0, 0}, &path),
              "and answers no path rather than crashing");

        // A vertical wall alone: nothing is walkable, so there is no mesh.
        Soup wall;
        wall.Box(Vec3{0.0f, 1.0f, 0.0f}, Vec3{2.0f, 1.0f, 0.05f});
        const eng::nav::NavMesh only_wall =
            eng::nav::NavMesh::Build(wall.vertices, wall.indices, Config());
        std::printf("    a bare wall produced %d polys\n", only_wall.PolyCount());
        // The box's TOP is walkable, so this is not zero -- but it must be
        // small, and nothing on the vertical faces may survive the slope test.
        Check(only_wall.PolyCount() < 20, "a bare wall's sides are not walkable");

        // A floor too small for the agent after erosion.
        Soup tiny;
        tiny.Floor(-0.2f, -0.2f, 0.2f, 0.2f, 0.0f);
        const eng::nav::NavMesh nothing =
            eng::nav::NavMesh::Build(tiny.vertices, tiny.indices, Config());
        Check(nothing.PolyCount() == 0,
              "and a floor narrower than the agent leaves nothing to walk on");
    }

    std::printf(g_failures == 0 ? "\nnavmesh_test: all checks passed\n"
                                : "\nnavmesh_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
