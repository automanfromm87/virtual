// Navigation: turning level geometry into somewhere a character can be told to
// walk, and finding the way.
//
// WHAT WAS MISSING WITHOUT IT. The isometric demo already lets you click a
// point on the ground and sends the character there -- in a straight line. It
// walks into the first wall between here and there and stops. Physics answers
// "what is in the way"; nothing answered "how do I get around it", and no
// amount of collision detection does, because the question is global and
// collision is local.
//
// THE PIPELINE, and why each stage exists:
//
//   1. VOXELIZE the geometry into a heightfield of solid spans. Working from
//      the triangles directly is possible and hopeless: real level geometry is
//      overlapping, non-manifold, and full of surfaces a character could never
//      stand on. Rasterising it throws all of that away and leaves a regular
//      grid, which every later stage can assume.
//   2. FILTER by slope, by headroom, and by ledge. What is left is exactly the
//      set of places the agent could stand.
//   3. ERODE by the agent's radius, so a path down the middle of a polygon is
//      a path the agent's body fits along. Doing this at the end -- shrinking
//      the final path instead -- does not work: the path has already been
//      routed through a gap too narrow to enter.
//   4. MERGE the surviving cells into CONVEX polygons, with PORTALS between
//      neighbours. Convexity is what makes the funnel in step 6 correct.
//   5. A* over the polygons.
//   6. STRING-PULL the corridor into a straight path with the funnel
//      algorithm. Without it a path is a staircase through cell centres, and a
//      character following one visibly zig-zags along an empty corridor.
//
// THE POLYGONS ARE AXIS-ALIGNED RECTANGLES, which is where this differs from
// Recast's watershed-and-contour build. Rectangles are convex by construction,
// so there is no region growing, no contour tracing and no convex partition --
// three stages, each with its own degenerate cases. The cost is more polygons
// along a diagonal wall than a contour build would produce. For a graph that is
// searched a few times a second that is a good trade; for one with a hundred
// thousand polygons it would not be.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "engine/core/math.h"

namespace eng::nav {

struct BuildConfig {
    // The grid the world is rasterised onto. Smaller finds narrower gaps and
    // costs the square of itself in memory and time; a third of the agent's
    // radius is the usual choice, because a gap narrower than that is one the
    // agent cannot use anyway.
    float cell_size = 0.15f;
    // Vertical quantisation. Wants to be well under max_climb or a step becomes
    // either impassable or invisible.
    float cell_height = 0.08f;

    // WHO IS WALKING. Every filter below is expressed in terms of these, so
    // rebuilding for a larger character is a parameter change rather than a
    // different pipeline.
    float agent_radius = 0.35f;
    float agent_height = 1.8f;
    // The tallest step that can be walked up without jumping.
    float agent_max_climb = 0.35f;
    float agent_max_slope_degrees = 50.0f;

    // Where to build. An empty box means "fit the geometry", which is what a
    // caller usually wants and is why it is the default.
    Vec3 bounds_min{0.0f, 0.0f, 0.0f};
    Vec3 bounds_max{0.0f, 0.0f, 0.0f};

    // Drop any polygon smaller than this many cells. Rasterising a world
    // produces slivers along every wall -- one cell wide, going nowhere -- and
    // they cost search time and produce paths that step sideways for no reason.
    int min_region_cells = 8;
};

// One convex piece of walkable floor. Axis-aligned in XZ; its height is sampled
// from the grid underneath, so a polygon spanning a ramp reports the ramp.
struct Poly {
    // Inclusive cell range, and the world-space rectangle it covers.
    Vec3 min{0.0f, 0.0f, 0.0f};
    Vec3 max{0.0f, 0.0f, 0.0f};
    float height_min = 0.0f;
    float height_max = 0.0f;
    // Indices into NavMesh::Portals() -- the edges shared with neighbours.
    int first_portal = 0;
    int portal_count = 0;
};

// The shared edge between two polygons, as a segment. ORDERED so that walking
// from `from` to `to`, `left` is on the left. The funnel algorithm depends on
// that ordering and produces a path that crosses itself without it.
struct Portal {
    int from = -1, to = -1;
    Vec3 left{0.0f, 0.0f, 0.0f};
    Vec3 right{0.0f, 0.0f, 0.0f};
};

struct BuildStats {
    int cells_total = 0;
    int cells_walkable = 0;
    int cells_after_erosion = 0;
    int polys = 0;
    int portals = 0;
    double build_seconds = 0.0;
};

class NavMesh {
  public:
    NavMesh();
    ~NavMesh();
    NavMesh(NavMesh&&) noexcept;
    NavMesh& operator=(NavMesh&&) noexcept;
    NavMesh(const NavMesh&) = delete;
    NavMesh& operator=(const NavMesh&) = delete;

    // Builds from a triangle soup. Indices are triples; the winding does not
    // matter, because the slope test uses the absolute normal -- a level whose
    // floors are wound inconsistently is normal, and refusing to walk on half
    // of them would be a mysterious failure.
    [[nodiscard]] static NavMesh Build(std::span<const Vec3> vertices,
                                       std::span<const std::uint32_t> indices,
                                       const BuildConfig&);

    [[nodiscard]] bool Valid() const;
    [[nodiscard]] int PolyCount() const;
    [[nodiscard]] const Poly& GetPoly(int) const;
    [[nodiscard]] std::span<const Portal> Portals() const;
    [[nodiscard]] const BuildStats& Stats() const;

    // The polygon containing `point`, or the nearest one within
    // `search_radius`. -1 when there is none, which is a real answer: a point
    // inside a wall or off the edge of the world has no polygon, and returning
    // the nearest one however far away would silently teleport an agent.
    [[nodiscard]] int FindPoly(Vec3 point, float search_radius = 2.0f) const;

    // The nearest point on the mesh. For snapping a clicked destination onto
    // walkable ground.
    [[nodiscard]] Vec3 ClosestPoint(Vec3, float search_radius = 4.0f) const;

    // The corridor of polygons from one point to another, and then the
    // string-pulled path through it.
    //
    // Returns false when there is no route -- which is a normal answer, not an
    // error, and the reason this is not a function that returns a path: a
    // caller has to be able to tell "no route" from "a route of length zero".
    [[nodiscard]] bool FindPath(Vec3 start, Vec3 end,
                                std::vector<Vec3>* out_path) const;
    // The polygon corridor alone, before string pulling. Exposed because it is
    // what a test can check independently -- a wrong path is either a wrong
    // corridor or a wrong funnel, and one number tells them apart.
    [[nodiscard]] bool FindCorridor(Vec3 start, Vec3 end,
                                    std::vector<int>* out_polys) const;

    // Turns a corridor into a straight path. Exposed for the same reason.
    static void StringPull(const NavMesh&, Vec3 start, Vec3 end,
                           std::span<const int> corridor, std::vector<Vec3>* out);

    // How many polygons the last search visited. The measurement that says
    // whether A* is actually being guided: an admissible heuristic visits far
    // fewer than a breadth-first search of the same graph, and a heuristic
    // wired up wrongly still finds the path.
    [[nodiscard]] int LastSearchVisited() const;

    // The ground height at a point on a polygon, from the grid the mesh was
    // built on rather than from the polygon's flat rectangle. A ramp is one
    // polygon and its height varies across it.
    [[nodiscard]] float HeightAt(int poly, Vec3 point) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eng::nav
