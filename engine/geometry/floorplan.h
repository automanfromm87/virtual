// Pure C++20. Turns a 2D floor plan into 3D geometry.
//
// Deliberately NOT a mesh importer. A floor plan already IS structured data —
// polygons and line segments — so extruding it directly skips the entire asset
// pipeline (glTF parsing, buffer views, image decode) and keeps the source of
// truth editable. Importing a model would throw that structure away and hand
// back a bag of triangles.
#pragma once

#include <vector>

#include "engine/core/math.h"
#include "engine/geometry/mesh.h"

namespace eng {

// A hole in a wall, measured ALONG the wall from its start point, in metres.
// A door is an opening whose bottom is 0; a window sits above a sill.
struct Opening {
    float start = 0.0f;
    float end = 0.0f;
    float bottom = 0.0f;
    float top = 2.1f;
};

struct WallSpec {
    Vec2 from{0.0f, 0.0f};
    Vec2 to{0.0f, 0.0f};
    std::vector<Opening> openings;
};

// Plan-space is XZ with Y up: `from`/`to` are (x, z) and the wall rises in +Y.
struct FloorPlan {
    std::vector<WallSpec> walls;
    // Counter-clockwise in plan space. Concave is fine; self-intersecting is not.
    std::vector<Vec2> floor_outline;
    float wall_height = 2.7f;
    float wall_thickness = 0.14f;
    float floor_thickness = 0.12f;
};

// Solid wall geometry with the openings cut out. Each opening splits its wall
// into a jamb on either side plus a sill below and a lintel above — building
// the gaps as separate boxes rather than subtracting them, because a CSG
// boolean is a great deal of machinery for a rectangle in a rectangle.
[[nodiscard]] Mesh MakeWalls(const FloorPlan&, Vec4 color);

// The floor slab: the outline triangulated by ear clipping, extruded down.
[[nodiscard]] Mesh MakeFloorSlab(const FloorPlan&, Vec4 color);

// One quad filling each window opening, for glass. Separate mesh because it
// needs a transparent material and therefore a different draw batch.
[[nodiscard]] Mesh MakeGlass(const FloorPlan&, Vec4 color);

// Gable roof over the plan's bounding rectangle: two slopes meeting at a ridge,
// plus a triangular gable wall closing each end.
//
//   rise     how far the ridge sits above the wall top, in metres
//   overhang how far the eaves project past the walls
//
// Rectangular plans only — the ridge runs along the longer side of the bounding
// box. A hip or a roof following an L-shaped outline is a straight-skeleton
// problem, which is a great deal more machinery than a demo needs.
[[nodiscard]] Mesh MakeGableRoof(const FloorPlan&, float rise, float overhang,
                                 Vec4 color);

// A door panel in every floor-level opening, hinged open by `ajar` radians.
// A doorway with nothing in it reads as a gap in a wall, not as a door.
[[nodiscard]] Mesh MakeDoorLeaves(const FloorPlan&, float ajar, Vec4 color);

// A frame around every window opening. Without one the glass floats in a hole
// and the building looks unfinished from outside.
[[nodiscard]] Mesh MakeWindowFrames(const FloorPlan&, float depth, Vec4 color);

// Ear clipping. Exposed for testing: a triangulator that silently drops a
// triangle produces a floor with an invisible hole in it.
[[nodiscard]] std::vector<std::uint16_t> TriangulatePolygon(
    const std::vector<Vec2>& outline);

}  // namespace eng
