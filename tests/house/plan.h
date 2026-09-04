// The demo apartment, in metres. Plan space is (x, z) with +Y up, centred on
// the origin so the shadow map's box lines up with the building without any
// extra bookkeeping.
//
// Shared between the windowed viewer and the offscreen test so both look at
// exactly the same building — a test that checks a different scene from the one
// you are looking at is worth very little.
#pragma once

#include "engine/geometry/floorplan.h"
#include "engine/geometry/mesh.h"
#include "engine/scene/scene.h"

namespace house {

// 10 x 7 m: living room and kitchen to the west, bedroom and bathroom east.
inline eng::FloorPlan Plan() {
    using eng::Opening;
    eng::FloorPlan p;
    p.wall_height = 2.7f;
    p.wall_thickness = 0.16f;
    p.floor_thickness = 0.15f;
    p.floor_outline = {{-5, -3.5f}, {5, -3.5f}, {5, 3.5f}, {-5, 3.5f}};

    // Openings are measured ALONG each wall from its `from` point, so the
    // direction a wall is declared in matters.
    p.walls = {
        // North exterior, west to east. Two windows.
        {{-5, -3.5f}, {5, -3.5f}, {{1.5f, 3.5f, 0.9f, 2.1f}, {6.5f, 8.5f, 0.9f, 2.1f}}},
        // East exterior.
        {{5, -3.5f}, {5, 3.5f}, {{1.5f, 3.0f, 0.9f, 2.1f}}},
        // South exterior, east to west. Front door (sill at the floor) + window.
        {{5, 3.5f}, {-5, 3.5f}, {{4.0f, 5.0f, 0.0f, 2.1f}, {7.0f, 8.5f, 0.9f, 2.1f}}},
        // West exterior.
        {{-5, 3.5f}, {-5, -3.5f}, {{2.0f, 4.0f, 0.9f, 2.1f}}},

        // Interior: living | bedroom, with a door.
        {{1.2f, -3.5f}, {1.2f, 0.5f}, {{2.5f, 3.4f, 0.0f, 2.05f}}},
        // Interior: bedroom | bathroom.
        {{1.2f, 0.5f}, {5, 0.5f}, {{1.0f, 1.9f, 0.0f, 2.05f}}},
        // Interior: living | kitchen, a wide cased opening.
        {{-5, 0.9f}, {1.2f, 0.9f}, {{2.0f, 3.8f, 0.0f, 2.2f}}},
    };
    return p;
}

// Furniture as plain boxes: half-extents, centre, colour. Enough to read the
// rooms as rooms, which is what a floor plan is for — an actual model library
// is an asset-pipeline problem, not a renderer one.
struct Block {
    eng::Vec3 half;
    eng::Vec3 at;
    eng::Vec4 color;
};

inline std::vector<Block> Furniture() {
    return {
        // Living room: sofa, coffee table, rug-height plinth.
        {{1.0f, 0.35f, 0.42f}, {-3.2f, 0.35f, -1.6f}, {0.35f, 0.38f, 0.48f, 1}},
        {{0.55f, 0.2f, 0.35f}, {-3.2f, 0.2f, -0.2f}, {0.45f, 0.32f, 0.22f, 1}},
        // Kitchen: counter run along the west wall.
        {{0.32f, 0.45f, 1.1f}, {-4.5f, 0.45f, 2.2f}, {0.72f, 0.70f, 0.66f, 1}},
        {{1.4f, 0.45f, 0.32f}, {-2.6f, 0.45f, 3.1f}, {0.72f, 0.70f, 0.66f, 1}},
        // Bedroom: bed and a nightstand.
        {{0.95f, 0.28f, 1.05f}, {3.3f, 0.28f, -1.9f}, {0.55f, 0.42f, 0.40f, 1}},
        {{0.25f, 0.28f, 0.25f}, {1.9f, 0.28f, -0.4f}, {0.42f, 0.34f, 0.30f, 1}},
        // Bathroom: tub.
        {{0.85f, 0.3f, 0.4f}, {3.6f, 0.3f, 1.6f}, {0.86f, 0.88f, 0.90f, 1}},
    };
}

}  // namespace house
