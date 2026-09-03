// Mesh simplification, for building levels of detail.
//
// THE PROBLEM LOD SOLVES. A thousand-triangle rock costs a thousand triangles
// whether it fills the screen or covers four pixels. Past a certain distance
// every triangle is smaller than a pixel, and a GPU rasterises a sub-pixel
// triangle at the same cost as a large one -- worse, actually, because quad
// utilisation collapses: the hardware shades in 2x2 blocks, so a triangle
// touching one pixel still runs four fragment shaders. A distant forest can
// spend more time on the trees you cannot make out than on the one in front.
//
// THE ALGORITHM: vertex clustering, with the representative chosen by
// minimising QUADRIC ERROR.
//
// Vertex clustering alone -- overlay a grid, collapse everything in a cell to
// its centroid -- is the simple version, and it is O(n), robust to any input
// including non-manifold soup, and produces recognisably wrong shapes: a
// centroid sits in the middle of a cell, so a flat wall becomes bumpy and a
// sharp edge becomes rounded.
//
// The quadric fixes that. Each vertex accumulates the sum of squared distances
// to the planes of its faces, as a 4x4 matrix; the point minimising that sum
// over a whole cluster is where the cluster's planes intersect -- so a cluster
// straddling a flat wall places its vertex ON the wall, and one on a corner
// places it AT the corner. It is the same error metric as Garland and Heckbert's
// edge collapse, applied to a grid instead of to a priority queue.
//
// WHY NOT the full edge-collapse version, which produces better meshes at the
// same triangle count: it is O(n log n) with a large constant, it needs
// manifold connectivity to avoid tearing, and it has to be told about
// boundaries, seams and attribute discontinuities or it welds them shut. This
// is for making a distant rock cheaper, and the distant rock does not need the
// difference.
#pragma once

#include <cstdint>
#include <vector>

#include "engine/geometry/mesh.h"

namespace eng {

struct SimplifyOptions {
    // The fraction of the original bounding-box diagonal one grid cell spans.
    // Bigger cells merge more and lose more. Around 1/32 halves the triangle
    // count on typical geometry; 1/8 takes it to a tenth.
    float cell_fraction = 1.0f / 32.0f;

    // Vertices whose NORMALS differ by more than this are never merged, even
    // inside one cell.
    //
    // Without it a cube collapses: its eight corners each carry three vertices
    // with three different normals, they all land in the same cell, and merging
    // them gives one vertex with an averaged normal -- so the cube's faces
    // stop being flat and its edges stop being sharp. Any model with a hard
    // edge has this problem, which is most of them.
    float normal_weld_degrees = 45.0f;

    // The same for colour, as a per-channel distance. A checkerboard whose
    // squares are separate vertices would otherwise average to grey.
    float color_weld = 0.25f;
};

// Returns a mesh with fewer vertices and triangles, or a copy of the input when
// nothing could be merged. Degenerate triangles -- the ones whose corners all
// landed in the same cluster -- are dropped rather than emitted as zero-area,
// which is what makes the triangle count actually fall.
[[nodiscard]] Mesh Simplify(const Mesh&, const SimplifyOptions& = {});

// A chain of increasingly coarse versions, starting with a copy of the input.
//
// `levels` counts the original, so 4 gives the original plus three reductions.
// Each level uses cells twice the size of the one before, which is roughly a
// quarter of the triangles per step -- matching the way the screen area of an
// object falls with the square of its distance.
[[nodiscard]] std::vector<Mesh> BuildLodChain(const Mesh&, int levels = 4,
                                              const SimplifyOptions& = {});

}  // namespace eng
