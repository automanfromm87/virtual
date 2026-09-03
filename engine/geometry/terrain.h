// Terrain: a height field, the meshes to draw it with, and the queries to walk
// on it.
//
// WHY IT IS NOT JUST A BIG MESH. A square kilometre at one-metre resolution is
// a million vertices and two million triangles. As one mesh it cannot be culled
// (it is always partly visible), cannot have levels of detail (they are a
// property of a whole mesh), and cannot be collided against without a spatial
// index over its own triangles. Split into chunks it gets all three for free
// from machinery that already exists: each chunk is an ordinary instance with
// ordinary bounds and an ordinary LOD chain.
//
// THE HEIGHTS STAY AS A GRID even though the drawing uses meshes, because the
// grid is what answers the questions gameplay actually asks -- how high is the
// ground here, which way does it slope -- in constant time and without touching
// a triangle.
//
// SKIRTS. A chunk drawn at a coarser level than its neighbour does not meet it:
// the coarse edge interpolates across a vertex the fine edge has, so a crack of
// background shows between them. The usual fixes are stitching the edges (which
// makes a chunk's mesh depend on its neighbours' levels, so a level change
// rebuilds several chunks) or skirts -- a wall dropped from the chunk's border
// straight down, wide enough to fill the gap. Skirts waste a few triangles and
// are visible if the camera gets under the terrain, and they are what this uses,
// because they make every chunk independent of every other.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "engine/core/math.h"
#include "engine/geometry/mesh.h"

namespace eng {

struct TerrainConfig {
    // Samples across the whole terrain, in each direction. A power of two plus
    // one is the friendly size: it divides evenly into chunks whose own edges
    // land on samples.
    int resolution = 257;
    // World size of the whole terrain, in metres.
    float world_size = 256.0f;
    // Where the corner at sample (0,0) sits.
    Vec3 origin{0.0f, 0.0f, 0.0f};
    // Samples per chunk edge. 33 gives a 32x32 quad grid, which is 2048
    // triangles -- enough that a draw call is worth making and few enough that
    // one chunk out of view is a real saving.
    int chunk_resolution = 33;
    // How far the skirt hangs below the chunk's lowest vertex. Has to exceed
    // the largest height difference a coarse level can miss, which is bounded
    // by the terrain's own slope over one coarse quad.
    float skirt_depth = 2.0f;
};

class Terrain {
  public:
    Terrain();
    ~Terrain();
    Terrain(Terrain&&) noexcept;
    Terrain& operator=(Terrain&&) noexcept;
    Terrain(const Terrain&) = delete;
    Terrain& operator=(const Terrain&) = delete;

    // From a function of world x and z. The generator is called once per
    // sample, so it must be cheap or the build is slow -- and it is called on
    // several threads, so it must not touch shared state.
    [[nodiscard]] static Terrain Generate(const TerrainConfig&,
                                          const std::function<float(float, float)>&);
    // From an existing grid, row-major, `resolution * resolution` values.
    [[nodiscard]] static Terrain FromHeights(const TerrainConfig&,
                                             std::span<const float> heights);
    // From an 8-bit greyscale image, scaled into `min_height`..`max_height`.
    // What a heightmap usually arrives as.
    [[nodiscard]] static Terrain FromImage(const TerrainConfig&,
                                           std::span<const std::uint8_t> grey,
                                           int width, int height, float min_height,
                                           float max_height);

    [[nodiscard]] bool Valid() const;
    [[nodiscard]] const TerrainConfig& Config() const;
    [[nodiscard]] int ChunksX() const;
    [[nodiscard]] int ChunksZ() const;

    // --- queries ---------------------------------------------------------------
    //
    // BILINEAR, from the four samples around the point. Not the triangle the
    // renderer drew: the mesh splits each quad along a diagonal, so a triangle
    // lookup gives a slightly different height depending on which way the split
    // went, and a character walking across a quad steps by a few millimetres at
    // the diagonal. Bilinear is smooth and is what every query here uses, which
    // means they all agree with each other even if none of them exactly matches
    // the drawn surface.
    [[nodiscard]] float HeightAt(float x, float z) const;
    [[nodiscard]] Vec3 NormalAt(float x, float z) const;
    // The steepest slope at a point, in degrees. For deciding what can be built
    // or walked on without reconstructing the normal at every call site.
    [[nodiscard]] float SlopeAt(float x, float z) const;

    // Where a ray meets the terrain. Marches the grid rather than testing every
    // cell, so the cost is the number of cells crossed and not the size of the
    // terrain.
    [[nodiscard]] bool Raycast(Vec3 origin, Vec3 direction, float max_distance,
                               float* out_t, Vec3* out_normal) const;

    // --- meshes ----------------------------------------------------------------
    //
    // `lod` of 0 is full resolution; each level after it takes every other
    // sample, so level n is 4^-n of the triangles.
    [[nodiscard]] Mesh BuildChunk(int chunk_x, int chunk_z, int lod = 0) const;
    // The chunk's world bounds, for culling, without building its mesh.
    void ChunkBounds(int chunk_x, int chunk_z, Vec3* out_min, Vec3* out_max) const;
    [[nodiscard]] int MaxLod() const;

    // The raw grid, for a caller that wants to sculpt it and rebuild.
    [[nodiscard]] std::span<const float> Heights() const;
    void SetHeight(int ix, int iz, float y);
    // Recomputes the cached per-chunk bounds. Call after SetHeight.
    void RefreshBounds();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eng
