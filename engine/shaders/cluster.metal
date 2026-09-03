// No #include here: shader_types.h is textually prepended by the loader.
//
// CLUSTERED LIGHT BINNING. One thread per cell of the view-frustum grid; it
// works out which lights touch that cell and writes their indices.
//
// The alternative this replaces is every fragment testing every light. That is
// fine at eight lights and quadratic misery at eight hundred: the fragment
// count is fixed by the screen, so the cost is (pixels x lights) and the
// overdraw multiplies it again. Binning pays (cells x lights) ONCE -- 3456
// against 921600 pixels -- and the fragment then reads a list that is almost
// always shorter than five.
//
// The geometry is the whole trick. A cell is the intersection of a screen-space
// tile with a depth slice, which in VIEW space is a frustum wedge. Its
// axis-aligned bounding box is what a sphere is tested against: conservative,
// so a light is never wrongly dropped, and cheap enough to build from four
// multiplies.

struct ClusterBox {
    float3 lo;
    float3 hi;
};

// The view-space AABB of cell (ix, iy, iz).
//
// View space here is right-handed with -z going into the screen, which is what
// the projection this engine builds produces. Distances along the view axis are
// therefore POSITIVE numbers that get negated when they become coordinates.
static ClusterBox CellBounds(uint ix, uint iy, uint iz, constant GpuClusters& c)
{
    const float nearZ = c.depth.x;
    const float farZ = c.depth.y;
    const float log_ratio = c.depth.z;  // log(far / near), precomputed
    const uint nz = uint(c.grid.z);

    // EXPONENTIAL slices. z(k) = near * (far/near)^(k/K). Linear slices would
    // put the first cell boundary metres from the eye and the last twenty at
    // distances nothing is ever lit at.
    const float z0 = nearZ * exp(log_ratio * float(iz) / float(nz));
    const float z1 = nearZ * exp(log_ratio * float(iz + 1) / float(nz));

    // The tile's NDC extent. y is flipped because the tile grid is indexed from
    // the top of the screen and clip space runs up it.
    const float x0 = 2.0f * float(ix) / c.grid.x - 1.0f;
    const float x1 = 2.0f * float(ix + 1) / c.grid.x - 1.0f;
    const float y1 = 1.0f - 2.0f * float(iy) / c.grid.y;
    const float y0 = 1.0f - 2.0f * float(iy + 1) / c.grid.y;

    // NDC to view, at a given distance. Both the projection's scale factors are
    // already in the uniforms for SSAO's benefit: f = 1/tan(fovY/2).
    const float f = c.slope.x;
    const float aspect = c.slope.y;
    const float sx = aspect / f;
    const float sy = 1.0f / f;

    // The extremes over the WHOLE slice, not at one depth. A wedge is widest at
    // its far face, so the box has to take x from z1 and not from z0 -- taking
    // it from the near face would produce a box that misses lights sitting just
    // inside the cell's far corners, and those show up as unlit patches that
    // appear and vanish as the camera moves.
    ClusterBox b;
    b.lo.x = min(x0 * sx * z0, x0 * sx * z1);
    b.hi.x = max(x1 * sx * z0, x1 * sx * z1);
    b.lo.y = min(y0 * sy * z0, y0 * sy * z1);
    b.hi.y = max(y1 * sy * z0, y1 * sy * z1);
    b.lo.z = -z1;
    b.hi.z = -z0;
    return b;
}

static float DistanceSquaredToBox(float3 p, ClusterBox b)
{
    const float3 d = max(max(b.lo - p, float3(0.0f)), p - b.hi);
    return dot(d, d);
}

kernel void cs_cluster_lights(uint3 gid [[thread_position_in_grid]],
                              constant GpuClusters& c [[buffer(0)]],
                              constant ENG_MAT4& view [[buffer(1)]],
                              device const GpuLight* lights [[buffer(2)]],
                              device uint* counts [[buffer(3)]],
                              device uint* indices [[buffer(4)]],
                              // Indices into the light buffer of the lights
                              // that survived a frustum test on the CPU. The
                              // loop below walks these, not every light: the
                              // binning cost is then proportional to what is
                              // ON SCREEN rather than to what exists, which is
                              // the difference between a level with a thousand
                              // lamps costing a thousand and costing the twenty
                              // in the room you are standing in.
                              device const uint* candidates [[buffer(5)]])
{
    const uint nx = uint(c.grid.x), ny = uint(c.grid.y), nz = uint(c.grid.z);
    if (gid.x >= nx || gid.y >= ny || gid.z >= nz) return;
    const uint cell = (gid.z * ny + gid.y) * nx + gid.x;

    const ClusterBox box = CellBounds(gid.x, gid.y, gid.z, c);
    const uint light_count = uint(c.screen.w);
    const uint capacity = uint(c.grid.w);

    uint n = 0;
    for (uint k = 0; k < light_count; ++k) {
        const uint i = candidates[k];
        // The light's position in VIEW space. Binning in view space rather than
        // world space is what makes the cell bounds trivial -- in world space
        // every cell would be an arbitrarily oriented box.
        const float4 vp = view * float4(lights[i].position.xyz, 1.0f);
        const float radius = lights[i].direction.w;  // range, where it hits zero
        if (DistanceSquaredToBox(vp.xyz, box) > radius * radius) continue;

        // FULL cells drop the rest rather than writing past the end. Losing a
        // light in a crowded cell dims that cell slightly; writing past the end
        // corrupts the next one, and the symptom is a lit patch in a completely
        // different part of the screen.
        if (n >= capacity) break;
        indices[cell * capacity + n] = i;
        ++n;
    }
    counts[cell] = n;
}
