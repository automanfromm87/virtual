// GPU FRUSTUM CULLING, straight into an indirect draw.
//
// Culling on the CPU means the CPU touches every object every frame -- reading
// a transform, testing six planes, appending to a list -- and then submits one
// draw per survivor. Both halves scale with the object count, and neither has
// anything to do with how many triangles end up on screen.
//
// This does the test on the GPU, one thread per instance, and writes the
// survivors into a compacted buffer. The draw that follows is INDIRECT: its
// instance count comes from the same buffer, so the CPU never learns how many
// survived and never waits to find out. A readback would cost a full pipeline
// stall, which is more than the culling saves.
//
// What it gives up: the CPU no longer knows what is visible, so anything that
// wanted that -- a "what did I draw" list, sorting by depth, LOD selection on
// the CPU -- has to move to the GPU too or be given up. Sorting is the one that
// matters, and it is why this path is for OPAQUE geometry only: blending needs
// back-to-front order, and there is none here.

struct CullParams {
    float4x4 viewProj;
    // Six frustum planes, xyz = normal, w = distance. Pointing INWARD, so a
    // point is inside when dot(plane.xyz, p) + plane.w >= 0 for all six.
    float4 planes[6];
    // .xyz camera position, .w near plane.
    float4 eye;
    // .xy screen pixels, .z pixels per world unit at one metre, .w Hi-Z mips.
    float4 screen;
    // Screen radius in PIXELS below which each level takes over. .x is where
    // level 1 starts, .y level 2, .z level 3; .w unused.
    float4 lod_px;
    // .x instance count, .y level count, .z per-level capacity in `visible`,
    // .w occlusion on.
    uint4 counts;
    // Index count of each level, so one indirect argument per level can be
    // filled without the CPU knowing which levels were chosen.
    uint4 index_counts;
};

// The LEVEL an instance of this projected size should draw at.
//
// By SCREEN RADIUS, not by distance. Distance thresholds have to be re-tuned
// for every object size and again for every field of view -- a boulder and a
// pebble at fifty metres need completely different numbers, and zooming in
// changes both. Screen size is the quantity that actually decides whether the
// detail is visible, and one set of thresholds works for the whole scene.
static uint SelectLod(float radius_px, constant CullParams& p) {
    const uint last = max(p.counts.y, 1u) - 1u;
    if (radius_px >= p.lod_px.x) return 0;
    if (radius_px >= p.lod_px.y) return min(1u, last);
    if (radius_px >= p.lod_px.z) return min(2u, last);
    return min(3u, last);
}

// Whether the sphere is entirely behind something already in the depth pyramid.
//
// THE PYRAMID HOLDS THE FARTHEST depth in each region -- the MINIMUM value,
// since this is reversed-Z. That is the conservative direction: a region's
// stored value is the depth of the nearest thing you could still be behind, so
// an object whose closest point is further away than that is hidden by
// everything in the region at once.
//
// Getting the direction backwards gives a test that culls whatever it feels
// like, and the symptom is objects flickering out at the edges of occluders.
static bool Occluded(float3 centre, float radius, texture2d<float> hiz,
                     constant CullParams& p) {
    const float distance = length(centre - p.eye.xyz);
    // Straddling the camera: no meaningful footprint, and the projection below
    // would divide through a w near zero.
    if (distance <= radius + p.eye.w) return false;

    const float4 clip = p.viewProj * float4(centre, 1.0);
    if (clip.w <= 0.0) return false;
    const float3 ndc = clip.xyz / clip.w;
    // The sphere's radius in NDC, from the projection's own scale rather than
    // from projecting a second point: projecting centre + right * radius is the
    // usual shortcut and it under-estimates for a sphere off to the side.
    const float r_ndc = radius * p.screen.z / max(distance, 1e-4) /
                        max(p.screen.y * 0.5, 1.0);
    const float2 uv_min = float2((ndc.x - r_ndc) * 0.5 + 0.5, 0.5 - (ndc.y + r_ndc) * 0.5);
    const float2 uv_max = float2((ndc.x + r_ndc) * 0.5 + 0.5, 0.5 - (ndc.y - r_ndc) * 0.5);
    if (uv_max.x < 0.0 || uv_min.x > 1.0 || uv_max.y < 0.0 || uv_min.y > 1.0)
        return false;  // off screen; the frustum test owns that case

    // THE MIP WHOSE TEXELS ARE AS BIG AS THE FOOTPRINT, so four samples cover
    // it. Choosing a finer level would need many more samples to be
    // conservative, and choosing a coarser one is still correct -- just less
    // effective, because the region includes more than the object.
    const float2 size_px = (uv_max - uv_min) * p.screen.xy;
    const float level = clamp(ceil(log2(max(max(size_px.x, size_px.y), 1.0))),
                              0.0, max(p.screen.w - 1.0, 0.0));
    const uint mip = uint(level);
    const uint2 dim = uint2(max(uint(p.screen.x) >> mip, 1u),
                            max(uint(p.screen.y) >> mip, 1u));
    const uint2 lo = uint2(clamp(uv_min * float2(dim), float2(0.0), float2(dim - 1)));
    const uint2 hi = uint2(clamp(uv_max * float2(dim), float2(0.0), float2(dim - 1)));

    // The FARTHEST occluder over the footprint. Taking the minimum of the
    // pyramid's already-minimised values keeps the test conservative: if any
    // part of the footprint has nothing behind it, this goes to zero and
    // nothing is culled.
    float farthest = 1.0;
    for (uint y = lo.y; y <= hi.y; ++y)
        for (uint x = lo.x; x <= hi.x; ++x)
            farthest = min(farthest, hiz.read(uint2(x, y), mip).r);

    // The sphere's NEAREST point, as a reversed-Z depth. Using the centre would
    // cull an object whose front half is poking out from behind a wall.
    const float near_depth = p.eye.w / max(distance - radius, 1e-4);
    return near_depth < farthest;
}

kernel void cs_cull(uint                      id        [[thread_position_in_grid]],
                    device const GpuInstance* instances [[buffer(0)]],
                    constant CullParams&      p         [[buffer(1)]],
                    device GpuInstance*       visible   [[buffer(2)]],
                    device atomic_uint*       counter   [[buffer(3)]],
                    device GpuDrawArgs*       args      [[buffer(4)]],
                    texture2d<float>          hiz       [[texture(0)]])
{
    // Thread 0 sets up the draw arguments that do not depend on the result --
    // one set per level, since a level with no survivors still needs a valid
    // argument block with a zero instance count.
    if (id == 0) {
        for (uint l = 0; l < 4u; ++l) {
            args[l].index_count = p.index_counts[l];
            args[l].index_start = 0;
            args[l].base_vertex = 0;
            args[l].base_instance = 0;
        }
    }
    if (id >= p.counts.x) return;

    const GpuInstance inst = instances[id];

    // The bounding sphere, taken to world space. A sphere only needs its centre
    // moved and its radius scaled, which is exactly why the bounds are a sphere
    // and not a box: a box would have to be re-fitted, and a re-fitted box
    // around a rotated box is bigger than either.
    const float4 c = inst.model * float4(inst.bounds.xyz, 1.0f);
    // The largest scale on any axis. Using one axis, or the determinant, gets a
    // non-uniformly scaled object culled while it is still on screen.
    const float3 sx = inst.model[0].xyz, sy = inst.model[1].xyz, sz = inst.model[2].xyz;
    const float scale = sqrt(max(max(dot(sx, sx), dot(sy, sy)), dot(sz, sz)));
    const float radius = inst.bounds.w * scale;

    for (uint i = 0; i < 6; ++i) {
        // Outside by MORE than the radius. Not "outside": a sphere straddling
        // the plane is partly visible, and rejecting it clips objects at the
        // edge of the screen.
        if (dot(p.planes[i].xyz, c.xyz) + p.planes[i].w < -radius) return;
    }

    // OCCLUSION, after the frustum test and before the level choice. Testing it
    // first would be wasted on everything off screen, which on a dense scene is
    // most of it.
    if (p.counts.w != 0 && !is_null_texture(hiz) && Occluded(c.xyz, radius, hiz, p))
        return;

    // The projected radius in pixels, which is what picks the level.
    const float distance = max(length(c.xyz - p.eye.xyz), 1e-4);
    const float radius_px = radius * p.screen.z / distance;
    const uint lod = SelectLod(radius_px, p);

    // Compacted PER LEVEL, in whatever order the threads happen to finish.
    // Order does not matter for opaque geometry -- the depth buffer resolves
    // it -- and insisting on a stable order would need a prefix sum instead of
    // an atomic, which is a great deal of machinery to preserve something
    // nothing reads.
    //
    // The levels share one buffer at a fixed stride rather than having one
    // buffer each: an instance can land in any of them, so all four would have
    // to be sized for the whole scene anyway.
    const uint slot = atomic_fetch_add_explicit(&counter[lod], 1u, memory_order_relaxed);
    if (slot >= p.counts.z) return;  // this level's region is full
    visible[lod * p.counts.z + slot] = inst;
}

// The counter is also the draw's instance count, and the two must be the same
// number rather than two numbers that agree. Copying it in a second dispatch --
// after every cull thread has finished -- is what guarantees that; writing it
// from within the cull kernel would race with the threads still counting.
kernel void cs_cull_finish(uint                id      [[thread_position_in_grid]],
                           device atomic_uint* counter [[buffer(0)]],
                           device GpuDrawArgs* args    [[buffer(1)]],
                           constant CullParams& p      [[buffer(2)]])
{
    if (id != 0) return;
    for (uint l = 0; l < 4u; ++l)
        args[l].instance_count =
            min(atomic_load_explicit(&counter[l], memory_order_relaxed), p.counts.z);
}

// --- the depth pyramid --------------------------------------------------------
//
// Level 0 is a copy of the depth buffer; every level after it is the MINIMUM of
// four texels -- the farthest, in reversed-Z.
//
// Not the average, and not MTLBlitCommandEncoder's generateMipmaps, which is an
// average. An averaged depth pyramid is not conservative: it reports a surface
// nearer than the farthest thing in the region, so an object genuinely behind
// the farthest occluder can compare as visible or, worse, one in front of it
// can compare as hidden. Occlusion culling has to err in exactly one direction,
// and a min-reduction is what guarantees it.

kernel void cs_hiz_copy(depth2d<float, access::read> src [[texture(0)]],
                        texture2d<float, access::write> dst [[texture(1)]],
                        constant uint4& size [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= size.x || gid.y >= size.y) return;
    dst.write(float4(src.read(gid), 0.0, 0.0, 0.0), gid);
}

kernel void cs_hiz_reduce(texture2d<float, access::read> src [[texture(0)]],
                          texture2d<float, access::write> dst [[texture(1)]],
                          constant uint4& size [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    // size.xy is the DESTINATION's size, size.zw the source's.
    if (gid.x >= size.x || gid.y >= size.y) return;
    const uint2 base = gid * 2;
    // An ODD source dimension means the last destination texel covers three
    // source texels, not two, and dropping the third leaves a strip of the
    // depth buffer unrepresented -- so an object hidden behind it would be
    // drawn. Clamping the extra reads to the edge covers it: reading a texel
    // twice is harmless for a minimum.
    const uint2 last = uint2(max(size.z, 1u) - 1u, max(size.w, 1u) - 1u);
    float v = 1.0;
    for (uint dy = 0; dy < 2; ++dy)
        for (uint dx = 0; dx < 2; ++dx)
            v = min(v, src.read(min(base + uint2(dx, dy), last)).r);
    // The odd-size third column and row.
    if ((size.z & 1u) != 0u && gid.x == size.x - 1u)
        for (uint dy = 0; dy < 2; ++dy)
            v = min(v, src.read(min(uint2(base.x + 2, base.y + dy), last)).r);
    if ((size.w & 1u) != 0u && gid.y == size.y - 1u)
        for (uint dx = 0; dx < 2; ++dx)
            v = min(v, src.read(min(uint2(base.x + dx, base.y + 2), last)).r);
    dst.write(float4(v, 0.0, 0.0, 0.0), gid);
}
