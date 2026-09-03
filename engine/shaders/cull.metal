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
    // Six frustum planes, xyz = normal, w = distance. Pointing INWARD, so a
    // point is inside when dot(plane.xyz, p) + plane.w >= 0 for all six.
    float4 planes[6];
    uint instance_count;
    uint index_count;
    uint pad0;
    uint pad1;
};

kernel void cs_cull(uint                      id        [[thread_position_in_grid]],
                    device const GpuInstance* instances [[buffer(0)]],
                    constant CullParams&      p         [[buffer(1)]],
                    device GpuInstance*       visible   [[buffer(2)]],
                    device atomic_uint*       counter   [[buffer(3)]],
                    device GpuDrawArgs*       args      [[buffer(4)]])
{
    // Thread 0 sets up the draw arguments that do not depend on the result.
    // The instance count is left to the atomic below -- it is the one field
    // that is not known until every thread has finished.
    if (id == 0) {
        args->index_count = p.index_count;
        args->index_start = 0;
        args->base_vertex = 0;
        args->base_instance = 0;
    }
    if (id >= p.instance_count) return;

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

    // Compacted, in whatever order the threads happen to finish. Order does not
    // matter for opaque geometry -- the depth buffer resolves it -- and
    // insisting on a stable order would need a prefix sum instead of an atomic,
    // which is a great deal of machinery to preserve something nothing reads.
    const uint slot = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    visible[slot] = inst;
}

// The counter is also the draw's instance count, and the two must be the same
// number rather than two numbers that agree. Copying it in a second dispatch --
// after every cull thread has finished -- is what guarantees that; writing it
// from within the cull kernel would race with the threads still counting.
kernel void cs_cull_finish(uint                id      [[thread_position_in_grid]],
                           device atomic_uint* counter [[buffer(0)]],
                           device GpuDrawArgs* args    [[buffer(1)]])
{
    if (id != 0) return;
    args->instance_count = atomic_load_explicit(counter, memory_order_relaxed);
}
