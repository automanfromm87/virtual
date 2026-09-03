// No #include here: shader_types.h is textually prepended by the loader.
//
// THE MESH-SHADER PATH. One object threadgroup per meshlet decides whether that
// meshlet is visible; the ones that survive launch a mesh threadgroup that
// writes their vertices and triangles directly, with no index buffer and no
// draw-call bookkeeping.
//
// What this buys over the vertex pipeline, in order of how much it matters:
//
//   A culled meshlet costs ONE object threadgroup. The vertex pipeline's unit
//   of culling is a whole object, so the back half of a rock is transformed,
//   clipped and rasterised into nothing.
//
//   The culling decision is made on the GPU and consumed on the GPU. The
//   indirect-draw route -- a compute pass writing draw arguments -- works, and
//   it needs a buffer per object and a barrier between the passes.
//
//   Triangle indices are BYTES. A meshlet has at most 64 vertices, so its
//   triangles cost three bytes each rather than twelve.

struct MeshletPayload {
    uint meshlet;
};

// Mirrors eng::Meshlet. Checked by a static_assert on the C++ side.
struct GpuMeshlet {
    uint vertex_offset;
    uint triangle_offset;
    uint vertex_count;
    uint triangle_count;
    float4 sphere;      // xyz centre, w radius, in OBJECT space
    float4 cone;        // xyz axis, w cutoff
    float4 cone_apex;   // x offset along the axis, yzw unused
};

// Object space -> is this meshlet worth drawing?
static bool MeshletVisible(GpuMeshlet m, constant FrameUniforms& u)
{
    // The sphere in WORLD space. `model` is rotation plus uniform scale here,
    // so the radius scales by the matrix's column length -- a non-uniform
    // scale would need the largest of the three and would make the sphere an
    // ellipsoid anyway.
    const float4 c = u.model * float4(m.sphere.xyz, 1.0f);
    const float scale = length(u.model[0].xyz);
    const float radius = m.sphere.w * scale;

    // BACKFACE CULLING, by cone. The apex is pushed back along the axis until
    // every triangle's plane passes through it, so this test is conservative:
    // it rejects only when no triangle in the meshlet can be front-facing.
    //
    // Without the apex a cluster off to one side of a curved object is culled
    // from angles it is plainly visible from, and the symptom is a hole that
    // opens and closes as the camera moves.
    if (m.cone.w <= 1.0f) {
        const float3 axis = normalize((u.model * float4(m.cone.xyz, 0.0f)).xyz);
        const float3 apex = c.xyz - axis * (m.cone_apex.x * scale);
        const float3 to_eye = normalize(u.eyePos.xyz - apex);
        if (dot(-to_eye, axis) >= m.cone.w) return false;
    }

    // FRUSTUM, from the view-projection's rows. Cheaper than passing six
    // planes, and it is the same test the CPU-side object culling does.
    const float4x4 vp = u.viewProj;
    for (int i = 0; i < 3; ++i) {
        // Row i plus and minus row 3 give the two planes of axis i. Reversed-Z
        // makes the near plane row3 - row2 and the far plane vanish, so only
        // the four side planes and the near one are tested.
        for (int sign = 0; sign < 2; ++sign) {
            const float4 plane =
                sign == 0 ? float4(vp[0][3] + vp[0][i], vp[1][3] + vp[1][i],
                                   vp[2][3] + vp[2][i], vp[3][3] + vp[3][i])
                          : float4(vp[0][3] - vp[0][i], vp[1][3] - vp[1][i],
                                   vp[2][3] - vp[2][i], vp[3][3] - vp[3][i]);
            const float len = length(plane.xyz);
            if (len < 1e-6f) continue;
            if ((dot(plane.xyz, c.xyz) + plane.w) / len < -radius) return false;
        }
    }
    return true;
}

[[object]] void os_meshlet(object_data MeshletPayload& payload [[payload]],
                           mesh_grid_properties grid,
                           uint gid [[threadgroup_position_in_grid]],
                           constant FrameUniforms& u [[buffer(1)]],
                           device const GpuMeshlet* meshlets [[buffer(4)]],
                           device atomic_uint* stats [[buffer(8)]])
{
    payload.meshlet = gid;
    const bool visible = MeshletVisible(meshlets[gid], u);
    // A COUNTER, so a test can see how many survived. It is an atomic add per
    // object threadgroup, which is one per meshlet -- a few thousand a frame at
    // most, and the alternative is having no way to tell culling that works
    // from culling that does nothing.
    if (visible) atomic_fetch_add_explicit(&stats[0], 1u, memory_order_relaxed);
    // ZERO threadgroups is how a meshlet is rejected. Nothing downstream runs:
    // no vertex is fetched, no triangle is set up, no fragment is shaded.
    grid.set_threadgroups_per_grid(uint3(visible ? 1u : 0u, 1u, 1u));
}

using MeshletMesh = mesh<VSOut, void, 64, 124, topology::triangle>;

[[mesh]] void ms_meshlet(MeshletMesh out,
                         const object_data MeshletPayload& payload [[payload]],
                         uint tid [[thread_position_in_threadgroup]],
                         constant FrameUniforms& u [[buffer(1)]],
                         device const VertexIn* verts [[buffer(0)]],
                         device const GpuMeshlet* meshlets [[buffer(4)]],
                         device const uint* meshlet_vertices [[buffer(5)]],
                         device const uchar* meshlet_triangles [[buffer(6)]])
{
    const GpuMeshlet m = meshlets[payload.meshlet];
    // ONE THREAD sets the counts, and it has to happen before any thread writes
    // a vertex or a primitive -- the counts are what tell the hardware how much
    // of the output to keep.
    if (tid == 0) out.set_primitive_count(m.triangle_count);
    // A BARRIER, because the count has to be established before any thread
    // writes a vertex or an index. Without it the other threads race the one
    // that sets it, and what the rasteriser sees is whatever the count happened
    // to be when it looked -- which on this hardware is nothing at all.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One vertex per thread, and one triangle per thread, with the threadgroup
    // sized to cover the larger of the two. A meshlet has at most 64 vertices
    // and 124 triangles, so 128 threads covers both with no loop.
    if (tid < m.vertex_count) {
        const VertexIn v = verts[meshlet_vertices[m.vertex_offset + tid]];
        VSOut o;
        const float4 worldPos = u.model * float4(v.position.xyz, 1.0f);
        o.position = u.viewProj * worldPos;
        o.worldPos = worldPos.xyz;
        o.normalW = (u.model * float4(v.normal.xyz, 0.0f)).xyz;
        o.color = v.color * u.tint;
        o.uv = v.uv.xy;
        o.lightClip = u.lightViewProj * worldPos;
        o.tangentW = float4((u.model * float4(v.tangent.xyz, 0.0f)).xyz, v.tangent.w);
        o.eyeW = u.eyePos.xyz;
        out.set_vertex(tid, o);
    }
    if (tid < m.triangle_count) {
        const uint base = (m.triangle_offset + tid) * 3;
        out.set_index(tid * 3 + 0, meshlet_triangles[base + 0]);
        out.set_index(tid * 3 + 1, meshlet_triangles[base + 1]);
        out.set_index(tid * 3 + 2, meshlet_triangles[base + 2]);
    }
}
