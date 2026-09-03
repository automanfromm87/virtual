// GPU SKINNING, written back to a buffer instead of consumed on the spot.
//
// The vertex shader already blends a skinned vertex, and for DRAWING that is
// the right place: the result is needed once, by the next stage, and never
// again. But it means the posed triangles exist only inside the rasteriser,
// and three things outside it want them:
//
//   * RAY TRACING. An acceleration structure is built from a buffer. Built from
//     the bind pose, it casts the shadow of a character standing still while
//     the character walks.
//   * COLLISION against the posed mesh, rather than against a proxy shape.
//   * Any second pass over the same geometry -- a depth prepass, a shadow map,
//     a G-buffer -- which currently re-runs the blend from scratch each time.
//
// So this does the same arithmetic in a compute kernel and stores it. The
// duplication with vs_skinned is real and deliberate: they run at different
// times for different consumers, and the test asserts they agree to the last
// bit rather than trusting that they will.

struct SkinParams {
    uint vertex_count;
    uint joint_count;
    uint pad0;
    uint pad1;
};

kernel void cs_skin(uint                     vid     [[thread_position_in_grid]],
                    device const VertexIn*   verts   [[buffer(0)]],
                    device const SkinIn*     skin    [[buffer(1)]],
                    device const float4x4*   palette [[buffer(2)]],
                    constant SkinParams&     p       [[buffer(3)]],
                    device VertexIn*         out     [[buffer(4)]])
{
    // The dispatch rounds the thread count UP to a whole threadgroup, so the
    // last group runs threads past the end of the mesh. Nothing else stops them
    // writing there.
    if (vid >= p.vertex_count) return;

    const SkinIn s = skin[vid];
    float4x4 blend = float4x4(0.0f);
    float total = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        const float w = s.weights[i];
        if (w <= 0.0f) continue;
        // An index past the end of the palette reads whatever follows it. The
        // vertex shader gets away without this check because its palette is
        // sized by the same code that sized the mesh; here the buffer is
        // written by a separate pass and the guard is cheap.
        if (s.joints[i] >= p.joint_count) continue;
        blend += palette[s.joints[i]] * w;
        total += w;
    }
    // No influences at all: leave the vertex where the modeller put it rather
    // than collapsing it onto the origin, which is what a zero matrix does.
    if (total <= 0.0f) blend = float4x4(1.0f);

    // VertexIn's vectors are float3 on the GPU -- padded to 16 bytes, which is
    // what the static_asserts in renderer.cc pin down, but three components
    // wide to the shader.
    VertexIn v = verts[vid];
    v.position = (blend * float4(verts[vid].position.xyz, 1.0f)).xyz;
    // w = 0 drops both translations: a normal is a direction.
    v.normal = (blend * float4(verts[vid].normal.xyz, 0.0f)).xyz;
    // Colour and uv ride along unchanged, so the output is a drop-in
    // replacement for the input buffer rather than a parallel array something
    // downstream has to know how to combine.
    out[vid] = v;
}
