// No #include here: shader_types.h is textually prepended by the loader.
// (An offline .metallib build would concatenate the same two files.)

struct VSOut {
    float4 position [[position]];
    float4 color;
};

vertex VSOut vs_main(uint                     vid   [[vertex_id]],
                     device const VertexIn*   verts [[buffer(0)]],
                     constant FrameUniforms&  u     [[buffer(1)]])
{
    VSOut o;
    o.position = u.viewProj * float4(verts[vid].position.xyz, 1.0f);
    o.color    = verts[vid].color * u.tint;
    return o;
}

fragment float4 fs_main(VSOut in [[stage_in]])
{
    return in.color;
}
