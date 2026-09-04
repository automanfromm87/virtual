// No #include here: shader_types.h is textually prepended by the loader.
//
// The shadow pass. Depth only — there is no fragment function at all, because
// nothing is being shaded: the pass exists purely to record, for every texel of
// the light's view, how far away the nearest surface is.
//
// TWO BLOCKS, not one. It read the whole of FrameUniforms to use three of its
// fields, which cost a 656-byte write and a slice of the frame-wide uniform
// ring for every caster in every cascade -- and that ring going dry is what
// makes a frame come out black. DepthPass is bound once per cascade; DepthDraw
// is the model matrix and goes straight into the command buffer.
//
// The model matrix still means the same thing it does in the lit pass, which is
// what the shared shader_types.h contract is for.

struct ShadowOut {
    float4 position [[position]];
    float worldY;  // for the section cut
};

vertex ShadowOut vs_shadow(uint                     vid   [[vertex_id]],
                           device const VertexIn*   verts [[buffer(0)]],
                           constant DepthPass&      p     [[buffer(1)]],
                           constant DepthDraw&      d     [[buffer(5)]])
{
    const float4 world = d.model * float4(verts[vid].position.xyz, 1.0f);
    ShadowOut o;
    // The LIGHT's viewProj in a cascade, the camera's in the prepass. One field
    // rather than two, because a pass is only ever one of the two.
    o.position = p.viewProj * world;
    o.worldY = world.y;
    return o;
}

// A skinned caster has to be posed before it is recorded, or its shadow stays
// in the bind pose while the character moves — a shadow that is the right shape
// and in the wrong place, which reads as a lighting bug.
vertex ShadowOut vs_shadow_skinned(uint                     vid     [[vertex_id]],
                                   device const VertexIn*   verts   [[buffer(0)]],
                                   constant DepthPass&      p       [[buffer(1)]],
                                   device const SkinIn*     skin    [[buffer(2)]],
                                   device const float4x4*   palette [[buffer(3)]],
                                   constant DepthDraw&      d       [[buffer(5)]])
{
    const SkinIn s = skin[vid];
    float4x4 blend = float4x4(0.0f);
    float total = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        const float w = s.weights[i];
        if (w <= 0.0f) continue;
        blend += palette[s.joints[i]] * w;
        total += w;
    }
    if (total <= 0.0f) blend = float4x4(1.0f);

    const float4 world = d.model * (blend * float4(verts[vid].position.xyz, 1.0f));
    ShadowOut o;
    o.position = p.viewProj * world;
    o.worldY = world.y;
    return o;
}

// The pass is still colour-less, but it is no longer purely a vertex stage:
// the SECTION CUT has to apply here too.
//
// Miss this and an architectural cutaway comes out pitch black inside. The lit
// pass discards the walls above the cut, but the shadow map still holds them at
// full height, so every room sits in the shadow of a roof you already removed.
// The symptom looks like a lighting bug and is a shadow-pass bug.
fragment void fs_shadow(ShadowOut in [[stage_in]],
                        constant DepthPass& p [[buffer(1)]])
{
    if (in.worldY > p.cut.x) discard_fragment();
}
