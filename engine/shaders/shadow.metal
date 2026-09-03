// No #include here: shader_types.h is textually prepended by the loader.
//
// The shadow pass. Depth only — there is no fragment function at all, because
// nothing is being shaded: the pass exists purely to record, for every texel of
// the light's view, how far away the nearest surface is.
//
// It reuses FrameUniforms rather than declaring its own block, so the model
// matrix means the same thing in both passes. Two structs that must agree is
// exactly the drift this engine's layout contract exists to prevent.

struct ShadowOut {
    float4 position [[position]];
    float worldY;  // for the section cut
};

vertex ShadowOut vs_shadow(uint                     vid   [[vertex_id]],
                           device const VertexIn*   verts [[buffer(0)]],
                           constant FrameUniforms&  u     [[buffer(1)]])
{
    const float4 world = u.model * float4(verts[vid].position.xyz, 1.0f);
    ShadowOut o;
    // lightViewProj, NOT viewProj: this is the world seen from the light.
    o.position = u.lightViewProj * world;
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
                        constant FrameUniforms& u [[buffer(1)]])
{
    if (in.worldY > u.surface.w) discard_fragment();
}
