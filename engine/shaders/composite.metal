// No #include here: shader_types.h is textually prepended by the loader.
//
// Fullscreen composite. Exists to give the render graph a SECOND pass with a
// real dependency on the first — a graph that orders one pass is a no-op.

struct CompositeOut {
    float4 position [[position]];
    float2 uv;
};

// No vertex buffer: one oversized triangle generated from the vertex id covers
// the whole viewport with three vertices instead of a quad's six, and avoids
// the diagonal seam a two-triangle quad puts down the middle.
vertex CompositeOut vs_composite(uint vid [[vertex_id]])
{
    const float2 corners[3] = {float2(-1.0f, -1.0f), float2(3.0f, -1.0f),
                               float2(-1.0f, 3.0f)};
    CompositeOut o;
    o.position = float4(corners[vid], 0.0f, 1.0f);
    o.uv = corners[vid] * 0.5f + 0.5f;
    // NDC +Y is the top of the screen, but texture v=0 is the FIRST row, which
    // is also the top. Flip so the copy is not upside down.
    o.uv.y = 1.0f - o.uv.y;
    return o;
}

fragment float4 fs_composite(CompositeOut in [[stage_in]],
                             texture2d<float> src [[texture(0)]],
                             texture2d<float> ao  [[texture(1)]],
                             sampler          smp [[sampler(0)]])
{
    // A real sampler object now, bound by the renderer. This used to be a
    // constexpr sampler declared inline — a workaround for the RHI not having
    // samplers at all.
    float3 color = src.sample(smp, in.uv).rgb;

    // Ambient occlusion, applied to the whole image rather than to the ambient
    // term alone. Strictly that is wrong — direct sunlight should not be
    // occluded by a nearby wall — but the ambient term is not separable without
    // a second render target, and indoors, where ambient dominates, the error
    // is far smaller than the gain. Defaults to a 1x1 white texture when the
    // SSAO pass is not in the graph.
    color *= ao.sample(smp, in.uv).r;

    // A vignette, on purpose: it is a visible, measurable effect, so a test can
    // prove the composite pass actually ran rather than assuming it did.
    const float2 centred = in.uv * 2.0f - 1.0f;
    const float r = length(centred);
    color *= mix(1.0f, 0.62f, smoothstep(0.65f, 1.35f, r));

    return float4(color, 1.0f);
}
