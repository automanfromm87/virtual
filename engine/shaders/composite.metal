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
                             constant FrameUniforms& u [[buffer(1)]],
                             texture2d<float> src   [[texture(0)]],
                             texture2d<float> ao    [[texture(1)]],
                             texture2d<float> bloom [[texture(2)]],
                             sampler          smp   [[sampler(0)]])
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

    // BLOOM, added while everything is still linear. Adding it after the tone
    // map would put a haze over the highlights instead of light around them —
    // the curve has already flattened that range, so the sum lands in the same
    // few codes. u.lighting.y is the strength; a null texture reads black and
    // the term vanishes.
    color += bloom.sample(smp, in.uv).rgb * u.lighting.y;

    // A vignette, on purpose: it is a visible, measurable effect, so a test can
    // prove the composite pass actually ran rather than assuming it did.
    const float2 centred = in.uv * 2.0f - 1.0f;
    const float r = length(centred);
    color *= mix(1.0f, mix(1.0f, 0.62f, u.lighting.z),
                 smoothstep(0.65f, 1.35f, r));

    // TONE MAP AND GAMMA, once, at the end of the frame.
    //
    // This used to live in the surface shader, which meant every pass after it
    // was working on display-referred colour that had already been clamped to
    // one. Doing it here is what makes an HDR scene target worth having.
    const float a1 = 2.51f, b1 = 0.03f, c1 = 2.43f, d1 = 0.59f, e1 = 0.14f;
    color = saturate((color * (a1 * color + b1)) / (color * (c1 * color + d1) + e1));
    color = pow(color, 1.0f / 2.2f);

    return float4(color, 1.0f);
}
