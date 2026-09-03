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

// Everything the grade needs, in one block. Separate from FrameUniforms because
// none of it means anything to a geometry pass, and putting display-referred
// parameters in the same struct as the lighting is how a renderer ends up with
// a surface shader that knows about contrast.
struct GradeParams {
    // .x lift, .y gamma, .z gain -- each a scalar applied to all channels after
    // the per-channel ones below. .w contrast pivot.
    float4 tone;
    float4 lift;        // .rgb per-channel shadows, .w unused
    float4 gamma;       // .rgb per-channel midtones
    float4 gain;        // .rgb per-channel highlights
    // .x contrast, .y saturation, .z temperature, .w tint
    float4 look;
};

fragment float4 fs_composite(CompositeOut in [[stage_in]],
                             constant FrameUniforms& u [[buffer(1)]],
                             constant GradeParams& g [[buffer(2)]],
                             device const float* exposure [[buffer(3)]],
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

    // EXPOSURE, applied while the image is still linear and BEFORE the tone
    // map. That order is the whole point: exposure decides which part of the
    // scene's range the curve is going to see, and applying it afterwards would
    // only brighten an image that had already been compressed -- which is what
    // a brightness slider does, and why a brightness slider cannot rescue a
    // blown-out sky.
    //
    // The value comes from a BUFFER the GPU wrote, so an automatic exposure
    // never travels to the CPU and back.
    color *= max(exposure[0], 1e-4f);

    // TONE MAP, once, at the end of the frame.
    //
    // This used to live in the surface shader, which meant every pass after it
    // was working on display-referred colour that had already been clamped to
    // one. Doing it here is what makes an HDR scene target worth having.
    const float a1 = 2.51f, b1 = 0.03f, c1 = 2.43f, d1 = 0.59f, e1 = 0.14f;
    color = saturate((color * (a1 * color + b1)) / (color * (c1 * color + d1) + e1));

    // --- the grade, in display-referred space ---------------------------------
    //
    // AFTER the tone map, and that is not the only defensible choice but it is
    // the one that matches how a colourist works: the curve has already decided
    // what white and black are, and the grade adjusts the picture between them.
    // Grading before the tone map means every adjustment also changes how the
    // curve compresses, so lifting the shadows desaturates the highlights.
    //
    // PARAMETRIC rather than a lookup table. A 3D LUT is what a film pipeline
    // uses because it can carry a look authored in a colour grading application;
    // this engine has no such application and no assets, so a table would be a
    // texture holding the same arithmetic done offline, plus interpolation
    // error. The parametric form is exact and can be animated.

    // WHITE BALANCE first, because it is a property of the light rather than a
    // look. Temperature shifts blue against red; tint shifts green against
    // magenta, which is the axis fluorescent lighting moves along.
    const float temp = g.look.z;
    color *= float3(1.0f + temp * 0.20f, 1.0f + g.look.w * 0.12f, 1.0f - temp * 0.20f);

    // LIFT, GAMMA, GAIN -- the standard three-way control. Lift moves the black
    // point, gain the white point, and gamma the curve between them without
    // moving either end.
    color = color * (g.gain.rgb * g.tone.z) + (g.lift.rgb + g.tone.x);
    color = max(color, 0.0f);
    color = pow(color, 1.0f / max(g.gamma.rgb * g.tone.y, 1e-3f));

    // CONTRAST about a pivot, not about zero. Scaling about zero darkens
    // everything as contrast rises, because every value moves away from black
    // rather than away from mid grey. 0.435 is 18% grey after the tone map.
    color = (color - g.tone.w) * g.look.x + g.tone.w;

    // SATURATION last, against the tone-mapped luminance. Rec.709 weights, so
    // desaturating leaves the perceived brightness alone -- an equal-weight
    // average would make greens go dark and blues go light.
    const float luma = dot(color, float3(0.2126f, 0.7152f, 0.0722f));
    color = mix(float3(luma), color, g.look.y);

    color = saturate(color);
    color = pow(color, 1.0f / 2.2f);

    return float4(color, 1.0f);
}
