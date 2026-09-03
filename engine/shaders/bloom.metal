// No #include here: shader_types.h is textually prepended by the loader.
//
// Bloom: the halo a real lens puts around anything much brighter than the rest
// of the frame. Light scatters inside the glass and on the sensor, so a bright
// source bleeds into its neighbours — and because every photograph and every
// eye does it, an image without it reads as synthetic even when the shading is
// right.
//
// Three stages, each its own pass: pull out what is bright, blur it wide, add
// it back. The blur is separable, so a wide radius costs two one-dimensional
// passes rather than one quadratic two-dimensional one.

struct FsOut {
    float4 position [[position]];
    float2 uv;
};

// The same oversized triangle the composite uses: three vertices, no buffer,
// and no diagonal seam down the middle.
vertex FsOut vs_bloom(uint vid [[vertex_id]])
{
    const float2 corners[3] = {float2(-1.0f, -1.0f), float2(3.0f, -1.0f),
                               float2(-1.0f, 3.0f)};
    FsOut o;
    o.position = float4(corners[vid], 0.0f, 1.0f);
    o.uv = corners[vid] * 0.5f + 0.5f;
    o.uv.y = 1.0f - o.uv.y;
    return o;
}

// BRIGHT PASS. Keeps what is above the threshold and nothing else.
//
// The knee matters. A hard cut makes a surface pop into bloom the instant it
// crosses the line, so a slow camera move produces a shimmering edge crawling
// across the image. Softening the first stop of the transition costs three
// lines and removes it entirely.
//
// u.ssao.x is the threshold and .y the knee width, both in linear radiance.
fragment float4 fs_bloom_bright(FsOut in [[stage_in]],
                                constant FrameUniforms& u [[buffer(1)]],
                                texture2d<float> src [[texture(0)]],
                                sampler smp [[sampler(0)]])
{
    float3 c = src.sample(smp, in.uv).rgb;

    // Reject anything not finite BEFORE it can spread.
    //
    // A blur turns one bad pixel into a bad screen, and the damage does not
    // stop at the bloom: an infinity multiplied by a zero bloom strength is a
    // NaN, which then survives the add, the tone map and saturate() all the way
    // to the framebuffer. That is exactly how this was found — binding a bloom
    // texture with the strength set to zero still zeroed a whole colour channel
    // across the entire image.
    c = select(float3(0.0f), c, isfinite(c));
    // FIREFLY CLAMP. A single pixel of enormous radiance -- a specular hit
    // right on a light -- spreads over the whole kernel and reads as a smear
    // rather than a glow. Capping it costs energy that was never going to look
    // like anything.
    c = min(c, float3(80.0f));

    const float threshold = u.ssao.x;
    const float knee = max(u.ssao.y, 1e-4f);
    // Brightness by the maximum channel, not by luminance: a saturated red
    // lamp has a low luminance and is unmistakably a light source.
    const float brightness = max(max(c.r, c.g), c.b);
    float contribution = brightness - threshold + knee;
    contribution = clamp(contribution, 0.0f, 2.0f * knee);
    contribution = contribution * contribution / (4.0f * knee);
    const float weight = max(contribution, brightness - threshold) /
                         max(brightness, 1e-4f);
    return float4(c * weight, 1.0f);
}

// SEPARABLE GAUSSIAN. u.ssao.zw is the step between taps, in uv — the caller
// puts (1/width, 0) in it for the horizontal pass and (0, 1/height) for the
// vertical one, so one shader serves both and the two cannot drift apart.
fragment float4 fs_bloom_blur(FsOut in [[stage_in]],
                              constant FrameUniforms& u [[buffer(1)]],
                              texture2d<float> src [[texture(0)]],
                              sampler smp [[sampler(0)]])
{
    const float2 step = u.ssao.zw;
    // Nine taps at five offsets, using the hardware's bilinear filter to fetch
    // two texels per sample at a weighted midpoint. Half the fetches for the
    // same kernel.
    const float weights[3] = {0.2270270270f, 0.3162162162f, 0.0702702703f};
    const float offsets[3] = {0.0f, 1.3846153846f, 3.2307692308f};

    float3 sum = src.sample(smp, in.uv).rgb * weights[0];
    for (int i = 1; i < 3; ++i) {
        const float2 d = step * offsets[i];
        sum += src.sample(smp, in.uv + d).rgb * weights[i];
        sum += src.sample(smp, in.uv - d).rgb * weights[i];
    }
    // Belt and braces: the bright pass filters its input, but a blur chain is
    // long enough that anything slipping in halfway through would still reach
    // the frame.
    return float4(select(float3(0.0f), sum, isfinite(sum)), 1.0f);
}
