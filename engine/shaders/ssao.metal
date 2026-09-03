// No #include here: shader_types.h is textually prepended by the loader.
//
// Screen-space ambient occlusion. Interiors are dominated by INDIRECT light,
// and a single directional lamp plus a constant ambient term cannot express
// that at all — every corner of every room comes out equally bright, so nothing
// reads as enclosed. AO is the cheapest approximation of "how much of the sky
// can this point actually see", and indoors it does more for believability than
// any amount of extra direct light.
//
// Depth only: no normal buffer. Normals are recovered from the depth
// derivatives, which costs a G-buffer we do not have at the price of being
// wrong along silhouettes — an acceptable trade at this scale.

struct SsaoOut {
    float4 position [[position]];
    float2 uv;
};

vertex SsaoOut vs_ssao(uint vid [[vertex_id]])
{
    const float2 corners[3] = {float2(-1.0f, -1.0f), float2(3.0f, -1.0f),
                               float2(-1.0f, 3.0f)};
    SsaoOut o;
    o.position = float4(corners[vid], 0.0f, 1.0f);
    o.uv = corners[vid] * 0.5f + 0.5f;
    o.uv.y = 1.0f - o.uv.y;
    return o;
}

// View-space position from a depth sample.
//
// REVERSED-Z with an infinite far plane makes this unusually cheap: the
// projection's row 2 is (0,0,0,nearZ), so z_ndc = nearZ / -z_view and the
// inverse is a single divide. No inverse projection matrix needed.
static inline float3 ViewPos(float2 uv, float depth, constant FrameUniforms& u)
{
    const float nearZ = u.ssao.x;
    const float f = u.ssao.y;
    const float aspect = u.ssao.z;
    const float viewZ = -nearZ / max(depth, 1e-6f);  // negative, into the screen
    // uv -> NDC, remembering v runs the other way.
    const float2 ndc = float2(uv.x * 2.0f - 1.0f, 1.0f - uv.y * 2.0f);
    return float3(ndc.x * aspect / f * -viewZ, ndc.y / f * -viewZ, viewZ);
}

fragment float4 fs_ssao(SsaoOut in [[stage_in]],
                        constant FrameUniforms& u [[buffer(1)]],
                        depth2d<float> depthMap [[texture(0)]],
                        sampler        smp      [[sampler(0)]])
{
    const float depth = depthMap.sample(smp, in.uv);
    // Reversed-Z: 0 is the far plane, i.e. nothing was drawn here.
    if (depth <= 1e-6f) return float4(1.0f);

    const float3 P = ViewPos(in.uv, depth, u);

    // Normal from the depth derivatives. Wrong across a silhouette, where the
    // derivative jumps between two surfaces — which is why the range check
    // below throws out samples that are implausibly far away.
    //
    // dfdy FIRST. UV runs DOWN the screen while view-space Y runs up, so
    // dfdy(P) is -Y_view and cross(dfdx, dfdy) = X x -Y = -Z: a normal pointing
    // away from the camera. With it flipped, every sample in FRONT of the
    // surface scores a negative cosine and is discarded, so a floor beside a
    // ball collects no occlusion at all and the ball darkens itself instead.
    // The buffer still looked plausible -- faint, grey, vaguely shaped like the
    // geometry -- which is how it survived.
    const float3 N = normalize(cross(dfdy(P), dfdx(P)));

    const float radius = u.ssao.w;
    const float2 texel = 1.0f / float2(depthMap.get_width(), depthMap.get_height());

    // A fixed spiral rather than random offsets: no per-pixel noise means no
    // denoise pass afterwards. It bands slightly; a real engine trades that for
    // noise plus a blur.
    constexpr int kSamples = 12;
    float occlusion = 0.0f;
    for (int i = 0; i < kSamples; ++i) {
        const float t = (float(i) + 0.5f) / float(kSamples);
        const float angle = t * 6.2831853f * 3.0f;  // three turns
        // Screen-space radius shrinks with distance, so the world-space size of
        // the sampled neighbourhood stays roughly constant.
        const float screenRadius = radius * u.ssao.y / max(-P.z, 0.01f);
        const float2 offset =
            float2(cos(angle), sin(angle)) * sqrt(t) * screenRadius * 0.5f;
        const float2 suv = in.uv + offset;
        if (suv.x < 0.0f || suv.x > 1.0f || suv.y < 0.0f || suv.y > 1.0f) continue;

        const float sd = depthMap.sample(smp, suv);
        if (sd <= 1e-6f) continue;
        const float3 S = ViewPos(suv, sd, u);

        const float3 delta = S - P;
        const float dist = length(delta);
        if (dist < 1e-4f || dist > radius) continue;  // too far to occlude

        // How much of P's hemisphere this sample blocks. Samples behind the
        // surface contribute nothing.
        const float cosine = max(dot(N, delta / dist), 0.0f);
        // Fade with distance so a wall across the room does not darken this one.
        occlusion += cosine * (1.0f - dist / radius);
    }

    // The strength multiplier is a look control, not physics — the sampling
    // above estimates a visibility fraction and this decides how much of the
    // ambient term that fraction is allowed to remove. Too low and corners stay
    // flat; too high and every crease turns into a black smear.
    float ao = 1.0f - occlusion / float(kSamples) * 3.2f;
    return float4(saturate(ao));
}

// Also declared here rather than in composite.metal: the composite pass needs a
// second texture now, and keeping both halves of the AO story in one file makes
// the multiply easy to find.
