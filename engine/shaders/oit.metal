// No #include here: shader_types.h is textually prepended by the loader.
//
// WEIGHTED-BLENDED ORDER-INDEPENDENT TRANSPARENCY (McGuire and Bavoil).
//
// THE PROBLEM. Alpha blending is not commutative -- src*a + dst*(1-a) gives a
// different answer depending on which surface arrives first -- so correct
// transparency needs the surfaces sorted back to front. Sorting by OBJECT is
// what this engine did, and it is wrong the moment two transparent objects
// intersect: there is no single order that is correct for every pixel of them,
// so one of them pops in front of the other along the intersection line.
// Sorting per triangle does not fix it either; two triangles can interpenetrate.
//
// THE TRICK. Approximate the sorted result with two quantities that CAN be
// accumulated in any order:
//
//   accum     sum of (colour * alpha * w)  and  sum of (alpha * w)
//   revealage product of (1 - alpha)
//
// Both are commutative, so no sort is needed at all. `w` is a weight that falls
// off with depth, standing in for the occlusion that a real sort would compute
// -- a near surface should dominate a far one, and the weight is how that gets
// expressed without knowing the order.
//
// WHAT IT GIVES UP. It is an approximation, and it shows most where alphas are
// high: a stack of nearly opaque sheets comes out too uniform, because the
// weight cannot fully express "the front one hides the rest". For glass, smoke,
// foliage and water -- everything transparency is actually used for -- it is
// far closer to right than an object sort, and it never pops.

struct OitOut {
    float4 accum [[color(0)]];
    // Single channel conceptually; written to all four so the multiply blend
    // has something defined in every one. Reading .r is what the resolve does.
    float4 revealage [[color(1)]];
};

// The depth weight. This is the whole approximation, and the shape matters:
// it has to fall off fast enough that a near surface dominates, and slowly
// enough that a distant one does not vanish into the float's denormals.
//
// The published function, with the alpha factored in so that a nearly
// transparent surface contributes nearly nothing however near it is.
static inline float OitWeight(float z, float alpha)
{
    // z is distance along the view axis, in metres. The 0.1 and 200 bracket the
    // range this engine's scenes live in; outside it the weight saturates
    // rather than overflowing, which is the failure to prefer -- an infinite
    // weight makes one surface swallow the whole pixel.
    const float d = clamp(z * 0.1f, 0.01f, 200.0f);
    return clamp(alpha * alpha * alpha * 1e8f * (1.0f / (d * d * d)), 1e-2f, 3e3f);
}

fragment OitOut fs_oit(VSOut in [[stage_in]],
                       constant FrameUniforms& u [[buffer(1)]],
                       device const GpuLight* lights [[buffer(2)]],
                       constant GpuCascades& cascades [[buffer(3)]],
                       texture2d<float> albedoMap    [[texture(0)]],
                       texture2d<float> roughnessMap [[texture(1)]],
                       depth2d<float>   shadowMap    [[texture(2)]],
                       depth2d<float>   shadowAtlas  [[texture(3)]],
                       texture2d<float> normalMap    [[texture(4)]],
                       texturecube<float> irradianceMap [[texture(5)]],
                       texturecube<float> specularMap   [[texture(6)]],
                       texture2d<float>   brdfLut       [[texture(7)]],
                       texture2d<float> metallicMap  [[texture(8)]],
                       texture2d<float> emissiveMap  [[texture(9)]],
                       texture2d<float> occlusionMap [[texture(10)]],
                       texture3d<float> giR [[texture(11)]],
                       texture3d<float> giG [[texture(12)]],
                       texture3d<float> giB [[texture(13)]],
                       sampler          smp          [[sampler(0)]],
                       sampler          envSmp       [[sampler(1)]],
                       sampler          shadowSmp    [[sampler(2)]],
                       sampler          giSmp        [[sampler(3)]],
                       constant GpuClusters* clusters       [[buffer(5)]],
                       device const uint*    clusterCounts  [[buffer(6)]],
                       device const uint*    clusterIndices [[buffer(7)]])
{
    if (in.worldPos.y > u.surface.w) discard_fragment();

    const float3 albedo =
        in.color.rgb * u.baseColor.rgb * albedoMap.sample(smp, in.uv).rgb;
    const float roughness =
        saturate(u.surface.x * roughnessMap.sample(smp, in.uv).r);
    const float metallic = saturate(u.surface.y * metallicMap.sample(smp, in.uv).r);
    const float ao = occlusionMap.sample(smp, in.uv).r;
    const float3 N = ApplyNormalMap(normalize(in.normalW), in.tangentW, in.uv,
                                    normalMap, smp, u.emissive.w);

    const float3 lit = ShadeSurface(in.worldPos, N, albedo, roughness, metallic,
                                    in.lightClip, u, lights, cascades, shadowMap,
                                    shadowAtlas, shadowSmp, irradianceMap,
                                    specularMap, brdfLut, envSmp, ao, clusters,
                                    clusterCounts, clusterIndices, in.position.xy,
                                    dot(in.worldPos - u.eyePos.xyz, u.viewDir.xyz),
                                    giR, giG, giB, giSmp);
    const float3 emit = u.emissive.rgb * emissiveMap.sample(smp, in.uv).rgb;
    const float3 colour = lit + emit;
    const float alpha = saturate(in.color.a);

    const float z = dot(in.worldPos - u.eyePos.xyz, u.viewDir.xyz);
    const float w = OitWeight(z, alpha);

    OitOut o;
    // PREMULTIPLIED, then weighted. The resolve divides by the accumulated
    // alpha*w, so the weight cancels out of the colour and only affects how
    // much each surface counts toward the average.
    o.accum = float4(colour * alpha * w, alpha * w);
    o.revealage = float4(alpha);
    return o;
}

// The resolve. Reads the two buffers and produces one premultiplied colour to
// blend over the opaque frame.
struct OitResolveOut {
    float4 position [[position]];
    float2 uv;
};

vertex OitResolveOut vs_oit_resolve(uint vid [[vertex_id]])
{
    const float2 corner = float2((vid << 1) & 2, vid & 2);
    OitResolveOut o;
    o.position = float4(corner * 2.0f - 1.0f, 0.0f, 1.0f);
    o.uv = float2(corner.x, 1.0f - corner.y);
    return o;
}

fragment float4 fs_oit_resolve(OitResolveOut in [[stage_in]],
                               texture2d<float> accumTex [[texture(0)]],
                               texture2d<float> revealTex [[texture(1)]],
                               sampler smp [[sampler(0)]])
{
    const float4 accum = accumTex.sample(smp, in.uv);
    const float reveal = revealTex.sample(smp, in.uv).r;

    // reveal is the product of (1 - alpha) over every surface, so 1 - reveal is
    // how much of the pixel the transparent surfaces cover in total. At 1 the
    // pixel is untouched and the opaque frame shows through unchanged.
    if (reveal > 0.9999f) return float4(0.0f);

    // The weighted average colour. Dividing by accum.a is what removes the
    // weight again -- it was only ever there to decide how much each surface
    // counted, never to change its colour. The max guards a pixel where every
    // contributing alpha was tiny; without it this is 0/0 and the result is a
    // NaN that spreads through the bloom and the tone map.
    const float3 average = accum.rgb / max(accum.a, 1e-5f);
    // PREMULTIPLIED out, so the caller blends with (ONE, ONE_MINUS_SRC_ALPHA)
    // and the coverage in .a does both jobs at once.
    return float4(average * (1.0f - reveal), 1.0f - reveal);
}
