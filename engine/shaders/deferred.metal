// DEFERRED SHADING: write the surface out, light it afterwards.
//
// The forward pass shades every fragment it rasterises, including the ones a
// later triangle covers up, and it re-runs the whole light loop for each. With
// N lights and M layers of overdraw that is N*M evaluations of the BRDF for
// every pixel of the screen, and only one layer of it survives.
//
// Deferred splits it: one pass writes what the surface IS -- albedo, normal,
// roughness, metallic -- into a G-buffer, and a second pass lights each pixel
// exactly once. The cost of lighting stops depending on the geometry entirely.
//
// What it gives up, and this is not a small list:
//
//   * TRANSPARENCY. A G-buffer holds one surface per pixel, and glass needs
//     two. Transparent geometry has to be drawn forward, afterwards.
//   * MSAA. The G-buffer would need every sample stored and lit, which is four
//     times the memory and four times the lighting -- and undoes the reason for
//     deferring. This engine's forward path keeps 4x MSAA; the deferred path
//     does not have it.
//   * Per-material variety. Everything shares one BRDF, because the lighting
//     pass sees the buffer and not the material.
//
// Neither is better. Forward wins with few lights, and deferred wins as the
// light count climbs -- which is why both exist here rather than one replacing
// the other.

struct GBufferOut {
    // Albedo can exceed 1: an emissive tint of 7 is how a light bulb is made to
    // bloom, so an 8-bit target would clip it back to white and take the colour
    // with it. Half-float throughout.
    float4 albedoRough [[color(0)]];  // rgb albedo, a roughness
    float4 normalMetal [[color(1)]];  // rgb world normal, a metallic
};

fragment GBufferOut fs_gbuffer(VSOut in [[stage_in]],
                               constant FrameUniforms& u [[buffer(1)]],
                               texture2d<float> albedoMap    [[texture(0)]],
                               texture2d<float> roughnessMap [[texture(1)]],
                               texture2d<float> normalMap    [[texture(4)]],
                               texture2d<float> metallicMap  [[texture(8)]],
                               sampler          smp          [[sampler(0)]])
{
    if (in.worldPos.y > u.surface.w) discard_fragment();

    GBufferOut o;
    o.albedoRough.rgb =
        in.color.rgb * u.baseColor.rgb * albedoMap.sample(smp, in.uv).rgb;
    o.albedoRough.a = saturate(u.surface.x * roughnessMap.sample(smp, in.uv).r);
    // The normal map is applied HERE, not in the lighting pass. The G-buffer's
    // whole contract is that it holds what the surface IS, and after a normal
    // map the surface faces somewhere else -- the lighting pass has no tangent
    // frame to apply one with, because a fullscreen triangle has no mesh.
    o.normalMetal.rgb = ApplyNormalMap(normalize(in.normalW), in.tangentW, in.uv,
                                       normalMap, smp, u.emissive.w);
    o.normalMetal.a = saturate(u.surface.y * metallicMap.sample(smp, in.uv).r);
    return o;
}

struct FullscreenOut {
    float4 position [[position]];
    float2 uv;
};

vertex FullscreenOut vs_deferred(uint vid [[vertex_id]])
{
    // One oversized triangle, not two triangles making a quad. A quad has a
    // seam down the diagonal where the two halves meet, and fragments there are
    // rasterised twice.
    const float2 corner = float2((vid << 1) & 2, vid & 2);
    FullscreenOut o;
    o.position = float4(corner * 2.0f - 1.0f, 0.0f, 1.0f);
    o.uv = float2(corner.x, 1.0f - corner.y);
    return o;
}

fragment float4 fs_deferred(FullscreenOut in [[stage_in]],
                            constant FrameUniforms& u [[buffer(1)]],
                            device const GpuLight* lights [[buffer(2)]],
                            constant GpuCascades& cascades [[buffer(3)]],
                            texture2d<float> albedoRough [[texture(0)]],
                            texture2d<float> normalMetal [[texture(1)]],
                            depth2d<float>   shadowMap   [[texture(2)]],
                            depth2d<float>   shadowAtlas [[texture(3)]],
                            depth2d<float>   sceneDepth  [[texture(4)]],
                            texturecube<float> irradianceMap [[texture(5)]],
                            texturecube<float> specularMap   [[texture(6)]],
                            texture2d<float>   brdfLut       [[texture(7)]],
                            sampler          smp         [[sampler(0)]],
                            sampler          envSmp      [[sampler(1)]],
                            sampler          shadowSmp   [[sampler(2)]],
                            constant GpuClusters* clusters       [[buffer(5)]],
                            device const uint*    clusterCounts  [[buffer(6)]],
                            device const uint*    clusterIndices [[buffer(7)]])
{
    const float depth = sceneDepth.sample(smp, in.uv);
    // Reversed-Z: 0 is the far plane, and it is what an untouched pixel holds.
    // Those pixels are background, and lighting them would apply the ambient
    // term to the sky and wash the whole image toward grey.
    //
    // DISCARD rather than return black. The forward path leaves background
    // pixels holding the pass's clear colour; writing black over them instead
    // makes deferred darker than forward everywhere the geometry does not
    // reach -- a systematic difference that is easy to mistake for a lighting
    // bug, because it looks like one.
    if (depth <= 0.0f) discard_fragment();

    // WORLD POSITION from depth. The G-buffer does not store it: a position is
    // three floats of the same magnitude as the scene, so half-float loses
    // centimetres at ten metres out and the shadows crawl. The depth buffer
    // already holds the information exactly, and one matrix multiply recovers
    // it.
    float2 ndc = in.uv * 2.0f - 1.0f;
    ndc.y = -ndc.y;  // uv runs down the screen, clip space runs up it
    const float4 world4 = u.invViewProj * float4(ndc, depth, 1.0f);
    const float3 worldPos = world4.xyz / world4.w;

    const float4 ar = albedoRough.sample(smp, in.uv);
    const float4 nm = normalMetal.sample(smp, in.uv);

    // The same ShadeSurface the forward pass calls. Not a copy of it: see the
    // top of shading.metal for why that distinction is the whole point.
    const float3 lit =
        ShadeSurface(worldPos, nm.xyz, ar.rgb, ar.a, nm.a,
                     u.lightViewProj * float4(worldPos, 1.0f), u, lights,
                     cascades, shadowMap, shadowAtlas, shadowSmp,
                     irradianceMap, specularMap, brdfLut, envSmp,
                     // No baked occlusion in the deferred path: the G-buffer
                     // has no channel left for it, and packing it into an
                     // alpha that already holds roughness would cost both.
                     1.0f, clusters, clusterCounts, clusterIndices,
                     in.position.xy,
                     dot(worldPos - u.eyePos.xyz, u.viewDir.xyz));
    return float4(lit, 1.0f);
}
