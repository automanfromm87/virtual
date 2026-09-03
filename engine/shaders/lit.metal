// No #include here: shader_types.h is textually prepended by the loader.
//
// Cook-Torrance microfacet BRDF: GGX normal distribution, Smith height-
// correlated geometry, Schlick Fresnel. One directional light plus a constant
// ambient term.
//
// The metallic workflow, which is the part people get wrong: a metal has NO
// diffuse lobe, and its specular colour IS its base colour. A dielectric has a
// diffuse lobe and a colourless 4% specular. `metallic` blends between those
// two physical facts — it is not a slider on a look.

struct VSOut {
    float4 position [[position]];
    float3 normalW;  // world space, NOT normalized (interpolation shortens it)
    float3 worldPos;
    float4 color;
    float2 uv;
    float4 lightClip;  // position in the light's clip space, for the shadow lookup
    // World-space tangent in xyz, the bitangent's handedness in w. Interpolated
    // like the normal, and like the normal it comes out short -- the fragment
    // renormalises. w is constant across a triangle in every mesh this engine
    // produces, so interpolating it is free and it survives.
    float4 tangentW;
};

// Perturbs a world normal by a tangent-space normal map sample.
//
// A no-op when nothing is bound: an unbound texture makes is_null_texture true
// and the interpolated vertex normal comes straight back out, so an untextured
// material costs one branch and no samples.
static inline float3 ApplyNormalMap(float3 N, float4 tangentW, float2 uv,
                                    texture2d<float> normalMap, sampler smp,
                                    float strength)
{
    if (is_null_texture(normalMap)) return N;

    // Re-orthogonalise against the INTERPOLATED normal. The tangent was made
    // perpendicular to the vertex normal at mesh time, but interpolation across
    // a triangle does not preserve that, and a TBN whose axes are not
    // orthogonal is not invertible by transpose -- which is how it is used
    // below.
    float3 T = tangentW.xyz - N * dot(N, tangentW.xyz);
    const float len = length(T);
    if (len < 1e-6f) return N;  // degenerate frame: better flat than NaN
    T /= len;
    const float3 B = cross(N, T) * tangentW.w;

    // Unpack from 0..1 to -1..1. Only xy are used to rebuild z: an 8-bit map's
    // blue channel is the least precise of the three and is redundant given the
    // other two are unit-length together, so reconstructing it is both cheaper
    // and more accurate than trusting it. It also makes two-channel maps (BC5,
    // which stores exactly xy) work with no special case.
    float2 xy = normalMap.sample(smp, uv).xy * 2.0f - 1.0f;
    // STRENGTH scales the tangent-space slope, not the final vector. Scaling
    // the vector and renormalising would rotate it toward the surface normal
    // non-linearly and saturate; scaling xy before rebuilding z is the same
    // thing as making the bumps taller or shallower.
    xy *= strength;
    const float z = sqrt(max(1.0f - dot(xy, xy), 0.0f));
    return normalize(T * xy.x + B * xy.y + N * z);
}

vertex VSOut vs_lit(uint                     vid   [[vertex_id]],
                    device const VertexIn*   verts [[buffer(0)]],
                    constant FrameUniforms&  u     [[buffer(1)]])
{
    VSOut o;
    const float4 worldPos = u.model * float4(verts[vid].position.xyz, 1.0f);
    o.position = u.viewProj * worldPos;
    o.worldPos = worldPos.xyz;

    // w=0 so the model matrix's translation column does not move the normal.
    // Correct only because `model` is rotation + uniform scale; a non-uniform
    // scale would need the inverse-transpose instead.
    o.normalW = (u.model * float4(verts[vid].normal.xyz, 0.0f)).xyz;

    o.color = verts[vid].color * u.tint;
    o.uv = verts[vid].uv.xy;
    o.lightClip = u.lightViewProj * worldPos;
    o.tangentW = float4((u.model * float4(verts[vid].tangent.xyz, 0.0f)).xyz,
                        verts[vid].tangent.w);
    return o;
}

// The SKINNED variant. Same output, same fragment stage — the only difference
// is where the vertex was before it got there.
//
// Linear blend skinning: sum the joint matrices weighted per vertex, then
// transform once. Summing the MATRICES rather than transforming four times and
// averaging the results is not an optimisation, it is the definition — the two
// agree for positions, and for normals the blended matrix is what keeps the
// basis consistent.
//
// The palette is world-relative already: palette[j] = jointWorld * inverseBind,
// so `model` still applies on top for where the character is standing.
// The same vertex stage, with model and tint read PER INSTANCE instead of from
// the per-draw uniform block. Everything else is identical, which is why this
// is a separate entry point rather than a branch: a branch would cost the
// non-instanced path a buffer read it never uses, and there is no version of
// this where the two disagree about lighting.
vertex VSOut vs_lit_instanced(uint                      vid       [[vertex_id]],
                              uint                      iid       [[instance_id]],
                              device const VertexIn*    verts     [[buffer(0)]],
                              constant FrameUniforms&   u         [[buffer(1)]],
                              device const GpuInstance* instances [[buffer(4)]])
{
    const GpuInstance inst = instances[iid];
    VSOut o;
    const float4 worldPos = inst.model * float4(verts[vid].position.xyz, 1.0f);
    o.position = u.viewProj * worldPos;
    o.worldPos = worldPos.xyz;
    o.normalW = (inst.model * float4(verts[vid].normal.xyz, 0.0f)).xyz;
    o.color = verts[vid].color * inst.tint;
    o.uv = verts[vid].uv.xy;
    o.lightClip = u.lightViewProj * worldPos;
    o.tangentW = float4((inst.model * float4(verts[vid].tangent.xyz, 0.0f)).xyz,
                        verts[vid].tangent.w);
    return o;
}

vertex VSOut vs_skinned(uint                     vid     [[vertex_id]],
                        device const VertexIn*   verts   [[buffer(0)]],
                        constant FrameUniforms&  u       [[buffer(1)]],
                        device const SkinIn*     skin    [[buffer(2)]],
                        device const float4x4*   palette [[buffer(3)]])
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
    // No influences at all: leave the vertex where the modeller put it rather
    // than collapsing it onto the origin, which is what a zero matrix does.
    if (total <= 0.0f) blend = float4x4(1.0f);

    VSOut o;
    const float4 skinned = blend * float4(verts[vid].position.xyz, 1.0f);
    const float4 worldPos = u.model * skinned;
    o.position = u.viewProj * worldPos;
    o.worldPos = worldPos.xyz;
    // w = 0 drops both translations: a normal is a direction.
    const float3 skinnedN = (blend * float4(verts[vid].normal.xyz, 0.0f)).xyz;
    o.normalW = (u.model * float4(skinnedN, 0.0f)).xyz;
    o.color = verts[vid].color * u.tint;
    o.uv = verts[vid].uv.xy;
    o.lightClip = u.lightViewProj * worldPos;
    // The tangent is skinned by the SAME blended matrix as the normal. Skinning
    // them with different matrices, or skinning only one, tears the frame apart
    // wherever a joint bends -- and a torn frame is a seam of wrong lighting
    // exactly at the elbow, which is where the eye is looking.
    const float3 skinnedT = (blend * float4(verts[vid].tangent.xyz, 0.0f)).xyz;
    o.tangentW = float4((u.model * float4(skinnedT, 0.0f)).xyz,
                        verts[vid].tangent.w);
    return o;
}

fragment float4 fs_lit(VSOut in [[stage_in]],
                       constant FrameUniforms& u [[buffer(1)]],
                       device const GpuLight* lights [[buffer(2)]],
                       constant GpuCascades& cascades [[buffer(3)]],
                       texture2d<float> albedoMap    [[texture(0)]],
                       texture2d<float> roughnessMap [[texture(1)]],
                       depth2d<float>   shadowMap    [[texture(2)]],
                       depth2d<float>   shadowAtlas  [[texture(3)]],
                       texture2d<float> normalMap    [[texture(4)]],
                       // The environment probe, slots 5-7. Left unbound when
                       // there is none, which ShadeSurface detects.
                       texturecube<float> irradianceMap [[texture(5)]],
                       texturecube<float> specularMap   [[texture(6)]],
                       texture2d<float>   brdfLut       [[texture(7)]],
                       texture2d<float> metallicMap  [[texture(8)]],
                       texture2d<float> emissiveMap  [[texture(9)]],
                       texture2d<float> occlusionMap [[texture(10)]],
                       sampler          smp          [[sampler(0)]],
                       sampler          envSmp       [[sampler(1)]],
                       sampler          shadowSmp    [[sampler(2)]],
                       // CLUSTERS. All three may be unbound, and the shading
                       // function falls back to the whole light buffer.
                       constant GpuClusters* clusters       [[buffer(5)]],
                       device const uint*    clusterCounts  [[buffer(6)]],
                       device const uint*    clusterIndices [[buffer(7)]],
                       // The baked irradiance volume, slots 11-13.
                       texture3d<float> giR [[texture(11)]],
                       texture3d<float> giG [[texture(12)]],
                       texture3d<float> giB [[texture(13)]],
                       sampler          giSmp [[sampler(3)]])
{
    // LOD CROSSFADE. A 4x4 ordered dither: the fragment survives when the
    // fade exceeds this pixel's threshold.
    //
    // Ordered rather than random. A hash of the pixel coordinate gives a
    // pattern that is different every frame if anything about the hash input
    // moves, and the dissolve then boils; an ordered matrix is fixed to the
    // screen, so a half-faded object looks like a static screen door and the
    // two halves of a crossfade are exact complements pixel for pixel.
    if (u.lighting.z < 0.999f) {
        // Bayer 4x4, scaled to (0.5/16 .. 15.5/16) so that fade 0 discards
        // everything and fade 1 keeps everything -- a matrix running 0..15/16
        // would keep one pixel in sixteen at fade 0.
        constexpr float kBayer[16] = {
            0.5f / 16, 8.5f / 16, 2.5f / 16, 10.5f / 16,
            12.5f / 16, 4.5f / 16, 14.5f / 16, 6.5f / 16,
            3.5f / 16, 11.5f / 16, 1.5f / 16, 9.5f / 16,
            15.5f / 16, 7.5f / 16, 13.5f / 16, 5.5f / 16};
        const uint bx = uint(in.position.x) & 3u;
        const uint by = uint(in.position.y) & 3u;
        const float t = kBayer[by * 4u + bx];
        const float f = u.lighting.z;
        const float share = abs(f);
        // POSITIVE takes the low thresholds, NEGATIVE takes the high ones, and
        // the two are exact complements: "share >= t" and "t > 1 - share" cover
        // every pixel between them exactly once when the shares sum to one.
        // Two positive fades would instead be nested, drawing the overlap twice
        // and leaving the rest bare.
        const bool keep = f >= 0.0f ? (share >= t) : (t > 1.0f - share);
        if (!keep) discard_fragment();
    }

    // SECTION CUT before anything else: no point shading a fragment that is
    // about to be thrown away. This is what lets you look inside a building
    // without deleting its roof from the scene.
    if (in.worldPos.y > u.surface.w) discard_fragment();

    // Both maps default to 1x1 white, so an untextured material multiplies by
    // one and needs no branch here.
    const float3 albedo =
        in.color.rgb * u.baseColor.rgb * albedoMap.sample(smp, in.uv).rgb;
    const float roughness =
        saturate(u.surface.x * roughnessMap.sample(smp, in.uv).r);
    const float metallic = saturate(u.surface.y * metallicMap.sample(smp, in.uv).r);
    const float ao = occlusionMap.sample(smp, in.uv).r;

    const float3 N = ApplyNormalMap(normalize(in.normalW), in.tangentW, in.uv,
                                    normalMap, smp, u.emissive.w);

    // LINEAR HDR out, un-tone-mapped and un-gamma-corrected. The scene target
    // is half-float and the composite does both at the end.
    //
    // This is not a rearrangement for tidiness. Tone mapping here clamps every
    // surface to one before anything downstream sees it, so a lamp and a sheet
    // of white paper arrive at the bloom pass identical and it glows off the
    // paper. Brightness has to survive to the end of the frame to be usable.
    const float3 worldPos = in.worldPos;
    const float3 lit = ShadeSurface(worldPos, N, albedo, roughness,
                                    metallic, in.lightClip, u, lights, cascades,
                                    shadowMap, shadowAtlas, shadowSmp,
                                    irradianceMap, specularMap, brdfLut, envSmp,
                                    ao, clusters, clusterCounts, clusterIndices,
                                    in.position.xy,
                                    // Distance ALONG the view axis, which is
                                    // what the exponential slicing is in terms
                                    // of -- not distance from the eye, which
                                    // would make the cell boundaries spherical
                                    // and disagree with the binning pass.
                                    dot(worldPos - u.eyePos.xyz, u.viewDir.xyz),
                                    giR, giG, giB, giSmp);
    // EMISSION last and unlit. It is radiance the surface produces, so nothing
    // shadows it, no light affects it and ambient occlusion does not dim it --
    // a glowing sign in a dark alcove is exactly as bright as one in the open.
    const float3 emit = u.emissive.rgb * emissiveMap.sample(smp, in.uv).rgb;
    return float4(lit + emit, in.color.a);
}
