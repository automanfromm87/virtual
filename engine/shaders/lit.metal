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
};

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
                       // The environment probe, slots 5-7. Left unbound when
                       // there is none, which ShadeSurface detects.
                       texturecube<float> irradianceMap [[texture(5)]],
                       texturecube<float> specularMap   [[texture(6)]],
                       texture2d<float>   brdfLut       [[texture(7)]],
                       sampler          smp          [[sampler(0)]],
                       sampler          envSmp       [[sampler(1)]])
{
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
    const float metallic = saturate(u.surface.y);

    // LINEAR HDR out, un-tone-mapped and un-gamma-corrected. The scene target
    // is half-float and the composite does both at the end.
    //
    // This is not a rearrangement for tidiness. Tone mapping here clamps every
    // surface to one before anything downstream sees it, so a lamp and a sheet
    // of white paper arrive at the bloom pass identical and it glows off the
    // paper. Brightness has to survive to the end of the frame to be usable.
    const float3 lit = ShadeSurface(in.worldPos, in.normalW, albedo, roughness,
                                    metallic, in.lightClip, u, lights, cascades,
                                    shadowMap, shadowAtlas, smp,
                                    irradianceMap, specularMap, brdfLut, envSmp);
    return float4(lit, in.color.a);
}
