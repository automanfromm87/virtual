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

// --- microfacet terms --------------------------------------------------------

// GGX / Trowbridge-Reitz. Describes what fraction of microfacets are oriented
// along the half vector: the shape of the highlight.
static inline float D_GGX(float NdotH, float a)
{
    const float a2 = a * a;
    const float d = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    return a2 / max(3.14159265f * d * d, 1e-7f);
}

// Smith height-correlated visibility, already divided by the 4*NdotL*NdotV
// denominator of the BRDF. Accounts for microfacets shadowing each other.
static inline float V_SmithGGX(float NdotV, float NdotL, float a)
{
    const float a2 = a * a;
    const float v = NdotL * sqrt(NdotV * NdotV * (1.0f - a2) + a2);
    const float l = NdotV * sqrt(NdotL * NdotL * (1.0f - a2) + a2);
    return 0.5f / max(v + l, 1e-7f);
}

// Schlick: reflectance climbs to 1 at grazing angles. This is why every
// material has a bright rim, and why leaving it out makes everything look like
// plastic lit from the front.
static inline float3 F_Schlick(float VdotH, float3 f0)
{
    const float f = pow(saturate(1.0f - VdotH), 5.0f);
    return f0 + (1.0f - f0) * f;
}

// Returns 1 where the surface is lit, 0 where the shadow map says something
// else was closer to the light.
static inline float ShadowFactor(float4 lightClip, depth2d<float> shadowMap,
                                 sampler smp, float enabled)
{
    if (enabled < 0.5f) return 1.0f;

    // Orthographic, so w is 1 and this divide is a formality — but doing it
    // keeps the code correct if the light ever becomes a spot.
    const float3 ndc = lightClip.xyz / lightClip.w;
    // Clip x/y are [-1,1]; texture uv is [0,1] with v flipped, same as the
    // fullscreen pass.
    float2 uv = ndc.xy * 0.5f + 0.5f;
    uv.y = 1.0f - uv.y;

    // Outside the shadow map's footprint: light it rather than shadowing it.
    // The opposite choice puts a hard black box around the lit region.
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) return 1.0f;

    // REVERSED-Z: nearer to the light means a GREATER depth value, so the
    // fragment is lit when its own depth is at least what was recorded. The
    // bias is subtracted for the same reason it would be added under
    // conventional Z — get this sign wrong and every surface shadows itself
    // into a mess of acne stripes.
    const float bias = 0.0015f;
    const float texel = 1.0f / float(shadowMap.get_width());

    // CONTACT HARDENING. A fixed-radius blur makes the shadow equally soft
    // where the object touches the ground and where it is a metre away — which
    // is the single most effective way to make a solid object look like it is
    // hovering. Real penumbra width grows with the gap between occluder and
    // receiver, so measure that gap first.
    //
    // Step 1: find how far in front of this fragment the blockers are. The
    // projection is orthographic and reversed-Z, so clip z is linear and a
    // plain difference is proportional to world distance along the light.
    float blockerSum = 0.0f;
    float blockerCount = 0.0f;
    for (int i = 0; i < 8; ++i) {
        // A fixed ring rather than a random disc: no noise to denoise later.
        const float ang = float(i) * 0.7853981f;  // 45 degrees apart
        const float2 o = float2(cos(ang), sin(ang)) * texel * 3.0f;
        const float d = shadowMap.sample(smp, uv + o);
        if (d > ndc.z + bias) {  // reversed-Z: greater means nearer the light
            blockerSum += d;
            blockerCount += 1.0f;
        }
    }
    if (blockerCount < 0.5f) return 1.0f;  // nothing in front: fully lit

    // Step 2: turn the gap into a filter radius. Clamped at the bottom so a
    // contact point still gets a little softening rather than a jagged edge,
    // and at the top so a distant caster does not smear across the map.
    const float gap = (blockerSum / blockerCount) - ndc.z;
    const float radius = clamp(gap * 45.0f, 0.6f, 6.0f) * texel;

    // Step 3: 3x3 PCF at that radius.
    float lit = 0.0f;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const float2 o = float2(float(dx), float(dy)) * radius;
            lit += (ndc.z >= shadowMap.sample(smp, uv + o) - bias) ? 1.0f : 0.0f;
        }
    return lit * (1.0f / 9.0f);
}

fragment float4 fs_lit(VSOut in [[stage_in]],
                       constant FrameUniforms& u [[buffer(1)]],
                       texture2d<float> albedoMap    [[texture(0)]],
                       texture2d<float> roughnessMap [[texture(1)]],
                       depth2d<float>   shadowMap    [[texture(2)]],
                       sampler          smp          [[sampler(0)]])
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

    // Renormalize per fragment: interpolating unit vectors across a triangle
    // does not preserve length, and the error is worst mid-face.
    const float3 N = normalize(in.normalW);
    const float3 L = normalize(u.lightDir.xyz);
    const float3 V = normalize(u.eyePos.xyz - in.worldPos);
    const float3 H = normalize(L + V);

    const float NdotL = saturate(dot(N, L));
    const float NdotV = saturate(dot(N, V)) + 1e-5f;
    const float NdotH = saturate(dot(N, H));
    const float VdotH = saturate(dot(V, H));

    // Perceptual roughness squared. Artists author "roughness"; the BRDF wants
    // alpha, and the square is what makes the slider feel linear.
    const float a = max(roughness * roughness, 1e-3f);

    // Dielectrics reflect ~4% white; metals reflect their own colour and have
    // no diffuse at all.
    const float3 f0 = mix(float3(0.04f), albedo, metallic);
    const float3 diffuseColor = albedo * (1.0f - metallic);

    const float3 F = F_Schlick(VdotH, f0);
    const float3 specular = F * (D_GGX(NdotH, a) * V_SmithGGX(NdotV, NdotL, a));

    // Energy conservation: whatever is reflected specularly cannot also be
    // diffused. Lambert's 1/pi is folded in.
    const float3 kD = (1.0f - F);
    const float3 diffuse = kD * diffuseColor * (1.0f / 3.14159265f);

    const float shadow = ShadowFactor(in.lightClip, shadowMap, smp, u.surface.z);
    const float3 direct =
        (diffuse + specular) * u.lightColor.rgb * NdotL * shadow;

    // HEMISPHERE AMBIENT: cool light from the sky above, dim warm bounce from
    // the ground below, blended by which way the normal points.
    //
    // This replaces a single constant, and the difference is not subtle. A
    // constant ambient lights the underside of an object exactly as brightly as
    // its top, which erases the one cue that says "this thing is sitting on
    // something". It is the cheapest useful approximation of image-based
    // lighting — a real IBL swaps these two colours for a prefiltered
    // environment probe and gains reflections as well.
    const float3 kSkyColor = float3(0.13f, 0.16f, 0.24f);
    const float3 kGroundColor = float3(0.055f, 0.045f, 0.035f);
    const float hemi = N.y * 0.5f + 0.5f;
    const float3 ambient = albedo * mix(kGroundColor, kSkyColor, hemi);

    // Reinhard tone map then gamma. Without these, a physically-scaled light
    // blows out to white and the whole point of the BRDF is invisible.
    float3 color = direct + ambient;
    color = color / (1.0f + color);
    color = pow(color, 1.0f / 2.2f);

    return float4(color, in.color.a);
}
