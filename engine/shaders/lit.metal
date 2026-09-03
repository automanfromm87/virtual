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

// The BRDF for ONE light direction, without the radiance or the cosine term —
// those belong to the light, and factoring them out is what lets the sun and a
// spot lamp share the same surface response instead of drifting apart.
static inline float3 Brdf(float3 N, float3 V, float3 L, float3 albedo,
                          float roughness, float metallic)
{
    const float3 H = normalize(L + V);
    const float NdotV = saturate(dot(N, V)) + 1e-5f;
    const float NdotL = saturate(dot(N, L));
    const float NdotH = saturate(dot(N, H));
    const float VdotH = saturate(dot(V, H));

    // Perceptual roughness squared. Artists author "roughness"; the BRDF wants
    // alpha, and the square is what makes the slider feel linear.
    const float a = max(roughness * roughness, 1e-3f);
    // Dielectrics reflect ~4% white; metals reflect their own colour and have
    // no diffuse at all.
    const float3 f0 = mix(float3(0.04f), albedo, metallic);
    const float3 F = F_Schlick(VdotH, f0);
    const float3 specular = F * (D_GGX(NdotH, a) * V_SmithGGX(NdotV, NdotL, a));
    // Energy conservation: whatever is reflected specularly cannot also be
    // diffused. Lambert's 1/pi is folded in.
    const float3 diffuse = (1.0f - F) * albedo * (1.0f - metallic) *
                           (1.0f / 3.14159265f);
    return diffuse + specular;
}

// Inverse-square falloff, windowed so it actually reaches zero at `range`.
//
// Pure 1/d² is the physical answer and never gets to zero, so a light would
// have to be evaluated for every fragment in the scene forever. The window is
// what makes a light have an extent you can cull against, and it is shaped to
// leave the near field — where the light is actually bright — untouched.
static inline float Falloff(float distance, float range)
{
    const float d2 = distance * distance;
    const float ratio = saturate(d2 / max(range * range, 1e-4f));
    const float window = saturate(1.0f - ratio * ratio);
    return window * window / max(d2, 1e-4f);
}

fragment float4 fs_lit(VSOut in [[stage_in]],
                       constant FrameUniforms& u [[buffer(1)]],
                       device const GpuLight* lights [[buffer(2)]],
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
    const float3 V = normalize(u.eyePos.xyz - in.worldPos);

    // --- the key light: directional, and the only one with a shadow map ------
    const float3 Lsun = normalize(u.lightDir.xyz);
    const float shadow = ShadowFactor(in.lightClip, shadowMap, smp, u.surface.z);
    float3 direct = Brdf(N, V, Lsun, albedo, roughness, metallic) *
                    u.lightColor.rgb * saturate(dot(N, Lsun)) * shadow;

    // --- local lights --------------------------------------------------------
    const uint light_count = min(uint(u.lighting.x), uint(ENG_MAX_LIGHTS));
    for (uint i = 0; i < light_count; ++i) {
        const GpuLight lt = lights[i];
        const float3 to_light = lt.position.xyz - in.worldPos;
        const float dist = length(to_light);
        if (dist > lt.direction.w) continue;  // outside its range entirely
        const float3 L = to_light / max(dist, 1e-4f);

        float attenuation = Falloff(dist, lt.direction.w);
        if (lt.position.w > 0.5f) {
            // Spot. The cone is measured against the direction the lamp SHINES,
            // so the fragment's bearing from the lamp is -L.
            const float cosine = dot(lt.direction.xyz, -L);
            // Smoothed between the two cosines rather than a hard cut, or the
            // cone edge is a jagged line that crawls as the light moves.
            const float t = saturate((cosine - lt.cone.y) /
                                     max(lt.cone.x - lt.cone.y, 1e-4f));
            attenuation *= t * t;
        }
        const float ndotl = saturate(dot(N, L));
        if (attenuation * ndotl <= 0.0f) continue;
        direct += Brdf(N, V, L, albedo, roughness, metallic) * lt.color.rgb *
                  ndotl * attenuation;
    }

    // AMBIENT, split the same way the direct term is: a diffuse part and a
    // specular part.
    //
    // Multiplying one hemisphere colour by the albedo was wrong twice over. A
    // metal has NO diffuse lobe, so that term should vanish for it — and then a
    // gold sphere in a dark room renders pure black, which is physically
    // correct and completely useless, because what a metal actually shows is
    // the room reflected in it. The specular half below is that reflection: the
    // same two-colour hemisphere, sampled in the mirror direction instead of
    // along the normal.
    //
    // This is image-based lighting with an environment of exactly two colours.
    // A real IBL swaps the hemisphere for a prefiltered probe and gains actual
    // reflections; the shape of the calculation does not change.
    const float3 f0amb = mix(float3(0.04f), albedo, metallic);
    const float hemi = N.y * 0.5f + 0.5f;
    const float3 skyAtN = mix(u.ambientGround.rgb, u.ambientSky.rgb, hemi);

    const float3 R = reflect(-V, N);
    const float hemiR = R.y * 0.5f + 0.5f;
    const float3 skyAtR = mix(u.ambientGround.rgb, u.ambientSky.rgb, hemiR);

    // Fresnel with a roughness-aware ceiling. Plain Schlick goes to white at
    // grazing angles even for a rough surface, which puts a bright rim on
    // everything; clamping it to (1 - roughness) is the standard cheap fix.
    const float NdotVamb = saturate(dot(N, V));
    const float3 Famb =
        f0amb + (max(float3(1.0f - roughness), f0amb) - f0amb) *
                    pow(1.0f - NdotVamb, 5.0f);

    const float3 ambient =
        albedo * (1.0f - metallic) * skyAtN * (1.0f - Famb) + skyAtR * Famb;

    // ACES filmic, then gamma.
    //
    // Reinhard was here first and it is why every earlier picture looked hazy:
    // x/(1+x) starts compressing at zero, so it lifts the blacks and pulls all
    // three channels toward each other, which desaturates exactly the coloured
    // lights that were worth having. This curve leaves the toe alone and rolls
    // the highlights off instead.
    float3 color = direct + ambient;
    const float a1 = 2.51f, b1 = 0.03f, c1 = 2.43f, d1 = 0.59f, e1 = 0.14f;
    color = saturate((color * (a1 * color + b1)) / (color * (c1 * color + d1) + e1));
    color = pow(color, 1.0f / 2.2f);

    return float4(color, in.color.a);
}
