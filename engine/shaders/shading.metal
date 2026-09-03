// No #include here: shader_types.h is textually prepended by the loader, and so
// is this file, ahead of every shader that shades a surface.
//
// WHY it is a separate file: the forward pass and the deferred pass must
// produce the SAME picture, and the only way to be sure of that is for them to
// run the same code. Two copies of a BRDF drift within a week -- someone fixes
// a Fresnel clamp in one of them -- and the difference shows up as a scene that
// looks subtly different depending on a setting nobody thinks affects looks.
//
// Cook-Torrance microfacet BRDF: GGX normal distribution, Smith height-
// correlated geometry, Schlick Fresnel.
//
// The metallic workflow, which is the part people get wrong: a metal has NO
// diffuse lobe, and its specular colour IS its base colour. A dielectric has a
// diffuse lobe and a colourless 4% specular. `metallic` blends between those
// two physical facts -- it is not a slider on a look.

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

// The uv rectangle of one tile of the atlas.
static inline float4 AtlasTile(float index, float per_side)
{
    const float span = 1.0f / per_side;
    const float i = floor(index + 0.5f);
    return float4(fmod(i, per_side) * span, floor(i / per_side) * span, span, 0.0f);
}

// A depth comparison against one tile, with 3x3 filtering.
//
// Clamped a texel inside the tile: a bilinear tap at the very edge reaches into
// the neighbouring light's map and draws a stripe of someone else's shadow
// along the border.
static inline float SampleTile(depth2d<float> atlas, sampler smp, float2 uv,
                               float4 tile, float depth, float bias)
{
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) return 1.0f;
    const float texel = 1.0f / float(atlas.get_width());
    const float2 lo = tile.xy + texel;
    const float2 hi = tile.xy + tile.z - texel;
    float lit = 0.0f;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            const float2 at = clamp(uv * tile.z + tile.xy +
                                        float2(float(dx), float(dy)) * texel,
                                    lo, hi);
            // Reversed-Z: nearer the light is a GREATER depth, so lit means "at
            // least as near as whatever was recorded".
            lit += (depth >= atlas.sample(smp, at) - bias) ? 1.0f : 0.0f;
        }
    return lit * (1.0f / 9.0f);
}

// A spot's lookup: one tile, through its own projection.
//
// Perspective, unlike the directional light's, so the divide by w is real work
// rather than a formality — a spot's rays diverge and depth is not linear in
// clip space.
static inline float SpotShadow(float4 clip, float4 place, depth2d<float> atlas,
                               sampler smp)
{
    if (clip.w <= 0.0f) return 1.0f;   // behind the lamp
    const float3 ndc = clip.xyz / clip.w;
    if (ndc.z <= 0.0f) return 1.0f;    // past the far plane
    float2 uv = ndc.xy * 0.5f + 0.5f;
    uv.y = 1.0f - uv.y;
    // The bias scales with depth: a perspective map's precision falls off with
    // distance in a way an orthographic one's does not, so a constant is either
    // useless near the lamp or lets the far end of the cone shadow itself.
    const float bias = 2.5e-4f + 4.0e-3f * ndc.z;
    return SampleTile(atlas, smp, uv, AtlasTile(place.x, place.y), ndc.z, bias);
}

// A point light's lookup: SIX tiles, and the direction chooses which.
//
// No matrix. The six faces are the axis directions with ninety-degree frusta,
// so the face is whichever component of the direction is largest and the uv is
// the other two divided by it — which is exactly what a cube map lookup does in
// hardware. Storing six matrices per light would work and would cost 384 bytes
// each; this costs a compare and a divide.
static inline float PointShadow(float3 to_frag, float4 place,
                                depth2d<float> atlas, sampler smp)
{
    const float3 a = abs(to_frag);
    const float major = max(max(a.x, a.y), a.z);
    if (major <= 1e-5f) return 1.0f;

    // Face order +X -X +Y -Y +Z -Z, and the `up` vectors below must match the
    // ones Light::CubeFaceViewProj rendered with — disagree and the lookup
    // samples a rotated copy of the right face, which reads as a shadow that
    // slides as the light turns.
    int face;
    float2 st;
    if (a.x >= a.y && a.x >= a.z) {
        face = to_frag.x > 0.0f ? 0 : 1;
        st = to_frag.x > 0.0f ? float2(-to_frag.z, -to_frag.y)
                              : float2(to_frag.z, -to_frag.y);
    } else if (a.y >= a.z) {
        face = to_frag.y > 0.0f ? 2 : 3;
        st = to_frag.y > 0.0f ? float2(to_frag.x, to_frag.z)
                              : float2(to_frag.x, -to_frag.z);
    } else {
        face = to_frag.z > 0.0f ? 4 : 5;
        st = to_frag.z > 0.0f ? float2(to_frag.x, -to_frag.y)
                              : float2(-to_frag.x, -to_frag.y);
    }
    float2 uv = st / major * 0.5f + 0.5f;
    uv.y = 1.0f - uv.y;

    // Reversed-Z with an infinite far plane: z = near / distance along the
    // face's forward axis, which for the chosen face is exactly `major`.
    const float depth = place.z / major;
    const float bias = 3.0e-4f + 6.0e-3f * depth;
    return SampleTile(atlas, smp, uv, AtlasTile(place.x + float(face), place.y),
                      depth, bias);
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

// EVERY light, applied to one surface point. The whole of the lighting model
// lives here and nowhere else.
//
// `lightClip` is only used by the single-cascade shadow path, where the vertex
// shader has already done the transform; the cascaded path and the deferred
// pass both recompute it from the world position, because a fullscreen pass has
// no vertex to have done it in.
static inline float3 ShadeSurface(float3 worldPos, float3 Nin, float3 albedo,
                                  float roughness, float metallic,
                                  float4 lightClip,
                                  constant FrameUniforms& u,
                                  device const GpuLight* lights,
                                  constant GpuCascades& cascades,
                                  depth2d<float> shadowMap,
                                  depth2d<float> shadowAtlas,
                                  sampler smp)
{
    // Renormalize per fragment: interpolating unit vectors across a triangle
    // does not preserve length, and the error is worst mid-face.
    const float3 N = normalize(Nin);
    const float3 V = normalize(u.eyePos.xyz - worldPos);

    // --- the key light: directional, and the only one with cascades ----------
    const float3 Lsun = normalize(u.lightDir.xyz);
    float shadow = 1.0f;
    if (u.surface.z > 0.5f) {
        const int count = int(cascades.info.x);
        if (count <= 1) {
            shadow = ShadowFactor(lightClip, shadowMap, smp, u.surface.z);
        } else {
            // Pick by distance from the eye. The FIRST cascade the fragment
            // fits inside — they are nested, so the nearest one that contains
            // it is also the highest resolution one that does.
            const float view_depth = length(worldPos - u.eyePos.xyz);
            int c = count - 1;
            for (int i = 0; i < count; ++i) {
                if (view_depth < cascades.splits[i]) { c = i; break; }
            }
            const float per_side = cascades.info.y;
            const float4 tile = AtlasTile(float(c), per_side);
            const float4 clip = cascades.viewProj[c] * float4(worldPos, 1.0f);
            const float3 ndc = clip.xyz / clip.w;
            float2 uv = ndc.xy * 0.5f + 0.5f;
            uv.y = 1.0f - uv.y;
            // Orthographic, so depth is linear and a constant bias is right —
            // unlike the perspective maps a spot or a point light uses.
            shadow = SampleTile(shadowMap, smp, uv, tile, ndc.z, 1.2e-3f);
        }
    }
    float3 direct = Brdf(N, V, Lsun, albedo, roughness, metallic) *
                    u.lightColor.rgb * saturate(dot(N, Lsun)) * shadow;

    // --- local lights --------------------------------------------------------
    const uint light_count = min(uint(u.lighting.x), uint(ENG_MAX_LIGHTS));
    for (uint i = 0; i < light_count; ++i) {
        const GpuLight lt = lights[i];
        const float3 to_light = lt.position.xyz - worldPos;
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

        if (lt.shadow.w > 1.5f) {
            attenuation *= PointShadow(worldPos - lt.position.xyz, lt.shadow,
                                       shadowAtlas, smp);
            if (attenuation <= 0.0f) continue;
        } else if (lt.shadow.w > 0.5f) {
            attenuation *= SpotShadow(lt.viewProj * float4(worldPos, 1.0f),
                                      lt.shadow, shadowAtlas, smp);
            if (attenuation <= 0.0f) continue;
        }
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

    // LINEAR HDR out, un-tone-mapped and un-gamma-corrected. The scene target
    // is half-float and the composite does both at the end.
    //
    // This is not a rearrangement for tidiness. Tone mapping here clamps every
    // surface to one before anything downstream sees it, so a lamp and a sheet
    // of white paper arrive at the bloom pass identical and it glows off the
    // paper. Brightness has to survive to the end of the frame to be usable.
    return direct + ambient;
}
