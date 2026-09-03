// No #include here: shader_types.h is textually prepended by the loader.
//
// FROXEL VOLUMETRIC LIGHTING. A 3D grid laid over the view frustum -- "frustum
// voxels" -- holding how much light is scattered toward the camera at every
// point in the air, and how much of the air's extinction lies in front of it.
//
// WHAT THE ANALYTIC HEIGHT FOG CANNOT DO. That integral assumes the air is
// uniformly lit, so it can only ever tint the distance a constant colour. It
// cannot make a shaft of light through a window, cannot put a lamp's glow in
// the air around it, and cannot darken the fog inside a shadow -- and those
// three are the whole reason to have fog that is not a colour ramp.
//
// THREE PASSES, and the split is what makes it affordable:
//
//   1. SCATTER, one thread per froxel. Works out how much light reaches that
//      point in the air -- sun through the cascades, plus every local light --
//      and how much of it heads toward the camera, which is the phase function.
//   2. INTEGRATE, one thread per froxel COLUMN. Walks the column front to back
//      accumulating in-scattered light and transmittance. This has to be a
//      separate pass because it is a prefix sum: froxel k needs the answer for
//      every froxel in front of it.
//   3. APPLY, per screen pixel, sampling the integrated volume at the
//      fragment's own depth.
//
// The grid is small -- a fraction of the screen's resolution -- and that is the
// point. Ray-marching per pixel costs (pixels x steps) shadow lookups; this
// costs (froxels) of them, once, and the result is filtered for free by the
// trilinear sample in pass 3.

// Froxel index -> the distance along the view axis of that slice's near face.
//
// EXPONENTIAL, like the light clustering's, and for a stronger reason here:
// scattering is integrated along the ray, so the slices nearest the camera
// contribute the most detail per metre and need to be the thinnest.
static inline float FroxelDepth(float slice, constant GpuVolumetrics& v)
{
    return v.range.x * exp(v.range.z * slice / v.grid.z);
}

// Henyey-Greenstein. `g` above zero throws light forward, which is what real
// air and real fog do -- it is why a shaft is bright when you look toward the
// lamp and faint when you look away from it. Isotropic scattering (g = 0)
// gives fog with no directionality at all, which reads as a grey wash.
// `cos_theta` is between the light's DIRECTION OF TRAVEL and the direction it
// scatters into -- not between the direction toward the light and the eye,
// which is its negative. Passing the latter turns a forward-scattering medium
// into a backward-scattering one: the fog is then dimmest looking into the sun
// and brightest looking away from it, which is exactly backwards and measures
// as 72.9 against 243.8.
static inline float PhaseHg(float cos_theta, float g)
{
    const float g2 = g * g;
    const float d = 1.0f + g2 - 2.0f * g * cos_theta;
    return (1.0f - g2) / (4.0f * 3.14159265f * max(d * sqrt(max(d, 1e-4f)), 1e-4f));
}

kernel void cs_volumetric_scatter(
    uint3 gid [[thread_position_in_grid]],
    constant GpuVolumetrics& v [[buffer(0)]],
    constant FrameUniforms& u [[buffer(1)]],
    device const GpuLight* lights [[buffer(2)]],
    constant GpuCascades& cascades [[buffer(3)]],
    depth2d<float> shadowMap [[texture(0)]],
    depth2d<float> shadowAtlas [[texture(1)]],
    sampler shadowSmp [[sampler(0)]],
    texture3d<float, access::write> outScatter [[texture(2)]])
{
    const uint nx = uint(v.grid.x), ny = uint(v.grid.y), nz = uint(v.grid.z);
    if (gid.x >= nx || gid.y >= ny || gid.z >= nz) return;

    // JITTER along the slice, one value per froxel, fixed per frame. Sampling
    // every slice at its exact centre puts a visible banding pattern across
    // every shaft -- the slices are thick, and a hard edge between two of them
    // reads as a stripe. Offsetting each one turns the banding into noise that
    // the trilinear filter in the apply pass smears out.
    const float jitter = v.range.w;
    const float z0 = FroxelDepth(float(gid.z) + jitter, v);
    const float z1 = FroxelDepth(float(gid.z) + 1.0f + jitter, v);
    const float z = (z0 + z1) * 0.5f;
    const float thickness = max(z1 - z0, 1e-4f);

    // The froxel's centre in world space, from its screen tile and its depth.
    const float2 uv = (float2(gid.xy) + 0.5f) / float2(float(nx), float(ny));
    float2 ndc = uv * 2.0f - 1.0f;
    ndc.y = -ndc.y;
    // Unprojecting at a KNOWN view depth rather than through the inverse
    // projection: the view ray through this pixel, scaled to reach z.
    const float f = u.ssao.y;      // 1/tan(fovY/2)
    const float aspect = u.ssao.z;
    const float3 right = normalize(cross(u.viewDir.xyz, float3(0.0f, 1.0f, 0.0f)));
    const float3 up = cross(right, u.viewDir.xyz);
    const float3 world = u.eyePos.xyz + u.viewDir.xyz * z +
                         right * (ndc.x * aspect / f * z) + up * (ndc.y / f * z);

    // DENSITY: a constant plus an exponential falling off with height, which is
    // what an atmosphere does. Both are in the caller's units per metre.
    const float density =
        v.medium.x + v.medium.y * exp(-max(world.y - v.medium.z, 0.0f) * v.medium.w);
    const float3 to_eye = normalize(u.eyePos.xyz - world);

    float3 scattered = float3(0.0f);

    // THE SUN, through the same cascades the surfaces use. Sharing them is what
    // makes a shaft line up with the shadow that casts it -- a separate shadow
    // lookup here, at a different resolution or bias, puts the edge of the
    // shaft somewhere the geometry's shadow is not.
    // The sun ALWAYS scatters; the shadow map only decides how much of it
    // reaches this froxel. Gating the whole contribution on shadows being
    // enabled -- which is what this did first -- means a scene with fog, a sun
    // and no shadow map has completely dark air, and the sun is the brightest
    // thing in most scenes.
    {
        float lit = 1.0f;
        if (u.surface.z > 0.5f) {
        const int count = int(cascades.info.x);
        int c = count - 1;
        for (int i = 0; i < count; ++i)
            if (z < cascades.splits[i]) { c = i; break; }
        const float4 lc = cascades.viewProj[c] * float4(world, 1.0f);
        if (lc.w > 0.0f) {
            float3 sn = lc.xyz / lc.w;
            float2 suv = float2(sn.x * 0.5f + 0.5f, 0.5f - sn.y * 0.5f);
            if (count > 1) {
                const float per_side = cascades.info.y;
                const float tile = cascades.info.z;
                suv = suv * tile + float2(float(c % int(per_side)) * tile,
                                          float(c / int(per_side)) * tile);
            }
            if (all(suv > 0.0f) && all(suv < 1.0f))
                lit = sn.z >= shadowMap.sample(shadowSmp, suv) - 1.5e-3f ? 1.0f : 0.0f;
        }
        }
        // lightDir points TOWARD the sun, so the photons travel along its
        // negative, and that is what the phase angle is measured from.
        const float phase =
            PhaseHg(dot(to_eye, -normalize(u.lightDir.xyz)), v.medium2.x);
        scattered += u.lightColor.rgb * lit * phase;
    }

    // LOCAL LIGHTS. Every one, not the clustered subset: the froxel grid is its
    // own spatial structure and reusing the screen-space clusters would need
    // this pass to agree with their depth slicing, which is a different
    // exponential over a different range.
    const uint light_count = min(uint(u.lighting.x), uint(ENG_MAX_LIGHTS));
    for (uint i = 0; i < light_count; ++i) {
        const GpuLight lt = lights[i];
        const float3 to_light = lt.position.xyz - world;
        const float dist = length(to_light);
        if (dist > lt.direction.w) continue;
        const float3 L = to_light / max(dist, 1e-4f);
        float atten = saturate(1.0f - dist / lt.direction.w);
        atten = atten * atten / max(dist * dist, 0.05f);
        if (lt.position.w > 0.5f) {
            const float cosine = dot(lt.direction.xyz, -L);
            const float t = saturate((cosine - lt.cone.y) /
                                     max(lt.cone.x - lt.cone.y, 1e-4f));
            atten *= t * t;
        }
        if (atten <= 0.0f) continue;
        // L points from the froxel toward the lamp, so the photons arrive
        // along -L. Same convention as the sun above.
        scattered += lt.color.rgb * atten * PhaseHg(dot(to_eye, -L), v.medium2.x);
    }

    // scattering coefficient x density, times the light that arrived. The
    // ambient term stands in for light that has already bounced -- without it
    // the inside of a shadow is perfectly black fog, which no real medium is.
    const float3 albedo_scatter = v.scatter.rgb * density;
    const float3 in_scatter = (scattered + v.ambient.rgb) * albedo_scatter;
    // Extinction in .a: how much this froxel's thickness absorbs. Stored rather
    // than the transmittance, because the integration pass needs to sum it.
    const float extinction = density * v.scatter.w * thickness;
    outScatter.write(float4(in_scatter * thickness, extinction),
                     uint3(gid.x, gid.y, gid.z));
}

// The prefix sum along each column. One thread per (x, y); it walks z.
kernel void cs_volumetric_integrate(
    uint2 gid [[thread_position_in_grid]],
    constant GpuVolumetrics& v [[buffer(0)]],
    texture3d<float, access::read> inScatter [[texture(0)]],
    texture3d<float, access::write> outVolume [[texture(1)]])
{
    const uint nx = uint(v.grid.x), ny = uint(v.grid.y), nz = uint(v.grid.z);
    if (gid.x >= nx || gid.y >= ny) return;

    float3 accumulated = float3(0.0f);
    float transmittance = 1.0f;
    for (uint k = 0; k < nz; ++k) {
        const float4 s = inScatter.read(uint3(gid.x, gid.y, k));
        // ANALYTIC integration across the slice rather than a rectangle rule.
        // The light scattered inside a slice is itself attenuated by the part
        // of the slice in front of it, and (1 - e^-sigma)/sigma is that
        // integral exactly. A plain multiply overestimates thick slices, and
        // the froxels near the far plane are very thick.
        const float sigma = max(s.a, 1e-6f);
        const float slice_t = exp(-sigma);
        accumulated += transmittance * s.rgb * (1.0f - slice_t) / sigma;
        transmittance *= slice_t;
        // rgb: light scattered toward the eye from here to the camera.
        // a: how much of what is BEHIND this froxel still gets through.
        outVolume.write(float4(accumulated, transmittance),
                        uint3(gid.x, gid.y, k));
    }
}

struct VolumeApplyOut {
    float4 position [[position]];
    float2 uv;
};

vertex VolumeApplyOut vs_volumetric_apply(uint vid [[vertex_id]])
{
    const float2 corner = float2((vid << 1) & 2, vid & 2);
    VolumeApplyOut o;
    o.position = float4(corner * 2.0f - 1.0f, 0.0f, 1.0f);
    o.uv = float2(corner.x, 1.0f - corner.y);
    return o;
}

fragment float4 fs_volumetric_apply(VolumeApplyOut in [[stage_in]],
                                    constant GpuVolumetrics& v [[buffer(0)]],
                                    constant FrameUniforms& u [[buffer(1)]],
                                    depth2d<float> sceneDepth [[texture(0)]],
                                    texture3d<float> volume [[texture(1)]],
                                    sampler smp [[sampler(0)]])
{
    const float depth = sceneDepth.sample(smp, in.uv);
    // Reversed-Z: 0 is the far plane. A pixel with nothing drawn on it gets the
    // whole depth of the volume, which is what puts fog in the sky.
    const float view_z = depth > 1e-6f ? u.ssao.x / depth : v.range.y;

    // Distance -> texture coordinate, the inverse of FroxelDepth.
    //
    // log(z/near) / log(far/near) is ALREADY normalised to 0..1 -- FroxelDepth
    // takes a froxel index and divides by the count itself, so dividing again
    // here samples the first slice at every depth and the fog is uniformly the
    // little that scattered in the nearest few centimetres.
    //
    // Then a half-texel inset, without which the outermost slices blend with
    // nothing and the fog fades out at the very front and very back.
    const float slice = log(max(view_z, v.range.x) / v.range.x) / v.range.z;
    const float w = clamp(slice, 0.5f / v.grid.z, 1.0f - 0.5f / v.grid.z);
    const float4 s = volume.sample(smp, float3(in.uv, w));

    // Premultiplied: the caller blends with (ONE, ONE_MINUS_SRC_ALPHA), so the
    // alpha carries "how much of the scene this fog hides" and the rgb carries
    // what it adds. One blend does both.
    return float4(s.rgb, 1.0f - s.a);
}
