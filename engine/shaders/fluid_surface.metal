// A continuous water surface out of SPH particles, in screen space.
//
// The billboard droplets in fluid.metal are a debug visualisation: discrete
// balls with visible gaps, matte shading, no refraction. Real water reads as
// a SURFACE, so this pass builds one -- without any mesh of the water, which
// would need topology the particles do not have.
//
// Three stages, all standard screen-space fluid rendering:
//
//   1. DEPTH. The particles drawn as instanced spheres into a depth buffer.
//      Real sphere geometry, not billboards: Metal cannot write depth from a
//      fragment shader, so a billboard can never record the hemisphere's true
//      depth, and without true depth there is nothing to smooth. This buffer
//      holds water ONLY, in its own target -- sharing the tank's would bake
//      the walls into the surface (tried: the whole tank shaded as water).
//      Tank occlusion is resolved in the shade pass, which reads both depths
//      under one uv convention instead of comparing them across passes.
//   2. SMOOTH. A bilateral filter turns the bumpy union of spheres into one
//      sheet. Bilateral, not Gaussian: a plain blur averages the water with
//      the background at every silhouette and rims the fluid with a halo of
//      half-depth soup. The edge stop keeps taps that disagree.
//   3. SHADE. Normals from screen-space derivatives of the smoothed depth,
//      then water optics: fresnel reflection of an analytic sky, refraction
//      of the background behind, and a sun glint. Opaque output -- the
//      refraction already contains the background, so there is nothing to
//      blend.
//
// What this is NOT: no thickness pass, so absorption is a constant rather
// than Beer-Lambert through the volume; no foam; the sky reflection is
// analytic rather than the IBL probe. Each of those is a known next step,
// not a gap in this one.

// GpuSurfaceParams lives in shader_types.h, shared with the C++ side like
// every other GPU struct in this engine.

// No fragment function at all (the pipeline takes an empty entry point):
// depth comes from the sphere geometry, and a void fragment would only add
// a stage that has nothing to say.

struct SurfaceOut {
    float4 position [[position]];
    float3 world;
};

vertex SurfaceOut vs_surface_depth(uint vid [[vertex_id]],
                                   uint iid [[instance_id]],
                                   device const VertexIn* sphere [[buffer(0)]],
                                   device const GpuFluidParticle* particles [[buffer(1)]],
                                   constant FrameUniforms& u [[buffer(2)]],
                                   constant GpuSurfaceParams& s [[buffer(3)]])
{
    const float3 centre = particles[iid].position.xyz;
    const float3 world =
        centre + sphere[vid].position.xyz * s.misc.x;
    SurfaceOut o;
    o.position = u.viewProj * float4(world, 1.0f);
    o.world = world;
    return o;
}

// No fragment function: this pass records depth only, like the shadow pass.
// Tank occlusion needs no shader code at all -- the tank draws first into the
// SAME depth buffer in the same pass, so the hardware depth test culls every
// sphere fragment behind a wall. A sampled-depth compare in a fragment stage
// would work too, and would be strictly worse: an extra texture, an extra
// convention to get upside down (see the y-flip in vs_surface_full below),
// and a manual bias to tune, all to reproduce what the depth unit does.

// ---------------------------------------------------------------- smooth --

struct SurfaceFullOut {
    float4 position [[position]];
    float2 uv;
};

vertex SurfaceFullOut vs_surface_full(uint vid [[vertex_id]])
{
    // The oversized-triangle fullscreen pass, same as the composite's: three
    // vertices, no diagonal seam, y flipped so texture v=0 is the top row.
    const float2 corners[3] = {float2(-1.0f, -1.0f), float2(3.0f, -1.0f),
                               float2(-1.0f, 3.0f)};
    SurfaceFullOut o;
    o.position = float4(corners[vid], 0.0f, 1.0f);
    o.uv = corners[vid] * 0.5f + 0.5f;
    o.uv.y = 1.0f - o.uv.y;
    return o;
}

// One axis of a separable bilateral smooth over the raw depth, writing LINEAR
// view distance in metres. The full 2D kernel is two of these passes; one
// 25x25 fragment pass would be 625 taps per pixel where 25+25 is 50.
//
// Bilateral, not Gaussian, for the silhouette reason stated at the top: taps
// whose linearised depth disagrees with the centre by more than the edge stop
// are the background (or another surface), and averaging them in rims every
// edge with half-depth soup. Invalid taps (cleared depth) never contribute;
// an invalid centre writes zero, which the shade pass reads as "no water".
//
// TWELVE taps each way, not six. Six spans about one particle spacing on
// screen at tank distance, which merges nothing -- the surface kept every
// lump and the sun sparkled off each one. Twelve spans two spacings, which
// is what actually fuses the sheet; the edge stop (not the radius) is what
// protects silhouettes, so widening is safe.
constexpr sampler smp_surface(coord::normalized, filter::nearest,
                                    address::clamp_to_edge);

// One axis of a separable bilateral smooth over the raw depth, writing LINEAR
// view distance in metres. The full 2D kernel is two of these passes; one
// 25x25 fragment pass would be 625 taps per pixel where 25+25 is 50.
//
// Bilateral, not Gaussian, for the silhouette reason stated at the top: taps
// whose linearised depth disagrees with the centre by more than the edge stop
// are the background (or another surface), and averaging them in rims every
// edge with half-depth soup. Invalid taps (cleared depth) never contribute;
// an invalid centre writes zero, which the shade pass reads as "no water".
//
// TWELVE taps each way, not six. Six spans about one particle spacing on
// screen at tank distance, which merges nothing -- the surface kept every
// lump and the sun sparkled off each one. Twelve spans two spacings, which
// is what actually fuses the sheet; the edge stop (not the radius) is what
// protects silhouettes, so widening is safe.
fragment float4 fs_surface_smooth_h(SurfaceFullOut in [[stage_in]],
                                    depth2d<float> depthTex [[texture(0)]],
                                    constant GpuSurfaceParams& s [[buffer(2)]])
{
    // Reversed-Z, infinite far: dist = near / raw.
    const float rawCentre = depthTex.sample(smp_surface, in.uv);
    if (rawCentre <= 1e-6f) return float4(0.0f);
    const float dc = s.misc.y / rawCentre;
    float sum = 0.0f, weight = 0.0f;
    for (int i = -12; i <= 12; ++i) {
        const float2 off = float2(float(i), 0.0f) * s.screen.zw;
        const float raw = depthTex.sample(smp_surface, in.uv + off);
        if (raw <= 1e-6f) continue;
        const float d = s.misc.y / raw;
        // The edge stop, in metres: taps on another surface are rejected, and
        // the weight is flat inside it rather than Gaussian -- the kernel is
        // a merger, not a blur, and distance-weighting would only shrink its
        // effective radius.
        if (abs(d - dc) > s.misc.z) continue;
        sum += d;
        weight += 1.0f;
    }
    if (weight <= 0.0f) return float4(0.0f);
    return float4(sum / weight, 0.0f, 0.0f, 1.0f);
}

// The vertical axis, reading the horizontal pass's LINEAR output rather than
// raw depth -- hence a second kernel rather than an axis uniform. Same edge
// stop, same units.
fragment float4 fs_surface_smooth_v(SurfaceFullOut in [[stage_in]],
                                    texture2d<float> linearTex [[texture(0)]],
                                    constant GpuSurfaceParams& s [[buffer(2)]])
{
    const float dc = linearTex.sample(smp_surface, in.uv).r;
    if (dc <= 1e-6f) return float4(0.0f);
    float sum = 0.0f, weight = 0.0f;
    for (int i = -12; i <= 12; ++i) {
        const float2 off = float2(0.0f, float(i)) * s.screen.zw;
        const float d = linearTex.sample(smp_surface, in.uv + off).r;
        if (d <= 1e-6f) continue;
        if (abs(d - dc) > s.misc.z) continue;
        sum += d;
        weight += 1.0f;
    }
    if (weight <= 0.0f) return float4(0.0f);
    return float4(sum / weight, 0.0f, 0.0f, 1.0f);
}

// ----------------------------------------------------------------- shade --

fragment float4 fs_surface_shade(SurfaceFullOut in [[stage_in]],
                                 texture2d<float> smoothTex [[texture(0)]],
                                 texture2d<float> background [[texture(1)]],
                                 depth2d<float> tankDepth [[texture(2)]],
                                 constant FrameUniforms& u [[buffer(1)]],
                                 constant GpuSurfaceParams& s [[buffer(2)]])
{
    const float dist = smoothTex.sample(smp_surface, in.uv).r;
    const float3 bg = background.sample(smp_surface, in.uv).rgb;
    // No water here: the background passes through untouched, so the surface
    // needs no blending and no alpha -- every pixel is either water or it is
    // the tank behind it.
    if (dist <= 1e-6f) return float4(bg, 1.0f);

    // The water's own depth buffer knows nothing of the tank (separate
    // buffer, by design), so a sphere behind a wall still has a depth here.
    // The tank's depth decides: water behind it is the background. Same uv,
    // same convention, no orientation to get wrong -- both textures are read
    // here, in this pass, rather than compared across passes.
    const float tankRaw = tankDepth.sample(smp_surface, in.uv);
    if (tankRaw > 1e-6f) {
        const float tankDist = s.misc.y / tankRaw;
        if (dist > tankDist + 0.02f) return float4(bg, 1.0f);
    }

    // Clip from the linear distance: reversed-Z with an infinite far plane
    // stores near/dist, and NDC y is up while texture v is down (the flip in
    // vs_surface_full above), so NDC.y = 1 - 2*uv.y.
    const float raw = s.misc.y / dist;
    const float4 clip =
        float4(in.uv.x * 2.0f - 1.0f, 1.0f - in.uv.y * 2.0f, raw, 1.0f);
    const float4 wpos = u.invViewProj * clip;
    const float3 world = wpos.xyz / wpos.w;

    // The surface normal is the slope of the smoothed sheet. Screen-space
    // derivatives, flipped to face the viewer -- the cross product's order
    // gives either the normal or its negation depending on the handedness,
    // and only one of them lights correctly.
    float3 n = normalize(cross(dfdx(world), dfdy(world)));
    const float3 v = normalize(u.eyePos.xyz - world);
    if (dot(n, v) < 0.0f) n = -n;

    // FRESNEL: straight-on you see into the water, at grazing angles the sky.
    // Schlick with F0 0.02, which is water's -- this is a physical constant,
    // not a tuning knob.
    const float fres =
        0.02f + 0.98f * pow(1.0f - saturate(dot(n, v)), 5.0f);
    // REFLECTION: an analytic sky, horizon to zenith by the reflected ray.
    // A probe sample would be better and is the known next step; this is
    // two lerps and has no capture to keep current.
    const float3 r = reflect(-v, n);
    const float3 refl =
        mix(s.sky.xyz, float3(0.30f, 0.50f, 0.80f), saturate(r.y * 0.5f + 0.5f));
    // REFRACTION: the background behind, pulled toward the pixel by the
    // surface slope -- a real Snell bend needs the indices and the thickness,
    // and without the thickness pass this linear pull is the honest
    // approximation. Tinted green-blue on the way through, which is the
    // cheapest absorption there is.
    const float3 refr =
        background.sample(smp_surface, in.uv + n.xy * s.water.w).rgb * s.water.xyz;
    // THE SUN, Blinn-Phong: water's highlight is small and white, and a
    // broad lobe here is what makes plastic look wet rather than water
    // look bright. 160, not 240: at 240 the glitter collapses to single
    // pixels that read as noise, because the smoothed sheet still carries
    // particle-scale lumps and each one flashes independently.
    const float3 sunDir = normalize(s.sun.xyz);
    const float3 h = normalize(sunDir + v);
    const float spec = pow(saturate(dot(n, h)), 160.0f) * s.sun.w;
    const float3 color =
        mix(refr, refl, fres) + spec * float3(1.0f, 0.96f, 0.90f);
    return float4(color, 1.0f);
}
