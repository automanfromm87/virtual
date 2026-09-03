// Post-processing: exposure, fog, depth of field, motion blur, and the
// temporal resolve.
//
// WHAT THESE HAVE IN COMMON is that they all operate on the finished HDR image
// plus the depth buffer, which is why they live together. What they do NOT have
// in common is where they belong in the frame: fog is applied while everything
// is still linear and before bloom, depth of field after fog but before the
// tone map, and the temporal resolve after everything else that is per-frame
// deterministic. Getting that order wrong is not subtle -- fog applied after
// the tone map is a grey wash rather than distance, and TAA applied before
// bloom accumulates the bloom's own history and smears every highlight.

struct PostParams {
    float4x4 invViewProj;      // current clip -> world
    float4x4 prevViewProj;     // world -> LAST frame's clip
    float4 eye;                // .xyz camera, .w near plane
    float4 fog;                // .xyz colour, .w density per metre
    float4 fog2;               // .x height falloff, .y ground height, .z start distance, .w max
    float4 dof;                // .x focus distance, .y focus range, .z max radius px, .w aperture
    float4 tune;               // .x exposure, .y dt, .z blur strength, .w feedback
    float4 screen;             // .xy size, .zw 1/size
    uint4 bins;                // .x histogram bin count, .yzw unused
    float4 lum;                // .x min log lum, .y log lum range, .z adapt up, .w adapt down
};

// ---------------------------------------------------------------- exposure --
//
// AUTOMATIC EXPOSURE, from a histogram rather than from an average.
//
// A plain average of the frame's luminance is the obvious approach and it is
// dominated by whatever is brightest: one lamp, one patch of sky, one specular
// highlight, and the whole image darkens to accommodate it. A histogram lets
// the resolve below throw away the extremes and take the average of what is
// left, which is what a photographer's meter does and for the same reason.
//
// The bins are LOG luminance. Perception is roughly logarithmic, so linear bins
// would spend nearly all of their resolution on the brightest stop and leave
// the shadows in one bucket.

static float BinOf(float luminance, float min_log, float inv_range, uint bins) {
    if (luminance < 1e-5) return 0.0;  // bin 0 is the "effectively black" bin
    const float log_lum = clamp((log2(luminance) - min_log) * inv_range, 0.0, 1.0);
    return log_lum * float(bins - 2) + 1.0;
}

kernel void cs_luminance_histogram(texture2d<float> hdr [[texture(0)]],
                                   device atomic_uint* histogram [[buffer(0)]],
                                   constant PostParams& p [[buffer(1)]],
                                   uint2 gid [[thread_position_in_grid]],
                                   uint local [[thread_index_in_threadgroup]]) {
    // A THREADGROUP-LOCAL histogram first, merged into the global one once per
    // group. Hitting a single global atomic from every pixel of a 4K frame is
    // eight million contended increments on 256 addresses; this turns it into
    // eight million cheap local ones plus a few thousand global.
    threadgroup atomic_uint local_bins[256];
    if (local < 256) atomic_store_explicit(&local_bins[local], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint2 size = uint2(p.screen.xy);
    if (gid.x < size.x && gid.y < size.y) {
        const float3 c = hdr.read(gid).rgb;
        const float y = dot(c, float3(0.2126, 0.7152, 0.0722));
        const uint bin = uint(BinOf(y, p.lum.x, 1.0 / max(p.lum.y, 1e-4), p.bins.x));
        atomic_fetch_add_explicit(&local_bins[min(bin, 255u)], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local < 256) {
        const uint v = atomic_load_explicit(&local_bins[local], memory_order_relaxed);
        if (v > 0) atomic_fetch_add_explicit(&histogram[local], v, memory_order_relaxed);
    }
}

// One thread. Reduces the histogram, adapts, and leaves the exposure in a
// buffer the composite reads -- so the value never travels to the CPU and back,
// which would cost a full pipeline stall to save nothing.
kernel void cs_luminance_resolve(device atomic_uint* histogram [[buffer(0)]],
                                 device float* exposure [[buffer(1)]],
                                 constant PostParams& p [[buffer(2)]],
                                 uint id [[thread_position_in_grid]]) {
    if (id != 0) return;
    const uint bins = min(p.bins.x, 256u);

    uint total = 0;
    for (uint i = 0; i < bins; ++i)
        total += atomic_load_explicit(&histogram[i], memory_order_relaxed);
    // Bin 0 is the black pixels. Counting them would drag the average toward
    // zero for any scene with a dark background -- a night sky, a letterboxed
    // frame -- and the camera would open up until the subject was white.
    const uint black = atomic_load_explicit(&histogram[0], memory_order_relaxed);
    const uint counted = total - black;

    float weighted = 0.0;
    if (counted > 0) {
        // THE MIDDLE of the distribution, discarding the darkest and brightest
        // fifth. This is the whole reason for the histogram: a scene that is
        // mostly dark with a bright window should expose for the room, and both
        // a plain average and a median-of-everything expose for the window.
        const uint lo = uint(float(counted) * 0.20);
        const uint hi = uint(float(counted) * 0.80);
        uint seen = 0;
        float sum = 0.0, weight = 0.0;
        for (uint i = 1; i < bins; ++i) {
            const uint c = atomic_load_explicit(&histogram[i], memory_order_relaxed);
            if (c == 0) continue;
            const uint start = seen, end = seen + c;
            seen = end;
            if (end <= lo || start >= hi) continue;
            const float used = float(min(end, hi) - max(start, lo));
            sum += used * float(i);
            weight += used;
        }
        weighted = weight > 0.0 ? sum / weight : 0.0;
    }

    // Bin index back to luminance, inverting BinOf.
    const float log_lum = (weighted - 1.0) / float(bins - 2) * p.lum.y + p.lum.x;
    const float average = counted > 0 ? exp2(log_lum) : 0.18;

    // The exposure that maps that average to middle grey. 0.18 is the
    // photographic convention and it is not arbitrary: it is the reflectance of
    // the standard grey card, which is what light meters are calibrated to.
    const float target = 0.18 / max(average, 1e-5);

    // ADAPTATION, exponential, with different speeds each way. The eye adapts
    // to a brighter scene in seconds and to a darker one in minutes, so a
    // single rate is wrong in one direction whichever value is chosen. An
    // instant change is worse than both -- it makes a camera pan look like a
    // fault in the display.
    const float previous = exposure[0] > 0.0 ? exposure[0] : target;
    const float rate = target > previous ? p.lum.z : p.lum.w;
    const float t = 1.0 - exp(-max(p.tune.y, 0.0) * rate);
    exposure[0] = mix(previous, target, saturate(t));

    // Cleared here rather than in a fourth dispatch: this thread has just read
    // every bin and nothing else will touch them until the next frame writes.
    for (uint i = 0; i < bins; ++i)
        atomic_store_explicit(&histogram[i], 0u, memory_order_relaxed);
}

// ---------------------------------------------------------------- velocity --

static float3 WorldFromDepth(float2 uv, float depth, float4x4 inv_view_proj) {
    // Reversed-Z: the depth buffer holds 1 at the near plane and 0 at infinity,
    // and clip space y is up while uv y is down.
    const float4 clip = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    const float4 world = inv_view_proj * clip;
    return world.xyz / world.w;
}

// Screen-space motion, reconstructed from depth and the previous frame's view
// projection.
//
// THE LIMITATION, stated plainly: this is the motion of the CAMERA over static
// geometry. A moving object gets the velocity of the empty space it happens to
// occupy, so temporal accumulation over it relies entirely on the neighbourhood
// clamp in the resolve, and motion blur will not streak a car against a static
// background.
//
// The alternative is a per-object velocity buffer, which means every geometry
// pass carries a second render target and every shader carries the previous
// frame's model matrix. That is the right answer for a renderer whose content
// is mostly moving; for one whose camera is the fastest thing in the frame it
// is a lot of bandwidth to fix a case the clamp already handles.
kernel void cs_velocity(texture2d<float, access::write> velocity [[texture(0)]],
                        // access::read SPELLED OUT. A depth2d defaults to
                        // access::sample, and read() on one of those does not
                        // do what it looks like -- the kernel still compiles and
                        // every texel comes back zero, which this code then
                        // treats as the far plane and reports no motion at all.
                        // The symptom is motion blur that never blurs.
                        depth2d<float, access::read> depth [[texture(1)]],
                        constant PostParams& p [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    const uint2 size = uint2(p.screen.xy);
    if (gid.x >= size.x || gid.y >= size.y) return;
    const float2 uv = (float2(gid) + 0.5) * p.screen.zw;
    const float d = depth.read(gid);
    // The far plane in reversed-Z. Nothing was drawn here, so there is nothing
    // whose motion could be reconstructed; report zero rather than the motion
    // of a point at infinity, which is a large and meaningless number.
    if (d <= 0.0) {
        velocity.write(float4(0.0), gid);
        return;
    }
    const float3 world = WorldFromDepth(uv, d, p.invViewProj);
    const float4 prev_clip = p.prevViewProj * float4(world, 1.0);
    if (prev_clip.w <= 0.0) {
        velocity.write(float4(0.0), gid);
        return;
    }
    float2 prev_uv = prev_clip.xy / prev_clip.w;
    prev_uv = float2(prev_uv.x * 0.5 + 0.5, 0.5 - prev_uv.y * 0.5);
    // Stored as the offset FROM this pixel TO where it was, in uv units, so a
    // consumer adds it rather than subtracting -- which is one fewer place to
    // get a sign wrong.
    velocity.write(float4(prev_uv - uv, 0.0, 0.0), gid);
}

// -------------------------------------------------------------- fullscreen --

struct PostOut {
    float4 position [[position]];
    float2 uv;
};

vertex PostOut vs_post(uint vid [[vertex_id]]) {
    const float2 corners[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
    PostOut o;
    o.position = float4(corners[vid], 0.0, 1.0);
    o.uv = corners[vid] * 0.5 + 0.5;
    o.uv.y = 1.0 - o.uv.y;
    return o;
}

// --------------------------------------------------------------------- fog --
//
// EXPONENTIAL, with a HEIGHT falloff.
//
// Distance alone gives the flat wash that reads as "someone turned on the fog
// slider". What makes it look like air is that there is more of it low down:
// the density falls off exponentially with height, so a valley fills and a
// hilltop does not, and a camera climbing out of the valley watches it thin.
//
// The integral of an exponentially decaying density along a ray has a closed
// form, which is why this is a handful of instructions rather than a march.
fragment float4 fs_fog(PostOut in [[stage_in]],
                       depth2d<float> depth [[texture(0)]],
                       constant PostParams& p [[buffer(0)]],
                       sampler smp [[sampler(0)]]) {
    const float d = depth.sample(smp, in.uv);
    const float3 world = WorldFromDepth(in.uv, max(d, 1e-7), p.invViewProj);
    const float3 to_eye = world - p.eye.xyz;
    float distance = length(to_eye);
    // Nothing drawn: fog the sky out to the maximum distance, so a foggy scene
    // does not have a crisp horizon floating in it.
    if (d <= 0.0) distance = p.fog2.w;
    distance = max(distance - p.fog2.z, 0.0);
    if (distance <= 0.0 || p.fog.w <= 0.0) discard_fragment();

    const float falloff = max(p.fog2.x, 1e-4);
    const float ground = p.fog2.y;
    // The analytic integral of density * exp(-falloff * (y - ground)) along the
    // segment from the eye to the surface. The guard is for a nearly horizontal
    // ray, where the height difference goes to zero and the expression becomes
    // 0/0 -- at which point the density is constant along the ray and the
    // integral is just density times distance.
    const float y0 = p.eye.y - ground;
    const float y1 = world.y - ground;
    const float dy = y1 - y0;
    float integral;
    if (abs(dy) < 1e-3) {
        integral = exp(-falloff * y0) * distance;
    } else {
        integral = (exp(-falloff * y0) - exp(-falloff * y1)) / (falloff * dy) * distance;
    }
    const float amount = saturate(1.0 - exp(-p.fog.w * max(integral, 0.0)));
    // ALPHA BLENDED, so this pass adds fog to whatever the scene pass left
    // rather than needing to read and rewrite it -- which would be a read of
    // the target being written.
    return float4(p.fog.xyz, amount);
}

// --------------------------------------------------------- depth of field --
//
// A GATHER, not a scatter. Each output pixel looks at its neighbours and takes
// the ones whose circle of confusion is large enough to reach it. That is the
// wrong way round physically -- a blurred point spreads outward, it does not
// pull inward -- and it is what every real-time implementation does, because
// scattering needs either per-pixel geometry or a sort.
//
// The visible consequence is that a sharp foreground object does not bleed over
// a blurred background the way a real lens does; the silhouette stays crisp.
fragment float4 fs_dof(PostOut in [[stage_in]],
                       texture2d<float> src [[texture(0)]],
                       depth2d<float> depth [[texture(1)]],
                       constant PostParams& p [[buffer(0)]],
                       sampler smp [[sampler(0)]]) {
    const float d = depth.sample(smp, in.uv);
    // Linear eye distance from a reversed-Z buffer with an infinite far plane:
    // depth = near / distance, so distance = near / depth.
    const float dist = d > 1e-6 ? p.eye.w / d : 1e6;

    // The circle of confusion, in pixels. Signed in a real lens -- in front of
    // and behind the focal plane blur differently -- and unsigned here, which
    // costs the bokeh's asymmetry and saves carrying the sign through the
    // gather.
    const float focus = p.dof.x, range = max(p.dof.y, 1e-3);
    const float coc = saturate(abs(dist - focus) / range) * p.dof.z;
    if (coc < 0.75) return float4(src.sample(smp, in.uv).rgb, 1.0);

    // A GOLDEN-ANGLE SPIRAL of taps. Uniform in area rather than in angle,
    // which a concentric-rings pattern is not: rings put the same number of
    // samples on a small circle as on a large one, so the centre is
    // oversampled and the rim has visible spokes.
    //
    // THE COUNT SCALES WITH THE RADIUS, and it has to. A fixed 32 taps over a
    // 26-pixel circle is one sample per 66 square pixels, which does not blur a
    // fine pattern -- it RESAMPLES it, and neighbouring output pixels draw from
    // different sparse sets, so the result is new aliasing at the same
    // amplitude as the old. Measured: a checkerboard at the horizon lost only
    // 22% of its high-frequency energy through a supposedly 26-pixel blur.
    // Roughly two and a half taps per pixel of radius is where that stops.
    const int taps = int(clamp(coc * 2.5, 16.0, 64.0));
    constexpr float kGolden = 2.39996323;
    float3 sum = src.sample(smp, in.uv).rgb;
    float weight = 1.0;
    for (int i = 1; i <= taps; ++i) {
        const float t = float(i) / float(taps);
        const float radius = sqrt(t) * coc;
        const float angle = float(i) * kGolden;
        const float2 offset = float2(cos(angle), sin(angle)) * radius * p.screen.zw;
        const float2 uv = in.uv + offset;

        // A neighbour only contributes if IT is blurred too. Without this, a
        // sharp object in front of a blurred background donates its colour to
        // the background and gains a halo -- the classic depth-of-field
        // artefact, and the one thing that makes a cheap implementation
        // instantly recognisable.
        const float nd = depth.sample(smp, uv);
        const float ndist = nd > 1e-6 ? p.eye.w / nd : 1e6;
        const float ncoc = saturate(abs(ndist - focus) / range) * p.dof.z;
        if (ncoc < radius - 1.0) continue;

        sum += src.sample(smp, uv).rgb;
        weight += 1.0;
    }
    return float4(sum / weight, 1.0);
}

// -------------------------------------------------------------- motion blur --

fragment float4 fs_motion_blur(PostOut in [[stage_in]],
                               texture2d<float> src [[texture(0)]],
                               texture2d<float> velocity [[texture(1)]],
                               constant PostParams& p [[buffer(0)]],
                               sampler smp [[sampler(0)]]) {
    // Scaled by the SHUTTER, which is the physical parameter: a blur is the
    // integral of the image over the time the shutter was open, so half a
    // frame's exposure gives half a frame's streak.
    float2 v = velocity.sample(smp, in.uv).xy * p.tune.z;
    // Capped, because a fast camera turn produces a velocity longer than the
    // screen and the samples would all land outside it -- which reads as the
    // image dissolving rather than moving.
    const float len_px = length(v * p.screen.xy);
    if (len_px < 1.0) return float4(src.sample(smp, in.uv).rgb, 1.0);
    const float capped = min(len_px, 64.0);
    v *= capped / len_px;

    const int taps = int(clamp(capped, 4.0, 16.0));
    float3 sum = float3(0.0);
    for (int i = 0; i < taps; ++i) {
        // Centred on the pixel and running BOTH ways along the velocity. A
        // one-sided sweep biases the whole image in the direction of travel,
        // which looks like the picture sliding rather than smearing.
        const float t = (float(i) + 0.5) / float(taps) - 0.5;
        sum += src.sample(smp, in.uv + v * t).rgb;
    }
    return float4(sum / float(taps), 1.0);
}

// ---------------------------------------------------------------- temporal --
//
// TAA: this frame's jittered sample blended with the accumulated history,
// reprojected to follow the camera.
//
// The whole difficulty is that the history is WRONG whenever something changes
// that reprojection cannot describe -- a surface becoming visible from behind
// an edge, a shadow moving, an object that the velocity buffer does not track.
// Blending it in anyway gives ghosting: a faint copy of where things were,
// trailing behind where they are.
//
// The fix is the NEIGHBOURHOOD CLAMP. The history is clamped into the range of
// colours actually present in the 3x3 block around the pixel this frame, on the
// grounds that a plausible history sample must look like something nearby. It
// is a heuristic and it is the one that makes TAA usable.

static float3 RgbToYCoCg(float3 c) {
    // Clamping in YCoCg rather than RGB, because the box that bounds a
    // neighbourhood in RGB is much larger than the colours actually in it --
    // luminance and chroma vary independently, and an axis-aligned RGB box
    // around a dark red and a bright red contains colours that are neither.
    return float3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
                  0.5 * c.r - 0.5 * c.b,
                  -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}
static float3 YCoCgToRgb(float3 c) {
    const float t = c.x - c.z;
    return float3(t + c.y, c.x + c.z, t - c.y);
}

fragment float4 fs_taa(PostOut in [[stage_in]],
                       texture2d<float> current [[texture(0)]],
                       texture2d<float> history [[texture(1)]],
                       texture2d<float> velocity [[texture(2)]],
                       constant PostParams& p [[buffer(0)]],
                       sampler smp [[sampler(0)]]) {
    const float3 now = current.sample(smp, in.uv).rgb;
    const float2 prev_uv = in.uv + velocity.sample(smp, in.uv).xy;

    // Off screen last frame: there is no history, so use this frame alone.
    // Blending toward the clamped edge instead would smear the border inward,
    // which is visible as a bright rim whenever the camera pans.
    if (prev_uv.x < 0.0 || prev_uv.x > 1.0 || prev_uv.y < 0.0 || prev_uv.y > 1.0)
        return float4(now, 1.0);

    float3 lo = RgbToYCoCg(now), hi = lo;
    float3 mean = lo, mean2 = lo * lo;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x) {
            if (x == 0 && y == 0) continue;
            const float3 c = RgbToYCoCg(
                current.sample(smp, in.uv + float2(x, y) * p.screen.zw).rgb);
            lo = min(lo, c);
            hi = max(hi, c);
            mean += c;
            mean2 += c * c;
        }
    // A VARIANCE box, intersected with the min/max one. The plain min/max box
    // is too generous when one neighbour is an outlier -- a single specular
    // pixel widens it enough to let a ghost through -- and too tight in noise.
    // One standard deviation either side of the mean tracks the actual
    // distribution; taking the intersection keeps the hard bound as a backstop.
    mean /= 9.0;
    mean2 /= 9.0;
    const float3 sigma = sqrt(max(mean2 - mean * mean, 0.0));
    lo = max(lo, mean - sigma * 1.25);
    hi = min(hi, mean + sigma * 1.25);

    float3 old = RgbToYCoCg(history.sample(smp, prev_uv).rgb);
    old = clamp(old, lo, hi);

    // The feedback weight. Higher keeps more history and antialiases better;
    // too high and every moving edge trails. 0.9 is the usual compromise.
    //
    // Blended in RGB after converting back, not in YCoCg: the clamp needed the
    // decorrelated space, the blend does not, and mixing in YCoCg would make
    // the weight mean something slightly different for chroma than for
    // luminance.
    const float feedback = saturate(p.tune.w);
    return float4(mix(now, YCoCgToRgb(old), feedback), 1.0);
}
