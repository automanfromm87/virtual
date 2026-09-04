// A physically based sky, from scattering rather than from a gradient.
//
// WHY NOT A GRADIENT. Two colours lerped by height is what most engines start
// with and it is wrong in the way that matters: it cannot produce a sunset. The
// red at the horizon is not a colour someone chose, it is what is left of the
// sunlight after it has crossed three hundred kilometres of atmosphere and had
// the blue scattered out of it, and no lerp between a zenith colour and a
// horizon colour reproduces the dependence on the sun's angle. Nor does a
// gradient give the bright ring around the sun, which is Mie scattering off
// aerosols and is most of what makes a sky read as photographic.
//
// THE MODEL is single-scattering Rayleigh plus Mie, integrated along the view
// ray -- the standard Nishita formulation. For each step along the ray it asks
// how much sunlight reaches that point (an optical-depth march toward the sun)
// and how much of it scatters toward the eye (the phase functions).
//
// What single scattering LEAVES OUT is the light that bounces more than once,
// which is why a physically correct single-scattering sky is too dark near the
// horizon at dusk and has a night side that is pure black. A ground bounce term
// stands in for the first of those; the second is why `night_sky` exists.

// Earth, in metres. Real numbers, because the scale height below is a real
// number and mixing an invented planet with a measured atmosphere gives a sky
// that is subtly the wrong colour at every angle.
constant float kGroundRadius = 6360000.0;
constant float kAtmosphereRadius = 6420000.0;
// Rayleigh scattering at sea level, per metre, for R/G/B. The famous 1/lambda^4:
// blue scatters six times as readily as red, which is the entire reason the sky
// is blue and the sun is yellow.
constant float3 kRayleigh = float3(5.8e-6, 13.5e-6, 33.1e-6);
// Mie is aerosols -- dust, water droplets -- which are large compared with the
// wavelength and so scatter all colours nearly equally. That is why haze is
// white and why the glow around the sun is not blue.
constant float kMie = 21e-6;
// How fast each falls off with altitude. Air is well mixed and thins slowly;
// aerosols sit near the ground.
constant float kRayleighScaleHeight = 8000.0;
constant float kMieScaleHeight = 1200.0;
// Mie's forward-scattering bias. Near 1 the glow tightens onto the sun.
constant float kMieG = 0.758;
// How much brighter the solar disc is than the `sun_intensity` that drives the
// scattering integral. It is a unit conversion: the integral treats
// sun_intensity as the irradiance arriving at the top of the atmosphere, while
// the disc needs a RADIANCE -- energy per solid angle -- and the two differ by
// the solid angle the disc covers, 1/(pi r^2) which is about 14700.
//
// IT WAS 60, which is 245 times too low, and that was load-bearing rather than
// careless: the irradiance convolution integrates this cube, the renderer also
// applies the sun as a directional light, and a disc at full strength is
// therefore counted twice -- the second time without a shadow, so it lights the
// inside of every shadow. Suppressing the disc was the cheap way to avoid that.
// cs_irradiance clamps now, so it no longer has to be.
//
// AND IT IS STILL NOT 14700, because the environment is stored as RGBA16Float
// and 22 * 14700 is 324000, which is five times the format's 65504 ceiling and
// arrives as +Inf -- which then propagates through the mip chain and the
// specular prefilter and turns whole reflections into NaN. 2000 puts the peak
// texel near 44000 with room to spare, leaves the disc about a thousand times
// the brightest sky, and costs nothing real: the sun's ENERGY is carried by the
// directional light, and what the cube needs is something that looks like a sun
// when it is reflected.
constant float kSunDiscGain = 2000.0;

// MULTIPLE SCATTERING, computed rather than fitted.
//
// The march below follows each photon exactly once: sun, one scattering event,
// eye. A real photon bounces many times before it arrives, and the ones that
// bounce more than once are most of a clear sky's light -- leaving them out
// does not make the sky slightly dim, it makes it dim by a factor of four, and
// that is the whole reason a physically correct renderer produces harsh
// outdoor images. The sun comes out right and every shadow is two stops too
// dark.
//
// THE FIRST VERSION OF THIS WAS A FITTED GAIN. An isotropic term times 3.4,
// with a second constant to whiten it with optical depth, both turned until the
// sun-to-sky ratio matched the real one. It worked, and it was two numbers with
// no derivation behind them -- change the turbidity or the scale heights and
// they would be wrong with nothing to say so.
//
// This is Hillaire's construction instead. Light that has scattered twice is
// computed directly by integrating over the sphere; light that has scattered
// three times is that result scaled by the same transfer fraction, four times
// by the square, and so on. The series is geometric, so the whole infinite tail
// sums in closed form:
//
//     Psi = L_2nd / (1 - F)
//
// where F is the fraction of light a point re-scatters rather than loses. Both
// terms come out of the medium's own coefficients, so turbidity and scale
// height feed through on their own and there is nothing left to tune.
//
// The spectrum falls out too. The first version needed an explicit whitening
// term because weighting multiple scattering by the single-scatter spectrum
// made the horizon bluer than the zenith, when it must be whiter; here F is
// per channel, so blue -- which scatters most -- also has the largest F and the
// strongest amplification, and the saturation that produces is the same
// saturation that whitens a deep sky. It is not corrected for, it is what the
// series does.
constant int kMsDirections = 64;   // sphere samples per LUT texel
constant int kMsSteps = 20;        // march steps per direction
constant float kIsotropicPhase = 0.0795774715;  // 1 / (4 pi)

// THE PRECISION PROBLEM, and why both functions below are written oddly.
//
// A ray-sphere intersection is t = -b +/- sqrt(b*b - c). At planetary scale
// with the eye two metres off the ground, b is about 6.4e6 and c is about
// 2.5e7 -- so b*b is 4.0e13, and float32 carries about seven significant
// digits, which at 4.0e13 means a resolution of roughly 2.4e6. Subtracting c
// from b*b therefore discards c ENTIRELY: the discriminant comes back as b*b,
// the root as -b +/- |b|, and the planet effectively is not there.
//
// This is not a subtle drift. It is the reason a naively written atmosphere
// shader has a horizon in the wrong place, or none at all, and it does not
// reproduce on a toy sphere of radius 1 -- which is where it gets tested.
//
// The fix is the textbook stable quadratic. For t^2 + 2bt + c = 0, compute
//
//     q = -b - sign(b) * sqrt(d)
//
// so that the two terms ADD rather than cancel, and then take the two roots as
// q and c/q. The root that used to be a difference of two nearly equal large
// numbers is now a quotient, and c -- the quantity that was being lost --
// appears in it directly, so every digit of it survives.
//
// The sign branch is not optional and getting it wrong is worse than the
// original. A first attempt here wrote the small root as c / (-b + sqrt(d))
// unconditionally with the denominator clamped away from zero. For a ray
// pointing UP that denominator is genuinely negative -- the ray misses the
// ground behind it -- and the clamp turned it into a tiny positive number, so
// every upward ray reported a ground hit two metres away, the march stopped
// immediately, and the entire sky went black. The clamp hid the sign, and the
// sign was the answer.
//
// `c` is also computed factored, as (|o| - r)(|o| + r) rather than |o|^2 - r^2,
// for the same cancellation reason one scale down.

static float SphereC(float3 origin, float radius) {
    const float len = length(origin);
    return (len - radius) * (len + radius);
}

// The two roots, stably. Returns them in ascending order; x is -1 for a miss.
static float2 SphereRoots(float3 origin, float3 dir, float radius) {
    const float b = dot(origin, dir);
    const float c = SphereC(origin, radius);
    const float d = b * b - c;
    if (d < 0.0) return float2(-1.0, -1.0);
    const float root = sqrt(d);
    const float q = (b < 0.0) ? (-b + root) : (-b - root);
    // q is zero only for a ray tangent to a sphere it starts on, where both
    // roots are zero anyway.
    const float other = (abs(q) > 1e-20) ? (c / q) : q;
    return float2(min(q, other), max(q, other));
}

// Where a ray leaves a sphere it is INSIDE.
static float RayAtmosphereExit(float3 origin, float3 dir, float radius) {
    const float2 t = SphereRoots(origin, dir, radius);
    // Inside the sphere c is negative, so the roots straddle zero and the exit
    // is the positive one -- always the larger.
    return t.y > 0.0 ? t.y : -1.0;
}

// Whether a ray hits the planet, and where. The sky below the horizon is not
// sky -- it is ground seen through air -- and integrating through the planet
// gives a bright band under the horizon that looks like a bug because it is.
static float RayGroundHit(float3 origin, float3 dir, float radius) {
    const float2 t = SphereRoots(origin, dir, radius);
    if (t.x > 0.0) return t.x;   // the near surface, looking down
    if (t.y > 0.0) return t.y;   // started underground, which should not happen
    return -1.0;                 // behind us: the ray is going up
}

static float RayleighPhase(float cos_theta) {
    // 3/(16pi) * (1 + cos^2). Symmetric forward and back, which is why the sky
    // opposite the sun is still blue rather than black.
    return 3.0 / (16.0 * M_PI_F) * (1.0 + cos_theta * cos_theta);
}

static float MiePhase(float cos_theta, float g) {
    // Henyey-Greenstein. Strongly forward-peaked, which is the sun's halo.
    const float g2 = g * g;
    const float denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return 3.0 / (8.0 * M_PI_F) * ((1.0 - g2) * (1.0 + cos_theta * cos_theta)) /
           ((2.0 + g2) * pow(max(denom, 1e-4), 1.5));
}

struct SkyParams {
    float4 sun_dir;       // .xyz toward the sun, .w sun angular radius (radians)
    float4 ground;        // .xyz albedo, .w turbidity
    float4 tune;          // .x sun intensity, .y exposure, .z night lift, .w unused
    uint4 size;           // .x cube face size, .y multi-scatter table size
};

// The SCATTERED radiance along `dir` -- the sky itself, with no solar disc and
// no ground bounce. Split out from the full model below because the ground term
// needs to know how bright the sky is in order to work out what it is
// reflecting, and a ground lit by a hand-picked constant instead is the reason
// most skies have a lower hemisphere that is either black or glowing.
// The optical depth from `x` along `dir` to the top of the atmosphere.
static float3 SunOpticalDepth(float3 x, float3 dir, float mie_scale, int steps) {
    const float t = RayAtmosphereExit(x, dir, kAtmosphereRadius);
    if (t <= 0.0) return float3(0.0);
    const float dt = t / float(steps);
    float r = 0.0, m = 0.0;
    for (int i = 0; i < steps; ++i) {
        const float h = max(length(x + dir * (dt * (float(i) + 0.5))) - kGroundRadius, 0.0);
        r += exp(-h / kRayleighScaleHeight) * dt;
        m += exp(-h / kMieScaleHeight) * dt;
    }
    return kRayleigh * r + float3(mie_scale * 1.1) * m;
}

// A Fibonacci sphere, which is the cheapest way to get directions that are
// evenly spread. Random ones cluster, and a latitude-longitude grid puts most
// of its samples at the poles -- both show up here as a LUT that is noisy in
// one place and smooth in another.
static float3 FibonacciDirection(int i, int n) {
    const float y = 1.0 - (float(i) + 0.5) / float(n) * 2.0;
    const float r = sqrt(max(1.0 - y * y, 0.0));
    const float phi = float(i) * 2.39996323;  // the golden angle
    return float3(cos(phi) * r, y, sin(phi) * r);
}

// One texel of the multiple-scattering table: the amplification a point at this
// altitude, under a sun at this angle, applies to the light it scatters.
//
// The isotropic phase everywhere is not a simplification made for speed. After
// two bounces the light has genuinely forgotten which way it came, and using
// the Rayleigh phase again would deepen the glow around the sun rather than
// lifting the whole dome -- which is the opposite of what multiple scattering
// does.
static float3 MultiScatterAt(float altitude, float mu_s, constant SkyParams& p) {
    const float mie_scale = max(p.ground.w, 0.0) * kMie;
    const float3 pos = float3(0.0, kGroundRadius + altitude, 0.0);
    const float3 sun = normalize(float3(sqrt(max(1.0 - mu_s * mu_s, 0.0)), mu_s, 0.0));

    float3 second = float3(0.0);    // light arriving after exactly two scatters
    float3 transfer = float3(0.0);  // the fraction this point re-scatters

    for (int d = 0; d < kMsDirections; ++d) {
        const float3 w = FibonacciDirection(d, kMsDirections);
        const float ground_t = RayGroundHit(pos, w, kGroundRadius);
        const float sky_t = RayAtmosphereExit(pos, w, kAtmosphereRadius);
        const float end = ground_t > 0.0 ? ground_t : sky_t;
        if (end <= 0.0) continue;
        const float dt = end / float(kMsSteps);

        float3 tau = float3(0.0);
        for (int i = 0; i < kMsSteps; ++i) {
            const float3 x = pos + w * (dt * (float(i) + 0.5));
            const float h = max(length(x) - kGroundRadius, 0.0);
            const float3 sigma_r = kRayleigh * exp(-h / kRayleighScaleHeight);
            const float3 sigma_m = float3(mie_scale) * exp(-h / kMieScaleHeight);
            const float3 sigma_s = sigma_r + sigma_m;
            const float3 sigma_t = sigma_r + sigma_m * 1.1;

            const float3 through = exp(-tau);  // reaching x from pos along w
            // The sun's contribution at x, if the planet is not in the way.
            if (RayGroundHit(x, sun, kGroundRadius) <= 0.0) {
                const float3 to_sun = exp(-SunOpticalDepth(x, sun, mie_scale, 6));
                second += through * to_sun * sigma_s * (kIsotropicPhase * dt);
            }
            // What a point scatters rather than absorbs or passes: this is the
            // F that makes the series geometric.
            transfer += through * sigma_s * dt;
            tau += sigma_t * dt;
        }
    }

    const float inv = 1.0 / float(kMsDirections);
    second *= inv * p.tune.x;
    transfer *= inv;
    // 1 / (1 - F), summing every order above the second. Clamped because a
    // medium that scattered everything and absorbed nothing would give an
    // infinite series -- physically it cannot, since photons escape, but the
    // integral above is discrete and can round past it.
    return second / max(float3(1.0) - transfer, float3(1e-3));
}

kernel void cs_multiscatter(texture2d<float, access::write> out [[texture(0)]],
                            constant SkyParams& p [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
    const uint n = p.size.y;
    if (gid.x >= n || gid.y >= n) return;
    // x is the sun's cosine from straight up, over the whole range: the sky
    // has to be right at sunset as well as at noon. y is altitude.
    const float mu_s = (float(gid.x) + 0.5) / float(n) * 2.0 - 1.0;
    const float altitude =
        (float(gid.y) + 0.5) / float(n) * (kAtmosphereRadius - kGroundRadius);
    out.write(float4(MultiScatterAt(altitude, mu_s, p), 1.0), gid);
}

static float3 ScatteredRadiance(float3 dir, constant SkyParams& p, thread float3& out_tau,
                                int view_steps, int sun_steps,
                                texture2d<float> ms, sampler ms_smp) {
    const float3 sun = normalize(p.sun_dir.xyz);
    const float3 eye = float3(0.0, kGroundRadius + 2.0, 0.0);
    out_tau = float3(0.0);

    const float atmos_t = RayAtmosphereExit(eye, dir, kAtmosphereRadius);
    if (atmos_t <= 0.0) return float3(0.0);
    const float ground_t = RayGroundHit(eye, dir, kGroundRadius);
    const float march_end = ground_t > 0.0 ? ground_t : atmos_t;
    const float mie_scale = max(p.ground.w, 0.0) * kMie;

    const float step_len = march_end / float(view_steps);

    float3 rayleigh_sum = float3(0.0);
    float3 mie_sum = float3(0.0);
    float optical_r = 0.0, optical_m = 0.0;

    for (int i = 0; i < view_steps; ++i) {
        const float3 pos = eye + dir * (step_len * (float(i) + 0.5));
        const float height = max(length(pos) - kGroundRadius, 0.0);
        const float dr = exp(-height / kRayleighScaleHeight) * step_len;
        const float dm = exp(-height / kMieScaleHeight) * step_len;
        optical_r += dr;
        optical_m += dm;

        if (RayGroundHit(pos, sun, kGroundRadius) > 0.0) continue;
        const float sun_t = RayAtmosphereExit(pos, sun, kAtmosphereRadius);
        if (sun_t <= 0.0) continue;
        float sun_r = 0.0, sun_m = 0.0;
        const float sun_step = sun_t / float(sun_steps);
        for (int j = 0; j < sun_steps; ++j) {
            const float3 sp = pos + sun * (sun_step * (float(j) + 0.5));
            const float sh = max(length(sp) - kGroundRadius, 0.0);
            sun_r += exp(-sh / kRayleighScaleHeight) * sun_step;
            sun_m += exp(-sh / kMieScaleHeight) * sun_step;
        }
        const float3 tau = kRayleigh * (optical_r + sun_r) +
                           float3(mie_scale * 1.1) * (optical_m + sun_m);
        const float3 attenuation = exp(-tau);
        rayleigh_sum += attenuation * dr;
        mie_sum += attenuation * dm;
    }

    // MULTIPLE SCATTERING, from the table. Walked separately from the loop
    // above because that one skips any step the planet shadows from the sun --
    // correct for single scattering, wrong for this: a point in the Earth's
    // shadow still receives light that bounced its way in, and that is most of
    // what makes a twilight sky bright rather than black.
    float3 multi = float3(0.0);
    if (!is_null_texture(ms)) {
        const float3 view_att_scale = float3(1.0);
        float3 view_tau = float3(0.0);
        for (int i = 0; i < view_steps; ++i) {
            const float3 pos = eye + dir * (step_len * (float(i) + 0.5));
            const float h = max(length(pos) - kGroundRadius, 0.0);
            const float3 sigma_r = kRayleigh * exp(-h / kRayleighScaleHeight);
            const float3 sigma_m = float3(mie_scale) * exp(-h / kMieScaleHeight);
            const float mu_s = dot(normalize(pos), sun);
            const float2 uv =
                float2(mu_s * 0.5 + 0.5,
                       clamp(h / (kAtmosphereRadius - kGroundRadius), 0.0, 1.0));
            const float3 psi = ms.sample(ms_smp, uv).rgb;
            multi += exp(-view_tau) * psi * (sigma_r + sigma_m) * step_len *
                     view_att_scale;
            view_tau += (sigma_r + sigma_m * 1.1) * step_len;
        }
    }

    // Handed back so the disc can be attenuated by exactly the air the view ray
    // passed through, rather than by a second march that would have to agree.
    out_tau = kRayleigh * optical_r + float3(mie_scale * 1.1) * optical_m;

    const float cos_theta = dot(dir, sun);
    const float3 single = rayleigh_sum * kRayleigh * RayleighPhase(cos_theta) +
                          mie_sum * mie_scale * MiePhase(cos_theta, kMieG);
    // No gain, no whitening term. Both were fitted; the table carries the
    // amplification and its spectrum comes out of the medium's own
    // coefficients, so there is nothing left here to turn.
    return float3(p.tune.x) * single + multi;
}

// The irradiance the SKY puts on level ground: a cosine-weighted average over
// the upper hemisphere, times pi.
//
// This was the zenith radiance alone, and that is wrong in precisely the case
// that matters most. At noon the zenith IS the sun's direction, so it sits on
// the Mie forward-scattering peak, which is about six times the hemisphere
// average -- and the ground came out BRIGHTER than the sky above it. Sampling
// one direction to stand for a hemisphere is only safe when the function is
// flat, and a sky is at its least flat exactly where the sun is.
//
// Nine directions: the zenith and a ring of eight at 55 degrees. Enough to
// dilute the peak to roughly its true share of the hemisphere, and cheap
// because these marches use a quarter of the main ray's steps -- the result is
// an ambient term spread over a hemisphere, so an error of a few percent in it
// is invisible, while an error of 600% was not.
//
// Recomputed per ground texel, which is redundant: it depends only on the sun.
// Hoisting it would mean a second dispatch and a buffer to carry one float3
// between them, and the bake happens when the sun moves rather than per frame.
static float3 SkyIrradianceOnGround(constant SkyParams& p,
                                    texture2d<float> ms, sampler ms_smp) {
    float3 sum = float3(0.0);
    float weight = 0.0;
    float3 ignored = float3(0.0);

    sum += ScatteredRadiance(float3(0, 1, 0), p, ignored, 6, 3, ms, ms_smp) * 1.0;
    weight += 1.0;

    const float theta = 0.96;  // 55 degrees from vertical
    const float st = sin(theta), ct = cos(theta);
    for (int i = 0; i < 8; ++i) {
        const float phi = float(i) * (2.0 * M_PI_F / 8.0);
        const float3 d = float3(st * cos(phi), ct, st * sin(phi));
        // Weighted by the cosine, which is Lambert's law: a direction 55
        // degrees off vertical delivers cos(55) of what the zenith does.
        sum += ScatteredRadiance(d, p, ignored, 6, 3, ms, ms_smp) * ct;
        weight += ct;
    }
    // Mean radiance times pi is the irradiance, because the cosine-weighted
    // solid angle of a hemisphere is exactly pi.
    return sum / weight * M_PI_F;
}

// The radiance arriving from `dir`. This is the whole sky model.
static float3 SkyRadiance(float3 dir, constant SkyParams& p,
                          texture2d<float> ms, sampler ms_smp) {
    const float3 sun = normalize(p.sun_dir.xyz);
    const float3 eye = float3(0.0, kGroundRadius + 2.0, 0.0);

    float3 tau = float3(0.0);
    float3 result = ScatteredRadiance(dir, p, tau, 24, 8, ms, ms_smp);
    const bool hits_ground = RayGroundHit(eye, dir, kGroundRadius) > 0.0;

    // THE SUN'S DISC, drawn only where the ground does not block it. Its
    // angular radius is a quarter of a degree, so at a 256-pixel cube face it
    // covers barely a texel -- which is exactly why it is added analytically
    // rather than left to whatever the ray march happens to sample. Without it
    // the environment carries no sun at all and every specular reflection is a
    // dull smear instead of a highlight.
    const float radius = max(p.sun_dir.w, 1e-4);

    // SPREAD TO AT LEAST A TEXEL, with the radiance dropped to match.
    //
    // The sun is a quarter of a degree across and a face of a 64-cube is 90
    // degrees over 64 texels, so a texel is 1.4 degrees and the disc is a fifth
    // of one. This is evaluated at texel CENTRES, and for a sun on a face's
    // axis the nearest centre is half a texel away -- so no sample ever landed
    // inside the disc and the cube contained no sun at all. Raising the gain
    // did nothing, measured: the brightest texel in the sun's own face did not
    // move by a digit when the gain went up by a factor of thirty.
    //
    // Widening it and scaling the radiance by the ratio of solid angles keeps
    // the total energy right, which is the whole point -- an aliased disc is
    // not "a slightly wrong sun", it is no sun, and a disc drawn at full
    // radiance across a whole texel would be a sun thirty times too bright.
    const float texel = 1.5707963 / max(float(p.size.x), 1.0);
    // 1.2 texels: half a texel diagonal is 0.71, so this guarantees a centre
    // inside the disc with enough margin that the smoothstep edge has somewhere
    // to fall rather than clipping to a single hard texel.
    const float spread = max(radius, texel * 1.2);
    const float conserve = (radius * radius) / (spread * spread);
    const float3 disc_radiance =
        exp(-tau) * float3(p.tune.x) * kSunDiscGain * conserve;
    if (!hits_ground) {
        const float angle = acos(clamp(dot(dir, sun), -1.0, 1.0));
        // Softened at the edge so a low-resolution cube does not alias the disc
        // into a square.
        const float disc = 1.0 - smoothstep(spread * 0.85, spread * 1.15, angle);
        if (disc > 0.0) result += disc_radiance * disc;
    }

    // The GROUND, as a Lambertian surface lit by what is actually above it,
    // rather than by a hand-picked fraction of the sun.
    //
    // Both terms are needed and they dominate at different times of day. The
    // sun's is the larger at noon; the SKY's is the only one left at twilight,
    // when the sun is below the horizon and the ground is still clearly lit --
    // which is the case a sun-only ground gets visibly wrong, going black while
    // the sky above it is still bright.
    //
    // Radiance out is albedo times irradiance in over pi, which is the
    // definition of a Lambertian reflector and is where the factor most
    // implementations fudge actually comes from.
    if (hits_ground) {
        const float3 sky_irradiance = SkyIrradianceOnGround(p, ms, ms_smp);
        // The sun's, as radiance times the solid angle it subtends times the
        // cosine at the ground. The solid angle is what makes this small: a
        // disc of a quarter degree covers 68 millionths of a steradian, and an
        // implementation that forgets it produces a ground brighter than the
        // sun itself.
        const float solid_angle = 2.0 * M_PI_F * (1.0 - cos(radius));
        const float3 sun_irradiance = disc_radiance * solid_angle * max(sun.y, 0.0);
        result += p.ground.xyz * (sky_irradiance + sun_irradiance) / M_PI_F;
    }

    // A floor, so a night sky is dark rather than absent. Pure black in an
    // environment map means a mirror reflects nothing and reads as a hole.
    return max(result * p.tune.y, float3(p.tune.z));
}

// One texel of one cube face. `gid.z` is the face, which is why the output is
// bound as a 2D array rather than as a cubemap -- see Device::CreateMipView.
kernel void cs_sky_cube(texture2d_array<float, access::write> out [[texture(0)]],
                        texture2d<float> ms [[texture(1)]],
                        sampler ms_smp [[sampler(0)]],
                        constant SkyParams& p [[buffer(0)]],
                        uint3 gid [[thread_position_in_grid]]) {
    const uint n = p.size.x;
    if (gid.x >= n || gid.y >= n || gid.z >= 6) return;

    // Texel centre to a direction, in Metal's cube face order:
    // +X -X +Y -Y +Z -Z, with the same handedness the sampler uses. Getting
    // this wrong is the classic cubemap bug and it does not look wrong -- it
    // looks like the sky is rotated, which reads as an art decision.
    const float2 uv = (float2(gid.xy) + 0.5) / float(n) * 2.0 - 1.0;
    float3 dir;
    switch (gid.z) {
        case 0: dir = float3( 1.0, -uv.y, -uv.x); break;
        case 1: dir = float3(-1.0, -uv.y,  uv.x); break;
        case 2: dir = float3( uv.x,  1.0,  uv.y); break;
        case 3: dir = float3( uv.x, -1.0, -uv.y); break;
        case 4: dir = float3( uv.x, -uv.y,  1.0); break;
        default: dir = float3(-uv.x, -uv.y, -1.0); break;
    }
    dir = normalize(dir);
    out.write(float4(SkyRadiance(dir, p, ms, ms_smp), 1.0), gid.xy, gid.z);
}

// --- drawing the sky behind the scene ---------------------------------------

struct SkyVertexOut {
    float4 position [[position]];
    float3 dir;
};

struct SkyDrawParams {
    float4x4 invViewProj;
    float4 eye;        // .xyz world position, .w intensity
};

// A FULLSCREEN TRIANGLE, not a cube. Drawing a box around the camera needs the
// box to be big enough not to clip and small enough not to lose depth
// precision, and it wastes fragments on the parts of each face that fall
// outside the view. Unprojecting a screen-filling triangle has neither problem
// and no vertex buffer at all.
vertex SkyVertexOut vs_sky(uint vid [[vertex_id]],
                           constant SkyDrawParams& p [[buffer(0)]]) {
    const float2 pos[3] = {float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0)};
    SkyVertexOut o;
    // REVERSED-Z: the far plane is zero, and the sky is at the far plane. A one
    // here would put it at the near plane and it would occlude the entire
    // scene, which is the sort of bug that looks like the scene failed to draw.
    o.position = float4(pos[vid], 0.0, 1.0);
    // Unproject the two ends of the ray through this pixel. Taking a single
    // far-plane point and subtracting the eye is the usual shortcut and it
    // breaks under an orthographic projection, where every ray is parallel and
    // the eye is not on any of them.
    const float4 near_h = p.invViewProj * float4(pos[vid], 1.0, 1.0);
    const float4 far_h = p.invViewProj * float4(pos[vid], 0.0, 1.0);
    // CROSS-MULTIPLIED, not two perspective divides. The obvious form is
    //     far.xyz / far.w - near.xyz / near.w
    // and under this projection far.w is EXACTLY ZERO for every pixel of every
    // camera: reversed-Z with an infinite far plane puts the far plane at
    // w = 0, which is the definition of a point at infinity. That divide is
    // 0/0 and +inf/-inf, the varying interpolates to NaN, and a cube sampled at
    // NaN returns one arbitrary texel -- a sky of a single flat colour.
    //
    // Multiplying through by far.w * near.w clears both divides. The result is
    // the same vector scaled by a positive number, and the fragment normalises
    // anyway, so the scale is free. Still correct under an orthographic
    // projection, where both w are finite and neither term drops out.
    o.dir = far_h.xyz * near_h.w - near_h.xyz * far_h.w;
    return o;
}

fragment float4 fs_sky(SkyVertexOut in [[stage_in]],
                       texturecube<float> env [[texture(0)]],
                       sampler samp [[sampler(0)]],
                       constant SkyDrawParams& p [[buffer(0)]]) {
    const float3 dir = normalize(in.dir);
    // Mip zero explicitly. The sky fills the screen and its derivatives across
    // a triangle that covers everything are enormous at the silhouette, so
    // letting the hardware choose picks a blurry level along the seams.
    const float3 radiance = env.sample(samp, dir, level(0.0)).rgb;
    return float4(radiance * p.eye.w, 1.0);
}
