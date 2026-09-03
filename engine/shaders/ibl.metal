// Image-based lighting: turning an environment map into the two things a PBR
// shader can actually use.
//
// THE PROBLEM. The reflectance equation integrates incoming radiance over the
// hemisphere, weighted by the BRDF. Doing that per pixel per frame would mean
// thousands of samples of the environment for every fragment. The whole of IBL
// is the observation that the integral depends on only a few parameters, so it
// can be precomputed -- once, when the environment changes -- and looked up.
//
// It splits in two because the BRDF does:
//
//   DIFFUSE is Lambertian, so the BRDF is a constant and the integral collapses
//   to a cosine-weighted average of the environment about the normal. That
//   depends on the normal ALONE, so it fits in one low-resolution cubemap. 32
//   pixels a side is plenty: the result is a very smooth function of direction,
//   and this is why an irradiance map looks like a blurry mess and is correct.
//
//   SPECULAR depends on the normal, the view direction, and the roughness --
//   three things, which is a four-dimensional table nobody can afford. Karis's
//   SPLIT-SUM approximation is what makes it tractable: assume the view
//   direction equals the normal, which lets the environment term be prefiltered
//   into a mip chain indexed by roughness, and put everything that depends on
//   the view into a separate two-dimensional lookup table that does not depend
//   on the environment at all.
//
// The assumption N = V = R is wrong at grazing angles -- it loses the stretched
// highlight you see reflecting off a wet road -- and it is the standard trade
// because the alternative costs an order of magnitude more memory.

constant float kPi = 3.14159265358979;

struct CubeParams {
    uint4 size;      // .x face size of the OUTPUT, .y source mip count, .zw unused
    float4 tune;     // .x roughness, .y source face size, .z sample count, .w unused
};

// Cube face texel -> direction. The one place the face order is written down;
// everything else calls this.
static float3 FaceDirection(uint face, uint2 xy, uint n) {
    const float2 uv = (float2(xy) + 0.5) / float(n) * 2.0 - 1.0;
    float3 dir;
    switch (face) {
        case 0: dir = float3( 1.0, -uv.y, -uv.x); break;
        case 1: dir = float3(-1.0, -uv.y,  uv.x); break;
        case 2: dir = float3( uv.x,  1.0,  uv.y); break;
        case 3: dir = float3( uv.x, -1.0, -uv.y); break;
        case 4: dir = float3( uv.x, -uv.y,  1.0); break;
        default: dir = float3(-uv.x, -uv.y, -1.0); break;
    }
    return normalize(dir);
}

// --- equirectangular source --------------------------------------------------

kernel void cs_equirect_to_cube(texture2d_array<float, access::write> out [[texture(0)]],
                                texture2d<float> src [[texture(1)]],
                                sampler samp [[sampler(0)]],
                                constant CubeParams& p [[buffer(0)]],
                                uint3 gid [[thread_position_in_grid]]) {
    const uint n = p.size.x;
    if (gid.x >= n || gid.y >= n || gid.z >= 6) return;
    const float3 d = FaceDirection(gid.z, gid.xy, n);
    // The standard latitude-longitude mapping. atan2(d.z, d.x) rather than
    // atan2(d.x, d.z) puts -Z at the centre of the image, which is where every
    // HDRI on earth puts the direction the photographer was facing.
    const float u = atan2(d.z, d.x) / (2.0 * kPi) + 0.5;
    const float v = acos(clamp(d.y, -1.0, 1.0)) / kPi;
    out.write(src.sample(samp, float2(u, v), level(0.0)), gid.xy, gid.z);
}

// --- diffuse irradiance ------------------------------------------------------

kernel void cs_irradiance(texture2d_array<float, access::write> out [[texture(0)]],
                          texturecube<float> env [[texture(1)]],
                          sampler samp [[sampler(0)]],
                          constant CubeParams& p [[buffer(0)]],
                          uint3 gid [[thread_position_in_grid]]) {
    const uint n = p.size.x;
    if (gid.x >= n || gid.y >= n || gid.z >= 6) return;
    const float3 normal = FaceDirection(gid.z, gid.xy, n);

    // An orthonormal basis about the normal. `up` is swapped near the poles
    // because a cross product with a parallel vector collapses -- and the
    // symptom is not subtle: two texels of the output cube come out black.
    const float3 up_ref = abs(normal.y) < 0.999 ? float3(0, 1, 0) : float3(1, 0, 0);
    const float3 right = normalize(cross(up_ref, normal));
    const float3 up = cross(normal, right);

    // UNIFORM STEPS in spherical coordinates, with the sin(theta) Jacobian
    // carried explicitly below. Not a random or a Hammersley sequence: the
    // integrand here is smooth and low frequency, so a regular grid converges
    // faster than sampling would and -- more usefully -- produces no noise at
    // all. An irradiance map with even slight noise shows up as blotches
    // crawling over every matte surface as the camera moves.
    const float delta = 0.025;
    float3 sum = float3(0.0);
    float weight = 0.0;
    for (float phi = 0.0; phi < 2.0 * kPi; phi += delta) {
        const float sin_phi = sin(phi), cos_phi = cos(phi);
        for (float theta = 0.0; theta < 0.5 * kPi; theta += delta) {
            const float sin_t = sin(theta), cos_t = cos(theta);
            const float3 tangent = float3(sin_t * cos_phi, sin_t * sin_phi, cos_t);
            const float3 dir = tangent.x * right + tangent.y * up + tangent.z * normal;
            // sin * cos is the two weights together: sin(theta) is the solid
            // angle of the ring, cos(theta) is Lambert's law. Dropping either
            // is a mistake that still produces a plausible-looking blur --
            // dropping the cosine makes the result too bright at grazing
            // angles, and nothing about the picture says which one is missing.
            sum += env.sample(samp, dir, level(0.0)).rgb * cos_t * sin_t;
            weight += cos_t * sin_t;
        }
    }
    // Normalised by the accumulated weight rather than by the analytic pi/N.
    // They agree only when the loop bounds divide exactly, and `delta` does not
    // divide pi/2 -- so the analytic constant would leave the result a few
    // percent bright, which is exactly the sort of error that survives review
    // because it looks fine.
    out.write(float4(weight > 0.0 ? sum / weight : float3(0.0), 1.0), gid.xy, gid.z);
}

// --- prefiltered specular ----------------------------------------------------

// Van der Corput radical inverse: the second dimension of a Hammersley set.
// Bit-reversal, which is what makes the sequence stratified at every prefix
// length rather than only at the end.
static float RadicalInverse(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

static float2 Hammersley(uint i, uint count) {
    return float2(float(i) / float(count), RadicalInverse(i));
}

// A half-vector drawn from the GGX distribution, so that samples land where the
// BRDF is large instead of uniformly over a hemisphere it mostly ignores. At
// roughness 0.1 uniform sampling would put fewer than one sample in a thousand
// inside the lobe.
static float3 ImportanceSampleGGX(float2 xi, float3 normal, float roughness) {
    const float a = roughness * roughness;
    const float phi = 2.0 * kPi * xi.x;
    const float cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    const float sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
    const float3 h_tangent =
        float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
    const float3 up_ref = abs(normal.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
    const float3 tx = normalize(cross(up_ref, normal));
    const float3 ty = cross(normal, tx);
    return normalize(tx * h_tangent.x + ty * h_tangent.y + normal * h_tangent.z);
}

static float DistributionGGX(float n_dot_h, float roughness) {
    const float a = roughness * roughness;
    const float a2 = a * a;
    const float d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / max(kPi * d * d, 1e-7);
}

kernel void cs_prefilter(texture2d_array<float, access::write> out [[texture(0)]],
                         texturecube<float> env [[texture(1)]],
                         sampler samp [[sampler(0)]],
                         constant CubeParams& p [[buffer(0)]],
                         uint3 gid [[thread_position_in_grid]]) {
    const uint n = p.size.x;
    if (gid.x >= n || gid.y >= n || gid.z >= 6) return;
    const float roughness = p.tune.x;
    const float3 normal = FaceDirection(gid.z, gid.xy, n);

    // Mip zero is a MIRROR, and copying it rather than integrating it is not an
    // optimisation. At roughness 0 the GGX lobe is a delta function, so every
    // importance sample returns the same direction and the sum is that texel
    // times N -- except that the mip selection below would still blur it,
    // because the solid angle of a delta lobe divided by the sample count is
    // not zero in floating point. The result is a "mirror" that is visibly soft.
    if (roughness <= 0.0) {
        out.write(env.sample(samp, normal, level(0.0)), gid.xy, gid.z);
        return;
    }

    const uint samples = uint(max(p.tune.z, 1.0));
    const float src_size = max(p.tune.y, 1.0);
    // The solid angle one source texel covers, for the mip selection below.
    const float texel_solid_angle = 4.0 * kPi / (6.0 * src_size * src_size);

    float3 sum = float3(0.0);
    float weight = 0.0;
    for (uint i = 0; i < samples; ++i) {
        const float2 xi = Hammersley(i, samples);
        const float3 h = ImportanceSampleGGX(xi, normal, roughness);
        // N = V is the split-sum assumption, so the view reflects to the normal
        // and the light direction is the normal mirrored about the half vector.
        const float3 l = normalize(2.0 * dot(normal, h) * h - normal);
        const float n_dot_l = dot(normal, l);
        if (n_dot_l <= 0.0) continue;

        // SAMPLE FROM A MIP, not from level zero. With a few hundred samples
        // spread over a wide lobe, reading full resolution means each sample is
        // a point read of a texture containing a sun that is a hundred times
        // brighter than its neighbours -- so whether a sample lands on it is
        // luck, and the result sparkles from texel to texel. Choosing the mip
        // whose texels match the sample's own solid angle prefilters that away.
        // This is the fix for the single most common IBL artefact.
        const float n_dot_h = max(dot(normal, h), 0.0);
        const float pdf = DistributionGGX(n_dot_h, roughness) * n_dot_h /
                          (4.0 * max(dot(h, l), 1e-4)) + 1e-4;
        const float sample_solid_angle = 1.0 / (float(samples) * pdf);
        const float mip = 0.5 * log2(sample_solid_angle / texel_solid_angle);

        sum += env.sample(samp, l, level(max(mip, 0.0))).rgb * n_dot_l;
        weight += n_dot_l;
    }
    out.write(float4(weight > 0.0 ? sum / weight : float3(0.0), 1.0), gid.xy, gid.z);
}

// --- the BRDF integration table ----------------------------------------------

// Smith's geometry term, IBL flavour. The k here is roughness^2/2 rather than
// the (roughness+1)^2/8 used for analytic lights, and they are genuinely
// different: the analytic one folds in a fudge that Disney introduced for
// direct lighting, and using it here makes the table wrong at low roughness.
static float GeometrySmithIBL(float n_dot_v, float n_dot_l, float roughness) {
    const float k = roughness * roughness / 2.0;
    const float gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
    const float gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return gv * gl;
}

// Independent of the environment entirely -- it is a property of the BRDF -- so
// it is baked once at startup and never again. The two channels are the SCALE
// and the BIAS applied to a surface's F0, which is what lets one table serve
// every material.
kernel void cs_brdf_lut(texture2d<float, access::write> out [[texture(0)]],
                        constant CubeParams& p [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    const uint n = p.size.x;
    if (gid.x >= n || gid.y >= n) return;
    // Both axes offset by half a texel, so the sampled value at the centre of
    // texel k is the integral AT that parameter rather than at its left edge.
    // Without the offset the table is half a texel wrong everywhere, which at
    // 256 entries is invisible and at 32 is a visible shift in the Fresnel.
    const float n_dot_v = max((float(gid.x) + 0.5) / float(n), 1e-3);
    const float roughness = (float(gid.y) + 0.5) / float(n);

    // The view vector, reconstructed in a frame where the normal is +Z. Any
    // frame will do: the integral is rotationally symmetric about the normal.
    const float3 v = float3(sqrt(1.0 - n_dot_v * n_dot_v), 0.0, n_dot_v);
    const float3 normal = float3(0.0, 0.0, 1.0);

    float scale = 0.0, bias = 0.0;
    constexpr uint kSamples = 1024;
    for (uint i = 0; i < kSamples; ++i) {
        const float2 xi = Hammersley(i, kSamples);
        const float3 h = ImportanceSampleGGX(xi, normal, roughness);
        const float3 l = normalize(2.0 * dot(v, h) * h - v);
        const float n_dot_l = max(l.z, 0.0);
        if (n_dot_l <= 0.0) continue;
        const float n_dot_h = max(h.z, 0.0);
        const float v_dot_h = max(dot(v, h), 0.0);

        const float g = GeometrySmithIBL(n_dot_v, n_dot_l, roughness);
        const float g_vis = g * v_dot_h / max(n_dot_h * n_dot_v, 1e-6);
        // Schlick's Fresnel, with F0 FACTORED OUT. That factoring is the whole
        // trick: F = F0 + (1-F0)*(1-cos)^5 is linear in F0, so the integral
        // splits into a term multiplying F0 and a term added to it, and those
        // two numbers are the same for every material.
        const float fc = pow(1.0 - v_dot_h, 5.0);
        scale += (1.0 - fc) * g_vis;
        bias += fc * g_vis;
    }
    out.write(float4(scale / float(kSamples), bias / float(kSamples), 0.0, 1.0), gid);
}

// --- readback ----------------------------------------------------------------

// Copies one face of one mip of a cube into a buffer, so a test can assert on
// it. Every property worth checking here -- energy conservation, the white
// furnace, whether the sun ended up in the right texel -- is a number, and a
// picture is the one form of evidence that cannot settle any of them.
kernel void cs_cube_readback(texturecube<float> env [[texture(0)]],
                             sampler samp [[sampler(0)]],
                             device float4* out [[buffer(0)]],
                             constant CubeParams& p [[buffer(1)]],
                             uint3 gid [[thread_position_in_grid]]) {
    const uint n = p.size.x;
    if (gid.x >= n || gid.y >= n || gid.z >= 6) return;
    const float3 dir = FaceDirection(gid.z, gid.xy, n);
    const float lod = p.tune.x;
    out[(gid.z * n + gid.y) * n + gid.x] = env.sample(samp, dir, level(lod));
}

kernel void cs_lut_readback(texture2d<float> lut [[texture(0)]],
                            sampler samp [[sampler(0)]],
                            device float4* out [[buffer(0)]],
                            constant CubeParams& p [[buffer(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
    const uint n = p.size.x;
    if (gid.x >= n || gid.y >= n) return;
    const float2 uv = (float2(gid) + 0.5) / float(n);
    out[gid.y * n + gid.x] = lut.sample(samp, uv, level(0.0));
}

// --- mip chain ---------------------------------------------------------------

// One level of the radiance cube from the one above it.
//
// Not MTLBlitCommandEncoder's generateMipmaps, which is the obvious answer and
// is wrong for a cube: it filters each face independently, so the texels along
// a face's edge average only the texels inside that face and lose the half of
// their footprint that lies on the neighbour. The seams show as bright or dark
// lines that get worse at every level -- and this chain is what the prefilter
// reads, so a seam in the chain becomes a seam in every reflection.
//
// Sampling by DIRECTION through the cube sampler instead lets the hardware do
// the cross-face filtering it already knows how to do. The child texel's centre
// falls exactly between four parent texels, so a bilinear read at the parent
// level is an equal-weight box filter -- exactly the right one for a 2x
// reduction -- everywhere except at an edge, where it becomes the correct
// cross-face blend instead of a wrong one-sided average.
kernel void cs_cube_downsample(texture2d_array<float, access::write> out [[texture(0)]],
                               texturecube<float> src [[texture(1)]],
                               sampler samp [[sampler(0)]],
                               constant CubeParams& p [[buffer(0)]],
                               uint3 gid [[thread_position_in_grid]]) {
    const uint n = p.size.x;
    if (gid.x >= n || gid.y >= n || gid.z >= 6) return;
    const float3 dir = FaceDirection(gid.z, gid.xy, n);
    out.write(src.sample(samp, dir, level(p.tune.x)), gid.xy, gid.z);
}
