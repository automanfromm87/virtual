// No #include here: shader_types.h is textually prepended by the loader.
//
// Fullscreen composite. Exists to give the render graph a SECOND pass with a
// real dependency on the first — a graph that orders one pass is a no-op.

struct CompositeOut {
    float4 position [[position]];
    float2 uv;
};

// No vertex buffer: one oversized triangle generated from the vertex id covers
// the whole viewport with three vertices instead of a quad's six, and avoids
// the diagonal seam a two-triangle quad puts down the middle.
vertex CompositeOut vs_composite(uint vid [[vertex_id]])
{
    const float2 corners[3] = {float2(-1.0f, -1.0f), float2(3.0f, -1.0f),
                               float2(-1.0f, 3.0f)};
    CompositeOut o;
    o.position = float4(corners[vid], 0.0f, 1.0f);
    o.uv = corners[vid] * 0.5f + 0.5f;
    // NDC +Y is the top of the screen, but texture v=0 is the FIRST row, which
    // is also the top. Flip so the copy is not upside down.
    o.uv.y = 1.0f - o.uv.y;
    return o;
}

// Everything the grade needs, in one block. Separate from FrameUniforms because
// none of it means anything to a geometry pass, and putting display-referred
// parameters in the same struct as the lighting is how a renderer ends up with
// a surface shader that knows about contrast.
struct GradeParams {
    // .x lift, .y gamma, .z gain -- each a scalar applied to all channels after
    // the per-channel ones below. .w contrast pivot.
    float4 tone;
    float4 lift;        // .rgb per-channel shadows, .w unused
    float4 gamma;       // .rgb per-channel midtones
    float4 gain;        // .rgb per-channel highlights
    // .x contrast, .y saturation, .z temperature, .w tint
    float4 look;
    // .x output mode (0 SDR, 1 extended linear, 2 PQ), .y display headroom in
    // multiples of reference white, .z where the highlight roll-off starts,
    // .w reference white in nits (PQ only).
    float4 output;
};

fragment float4 fs_composite(CompositeOut in [[stage_in]],
                             constant FrameUniforms& u [[buffer(1)]],
                             constant GradeParams& g [[buffer(2)]],
                             device const float* exposure [[buffer(3)]],
                             texture2d<float> src   [[texture(0)]],
                             texture2d<float> ao    [[texture(1)]],
                             texture2d<float> bloom [[texture(2)]],
                             sampler          smp   [[sampler(0)]])
{
    // A real sampler object now, bound by the renderer. This used to be a
    // constexpr sampler declared inline — a workaround for the RHI not having
    // samplers at all.
    float3 color = src.sample(smp, in.uv).rgb;

    // Ambient occlusion, applied to the whole image rather than to the ambient
    // term alone. Strictly that is wrong — direct sunlight should not be
    // occluded by a nearby wall — but the ambient term is not separable without
    // a second render target, and indoors, where ambient dominates, the error
    // is far smaller than the gain. Defaults to a 1x1 white texture when the
    // SSAO pass is not in the graph.
    color *= ao.sample(smp, in.uv).r;

    // BLOOM, added while everything is still linear. Adding it after the tone
    // map would put a haze over the highlights instead of light around them —
    // the curve has already flattened that range, so the sum lands in the same
    // few codes. u.lighting.y is the strength; a null texture reads black and
    // the term vanishes.
    color += bloom.sample(smp, in.uv).rgb * u.lighting.y;

    // A vignette, on purpose: it is a visible, measurable effect, so a test can
    // prove the composite pass actually ran rather than assuming it did.
    const float2 centred = in.uv * 2.0f - 1.0f;
    const float r = length(centred);
    color *= mix(1.0f, mix(1.0f, 0.62f, u.lighting.z),
                 smoothstep(0.65f, 1.35f, r));

    // EXPOSURE, applied while the image is still linear and BEFORE the tone
    // map. That order is the whole point: exposure decides which part of the
    // scene's range the curve is going to see, and applying it afterwards would
    // only brighten an image that had already been compressed -- which is what
    // a brightness slider does, and why a brightness slider cannot rescue a
    // blown-out sky.
    //
    // The value comes from a BUFFER the GPU wrote, so an automatic exposure
    // never travels to the CPU and back.
    color *= max(exposure[0], 1e-4f);

    // TONE MAP, once, at the end of the frame.
    //
    // This used to live in the surface shader, which meant every pass after it
    // was working on display-referred colour that had already been clamped to
    // one. Doing it here is what makes an HDR scene target worth having.
    //
    // g.output.x picks the destination:
    //   0  SDR. The ACES curve clamped to 0..1 and gamma-encoded. What a
    //      conventional display wants.
    //   1  extended-range linear (scRGB). NO clamp and NO gamma: the display
    //      pipeline is linear and 1.0 means the reference white, so a specular
    //      highlight at 6.0 is six times reference white and stays that way.
    //   2  Rec.2100 PQ. Absolute luminance in nits through the SMPTE ST 2084
    //      curve, which is what an HDR10 signal is.
    //
    // The KEY POINT, and the reason this is not just "skip the clamp": an SDR
    // frame has already thrown the highlights away by the time it is written,
    // and no amount of expanding it afterwards puts them back. The tone curve
    // has to know what it is mapping to.
    const int mode = int(g.output.x + 0.5f);
    if (mode == 0) {
        // THE KHRONOS PBR NEUTRAL CURVE, and the reason it replaced a
        // per-channel fit of ACES is one measurement.
        //
        // A per-channel curve compresses each channel on its own, so the
        // strongest channel of a saturated colour is squeezed hardest and the
        // weakest barely at all. The ratio between them -- which IS the hue --
        // is destroyed in proportion to how bright the pixel is. Measured as
        // the fraction of the input's saturation still present at a matched
        // output brightness:
        //
        //     output peak      0.50    0.70    0.85    0.95
        //     ACES per-channel 0.945   0.660   0.385   0.175
        //     this curve       1.080   1.057   1.036   0.924
        //
        // world2's grass sat at 0.84 and kept a third of its green. It did not
        // clip -- nothing in that frame was above 250 -- it simply had its hue
        // dissolved by the curve, which reads as haze and cannot be fixed by
        // exposure, because exposure only chooses where on the curve to sit.
        //
        // The compression is applied to the PEAK CHANNEL and the others are
        // scaled with it, so hue survives by construction rather than by fit.
        // Highlights still go white, which is correct -- a bright enough
        // surface does -- but only in the top few percent, and by an explicit
        // desaturation term rather than as a side effect.
        constexpr float kStart = 0.76f;   // below this the curve is the identity
        constexpr float kDesat = 0.15f;   // how fast highlights roll to white

        // A small black offset, so the darkest channel reaches zero rather than
        // sitting on a grey floor. Quadratic near zero to avoid a visible kink.
        const float dark = min(color.r, min(color.g, color.b));
        color -= dark < 0.08f ? dark - 6.25f * dark * dark : 0.04f;

        const float peak = max(color.r, max(color.g, color.b));
        if (peak >= kStart) {
            const float d = 1.0f - kStart;
            // A hyperbola through (kStart, kStart) that approaches 1 without
            // reaching it, so there is headroom for any finite input.
            const float mapped = 1.0f - d * d / (peak + d - kStart);
            color *= mapped / peak;
            // Then toward white, by how far the peak had to be compressed.
            const float w = 1.0f - 1.0f / (kDesat * (peak - mapped) + 1.0f);
            color = mix(color, float3(mapped), w);
        }
        color = saturate(color);
    } else {
        // A SOFT SHOULDER instead of the SDR curve, rolling off only above the
        // headroom the display actually has. Below reference white the mapping
        // is the identity, so an SDR-looking image on an HDR display looks the
        // same rather than washed out -- which is the failure everyone
        // remembers from early HDR games.
        const float peak = max(g.output.y, 1.0f);   // display headroom, x white
        const float knee = min(g.output.z, peak * 0.99f);
        const float3 over = max(color - knee, 0.0f);
        const float range = max(peak - knee, 1e-3f);
        color = min(color, knee) + range * over / (range + over);
    }

    // --- the grade, in display-referred space ---------------------------------
    //
    // AFTER the tone map, and that is not the only defensible choice but it is
    // the one that matches how a colourist works: the curve has already decided
    // what white and black are, and the grade adjusts the picture between them.
    // Grading before the tone map means every adjustment also changes how the
    // curve compresses, so lifting the shadows desaturates the highlights.
    //
    // PARAMETRIC rather than a lookup table. A 3D LUT is what a film pipeline
    // uses because it can carry a look authored in a colour grading application;
    // this engine has no such application and no assets, so a table would be a
    // texture holding the same arithmetic done offline, plus interpolation
    // error. The parametric form is exact and can be animated.

    // WHITE BALANCE first, because it is a property of the light rather than a
    // look. Temperature shifts blue against red; tint shifts green against
    // magenta, which is the axis fluorescent lighting moves along.
    const float temp = g.look.z;
    color *= float3(1.0f + temp * 0.20f, 1.0f + g.look.w * 0.12f, 1.0f - temp * 0.20f);

    // LIFT, GAMMA, GAIN -- the standard three-way control. Lift moves the black
    // point, gain the white point, and gamma the curve between them without
    // moving either end.
    color = color * (g.gain.rgb * g.tone.z) + (g.lift.rgb + g.tone.x);
    color = max(color, 0.0f);
    color = pow(color, 1.0f / max(g.gamma.rgb * g.tone.y, 1e-3f));

    // CONTRAST about a pivot, not about zero. Scaling about zero darkens
    // everything as contrast rises, because every value moves away from black
    // rather than away from mid grey. 0.435 is 18% grey after the tone map.
    color = (color - g.tone.w) * g.look.x + g.tone.w;

    // SATURATION last, against the tone-mapped luminance. Rec.709 weights, so
    // desaturating leaves the perceived brightness alone -- an equal-weight
    // average would make greens go dark and blues go light.
    const float luma = dot(color, float3(0.2126f, 0.7152f, 0.0722f));
    color = mix(float3(luma), color, g.look.y);

    if (mode == 0) {
        color = saturate(color);
        // The display's own transfer function, undone. 2.2 rather than the
        // piecewise sRGB curve: the difference is confined to the darkest few
        // codes and the whole pipeline is consistent about it.
        color = pow(color, 1.0f / 2.2f);
    } else if (mode == 2) {
        // SMPTE ST 2084, the PQ curve. Its input is ABSOLUTE luminance in nits
        // divided by 10000, which is why the reference white has to be stated
        // in nits -- unlike every other stage here, PQ is not relative to
        // anything and 0.5 means a specific brightness.
        const float3 nits = max(color, 0.0f) * g.output.w;
        const float3 y = nits / 10000.0f;
        const float m1 = 0.1593017578125f, m2 = 78.84375f;
        const float c1p = 0.8359375f, c2 = 18.8515625f, c3 = 18.6875f;
        const float3 yp = pow(y, m1);
        color = pow((c1p + c2 * yp) / (1.0f + c3 * yp), m2);
    }
    // Mode 1 falls through unencoded, which is what extended-range linear
    // means: the value IS the light, in units of reference white.

    return float4(color, 1.0f);
}
