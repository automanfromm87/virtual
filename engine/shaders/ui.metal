// The UI's one pipeline: textured, tinted, alpha-blended quads in pixel space.
//
// One pipeline for text AND solid rectangles, which is why every vertex carries
// a uv even when it is a plain colour: a solid quad points at a single white
// texel in the atlas, so the shader multiplies by one and needs no branch and
// no second pipeline. The alternative is two pipelines and a state change every
// time a label sits on a panel.

// Included directly, unlike every other shader here: those are concatenated
// after shader_types.h, which brings in metal_stdlib for them. This one shares
// no types with the CPU and is compiled alone, so it brings in its own.
#include <metal_stdlib>
using namespace metal;

struct UiVertex {
    float2 position;  // PIXELS, origin top left
    float2 uv;        // atlas texels, normalised
    float4 colour;
};

struct UiOut {
    float4 position [[position]];
    float2 uv;
    float4 colour;
};

vertex UiOut vs_ui(uint vid [[vertex_id]],
                   device const UiVertex* verts [[buffer(0)]],
                   constant float2& viewport [[buffer(1)]])
{
    const UiVertex v = verts[vid];
    UiOut o;
    // Pixels to clip space. The y flip is here rather than at every call site:
    // the framebuffer, the cursor and this layer all agree that y grows
    // downward, and clip space is the only thing that disagrees.
    o.position = float4(v.position.x / viewport.x * 2.0f - 1.0f,
                        1.0f - v.position.y / viewport.y * 2.0f, 0.0f, 1.0f);
    o.uv = v.uv;
    o.colour = v.colour;
    return o;
}

fragment float4 fs_ui(UiOut in [[stage_in]],
                      texture2d<float> atlas [[texture(0)]],
                      sampler smp [[sampler(0)]])
{
    // The atlas holds COVERAGE, not colour, so it multiplies the alpha and
    // leaves the hue alone. One atlas therefore serves text in every colour
    // anything is ever drawn in.
    const float coverage = atlas.sample(smp, in.uv).r;
    return float4(in.colour.rgb, in.colour.a * coverage);
}
