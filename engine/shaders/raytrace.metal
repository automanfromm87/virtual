// RAY-TRACED SHADOWS.
//
// A shadow map answers "is this point lit" by rendering the scene from the
// light, storing the nearest distance per texel, and comparing. Everything
// awkward about shadow maps follows from that being an APPROXIMATION sampled on
// a grid that has nothing to do with the camera's:
//
//   * Resolution. One texel covers many pixels somewhere in the frame, and the
//     shadow edge is that texel's staircase. Cascades reduce it; nothing
//     removes it.
//   * BIAS. The stored depth is the surface's own, so a surface shadows itself
//     unless the comparison is offset -- and the offset that stops the acne is
//     the offset that detaches the shadow from the object's feet.
//   * Everything outside the map. A caster beyond the box casts nothing.
//
// A ray does not approximate. It asks the geometry directly, so there is no
// resolution, no bias to tune, and no box to fall out of. What it costs is a
// BVH over the scene, kept up to date as things move, and a traversal per pixel
// -- which is why this is a choice and not a replacement.
//
// This traces one ray toward the sun and returns visibility, which the lighting
// pass multiplies in exactly where it would have multiplied a shadow-map
// lookup.

#include <metal_raytracing>

using namespace metal;
using namespace raytracing;

struct RtOut {
    float4 position [[position]];
    float2 uv;
};

vertex RtOut vs_rt_shadow(uint vid [[vertex_id]])
{
    const float2 corner = float2((vid << 1) & 2, vid & 2);
    RtOut o;
    o.position = float4(corner * 2.0f - 1.0f, 0.0f, 1.0f);
    o.uv = float2(corner.x, 1.0f - corner.y);
    return o;
}

fragment float4 fs_rt_shadow(RtOut in [[stage_in]],
                             constant FrameUniforms& u [[buffer(1)]],
                             instance_acceleration_structure scene [[buffer(4)]],
                             depth2d<float> sceneDepth [[texture(0)]],
                             texture2d<float> gbufNormal [[texture(1)]],
                             sampler smp [[sampler(0)]])
{
    const float depth = sceneDepth.sample(smp, in.uv);
    // Background: nothing there to shadow, and fully lit is the value that
    // makes a later multiply a no-op.
    if (depth <= 0.0f) return float4(1.0f, 1.0f, 1.0f, 1.0f);

    float2 ndc = in.uv * 2.0f - 1.0f;
    ndc.y = -ndc.y;
    const float4 world4 = u.invViewProj * float4(ndc, depth, 1.0f);
    const float3 worldPos = world4.xyz / world4.w;
    const float3 N = normalize(gbufNormal.sample(smp, in.uv).xyz);
    const float3 L = normalize(u.lightDir.xyz);

    // Facing away from the light needs no ray: it is in shadow by geometry, and
    // tracing from a back face starts the ray inside the surface.
    const float ndotl = dot(N, L);
    if (ndotl <= 0.0f) return float4(0.0f, 0.0f, 0.0f, 1.0f);

    ray r;
    // Offset along the NORMAL, not along the ray. This is the one bias a ray
    // tracer still needs, and it is a very different thing from a shadow map's:
    // it exists only because the world position was reconstructed from a
    // quantised depth buffer and may sit a fraction of a millimetre inside the
    // surface. It does not scale with distance, it does not depend on the angle
    // to the light, and it does not detach the shadow -- which are the three
    // things that make shadow-map bias a permanent tuning problem.
    r.origin = worldPos + N * 1e-3f;
    r.direction = L;
    r.min_distance = 0.0f;
    r.max_distance = u.ssao.w > 0.0f ? u.ssao.w : 1e4f;

    // ACCEPT_ANY_INTERSECTION: a shadow ray does not care WHICH surface blocks
    // it or how far away that surface is, only that something does. Telling the
    // intersector that lets it stop at the first hit instead of finding the
    // nearest one, which is most of the cost.
    intersector<instancing, triangle_data> isect;
    isect.accept_any_intersection(true);
    isect.assume_geometry_type(geometry_type::triangle);

    const intersection_result<instancing, triangle_data> hit =
        isect.intersect(r, scene, 0xFF);
    const float visible = hit.type == intersection_type::none ? 1.0f : 0.0f;
    return float4(visible, visible, visible, 1.0f);
}
