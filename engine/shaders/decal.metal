// DEFERRED DECALS: a bullet hole, a puddle, a crack, a poster -- projected onto
// whatever geometry is already there.
//
// WHY NOT JUST DRAW A QUAD ON THE WALL. Because the wall is not flat. A quad
// placed on a surface z-fights with it, does not follow a curve, does not wrap
// around a corner, and has to be placed by whatever code knew where the surface
// was -- which for a bullet hole means the impact point AND the exact local
// orientation AND a guarantee that nothing else is within a few centimetres.
//
// A projected decal has none of those problems. It draws a BOX, and for every
// pixel the box covers it reconstructs the world position from the depth buffer
// and asks whether that position is inside the box. If it is, the decal's
// texture is sampled by where the position sits in the box's own space, and the
// result is blended into the G-buffer's albedo. The decal lands on whatever was
// there, at whatever angle, across as many surfaces as the box touches.
//
// WHAT IT COSTS. It only works on the deferred path, because it writes into the
// G-buffer after the geometry and before the lighting -- there is no equivalent
// moment in a forward renderer. And it is fill-rate: a large decal box covers a
// lot of screen, and every pixel does the reconstruction whether or not it ends
// up inside.

struct DecalParams {
    // Clip -> world, for turning the depth buffer back into positions.
    float4x4 invViewProj;
    float4x4 viewProj;
    float4 screen;  // .xy size, .zw 1/size
};

// One decal. Instanced, so a scene's decals are one draw call.
struct GpuDecal {
    // World -> the decal's own space, where the box is the unit cube centred on
    // the origin. An inverse rather than the forward transform because that is
    // the direction the shader needs, and inverting a matrix per pixel is not
    // an option.
    float4x4 invModel;
    float4x4 model;
    float4 tint;
    // .x normal-fade cosine, .y opacity, .z atlas index, .w unused.
    float4 params;
};

struct DecalOut {
    float4 position [[position]];
    // The decal this fragment belongs to. Flat, because interpolating an index
    // between vertices of different instances would be meaningless -- and
    // without the qualifier the compiler is entitled to do exactly that.
    uint index [[flat]];
};

// A UNIT CUBE from the vertex id, so there is no vertex buffer. Fourteen
// vertices as a triangle strip is the shortest cube there is; this uses 36 as a
// list, because the indirect and instanced paths here all draw lists and one
// primitive type is one fewer thing for a pipeline to disagree about.
constant float3 kCubeVerts[36] = {
    // -Z
    float3(-0.5,-0.5,-0.5), float3( 0.5, 0.5,-0.5), float3( 0.5,-0.5,-0.5),
    float3(-0.5,-0.5,-0.5), float3(-0.5, 0.5,-0.5), float3( 0.5, 0.5,-0.5),
    // +Z
    float3(-0.5,-0.5, 0.5), float3( 0.5,-0.5, 0.5), float3( 0.5, 0.5, 0.5),
    float3(-0.5,-0.5, 0.5), float3( 0.5, 0.5, 0.5), float3(-0.5, 0.5, 0.5),
    // -X
    float3(-0.5,-0.5,-0.5), float3(-0.5,-0.5, 0.5), float3(-0.5, 0.5, 0.5),
    float3(-0.5,-0.5,-0.5), float3(-0.5, 0.5, 0.5), float3(-0.5, 0.5,-0.5),
    // +X
    float3( 0.5,-0.5,-0.5), float3( 0.5, 0.5,-0.5), float3( 0.5, 0.5, 0.5),
    float3( 0.5,-0.5,-0.5), float3( 0.5, 0.5, 0.5), float3( 0.5,-0.5, 0.5),
    // -Y
    float3(-0.5,-0.5,-0.5), float3( 0.5,-0.5,-0.5), float3( 0.5,-0.5, 0.5),
    float3(-0.5,-0.5,-0.5), float3( 0.5,-0.5, 0.5), float3(-0.5,-0.5, 0.5),
    // +Y
    float3(-0.5, 0.5,-0.5), float3(-0.5, 0.5, 0.5), float3( 0.5, 0.5, 0.5),
    float3(-0.5, 0.5,-0.5), float3( 0.5, 0.5, 0.5), float3( 0.5, 0.5,-0.5),
};

vertex DecalOut vs_decal(uint vid [[vertex_id]], uint iid [[instance_id]],
                         constant DecalParams& p [[buffer(0)]],
                         device const GpuDecal* decals [[buffer(1)]]) {
    DecalOut o;
    const float3 local = kCubeVerts[vid % 36];
    const float4 world = decals[iid].model * float4(local, 1.0);
    o.position = p.viewProj * world;
    o.index = iid;
    return o;
}

fragment float4 fs_decal(DecalOut in [[stage_in]],
                         depth2d<float> depth [[texture(0)]],
                         texture2d<float> normal_metal [[texture(1)]],
                         texture2d<float> decal_tex [[texture(2)]],
                         constant DecalParams& p [[buffer(0)]],
                         device const GpuDecal* decals [[buffer(1)]],
                         sampler smp [[sampler(0)]]) {
    const float2 uv = in.position.xy * p.screen.zw;
    const float d = depth.sample(smp, uv);
    // Nothing was drawn here, so there is no surface to project onto. A decal
    // that painted the sky would be a rectangle floating in the air.
    if (d <= 0.0) discard_fragment();

    const float4 clip = float4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, d, 1.0);
    const float4 world4 = p.invViewProj * clip;
    const float3 world = world4.xyz / world4.w;

    const GpuDecal dec = decals[in.index];
    const float3 local = (dec.invModel * float4(world, 1.0)).xyz;
    // OUTSIDE THE BOX. The box's front faces are what got us here; the surface
    // behind them may be anywhere. This is the actual projection test and it is
    // the whole technique.
    if (any(abs(local) > 0.5)) discard_fragment();

    // THE NORMAL FADE, and without it a decal sprayed on a floor also appears,
    // stretched into stripes, on every wall the box happens to intersect. A
    // bullet hole projected onto a surface facing 80 degrees away from the
    // projection is not a bullet hole, it is a smear.
    const float3 n = normalize(normal_metal.sample(smp, uv).xyz);
    // The decal's own forward is its local -Z, taken to world space.
    const float3 forward = normalize(-dec.model[2].xyz);
    const float facing = dot(n, -forward);
    if (facing < dec.params.x) discard_fragment();
    // Faded rather than cut, so the edge of the acceptable range is not a hard
    // line across a curved surface.
    const float fade = smoothstep(dec.params.x, mix(dec.params.x, 1.0, 0.35), facing);

    // Local space to texture coordinates. +Y is up in the box and v = 0 is the
    // top of the image, hence the flip -- getting it wrong shows a decal upside
    // down, which for a symmetric texture is invisible until someone uses an
    // arrow.
    const float2 tex_uv = float2(local.x + 0.5, 0.5 - local.y);
    float4 texel = decal_tex.sample(smp, tex_uv);
    texel *= dec.tint;
    texel.a *= dec.params.y * fade;

    // EDGE FADE inside the box, so a decal whose texture reaches its border
    // does not end in a straight cut. Cheap and it is what makes a scorch mark
    // read as a scorch mark.
    const float2 edge = 1.0 - saturate((abs(local.xy) - 0.35) / 0.15);
    texel.a *= min(edge.x, edge.y);

    if (texel.a <= 0.001) discard_fragment();
    return texel;
}
