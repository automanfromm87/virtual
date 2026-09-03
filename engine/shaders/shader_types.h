// The CPU/GPU layout contract. Included as a real header by C++, and textually
// prepended to every shader before runtime compilation.
//
// RULE: every type used below must be byte-identical on both sides. When in
// doubt, use float4 and pad by hand. Never put a bare 3-float struct here —
// MSL's float3 occupies 16 bytes, not 12.
#ifndef ENGINE_SHADER_TYPES_H
#define ENGINE_SHADER_TYPES_H

#ifdef __METAL_VERSION__
#  include <metal_stdlib>
using namespace metal;
#  define ENG_MAT4 float4x4
#  define ENG_VEC4 float4
#  define ENG_VEC3 float3  // 16 bytes
#  define ENG_UVEC4 uint4
#else
#  include "engine/core/math.h"
#  define ENG_MAT4 ::eng::Mat4
#  define ENG_VEC4 ::eng::Vec4
#  define ENG_VEC3 ::eng::Vec4  // deliberately Vec4: matches MSL float3's 16 bytes
struct EngUVec4 { std::uint32_t x = 0, y = 0, z = 0, w = 0; };
#  define ENG_UVEC4 ::EngUVec4
#endif

struct FrameUniforms {
    ENG_MAT4 viewProj;
    ENG_MAT4 model;  // object -> world. Normals use this too (rotation only).
    // World -> the light's clip space. Orthographic, because a directional
    // light's rays are parallel and have no vanishing point.
    ENG_MAT4 lightViewProj;
    ENG_VEC4 tint;   // per-instance colour multiplier
    // Unit vector pointing FROM the surface TOWARD the light, in world space.
    // Named for where the light IS, not where its photons go — the other
    // convention costs you a sign error every single time.
    ENG_VEC4 lightDir;
    ENG_VEC4 lightColor;  // rgb = radiance, .w unused

    // --- material ------------------------------------------------------------
    ENG_VEC4 baseColor;  // multiplied by the albedo map and by tint
    // .x roughness, .y metallic, .z shadows-on (0 or 1), .w SECTION CUT height
    // in world Y — fragments above it are discarded. Roughness
    // and metallic are multiplied by the corresponding map, which defaults to
    // 1x1 white — so an untextured material needs no branch in the shader.
    ENG_VEC4 surface;
    // World-space camera position. Diffuse-only shading never needed it;
    // anything with a specular lobe does, because the highlight depends on
    // where you are standing.
    ENG_VEC4 eyePos;
    // Depth reconstruction for the SSAO pass: .x nearZ, .y 1/tan(fovY/2),
    // .z aspect, .w sample radius in world metres.
    ENG_VEC4 ssao;
    // .x how many entries of the light buffer are live. The rest is reserved.
    ENG_VEC4 lighting;
    // Hemisphere ambient: sky above, bounce below. Scene-controlled, because
    // the right value differs by an order of magnitude between a daylit
    // exterior and a room lit by lamps — and when it is wrong the wrong way,
    // it does the lighting and every actual light in the scene stops mattering.
    ENG_VEC4 ambientSky;
    ENG_VEC4 ambientGround;
};

// A local light. Sixteen bytes times four, so the array packs with no padding
// and the C++ and MSL views cannot drift.
//
// The directional key light is NOT in here — it lives in FrameUniforms because
// it is the one that casts the shadow map, and it needs a matrix rather than a
// position. See the note on Scene::lights.
struct GpuLight {
    // .xyz world position, .w the type: 0 point, 1 spot.
    ENG_VEC4 position;
    // .xyz the direction the light SHINES, from the lamp into the scene, which
    // is the opposite of Scene::lightDir. A directional light has no position,
    // so the only useful thing to record is where it is; a spot has one, so the
    // only useful thing to record is where it aims. .w is the range in metres.
    ENG_VEC4 direction;
    ENG_VEC4 color;  // .rgb radiance at one metre
    // .x cos(inner cone), .y cos(outer cone). Equal for a point light, where
    // they are ignored.
    ENG_VEC4 cone;
};

// Lights the fragment stage can read in one pass. A forward renderer pays for
// every light on every fragment, so this is a budget rather than a limit of the
// format — past a few dozen the answer is to cluster them, not to raise this.
#define ENG_MAX_LIGHTS 32

// Skinning attributes, in their OWN buffer rather than inside VertexIn. A
// static mesh vastly outnumbers a skinned one in most scenes, and adding this
// to every vertex would cost 32 bytes each for four weights that are all zero.
//
// 32-bit joint indices where 16 would do: an MSL ushort4 is 8 bytes and still
// forces the float4 that follows onto a 16-byte boundary, so the smaller field
// buys nothing but a chance to get the padding wrong.
struct SkinIn {
    ENG_UVEC4 joints;
    ENG_VEC4 weights;
};

struct VertexIn {
    ENG_VEC3 position;  // .w unused on the C++ side
    ENG_VEC3 normal;    // object space, unit length; .w unused
    ENG_VEC4 color;
    // .xy = texture coordinates, origin top-left to match Metal's texture
    // convention. .zw reserved (tangent sign / a second uv set later).
    //
    // A bare float2 would be tempting, but MSL would align the struct to 16
    // anyway and the padding lands somewhere less useful. The unused .w lanes
    // of position and normal cannot be used instead: they are float3 on the
    // MSL side, which occupies 16 bytes but exposes only .xyz.
    ENG_VEC4 uv;
};

#endif  // ENGINE_SHADER_TYPES_H
