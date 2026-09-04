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
    // Clip -> world, for reconstructing a world position from a depth buffer.
    // Deferred shading needs it; nothing else does, and it is here rather than
    // in its own block because a fullscreen lighting pass already binds this
    // one and a second block would be a second binding for one matrix.
    ENG_MAT4 invViewProj;
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
    // Unit vector the camera looks ALONG. Clustered lighting needs it to turn
    // a world position into a distance along the view axis, which is the axis
    // the depth slices are cut on -- using the straight-line distance from the
    // eye instead would make the fragment's idea of a cell spherical while the
    // binning pass's is planar, and the two disagree most at the screen edges.
    ENG_VEC4 viewDir;
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
    // .rgb emissive radiance, multiplied by the emissive map. .w is the normal
    // map's strength.
    //
    // Radiance and not a colour: it is added to the lit result AFTER everything
    // else and before the tone map, so a value of 40 is a filament and a value
    // of 1 is a surface no brighter than a lit wall. Anything at or below 1
    // will not bloom, because the bloom threshold is in the same linear units.
    ENG_VEC4 emissive;
    // The irradiance volume's placement. xyz = world origin of probe (0,0,0),
    // w = 1 when a volume is bound.
    ENG_VEC4 giOrigin;
    // xyz = cell size in metres, w unused.
    ENG_VEC4 giSpacing;
    // xyz = probe counts, w unused. Needed to turn a world position into a
    // texture coordinate: a 3D texture samples in 0..1, and the half-texel
    // inset is what stops the outermost probes being blended with nothing.
    ENG_VEC4 giCounts;
    // PARALLAX CORRECTION for the reflection probe. xyz = the box's minimum
    // corner, w = 1 when correction is on.
    //
    // A cubemap is a picture of the world from ONE point, and sampling it by
    // direction alone says every reflected thing is infinitely far away. That
    // is right for a sky and wrong for a room: a mirror ball moved across a
    // room reflects exactly the same walls in exactly the same places, so it
    // reads as a ball with a sky painted on it rather than as a ball in a room.
    ENG_VEC4 probeBoxMin;
    ENG_VEC4 probeBoxMax;   // xyz, w unused
    ENG_VEC4 probePosition; // where the cube was captured from
    // STEREO. The RIGHT eye's view-projection and its eye position; the left
    // eye's are the ordinary viewProj and eyePos above.
    //
    // Both in the per-draw block rather than a second one, because a stereo
    // pass needs them on the same draw as the left eye's -- the whole point of
    // vertex amplification is that there IS only one draw.
    ENG_MAT4 viewProjRight;
    ENG_VEC4 eyePosRight;
    // UV TILING for this material. .xy multiplies the mesh's uv, .zw offsets it.
    //
    // Needed because mesh UVs describe the SURFACE and a detail texture
    // describes the MATERIAL, and the two have no reason to be at the same
    // scale. A terrain chunk lays 0..1 across 128 metres; grass repeats every
    // few metres. Without this the only options are to bake the tiling into
    // every mesh that uses the material or to give up on detail textures.
    ENG_VEC4 uvScale;
    // .rgb the colour transmitted through a thin surface, .w how much. Zero w
    // is opaque, which is everything that is not a leaf.
    ENG_VEC4 transmission;
};

// A local light. Sixteen bytes times four, so the array packs with no padding
// and the C++ and MSL views cannot drift.
//
// The directional key light is NOT in here — it lives in FrameUniforms because
// it is the one that casts the shadow map, and it needs a matrix rather than a
// position. See the note on Scene::lights.
struct GpuLight {
    // World -> this light's clip space. Only meaningful when it has a shadow
    // tile; a point light would need six of these and does not get one yet.
    ENG_MAT4 viewProj;
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
    // Shadow map placement. .x the FIRST tile index this light owns, .y how
    // many tiles the atlas has per side, .z the near plane its maps were drawn
    // with, .w the kind: 0 none, 1 spot, 2 point.
    //
    // A tile INDEX rather than a uv rectangle, because a point light owns six
    // consecutive tiles and the shader picks among them from a direction. Six
    // matrices per light would also work and would cost 384 bytes each.
    ENG_VEC4 shadow;
};

// The directional light's cascades, bound once per pass. Not in FrameUniforms:
// four matrices is 256 bytes and every one of the scene's instances would carry
// a copy of them.
struct GpuCascades {
    ENG_MAT4 viewProj[4];
    // .x..w the far distance of each cascade, in view units. A fragment picks
    // the first one it fits inside.
    ENG_VEC4 splits;
    // .x how many are live, .y tiles per side of the shadow map, .z the size of
    // one tile in uv.
    ENG_VEC4 info;
};

// Lights the fragment stage can read in one pass.
//
// This is the size of the light BUFFER, not the number a fragment loops over.
// With clustering on, a fragment reads only the handful its cluster overlaps,
// so this is just how many lights a scene may contain at once.
#define ENG_MAX_LIGHTS 256

// CLUSTERED LIGHTING. The view frustum is cut into a 3D grid; a compute pass
// works out which lights touch which cell; a fragment reads only its own cell's
// list.
//
// Why the frustum and not the screen: a tile in screen space is an infinitely
// long wedge, and every light anywhere along it counts as overlapping. Slicing
// by depth as well turns that wedge into a box a few metres across, and the
// number of lights a fragment actually evaluates stops depending on how deep
// the scene is.
//
// 16 x 9 x 24 = 3456 cells. The x/y split matches a 16:9 frame so cells are
// roughly square; the depth split is EXPONENTIAL, because a linear one puts
// almost every cell beyond the distance where lights matter.
#define ENG_CLUSTER_X 16
#define ENG_CLUSTER_Y 9
#define ENG_CLUSTER_Z 24
#define ENG_CLUSTER_COUNT (ENG_CLUSTER_X * ENG_CLUSTER_Y * ENG_CLUSTER_Z)
// Per cell. 64 x 3456 x 4 bytes is 884 KB, which is nothing, and overflowing a
// cell drops lights rather than corrupting anything -- the binning pass stops
// writing when it is full.
#define ENG_CLUSTER_CAPACITY 64

// The grid's placement in the view frustum, bound once per pass.
//
// Its own block rather than more fields on FrameUniforms: FrameUniforms is
// written PER DRAW, and this is identical for every draw in a frame -- adding
// it there would copy 32 bytes per object for a value that never changes.
struct GpuClusters {
    // x tiles across, y tiles down, z depth slices, w the per-cell capacity.
    ENG_VEC4 grid;
    // x near plane, y the far distance the slices reach, z log(far/near)
    // precomputed, w screen width in pixels.
    ENG_VEC4 depth;
    // x screen height, y tile width in pixels, z tile height in pixels, w
    // how many lights the buffer holds.
    ENG_VEC4 screen;
    // x = 1/tan(fovY/2), y = aspect. The projection's two scale factors, which
    // are what turn a cell's NDC rectangle into view-space metres.
    ENG_VEC4 slope;
};

// FROXEL VOLUMETRICS. The grid's placement and the medium's properties, bound
// once per pass.
//
// 160 x 90 x 64 froxels is 921600 of them -- the same count as a 1280x720
// screen has pixels, but each one is doing a shadow lookup ONCE instead of once
// per step of a per-pixel ray march, and the trilinear sample in the apply pass
// filters the result for free.
#define ENG_FROXEL_X 160
#define ENG_FROXEL_Y 90
#define ENG_FROXEL_Z 64

struct GpuVolumetrics {
    // xyz = froxel counts, w unused.
    ENG_VEC4 grid;
    // x near, y far, z log(far/near) precomputed, w this frame's slice jitter
    // in 0..1.
    ENG_VEC4 range;
    // rgb = scattering coefficient per metre, a = extinction per metre.
    //
    // SEPARATE, not one coefficient. A medium that scatters as much as it
    // absorbs is a very particular medium; smoke absorbs far more than it
    // scatters and clean air the reverse, and tying them together makes dense
    // fog necessarily bright.
    ENG_VEC4 scatter;
    // rgb = ambient radiance in the medium, a unused. What stands in for light
    // that has already bounced -- without it the inside of a shadow is
    // perfectly black fog, which no real medium is.
    ENG_VEC4 ambient;
    // x = constant density per metre, y = the exponential term's density at
    // the reference height, z = that reference height in world Y, w = the
    // falloff rate per metre.
    ENG_VEC4 medium;
    // x = the Henyey-Greenstein anisotropy g. Above zero throws light forward,
    // which is why a shaft is bright looking toward the lamp and faint looking
    // away. yzw spare.
    ENG_VEC4 medium2;
};

// Skinning attributes, in their OWN buffer rather than inside VertexIn. A
// static mesh vastly outnumbers a skinned one in most scenes, and adding this
// to every vertex would cost 32 bytes each for four weights that are all zero.
//
// 32-bit joint indices where 16 would do: an MSL ushort4 is 8 bytes and still
// forces the float4 that follows onto a 16-byte boundary, so the smaller field
// buys nothing but a chance to get the padding wrong.
// One instance of a mesh, for INSTANCED drawing.
//
// The per-draw uniform block carries model and tint today, which means one
// draw call and one uniform slice per object. That is fine at fourteen objects
// and hopeless at fourteen thousand: the cost is entirely in the submission,
// not in the triangles. Moving the two things that actually differ per object
// into a buffer indexed by instance_id turns N draws into one.
//
// `bounds` rides along because GPU culling needs it and the vertex stage
// ignores it: putting it in a second parallel array would mean two buffers
// kept in step by hand.
struct GpuInstance {
    ENG_MAT4 model;
    ENG_VEC4 tint;
    ENG_VEC4 bounds;  // xyz = centre in OBJECT space, w = radius
};

// What a GPU culling pass writes, and what an indirect draw reads. The layout
// is Metal's MTLDrawIndexedPrimitivesIndirectArguments, field for field --
// the GPU reads it as that struct, so a mismatch here is not a compile error
// anywhere, just a draw of the wrong thing.
struct GpuDrawArgs {
    unsigned int index_count;
    unsigned int instance_count;
    unsigned int index_start;
    int base_vertex;
    unsigned int base_instance;
};

// One particle. 64 bytes, and the packing is not tidiness -- a million
// particles is 64 MB, and every field that earns its place costs another 4 MB.
//
// `life` counts DOWN, and life <= 0 is the only definition of dead. Keeping a
// separate alive flag means two things that can disagree, and the one that
// disagrees is always the one the renderer read.
struct GpuParticle {
    ENG_VEC4 position;  // xyz world, w = seconds of life remaining
    ENG_VEC4 velocity;  // xyz m/s,   w = size in metres
    ENG_VEC4 color;     // rgba at birth
    // x = the lifetime it was born with, so the shader can compute normalised
    // age without a second buffer. y = a per-particle random seed, kept so that
    // anything wanting variation reads it instead of re-hashing the index and
    // getting the same number in two places. zw spare.
    ENG_VEC4 birth;
};

// Everything the simulation kernel needs. One block, uploaded per step.
struct GpuParticleParams {
    ENG_VEC4 origin;     // xyz emitter position, w = cone half-angle in radians
    ENG_VEC4 direction;  // xyz mean emit direction (unit), w = speed
    ENG_VEC4 gravity;    // xyz m/s^2, w = linear drag per second
    ENG_VEC4 color;      // rgba at birth
    // x dt, y speed variance, z lifetime, w lifetime variance
    ENG_VEC4 motion;
    // x size, y size variance, z how many to spawn this step, w frame index
    // (the random seed's other half -- without it every frame emits the same
    // particles into the same slots).
    ENG_VEC4 emit;
    // x capacity, yzw spare.
    ENG_VEC4 limits;
};

// One SPH particle. Density and pressure ride in the w components because they
// are computed and consumed within a single step and never leave the GPU --
// giving them their own buffers would double the number of bindings for data
// with the same lifetime.
struct GpuFluidParticle {
    ENG_VEC4 position;  // xyz, w = density
    ENG_VEC4 velocity;  // xyz, w = pressure
};

struct GpuFluidParams {
    ENG_VEC4 bounds_min;  // xyz, w = smoothing radius h
    ENG_VEC4 bounds_max;  // xyz, w = particle mass
    ENG_VEC4 grid;        // xyz = cells per axis, w = cell size (== h)
    // x rest density, y stiffness, z viscosity, w dt
    ENG_VEC4 physics;
    // x particle count, y wall restitution, z bucket capacity, w gravity (down)
    ENG_VEC4 misc;
    // Precomputed kernel normalisations. They involve h to the ninth power and
    // a division by pi; computing them per particle per neighbour would be
    // three transcendentals in the innermost loop of the whole simulation.
    ENG_VEC4 kernels;  // x poly6, y spiky gradient, z viscosity laplacian, w unused
    // x = artificial viscosity alpha, y = sound speed sqrt(stiffness), zw spare.
    ENG_VEC4 artificial;
};

struct SkinIn {
    ENG_UVEC4 joints;
    ENG_VEC4 weights;
};

struct VertexIn {
    ENG_VEC3 position;  // .w unused on the C++ side
    ENG_VEC3 normal;    // object space, unit length; .w unused
    ENG_VEC4 color;
    // .xy = texture coordinates, origin top-left to match Metal's texture
    // convention. .zw reserved (a second uv set later).
    //
    // A bare float2 would be tempting, but MSL would align the struct to 16
    // anyway and the padding lands somewhere less useful. The unused .w lanes
    // of position and normal cannot be used instead: they are float3 on the
    // MSL side, which occupies 16 bytes but exposes only .xyz.
    ENG_VEC4 uv;
    // TANGENT, object space. .xyz is the direction in which u increases across
    // the surface; .w is the handedness of the bitangent, +1 or -1.
    //
    // Why a whole extra 16 bytes per vertex: a normal map stores a perturbation
    // in TANGENT space -- relative to the surface's own uv axes -- and there is
    // no way to recover those axes from position and normal alone. Two meshes
    // with identical geometry and mirrored uvs need opposite bitangents, which
    // is exactly what .w records; deriving the bitangent as cross(N, T) without
    // it inverts every mirrored shell, and a mirrored shell is what half of
    // every character model is.
    //
    // Computing the frame in the fragment shader from screen derivatives is the
    // other option and it is worse: it costs four derivatives per pixel, it is
    // wrong wherever the derivative crosses a uv seam, and it cannot be
    // orthogonalised consistently between neighbouring triangles.
    ENG_VEC4 tangent;
};

#endif  // ENGINE_SHADER_TYPES_H
