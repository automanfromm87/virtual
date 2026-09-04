# virtual

A 3D game engine written from scratch in C++20 and Metal, for macOS on Apple
Silicon. No third-party dependencies of any kind: no maths library, no image
loaders, no physics engine, no UI toolkit. Everything below is in this
repository, and the only outside code is Apple's own frameworks — Metal for the
GPU, CoreText for glyph outlines, AudioToolbox for the speaker and for decoding
compressed audio.

Around 68,000 lines, 70 test targets, everything built with Bazel.

## Running it

```
bazel run -c opt //apps/world:world                 # the demo: terrain + navmesh + character + sky + fog + post
bazel run -c opt //apps/world:world -- --shot /tmp/a.png   # headless capture, opens no window
bazel test //...
```

`-c opt` is not optional; the demo is an order of magnitude slower without it.
Left-click to walk, `wasd` to move (`space` to run), right-drag to orbit,
scroll to zoom, `0`–`5` to fast-travel between districts, `[`/`]` to move the
sun, `f` fog, `t` TAA, `o` SSAO, `v` light shafts, `e`/`q` focus height,
`r` reset, `esc` to quit. `--shot` renders a fixed number of frames offscreen
to a PNG; `viewer.cpp` parses further capture flags (`--frames`, `--sun`,
`--nossao`, `--noshafts`, `--shadowpx`, …).

The per-system showcases that used to be separate demos now live as offscreen
tests under `tests/` (`gallery`, `ibl`, `lights`, `particles`, `fluid`,
`skinned`, …) — same scenes, asserted by measurement rather than by looking.

## What is in it

**Rendering.** Forward and deferred paths that produce the same image from one
shared shading function. Cook-Torrance PBR with multiple-scattering
compensation — a white metal in a white furnace stays at 231/255 across every
roughness, where single scattering falls to 178. HDR with a tone map at the
end, bloom, SSAO, 4× MSAA. Shadows: cascaded for the sun, perspective for
spots, cube maps for point lights, and hardware ray-traced as an alternative.
GPU-driven drawing — instancing, a compute culling pass, indirect draws — which
takes 6000 objects from 3428 draw calls at 3.38 ms to 3 draws at 0.98 ms.

**Surfaces.** Tangents on every vertex and tangent-space normal maps, with
metallic, emissive and baked-occlusion maps beside them. Mip chains built on
upload and 16× anisotropic filtering: the aliasing on a receding checkerboard
floor drops to 4.9% of its unfiltered value, and anisotropy keeps 14.0 of
contrast in the band where an isotropic mip keeps 8.1. Block compression —
BC1, BC3 and BC5 with the encoder — at 8:1 against RGBA8, and BC5 holds a
normal map to 1.52/255 where BC1 manages 17.75. Indices are 32-bit, so a mesh
has no vertex ceiling.

**Lighting at scale.** Clustered lighting over a 16×9×24 frustum grid with a
CPU frustum pre-cull: 256 lights render pixel-identically to the brute-force
loop, 12.4× faster, and the curve is flat because the bins are byte-identical
whatever the total. Baked irradiance volumes for indirect light, path-traced on
the CPU with real bounces — a red wall turns the floor beside it red and
nothing nine metres away. Froxel volumetrics for shafts, glow and fog a shadow
darkens.

**Mesh shaders.** Meshlets of 64 vertices and 124 triangles with bounding
spheres and normal cones, culled by an object stage that launches zero or one
mesh threadgroups. Bit-identical to the vertex path while rejecting 19 of 101
meshlets — so those nineteen were provably invisible.

**Transparency.** Weighted-blended order-independent transparency. Three
intersecting panes give the same pixels in any submission order to within one
level, where the back-to-front sort differs by 78 because their depths tie.

**Display.** SDR, extended-range linear and Rec.2100 PQ. The PQ path matches
SMPTE ST 2084 to 0.0004 — a tenth of a 10-bit code step — and the HDR modes
are the identity below the roll-off, so an SDR-looking image does not wash out.

**Stereo.** Both eyes from one pass by vertex amplification, into a two-layer
target. Each slice is bit-identical to its own single-eye pass, at 0.075 ms
against 0.103 for two, and a 64 mm eye offset moves the image by exactly what
a 64 mm object offset does.

**Streaming.** Residency by angular size under a byte budget, loaded on worker
threads and evicted with hysteresis: 200 frames stepping across a level
boundary cause zero loads and zero evictions, and four threads reach
byte-identical residency to the synchronous path.

**Image-based lighting.** A physically based sky — Rayleigh and Mie scattering
integrated along the view ray — baked into a radiance cube, then convolved into
a diffuse probe and a prefiltered specular chain with a BRDF table. The sun's
colour comes out of the same model, so a sunset cannot have a white key light.
Checked with a white furnace: a uniform environment of 1.0 comes back as 1.0
from every stage.

**Post-processing.** Auto-exposure from a GPU luminance histogram, height fog
integrated analytically along the ray, depth of field, motion blur, temporal
antialiasing with a YCoCg variance clamp, screen-space reflections,
parallax-corrected reflection probes, and a parametric colour grade whose
defaults are bit-for-bit a no-op.

**Scale.** Levels of detail from a quadric-error simplifier, selected on the GPU
by screen radius; Hi-Z occlusion culling against a min-reduced depth pyramid
(144 objects behind a wall go from 145 survivors to 1); projected decals; and
chunked terrain with skirts, whose height query, mesh and raycast agree to a
ten-thousandth of a metre; and a dithered crossfade between levels of detail
whose two halves cover every pixel exactly once — no holes and no doubling.

**Compute.** GPU skinning written back to a buffer (so skinned meshes can cast
ray-traced shadows), a 60,000-particle system, and an SPH fluid.

**Physics.** Rigid bodies with angular dynamics, sphere/box/capsule/convex-hull
and height-field shapes, GJK and EPA, a convex hull builder, joints, sleeping
islands, continuous collision, raycasts and overlap queries, trigger volumes,
and a kinematic capsule character controller. A BVH broadphase, rebuilt per
step, keeps the pairs tested at about ten per body whether there are 125 bodies
or 1728 — brute force goes from 62 to 864.

**Animation.** Skeletal animation with a middle layer that makes it usable:
crossfades that do not jump when interrupted, masked layers, additive poses, a
blend space that synchronises phase across clips of different lengths, root
motion, two-bone IK with a pole vector, foot placement, and look-at.

**Navigation.** A navmesh built by voxelising level geometry into solid spans,
filtering by slope and headroom and ledge, eroding by the agent's radius, and
merging into convex polygons with portals; then A* and a funnel string-pull. A
wall with a three-metre doorway routes 16.58 m against a 10 m direct line, and
an unobstructed path is two points and exactly the straight-line length.

**Networking.** UDP with sequence numbers, an ack bitfield, round-trip
estimation and opt-in reliability; snapshot replication with delta compression
against the tick the client confirmed; client-side interpolation and prediction
with reconciliation. Tested against a deterministic network simulator that
loses, delays, jitters, duplicates and reorders on purpose — which is where
three of its bugs were found.

**Assets.** PNG (including Adam7 and sub-8-bit depths) and baseline JPEG, both
hand-written; glTF 2.0 with `.glb`, skins, animations, morph targets and sparse
accessors; WAV; and anything else the OS can decode.

**Tooling.** A baked asset package that loads without parsing — the vertex block
in it is byte-identical to what goes into a GPU buffer — and hot reload that
waits for a file to stop changing before firing, so an editor's save produces
one event rather than five and never a half-written file.

**The rest.** An ECS with a transform hierarchy, scene serialisation, a render
graph, an immediate-mode UI over a CoreText-rasterised font atlas, an audio
mixer with equal-power panning and distance attenuation behind a lock-free
queue, a job system that gets 10.9x on sixteen threads, per-pass GPU timings,
and gamepad support with a radial deadzone.

## How it is tested

Every module is tested by MEASUREMENT rather than by looking at the output, and
the tests are the interesting part of the repository. A fluid solver that is 30%
over-compressed makes a convincing picture for several seconds; a mixer with
linear panning sounds fine until a sound crosses the centre; a font atlas of
upside-down letters reads as a font that renders slightly badly. So the checks
are things like: total power is constant across a stereo pan, the settled
density sits at the rest density, a ballistic arc matches semi-implicit Euler to
five decimal places, deferred and forward agree to a mean of 0.049 of 255.

Where a bound is loose, the comment says what was measured and why the number is
what it is. Where a mutation of the code survived the tests, the comment says
that too.

The commit messages are written the same way: each one records what the
measurements said, including the several occasions where a test was wrong before
the code was.
