# virtual

A 3D game engine written from scratch in C++20 and Metal, for macOS on Apple
Silicon. No third-party dependencies of any kind: no maths library, no image
loaders, no physics engine, no UI toolkit. Everything below is in this
repository, and the only outside code is Apple's own frameworks — Metal for the
GPU, CoreText for glyph outlines, AudioToolbox for the speaker and for decoding
compressed audio.

Around 25,000 lines, 36 test targets, everything built with Bazel.

## Running it

```
bazel run -c opt //apps/iso        # isometric game: click to walk, collect coins
bazel run -c opt //apps/walk       # first person, raycast crosshair
bazel run -c opt //apps/gallery:viewer   # the rendering showcase
bazel run -c opt //apps/particles:viewer # 60k GPU particles
bazel run -c opt //apps/fluid:viewer     # SPH fluid, dam break
bazel run -c opt //apps/skinned:viewer   # skeletal animation
bazel run -c opt //apps/world:viewer     # physics + ECS + glTF

bazel test //...
```

`-c opt` is not optional for the fluid; it is an order of magnitude slower
without it. Drag to look, `wasd` to move, `esc` to quit. Set `VIRTUAL_MUSIC` to
an audio file to give the isometric demo a soundtrack.

## What is in it

**Rendering.** Forward and deferred paths that produce the same image from one
shared shading function. Cook-Torrance PBR, HDR with a tone map at the end,
bloom, SSAO, 4× MSAA. Shadows: cascaded for the sun, perspective for spots, cube
maps for point lights, and hardware ray-traced as an alternative. GPU-driven
drawing — instancing, a compute culling pass, indirect draws — which takes 6000
objects from 3428 draw calls at 3.38 ms to 3 draws at 0.98 ms.

**Compute.** GPU skinning written back to a buffer (so skinned meshes can cast
ray-traced shadows), a 60,000-particle system, and an SPH fluid.

**Physics.** Rigid bodies with angular dynamics, sphere/box/capsule/convex-hull
shapes, GJK and EPA, a convex hull builder, joints, sleeping islands, continuous
collision, raycasts and overlap queries, trigger volumes, and a kinematic
capsule character controller.

**Assets.** PNG (including Adam7 and sub-8-bit depths) and baseline JPEG, both
hand-written; glTF 2.0 with `.glb`, skins, animations, morph targets and sparse
accessors; WAV; and anything else the OS can decode.

**The rest.** An ECS with a transform hierarchy, scene serialisation, a render
graph, an immediate-mode UI over a CoreText-rasterised font atlas, and an audio
mixer with equal-power panning and distance attenuation behind a lock-free
queue.

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
