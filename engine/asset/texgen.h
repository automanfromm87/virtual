// Pure C++20. Procedural surface textures: albedo and tangent-space normals,
// computed rather than loaded.
//
// WHY. This engine ships no assets, and an untextured world is the single
// biggest thing separating a correct render from a photograph. Real surfaces
// have detail at every scale -- a lawn is not one green, bark is not one brown
// -- and without it the shading is doing its job on a subject that has nothing
// to shade. Every material here is a few hundred lines of arithmetic and costs
// a megabyte of GPU memory, which is a better trade than it sounds: the same
// texture at 512 is indistinguishable from one painted by hand once it is on a
// surface, at a fraction of a JPEG decoder's complexity.
//
// EVERYTHING IS TILEABLE. A detail texture is repeated tens of times across a
// surface, and a texture that does not wrap shows a grid of seams that is far
// more objectionable than no texture at all. The noise below wraps by hashing
// integer coordinates modulo a period, so the right edge and the left edge are
// generated from the same lattice points and match exactly rather than
// approximately.
//
// ALBEDO MAPS MODULATE, they do not replace. The shader multiplies the map into
// the material's base colour, so these are built with a mean near 1 and carry
// variation rather than absolute colour. That keeps MaterialDesc::base_color
// meaning what it says and lets one ground texture serve grass, sand and dirt
// by changing four numbers instead of regenerating a megabyte.
#pragma once

#include <cstdint>
#include <vector>

#include "engine/core/math.h"

namespace eng::texgen {

struct Image {
    int width = 0, height = 0;
    std::vector<std::uint8_t> rgba;  // 8 bits per channel, tightly packed

    [[nodiscard]] bool Valid() const {
        return width > 0 && height > 0 &&
               rgba.size() == std::size_t(width) * std::size_t(height) * 4;
    }
};

// A tileable fractional-Brownian-motion field, normalised to roughly [0, 1].
//
// `period_x` and `period_y` are the lattice size of the FIRST octave, in cells,
// on each axis. Each octave doubles both, so the whole stack wraps at `size` as
// long as `size` is a multiple of the period times two to the octave count --
// powers of two for both and it always is. A period that does not divide leaves
// a seam, which is the one way to misuse this and is why it is not hidden.
//
// THE TWO AXES ARE SEPARATE so that noise can be STRETCHED. A field with a
// coarser lattice on one axis has features elongated along it, which is the
// only way to build a directional material -- wood grain, brushed metal, bark
// -- out of noise. Equal periods give the isotropic case.
[[nodiscard]] std::vector<float> Fbm(int size, std::uint32_t seed, int period_x,
                                     int period_y, int octaves, float gain);

// A height field turned into a tangent-space normal map, wrapping at the edges
// so the result tiles wherever its input did.
//
// CENTRAL DIFFERENCES, not Sobel. Sobel's extra taps buy noise rejection on a
// photographed height field; this one is analytic and noise-free, so the wider
// kernel would only blur detail that is already exactly right.
//
// `strength` scales the slope before the vector is built, so 0 is flat and 2 is
// twice as deep. Scaling the finished vector instead rotates it toward the
// surface normal and saturates.
[[nodiscard]] Image NormalFromHeight(const std::vector<float>& height, int size,
                                     float strength);

// An albedo/normal pair, which is how these are always used.
struct Surface {
    Image albedo;
    Image normal;
};

// Ground cover: fine blades over broad dry and damp patches. `dry` tints the
// patches, `blade` is how strong the high-frequency detail is.
// A SUBTLE dry tint by default. The first version used {1.18, 1.06, 0.72} and
// it was far too strong: this texture tiles about thirty times across a 128 m
// terrain, and a patch pattern that is individually convincing becomes a
// repeating yellow grid at that count. The broad scale has to be quiet enough
// that its repetition is not the thing the eye finds.
[[nodiscard]] Surface Ground(int size, std::uint32_t seed, Vec3 dry = {1.09f, 1.03f, 0.87f},
                             float blade = 0.40f);

// Bark: ridges running ALONG the branch, which means varying across u and
// holding nearly constant along v -- the tube generator lays u around the
// circumference and v along the length, so this is the way round that puts the
// grain where a tree's grain is.
[[nodiscard]] Surface Bark(int size, std::uint32_t seed);

// Foliage: leaf-scale mottling and vein-like streaks. No normal map -- a leaf
// blob is already an approximation of thousands of leaves, and a normal map on
// it would be detail at the wrong scale pretending to be detail at the right
// one.
[[nodiscard]] Image Foliage(int size, std::uint32_t seed);

}  // namespace eng::texgen
