// Pure C++20. Block-compressed textures: BC1, BC3 and BC5, encoder included.
//
// WHY: an RGBA8 texture is four bytes a texel and a 2048-square albedo is 16 MB
// before its mip chain and 21 MB after. Thirty of those is most of a GPU. Block
// compression stores a 4x4 tile in 8 or 16 bytes and the hardware decodes it in
// the sampler at no cost -- it is the only lossy step in a renderer that pays
// for itself several times over.
//
// The three formats and what each is for:
//
//   BC1  RGB, 8 bytes per block, 6:1. Albedo, emissive, anything opaque.
//        Two endpoint colours in RGB565 and a 2-bit index per texel selecting
//        one of four points along the line between them.
//   BC3  RGBA, 16 bytes, 3:1. BC1 colour plus an independently compressed
//        alpha channel, which is why it beats BC1's 1-bit alpha for anything
//        with a soft edge.
//   BC5  Two channels, 16 bytes, 2:1 against RG8 and 4:1 against RGBA8. THE
//        normal map format: it stores x and y at BC4 quality each and the
//        shader rebuilds z, which it already does. Storing a normal as BC1 is
//        the classic mistake -- the colour endpoints are a line in RGB and a
//        normal map's texels are a sphere, so a BC1 normal map has visible
//        banding on every curved surface.
//
// NOT a general image codec. There is no BC7, which needs mode selection and a
// real optimiser; BC1 at 6:1 against BC7 at 4:1 is most of the win for a
// fraction of the code.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "engine/texture/texture.h"

namespace eng {

enum class BlockFormat : std::uint8_t {
    BC1,  // RGB
    BC3,  // RGBA
    BC5,  // two channels, from R and G
};

// Bytes per 4x4 block.
[[nodiscard]] constexpr int BlockBytes(BlockFormat f) {
    return f == BlockFormat::BC1 ? 8 : 16;
}

struct CompressedTexture {
    int width = 0;
    int height = 0;
    BlockFormat format = BlockFormat::BC1;
    // Byte offset of each mip level into `data`. Size is the level count.
    std::vector<std::size_t> level_offsets;
    std::vector<std::uint8_t> data;

    [[nodiscard]] int Levels() const { return int(level_offsets.size()); }
    [[nodiscard]] bool Empty() const { return data.empty(); }
};

// Compresses `src`, optionally building the whole mip chain down to 1x1.
//
// MIPS ARE BUILT HERE, on the CPU, and not by the GPU's generateMipmaps: that
// call cannot write into a compressed texture. So each level is box-filtered
// from the level above and then compressed separately.
//
// `srgb` makes the box filter decode to linear before averaging and re-encode
// after. It matters more than it sounds: averaging four sRGB bytes directly
// darkens every level, and by the fifth mip a mid-grey has drifted a long way.
// Set it for albedo and emissive; leave it off for normal, roughness,
// metallic and occlusion maps, which store numbers rather than colours.
//
// A source whose dimensions are not multiples of four is padded by REPLICATING
// its edge texels into the partial block, which is what stops a garbage column
// appearing along the right-hand edge.
[[nodiscard]] CompressedTexture Compress(const Texture2D& src, BlockFormat,
                                         bool mips = true, bool srgb = false);

// Decodes back to RGBA8. Exists for testing: the GPU decodes these in the
// sampler, so nothing in a frame calls this -- but "the encoder produces
// something the hardware likes" and "the encoder produces the right colours"
// are different claims, and only a decoder can separate them.
//
// BC5 decodes with z rebuilt as sqrt(1 - x^2 - y^2) into blue, matching what
// the shader does, so a round trip can be compared against the source.
[[nodiscard]] Texture2D DecodeBlocks(std::span<const std::uint8_t> blocks,
                                     int width, int height, BlockFormat);

}  // namespace eng
