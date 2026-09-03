// Pure C++20. A PNG decoder, including the DEFLATE decompressor it needs.
//
// WHY write one: this engine takes no third-party code, and until now the only
// textures it could have were the procedural ones in engine/texture. That makes
// the glTF importer half a feature — it reads a model's geometry and materials
// and then has to throw away every image the file points at.
//
// The work is really two things stacked. DEFLATE (RFC 1951) is the hard half
// and has nothing to do with images; PNG (chunks, filters, colour types) is the
// easy half and is where all the format-specific detail lives. They are tested
// separately for that reason.
//
// SUPPORTED: bit depth 8 and 16; colour types greyscale, RGB, palette,
// greyscale+alpha and RGBA; tRNS transparency; any mix of the five scanline
// filters; multiple IDAT chunks.
//
// NOT SUPPORTED, each rejected with a message rather than mis-decoded:
// interlaced (Adam7) images, and bit depths below 8. Both are rare in the
// textures a 3D exporter writes, and both are real work — Adam7 is seven
// separate sub-images with their own filtering.
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "engine/texture/texture.h"

namespace eng::png {

// Returns an empty Texture2D and fills `error` on anything malformed.
[[nodiscard]] Texture2D Decode(std::span<const std::uint8_t> bytes,
                               std::string& error);

[[nodiscard]] Texture2D DecodeFile(const std::string& path, std::string& error);

// Encodes 8-bit RGBA — `rgba` is width*height*4, top row first — into a PNG
// byte stream.
//
// The DEFLATE stream is STORED blocks: the framing a zlib reader expects, with
// no compression inside it. Writing a Huffman coder is a project of its own and
// this exists for exactly one job — getting a rendered frame out of the GPU and
// onto disk where a human can look at it. A screenshot 1.0004x the size of its
// raw pixels costs nothing. A subtly wrong compressor costs an afternoon, and
// the failure mode is a file that opens fine in one viewer and not another.
[[nodiscard]] std::vector<std::uint8_t> Encode(std::span<const std::uint8_t> rgba,
                                               int width, int height);

[[nodiscard]] bool EncodeFile(const std::string& path,
                              std::span<const std::uint8_t> rgba, int width,
                              int height, std::string& error);

// True if `bytes` starts with the PNG signature. Cheap enough to call before
// deciding what a blob is.
[[nodiscard]] bool IsPng(std::span<const std::uint8_t> bytes);

// Raw DEFLATE, no wrapper. Exposed because it is the half most likely to be
// wrong and the half that can be tested against any other implementation.
//
// `max_output` is a hard cap, not a hint. DEFLATE compresses about 1000:1 in
// the limit — 24 KB of input expands to 24 MB of zeros — so a decompressor with
// no bound turns a small hostile file into an arbitrary allocation. PNG knows
// exactly how many bytes it needs before it starts, so it passes that.
//
// APPENDS to `out` rather than clearing it, and back-references may reach into
// whatever was already there. Pass an empty vector unless that is what you
// want.
[[nodiscard]] bool Inflate(std::span<const std::uint8_t> in,
                           std::vector<std::uint8_t>& out, std::string& error,
                           std::size_t max_output = ~std::size_t{0});

// zlib-wrapped DEFLATE (RFC 1950): two header bytes and a trailing Adler-32,
// which is what PNG's IDAT stream actually contains.
[[nodiscard]] bool ZlibInflate(std::span<const std::uint8_t> in,
                               std::vector<std::uint8_t>& out, std::string& error,
                               std::size_t max_output = ~std::size_t{0});

}  // namespace eng::png
