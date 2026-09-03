// Baseline JPEG, decoded to RGBA8.
//
// The half of the format that matters for textures: SOF0/SOF1 sequential DCT,
// 8-bit samples, Huffman coding, greyscale or YCbCr, any of the usual chroma
// subsamplings, with restart intervals. That is what every camera, every
// exporter and every glTF file with an "image/jpeg" buffer emits.
//
// What it deliberately does NOT do, each refused with a message that says which
// it was rather than producing a wrong picture:
//
//   * PROGRESSIVE (SOF2). A different scan structure -- coefficients arrive
//     across many passes, each refining the last -- not a variation on this
//     one. Worth adding when a file needs it; guessing at it is not.
//   * Arithmetic coding, lossless and hierarchical modes. Effectively extinct.
//   * 12-bit samples, and CMYK/YCCK (Adobe four-component) images.
//
// The output is always RGBA8 with alpha 255: JPEG has no alpha, and inventing
// one from the luma is a guess that belongs to whoever authored the asset.
#pragma once

#include <cstdint>
#include <span>
#include <string>

#include "engine/asset/png.h"  // for Texture2D

namespace eng::jpeg {

// True if `bytes` starts with SOI. Cheap, and does not validate the rest -- it
// exists so a loader can pick a decoder, not so it can trust the file.
[[nodiscard]] bool IsJpeg(std::span<const std::uint8_t> bytes);

// Decodes to RGBA8. On failure returns an empty texture and sets `error`.
[[nodiscard]] Texture2D Decode(std::span<const std::uint8_t> bytes,
                               std::string& error);

}  // namespace eng::jpeg
