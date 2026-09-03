#include "engine/asset/png.h"

#include <array>
#include <cstdio>
#include <cstring>

namespace eng::png {
namespace {

// ------------------------------------------------------------------ DEFLATE --

// LSB-first bit reader. DEFLATE packs Huffman codes from the least significant
// bit up, which is the opposite of how the codes are written down, and mixing
// the two conventions is the classic way to get a decoder that almost works.
class BitReader {
  public:
    explicit BitReader(std::span<const std::uint8_t> in) : in_(in) {}

    // Returns -1 once the input is exhausted, which every caller must check:
    // a truncated stream otherwise decodes as an endless run of zero bits.
    int Bit() {
        if (held_ == 0) {
            if (pos_ >= in_.size()) {
                overrun_ = true;
                return -1;
            }
            bits_ = in_[pos_++];
            held_ = 8;
        }
        const int b = bits_ & 1;
        bits_ >>= 1;
        --held_;
        return b;
    }

    // `n` bits, least significant first.
    int Bits(int n) {
        int v = 0;
        for (int i = 0; i < n; ++i) {
            const int b = Bit();
            if (b < 0) return -1;
            v |= b << i;
        }
        return v;
    }

    void AlignToByte() { held_ = 0; }

    [[nodiscard]] std::size_t BytePos() const { return pos_; }
    [[nodiscard]] bool Overrun() const { return overrun_; }
    [[nodiscard]] const std::uint8_t* At(std::size_t i) const { return &in_[i]; }
    [[nodiscard]] std::size_t Size() const { return in_.size(); }
    void SeekByte(std::size_t p) {
        pos_ = p;
        held_ = 0;
    }

  private:
    std::span<const std::uint8_t> in_;
    std::size_t pos_ = 0;
    std::uint32_t bits_ = 0;
    int held_ = 0;
    bool overrun_ = false;
};

constexpr int kMaxBits = 15;

// Canonical Huffman, decoded a bit at a time. Slower than a lookup table and
// about a fifth the code; a texture is decoded once at load, not per frame.
struct Huffman {
    std::array<std::uint16_t, kMaxBits + 1> counts{};
    std::vector<std::uint16_t> symbols;

    bool Build(const std::uint8_t* lengths, int n) {
        counts.fill(0);
        for (int i = 0; i < n; ++i) ++counts[lengths[i]];
        counts[0] = 0;  // symbols with length 0 are simply absent

        // A code set is valid only if it is neither over- nor under-subscribed.
        // Checking is what stops a corrupt table from producing a decoder that
        // silently maps several symbols to the same code.
        int left = 1;
        for (int len = 1; len <= kMaxBits; ++len) {
            left <<= 1;
            left -= counts[len];
            if (left < 0) return false;  // over-subscribed
        }

        std::array<std::uint16_t, kMaxBits + 1> offsets{};
        for (int len = 1; len < kMaxBits; ++len)
            offsets[len + 1] = std::uint16_t(offsets[len] + counts[len]);

        symbols.assign(std::size_t(n), 0);
        for (int i = 0; i < n; ++i)
            if (lengths[i]) symbols[offsets[lengths[i]]++] = std::uint16_t(i);
        return true;
    }

    // -1 on a bad code or a truncated stream.
    int Decode(BitReader& br) const {
        int code = 0, first = 0, index = 0;
        for (int len = 1; len <= kMaxBits; ++len) {
            const int b = br.Bit();
            if (b < 0) return -1;
            code |= b;
            const int count = counts[len];
            if (code - first < count) return symbols[std::size_t(index + code - first)];
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        return -1;
    }
};

constexpr std::uint16_t kLenBase[29] = {3,  4,  5,  6,  7,  8,  9,  10,  11,  13,
                                        15, 17, 19, 23, 27, 31, 35, 43,  51,  59,
                                        67, 83, 99, 115, 131, 163, 195, 227, 258};
constexpr std::uint8_t kLenExtra[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                        2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
constexpr std::uint16_t kDistBase[30] = {
    1,    2,    3,    4,    5,    7,     9,     13,    17,   25,
    33,   49,   65,   97,   129,  193,   257,   385,   513,  769,
    1025, 1537, 2049, 3073, 4097, 6145,  8193,  12289, 16385, 24577};
constexpr std::uint8_t kDistExtra[30] = {0, 0, 0,  0,  1,  1,  2,  2,  3,  3,
                                         4, 4, 5,  5,  6,  6,  7,  7,  8,  8,
                                         9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

bool InflateBlockData(BitReader& br, const Huffman& lit, const Huffman& dist,
                      std::vector<std::uint8_t>& out, std::string& error,
                      std::size_t max_output) {
    for (;;) {
        const int sym = lit.Decode(br);
        if (sym < 0) {
            error = "deflate: bad literal/length code";
            return false;
        }
        if (sym < 256) {
            if (out.size() >= max_output) {
                error = "deflate: output exceeds the expected size";
                return false;
            }
            out.push_back(std::uint8_t(sym));
            continue;
        }
        if (sym == 256) return true;  // end of block

        const int li = sym - 257;
        if (li >= 29) {
            error = "deflate: length symbol out of range";
            return false;
        }
        const int extra = br.Bits(kLenExtra[li]);
        if (extra < 0) {
            error = "deflate: truncated length";
            return false;
        }
        const int length = kLenBase[li] + extra;

        const int dsym = dist.Decode(br);
        if (dsym < 0 || dsym >= 30) {
            error = "deflate: bad distance code";
            return false;
        }
        const int dextra = br.Bits(kDistExtra[dsym]);
        if (dextra < 0) {
            error = "deflate: truncated distance";
            return false;
        }
        const std::size_t distance = std::size_t(kDistBase[dsym] + dextra);
        if (distance > out.size()) {
            error = "deflate: back-reference before start of output";
            return false;
        }

        if (out.size() + std::size_t(length) > max_output) {
            error = "deflate: output exceeds the expected size";
            return false;
        }
        // Byte at a time on purpose: a match may overlap its own output, which
        // is how DEFLATE encodes runs. memcpy would read the pre-copy bytes.
        std::size_t from = out.size() - distance;
        for (int i = 0; i < length; ++i) out.push_back(out[from++]);
    }
}

bool FixedTables(Huffman& lit, Huffman& dist) {
    std::uint8_t l[288];
    for (int i = 0; i < 144; ++i) l[i] = 8;
    for (int i = 144; i < 256; ++i) l[i] = 9;
    for (int i = 256; i < 280; ++i) l[i] = 7;
    for (int i = 280; i < 288; ++i) l[i] = 8;
    std::uint8_t d[30];
    for (int i = 0; i < 30; ++i) d[i] = 5;
    return lit.Build(l, 288) && dist.Build(d, 30);
}

bool DynamicTables(BitReader& br, Huffman& lit, Huffman& dist, std::string& error) {
    const int hlit = br.Bits(5);
    const int hdist = br.Bits(5);
    const int hclen = br.Bits(4);
    if (hlit < 0 || hdist < 0 || hclen < 0) {
        error = "deflate: truncated dynamic header";
        return false;
    }
    const int nlit = hlit + 257, ndist = hdist + 1, nclen = hclen + 4;

    // The code-length alphabet is itself Huffman-coded, and its lengths arrive
    // in this permuted order so the common ones come first and the tail can be
    // omitted.
    static constexpr int kOrder[19] = {16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                                       11, 4,  12, 3, 13, 2, 14, 1, 15};
    std::uint8_t clen[19] = {};
    for (int i = 0; i < nclen; ++i) {
        const int v = br.Bits(3);
        if (v < 0) {
            error = "deflate: truncated code lengths";
            return false;
        }
        clen[kOrder[i]] = std::uint8_t(v);
    }
    Huffman clh;
    if (!clh.Build(clen, 19)) {
        error = "deflate: invalid code-length table";
        return false;
    }

    std::uint8_t lengths[288 + 32] = {};
    int i = 0;
    while (i < nlit + ndist) {
        const int sym = clh.Decode(br);
        if (sym < 0) {
            error = "deflate: bad code-length symbol";
            return false;
        }
        if (sym < 16) {
            lengths[i++] = std::uint8_t(sym);
            continue;
        }
        int repeat = 0;
        std::uint8_t value = 0;
        if (sym == 16) {
            if (i == 0) {
                error = "deflate: repeat with nothing to repeat";
                return false;
            }
            value = lengths[i - 1];
            repeat = 3 + br.Bits(2);
        } else if (sym == 17) {
            repeat = 3 + br.Bits(3);
        } else {
            repeat = 11 + br.Bits(7);
        }
        if (br.Overrun() || i + repeat > nlit + ndist) {
            error = "deflate: code-length repeat overruns";
            return false;
        }
        while (repeat-- > 0) lengths[i++] = value;
    }
    if (lengths[256] == 0) {
        error = "deflate: no end-of-block code";
        return false;
    }
    if (!lit.Build(lengths, nlit) || !dist.Build(lengths + nlit, ndist)) {
        error = "deflate: invalid Huffman table";
        return false;
    }
    return true;
}

std::uint32_t Adler32(std::span<const std::uint8_t> data) {
    std::uint32_t a = 1, b = 0;
    for (std::uint8_t byte : data) {
        a = (a + byte) % 65521;
        b = (b + a) % 65521;
    }
    return (b << 16) | a;
}

std::uint32_t Crc32(std::span<const std::uint8_t> data) {
    static std::uint32_t table[256];
    static bool built = false;
    if (!built) {
        for (std::uint32_t n = 0; n < 256; ++n) {
            std::uint32_t c = n;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[n] = c;
        }
        built = true;
    }
    std::uint32_t c = 0xFFFFFFFFu;
    for (std::uint8_t byte : data) c = table[(c ^ byte) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

// ---------------------------------------------------------------------- PNG --

std::uint32_t ReadBE32(const std::uint8_t* p) {
    return (std::uint32_t(p[0]) << 24) | (std::uint32_t(p[1]) << 16) |
           (std::uint32_t(p[2]) << 8) | std::uint32_t(p[3]);
}

int Paeth(int a, int b, int c) {
    const int p = a + b - c;
    const int pa = p > a ? p - a : a - p;
    const int pb = p > b ? p - b : b - p;
    const int pc = p > c ? p - c : c - p;
    if (pa <= pb && pa <= pc) return a;
    return pb <= pc ? b : c;
}

struct Header {
    std::uint32_t width = 0, height = 0;
    int bit_depth = 8;
    int color_type = 6;
    int interlace = 0;
};

int ChannelsFor(int color_type) {
    switch (color_type) {
        case 0: return 1;  // greyscale
        case 2: return 3;  // rgb
        case 3: return 1;  // palette index
        case 4: return 2;  // greyscale + alpha
        case 6: return 4;  // rgba
        default: return 0;
    }
}

}  // namespace

bool Inflate(std::span<const std::uint8_t> in, std::vector<std::uint8_t>& out,
             std::string& error, std::size_t max_output) {
    BitReader br(in);
    for (;;) {
        const int final_block = br.Bit();
        const int type = br.Bits(2);
        if (final_block < 0 || type < 0) {
            error = "deflate: truncated block header";
            return false;
        }
        if (type == 0) {
            // Stored: byte-aligned, with a length and its complement.
            br.AlignToByte();
            const std::size_t p = br.BytePos();
            if (p + 4 > br.Size()) {
                error = "deflate: truncated stored block";
                return false;
            }
            const std::uint8_t* h = br.At(p);
            const std::size_t len = std::size_t(h[0]) | (std::size_t(h[1]) << 8);
            const std::size_t nlen = std::size_t(h[2]) | (std::size_t(h[3]) << 8);
            if ((len ^ 0xFFFFu) != nlen) {
                error = "deflate: stored block length check failed";
                return false;
            }
            if (p + 4 + len > br.Size()) {
                error = "deflate: stored block runs past end";
                return false;
            }
            if (out.size() + len > max_output) {
                error = "deflate: output exceeds the expected size";
                return false;
            }
            out.insert(out.end(), br.At(p + 4), br.At(p + 4) + len);
            br.SeekByte(p + 4 + len);
        } else if (type == 1 || type == 2) {
            Huffman lit, dist;
            if (type == 1) {
                if (!FixedTables(lit, dist)) {
                    error = "deflate: bad fixed tables";
                    return false;
                }
            } else if (!DynamicTables(br, lit, dist, error)) {
                return false;
            }
            if (!InflateBlockData(br, lit, dist, out, error, max_output)) return false;
        } else {
            error = "deflate: reserved block type";
            return false;
        }
        if (final_block) return true;
    }
}

bool ZlibInflate(std::span<const std::uint8_t> in, std::vector<std::uint8_t>& out,
                 std::string& error, std::size_t max_output) {
    if (in.size() < 6) {
        error = "zlib: stream too short";
        return false;
    }
    const int cmf = in[0], flg = in[1];
    if ((cmf & 0x0F) != 8) {
        error = "zlib: not deflate";
        return false;
    }
    if (((cmf << 8) | flg) % 31 != 0) {
        error = "zlib: header check failed";
        return false;
    }
    if (flg & 0x20) {
        error = "zlib: preset dictionary not supported";
        return false;
    }
    const std::size_t body = in.size() - 4;
    if (!Inflate(in.subspan(2, body - 2), out, error, max_output)) return false;

    // The checksum is the difference between "decompressed" and "decompressed
    // correctly". Skipping it turns a corrupt file into a plausible image.
    const std::uint32_t want = ReadBE32(in.data() + body);
    const std::uint32_t got = Adler32(out);
    if (want != got) {
        error = "zlib: adler32 mismatch";
        return false;
    }
    return true;
}

bool IsPng(std::span<const std::uint8_t> bytes) {
    static constexpr std::uint8_t kSig[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    return bytes.size() >= 8 && std::memcmp(bytes.data(), kSig, 8) == 0;
}

Texture2D Decode(std::span<const std::uint8_t> bytes, std::string& error) {
    Texture2D img;
    if (!IsPng(bytes)) {
        error = "png: bad signature";
        return img;
    }

    Header hdr;
    bool have_header = false;
    std::vector<std::uint8_t> idat;
    std::vector<std::uint8_t> palette;   // rgb triples
    std::vector<std::uint8_t> trns;      // palette alpha, or a key colour

    std::size_t p = 8;
    while (p + 8 <= bytes.size()) {
        const std::uint32_t len = ReadBE32(bytes.data() + p);
        const std::uint8_t* type = bytes.data() + p + 4;
        if (p + 12 + len > bytes.size()) {
            error = "png: chunk runs past end of file";
            return img;
        }
        const std::uint8_t* data = bytes.data() + p + 8;

        const std::uint32_t want_crc = ReadBE32(data + len);
        const std::uint32_t got_crc = Crc32({type, std::size_t(len) + 4});
        if (want_crc != got_crc) {
            error = "png: chunk CRC mismatch";
            return img;
        }

        auto is = [&](const char* t) { return std::memcmp(type, t, 4) == 0; };
        if (is("IHDR")) {
            if (len != 13) {
                error = "png: bad IHDR length";
                return img;
            }
            hdr.width = ReadBE32(data);
            hdr.height = ReadBE32(data + 4);
            hdr.bit_depth = data[8];
            hdr.color_type = data[9];
            hdr.interlace = data[12];
            have_header = true;

            if (hdr.width == 0 || hdr.height == 0) {
                error = "png: zero-sized image";
                return img;
            }
            if (data[10] != 0 || data[11] != 0) {
                error = "png: unknown compression or filter method";
                return img;
            }
            if (hdr.interlace != 0 && hdr.interlace != 1) {
                error = "png: unknown interlace method " +
                        std::to_string(hdr.interlace);
                return img;
            }
            if (ChannelsFor(hdr.color_type) == 0) {
                error = "png: unknown colour type " + std::to_string(hdr.color_type);
                return img;
            }
            // Which depths are legal depends on the colour type, and the spec
            // is not uniform about it: greyscale goes down to 1 bit, a palette
            // goes down to 1 bit but never up to 16 (an index is not a
            // brightness), and anything with real colour or alpha channels is
            // 8 or 16 only. Accepting a combination the spec forbids means
            // decoding whatever the encoder happened to emit.
            const bool depth_ok = [&] {
                switch (hdr.color_type) {
                    case 0: return hdr.bit_depth == 1 || hdr.bit_depth == 2 ||
                                   hdr.bit_depth == 4 || hdr.bit_depth == 8 ||
                                   hdr.bit_depth == 16;
                    case 3: return hdr.bit_depth == 1 || hdr.bit_depth == 2 ||
                                   hdr.bit_depth == 4 || hdr.bit_depth == 8;
                    default: return hdr.bit_depth == 8 || hdr.bit_depth == 16;
                }
            }();
            if (!depth_ok) {
                error = "png: bit depth " + std::to_string(hdr.bit_depth) +
                        " is not legal for colour type " +
                        std::to_string(hdr.color_type);
                return img;
            }
            // 4 bytes per pixel out, and the filtered rows are wider still.
            // Checked before allocating so a header claiming 60000x60000 is an
            // error rather than an out-of-memory abort.
            if (std::uint64_t(hdr.width) * hdr.height > 64u * 1024 * 1024) {
                error = "png: image too large";
                return img;
            }
        } else if (is("PLTE")) {
            palette.assign(data, data + len);
        } else if (is("tRNS")) {
            trns.assign(data, data + len);
        } else if (is("IDAT")) {
            idat.insert(idat.end(), data, data + len);
        } else if (is("IEND")) {
            break;
        }
        p += 12 + len;
    }

    if (!have_header) {
        error = "png: no IHDR";
        return img;
    }
    if (idat.empty()) {
        error = "png: no image data";
        return img;
    }
    if (hdr.color_type == 3 && palette.empty()) {
        error = "png: palette image with no PLTE";
        return img;
    }

    const int channels = ChannelsFor(hdr.color_type);
    const int depth = hdr.bit_depth;
    // SUB-BYTE DEPTHS. At 1, 2 or 4 bits a pixel is a fraction of a byte, and
    // two different widths follow from that:
    //
    //   filt_bpp  what the FILTER calls "the pixel to the left". The spec
    //             rounds it up to one whole byte, so at 1 bit the filter
    //             reaches back eight pixels, not one. Using the true fractional
    //             width here decodes a plausible-looking but wrong image.
    //   bpp       what everything downstream sees, AFTER unpacking to one byte
    //             per sample. Unpacking early is what keeps the palette lookup,
    //             the tRNS key and the scatter from each needing a bit-reader.
    const int filt_bpp = std::max(1, channels * depth / 8);
    const int sample_bytes = depth == 16 ? 2 : 1;
    const int bpp = channels * sample_bytes;
    const std::size_t stride = std::size_t(hdr.width) * std::size_t(bpp);
    // Bytes in one packed row of `w` pixels, rounded up: the last byte of a row
    // is padded, and those spare bits belong to no pixel.
    const auto packed_stride = [&](std::uint32_t w) {
        return (std::size_t(w) * channels * depth + 7) / 8;
    };
    // Every byte the image can possibly need, known from IHDR alone. Handing
    // this to the decompressor as a cap is what stops a 24 KB file from
    // demanding 24 MB — the size is not a guess, it is arithmetic.
    // ADAM7. An interlaced image is seven smaller images, each with its own
    // rows, its own filtering and its own scanline width — a coarse pass first
    // so a slow download shows something early. Every pass has to be defiltered
    // in isolation: filters refer to the previous row OF THAT PASS, which is
    // eight pixels away in the final image, not one.
    static constexpr int kPassX0[7] = {0, 4, 0, 2, 0, 1, 0};
    static constexpr int kPassY0[7] = {0, 0, 4, 0, 2, 0, 1};
    static constexpr int kPassDX[7] = {8, 8, 4, 4, 2, 2, 1};
    static constexpr int kPassDY[7] = {8, 8, 8, 4, 4, 2, 2};

    const int passes = hdr.interlace ? 7 : 1;
    struct PassGeom {
        std::uint32_t w = 0, h = 0;
        std::size_t stride = 0;
    };
    PassGeom geom[7];
    std::size_t expect = 0;
    for (int p = 0; p < passes; ++p) {
        if (!hdr.interlace) {
            geom[0] = {hdr.width, hdr.height, packed_stride(hdr.width)};
        } else {
            geom[p].w = std::uint32_t(
                (hdr.width + kPassDX[p] - 1 - kPassX0[p]) / kPassDX[p]);
            geom[p].h = std::uint32_t(
                (hdr.height + kPassDY[p] - 1 - kPassY0[p]) / kPassDY[p]);
            geom[p].stride = packed_stride(geom[p].w);
        }
        // An empty pass contributes NOTHING, not even a filter byte. A small
        // image legitimately has several, and counting a byte for each is the
        // classic way to decode every interlaced thumbnail one row short.
        if (geom[p].w == 0 || geom[p].h == 0) continue;
        expect += (geom[p].stride + 1) * geom[p].h;
    }

    std::vector<std::uint8_t> raw;
    if (!ZlibInflate(idat, raw, error, expect)) return img;

    if (raw.size() < expect) {
        error = "png: decompressed data is short (" + std::to_string(raw.size()) +
                " of " + std::to_string(expect) + ")";
        return img;
    }

    // Defiltered, then scattered into place. `lines` is always the final
    // full-size image; a non-interlaced file is simply the one-pass case.
    std::vector<std::uint8_t> lines(stride * hdr.height, 0);
    std::size_t read_at = 0;
    for (int p = 0; p < passes; ++p) {
        const PassGeom& g = geom[p];
        if (g.w == 0 || g.h == 0) continue;
        std::vector<std::uint8_t> pass(g.stride * g.h);
        std::size_t g_stride_out = g.stride;
        for (std::uint32_t y = 0; y < g.h; ++y) {
            const std::uint8_t filter = raw[read_at];
            const std::uint8_t* src = &raw[read_at + 1];
            read_at += g.stride + 1;
            std::uint8_t* dst = &pass[g.stride * y];
            const std::uint8_t* prev = y ? &pass[g.stride * (y - 1)] : nullptr;

            for (std::size_t x = 0; x < g.stride; ++x) {
                const int a = x >= std::size_t(filt_bpp) ? dst[x - filt_bpp] : 0;
                const int b = prev ? prev[x] : 0;
                const int c =
                    (prev && x >= std::size_t(filt_bpp)) ? prev[x - filt_bpp] : 0;
                int v = src[x];
                switch (filter) {
                    case 0: break;
                    case 1: v += a; break;
                    case 2: v += b; break;
                    case 3: v += (a + b) / 2; break;
                    case 4: v += Paeth(a, b, c); break;
                    default:
                        error = "png: unknown row filter " + std::to_string(filter);
                        return img;
                }
                dst[x] = std::uint8_t(v);
            }
        }
        // UNPACK, per pass and before the scatter. Doing it once at the end
        // instead would work for a non-interlaced image and quietly corrupt an
        // interlaced one: the scatter moves whole pixels, and below 8 bits a
        // pixel does not start on a byte boundary.
        if (depth < 8) {
            std::vector<std::uint8_t> wide(std::size_t(g.w) * channels * g.h);
            const int per_byte = 8 / depth;
            const int mask = (1 << depth) - 1;
            for (std::uint32_t y = 0; y < g.h; ++y)
                for (std::size_t i = 0; i < std::size_t(g.w) * channels; ++i) {
                    // MSB first within the byte, which is the order the spec
                    // uses and the opposite of what a bit-shift loop written
                    // from intuition produces.
                    const std::uint8_t byte = pass[g.stride * y + i / per_byte];
                    const int shift = (per_byte - 1 - int(i % per_byte)) * depth;
                    wide[std::size_t(g.w) * channels * y + i] =
                        std::uint8_t((byte >> shift) & mask);
                }
            pass.swap(wide);
            g_stride_out = std::size_t(g.w) * channels;
        } else {
            g_stride_out = g.stride;
        }
        if (!hdr.interlace) {
            lines.swap(pass);
            continue;
        }
        for (std::uint32_t y = 0; y < g.h; ++y)
            for (std::uint32_t x = 0; x < g.w; ++x) {
                const std::uint32_t fx =
                    std::uint32_t(kPassX0[p]) + x * std::uint32_t(kPassDX[p]);
                const std::uint32_t fy =
                    std::uint32_t(kPassY0[p]) + y * std::uint32_t(kPassDY[p]);
                std::memcpy(&lines[stride * fy + std::size_t(fx) * bpp],
                            &pass[g_stride_out * y + std::size_t(x) * bpp],
                            std::size_t(bpp));
            }
    }

    // Expand to RGBA8. 16-bit samples are truncated to their high byte: this
    // renderer has no 16-bit texture format, so keeping the low byte would only
    // mean throwing it away one layer further up.
    img.width = int(hdr.width);
    img.height = int(hdr.height);
    img.rgba.assign(std::size_t(hdr.width) * hdr.height * 4, 255);

    // COLOUR-KEY transparency. On a palette image tRNS is a per-entry alpha,
    // but on greyscale or RGB it names one colour that is fully transparent.
    // Only the palette form was handled, so a keyed image came out completely
    // opaque — a wrong picture with no error, which is worse than refusing it.
    int key_r = -1, key_g = -1, key_b = -1;
    if (hdr.color_type == 0 && trns.size() >= 2) {
        key_r = (int(trns[0]) << 8) | trns[1];
    } else if (hdr.color_type == 2 && trns.size() >= 6) {
        key_r = (int(trns[0]) << 8) | trns[1];
        key_g = (int(trns[2]) << 8) | trns[3];
        key_b = (int(trns[4]) << 8) | trns[5];
    }
    // The key is always stored as 16 bits, even for an 8-bit image, so the
    // comparison has to happen at full sample width rather than after the
    // truncation to RGBA8 below.
    const auto full = [&](const std::uint8_t* s, int ch) {
        return sample_bytes == 2 ? ((int(s[ch * 2]) << 8) | s[ch * 2 + 1])
                                 : int(s[ch]);
    };

    const std::size_t pixels = std::size_t(hdr.width) * hdr.height;
    for (std::size_t i = 0; i < pixels; ++i) {
        const std::uint8_t* s = &lines[i * std::size_t(bpp)];
        std::uint8_t* d = &img.rgba[i * 4];
        auto sample = [&](int ch) { return s[ch * sample_bytes]; };  // high byte
        switch (hdr.color_type) {
            case 0:  // greyscale
                // SCALED to fill the byte. A 1-bit image is black and WHITE; a
                // raw sample of 1 left alone is black and very-slightly-less
                // black. Palette indices below get no such treatment -- an
                // index is not a brightness, and scaling one is a lookup into
                // the wrong entry.
                d[0] = d[1] = d[2] =
                    depth < 8 ? std::uint8_t(sample(0) * 255 / ((1 << depth) - 1))
                              : sample(0);
                if (key_r >= 0 && full(s, 0) == key_r) d[3] = 0;
                break;
            case 2:  // rgb
                d[0] = sample(0);
                d[1] = sample(1);
                d[2] = sample(2);
                if (key_r >= 0 && full(s, 0) == key_r && full(s, 1) == key_g &&
                    full(s, 2) == key_b)
                    d[3] = 0;
                break;
            case 3: {  // palette
                const std::size_t idx = s[0];
                if (idx * 3 + 2 >= palette.size()) {
                    error = "png: palette index out of range";
                    return Texture2D{};
                }
                d[0] = palette[idx * 3];
                d[1] = palette[idx * 3 + 1];
                d[2] = palette[idx * 3 + 2];
                if (idx < trns.size()) d[3] = trns[idx];
                break;
            }
            case 4:  // greyscale + alpha
                d[0] = d[1] = d[2] = sample(0);
                d[3] = sample(1);
                break;
            case 6:  // rgba
                d[0] = sample(0);
                d[1] = sample(1);
                d[2] = sample(2);
                d[3] = sample(3);
                break;
            default: break;
        }
    }
    return img;
}

Texture2D DecodeFile(const std::string& path, std::string& error) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        error = "cannot open " + path;
        return Texture2D{};
    }
    std::vector<std::uint8_t> bytes;
    std::uint8_t buf[8192];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) bytes.insert(bytes.end(), buf, buf + n);
    std::fclose(f);
    return Decode(bytes, error);
}

}  // namespace eng::png
