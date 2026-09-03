#include "engine/asset/jpeg.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace eng::jpeg {
namespace {

// Coefficients arrive in zigzag order -- low frequencies first, so that the
// long run of zeroes a photograph produces lands at the end where the
// run-length coding can collapse it. This maps that order back to the 8x8 grid.
constexpr int kZigzag[64] = {
    0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

struct Huffman {
    bool present = false;
    // Canonical Huffman, decoded by length. mincode/maxcode/valptr are the
    // spec's own formulation: at each length, the codes in use form one
    // contiguous numeric range, so a decode is a compare and an index rather
    // than a tree walk.
    int mincode[17] = {};
    int maxcode[17] = {};  // -1 where no code has that length
    int valptr[17] = {};
    std::vector<std::uint8_t> values;
};

struct Component {
    int id = 0;
    int h = 1, v = 1;   // sampling factors
    int tq = 0;         // quantisation table
    int td = 0, ta = 0; // DC and AC Huffman tables
    int dc_pred = 0;
    // Plane at the component's own resolution, padded out to whole MCUs.
    int plane_w = 0, plane_h = 0;
    // The part of that plane the image actually occupies. Sampling into the
    // padding pulls in whatever the last block decoded to, which shows as a
    // smear of the right-hand column along the edge.
    int valid_w = 0, valid_h = 0;
    std::vector<std::uint8_t> plane;
};

// A JPEG entropy stream is bytes with 0xFF escaped as 0xFF 0x00, because 0xFF
// starts a marker. Reading it as plain bits desynchronises on the first white
// pixel run, which is why a decoder that skips this looks fine on a dark test
// image and falls apart on a real one.
class BitReader {
  public:
    BitReader(const std::uint8_t* p, std::size_t n) : p_(p), n_(n) {}

    // Returns 0 past the end rather than failing: a truncated file should give
    // back the part that decoded, and the caller checks `marker_` anyway.
    int Bit() {
        if (count_ == 0) {
            if (!Fill()) return 0;
        }
        --count_;
        return (byte_ >> count_) & 1;
    }

    int Bits(int n) {
        int v = 0;
        for (int i = 0; i < n; ++i) v = (v << 1) | Bit();
        return v;
    }

    // Drops any partial byte and steps over an RST marker. Restart intervals
    // exist so a corrupt stream can resynchronise; the price is that the bit
    // position is only defined at a byte boundary here.
    bool Restart() {
        count_ = 0;
        while (at_ + 1 < n_) {
            if (p_[at_] == 0xFF && p_[at_ + 1] >= 0xD0 && p_[at_ + 1] <= 0xD7) {
                at_ += 2;
                hit_marker_ = false;
                return true;
            }
            ++at_;
        }
        return false;
    }

    [[nodiscard]] std::size_t Position() const { return at_; }
    [[nodiscard]] bool HitMarker() const { return hit_marker_; }

  private:
    bool Fill() {
        if (at_ >= n_) { hit_marker_ = true; return false; }
        std::uint8_t b = p_[at_++];
        if (b == 0xFF) {
            const std::uint8_t next = at_ < n_ ? p_[at_] : 0xD9;
            if (next == 0x00) {
                ++at_;  // stuffed: the 0xFF was data
            } else {
                // A real marker. Back up so the scan loop can see it, and feed
                // out zeroes -- the spec's own recommendation for a stream that
                // ends early.
                --at_;
                hit_marker_ = true;
                return false;
            }
        }
        byte_ = b;
        count_ = 8;
        return true;
    }

    const std::uint8_t* p_;
    std::size_t n_;
    std::size_t at_ = 0;
    std::uint8_t byte_ = 0;
    int count_ = 0;
    bool hit_marker_ = false;
};

int Decode(BitReader& br, const Huffman& h) {
    int code = br.Bit();
    for (int len = 1; len <= 16; ++len) {
        if (h.maxcode[len] >= 0 && code <= h.maxcode[len])
            return h.values[std::size_t(h.valptr[len] + code - h.mincode[len])];
        code = (code << 1) | br.Bit();
    }
    return -1;  // no code of any length matches: the stream is corrupt
}

// A value of `s` bits, sign-extended the way JPEG defines it: the top half of
// the range is positive and the bottom half is negative, offset so that zero
// never needs to be coded.
int Extend(int v, int s) {
    return (s == 0) ? 0 : (v < (1 << (s - 1)) ? v - (1 << s) + 1 : v);
}

// The 8x8 inverse DCT, separable: one pass over rows, one over columns.
//
// B[x][u] = C(u)/2 * cos((2x+1) u pi / 16), so the whole transform is two
// matrix multiplies. Slower than a factored AAN butterfly and about thirty
// lines shorter; a decode is dominated by Huffman anyway.
struct Basis {
    float b[8][8];
    Basis() {
        for (int x = 0; x < 8; ++x)
            for (int u = 0; u < 8; ++u) {
                const float c = u == 0 ? 0.70710678f : 1.0f;
                b[x][u] = 0.5f * c *
                          std::cos(float((2 * x + 1) * u) * 3.14159265f / 16.0f);
            }
    }
};
const Basis& TheBasis() {
    static const Basis basis;
    return basis;
}

void Idct(const float* in, std::uint8_t* out, int stride) {
    const Basis& B = TheBasis();
    float tmp[64];
    for (int v = 0; v < 8; ++v)
        for (int x = 0; x < 8; ++x) {
            float s = 0.0f;
            for (int u = 0; u < 8; ++u) s += B.b[x][u] * in[v * 8 + u];
            tmp[v * 8 + x] = s;
        }
    for (int x = 0; x < 8; ++x)
        for (int y = 0; y < 8; ++y) {
            float s = 0.0f;
            for (int v = 0; v < 8; ++v) s += B.b[y][v] * tmp[v * 8 + x];
            // Level shift: JPEG codes samples centred on zero, so 0 means 128.
            const int q = int(std::lround(s)) + 128;
            out[y * stride + x] = std::uint8_t(std::clamp(q, 0, 255));
        }
}

std::uint8_t Clamp8(float v) {
    return std::uint8_t(std::clamp(int(std::lround(v)), 0, 255));
}

}  // namespace

bool IsJpeg(std::span<const std::uint8_t> bytes) {
    return bytes.size() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xD8;
}

Texture2D Decode(std::span<const std::uint8_t> bytes, std::string& error) {
    Texture2D img;
    error.clear();
    if (!IsJpeg(bytes)) {
        error = "jpeg: not a jpeg (no SOI)";
        return img;
    }

    std::uint16_t quant[4][64] = {};
    bool quant_seen[4] = {};
    Huffman dc[4], ac[4];
    std::vector<Component> comps;
    int width = 0, height = 0;
    int restart_interval = 0;
    bool got_frame = false;

    std::size_t p = 2;
    const auto u16 = [&](std::size_t at) {
        return int(bytes[at]) << 8 | bytes[at + 1];
    };

    while (p + 1 < bytes.size()) {
        if (bytes[p] != 0xFF) { ++p; continue; }  // resync over fill bytes
        const std::uint8_t marker = bytes[p + 1];
        p += 2;
        if (marker == 0xD8 || marker == 0x01 || (marker >= 0xD0 && marker <= 0xD7))
            continue;  // standalone, no payload
        if (marker == 0xD9) break;  // EOI
        if (marker == 0xFF) { --p; continue; }  // fill

        if (p + 1 >= bytes.size()) { error = "jpeg: truncated segment header"; return img; }
        const int len = u16(p);
        if (len < 2 || p + std::size_t(len) > bytes.size()) {
            error = "jpeg: segment length runs past the end of the file";
            return img;
        }
        const std::size_t seg = p + 2, seg_end = p + std::size_t(len);

        switch (marker) {
            case 0xDB: {  // DQT
                std::size_t q = seg;
                while (q < seg_end) {
                    const int pq = bytes[q] >> 4, tq = bytes[q] & 15;
                    ++q;
                    if (tq > 3) { error = "jpeg: quantisation table id out of range"; return img; }
                    const std::size_t need = pq ? 128u : 64u;
                    if (q + need > seg_end) { error = "jpeg: short quantisation table"; return img; }
                    for (int i = 0; i < 64; ++i)
                        quant[tq][i] = pq ? std::uint16_t(int(bytes[q + std::size_t(i) * 2]) << 8 |
                                                          bytes[q + std::size_t(i) * 2 + 1])
                                          : bytes[q + std::size_t(i)];
                    quant_seen[tq] = true;
                    q += need;
                }
                break;
            }
            case 0xC4: {  // DHT
                std::size_t q = seg;
                while (q < seg_end) {
                    const int tc = bytes[q] >> 4, th = bytes[q] & 15;
                    ++q;
                    if (tc > 1 || th > 3) { error = "jpeg: huffman table id out of range"; return img; }
                    if (q + 16 > seg_end) { error = "jpeg: short huffman table"; return img; }
                    int counts[17] = {};
                    int total = 0;
                    for (int i = 1; i <= 16; ++i) {
                        counts[i] = bytes[q + std::size_t(i) - 1];
                        total += counts[i];
                    }
                    q += 16;
                    if (q + std::size_t(total) > seg_end) { error = "jpeg: short huffman values"; return img; }
                    Huffman& t = (tc == 0 ? dc[th] : ac[th]);
                    t.present = true;
                    t.values.assign(bytes.begin() + std::ptrdiff_t(q),
                                    bytes.begin() + std::ptrdiff_t(q + std::size_t(total)));
                    q += std::size_t(total);
                    // Canonical code assignment: codes of each length are
                    // consecutive, and moving to the next length shifts left.
                    int code = 0, k = 0;
                    for (int len_bits = 1; len_bits <= 16; ++len_bits) {
                        if (counts[len_bits] == 0) {
                            t.maxcode[len_bits] = -1;
                        } else {
                            t.valptr[len_bits] = k;
                            t.mincode[len_bits] = code;
                            code += counts[len_bits];
                            k += counts[len_bits];
                            t.maxcode[len_bits] = code - 1;
                        }
                        code <<= 1;
                    }
                }
                break;
            }
            case 0xDD:  // DRI
                if (seg + 2 > seg_end) { error = "jpeg: short restart interval"; return img; }
                restart_interval = u16(seg);
                break;
            case 0xC0:
            case 0xC1: {  // SOF0 baseline, SOF1 extended sequential
                if (seg + 6 > seg_end) { error = "jpeg: short frame header"; return img; }
                if (bytes[seg] != 8) {
                    error = "jpeg: only 8-bit samples are supported, got " +
                            std::to_string(int(bytes[seg]));
                    return img;
                }
                height = u16(seg + 1);
                width = u16(seg + 3);
                const int nf = bytes[seg + 5];
                if (width <= 0 || height <= 0) { error = "jpeg: zero-sized image"; return img; }
                if (std::uint64_t(width) * height > 64ull * 1024 * 1024) {
                    error = "jpeg: image too large";
                    return img;
                }
                if (nf != 1 && nf != 3) {
                    error = "jpeg: " + std::to_string(nf) +
                            "-component images (CMYK/YCCK) are not supported";
                    return img;
                }
                if (seg + 6 + std::size_t(nf) * 3 > seg_end) { error = "jpeg: short component list"; return img; }
                comps.clear();
                for (int i = 0; i < nf; ++i) {
                    const std::size_t at = seg + 6 + std::size_t(i) * 3;
                    Component c;
                    c.id = bytes[at];
                    c.h = bytes[at + 1] >> 4;
                    c.v = bytes[at + 1] & 15;
                    c.tq = bytes[at + 2];
                    if (c.h < 1 || c.h > 4 || c.v < 1 || c.v > 4 || c.tq > 3) {
                        error = "jpeg: bad component sampling or table id";
                        return img;
                    }
                    comps.push_back(c);
                }
                got_frame = true;
                break;
            }
            case 0xC2:
                error = "jpeg: progressive images (SOF2) are not supported";
                return img;
            case 0xC3: case 0xC5: case 0xC6: case 0xC7:
            case 0xC9: case 0xCA: case 0xCB:
            case 0xCD: case 0xCE: case 0xCF:
                error = "jpeg: unsupported coding process (marker 0x" +
                        std::to_string(int(marker)) + ")";
                return img;
            case 0xDA: {  // SOS -- the scan itself
                if (!got_frame) { error = "jpeg: scan before frame header"; return img; }
                if (seg >= seg_end) { error = "jpeg: short scan header"; return img; }
                const int ns = bytes[seg];
                if (ns != int(comps.size())) {
                    // A baseline file has exactly one scan covering every
                    // component. Fewer means the file is progressive-shaped, and
                    // decoding one component's worth would give a green image.
                    error = "jpeg: scan covers " + std::to_string(ns) + " of " +
                            std::to_string(comps.size()) +
                            " components; only single-scan baseline is supported";
                    return img;
                }
                if (seg + 1 + std::size_t(ns) * 2 + 3 > seg_end) { error = "jpeg: short scan header"; return img; }
                for (int i = 0; i < ns; ++i) {
                    const std::size_t at = seg + 1 + std::size_t(i) * 2;
                    const int cs = bytes[at];
                    Component* found = nullptr;
                    for (Component& c : comps)
                        if (c.id == cs) found = &c;
                    if (!found) { error = "jpeg: scan names an unknown component"; return img; }
                    found->td = bytes[at + 1] >> 4;
                    found->ta = bytes[at + 1] & 15;
                    if (found->td > 3 || found->ta > 3) { error = "jpeg: bad scan table id"; return img; }
                }

                for (const Component& c : comps) {
                    if (!quant_seen[c.tq]) { error = "jpeg: a component names a missing quantisation table"; return img; }
                    if (!dc[c.td].present || !ac[c.ta].present) {
                        error = "jpeg: a component names a missing huffman table";
                        return img;
                    }
                }

                // MCU geometry. The MCU is as wide as the most-sampled
                // component needs, and every other component contributes fewer
                // blocks to it -- that IS chroma subsampling.
                int hmax = 1, vmax = 1;
                for (const Component& c : comps) {
                    hmax = std::max(hmax, c.h);
                    vmax = std::max(vmax, c.v);
                }
                const int mcus_x = (width + 8 * hmax - 1) / (8 * hmax);
                const int mcus_y = (height + 8 * vmax - 1) / (8 * vmax);
                for (Component& c : comps) {
                    c.plane_w = mcus_x * c.h * 8;
                    c.plane_h = mcus_y * c.v * 8;
                    c.valid_w = (width * c.h + hmax - 1) / hmax;
                    c.valid_h = (height * c.v + vmax - 1) / vmax;
                    c.plane.assign(std::size_t(c.plane_w) * c.plane_h, 128);
                    c.dc_pred = 0;
                }

                BitReader br(bytes.data() + seg_end, bytes.size() - seg_end);
                float block[64];
                int since_restart = 0;
                for (int my = 0; my < mcus_y; ++my) {
                    for (int mx = 0; mx < mcus_x; ++mx) {
                        if (restart_interval > 0 && since_restart == restart_interval) {
                            if (!br.Restart()) { error = "jpeg: expected a restart marker"; return img; }
                            for (Component& c : comps) c.dc_pred = 0;
                            since_restart = 0;
                        }
                        ++since_restart;
                        for (Component& c : comps) {
                            for (int by = 0; by < c.v; ++by)
                                for (int bx = 0; bx < c.h; ++bx) {
                                    std::memset(block, 0, sizeof(block));
                                    const std::uint16_t* qt = quant[c.tq];

                                    const int t = Decode(br, dc[c.td]);
                                    if (t < 0 || t > 15) { error = "jpeg: corrupt DC code"; return img; }
                                    c.dc_pred += Extend(br.Bits(t), t);
                                    block[0] = float(c.dc_pred) * float(qt[0]);

                                    for (int k = 1; k < 64;) {
                                        const int rs = Decode(br, ac[c.ta]);
                                        if (rs < 0) { error = "jpeg: corrupt AC code"; return img; }
                                        const int r = rs >> 4, s = rs & 15;
                                        if (s == 0) {
                                            if (r != 15) break;  // EOB
                                            k += 16;             // ZRL: 16 zeroes
                                            continue;
                                        }
                                        k += r;
                                        if (k > 63) { error = "jpeg: AC run overruns the block"; return img; }
                                        block[kZigzag[k]] =
                                            float(Extend(br.Bits(s), s)) * float(qt[k]);
                                        ++k;
                                    }

                                    const int px = (mx * c.h + bx) * 8;
                                    const int py = (my * c.v + by) * 8;
                                    Idct(block,
                                         &c.plane[std::size_t(py) * c.plane_w + px],
                                         c.plane_w);
                                }
                        }
                    }
                }

                // --- colour -------------------------------------------------
                img.width = width;
                img.height = height;
                img.rgba.assign(std::size_t(width) * height * 4, 255);
                for (int y = 0; y < height; ++y)
                    for (int x = 0; x < width; ++x) {
                        std::uint8_t* d = &img.rgba[(std::size_t(y) * width + x) * 4];
                        // TRIANGULAR upsampling of the subsampled planes.
                        //
                        // A chroma sample sits at the CENTRE of the block of
                        // luma pixels it covers, so the mapping carries a half
                        // pixel at each end: output x reads source
                        // (x + 0.5)*h/hmax - 0.5. Getting that offset wrong
                        // shifts colour half a pixel against luminance, which
                        // looks like chromatic aberration along every edge.
                        //
                        // Where the component is already full resolution the
                        // coordinate lands exactly on a sample and the weights
                        // collapse to 1 and 0, so luma is never filtered.
                        // Nearest-neighbour instead would be legal and much
                        // worse: it puts a hard chroma staircase on every
                        // diagonal, and disagrees with every other decoder by
                        // up to ninety levels at a saturated edge.
                        const auto at = [&](const Component& c) {
                            const float fx =
                                (float(x) + 0.5f) * float(c.h) / float(hmax) - 0.5f;
                            const float fy =
                                (float(y) + 0.5f) * float(c.v) / float(vmax) - 0.5f;
                            const int x0 = int(std::floor(fx)), y0 = int(std::floor(fy));
                            const float tx = fx - float(x0), ty = fy - float(y0);
                            const auto tap = [&](int sx, int sy) {
                                sx = std::clamp(sx, 0, c.valid_w - 1);
                                sy = std::clamp(sy, 0, c.valid_h - 1);
                                return float(c.plane[std::size_t(sy) * c.plane_w + sx]);
                            };
                            return (tap(x0, y0) * (1 - tx) + tap(x0 + 1, y0) * tx) *
                                       (1 - ty) +
                                   (tap(x0, y0 + 1) * (1 - tx) +
                                    tap(x0 + 1, y0 + 1) * tx) * ty;
                        };
                        if (comps.size() == 1) {
                            d[0] = d[1] = d[2] = Clamp8(at(comps[0]));
                        } else {
                            const float Y = at(comps[0]);
                            const float cb = at(comps[1]) - 128.0f;
                            const float cr = at(comps[2]) - 128.0f;
                            d[0] = Clamp8(Y + 1.402f * cr);
                            d[1] = Clamp8(Y - 0.344136f * cb - 0.714136f * cr);
                            d[2] = Clamp8(Y + 1.772f * cb);
                        }
                    }
                return img;
            }
            default:
                break;  // APPn, COM, and anything else with a length: skipped
        }
        p = seg_end;
    }

    error = got_frame ? "jpeg: no scan data" : "jpeg: no frame header";
    return img;
}

}  // namespace eng::jpeg
