#include "engine/geometry/terrain.h"

#include <algorithm>
#include <cmath>

#include "engine/core/jobs.h"

namespace eng {
namespace {

int ChunkCount(int resolution, int chunk_resolution) {
    const int quads = std::max(1, resolution - 1);
    const int per_chunk = std::max(1, chunk_resolution - 1);
    return std::max(1, (quads + per_chunk - 1) / per_chunk);
}

}  // namespace

struct Terrain::Impl {
    TerrainConfig config;
    std::vector<float> heights;
    int chunks_x = 0, chunks_z = 0;
    // Per-chunk height range, so culling and the skirt depth do not have to
    // rescan the grid.
    std::vector<Vec2> chunk_range;
    float spacing = 1.0f;

    [[nodiscard]] float At(int ix, int iz) const {
        const int n = config.resolution;
        ix = std::clamp(ix, 0, n - 1);
        iz = std::clamp(iz, 0, n - 1);
        return heights[std::size_t(iz) * std::size_t(n) + std::size_t(ix)];
    }
};

Terrain::Terrain() : impl_(std::make_unique<Impl>()) {}
Terrain::~Terrain() = default;
Terrain::Terrain(Terrain&&) noexcept = default;
Terrain& Terrain::operator=(Terrain&&) noexcept = default;

bool Terrain::Valid() const { return !impl_->heights.empty(); }
const TerrainConfig& Terrain::Config() const { return impl_->config; }
int Terrain::ChunksX() const { return impl_->chunks_x; }
int Terrain::ChunksZ() const { return impl_->chunks_z; }
std::span<const float> Terrain::Heights() const {
    return {impl_->heights.data(), impl_->heights.size()};
}

void Terrain::SetHeight(int ix, int iz, float y) {
    const int n = impl_->config.resolution;
    if (ix < 0 || iz < 0 || ix >= n || iz >= n) return;
    impl_->heights[std::size_t(iz) * std::size_t(n) + std::size_t(ix)] = y;
}

void Terrain::RefreshBounds() {
    Impl& im = *impl_;
    const int per_chunk = std::max(1, im.config.chunk_resolution - 1);
    im.chunk_range.assign(std::size_t(im.chunks_x) * std::size_t(im.chunks_z),
                          Vec2{0.0f, 0.0f});
    for (int cz = 0; cz < im.chunks_z; ++cz)
        for (int cx = 0; cx < im.chunks_x; ++cx) {
            float lo = 1e30f, hi = -1e30f;
            const int x0 = cx * per_chunk, z0 = cz * per_chunk;
            for (int z = z0; z <= z0 + per_chunk; ++z)
                for (int x = x0; x <= x0 + per_chunk; ++x) {
                    const float h = im.At(x, z);
                    lo = std::min(lo, h);
                    hi = std::max(hi, h);
                }
            im.chunk_range[std::size_t(cz) * std::size_t(im.chunks_x) +
                           std::size_t(cx)] = Vec2{lo, hi};
        }
}

Terrain Terrain::FromHeights(const TerrainConfig& config,
                             std::span<const float> heights) {
    Terrain t;
    Impl& im = *t.impl_;
    im.config = config;
    im.config.resolution = std::max(2, config.resolution);
    im.config.chunk_resolution = std::max(2, config.chunk_resolution);
    const std::size_t need =
        std::size_t(im.config.resolution) * std::size_t(im.config.resolution);
    if (heights.size() < need) return t;  // invalid: Valid() stays false
    im.heights.assign(heights.begin(), heights.begin() + std::ptrdiff_t(need));
    im.spacing = im.config.world_size / float(im.config.resolution - 1);
    im.chunks_x = ChunkCount(im.config.resolution, im.config.chunk_resolution);
    im.chunks_z = im.chunks_x;
    t.RefreshBounds();
    return t;
}

Terrain Terrain::Generate(const TerrainConfig& config,
                          const std::function<float(float, float)>& fn) {
    const int n = std::max(2, config.resolution);
    std::vector<float> heights(std::size_t(n) * std::size_t(n));
    const float spacing = config.world_size / float(n - 1);
    // PARALLEL BY ROW. The generator is usually several octaves of noise per
    // sample, which at 257x257 is 66,000 calls -- enough that one thread is a
    // visible pause at load time and eight is not.
    ParallelRanges(n, 8, [&](int z0, int z1) {
        for (int z = z0; z < z1; ++z)
            for (int x = 0; x < n; ++x)
                heights[std::size_t(z) * std::size_t(n) + std::size_t(x)] =
                    fn(config.origin.x + float(x) * spacing,
                       config.origin.z + float(z) * spacing);
    });
    TerrainConfig c = config;
    c.resolution = n;
    return FromHeights(c, heights);
}

Terrain Terrain::FromImage(const TerrainConfig& config,
                           std::span<const std::uint8_t> grey, int width,
                           int height, float min_height, float max_height) {
    const int n = std::max(2, config.resolution);
    std::vector<float> heights(std::size_t(n) * std::size_t(n), min_height);
    if (width <= 0 || height <= 0 ||
        grey.size() < std::size_t(width) * std::size_t(height)) {
        TerrainConfig c = config;
        c.resolution = n;
        return FromHeights(c, heights);
    }
    // BILINEAR from the image, so a heightmap smaller than the terrain's
    // resolution does not come out as visible steps. Nearest sampling here is
    // the usual shortcut and it turns a 256-pixel image into a staircase with
    // 256 treads whatever the mesh resolution.
    for (int z = 0; z < n; ++z)
        for (int x = 0; x < n; ++x) {
            const float u = float(x) / float(n - 1) * float(width - 1);
            const float v = float(z) / float(n - 1) * float(height - 1);
            const int x0 = std::clamp(int(u), 0, width - 1);
            const int z0 = std::clamp(int(v), 0, height - 1);
            const int x1 = std::min(x0 + 1, width - 1);
            const int z1 = std::min(z0 + 1, height - 1);
            const float fx = u - float(x0), fz = v - float(z0);
            const auto at = [&](int a, int b) {
                return float(grey[std::size_t(b) * std::size_t(width) + std::size_t(a)]) /
                       255.0f;
            };
            const float top = at(x0, z0) + (at(x1, z0) - at(x0, z0)) * fx;
            const float bottom = at(x0, z1) + (at(x1, z1) - at(x0, z1)) * fx;
            const float value = top + (bottom - top) * fz;
            heights[std::size_t(z) * std::size_t(n) + std::size_t(x)] =
                min_height + (max_height - min_height) * value;
        }
    TerrainConfig c = config;
    c.resolution = n;
    return FromHeights(c, heights);
}

float Terrain::HeightAt(float x, float z) const {
    const Impl& im = *impl_;
    if (im.heights.empty()) return 0.0f;
    const float fx = (x - im.config.origin.x) / im.spacing;
    const float fz = (z - im.config.origin.z) / im.spacing;
    const int ix = int(std::floor(fx));
    const int iz = int(std::floor(fz));
    const float tx = fx - float(ix);
    const float tz = fz - float(iz);
    const float h00 = im.At(ix, iz), h10 = im.At(ix + 1, iz);
    const float h01 = im.At(ix, iz + 1), h11 = im.At(ix + 1, iz + 1);
    const float top = h00 + (h10 - h00) * tx;
    const float bottom = h01 + (h11 - h01) * tx;
    return top + (bottom - top) * tz;
}

Vec3 Terrain::NormalAt(float x, float z) const {
    const Impl& im = *impl_;
    if (im.heights.empty()) return Vec3{0.0f, 1.0f, 0.0f};
    // CENTRAL DIFFERENCES over one grid spacing. A forward difference is half
    // the code and biased: it reports the slope of the quad to the +x side
    // rather than the slope at the point, so a normal computed on one side of a
    // ridge differs from one computed a millimetre away on the other.
    const float d = im.spacing;
    const float hl = HeightAt(x - d, z), hr = HeightAt(x + d, z);
    const float hd = HeightAt(x, z - d), hu = HeightAt(x, z + d);
    return Normalize(Vec3{hl - hr, 2.0f * d, hd - hu});
}

float Terrain::SlopeAt(float x, float z) const {
    const Vec3 n = NormalAt(x, z);
    return std::acos(std::clamp(n.y, -1.0f, 1.0f)) * 57.29578f;
}

bool Terrain::Raycast(Vec3 origin, Vec3 direction, float max_distance,
                      float* out_t, Vec3* out_normal) const {
    const Impl& im = *impl_;
    if (im.heights.empty() || max_distance <= 0.0f) return false;
    const float len = Length(direction);
    if (len < 1e-9f) return false;
    const Vec3 dir = direction * (1.0f / len);

    // A LINEAR MARCH at the grid spacing, refined by bisection once the ray
    // crosses the surface.
    //
    // Not a DDA over cells with an exact per-cell test, which is the faster
    // and more precise answer -- and which needs the ray clipped to each cell,
    // both triangles intersected, and the diagonal split direction to match the
    // mesh builder's. The march agrees with HeightAt by construction, and this
    // module's whole point is that its queries agree with each other.
    const float step = im.spacing;
    float t = 0.0f;
    float previous_gap = origin.y - HeightAt(origin.x, origin.z);
    // Starting underground: report the surface right here rather than marching
    // forward and coming out the far side of a hill.
    if (previous_gap <= 0.0f) {
        if (out_t) *out_t = 0.0f;
        if (out_normal) *out_normal = NormalAt(origin.x, origin.z);
        return true;
    }

    while (t < max_distance) {
        const float next_t = std::min(t + step, max_distance);
        const Vec3 p = origin + dir * next_t;
        const float gap = p.y - HeightAt(p.x, p.z);
        if (gap <= 0.0f) {
            // Bisect between the last point above and this one below.
            float lo = t, hi = next_t;
            for (int i = 0; i < 24; ++i) {
                const float mid = (lo + hi) * 0.5f;
                const Vec3 q = origin + dir * mid;
                if (q.y - HeightAt(q.x, q.z) > 0.0f) lo = mid;
                else hi = mid;
            }
            const Vec3 hit = origin + dir * hi;
            if (out_t) *out_t = hi;
            if (out_normal) *out_normal = NormalAt(hit.x, hit.z);
            return true;
        }
        previous_gap = gap;
        t = next_t;
        if (next_t >= max_distance) break;
    }
    return false;
}

int Terrain::MaxLod() const {
    const int per_chunk = std::max(1, impl_->config.chunk_resolution - 1);
    int lod = 0;
    for (int step = 1; step < per_chunk; step *= 2) ++lod;
    return lod;
}

void Terrain::ChunkBounds(int chunk_x, int chunk_z, Vec3* out_min,
                          Vec3* out_max) const {
    const Impl& im = *impl_;
    if (!out_min || !out_max) return;
    const int per_chunk = std::max(1, im.config.chunk_resolution - 1);
    const float x0 = im.config.origin.x + float(chunk_x * per_chunk) * im.spacing;
    const float z0 = im.config.origin.z + float(chunk_z * per_chunk) * im.spacing;
    const float size = float(per_chunk) * im.spacing;
    Vec2 range{0.0f, 0.0f};
    if (chunk_x >= 0 && chunk_z >= 0 && chunk_x < im.chunks_x && chunk_z < im.chunks_z)
        range = im.chunk_range[std::size_t(chunk_z) * std::size_t(im.chunks_x) +
                               std::size_t(chunk_x)];
    *out_min = Vec3{x0, range.x - im.config.skirt_depth, z0};
    *out_max = Vec3{x0 + size, range.y, z0 + size};
}

Mesh Terrain::BuildChunk(int chunk_x, int chunk_z, int lod) const {
    const Impl& im = *impl_;
    Mesh mesh;
    if (im.heights.empty()) return mesh;
    const int per_chunk = std::max(1, im.config.chunk_resolution - 1);
    if (chunk_x < 0 || chunk_z < 0 || chunk_x >= im.chunks_x || chunk_z >= im.chunks_z)
        return mesh;

    const int step = 1 << std::clamp(lod, 0, MaxLod());
    const int quads = std::max(1, per_chunk / step);
    const int verts = quads + 1;
    const int base_x = chunk_x * per_chunk;
    const int base_z = chunk_z * per_chunk;

    // No vertex cap. This used to refuse anything over 65000 vertices because
    // indices were 16 bits and the alternative was silently wrapped indices --
    // a plausible and completely wrong surface. Indices are 32-bit now, so the
    // only cost of a large chunk is that it is one draw call and one LOD
    // decision, which is the caller's trade to make.
    mesh.vertices.reserve(std::size_t(verts) * std::size_t(verts) +
                          std::size_t(verts) * 4);
    float lo = 1e30f, hi = -1e30f;
    for (int z = 0; z < verts; ++z)
        for (int x = 0; x < verts; ++x) {
            const int gx = std::min(base_x + x * step, im.config.resolution - 1);
            const int gz = std::min(base_z + z * step, im.config.resolution - 1);
            const float wx = im.config.origin.x + float(gx) * im.spacing;
            const float wz = im.config.origin.z + float(gz) * im.spacing;
            const float wy = im.At(gx, gz);
            lo = std::min(lo, wy);
            hi = std::max(hi, wy);

            VertexIn v{};
            v.position = Vec4{wx, wy + im.config.origin.y, wz, 0.0f};
            // From the FULL-RESOLUTION grid, not from this level's neighbours.
            // A coarse chunk whose normals came from its own sparse samples has
            // visibly different shading from the fine chunk beside it, and the
            // seam shows even when the geometry lines up perfectly.
            const Vec3 n = NormalAt(wx, wz);
            v.normal = Vec4{n.x, n.y, n.z, 0.0f};
            v.color = Vec4{1.0f, 1.0f, 1.0f, 1.0f};
            v.uv = Vec4{float(gx) / float(im.config.resolution - 1),
                        float(gz) / float(im.config.resolution - 1), 0.0f, 0.0f};
            mesh.vertices.push_back(v);
        }

    mesh.indices.reserve(std::size_t(quads) * std::size_t(quads) * 6 +
                         std::size_t(quads) * 24);
    for (int z = 0; z < quads; ++z)
        for (int x = 0; x < quads; ++x) {
            const std::uint32_t a = std::uint32_t(z * verts + x);
            const std::uint32_t b = std::uint32_t(a + 1);
            const std::uint32_t c = std::uint32_t(a + verts);
            const std::uint32_t d = std::uint32_t(c + 1);
            // The diagonal ALTERNATES with the checker parity. A fixed diagonal
            // gives every quad the same bias, and on a regular slope that shows
            // as a herringbone running across the whole terrain.
            if (((x + z) & 1) == 0) {
                mesh.indices.insert(mesh.indices.end(), {a, c, b, b, c, d});
            } else {
                mesh.indices.insert(mesh.indices.end(), {a, c, d, a, d, b});
            }
        }

    // --- the skirt -----------------------------------------------------------
    //
    // A wall dropped straight down from each border vertex. It is only ever
    // seen edge-on through a crack, so its normals point outward and its
    // texture coordinates match the border -- whatever shows through the gap
    // then looks like the terrain rather than like a grey wall.
    const float skirt_y = lo - im.config.skirt_depth + im.config.origin.y;
    const auto add_skirt = [&](int start_vertex, int stride, int count, Vec3 outward) {
        const std::uint32_t first = std::uint32_t(mesh.vertices.size());
        for (int i = 0; i < count; ++i) {
            VertexIn v = mesh.vertices[std::size_t(start_vertex + i * stride)];
            v.position.y = skirt_y;
            v.normal = Vec4{outward.x, outward.y, outward.z, 0.0f};
            mesh.vertices.push_back(v);
        }
        for (int i = 0; i + 1 < count; ++i) {
            const std::uint32_t top0 = std::uint32_t(start_vertex + i * stride);
            const std::uint32_t top1 = std::uint32_t(start_vertex + (i + 1) * stride);
            const std::uint32_t bot0 = std::uint32_t(first + i);
            const std::uint32_t bot1 = std::uint32_t(first + i + 1);
            mesh.indices.insert(mesh.indices.end(),
                                {top0, bot0, top1, top1, bot0, bot1});
        }
    };
    add_skirt(0, 1, verts, Vec3{0.0f, 0.0f, -1.0f});                       // -Z
    add_skirt((verts - 1) * verts, 1, verts, Vec3{0.0f, 0.0f, 1.0f});      // +Z
    add_skirt(0, verts, verts, Vec3{-1.0f, 0.0f, 0.0f});                   // -X
    add_skirt(verts - 1, verts, verts, Vec3{1.0f, 0.0f, 0.0f});            // +X

    // Bounds from the SURFACE, not from the skirt. The skirt hangs below the
    // terrain and nothing can be standing on it, so including it would inflate
    // every chunk's culling sphere by the skirt depth for no benefit.
    const float size = float(per_chunk) * im.spacing;
    const Vec3 centre{im.config.origin.x + float(base_x) * im.spacing + size * 0.5f,
                      (lo + hi) * 0.5f + im.config.origin.y,
                      im.config.origin.z + float(base_z) * im.spacing + size * 0.5f};
    mesh.bounds.center = centre;
    mesh.bounds.radius =
        Length(Vec3{size * 0.5f, (hi - lo) * 0.5f + im.config.skirt_depth, size * 0.5f});
    return mesh;
}

}  // namespace eng
