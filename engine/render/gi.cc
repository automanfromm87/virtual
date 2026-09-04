#include "engine/render/gi.h"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <thread>

#include "engine/physics/bvh.h"

using eng::physics::Aabb;
using eng::physics::Bvh;

namespace eng {
namespace {

constexpr float kPi = std::numbers::pi_v<float>;

// SH-L1 basis, evaluated in a direction. The constants are the normalisation
// that makes the basis orthonormal over the sphere; getting them wrong does not
// produce an obviously broken picture, it produces indirect light that is a
// constant factor too bright or too directional.
inline void ShBasis(Vec3 d, float out[4]) {
    out[0] = 0.282095f;        // 1/(2 sqrt(pi))
    out[1] = 0.488603f * d.y;  // sqrt(3/(4 pi))
    out[2] = 0.488603f * d.z;
    out[3] = 0.488603f * d.x;
}

// Möller-Trumbore. Returns the distance along `dir`, or -1.
//
// Single-sided is deliberately NOT used: a bake has to see the inside of a
// closed room, and a room is a box whose faces point outward. Culling back
// faces would make every probe inside it see nothing at all.
inline float RayTriangle(Vec3 o, Vec3 dir, Vec3 a, Vec3 b, Vec3 c, Vec3* normal) {
    const Vec3 e1 = b - a, e2 = c - a;
    const Vec3 p = Cross(dir, e2);
    const float det = Dot(e1, p);
    if (std::fabs(det) < 1e-9f) return -1.0f;  // ray parallel to the triangle
    const float inv = 1.0f / det;
    const Vec3 tv = o - a;
    const float u = Dot(tv, p) * inv;
    if (u < 0.0f || u > 1.0f) return -1.0f;
    const Vec3 q = Cross(tv, e1);
    const float v = Dot(dir, q) * inv;
    if (v < 0.0f || u + v > 1.0f) return -1.0f;
    const float t = Dot(e2, q) * inv;
    if (t <= 1e-4f) return -1.0f;
    *normal = Normalize(Cross(e1, e2));
    return t;
}

// Evenly spread directions. A Fibonacci sphere rather than random sampling:
// the bake has to be DETERMINISTIC -- two runs producing slightly different
// indirect light is impossible to distinguish from a bug, and a stochastic
// bake also means every test needs a tolerance chosen by eye.
std::vector<Vec3> FibonacciSphere(int n) {
    std::vector<Vec3> out;
    out.reserve(std::size_t(std::max(n, 1)));
    const float golden = kPi * (3.0f - std::sqrt(5.0f));
    for (int i = 0; i < n; ++i) {
        const float y = 1.0f - 2.0f * (float(i) + 0.5f) / float(n);
        const float r = std::sqrt(std::max(1.0f - y * y, 0.0f));
        const float theta = golden * float(i);
        out.push_back(Vec3{std::cos(theta) * r, y, std::sin(theta) * r});
    }
    return out;
}

struct Tracer {
    std::span<const GiTriangle> tris;
    Bvh bvh;

    struct Hit {
        float t = -1.0f;
        Vec3 normal{0, 1, 0};
        int index = -1;
    };

    void Build() {
        std::vector<Aabb> boxes;
        boxes.reserve(tris.size());
        for (const GiTriangle& t : tris) {
            Aabb b;
            b.lo = Vec3{std::min({t.a.x, t.b.x, t.c.x}), std::min({t.a.y, t.b.y, t.c.y}),
                        std::min({t.a.z, t.b.z, t.c.z})};
            b.hi = Vec3{std::max({t.a.x, t.b.x, t.c.x}), std::max({t.a.y, t.b.y, t.c.y}),
                        std::max({t.a.z, t.b.z, t.c.z})};
            // A triangle lying exactly in an axis plane -- a flat floor -- has a
            // zero-thickness box, and a ray travelling in that plane hits it
            // only by luck of the floating point. The margin costs a few false
            // candidates and removes the whole class of missed hits.
            const Vec3 pad{1e-4f, 1e-4f, 1e-4f};
            b.lo = b.lo - pad;
            b.hi = b.hi + pad;
            boxes.push_back(b);
        }
        bvh.Build(boxes);
    }

    [[nodiscard]] Hit Trace(Vec3 origin, Vec3 dir, float tmax) const {
        Hit best;
        best.t = tmax;
        bvh.QueryRay(origin, dir, tmax, [&](int i) {
            Vec3 n;
            const float t = RayTriangle(origin, dir, tris[std::size_t(i)].a,
                                        tris[std::size_t(i)].b,
                                        tris[std::size_t(i)].c, &n);
            if (t > 0.0f && t < best.t) {
                best.t = t;
                best.normal = n;
                best.index = i;
            }
        });
        if (best.index < 0) best.t = -1.0f;
        return best;
    }

    // Any hit at all, for shadow rays. Not the nearest one: a shadow ray only
    // has to know whether something is in the way, and stopping at the first
    // blocker is most of the cost of a bake.
    //
    // QueryRay visits every candidate box regardless, so the saving here is the
    // triangle test and not the traversal -- the BVH has no early-out interface
    // and adding one would change a container the physics broadphase shares.
    [[nodiscard]] bool Occluded(Vec3 origin, Vec3 dir, float tmax) const {
        bool hit = false;
        bvh.QueryRay(origin, dir, tmax, [&](int i) {
            if (hit) return;
            Vec3 n;
            const float t = RayTriangle(origin, dir, tris[std::size_t(i)].a,
                                        tris[std::size_t(i)].b,
                                        tris[std::size_t(i)].c, &n);
            if (t > 0.0f && t < tmax) hit = true;
        });
        return hit;
    }
};

Vec3 SkyRadiance(Vec3 dir, const GiBakeConfig& cfg) {
    const float t = std::clamp(dir.y * 0.5f + 0.5f, 0.0f, 1.0f);
    return cfg.sky_bottom * (1.0f - t) + cfg.sky_top * t;
}

}  // namespace

void ShAccumulate(ShProbe& into, Vec3 dir, Vec3 radiance, float weight) {
    float y[4];
    ShBasis(dir, y);
    for (int i = 0; i < 4; ++i) {
        (&into.r.x)[i] += radiance.x * y[i] * weight;
        (&into.g.x)[i] += radiance.y * y[i] * weight;
        (&into.b.x)[i] += radiance.z * y[i] * weight;
    }
}

Vec3 ShIrradiance(const ShProbe& p, Vec3 normal) {
    const Vec3 n = Normalize(normal);
    float y[4];
    ShBasis(n, y);
    // The COSINE LOBE convolution. Irradiance is not the radiance field
    // evaluated in a direction -- it is that field convolved with a clamped
    // cosine, and in SH that convolution is one constant per band: pi for band
    // 0 and 2pi/3 for band 1. Leaving them out gives indirect light that is
    // roughly a third as bright and far too directional, which reads as "the
    // GI is too subtle" rather than as an error.
    //
    // The trailing 1/pi turns irradiance into the outgoing radiance of a
    // Lambertian surface of albedo 1, which is what every caller wants.
    constexpr float kA0 = kPi;
    constexpr float kA1 = 2.0f * kPi / 3.0f;
    const float w[4] = {kA0, kA1, kA1, kA1};
    Vec3 out{0, 0, 0};
    for (int i = 0; i < 4; ++i) {
        out.x += (&p.r.x)[i] * y[i] * w[i];
        out.y += (&p.g.x)[i] * y[i] * w[i];
        out.z += (&p.b.x)[i] * y[i] * w[i];
    }
    const float inv_pi = 1.0f / kPi;
    // NEGATIVE irradiance is possible and meaningless. An L1 reconstruction of
    // a sharply directional field undershoots on the far side -- the classic
    // "ringing" -- and a negative ambient term subtracts light from a surface,
    // which shows up as black patches on the side of an object facing away from
    // a bright window.
    return Vec3{std::max(out.x * inv_pi, 0.0f), std::max(out.y * inv_pi, 0.0f),
                std::max(out.z * inv_pi, 0.0f)};
}

IrradianceVolume IrradianceVolume::Bake(std::span<const GiTriangle> tris,
                                        const GiBakeConfig& cfg) {
    IrradianceVolume vol;
    vol.cfg_ = cfg;
    const int nx = std::max(cfg.nx, 1), ny = std::max(cfg.ny, 1),
              nz = std::max(cfg.nz, 1);
    vol.cfg_.nx = nx;
    vol.cfg_.ny = ny;
    vol.cfg_.nz = nz;
    const std::size_t count = std::size_t(nx) * ny * nz;
    vol.probes_.assign(count, ShProbe{});

    Tracer tracer;
    tracer.tris = tris;
    if (!tris.empty()) tracer.Build();

    const std::vector<Vec3> dirs = FibonacciSphere(std::max(cfg.rays, 1));
    const float weight = 4.0f * kPi / float(dirs.size());
    const Vec3 sun = Normalize(cfg.sun_direction);

    const auto probe_position = [&](std::size_t i) {
        const int x = int(i % std::size_t(nx));
        const int y = int((i / std::size_t(nx)) % std::size_t(ny));
        const int z = int(i / (std::size_t(nx) * std::size_t(ny)));
        return Vec3{cfg.origin.x + float(x) * cfg.spacing.x,
                    cfg.origin.y + float(y) * cfg.spacing.y,
                    cfg.origin.z + float(z) * cfg.spacing.z};
    };

    // BOUNCE ZERO plus `bounces` more. Each pass reads the PREVIOUS pass's
    // volume at every hit point and writes a new one, so the passes cannot
    // share a buffer -- reading and writing the same probes would make the
    // result depend on the order the probes happened to be visited, and with
    // threads on, on which thread got there first.
    std::vector<ShProbe> previous(count, ShProbe{});
    const int passes = std::max(cfg.bounces, 0) + 1;

    for (int pass = 0; pass < passes; ++pass) {
        std::vector<ShProbe> current(count, ShProbe{});

        const auto do_range = [&](std::size_t from, std::size_t to) {
            for (std::size_t i = from; i < to; ++i) {
                const Vec3 p = probe_position(i);
                ShProbe acc;


                for (const Vec3& d : dirs) {
                    const Tracer::Hit h = tracer.Trace(p, d, 1e6f);
                    if (h.index < 0) {
                        ShAccumulate(acc, d, SkyRadiance(d, cfg), weight);
                        continue;
                    }

                    const GiTriangle& tri = tris[std::size_t(h.index)];
                    // Face the normal back toward the ray. Geometry is not
                    // consistently wound here -- and even when it is, a probe
                    // inside a closed room legitimately sees the backs of every
                    // wall.
                    // Face the normal back toward the ray. Geometry is not
                    // consistently wound here -- and even when it is, a probe
                    // inside a closed room legitimately sees the backs of every
                    // wall.
                    Vec3 n = h.normal;
                    if (Dot(n, d) > 0.0f) n = Vec3{-n.x, -n.y, -n.z};
                    const Vec3 hit_p = p + d * h.t + n * 1e-3f;

                    // DIRECT: the sun, shadowed.
                    Vec3 radiance = tri.emissive;
                    const float ndotl = Dot(n, sun);
                    if (ndotl > 0.0f && !tracer.Occluded(hit_p, sun, 1e5f))
                        radiance = radiance +
                                   Vec3{tri.albedo.x * cfg.sun_color.x,
                                        tri.albedo.y * cfg.sun_color.y,
                                        tri.albedo.z * cfg.sun_color.z} *
                                       (ndotl / kPi);

                    // INDIRECT: whatever the last pass found here. This one
                    // line is the whole difference between "ambient occlusion
                    // with colour" and global illumination -- it is what makes
                    // a white wall beside a red one turn pink.
                    if (pass > 0) {
                        const Vec3 bounce = vol.SampleFrom(previous, hit_p, n);
                        radiance = radiance + Vec3{tri.albedo.x * bounce.x,
                                                   tri.albedo.y * bounce.y,
                                                   tri.albedo.z * bounce.z};
                    }
                    ShAccumulate(acc, d, radiance, weight);
                }

                current[i] = acc;
            }
        };

        const int threads = std::max(cfg.threads, 0);
        if (threads <= 1) {
            do_range(0, count);
        } else {
            std::vector<std::thread> pool;
            const std::size_t chunk = (count + std::size_t(threads) - 1) /
                                      std::size_t(threads);
            for (int t = 0; t < threads; ++t) {
                const std::size_t from = std::min(count, std::size_t(t) * chunk);
                const std::size_t to = std::min(count, from + chunk);
                if (from >= to) break;
                pool.emplace_back([&, from, to] { do_range(from, to); });
            }
            for (std::thread& t : pool) t.join();
        }

        previous.swap(current);
        if (pass == passes - 1) {
            vol.probes_ = previous;
            // DARK OUTLIERS, found once at the end from the baked result.
            // The band-0 coefficient is the average radiance over the sphere,
            // so its luminance is a single honest number for "how much light
            // is here" -- and a probe inside geometry has almost none of it.
            std::vector<float> lum;
            lum.reserve(count);
            for (const ShProbe& pr : vol.probes_)
                lum.push_back(0.2126f * pr.r.x + 0.7152f * pr.g.x +
                              0.0722f * pr.b.x);
            std::vector<float> sorted = lum;
            std::nth_element(sorted.begin(), sorted.begin() + long(count / 2),
                             sorted.end());
            const float median = sorted[count / 2];
            vol.dark_ = 0;
            // A volume that is dark everywhere -- a night scene, or one baked
            // with no lights at all -- has no outliers by definition, and
            // flagging every probe in it would be worse than flagging none.
            if (median > 1e-4f)
                for (float l : lum)
                    if (l < median / 7.0f) ++vol.dark_;
        }
    }
    return vol;
}

Vec3 IrradianceVolume::Sample(Vec3 world, Vec3 normal) const {
    return SampleFrom(probes_, world, normal);
}

Vec3 IrradianceVolume::SampleFrom(const std::vector<ShProbe>& probes, Vec3 world,
                                  Vec3 normal) const {
    if (probes.empty()) return Vec3{0, 0, 0};
    const int nx = cfg_.nx, ny = cfg_.ny, nz = cfg_.nz;

    // Grid coordinates, CLAMPED. A surface just outside the volume takes the
    // nearest probe rather than zero -- an object half a metre past the edge of
    // the grid should look like the place next to it, not like a hole.
    const Vec3 g{(world.x - cfg_.origin.x) / std::max(cfg_.spacing.x, 1e-6f),
                 (world.y - cfg_.origin.y) / std::max(cfg_.spacing.y, 1e-6f),
                 (world.z - cfg_.origin.z) / std::max(cfg_.spacing.z, 1e-6f)};
    const float cx = std::clamp(g.x, 0.0f, float(nx - 1));
    const float cy = std::clamp(g.y, 0.0f, float(ny - 1));
    const float cz = std::clamp(g.z, 0.0f, float(nz - 1));
    const int x0 = std::min(int(cx), std::max(nx - 2, 0));
    const int y0 = std::min(int(cy), std::max(ny - 2, 0));
    const int z0 = std::min(int(cz), std::max(nz - 2, 0));
    const int x1 = std::min(x0 + 1, nx - 1);
    const int y1 = std::min(y0 + 1, ny - 1);
    const int z1 = std::min(z0 + 1, nz - 1);
    const float fx = cx - float(x0), fy = cy - float(y0), fz = cz - float(z0);

    // Interpolate the COEFFICIENTS and evaluate once, not the other way round.
    // Both give the same answer for the SH sum itself, but the clamp against
    // negatives in ShIrradiance is not linear -- evaluating eight probes and
    // blending the clamped results loses the cancellation that makes the
    // interpolation smooth, and the seams between cells become visible.
    ShProbe blended;
    const auto add = [&](int x, int y, int z, float w) {
        if (w <= 0.0f) return;
        const std::size_t i = std::size_t(z) * std::size_t(nx) * std::size_t(ny) +
                              std::size_t(y) * std::size_t(nx) + std::size_t(x);
        const ShProbe& p = probes[i];
        for (int k = 0; k < 4; ++k) {
            (&blended.r.x)[k] += (&p.r.x)[k] * w;
            (&blended.g.x)[k] += (&p.g.x)[k] * w;
            (&blended.b.x)[k] += (&p.b.x)[k] * w;
        }
    };
    add(x0, y0, z0, (1 - fx) * (1 - fy) * (1 - fz));
    add(x1, y0, z0, fx * (1 - fy) * (1 - fz));
    add(x0, y1, z0, (1 - fx) * fy * (1 - fz));
    add(x1, y1, z0, fx * fy * (1 - fz));
    add(x0, y0, z1, (1 - fx) * (1 - fy) * fz);
    add(x1, y0, z1, fx * (1 - fy) * fz);
    add(x0, y1, z1, (1 - fx) * fy * fz);
    add(x1, y1, z1, fx * fy * fz);
    return ShIrradiance(blended, normal);
}

int IrradianceVolume::FillDark() {
    const std::size_t count = probes_.size();
    if (count == 0 || dark_ == 0) return 0;

    // The same criterion the bake reported with, recomputed rather than stored
    // per probe: two definitions of "dark" that could drift apart is exactly
    // the kind of thing that makes a repair pass fix the wrong probes.
    std::vector<float> lum;
    lum.reserve(count);
    for (const ShProbe& pr : probes_)
        lum.push_back(0.2126f * pr.r.x + 0.7152f * pr.g.x + 0.0722f * pr.b.x);
    std::vector<float> sorted = lum;
    std::nth_element(sorted.begin(), sorted.begin() + long(count / 2), sorted.end());
    const float median = sorted[count / 2];
    if (median <= 1e-4f) return 0;  // dark everywhere: nothing to borrow from
    const float threshold = median / 7.0f;

    std::vector<char> lit(count);
    for (std::size_t i = 0; i < count; ++i) lit[i] = lum[i] >= threshold ? 1 : 0;

    const int nx = cfg_.nx, ny = cfg_.ny, nz = cfg_.nz;
    const auto index = [&](int x, int y, int z) {
        return std::size_t((z * ny + y) * nx + x);
    };

    int filled = 0;
    // Bounded rather than while(true): a volume with no lit probe at all would
    // otherwise spin, and the bound is the largest number of sweeps it can take
    // to cross the grid diagonally.
    const int max_sweeps = nx + ny + nz;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        std::vector<std::size_t> repaired;
        std::vector<ShProbe> values;
        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y)
                for (int x = 0; x < nx; ++x) {
                    const std::size_t i = index(x, y, z);
                    if (lit[i]) continue;
                    ShProbe sum{};
                    int n = 0;
                    for (int dz = -1; dz <= 1; ++dz)
                        for (int dy = -1; dy <= 1; ++dy)
                            for (int dx = -1; dx <= 1; ++dx) {
                                if (dx == 0 && dy == 0 && dz == 0) continue;
                                const int px = x + dx, py = y + dy, pz = z + dz;
                                if (px < 0 || py < 0 || pz < 0 || px >= nx || py >= ny ||
                                    pz >= nz)
                                    continue;
                                const std::size_t j = index(px, py, pz);
                                if (!lit[j]) continue;
                                sum.r = sum.r + probes_[j].r;
                                sum.g = sum.g + probes_[j].g;
                                sum.b = sum.b + probes_[j].b;
                                ++n;
                            }
                    if (n == 0) continue;
                    const float inv = 1.0f / float(n);
                    ShProbe out;
                    out.r = sum.r * inv;
                    out.g = sum.g * inv;
                    out.b = sum.b * inv;
                    repaired.push_back(i);
                    values.push_back(out);
                }
        if (repaired.empty()) break;
        // APPLIED AFTER THE WHOLE SWEEP, not during it. Writing as we go would
        // let a probe filled early in the sweep feed one filled later, so the
        // result would depend on the order the grid was walked -- and the light
        // would smear along +x, which is a visible directional bias.
        for (std::size_t k = 0; k < repaired.size(); ++k) {
            probes_[repaired[k]] = values[k];
            lit[repaired[k]] = 1;
            ++filled;
        }
    }
    dark_ -= filled;
    if (dark_ < 0) dark_ = 0;
    return filled;
}

}  // namespace eng
