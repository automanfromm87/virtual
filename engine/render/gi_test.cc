// The irradiance volume, checked against answers that are known exactly or
// known by construction.
//
// Every failure mode of a GI bake produces a picture that looks plausible: too
// dark reads as a stylistic choice, too bright reads as a stylistic choice, and
// light bleeding through a wall reads as a light you forgot about. So none of
// the checks here are "does it look lit". They are:
//
//   - a uniform environment has an exact answer, and it is the environment
//   - a wall blocks light, by a measurable factor
//   - a red wall makes a white one red, which nothing but a bounce can do
//   - each extra bounce adds light, and the series converges
#include "engine/render/gi.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

using eng::GiTriangle;
using eng::Vec3;

// An axis-aligned quad as two triangles, with a material.
void AddQuad(std::vector<GiTriangle>& out, Vec3 a, Vec3 b, Vec3 c, Vec3 d,
             Vec3 albedo, Vec3 emissive = Vec3{0, 0, 0}) {
    out.push_back(GiTriangle{a, b, c, albedo, emissive});
    out.push_back(GiTriangle{a, c, d, albedo, emissive});
}

}  // namespace

int main() {
    using namespace eng;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    {
        // THE UNIFORM CASE, which has an exact answer. A probe with nothing
        // around it, under a sky of one constant radiance L in every direction,
        // must report irradiance L on every normal -- the cosine integral of a
        // constant is that constant, whichever way the surface faces.
        //
        // This one check pins the SH normalisation, the cosine-lobe
        // convolution, the 1/pi and the ray weighting all at once. Get any of
        // them wrong and the answer is a constant factor off, which is
        // invisible in a picture and obvious here.
        std::printf("a constant environment integrates to itself\n");
        GiBakeConfig cfg;
        cfg.nx = cfg.ny = cfg.nz = 2;
        cfg.rays = 512;
        cfg.bounces = 0;
        cfg.sky_top = Vec3{0.6f, 0.6f, 0.6f};
        cfg.sky_bottom = Vec3{0.6f, 0.6f, 0.6f};
        const IrradianceVolume v = IrradianceVolume::Bake({}, cfg);

        float worst = 0.0f;
        const Vec3 normals[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
                                 {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        for (const Vec3& n : normals) {
            const Vec3 e = v.Sample(Vec3{0.5f, 0.5f, 0.5f}, n);
            worst = std::max(worst, std::fabs(e.x - 0.6f));
        }
        std::printf("    sky 0.600 -> irradiance %.4f, worst error over six "
                    "normals %.5f\n",
                    v.Sample(Vec3{0.5f, 0.5f, 0.5f}, Vec3{0, 1, 0}).x, double(worst));
        Check(worst < 0.01f,
              "irradiance equals the sky's radiance, on every normal");
    }

    {
        std::printf("\na wall blocks light\n");
        // A ground plane under a bright sky, and a large opaque slab above one
        // half of it. The probe under the slab must be much darker.
        std::vector<GiTriangle> tris;
        AddQuad(tris, Vec3{-20, 0, -20}, Vec3{20, 0, -20}, Vec3{20, 0, 20},
                Vec3{-20, 0, 20}, Vec3{0.5f, 0.5f, 0.5f});
        // The roof, over x in [0, 20] only.
        AddQuad(tris, Vec3{0, 6, -20}, Vec3{20, 6, -20}, Vec3{20, 6, 20},
                Vec3{0, 6, 20}, Vec3{0.05f, 0.05f, 0.05f});

        GiBakeConfig cfg;
        cfg.nx = 9;
        cfg.ny = 1;
        cfg.nz = 1;
        cfg.origin = Vec3{-8.0f, 2.0f, 0.0f};
        cfg.spacing = Vec3{2.0f, 1.0f, 1.0f};
        cfg.rays = 512;
        cfg.bounces = 0;
        cfg.sun_color = Vec3{0, 0, 0};  // sky only: this is about occlusion
        cfg.sky_top = Vec3{1.0f, 1.0f, 1.0f};
        cfg.sky_bottom = Vec3{0.2f, 0.2f, 0.2f};
        const IrradianceVolume v = IrradianceVolume::Bake(tris, cfg);

        const Vec3 up{0, 1, 0};
        const float open = v.Sample(Vec3{-6.0f, 2.0f, 0.0f}, up).x;
        const float covered = v.Sample(Vec3{6.0f, 2.0f, 0.0f}, up).x;
        std::printf("    open sky above: %.4f, under the roof: %.4f (%.0f%% "
                    "darker)\n",
                    double(open), double(covered),
                    100.0 * (1.0 - double(covered) / double(open)));
        Check(open > 0.3f, "the open probe sees the sky");
        Check(covered < open * 0.25f, "and the covered one is far darker");
    }

    {
        std::printf("\ncolour bleeding: a red wall makes a white one red\n");
        // The Cornell arrangement, cut down. A white floor with a saturated red
        // wall standing on it, lit from directly above. With no bounce the
        // floor is grey; with one bounce the part next to the wall is pink.
        //
        // NOTHING ELSE CAN DO THIS. Ambient occlusion darkens near the wall,
        // an image-based probe tints the whole scene equally, and a second
        // light source would have to be placed by hand. A red channel that
        // rises next to a red wall and nowhere else is a bounce, by
        // construction.
        std::vector<GiTriangle> tris;
        AddQuad(tris, Vec3{-5, 0, -5}, Vec3{5, 0, -5}, Vec3{5, 0, 5},
                Vec3{-5, 0, 5}, Vec3{0.8f, 0.8f, 0.8f});  // white floor
        AddQuad(tris, Vec3{-5, 0, -1}, Vec3{-5, 4, -1}, Vec3{-5, 4, 1},
                Vec3{-5, 0, 1}, Vec3{0.75f, 0.05f, 0.05f});  // red wall at x=-5

        GiBakeConfig cfg;
        cfg.nx = 11;
        cfg.ny = 1;
        cfg.nz = 1;
        cfg.origin = Vec3{-4.8f, 0.4f, 0.0f};
        cfg.spacing = Vec3{0.9f, 1.0f, 1.0f};
        cfg.rays = 1024;
        // The sun comes over the wall's shoulder, so the wall's inward face
        // (+x) is lit: ndotl = 0.7. With the sun straight up the wall gets
        // nothing at all, the direct reference is identically zero, and the
        // red/green ratio it is compared against is a division by nothing.
        cfg.sun_direction = Vec3{0.7f, 0.7f, 0.0f};
        cfg.sun_color = Vec3{4.0f, 4.0f, 4.0f};
        // A little sky, so the no-bounce reference is a real neutral number
        // rather than zero.
        cfg.sky_top = Vec3{0.10f, 0.10f, 0.10f};
        cfg.sky_bottom = Vec3{0.06f, 0.06f, 0.06f};

        cfg.bounces = 0;
        const IrradianceVolume direct = IrradianceVolume::Bake(tris, cfg);
        cfg.bounces = 1;
        const IrradianceVolume bounced = IrradianceVolume::Bake(tris, cfg);

        const Vec3 up{0, 1, 0};
        const Vec3 near_wall{-4.4f, 0.4f, 0.0f};
        const Vec3 far_wall{4.0f, 0.4f, 0.0f};
        const Vec3 d_near = direct.Sample(near_wall, up);
        const Vec3 b_near = bounced.Sample(near_wall, up);
        const Vec3 b_far = bounced.Sample(far_wall, up);

        const auto ratio = [](Vec3 c) { return c.x / std::max(c.y, 1e-6f); };
        std::printf("    beside the wall, no bounce: rgb %.4f %.4f %.4f  "
                    "(r/g %.3f)\n",
                    double(d_near.x), double(d_near.y), double(d_near.z),
                    double(ratio(d_near)));
        std::printf("    beside the wall, one bounce: rgb %.4f %.4f %.4f  "
                    "(r/g %.3f)\n",
                    double(b_near.x), double(b_near.y), double(b_near.z),
                    double(ratio(b_near)));
        std::printf("    nine metres away, one bounce: rgb %.4f %.4f %.4f  "
                    "(r/g %.3f)\n",
                    double(b_far.x), double(b_far.y), double(b_far.z),
                    double(ratio(b_far)));

        // The no-bounce probe is ALREADY red, and that is correct rather than
        // a failure: standing 60 cm from a directly-lit red wall, most of what
        // the probe can see IS the red wall. It is not colour bleeding -- the
        // floor has not changed -- but it means the absolute colour cannot be
        // the measurement.
        //
        // The BOUNCE is the difference between the two bakes, so that is what
        // gets measured. It is red beside the wall and neutral away from it,
        // and nothing but a bounce produces that: ambient occlusion only
        // darkens, an image-based probe tints everything equally, and a second
        // light would have to be placed by hand.
        const Vec3 bounce_near{b_near.x - d_near.x, b_near.y - d_near.y,
                               b_near.z - d_near.z};
        const Vec3 d_far = direct.Sample(far_wall, up);
        const Vec3 bounce_far{b_far.x - d_far.x, b_far.y - d_far.y,
                              b_far.z - d_far.z};
        std::printf("    what the bounce ADDED: beside the wall %.4f %.4f %.4f "
                    "(r/g %.2f), nine metres away %.4f %.4f %.4f (r/g %.2f)\n",
                    double(bounce_near.x), double(bounce_near.y),
                    double(bounce_near.z), double(ratio(bounce_near)),
                    double(bounce_far.x), double(bounce_far.y),
                    double(bounce_far.z), double(ratio(bounce_far)));
        Check(bounce_near.x > 0.02f, "the bounce adds real light beside the wall");
        Check(ratio(bounce_near) > 3.0f, "and what it adds is strongly red");
        // AND ONLY NEAR THE WALL. A bake that simply tinted everything red --
        // an ambient term set wrongly, say -- would pass the check above.
        Check(bounce_near.x > bounce_far.x * 3.0f,
              "and it falls off with distance from the wall");
    }

    {
        std::printf("\nbounces add light and the series converges\n");
        // A closed white box. Every bounce puts more light in and the total has
        // to settle: the geometric series of albedo 0.8 converges, and a bake
        // whose bounces grew without bound would mean energy is being created.
        std::vector<GiTriangle> tris;
        const Vec3 white{0.8f, 0.8f, 0.8f};
        const float s = 4.0f;
        AddQuad(tris, Vec3{-s, 0, -s}, Vec3{s, 0, -s}, Vec3{s, 0, s}, Vec3{-s, 0, s}, white);
        AddQuad(tris, Vec3{-s, 2 * s, -s}, Vec3{s, 2 * s, -s}, Vec3{s, 2 * s, s},
                Vec3{-s, 2 * s, s}, white, Vec3{2.0f, 2.0f, 2.0f});  // glowing ceiling
        AddQuad(tris, Vec3{-s, 0, -s}, Vec3{-s, 2 * s, -s}, Vec3{-s, 2 * s, s},
                Vec3{-s, 0, s}, white);
        AddQuad(tris, Vec3{s, 0, -s}, Vec3{s, 2 * s, -s}, Vec3{s, 2 * s, s},
                Vec3{s, 0, s}, white);
        AddQuad(tris, Vec3{-s, 0, -s}, Vec3{-s, 2 * s, -s}, Vec3{s, 2 * s, -s},
                Vec3{s, 0, -s}, white);
        AddQuad(tris, Vec3{-s, 0, s}, Vec3{-s, 2 * s, s}, Vec3{s, 2 * s, s},
                Vec3{s, 0, s}, white);

        GiBakeConfig cfg;
        cfg.nx = cfg.ny = cfg.nz = 3;
        cfg.origin = Vec3{-2.0f, 2.0f, -2.0f};
        cfg.spacing = Vec3{2.0f, 2.0f, 2.0f};
        cfg.rays = 512;
        cfg.sun_color = Vec3{0, 0, 0};   // the ceiling is the only source
        cfg.sky_top = Vec3{0, 0, 0};
        cfg.sky_bottom = Vec3{0, 0, 0};

        std::vector<float> at;
        for (int b = 0; b <= 4; ++b) {
            cfg.bounces = b;
            const IrradianceVolume v = IrradianceVolume::Bake(tris, cfg);
            at.push_back(v.Sample(Vec3{0, 4, 0}, Vec3{0, 1, 0}).x);
            std::printf("    %d bounce%s: %.4f\n", b, b == 1 ? " " : "s", double(at.back()));
        }
        bool increasing = true;
        for (std::size_t i = 1; i < at.size(); ++i)
            if (at[i] <= at[i - 1]) increasing = false;
        Check(increasing, "every extra bounce adds light");
        // CONVERGENCE. Each bounce multiplies by roughly the albedo, so the
        // increments have to shrink. A bake that added the SAME amount each
        // time would be creating energy, and the symptom is a room that gets
        // brighter the longer you bake it.
        const float first = at[1] - at[0];
        const float last = at[4] - at[3];
        std::printf("    the first bounce adds %.4f, the fourth adds %.4f "
                    "(ratio %.3f)\n",
                    double(first), double(last), double(last / first));
        // Each bounce multiplies by roughly the albedo, so three more bounces
        // give 0.8^3 = 0.51 -- the measured 0.54 is that, and a threshold of
        // 0.5 was tighter than the physics allows. What matters is that the
        // increments SHRINK monotonically; a bake adding the same amount each
        // time would be creating energy.
        bool shrinking = true;
        for (std::size_t i = 2; i < at.size(); ++i)
            if (at[i] - at[i - 1] >= at[i - 1] - at[i - 2]) shrinking = false;
        Check(shrinking, "and each adds less than the one before");
        Check(last < first * 0.7f, "by a clear margin, not by a rounding error");
        // Against the analytic answer for a closed diffuse box: the series
        // sums to 1/(1 - albedo), so the total cannot exceed five times the
        // direct term at albedo 0.8.
        Check(at[4] < at[0] * 5.0f,
              "and the total stays under the geometric series' limit");
    }

    {
        std::printf("\nprobes that baked dark are reported\n");
        // A probe inside a wall bakes to near black and then drags down every
        // surface that interpolates against it -- the classic dark blotch along
        // a skirting board. It is reported rather than corrected, because the
        // fix is to move the grid and that is the caller's decision.
        //
        // The hard part is telling "inside a wall" from "standing in a room".
        // Both see the backs of surfaces -- a room is a box wound outward and
        // you are inside it -- so a back-face count cannot separate them, and
        // "most rays hit something close" fails on a thin slab because most
        // rays escape sideways. Both were tried against this scene and both
        // flagged nothing. What DOES separate them is that one is dark and the
        // other is lit, so that is what is measured.
        std::vector<GiTriangle> tris;
        const Vec3 grey{0.5f, 0.5f, 0.5f};
        // A thick slab from y = 0 to y = 1, closed on all six sides. The
        // winding is not consistently outward and deliberately so: the check
        // this exercises must not depend on it, because nothing else in this
        // engine guarantees it either.
        const float bx = 10.0f;
        AddQuad(tris, Vec3{-bx, 0, -bx}, Vec3{bx, 0, -bx}, Vec3{bx, 0, bx},
                Vec3{-bx, 0, bx}, grey);
        AddQuad(tris, Vec3{-bx, 1, -bx}, Vec3{bx, 1, -bx}, Vec3{bx, 1, bx},
                Vec3{-bx, 1, bx}, grey);
        for (float x : {-bx, bx})
            AddQuad(tris, Vec3{x, 0, -bx}, Vec3{x, 1, -bx}, Vec3{x, 1, bx},
                    Vec3{x, 0, bx}, grey);
        for (float z : {-bx, bx})
            AddQuad(tris, Vec3{-bx, 0, z}, Vec3{-bx, 1, z}, Vec3{bx, 1, z},
                    Vec3{bx, 0, z}, grey);
        // And a large courtyard around it, whose walls are twenty metres away.
        // OPEN TO THE SKY on purpose: sealed, every probe in the scene bakes
        // dark, the median is zero and the outlier test correctly finds no
        // outliers -- which is the right answer to the wrong question. The
        // check needs a scene where the dark probes are actually unusual.
        const float rx = 30.0f;
        AddQuad(tris, Vec3{-rx, -5, -rx}, Vec3{rx, -5, -rx}, Vec3{rx, -5, rx},
                Vec3{-rx, -5, rx}, grey);
        for (float x : {-rx, rx})
            AddQuad(tris, Vec3{x, -5, -rx}, Vec3{x, 25, -rx}, Vec3{x, 25, rx},
                    Vec3{x, -5, rx}, grey);
        for (float z : {-rx, rx})
            AddQuad(tris, Vec3{-rx, -5, z}, Vec3{-rx, 25, z}, Vec3{rx, 25, z},
                    Vec3{rx, -5, z}, grey);

        GiBakeConfig cfg;
        cfg.nx = 3;
        cfg.ny = 3;
        cfg.nz = 3;
        // y = 0.5 is inside the slab; y = 5.5 and 10.5 are in the open room.
        cfg.origin = Vec3{-2.0f, 0.5f, -2.0f};
        cfg.spacing = Vec3{2.0f, 5.0f, 2.0f};
        cfg.rays = 128;
        cfg.bounces = 0;
        cfg.sun_direction = Vec3{0.3f, 0.9f, 0.2f};
        cfg.sun_color = Vec3{3.0f, 3.0f, 3.0f};
        const IrradianceVolume v = IrradianceVolume::Bake(tris, cfg);
        std::printf("    %d of %d probes baked dark "
                    "(9 are inside the slab, 18 are in the open room)\n",
                    v.DarkProbes(), cfg.nx * cfg.ny * cfg.nz);
        Check(v.DarkProbes() == 9,
              "the nine probes inside the slab are flagged, and no others");
    }

    {
        std::printf("\nthe bake is deterministic, threaded or not\n");
        std::vector<GiTriangle> tris;
        AddQuad(tris, Vec3{-6, 0, -6}, Vec3{6, 0, -6}, Vec3{6, 0, 6}, Vec3{-6, 0, 6},
                Vec3{0.7f, 0.3f, 0.2f});
        AddQuad(tris, Vec3{-6, 0, -6}, Vec3{-6, 5, -6}, Vec3{-6, 5, 6}, Vec3{-6, 0, 6},
                Vec3{0.2f, 0.7f, 0.3f});

        GiBakeConfig cfg;
        cfg.nx = 5;
        cfg.ny = 3;
        cfg.nz = 5;
        cfg.origin = Vec3{-4.0f, 0.5f, -4.0f};
        cfg.spacing = Vec3{2.0f, 1.5f, 2.0f};
        cfg.rays = 256;
        cfg.bounces = 2;

        cfg.threads = 0;
        const IrradianceVolume a = IrradianceVolume::Bake(tris, cfg);
        cfg.threads = 6;
        const IrradianceVolume b = IrradianceVolume::Bake(tris, cfg);

        double worst = 0.0;
        for (std::size_t i = 0; i < a.Probes().size(); ++i)
            for (int k = 0; k < 4; ++k) {
                worst = std::max(worst, std::fabs(double((&a.Probes()[i].r.x)[k]) -
                                                  double((&b.Probes()[i].r.x)[k])));
                worst = std::max(worst, std::fabs(double((&a.Probes()[i].g.x)[k]) -
                                                  double((&b.Probes()[i].g.x)[k])));
            }
        std::printf("    %zu probes, worst coefficient difference %.3e\n",
                    a.Probes().size(), worst);
        // EXACTLY zero. Every thread reads the previous pass and writes its own
        // slice of this one, so there is nothing for the schedule to change --
        // and a bake that differed by a little would be a bake with a race, not
        // a bake with rounding.
        Check(worst == 0.0, "six threads produce a bit-identical bake");
    }

    std::printf(g_failures == 0 ? "\ngi_test: all checks passed\n"
                                : "\ngi_test: %d FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
