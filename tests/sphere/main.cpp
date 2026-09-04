// Pure C++20. Knows nothing about Metal or Objective-C.
// Renders the sphere offscreen and self-checks the pixels.
//
// Unlike apps/headless, these checks are ORIENTATION-SENSITIVE. A colour
// histogram is invariant under any permutation of the framebuffer, so it cannot
// catch a flipped Y, an inside-out cull, or an inverted normal. Checking that
// the lit side is where the light is does catch all three.
#include "engine/render/renderer.h"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <span>
#include <string>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-46s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

}  // namespace

int main(int argc, char** argv) {
    constexpr int kW = 512, kH = 512;
    // Optional rotation in radians, so you can eyeball the model matrix without
    // rebuilding. The checks below hold at any angle — a sphere is a sphere.
    const float angle = argc > 1 ? float(std::atof(argv[1])) : 0.7f;

    std::string error;
    const eng::Image img = eng::RenderSphereOffscreen(kW, kH, angle, error);
    if (img.rgba.empty()) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }

    const std::span px{img.rgba};
    auto at = [&](int x, int y) { return std::size_t(y) * kW * 4 + std::size_t(x) * 4; };

    // The clear colour is whatever the renderer used; read it from a corner
    // rather than hardcoding it and coupling two files together.
    const std::size_t c0 = at(0, 0);
    const std::uint8_t bg[3] = {px[c0], px[c0 + 1], px[c0 + 2]};
    auto isSphere = [&](std::size_t i) {
        return px[i] != bg[0] || px[i + 1] != bg[1] || px[i + 2] != bg[2];
    };

    std::set<std::uint32_t> distinct;
    std::size_t covered = 0;
    int minx = kW, maxx = -1, miny = kH, maxy = -1;
    double litSum = 0, darkSum = 0;
    std::size_t litN = 0, darkN = 0;

    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const std::size_t i = at(x, y);
            if (!isSphere(i)) continue;
            ++covered;
            distinct.insert((std::uint32_t(px[i]) << 16) |
                            (std::uint32_t(px[i + 1]) << 8) | std::uint32_t(px[i + 2]));
            if (x < minx) minx = x;
            if (x > maxx) maxx = x;
            if (y < miny) miny = y;
            if (y > maxy) maxy = y;

            // Light points up (+Y) and right (+X); row 0 is the TOP of the
            // image, so the lit quadrant is upper-right.
            const double lum = 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2];
            if (x > kW / 2 && y < kH / 2) { litSum += lum; ++litN; }
            if (x < kW / 2 && y > kH / 2) { darkSum += lum; ++darkN; }
        }
    }

    if (std::FILE* f = std::fopen("sphere.ppm", "wb")) {
        std::fprintf(f, "P6\n%d %d\n255\n", img.width, img.height);
        for (std::size_t i = 0; i + 3 < img.rgba.size(); i += 4)
            std::fwrite(&img.rgba[i], 1, 3, f);
        std::fclose(f);
    }

    // The point of the disc that faces the camera dead-on. Its normal is
    // (0,0,+1) if we are seeing the NEAR hemisphere and (0,0,-1) if back-face
    // culling is inverted and we are seeing the inside of the far one. The
    // light has a positive z component, so the first is lit and the second
    // falls all the way back to ambient. Everything else about the two images
    // — silhouette, coverage, roundness, even which side looks brighter — is
    // identical, so this is the only cheap thing that tells them apart.
    double centreSum = 0;
    std::size_t centreN = 0;
    for (int y = kH / 2 - 12; y < kH / 2 + 12; ++y) {
        for (int x = kW / 2 - 12; x < kW / 2 + 12; ++x) {
            const std::size_t i = at(x, y);
            centreSum += 0.2126 * px[i] + 0.7152 * px[i + 1] + 0.0722 * px[i + 2];
            ++centreN;
        }
    }
    const double centre = centreSum / double(centreN);

    const double coverage = 100.0 * double(covered) / double(kW * kH);
    const double boxW = maxx - minx + 1, boxH = maxy - miny + 1;
    const double aspect = boxW / boxH;
    const double lit = litN ? litSum / double(litN) : 0.0;
    const double dark = darkN ? darkSum / double(darkN) : 0.0;

    std::printf(
        "%dx%d  coverage=%.1f%%  distinct=%zu  bbox=%.0fx%.0f  lit=%.1f dark=%.1f centre=%.1f\n",
        img.width, img.height, coverage, distinct.size(), boxW, boxH, lit, dark, centre);

    // A sphere of radius 1 seen from 3 units away through a 60-degree vertical
    // FOV subtends asin(1/3); its silhouette is a disc of NDC radius
    // tan(asin(1/3))/tan(30 deg) = 0.6124, i.e. pi*0.6124^2/4 = 29.5% of frame.
    Check(coverage > 27.0 && coverage < 32.0, "coverage matches analytic disc (29.5%)");
    Check(distinct.size() > 500, "shaded, not flat (>500 distinct colours)");
    Check(aspect > 0.97 && aspect < 1.03, "silhouette is round, not stretched");
    Check(lit > dark * 1.5, "lit side faces the light (normals + winding)");
    Check(centre > 25.0, "camera-facing point is lit (near hemisphere, not far)");

    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
