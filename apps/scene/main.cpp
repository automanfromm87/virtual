// Pure C++20. Proves the DEPTH BUFFER works.
//
// Nothing before this actually tested depth. A single convex sphere with
// back-face culling looks identical whether the depth test is correct,
// inverted, or absent — culling alone leaves exactly one surface per pixel.
// You need two objects whose screen discs overlap.
//
// The method is differential: render the scene, then render it again with the
// occluder deleted, and compare how much of the occluded object survives. That
// needs no analytic pixel prediction and no magic thresholds.
#include "engine/render/renderer.h"
#include "engine/scene/scene.h"

#include <cstdio>
#include <span>
#include <string>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-52s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

struct Counts {
    std::size_t warm = 0;    // near sphere, tint (1.00, 0.45, 0.35)
    std::size_t green = 0;   // far sphere,  tint (0.35, 1.00, 0.50)
    std::size_t violet = 0;  // top CUBE,    tint (0.70, 0.55, 1.00)
};

// Lambert shading multiplies each tint by a scalar, so the dominant channel is
// preserved all the way down to the terminator. That makes "which sphere is
// this pixel" a pure hue question.
Counts Classify(const eng::Image& img) {
    Counts c;
    const std::span px{img.rgba};
    const std::uint8_t bg[3] = {px[0], px[1], px[2]};  // corner is always sky
    for (std::size_t i = 0; i + 3 < px.size(); i += 4) {
        const int r = px[i], g = px[i + 1], b = px[i + 2];
        if (r == bg[0] && g == bg[1] && b == bg[2]) continue;
        if (r > g && r > b) ++c.warm;
        else if (g > r && g > b) ++c.green;
        else if (b > r && b > g) ++c.violet;
    }
    return c;
}

eng::Image Render(const eng::Scene& s, int w, int h) {
    std::string error;
    eng::Image img = eng::RenderSceneOffscreen(s, w, h, error);
    if (img.rgba.empty()) {
        std::fprintf(stderr, "render failed: %s\n", error.c_str());
        std::exit(1);
    }
    return img;
}

}  // namespace

int main() {
    constexpr int kW = 640, kH = 640;
    constexpr float kT = 0.0f;  // fixed time: the test must be deterministic

    const eng::Image full = Render(eng::ShapesDemo(kT), kW, kH);
    // Instance 0 is the near sphere, instance 1 the far one.
    const eng::Image noNear = Render(eng::ShapesDemoWithout(kT, 0), kW, kH);
    const eng::Image noFar = Render(eng::ShapesDemoWithout(kT, 1), kW, kH);

    const Counts f = Classify(full);
    const Counts a = Classify(noNear);
    const Counts b = Classify(noFar);

    if (std::FILE* fp = std::fopen("scene.ppm", "wb")) {
        std::fprintf(fp, "P6\n%d %d\n255\n", full.width, full.height);
        for (std::size_t i = 0; i + 3 < full.rgba.size(); i += 4)
            std::fwrite(&full.rgba[i], 1, 3, fp);
        std::fclose(fp);
    }

    std::printf("%dx%d  full: warm=%zu green=%zu violet=%zu\n", kW, kH, f.warm,
                f.green, f.violet);
    std::printf("        far sphere alone: green=%zu   near alone: warm=%zu\n",
                a.green, b.warm);

    Check(f.warm > 0 && f.green > 0 && f.violet > 0, "all three objects are visible");

    // THE depth test. The near sphere covers part of the far one, so deleting
    // the near sphere must reveal MORE green.
    Check(f.green < a.green, "near sphere occludes part of the far sphere");
    Check(f.green > a.green / 2, "far sphere is only PARTLY hidden, not erased");

    // THE reversed-Z direction test. The near sphere is in front, so the far
    // sphere must never take a bite out of it. If depthCompare were Less
    // instead of Greater — or clearDepth were 1.0 instead of 0.0 — the far
    // sphere would win these pixels and warm would drop.
    // The `b.warm > 0` term matters: without it this check passes vacuously
    // when the depth test rejects everything and both counts are zero.
    const long drift = long(f.warm) - long(b.warm);
    Check(b.warm > 0 && drift > -8 && drift < 8,
          "near sphere is never occluded (reversed-Z ok)");

    std::printf("  occluded green pixels: %zu (%.1f%% of the far sphere)\n",
                a.green - f.green, 100.0 * double(a.green - f.green) / double(a.green));
    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
