// Pure C++20. Knows nothing about Metal or Objective-C.
// Renders offscreen and self-checks the pixels, so it can run in CI with no
// window server and no GPU debugger.
#include "engine/render/renderer.h"

#include <cstdio>
#include <set>
#include <span>
#include <string>

int main() {
    constexpr int kW = 256, kH = 256;

    std::string error;
    const eng::Image img = eng::RenderTriangleOffscreen(kW, kH, error);
    if (img.rgba.empty()) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }

    // A render that "succeeds" but produces a solid clear color is the most
    // common silent failure in graphics. Check the pixels, every time.
    std::set<std::uint32_t> distinct;
    std::size_t covered = 0;
    const std::span pixels{img.rgba};
    for (std::size_t i = 0; i + 3 < pixels.size(); i += 4) {
        distinct.insert((std::uint32_t(pixels[i + 0]) << 24) |
                        (std::uint32_t(pixels[i + 1]) << 16) |
                        (std::uint32_t(pixels[i + 2]) << 8) |
                        std::uint32_t(pixels[i + 3]));
        if (pixels[i] > 40 || pixels[i + 1] > 40) ++covered;
    }

    if (std::FILE* f = std::fopen("triangle.ppm", "wb")) {
        std::fprintf(f, "P6\n%d %d\n255\n", img.width, img.height);
        for (std::size_t i = 0; i + 3 < img.rgba.size(); i += 4)
            std::fwrite(&img.rgba[i], 1, 3, f);
        std::fclose(f);
    }

    const double coverage = 100.0 * double(covered) / double(kW * kH);
    std::printf("%dx%d  distinct colors=%zu  coverage=%.1f%%\n", img.width,
                img.height, distinct.size(), coverage);

    const bool ok = distinct.size() > 100 && coverage > 5.0 && coverage < 60.0;
    std::printf("%s\n", ok ? "PASS" : "FAIL: that does not look like a triangle");
    return ok ? 0 : 1;
}
