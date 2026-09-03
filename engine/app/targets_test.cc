// The property worth testing here is NEGATIVE: a steady-state frame must not
// allocate. That is invisible in a picture — an app that recreates its render
// targets every frame draws exactly the right image and just gets slower.
#include "engine/app/targets.h"

#include <cstdio>
#include <string>

namespace {

int g_failures = 0;

void Fail(const char* what, int line) {
    std::fprintf(stderr, "targets_test.cc:%d  %s\n", line, what);
    ++g_failures;
}

#define CHECK(cond) \
    do { if (!(cond)) Fail(#cond, __LINE__); } while (0)

using namespace eng;

}  // namespace

int main() {
    std::string error;
    auto dev = rhi::Device::Create(error);
    if (!dev) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }

    {
        app::FrameTargets t(*dev, rhi::Format::RGBA8Unorm);
        t.Resize(800, 600);
        const rhi::TextureId color = t.Color("scene");
        const rhi::TextureId depth = t.Depth("scene");
        CHECK(Valid(color) && Valid(depth));
        CHECK(t.Allocations() == 2);
        CHECK(t.Width() == 800 && t.Height() == 600);

        // A THOUSAND steady frames allocate nothing more, and hand back the
        // same textures every time.
        for (int i = 0; i < 1000; ++i) {
            t.Resize(800, 600);
            CHECK(t.Color("scene") == color);
            CHECK(t.Depth("scene") == depth);
        }
        CHECK(t.Allocations() == 2);

        // Distinct names are distinct textures. Sharing one by accident would
        // make a pass read what it just wrote.
        const rhi::TextureId ao = t.Color("ao");
        CHECK(Valid(ao) && ao != color);
        CHECK(t.Allocations() == 3);
        CHECK(t.Live() == 3);

        // A real resize replaces them, once each.
        t.Resize(1024, 768);
        const rhi::TextureId color2 = t.Color("scene");
        CHECK(Valid(color2));
        CHECK(t.Width() == 1024);
        // "ao" was dropped at the resize and is only remade when asked for, so
        // an app that stopped using it stops paying for it.
        CHECK(t.Live() == 1);
        CHECK(Valid(t.Color("ao")));
        CHECK(t.Live() == 2);
        CHECK(t.Allocations() == 5);

        // Sampleable depth is a different request from ordinary depth.
        CHECK(Valid(t.Depth("readable", /*sampleable=*/true)));

        // COLOUR AND DEPTH ARE DIFFERENT NAMESPACES. They were not: both used
        // the bare name as a key, so Depth("scene") returned the colour
        // texture created by Color("scene") and an app would have bound a
        // colour target as its depth buffer.
        const rhi::TextureId c = t.Color("shared");
        const rhi::TextureId d = t.Depth("shared");
        CHECK(Valid(c) && Valid(d));
        CHECK(c != d);
    }

    // A zero-sized window (minimised) must not create anything or crash.
    {
        app::FrameTargets t(*dev, rhi::Format::RGBA8Unorm);
        t.Resize(0, 0);
        CHECK(!Valid(t.Color("scene")));
        CHECK(t.Allocations() == 0);
        t.Resize(64, 64);
        CHECK(Valid(t.Color("scene")));
        CHECK(t.Allocations() == 1);
    }

    if (g_failures == 0) std::printf("targets_test: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
