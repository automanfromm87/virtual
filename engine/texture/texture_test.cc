// No test framework — from scratch means from scratch.
#include "engine/texture/texture.h"

#include <cstdio>

namespace {

int g_failures = 0;

void Fail(const char* what, int line) {
    std::fprintf(stderr, "texture_test.cc:%d  %s\n", line, what);
    ++g_failures;
}

#define CHECK(cond) \
    do { if (!(cond)) Fail(#cond, __LINE__); } while (0)

int R(const eng::Texture2D& t, int x, int y) {
    return t.rgba[(std::size_t(y) * t.width + x) * 4];
}
int A(const eng::Texture2D& t, int x, int y) {
    return t.rgba[(std::size_t(y) * t.width + x) * 4 + 3];
}

}  // namespace

int main() {
    using namespace eng;

    {
        constexpr int kSize = 64, kSquares = 8;
        const Texture2D t = MakeChecker(kSize, kSquares, Vec4{1, 1, 1, 1},
                                        Vec4{0, 0, 0, 1});
        CHECK(t.width == kSize && t.height == kSize);
        CHECK(t.rgba.size() == std::size_t(kSize) * kSize * 4);

        const int tile = kSize / kSquares;  // 8 px per square
        // Corner tile is `a` (white); its right and lower neighbours flip.
        CHECK(R(t, 0, 0) == 255);
        CHECK(R(t, tile, 0) == 0);
        CHECK(R(t, 0, tile) == 0);
        // Diagonal neighbour flips twice, so it is back to `a`.
        CHECK(R(t, tile, tile) == 255);
        // Opaque everywhere.
        CHECK(A(t, 13, 41) == 255);

        // Exactly half the pixels are each colour on an even checker.
        int white = 0;
        for (int y = 0; y < kSize; ++y)
            for (int x = 0; x < kSize; ++x)
                if (R(t, x, y) == 255) ++white;
        CHECK(white == kSize * kSize / 2);
    }

    {
        constexpr int kSize = 32, kBands = 4;
        const Texture2D t = MakeRoughnessRamp(kSize, kBands);
        // Roughness lives in RED and must span the full 0..1 range, or the
        // whole point of the ramp (showing every roughness at once) is lost.
        CHECK(R(t, 0, 0) == 0);
        CHECK(R(t, 0, kSize - 1) == 255);
        // Monotonically non-decreasing down the image.
        bool monotonic = true;
        for (int y = 1; y < kSize; ++y)
            if (R(t, 0, y) < R(t, 0, y - 1)) monotonic = false;
        CHECK(monotonic);
        // Constant across each row.
        CHECK(R(t, 0, 7) == R(t, kSize - 1, 7));
    }

    {
        const Texture2D t = MakeUVDebug(16);
        CHECK(R(t, 0, 0) == 0);          // u = 0 at the left
        CHECK(R(t, 15, 0) == 255);       // u = 1 at the right
        // v lives in green and grows downward, matching Metal's top-left origin.
        CHECK(t.rgba[(std::size_t(15) * 16 + 0) * 4 + 1] == 255);
        CHECK(t.rgba[1] == 0);
    }

    {
        // Degenerate inputs give an empty texture rather than a crash.
        CHECK(MakeChecker(0, 8, Vec4{}, Vec4{}).Empty());
        CHECK(MakeChecker(16, 0, Vec4{}, Vec4{}).Empty());
        CHECK(MakeRoughnessRamp(-4, 4).Empty());
        CHECK(MakeUVDebug(0).Empty());
    }

    if (g_failures == 0) std::printf("texture_test: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
