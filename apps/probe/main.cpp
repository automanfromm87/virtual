// Two quality features whose failure mode is "looks fine": parallax-corrected
// reflection probes, and the dithered crossfade that hides a level-of-detail
// swap.
//
// Both are checked by DIFFERENCE rather than by appearance. A probe with no
// parallax correction produces a perfectly convincing reflection -- it is just
// the same reflection wherever the object stands, which nobody notices until
// the object moves. A crossfade that does nothing produces a perfectly
// convincing object, which is the one it was already drawing.
#include "engine/geometry/mesh.h"
#include "engine/render/ibl.h"
#include "engine/render/renderer.h"
#include "engine/rhi/rhi.h"
#include "engine/scene/scene.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-60s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

constexpr int kW = 320, kH = 320;

double Luma(const std::vector<std::uint8_t>& p, std::size_t i) {
    return 0.2126 * p[i] + 0.7152 * p[i + 1] + 0.0722 * p[i + 2];
}

}  // namespace

int main() {
    using namespace eng;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string error;
    auto dev = rhi::Device::Create(error);
    if (!dev) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }
    const auto kFmt = rhi::Format::RGBA8Unorm;
    auto r = Renderer::Create(*dev, kFmt, error, 1);
    auto env = Environment::Create(*dev, error, 64, Renderer::kSceneFormat, 1);
    if (!r || !env) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }

    const rhi::TextureId hdr =
        dev->CreateRenderTarget(kW, kH, Renderer::kSceneFormat);
    const rhi::TextureId ldr =
        dev->CreateRenderTarget(kW, kH, kFmt, /*cpu_readable=*/true);
    const rhi::TextureId depth = dev->CreateDepthTarget(kW, kH);
    std::vector<std::uint8_t> px(std::size_t(kW) * kH * 4);

    const MeshHandle ball =
        r->UploadMesh(MakeUVSphere(0.6f, 40, 56, Vec4{1, 1, 1, 1}, Vec4{1, 1, 1, 1}));

    const auto render = [&](const Scene& scene) -> bool {
        dev->BeginFrame();
        {
            rhi::PassDesc pd;
            pd.color = hdr;
            pd.depth = depth;
            pd.clear_depth = 0.0f;
            auto e = dev->BeginPass(pd);
            r->DrawScene(e, scene, kW, kH, {});
            dev->EndPass();
        }
        {
            rhi::PassDesc pd;
            pd.color = ldr;
            auto e = dev->BeginPass(pd);
            r->DrawComposite(e, hdr, {}, {}, 0.0f, /*vignette=*/0.0f);
            dev->EndPass();
        }
        if (!dev->CommitAndWait(error)) {
            std::fprintf(stderr, "FAIL: submit: %s\n", error.c_str());
            return false;
        }
        return dev->ReadPixels(ldr, kW, kH, px);
    };

    // ------------------------------------------------------- parallax probes --
    {
        std::printf("parallax-corrected reflection probes\n");
        // A room painted differently on each wall, baked into the probe from
        // its centre. Then a mirror ball at two positions inside it.
        //
        // WITHOUT correction the cubemap is read by direction alone, so the
        // ball reflects the same walls in the same places wherever it stands --
        // convincing, and wrong the moment anything moves. WITH it, the
        // reflection is re-aimed at where the ray actually meets the room, so
        // a ball near the red wall shows more red.
        constexpr int kEW = 64, kEH = 32;
        std::vector<float> equirect(std::size_t(kEW) * kEH * 4, 0.0f);
        for (int y = 0; y < kEH; ++y)
            for (int x = 0; x < kEW; ++x) {
                const float u = (float(x) + 0.5f) / float(kEW);
                const std::size_t i = (std::size_t(y) * kEW + x) * 4;
                // Four coloured walls, by longitude.
                const int quadrant = int(u * 4.0f) & 3;
                const float c[4][3] = {{2.5f, 0.15f, 0.15f},
                                       {0.15f, 2.5f, 0.15f},
                                       {0.15f, 0.15f, 2.5f},
                                       {2.0f, 2.0f, 0.2f}};
                equirect[i + 0] = c[quadrant][0];
                equirect[i + 1] = c[quadrant][1];
                equirect[i + 2] = c[quadrant][2];
                equirect[i + 3] = 1.0f;
            }
        const rhi::TextureId eq =
            dev->CreateTexture2DFloat(kEW, kEH, equirect.data());
        dev->BeginFrame();
        {
            auto e = dev->BeginCompute({});
            env->BakeEquirect(e, eq);
            dev->EndCompute();
        }
        if (!dev->CommitAndWait(error)) {
            std::fprintf(stderr, "FAIL: bake: %s\n", error.c_str());
            return 1;
        }

        MaterialDesc mirror;
        mirror.base_color = Vec4{1, 1, 1, 1};
        mirror.roughness = 0.06f;
        mirror.metallic = 1.0f;
        const MaterialHandle mirror_mat = r->CreateMaterial(mirror, error);

        const auto scene_at = [&](float x) {
            Scene s;
            // The CAMERA FOLLOWS the ball, so it lands in the same pixels
            // either way and the only thing that differs between the two
            // renders is where the ball is in the ROOM. Leaving the camera
            // still would move the ball out of the measured patch and every
            // reading would be the black background -- which is what the first
            // version of this did.
            s.camera.eye = Vec3{x, 0.0f, 4.5f};
            s.camera.target = Vec3{x, 0.0f, 0.0f};
            s.lightColor = Vec4{0, 0, 0, 1};
            s.ambientSky = Vec3{0, 0, 0};
            s.ambientGround = Vec3{0, 0, 0};
            Instance in;
            in.mesh = ball;
            in.material = mirror_mat;
            in.model = Mat4::Translation({x, 0.0f, 0.0f});
            s.instances.push_back(in);
            return s;
        };

        const auto measure = [&]() {
            double rr = 0, gg = 0, bb = 0;
            int n = 0;
            for (int y = kH / 2 - 24; y < kH / 2 + 24; ++y)
                for (int x = kW / 2 - 24; x < kW / 2 + 24; ++x, ++n) {
                    const std::size_t i = (std::size_t(y) * kW + x) * 4;
                    rr += px[i];
                    gg += px[i + 1];
                    bb += px[i + 2];
                }
            return Vec3{float(rr / n), float(gg / n), float(bb / n)};
        };

        EnvironmentBindings b = env->Bindings();
        b.parallax = false;
        r->SetEnvironment(b);
        // The ball is centred in the frame either way -- the camera follows it
        // -- so any difference between these two is the probe and not the view.
        if (!render(scene_at(-1.6f))) return 1;
        const Vec3 off_left = measure();
        if (!render(scene_at(1.6f))) return 1;
        const Vec3 off_right = measure();

        b.parallax = true;
        b.box_min = Vec3{-3.0f, -3.0f, -3.0f};
        b.box_max = Vec3{3.0f, 3.0f, 3.0f};
        b.capture_position = Vec3{0.0f, 0.0f, 0.0f};
        r->SetEnvironment(b);
        if (!render(scene_at(-1.6f))) return 1;
        const Vec3 on_left = measure();
        if (!render(scene_at(1.6f))) return 1;
        const Vec3 on_right = measure();

        const auto shift = [](Vec3 a, Vec3 c) {
            return std::fabs(a.x - c.x) + std::fabs(a.y - c.y) +
                   std::fabs(a.z - c.z);
        };
        std::printf("    no parallax: left rgb %.1f %.1f %.1f, "
                    "right %.1f %.1f %.1f  (moved %.1f)\n",
                    double(off_left.x), double(off_left.y), double(off_left.z),
                    double(off_right.x), double(off_right.y), double(off_right.z),
                    double(shift(off_left, off_right)));
        std::printf("    parallax:    left rgb %.1f %.1f %.1f, "
                    "right %.1f %.1f %.1f  (moved %.1f)\n",
                    double(on_left.x), double(on_left.y), double(on_left.z),
                    double(on_right.x), double(on_right.y), double(on_right.z),
                    double(shift(on_left, on_right)));

        Check(shift(off_left, off_right) < 3.0,
              "without correction the reflection does not move with the object");
        Check(shift(on_left, on_right) > shift(off_left, off_right) * 4.0,
              "and with it, the object's position changes what it reflects");
        // AND IT MOVES THE RIGHT WAY. A correction that aimed the ray anywhere
        // at all would pass the check above; this one says the ball nearer a
        // wall shows more of THAT wall. The room is red at -x and blue at +x,
        // so the red/blue balance has to flip between the two positions.
        //
        // A saturation check would have been the obvious thing here and would
        // have been meaningless: the patch averages 48x48 pixels of a sphere,
        // which covers most of the hemisphere of reflection directions, so the
        // MEAN is desaturated however saturated the walls are.
        std::printf("    r-b balance: %.1f on the left, %.1f on the right\n",
                    double(on_left.x - on_left.z), double(on_right.x - on_right.z));
        Check(on_left.x > on_left.z && on_right.z > on_right.x,
              "and the ball nearer a wall shows more of that wall");
    }

    // ------------------------------------------------------- LOD crossfade --
    {
        std::printf("\nthe dithered level-of-detail crossfade\n");
        MaterialDesc md;
        md.base_color = Vec4{0.9f, 0.9f, 0.9f, 1.0f};
        md.roughness = 0.5f;
        const MaterialHandle mat = r->CreateMaterial(md, error);
        r->ClearEnvironment();

        const auto scene_with = [&](std::vector<float> fades) {
            Scene s;
            s.camera.eye = Vec3{0.0f, 0.0f, 3.0f};
            s.camera.target = Vec3{0.0f, 0.0f, 0.0f};
            s.lightDir = Vec4{0.3f, 0.5f, 0.8f, 0.0f};
            s.lightColor = Vec4{3.0f, 3.0f, 3.0f, 1.0f};
            s.ambientSky = Vec3{0.0f, 0.0f, 0.0f};
            s.ambientGround = Vec3{0.0f, 0.0f, 0.0f};
            for (float f : fades) {
                Instance in;
                in.mesh = ball;
                in.material = mat;
                in.lod_fade = f;
                s.instances.push_back(in);
            }
            return s;
        };
        // Coverage over the sphere's own footprint, not the whole frame: the
        // background is black and counting it would swamp the signal.
        const auto coverage = [&]() {
            int n = 0;
            for (int y = kH / 2 - 50; y < kH / 2 + 50; ++y)
                for (int x = kW / 2 - 50; x < kW / 2 + 50; ++x)
                    if (Luma(px, (std::size_t(y) * kW + x) * 4) > 8.0) ++n;
            return n;
        };

        if (!render(scene_with({1.0f}))) return 1;
        const int full = coverage();
        if (!render(scene_with({0.5f}))) return 1;
        const int half = coverage();
        if (!render(scene_with({0.0f}))) return 1;
        const int none = coverage();
        std::printf("    fade 1.0 covers %d px, 0.5 covers %d, 0.0 covers %d\n",
                    full, half, none);
        Check(full > 4000, "a fully faded-in object draws");
        Check(none == 0, "a fully faded-out one draws nothing at all");
        Check(std::abs(half - full / 2) < full / 12,
              "and a half fade covers half the pixels");

        // THE POINT. Two levels at complementary fades must between them cover
        // every pixel exactly once -- no holes and no doubling -- because that
        // is what turns a pop into a dissolve. An ordered dither gives it and a
        // random one does not.
        //
        // Measured as SETS of pixels from two separate renders, not by drawing
        // both at once. Drawing both at once measures the wrong thing: two
        // instances at the same position are at the same depth, reversed-Z
        // compares Greater, equal is not greater, and the second one is
        // rejected entirely. That is a property of the depth test, not of the
        // dither.
        const auto mask = [&](float f) {
            if (!render(scene_with({f}))) std::exit(1);
            std::vector<std::uint8_t> m(std::size_t(100) * 100, 0);
            for (int y = 0; y < 100; ++y)
                for (int x = 0; x < 100; ++x) {
                    const std::size_t i =
                        (std::size_t(y + kH / 2 - 50) * kW + x + kW / 2 - 50) * 4;
                    m[std::size_t(y) * 100 + x] = Luma(px, i) > 8.0 ? 1 : 0;
                }
            return m;
        };
        const std::vector<std::uint8_t> solid = mask(1.0f);
        // +0.375 for the incoming level and -0.625 for the outgoing one. Both
        // magnitudes are that level's share of the pixels; the signs are what
        // make the two patterns complements rather than nested.
        const std::vector<std::uint8_t> a = mask(0.375f);
        const std::vector<std::uint8_t> b = mask(-0.625f);
        int overlap = 0, union_count = 0, holes = 0;
        for (std::size_t i = 0; i < solid.size(); ++i) {
            if (a[i] && b[i]) ++overlap;
            if (a[i] || b[i]) ++union_count;
            if (solid[i] && !a[i] && !b[i]) ++holes;
        }
        std::printf("    fades +0.375 and -0.625: union %d px, overlap %d, "
                    "holes %d (a solid draw covers %d)\n",
                    union_count, overlap, holes, full);
        Check(overlap == 0, "complementary fades never cover the same pixel twice");
        Check(holes == 0, "and leave no pixel uncovered");
        Check(union_count == full,
              "so between them they cover exactly what one solid draw does");

        // A sweep, to show the dissolve is smooth rather than a step. A fade
        // implemented as a threshold on the whole object would give 0 then
        // jump to `full`.
        std::printf("    ");
        int prev = -1;
        bool monotonic = true;
        for (int k = 0; k <= 4; ++k) {
            const float f = float(k) / 4.0f;
            if (!render(scene_with({f}))) return 1;
            const int c = coverage();
            std::printf("%.2f:%d  ", double(f), c);
            if (prev >= 0 && c <= prev) monotonic = false;
            prev = c;
        }
        std::printf("\n");
        Check(monotonic, "and coverage rises smoothly with the fade");
    }

    std::printf(g_failures == 0 ? "\nprobe_test: all checks passed\n"
                                : "\nprobe_test: %d FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
