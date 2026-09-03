// Pure C++20. The surface-detail layer, checked as numbers.
//
// Four things went in together because they all touch the vertex format and the
// material, and each one is measured here by the quantity it is supposed to
// move:
//
//   NORMAL MAPS   a flat map must be indistinguishable from no map, a bumpy one
//                 must raise the variance, and moving the LIGHT must move which
//                 side of each bump is lit. The third is the one that proves
//                 the tangent frame is oriented rather than merely non-zero.
//   MIP CHAINS    a fine texture at a grazing angle, measured as the
//                 pixel-to-pixel variance of the far half of the floor. Without
//                 a chain that number is the aliasing.
//   32-BIT INDEX  a mesh with more than 65535 vertices has to draw as itself.
//                 Under 16-bit indices it drew as a plausible mess.
//   EMISSIVE      a surface with no light on it at all has to be bright.
#include "engine/asset/png.h"
#include "engine/geometry/mesh.h"
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
    std::printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

constexpr int kW = 400, kH = 400;

double Luma(const std::vector<std::uint8_t>& p, std::size_t i) {
    return 0.2126 * p[i] + 0.7152 * p[i + 1] + 0.0722 * p[i + 2];
}

// Mean and variance of luminance over a rectangle.
struct Region {
    double mean = 0.0, variance = 0.0;
    int count = 0;
};

Region Measure(const std::vector<std::uint8_t>& px, int x0, int y0, int x1,
               int y1) {
    Region r;
    double sum = 0.0, sum2 = 0.0;
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) {
            const double l = Luma(px, (std::size_t(y) * kW + x) * 4);
            sum += l;
            sum2 += l * l;
            ++r.count;
        }
    if (r.count == 0) return r;
    r.mean = sum / r.count;
    r.variance = sum2 / r.count - r.mean * r.mean;
    return r;
}

// Mean squared difference between horizontally adjacent pixels. This is the
// aliasing measure: a correctly filtered minified texture is SMOOTH across the
// screen even though the texture itself is not, while an unfiltered one jumps
// between texels from one pixel to the next.
//
// Not the plain variance, which a mip chain also lowers by blurring the whole
// image -- this looks only at the highest frequency the screen can carry, which
// is exactly the band that aliases.
double HighFrequency(const std::vector<std::uint8_t>& px, int x0, int y0, int x1,
                     int y1) {
    double sum = 0.0;
    int n = 0;
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x + 1 < x1; ++x) {
            const double a = Luma(px, (std::size_t(y) * kW + x) * 4);
            const double b = Luma(px, (std::size_t(y) * kW + x + 1) * 4);
            sum += (a - b) * (a - b);
            ++n;
        }
    return n ? sum / n : 0.0;
}

// A normal map of round bumps on a grid. Tangent-space, so a flat surface is
// (128, 128, 255) and a slope is a displacement of x and y.
std::vector<std::uint8_t> BumpNormals(int size, int bumps) {
    std::vector<std::uint8_t> px(std::size_t(size) * size * 4, 0);
    const float period = float(size) / float(bumps);
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x) {
            // Position within one cell, in -1..1.
            const float u = std::fmod(float(x), period) / period * 2.0f - 1.0f;
            const float v = std::fmod(float(y), period) / period * 2.0f - 1.0f;
            const float r2 = u * u + v * v;
            float nx = 0.0f, ny = 0.0f;
            if (r2 < 1.0f) {
                // The gradient of a hemisphere: the surface normal of a dome of
                // radius 1 at (u, v) is exactly (u, v, sqrt(1 - r^2)).
                nx = u;
                ny = v;
            }
            const std::size_t i = (std::size_t(y) * size + x) * 4;
            px[i + 0] = std::uint8_t(std::clamp((nx * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
            px[i + 1] = std::uint8_t(std::clamp((ny * 0.5f + 0.5f) * 255.0f, 0.0f, 255.0f));
            px[i + 2] = std::uint8_t(
                std::clamp((std::sqrt(std::max(1.0f - r2, 0.0f)) * 0.5f + 0.5f) * 255.0f,
                           0.0f, 255.0f));
            px[i + 3] = 255;
        }
    return px;
}

std::vector<std::uint8_t> Checker(int size, int squares) {
    std::vector<std::uint8_t> px(std::size_t(size) * size * 4, 0);
    const int cell = std::max(1, size / squares);
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x) {
            const bool on = ((x / cell) + (y / cell)) & 1;
            const std::size_t i = (std::size_t(y) * size + x) * 4;
            px[i + 0] = px[i + 1] = px[i + 2] = on ? 245 : 20;
            px[i + 3] = 255;
        }
    return px;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string error;
    auto dev = eng::rhi::Device::Create(error);
    if (!dev) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }
    const auto kFmt = eng::rhi::Format::RGBA8Unorm;
    auto r = eng::Renderer::Create(*dev, kFmt, error, 1);
    if (!r) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }

    const eng::rhi::TextureId hdr =
        dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
    const eng::rhi::TextureId ldr =
        dev->CreateRenderTarget(kW, kH, kFmt, /*cpu_readable=*/true);
    const eng::rhi::TextureId depth = dev->CreateDepthTarget(kW, kH);
    std::vector<std::uint8_t> px(std::size_t(kW) * kH * 4);

    const auto render = [&](const eng::Scene& scene) -> bool {
        dev->BeginFrame();
        {
            eng::rhi::PassDesc pd;
            pd.color = hdr;
            pd.depth = depth;
            pd.clear_depth = 0.0f;
            auto e = dev->BeginPass(pd);
            r->DrawScene(e, scene, kW, kH, {});
            dev->EndPass();
        }
        {
            eng::rhi::PassDesc pd;
            pd.color = ldr;
            auto e = dev->BeginPass(pd);
            r->DrawComposite(e, hdr, {}, {}, 0.0f, /*vignette=*/0.0f);
            dev->EndPass();
        }
        if (!dev->CommitAndWait(error)) {
            std::fprintf(stderr, "FAIL: submit: %s\n", error.c_str());
            return false;
        }
        if (!dev->ReadPixels(ldr, kW, kH, px)) {
            std::fprintf(stderr, "FAIL: readback\n");
            return false;
        }
        return true;
    };

    // ---------------------------------------------------------------- normals --
    {
        std::printf("normal maps\n");
        constexpr int kMapSize = 256;
        const std::vector<std::uint8_t> bumps = BumpNormals(kMapSize, 8);
        std::vector<std::uint8_t> flat(std::size_t(kMapSize) * kMapSize * 4);
        for (std::size_t i = 0; i < std::size_t(kMapSize) * kMapSize; ++i) {
            flat[i * 4 + 0] = 128;
            flat[i * 4 + 1] = 128;
            flat[i * 4 + 2] = 255;
            flat[i * 4 + 3] = 255;
        }
        // srgb = FALSE on both. A normal map stores a direction, not a colour,
        // and an sRGB decode would bend every component toward zero -- which
        // reads as bumps that are subtly too shallow near flat and far too
        // steep at the edges.
        const eng::rhi::TextureId bump_tex =
            dev->CreateTexture2D(kMapSize, kMapSize, bumps.data(), true, false);
        const eng::rhi::TextureId flat_tex =
            dev->CreateTexture2D(kMapSize, kMapSize, flat.data(), true, false);

        // A single quad facing the camera, filling most of the frame. Flat
        // geometry on purpose: everything the shading does here has to come
        // from the map, so any variation IS the map.
        const eng::MeshHandle quad =
            r->UploadMesh(eng::MakeBox(eng::Vec3{2.0f, 2.0f, 0.05f},
                                       eng::Vec4{1, 1, 1, 1}));

        const auto build = [&](eng::rhi::TextureId nmap, float light_x) {
            eng::Scene s;
            s.camera.eye = eng::Vec3{0.0f, 0.0f, 5.0f};
            s.camera.target = eng::Vec3{0.0f, 0.0f, 0.0f};
            // GRAZING light. A light straight down the view axis lands on the
            // top of every bump equally and the map barely shows; the whole
            // effect of a normal map is the terminator moving across each bump.
            s.lightDir = eng::Vec4{light_x, 0.25f, 0.45f, 0.0f};
            s.lightColor = eng::Vec4{3.0f, 3.0f, 3.0f, 1.0f};
            s.ambientSky = eng::Vec3{0.02f, 0.02f, 0.02f};
            s.ambientGround = eng::Vec3{0.02f, 0.02f, 0.02f};
            eng::MaterialDesc md;
            md.base_color = eng::Vec4{0.8f, 0.8f, 0.8f, 1.0f};
            md.roughness = 0.55f;
            md.metallic = 0.0f;
            md.normal_map = nmap;
            const eng::MaterialHandle m = r->CreateMaterial(md, error);
            eng::Instance in;
            in.mesh = quad;
            in.material = m;
            s.instances.push_back(in);
            return s;
        };

        if (!render(build({}, -0.85f))) return 1;
        const Region none = Measure(px, 120, 120, 280, 280);
        const std::vector<std::uint8_t> none_px = px;

        if (!render(build(flat_tex, -0.85f))) return 1;
        const Region flatr = Measure(px, 120, 120, 280, 280);
        double worst_flat = 0.0;
        for (int y = 120; y < 280; ++y)
            for (int x = 120; x < 280; ++x) {
                const std::size_t i = (std::size_t(y) * kW + x) * 4;
                worst_flat = std::max(worst_flat, std::fabs(Luma(px, i) - Luma(none_px, i)));
            }

        if (!render(build(bump_tex, -0.85f))) return 1;
        const Region bumpy = Measure(px, 120, 120, 280, 280);
        const std::vector<std::uint8_t> lit_left = px;

        if (!render(build(bump_tex, +0.85f))) return 1;
        const Region bumpy_r = Measure(px, 120, 120, 280, 280);

        std::printf("    no map:   mean %6.1f  variance %8.1f\n", none.mean,
                    none.variance);
        std::printf("    flat map: mean %6.1f  variance %8.1f  worst pixel diff %.1f\n",
                    flatr.mean, flatr.variance, worst_flat);
        std::printf("    bumpy:    mean %6.1f  variance %8.1f\n", bumpy.mean,
                    bumpy.variance);

        // A FLAT map is the identity. If the tangent frame were wrong -- wrong
        // handedness, unnormalised, axes swapped -- this is where it shows,
        // because (128,128,255) unpacks to (0,0,1) and the TBN must send that
        // straight back to the vertex normal.
        Check(worst_flat <= 2.0,
              "a flat normal map is indistinguishable from no map");
        // And a real one has to actually do something.
        Check(bumpy.variance > none.variance * 20.0,
              "a bumpy map multiplies the shading variance many times over");

        // THE ORIENTATION TEST. Swing the light from the left to the right and
        // the lit side of every bump must swap. Compare pixel by pixel: a
        // frame that is merely noisy in the right amount would pass the
        // variance check above and fail this one.
        int moved = 0, total = 0;
        for (int y = 130; y < 270; ++y)
            for (int x = 130; x < 270; ++x) {
                const std::size_t i = (std::size_t(y) * kW + x) * 4;
                if (std::fabs(Luma(lit_left, i) - Luma(px, i)) > 12.0) ++moved;
                ++total;
            }
        std::printf("    swinging the light moved %d of %d pixels by >12/255\n",
                    moved, total);
        Check(moved > total / 4,
              "and the lit side of each bump follows the light");
        Check(std::fabs(bumpy.mean - bumpy_r.mean) < bumpy.mean * 0.35,
              "with roughly the same total energy from either side");

        // THE EQUIVALENCE. A uniform normal map encoding a tilt of theta about
        // the tangent axis must shade EXACTLY like the same quad with its
        // geometry rotated by theta. No BRDF is modelled here and no tolerance
        // is guessed from a picture -- the renderer is being asked to agree
        // with itself under two descriptions of the same surface.
        //
        // This is the check that pins everything at once. It fails if z is not
        // rebuilt as sqrt(1 - x^2 - y^2), if the tangent points the wrong way,
        // if the strength scales the vector instead of the slope, or if the
        // TBN is transposed -- none of which the variance checks above can see,
        // because all of them still produce variation, just the wrong variation.
        constexpr float kTilt = 0.45f;  // radians, ~26 degrees
        std::vector<std::uint8_t> tilted(std::size_t(kMapSize) * kMapSize * 4);
        for (std::size_t i = 0; i < std::size_t(kMapSize) * kMapSize; ++i) {
            tilted[i * 4 + 0] =
                std::uint8_t(std::lround((std::sin(kTilt) * 0.5f + 0.5f) * 255.0f));
            tilted[i * 4 + 1] = 128;
            tilted[i * 4 + 2] =
                std::uint8_t(std::lround((std::cos(kTilt) * 0.5f + 0.5f) * 255.0f));
            tilted[i * 4 + 3] = 255;
        }
        const eng::rhi::TextureId tilt_tex =
            dev->CreateTexture2D(kMapSize, kMapSize, tilted.data(), true, false);

        // Pure diffuse. A specular lobe depends on the VIEW direction, which
        // the two cases do not share -- rotating the geometry moves the surface
        // and rotating only the normal does not.
        const auto tilt_scene = [&](eng::rhi::TextureId nmap, float geom_angle) {
            eng::Scene sc;
            sc.camera.eye = eng::Vec3{0.0f, 0.0f, 6.0f};
            sc.camera.target = eng::Vec3{0.0f, 0.0f, 0.0f};
            sc.lightDir = eng::Vec4{-0.6f, 0.3f, 0.74f, 0.0f};
            sc.lightColor = eng::Vec4{2.0f, 2.0f, 2.0f, 1.0f};
            sc.ambientSky = eng::Vec3{0.0f, 0.0f, 0.0f};
            sc.ambientGround = eng::Vec3{0.0f, 0.0f, 0.0f};
            eng::MaterialDesc md;
            md.base_color = eng::Vec4{0.7f, 0.7f, 0.7f, 1.0f};
            md.roughness = 1.0f;
            md.metallic = 0.0f;
            md.normal_map = nmap;
            eng::Instance in;
            in.mesh = quad;
            in.material = r->CreateMaterial(md, error);
            in.model = eng::Mat4::RotationY(geom_angle);
            sc.instances.push_back(in);
            return sc;
        };

        // MakeBox's +Z face runs u along +X, so a tangent-space tilt toward +x
        // is a world-space tilt toward +X -- which is exactly what RotationY
        // does to the +Z normal.
        if (!render(tilt_scene({}, kTilt))) return 1;
        const Region by_geometry = Measure(px, 185, 185, 215, 215);
        if (!render(tilt_scene(tilt_tex, 0.0f))) return 1;
        const Region by_map = Measure(px, 185, 185, 215, 215);
        // The control: the SAME map on a quad that is not tilted at all, so a
        // check that merely compared two similar numbers would not pass.
        if (!render(tilt_scene({}, 0.0f))) return 1;
        const Region untilted = Measure(px, 185, 185, 215, 215);

        std::printf("    tilt %.2f rad: geometry %.1f, normal map %.1f, "
                    "no tilt %.1f\n",
                    double(kTilt), by_geometry.mean, by_map.mean, untilted.mean);
        Check(std::fabs(by_map.mean - by_geometry.mean) < by_geometry.mean * 0.03,
              "a normal-map tilt shades the same as a geometry tilt");
        Check(std::fabs(untilted.mean - by_geometry.mean) > by_geometry.mean * 0.10,
              "and the tilt is large enough that agreeing means something");

        // STRENGTH scales the tangent-space SLOPE, before z is rebuilt. So 0 is
        // exactly flat and the values between interpolate the angle, not the
        // vector -- scaling the unpacked vector and renormalising would rotate
        // it toward the surface normal along a different curve and would never
        // reach flat at all.
        const auto at_strength = [&](float k) {
            eng::Scene sc = tilt_scene(tilt_tex, 0.0f);
            eng::MaterialDesc md;
            md.base_color = eng::Vec4{0.7f, 0.7f, 0.7f, 1.0f};
            md.roughness = 1.0f;
            md.normal_map = tilt_tex;
            md.normal_strength = k;
            sc.instances[0].material = r->CreateMaterial(md, error);
            return sc;
        };
        double at[4] = {};
        const float kStrengths[4] = {0.0f, 0.5f, 1.0f, 1.6f};
        for (int k = 0; k < 4; ++k) {
            if (!render(at_strength(kStrengths[k]))) return 1;
            at[k] = Measure(px, 185, 185, 215, 215).mean;
        }
        std::printf("    strength 0/0.5/1/1.6 -> %.1f %.1f %.1f %.1f "
                    "(no map at all: %.1f)\n",
                    at[0], at[1], at[2], at[3], untilted.mean);
        Check(std::fabs(at[0] - untilted.mean) < 1.5,
              "normal_strength 0 is exactly the same as no map");
        Check(at[1] < at[0] && at[2] < at[1] && at[3] < at[2],
              "and raising it deepens the tilt monotonically");

        // HANDEDNESS, in the shader this time, and it needs a map that tilts
        // along the BITANGENT -- a tilt along the tangent never reads w at all.
        //
        // The construction: mirroring u flips the tangent AND the stored sign,
        // and cross(N, T) * w is therefore unchanged. So a uniform v-tilt must
        // render IDENTICALLY on a mirrored shell. Drop the sign and the
        // bitangent flips instead, and the surface tilts the other way.
        //
        // A uniform map rather than the bump pattern on purpose: the bumps are
        // only symmetric under mirroring to within one texel, and that residue
        // is larger than the effect being measured. This map has no pattern to
        // be asymmetric.
        std::vector<std::uint8_t> vtilt(std::size_t(kMapSize) * kMapSize * 4);
        for (std::size_t i = 0; i < std::size_t(kMapSize) * kMapSize; ++i) {
            vtilt[i * 4 + 0] = 128;
            vtilt[i * 4 + 1] =
                std::uint8_t(std::lround((std::sin(kTilt) * 0.5f + 0.5f) * 255.0f));
            vtilt[i * 4 + 2] =
                std::uint8_t(std::lround((std::cos(kTilt) * 0.5f + 0.5f) * 255.0f));
            vtilt[i * 4 + 3] = 255;
        }
        const eng::rhi::TextureId vtilt_tex =
            dev->CreateTexture2D(kMapSize, kMapSize, vtilt.data(), true, false);

        eng::Mesh mirrored_quad = eng::MakeBox(eng::Vec3{2.0f, 2.0f, 0.05f},
                                               eng::Vec4{1, 1, 1, 1});
        for (VertexIn& v : mirrored_quad.vertices) v.uv.x = 1.0f - v.uv.x;
        eng::GenerateTangents(mirrored_quad);
        const eng::MeshHandle mq = r->UploadMesh(mirrored_quad);

        // The light is tilted out of the tangent plane's u axis so that a
        // v-tilt actually changes the shading -- with the light in the u-N
        // plane, tilting along v is nearly invisible.
        const auto vscene = [&](eng::MeshHandle mesh, eng::rhi::TextureId nmap) {
            eng::Scene sc = tilt_scene(nmap, 0.0f);
            sc.lightDir = eng::Vec4{0.15f, -0.62f, 0.77f, 0.0f};
            sc.instances[0].mesh = mesh;
            return sc;
        };

        if (!render(vscene(quad, vtilt_tex))) return 1;
        const std::vector<std::uint8_t> vt_normal = px;
        const Region vt_normal_r = Measure(px, 185, 185, 215, 215);
        if (!render(vscene(mq, vtilt_tex))) return 1;
        const Region vt_mirror_r = Measure(px, 185, 185, 215, 215);
        double rms = 0.0;
        int n = 0;
        for (int y = 150; y < 250; ++y)
            for (int x = 150; x < 250; ++x, ++n) {
                const std::size_t i = (std::size_t(y) * kW + x) * 4;
                const double d = Luma(vt_normal, i) - Luma(px, i);
                rms += d * d;
            }
        rms = std::sqrt(rms / n);
        // The control: the same quad with NO tilt. If a v-tilt barely moved the
        // shading, "mirrored matches original" would be true for the wrong
        // reason and the check would be worthless.
        if (!render(vscene(quad, {}))) return 1;
        const Region vt_flat_r = Measure(px, 185, 185, 215, 215);

        std::printf("    v-tilt: original %.1f, mirrored uv %.1f, untilted %.1f"
                    " (rms %.2f)\n",
                    vt_normal_r.mean, vt_mirror_r.mean, vt_flat_r.mean, rms);
        Check(std::fabs(vt_flat_r.mean - vt_normal_r.mean) > 8.0,
              "a bitangent-direction tilt visibly changes the shading");
        Check(rms < 2.0,
              "and a mirrored uv shell shades identically, as the sign demands");
    }

    // ------------------------------------------------------------------- mips --
    {
        std::printf("\nmip chains and anisotropy\n");
        constexpr int kTex = 512;
        const std::vector<std::uint8_t> checks = Checker(kTex, 64);
        const eng::rhi::TextureId with_mips =
            dev->CreateTexture2D(kTex, kTex, checks.data(), /*mips=*/true, false);
        const eng::rhi::TextureId no_mips =
            dev->CreateTexture2D(kTex, kTex, checks.data(), /*mips=*/false, false);

        // A long floor running away from a low camera. The far end is where a
        // texel is far smaller than a pixel, which is the whole point.
        eng::Mesh floor = eng::MakeBox(eng::Vec3{20.0f, 0.1f, 90.0f},
                                       eng::Vec4{1, 1, 1, 1});
        // Tile the texture many times over it: MakeBox gives each face 0..1,
        // and one checkerboard stretched over 180 metres never minifies.
        for (VertexIn& v : floor.vertices) {
            v.uv.x *= 40.0f;
            v.uv.y *= 180.0f;
        }
        const eng::MeshHandle fm = r->UploadMesh(floor);

        const auto build = [&](eng::rhi::TextureId tex) {
            eng::Scene s;
            s.camera.eye = eng::Vec3{0.0f, 1.4f, 6.0f};
            s.camera.target = eng::Vec3{0.0f, 1.15f, -30.0f};
            s.camera.fovY = 1.0f;
            s.lightDir = eng::Vec4{0.2f, 0.9f, 0.3f, 0.0f};
            s.lightColor = eng::Vec4{2.2f, 2.2f, 2.2f, 1.0f};
            eng::MaterialDesc md;
            md.base_color = eng::Vec4{1, 1, 1, 1};
            md.roughness = 0.8f;
            md.albedo = tex;
            const eng::MaterialHandle m = r->CreateMaterial(md, error);
            eng::Instance in;
            in.mesh = fm;
            in.material = m;
            in.model = eng::Mat4::Translation({0.0f, -0.1f, -60.0f});
            s.instances.push_back(in);
            return s;
        };

        // ROWS 250-390, chosen by measuring rather than by reasoning. The
        // horizon sits at row 200 and the intuition -- "the far band aliases
        // worst" -- is wrong here: right at the horizon a checker square is far
        // below a pixel and the samples average toward mid grey, while the
        // middle distance is where a square is two or three pixels across.
        // That is the Nyquist band, and it is where the shimmer lives.
        constexpr int kY0 = 250, kY1 = 390;
        if (!render(build(no_mips))) return 1;
        const double alias_off = HighFrequency(px, 40, kY0, 360, kY1);
        const Region off = Measure(px, 40, kY0, 360, kY1);

        if (!render(build(with_mips))) return 1;
        const double alias_on = HighFrequency(px, 40, kY0, 360, kY1);
        const Region on = Measure(px, 40, kY0, 360, kY1);

        std::printf("    no mips:   adjacent-pixel energy %9.1f  mean %.1f\n",
                    alias_off, off.mean);
        std::printf("    with mips: adjacent-pixel energy %9.1f  mean %.1f\n",
                    alias_on, on.mean);
        std::printf("    aliasing reduced to %.1f%%\n",
                    alias_off > 0 ? 100.0 * alias_on / alias_off : 0.0);
        Check(alias_off > 100.0, "the unfiltered floor really is aliasing");
        Check(alias_on < alias_off * 0.25,
              "a mip chain removes most of the high-frequency energy");
        // NOT just darker. A mip chain that averaged wrongly -- in gamma space,
        // or with the wrong box -- would also lower the number above, by
        // turning the whole floor grey. The mean has to survive.
        Check(std::fabs(on.mean - off.mean) < off.mean * 0.30,
              "and does it by averaging, not by dimming the floor");

        // ANISOTROPY, which is the OTHER half. A mip chain picks one level for
        // both axes, so a floor at a grazing angle -- squeezed hard along the
        // view direction and barely at all across it -- has to choose a level
        // that kills the aliasing along the squeezed axis, and that level is
        // far too blurry for the other one. Anisotropic filtering takes several
        // samples along the squeezed axis instead and keeps the rest sharp.
        //
        // So the measure is CONTRAST RETAINED at distance, not aliasing
        // removed: both settings are alias-free, and the anisotropic one still
        // has a visible checkerboard where the isotropic one has grey.
        // ROWS 260-390, chosen by scanning. Nearer the horizon than that, the
        // squeeze is far past 16:1 and both settings clamp to the same blurry
        // level; nearer the camera than that, there is barely any squeeze to
        // exploit. In between is where anisotropy is the whole difference
        // between a visible checkerboard and flat grey.
        constexpr int kFarY0 = 260, kFarY1 = 390;
        r->SetAnisotropy(1);
        if (!render(build(with_mips))) return 1;
        const Region iso = Measure(px, 40, kFarY0, 360, kFarY1);
        r->SetAnisotropy(16);
        if (!render(build(with_mips))) return 1;
        const Region aniso = Measure(px, 40, kFarY0, 360, kFarY1);
        std::printf("    far band contrast: isotropic %.1f, 16x anisotropic %.1f\n",
                    std::sqrt(iso.variance), std::sqrt(aniso.variance));
        Check(r->Anisotropy() == 16, "the anisotropy setting takes effect");
        Check(std::sqrt(aniso.variance) > std::sqrt(iso.variance) * 1.25,
              "anisotropic filtering keeps far detail an isotropic mip loses");
        // And it must not bring the aliasing back: sharper along the
        // unsqueezed axis is the point, sharper along the squeezed one is the
        // bug it would be if the sample count were applied to the wrong axis.
        const double alias_aniso = HighFrequency(px, 40, kY0, 360, kY1);
        std::printf("    and the mid-band aliasing stays at %.1f\n", alias_aniso);
        Check(alias_aniso < alias_off * 0.25,
              "without putting the mid-band aliasing back");
    }

    // ------------------------------------------------------------ big meshes --
    {
        std::printf("\n32-bit indices\n");
        // 301 x 301 = 90601 vertices, comfortably past the old 65535 ceiling.
        const eng::Mesh big = eng::MakeUVSphere(1.0f, 300, 300, eng::Vec4{1, 1, 1, 1},
                                                eng::Vec4{1, 1, 1, 1});
        const eng::Mesh small = eng::MakeUVSphere(1.0f, 24, 32, eng::Vec4{1, 1, 1, 1},
                                                  eng::Vec4{1, 1, 1, 1});
        std::printf("    %zu vertices, %zu indices, max index %u\n",
                    big.vertices.size(), big.indices.size(),
                    *std::max_element(big.indices.begin(), big.indices.end()));
        Check(big.vertices.size() > 65535,
              "the test mesh really is past the 16-bit ceiling");

        const eng::MeshHandle bh = r->UploadMesh(big);
        const eng::MeshHandle sh = r->UploadMesh(small);

        const auto build = [&](eng::MeshHandle mesh) {
            eng::Scene s;
            s.camera.eye = eng::Vec3{0.0f, 0.0f, 3.0f};
            s.camera.target = eng::Vec3{0.0f, 0.0f, 0.0f};
            s.lightDir = eng::Vec4{0.3f, 0.5f, 0.8f, 0.0f};
            s.lightColor = eng::Vec4{3.0f, 3.0f, 3.0f, 1.0f};
            eng::MaterialDesc md;
            md.base_color = eng::Vec4{0.9f, 0.9f, 0.9f, 1.0f};
            md.roughness = 0.5f;
            const eng::MaterialHandle m = r->CreateMaterial(md, error);
            eng::Instance in;
            in.mesh = mesh;
            in.material = m;
            s.instances.push_back(in);
            return s;
        };

        const auto coverage = [&]() {
            int n = 0;
            for (int i = 0; i < kW * kH; ++i)
                if (Luma(px, std::size_t(i) * 4) > 30.0) ++n;
            return n;
        };

        if (!render(build(sh))) return 1;
        const int small_cov = coverage();
        if (!render(build(bh))) return 1;
        const int big_cov = coverage();

        std::printf("    24x32 sphere covers %d px, 300x300 sphere covers %d px\n",
                    small_cov, big_cov);
        Check(big_cov > 1000, "a 90601-vertex mesh draws at all");
        // THE REAL CHECK. Two spheres of the same radius must cover the same
        // area. Sixteen-bit indices would not have failed to draw -- they would
        // have wrapped, connecting vertex 65536 back to vertex 0 and stretching
        // triangles across the whole model. That covers a DIFFERENT area, and
        // it is the only symptom the old code had.
        Check(std::fabs(double(big_cov - small_cov)) < small_cov * 0.03,
              "and covers the same silhouette as the coarse one");
    }

    // --------------------------------------------------------------- emissive --
    {
        std::printf("\nemissive\n");
        const eng::MeshHandle ball =
            r->UploadMesh(eng::MakeUVSphere(1.0f, 32, 48, eng::Vec4{1, 1, 1, 1},
                                            eng::Vec4{1, 1, 1, 1}));

        const auto build = [&](eng::Vec3 emit) {
            eng::Scene s;
            s.camera.eye = eng::Vec3{0.0f, 0.0f, 3.0f};
            s.camera.target = eng::Vec3{0.0f, 0.0f, 0.0f};
            // TOTAL DARKNESS. No key light, no ambient, no probe. Anything that
            // arrives at the film came out of the surface itself.
            s.lightColor = eng::Vec4{0.0f, 0.0f, 0.0f, 1.0f};
            s.ambientSky = eng::Vec3{0.0f, 0.0f, 0.0f};
            s.ambientGround = eng::Vec3{0.0f, 0.0f, 0.0f};
            eng::MaterialDesc md;
            md.base_color = eng::Vec4{0.5f, 0.5f, 0.5f, 1.0f};
            md.emissive = emit;
            const eng::MaterialHandle m = r->CreateMaterial(md, error);
            eng::Instance in;
            in.mesh = ball;
            in.material = m;
            s.instances.push_back(in);
            return s;
        };

        if (!render(build(eng::Vec3{0.0f, 0.0f, 0.0f}))) return 1;
        const Region dark = Measure(px, 180, 180, 220, 220);
        // A SWEEP, because the interesting property is the shape of the curve
        // and no single value shows it. The tone map saturates hard past about
        // 4, so a check written against one bright value would pass whether
        // emission were linear radiance or a 0..1 colour.
        double mean_at[6] = {};
        int rgb_at[6][3] = {};
        const float kSweep[6] = {0.3f, 0.9f, 2.0f, 4.0f, 8.0f, 22.0f};
        for (int k = 0; k < 6; ++k) {
            const float e = kSweep[k];
            if (!render(build(eng::Vec3{e, e * 0.32f, e * 0.09f}))) return 1;
            const Region q = Measure(px, 190, 190, 210, 210);
            const std::size_t c = (std::size_t(200) * kW + 200) * 4;
            mean_at[k] = q.mean;
            rgb_at[k][0] = px[c];
            rgb_at[k][1] = px[c + 1];
            rgb_at[k][2] = px[c + 2];
            std::printf("    emissive %5.1f -> mean %6.2f  rgb %3d %3d %3d\n", e,
                        q.mean, px[c], px[c + 1], px[c + 2]);
        }

        std::printf("    unlit and non-emissive: mean %.2f\n", dark.mean);
        Check(dark.mean < 1.0, "with no light and no emission the ball is black");
        Check(mean_at[2] > 150.0,
              "with emission and still no light it is bright");
        // COLOUR, read at the dimmest sample. At 22 the tone map has pushed
        // every channel to white and a swizzled or greyscale emissive would
        // pass; at 0.3 the ratio 1 : 0.32 : 0.09 is still visible in the bytes.
        Check(rgb_at[0][0] > rgb_at[0][1] && rgb_at[0][1] > rgb_at[0][2],
              "and it keeps the colour it was given, not white");
        // RADIANCE, not a colour. A 0..1 colour would clamp and the curve would
        // be flat from 1 upward; linear radiance keeps climbing through the
        // tone map long after the byte value stops being interesting.
        std::printf("    0.3 -> %.1f, 8.0 -> %.1f (a 0..1 colour would clamp)\n",
                    mean_at[0], mean_at[4]);
        Check(mean_at[4] > mean_at[0] * 2.0,
              "and it is radiance: 8.0 is far brighter than 0.3");
        bool monotonic = true;
        for (int k = 1; k < 6; ++k)
            if (mean_at[k] <= mean_at[k - 1]) monotonic = false;
        Check(monotonic, "brighter emission is monotonically brighter output");

        // sRGB DECODING, measured through the emissive map because that is the
        // one path where a texture's value reaches the film without a BRDF in
        // between. The same bytes, read two ways: 128 is 0.502 as a linear
        // value and 0.216 once decoded from sRGB. Reading an authored albedo
        // as linear is why untextured renders and textured ones never quite
        // match in brightness.
        std::vector<std::uint8_t> grey(64 * 64 * 4, 128);
        for (std::size_t i = 3; i < grey.size(); i += 4) grey[i] = 255;
        const eng::rhi::TextureId lin_tex =
            dev->CreateTexture2D(64, 64, grey.data(), true, /*srgb=*/false);
        const eng::rhi::TextureId srgb_tex =
            dev->CreateTexture2D(64, 64, grey.data(), true, /*srgb=*/true);

        const auto build_map = [&](eng::rhi::TextureId t) {
            eng::Scene sc = build(eng::Vec3{1.0f, 1.0f, 1.0f});
            eng::MaterialDesc md;
            md.base_color = eng::Vec4{0.5f, 0.5f, 0.5f, 1.0f};
            md.emissive = eng::Vec3{1.0f, 1.0f, 1.0f};
            md.emissive_map = t;
            sc.instances[0].material = r->CreateMaterial(md, error);
            return sc;
        };
        if (!render(build_map(lin_tex))) return 1;
        const Region as_linear = Measure(px, 190, 190, 210, 210);
        if (!render(build_map(srgb_tex))) return 1;
        const Region as_srgb = Measure(px, 190, 190, 210, 210);
        std::printf("    grey 128 read linear: %.1f   read as sRGB: %.1f\n",
                    as_linear.mean, as_srgb.mean);
        Check(as_srgb.mean < as_linear.mean * 0.80,
              "an sRGB texture decodes darker than the same bytes read linear");
    }

    std::printf(g_failures == 0 ? "\nsurface_test: all checks passed\n"
                                : "\nsurface_test: %d FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
