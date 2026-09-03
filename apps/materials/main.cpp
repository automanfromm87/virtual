// Pure C++20. A 5x5 material grid: roughness left-to-right, metallic
// bottom-to-top, plus two textured spheres.
//
// The checks are PHYSICS, not screenshots. A Cook-Torrance BRDF makes two
// predictions that a wrong implementation cannot fake:
//   1. Peak brightness falls as roughness rises. The same energy gets spread
//      over a wider lobe, so the highlight dims as it grows.
//   2. A metal is darker overall than a dielectric of the same roughness,
//      because it has no diffuse lobe at all — only the highlight survives.
// Assert those and a Blinn-Phong stand-in, or a Fresnel term with the wrong
// sign, fails.
#include "engine/render/renderer.h"
#include "engine/rhi/rhi.h"
#include "engine/geometry/mesh.h"
#include "engine/scene/scene.h"
#include "engine/texture/texture.h"

#include <cstdio>
#include <set>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-54s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

double Luma(const std::vector<std::uint8_t>& p, std::size_t i) {
    return 0.2126 * p[i] + 0.7152 * p[i + 1] + 0.0722 * p[i + 2];
}

constexpr int kW = 640, kH = 640;
constexpr int kGrid = 5;
constexpr float kSpacing = 1.2f;
constexpr float kRadius = 0.5f;
constexpr float kEyeZ = 9.0f;

// Where a grid cell's centre lands in pixels. f = 1/tan(fovY/2) with the
// engine's 60-degree default; the sphere sits on the z = 0 plane, so its view
// depth is exactly kEyeZ.
void CellPixel(int i, int j, int* px, int* py) {
    constexpr float kF = 1.7320508f;
    const float x = (float(i) - 2.0f) * kSpacing;
    const float y = (float(j) - 2.0f) * kSpacing;
    const float ndc_x = kF * x / kEyeZ;
    const float ndc_y = kF * y / kEyeZ;
    *px = int(float(kW) * 0.5f * (1.0f + ndc_x));
    *py = int(float(kH) * 0.5f * (1.0f - ndc_y));  // row 0 is the TOP
}

}  // namespace

int main() {
    std::string error;
    auto dev = eng::rhi::Device::Create(error);
    if (!dev) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
    auto renderer = eng::Renderer::Create(*dev, eng::rhi::Format::RGBA8Unorm, error);
    if (!renderer) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    // --- textures ------------------------------------------------------------
    const eng::Texture2D checker =
        eng::MakeChecker(256, 8, eng::Vec4{0.9f, 0.9f, 0.9f, 1}, eng::Vec4{0.15f, 0.35f, 0.8f, 1});
    const eng::Texture2D ramp = eng::MakeRoughnessRamp(256, 8);
    const eng::rhi::TextureId albedo_tex =
        dev->CreateTexture2D(checker.width, checker.height, checker.rgba.data());
    const eng::rhi::TextureId rough_tex =
        dev->CreateTexture2D(ramp.width, ramp.height, ramp.rgba.data());
    if (!Valid(albedo_tex) || !Valid(rough_tex)) {
        std::fprintf(stderr, "FAIL: texture upload\n");
        return 1;
    }

    // --- the grid ------------------------------------------------------------
    // Building the scene is the APP's job: engine/scene cannot create materials
    // (that is the renderer's table) and engine/render does not know what a
    // demo is.
    eng::Scene scene;
    scene.camera.eye = eng::Vec3{0.0f, 0.0f, kEyeZ};

    // A PLAIN sphere, uploaded by the app rather than using the built-in. The
    // built-in carries a luminance checker in its vertex colours, which
    // multiplies into albedo and would scribble over the very thing this grid
    // is meant to show. Uploading a mesh from an app is the resource system
    // working as designed.
    const eng::Mesh plain = eng::MakeUVSphere(1.0f, 48, 96, eng::Vec4{1, 1, 1, 1},
                                              eng::Vec4{1, 1, 1, 1});
    const eng::MeshHandle plain_sphere = renderer->UploadMesh(plain);
    if (!Valid(plain_sphere)) { std::fprintf(stderr, "FAIL: mesh upload\n"); return 1; }
    eng::MaterialHandle grid_mat[kGrid][kGrid];
    for (int j = 0; j < kGrid; ++j) {
        for (int i = 0; i < kGrid; ++i) {
            eng::MaterialDesc d;
            d.base_color = eng::Vec4{0.95f, 0.72f, 0.35f, 1.0f};  // gold-ish
            // Clamped away from 0: a perfectly smooth surface has a highlight
            // narrower than one pixel, which aliases into noise rather than
            // showing the trend.
            d.roughness = 0.06f + 0.94f * float(i) / float(kGrid - 1);
            d.metallic = float(j) / float(kGrid - 1);
            grid_mat[j][i] = renderer->CreateMaterial(d, error);
            if (!Valid(grid_mat[j][i])) {
                std::fprintf(stderr, "FAIL: %s\n", error.c_str());
                return 1;
            }
            eng::Instance inst;
            inst.mesh = plain_sphere;
            inst.material = grid_mat[j][i];
            inst.model = eng::Mat4::Translation({(float(i) - 2.0f) * kSpacing,
                                                 (float(j) - 2.0f) * kSpacing, 0.0f}) *
                         eng::Mat4::Scale(kRadius);
            scene.instances.push_back(inst);
        }
    }

    // Two textured spheres off to the side of the grid.
    eng::MaterialDesc textured;
    textured.albedo = albedo_tex;
    textured.roughness = 0.35f;
    const eng::MaterialHandle m_tex = renderer->CreateMaterial(textured, error);
    eng::MaterialDesc ramped;
    ramped.base_color = eng::Vec4{0.9f, 0.9f, 0.95f, 1.0f};
    ramped.metallic = 1.0f;
    ramped.roughness_map = rough_tex;  // roughness varies across the surface
    const eng::MaterialHandle m_ramp = renderer->CreateMaterial(ramped, error);
    if (!Valid(m_tex) || !Valid(m_ramp)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }
    for (int k = 0; k < 2; ++k) {
        eng::Instance inst;
        inst.mesh = plain_sphere;
        inst.material = k ? m_ramp : m_tex;
        inst.model = eng::Mat4::Translation({(k ? 1.2f : -1.2f), -3.6f, 0.0f}) *
                     eng::Mat4::RotationY(0.6f) * eng::Mat4::Scale(0.75f);
        scene.instances.push_back(inst);
    }

    // --- render --------------------------------------------------------------
    eng::rhi::PassDesc pass;
    pass.color = dev->CreateRenderTarget(kW, kH, eng::rhi::Format::RGBA8Unorm, true);
    pass.depth = dev->CreateDepthTarget(kW, kH);
    for (int i = 0; i < 4; ++i) pass.clear_color[i] = eng::kClearColor[i];
    pass.clear_depth = 0.0f;

    dev->BeginFrame();
    eng::rhi::Encoder enc = dev->BeginPass(pass);
    renderer->DrawScene(enc, scene, kW, kH);
    dev->EndPass();
    const eng::RenderStats stats = renderer->LastStats();
    if (!dev->CommitAndWait(error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    std::vector<std::uint8_t> px(std::size_t(kW) * kH * 4);
    if (!dev->ReadPixels(pass.color, kW, kH, px)) {
        std::fprintf(stderr, "FAIL: readback\n");
        return 1;
    }
    if (std::FILE* fp = std::fopen("materials.ppm", "wb")) {
        std::fprintf(fp, "P6\n%d %d\n255\n", kW, kH);
        for (std::size_t i = 0; i + 3 < px.size(); i += 4)
            std::fwrite(&px[i], 1, 3, fp);
        std::fclose(fp);
    }

    // --- measure -------------------------------------------------------------
    // Peak and mean luminance over a box centred on each grid cell.
    double peak[kGrid][kGrid] = {};
    double mean[kGrid][kGrid] = {};
    int area[kGrid][kGrid] = {};
    for (int j = 0; j < kGrid; ++j) {
        for (int i = 0; i < kGrid; ++i) {
            int cx = 0, cy = 0;
            CellPixel(i, j, &cx, &cy);
            double hi = 0, sum = 0;
            int n = 0;
            for (int y = cy - 20; y <= cy + 20; ++y) {
                for (int x = cx - 20; x <= cx + 20; ++x) {
                    if (x < 0 || y < 0 || x >= kW || y >= kH) continue;
                    const double l = Luma(px, (std::size_t(y) * kW + x) * 4);
                    if (l > hi) hi = l;
                    sum += l;
                    ++n;
                }
            }
            peak[j][i] = hi;
            mean[j][i] = n ? sum / n : 0.0;
            area[j][i] = 0;
            for (int y = cy - 20; y <= cy + 20; ++y)
                for (int x = cx - 20; x <= cx + 20; ++x) {
                    if (x < 0 || y < 0 || x >= kW || y >= kH) continue;
                    if (Luma(px, (std::size_t(y) * kW + x) * 4) > 150.0) ++area[j][i];
                }
        }
    }

    std::printf("%dx%d  %d instances, %d draws\n", kW, kH, stats.submitted, stats.draws);
    std::printf("  peak luminance by roughness (metallic rows):\n");
    for (int j = kGrid - 1; j >= 0; --j) {
        std::printf("    metallic %.2f :", float(j) / float(kGrid - 1));
        for (int i = 0; i < kGrid; ++i) std::printf(" %6.1f", peak[j][i]);
        std::printf("\n");
    }

    Check(stats.draws == kGrid * kGrid + 2, "every material instance drew");

    std::printf("  highlight area (px > 150 luma), metal row:");
    for (int i = 0; i < kGrid; ++i) std::printf(" %5d", area[kGrid - 1][i]);
    std::printf("\n");

    // PREDICTION 1: rougher spreads the same energy over a wider lobe, so the
    // peak falls. Checked on the fully metallic row, where the highlight is the
    // whole story and no diffuse term muddies it.
    //
    // Only over the UNSATURATED range. A tone mapper's whole job is to fold an
    // unbounded radiance into a display, so a peak of 5 and a peak of 20 land
    // within a few 8-bit codes of each other and carry no ordering. That is a
    // property of the curve, not of the BRDF, and asserting through it would be
    // asserting noise.
    //
    // The filmic curve this engine uses has a harder shoulder than the Reinhard
    // one it replaced, so MORE of the sweep saturates — which is why the check
    // below starts at `first` rather than at column zero. It used to start at
    // zero, contradicting this very comment, and survived only because the old
    // curve happened to compress gently enough.
    constexpr double kSaturated = 240.0;
    int first = 0;
    while (first < kGrid && peak[kGrid - 1][first] >= kSaturated) ++first;
    bool falls = true;
    for (int i = first + 1; i < kGrid; ++i)
        if (peak[kGrid - 1][i] > peak[kGrid - 1][i - 1] + 1.0) falls = false;
    std::printf("  unsaturated from column %d of %d\n", first, kGrid);
    Check(first < kGrid - 1, "the roughness sweep leaves the saturated region");
    Check(falls, "peak highlight dims as roughness rises (unsaturated range)");
    // NOT asserted: that the smoothest metal's peak beats the roughest one's.
    //
    // It cannot be measured from an 8-bit image at any exposure. A near-mirror
    // GGX lobe is close to a delta function -- its peak radiance goes as
    // 1/(pi*alpha^2), which at roughness 0.1 is over three thousand times the
    // incoming light -- so it reads 255 until the exposure is turned down far
    // enough that every rougher sample in the sweep is black. One image cannot
    // hold both ends. Measuring it would need an HDR readback, and the claim it
    // would establish is the one the AREA check below already makes.
    //
    // This assertion used to exist and passed only because the previous tone
    // curve compressed gently enough from zero to leave a few codes of ordering
    // in the blown-out region. That was reading the curve, not the BRDF.

    // PREDICTION 1b, the half that does NOT saturate: the same energy over a
    // wider lobe covers more pixels. Peak brightness and highlight area move in
    // opposite directions — that pairing is what energy conservation looks like
    // on screen, and a normalisation error breaks one or the other.
    // Measured at 14 pixels for the smoothest and 529 for the middle of the
    // sweep -- a factor of thirty-eight. This is the lobe width, and it is the
    // half of energy conservation that an LDR image CAN see.
    Check(area[kGrid - 1][kGrid / 2] > area[kGrid - 1][0] * 8,
          "highlight area grows sharply as the peak saturates (energy conserved)");

    // PREDICTION 2: metals have no diffuse lobe, so away from the highlight
    // they are darker than a dielectric with identical roughness.
    int darker = 0;
    for (int i = 0; i < kGrid; ++i)
        if (mean[kGrid - 1][i] < mean[0][i]) ++darker;
    std::printf("  metal darker than dielectric in %d of %d roughness columns\n",
                darker, kGrid);
    Check(darker == kGrid, "metals are darker than dielectrics (no diffuse lobe)");

    // The albedo map actually reaches the shader: the checkered sphere must
    // carry both of its colours.
    {
        int cx = kW / 2 - 100, cy = int(kH * 0.5f * (1.0f + 1.7320508f * 3.6f / kEyeZ));
        std::set<std::uint32_t> hues;
        for (int y = cy - 30; y <= cy + 30; ++y)
            for (int x = cx - 30; x <= cx + 30; ++x) {
                if (x < 0 || y < 0 || x >= kW || y >= kH) continue;
                const std::size_t i = (std::size_t(y) * kW + x) * 4;
                if (px[i] + px[i + 1] + px[i + 2] < 40) continue;
                // Quantise hard so shading gradients do not count as variety.
                hues.insert(((px[i] >> 5) << 10) | ((px[i + 1] >> 5) << 5) | (px[i + 2] >> 5));
            }
        std::printf("  textured sphere quantised hues: %zu\n", hues.size());
        Check(hues.size() >= 4, "albedo map reaches the shader (checker visible)");
    }

    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
