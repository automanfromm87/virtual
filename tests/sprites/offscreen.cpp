// Sprites, seen: three quads from one SpriteBatch draw, offscreen.
//
// A batch that puts the wrong corner first still shows a quad roughly where
// it belongs, so the CPU test cannot catch everything. This renders the real
// path -- batch, upload, ortho camera, textured Lit material -- and measures
// the pixels: red left, blue right (rotated), green on top in the middle.
#include "engine/geometry/mesh.h"
#include "engine/render/renderer.h"
#include "engine/rhi/rhi.h"
#include "engine/scene/scene.h"
#include "engine/sprite/sprite.h"
#include "engine/texture/texture.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-60s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

constexpr int kW = 640, kH = 480;

struct Accum {
    double r = 0, g = 0, b = 0, n = 0;
    void Add(std::uint8_t R, std::uint8_t G, std::uint8_t B) {
        r += R;
        g += G;
        b += B;
        n += 1;
    }
    // Means, so thresholds read as colours rather than pixel counts.
    double R() const { return n > 0 ? r / n : 0; }
    double G() const { return n > 0 ? g / n : 0; }
    double B() const { return n > 0 ? b / n : 0; }
};

// Mean colour over a world-space box. Ortho height 3 over 480 rows maps
// y in [-3, 3] to rows [479, 0]; x in [-4, 4] to cols [0, 639].
Accum Region(const std::vector<std::uint8_t>& px, float x0, float x1, float y0,
             float y1) {
    Accum a;
    const int c0 = int((x0 + 4.0f) / 8.0f * kW), c1 = int((x1 + 4.0f) / 8.0f * kW);
    const int r0 = int((3.0f - y1) / 6.0f * kH), r1 = int((3.0f - y0) / 6.0f * kH);
    for (int r = std::max(0, r0); r < std::min(kH, r1); ++r)
        for (int c = std::max(0, c0); c < std::min(kW, c1); ++c) {
            const std::size_t i = (std::size_t(r) * kW + c) * 4;
            a.Add(px[i], px[i + 1], px[i + 2]);
        }
    return a;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("sprites offscreen\n");

    std::string error;
    auto dev = eng::rhi::Device::Create(error);
    if (!dev) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }
    auto renderer =
        eng::Renderer::Create(*dev, eng::rhi::Format::RGBA8Unorm, error);
    if (!renderer) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }

    // Three sprites, one batch, one draw: red left, blue right rotated half
    // a radian, green overlapping both on a higher layer.
    eng::sprite::SpriteBatch batch;
    eng::sprite::Sprite a;
    a.center = eng::Vec2{-2.0f, 0.0f};
    a.size = eng::Vec2{3.0f, 3.0f};
    a.tint = eng::Vec4{1.0f, 0.3f, 0.3f, 1.0f};
    batch.Push(a);
    eng::sprite::Sprite b;
    b.center = eng::Vec2{2.0f, 0.5f};
    b.size = eng::Vec2{2.5f, 2.5f};
    b.rotation = 0.5f;
    b.uv = eng::Vec4{0.0f, 0.0f, 0.5f, 1.0f};
    b.tint = eng::Vec4{0.4f, 0.6f, 1.0f, 1.0f};
    batch.Push(b);
    eng::sprite::Sprite c;
    c.center = eng::Vec2{0.0f, -0.5f};
    c.size = eng::Vec2{2.0f, 2.0f};
    c.uv = eng::Vec4{0.5f, 0.0f, 1.0f, 0.5f};
    c.tint = eng::Vec4{0.4f, 1.0f, 0.5f, 1.0f};
    c.layer = 1;
    batch.Push(c);
    batch.Bake();
    Check(batch.vertices().size() == 12, "three sprites baked");

    const eng::Mesh mesh = batch.BuildMesh();
    const eng::MeshHandle mesh_handle = renderer->UploadMesh(mesh);
    Check(eng::Valid(mesh_handle), "the batch uploaded");

    const eng::Texture2D checker = eng::MakeChecker(
        256, 8, eng::Vec4{0.9f, 0.9f, 0.9f, 1}, eng::Vec4{0.15f, 0.35f, 0.8f, 1});
    const eng::rhi::TextureId checker_tex = dev->CreateTexture2D(
        checker.width, checker.height, checker.rgba.data());
    Check(eng::rhi::Valid(checker_tex), "the checker uploaded");

    eng::MaterialDesc md;
    md.shading = eng::Shading::Lit;
    md.base_color = eng::Vec4{1.0f, 1.0f, 1.0f, 1.0f};
    md.roughness = 1.0f;
    md.albedo = checker_tex;
    const eng::MaterialHandle mat = renderer->CreateMaterial(md, error);
    Check(eng::Valid(mat), "the sprite material built");

    eng::Scene scene;
    scene.camera.projection = eng::Projection::Orthographic;
    scene.camera.orthoHeight = 3.0f;
    scene.camera.eye = eng::Vec3{0.0f, 0.0f, 10.0f};
    scene.camera.target = eng::Vec3{0.0f, 0.0f, 0.0f};
    // Straight-on key light: shading differences across the frame come from
    // the sprites, not from an angle.
    scene.lightDir = eng::Vec4{0.0f, 0.0f, 1.0f, 0.0f};
    scene.lightColor = eng::Vec4{1.0f, 1.0f, 1.0f, 1.0f};
    eng::Instance inst;
    inst.mesh = mesh_handle;
    inst.material = mat;
    scene.instances.push_back(inst);

    eng::rhi::PassDesc pass;
    pass.color = dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
    pass.depth = dev->CreateDepthTarget(kW, kH);
    const eng::rhi::TextureId readable =
        dev->CreateRenderTarget(kW, kH, eng::rhi::Format::RGBA8Unorm, true);
    for (int i = 0; i < 4; ++i) pass.clear_color[i] = eng::kClearColor[i];
    pass.clear_depth = 0.0f;

    dev->BeginFrame();
    eng::rhi::Encoder enc = dev->BeginPass(pass);
    renderer->DrawScene(enc, scene, kW, kH);
    dev->EndPass();
    {
        eng::rhi::PassDesc resolve;
        resolve.color = readable;
        eng::rhi::Encoder re = dev->BeginPass(resolve);
        renderer->DrawComposite(re, pass.color, {}, {}, 0.0f, /*vignette=*/0.0f);
        dev->EndPass();
    }
    if (!dev->CommitAndWait(error)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
    }
    std::vector<std::uint8_t> px(std::size_t(kW) * kH * 4);
    Check(dev->ReadPixels(readable, kW, kH, px), "the frame reads back");
    if (std::FILE* fp = std::fopen("sprites.ppm", "wb")) {
        std::fprintf(fp, "P6\n%d %d\n255\n", kW, kH);
        for (std::size_t i = 0; i + 3 < px.size(); i += 4)
            std::fwrite(&px[i], 1, 3, fp);
        std::fclose(fp);
    }

    // Left of centre is the red sprite: red beats each other channel. (Not
    // their sum: half the checker squares are blue-dark by design.)
    const Accum left = Region(px, -3.5f, -1.0f, -1.0f, 1.0f);
    Check(left.n > 0 && left.R() > left.G() && left.R() > left.B(),
          "red covers the left");
    // Right holds the rotated blue sprite.
    const Accum right = Region(px, 1.0f, 3.5f, -0.5f, 1.5f);
    Check(right.n > 0 && right.B() > right.R() && right.B() > right.G(),
          "blue covers the right");
    // The middle bottom is the green top layer over both.
    const Accum mid = Region(px, -0.5f, 0.5f, -1.2f, -0.2f);
    Check(mid.n > 0 && mid.G() > mid.R() && mid.G() > mid.B(), "green sits on top");
    // The corners are background, not sprite spill.
    const Accum corner = Region(px, -4.0f, -3.4f, 2.4f, 3.0f);
    Check(corner.n > 0 && corner.R() < 60 && corner.G() < 60 && corner.B() < 60,
          "the corner stays background");

    if (g_failures == 0) std::printf("sprites: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
