// The material line-up, shared by the interactive viewer and the offscreen
// check so both look at exactly the same five spheres.
#pragma once

#include <string>
#include <vector>

#include "engine/geometry/mesh.h"
#include "engine/render/renderer.h"
#include "engine/rhi/rhi.h"
#include "engine/scene/scene.h"
#include "engine/texture/texture.h"

namespace demo {

constexpr int kCount = 5;
constexpr float kSpacing = 2.4f;
constexpr float kRadius = 0.9f;

struct Assets {
    eng::MeshHandle sphere, ground;
    std::vector<eng::MaterialHandle> materials;
    std::vector<std::string> names;
    eng::MaterialHandle floor_mat;
    bool ok = false;
};

inline Assets Build(eng::Renderer& r, eng::rhi::Device& dev, std::string& error) {
    Assets a;

    // High tessellation on purpose: a coarse sphere puts facet edges right
    // where the highlight is, and then every material looks faceted instead of
    // smooth. Roughness is the thing being judged here, so the silhouette and
    // the shading normals have to be out of the way.
    a.sphere = r.UploadMesh(eng::MakeUVSphere(kRadius, 64, 128, eng::Vec4{1, 1, 1, 1},
                                              eng::Vec4{1, 1, 1, 1}));
    a.ground = r.UploadMesh(
        eng::MakeBox(eng::Vec3{14.0f, 0.4f, 9.0f}, eng::Vec4{1, 1, 1, 1}));

    const eng::Texture2D checker = eng::MakeChecker(
        512, 10, eng::Vec4{0.92f, 0.90f, 0.86f, 1}, eng::Vec4{0.12f, 0.30f, 0.55f, 1});
    const eng::Texture2D ramp = eng::MakeRoughnessRamp(512, 10);
    const eng::rhi::TextureId tex_checker =
        dev.CreateTexture2D(checker.width, checker.height, checker.rgba.data());
    const eng::rhi::TextureId tex_ramp =
        dev.CreateTexture2D(ramp.width, ramp.height, ramp.rgba.data());

    // Chosen to separate the two axes people conflate. 0 and 1 differ ONLY in
    // roughness and are both metal; 2 and 3 differ only in roughness and are
    // both dielectric; 4 varies roughness across a single surface.
    struct Entry {
        const char* name;
        eng::Vec4 base;
        float roughness;
        float metallic;
        bool textured;
    };
    const Entry table[kCount] = {
        {"polished gold", {1.00f, 0.78f, 0.34f, 1}, 0.10f, 1.0f, false},
        {"brushed gold", {1.00f, 0.78f, 0.34f, 1}, 0.45f, 1.0f, false},
        {"glossy lacquer", {0.80f, 0.15f, 0.13f, 1}, 0.12f, 0.0f, false},
        {"matte paint", {0.22f, 0.38f, 0.70f, 1}, 0.90f, 0.0f, false},
        {"textured + roughness map", {1.0f, 1.0f, 1.0f, 1}, 1.0f, 1.0f, true},
    };
    for (const Entry& e : table) {
        eng::MaterialDesc m;
        m.base_color = e.base;
        m.roughness = e.roughness;  // scaled by the map when there is one
        m.metallic = e.metallic;
        if (e.textured) {
            m.albedo = tex_checker;
            m.roughness_map = tex_ramp;
        }
        a.materials.push_back(r.CreateMaterial(m, error));
        a.names.push_back(e.name);
    }

    eng::MaterialDesc floor;
    floor.base_color = eng::Vec4{0.30f, 0.31f, 0.33f, 1};
    floor.roughness = 0.7f;
    a.floor_mat = r.CreateMaterial(floor, error);

    a.ok = Valid(a.sphere) && Valid(a.ground) && Valid(a.floor_mat) &&
           Valid(tex_checker) && Valid(tex_ramp);
    for (eng::MaterialHandle h : a.materials)
        if (!Valid(h)) a.ok = false;
    if (!a.ok && error.empty()) error = "failed to build the material demo assets";
    return a;
}

inline eng::Scene MakeScene(const Assets& a, float spin, float sun_azimuth,
                            bool shadows) {
    eng::Scene s;
    constexpr float kElevation = 0.62f;
    const float ce = std::cos(kElevation);
    s.lightDir = eng::Vec4{ce * std::cos(sun_azimuth), std::sin(kElevation),
                           ce * std::sin(sun_azimuth), 0.0f};
    s.lightColor = eng::Vec4{4.5f, 4.3f, 4.0f, 1.0f};
    s.shadowExtent = shadows ? 8.0f : 0.0f;

    s.instances.push_back({a.ground, a.floor_mat,
                           eng::Mat4::Translation({0.0f, -0.4f - kRadius, 0.0f}),
                           eng::Vec4{1, 1, 1, 1}});
    for (int i = 0; i < kCount; ++i) {
        const float x = (float(i) - float(kCount - 1) * 0.5f) * kSpacing;
        s.instances.push_back({a.sphere, a.materials[std::size_t(i)],
                               eng::Mat4::Translation({x, 0.0f, 0.0f}) *
                                   eng::Mat4::RotationY(spin),
                               eng::Vec4{1, 1, 1, 1}});
    }
    return s;
}

// Where sphere `i` lands in pixels, for the offscreen check. Perspective, so
// this only holds for the default camera the check sets up.
inline void SpherePixel(int i, int width, int height, const eng::Camera& cam,
                        int* px, int* py) {
    const float x = (float(i) - float(kCount - 1) * 0.5f) * kSpacing;
    const eng::Mat4 vp = cam.ViewProj(float(width) / float(height));
    const eng::Vec4 clip = vp * eng::Vec4{x, 0.0f, 0.0f, 1.0f};
    const float nx = clip.x / clip.w;
    const float ny = clip.y / clip.w;
    *px = int(float(width) * 0.5f * (1.0f + nx));
    *py = int(float(height) * 0.5f * (1.0f - ny));  // row 0 is the TOP
}

}  // namespace demo
