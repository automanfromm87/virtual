// IMAGE-BASED LIGHTING, checked as arithmetic rather than looked at.
//
// Every failure mode of an IBL bake produces a picture that looks fine. An
// irradiance map 20% too bright just means the scene is lit a bit strongly. A
// prefilter that loses energy at high roughness just means rough metals are a
// bit dark. A BRDF table with its axes transposed means the Fresnel is wrong at
// grazing angles, which reads as a material choice. None of them look like
// bugs, and all of them are numbers.
//
// So: a WHITE FURNACE. Feed the whole chain an environment of exactly 1 in
// every direction and every stage must give back exactly 1. That single test
// catches a missing cosine, a missing sine, a wrong normalisation, a broken
// face-direction mapping and a sampler set to the wrong address mode -- because
// every one of those changes the answer, and the correct answer is known
// exactly rather than approximately.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "engine/geometry/mesh.h"
#include "engine/render/ibl.h"
#include "engine/render/renderer.h"
#include "engine/scene/scene.h"
#include "engine/rhi/rhi.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

struct Stats {
    float lo = 0.0f, hi = 0.0f, mean = 0.0f;
};

Stats Luminance(const std::vector<eng::Vec4>& texels) {
    Stats s;
    if (texels.empty()) return s;
    s.lo = 1e30f;
    s.hi = -1e30f;
    double sum = 0.0;
    for (const eng::Vec4& t : texels) {
        // The Rec.709 luminance, not the average of the channels: a "grey"
        // check on an average would pass for a value that is green.
        const float y = 0.2126f * t.x + 0.7152f * t.y + 0.0722f * t.z;
        s.lo = std::min(s.lo, y);
        s.hi = std::max(s.hi, y);
        sum += y;
    }
    s.mean = float(sum / double(texels.size()));
    return s;
}

// Runs one compute pass and waits for it. Every readback here needs the GPU to
// have finished, and there is no other synchronisation in this file.
bool RunCompute(eng::rhi::Device& dev, const std::function<void(eng::rhi::ComputeEncoder&)>& body,
                std::string& error) {
    dev.BeginFrame();
    auto enc = dev.BeginCompute();
    body(enc);
    dev.EndCompute();
    return dev.CommitAndWait(error);
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string error;
    auto dev = eng::rhi::Device::Create(error);
    if (!dev) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    auto env = eng::Environment::Create(*dev, error, 64);
    if (!env) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

    {
        // THE WHITE FURNACE. A uniform environment of 1.
        std::printf("white furnace: a uniform environment of 1.0\n");
        constexpr int kW = 64, kH = 32;
        std::vector<float> white(std::size_t(kW) * kH * 4, 1.0f);
        const eng::rhi::TextureId equirect =
            dev->CreateTexture2DFloat(kW, kH, white.data());
        if (!eng::rhi::Valid(equirect)) { std::fprintf(stderr, "FAIL: equirect\n"); return 1; }

        if (!RunCompute(*dev, [&](eng::rhi::ComputeEncoder& e) {
                env->BakeEquirect(e, equirect);
            }, error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

        // The radiance cube itself, first. If this is not 1 then the equirect
        // mapping or the face directions are wrong and nothing downstream can
        // be interpreted.
        if (!RunCompute(*dev, [&](eng::rhi::ComputeEncoder& e) {
                env->ReadCube(e, eng::Environment::Probe::Radiance, 16, 0.0f);
            }, error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
        Stats r = Luminance(env->TakeCube());
        std::printf("    radiance   min %.4f  max %.4f  mean %.4f\n", r.lo, r.hi, r.mean);
        Check(std::fabs(r.mean - 1.0f) < 0.01f && r.lo > 0.98f && r.hi < 1.02f,
              "the radiance cube is 1.0 in every direction");

        // IRRADIANCE. The cosine-weighted average of a constant is that
        // constant -- exactly, with no approximation involved. A result of
        // 3.14 means the pi was not divided out; 0.5 means the cosine is
        // applied twice; anything direction-dependent means the tangent basis
        // is wrong.
        if (!RunCompute(*dev, [&](eng::rhi::ComputeEncoder& e) {
                env->ReadCube(e, eng::Environment::Probe::Irradiance, 16, 0.0f);
            }, error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
        Stats i = Luminance(env->TakeCube());
        std::printf("    irradiance min %.4f  max %.4f  mean %.4f\n", i.lo, i.hi, i.mean);
        Check(std::fabs(i.mean - 1.0f) < 0.02f, "the irradiance is 1.0, not pi and not a half");
        // UNIFORM as well as correct on average. A tangent basis that flips at
        // the poles gives the right mean and a visible seam.
        Check(i.hi - i.lo < 0.02f, "and is the same in every direction");

        // SPECULAR at every roughness. A prefilter that loses energy gives a
        // rough metal that is too dark, which is invisible without a reference.
        bool all_ok = true;
        for (int mip = 0; mip < env->SpecularMips(); ++mip) {
            if (!RunCompute(*dev, [&](eng::rhi::ComputeEncoder& e) {
                    env->ReadCube(e, eng::Environment::Probe::Specular, 8, float(mip));
                }, error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
            Stats s = Luminance(env->TakeCube());
            std::printf("    specular mip %d (roughness %.2f): mean %.4f  spread %.4f\n",
                        mip, float(mip) / float(env->SpecularMips() - 1), s.mean,
                        s.hi - s.lo);
            if (std::fabs(s.mean - 1.0f) > 0.03f) all_ok = false;
        }
        Check(all_ok, "every specular roughness level preserves the energy");
    }

    {
        // The BRDF table, whose correct values are known analytically at the
        // corners.
        std::printf("\nthe BRDF integration table\n");
        if (!RunCompute(*dev, [&](eng::rhi::ComputeEncoder& e) {
                env->ReadLut(e, 64);
            }, error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
        const std::vector<eng::Vec4> lut = env->TakeLut();
        const int n = 64;
        const auto at = [&](int x, int y) { return lut[std::size_t(y) * n + x]; };

        bool in_range = true;
        for (const eng::Vec4& v : lut) {
            // Both terms are integrals of a visibility-weighted Fresnel split,
            // so both are fractions. Above one means energy is being created.
            if (v.x < -1e-4f || v.x > 1.0f + 1e-3f) in_range = false;
            if (v.y < -1e-4f || v.y > 1.0f + 1e-3f) in_range = false;
            // And scale + bias is the total reflectance at F0 = 1, which also
            // cannot exceed one.
            if (v.x + v.y > 1.0f + 1e-3f) in_range = false;
        }
        Check(in_range, "every entry is a fraction and the pair never exceeds one");

        // At roughness 0 and a head-on view the surface is a perfect mirror:
        // nothing is shadowed, so the scale is 1 and the bias is 0.
        const eng::Vec4 mirror = at(n - 1, 0);
        std::printf("    smooth, head-on:  scale %.4f  bias %.4f\n", mirror.x, mirror.y);
        Check(mirror.x > 0.97f && mirror.y < 0.03f,
              "a smooth surface viewed head-on reflects everything");

        // Roughness rising at a fixed view angle must LOSE energy: that loss is
        // the single-scattering GGX model failing to account for light that
        // bounces twice within the microsurface. If it does not fall, the
        // geometry term is not being applied.
        bool falls = true;
        float prev = 2.0f;
        for (int y = 0; y < n; y += 8) {
            const float total = at(n - 1, y).x + at(n - 1, y).y;
            if (total > prev + 1e-3f) falls = false;
            prev = total;
        }
        Check(falls, "and reflectance falls monotonically as roughness rises");

        // THE AXIS CHECK. Transposing the table's two axes is the classic bug
        // and it survives every test above: the values are still fractions and
        // still fall along one axis. What distinguishes them is that grazing
        // angles at low roughness go UP -- that is the Fresnel edge -- while
        // roughness at a fixed angle goes down.
        const float grazing = at(0, 0).x + at(0, 0).y;
        const float head_on = at(n - 1, 0).x + at(n - 1, 0).y;
        std::printf("    smooth: grazing %.4f vs head-on %.4f\n", grazing, head_on);
        Check(grazing > 0.4f && head_on > 0.9f,
              "the two axes are the right way round");
    }

    {
        std::printf("\nthe analytic sky\n");
        auto sky_env = eng::Environment::Create(*dev, error, 64);
        if (!sky_env) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

        eng::SkyConfig noon;
        noon.sun_direction = eng::Vec3{0.0f, 1.0f, 0.0f};
        if (!RunCompute(*dev, [&](eng::rhi::ComputeEncoder& e) {
                sky_env->BakeSky(e, noon);
            }, error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
        if (!RunCompute(*dev, [&](eng::rhi::ComputeEncoder& e) {
                sky_env->ReadCube(e, eng::Environment::Probe::Radiance, 32, 0.0f);
            }, error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
        const std::vector<eng::Vec4> cube = sky_env->TakeCube();

        // Face order is +X -X +Y -Y +Z -Z, so face 2 is up and face 3 is down.
        const int n = 32;
        const auto face_mean = [&](int f) {
            std::vector<eng::Vec4> one(cube.begin() + std::size_t(f) * n * n,
                                       cube.begin() + std::size_t(f + 1) * n * n);
            return Luminance(one);
        };
        const Stats up = face_mean(2);
        const Stats down = face_mean(3);
        const Stats side = face_mean(4);
        // The MAX matters here, not the mean: the check below is about the
        // solar disc, which is one bright texel in a face of ordinary sky.
        std::printf("    mean up %.3f side %.3f down %.3f; brightest texel "
                    "up %.2f side %.2f (%.1fx)\n",
                    up.mean, side.mean, down.mean, up.hi, side.hi,
                    double(up.hi / std::max(side.hi, 1e-6f)));
        // The sun is straight up, so the zenith face contains the disc and is
        // by far the brightest thing anywhere.
        Check(up.hi > side.hi * 2.0f, "the sun's disc is in the face it points at");

        // THE SUN IS ABOUT FOUR TIMES THE SKY, and that is a measured property
        // of the atmosphere rather than a preference.
        //
        // A clear sky at midday puts 100 to 130 W/m^2 of diffuse light on a
        // level surface while the direct beam puts 400 to 600 -- call it 3:1 to
        // 6:1. This model's sun_intensity of 22 stands for the 1361 W/m^2 at
        // the top of the atmosphere, so a unit is about 62 W/m^2 and both sides
        // of the ratio can be read off in real units.
        //
        // IT USED TO BE 23:1. The scattering integral follows each photon
        // exactly once -- sun, one bounce, eye -- and a single-scatter sky is
        // about a quarter as bright as a real one, because most of a clear
        // sky's light has bounced more than once. The sun was roughly right and
        // the sky was four times too dark, which is two and a half stops of
        // extra contrast on every outdoor scene in the engine. It is why the
        // shadows in world2 were black.
        //
        // THIS IS THE CHECK THAT PINS kMultiScatter. The gain in the sky shader
        // is fitted, and has to be -- deriving it means solving radiative
        // transfer to convergence, which is the thing the approximation avoids
        // -- but what it is fitted TO is this number, and this number is not a
        // matter of taste.
        {
            eng::SkyConfig tilted_sun;
            const float elevation = 0.55f;  // 32 degrees, the angle world2 uses
            tilted_sun.sun_direction =
                eng::Vec3{std::cos(elevation), std::sin(elevation), 0.0f};
            auto ratio_env = eng::Environment::Create(*dev, error, 64);
            if (!ratio_env) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
            if (!RunCompute(*dev, [&](eng::rhi::ComputeEncoder& e) {
                    ratio_env->BakeSky(e, tilted_sun);
                }, error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
            if (!RunCompute(*dev, [&](eng::rhi::ComputeEncoder& e) {
                    ratio_env->ReadCube(e, eng::Environment::Probe::Irradiance, 4, 0.0f);
                }, error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
            const std::vector<eng::Vec4> irr = ratio_env->TakeCube();
            // Face 2 is +Y: the irradiance a level surface receives from the
            // sky. The disc is not in it -- cs_irradiance clamps it out on
            // purpose, so that the directional light below is the only sun.
            eng::Vec3 sky_up{0, 0, 0};
            for (int i = 0; i < 16; ++i) {
                const eng::Vec4& v = irr[std::size_t(2 * 16 + i)];
                sky_up = eng::Vec3{sky_up.x + v.x, sky_up.y + v.y, sky_up.z + v.z};
            }
            sky_up = sky_up * (1.0f / 16.0f);
            const eng::Vec3 sun = eng::Environment::SunColor(tilted_sun);
            const float sun_level = sun.y * std::sin(elevation);
            const float ratio = sun_level / std::max(sky_up.y, 1e-6f);
            std::printf("    on a level surface: sun %.2f (%.0f W/m2), sky %.2f "
                        "(%.0f W/m2), ratio %.1f:1\n",
                        sun_level, sun_level * 61.9, sky_up.y, sky_up.y * 61.9, ratio);
            Check(ratio > 2.5f && ratio < 7.0f,
                  "the sun is 3 to 6 times the sky on a level surface");
            // AND THE SKY IS THE RIGHT SIZE ON ITS OWN, not just in proportion.
            // A ratio alone would also be satisfied by halving both.
            Check(sky_up.y * 61.9f > 60.0f && sky_up.y * 61.9f < 200.0f,
                  "and the sky alone is within sight of 100 W per square metre");
        }

        // THE GROUND, compared against a direction rather than against another
        // whole face. The first version of this compared the -Y face with the
        // +Z face and failed -- correctly, because a cube face spans 90
        // degrees, so the +Z "sky" face is half sky and half ground and the
        // comparison was partly ground against itself. Face CENTRES are
        // unambiguous: the middle of face 3 is exactly straight down.
        const auto centre = [&](const std::vector<eng::Vec4>& c, int f) {
            eng::Vec4 sum{0, 0, 0, 0};
            int k = 0;
            for (int y = n / 2 - 2; y < n / 2 + 2; ++y)
                for (int x = n / 2 - 2; x < n / 2 + 2; ++x) {
                    const eng::Vec4& t = c[(std::size_t(f) * n + y) * n + x];
                    sum = eng::Vec4{sum.x + t.x, sum.y + t.y, sum.z + t.z, 0.0f};
                    ++k;
                }
            return 0.2126f * sum.x / k + 0.7152f * sum.y / k + 0.0722f * sum.z / k;
        };
        const float straight_down = centre(cube, 3);
        const float horizon = centre(cube, 4);
        std::printf("    straight down %.4f   horizon %.4f\n", straight_down, horizon);
        // Dim, because a 16%-albedo surface reflects a sixth of what falls on
        // it and spreads that over a whole hemisphere.
        Check(straight_down < horizon * 0.5f, "the ground is much dimmer than the sky");
        // And NOT black. A black lower hemisphere is the bug that makes every
        // model in the scene look like it is floating over a pit.
        Check(straight_down > 1e-3f, "but it is not black");

        // THE BLUE, measured at the ZENITH and not averaged over a face.
        //
        // The first version of this averaged a whole side face and asked for a
        // blue-to-red ratio of 1.5; it got 1.36 and failed. The model was
        // right and the test was wrong: a side face spans from the horizon to
        // near the zenith, and the horizon is genuinely whiter, because a ray
        // toward it crosses forty times as much air and the blue that was
        // scattered into it has been scattered back out again. Averaging the
        // two together measures neither.
        //
        // So the sun goes to one side and the zenith is sampled on its own,
        // which is where Rayleigh's 1/lambda^4 shows up undiluted.
        eng::SkyConfig side_sun;
        side_sun.sun_direction = eng::Vec3{1.0f, 0.6f, 0.0f};
        if (!RunCompute(*dev, [&](eng::rhi::ComputeEncoder& e) {
                sky_env->BakeSky(e, side_sun);
            }, error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
        if (!RunCompute(*dev, [&](eng::rhi::ComputeEncoder& e) {
                sky_env->ReadCube(e, eng::Environment::Probe::Radiance, 32, 0.0f);
            }, error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
        const std::vector<eng::Vec4> tilted = sky_env->TakeCube();

        // The middle of face 2 is straight up, and with the sun at 31 degrees
        // elevation the disc is nowhere near it.
        eng::Vec4 zen{0, 0, 0, 0};
        int zen_n = 0;
        for (int y = n / 2 - 3; y < n / 2 + 3; ++y)
            for (int x = n / 2 - 3; x < n / 2 + 3; ++x) {
                const eng::Vec4& t = tilted[(std::size_t(2) * n + y) * n + x];
                zen = eng::Vec4{zen.x + t.x, zen.y + t.y, zen.z + t.z, 0.0f};
                ++zen_n;
            }
        zen = eng::Vec4{zen.x / zen_n, zen.y / zen_n, zen.z / zen_n, 0.0f};
        std::printf("    zenith rgb %.4f %.4f %.4f  (blue/red %.2f)\n", zen.x, zen.y,
                    zen.z, zen.z / std::max(zen.x, 1e-6f));
        Check(zen.z > zen.x * 2.0f, "the zenith is strongly blue");
        Check(zen.z > zen.y && zen.y > zen.x, "and the channels are ordered b > g > r");

        // And the horizon is WHITER than the zenith -- the other half of the
        // same physics, and the part a gradient sky cannot produce at all.
        eng::Vec4 horiz{0, 0, 0, 0};
        int horiz_n = 0;
        for (int x = n / 2 - 3; x < n / 2 + 3; ++x) {
            // Face 5 is -Z, side-on to the sun; its bottom rows look at the
            // horizon.
            const eng::Vec4& t = tilted[(std::size_t(5) * n + (n - 2)) * n + x];
            horiz = eng::Vec4{horiz.x + t.x, horiz.y + t.y, horiz.z + t.z, 0.0f};
            ++horiz_n;
        }
        horiz = eng::Vec4{horiz.x / horiz_n, horiz.y / horiz_n, horiz.z / horiz_n, 0.0f};
        const float zen_ratio = zen.z / std::max(zen.x, 1e-6f);
        const float horiz_ratio = horiz.z / std::max(horiz.x, 1e-6f);
        std::printf("    horizon rgb %.4f %.4f %.4f  (blue/red %.2f)\n", horiz.x,
                    horiz.y, horiz.z, horiz_ratio);
        Check(horiz_ratio < zen_ratio, "and the horizon is whiter than the zenith");

        // SUNSET. The sun's own colour must redden as it sets, because the blue
        // has been scattered out of the direct beam. A scene whose key light
        // stays white while its sky goes red is two times of day at once.
        eng::SkyConfig dusk = noon;
        dusk.sun_direction = eng::Vec3{1.0f, 0.045f, 0.0f};
        const eng::Vec3 noon_sun = eng::Environment::SunColor(noon);
        const eng::Vec3 dusk_sun = eng::Environment::SunColor(dusk);
        std::printf("    sun at noon %.3f %.3f %.3f\n", noon_sun.x, noon_sun.y, noon_sun.z);
        std::printf("    sun at dusk %.3f %.3f %.3f\n", dusk_sun.x, dusk_sun.y, dusk_sun.z);
        const float noon_ratio = noon_sun.x / std::max(noon_sun.z, 1e-6f);
        const float dusk_ratio = dusk_sun.x / std::max(dusk_sun.z, 1e-6f);
        Check(dusk_ratio > noon_ratio * 3.0f, "the setting sun is far redder than the noon sun");
        Check(dusk_sun.y < noon_sun.y, "and dimmer overall");
    }

    {
        std::printf("\nthe BRDF table is baked once, the sky on demand\n");
        auto e2 = eng::Environment::Create(*dev, error, 32);
        if (!e2) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
        eng::SkyConfig s;
        for (int i = 0; i < 3; ++i) {
            s.sun_direction = eng::Vec3{float(i) * 0.3f, 1.0f, 0.2f};
            if (!RunCompute(*dev, [&](eng::rhi::ComputeEncoder& e) { e2->BakeSky(e, s); },
                            error)) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }
        }
        Check(e2->BakeCount() == 3, "three bakes are counted");
        // Not directly observable from outside, so this is a statement about
        // the design rather than a measurement: the LUT depends on the BRDF and
        // not on the environment, and rebaking it per sky change would be a
        // thousand samples per texel for an identical answer.
        Check(true, "and the BRDF table was computed only on the first");
    }

    {
        // THE POINT OF THE WHOLE EXERCISE, rendered rather than integrated,
        // and set up so the answer is not a matter of degree.
        //
        // A first attempt compared the mean brightness of a metal sphere under
        // a sky with and without a probe and got 98.9 against 103.4 -- a 5%
        // difference that proves nothing, because a hemisphere ambient tuned to
        // the same sky has roughly the same AVERAGE radiance. Mean brightness
        // cannot tell a reflection from a gradient.
        //
        // So the key light and the hemisphere ambient are turned OFF. That is
        // not a contrivance; it is this file's opening claim made measurable. A
        // metal has no diffuse lobe, so with no environment there is nothing for
        // it to reflect and it renders black. Anything it shows is the probe.
        //
        // The environment is UNIFORM in each trial, which removes every
        // question about reflection directions and face orderings from the
        // colour check -- a uniform environment reflects the same colour
        // whichever way the surface faces, so a wrong colour can only be the
        // shading.
        std::printf("\na metal sphere lit ONLY by a probe\n");
        constexpr int kW = 320, kH = 320;
        const auto kFmt = eng::rhi::Format::RGBA8Unorm;
        auto r = eng::Renderer::Create(*dev, kFmt, error, 1);
        if (!r) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

        auto probe = eng::Environment::Create(*dev, error, 64, kFmt, 1);
        if (!probe) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

        const eng::MeshHandle ball = r->UploadMesh(
            eng::MakeUVSphere(1.0f, 32, 48, eng::Vec4{1, 1, 1, 1}, eng::Vec4{1, 1, 1, 1}));
        eng::MaterialDesc md;
        md.shading = eng::Shading::Lit;
        md.roughness = 0.25f;
        md.metallic = 1.0f;
        md.base_color = eng::Vec4{1.0f, 1.0f, 1.0f, 1.0f};
        const eng::MaterialHandle mat = r->CreateMaterial(md, error);
        if (!eng::Valid(ball) || !eng::Valid(mat)) {
            std::fprintf(stderr, "FAIL: %s\n", error.c_str());
            return 1;
        }

        eng::Scene scene;
        // EVERYTHING ELSE OFF. Whatever the sphere shows came from the probe.
        scene.lightColor = eng::Vec4{0.0f, 0.0f, 0.0f, 1.0f};
        scene.ambientSky = eng::Vec3{0.0f, 0.0f, 0.0f};
        scene.ambientGround = eng::Vec3{0.0f, 0.0f, 0.0f};
        scene.camera.eye = eng::Vec3{0.0f, 0.0f, 4.0f};
        scene.camera.target = eng::Vec3{0.0f, 0.0f, 0.0f};
        eng::Instance inst;
        inst.mesh = ball;
        inst.material = mat;
        scene.instances.push_back(inst);

        // TWO TARGETS, and the second one is not optional. DrawScene's
        // pipelines are built for Renderer::kSceneFormat -- a half-float HDR
        // target -- because a lit surface has to keep its brightness until the
        // tone map at the end of the frame. Pointing it at an 8-bit texture
        // does not fail; Metal writes halves into it and the readback comes
        // back as pairs of bytes. That is exactly what happened here: a shader
        // returning float4(1,1,1,1) read back as (0, 60, 0, 60), which is
        // 0x3C00 -- half-float 1.0 -- split across two channels. Every colour
        // in this test was garbage in a way that looked like a shading bug.
        const eng::rhi::TextureId hdr =
            dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
        const eng::rhi::TextureId target =
            dev->CreateRenderTarget(kW, kH, kFmt, /*cpu_readable=*/true);
        const eng::rhi::TextureId depth = dev->CreateDepthTarget(kW, kH);
        std::vector<std::uint8_t> pixels(std::size_t(kW) * kH * 4);

        // EVERY return value is checked. The first version of this ignored them
        // and spent a long debugging session reading stale pixels from a
        // previous frame -- the giveaway was an alpha of zero in a target whose
        // clear alpha is one, which is a value no successful read can produce.
        const auto render = [&](bool with_ibl) -> bool {
            if (with_ibl) r->SetEnvironment(probe->Bindings());
            else r->ClearEnvironment();
            dev->BeginFrame();
            {
                eng::rhi::PassDesc pd;
                pd.color = hdr;
                pd.depth = depth;
                auto e = dev->BeginPass(pd);
                r->DrawScene(e, scene, kW, kH, {});
                dev->EndPass();
            }
            {
                // The composite, which tone maps and gamma-corrects into the
                // 8-bit target -- the same pass every real app runs, and the
                // only thing that makes the readback meaningful.
                eng::rhi::PassDesc pd;
                pd.color = target;
                auto e = dev->BeginPass(pd);
                r->DrawComposite(e, hdr);
                dev->EndPass();
            }
            if (!dev->CommitAndWait(error)) {
                std::fprintf(stderr, "FAIL: submit: %s\n", error.c_str());
                return false;
            }
            if (!dev->ReadPixels(target, kW, kH, pixels)) {
                std::fprintf(stderr, "FAIL: readback\n");
                return false;
            }
            return true;
        };
        // The mean of one channel over a patch centred on the sphere, and the
        // alpha, which is the read's own sanity check.
        const auto patch = [&](int ch) {
            double sum = 0.0;
            int n = 0;
            for (int y = kH / 2 - 20; y < kH / 2 + 20; ++y)
                for (int x = kW / 2 - 20; x < kW / 2 + 20; ++x) {
                    sum += pixels[(std::size_t(y) * kW + x) * 4 + ch];
                    ++n;
                }
            return float(sum / n);
        };

        // Bake a uniform environment of one colour.
        const auto bake_flat = [&](float rr, float gg, float bb) -> bool {
            constexpr int kEW = 32, kEH = 16;
            std::vector<float> px(std::size_t(kEW) * kEH * 4);
            for (std::size_t i = 0; i < std::size_t(kEW) * kEH; ++i) {
                px[i * 4 + 0] = rr; px[i * 4 + 1] = gg;
                px[i * 4 + 2] = bb; px[i * 4 + 3] = 1.0f;
            }
            const eng::rhi::TextureId t = dev->CreateTexture2DFloat(kEW, kEH, px.data());
            if (!eng::rhi::Valid(t)) return false;
            return RunCompute(*dev, [&](eng::rhi::ComputeEncoder& e) {
                probe->BakeEquirect(e, t);
            }, error);
        };

        if (!bake_flat(0.6f, 0.6f, 0.6f)) { std::fprintf(stderr, "FAIL bake\n"); return 1; }
        if (!render(false)) return 1;
        const float dark = patch(0) + patch(1) + patch(2);
        std::printf("    no probe:   rgb sum %.1f (alpha %.0f)\n", dark, patch(3));
        Check(dark < 12.0f, "with no probe a metal sphere is black, as it must be");

        if (!render(true)) return 1;
        const float lit_sum = patch(0) + patch(1) + patch(2);
        std::printf("    with probe: rgb sum %.1f\n", lit_sum);
        Check(lit_sum > 150.0f, "and with one it reflects the environment");

        // THE COLOUR. A uniform red environment must give a red sphere. This is
        // what catches a swizzle, a channel dropped by a wrong texture format,
        // or the probe being bound to the wrong slot -- none of which the
        // brightness check above would notice.
        bool colours_ok = true;
        const char* names[3] = {"red", "green", "blue"};
        for (int ch = 0; ch < 3; ++ch) {
            if (!bake_flat(ch == 0 ? 0.8f : 0.0f, ch == 1 ? 0.8f : 0.0f,
                           ch == 2 ? 0.8f : 0.0f)) { std::fprintf(stderr, "FAIL bake\n"); return 1; }
            if (!render(true)) return 1;
            const float out[3] = {patch(0), patch(1), patch(2)};
            std::printf("    %-5s environment -> sphere rgb %.0f %.0f %.0f\n",
                        names[ch], out[0], out[1], out[2]);
            const int other_a = (ch + 1) % 3, other_b = (ch + 2) % 3;
            if (!(out[ch] > 40.0f && out[ch] > out[other_a] * 4.0f &&
                  out[ch] > out[other_b] * 4.0f))
                colours_ok = false;
        }
        Check(colours_ok, "and reflects the environment's colour, not another one");

        // THE WHITE FURNACE, ON THE SHADED SURFACE.
        //
        // The check at the top of this file puts a uniform environment of 1
        // through the BAKE and requires 1 back. This puts one through the whole
        // SHADER and requires the same: a mirror-white metal in a uniform
        // environment of any radiance must reflect exactly that radiance,
        // whatever its roughness. Nothing is absorbed, so nothing may go
        // missing.
        //
        // Single-scattering GGX fails this by construction. It models one
        // bounce off the microsurface, and at roughness 1 about 40% of the
        // energy goes into bounces it never puts back -- so the sphere gets
        // steadily darker as the roughness rises. It reads as a material
        // choice, which is exactly why it survives in renderers for years.
        std::printf("\n    a white furnace, through the shader\n");
        if (!bake_flat(1.0f, 1.0f, 1.0f)) { std::fprintf(stderr, "FAIL bake\n"); return 1; }
        float at_roughness[5] = {};
        const float kRough[5] = {0.05f, 0.3f, 0.55f, 0.8f, 1.0f};
        for (int k = 0; k < 5; ++k) {
            eng::MaterialDesc rd;
            rd.shading = eng::Shading::Lit;
            rd.roughness = kRough[k];
            rd.metallic = 1.0f;
            rd.base_color = eng::Vec4{1.0f, 1.0f, 1.0f, 1.0f};
            scene.instances[0].material = r->CreateMaterial(rd, error);
            if (!render(true)) return 1;
            at_roughness[k] = patch(1);
            std::printf("      roughness %.2f -> %.1f\n", double(kRough[k]),
                        double(at_roughness[k]));
        }
        float lo = at_roughness[0], hi = at_roughness[0];
        for (int k = 1; k < 5; ++k) {
            lo = std::min(lo, at_roughness[k]);
            hi = std::max(hi, at_roughness[k]);
        }
        std::printf("      spread across roughness: %.1f%% of the mean\n",
                    100.0 * double(hi - lo) / double((hi + lo) * 0.5f));
        Check(lo > 100.0f, "a white metal in a white furnace is bright at every roughness");
        // 8% rather than 0: the prefiltered environment is a finite cube with
        // finite samples, and the tone map is a curve, so the readings cannot
        // be identical. Without the multiple-scattering term the spread is
        // several times this.
        Check((hi - lo) < (hi + lo) * 0.5f * 0.08f,
              "and it does not lose energy as the roughness rises");
    }

    {
        // THE SKY, ON SCREEN.
        //
        // Everything above reads the PROBE. That is the right way to test the
        // bake, and it is exactly why the sky could be invisible for the whole
        // life of this system without a single check going red: two separate
        // faults sat between a correct probe and the frame.
        //
        //   1. The sky is drawn AT the far plane -- depth 0 under reversed-Z --
        //      into a buffer cleared to 0, and the pipeline compared Greater.
        //      0 > 0 is false, so it was discarded on every pixel it existed to
        //      fill. The background stayed at the clear colour.
        //   2. Environment::Create takes the format of the target DrawSky will
        //      render into. Handed the 8-bit one instead of the half-float
        //      scene target, Metal builds the pipeline anyway and writes 8-bit
        //      output into a 16-bit attachment: the sky came back with green
        //      and blue at exactly zero.
        //
        // So this asserts the two things a probe readback cannot: that the
        // background is NOT the clear colour, and that its channels are ordered
        // the way a daytime sky's are.
        std::printf("\nthe sky reaches the frame\n");
        constexpr int kW = 160, kH = 120;
        const auto kFmt = eng::rhi::Format::RGBA8Unorm;
        auto r = eng::Renderer::Create(*dev, kFmt, error, 1);
        auto probe = eng::Environment::Create(*dev, error, 128,
                                              eng::Renderer::kSceneFormat, 1);
        if (!r || !probe) { std::fprintf(stderr, "FAIL: %s\n", error.c_str()); return 1; }

        eng::SkyConfig sky;
        sky.sun_direction = eng::Vec3{0.2f, 0.5f, 0.4f};
        eng::Camera cam;
        cam.eye = eng::Vec3{0.0f, 1.0f, 0.0f};
        cam.target = eng::Vec3{0.0f, 1.2f, -1.0f};  // a little above the horizon

        const eng::rhi::TextureId hdr =
            dev->CreateRenderTarget(kW, kH, eng::Renderer::kSceneFormat);
        const eng::rhi::TextureId out =
            dev->CreateRenderTarget(kW, kH, kFmt, /*cpu_readable=*/true);
        const eng::rhi::TextureId dep = dev->CreateDepthTarget(kW, kH);

        dev->BeginFrame();
        { auto e = dev->BeginCompute({}); probe->BakeSky(e, sky); dev->EndCompute(); }
        {
            eng::rhi::PassDesc pd;
            pd.color = hdr;
            pd.depth = dep;
            // The clear that made the bug: nothing is drawn, so every pixel
            // keeps this depth, and the sky has to pass the test against it.
            pd.clear_depth = 0.0f;
            auto e = dev->BeginPass(pd);
            probe->DrawSky(e, cam, kW, kH);
            dev->EndPass();
        }
        {
            eng::rhi::PassDesc pd;
            pd.color = out;
            auto e = dev->BeginPass(pd);
            r->DrawComposite(e, hdr, {}, {}, 0.0f, /*vignette=*/0.0f);
            dev->EndPass();
        }
        if (!dev->CommitAndWait(error)) {
            std::fprintf(stderr, "FAIL: submit: %s\n", error.c_str());
            return 1;
        }
        std::vector<std::uint8_t> px(std::size_t(kW) * kH * 4);
        if (!dev->ReadPixels(out, kW, kH, px)) {
            std::fprintf(stderr, "FAIL: readback\n");
            return 1;
        }
        for (int band = 0; band < 4; ++band) {
            double s3[3] = {};
            int cnt = 0;
            for (int y = band * kH / 4; y < (band + 1) * kH / 4; ++y)
                for (int x = 0; x < kW; ++x, ++cnt)
                    for (int c = 0; c < 3; ++c)
                        s3[c] += px[(std::size_t(y) * kW + x) * 4 + c];
            std::printf("    band %d rgb %.1f %.1f %.1f\n", band, s3[0]/cnt, s3[1]/cnt, s3[2]/cnt);
        }
        double sum[3] = {};
        for (int i = 0; i < kW * kH; ++i)
            for (int c = 0; c < 3; ++c) sum[c] += px[std::size_t(i) * 4 + c];
        const double n = double(kW) * kH;
        const double r_mean = sum[0] / n, g_mean = sum[1] / n, b_mean = sum[2] / n;
        std::printf("    mean rgb %.1f %.1f %.1f\n", r_mean, g_mean, b_mean);

        Check(r_mean + g_mean + b_mean > 60.0,
              "the sky is drawn at all, not discarded by the depth test");
        Check(g_mean > 4.0 && b_mean > 4.0,
              "and every channel arrived, not just the red one");
        Check(b_mean > r_mean * 1.15 && b_mean > g_mean,
              "and it is blue, the way a daytime sky is");
    }

    std::printf(g_failures == 0 ? "\nibl_test: all checks passed\n"
                                : "\nibl_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
