// Terrain collision: a height field as a physics body.
//
// The thing that has to be true is AGREEMENT. The renderer draws the terrain
// from eng::Terrain, the character stands on it through physics, and a click
// picks a point on it through a raycast. If those three disagree by a few
// centimetres the character's feet sink into a visibly solid slope, and the
// diagnosis is somewhere between "the animation is wrong" and "the physics is
// wrong" -- both false.
//
// So every check here is against the analytic surface the field was built from.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "engine/geometry/terrain.h"
#include "engine/physics/character.h"
#include "engine/physics/physics.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

using eng::Vec3;

// A gentle valley running along Z, so a ball placed on the side rolls toward
// the middle and a character can walk up out of it.
float Valley(float x, float z) {
    (void)z;
    return 0.02f * x * x - 1.0f;
}

// world2's terrain, verbatim, because that is where a character was found
// bouncing forever and the smooth valley above does not reproduce it. Four
// octaves of value noise in a bowl -- the difference that matters is that this
// one has SLOPE at every scale, including the scale of a single cell.
std::uint32_t LandHash32(int x, int z) {
    std::uint32_t h = std::uint32_t(x) * 374761393u + std::uint32_t(z) * 668265263u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return h ^ (h >> 16);
}
float LandNoise(float x, float z) {
    const int ix = int(std::floor(x)), iz = int(std::floor(z));
    const float fx = x - float(ix), fz = z - float(iz);
    const float sx = fx * fx * (3.0f - 2.0f * fx);
    const float sz = fz * fz * (3.0f - 2.0f * fz);
    const auto h = [](int a, int b) {
        return float(LandHash32(a, b) & 0xFFFFFFu) / 16777215.0f;
    };
    const float a = h(ix, iz), b = h(ix + 1, iz);
    const float c = h(ix, iz + 1), d = h(ix + 1, iz + 1);
    const float top = a + (b - a) * sx;
    return top + ((c + (d - c) * sx) - top) * sz;
}
float Landscape(float x, float z) {
    float height = 0.0f, amplitude = 6.0f, frequency = 0.012f;
    for (int octave = 0; octave < 4; ++octave) {
        height += (LandNoise(x * frequency, z * frequency) - 0.5f) * amplitude;
        amplitude *= 0.5f;
        frequency *= 2.1f;
    }
    const float r = std::sqrt(x * x + z * z);
    return height + std::max(0.0f, (r - 46.0f) * 0.55f);
}

std::shared_ptr<const eng::physics::HeightfieldData> MakeFieldFrom(
    int n, float spacing, Vec3 origin, float (*fn)(float, float)) {
    auto data = std::make_shared<eng::physics::HeightfieldData>();
    data->resolution = n;
    data->spacing = spacing;
    data->origin = origin;
    data->heights.resize(std::size_t(n) * std::size_t(n));
    for (int z = 0; z < n; ++z)
        for (int x = 0; x < n; ++x)
            data->heights[std::size_t(z) * std::size_t(n) + std::size_t(x)] =
                fn(origin.x + float(x) * spacing, origin.z + float(z) * spacing);
    return data;
}

std::shared_ptr<const eng::physics::HeightfieldData> MakeField(int n, float spacing,
                                                               Vec3 origin) {
    auto data = std::make_shared<eng::physics::HeightfieldData>();
    data->resolution = n;
    data->spacing = spacing;
    data->origin = origin;
    data->heights.resize(std::size_t(n) * std::size_t(n));
    for (int z = 0; z < n; ++z)
        for (int x = 0; x < n; ++x)
            data->heights[std::size_t(z) * std::size_t(n) + std::size_t(x)] =
                Valley(origin.x + float(x) * spacing, origin.z + float(z) * spacing);
    return data;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    constexpr int kN = 65;
    constexpr float kSpacing = 0.5f;
    const Vec3 kOrigin{-16.0f, 0.0f, -16.0f};
    const auto field = MakeField(kN, kSpacing, kOrigin);

    {
        std::printf("the height query matches the surface it was built from\n");
        float worst = 0.0f;
        for (float z = -15.0f; z < 15.0f; z += 1.7f)
            for (float x = -15.0f; x < 15.0f; x += 1.7f)
                worst = std::max(worst, std::fabs(field->HeightAt(x, z) - Valley(x, z)));
        std::printf("    worst error: %.5f m\n", worst);
        // A parabola is not exactly reproduced by bilinear interpolation, so
        // this is bounded by the curvature over one cell rather than by zero.
        Check(worst < 0.01f, "bilinear over a half-metre grid is within a centimetre");
    }

    {
        std::printf("\nthe physics field and eng::Terrain agree\n");
        // The two have separate implementations of the same bilinear lookup,
        // and a character standing on one while looking at the other is the
        // whole point of checking.
        eng::TerrainConfig tc;
        tc.resolution = kN;
        tc.world_size = float(kN - 1) * kSpacing;
        tc.origin = kOrigin;
        tc.chunk_resolution = 33;
        const eng::Terrain terrain = eng::Terrain::FromHeights(tc, field->heights);
        float worst = 0.0f;
        for (float z = -15.0f; z < 15.0f; z += 1.3f)
            for (float x = -15.0f; x < 15.0f; x += 1.3f)
                worst = std::max(worst,
                                 std::fabs(field->HeightAt(x, z) - terrain.HeightAt(x, z)));
        std::printf("    worst disagreement: %.7f m\n", worst);
        Check(worst < 1e-5f, "the renderer's terrain and the collider return the same height");
    }

    {
        std::printf("\na raycast lands on the surface\n");
        eng::physics::World world;
        eng::physics::Body ground;
        ground.shape = eng::physics::Shape::MakeHeightfield(field);
        ground.inverse_mass = 0.0f;
        world.Add(ground);

        int hits = 0, tested = 0;
        float worst = 0.0f;
        for (float z = -12.0f; z < 12.0f; z += 3.1f)
            for (float x = -12.0f; x < 12.0f; x += 3.1f) {
                ++tested;
                eng::physics::RayHit hit;
                if (!world.Raycast(Vec3{x, 20.0f, z}, Vec3{0, -1, 0}, 60.0f, &hit))
                    continue;
                ++hits;
                worst = std::max(worst, std::fabs(hit.point.y - field->HeightAt(x, z)));
            }
        std::printf("    %d of %d rays hit, worst error %.5f m\n", hits, tested, worst);
        Check(hits == tested, "every downward ray finds the ground");
        Check(worst < 1e-3f, "and lands on the surface");

        // The NORMAL points up out of the slope. On the valley's side at
        // x = 10 the surface rises steeply, so the normal leans toward -x.
        eng::physics::RayHit slope;
        Check(world.Raycast(Vec3{10.0f, 20.0f, 0.0f}, Vec3{0, -1, 0}, 60.0f, &slope),
              "a ray on the valley wall hits");
        std::printf("    normal on the wall: (%.3f, %.3f, %.3f)\n", slope.normal.x,
                    slope.normal.y, slope.normal.z);
        Check(slope.normal.y > 0.5f, "and its normal points upward");
        Check(slope.normal.x < -0.2f, "and leans downhill");
    }

    {
        std::printf("\na ball rests on the terrain rather than falling through\n");
        eng::physics::World world;
        world.sleep_after = 0.0f;
        eng::physics::Body ground;
        ground.shape = eng::physics::Shape::MakeHeightfield(field);
        ground.inverse_mass = 0.0f;
        world.Add(ground);

        eng::physics::Body ball;
        ball.shape = eng::physics::Shape::MakeSphere(0.4f);
        ball.position = Vec3{0.0f, 4.0f, 0.0f};
        ball.restitution = 0.0f;
        ball.SetMass(1.0f);
        const int index = world.Add(ball);
        for (int i = 0; i < 400; ++i) world.StepFixed();
        const Vec3 p = world[index].position;
        const float ground_y = field->HeightAt(p.x, p.z);
        std::printf("    settled at y = %.3f, ground is %.3f, radius 0.4\n", p.y,
                    ground_y);
        // ON the surface, one radius up. A ball that fell through would be far
        // below; one that never touched would still be at 4.
        Check(p.y > ground_y + 0.2f, "it did not fall through");
        Check(p.y < ground_y + 0.7f, "and it is resting on the surface");
        Check(world.Contacts().size() >= 1, "with a contact against the terrain");
    }

    {
        std::printf("\na ball on a slope rolls downhill\n");
        eng::physics::World world;
        world.sleep_after = 0.0f;
        eng::physics::Body ground;
        ground.shape = eng::physics::Shape::MakeHeightfield(field);
        ground.inverse_mass = 0.0f;
        world.Add(ground);

        eng::physics::Body ball;
        ball.shape = eng::physics::Shape::MakeSphere(0.35f);
        ball.position = Vec3{9.0f, field->HeightAt(9.0f, 0.0f) + 0.4f, 0.0f};
        ball.restitution = 0.0f;
        ball.friction = 0.3f;
        ball.SetMass(1.0f);
        const int index = world.Add(ball);
        const float start_x = ball.position.x;

        for (int i = 0; i < 600; ++i) world.StepFixed();
        const Vec3 p = world[index].position;
        std::printf("    started at x = %.2f, ended at x = %.2f\n", start_x, p.x);
        // DOWNHILL, which for this valley means toward x = 0. A ball that
        // stayed put means the contact normal came back as straight up on a
        // slope, which is what happens when the triangles are ignored and only
        // the bounding box is tested.
        Check(p.x < start_x - 2.0f, "it rolled a long way toward the valley floor");
        Check(p.y > field->HeightAt(p.x, p.z) - 0.1f, "and stayed on the surface");
    }

    {
        std::printf("\na character walks up the valley wall\n");
        eng::physics::World world;
        eng::physics::Body ground;
        ground.shape = eng::physics::Shape::MakeHeightfield(field);
        ground.inverse_mass = 0.0f;
        world.Add(ground);

        eng::physics::CharacterConfig cc;
        cc.radius = 0.3f;
        cc.height = 1.7f;
        cc.slope_limit_degrees = 60.0f;
        eng::physics::CharacterController player(cc);
        player.Teleport(Vec3{0.0f, field->HeightAt(0.0f, 0.0f) + 0.1f, 0.0f});

        // Settle first, so the walk starts from the ground rather than in mid
        // air -- the step-up needs footing established before the move.
        for (int i = 0; i < 30; ++i)
            player.Move(world, Vec3{0.0f, -9.81f, 0.0f} * (1.0f / 60.0f) * (1.0f / 60.0f));
        std::printf("    settled at y = %.3f, ground is %.3f\n", player.Feet().y,
                    field->HeightAt(player.Feet().x, player.Feet().z));
        Check(player.Grounded(), "the character finds the ground");
        Check(std::fabs(player.Feet().y - field->HeightAt(0.0f, 0.0f)) < 0.1f,
              "and stands on it rather than in it");

        // Walk toward +x, which is uphill.
        float worst_sink = 0.0f;
        for (int i = 0; i < 400; ++i) {
            player.Move(world, Vec3{2.0f, -9.81f * 0.016f, 0.0f} * (1.0f / 60.0f));
            const float ground_y = field->HeightAt(player.Feet().x, player.Feet().z);
            worst_sink = std::max(worst_sink, ground_y - player.Feet().y);
        }
        const Vec3 feet = player.Feet();
        std::printf("    walked to x = %.2f, y = %.2f (ground %.2f), worst sink "
                    "%.3f m\n", feet.x, feet.y, field->HeightAt(feet.x, feet.z),
                    worst_sink);
        Check(feet.x > 4.0f, "the character makes progress uphill");
        Check(feet.y > field->HeightAt(0.0f, 0.0f) + 0.3f, "and gains height doing it");
        // THE MEASUREMENT THAT MATTERS. Feet below the surface is the symptom
        // everyone sees, and it is what a collider that only tested the
        // terrain's bounding box would produce for the whole walk.
        Check(worst_sink < 0.12f, "without its feet ever sinking into the ground");
    }

    {
        std::printf("\na standing character stands still\n");
        // THE INVARIANT world2 now depends on, after its character was found
        // bouncing 14 mm forever while apparently at rest -- feet.y alternating
        // between -5.609918 and -5.623887 for as long as the demo ran. The
        // camera was attached to it, so what anyone saw was the PICTURE
        // flickering: 0.68% of pixels changed between two consecutive frames of
        // a completely still shot, some by three quarters of the range.
        //
        // IT IS NOT REPRODUCED HERE, and that is worth writing down rather than
        // hiding. Two mechanisms were tried and neither did it on this field: a
        // step longer than the skin, and the caller's 9.81 * dt * dt * 6 with a
        // frame time that jitters. Both come out under a hundredth of a
        // millimetre below. world2's terrain is four octaves of noise in a bowl
        // and this one is smooth, so the trigger is very likely a particular
        // slope or triangle boundary that this field does not have.
        //
        // What IS established is the property the fix relies on, and it is the
        // one worth guarding: a constant ground stick holds the character
        // exactly still. If that ever stops being true, every camera attached
        // to a character starts shaking.
        eng::physics::World world;
        eng::physics::Body ground;
        ground.shape = eng::physics::Shape::MakeHeightfield(field);
        ground.inverse_mass = 0.0f;
        world.Add(ground);

        // world2's numbers exactly, because a controller can behave differently
        // at a different radius or step height.
        eng::physics::CharacterConfig cc;
        cc.radius = 0.4f;
        cc.height = 1.8f;
        cc.slope_limit_degrees = 45.0f;
        cc.step_height = 0.4f;
        eng::physics::CharacterController player(cc);
        player.Teleport(Vec3{0.0f, field->HeightAt(0.0f, 0.0f) + 0.5f, 0.0f});

        const auto swing = [&](float stick, int frames) {
            float lo = 1e30f, hi = -1e30f;
            for (int i = 0; i < frames; ++i) {
                player.Move(world, Vec3{0.0f, -stick, 0.0f});
                lo = std::min(lo, player.Feet().y);
                hi = std::max(hi, player.Feet().y);
            }
            return hi - lo;
        };
        swing(cc.skin * 0.5f, 60);  // settle
        const float inside = swing(cc.skin * 0.5f, 200);
        std::printf("    ground stick %.1f mm, held for 200 frames: moves %.4f mm\n",
                    double(cc.skin * 0.5f * 1000.0f), double(inside * 1000.0f));
        // EXACTLY still, not nearly. A tenth of a millimetre of creep per frame
        // is 20 mm over a walk across the map, and it is a camera that never
        // stops drifting.
        Check(inside == 0.0f, "a constant ground stick leaves it exactly where it is");

        // AND IT DOES NOT CREEP SIDEWAYS either. A purely vertical push that
        // produces horizontal motion means the contact normal is being applied
        // asymmetrically, and the character slides off a flat surface on its
        // own -- which is the same class of bug seen from a different axis.
        const Vec3 before = player.Feet();
        swing(cc.skin * 0.5f, 200);
        const Vec3 after = player.Feet();
        const float drift = std::sqrt((after.x - before.x) * (after.x - before.x) +
                                      (after.z - before.z) * (after.z - before.z));
        std::printf("    and drifts %.4f mm horizontally over another 200\n",
                    double(drift * 1000.0f));
        Check(drift == 0.0f, "and does not slide across a surface it is resting on");
    }

    {
        std::printf("\nthe ground stick has a window, and it is narrow\n");
        // THE MECHANISM behind a character that bounced forever in world2, and
        // it needed world2's OWN terrain to show: the smooth valley above does
        // not reproduce it at any step size. What is different is that this
        // ground has slope at every scale.
        //
        // Swing over 200 frames while standing still, in millimetres:
        //
        //     stick     0.3 deg slope    10.2 deg slope
        //      1 mm        199              199
        //      5 mm        185              181
        //      7 mm         65               61
        //    7.5 mm          0                0
        //     12 mm          0                0
        //     13 mm          0               64
        //   14.9 mm          0              112
        //     30 mm          0              219
        //
        // TWO different failures with a good window between them.
        //
        // TOO SMALL and the character is never caught: the swing is very nearly
        // the step times the frame count, so it is simply falling. Something in
        // the ground probe needs a minimum push to register a contact.
        //
        // TOO LARGE and it penetrates, and depenetration pushes it back out
        // along the surface NORMAL -- which on a slope has a horizontal
        // component. So it moves sideways, lands on ground at a different
        // height, and does it again. That is why the swing grows with the slope
        // and vanishes on the flat, and it is why world2's character was
        // drifting in x and z as well as bouncing.
        //
        // world2 pushed 16.3 mm, which is past the upper edge. The first fix
        // used half the skin -- 7.5 mm -- which is EXACTLY the lower edge, one
        // step from a 65 mm failure. Right by luck. Two thirds of the skin sits
        // in the middle of the window, and the check below is on the MARGIN
        // rather than on the value, because a number that only works at one
        // point is not a fix.
        const auto rough = MakeFieldFrom(257, 0.5f, Vec3{-64.0f, 0.0f, -64.0f},
                                         Landscape);
        eng::physics::World world;
        eng::physics::Body ground;
        ground.shape = eng::physics::Shape::MakeHeightfield(rough);
        ground.inverse_mass = 0.0f;
        world.Add(ground);

        eng::physics::CharacterConfig cc;
        cc.radius = 0.4f;
        cc.height = 1.8f;
        cc.slope_limit_degrees = 45.0f;
        cc.step_height = 0.4f;

        const auto swing_at = [&](Vec3 where, float stick) {
            eng::physics::CharacterController p(cc);
            p.Teleport(Vec3{where.x, rough->HeightAt(where.x, where.z) + 0.5f, where.z});
            for (int i = 0; i < 60; ++i) p.Move(world, Vec3{0.0f, -stick, 0.0f});
            float lo = 1e30f, hi = -1e30f;
            for (int i = 0; i < 200; ++i) {
                p.Move(world, Vec3{0.0f, -stick, 0.0f});
                lo = std::min(lo, p.Feet().y);
                hi = std::max(hi, p.Feet().y);
            }
            return hi - lo;
        };

        // Five places, because a failure that depends on the local slope will
        // not show at one spot chosen by hand -- and the flat one would have
        // passed every step size in the table above.
        const Vec3 spots[] = {{0, 0, 0}, {12, 0, -7}, {-20, 0, 15},
                              {30, 0, 30}, {-35, 0, -5}};
        const float stick = cc.skin * (2.0f / 3.0f);
        float worst = 0.0f, worst_low = 0.0f, worst_high = 0.0f, worst_old = 0.0f;
        for (const Vec3& v : spots) {
            worst = std::max(worst, swing_at(v, stick));
            worst_low = std::max(worst_low, swing_at(v, stick * 0.8f));
            worst_high = std::max(worst_high, swing_at(v, stick * 1.2f));
            // 9.81 * dt * dt * 6 at 60 Hz, which is what world2 had.
            worst_old = std::max(worst_old, swing_at(v, 9.81f / 3600.0f * 6.0f));
        }
        std::printf("    worst swing over 5 spots: %.1f mm at the old 16.3 mm step, "
                    "%.4f mm at %.1f mm\n",
                    double(worst_old * 1000.0f), double(worst * 1000.0f),
                    double(stick * 1000.0f));
        std::printf("    and %.1f / %.1f nm at 20%% either side of it\n",
                    double(worst_low * 1e9f), double(worst_high * 1e9f));
        Check(worst == 0.0f, "the chosen stick holds the character exactly still");
        // THE MARGIN, which is the part worth guarding. If the controller ever
        // changes so that the window moves or narrows, this fails while the
        // check above still passes -- and a stick sitting on a cliff edge is a
        // bug that appears on somebody else's terrain, not on this one.
        // A MICRON, not zero, for the off-centre pair. They come out at tens
        // of NANOMETRES -- about one float bit at this magnitude -- because a
        // different step size is different arithmetic and lands on a different
        // last digit. The chosen value above is bit-exact and is asserted as
        // such; requiring that of its neighbours would be asserting a property
        // of float rounding rather than of the controller.
        Check(worst_low < 1e-6f && worst_high < 1e-6f,
              "and so does 20% either side of it, so it is not on an edge");
    }

    {
        std::printf("\noverlap queries find the terrain\n");
        eng::physics::World world;
        eng::physics::Body ground;
        ground.shape = eng::physics::Shape::MakeHeightfield(field);
        ground.inverse_mass = 0.0f;
        const int terrain_index = world.Add(ground);

        std::vector<int> found;
        // A sphere straddling the surface.
        const float y = field->HeightAt(3.0f, 3.0f);
        world.OverlapSphere(Vec3{3.0f, y, 3.0f}, 0.5f, &found);
        std::printf("    a sphere on the surface found %zu bodies\n", found.size());
        Check(found.size() == 1 && found[0] == terrain_index,
              "a sphere touching the ground overlaps it");

        // And one well above it does not. An overlap that always succeeded
        // would make every ground probe report a hit and the character would
        // hover.
        found.clear();
        world.OverlapSphere(Vec3{3.0f, y + 5.0f, 3.0f}, 0.5f, &found);
        Check(found.empty(), "and one five metres up does not");
    }

    {
        std::printf("\ndegenerate fields\n");
        eng::physics::World world;
        eng::physics::Body empty;
        empty.shape = eng::physics::Shape::MakeHeightfield(nullptr);
        empty.inverse_mass = 0.0f;
        world.Add(empty);
        eng::physics::RayHit hit;
        Check(!world.Raycast(Vec3{0, 10, 0}, Vec3{0, -1, 0}, 100.0f, &hit),
              "a null height field hits nothing");
        world.StepFixed();
        Check(world.Contacts().empty(), "and collides with nothing");
    }

    std::printf(g_failures == 0 ? "\nheightfield_test: all checks passed\n"
                                : "\nheightfield_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
