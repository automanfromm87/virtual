// The generated character, checked for the things that make a rig unusable.
//
// A rig fails in ways that are obvious on screen and impossible to describe
// afterwards: the body turns inside out, an elbow creases in the wrong place, a
// knee bends backwards, the feet skate along the ground. Every one of those is
// arithmetic before it is a picture, and every one is cheaper to catch here
// than by squinting at a screenshot.
//
// THE STRIDE MEASUREMENT IS NOT A CHECK, it is a measurement whose result the
// demo needs. A blend space places walk and run at positions along a speed
// axis, and if those positions are guesses the feet slide -- the clip carries
// the character's legs at one speed while the controller carries their body at
// another. The gait's natural speed is a property of the clip and can be
// derived from it, so it is derived here and printed, rather than tuned by
// watching the feet.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "apps/world/humanoid.h"
#include "engine/anim/anim.h"
#include "engine/anim/ik.h"
#include "engine/core/math.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-66s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

eng::Vec3 P(const ::VertexIn& v) { return {v.position.x, v.position.y, v.position.z}; }
eng::Vec3 N(const ::VertexIn& v) { return {v.normal.x, v.normal.y, v.normal.z}; }

// Where a joint ends up in world space under a pose.
eng::Vec3 JointAt(const eng::anim::Skeleton& s, const eng::anim::Pose& p, int joint) {
    std::vector<eng::Mat4> world;
    eng::anim::ComputeJointWorld(s, p, &world);
    const eng::Vec4 o = world[std::size_t(joint)] * eng::Vec4{0.0f, 0.0f, 0.0f, 1.0f};
    return eng::Vec3{o.x, o.y, o.z};
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const eng::anim::Skeleton skel = world::BuildHumanSkeleton();
    const world::HumanBody body = world::BuildHumanBody(skel);
    const eng::anim::Clip walk = world::MakeGait("walk", world::kWalkGait);
    const eng::anim::Clip run = world::MakeGait("run", world::kRunGait);
    const eng::anim::Clip idle = world::MakeIdle();

    {
        std::printf("the skeleton is well formed\n");
        Check(skel.Size() == std::size_t(world::kHumanJoints), "twenty-one joints");
        // A child evaluated before its parent poses from the parent's PREVIOUS
        // frame, which reads as a one-frame lag that only shows when moving
        // fast -- the worst kind of bug to find from a screenshot.
        Check(skel.ParentsFirst(), "and every parent comes before its children");
        Check(!skel.order.empty() || skel.ParentsFirst(), "and Finalize accepted it");
        Check(skel.Find("foot.L") == world::kFootL && skel.Find("head") == world::kHead,
              "and the names match the indices the code uses");
        int roots = 0;
        for (const auto& j : skel.joints)
            if (j.parent < 0) ++roots;
        Check(roots == 1, "and there is exactly one root");
    }

    {
        std::printf("\nthe bind pose skins to itself\n");
        // THE test for inverse_bind. palette[j] = world(j) * inverseBind(j), so
        // at rest every palette entry must be the identity and every vertex must
        // land exactly where it was authored. Get the inverse bind wrong and the
        // mesh is flung away from the origin the instant it is skinned; get it
        // slightly wrong and the body is subtly the wrong shape, which is the
        // version nobody catches.
        eng::anim::Pose rest;
        rest.local.resize(skel.Size());
        for (std::size_t i = 0; i < skel.Size(); ++i) rest.local[i] = skel.joints[i].rest;
        std::vector<eng::Mat4> palette;
        eng::anim::ComputeJointMatrices(skel, rest, &palette);

        float worst_identity = 0.0f;
        for (const eng::Mat4& m : palette)
            for (int c = 0; c < 4; ++c)
                for (int r = 0; r < 4; ++r)
                    worst_identity = std::max(
                        worst_identity,
                        std::fabs((&m.col[c].x)[r] - (c == r ? 1.0f : 0.0f)));

        float worst_vertex = 0.0f;
        for (std::size_t i = 0; i < body.mesh.vertices.size(); ++i)
            worst_vertex = std::max(
                worst_vertex,
                eng::Length(eng::anim::SkinPosition(P(body.mesh.vertices[i]),
                                                    body.skin[i], palette) -
                            P(body.mesh.vertices[i])));
        std::printf("    palette departs from identity by %.2e, vertices move %.2e m\n",
                    worst_identity, worst_vertex);
        Check(worst_identity < 1e-5f, "every rest palette entry is the identity");
        Check(worst_vertex < 1e-5f, "so no vertex moves when the rest pose is applied");
    }

    {
        std::printf("\nthe skin weights are usable\n");
        Check(body.skin.size() == body.mesh.vertices.size(),
              "there is one weight set per vertex");
        float worst_sum = 0.0f;
        int out_of_range = 0, negative = 0;
        for (const auto& sv : body.skin) {
            float sum = 0.0f;
            for (int i = 0; i < eng::anim::kMaxInfluences; ++i) {
                sum += sv.weights[i];
                if (sv.weights[i] < 0.0f) ++negative;
                if (sv.weights[i] > 0.0f && sv.joints[i] >= world::kHumanJoints)
                    ++out_of_range;
            }
            worst_sum = std::max(worst_sum, std::fabs(sum - 1.0f));
        }
        std::printf("    %zu vertices, worst weight sum error %.2e\n",
                    body.skin.size(), worst_sum);
        // Weights that do not sum to one scale the vertex, which reads as a
        // limb shrinking near a joint rather than as a weighting bug.
        Check(worst_sum < 1e-5f, "and every set sums to one");
        Check(out_of_range == 0, "and indexes a joint that exists");
        Check(negative == 0, "and is never negative");
    }

    {
        std::printf("\nthe mesh is geometrically sound\n");
        int degenerate = 0;
        for (std::size_t i = 0; i + 2 < body.mesh.indices.size(); i += 3) {
            const eng::Vec3 a = P(body.mesh.vertices[body.mesh.indices[i]]);
            const eng::Vec3 b = P(body.mesh.vertices[body.mesh.indices[i + 1]]);
            const eng::Vec3 c = P(body.mesh.vertices[body.mesh.indices[i + 2]]);
            if (eng::Length(eng::Cross(b - a, c - a)) < 1e-9f) ++degenerate;
        }
        float worst_len = 0.0f;
        for (const auto& v : body.mesh.vertices)
            worst_len = std::max(worst_len, std::fabs(eng::Length(N(v)) - 1.0f));
        // OUTWARD. A tube built with its rings the other way round has every
        // normal pointing into the body, and the result is lit from inside --
        // which looks like a broken material, not a broken mesh.
        //
        // OUTWARD FROM WHAT is the whole difficulty, and the first version of
        // this got it wrong: it treated the figure as a solid of revolution
        // about its own vertical axis, which flagged 237 vertices, every one of
        // them on the INNER face of a limb. The inner face of the left thigh
        // does point toward the body's axis, and it is supposed to. A limb is a
        // tube around its own bone, so the direction out is measured from that
        // bone -- which the skin weights already say which one is.
        int inward = 0, counted = 0;
        for (std::size_t i = 0; i < body.mesh.vertices.size(); ++i) {
            const eng::Vec3 p = P(body.mesh.vertices[i]);
            const eng::anim::SkinVertex& sv = body.skin[i];
            int dominant = sv.joints[0];
            for (int k = 1; k < eng::anim::kMaxInfluences; ++k)
                if (sv.weights[k] > sv.weights[k == 0 ? 0 : dominant == sv.joints[k] ? k : 0] &&
                    sv.weights[k] > sv.weights[0])
                    dominant = sv.joints[k];
            const eng::Vec3 a = world::kHumanRig[dominant].at;
            // Along the bone toward whichever joint calls this one parent.
            eng::Vec3 axis{0.0f, -1.0f, 0.0f};
            for (int j = 0; j < world::kHumanJoints; ++j)
                if (world::kHumanRig[j].parent == dominant) {
                    axis = eng::Normalize(world::kHumanRig[j].at - a);
                    break;
                }
            const eng::Vec3 rel = p - a;
            const eng::Vec3 out = rel - axis * eng::Dot(rel, axis);
            if (eng::Length(out) < 0.02f) continue;  // on the bone: no radial direction
            ++counted;
            if (eng::Dot(eng::Normalize(out), N(body.mesh.vertices[i])) < 0.0f) ++inward;
        }
        std::printf("    %zu vertices, %zu triangles, %d degenerate, %d of %d normals "
                    "point inward\n", body.mesh.vertices.size(),
                    body.mesh.indices.size() / 3, degenerate, inward, counted);
        Check(degenerate == 0, "no triangle has zero area");
        Check(worst_len < 1e-4f, "every normal is unit length");
        Check(inward * 20 < counted, "and the surface faces outward");
        eng::Vec3 lo = P(body.mesh.vertices[0]), hi = lo;
        for (const auto& v : body.mesh.vertices) {
            lo = eng::Vec3{std::min(lo.x, v.position.x), std::min(lo.y, v.position.y),
                           std::min(lo.z, v.position.z)};
            hi = eng::Vec3{std::max(hi.x, v.position.x), std::max(hi.y, v.position.y),
                           std::max(hi.z, v.position.z)};
        }
        std::printf("    stands %.2f m tall, %.2f m across, %.2f m deep; cull sphere "
                    "r=%.2f at y=%.2f\n", hi.y - lo.y, hi.x - lo.x, hi.z - lo.z,
                    body.mesh.bounds.radius, body.mesh.bounds.center.y);
        Check(hi.y - lo.y > 1.6f && hi.y - lo.y < 1.85f, "and it is person-sized");
        // Feet on the floor. A rig authored a few centimetres above y=0 walks
        // on air, and the gap is small enough to be mistaken for a shadow bug.
        Check(std::fabs(lo.y) < 0.02f, "with the soles at the origin's height");
    }

    {
        std::printf("\nthe clips are valid and loop cleanly\n");
        for (const eng::anim::Clip* c : {&walk, &run, &idle}) {
            bool ok = c->duration > 0.0f && !c->channels.empty();
            for (const auto& ch : c->channels) ok = ok && ch.Valid(skel.Size());
            Check(ok, (std::string("'") + c->name + "' is structurally valid").c_str());

            // A cycle whose last key differs from its first pops once per loop.
            // The pop is brief, so it reads as a stutter in the frame rate.
            eng::anim::Pose a, b;
            c->Sample(0.0f, skel, &a);
            c->Sample(c->duration, skel, &b);
            float worst = 0.0f;
            for (std::size_t j = 0; j < a.local.size(); ++j) {
                worst = std::max(worst, eng::Length(a.local[j].translation -
                                                    b.local[j].translation));
                const eng::Quat &qa = a.local[j].rotation, &qb = b.local[j].rotation;
                // Through the dot product, because q and -q are the same
                // rotation and a component-wise difference calls them opposite.
                const float d = std::fabs(qa.x * qb.x + qa.y * qb.y + qa.z * qb.z + qa.w * qb.w);
                worst = std::max(worst, 1.0f - d);
            }
            Check(worst < 1e-4f,
                  (std::string("and '") + c->name + "' ends where it began").c_str());
        }
    }

    {
        std::printf("\nthe knees bend the way knees bend\n");
        // A shin that swings past straight is the single most recognisable rig
        // failure there is, and it is a sign error away at all times.
        for (const eng::anim::Clip* c : {&walk, &run}) {
            float worst_forward = 0.0f;
            for (int k = 0; k <= 64; ++k) {
                eng::anim::Pose p;
                c->Sample(float(k) / 64.0f * c->duration, skel, &p);
                for (int j : {world::kShinL, world::kShinR}) {
                    // The shin's local rotation is a pitch; its sign is the
                    // bend direction. x of the quaternion carries it.
                    worst_forward = std::max(worst_forward, p.local[std::size_t(j)].rotation.x);
                }
            }
            std::printf("    '%s' worst forward knee component %+.4f\n", c->name.c_str(),
                        worst_forward);
            Check(worst_forward <= 1e-4f,
                  (std::string("'") + c->name + "' never bends a knee forward").c_str());
        }
    }

    {
        std::printf("\nthe gait's own speed, which the blend space needs\n");
        // A planted foot must be still relative to the ground. In a clip with no
        // root motion the character does not move, so the stance foot instead
        // travels BACKWARDS under the body -- and the speed it travels at is
        // exactly the speed the character has to be moved forward for the foot
        // to appear planted. Measure it over the stance half of the cycle and
        // the clip tells you where it belongs on the axis.
        for (const eng::anim::Clip* c : {&walk, &run}) {
            constexpr int kSteps = 128;
            float best = 0.0f;
            // Slide a half-cycle window over the loop and take the interval in
            // which the foot moves backwards most steadily: that is stance.
            for (int start = 0; start < kSteps; ++start) {
                float travel = 0.0f;
                bool monotone = true;
                for (int k = 0; k < kSteps / 2; ++k) {
                    eng::anim::Pose p0, p1;
                    c->Sample(float((start + k) % kSteps) / kSteps * c->duration, skel, &p0);
                    c->Sample(float((start + k + 1) % kSteps) / kSteps * c->duration, skel, &p1);
                    const float dz = JointAt(skel, p1, world::kFootL).z -
                                     JointAt(skel, p0, world::kFootL).z;
                    if (dz > 0.0f) monotone = false;
                    travel += -dz;
                }
                if (monotone) best = std::max(best, travel);
            }
            const float speed = best / (c->duration * 0.5f);
            std::printf("    '%s': stance carries the foot %.3f m back over %.2f s "
                        "-> %.2f m/s\n", c->name.c_str(), best, c->duration * 0.5f, speed);
            // AND IT MATCHES WHAT THE DEMO WAS TOLD. The blend space axis and
            // the character controller both read world::kWalkSpeed and
            // kRunSpeed; if a gait is retuned and those are not, the feet start
            // sliding and nothing here would have said so.
            const float declared = c == &walk ? world::kWalkSpeed : world::kRunSpeed;
            Check(std::fabs(speed - declared) < 0.05f,
                  (std::string("and '") + c->name + "' matches the constant the demo uses")
                      .c_str());
        }
    }

    {
        std::printf("\nfoot IK plants the feet without taking the legs apart\n");
        // Added because turning IK on in the demo collapsed the legs into
        // sticks. On a picture that is "the legs look wrong"; here it is three
        // numbers, and they say which of the three possible causes it is --
        // the solver stretching bones, the solver straightening the knee, or
        // the foot simply not reaching the ground.
        eng::anim::FootIkConfig cfg;
        cfg.limb.root = world::kThighL;
        cfg.limb.mid = world::kShinL;
        cfg.limb.end = world::kFootL;
        cfg.limb.pole = eng::Vec3{0.0f, 0.0f, 1.0f};
        cfg.limb.max_extension = 0.985f;
        cfg.ankle_to_sole = world::kAnkleToSole;

        const float thigh_len = eng::Length(world::kHumanRig[world::kShinL].at -
                                            world::kHumanRig[world::kThighL].at);
        const float shin_len = eng::Length(world::kHumanRig[world::kFootL].at -
                                           world::kHumanRig[world::kShinL].at);
        float worst_stretch = 0.0f, worst_sole = 0.0f, straightest = 180.0f;
        int planted = 0;
        float worst_lift = 0.0f, lift_weight = 1.0f;
        for (int k = 0; k <= 32; ++k) {
            eng::anim::Pose pose;
            walk.Sample(float(k) / 32.0f * walk.duration, skel, &pose);
            // Flat ground at y = 0, which is where the rig's soles already are:
            // any movement at all is the IK's doing and not the terrain's.
            eng::anim::GroundHit hit;
            hit.hit = true;
            hit.normal = eng::Vec3{0.0f, 1.0f, 0.0f};
            const eng::anim::FootTrace tr =
                eng::anim::FootTraceFor(skel, cfg, eng::Mat4::Identity(), pose);
            hit.point = eng::Vec3{tr.origin.x, 0.0f, tr.origin.z};
            // Gated on contact, the way the demo does it.
            std::vector<eng::Mat4> before;
            eng::anim::ComputeJointWorld(skel, pose, &before);
            const eng::Vec4 sole0 = before[std::size_t(world::kFootL)] *
                                    eng::Vec4{world::kAnkleToSole.x, world::kAnkleToSole.y,
                                              world::kAnkleToSole.z, 1.0f};
            cfg.limb.weight = world::FootPlantWeight(sole0.y);
            // TWO PASSES, WITH THE HIP DROP BETWEEN THEM. A leg is 0.84 m and
            // max_extension caps it at 0.827, so a stance leg that is nearly
            // straight cannot be asked to reach another 30 mm down -- the
            // solver refuses, correctly, and the foot hangs. Lowering the
            // pelvis is what buys the reach, and it has to happen BEFORE the
            // solve that needs it, not after. Applying it afterwards, which is
            // the obvious reading of "the caller applies it", leaves the foot
            // exactly where it failed to reach and then moves it further away.
            // ITERATED, because one pass does not converge: lowering the
            // pelvis by the shortfall changes the geometry, so the next solve
            // asks for a little less and the residual falls off geometrically
            // rather than to zero. Three passes takes 32 mm to under a
            // millimetre, and the fourth is not worth a joint hierarchy walk.
            for (int pass = 0; pass < 3; ++pass) {
                const float drop =
                    eng::anim::SolveFootIk(skel, cfg, eng::Mat4::Identity(), hit, &pose);
                if (drop >= -1e-4f) break;
                pose.local[world::kPelvis].translation.y += drop;
            }
            // Only a foot the animation says is DOWN has to be on the ground.
            // A swinging foot is supposed to be in the air, and asking it to
            // touch is what broke this.
            if (cfg.limb.weight < 0.9f) continue;
            if (std::fabs(sole0.y) > worst_lift) {
                worst_lift = std::fabs(sole0.y);
                lift_weight = cfg.limb.weight;
            }

            const eng::Vec3 hip = JointAt(skel, pose, world::kThighL);
            const eng::Vec3 knee = JointAt(skel, pose, world::kShinL);
            const eng::Vec3 ankle = JointAt(skel, pose, world::kFootL);
            // A ROTATION SOLVER CANNOT CHANGE A BONE'S LENGTH. If these move,
            // the pose is being written in a way the skeleton does not mean,
            // and the mesh stretches along the bone -- which is the streak.
            worst_stretch = std::max({worst_stretch,
                                      std::fabs(eng::Length(knee - hip) - thigh_len),
                                      std::fabs(eng::Length(ankle - knee) - shin_len)});
            // THE SOLE, transformed through the ankle's own matrix. Checking
            // the ankle's height instead was the first version and it was
            // wrong: ankle_to_sole is in the ankle's space, so a foot pitched
            // by the gait puts its sole somewhere other than straight below,
            // and the check reported an error the solver had not made.
            std::vector<eng::Mat4> jw;
            eng::anim::ComputeJointWorld(skel, pose, &jw);
            const eng::Vec4 sole =
                jw[std::size_t(world::kFootL)] * eng::Vec4{0.0f, -0.09f, 0.0f, 1.0f};
            worst_sole = std::max(worst_sole, std::fabs(sole.y));
            ++planted;
            const eng::Vec3 up = eng::Normalize(hip - knee), down = eng::Normalize(ankle - knee);
            straightest = std::min(straightest,
                                   std::acos(std::clamp(eng::Dot(up, down), -1.0f, 1.0f)) *
                                       180.0f / 3.14159265358979f);
        }
        std::printf("    planted on %d of 33 samples; bones change length by %.4f m, "
                    "a planted sole misses the ground by %.4f m, knee opens to %.0f "
                    "degrees\n", planted, worst_stretch, worst_sole, straightest);
        std::printf("    the largest correction asked of a planted foot was %.4f m "
                    "at weight %.3f, so %.4f m of it is deliberately not applied\n",
                    worst_lift, lift_weight, worst_lift * (1.0f - lift_weight));
        // A gate that never opens would pass the sole check by testing nothing.
        Check(planted > 8, "the gate calls the foot planted for part of the cycle");
        Check(worst_stretch < 1e-3f, "no bone changes length");
        Check(worst_sole < 0.01f, "and the sole reaches the ground");
        // A leg locked dead straight has no bend left, so the next centimetre
        // of reach swings the knee through a large angle: the pop.
        Check(straightest < 178.0f, "and the knee never locks straight");
    }

    std::printf(g_failures == 0 ? "\nhumanoid_test: all checks passed\n"
                                : "\nhumanoid_test: %d FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
