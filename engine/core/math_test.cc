// No test framework — from scratch means from scratch. A failing CHECK prints
// the file and line and makes the binary exit non-zero, which is all Bazel's
// cc_test needs.
#include "engine/core/math.h"

#include <cmath>
#include <cstdio>

namespace {

int g_failures = 0;

void CheckNear(float got, float want, const char* what, int line) {
    // Inverted on purpose. `fabs(got - want) > eps` is FALSE when got is NaN,
    // so the natural spelling reports "all checks passed" on a NaN result.
    if (!(std::fabs(got - want) <= 1e-5f)) {
        std::fprintf(stderr, "math_test.cc:%d  %s: got %f, want %f\n", line, what,
                     got, want);
        ++g_failures;
    }
}

#define CHECK_NEAR(got, want) CheckNear((got), (want), #got, __LINE__)

void CheckTrue(bool ok, const char* what, int line) {
    if (!ok) {
        std::fprintf(stderr, "math_test.cc:%d  %s\n", line, what);
        ++g_failures;
    }
}

#define CHECK(cond) CheckTrue((cond), #cond, __LINE__)

}  // namespace

int main() {
    using namespace eng;

    // Identity leaves a vector alone.
    const Vec4 v{1, 2, 3, 1};
    const Vec4 iv = Mat4::Identity() * v;
    CHECK_NEAR(iv.x, 1);
    CHECK_NEAR(iv.y, 2);
    CHECK_NEAR(iv.z, 3);
    // w too: a broken bottom row is invisible if you only ever look at xyz, and
    // it silently destroys the perspective divide.
    CHECK_NEAR(iv.w, 1);

    // Translation is in the last COLUMN, and v' = M * v applies it.
    const Vec4 tv = Mat4::Translation({10, 0, 0}) * v;
    CHECK_NEAR(tv.x, 11);
    CHECK_NEAR(tv.w, 1);
    // A DIRECTION (w = 0) must not be translated.
    const Vec4 dirv = Mat4::Translation({10, 0, 0}) * Vec4{1, 0, 0, 0};
    CHECK_NEAR(dirv.x, 1);
    CHECK_NEAR(dirv.w, 0);

    // Composition order: (A * B) * v applies B first.
    //
    // This MUST mix a rotation with a translation. Two translations commute
    // exactly — T(a)*T(b) == T(b)*T(a) — so a test built from them passes under
    // either convention and catches nothing.
    constexpr float kHalfPi = 1.5707963f;
    const Mat4 rot = Mat4::RotationY(kHalfPi);  // +X -> -Z
    const Mat4 tr = Mat4::Translation({2, 0, 0});

    // Translate first, then rotate: origin -> (2,0,0) -> (0,0,-2).
    const Vec4 tr_then_rot = (rot * tr) * Vec4{0, 0, 0, 1};
    CHECK_NEAR(tr_then_rot.x, 0);
    CHECK_NEAR(tr_then_rot.z, -2);

    // Rotate first (origin is a fixed point), then translate: -> (2,0,0).
    const Vec4 rot_then_tr = (tr * rot) * Vec4{0, 0, 0, 1};
    CHECK_NEAR(rot_then_tr.x, 2);
    CHECK_NEAR(rot_then_tr.z, 0);

    // Rotation direction: right-handed about +Y sends +X to -Z, not +Z.
    const Vec4 spun = rot * Vec4{1, 0, 0, 0};
    CHECK_NEAR(spun.z, -1);

    // RotationX, which had no test at all. Right-handed about +X sends +Y to
    // +Z. A transposed implementation sends it to -Z and everything else here
    // still passes.
    const Vec4 tilted = Mat4::RotationX(kHalfPi) * Vec4{0, 1, 0, 0};
    CHECK_NEAR(tilted.y, 0);
    CHECK_NEAR(tilted.z, 1);
    // ...and leaves the axis it turns about alone.
    const Vec4 axis = Mat4::RotationX(kHalfPi) * Vec4{1, 0, 0, 0};
    CHECK_NEAR(axis.x, 1);

    // Rotations are rigid: no scale, no shear.
    const Vec4 rigid = Mat4::RotationX(0.7f) * Vec4{0.3f, -0.8f, 0.5f, 0};
    CHECK_NEAR(std::sqrt(rigid.x * rigid.x + rigid.y * rigid.y + rigid.z * rigid.z),
               std::sqrt(0.3f * 0.3f + 0.8f * 0.8f + 0.5f * 0.5f));

    // Uniform scale composes on the right (applied first).
    const Vec4 scaled = (tr * Mat4::Scale(3.0f)) * Vec4{1, 0, 0, 1};
    CHECK_NEAR(scaled.x, 5);  // 1*3 then +2

    // Reversed-Z: a point ON the near plane must land at z_ndc == 1.
    const Mat4 p = Mat4::PerspectiveReverseZ(1.0472f, 1.0f, 0.1f);
    const Vec4 nearPt = p * Vec4{0, 0, -0.1f, 1};
    CHECK_NEAR(nearPt.z / nearPt.w, 1.0f);

    // ...and a very distant point approaches z_ndc == 0. At -10000 the result
    // is 0.1/10000 == 1e-5f exactly, which is bit-identical to the tolerance —
    // the check would pass only because the comparison is strict. Go further
    // out so there is real margin.
    const Vec4 farPt = p * Vec4{0, 0, -1.0e6f, 1};
    CHECK_NEAR(farPt.z / farPt.w, 0.0f);

    // The x/y half of the projection: nothing above touches it, because every
    // probe so far sits on the axis with aspect == 1.
    //
    // fovY = 90 deg makes f = 1/tan(45) = 1, so a point one unit up at one unit
    // of depth lands exactly on the top edge (y_ndc == 1), and aspect == 2
    // halves the x term.
    const Mat4 wide = Mat4::PerspectiveReverseZ(kHalfPi, 2.0f, 0.1f);
    const Vec4 corner = wide * Vec4{1, 1, -1, 1};
    CHECK_NEAR(corner.w, 1);
    CHECK_NEAR(corner.x / corner.w, 0.5f);
    CHECK_NEAR(corner.y / corner.w, 1.0f);

    // --- quaternions ---------------------------------------------------------
    // Identity does nothing.
    const Vec4 qi = QuatToMat4(Quat{}) * Vec4{1, 2, 3, 1};
    CHECK_NEAR(qi.x, 1); CHECK_NEAR(qi.y, 2); CHECK_NEAR(qi.z, 3);

    // 90 degrees about +Y must agree with Mat4::RotationY, or glTF nodes and
    // hand-built transforms would disagree about which way is which.
    const Quat qy{0.0f, std::sin(kHalfPi * 0.5f), 0.0f, std::cos(kHalfPi * 0.5f)};
    const Vec4 fromQuat = QuatToMat4(qy) * Vec4{1, 0, 0, 0};
    const Vec4 fromEuler = Mat4::RotationY(kHalfPi) * Vec4{1, 0, 0, 0};
    CHECK_NEAR(fromQuat.x, fromEuler.x);
    CHECK_NEAR(fromQuat.z, fromEuler.z);
    CHECK_NEAR(fromQuat.z, -1.0f);  // +X -> -Z, right-handed

    // Rotations are rigid.
    const Quat qa{0.3f, -0.5f, 0.2f, 0.8f};
    const Vec4 qrot = QuatToMat4(qa) * Vec4{0.4f, 0.7f, -0.2f, 0};
    CHECK_NEAR(std::sqrt(qrot.x * qrot.x + qrot.y * qrot.y + qrot.z * qrot.z),
               std::sqrt(0.4f * 0.4f + 0.7f * 0.7f + 0.2f * 0.2f));

    // Slerp hits both ends exactly and stays unit-length in between.
    const Quat qb{0.0f, 0.0f, std::sin(0.6f), std::cos(0.6f)};
    const Quat s0 = Slerp(qa, qb, 0.0f), s1 = Slerp(qa, qb, 1.0f);
    const Quat na = Normalize(qa), nb = Normalize(qb);
    CHECK_NEAR(s0.x, na.x); CHECK_NEAR(s0.w, na.w);
    CHECK_NEAR(s1.x, nb.x); CHECK_NEAR(s1.w, nb.w);
    const Quat mid = Slerp(qa, qb, 0.37f);
    CHECK_NEAR(std::sqrt(mid.x * mid.x + mid.y * mid.y + mid.z * mid.z + mid.w * mid.w), 1.0f);

    // q and -q are the SAME rotation, so slerp must take the short way round
    // rather than spinning almost all the way about.
    const Quat neg{-nb.x, -nb.y, -nb.z, -nb.w};
    const Quat viaPos = Slerp(na, nb, 0.5f);
    const Quat viaNeg = Slerp(na, neg, 0.5f);
    const float agree = std::fabs(viaPos.x * viaNeg.x + viaPos.y * viaNeg.y +
                                  viaPos.z * viaNeg.z + viaPos.w * viaNeg.w);
    CHECK_NEAR(agree, 1.0f);

    // Orthographic reversed-Z: the near plane maps to 1, the far plane to 0,
    // and unlike the perspective version w stays 1 so there is no divide.
    const Mat4 ortho = Mat4::OrthographicReverseZ(2.0f, 2.0f, 1.0f, 11.0f);
    const Vec4 oNear = ortho * Vec4{0, 0, -1.0f, 1};
    CHECK_NEAR(oNear.w, 1);
    CHECK_NEAR(oNear.z, 1.0f);
    const Vec4 oFar = ortho * Vec4{0, 0, -11.0f, 1};
    CHECK_NEAR(oFar.z, 0.0f);
    // Halfway in depth lands halfway in z: an ortho projection is linear, which
    // is exactly why it does not need the reversed-Z precision trick that the
    // perspective one does.
    const Vec4 oMid = ortho * Vec4{0, 0, -6.0f, 1};
    CHECK_NEAR(oMid.z, 0.5f);
    // x/y scale by the half extents, with no perspective divide.
    const Vec4 oCorner = ortho * Vec4{2.0f, -2.0f, -5.0f, 1};
    CHECK_NEAR(oCorner.x, 1.0f);
    CHECK_NEAR(oCorner.y, -1.0f);

    // THE defining property of a parallel projection: moving a point away from
    // the camera does not change where it lands on screen. This is what makes
    // an orthographic plan view measurable, and it is a statement about the
    // matrix — checking it by measuring pixels only tells you how wide the
    // largest object in the frame happens to be.
    const Vec4 oNearXY = ortho * Vec4{1.5f, 0.5f, -2.0f, 1};
    const Vec4 oFarXY = ortho * Vec4{1.5f, 0.5f, -9.0f, 1};
    CHECK_NEAR(oNearXY.x / oNearXY.w, oFarXY.x / oFarXY.w);
    CHECK_NEAR(oNearXY.y / oNearXY.w, oFarXY.y / oFarXY.w);

    // ...and perspective must NOT have it, or the test above would pass on a
    // projection that is not orthographic at all.
    const Mat4 persp = Mat4::PerspectiveReverseZ(1.0472f, 1.0f, 0.1f);
    const Vec4 pNear = persp * Vec4{1.5f, 0.5f, -2.0f, 1};
    const Vec4 pFar = persp * Vec4{1.5f, 0.5f, -9.0f, 1};
    CHECK(std::fabs(pNear.x / pNear.w - pFar.x / pFar.w) > 0.5f);

    // LookAt puts the target straight down -Z in view space.
    const Mat4 view = Mat4::LookAt({0, 0, 5}, {0, 0, 0}, {0, 1, 0});
    const Vec4 origin = view * Vec4{0, 0, 0, 1};
    CHECK_NEAR(origin.x, 0);
    CHECK_NEAR(origin.y, 0);
    CHECK_NEAR(origin.z, -5);

    // Looking down +X instead: the axis-aligned case above cannot distinguish a
    // correct basis from a transposed one, because it is nearly the identity.
    const Mat4 side = Mat4::LookAt({5, 0, 0}, {0, 0, 0}, {0, 1, 0});
    const Vec4 target = side * Vec4{0, 0, 0, 1};
    CHECK_NEAR(target.x, 0);
    CHECK_NEAR(target.y, 0);
    CHECK_NEAR(target.z, -5);

    // `up` survives as view-space +Y. Note w = 0: this is a direction, so the
    // translation column must not touch it.
    const Vec4 upInView = side * Vec4{0, 1, 0, 0};
    CHECK_NEAR(upInView.x, 0);
    CHECK_NEAR(upInView.y, 1);
    CHECK_NEAR(upInView.z, 0);

    // A view matrix is rigid: it must not stretch anything.
    const Vec4 dir = side * Vec4{0, 0, 1, 0};
    CHECK_NEAR(std::sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z), 1.0f);

    // --- quaternion composition and rotation ---------------------------------
    {
        const Quat rx = QuatFromAxisAngle(Vec3{1, 0, 0}, kHalfPi);
        const Quat ry = QuatFromAxisAngle(Vec3{0, 1, 0}, kHalfPi);

        // Rotate() must agree with the matrix form. Two ways to say the same
        // thing, and the engine uses both — physics rotates vectors directly,
        // the ECS builds a matrix.
        const Vec3 v{0.4f, 0.7f, -0.2f};
        const Vec4 viaMat = QuatToMat4(ry) * Vec4{v.x, v.y, v.z, 0.0f};
        const Vec3 viaQuat = Rotate(ry, v);
        CHECK_NEAR(viaQuat.x, viaMat.x);
        CHECK_NEAR(viaQuat.y, viaMat.y);
        CHECK_NEAR(viaQuat.z, viaMat.z);

        // Composition is in the same order as the matrix product: (a*b) applies
        // b first. Getting this backwards makes every parent-child transform
        // wrong in a way that looks almost right.
        const Vec3 composed = Rotate(rx * ry, v);
        const Vec3 stepwise = Rotate(rx, Rotate(ry, v));
        CHECK_NEAR(composed.x, stepwise.x);
        CHECK_NEAR(composed.y, stepwise.y);
        CHECK_NEAR(composed.z, stepwise.z);
        // ...and it is NOT commutative, so the check above has real content.
        const Vec3 flipped = Rotate(ry * rx, v);
        CHECK(std::fabs(flipped.z - composed.z) > 0.1f);

        // Inverse round-trips.
        const Vec3 back = RotateInverse(ry, Rotate(ry, v));
        CHECK_NEAR(back.x, v.x);
        CHECK_NEAR(back.z, v.z);

        // A rotation preserves length, and a quarter turn about +y sends +x
        // to -z — the same handedness the matrix path uses.
        const Vec3 xr = Rotate(ry, Vec3{1, 0, 0});
        CHECK_NEAR(Length(xr), 1.0f);
        CHECK_NEAR(xr.z, -1.0f);

        // Zero axis must not divide by zero.
        const Quat degenerate = QuatFromAxisAngle(Vec3{0, 0, 0}, 1.0f);
        CHECK_NEAR(degenerate.w, 1.0f);
        CHECK(std::isfinite(degenerate.x));
    }

    if (g_failures == 0) std::printf("math_test: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
