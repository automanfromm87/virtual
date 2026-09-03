#include "engine/physics/physics.h"

#include <algorithm>
#include <cmath>

namespace eng::physics {
namespace {

Vec3 Clamp(Vec3 v, Vec3 lo, Vec3 hi) {
    return Vec3{std::clamp(v.x, lo.x, hi.x), std::clamp(v.y, lo.y, hi.y),
                std::clamp(v.z, lo.z, hi.z)};
}

Vec3 Mul(Vec3 a, Vec3 b) { return Vec3{a.x * b.x, a.y * b.y, a.z * b.z}; }

Vec3 Axis(int i, float sign) {
    return Vec3{i == 0 ? sign : 0.0f, i == 1 ? sign : 0.0f, i == 2 ? sign : 0.0f};
}

float Component(Vec3 v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }

// Applies an impulse at world offset `r` from the centre of mass. The angular
// half is the whole reason a ball rolls: an impulse through the centre changes
// only velocity, one at the rim also spins it.
void ApplyImpulse(Body& b, Vec3 impulse, Vec3 r) {
    if (b.IsStatic()) return;
    b.velocity = b.velocity + impulse * b.inverse_mass;
    b.angular_velocity = b.angular_velocity + b.ApplyInverseInertia(Cross(r, impulse));
}

bool PointInsideBox(const Body& box, Vec3 p) {
    const Vec3 local = RotateInverse(box.orientation, p - box.position);
    const Vec3 h = box.shape.half_extents;
    return std::fabs(local.x) <= h.x && std::fabs(local.y) <= h.y &&
           std::fabs(local.z) <= h.z;
}

// The eight corners of an oriented box, in world space.
void BoxCorners(const Body& box, Vec3 out[8]) {
    const Vec3 h = box.shape.half_extents;
    int n = 0;
    for (int sx = -1; sx <= 1; sx += 2)
        for (int sy = -1; sy <= 1; sy += 2)
            for (int sz = -1; sz <= 1; sz += 2)
                out[n++] = box.position +
                           Rotate(box.orientation,
                                  Vec3{h.x * float(sx), h.y * float(sy),
                                       h.z * float(sz)});
}

void BoxAxes(const Body& box, Vec3 out[3]) {
    out[0] = Rotate(box.orientation, Vec3{1, 0, 0});
    out[1] = Rotate(box.orientation, Vec3{0, 1, 0});
    out[2] = Rotate(box.orientation, Vec3{0, 0, 1});
}

// Half-width of the box's shadow on the line `l`.
float BoxRadius(const Vec3 axes[3], Vec3 h, Vec3 l) {
    return std::fabs(Dot(axes[0], l)) * h.x + std::fabs(Dot(axes[1], l)) * h.y +
           std::fabs(Dot(axes[2], l)) * h.z;
}

// Separating-axis test for two oriented boxes. `n_out` is unit and points from
// a toward b; `depth_out` is the least overlap found, which is the shortest
// push that would separate them.
//
// Fifteen axes, not six: two boxes can be apart with all six face directions
// overlapping, and only a cross product of one edge from each finds the gap.
// Testing only the faces reports a collision that is not there.
bool BoxBoxSat(const Body& a, const Body& b, Vec3* n_out, float* depth_out) {
    Vec3 ax[3], bx[3];
    BoxAxes(a, ax);
    BoxAxes(b, bx);
    const Vec3 d = b.position - a.position;

    float best = 1e30f;
    Vec3 best_axis{0.0f, 1.0f, 0.0f};
    bool found = false;

    // Returns false the moment an axis separates them.
    auto test = [&](Vec3 l, bool is_cross) {
        const float len2 = Dot(l, l);
        // Parallel edges give a zero-length cross product. That is not a
        // separating axis, it is a missing one — the parallel case is already
        // covered by the face axes.
        if (len2 < 1e-8f) return true;
        l = l * (1.0f / std::sqrt(len2));
        const float overlap = BoxRadius(ax, a.shape.half_extents, l) +
                              BoxRadius(bx, b.shape.half_extents, l) -
                              std::fabs(Dot(d, l));
        if (overlap <= 0.0f) return false;
        // Face axes win near-ties. An edge-edge normal is the noisier of the
        // two, and preferring it when a face is just as shallow makes a stack
        // of crates shiver instead of resting.
        const float biased = is_cross ? overlap * 1.02f : overlap;
        if (biased < best) {
            best = biased;
            best_axis = l;
            found = true;
        }
        return true;
    };

    for (int i = 0; i < 3; ++i)
        if (!test(ax[i], false)) return false;
    for (int i = 0; i < 3; ++i)
        if (!test(bx[i], false)) return false;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (!test(Cross(ax[i], bx[j]), true)) return false;
    if (!found) return false;

    if (Dot(d, best_axis) < 0.0f) best_axis = best_axis * -1.0f;
    *n_out = best_axis;
    // Recomputed rather than reusing `best`, which may carry the tie-break bias.
    *depth_out = BoxRadius(ax, a.shape.half_extents, best_axis) +
                 BoxRadius(bx, b.shape.half_extents, best_axis) -
                 std::fabs(Dot(d, best_axis));
    return true;
}

// Contact points for a box pair whose separating axis is already known: every
// corner of one box that lies inside the other, with its depth measured along
// the shared normal rather than per-corner.
//
// Per-corner normals would be wrong. A corner's own nearest face is not the
// pair's separating axis, so two boxes overlapping slightly on their sides
// would each get pushed along whichever face that corner happened to be near.
int BoxBoxPoints(const Body& a, const Body& b, Vec3 n, Contact* out, int max_out) {
    Vec3 ax[3], bx[3];
    BoxAxes(a, ax);
    BoxAxes(b, bx);
    const float a_far = Dot(a.position, n) + BoxRadius(ax, a.shape.half_extents, n);
    const float b_near = Dot(b.position, n) - BoxRadius(bx, b.shape.half_extents, n);

    int n_out = 0;
    Vec3 corners[8];

    BoxCorners(b, corners);
    for (const Vec3& p : corners) {
        if (n_out >= max_out) break;
        if (!PointInsideBox(a, p)) continue;
        out[n_out].normal = n;
        out[n_out].point = p;
        out[n_out].depth = a_far - Dot(p, n);
        ++n_out;
    }
    BoxCorners(a, corners);
    for (const Vec3& p : corners) {
        if (n_out >= max_out) break;
        if (!PointInsideBox(b, p)) continue;
        out[n_out].normal = n;
        out[n_out].point = p;
        out[n_out].depth = Dot(p, n) - b_near;
        ++n_out;
    }
    return n_out;
}

}  // namespace

Shape Shape::MakeHull(const geom::Hull& hull) {
    Shape s;
    s.type = ShapeType::Hull;
    if (hull.Empty()) return s;

    const Vec3 centre = hull.Centroid();
    s.centre_offset = centre;
    s.points.reserve(hull.vertices.size());
    for (const Vec3& p : hull.vertices) s.points.push_back(p - centre);

    float r = 0.0f;
    Vec3 half{0, 0, 0};
    for (const Vec3& p : s.points) {
        r = std::max(r, Length(p));
        half.x = std::max(half.x, std::fabs(p.x));
        half.y = std::max(half.y, std::fabs(p.y));
        half.z = std::max(half.z, std::fabs(p.z));
    }
    s.radius = r;
    s.half_extents = half;

    // Per unit MASS, so that SetMass can scale it. Hull::Inertia() is for unit
    // DENSITY, and dividing by the volume is what converts between the two --
    // without it a small dense object and a large light one of the same mass
    // would spin identically.
    const float vol = hull.Volume();
    if (vol > 1e-9f) {
        const Vec3 d = hull.Inertia().Diagonal();
        s.unit_inertia = d * (1.0f / vol);
    }
    return s;
}

void Body::SetMass(float mass) {
    if (mass <= 0.0f) {
        inverse_mass = 0.0f;
        inverse_inertia = Vec3{0.0f, 0.0f, 0.0f};
        return;
    }
    inverse_mass = 1.0f / mass;
    if (shape.type == ShapeType::Hull && shape.unit_inertia.x >= 0.0f) {
        // The real tensor's diagonal, from the hull's own geometry. The
        // off-diagonal terms are dropped: they are zero for any hull whose own
        // axes are its principal axes, and small for anything roughly
        // symmetric. A deliberately skewed hull will precess slightly wrong.
        const Vec3 i = shape.unit_inertia * mass;
        inverse_inertia = Vec3{i.x > 0.0f ? 1.0f / i.x : 0.0f,
                               i.y > 0.0f ? 1.0f / i.y : 0.0f,
                               i.z > 0.0f ? 1.0f / i.z : 0.0f};
    } else if (shape.type == ShapeType::Sphere) {
        // Solid sphere: I = 2/5 m r², the same about every axis.
        const float i = 0.4f * mass * shape.radius * shape.radius;
        const float inv = i > 0.0f ? 1.0f / i : 0.0f;
        inverse_inertia = Vec3{inv, inv, inv};
    } else {
        // Solid box, half-extents h: I_x = m(h_y² + h_z²)/3. Diagonal in the
        // body's own frame because the box's axes ARE its principal axes.
        const Vec3 h = shape.half_extents;
        const Vec3 i{mass * (h.y * h.y + h.z * h.z) / 3.0f,
                     mass * (h.x * h.x + h.z * h.z) / 3.0f,
                     mass * (h.x * h.x + h.y * h.y) / 3.0f};
        inverse_inertia = Vec3{i.x > 0.0f ? 1.0f / i.x : 0.0f,
                               i.y > 0.0f ? 1.0f / i.y : 0.0f,
                               i.z > 0.0f ? 1.0f / i.z : 0.0f};
    }
}

Vec3 Body::ApplyInverseInertia(Vec3 v) const {
    // R · diag · Rᵀ · v. Rotating into the body frame, scaling by the diagonal,
    // and rotating back is the same thing as the full tensor product, without
    // building a 3x3 or keeping it in sync as the body turns.
    return Rotate(orientation, Mul(inverse_inertia, RotateInverse(orientation, v)));
}

bool CollideSphereSphere(const Body& a, const Body& b, Contact* out) {
    const Vec3 d = b.position - a.position;
    const float r = a.shape.radius + b.shape.radius;
    const float dist2 = Dot(d, d);
    if (dist2 >= r * r) return false;

    const float dist = std::sqrt(dist2);
    // Exactly concentric: any direction is as good as another, and normalising
    // a zero vector would hand the solver a NaN it never recovers from.
    out->normal = dist > 1e-6f ? d * (1.0f / dist) : Vec3{0.0f, 1.0f, 0.0f};
    out->depth = r - dist;
    // Midway into the overlap, so both surfaces are represented.
    out->point = a.position + out->normal * (a.shape.radius - out->depth * 0.5f);
    return true;
}

bool CollideSphereBox(const Body& sphere, const Body& box, Contact* out) {
    // Work in the BOX's frame: an oriented box is an axis-aligned one seen from
    // the right place, so the whole test stays the cheap clamp it always was.
    const Vec3 h = box.shape.half_extents;
    const Vec3 local = RotateInverse(box.orientation, sphere.position - box.position);
    const Vec3 closest = Clamp(local, h * -1.0f, h);
    const Vec3 d = closest - local;
    const float dist2 = Dot(d, d);
    const float r = sphere.shape.radius;

    if (dist2 > r * r) return false;

    Vec3 normal_local, point_local;
    if (dist2 > 1e-12f) {
        // Centre is outside the box: the contact normal is the direction to the
        // closest surface point.
        const float dist = std::sqrt(dist2);
        normal_local = d * (1.0f / dist);
        out->depth = r - dist;
        point_local = closest;
    } else {
        // Centre is INSIDE the box. There is no direction to the closest point,
        // so push out along whichever face is nearest — the shallowest escape.
        int best_axis = 0;
        float best = h.x - std::fabs(local.x);
        for (int i = 1; i < 3; ++i) {
            const float e = Component(h, i) - std::fabs(Component(local, i));
            if (e < best) {
                best = e;
                best_axis = i;
            }
        }
        const float sign = Component(local, best_axis) < 0.0f ? -1.0f : 1.0f;
        const Vec3 out_axis = Axis(best_axis, sign);
        // Normal points from the sphere toward the box, matching the outside
        // case, so it is the INWARD direction here.
        normal_local = out_axis * -1.0f;
        out->depth = r + best;
        point_local = local + out_axis * best;
    }
    out->normal = Rotate(box.orientation, normal_local);
    out->point = box.position + Rotate(box.orientation, point_local);
    return true;
}

bool CollideBoxBox(const Body& a, const Body& b, Contact* out) {
    Vec3 n;
    float depth;
    if (!BoxBoxSat(a, b, &n, &depth)) return false;

    Contact points[16];
    const int count = BoxBoxPoints(a, b, n, points, 16);
    out->normal = n;
    out->depth = depth;
    if (count == 0) {
        // Edge crossing, with no corner of either box inside the other — two
        // beams making an X. Put the single contact at the point of b that
        // reaches deepest into a.
        Vec3 bx[3];
        BoxAxes(b, bx);
        const Vec3 h = b.shape.half_extents;
        out->point = b.position -
                     Rotate(b.orientation,
                            Vec3{Dot(bx[0], n) > 0.0f ? h.x : -h.x,
                                 Dot(bx[1], n) > 0.0f ? h.y : -h.y,
                                 Dot(bx[2], n) > 0.0f ? h.z : -h.z});
        return true;
    }
    // Deepest of the manifold, for callers that want one representative point.
    int best = 0;
    for (int i = 1; i < count; ++i)
        if (points[i].depth > points[best].depth) best = i;
    out->point = points[best].point;
    return true;
}

int World::AddJoint(const Joint& j) {
    joints_.push_back(j);
    return int(joints_.size()) - 1;
}

void World::Wake(int i) {
    if (i < 0 || std::size_t(i) >= bodies_.size()) return;
    bodies_[std::size_t(i)].sleeping = false;
    bodies_[std::size_t(i)].still_for = 0.0f;
}

void World::WakeAll() {
    for (Body& b : bodies_) {
        b.sleeping = false;
        b.still_for = 0.0f;
    }
}

int World::SleepingCount() const {
    int n = 0;
    for (const Body& b : bodies_)
        if (b.sleeping) ++n;
    return n;
}

int World::Add(const Body& b) {
    bodies_.push_back(b);
    Body& added = bodies_.back();
    // A dynamic body whose inertia was never filled in would have every axis
    // locked and slide around without ever turning — the failure this whole
    // pass exists to remove. Derive it from the shape rather than let that
    // happen silently. Locking an axis deliberately means zeroing it after Add.
    if (!added.IsStatic() && Dot(added.inverse_inertia, added.inverse_inertia) == 0.0f)
        added.SetMass(1.0f / added.inverse_mass);
    return int(bodies_.size()) - 1;
}

void World::Clear() {
    bodies_.clear();
    joints_.clear();
    contacts_.clear();
    stats_ = WorldStats{};
    accumulator_ = 0.0f;
}

float World::Energy() const {
    float e = 0.0f;
    for (const Body& b : bodies_) {
        if (b.IsStatic()) continue;  // static bodies hold no energy
        const float mass = 1.0f / b.inverse_mass;
        e += 0.5f * mass * Dot(b.velocity, b.velocity);
        e += mass * (-gravity.y) * b.position.y;
        // Rotational: ½ ω·(Iω), with I the inverse of the inverse. An axis with
        // zero inverse inertia is locked, so ω about it is zero and it carries
        // no energy — skipping it is exact, not an approximation.
        const Vec3 wl = RotateInverse(b.orientation, b.angular_velocity);
        const Vec3 ii = b.inverse_inertia;
        if (ii.x > 0.0f) e += 0.5f * wl.x * wl.x / ii.x;
        if (ii.y > 0.0f) e += 0.5f * wl.y * wl.y / ii.y;
        if (ii.z > 0.0f) e += 0.5f * wl.z * wl.z / ii.z;
    }
    return e;
}

Vec3 World::AngularMomentum() const {
    Vec3 l{0.0f, 0.0f, 0.0f};
    for (const Body& b : bodies_) {
        if (b.IsStatic()) continue;
        const float mass = 1.0f / b.inverse_mass;
        // Orbital part, about the origin.
        l = l + Cross(b.position, b.velocity) * mass;
        // Spin part: I ω, built the same way ApplyInverseInertia builds I⁻¹ v.
        const Vec3 wl = RotateInverse(b.orientation, b.angular_velocity);
        const Vec3 ii = b.inverse_inertia;
        const Vec3 il{ii.x > 0.0f ? wl.x / ii.x : 0.0f,
                      ii.y > 0.0f ? wl.y / ii.y : 0.0f,
                      ii.z > 0.0f ? wl.z / ii.z : 0.0f};
        l = l + Rotate(b.orientation, il);
    }
    return l;
}

// ---------------------------------------------------------------- GJK/EPA ---

Vec3 Support(const Body& body, Vec3 dir) {
    switch (body.shape.type) {
        case ShapeType::Sphere: {
            const float len = Length(dir);
            // A sphere's support is its centre plus r in the direction asked
            // for. Zero direction happens on the very first GJK iteration when
            // two centres coincide, and any unit vector is a correct answer.
            const Vec3 unit = len > 1e-12f ? dir * (1.0f / len) : Vec3{1, 0, 0};
            return body.position + unit * body.shape.radius;
        }
        case ShapeType::Box: {
            // In the body's frame the answer is a corner, chosen per axis by
            // the sign of the direction.
            const Vec3 local = RotateInverse(body.orientation, dir);
            const Vec3 h = body.shape.half_extents;
            const Vec3 corner{local.x >= 0.0f ? h.x : -h.x,
                              local.y >= 0.0f ? h.y : -h.y,
                              local.z >= 0.0f ? h.z : -h.z};
            return body.position + Rotate(body.orientation, corner);
        }
        case ShapeType::Hull: {
            if (body.shape.points.empty()) return body.position;
            const Vec3 local = RotateInverse(body.orientation, dir);
            const Vec3* best = &body.shape.points[0];
            float best_dot = Dot(local, *best);
            for (const Vec3& p : body.shape.points) {
                const float d = Dot(local, p);
                if (d > best_dot) { best_dot = d; best = &p; }
            }
            return body.position + Rotate(body.orientation, *best);
        }
    }
    return body.position;
}

namespace {

// A point of the Minkowski difference, with the two points it came from. The
// witnesses are what turn a penetration depth into a contact POINT: without
// them EPA gives a direction and a distance and no place to apply the impulse.
struct SupportPoint {
    Vec3 p;  // pa - pb
    Vec3 pa, pb;
};

SupportPoint MinkowskiSupport(const Body& a, const Body& b, Vec3 dir) {
    SupportPoint s;
    s.pa = Support(a, dir);
    s.pb = Support(b, dir * -1.0f);
    s.p = s.pa - s.pb;
    return s;
}

// Triple product, the "vector perpendicular to ab in the direction of ac" that
// every GJK simplex case is written in terms of.
Vec3 TripleCross(Vec3 a, Vec3 b, Vec3 c) { return Cross(Cross(a, b), c); }

// One GJK iteration on the current simplex: reduces it to the feature closest
// to the origin and points `dir` at the origin from it. Returns true when the
// simplex is a tetrahedron containing the origin.
bool DoSimplex(SupportPoint* s, int& n, Vec3& dir) {
    const Vec3 a = s[n - 1].p;         // the point just added
    const Vec3 ao = a * -1.0f;         // toward the origin

    if (n == 2) {
        const Vec3 ab = s[0].p - a;
        // Perpendicular to ab, in the plane containing the origin. Degenerate
        // when the origin is ON the segment, which is a touching contact.
        dir = TripleCross(ab, ao, ab);
        if (Dot(dir, dir) < 1e-12f) dir = Cross(ab, Vec3{1, 0, 0});
        if (Dot(dir, dir) < 1e-12f) dir = Cross(ab, Vec3{0, 1, 0});
        return false;
    }

    if (n == 3) {
        const Vec3 b = s[1].p, c = s[0].p;
        const Vec3 ab = b - a, ac = c - a;
        const Vec3 abc = Cross(ab, ac);
        if (Dot(Cross(abc, ac), ao) > 0.0f) {
            // Outside edge ac: drop b.
            s[1] = s[2];
            n = 2;
            dir = TripleCross(ac, ao, ac);
        } else if (Dot(Cross(ab, abc), ao) > 0.0f) {
            // Outside edge ab: drop c.
            s[0] = s[1];
            s[1] = s[2];
            n = 2;
            dir = TripleCross(ab, ao, ab);
        } else {
            // Above or below the triangle.
            dir = Dot(abc, ao) > 0.0f ? abc : abc * -1.0f;
            if (Dot(abc, ao) <= 0.0f) std::swap(s[0], s[1]);
        }
        return false;
    }

    // n == 4: the origin is inside unless it is outside one of the three faces
    // that touch the newest point.
    const Vec3 b = s[2].p, c = s[1].p, d = s[0].p;
    const Vec3 abc = Cross(b - a, c - a);
    const Vec3 acd = Cross(c - a, d - a);
    const Vec3 adb = Cross(d - a, b - a);
    if (Dot(abc, ao) > 0.0f) {
        s[0] = s[1]; s[1] = s[2]; s[2] = s[3];
        n = 3;
        dir = abc;
        return false;
    }
    if (Dot(acd, ao) > 0.0f) {
        s[2] = s[3];
        n = 3;
        dir = acd;
        return false;
    }
    if (Dot(adb, ao) > 0.0f) {
        s[1] = s[0]; s[0] = s[2]; s[2] = s[3];
        n = 3;
        dir = adb;
        return false;
    }
    return true;
}

struct EpaFace {
    int a, b, c;
    Vec3 normal;
    float dist;
};

// Barycentric coordinates of the projection of the origin onto triangle pqr.
// Used to carry the witness points along: the contact point is the same blend
// of the A-side witnesses that the origin's projection is of the triangle.
void Barycentric(Vec3 p, Vec3 q, Vec3 r, float* u, float* v, float* w) {
    // origin - p. The origin is not in the triangle's plane, but only the
    // in-plane part survives the dots with v0 and v1, so this gives the
    // barycentric coordinates of its PROJECTION -- which is what is wanted.
    const Vec3 v0 = q - p, v1 = r - p, v2 = p * -1.0f;
    const float d00 = Dot(v0, v0), d01 = Dot(v0, v1), d11 = Dot(v1, v1);
    const float d20 = Dot(v2, v0), d21 = Dot(v2, v1);
    const float denom = d00 * d11 - d01 * d01;
    if (std::fabs(denom) < 1e-20f) { *u = 1.0f; *v = *w = 0.0f; return; }
    *v = (d11 * d20 - d01 * d21) / denom;
    *w = (d00 * d21 - d01 * d20) / denom;
    *u = 1.0f - *v - *w;
}

}  // namespace

bool CollideConvex(const Body& a, const Body& b, Contact* out) {
    // --- GJK ----------------------------------------------------------------
    Vec3 dir = b.position - a.position;
    if (Dot(dir, dir) < 1e-12f) dir = Vec3{1, 0, 0};

    SupportPoint simplex[4];
    simplex[0] = MinkowskiSupport(a, b, dir);
    int n = 1;
    dir = simplex[0].p * -1.0f;

    // Bounded, not while(true). A support function that returns the same point
    // twice -- which floating point makes possible on a flat face -- would spin
    // forever, and a physics step that never returns is worse than a missed
    // contact.
    for (int iter = 0; iter < 64; ++iter) {
        if (Dot(dir, dir) < 1e-24f) break;
        const SupportPoint s = MinkowskiSupport(a, b, dir);
        // The new point did not pass the origin, so the origin is outside the
        // Minkowski difference and the shapes are apart.
        if (Dot(s.p, dir) < 0.0f) return false;
        simplex[n++] = s;
        if (DoSimplex(simplex, n, dir)) {
            // --- EPA --------------------------------------------------------
            std::vector<SupportPoint> poly(simplex, simplex + 4);
            std::vector<EpaFace> faces;
            // Winding must be CONSISTENT across the whole polytope, not merely
            // outward on each face taken alone. The horizon walk below
            // identifies interior edges by finding the same edge traversed in
            // opposite directions by two deleted faces; flipping a face to make
            // its normal point away from the origin breaks that pairing, and
            // the polytope silently develops holes. With holes, "the closest
            // face" is chosen from an incomplete set and EPA returns a plane
            // that cuts through the shape -- a normal a few degrees off and a
            // depth a little short, which is exactly the wrong kind of wrong:
            // small enough to look like tolerance, large enough to matter.
            const auto add_face = [&](int i, int j, int k) {
                EpaFace f{i, j, k, {}, 0.0f};
                const Vec3 nrm = Cross(poly[std::size_t(j)].p - poly[std::size_t(i)].p,
                                       poly[std::size_t(k)].p - poly[std::size_t(i)].p);
                const float len = Length(nrm);
                if (len < 1e-12f) return;  // sliver: contributes no plane
                f.normal = nrm * (1.0f / len);
                f.dist = Dot(f.normal, poly[std::size_t(i)].p);
                faces.push_back(f);
            };
            // The seed tetrahedron, wound outward by testing the OPPOSITE
            // vertex rather than the origin. Same rule the hull builder uses,
            // and unlike the origin test it stays correct when the origin sits
            // exactly on a face -- which is a touching contact, not a rarity.
            const auto seed = [&](int i, int j, int k, int opposite) {
                const Vec3 nrm = Cross(poly[std::size_t(j)].p - poly[std::size_t(i)].p,
                                       poly[std::size_t(k)].p - poly[std::size_t(i)].p);
                if (Dot(nrm, poly[std::size_t(opposite)].p - poly[std::size_t(i)].p) > 0.0f)
                    add_face(i, k, j);
                else
                    add_face(i, j, k);
            };
            seed(0, 1, 2, 3);
            seed(0, 3, 1, 2);
            seed(0, 2, 3, 1);
            seed(1, 3, 2, 0);
            if (faces.size() < 4) return false;  // degenerate simplex

            for (int step = 0; step < 64; ++step) {
                // The face closest to the origin is the current best guess at
                // the penetration.
                std::size_t best = 0;
                for (std::size_t f = 1; f < faces.size(); ++f)
                    if (faces[f].dist < faces[best].dist) best = f;
                const EpaFace closest = faces[best];

                const SupportPoint s2 = MinkowskiSupport(a, b, closest.normal);
                const float reach = Dot(s2.p, closest.normal);
                if (reach - closest.dist < 1e-4f || step == 63) {
                    // Converged. The contact point is the origin's projection
                    // onto this face, expressed through the A-side witnesses.
                    float u, v, w;
                    Barycentric(poly[std::size_t(closest.a)].p,
                                poly[std::size_t(closest.b)].p,
                                poly[std::size_t(closest.c)].p, &u, &v, &w);
                    const Vec3 pa = poly[std::size_t(closest.a)].pa * u +
                                    poly[std::size_t(closest.b)].pa * v +
                                    poly[std::size_t(closest.c)].pa * w;
                    const Vec3 pb = poly[std::size_t(closest.a)].pb * u +
                                    poly[std::size_t(closest.b)].pb * v +
                                    poly[std::size_t(closest.c)].pb * w;
                    out->normal = closest.normal;
                    out->depth = closest.dist;
                    // Halfway between the witnesses: they are the same point
                    // when the shapes just touch, and straddle the overlap when
                    // they do not.
                    out->point = (pa + pb) * 0.5f;
                    return closest.dist > 0.0f;
                }

                // Expand: delete every face the new point can see, and close
                // the hole with a fan from the horizon. Same cancellation rule
                // as the hull builder -- an edge shared by two deleted faces is
                // interior, not horizon.
                const int added = int(poly.size());
                poly.push_back(s2);
                std::vector<std::pair<int, int>> horizon;
                std::vector<EpaFace> kept;
                for (const EpaFace& f : faces) {
                    if (Dot(f.normal, s2.p) - f.dist <= 0.0f) {
                        kept.push_back(f);
                        continue;
                    }
                    const int tri[3][2] = {{f.a, f.b}, {f.b, f.c}, {f.c, f.a}};
                    for (const auto& e : tri) {
                        bool cancelled = false;
                        for (std::size_t q = 0; q < horizon.size(); ++q)
                            if (horizon[q].first == e[1] && horizon[q].second == e[0]) {
                                horizon.erase(horizon.begin() + std::ptrdiff_t(q));
                                cancelled = true;
                                break;
                            }
                        if (!cancelled) horizon.emplace_back(e[0], e[1]);
                    }
                }
                faces.swap(kept);
                for (const auto& e : horizon) add_face(e.first, e.second, added);
                if (faces.empty()) return false;
                // A closed polytope on V vertices has 2V-4 faces, and EPA adds
                // at most one vertex per step, so this cannot be reached by a
                // correct expansion. It IS reached the moment the horizon walk
                // leaks: every step then triples the face count, and what would
                // otherwise be a wrong answer becomes an out-of-memory a few
                // seconds later, in a function with no obvious connection to
                // the cause. Failing here keeps it a collision result.
                if (faces.size() > 4 * (poly.size() + 2)) return false;
            }
            return false;
        }
    }
    return false;
}

void World::Collide() {
    contacts_.clear();
    stats_.pairs_tested = 0;

    // Brute force. At a few hundred bodies this is faster than any acceleration
    // structure that has to be rebuilt every step; past that it is the first
    // thing to replace.
    for (int i = 0; i < int(bodies_.size()); ++i) {
        for (int j = i + 1; j < int(bodies_.size()); ++j) {
            const Body& a = bodies_[std::size_t(i)];
            const Body& b = bodies_[std::size_t(j)];
            // Nothing that could move: two static bodies, two sleeping ones, or
            // one of each. A contact between them has nothing to resolve, and
            // skipping the narrowphase here is what makes sleeping cost less
            // rather than merely look calmer — without it the broadphase still
            // tests every pair in a settled scene and dominates the step.
            //
            // Safe because waking is contact-driven from the OTHER side: a
            // moving body still generates a contact against a sleeping one, and
            // that is what wakes it.
            const bool a_inert = a.IsStatic() || a.sleeping;
            const bool b_inert = b.IsStatic() || b.sleeping;
            if (a_inert && b_inert) continue;
            ++stats_.pairs_tested;

            Contact c;
            // A HULL first, before anything else. The sphere branches below
            // hand their second argument to CollideSphereBox, which would
            // silently collide against the hull's bounding box -- a shape that
            // is always bigger and never right.
            if (a.shape.type == ShapeType::Hull || b.shape.type == ShapeType::Hull) {
                // GJK/EPA, which needs no case analysis: it asks only for
                // support points, so one path covers hull against hull, box or
                // sphere.
                if (CollideConvex(a, b, &c)) {
                    c.a = i;
                    c.b = j;
                    contacts_.push_back(c);
                }
            } else if (a.shape.type == ShapeType::Sphere &&
                       b.shape.type == ShapeType::Sphere) {
                if (CollideSphereSphere(a, b, &c)) {
                    c.a = i;
                    c.b = j;
                    contacts_.push_back(c);
                }
            } else if (a.shape.type == ShapeType::Sphere) {
                if (CollideSphereBox(a, b, &c)) {
                    c.a = i;
                    c.b = j;
                    contacts_.push_back(c);
                }
            } else if (b.shape.type == ShapeType::Sphere) {
                // Swap so the sphere is always `a`; the normal comes back in
                // that pair's orientation and c.a/c.b are labelled to match.
                if (CollideSphereBox(b, a, &c)) {
                    c.a = j;
                    c.b = i;
                    contacts_.push_back(c);
                }
            } else {
                // MULTIPLE contacts for a box pair, one per penetrating corner.
                // A single deepest point is enough to stop interpenetration but
                // not to hold a box still: one contact is a pivot, so a crate
                // resting flat rocks forever. Four corners is a manifold.
                Vec3 n;
                float depth;
                if (!BoxBoxSat(a, b, &n, &depth)) continue;
                Contact points[16];
                const int count = BoxBoxPoints(a, b, n, points, 16);
                if (count == 0) {
                    // Edge crossing with no corner inside either box. Falls back
                    // to the single representative contact.
                    if (CollideBoxBox(a, b, &c)) {
                        c.a = i;
                        c.b = j;
                        contacts_.push_back(c);
                    }
                    continue;
                }
                for (int k = 0; k < count; ++k) {
                    points[k].a = i;
                    points[k].b = j;
                    contacts_.push_back(points[k]);
                }
            }
        }
    }
    stats_.contacts = int(contacts_.size());
}

void World::Resolve() {
    // Sequential impulses: walk the contact list repeatedly, each pass nudging
    // the velocities closer to satisfying every constraint at once. Cheap, and
    // it degrades into "slightly soft" rather than exploding when it runs out
    // of iterations.
    for (int iter = 0; iter < solver_iterations; ++iter) {
        for (const Contact& c : contacts_) {
            Body& a = bodies_[std::size_t(c.a)];
            Body& b = bodies_[std::size_t(c.b)];
            const Vec3 ra = c.point - a.position;
            const Vec3 rb = c.point - b.position;

            const Vec3 rel = b.PointVelocity(c.point) - a.PointVelocity(c.point);
            const float vn = Dot(rel, c.normal);
            // Already separating: an impulse here would suck them together.
            if (vn > 0.0f) continue;

            // Effective mass along the normal. The two cross-product terms are
            // the angular share: a contact far from the centre of mass is
            // "heavier" because part of the impulse goes into spin.
            const float kn =
                a.inverse_mass + b.inverse_mass +
                Dot(c.normal, Cross(a.ApplyInverseInertia(Cross(ra, c.normal)), ra)) +
                Dot(c.normal, Cross(b.ApplyInverseInertia(Cross(rb, c.normal)), rb));
            if (kn <= 0.0f) continue;

            // Restitution is switched off below a threshold. A body at rest is
            // in contact every step with a tiny approach velocity from gravity;
            // bouncing that back is what makes a settled pile hum.
            const float e = -vn < restitution_threshold
                                ? 0.0f
                                : std::min(a.restitution, b.restitution);
            const float jn = -(1.0f + e) * vn / kn;
            if (jn <= 0.0f) continue;
            const Vec3 impulse = c.normal * jn;
            ApplyImpulse(a, impulse * -1.0f, ra);
            ApplyImpulse(b, impulse, rb);

            // Coulomb friction, on the tangential motion of the contact POINT.
            // Applying it there rather than at the centre is the whole
            // difference between a ball that rolls and a ball that slides: the
            // same impulse that stops the surface sliding also spins the body.
            const Vec3 rel2 = b.PointVelocity(c.point) - a.PointVelocity(c.point);
            Vec3 tangent = rel2 - c.normal * Dot(rel2, c.normal);
            const float tlen = Length(tangent);
            if (tlen <= 1e-6f) continue;
            tangent = tangent * (1.0f / tlen);

            const float kt =
                a.inverse_mass + b.inverse_mass +
                Dot(tangent, Cross(a.ApplyInverseInertia(Cross(ra, tangent)), ra)) +
                Dot(tangent, Cross(b.ApplyInverseInertia(Cross(rb, tangent)), rb));
            if (kt <= 0.0f) continue;

            const float mu = std::sqrt(a.friction * b.friction);
            float jt = -Dot(rel2, tangent) / kt;
            jt = std::clamp(jt, -mu * jn, mu * jn);
            const Vec3 fimp = tangent * jt;
            ApplyImpulse(a, fimp * -1.0f, ra);
            ApplyImpulse(b, fimp, rb);
        }
    }

    // Positional correction, separate from the velocity solve. Impulses alone
    // cannot undo an overlap that already exists, and letting it persist makes
    // bodies sink through each other over time.
    //
    // Linear only, deliberately: rotating a body to fix penetration adds energy
    // the solver never accounted for and makes stacks squirm.
    //
    // DIVIDED BY THE NUMBER OF CONTACTS ON EACH BODY. A box resting on the floor
    // produces one contact per corner, and pushing the full correction at every
    // one of them moves it four times as far as the penetration it is fixing —
    // measured at 0.312 of lift for 0.2 of overlap, so the box is ejected
    // upward and lands again next step. A sphere has a single contact and never
    // showed it.
    touches_.assign(bodies_.size(), 0);
    for (const Contact& c : contacts_) {
        ++touches_[std::size_t(c.a)];
        ++touches_[std::size_t(c.b)];
    }
    for (const Contact& c : contacts_) {
        Body& a = bodies_[std::size_t(c.a)];
        Body& b = bodies_[std::size_t(c.b)];
        const float inv_sum = a.inverse_mass + b.inverse_mass;
        if (inv_sum <= 0.0f) continue;
        const float excess = std::max(c.depth - penetration_slop, 0.0f);
        const Vec3 push = c.normal * (excess * penetration_correction / inv_sum);
        if (a.sleeping && b.sleeping) continue;
        const float share_a = 1.0f / float(std::max(touches_[std::size_t(c.a)], 1));
        const float share_b = 1.0f / float(std::max(touches_[std::size_t(c.b)], 1));
        a.position = a.position - push * (a.inverse_mass * share_a);
        b.position = b.position + push * (b.inverse_mass * share_b);
    }
}

void World::SolveJoints() {
    // Same shape as a contact: find how far the constraint is violated along a
    // direction, work out the effective mass along it, and apply an impulse.
    // A joint differs only in being two-sided — it pulls as readily as it
    // pushes, where a contact can only push.
    for (int iter = 0; iter < solver_iterations; ++iter) {
        for (const Joint& j : joints_) {
            if (j.a < 0 || j.b < 0 || std::size_t(j.a) >= bodies_.size() ||
                std::size_t(j.b) >= bodies_.size())
                continue;
            Body& a = bodies_[std::size_t(j.a)];
            Body& b = bodies_[std::size_t(j.b)];
            if (a.IsStatic() && b.IsStatic()) continue;

            const Vec3 ra = Rotate(a.orientation, j.anchor_a);
            const Vec3 rb = Rotate(b.orientation, j.anchor_b);
            const Vec3 pa = a.position + ra;
            const Vec3 pb = b.position + rb;
            const Vec3 d = pb - pa;
            const float len = Length(d);
            // Coincident anchors on a ball socket: any direction is as good as
            // another, and normalising zero hands the solver a NaN.
            const Vec3 n = len > 1e-6f ? d * (1.0f / len) : Vec3{0.0f, 1.0f, 0.0f};
            const float error = len - j.distance;
            // A rope only resists being stretched.
            if (j.rope && error <= 0.0f) continue;

            const float inv_sum =
                a.inverse_mass + b.inverse_mass +
                Dot(n, Cross(a.ApplyInverseInertia(Cross(ra, n)), ra)) +
                Dot(n, Cross(b.ApplyInverseInertia(Cross(rb, n)), rb));
            if (inv_sum <= 0.0f) continue;

            // Velocity first: cancel the relative motion ALONG the constraint,
            // so the joint stops being violated rather than being repeatedly
            // pulled back into place.
            const Vec3 rel = b.PointVelocity(pb) - a.PointVelocity(pa);
            const float along = Dot(rel, n);
            const Vec3 impulse = n * (-along / inv_sum);
            ApplyImpulse(a, impulse * -1.0f, ra);
            ApplyImpulse(b, impulse, rb);

            // Then position, at a fraction of the error. Correcting all of it
            // makes the joint rigid enough to fight the contact solver, and a
            // body held by both squirms.
            const Vec3 push = n * (error * j.stiffness / inv_sum);
            a.position = a.position + push * a.inverse_mass;
            b.position = b.position - push * b.inverse_mass;
        }
    }
}

void World::UpdateSleep() {
    if (sleep_after <= 0.0f) {
        for (Body& b : bodies_) b.sleeping = false;
        return;
    }

    // Every body's own stillness timer first.
    for (Body& b : bodies_) {
        if (b.IsStatic()) continue;
        const bool still = Length(b.velocity) < sleep_linear &&
                           Length(b.angular_velocity) < sleep_angular;
        b.still_for = still ? b.still_for + fixed_dt : 0.0f;
    }

    // ISLANDS. Bodies connected by contacts sleep together or not at all.
    //
    // The obvious rule — "do not sleep while touching something awake" — is a
    // deadlock. Two balls resting on each other are both awake and both still,
    // and each holds the other up forever, so a settled stack never stops being
    // simulated, which is the one thing sleeping exists to prevent. Refining it
    // to "touching something MOVING" fails the other way: contact resolution
    // has already cancelled the velocity of the body landing on the stack by
    // the time this runs, so the thing arriving looks stationary.
    //
    // Union-find over the contact graph, with static bodies left out — the
    // floor touches everything, and putting it in one island would mean the
    // whole scene sleeps together or never.
    const std::size_t n = bodies_.size();
    islands_.resize(n);
    for (std::size_t i = 0; i < n; ++i) islands_[i] = int(i);
    // Path-halving find: no recursion, and it flattens as it goes.
    auto find = [&](int x) {
        while (islands_[std::size_t(x)] != x) {
            islands_[std::size_t(x)] = islands_[std::size_t(islands_[std::size_t(x)])];
            x = islands_[std::size_t(x)];
        }
        return x;
    };
    for (const Contact& c : contacts_) {
        if (bodies_[std::size_t(c.a)].IsStatic() ||
            bodies_[std::size_t(c.b)].IsStatic())
            continue;
        const int ra = find(c.a), rb = find(c.b);
        if (ra != rb) islands_[std::size_t(ra)] = rb;
    }

    // An island sleeps when its LEAST settled member has been still long
    // enough. Waking one body therefore wakes everything it is resting on,
    // which is what Wake() being called on an arriving body has to mean.
    std::vector<float> island_min(n, 1e30f);
    for (std::size_t i = 0; i < n; ++i) {
        if (bodies_[i].IsStatic()) continue;
        const std::size_t root = std::size_t(find(int(i)));
        island_min[root] = std::min(island_min[root], bodies_[i].still_for);
    }
    for (std::size_t i = 0; i < n; ++i) {
        Body& b = bodies_[i];
        if (b.IsStatic()) continue;
        const std::size_t root = std::size_t(find(int(i)));
        if (island_min[root] >= sleep_after) {
            b.sleeping = true;
            // Zeroed rather than left as they are: a body that sleeps with a
            // millimetre per second on it wakes up having drifted.
            b.velocity = Vec3{0.0f, 0.0f, 0.0f};
            b.angular_velocity = Vec3{0.0f, 0.0f, 0.0f};
        } else {
            b.sleeping = false;
        }
    }
}

void World::StepFixed() {
    // Semi-implicit Euler: velocity first, then integrate position with the NEW
    // velocity. Explicit Euler (position from the old velocity) pumps energy
    // into every orbit and bounce, and no amount of solver tuning hides it.
    for (Body& b : bodies_) {
        if (b.IsStatic() || b.sleeping) continue;
        b.velocity = b.velocity + gravity * fixed_dt;
    }
    Collide();
    // A contact with a sleeping body wakes it. Without this a ball rolls into
    // a settled crate and passes through the place it should have hit — the
    // crate is still solid, but nothing ever gives it the impulse.
    for (const Contact& c : contacts_) {
        Body& a = bodies_[std::size_t(c.a)];
        Body& b = bodies_[std::size_t(c.b)];
        const bool a_moving = !a.sleeping && !a.IsStatic();
        const bool b_moving = !b.sleeping && !b.IsStatic();
        if (a_moving && b.sleeping) Wake(c.b);
        if (b_moving && a.sleeping) Wake(c.a);
    }
    Resolve();
    SolveJoints();
    // Damping AFTER the solve, so it never fights the contact impulses — a
    // body damped before resolution needs a bigger impulse to hold it up, and
    // the two chase each other into jitter.
    const float lin = 1.0f / (1.0f + linear_damping * fixed_dt);
    const float ang = 1.0f / (1.0f + angular_damping * fixed_dt);
    for (Body& b : bodies_) {
        if (b.IsStatic() || b.sleeping) continue;
        if (linear_damping > 0.0f) b.velocity = b.velocity * lin;
        if (angular_damping > 0.0f) b.angular_velocity = b.angular_velocity * ang;
        b.position = b.position + b.velocity * fixed_dt;

        // Orientation from angular velocity: q̇ = ½ ω q, integrated and
        // renormalised. The renormalise is not optional — the first-order step
        // leaves the quaternion slightly long every time, and the drift shows
        // up as a body that visibly grows or shears.
        const Vec3 w = b.angular_velocity;
        const Quat spin = Quat{w.x, w.y, w.z, 0.0f} * b.orientation;
        const float half = 0.5f * fixed_dt;
        b.orientation = Normalize(Quat{b.orientation.x + spin.x * half,
                                       b.orientation.y + spin.y * half,
                                       b.orientation.z + spin.z * half,
                                       b.orientation.w + spin.w * half});
    }
    UpdateSleep();
    ++stats_.steps;
}

int World::Step(float dt) {
    if (dt <= 0.0f) return 0;
    accumulator_ += dt;
    // Cap the catch-up. After a long stall — a breakpoint, a window drag —
    // running every missed step at once produces a burst of simulation that
    // looks like an explosion and can take longer than the stall did.
    constexpr int kMaxSteps = 8;
    int ran = 0;
    while (accumulator_ >= fixed_dt && ran < kMaxSteps) {
        StepFixed();
        accumulator_ -= fixed_dt;
        ++ran;
    }
    if (ran == kMaxSteps) accumulator_ = 0.0f;
    return ran;
}

}  // namespace eng::physics
