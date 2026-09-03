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

void Body::SetMass(float mass) {
    if (mass <= 0.0f) {
        inverse_mass = 0.0f;
        inverse_inertia = Vec3{0.0f, 0.0f, 0.0f};
        return;
    }
    inverse_mass = 1.0f / mass;
    if (shape.type == ShapeType::Sphere) {
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
            // Two static bodies can never move, so a contact between them has
            // nothing to resolve.
            if (a.IsStatic() && b.IsStatic()) continue;
            ++stats_.pairs_tested;

            Contact c;
            if (a.shape.type == ShapeType::Sphere && b.shape.type == ShapeType::Sphere) {
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
        const float share_a = 1.0f / float(std::max(touches_[std::size_t(c.a)], 1));
        const float share_b = 1.0f / float(std::max(touches_[std::size_t(c.b)], 1));
        a.position = a.position - push * (a.inverse_mass * share_a);
        b.position = b.position + push * (b.inverse_mass * share_b);
    }
}

void World::StepFixed() {
    // Semi-implicit Euler: velocity first, then integrate position with the NEW
    // velocity. Explicit Euler (position from the old velocity) pumps energy
    // into every orbit and bounce, and no amount of solver tuning hides it.
    for (Body& b : bodies_) {
        if (b.IsStatic()) continue;
        b.velocity = b.velocity + gravity * fixed_dt;
    }
    Collide();
    Resolve();
    // Damping AFTER the solve, so it never fights the contact impulses — a
    // body damped before resolution needs a bigger impulse to hold it up, and
    // the two chase each other into jitter.
    const float lin = 1.0f / (1.0f + linear_damping * fixed_dt);
    const float ang = 1.0f / (1.0f + angular_damping * fixed_dt);
    for (Body& b : bodies_) {
        if (b.IsStatic()) continue;
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
