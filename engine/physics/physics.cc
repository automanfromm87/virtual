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
    } else if (shape.type == ShapeType::Capsule) {
        // A cylinder plus two hemispheres, done as the cylinder alone.
        //
        // The caps contribute maybe a tenth of the inertia of a character-sized
        // capsule, and the exact formula is four terms with a parallel-axis
        // shift on each. This is an APPROXIMATION and the number it gets wrong
        // is how readily the thing tumbles -- which for the case a capsule
        // exists to serve, an upright character, is locked to zero anyway.
        const float r = shape.radius;
        const float h = shape.half_extents.y * 2.0f;
        const float ix = mass * (3.0f * r * r + h * h) / 12.0f;
        const float iy = 0.5f * mass * r * r;
        inverse_inertia = Vec3{ix > 0.0f ? 1.0f / ix : 0.0f,
                               iy > 0.0f ? 1.0f / iy : 0.0f,
                               ix > 0.0f ? 1.0f / ix : 0.0f};
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

float HeightfieldData::HeightAt(float x, float z) const {
    if (!Valid()) return 0.0f;
    const float fx = (x - origin.x) / spacing;
    const float fz = (z - origin.z) / spacing;
    const int ix = int(std::floor(fx));
    const int iz = int(std::floor(fz));
    const float tx = fx - float(ix);
    const float tz = fz - float(iz);
    const float h00 = At(ix, iz), h10 = At(ix + 1, iz);
    const float h01 = At(ix, iz + 1), h11 = At(ix + 1, iz + 1);
    const float top = h00 + (h10 - h00) * tx;
    const float bottom = h01 + (h11 - h01) * tx;
    return origin.y + top + (bottom - top) * tz;
}

Shape Shape::MakeHeightfield(std::shared_ptr<const HeightfieldData> data) {
    Shape s;
    s.type = ShapeType::Heightfield;
    s.heightfield = std::move(data);
    if (s.heightfield && s.heightfield->Valid()) {
        const HeightfieldData& h = *s.heightfield;
        float lo = 1e30f, hi = -1e30f;
        for (float v : h.heights) {
            lo = std::min(lo, v);
            hi = std::max(hi, v);
        }
        const float span = float(h.resolution - 1) * h.spacing;
        s.half_extents = Vec3{span * 0.5f, (hi - lo) * 0.5f + 0.01f, span * 0.5f};
        // The bounding sphere of the WHOLE terrain, which is enormous -- and
        // that is honest rather than a problem: every cheap reject that reads
        // it will pass, and the exact test below is the one that matters. A
        // small lie here would cull the ground out from under something.
        s.bounds_radius = Length(s.half_extents);
        // Positioned so the body's own position is the terrain's CENTRE, which
        // is what BodyBounds and the broadphase assume of every other shape.
        s.bounds_centre = Vec3{h.origin.x + span * 0.5f, h.origin.y + (lo + hi) * 0.5f,
                               h.origin.z + span * 0.5f};
    }
    return s;
}

namespace {

// The triangles of a height field overlapping a world-space box, as an array of
// nine floats each (three corners).
//
// The diagonal split ALTERNATES with the checker parity, matching
// Terrain::BuildChunk. If the two disagreed, a character would collide with a
// surface a few millimetres away from the one on screen -- which shows up as
// feet sinking into a slope in one direction and floating in the other.
int GatherHeightfieldTriangles(const HeightfieldData& h, const Aabb& box,
                               Vec3* out, int max_triangles) {
    if (!h.Valid() || max_triangles <= 0) return 0;
    const int last = h.resolution - 1;
    const int x0 = std::clamp(int(std::floor((box.lo.x - h.origin.x) / h.spacing)), 0, last - 1);
    const int x1 = std::clamp(int(std::ceil((box.hi.x - h.origin.x) / h.spacing)), 0, last);
    const int z0 = std::clamp(int(std::floor((box.lo.z - h.origin.z) / h.spacing)), 0, last - 1);
    const int z1 = std::clamp(int(std::ceil((box.hi.z - h.origin.z) / h.spacing)), 0, last);

    int count = 0;
    for (int z = z0; z < z1 && count + 1 < max_triangles; ++z)
        for (int x = x0; x < x1 && count + 1 < max_triangles; ++x) {
            const float wx = h.origin.x + float(x) * h.spacing;
            const float wz = h.origin.z + float(z) * h.spacing;
            const Vec3 a{wx, h.origin.y + h.At(x, z), wz};
            const Vec3 b{wx + h.spacing, h.origin.y + h.At(x + 1, z), wz};
            const Vec3 c{wx, h.origin.y + h.At(x, z + 1), wz + h.spacing};
            const Vec3 d{wx + h.spacing, h.origin.y + h.At(x + 1, z + 1), wz + h.spacing};
            // Skip a cell entirely below or above the query box: at a hundred
            // cells a step this is most of them.
            const float cell_lo = std::min({a.y, b.y, c.y, d.y});
            const float cell_hi = std::max({a.y, b.y, c.y, d.y});
            if (cell_hi < box.lo.y || cell_lo > box.hi.y) continue;
            if (((x + z) & 1) == 0) {
                out[count * 3 + 0] = a; out[count * 3 + 1] = c; out[count * 3 + 2] = b;
                ++count;
                out[count * 3 + 0] = b; out[count * 3 + 1] = c; out[count * 3 + 2] = d;
                ++count;
            } else {
                out[count * 3 + 0] = a; out[count * 3 + 1] = c; out[count * 3 + 2] = d;
                ++count;
                out[count * 3 + 0] = a; out[count * 3 + 1] = d; out[count * 3 + 2] = b;
                ++count;
            }
        }
    return count;
}

}  // namespace

Aabb BodyBounds(const Body& b, float margin) {
    Aabb box;
    switch (b.shape.type) {
        case ShapeType::Sphere:
            // Rotation-invariant, so the exact box is the trivial one. Not a
            // special case for speed -- the general path below would give the
            // same answer -- but because a sphere's half_extents field is
            // meaningless and reading it would be a bug waiting for someone to
            // set it.
            box.lo = b.position - Vec3{b.shape.radius, b.shape.radius, b.shape.radius};
            box.hi = b.position + Vec3{b.shape.radius, b.shape.radius, b.shape.radius};
            break;
        case ShapeType::Capsule: {
            // The two cap CENTRES, swept by the radius. Exact, and much tighter
            // than treating the capsule as a sphere of its bounding radius --
            // which for a 1.8 m character is a 1.8 m wide box instead of a
            // 0.7 m one, and every extra centimetre is broadphase pairs.
            const Vec3 axis = Rotate(b.orientation, Vec3{0.0f, b.shape.half_extents.y, 0.0f});
            const Vec3 a = b.position + axis, c = b.position - axis;
            box.Add(a);
            box.Add(c);
            box.Expand(b.shape.radius);
            break;
        }
        case ShapeType::Heightfield: {
            // The whole grid, in world space, from the extents cached at
            // construction -- rescanning a 129x129 field every frame to find
            // its own height range would be the most expensive thing in the
            // broadphase.
            //
            // A height field does not rotate, so this is exact rather than a
            // fitted box, and `bounds_centre` is where the grid sits relative
            // to the body: unlike every other shape, a terrain's geometry is
            // not centred on its own position.
            const Vec3 c = b.position + b.shape.bounds_centre;
            box.lo = c - b.shape.half_extents;
            box.hi = c + b.shape.half_extents;
            break;
        }
        case ShapeType::Box:
        case ShapeType::Hull: {
            // The rotated box's extent along each world axis is the row of the
            // absolute rotation matrix dotted with the half extents. Cheaper
            // than transforming eight corners and exactly as tight, and for a
            // hull it is the AABB of the hull's own box -- conservative, since
            // the hull is inside it, which is all a broadphase needs.
            const Vec3 h = b.shape.half_extents;
            const Vec3 rx = Rotate(b.orientation, Vec3{1.0f, 0.0f, 0.0f});
            const Vec3 ry = Rotate(b.orientation, Vec3{0.0f, 1.0f, 0.0f});
            const Vec3 rz = Rotate(b.orientation, Vec3{0.0f, 0.0f, 1.0f});
            const Vec3 e{
                std::fabs(rx.x) * h.x + std::fabs(ry.x) * h.y + std::fabs(rz.x) * h.z,
                std::fabs(rx.y) * h.x + std::fabs(ry.y) * h.y + std::fabs(rz.y) * h.z,
                std::fabs(rx.z) * h.x + std::fabs(ry.z) * h.y + std::fabs(rz.z) * h.z};
            box.lo = b.position - e;
            box.hi = b.position + e;
            break;
        }
    }
    if (margin > 0.0f) box.Expand(margin);
    return box;
}

void World::RefreshBroadphase() const {
    if (!bvh_dirty_) return;
    body_boxes_.resize(bodies_.size());
    for (std::size_t i = 0; i < bodies_.size(); ++i) {
        const Body& b = bodies_[i];
        // A body's box is grown by the distance it can travel in one step, so
        // the tree built before the integrator runs is still valid after it.
        // Static and sleeping bodies get no motion margin because they have no
        // motion, and a static floor is the body most worth keeping tight.
        const float motion = (b.IsStatic() || b.sleeping)
                                 ? 0.0f
                                 : Length(b.velocity) * fixed_dt;
        body_boxes_[i] = BodyBounds(b, broadphase_margin_ + motion);
    }
    bvh_.Build(body_boxes_);
    bvh_dirty_ = false;
    ++stats_.bvh_rebuilds;
    stats_.bvh_nodes = bvh_.NodeCount();
    stats_.bvh_depth = bvh_.MaxDepth();
}

void World::Wake(int i) {
    if (i < 0 || std::size_t(i) >= bodies_.size()) return;
    bodies_[std::size_t(i)].sleeping = false;
    bodies_[std::size_t(i)].still_for = 0.0f;
    // Waking changes the motion margin the tree was built with -- a sleeping
    // body got none -- so the box it is in is now too tight for where it is
    // about to go.
    bvh_dirty_ = true;
}

void World::WakeAll() {
    bvh_dirty_ = true;
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
    bvh_dirty_ = true;
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
    bvh_.Clear();
    body_boxes_.clear();
    bvh_dirty_ = true;
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
        case ShapeType::Capsule: {
            const float len = Length(dir);
            const Vec3 unit = len > 1e-12f ? dir * (1.0f / len) : Vec3{1, 0, 0};
            // The segment's furthest END, then the cap's radius along the
            // direction. A capsule IS a swept sphere, and its support function
            // is the segment's support plus the sphere's -- which is why one
            // line covers it and a box needs three.
            const Vec3 local = RotateInverse(body.orientation, dir);
            const float h = local.y >= 0.0f ? body.shape.half_extents.y
                                            : -body.shape.half_extents.y;
            return body.position + Rotate(body.orientation, Vec3{0.0f, h, 0.0f}) +
                   unit * body.shape.radius;
        }
        case ShapeType::Heightfield:
            // NEVER REACHED, and the case is here so the compiler says so if
            // that stops being true. A height field is not convex, so it is
            // never a GJK operand -- CollideHeightfield hands the algorithm one
            // triangle at a time instead. Falling through to the default would
            // return the body's position, which GJK would happily treat as a
            // single-point shape and report no collision with anything.
            return body.position;
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

// ------------------------------------------------------------- distance -----

namespace {

// The point of a simplex closest to the ORIGIN, with the simplex reduced to
// only the vertices that actually support that point.
//
// This is the half of GJK that a boolean intersection test does not need, and
// it is where the fiddly cases live: the answer may be a vertex, an edge, a
// face or the interior, and carrying along vertices that do not contribute is
// what makes the next iteration pick a direction it has already tried.
Vec3 ClosestOnSimplex(SupportPoint* s, int& n) {
    const auto keep = [&](int count, const int* which) {
        SupportPoint tmp[4];
        for (int i = 0; i < count; ++i) tmp[i] = s[which[i]];
        for (int i = 0; i < count; ++i) s[i] = tmp[i];
        n = count;
    };

    if (n == 1) return s[0].p;

    if (n == 2) {
        const Vec3 a = s[0].p, b = s[1].p, ab = b - a;
        const float denom = Dot(ab, ab);
        if (denom < 1e-20f) { const int k[1] = {0}; keep(1, k); return a; }
        const float t = std::clamp(Dot(a * -1.0f, ab) / denom, 0.0f, 1.0f);
        if (t <= 0.0f) { const int k[1] = {0}; keep(1, k); return a; }
        if (t >= 1.0f) { const int k[1] = {1}; keep(1, k); return b; }
        return a + ab * t;
    }

    if (n == 3) {
        const Vec3 a = s[0].p, b = s[1].p, c = s[2].p;
        const Vec3 ab = b - a, ac = c - a, ao = a * -1.0f;
        const float d1 = Dot(ab, ao), d2 = Dot(ac, ao);
        if (d1 <= 0.0f && d2 <= 0.0f) { const int k[1] = {0}; keep(1, k); return a; }
        const Vec3 bo = b * -1.0f;
        const float d3 = Dot(ab, bo), d4 = Dot(ac, bo);
        if (d3 >= 0.0f && d4 <= d3) { const int k[1] = {1}; keep(1, k); return b; }
        const float vc = d1 * d4 - d3 * d2;
        if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
            const int k[2] = {0, 1};
            const float t = d1 / (d1 - d3);
            keep(2, k);
            return a + ab * t;
        }
        const Vec3 co = c * -1.0f;
        const float d5 = Dot(ab, co), d6 = Dot(ac, co);
        if (d6 >= 0.0f && d5 <= d6) { const int k[1] = {2}; keep(1, k); return c; }
        const float vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
            const int k[2] = {0, 2};
            const float t = d2 / (d2 - d6);
            keep(2, k);
            return a + ac * t;
        }
        const float va = d3 * d6 - d5 * d4;
        if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
            const int k[2] = {1, 2};
            const float t = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            keep(2, k);
            return b + (c - b) * t;
        }
        const float denom = 1.0f / (va + vb + vc);
        return a + ab * (vb * denom) + ac * (vc * denom);
    }

    // n == 4. The origin is inside unless it is outside a face; test each face
    // that it is outside of and take the nearest.
    const Vec3 p[4] = {s[0].p, s[1].p, s[2].p, s[3].p};

    // DEGENERATE TETRAHEDRON. Four coplanar points enclose nothing, so the
    // origin cannot be inside one -- but each face test then decides on the
    // sign of a number that is zero to within rounding, and they can all come
    // back "inside". The caller reads that as the shapes overlapping.
    //
    // This is not a rare shape. GJK against a BOX collects support points from
    // whichever face is pointing at the query, and a box's face has four
    // coplanar corners -- so a point approaching a cube face-on produces
    // exactly this. Measured before the guard: the distance from a point to a
    // unit cube came back as zero for every point within 2.7 units of it, and
    // correct beyond that, which made every hull raycast stop short.
    const float vol = Dot(p[1] - p[0], Cross(p[2] - p[0], p[3] - p[0]));
    const float scale = Length(p[1] - p[0]) * Length(p[2] - p[0]) *
                        Length(p[3] - p[0]);
    const bool flat = std::fabs(vol) <= 1e-5f * std::max(scale, 1e-12f);
    static const int kFace[4][3] = {{0, 1, 2}, {0, 2, 3}, {0, 3, 1}, {1, 3, 2}};
    Vec3 best{0, 0, 0};
    float best_d2 = 1e30f;
    int best_keep[3] = {0, 1, 2};
    int best_n = 0;
    for (const auto& f : kFace) {
        const Vec3 fa = p[f[0]], fb = p[f[1]], fc = p[f[2]];
        const Vec3 nrm = Cross(fb - fa, fc - fa);
        // The opposite vertex tells us which side is "in". On a flat
        // tetrahedron it tells us nothing, so every face is a candidate.
        int opp = 0;
        for (int i = 0; i < 4; ++i)
            if (i != f[0] && i != f[1] && i != f[2]) opp = i;
        const float side = Dot(nrm, p[opp] - fa);
        const float here = Dot(nrm, fa * -1.0f);
        // Origin on the same side as the opposite vertex: inside this face.
        if (!flat && side * here >= 0.0f) continue;

        SupportPoint sub[3] = {s[f[0]], s[f[1]], s[f[2]]};
        int sn = 3;
        const Vec3 q = ClosestOnSimplex(sub, sn);
        const float d2 = Dot(q, q);
        if (d2 < best_d2) {
            best_d2 = d2;
            best = q;
            best_n = sn;
            for (int i = 0; i < sn; ++i) {
                // Map the reduced sub-simplex back to indices in `s`.
                for (int j = 0; j < 3; ++j)
                    if (Dot(sub[i].p - p[f[j]], sub[i].p - p[f[j]]) < 1e-20f)
                        best_keep[i] = f[j];
            }
        }
    }
    if (best_n == 0) { n = 4; return Vec3{0, 0, 0}; }  // origin is inside
    keep(best_n, best_keep);
    return best;
}

}  // namespace

float Distance(const Body& a, const Body& b, Vec3* normal, Vec3* point_a,
               Vec3* point_b) {
    Vec3 dir = b.position - a.position;
    if (Dot(dir, dir) < 1e-12f) dir = Vec3{1, 0, 0};

    SupportPoint s[4];
    s[0] = MinkowskiSupport(a, b, dir);
    int n = 1;
    Vec3 closest = s[0].p;

    for (int iter = 0; iter < 48; ++iter) {
        closest = ClosestOnSimplex(s, n);
        const float d2 = Dot(closest, closest);
        if (d2 < 1e-12f || n == 4) {
            if (normal) *normal = Vec3{0, 1, 0};
            return 0.0f;  // overlapping; use CollideConvex for the depth
        }
        dir = closest * -1.0f;
        const SupportPoint w = MinkowskiSupport(a, b, dir);
        // No progress: the support in the direction of the origin is no closer
        // than the point we already have, so this IS the closest feature.
        // Comparing progress rather than iterating a fixed number of times is
        // what makes the result exact for a flat face instead of nearly right.
        const float progress = Dot(closest, closest) - Dot(w.p, closest);
        if (progress <= 1e-9f * std::sqrt(d2)) break;
        s[n++] = w;
        if (n > 4) break;  // cannot happen with a reduced simplex; belt and braces
    }

    const float dist = Length(closest);
    if (normal) *normal = dist > 1e-9f ? closest * (-1.0f / dist) : Vec3{0, 1, 0};
    // Witnesses, by the same barycentric blend EPA uses. Only meaningful when
    // the simplex has been reduced to the supporting feature, which it has.
    if (point_a || point_b) {
        Vec3 pa{0, 0, 0}, pb{0, 0, 0};
        if (n == 1) {
            pa = s[0].pa;
            pb = s[0].pb;
        } else if (n == 2) {
            const Vec3 ab = s[1].p - s[0].p;
            const float denom = Dot(ab, ab);
            const float t = denom > 1e-20f
                                ? std::clamp(Dot(closest - s[0].p, ab) / denom, 0.0f, 1.0f)
                                : 0.0f;
            pa = s[0].pa + (s[1].pa - s[0].pa) * t;
            pb = s[0].pb + (s[1].pb - s[0].pb) * t;
        } else {
            float u, v, w2;
            Barycentric(s[0].p, s[1].p, s[2].p, &u, &v, &w2);
            pa = s[0].pa * u + s[1].pa * v + s[2].pa * w2;
            pb = s[0].pb * u + s[1].pb * v + s[2].pb * w2;
        }
        if (point_a) *point_a = pa;
        if (point_b) *point_b = pb;
    }
    return dist;
}

// ------------------------------------------------------------------ CCD -----

float TimeOfImpact(const Body& a, const Body& b, Vec3 motion_a, Vec3 motion_b,
                   float tolerance) {
    // CONSERVATIVE ADVANCEMENT. Given the distance between two shapes and an
    // upper bound on how fast that distance can close, there is a span of time
    // in which they certainly cannot touch. Advance by exactly that, and
    // repeat. Each step is provably safe, so the result never skips an impact
    // -- which a fixed number of intermediate samples always eventually does,
    // and only for the fast objects where it matters most.
    const Vec3 relative = motion_a - motion_b;
    const float speed = Length(relative);
    if (speed < 1e-9f) return 1.0f;

    Body ma = a, mb = b;
    float t = 0.0f;
    for (int iter = 0; iter < 32; ++iter) {
        ma.position = a.position + motion_a * t;
        mb.position = b.position + motion_b * t;
        Vec3 n;
        const float dist = Distance(ma, mb, &n, nullptr, nullptr);
        if (dist <= tolerance) return t;
        // The distance closes at most at the full relative speed -- it closes
        // exactly that fast only when the motion is straight along the
        // separating direction. Using the projected speed instead would be
        // tighter and would also be WRONG here, because the direction turns as
        // the bodies move.
        const float advance = (dist - tolerance) / speed;
        t += advance;
        if (t >= 1.0f) return 1.0f;
    }
    return t;
}

// ---------------------------------------------------------------- queries ---

namespace {

// Ray against a sphere, solved directly. `dir` need not be unit; `t` comes back
// in its units.
bool RaySphere(Vec3 origin, Vec3 dir, Vec3 centre, float radius, float max_t,
               float* t_out, Vec3* normal_out) {
    const Vec3 m = origin - centre;
    const float a = Dot(dir, dir);
    if (a < 1e-20f) return false;
    const float b = 2.0f * Dot(m, dir);
    const float c = Dot(m, m) - radius * radius;
    // Starting INSIDE is a hit at t = 0 with no meaningful normal. Reporting a
    // miss instead is the more common choice and the wrong one: a character
    // that spawns inside geometry then has no way to discover it.
    if (c <= 0.0f) {
        *t_out = 0.0f;
        *normal_out = Dot(m, m) > 1e-12f ? Normalize(m) : Vec3{0, 1, 0};
        return true;
    }
    const float disc = b * b - 4.0f * a * c;
    if (disc < 0.0f) return false;
    const float root = std::sqrt(disc);
    // The NEAR root. Both are positive here because c > 0 means the origin is
    // outside, so taking the far one would report the exit wound.
    const float t = (-b - root) / (2.0f * a);
    if (t < 0.0f || t > max_t) return false;
    *t_out = t;
    *normal_out = Normalize(origin + dir * t - centre);
    return true;
}

// Ray against an oriented box, by the slab method in the box's own frame --
// where it is axis-aligned and the test is three interval intersections.
bool RayBox(Vec3 origin, Vec3 dir, const Body& b, float max_t, float* t_out,
            Vec3* normal_out) {
    const Vec3 o = RotateInverse(b.orientation, origin - b.position);
    const Vec3 d = RotateInverse(b.orientation, dir);
    const Vec3 h = b.shape.half_extents;

    float tmin = 0.0f, tmax = max_t;
    int axis = 0;
    float sign = 1.0f;
    for (int i = 0; i < 3; ++i) {
        const float oi = (&o.x)[i], di = (&d.x)[i], hi = (&h.x)[i];
        if (std::fabs(di) < 1e-9f) {
            // Parallel to this pair of faces: either inside the slab forever or
            // outside it forever, and no finite t divides the two.
            //
            // DEFENSIVE, and honestly so: removing it changes no answer that
            // this engine's tests can produce. Dividing by zero gives a signed
            // infinity, the interval arithmetic below comes out right for both
            // inside and outside, and a ray exactly ON a face makes one bound
            // 0/0 -- a NaN that then vanishes, because `NaN > tmin` is false
            // and `std::min(tmax, NaN)` returns tmax by specification. So the
            // NaN never reaches either bound.
            //
            // It stays because that is three coincidences deep, two of them in
            // the standard library's argument order, and the cost is one
            // compare per axis.
            if (oi < -hi || oi > hi) return false;
            continue;
        }
        const float inv = 1.0f / di;
        float t1 = (-hi - oi) * inv, t2 = (hi - oi) * inv;
        float s = -1.0f;
        if (t1 > t2) { std::swap(t1, t2); s = 1.0f; }
        if (t1 > tmin) { tmin = t1; axis = i; sign = s; }
        tmax = std::min(tmax, t2);
        if (tmin > tmax) return false;
    }
    *t_out = tmin;
    Vec3 n{0, 0, 0};
    (&n.x)[axis] = sign;
    // Started inside: tmin stayed at zero and no axis was chosen, so the normal
    // above is arbitrary. Say so by pointing back along the ray.
    if (tmin <= 0.0f) n = Normalize(dir) * -1.0f;
    *normal_out = Rotate(b.orientation, n);
    return true;
}

}  // namespace

bool CollideAny(const Body& a, const Body& b, Contact* out) {
    const bool a_field = a.shape.type == ShapeType::Heightfield;
    const bool b_field = b.shape.type == ShapeType::Heightfield;
    if (a_field && b_field) return false;  // two terrains have nothing to say
    if (b_field) return CollideHeightfield(a, b, out);
    if (a_field) {
        // Answered for (convex, field), so the normal comes back pointing from
        // the convex body toward the terrain -- and the caller asked for a to b.
        if (!CollideHeightfield(b, a, out)) return false;
        out->normal = out->normal * -1.0f;
        return true;
    }
    return CollideConvex(a, b, out);
}

// A convex body against a height field.
//
// ONE CONTACT, the deepest, rather than one per overlapping triangle. A capsule
// standing on a slope overlaps four or six triangles at once, and emitting a
// contact for each gives the solver six impulses pushing the same body out of
// the same ground -- which resolves to six times the correction and launches it.
//
// The alternative, a proper manifold, needs the contacts deduplicated by normal
// and their impulses shared. That is what the box-box path does, and it is worth
// it there because a crate resting flat on one contact rocks. A capsule has no
// flat face to rest on, so one contact holds it still.
bool CollideHeightfield(const Body& convex, const Body& field, Contact* out) {
    const HeightfieldData* h = field.shape.heightfield.get();
    if (!h || !h->Valid() || !out) return false;

    // The query's box, in the FIELD's own space.
    Aabb box = BodyBounds(convex, 0.02f);
    box.lo = box.lo - field.position;
    box.hi = box.hi - field.position;

    constexpr int kMaxTriangles = 256;
    Vec3 corners[kMaxTriangles * 3];
    const int count = GatherHeightfieldTriangles(*h, box, corners, kMaxTriangles);
    if (count == 0) return false;

    Body tri;
    tri.inverse_mass = 0.0f;
    tri.position = field.position;
    bool any = false;
    Contact best;
    best.depth = 0.0f;
    for (int i = 0; i < count; ++i) {
        // A TRIANGLE AS A THREE-POINT HULL. GJK asks a shape only for its
        // support point, and the support of a triangle is whichever of three
        // vertices is furthest along the direction -- so the general convex
        // path already handles it and no triangle-specific code is needed.
        //
        // The vertices go in relative to the body's position, because
        // Support() adds that back.
        tri.shape.type = ShapeType::Hull;
        tri.shape.points = {corners[i * 3 + 0], corners[i * 3 + 1], corners[i * 3 + 2]};
        float r = 0.0f;
        for (const Vec3& p : tri.shape.points) r = std::max(r, Length(p));
        tri.shape.radius = r;
        tri.shape.bounds_radius = r;

        Contact c;
        if (!CollideConvex(convex, tri, &c)) continue;
        if (c.depth <= best.depth) continue;
        best = c;
        any = true;
    }
    if (!any) return false;
    *out = best;
    return true;
}

bool World::Raycast(Vec3 origin, Vec3 direction, float max_distance, RayHit* out,
                    const QueryFilter& filter) const {
    if (!out) return false;
    *out = RayHit{};
    const float len = Length(direction);
    if (len < 1e-12f || max_distance <= 0.0f) return false;

    RefreshBroadphase();

    float best_t = max_distance;
    // THE TREE, not every body. `best_t` shrinks as hits are found and the slab
    // test reads it, so a ray down a corridor of a thousand crates stops
    // descending into the ones behind the first wall.
    //
    // Note this is NOT an ordered traversal -- the nearest child is not visited
    // first -- so `best_t` only tightens as luck has it. Ordering the descent
    // would help a long ray in a dense scene and costs a comparison and a swap
    // per node; it is the next thing to do here if a profile ever asks.
    bvh_.QueryRay(origin, direction, max_distance, [&](int i) {
        if (i == filter.ignore) return;
        const Body& b = bodies_[std::size_t(i)];
        if ((b.layer & filter.mask) == 0u) return;
        if (b.trigger && !filter.hit_triggers) return;

        // A cheap reject against the bounding sphere first. Every shape keeps
        // its radius up to date, and the exact tests below are ten times the
        // work of this one.
        float t = 0.0f;
        Vec3 n{0, 1, 0};
        // From the shape's CENTRE, which is the body's position for everything
        // except a height field -- whose grid extends from its own origin, so a
        // sphere centred on the body would sit half a terrain away and reject
        // every ray that hits the ground.
        if (!RaySphere(origin, direction, b.position + b.shape.bounds_centre,
                       b.shape.bounds_radius, best_t, &t, &n))
            return;

        switch (b.shape.type) {
            case ShapeType::Sphere:
                break;  // the bounding test WAS the exact test
            case ShapeType::Box:
                if (!RayBox(origin, direction, b, best_t, &t, &n)) return;
                break;
            case ShapeType::Heightfield: {
                // A MARCH of the height field, then a bisection -- the same
                // method eng::Terrain::Raycast uses, and deliberately so: the
                // two have to agree or a click that the renderer's terrain says
                // hit the ground would miss in physics.
                const HeightfieldData* h = b.shape.heightfield.get();
                if (!h || !h->Valid()) return;
                const Vec3 local_origin = origin - b.position;
                const Vec3 unit = direction * (1.0f / len);
                const float step = h->spacing;
                float march = 0.0f;
                bool hit = false;
                float gap = local_origin.y - h->HeightAt(local_origin.x, local_origin.z);
                if (gap <= 0.0f) {
                    t = 0.0f;
                    hit = true;
                } else {
                    while (march < best_t * len) {
                        const float next = std::min(march + step, best_t * len);
                        const Vec3 p = local_origin + unit * next;
                        if (p.y - h->HeightAt(p.x, p.z) <= 0.0f) {
                            float lo2 = march, hi2 = next;
                            for (int k = 0; k < 24; ++k) {
                                const float mid = (lo2 + hi2) * 0.5f;
                                const Vec3 q = local_origin + unit * mid;
                                if (q.y - h->HeightAt(q.x, q.z) > 0.0f) lo2 = mid;
                                else hi2 = mid;
                            }
                            t = hi2 / len;
                            hit = true;
                            break;
                        }
                        march = next;
                        if (next >= best_t * len) break;
                    }
                }
                if (!hit) return;
                const Vec3 p = local_origin + unit * (t * len);
                const float d = h->spacing;
                n = Normalize(Vec3{h->HeightAt(p.x - d, p.z) - h->HeightAt(p.x + d, p.z),
                                   2.0f * d,
                                   h->HeightAt(p.x, p.z - d) - h->HeightAt(p.x, p.z + d)});
                break;
            }
            case ShapeType::Capsule:
            case ShapeType::Hull: {
                // CONSERVATIVE ADVANCEMENT, the same idea the continuous
                // collision code uses: the distance from a point to a convex
                // body is a lower bound on how far the point may travel before
                // touching it, so advancing by exactly that is always safe and
                // never skips the surface.
                //
                // The alternative is clipping the ray against the hull's face
                // planes, which is faster and needs the FACES -- and a hull
                // shape deliberately stores only its vertices, because that is
                // all a support function needs. Paying a few iterations here
                // keeps that true.
                Body probe;
                probe.shape = Shape::MakeSphere(0.0f);
                float march = 0.0f;
                bool hit = false;
                for (int iter = 0; iter < 32; ++iter) {
                    probe.position = origin + direction * march;
                    Vec3 dir_to;
                    const float dist = Distance(probe, b, &dir_to, nullptr, nullptr);
                    if (dist <= 1e-4f) { hit = true; break; }
                    march += dist / len;
                    if (march > best_t) break;
                    n = dir_to * -1.0f;  // Distance points probe -> body
                }
                if (!hit || march > best_t) return;
                t = march;
                break;
            }
        }

        // Also belt-and-braces: every shape test above is given `best_t` as
        // its own limit and rejects anything beyond it, so a farther body
        // cannot get this far. Kept because that is a property of three
        // separate routines agreeing, not of this loop.
        if (t < best_t) {
            best_t = t;
            out->body = i;
            out->t = t;
            out->point = origin + direction * t;
            out->normal = n;
        }
    });
    return out->Hit();
}

int World::OverlapShape(const Shape& shape, Vec3 position, Quat orientation,
                        std::vector<int>* out, const QueryFilter& filter) const {
    if (!out) return 0;
    Body probe;
    probe.shape = shape;
    probe.position = position;
    probe.orientation = orientation;
    probe.inverse_mass = 0.0f;

    RefreshBroadphase();
    // The probe's own box is what the tree is asked for. No margin: the tree's
    // leaves already carry one, so a body whose true shape is just outside the
    // probe is still reported here and rejected by the exact test below.
    const Aabb probe_box = BodyBounds(probe);

    int found = 0;
    bvh_.QueryBox(probe_box, [&](int i) {
        if (i == filter.ignore) return;
        const Body& b = bodies_[std::size_t(i)];
        if ((b.layer & filter.mask) == 0u) return;
        if (b.trigger && !filter.hit_triggers) return;
        // Bounding spheres first, for the same reason as the raycast. Measured
        // from the shape's CENTRE, which for a height field is not its position
        // -- the grid extends from its own origin, so a terrain body at (0,0,0)
        // has its centre half a kilometre away and a test against `position`
        // rejects the ground under your feet.
        const Vec3 other_centre = b.position + b.shape.bounds_centre;
        const float reach = probe.shape.bounds_radius + b.shape.bounds_radius;
        if (Dot(other_centre - position, other_centre - position) > reach * reach)
            return;
        // The general convex test, not the specialised ones: a query shape can
        // be any of the three against any of the three, and CollideConvex is
        // the path that needs no case analysis. A height field is the exception
        // -- it is not convex.
        Contact c;
        if (!CollideAny(probe, b, &c)) return;
        out->push_back(i);
        ++found;
    });
    // SORTED, because a tree traversal reports in whatever order the nodes
    // happen to lie and the old linear scan reported in index order. Callers
    // that iterate the result and take the first, or that compare two runs,
    // were relying on that without anyone writing it down.
    std::sort(out->end() - found, out->end());
    return found;
}

int World::OverlapSphere(Vec3 centre, float radius, std::vector<int>* out,
                         const QueryFilter& filter) const {
    return OverlapShape(Shape::MakeSphere(radius), centre, Quat{}, out, filter);
}

int World::OverlapBox(Vec3 centre, Vec3 half_extents, std::vector<int>* out,
                      const QueryFilter& filter) const {
    return OverlapShape(Shape::MakeBox(half_extents), centre, Quat{}, out, filter);
}

void World::Collide() {
    contacts_.clear();
    stats_.pairs_tested = 0;

    // --- broadphase ------------------------------------------------------------
    //
    // The tree, queried once per body that could move. A settled scene queries
    // almost nothing; a scene of a thousand scattered crates queries a thousand
    // small boxes against a balanced tree instead of testing half a million
    // pairs.
    RefreshBroadphase();
    pairs_.clear();
    for (int q = 0; q < int(bodies_.size()); ++q) {
        const Body& a = bodies_[std::size_t(q)];
        // Nothing that could move: two static bodies, two sleeping ones, or one
        // of each. A contact between them has nothing to resolve, and skipping
        // the narrowphase is what makes sleeping cost less rather than merely
        // look calmer — without it the broadphase still walks every body in a
        // settled scene and dominates the step.
        //
        // Safe because waking is contact-driven from the OTHER side: a moving
        // body still generates a contact against a sleeping one, and that is
        // what wakes it. Which is also why only NON-inert bodies query at all:
        // every pair an inert body is in is found from the other end.
        if (a.IsStatic() || a.sleeping) continue;
        bvh_.QueryBox(body_boxes_[std::size_t(q)], [&](int j) {
            if (j == q) return;
            const Body& b = bodies_[std::size_t(j)];
            const bool b_inert = b.IsStatic() || b.sleeping;
            // Each pair recorded exactly ONCE. Two non-inert bodies are seen
            // from both ends, so only the ascending direction is kept; an inert
            // partner is only ever seen from here, so it is kept whichever way
            // round the indices are, and then normalised below.
            if (!b_inert && j < q) return;
            const int lo = std::min(q, j), hi = std::max(q, j);
            pairs_.push_back((std::uint64_t(lo) << 32) | std::uint32_t(hi));
        });
    }

    // SORTED, and this is not tidiness. A tree traversal visits nodes in
    // whatever order the build laid them out, so without this the contact list
    // — and therefore the order the solver applies impulses, and therefore the
    // answer — would depend on the shape of the tree. Sorting restores exactly
    // the order the brute-force double loop produced, which is what makes every
    // existing physics test a regression test on this change rather than a set
    // of numbers that had to be re-tuned to match a new ordering.
    std::sort(pairs_.begin(), pairs_.end());

    for (std::uint64_t packed : pairs_) {
        const int i = int(packed >> 32);
        const int j = int(packed & 0xFFFFFFFFu);
        {
            const Body& a = bodies_[std::size_t(i)];
            const Body& b = bodies_[std::size_t(j)];
            ++stats_.pairs_tested;

            Contact c;
            // A HEIGHTFIELD before anything else, because it is the one shape
            // GJK cannot be run against directly -- it is not convex, and every
            // branch below assumes both sides are.
            if (a.shape.type == ShapeType::Heightfield ||
                b.shape.type == ShapeType::Heightfield) {
                const bool a_field = a.shape.type == ShapeType::Heightfield;
                // Two height fields have nothing to say to each other, and both
                // are static anyway.
                if (a_field && b.shape.type == ShapeType::Heightfield) continue;
                const Body& convex = a_field ? b : a;
                const Body& field = a_field ? a : b;
                if (CollideHeightfield(convex, field, &c)) {
                    // Labelled so the normal points from `a` toward `b`, which
                    // is what the solver assumes. CollideHeightfield answers
                    // for (convex, field), so a swap flips the normal too.
                    if (a_field) {
                        c.a = j;
                        c.b = i;
                    } else {
                        c.a = i;
                        c.b = j;
                    }
                    contacts_.push_back(c);
                }
                continue;
            }
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
    // TRIGGERS out of the solver's list, in one pass rather than at each of the
    // six places a contact is appended -- one of those sits after an early
    // "continue" and would have been missed.
    trigger_contacts_.clear();
    contacts_.erase(
        std::remove_if(contacts_.begin(), contacts_.end(),
                       [&](const Contact& c) {
                           if (!bodies_[std::size_t(c.a)].trigger &&
                               !bodies_[std::size_t(c.b)].trigger)
                               return false;
                           trigger_contacts_.push_back(c);
                           return true;
                       }),
        contacts_.end());

    stats_.contacts = int(contacts_.size());
    BuildTouchEvents();
}

void World::BuildTouchEvents() {
    // CLEARED first. Without this the list accumulates for the life of the
    // world -- which does not look like a bug from the outside, because the
    // events in it are all real: a caller iterating them sees last frame's
    // Begin again this frame, and every frame after.
    touch_events_.clear();

    // Pairs touching NOW, from both lists: a solid collision beginning is as
    // useful an event as a trigger being entered -- it is when the impact
    // sound plays.
    const auto key = [](int a, int b) {
        const std::uint64_t lo = std::uint64_t(std::min(a, b));
        const std::uint64_t hi = std::uint64_t(std::max(a, b));
        // Ordered, so a pair has ONE key however the narrowphase happened to
        // label it -- the sphere-box path swaps its arguments so the sphere is
        // always first, which means the same two bodies come back as (3,0) or
        // (0,3) depending on which is which.
        //
        // It also gives callers a guarantee worth having: every event reports
        // a < b, so a pair can be used as a map key without normalising it at
        // each of the call sites that does.
        return (lo << 32) | hi;
    };

    struct Seen {
        std::uint64_t k;
        Vec3 point, normal;
    };
    std::vector<Seen> now;
    now.reserve(contacts_.size() + trigger_contacts_.size());
    for (const Contact& c : contacts_) now.push_back({key(c.a, c.b), c.point, c.normal});
    for (const Contact& c : trigger_contacts_)
        now.push_back({key(c.a, c.b), c.point, c.normal});
    std::sort(now.begin(), now.end(),
              [](const Seen& x, const Seen& y) { return x.k < y.k; });
    now.erase(std::unique(now.begin(), now.end(),
                          [](const Seen& x, const Seen& y) { return x.k == y.k; }),
              now.end());

    // Built during the merge below rather than copied from `now`, because a
    // pair can survive into it without being in `now` -- see the sleeping case.
    touching_next_.clear();
    touching_next_.reserve(now.size());

    const auto unpack = [](std::uint64_t k, int* a, int* b) {
        *a = int(k >> 32);
        *b = int(k & 0xFFFFFFFFu);
    };

    // A linear merge over two sorted lists, which is the whole operation: what
    // is in one and not the other is a transition, and what is in both is not.
    std::size_t i = 0, j = 0;
    while (i < now.size() || j < touching_.size()) {
        const bool have_new = i < now.size();
        const bool have_old = j < touching_.size();
        if (have_new && (!have_old || now[i].k < touching_[j])) {
            TouchEvent e;
            unpack(now[i].k, &e.a, &e.b);
            e.phase = TouchPhase::Begin;
            e.point = now[i].point;
            e.normal = now[i].normal;
            touch_events_.push_back(e);
            touching_next_.push_back(now[i].k);
            ++i;
        } else if (have_old && (!have_new || touching_[j] < now[i].k)) {
            int a = 0, b = 0;
            unpack(touching_[j], &a, &b);
            // A pair can vanish from the contact list without stopping
            // touching: the broadphase skips two bodies that are both static
            // or asleep, because there is nothing to resolve between them.
            //
            // A ball that lands and settles does exactly that, and reporting
            // End for it says "the ball left the floor" at the moment it
            // finally came to rest. So a pair whose bodies are all inert is
            // carried forward as still-touching and produces no event at all;
            // waking either of them puts it back in the contact list, which
            // produces a Stay rather than a second Begin.
            const Body& ba = bodies_[std::size_t(a)];
            const Body& bb = bodies_[std::size_t(b)];
            const bool a_inert = ba.IsStatic() || ba.sleeping;
            const bool b_inert = bb.IsStatic() || bb.sleeping;
            if (a_inert && b_inert) {
                touching_next_.push_back(touching_[j]);
            } else {
                TouchEvent e;
                e.a = a;
                e.b = b;
                e.phase = TouchPhase::End;
                // Point and normal stay zeroed: the bodies are apart, and a
                // position from two steps ago is worse than nothing because it
                // looks usable.
                touch_events_.push_back(e);
            }
            ++j;
        } else {
            TouchEvent e;
            unpack(now[i].k, &e.a, &e.b);
            e.phase = TouchPhase::Stay;
            e.point = now[i].point;
            e.normal = now[i].normal;
            touch_events_.push_back(e);
            touching_next_.push_back(now[i].k);
            ++i;
            ++j;
        }
    }
    // The next step's merge is a linear scan that requires both sides sorted,
    // and nothing else enforces it.
    //
    // As written this sort is a NO-OP and measurably so: the merge above always
    // pushes whichever of the two fronts is smaller, including the carried-over
    // pairs, so it emits in ascending order by construction. Removing the sort
    // changes no test. It stays because that property belongs to the merge's
    // control flow rather than to anything declared, and the failure if it ever
    // stops holding is pairs churning between Begin and End forever -- with no
    // symptom until something goes to sleep.
    std::sort(touching_next_.begin(), touching_next_.end());
    touching_.swap(touching_next_);
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
    contacts_per_body_.assign(bodies_.size(), 0);
    for (const Contact& c : contacts_) {
        ++contacts_per_body_[std::size_t(c.a)];
        ++contacts_per_body_[std::size_t(c.b)];
    }
    for (const Contact& c : contacts_) {
        Body& a = bodies_[std::size_t(c.a)];
        Body& b = bodies_[std::size_t(c.b)];
        const float inv_sum = a.inverse_mass + b.inverse_mass;
        if (inv_sum <= 0.0f) continue;
        const float excess = std::max(c.depth - penetration_slop, 0.0f);
        const Vec3 push = c.normal * (excess * penetration_correction / inv_sum);
        if (a.sleeping && b.sleeping) continue;
        const float share_a = 1.0f / float(std::max(contacts_per_body_[std::size_t(c.a)], 1));
        const float share_b = 1.0f / float(std::max(contacts_per_body_[std::size_t(c.b)], 1));
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
    // The integrator below moves bodies directly, not through operator[], so
    // nothing else would notice. Marked here rather than at each of the four
    // loops that write a position.
    bvh_dirty_ = true;
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
        // CONTINUOUS collision, for the bodies that asked for it. The step is
        // clipped at the first impact so the body lands ON the surface instead
        // of past it; the discrete solver then resolves the contact next step,
        // exactly as it would for anything slow. Nothing else in the pipeline
        // has to know.
        Vec3 motion = b.velocity * fixed_dt;
        if (b.bullet && ccd_enabled) {
            float toi = 1.0f;
            for (const Body& other : bodies_) {
                if (&other == &b) continue;
                // AGAINST MOVING BODIES TOO. This used to skip anything awake
                // and dynamic, on the stated grounds that "pretending otherwise
                // by sweeping against a stale position would be worse than the
                // honest gap" -- and that justification was simply wrong.
                // TimeOfImpact takes motion_a AND motion_b, advances both, and
                // works on their relative motion; this call site was passing
                // zero for the second one. The correct answer was already
                // implemented and was not being asked for.
                //
                // It matters most in the case the gap was named after: two
                // bullets head-on, each covering half the distance in a step.
                // Neither one's own sweep sees the other move, so both discrete
                // tests miss and they pass through each other.
                const Vec3 other_motion =
                    other.IsStatic() ? Vec3{0.0f, 0.0f, 0.0f} : other.velocity * fixed_dt;
                // A cheap reject first. If the swept sphere of one cannot reach
                // the bounding sphere of the other, no sweep is needed -- and
                // that is the overwhelmingly common case, which is what keeps
                // this affordable at all.
                //
                // RELATIVE motion in the reach, not just this body's: two
                // bodies closing at a combined speed cover more ground between
                // them than either one does alone, and using one side's motion
                // rejects the pair that needs the sweep most.
                const float reach = b.shape.bounds_radius +
                                    other.shape.bounds_radius +
                                    Length(motion - other_motion);
                if (Dot(other.position - b.position, other.position - b.position) >
                    reach * reach)
                    continue;
                toi = std::fmin(toi, TimeOfImpact(b, other, motion, other_motion));
            }
            if (toi < 1.0f) {
                // Stop a little PAST the touch point, not exactly on it.
                //
                // The discrete narrowphase needs an actual overlap to produce a
                // contact. Landing exactly on the surface produces none, so the
                // solver never runs, the velocity is never resolved, and next
                // step the time of impact is zero again -- the body sits
                // against the wall frozen, holding its full speed, and leaps
                // forward if the wall ever moves. That is a worse failure than
                // tunnelling, because it looks correct.
                //
                // Overshooting by a few millimetres hands the body to the
                // ordinary contact path with a shallow penetration, which is
                // exactly the state a slow-moving body would have arrived in.
                constexpr float kSlop = 4e-3f;
                const float len = Length(motion);
                const float extra = len > 1e-9f ? kSlop / len : 0.0f;
                motion = motion * std::fmin(1.0f, toi + extra);
                // The velocity is NOT zeroed. The body has only been stopped
                // early in time, not in space: it is touching the surface with
                // its speed intact, and the next step's contact resolution is
                // what decides whether it bounces, slides or stops. Killing the
                // velocity here would make every fast impact a dead stop and
                // quietly disable restitution for exactly the objects most
                // likely to need it.
                ++stats_.toi_clamps;
            }
        }
        b.position = b.position + motion;

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
