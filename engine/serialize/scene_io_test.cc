// No test framework — from scratch means from scratch.
//
// The property that matters for a serialiser is ROUND-TRIP FIDELITY, and the
// sharp version of it is textual: save, load, save again, and the two strings
// must match byte for byte. Comparing fields by hand only tests the fields
// someone remembered to compare, and the ones that get forgotten are exactly
// the ones that get dropped.
#include "engine/serialize/scene_io.h"

#include <cmath>
#include <cstdio>
#include <string>

namespace {

int g_failures = 0;

void Fail(const char* what, int line) {
    std::fprintf(stderr, "scene_io_test.cc:%d  %s\n", line, what);
    ++g_failures;
}

#define CHECK(cond) \
    do { if (!(cond)) Fail(#cond, __LINE__); } while (0)

using namespace eng;
using namespace eng::serialize;

Registry MakeRegistry() {
    Registry r;
    r.AddMesh("sphere", MeshHandle{1});
    r.AddMesh("box", MeshHandle{2});
    r.AddMaterial("lit", MaterialHandle{1});
    r.AddMaterial("metal", MaterialHandle{2});
    return r;
}

// A scene with one of everything the format can express.
void BuildSample(ecs::World& w, physics::World& p) {
    p.gravity = Vec3{0.0f, -9.81f, 0.0f};
    p.angular_damping = 0.7f;
    p.solver_iterations = 6;

    const ecs::Entity floor = w.Create();
    w.names.Set(floor, ecs::Name{"floor"});
    ecs::Transform ft;
    ft.position = Vec3{0.0f, -0.5f, 0.0f};
    w.transforms.Set(floor, ft);
    w.renderables.Set(floor, {MeshHandle{2}, MaterialHandle{1}, Vec4{1, 1, 1, 1}, true});
    {
        physics::Body b;
        b.shape = physics::Shape::MakeBox(Vec3{9.0f, 0.5f, 9.0f});
        b.position = ft.position;
        b.restitution = 0.45f;
        b.friction = 0.6f;
        b.SetMass(0.0f);  // static
        w.bodies.Set(floor, {p.Add(b)});
    }

    const ecs::Entity pivot = w.Create();
    w.names.Set(pivot, ecs::Name{"pivot"});
    ecs::Transform pt;
    pt.position = Vec3{0.0f, 3.1f, 0.0f};
    pt.rotation = QuatFromAxisAngle(Vec3{0, 1, 0}, 0.9f);
    w.transforms.Set(pivot, pt);  // no renderable, no body: a bare pivot

    const ecs::Entity arm = w.Create();
    w.names.Set(arm, ecs::Name{"arm"});
    ecs::Transform at;
    at.position = Vec3{2.4f, 0.0f, 0.0f};
    at.scale = 1.6f;
    w.transforms.Set(arm, at);
    w.renderables.Set(arm, {MeshHandle{1}, MaterialHandle{2},
                            Vec4{0.9f, 0.3f, 0.2f, 1.0f}, false});
    w.SetParent(arm, pivot);

    const ecs::Entity ball = w.Create();
    ecs::Transform bt;
    bt.position = Vec3{1.0f, 5.0f, -2.0f};
    bt.rotation = QuatFromAxisAngle(Vec3{0.3f, 1.0f, 0.2f}, 0.4f);
    w.transforms.Set(ball, bt);
    w.renderables.Set(ball, {MeshHandle{1}, MaterialHandle{1}, Vec4{1, 1, 1, 1}, true});
    {
        physics::Body b;
        b.shape = physics::Shape::MakeSphere(0.42f);
        b.position = bt.position;
        b.orientation = bt.rotation;
        b.velocity = Vec3{1.5f, 0.0f, -0.25f};
        b.angular_velocity = Vec3{0.0f, 0.0f, -3.0f};
        b.restitution = 0.35f;
        b.friction = 0.7f;
        b.SetMass(2.0f);
        w.bodies.Set(ball, {p.Add(b)});
    }
}

}  // namespace

int main() {
    const Registry reg = MakeRegistry();

    // --- save, load, save: the text must be identical -------------------------
    std::string once;
    {
        ecs::World w;
        physics::World p;
        BuildSample(w, p);
        once = Save(w, p, reg);

        ecs::World w2;
        physics::World p2;
        std::string error;
        CHECK(Load(once, reg, Scene{&w2, &p2}, error));
        CHECK(error.empty());
        const std::string twice = Save(w2, p2, reg);
        CHECK(once == twice);
        if (once != twice) {
            std::fprintf(stderr, "--- first ---\n%s\n--- second ---\n%s\n",
                         once.c_str(), twice.c_str());
        }
    }

    // --- and the loaded world really is the same world ------------------------
    {
        ecs::World w;
        physics::World p;
        std::string error;
        std::vector<Named> names;
        CHECK(Load(once, reg, Scene{&w, &p}, error, &names));

        CHECK(w.AliveCount() == 4);
        CHECK(p.Count() == 2);
        CHECK(names.size() == 3);  // the ball has no name

        // Physics settings, not just entities.
        CHECK(std::fabs(p.angular_damping - 0.7f) < 1e-6f);
        CHECK(p.solver_iterations == 6);
        CHECK(std::fabs(p.gravity.y + 9.81f) < 1e-6f);

        auto find = [&](const char* n) {
            for (const Named& e : names)
                if (e.name == n) return e.entity;
            return ecs::kNoEntity;
        };
        const ecs::Entity pivot = find("pivot");
        const ecs::Entity arm = find("arm");
        CHECK(w.Alive(pivot) && w.Alive(arm));

        // The HIERARCHY survived, which is the part a flat list of entities
        // silently drops.
        CHECK(w.transforms.Get(arm)->parent == pivot);
        CHECK(w.Depth(arm) == 1);
        CHECK(std::fabs(w.transforms.Get(arm)->scale - 1.6f) < 1e-6f);
        // ...and it was applied: the arm's world position is the rotated offset,
        // not its local one.
        const Vec4 o = w.WorldOf(arm) * Vec4{0, 0, 0, 1};
        CHECK(std::fabs(o.x - 2.4f) > 0.1f);
        CHECK(std::fabs(o.y - 3.1f) < 1e-4f);

        // A bare pivot stays bare rather than acquiring a default mesh.
        CHECK(!w.renderables.Has(pivot));
        CHECK(!w.bodies.Has(pivot));

        // Non-default renderable fields.
        const ecs::Renderable* r = w.renderables.Get(arm);
        CHECK(r->visible == false);
        CHECK(std::fabs(r->tint.x - 0.9f) < 1e-6f);
        CHECK(r->mesh == MeshHandle{1} && r->material == MaterialHandle{2});

        // Bodies: static stays static, and mass is not confused with its
        // inverse — a file saying "mass: 0" must not load as mass 0 dynamic.
        const ecs::Entity floor = find("floor");
        CHECK(p[w.bodies.Get(floor)->body].IsStatic());
        CHECK(p[w.bodies.Get(floor)->body].shape.type == physics::ShapeType::Box);

        // The ball's motion round-tripped, so a paused simulation can be saved
        // and resumed rather than restarted.
        int ball_body = -1;
        for (std::size_t i = 0; i < w.bodies.Size(); ++i) {
            const int idx = w.bodies.At(i).body;
            if (p[idx].shape.type == physics::ShapeType::Sphere) ball_body = idx;
        }
        CHECK(ball_body >= 0);
        const physics::Body& b = p[ball_body];
        CHECK(std::fabs(b.velocity.x - 1.5f) < 1e-6f);
        CHECK(std::fabs(b.angular_velocity.z + 3.0f) < 1e-6f);
        CHECK(std::fabs(1.0f / b.inverse_mass - 2.0f) < 1e-4f);
        // Inertia was recomputed from mass and shape, not stored and trusted.
        CHECK(b.inverse_inertia.x > 0.0f);
        CHECK(std::fabs(1.0f / b.inverse_inertia.x - 0.4f * 2.0f * 0.42f * 0.42f) < 1e-4f);
    }

    // --- loading twice into one world appends, it does not clobber -------------
    {
        ecs::World w;
        physics::World p;
        std::string error;
        CHECK(Load(once, reg, Scene{&w, &p}, error));
        CHECK(Load(once, reg, Scene{&w, &p}, error));
        CHECK(w.AliveCount() == 8);
        CHECK(p.Count() == 4);
        // The second copy's parent link points at the SECOND copy's pivot, not
        // the first. Parents are file-relative, and getting that wrong makes
        // two loaded levels quietly share a transform.
        int roots = 0, children = 0;
        for (std::size_t i = 0; i < w.transforms.Size(); ++i) {
            if (w.Depth(w.transforms.Owner(i)) == 0) ++roots;
            else ++children;
        }
        CHECK(children == 2);
        CHECK(roots == 6);
    }

    // --- bad input is rejected, not half-swallowed ----------------------------
    {
        ecs::World w;
        physics::World p;
        std::string error;

        CHECK(!Load("{ not json", reg, Scene{&w, &p}, error));
        CHECK(!error.empty());

        CHECK(!Load(R"({"version":99,"entities":[]})", reg, Scene{&w, &p}, error));
        CHECK(error.find("version") != std::string::npos);

        // An unregistered mesh name must fail loudly. Substituting the null
        // handle gives a scene that loads clean and draws nothing.
        CHECK(!Load(R"({"version":1,"entities":[{"position":[0,0,0],
              "renderable":{"mesh":"nope","material":"lit"}}]})",
                    reg, Scene{&w, &p}, error));
        CHECK(error.find("nope") != std::string::npos);

        CHECK(!Load(R"({"version":1,"entities":[{"position":[0,0,0],"parent":7}]})",
                    reg, Scene{&w, &p}, error));
        CHECK(error.find("parent") != std::string::npos);

        // A body with nowhere to go is an error rather than a silent drop.
        CHECK(!Load(R"({"version":1,"entities":[{"position":[0,0,0],
              "body":{"shape":"sphere","radius":1,"mass":1}}]})",
                    reg, Scene{nullptr, nullptr}, error));
        ecs::World only_ecs;
        CHECK(!Load(R"({"version":1,"entities":[{"position":[0,0,0],
              "body":{"shape":"sphere","radius":1,"mass":1}}]})",
                    reg, Scene{&only_ecs, nullptr}, error));
    }

    // --- defaults are omitted, so a file stays readable ------------------------
    {
        ecs::World w;
        physics::World p;
        const ecs::Entity e = w.Create();
        w.transforms.Set(e, ecs::Transform{});
        const std::string text = Save(w, p, reg);
        // An identity rotation and unit scale carry no information.
        CHECK(text.find("rotation") == std::string::npos);
        CHECK(text.find("scale") == std::string::npos);
        CHECK(text.find("position") != std::string::npos);
        // But they still load back as the defaults.
        ecs::World w2;
        physics::World p2;
        std::string error;
        CHECK(Load(text, reg, Scene{&w2, &p2}, error));
        const ecs::Transform* t = w2.transforms.Get(w2.transforms.Owner(0));
        CHECK(t->scale == 1.0f && t->rotation.w == 1.0f);
    }

    // --- a body on a CHILD entity loads at its WORLD position -----------------
    {
        // Regression test. Load used to read `position` straight off the local
        // Transform, so a child one metre from a parent five metres out got a
        // body at x = 1 instead of x = 6 — and the first physics step then
        // dragged the visible object back to meet it.
        ecs::World w;
        physics::World p;
        const ecs::Entity parent = w.Create();
        ecs::Transform pt;
        pt.position = Vec3{5.0f, 0.0f, 0.0f};
        w.transforms.Set(parent, pt);
        w.names.Set(parent, ecs::Name{"parent"});

        const ecs::Entity child = w.Create();
        ecs::Transform ct;
        ct.position = Vec3{1.0f, 0.0f, 0.0f};  // local; world is (6,0,0)
        ct.rotation = QuatFromAxisAngle(Vec3{0, 1, 0}, 0.5f);
        w.transforms.Set(child, ct);
        w.names.Set(child, ecs::Name{"child"});
        CHECK(w.SetParent(child, parent));
        {
            physics::Body b;
            b.shape = physics::Shape::MakeSphere(0.5f);
            b.SetMass(1.0f);
            w.bodies.Set(child, {p.Add(b)});
        }
        w.UpdateTransforms();

        ecs::World w2;
        physics::World p2;
        std::string error;
        std::vector<Named> names;
        CHECK(Load(Save(w, p, reg), reg, Scene{&w2, &p2}, error, &names));

        ecs::Entity c2 = ecs::kNoEntity;
        for (const Named& n : names)
            if (n.name == "child") c2 = n.entity;
        CHECK(w2.Alive(c2));
        const physics::Body& b2 = p2[w2.bodies.Get(c2)->body];
        CHECK(std::fabs(b2.position.x - 6.0f) < 1e-4f);
        // The transform stayed LOCAL, so this is a real conversion rather than
        // the two happening to agree.
        CHECK(std::fabs(w2.transforms.Get(c2)->position.x - 1.0f) < 1e-4f);
        // Orientation came through the chain as well.
        CHECK(std::fabs(b2.orientation.y - ct.rotation.y) < 1e-4f);
    }

    if (g_failures == 0) std::printf("scene_io_test: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
