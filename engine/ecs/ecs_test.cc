// No test framework — from scratch means from scratch.
//
// The properties worth testing here are the ones that make handles better than
// pointers: a destroyed entity's handle must be detectably dead, a recycled
// slot must not resurrect the old occupant, and the hierarchy must refuse to
// form a cycle rather than hang the first time it is walked.
#include "engine/ecs/ecs.h"

#include <cmath>
#include <cstdio>

namespace {

int g_failures = 0;

void Fail(const char* what, int line) {
    std::fprintf(stderr, "ecs_test.cc:%d  %s\n", line, what);
    ++g_failures;
}

#define CHECK(cond) \
    do { if (!(cond)) Fail(#cond, __LINE__); } while (0)

bool Near(float a, float b) { return std::fabs(a - b) < 1e-4f; }

}  // namespace

int main() {
    using namespace eng;
    using namespace eng::ecs;

    // --- lifetime ------------------------------------------------------------
    {
        World w;
        const Entity a = w.Create();
        const Entity b = w.Create();
        CHECK(w.Alive(a) && w.Alive(b));
        CHECK(w.AliveCount() == 2);
        CHECK(a != b);
        // A default-constructed handle is never live. Zero-initialised memory
        // must not read as a valid entity.
        CHECK(!w.Alive(Entity{}));

        w.Destroy(a);
        CHECK(!w.Alive(a));
        CHECK(w.AliveCount() == 1);
        w.Destroy(a);  // idempotent
        CHECK(w.AliveCount() == 1);

        // THE reason for generations: the freed slot gets reused, and the OLD
        // handle must not point at the new occupant. With raw indices this is
        // the bug that corrupts a scene silently.
        const Entity c = w.Create();
        CHECK(c.index == a.index);
        CHECK(c.generation != a.generation);
        CHECK(w.Alive(c) && !w.Alive(a));
    }

    // --- component storage ---------------------------------------------------
    {
        World w;
        const Entity a = w.Create(), b = w.Create(), c = w.Create();
        w.transforms.Set(a, Transform{Vec3{1, 0, 0}});
        w.transforms.Set(b, Transform{Vec3{2, 0, 0}});
        w.transforms.Set(c, Transform{Vec3{3, 0, 0}});
        CHECK(w.transforms.Size() == 3);
        CHECK(w.transforms.Has(a) && w.transforms.Has(c));
        CHECK(Near(w.transforms.Get(b)->position.x, 2));

        // Overwriting an existing component must not add a second one.
        w.transforms.Set(b, Transform{Vec3{9, 0, 0}});
        CHECK(w.transforms.Size() == 3);
        CHECK(Near(w.transforms.Get(b)->position.x, 9));

        // Removal swaps the last element into the hole, so the SURVIVORS must
        // still be findable — a broken sparse fixup shows up exactly here.
        w.transforms.Remove(a);
        CHECK(w.transforms.Size() == 2);
        CHECK(!w.transforms.Has(a));
        CHECK(w.transforms.Get(a) == nullptr);
        CHECK(Near(w.transforms.Get(b)->position.x, 9));
        CHECK(Near(w.transforms.Get(c)->position.x, 3));

        // Destroying an entity takes its components with it.
        w.renderables.Set(c, Renderable{});
        w.Destroy(c);
        CHECK(!w.transforms.Has(c));
        CHECK(!w.renderables.Has(c));
    }

    // --- hierarchy -----------------------------------------------------------
    {
        World w;
        const Entity root = w.Create();
        const Entity mid = w.Create();
        const Entity leaf = w.Create();

        Transform tr;
        tr.position = Vec3{10, 0, 0};
        w.transforms.Set(root, tr);

        Transform tm;
        // 90 degrees about +Y.
        tm.rotation = Quat{0.0f, std::sin(0.7853982f), 0.0f, std::cos(0.7853982f)};
        w.transforms.Set(mid, tm);

        Transform tl;
        tl.position = Vec3{2, 0, 0};
        w.transforms.Set(leaf, tl);

        CHECK(w.SetParent(mid, root));
        CHECK(w.SetParent(leaf, mid));
        CHECK(w.Depth(root) == 0 && w.Depth(mid) == 1 && w.Depth(leaf) == 2);

        w.UpdateTransforms();
        // leaf sits 2 along its own +X; mid turns that to -Z; root shifts it to
        // x = 10. Getting the composition order backwards puts it somewhere
        // else entirely, which is the whole point of having a hierarchy.
        const Vec4 p = w.WorldOf(leaf) * Vec4{0, 0, 0, 1};
        CHECK(Near(p.x, 10.0f));
        CHECK(Near(p.y, 0.0f));
        CHECK(Near(p.z, -2.0f));

        // Moving the ROOT moves the leaf with it. This is the property that a
        // flat instance list cannot express at all.
        w.transforms.Get(root)->position = Vec3{10, 5, 0};
        w.UpdateTransforms();
        const Vec4 p2 = w.WorldOf(leaf) * Vec4{0, 0, 0, 1};
        CHECK(Near(p2.y, 5.0f));
        CHECK(Near(p2.z, -2.0f));

        // Reparenting to nothing leaves the leaf at its local transform.
        CHECK(w.SetParent(leaf, kNoEntity));
        w.UpdateTransforms();
        const Vec4 p3 = w.WorldOf(leaf) * Vec4{0, 0, 0, 1};
        CHECK(Near(p3.x, 2.0f) && Near(p3.y, 0.0f) && Near(p3.z, 0.0f));
    }

    // --- cycles are REFUSED, not discovered later ----------------------------
    {
        World w;
        const Entity a = w.Create(), b = w.Create(), c = w.Create();
        w.transforms.Set(a, Transform{});
        w.transforms.Set(b, Transform{});
        w.transforms.Set(c, Transform{});
        CHECK(w.SetParent(b, a));
        CHECK(w.SetParent(c, b));
        // a -> b -> c already; making a a child of c would close the loop.
        CHECK(!w.SetParent(a, c));
        CHECK(!w.SetParent(a, a));  // self-parenting too
        // The rejected edge must leave the hierarchy untouched, and this must
        // still terminate.
        w.UpdateTransforms();
        CHECK(w.Depth(c) == 2);
    }

    // --- ordering does not depend on creation order --------------------------
    {
        // Children created BEFORE their parents. A naive single pass in
        // insertion order evaluates the child against a stale parent matrix.
        World w;
        const Entity leaf = w.Create();
        const Entity root = w.Create();
        Transform tl; tl.position = Vec3{1, 0, 0};
        Transform tr2; tr2.position = Vec3{0, 7, 0};
        w.transforms.Set(leaf, tl);
        w.transforms.Set(root, tr2);
        CHECK(w.SetParent(leaf, root));
        w.UpdateTransforms();
        const Vec4 p = w.WorldOf(leaf) * Vec4{0, 0, 0, 1};
        CHECK(Near(p.x, 1.0f) && Near(p.y, 7.0f));
    }

    // --- a deleted parent orphans its child rather than crashing -------------
    {
        World w;
        const Entity root = w.Create(), child = w.Create();
        Transform tr; tr.position = Vec3{5, 0, 0};
        w.transforms.Set(root, tr);
        Transform tc; tc.position = Vec3{1, 0, 0};
        w.transforms.Set(child, tc);
        CHECK(w.SetParent(child, root));
        w.Destroy(root);
        w.UpdateTransforms();  // the child still holds a stale parent handle
        const Vec4 p = w.WorldOf(child) * Vec4{0, 0, 0, 1};
        CHECK(Near(p.x, 1.0f));  // treated as a root
        CHECK(w.Depth(child) == 0);
    }

    // --- a world pose written onto a CHILD lands in the right place -----------
    {
        // Physics is world-space; a Transform is not. Assigning one to the
        // other directly is right for a root and wrong for everything else,
        // and the failure is invisible until something gets parented.
        World w;
        const Entity parent = w.Create();
        Transform pt;
        pt.position = Vec3{5.0f, 0.0f, 0.0f};
        pt.rotation = QuatFromAxisAngle(Vec3{0, 1, 0}, 1.5707963f);
        pt.scale = 2.0f;
        w.transforms.Set(parent, pt);

        const Entity child = w.Create();
        w.transforms.Set(child, Transform{});
        CHECK(w.SetParent(child, parent));
        w.UpdateTransforms();

        const Vec3 want_pos{6.0f, 1.0f, -2.0f};
        const Quat want_rot = QuatFromAxisAngle(Vec3{0, 0, 1}, 0.6f);
        w.SetWorldPose(child, want_pos, want_rot);
        w.UpdateTransforms();

        const Vec4 got = w.WorldOf(child) * Vec4{0, 0, 0, 1};
        CHECK(std::fabs(got.x - want_pos.x) < 1e-4f);
        CHECK(std::fabs(got.y - want_pos.y) < 1e-4f);
        CHECK(std::fabs(got.z - want_pos.z) < 1e-4f);
        // The LOCAL transform is not the world one — otherwise this test would
        // pass against the bug it exists to catch.
        CHECK(std::fabs(w.transforms.Get(child)->position.x - want_pos.x) > 0.5f);

        // Orientation survives the round trip too.
        const Quat wr = w.WorldRotationOf(child);
        CHECK(std::fabs(wr.x - want_rot.x) < 1e-4f);
        CHECK(std::fabs(wr.w - want_rot.w) < 1e-4f);
        CHECK(std::fabs(w.WorldScaleOf(child) - 2.0f) < 1e-6f);

        // On a root it is a plain assignment.
        w.SetWorldPose(parent, Vec3{1, 2, 3}, Quat{});
        CHECK(std::fabs(w.transforms.Get(parent)->position.y - 2.0f) < 1e-6f);
    }

    // --- Destroy clears every pool, including names ---------------------------
    {
        World w;
        const Entity e = w.Create();
        w.transforms.Set(e, Transform{});
        w.names.Set(e, Name{"doomed"});
        w.Destroy(e);
        CHECK(w.names.Size() == 0);
        CHECK(w.transforms.Size() == 0);
    }

    if (g_failures == 0) std::printf("ecs_test: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
