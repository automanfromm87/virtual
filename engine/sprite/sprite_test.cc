// The sprite batch, checked without a GPU: positions, uvs, tints and the
// order they come out in are all CPU facts, and a batch that puts the
// wrong corner first is invisible in any screenshot that still shows a
// quad roughly where it belongs.
#include "engine/sprite/sprite.h"

#include <cmath>
#include <cstdio>

namespace {

int g_failures = 0;

void Fail(const char* what, int line) {
    std::fprintf(stderr, "sprite_test.cc:%d  %s\n", line, what);
    ++g_failures;
}

#define CHECK(cond) \
    do { if (!(cond)) Fail(#cond, __LINE__); } while (0)

bool Near(float a, float b, float eps = 1e-5f) { return std::fabs(a - b) < eps; }

}  // namespace

int main() {
    using namespace eng;

    // --- empty -----------------------------------------------------------------
    {
        sprite::SpriteBatch b;
        b.Bake();
        CHECK(b.Count() == 0);
        CHECK(b.vertices().empty());
        CHECK(b.indices().empty());
        const Mesh empty = b.BuildMesh();
        CHECK(empty.vertices.empty() && empty.indices.empty());
        CHECK(Near(empty.bounds.radius, 0.0f));
    }

    // --- one axis-aligned sprite ------------------------------------------------
    // Center (1, 2), size (4, 2): corners (-1, 1), (3, 1), (3, 3), (-1, 3).
    {
        sprite::SpriteBatch b;
        sprite::Sprite s;
        s.center = Vec2{1.0f, 2.0f};
        s.size = Vec2{4.0f, 2.0f};
        s.uv = Vec4{0.25f, 0.0f, 0.75f, 1.0f};
        s.tint = Vec4{1.0f, 0.5f, 0.25f, 1.0f};
        b.Push(s);
        b.Bake();
        CHECK(b.vertices().size() == 4);
        CHECK(b.indices().size() == 6);
        const auto& v = b.vertices();
        CHECK(Near(v[0].position.x, -1.0f) && Near(v[0].position.y, 1.0f));
        CHECK(Near(v[2].position.x, 3.0f) && Near(v[2].position.y, 3.0f));
        // v0 is the TOP edge: bottom corners carry v = 1, top corners v = 0.
        CHECK(Near(v[0].uv.x, 0.25f) && Near(v[0].uv.y, 1.0f));
        CHECK(Near(v[2].uv.x, 0.75f) && Near(v[2].uv.y, 0.0f));
        for (const VertexIn& q : v) {
            CHECK(Near(q.normal.z, 1.0f));
            CHECK(Near(q.position.z, 0.0f));
            CHECK(Near(q.color.y, 0.5f) && Near(q.color.z, 0.25f));
        }
        const std::uint32_t* ix = b.indices().data();
        CHECK(ix[0] == 0 && ix[1] == 1 && ix[2] == 2);
        CHECK(ix[3] == 0 && ix[4] == 2 && ix[5] == 3);
        const Mesh m = b.BuildMesh();
        CHECK(Near(m.bounds.center.x, 1.0f) && Near(m.bounds.center.y, 2.0f));
        // Half-extents (2, 1, 0): radius sqrt(5).
        CHECK(Near(m.bounds.radius, std::sqrt(5.0f)));
    }

    // --- rotation ----------------------------------------------------------------
    // 90 degrees CCW maps local (x, y) to (-y, x).
    {
        sprite::SpriteBatch b;
        sprite::Sprite s;
        s.center = Vec2{0.0f, 0.0f};
        s.size = Vec2{2.0f, 4.0f};
        s.rotation = 1.5707963f;
        b.Push(s);
        b.Bake();
        const auto& v = b.vertices();
        // Local bottom-left (-1, -2) -> world (2, -1).
        CHECK(Near(v[0].position.x, 2.0f) && Near(v[0].position.y, -1.0f));
        // Local top-right (1, 2) -> world (-2, 1).
        CHECK(Near(v[2].position.x, -2.0f) && Near(v[2].position.y, 1.0f));
    }

    // --- order: texture first, then layer, ties keep push order -------------------
    {
        sprite::SpriteBatch b;
        sprite::Sprite back, mid, other;
        back.center = Vec2{100.0f, 0.0f};
        back.layer = 1;
        mid.center = Vec2{200.0f, 0.0f};
        mid.layer = 0;
        other.center = Vec2{300.0f, 0.0f};
        other.texture = 1;
        other.layer = 0;
        b.Push(back);   // tex 0, layer 1 -- pushed first, baked second
        b.Push(mid);    // tex 0, layer 0
        b.Push(other);  // tex 1
        b.Bake();
        const auto& v = b.vertices();
        CHECK(Near(v[0].position.x, 200.0f - 0.5f));
        CHECK(Near(v[4].position.x, 100.0f - 0.5f));
        CHECK(Near(v[8].position.x, 300.0f - 0.5f));
    }

    // --- ties keep push order -------------------------------------------------------
    {
        sprite::SpriteBatch b;
        sprite::Sprite a, c;
        a.center = Vec2{10.0f, 0.0f};
        c.center = Vec2{20.0f, 0.0f};
        b.Push(a);
        b.Push(c);
        b.Bake();
        CHECK(Near(b.vertices()[0].position.x, 10.0f - 0.5f));
        CHECK(Near(b.vertices()[4].position.x, 20.0f - 0.5f));
        b.Clear();
        CHECK(b.Count() == 0);
        b.Bake();
        CHECK(b.vertices().empty());
    }

    if (g_failures == 0) std::printf("sprite batch: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
