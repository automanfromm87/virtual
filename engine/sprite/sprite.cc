#include "engine/sprite/sprite.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace eng::sprite {

void SpriteBatch::Push(const Sprite& s) { pending_.push_back(s); }

void SpriteBatch::Clear() {
    pending_.clear();
    vertices_.clear();
    indices_.clear();
}

void SpriteBatch::Bake() {
    std::vector<std::size_t> order(pending_.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (pending_[a].texture != pending_[b].texture)
            return pending_[a].texture < pending_[b].texture;
        return pending_[a].layer < pending_[b].layer;
    });
    vertices_.clear();
    indices_.clear();
    vertices_.reserve(pending_.size() * 4);
    indices_.reserve(pending_.size() * 6);
    for (std::size_t k : order) {
        const Sprite& s = pending_[k];
        const float c = std::cos(s.rotation), n = std::sin(s.rotation);
        const float hx = s.size.x * 0.5f, hy = s.size.y * 0.5f;
        // Bottom-left, bottom-right, top-right, top-left -- counter-clockwise
        // facing +Z, with v0 on the top edge (VertexIn's top-left origin).
        const float lx[4] = {-hx, hx, hx, -hx};
        const float ly[4] = {-hy, -hy, hy, hy};
        const float uu[4] = {s.uv.x, s.uv.z, s.uv.z, s.uv.x};
        const float vv[4] = {s.uv.w, s.uv.w, s.uv.y, s.uv.y};
        const std::uint32_t base = std::uint32_t(vertices_.size());
        for (int i = 0; i < 4; ++i) {
            VertexIn v{};
            v.position = Vec4{s.center.x + lx[i] * c - ly[i] * n,
                              s.center.y + lx[i] * n + ly[i] * c, 0.0f, 0.0f};
            v.normal = Vec4{0.0f, 0.0f, 1.0f, 0.0f};
            v.color = s.tint;
            v.uv = Vec4{uu[i], vv[i], 0.0f, 0.0f};
            v.tangent = Vec4{c, n, 0.0f, 1.0f};
            vertices_.push_back(v);
        }
        indices_.push_back(base + 0);
        indices_.push_back(base + 1);
        indices_.push_back(base + 2);
        indices_.push_back(base + 0);
        indices_.push_back(base + 2);
        indices_.push_back(base + 3);
    }
}

std::size_t SpriteBatch::Count() const { return pending_.size(); }

const std::vector<VertexIn>& SpriteBatch::vertices() const { return vertices_; }

const std::vector<std::uint32_t>& SpriteBatch::indices() const { return indices_; }

Mesh SpriteBatch::BuildMesh() const {
    Mesh m;
    m.vertices = vertices_;
    m.indices = indices_;
    if (m.vertices.empty()) {
        m.bounds = Bounds{};
        return m;
    }
    Vec3 lo = Vec3{m.vertices[0].position.x, m.vertices[0].position.y,
                   m.vertices[0].position.z};
    Vec3 hi = lo;
    for (const VertexIn& v : m.vertices) {
        lo.x = std::min(lo.x, v.position.x);
        lo.y = std::min(lo.y, v.position.y);
        lo.z = std::min(lo.z, v.position.z);
        hi.x = std::max(hi.x, v.position.x);
        hi.y = std::max(hi.y, v.position.y);
        hi.z = std::max(hi.z, v.position.z);
    }
    m.bounds.center = (lo + hi) * 0.5f;
    const Vec3 e = (hi - lo) * 0.5f;
    m.bounds.radius = std::sqrt(e.x * e.x + e.y * e.y + e.z * e.z);
    return m;
}

}  // namespace eng::sprite
