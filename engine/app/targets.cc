#include "engine/app/targets.h"

namespace eng::app {

FrameTargets::FrameTargets(rhi::Device& device, rhi::Format color)
    : device_(device), color_(color) {}

FrameTargets::~FrameTargets() { DestroyAll(); }

void FrameTargets::DestroyAll() {
    for (auto& [name, t] : targets_)
        if (Valid(t.id)) device_.DestroyTexture(t.id);
    targets_.clear();
}

void FrameTargets::Resize(int width, int height) {
    if (width == width_ && height == height_) return;
    // Everything screen-sized is wrong at the new size, so the whole set goes.
    // Keeping the map and reallocating lazily would leave a stale handle live
    // for anyone who cached one across the resize.
    DestroyAll();
    width_ = width;
    height_ = height;
}

FrameTargets::Target& FrameTargets::Lookup(std::string_view name, bool is_depth,
                                           bool sampleable) {
    // The kind is part of the key. Without it Color("scene") and Depth("scene")
    // collide and the second call hands back the first one's texture — a colour
    // target bound as a depth buffer, which is not a subtle failure but is a
    // silent one at this layer.
    std::string key = is_depth ? "d:" : "c:";
    key += name;

    auto it = targets_.find(key);
    if (it != targets_.end() && Valid(it->second.id)) {
        // Asking for the same depth target both sampleable and not is a real
        // change of resource, not a flag on one. Recreating rather than
        // returning the wrong kind keeps it correct; the churn shows up in
        // Allocations() for anyone who does it every frame by accident.
        if (it->second.sampleable == sampleable) return it->second;
        device_.DestroyTexture(it->second.id);
        targets_.erase(it);
    }

    Target t;
    t.is_depth = is_depth;
    t.sampleable = sampleable;
    if (width_ > 0 && height_ > 0) {
        t.id = is_depth ? device_.CreateDepthTarget(width_, height_, sampleable)
                        : device_.CreateRenderTarget(width_, height_, color_);
        if (Valid(t.id)) ++allocations_;
    }
    return targets_[key] = t;
}

rhi::TextureId FrameTargets::Color(std::string_view name) {
    return Lookup(name, /*is_depth=*/false, /*sampleable=*/false).id;
}

rhi::TextureId FrameTargets::Depth(std::string_view name, bool sampleable) {
    return Lookup(name, /*is_depth=*/true, sampleable).id;
}

}  // namespace eng::app
