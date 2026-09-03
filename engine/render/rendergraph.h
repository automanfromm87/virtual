// Pure C++20. Orders a frame's passes from their resource dependencies.
//
// Honest scope note: with a single pass this is dead weight, and it stays dead
// weight until a frame has passes that feed each other. It earns its place the
// moment you add shadows, or post-processing, or anything that renders to a
// texture something else then samples — because at that point the ORDER stops
// being obvious and hand-maintaining it is how frames end up reading garbage.
//
// What it does: you declare what each pass writes and what it reads; the graph
// derives the order. What it deliberately does NOT do yet: allocate or alias
// transient resources, insert barriers, or cull passes whose output nobody
// consumes. Those are the next things, not today's things.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "engine/rhi/rhi.h"

namespace eng {

class RenderGraph {
  public:
    // Records draws for one pass. The encoder's pass is already open.
    using ExecuteFn = std::function<void(rhi::Encoder&)>;

    struct Pass {
        std::string name;
        // A depth-only pass (a shadow map) legitimately has no colour target,
        // so `color` may be null when `depth` is not.
        rhi::TextureId color;                 // written by this pass
        // Where a multisample `color` is averaged down to at end of pass. For
        // dependency purposes this counts as WRITTEN by the pass — a later pass
        // reading the resolve target depends on this one, and the graph would
        // otherwise reject it as reading a texture nobody writes.
        rhi::TextureId resolve;
        rhi::TextureId depth;                 // written; null = no depth
        std::vector<rhi::TextureId> reads;    // sampled inputs
        float clear_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        float clear_depth = 0.0f;             // reversed-Z far value
        bool keep_depth = false;              // true for a shadow map
        ExecuteFn execute;
    };

    void AddPass(Pass p) { passes_.push_back(std::move(p)); }
    void Clear();

    // Topologically orders the passes. Fails on a dependency cycle, or on a
    // read of a texture no declared pass writes.
    [[nodiscard]] bool Compile(std::string& error);

    // Runs the compiled order. Caller owns BeginFrame/Commit — the graph
    // organises a frame, it does not own one.
    void Execute(rhi::Device&);

    // Compiled execution order, by name. Exposed so a test can assert the graph
    // actually reordered something rather than replaying insertion order.
    [[nodiscard]] const std::vector<std::string>& Order() const { return order_names_; }

  private:
    std::vector<Pass> passes_;
    std::vector<int> order_;
    std::vector<std::string> order_names_;
};

}  // namespace eng
