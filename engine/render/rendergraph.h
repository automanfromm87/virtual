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
    // The same for a COMPUTE pass. A separate type because the two encoders
    // cannot be substituted -- see rhi::ComputeEncoder for why that is a
    // hardware fact rather than a rule invented here.
    using ComputeFn = std::function<void(rhi::ComputeEncoder&)>;

    struct Pass {
        std::string name;
        // Names this pass in the GPU timing report. Null means untimed.
        const char* timer = nullptr;
        // A depth-only pass (a shadow map) legitimately has no colour target,
        // so `color` may be null when `depth` is not.
        rhi::TextureId color;                 // written by this pass
        // Where a multisample `color` is averaged down to at end of pass. For
        // dependency purposes this counts as WRITTEN by the pass — a later pass
        // reading the resolve target depends on this one, and the graph would
        // otherwise reject it as reading a texture nobody writes.
        rhi::TextureId resolve;
        // Additional colour attachments, 1..n. A deferred G-buffer pass writes
        // several targets at once; each counts as WRITTEN, so a later pass may
        // read any of them.
        std::vector<rhi::TextureId> extra_colors;
        rhi::TextureId depth;                 // written; null = no depth
        // Where a multisample depth attachment is resolved to. Counts as
        // WRITTEN by this pass, exactly as `resolve` does, so a later pass that
        // samples the frame's depth depends on the pass that produced it.
        rhi::TextureId depth_resolve;
        // Keep the depth attachment's existing contents rather than clearing.
        // The depth twin of `load`.
        bool load_depth = false;
        std::vector<rhi::TextureId> reads;    // sampled inputs
        float clear_color[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        // Keep the colour attachment's existing contents. Required for any pass
        // that blends over an earlier one -- see rhi::PassDesc::load.
        bool load = false;
        float clear_depth = 0.0f;             // reversed-Z far value
        bool keep_depth = false;              // true for a shadow map
        // A target this pass BLENDS INTO rather than replaces: fog over the
        // scene, decals over a G-buffer, reflections added to the lit image.
        //
        // Distinct from `color` because such a pass is both a reader and a
        // writer of the same texture, and the single-writer rule exists to
        // reject exactly that -- two passes writing one target have no defined
        // order. A modification does have one: it comes after whoever produced
        // the texture, and before whoever reads it next, and if several passes
        // modify the same target they run in DECLARATION order.
        //
        // That is a restricted form of the resource versioning a full frame
        // graph does, and it is the restriction that makes it a dozen lines
        // instead of a rewrite: a chain of modifications is a total order, so
        // no version numbers are needed to name which one a reader wants.
        //
        // Implies `load` -- there is nothing to blend into otherwise.
        rhi::TextureId modifies;
        // Resources this pass writes that are NOT attachments. A compute pass
        // has no attachments at all, so this is the only thing that can order
        // it against the passes that read what it produced -- an environment
        // probe baked in compute and sampled by the lit pass is exactly that.
        std::vector<rhi::TextureId> writes;
        ExecuteFn execute;
        // Set INSTEAD of `execute` to make this a compute pass. Both set is
        // rejected at Compile: a pass is one or the other, and silently
        // preferring one would make the other's draws vanish.
        ComputeFn compute;
    };

    void AddPass(Pass p) { passes_.push_back(std::move(p)); }

    // A compute pass in one line, which is the shape nearly every one takes:
    // it writes some textures and runs some dispatches.
    void AddCompute(std::string name, std::vector<rhi::TextureId> writes,
                    ComputeFn fn, const char* timer = nullptr) {
        Pass p;
        p.name = std::move(name);
        p.writes = std::move(writes);
        p.compute = std::move(fn);
        p.timer = timer;
        passes_.push_back(std::move(p));
    }

    // Declares a texture that already holds what this graph needs, written
    // before it -- last frame, or at load time. Without it the graph rejects
    // any read it cannot trace to a pass, which is the right default: that
    // check is what catches a pass wired to a target nobody filled.
    //
    // So an import is an ASSERTION by the caller, not a way around the check.
    // Re-lighting a G-buffer from an earlier frame is the case that needs one,
    // and a static lookup table loaded once is another.
    void Import(rhi::TextureId t) { imported_.push_back(t); }

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
    std::vector<rhi::TextureId> imported_;
    std::vector<Pass> passes_;
    std::vector<int> order_;
    std::vector<std::string> order_names_;
};

}  // namespace eng
