#include "engine/render/rendergraph.h"

#include <algorithm>

#include <unordered_map>

namespace eng {

void RenderGraph::Clear() {
    imported_.clear();
    passes_.clear();
    order_.clear();
    order_names_.clear();
}

bool RenderGraph::Compile(std::string& error) {
    order_.clear();
    order_names_.clear();
    const int n = int(passes_.size());
    if (n == 0) return true;

    // Who produces each texture.
    //
    // SINGLE WRITER, enforced. Two passes writing one texture is genuinely
    // ambiguous here: passes are declared in arbitrary order (that is the whole
    // point of deriving the order from dependencies), so nothing says which
    // write should land first. A real frame graph resolves this by VERSIONING —
    // every write produces a new virtual resource and a reader names the
    // version it wants. Until that exists, silently picking one order would
    // hand you a frame that renders differently depending on declaration order.
    // Rejecting is the honest behaviour.
    std::unordered_map<std::uint32_t, int> producer;
    for (int i = 0; i < n; ++i) {
        if (passes_[i].execute && passes_[i].compute) {
            error = "pass '" + passes_[i].name +
                    "' has both a render body and a compute body; it is one or "
                    "the other, and running one would silently drop the other";
            return false;
        }
        // Anything named in `writes` produces that texture, whatever kind of
        // pass it is. Entered first so a compute pass with no attachments is
        // still a producer.
        for (const rhi::TextureId& w : passes_[i].writes) {
            if (!Valid(w)) continue;
            auto [wit, wok] = producer.emplace(w.v, i);
            if (!wok) {
                error = "passes '" + passes_[wit->second].name + "' and '" +
                        passes_[i].name + "' both write the same texture";
                return false;
            }
        }
        // A pass must write SOMEWHERE. Colour is the usual answer; a
        // depth-only shadow pass writes depth instead, and a compute pass
        // writes whatever it declared.
        if (!Valid(passes_[i].color)) {
            // A modification's target is handled below, after every producer is
            // known -- it has to be, because it takes over as the producer of a
            // texture some other pass wrote.
            if (Valid(passes_[i].modifies)) continue;
            if (!Valid(passes_[i].depth) && passes_[i].writes.empty()) {
                error = "pass '" + passes_[i].name +
                        "' writes neither colour nor depth nor any declared "
                        "texture, so nothing can depend on it and it would run "
                        "in an arbitrary place";
                return false;
            }
            continue;  // nothing more to enter in the colour producer map
        }
        auto [it, inserted] = producer.emplace(passes_[i].color.v, i);
        if (!inserted) {
            error = "passes '" + passes_[it->second].name + "' and '" +
                    passes_[i].name +
                    "' both write the same colour target; resource versioning is "
                    "not implemented, so their order would be undefined";
            return false;
        }
        // A resolve target is written by this pass too, and is the one a later
        // pass will actually read — the multisample attachment itself cannot
        // be sampled. Leaving it out means the graph rejects every consumer of
        // an anti-aliased scene as reading a texture nobody writes.
        if (Valid(passes_[i].resolve)) {
            auto [rit, rok] = producer.emplace(passes_[i].resolve.v, i);
            if (!rok) {
                error = "passes '" + passes_[rit->second].name + "' and '" +
                        passes_[i].name + "' both write the same resolve target";
                return false;
            }
        }
        // The DEPTH resolve, on exactly the same footing and for exactly the
        // same reason: it is the only form of this pass's depth that anything
        // downstream can sample.
        if (Valid(passes_[i].depth_resolve)) {
            auto [dit, dok] = producer.emplace(passes_[i].depth_resolve.v, i);
            if (!dok) {
                error = "passes '" + passes_[dit->second].name + "' and '" +
                        passes_[i].name + "' both write the same depth resolve target";
                return false;
            }
        }
        // The G-buffer's extra attachments, on the same footing as the first:
        // a lighting pass reads all of them, and leaving them out of the map
        // would have the graph reject it for reading a texture nobody writes.
        for (const rhi::TextureId& extra : passes_[i].extra_colors) {
            if (!Valid(extra)) continue;
            auto [eit, eok] = producer.emplace(extra.v, i);
            if (!eok) {
                error = "passes '" + passes_[eit->second].name + "' and '" +
                        passes_[i].name +
                        "' both write the same colour attachment";
                return false;
            }
        }
    }

    // Same rule for depth. Two passes sharing a depth buffer are ordered only
    // by their colour dependencies, which is not enough to make the depth
    // contents deterministic.
    std::unordered_map<std::uint32_t, int> depth_writer;
    for (int i = 0; i < n; ++i) {
        if (!Valid(passes_[i].depth)) continue;
        auto [it, inserted] = depth_writer.emplace(passes_[i].depth.v, i);
        if (!inserted) {
            error = "passes '" + passes_[it->second].name + "' and '" +
                    passes_[i].name + "' share a depth target; ordering would be "
                    "undefined";
            return false;
        }
    }

    // A written depth target counts as a producer too, so a pass that SAMPLES
    // a shadow map is ordered after the pass that rendered it.
    for (const auto& [tex, pass_index] : depth_writer)
        producer.emplace(tex, pass_index);

    // MODIFICATIONS, in declaration order. Each takes over as the producer of
    // its target, so a later reader depends on the last modification rather
    // than on the pass that first wrote it -- which is the whole point: a pass
    // reading the scene after the fog has to see the fog.
    //
    // Resolved here, after every ordinary producer is known, because a
    // modification by definition follows one.
    // Braces: with parens, `std::size_t(n)` reads as a parameter declaration
    // and this becomes a function declaration. The same most-vexing-parse trap
    // as `std::vector<T> v(std::size_t(count))`, and the error it produces --
    // "subscript of pointer to function type" -- names the symptom rather than
    // the cause.
    std::vector<std::vector<int>> modify_after{std::size_t(n), std::vector<int>{}};
    for (int i = 0; i < n; ++i) {
        if (!Valid(passes_[i].modifies)) continue;
        auto it = producer.find(passes_[i].modifies.v);
        if (it == producer.end()) {
            const bool imported =
                std::find_if(imported_.begin(), imported_.end(), [&](rhi::TextureId t) {
                    return t.v == passes_[i].modifies.v;
                }) != imported_.end();
            if (!imported) {
                error = "pass '" + passes_[i].name +
                        "' modifies a texture no pass in this graph writes and no "
                        "Import() declares";
                return false;
            }
        } else {
            modify_after[std::size_t(i)].push_back(it->second);
        }
        producer[passes_[i].modifies.v] = i;
    }

    // Edge producer -> consumer for every read.
    std::vector<std::vector<int>> out(n);
    std::vector<int> indegree(n, 0);
    for (int i = 0; i < n; ++i) {
        for (rhi::TextureId r : passes_[i].reads) {
            auto it = producer.find(r.v);
            if (it == producer.end()) {
                // An IMPORTED texture was filled before this graph ran, so it
                // has no producer here and needs no edge. Anything else is a
                // pass wired to a target nobody writes.
                if (std::find_if(imported_.begin(), imported_.end(),
                                 [&](rhi::TextureId t) { return t.v == r.v; }) !=
                    imported_.end())
                    continue;
                error = "pass '" + passes_[i].name +
                        "' reads a texture no pass in this graph writes and no "
                        "Import() declares";
                return false;
            }
            if (it->second == i) {
                // Sampling the texture you are simultaneously rendering into is
                // undefined in every graphics API. The fix is a ping-pong pair
                // of targets, which needs resource versioning — so refuse
                // rather than produce a frame that works by luck.
                error = "pass '" + passes_[i].name +
                        "' samples the same texture it renders into";
                return false;
            }
            out[it->second].push_back(i);
            ++indegree[i];
        }
    }

    for (int i = 0; i < n; ++i)
        for (int from : modify_after[std::size_t(i)]) {
            out[std::size_t(from)].push_back(i);
            ++indegree[std::size_t(i)];
        }

    // Kahn. Ties break on insertion order so the result is deterministic —
    // a graph that shuffles equivalent passes run to run is untestable.
    std::vector<int> ready;
    for (int i = 0; i < n; ++i)
        if (indegree[i] == 0) ready.push_back(i);

    while (!ready.empty()) {
        int best = 0;
        for (int k = 1; k < int(ready.size()); ++k)
            if (ready[k] < ready[best]) best = k;
        const int i = ready[best];
        ready.erase(ready.begin() + best);

        order_.push_back(i);
        order_names_.push_back(passes_[i].name);
        for (int j : out[i])
            if (--indegree[j] == 0) ready.push_back(j);
    }

    if (int(order_.size()) != n) {
        error = "render graph has a dependency cycle";
        order_.clear();
        order_names_.clear();
        return false;
    }
    return true;
}

void RenderGraph::Execute(rhi::Device& dev) {
    for (int i : order_) {
        const Pass& p = passes_[i];
        if (p.compute) {
            rhi::ComputeEncoder enc = dev.BeginCompute(p.timer, p.name.c_str());
            p.compute(enc);
            dev.EndCompute();
            continue;
        }

        rhi::PassDesc desc;
        desc.timer = p.timer;
        // The graph already knows every pass's name; a capture may as well use
        // it. Valid for the duration of Execute, which is the only place the
        // descriptor lives.
        desc.label = p.name.c_str();
        // A modification IS the colour attachment, loaded rather than cleared.
        desc.color = Valid(p.modifies) ? p.modifies : p.color;
        desc.load = p.load || Valid(p.modifies);
        desc.resolve = p.resolve;
        desc.extra_colors = p.extra_colors;
        desc.depth = p.depth;
        desc.depth_resolve = p.depth_resolve;
        desc.load_depth = p.load_depth;
        for (int c = 0; c < 4; ++c) desc.clear_color[c] = p.clear_color[c];
        desc.clear_depth = p.clear_depth;
        desc.keep_depth = p.keep_depth;

        rhi::Encoder enc = dev.BeginPass(desc);
        if (p.execute) p.execute(enc);
        dev.EndPass();
    }
}

}  // namespace eng
