// Pure C++20. Streaming residency: what should be in memory right now, loaded
// in the background, evicted under a budget.
//
// THE PROBLEM. Everything in this engine is loaded up front, so the largest
// world it can hold is the smallest amount of memory any target machine has.
// That is not a size limit you can work around by being careful with assets --
// it is a hard ceiling that has nothing to do with what is on screen.
//
// WHAT THIS IS, and what it is not. This decides RESIDENCY: which resources,
// and at which level of detail, should be in memory given where the viewer is
// and how much memory there is. It does not know what a texture or a mesh is,
// it does not touch the GPU, and it does not read files -- the caller supplies
// a load function and does the upload. That split is deliberate: residency is a
// scheduling problem with no graphics in it, and mixing the two would make it
// testable only through a device.
//
// LEVELS. A resource has one or more levels, smallest first, and loading is
// cumulative: level 2 resident means levels 0, 1 and 2 are. That is exactly a
// mip chain read from the top down, which is the case this exists for -- a
// distant object needs its 32x32 mip and nothing else, and that mip is a
// thousandth of the full texture.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <vector>

#include "engine/core/math.h"

namespace eng {

// A handle into the streamer's table. Never zero for a live resource.
//
// GENERATION COUNTED: `v` is the table slot and `gen` counts how many times
// the slot has been recycled by Remove/Add. Two ids are equal only when both
// fields match, and every lookup verifies the generation, so an id kept past
// Remove() can never silently name the resource that took its slot -- it
// simply stops resolving (ResidentLevel/TargetLevel return -1 for it).
struct StreamId {
    std::uint32_t v = 0;
    std::uint32_t gen = 0;
};
inline bool Valid(StreamId s) { return s.v != 0; }
inline bool operator==(StreamId a, StreamId b) {
    return a.v == b.v && a.gen == b.gen;
}
inline bool operator!=(StreamId a, StreamId b) { return !(a == b); }

struct StreamConfig {
    // Bytes the resident set may occupy. Exceeded only transiently, while a
    // load that was issued under the old budget is still in flight.
    std::size_t budget = 64u * 1024u * 1024u;
    // Worker threads. Loading is IO-bound, so more than a couple buys little,
    // and zero makes every load run on the calling thread inside Update --
    // which is what a test wants, because it removes the timing entirely.
    int threads = 2;
    // HYSTERESIS, in LEVELS. A resident level is held until the level the
    // viewer's distance actually calls for has fallen this far below it.
    // Rising is immediate; falling is sticky.
    //
    // Without it, a viewer stepping back and forth across a level boundary
    // loads and evicts the same level every frame forever -- measured at 99
    // loads and 100 evictions over 200 frames. The picture is correct
    // throughout, so the only symptom is a disk that never stops and a frame
    // time that spikes at random.
    //
    // In LEVELS and not as a priority ratio, which is what this was first
    // written as. One level is a factor of four in distance, so the 1.35x ratio
    // it started with converted to zero whole levels of slack and the whole
    // mechanism did nothing. A margin has to be in the same units as the
    // decision it is damping.
    float evict_levels = 0.5f;
};

// Priority is screen-space size: radius over distance, which is proportional to
// the angle the resource subtends. Not distance alone -- a mountain at two
// kilometres needs its detail more than a teacup at ten metres.
class Streamer {
  public:
    // How many finished loads may wait undrained in the Ready queue before
    // Update stops issuing new ones. 256 finished levels is many frames of
    // backlog for any caller that drains per frame, and a caller that never
    // drains is exactly what the backpressure is for.
    static constexpr std::size_t kMaxReadyLoads = 256;

    // Fills `out` with the bytes of one LEVEL of one resource. Called on a
    // worker thread, so it must not touch anything the main thread owns.
    // Returning false marks the level failed; it will be retried when the
    // resource next becomes wanted, not in a tight loop.
    using LoadFn =
        std::function<bool(StreamId, int level, std::vector<std::uint8_t>& out)>;

    [[nodiscard]] static std::unique_ptr<Streamer> Create(const StreamConfig&,
                                                          LoadFn);
    ~Streamer();
    Streamer(const Streamer&) = delete;
    Streamer& operator=(const Streamer&) = delete;

    // Registers a resource. `level_bytes` is the ADDITIONAL cost of each level
    // over the one below it, smallest first. A single resource may not exceed
    // the budget on its own: a level chain whose smallest cumulative size is
    // already over budget can never become resident, and Add rejects it
    // outright (returns an invalid id) rather than admitting a resource that
    // loads forever and settles nowhere.
    [[nodiscard]] StreamId Add(Vec3 position, float radius,
                               std::span<const std::size_t> level_bytes);

    // Unregisters a resource: cancels its in-flight loads, frees its resident
    // bytes, drops its undrained Ready entries and recycles the slot for a
    // later Add. Without this a long session -- a world streamed for hours --
    // grows the table without bound, since every visited resource stayed
    // registered forever. Removing an unknown or already-removed id is a
    // no-op that returns false; ids issued before the call keep their value
    // but stop resolving (see the generation count on StreamId).
    [[nodiscard]] bool Remove(StreamId);

    // Recomputes what should be resident and issues the work. Returns without
    // blocking; finished loads arrive through NextReady.
    //
    // BACKPRESSURE: when the undrained Ready backlog reaches kMaxReadyLoads,
    // Update still recomputes targets and evicts, but issues no new loads --
    // a caller that never drains would otherwise turn the budget advisory no
    // matter how carefully Update accounts for it.
    void Update(Vec3 viewer);

    // One finished load, or false when there are none. Drain this every frame
    // on the thread that owns the GPU -- the streamer holds the bytes until
    // then, so leaving them undrained is what makes the budget overshoot.
    //
    // At most kMaxReadyLoads entries wait here; past that Update stops issuing
    // (see above) instead of queueing without bound.
    struct Ready {
        StreamId id;
        int level = 0;
        std::vector<std::uint8_t> bytes;
    };
    [[nodiscard]] bool NextReady(Ready&);

    // Blocks until nothing is in flight. For tests and for a load screen; a
    // frame loop must never call it.
    void Wait();

    // The level currently resident, or -1 for nothing. After a Ready has been
    // drained, not when the load was issued.
    [[nodiscard]] int ResidentLevel(StreamId) const;
    // What Update most recently decided this resource ought to be at.
    [[nodiscard]] int TargetLevel(StreamId) const;

    struct Stats {
        std::size_t resident_bytes = 0;
        int resident = 0;      // resources with at least level 0
        int in_flight = 0;     // loads issued and not yet drained
        std::uint64_t loads = 0;    // levels loaded, ever
        std::uint64_t evictions = 0;  // levels dropped, ever
        std::uint64_t failures = 0;
    };
    [[nodiscard]] Stats GetStats() const;

  private:
    Streamer();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eng
