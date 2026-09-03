// A job system: N worker threads, and a fork-join primitive over them.
//
// WHY this exists at all. Every other system in this engine is single
// threaded, and most of them should stay that way -- a render pass is a
// sequence, a physics solver iterating to convergence is a sequence. What is
// NOT a sequence is the shape that keeps recurring: the same independent
// arithmetic over ten thousand elements. Skinning vertices. Culling instances.
// Building a BVH's leaves. Rasterising a navmesh's voxels. Those are the loops
// that dominate a frame, they have no cross-element dependencies, and running
// them on one of eight cores wastes seven.
//
// WHY NOT a general task graph with futures and continuations. A task graph is
// the right answer when the work is heterogeneous and the dependencies are
// discovered at runtime. Here they are not: the caller knows the shape of the
// parallelism when they write the loop. ParallelFor is what that knowledge
// looks like as an API, and it is the one primitive that cannot be misused into
// a deadlock -- it has no way to express one.
//
// THE CALLING THREAD PARTICIPATES. It is not a spectator that blocks on a
// condition variable while the workers do the job. Three reasons, and the third
// is the one that matters: it means a machine reporting one core still works,
// it means the common case of a small batch never pays a wake-up, and it means
// a ParallelFor nested inside another cannot deadlock by waiting for a worker
// that is itself blocked inside the outer one.
#pragma once

#include <cstddef>
#include <functional>
#include <memory>

namespace eng {

class JobSystem {
  public:
    // The process-wide instance, started on first use with one worker per
    // hardware thread beyond the caller's. A singleton because threads are a
    // process resource: two systems each sizing themselves to the machine would
    // between them oversubscribe it by two, and oversubscription is worse than
    // no threading at all -- the cores are still shared, but now with context
    // switches on top.
    [[nodiscard]] static JobSystem& Get();

    // `workers` is the number of threads BESIDES the caller. 0 is legal and
    // makes everything run inline, which is what the tests use to prove that
    // the parallel and serial answers agree.
    explicit JobSystem(int workers);
    ~JobSystem();

    JobSystem(const JobSystem&) = delete;
    JobSystem& operator=(const JobSystem&) = delete;

    // Threads besides the caller. Total concurrency is this plus one.
    [[nodiscard]] int Workers() const;

    // Runs `fn(i)` for every i in [0, count), and does not return until all of
    // them have finished.
    //
    // `grain` is the SMALLEST number of iterations worth handing to a thread.
    // It is not a tuning knob to be left at a default: the right value is
    // whatever makes a chunk cost more than the roughly 100 ns it takes to
    // claim one, and that depends entirely on what `fn` does. A grain of 1 on a
    // trivial body is slower than a plain loop, and measurably so.
    void ParallelFor(int count, int grain, const std::function<void(int)>& fn);

    // The same, but `fn(begin, end)` gets a whole chunk. Use this when the body
    // has per-chunk setup worth hoisting -- a scratch buffer, an accumulator,
    // an RNG -- which is most non-trivial parallel work. ParallelFor is written
    // in terms of it.
    void ParallelRanges(int count, int grain,
                        const std::function<void(int begin, int end)>& fn);

    // --- diagnostics ---------------------------------------------------------
    //
    // How many chunks the last ParallelRanges split into, and how many of them
    // the CALLING thread ran. The second is the one that tells you whether the
    // work actually spread: a caller that ran all of them means the workers
    // never got there, which happens when the batch is too small to be worth
    // splitting and is fine -- and also when the workers are wedged, which is
    // not. There is no other way to tell those apart from outside.
    [[nodiscard]] int LastChunkCount() const;
    [[nodiscard]] int LastChunksRunOnCaller() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Convenience over JobSystem::Get(). The overwhelmingly common call.
inline void ParallelFor(int count, int grain, const std::function<void(int)>& fn) {
    JobSystem::Get().ParallelFor(count, grain, fn);
}
inline void ParallelRanges(int count, int grain,
                           const std::function<void(int, int)>& fn) {
    JobSystem::Get().ParallelRanges(count, grain, fn);
}

}  // namespace eng
