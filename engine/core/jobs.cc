#include "engine/core/jobs.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace eng {
namespace {

// One outstanding ParallelRanges. Workers claim chunks out of it with a single
// atomic increment, which is the whole scheduling algorithm: there is no queue
// per thread, no stealing, and nothing to balance, because a chunk index is
// already the finest granularity anything could steal at.
struct Batch {
    const std::function<void(int, int)>* fn = nullptr;
    int count = 0;
    int chunk = 1;
    int chunks = 0;
    std::atomic<int> next{0};
    // Chunks STARTED but not finished, plus one held by the submitter until it
    // has claimed everything it is going to. Without that extra one, a worker
    // that finishes the last chunk before the submitter has claimed its first
    // would drop the count to zero and let Wait return early.
    std::atomic<int> outstanding{0};
};

// Whether this thread is inside a ParallelRanges. Nested batches run INLINE on
// the thread that submitted them: a worker that published a nested batch and
// then waited for it would be occupying a thread that the nested batch needs,
// and with every worker doing the same the system deadlocks with all threads
// waiting on work no thread is free to run. Running inline is not a
// degradation to apologise for -- the outer loop is already using every core,
// so there is no parallelism left for the inner one to find.
thread_local bool t_in_batch = false;

}  // namespace

struct JobSystem::Impl {
    std::vector<std::thread> threads;
    std::mutex m;
    std::condition_variable cv;
    // The batches with chunks still unclaimed. At most a handful deep -- one
    // per thread that is simultaneously inside a top-level ParallelRanges.
    std::vector<Batch*> open;
    bool quit = false;

    std::atomic<int> last_chunks{0};
    std::atomic<int> last_on_caller{0};

    // Runs one chunk of `b` if any remain, on the SUBMITTING thread. Returns
    // false when the batch is exhausted.
    //
    // It does not touch `outstanding`, and that is deliberate rather than an
    // omission: the submitter already holds a reference for the whole time it
    // is claiming, so counting its chunks individually would be counting the
    // same thing twice.
    static bool RunOne(Batch* b) {
        const int i = b->next.fetch_add(1, std::memory_order_relaxed);
        if (i >= b->chunks) return false;
        const int begin = i * b->chunk;
        const int end = std::min(begin + b->chunk, b->count);
        const bool saved = t_in_batch;
        t_in_batch = true;
        (*b->fn)(begin, end);
        t_in_batch = saved;
        return true;
    }

    void Worker() {
        std::unique_lock<std::mutex> lock(m);
        for (;;) {
            if (quit) return;
            if (open.empty()) {
                cv.wait(lock);
                continue;
            }
            // The BACK, not the front. The most recently submitted batch is the
            // one whose submitter is currently blocked waiting for it, so
            // finishing that one first is what unblocks a thread; draining an
            // older batch first would leave the newer submitter waiting on
            // work no one has started.
            Batch* b = open.back();
            // Claimed and counted while still holding the lock, so `open` can
            // never name a batch whose storage the submitter has reclaimed.
            b->outstanding.fetch_add(1, std::memory_order_relaxed);
            const int i = b->next.fetch_add(1, std::memory_order_relaxed);
            if (i >= b->chunks) {
                b->outstanding.fetch_sub(1, std::memory_order_release);
                // Exhausted: take it off the list so nothing looks at it again.
                // The submitter cannot do this itself -- it does not know when
                // the last chunk was claimed, only when the last one finished.
                auto it = std::find(open.begin(), open.end(), b);
                if (it != open.end()) open.erase(it);
                continue;
            }
            lock.unlock();
            const int begin = i * b->chunk;
            const int end = std::min(begin + b->chunk, b->count);
            t_in_batch = true;
            (*b->fn)(begin, end);
            t_in_batch = false;
            b->outstanding.fetch_sub(1, std::memory_order_release);
            lock.lock();
        }
    }
};

JobSystem::JobSystem(int workers) : impl_(std::make_unique<Impl>()) {
    workers = std::max(0, workers);
    impl_->threads.reserve(std::size_t(workers));
    for (int i = 0; i < workers; ++i)
        impl_->threads.emplace_back([this] { impl_->Worker(); });
}

JobSystem::~JobSystem() {
    {
        std::lock_guard<std::mutex> lock(impl_->m);
        impl_->quit = true;
    }
    impl_->cv.notify_all();
    for (std::thread& t : impl_->threads) t.join();
}

int JobSystem::Workers() const { return int(impl_->threads.size()); }
int JobSystem::LastChunkCount() const {
    return impl_->last_chunks.load(std::memory_order_relaxed);
}
int JobSystem::LastChunksRunOnCaller() const {
    return impl_->last_on_caller.load(std::memory_order_relaxed);
}

JobSystem& JobSystem::Get() {
    // One worker per hardware thread BEYOND the caller. hardware_concurrency
    // counts the caller's core too, and spawning that many workers on top of a
    // participating caller oversubscribes by one -- which on a fully loaded
    // fork-join costs a context switch on every chunk boundary.
    //
    // Function-local static: thread-safe initialisation is the standard's
    // problem, and the destructor runs at exit, which is what joins the
    // threads. A namespace-scope instance would start threads before main.
    static JobSystem system(
        std::max(0, int(std::thread::hardware_concurrency()) - 1));
    return system;
}

void JobSystem::ParallelRanges(int count, int grain,
                               const std::function<void(int, int)>& fn) {
    if (count <= 0) return;
    grain = std::max(1, grain);

    const int workers = int(impl_->threads.size());
    // Inline when there is no one to hand work to, when the batch is smaller
    // than a single chunk, or when this is already a nested call. All three end
    // up at the same place: run it here, now, with no atomics at all.
    if (workers == 0 || t_in_batch || count <= grain) {
        impl_->last_chunks.store(1, std::memory_order_relaxed);
        impl_->last_on_caller.store(1, std::memory_order_relaxed);
        const bool saved = t_in_batch;
        t_in_batch = true;
        fn(0, count);
        t_in_batch = saved;
        return;
    }

    // FOUR CHUNKS PER THREAD, not one. One chunk each is the obvious split and
    // it is wrong whenever the iterations are not equally expensive: the thread
    // that draws the slow quarter finishes last and everyone else idles. Four
    // gives the atomic counter something to balance with, at the cost of three
    // more claims per thread -- about 300 ns total, against the milliseconds
    // this is for.
    const int target = (workers + 1) * 4;
    int chunk = (count + target - 1) / target;
    chunk = std::max(chunk, grain);
    const int chunks = (count + chunk - 1) / chunk;

    Batch b;
    b.fn = &fn;
    b.count = count;
    b.chunk = chunk;
    b.chunks = chunks;
    // The submitter's own reference. Released once it has stopped claiming, so
    // that a worker finishing the final chunk in between cannot see zero
    // outstanding while this thread is still about to run one.
    b.outstanding.store(1, std::memory_order_relaxed);

    {
        std::lock_guard<std::mutex> lock(impl_->m);
        impl_->open.push_back(&b);
    }
    // notify_all, not notify_one: there are `chunks` pieces of work and waking
    // a single thread to do all of them is a serial loop with extra steps.
    impl_->cv.notify_all();

    int mine = 0;
    while (Impl::RunOne(&b)) ++mine;
    b.outstanding.fetch_sub(1, std::memory_order_release);

    // SPIN, not a condition variable. By the time this is reached the caller
    // has already run every chunk it could claim, so what it is waiting on is
    // one or two chunks already in flight on other threads -- microseconds. A
    // condvar round trip costs more than the wait, and the yield keeps a
    // single-core machine from starving the very thread it is waiting for.
    while (b.outstanding.load(std::memory_order_acquire) != 0)
        std::this_thread::yield();

    // The batch may still be listed if the workers never drained it. Removing
    // it here is not an optimisation -- `b` is a stack local and is about to go
    // away, so a worker holding a pointer to it would read freed memory.
    {
        std::lock_guard<std::mutex> lock(impl_->m);
        auto it = std::find(impl_->open.begin(), impl_->open.end(), &b);
        if (it != impl_->open.end()) impl_->open.erase(it);
    }

    impl_->last_chunks.store(chunks, std::memory_order_relaxed);
    impl_->last_on_caller.store(mine, std::memory_order_relaxed);
}

void JobSystem::ParallelFor(int count, int grain,
                            const std::function<void(int)>& fn) {
    // Wrapped rather than duplicated. The per-item call through std::function
    // is a real cost -- an indirect call per iteration -- and it is exactly why
    // ParallelRanges is the primitive and this is the convenience.
    const std::function<void(int, int)> range = [&fn](int begin, int end) {
        for (int i = begin; i < end; ++i) fn(i);
    };
    ParallelRanges(count, grain, range);
}

}  // namespace eng
