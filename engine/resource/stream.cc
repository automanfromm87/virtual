#include "engine/resource/stream.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace eng {
namespace {

struct Resource {
    Vec3 position{0.0f, 0.0f, 0.0f};
    float radius = 1.0f;
    std::vector<std::size_t> level_bytes;  // additional cost per level
    std::vector<std::size_t> cumulative;   // bytes for levels 0..i inclusive

    int resident = -1;   // highest level actually in memory
    int target = -1;     // what Update wants
    int in_flight = -1;  // level currently being loaded, or -1
    float priority = 0.0f;
    // Set when a load returns false. Cleared when the resource drops out of the
    // wanted set, so a permanent failure costs one attempt per visit rather
    // than one per frame -- a missing file on a hot path would otherwise
    // saturate every worker thread with retries.
    bool failed = false;
};

}  // namespace

struct Streamer::Impl {
    StreamConfig cfg;
    LoadFn load;

    mutable std::mutex mu;
    std::vector<Resource> res{Resource{}};  // index 0 is the null handle

    // Work waiting for a thread, highest priority first. A deque rather than a
    // priority_queue because Update REPLACES the whole queue every frame: last
    // frame's ordering is stale the moment the viewer moves, and re-sorting a
    // heap in place is more code than rebuilding a sorted vector.
    struct Job {
        StreamId id;
        int level;
        float priority;
    };
    std::vector<Job> queue;
    std::condition_variable work;

    std::deque<Ready> ready;
    std::vector<std::thread> workers;
    bool stopping = false;
    int active = 0;  // workers inside a load call

    std::size_t resident_bytes = 0;
    std::uint64_t loads = 0, evictions = 0, failures = 0;
    std::condition_variable idle;

    void Worker();
    // Runs one job if there is one. Returns false when the queue is empty.
    // Shared by the worker threads and by the zero-thread path, which runs it
    // inline from Update -- one implementation, so a test with no threads
    // exercises the same code a shipped build does.
    bool RunOne(std::unique_lock<std::mutex>& lock);
};

Streamer::Streamer() : impl_(std::make_unique<Impl>()) {}

Streamer::~Streamer() {
    {
        std::lock_guard<std::mutex> g(impl_->mu);
        impl_->stopping = true;
    }
    impl_->work.notify_all();
    for (std::thread& t : impl_->workers)
        if (t.joinable()) t.join();
}

std::unique_ptr<Streamer> Streamer::Create(const StreamConfig& cfg, LoadFn load) {
    if (!load) return nullptr;
    std::unique_ptr<Streamer> s(new Streamer());
    s->impl_->cfg = cfg;
    s->impl_->load = std::move(load);
    const int n = std::max(0, std::min(cfg.threads, 32));
    for (int i = 0; i < n; ++i)
        s->impl_->workers.emplace_back([im = s->impl_.get()] { im->Worker(); });
    return s;
}

bool Streamer::Impl::RunOne(std::unique_lock<std::mutex>& lock) {
    if (queue.empty()) return false;
    const Job job = queue.back();  // sorted ascending, so the back is the best
    queue.pop_back();
    ++active;

    // The load runs UNLOCKED. It is the only slow thing here, and holding the
    // mutex across it would serialise every worker and block Update as well --
    // which would make the whole thing a synchronous loader with extra threads.
    lock.unlock();
    std::vector<std::uint8_t> bytes;
    const bool ok = load(job.id, job.level, bytes);
    lock.lock();

    Resource& r = res[job.id.v];
    if (ok) {
        ++loads;
        // Charged when the load LANDS, not when it is issued. The alternative
        // makes the budget a promise about work in flight rather than about
        // memory, and a burst of issued-but-unfinished loads then blocks
        // eviction of things that are genuinely resident.
        resident_bytes += r.level_bytes[std::size_t(job.level)];
        ready.push_back(Ready{job.id, job.level, std::move(bytes)});
    } else {
        ++failures;
        r.failed = true;
    }
    r.in_flight = -1;
    --active;
    if (active == 0 && queue.empty()) idle.notify_all();
    return true;
}

void Streamer::Impl::Worker() {
    std::unique_lock<std::mutex> lock(mu);
    for (;;) {
        work.wait(lock, [this] { return stopping || !queue.empty(); });
        if (stopping) return;
        RunOne(lock);
    }
}

StreamId Streamer::Add(Vec3 position, float radius,
                       std::span<const std::size_t> level_bytes) {
    if (level_bytes.empty()) return {};
    std::lock_guard<std::mutex> g(impl_->mu);
    Resource r;
    r.position = position;
    r.radius = std::max(radius, 1e-4f);
    r.level_bytes.assign(level_bytes.begin(), level_bytes.end());
    std::size_t total = 0;
    for (std::size_t b : r.level_bytes) {
        total += b;
        r.cumulative.push_back(total);
    }
    impl_->res.push_back(std::move(r));
    return StreamId{std::uint32_t(impl_->res.size() - 1)};
}

void Streamer::Update(Vec3 viewer) {
    std::unique_lock<std::mutex> lock(impl_->mu);
    const int n = int(impl_->res.size());
    if (n <= 1) return;

    // --- 1. priority, and the level each resource would like ------------------
    struct Want {
        int index;
        float priority;
    };
    std::vector<Want> wants;
    wants.reserve(std::size_t(n));
    for (int i = 1; i < n; ++i) {
        Resource& r = impl_->res[std::size_t(i)];
        const Vec3 d = r.position - viewer;
        // The ANGLE the resource subtends, near enough: radius over distance.
        // Distance alone gets the ordering wrong in the case that matters --
        // a mountain two kilometres away needs its detail more than a teacup
        // ten metres away, and a distance sort puts the teacup first.
        const float dist = std::max(Length(d), r.radius);
        r.priority = r.radius / dist;
        wants.push_back(Want{i, r.priority});
    }
    std::sort(wants.begin(), wants.end(), [](const Want& a, const Want& b) {
        // Ties broken by index so the residency set is deterministic. Without
        // it, two resources of identical priority swap places between frames
        // and the one that loses gets evicted and reloaded forever.
        if (a.priority != b.priority) return a.priority > b.priority;
        return a.index < b.index;
    });

    // --- 2. fill the budget, most important first -----------------------------
    //
    // Each resource takes the highest level that still fits. Walking in
    // priority order means the budget is spent on what matters, and a resource
    // that does not fit at all gets target -1 rather than a partial level --
    // levels are cumulative, so half of level 3 is not a usable anything.
    std::size_t used = 0;
    for (const Want& w : wants) {
        Resource& r = impl_->res[std::size_t(w.index)];
        const int levels = int(r.level_bytes.size());
        // The level a resource DESERVES from its screen size, before the budget
        // has a say. priority 1 is "as wide as it is far away", which is very
        // close indeed; the log maps each halving of angular size to one level.
        const float p = std::max(r.priority, 1e-6f);
        const float want_f = float(levels - 1) + std::log2(p) * 0.5f;
        int want = std::clamp(int(std::floor(want_f)), 0, levels - 1);

        // HOLD what is already there until the desire has fallen a clear margin
        // below it. Rising is immediate -- a level that is suddenly needed is
        // needed now -- and falling is sticky, because the cost of dropping a
        // level you are about to want again is paid twice.
        if (r.resident > want &&
            want_f + impl_->cfg.evict_levels >= float(r.resident))
            want = r.resident;

        // The BUDGET still overrides. Memory pressure is not a preference, and
        // holding a level nobody can afford would let the resident set exceed
        // the budget for as long as the viewer stood still.
        while (want >= 0 && used + r.cumulative[std::size_t(want)] > impl_->cfg.budget)
            --want;
        r.target = want;
        if (want >= 0) used += r.cumulative[std::size_t(want)];
    }

    // --- 3. evictions ---------------------------------------------------------
    for (int i = 1; i < n; ++i) {
        Resource& r = impl_->res[std::size_t(i)];
        // The failure flag clears when the resource leaves the WANTED set, not
        // when it has nothing resident. A failed load leaves it with nothing
        // resident by definition, so keying on that cleared the flag every
        // frame and a missing file was retried on every one of them -- fifty
        // frames, fifty attempts, and on a real disk fifty stalls.
        if (r.target < 0) r.failed = false;
        if (r.resident < 0) continue;
        // Straight down to the target. The hysteresis lives in how the target
        // was CHOSEN, above -- putting it here as well would be two dampers in
        // series and the second one would make the budget advisory.
        while (r.resident > r.target) {
            impl_->resident_bytes -= r.level_bytes[std::size_t(r.resident)];
            --r.resident;
            ++impl_->evictions;
        }
        if (r.target < 0 && r.resident >= 0) {
            // Nothing wanted at all: drop every level, including zero.
            while (r.resident >= 0) {
                impl_->resident_bytes -= r.level_bytes[std::size_t(r.resident)];
                --r.resident;
                ++impl_->evictions;
            }
        }
    }

    // --- 4. the work queue ----------------------------------------------------
    //
    // Rebuilt from scratch every frame. Last frame's ordering is stale the
    // moment the viewer moves, and a job for a resource that has since dropped
    // out of the wanted set is work nobody will use.
    impl_->queue.clear();
    for (const Want& w : wants) {
        Resource& r = impl_->res[std::size_t(w.index)];
        if (r.failed || r.in_flight >= 0) continue;
        if (r.target <= r.resident) continue;
        // ONE LEVEL at a time, the next one up. Levels are cumulative and the
        // smaller ones are nearly free, so loading level 0 first puts something
        // usable on screen while the rest arrives -- jumping straight to the
        // target means the object is missing for the whole load instead.
        impl_->queue.push_back(Impl::Job{StreamId{std::uint32_t(w.index)},
                                         r.resident + 1, r.priority});
    }
    // Ascending, because RunOne takes from the back.
    std::sort(impl_->queue.begin(), impl_->queue.end(),
              [](const Impl::Job& a, const Impl::Job& b) {
                  if (a.priority != b.priority) return a.priority < b.priority;
                  return a.id.v > b.id.v;
              });
    for (Impl::Job& j : impl_->queue) impl_->res[j.id.v].in_flight = j.level;

    if (impl_->workers.empty()) {
        // NO THREADS: run the queue inline. Not a fallback -- it is what a test
        // uses, because it removes the timing from the problem entirely, and it
        // runs the same RunOne the workers do.
        while (impl_->RunOne(lock)) {
        }
    } else {
        lock.unlock();
        impl_->work.notify_all();
    }
}

bool Streamer::NextReady(Ready& out) {
    std::lock_guard<std::mutex> g(impl_->mu);
    if (impl_->ready.empty()) return false;
    out = std::move(impl_->ready.front());
    impl_->ready.pop_front();
    // The resident level rises HERE, when the caller takes the bytes, and not
    // when the load finished. Until the caller has uploaded them, the resource
    // is not usable, and reporting it resident would have the renderer draw
    // with a level that does not exist yet.
    Resource& r = impl_->res[out.id.v];
    if (out.level == r.resident + 1) r.resident = out.level;
    return true;
}

void Streamer::Wait() {
    std::unique_lock<std::mutex> lock(impl_->mu);
    impl_->idle.wait(lock,
                     [this] { return impl_->queue.empty() && impl_->active == 0; });
}

int Streamer::ResidentLevel(StreamId id) const {
    std::lock_guard<std::mutex> g(impl_->mu);
    if (!Valid(id) || id.v >= impl_->res.size()) return -1;
    return impl_->res[id.v].resident;
}

int Streamer::TargetLevel(StreamId id) const {
    std::lock_guard<std::mutex> g(impl_->mu);
    if (!Valid(id) || id.v >= impl_->res.size()) return -1;
    return impl_->res[id.v].target;
}

Streamer::Stats Streamer::GetStats() const {
    std::lock_guard<std::mutex> g(impl_->mu);
    Stats s;
    s.resident_bytes = impl_->resident_bytes;
    s.loads = impl_->loads;
    s.evictions = impl_->evictions;
    s.failures = impl_->failures;
    s.in_flight = int(impl_->queue.size()) + impl_->active + int(impl_->ready.size());
    for (std::size_t i = 1; i < impl_->res.size(); ++i)
        if (impl_->res[i].resident >= 0) ++s.resident;
    return s;
}

}  // namespace eng
