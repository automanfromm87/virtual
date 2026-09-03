// The job system, checked for the three things that can be wrong with one:
// it visits every item exactly once, it actually uses more than one core, and
// it does not deadlock when a caller does something reasonable.
//
// The first is the only one most test suites check, and it is the one least
// likely to be broken. A scheduler that silently runs everything on the calling
// thread passes it perfectly.

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <thread>
#include <vector>

#include "engine/core/jobs.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

// Deliberately expensive and impossible to optimise away: the parallel speedup
// measurement needs work that actually takes time, and a compiler that hoists
// the loop out would make the test measure scheduling overhead alone.
double Burn(int seed) {
    double x = double(seed) * 0.5;
    for (int i = 0; i < 400; ++i) x = std::sin(x) * 1.000001 + 0.5;
    return x;
}

double Seconds(void (*f)()) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
        .count();
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    {
        std::printf("every index is visited exactly once\n");
        constexpr int kN = 100000;
        std::vector<std::atomic<int>> seen(kN);
        for (auto& s : seen) s.store(0, std::memory_order_relaxed);
        eng::JobSystem js(4);
        js.ParallelFor(kN, 64, [&](int i) {
            seen[std::size_t(i)].fetch_add(1, std::memory_order_relaxed);
        });
        int missing = 0, twice = 0;
        for (auto& s : seen) {
            const int n = s.load(std::memory_order_relaxed);
            if (n == 0) ++missing;
            if (n > 1) ++twice;
        }
        std::printf("    %d chunks, %d run on the caller\n", js.LastChunkCount(),
                    js.LastChunksRunOnCaller());
        Check(missing == 0, "no index is skipped");
        Check(twice == 0, "and none is run twice");
        // THE CHECK THAT CATCHES A SCHEDULER THAT DOES NOTHING. Running every
        // chunk on the calling thread satisfies both assertions above.
        Check(js.LastChunksRunOnCaller() < js.LastChunkCount(),
              "and the workers ran some of them");
    }

    {
        // The arithmetic here is the part that is easy to get wrong: a chunk
        // size that does not divide the count, a count smaller than the grain,
        // a count of exactly one. Every one of them has an off-by-one available.
        std::printf("\nawkward sizes are covered exactly\n");
        eng::JobSystem js(3);
        bool all_ok = true;
        for (int count : {0, 1, 2, 3, 7, 8, 9, 63, 64, 65, 1000, 1001}) {
            for (int grain : {1, 3, 8, 64, 1000}) {
                std::vector<int> hits(std::size_t(std::max(count, 1)), 0);
                std::atomic<int> total{0};
                js.ParallelRanges(count, grain, [&](int b, int e) {
                    for (int i = b; i < e; ++i) hits[std::size_t(i)] = 1;
                    total.fetch_add(e - b, std::memory_order_relaxed);
                });
                if (total.load() != count) all_ok = false;
                for (int i = 0; i < count; ++i)
                    if (hits[std::size_t(i)] != 1) all_ok = false;
            }
        }
        Check(all_ok, "60 count/grain combinations each cover their range once");
    }

    {
        std::printf("\nzero workers runs inline and gives the same answer\n");
        constexpr int kN = 5000;
        std::vector<double> parallel(kN), serial(kN);
        {
            eng::JobSystem js(4);
            js.ParallelFor(kN, 32, [&](int i) { parallel[std::size_t(i)] = Burn(i); });
        }
        {
            eng::JobSystem js(0);
            js.ParallelFor(kN, 32, [&](int i) { serial[std::size_t(i)] = Burn(i); });
            Check(js.Workers() == 0, "a zero-worker system reports no workers");
            Check(js.LastChunksRunOnCaller() == 1,
                  "and runs the whole range in one inline chunk");
        }
        bool same = true;
        for (int i = 0; i < kN; ++i)
            if (parallel[std::size_t(i)] != serial[std::size_t(i)]) same = false;
        // BIT-IDENTICAL, not close. Nothing here is order-dependent, so any
        // difference at all would mean the chunking changed what was computed.
        Check(same, "and every result matches the parallel run bit for bit");
    }

    {
        // The measurement the whole thing exists for. Not a tight bound: a
        // shared CI machine is noisy and a laptop may be thermally throttled,
        // so this asks for 1.5x on a machine with at least four cores rather
        // than the ~3.5x it actually gets. What it rules out is the failure
        // that matters -- no speedup at all.
        const int cores = int(std::thread::hardware_concurrency());
        std::printf("\nit is actually parallel (%d hardware threads)\n", cores);
        constexpr int kN = 20000;
        std::vector<double> out(kN);

        eng::JobSystem one(0);
        const auto t0 = std::chrono::steady_clock::now();
        one.ParallelFor(kN, 64, [&](int i) { out[std::size_t(i)] = Burn(i); });
        const double serial =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        eng::JobSystem many(std::max(1, cores - 1));
        const auto t1 = std::chrono::steady_clock::now();
        many.ParallelFor(kN, 64, [&](int i) { out[std::size_t(i)] = Burn(i); });
        const double par =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();

        std::printf("    serial %.1f ms, %d threads %.1f ms, speedup %.2fx\n",
                    serial * 1e3, many.Workers() + 1, par * 1e3, serial / par);
        if (cores >= 4) {
            Check(serial / par > 1.5, "at least 1.5x faster than one thread");
        } else {
            std::printf("    (fewer than four cores; speedup not asserted)\n");
        }
    }

    {
        // A nested ParallelFor is not exotic -- it is what happens the first
        // time someone parallelises a loop whose body already called a
        // parallelised helper. The naive implementation deadlocks: every worker
        // is blocked inside an outer chunk waiting for an inner batch that
        // needs a worker to run it.
        std::printf("\na nested ParallelFor does not deadlock\n");
        eng::JobSystem js(4);
        std::atomic<int> total{0};
        js.ParallelFor(64, 1, [&](int) {
            js.ParallelFor(64, 1, [&](int) {
                total.fetch_add(1, std::memory_order_relaxed);
            });
        });
        Check(total.load() == 64 * 64, "and every inner iteration runs");
    }

    {
        // A system destroyed with idle workers must join them, not detach or
        // abandon them. If this is wrong the test binary hangs at exit rather
        // than failing, which is why it is here and not left to chance.
        std::printf("\nconstruction and destruction are clean\n");
        for (int i = 0; i < 20; ++i) {
            eng::JobSystem js(4);
            std::atomic<int> n{0};
            js.ParallelFor(1000, 16, [&](int) { n.fetch_add(1); });
            if (n.load() != 1000) g_failures++;
        }
        Check(true, "twenty systems created, used and destroyed");
    }

    {
        std::printf("\nthe shared instance works and is one instance\n");
        eng::JobSystem& a = eng::JobSystem::Get();
        eng::JobSystem& b = eng::JobSystem::Get();
        Check(&a == &b, "Get() returns the same system every time");
        std::atomic<long long> sum{0};
        eng::ParallelFor(10000, 64,
                         [&](int i) { sum.fetch_add(i, std::memory_order_relaxed); });
        Check(sum.load() == 10000LL * 9999LL / 2, "and the free function sums correctly");
    }

    std::printf(g_failures == 0 ? "\njobs_test: all checks passed\n"
                                : "\njobs_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
