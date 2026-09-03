// Streaming residency. Everything here is a scheduling property, so everything
// here is exact -- there is no image and no tolerance.
//
// The checks are ordered by what they would let through if they were absent:
// the budget being respected (or memory grows without bound), priority ordering
// (or the wrong things are resident and the near ones pop), hysteresis (or the
// disk never stops), and determinism (or nothing above is reproducible).
#include "engine/resource/stream.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <map>
#include <vector>

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

// Levels that quadruple, which is what a mip chain does: level 0 is a 32x32
// mip at 4 KB and each one above is four times the last.
std::vector<std::size_t> MipLevels(int count, std::size_t base) {
    std::vector<std::size_t> v;
    std::size_t b = base;
    for (int i = 0; i < count; ++i) {
        v.push_back(b);
        b *= 4;
    }
    return v;
}

// Drains everything the streamer has finished, as a frame loop would.
int DrainAll(eng::Streamer& s) {
    eng::Streamer::Ready r;
    int n = 0;
    while (s.NextReady(r)) ++n;
    return n;
}

}  // namespace

int main() {
    using namespace eng;
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    // A loader that records what it was asked for. SYNCHRONOUS, because
    // threads = 0 makes Update run the queue inline -- the scheduling
    // properties below are what is being tested and thread timing would only
    // make them intermittent.
    std::map<std::pair<std::uint32_t, int>, int> requested;
    std::atomic<int> total_requests{0};
    const auto loader = [&](StreamId id, int level, std::vector<std::uint8_t>& out) {
        ++requested[{id.v, level}];
        ++total_requests;
        out.assign(16, std::uint8_t(level));
        return true;
    };

    {
        std::printf("the budget is respected\n");
        StreamConfig cfg;
        cfg.threads = 0;
        cfg.budget = 200 * 1024;
        auto s = Streamer::Create(cfg, loader);
        if (!s) {
            std::fprintf(stderr, "FAIL: could not create the streamer\n");
            return 1;
        }

        // Forty resources of five levels each: 4 + 16 + 64 + 256 + 1024 KB, so
        // one resource at full detail is over six times the whole budget. The
        // scheduler has to be choosing levels, not just choosing resources.
        const std::vector<std::size_t> levels = MipLevels(5, 4 * 1024);
        std::vector<StreamId> ids;
        for (int i = 0; i < 40; ++i)
            ids.push_back(s->Add(Vec3{float(i) * 10.0f, 0.0f, 0.0f}, 5.0f, levels));

        for (int frame = 0; frame < 12; ++frame) {
            s->Update(Vec3{0.0f, 0.0f, 0.0f});
            DrainAll(*s);
        }
        const Streamer::Stats st = s->GetStats();
        std::printf("    %zu KB resident of a %zu KB budget, %d of 40 resources, "
                    "%llu loads\n",
                    st.resident_bytes / 1024, cfg.budget / 1024, st.resident,
                    (unsigned long long)st.loads);
        Check(st.resident_bytes <= cfg.budget,
              "resident bytes never exceed the budget");
        Check(st.resident > 0, "and something is actually resident");
        // The budget must be USED, not merely respected. A scheduler that
        // loaded nothing would pass the check above.
        Check(st.resident_bytes > cfg.budget / 2,
              "and most of the budget is spent rather than left idle");

        // PRIORITY. The viewer is at the origin and the resources march away
        // along x, so residency has to fall off with distance -- monotonically,
        // because they are otherwise identical.
        int nearest = s->ResidentLevel(ids[0]);
        int furthest = s->ResidentLevel(ids[39]);
        std::printf("    nearest resource at level %d, furthest at level %d\n",
                    nearest, furthest);
        Check(nearest > furthest, "the nearest resource has more detail than the furthest");
        bool monotonic = true;
        for (std::size_t i = 1; i < ids.size(); ++i)
            if (s->ResidentLevel(ids[i]) > s->ResidentLevel(ids[i - 1]))
                monotonic = false;
        Check(monotonic, "and detail falls off monotonically with distance");
    }

    {
        std::printf("\npriority is angular size, not distance\n");
        StreamConfig cfg;
        cfg.threads = 0;
        cfg.budget = 100 * 1024;
        auto s = Streamer::Create(cfg, loader);
        const std::vector<std::size_t> levels = MipLevels(4, 4 * 1024);
        // A teacup ten metres away and a mountain two kilometres away. Sorting
        // by distance puts the teacup first and is wrong: the mountain fills a
        // quarter of the screen and the teacup fills none of it.
        const StreamId cup = s->Add(Vec3{10.0f, 0.0f, 0.0f}, 0.05f, levels);
        const StreamId mountain = s->Add(Vec3{2000.0f, 0.0f, 0.0f}, 400.0f, levels);
        for (int i = 0; i < 8; ++i) {
            s->Update(Vec3{0.0f, 0.0f, 0.0f});
            DrainAll(*s);
        }
        std::printf("    teacup (r=0.05 at 10 m) level %d, "
                    "mountain (r=400 at 2 km) level %d\n",
                    s->ResidentLevel(cup), s->ResidentLevel(mountain));
        Check(s->ResidentLevel(mountain) > s->ResidentLevel(cup),
              "the mountain outranks the nearer teacup");
    }

    {
        std::printf("\nlevels arrive smallest first\n");
        std::vector<int> order;
        StreamConfig cfg;
        cfg.threads = 0;
        cfg.budget = 10 * 1024 * 1024;
        auto s = Streamer::Create(cfg, [&](StreamId, int level,
                                           std::vector<std::uint8_t>& out) {
            order.push_back(level);
            out.assign(4, 0);
            return true;
        });
        const std::vector<std::size_t> levels = MipLevels(5, 1024);
        const StreamId id = s->Add(Vec3{1.0f, 0.0f, 0.0f}, 4.0f, levels);
        for (int i = 0; i < 10; ++i) {
            s->Update(Vec3{0.0f, 0.0f, 0.0f});
            DrainAll(*s);
        }
        std::printf("    load order:");
        for (int l : order) std::printf(" %d", l);
        std::printf("  (resident level %d)\n", s->ResidentLevel(id));
        // One level per frame, in order. Jumping straight to the target would
        // leave the object with nothing on screen for the whole load, where
        // this way the 1 KB mip is usable almost immediately.
        bool ascending = true;
        for (std::size_t i = 0; i < order.size(); ++i)
            if (order[i] != int(i)) ascending = false;
        Check(ascending && !order.empty(),
              "each level is requested exactly once, in ascending order");
        Check(s->ResidentLevel(id) == 4, "and the chain reaches the top");
    }

    {
        std::printf("\nhysteresis: standing on a boundary does not thrash\n");
        StreamConfig cfg;
        cfg.threads = 0;
        cfg.budget = 4 * 1024 * 1024;  // roomy: this is about the LEVEL boundary
        auto s = Streamer::Create(cfg, loader);
        const std::vector<std::size_t> levels = MipLevels(6, 1024);
        const StreamId id = s->Add(Vec3{0.0f, 0.0f, 0.0f}, 4.0f, levels);

        // FIND the boundary rather than guessing one. Jittering at an arbitrary
        // distance tests nothing: the target level only changes at particular
        // distances, and a few centimetres anywhere else leaves it untouched
        // whether there is hysteresis or not. That is what made the first
        // version of this check pass with the hysteresis removed.
        // A PAIR of distances that straddle a boundary, found by asking the
        // streamer rather than by computing one. Two things went wrong in the
        // first attempt at this: the search started at the resource's own
        // radius, where priority is clamped to 1 and the first "boundary" it
        // found was that clamp; and the jitter was a fixed fraction of the
        // distance, which at a boundary found elsewhere did not straddle it.
        // Both made the check pass with the hysteresis deleted.
        float lo_d = 0.0f, hi_d = 0.0f;
        for (float d = 12.0f; d < 400.0f; d *= 1.01f) {
            s->Update(Vec3{d, 0.0f, 0.0f});
            const int a = s->TargetLevel(id);
            s->Update(Vec3{d * 1.01f, 0.0f, 0.0f});
            const int b = s->TargetLevel(id);
            if (a != b) {
                lo_d = d;
                hi_d = d * 1.01f;
                break;
            }
        }
        std::printf("    the target level changes between %.2f m and %.2f m\n",
                    double(lo_d), double(hi_d));
        Check(lo_d > 0.0f, "there is a level boundary to stand on");

        // Settle on the near side, then step back and forth across it.
        for (int i = 0; i < 30; ++i) {
            s->Update(Vec3{lo_d, 0.0f, 0.0f});
            DrainAll(*s);
        }
        const Streamer::Stats before = s->GetStats();
        for (int i = 0; i < 200; ++i) {
            s->Update(Vec3{i % 2 ? hi_d : lo_d, 0.0f, 0.0f});
            DrainAll(*s);
        }
        const Streamer::Stats after = s->GetStats();
        std::printf("    200 frames stepping across it: %llu loads, %llu evictions\n",
                    (unsigned long long)(after.loads - before.loads),
                    (unsigned long long)(after.evictions - before.evictions));
        // Without the margin this is 100 evictions and 100 reloads -- the
        // picture stays correct throughout, so the only symptom is a disk that
        // never stops and a frame time that spikes at random.
        Check(after.evictions - before.evictions == 0,
              "crossing a level boundary every frame causes no evictions");
        Check(after.loads - before.loads == 0, "and no reloads");
    }

    {
        std::printf("\nequal priorities resolve the same way every time\n");
        StreamConfig cfg;
        cfg.threads = 0;
        cfg.budget = 24 * 1024;  // tight: not everything can be resident
        const std::vector<std::size_t> levels = MipLevels(3, 4 * 1024);
        // TWELVE resources at exactly the same distance and radius, so their
        // priorities are bit-identical and the sort has nothing to separate
        // them but the tie-break. With the tie-break gone the order is
        // unspecified, it differs between runs, and the resources that lose are
        // evicted and reloaded forever -- a thrash with no camera movement at
        // all behind it.
        std::vector<int> first_run;
        for (int run = 0; run < 2; ++run) {
            auto s = Streamer::Create(cfg, loader);
            std::vector<StreamId> ids;
            for (int i = 0; i < 12; ++i)
                ids.push_back(s->Add(Vec3{0.0f, 0.0f, 40.0f}, 6.0f, levels));
            for (int f = 0; f < 20; ++f) {
                s->Update(Vec3{0.0f, 0.0f, 0.0f});
                DrainAll(*s);
            }
            std::vector<int> got;
            for (StreamId id : ids) got.push_back(s->ResidentLevel(id));
            if (run == 0) {
                first_run = got;
                std::printf("    residency across 12 identical resources:");
                for (int l : got) std::printf(" %d", l);
                std::printf("\n");
                // Settle, then keep going: an unstable tie-break shows up as
                // continuing churn with a completely static camera.
                const Streamer::Stats a = s->GetStats();
                for (int f = 0; f < 60; ++f) {
                    s->Update(Vec3{0.0f, 0.0f, 0.0f});
                    DrainAll(*s);
                }
                const Streamer::Stats b = s->GetStats();
                std::printf("    60 more frames with a static camera: %llu loads, "
                            "%llu evictions\n",
                            (unsigned long long)(b.loads - a.loads),
                            (unsigned long long)(b.evictions - a.evictions));
                Check(b.loads - a.loads == 0 && b.evictions - a.evictions == 0,
                      "a static camera over identical resources causes no churn");
            } else {
                Check(got == first_run,
                      "and two runs pick exactly the same ones");
            }
        }
    }

    {
        std::printf("\nand moving away really does evict\n");
        StreamConfig cfg;
        cfg.threads = 0;
        cfg.budget = 60 * 1024;
        auto s = Streamer::Create(cfg, loader);
        const std::vector<std::size_t> levels = MipLevels(5, 2 * 1024);
        const StreamId near_id = s->Add(Vec3{0.0f, 0.0f, 0.0f}, 8.0f, levels);
        for (int i = 0; i < 15; ++i) {
            s->Update(Vec3{5.0f, 0.0f, 0.0f});
            DrainAll(*s);
        }
        const int close = s->ResidentLevel(near_id);
        const std::size_t close_bytes = s->GetStats().resident_bytes;
        for (int i = 0; i < 15; ++i) {
            s->Update(Vec3{4000.0f, 0.0f, 0.0f});
            DrainAll(*s);
        }
        const int far = s->ResidentLevel(near_id);
        std::printf("    5 m away: level %d, %zu KB.  4 km away: level %d, %zu KB\n",
                    close, close_bytes / 1024, far,
                    s->GetStats().resident_bytes / 1024);
        Check(far < close, "walking four kilometres away drops levels");
        Check(s->GetStats().resident_bytes < close_bytes,
              "and the memory comes back");
    }

    {
        std::printf("\na failing load is retried once, not forever\n");
        int attempts = 0;
        StreamConfig cfg;
        cfg.threads = 0;
        cfg.budget = 1024 * 1024;
        auto s = Streamer::Create(cfg, [&](StreamId, int, std::vector<std::uint8_t>&) {
            ++attempts;
            return false;
        });
        const std::vector<std::size_t> levels = MipLevels(3, 1024);
        s->Add(Vec3{0.0f, 0.0f, 0.0f}, 4.0f, levels);
        for (int i = 0; i < 50; ++i) {
            s->Update(Vec3{1.0f, 0.0f, 0.0f});
            DrainAll(*s);
        }
        std::printf("    50 frames against a missing file: %d attempts\n", attempts);
        Check(attempts == 1,
              "a missing file costs one attempt, not one per frame");
        Check(s->GetStats().failures == 1, "and is counted");
    }

    {
        std::printf("\nthreaded, and it agrees with the synchronous answer\n");
        // THE POINT of this one: threads must change the timing and nothing
        // else. Running the same camera path with workers and without, and
        // comparing the settled residency exactly, is the only way to know that
        // -- a race in the scheduler produces a plausible residency set, not a
        // crash.
        const std::vector<std::size_t> levels = MipLevels(5, 4 * 1024);
        const auto run = [&](int threads) {
            StreamConfig cfg;
            cfg.threads = threads;
            cfg.budget = 300 * 1024;
            auto s = Streamer::Create(cfg, loader);
            std::vector<StreamId> ids;
            for (int i = 0; i < 30; ++i)
                ids.push_back(
                    s->Add(Vec3{float(i) * 7.0f, float(i % 5), 0.0f}, 4.0f, levels));
            for (int f = 0; f < 25; ++f) {
                s->Update(Vec3{float(f) * 2.0f, 0.0f, 0.0f});
                s->Wait();
                DrainAll(*s);
            }
            std::vector<int> out;
            for (StreamId id : ids) out.push_back(s->ResidentLevel(id));
            return std::pair<std::vector<int>, std::size_t>{
                out, s->GetStats().resident_bytes};
        };
        const auto [sync_levels, sync_bytes] = run(0);
        const auto [par_levels, par_bytes] = run(4);
        int differing = 0;
        for (std::size_t i = 0; i < sync_levels.size(); ++i)
            if (sync_levels[i] != par_levels[i]) ++differing;
        std::printf("    %zu KB vs %zu KB resident, %d of %zu levels differ\n",
                    sync_bytes / 1024, par_bytes / 1024, differing,
                    sync_levels.size());
        Check(differing == 0, "four worker threads reach the identical residency");
        Check(sync_bytes == par_bytes, "down to the byte");
    }

    std::printf(g_failures == 0 ? "\nstream_test: all checks passed\n"
                                : "\nstream_test: %d FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
