// The one thing that crosses between the game thread and the audio thread.
//
// A ring buffer is easy to write so that it works until it wraps, works until
// it fills, or works on x86 and fails on ARM. The first two are deterministic
// and checked exactly; the third is checked by hammering it from two real
// threads, which cannot prove correctness but does reliably catch a missing
// release barrier on this hardware.

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

#include "engine/audio/queue.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

struct Item {
    int a = 0;
    int b = 0;
};

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::printf("spsc queue\n");

    // --- empty, full, and the difference ---------------------------------------
    {
        eng::audio::SpscQueue<Item, 8> q;
        Item out;
        Check(!q.Pop(&out), "an empty queue pops nothing");
        Check(q.Size() == 0, "and is empty");

        // Capacity is one LESS than the array, which is what keeps full and
        // empty distinguishable without a third variable both threads would
        // have to agree about.
        int pushed = 0;
        while (q.Push(Item{pushed, -pushed})) ++pushed;
        std::printf("    an 8-slot ring accepted %d items\n", pushed);
        Check(pushed == 7, "a ring of 8 holds 7");
        Check(q.Size() == 7, "and says so");
        Check(!q.Push(Item{99, 99}), "and refuses the eighth rather than wrapping");

        for (int i = 0; i < 7; ++i) {
            Check(q.Pop(&out), "each item comes back");
            if (out.a != i || out.b != -i) Check(false, "in order and intact");
        }
        Check(!q.Pop(&out), "and then it is empty again");
    }

    // --- wrapping -----------------------------------------------------------------
    //
    // The case that works right up until the indices pass the end of the array.
    // Pushing and popping alternately for many times the capacity walks every
    // slot repeatedly.
    {
        eng::audio::SpscQueue<Item, 8> q;
        Item out;
        bool ok = true;
        for (int i = 0; i < 10000; ++i) {
            if (!q.Push(Item{i, i * 2})) ok = false;
            if (!q.Pop(&out) || out.a != i || out.b != i * 2) ok = false;
        }
        Check(ok, "10000 push/pop pairs wrap cleanly");
    }
    {
        // ...and partially full, so head and tail wrap at different times.
        eng::audio::SpscQueue<Item, 8> q;
        Item out;
        bool ok = true;
        int next_in = 0, next_out = 0;
        for (int round = 0; round < 3000; ++round) {
            for (int k = 0; k < 3; ++k)
                if (q.Push(Item{next_in, 0})) ++next_in;
            for (int k = 0; k < 3; ++k)
                if (q.Pop(&out)) {
                    if (out.a != next_out) ok = false;
                    ++next_out;
                }
        }
        std::printf("    interleaved: %d in, %d out\n", next_in, next_out);
        Check(ok, "an interleaved producer and consumer stay in order");
    }

    // --- two real threads -------------------------------------------------------
    //
    // Cannot prove the memory ordering is right. Does catch a missing release
    // on this hardware, which is weakly ordered -- the store of the payload and
    // the store of the index can be observed out of order, and the consumer
    // then reads a slot the producer has not finished writing.
    //
    // The payload is two fields that must agree. A torn read shows up as b
    // not being a's double, which a single-field payload could never reveal.
    {
        eng::audio::SpscQueue<Item, 1024> q;
        constexpr int kCount = 200000;
        std::atomic<int> torn{0};
        std::atomic<int> received{0};

        std::thread consumer([&] {
            Item out;
            int seen = 0;
            while (seen < kCount) {
                if (!q.Pop(&out)) continue;
                if (out.b != out.a * 2) torn.fetch_add(1);
                ++seen;
            }
            received.store(seen);
        });

        int sent = 0;
        while (sent < kCount)
            if (q.Push(Item{sent, sent * 2})) ++sent;
        consumer.join();

        std::printf("    %d items across two threads, %d torn\n",
                    received.load(), torn.load());
        Check(received.load() == kCount, "every item crossed");
        Check(torn.load() == 0, "and none was read half-written");
    }

    std::printf(g_failures == 0 ? "\nqueue_test: all checks passed\n"
                                : "\nqueue_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
