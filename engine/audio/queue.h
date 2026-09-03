// A single-producer, single-consumer ring of fixed-size commands.
//
// The only thing that crosses between the game thread and the audio thread.
//
// WHY not a mutex: the audio callback has a hard deadline of a few
// milliseconds and misses it audibly. A mutex it has to wait for is a dropout,
// and the wait need not even be long -- the game thread being descheduled while
// holding the lock is enough. Lock-free here is not an optimisation; it is the
// requirement.
//
// Single producer and single consumer is what makes it this short. There is
// exactly one game thread posting and exactly one audio thread draining, so
// each index is written by one thread only and the atomics are a release and an
// acquire rather than a compare-exchange loop.
#pragma once

#include <array>
#include <atomic>
#include <cstddef>

namespace eng::audio {

template <class T, std::size_t kCapacity>
class SpscQueue {
  public:
    static_assert((kCapacity & (kCapacity - 1)) == 0,
                  "capacity must be a power of two, so the wrap is a mask");

    // Producer side. False when the ring is full: the command is DROPPED and
    // the caller told, rather than the producer blocking -- a game thread that
    // waits on the audio thread has the deadlock the other way round.
    [[nodiscard]] bool Push(const T& value) {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t next = (head + 1) & (kCapacity - 1);
        // Acquire on the consumer's index: everything the consumer did before
        // publishing it must be visible before the slot is reused.
        if (next == tail_.load(std::memory_order_acquire)) return false;
        slots_[head] = value;
        // RELEASE, so the write above cannot be reordered past the publish. Get
        // this wrong and the consumer occasionally reads a slot the producer
        // has not finished filling -- on a weakly ordered machine, which every
        // Apple Silicon machine is.
        head_.store(next, std::memory_order_release);
        return true;
    }

    // Consumer side.
    [[nodiscard]] bool Pop(T* out) {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        if (tail == head_.load(std::memory_order_acquire)) return false;
        *out = slots_[tail];
        tail_.store((tail + 1) & (kCapacity - 1), std::memory_order_release);
        return true;
    }

    [[nodiscard]] std::size_t Size() const {
        const std::size_t h = head_.load(std::memory_order_relaxed);
        const std::size_t t = tail_.load(std::memory_order_relaxed);
        return (h - t) & (kCapacity - 1);
    }
    [[nodiscard]] static constexpr std::size_t Capacity() { return kCapacity - 1; }

  private:
    // A cache line. Not std::hardware_destructive_interference_size, which is
    // a constant the compiler bakes in and which libc++ still does not define;
    // 64 is right on Apple Silicon and on x86-64 both.
    static constexpr std::size_t kLine = 64;

    // One slot is always left empty, which is what makes full and empty
    // distinguishable without a third variable that both threads would have to
    // agree on.
    alignas(kLine) std::array<T, kCapacity> slots_{};
    // EACH INDEX ON ITS OWN LINE. Written 8 bytes apart -- which is what
    // declaring them adjacently gives -- the producer's store to head_ dirties
    // the line holding tail_, and the audio thread's next load of either goes
    // through a coherence miss. That is a stall on the one thread with a hard
    // deadline, and it is exactly the cost this file's header argues a mutex
    // would impose. A few hundred bytes of padding on a ~49 KB object.
    alignas(kLine) std::atomic<std::size_t> head_{0};
    alignas(kLine) std::atomic<std::size_t> tail_{0};
};

}  // namespace eng::audio
