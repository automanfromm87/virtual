// Pure C++20. What "a frame" means, in one place.
//
// Every windowed app in this engine had invented its own answer: one clamped dt
// to 0.1 s, one had no clock at all, one clamped somewhere else. That is not a
// tidiness problem — a simulation whose timing policy depends on which demo is
// hosting it cannot be reasoned about, and the bugs it produces (a body
// tunnelling after a window drag, an animation that jumps on the first frame)
// look like physics bugs.
#pragma once

#include <cstdint>

namespace eng::app {

class Clock {
  public:
    // Longest step the rest of the engine will ever be handed.
    //
    // A frame is not always a frame: dragging a window, hitting a breakpoint or
    // waking from sleep can put seconds between two Tick calls. Passing that
    // through means every moving thing teleports, and anything doing collision
    // moves further than its own size in one step. Clamping loses real time on
    // purpose — a stall is not a simulation event.
    float max_dt = 0.1f;

    // `now` is any monotonically increasing clock, in seconds. The first call
    // establishes the origin and yields dt = 0: there is no previous frame to
    // measure against, and inventing one is how demos get a jolt on startup.
    float Tick(double now);

    // Frozen time still advances the frame counter and still reports the real
    // elapsed dt, it just hands out zero. A paused app must keep drawing and
    // keep responding to input.
    void SetPaused(bool paused) { paused_ = paused; }
    [[nodiscard]] bool Paused() const { return paused_; }

    // Seconds to advance the world by. Zero while paused, and never above
    // max_dt.
    [[nodiscard]] float Dt() const { return dt_; }
    // Wall time this frame took, ignoring both the clamp and the pause. For
    // measuring, not for simulating — using this to move things is exactly the
    // mistake max_dt exists to prevent.
    [[nodiscard]] float RawDt() const { return raw_dt_; }
    // Accumulated Dt(), so it does not run while paused. Anything driven off
    // this stays in step with the simulation rather than with the wall.
    [[nodiscard]] float Total() const { return total_; }
    [[nodiscard]] std::uint64_t Frame() const { return frame_; }

    // Exponentially smoothed, from RawDt. A per-frame reciprocal is unreadable.
    [[nodiscard]] float Fps() const;

  private:
    bool started_ = false;
    bool paused_ = false;
    double last_ = 0.0;
    float dt_ = 0.0f;
    float raw_dt_ = 0.0f;
    float total_ = 0.0f;
    float smoothed_ = 0.0f;
    std::uint64_t frame_ = 0;
};

}  // namespace eng::app
