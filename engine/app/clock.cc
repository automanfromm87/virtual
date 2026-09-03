#include "engine/app/clock.h"

#include <algorithm>

namespace eng::app {

float Clock::Tick(double now) {
    ++frame_;
    if (!started_) {
        started_ = true;
        last_ = now;
        raw_dt_ = 0.0f;
        dt_ = 0.0f;
        return 0.0f;
    }
    // A clock that went backwards is a clock that was reset, not time running
    // in reverse. Zero is the only sane answer.
    raw_dt_ = float(std::max(now - last_, 0.0));
    last_ = now;
    dt_ = paused_ ? 0.0f : std::min(raw_dt_, max_dt);
    total_ += dt_;

    // Roughly a half-second window at 60 Hz. Seeded from the first real frame
    // so the readout does not spend a second climbing from zero.
    constexpr float kBlend = 0.05f;
    if (raw_dt_ > 0.0f) {
        smoothed_ = smoothed_ > 0.0f ? smoothed_ + (raw_dt_ - smoothed_) * kBlend
                                     : raw_dt_;
    }
    return dt_;
}

float Clock::Fps() const { return smoothed_ > 0.0f ? 1.0f / smoothed_ : 0.0f; }

}  // namespace eng::app
