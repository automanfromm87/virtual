// Snapshot replication: getting the server's world onto the client's screen.
//
// THE THREE PROBLEMS, and why each needs its own mechanism.
//
// 1. BANDWIDTH. Sending every entity's full state at 30 Hz is more than a
//    connection has. DELTA COMPRESSION sends only what changed since a snapshot
//    the client has confirmed it holds -- and against a CONFIRMED one, not the
//    last one sent, because a delta against a snapshot that was lost cannot be
//    applied and every snapshot after it would be undecodable too.
//
// 2. JITTER. Snapshots arrive unevenly and sometimes not at all. Drawing each
//    one as it lands makes everything stutter. INTERPOLATION draws the world
//    slightly in the past -- far enough back that there is always a snapshot on
//    each side to interpolate between -- so motion is smooth at the cost of
//    seeing everything a fraction of a second late.
//
// 3. LATENCY on your own character. Waiting for the server to confirm your own
//    movement puts a round trip between pressing a key and moving, which at
//    100 ms is the difference between a game that feels responsive and one that
//    does not. PREDICTION applies your input immediately and RECONCILIATION
//    fixes it up when the server disagrees: rewind to the server's state and
//    replay every input it had not yet seen.
//
// WHAT IS NOT HERE: lag compensation -- rewinding other players to where the
// shooter saw them. It belongs on the server, needs the server to keep a
// history of every entity, and is a design decision (whose reality wins) rather
// than a mechanism.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "engine/core/math.h"

namespace eng::net {

// One replicated object. Deliberately fixed: a general property system would
// need reflection, and the state a game actually replicates per entity is
// almost always a transform plus a handful of numbers.
struct EntityState {
    std::uint32_t id = 0;
    Vec3 position{0.0f, 0.0f, 0.0f};
    Quat rotation;
    Vec3 velocity{0.0f, 0.0f, 0.0f};
    // Whatever the game wants: health, animation state, a team. Replicated
    // whole, so a change to any bit sends all four bytes.
    std::uint32_t flags = 0;

    [[nodiscard]] bool operator==(const EntityState&) const;
};

struct Snapshot {
    std::uint32_t tick = 0;
    std::vector<EntityState> entities;  // sorted by id

    [[nodiscard]] const EntityState* Find(std::uint32_t id) const;
};

// --- encoding -----------------------------------------------------------------

// Writes `current` as a delta against `baseline`, or in full when `baseline` is
// null.
//
// The encoding is a list of changed entities plus a list of removed ids. An
// entity whose state is identical to the baseline's costs nothing at all, which
// is what makes a world of mostly-still objects cheap -- and is why the
// comparison is exact rather than approximate: a position that differs in its
// last bit costs the same as one that moved a metre, but a "close enough"
// comparison would let error accumulate with nothing to correct it.
[[nodiscard]] std::vector<std::uint8_t> EncodeSnapshot(const Snapshot& current,
                                                       const Snapshot* baseline);

// Applies a delta to `baseline`, producing the full snapshot.
[[nodiscard]] bool DecodeSnapshot(std::span<const std::uint8_t> bytes,
                                  const Snapshot* baseline, Snapshot* out);

// The tick a packet says it is a delta against, without decoding it. The client
// needs this to find the right baseline, and the server needs it to know what
// the client has confirmed.
[[nodiscard]] bool PeekSnapshotHeader(std::span<const std::uint8_t> bytes,
                                      std::uint32_t* out_tick,
                                      std::uint32_t* out_baseline_tick);

// --- the server's side --------------------------------------------------------

class SnapshotHistory {
  public:
    explicit SnapshotHistory(int capacity = 64);
    ~SnapshotHistory();

    void Add(const Snapshot&);
    [[nodiscard]] const Snapshot* Get(std::uint32_t tick) const;
    [[nodiscard]] const Snapshot* Newest() const;
    [[nodiscard]] int Count() const;

    // Encodes the newest snapshot against whatever the client last confirmed.
    // `acked_tick` of 0 means it has nothing, so the result is a full snapshot.
    [[nodiscard]] std::vector<std::uint8_t> EncodeFor(std::uint32_t acked_tick) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// --- the client's side --------------------------------------------------------

struct InterpolationConfig {
    // How far behind the newest snapshot to draw, in seconds.
    //
    // Has to exceed the gap between snapshots plus the jitter, or the buffer
    // runs dry and motion stalls. Two snapshot intervals is the usual choice:
    // at 20 Hz that is 100 ms of added latency on everything except your own
    // character, which is what prediction is for.
    float delay_seconds = 0.1f;
    // Snapshots kept. Enough to cover the delay plus a burst of loss.
    int history = 32;
    // Beyond this much missing data the buffer stops extrapolating and holds
    // the last known state. Extrapolating indefinitely sends everything
    // sliding off in a straight line, which looks far worse than a pause.
    float max_extrapolation_seconds = 0.25f;
};

class InterpolationBuffer {
  public:
    explicit InterpolationBuffer(const InterpolationConfig& = {});
    ~InterpolationBuffer();

    // `arrival_time` is the local clock when the snapshot was received. The
    // server's own tick rate is not enough: what matters for smoothing is when
    // the data got HERE.
    void Add(const Snapshot&, float arrival_time);

    // The world as it should be drawn at `now`. Interpolates between the two
    // snapshots that straddle `now - delay`.
    [[nodiscard]] bool Sample(float now, Snapshot* out) const;

    [[nodiscard]] int Count() const;
    [[nodiscard]] std::uint32_t NewestTick() const;
    // Whether the last Sample had to extrapolate or hold. The number to put on
    // a connection-quality indicator: a buffer that is interpolating is healthy
    // however bad the ping, and one that is extrapolating is not.
    [[nodiscard]] bool Starved() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// --- prediction ---------------------------------------------------------------

// One frame of the local player's input, kept until the server confirms it.
struct InputCommand {
    std::uint32_t sequence = 0;
    float dt = 0.0f;
    Vec3 move{0.0f, 0.0f, 0.0f};
    std::uint32_t buttons = 0;
};

// Applies an input to a state. The SAME function must run on the client and the
// server, or reconciliation corrects a disagreement that is not the network's
// fault -- and the correction happens every frame, which reads as rubber
// banding on a perfect connection.
using SimulateFn = void (*)(EntityState*, const InputCommand&);

class Predictor {
  public:
    Predictor();
    ~Predictor();

    // Records an input and applies it immediately to the predicted state.
    void Apply(const InputCommand&, SimulateFn);
    [[nodiscard]] const EntityState& Predicted() const;
    void SetPredicted(const EntityState&);

    // The server has confirmed the state after `acked_sequence`. Discards the
    // inputs it has seen, snaps to its answer, and replays the rest.
    //
    // Returns how far the prediction was wrong, in metres. Zero on a correct
    // prediction, which on a deterministic simulation with no loss is the
    // normal case -- and a value that is never zero means the client and server
    // are running different code.
    float Reconcile(std::uint32_t acked_sequence, const EntityState& server_state,
                    SimulateFn);

    [[nodiscard]] int PendingInputs() const;
    [[nodiscard]] std::uint32_t LastAcked() const;
    // How many inputs have been replayed in total. A number that grows fast
    // means the connection is losing acknowledgements, and every replay is
    // wasted work.
    [[nodiscard]] int ReplayCount() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eng::net
