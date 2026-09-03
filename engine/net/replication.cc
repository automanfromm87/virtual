#include "engine/net/replication.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>

namespace eng::net {
namespace {

struct SnapshotHeader {
    std::uint32_t tick;
    // 0 means "this is complete, not a delta". The client uses it to find the
    // baseline; a delta whose baseline the client no longer has is undecodable
    // and has to be dropped rather than applied to the wrong one.
    std::uint32_t baseline_tick;
    std::uint16_t changed_count;
    std::uint16_t removed_count;
};
static_assert(sizeof(SnapshotHeader) == 12, "the snapshot header must not drift");

// The wire form of an entity. Written as raw floats rather than quantised.
//
// QUANTISING -- a position as three 16-bit fixed-point values, a rotation as
// three of four quaternion components -- is the next thing to do here and would
// roughly halve this. It is not free: quantisation error has to be small enough
// that the client's copy does not visibly disagree with the server's, and the
// bounds it is relative to have to be part of the protocol. Full floats are
// honest and correct, and the delta above already removes the entities that did
// not move, which is the larger saving.
struct WireEntity {
    std::uint32_t id;
    float px, py, pz;
    float qx, qy, qz, qw;
    float vx, vy, vz;
    std::uint32_t flags;
};
static_assert(sizeof(WireEntity) == 48, "the wire entity must not drift");

void Append(std::vector<std::uint8_t>* out, const void* data, std::size_t bytes) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    out->insert(out->end(), p, p + bytes);
}

WireEntity ToWire(const EntityState& e) {
    WireEntity w{};
    w.id = e.id;
    w.px = e.position.x; w.py = e.position.y; w.pz = e.position.z;
    w.qx = e.rotation.x; w.qy = e.rotation.y; w.qz = e.rotation.z; w.qw = e.rotation.w;
    w.vx = e.velocity.x; w.vy = e.velocity.y; w.vz = e.velocity.z;
    w.flags = e.flags;
    return w;
}

EntityState FromWire(const WireEntity& w) {
    EntityState e;
    e.id = w.id;
    e.position = Vec3{w.px, w.py, w.pz};
    e.rotation = Quat{w.qx, w.qy, w.qz, w.qw};
    e.velocity = Vec3{w.vx, w.vy, w.vz};
    e.flags = w.flags;
    return e;
}

}  // namespace

bool EntityState::operator==(const EntityState& o) const {
    // EXACT, not approximate. A tolerance here would mean an entity drifting by
    // less than the tolerance every tick is never sent, and the error
    // accumulates forever with nothing to correct it -- an object that slowly
    // walks away from where the server has it.
    return id == o.id && position.x == o.position.x && position.y == o.position.y &&
           position.z == o.position.z && rotation.x == o.rotation.x &&
           rotation.y == o.rotation.y && rotation.z == o.rotation.z &&
           rotation.w == o.rotation.w && velocity.x == o.velocity.x &&
           velocity.y == o.velocity.y && velocity.z == o.velocity.z &&
           flags == o.flags;
}

const EntityState* Snapshot::Find(std::uint32_t id) const {
    const auto it = std::lower_bound(
        entities.begin(), entities.end(), id,
        [](const EntityState& e, std::uint32_t v) { return e.id < v; });
    if (it == entities.end() || it->id != id) return nullptr;
    return &*it;
}

std::vector<std::uint8_t> EncodeSnapshot(const Snapshot& current,
                                         const Snapshot* baseline) {
    std::vector<std::uint8_t> out;
    std::vector<WireEntity> changed;
    std::vector<std::uint32_t> removed;

    for (const EntityState& e : current.entities) {
        const EntityState* was = baseline ? baseline->Find(e.id) : nullptr;
        if (was && *was == e) continue;  // unchanged: costs nothing
        changed.push_back(ToWire(e));
    }
    if (baseline)
        for (const EntityState& e : baseline->entities)
            if (!current.Find(e.id)) removed.push_back(e.id);

    SnapshotHeader h{};
    h.tick = current.tick;
    h.baseline_tick = baseline ? baseline->tick : 0;
    h.changed_count = std::uint16_t(changed.size());
    h.removed_count = std::uint16_t(removed.size());

    out.reserve(sizeof(h) + changed.size() * sizeof(WireEntity) +
                removed.size() * sizeof(std::uint32_t));
    Append(&out, &h, sizeof(h));
    if (!changed.empty())
        Append(&out, changed.data(), changed.size() * sizeof(WireEntity));
    if (!removed.empty())
        Append(&out, removed.data(), removed.size() * sizeof(std::uint32_t));
    return out;
}

bool PeekSnapshotHeader(std::span<const std::uint8_t> bytes,
                        std::uint32_t* out_tick, std::uint32_t* out_baseline_tick) {
    if (bytes.size() < sizeof(SnapshotHeader)) return false;
    SnapshotHeader h{};
    std::memcpy(&h, bytes.data(), sizeof(h));
    if (out_tick) *out_tick = h.tick;
    if (out_baseline_tick) *out_baseline_tick = h.baseline_tick;
    return true;
}

bool DecodeSnapshot(std::span<const std::uint8_t> bytes, const Snapshot* baseline,
                    Snapshot* out) {
    if (!out || bytes.size() < sizeof(SnapshotHeader)) return false;
    SnapshotHeader h{};
    std::memcpy(&h, bytes.data(), sizeof(h));

    // THE BASELINE MUST BE THE ONE THE DELTA WAS BUILT AGAINST. Applying a
    // delta to a different snapshot produces a world that is plausible and
    // wrong, and every subsequent delta compounds it -- so this is refused
    // rather than approximated.
    if (h.baseline_tick != 0 && (!baseline || baseline->tick != h.baseline_tick))
        return false;
    if (h.baseline_tick == 0 && baseline != nullptr && baseline->tick != 0) {
        // A full snapshot with a baseline supplied is fine -- the baseline is
        // simply ignored -- but say so rather than silently merging them.
        baseline = nullptr;
    }

    const std::size_t need = sizeof(h) + std::size_t(h.changed_count) * sizeof(WireEntity) +
                             std::size_t(h.removed_count) * sizeof(std::uint32_t);
    if (bytes.size() < need) return false;

    out->tick = h.tick;
    out->entities.clear();
    if (baseline) out->entities = baseline->entities;

    const auto* wire = reinterpret_cast<const WireEntity*>(bytes.data() + sizeof(h));
    for (int i = 0; i < h.changed_count; ++i) {
        WireEntity w{};
        std::memcpy(&w, wire + i, sizeof(w));
        const EntityState e = FromWire(w);
        const auto it = std::lower_bound(
            out->entities.begin(), out->entities.end(), e.id,
            [](const EntityState& a, std::uint32_t v) { return a.id < v; });
        if (it != out->entities.end() && it->id == e.id) *it = e;
        else out->entities.insert(it, e);
    }

    const auto* removed = reinterpret_cast<const std::uint32_t*>(
        bytes.data() + sizeof(h) + std::size_t(h.changed_count) * sizeof(WireEntity));
    for (int i = 0; i < h.removed_count; ++i) {
        std::uint32_t id = 0;
        std::memcpy(&id, removed + i, sizeof(id));
        const auto it = std::lower_bound(
            out->entities.begin(), out->entities.end(), id,
            [](const EntityState& a, std::uint32_t v) { return a.id < v; });
        if (it != out->entities.end() && it->id == id) out->entities.erase(it);
    }
    return true;
}

// ----------------------------------------------------------------- the server

struct SnapshotHistory::Impl {
    std::deque<Snapshot> snapshots;
    int capacity = 64;
};

SnapshotHistory::SnapshotHistory(int capacity) : impl_(std::make_unique<Impl>()) {
    impl_->capacity = std::max(2, capacity);
}
SnapshotHistory::~SnapshotHistory() = default;

void SnapshotHistory::Add(const Snapshot& s) {
    Impl& im = *impl_;
    im.snapshots.push_back(s);
    // SORTED BY ID on the way in, once, so every Find is a binary search and
    // every delta is a merge. A history that stored them unsorted would sort
    // them again on every encode, per client.
    std::sort(im.snapshots.back().entities.begin(), im.snapshots.back().entities.end(),
              [](const EntityState& a, const EntityState& b) { return a.id < b.id; });
    while (int(im.snapshots.size()) > im.capacity) im.snapshots.pop_front();
}

const Snapshot* SnapshotHistory::Get(std::uint32_t tick) const {
    for (const Snapshot& s : impl_->snapshots)
        if (s.tick == tick) return &s;
    return nullptr;
}

const Snapshot* SnapshotHistory::Newest() const {
    return impl_->snapshots.empty() ? nullptr : &impl_->snapshots.back();
}

int SnapshotHistory::Count() const { return int(impl_->snapshots.size()); }

std::vector<std::uint8_t> SnapshotHistory::EncodeFor(std::uint32_t acked_tick) const {
    const Snapshot* newest = Newest();
    if (!newest) return {};
    // AGAINST WHAT THE CLIENT CONFIRMED, not against the last thing sent. A
    // delta against a snapshot that never arrived cannot be applied, and the
    // client would then be unable to decode anything until a full snapshot came
    // -- which, if the server kept deltaing against its own last send, would be
    // never.
    const Snapshot* baseline = acked_tick != 0 ? Get(acked_tick) : nullptr;
    return EncodeSnapshot(*newest, baseline);
}

// ----------------------------------------------------------------- the client

namespace {
struct Timed {
    Snapshot snapshot;
    float arrival = 0.0f;
};

Vec3 Lerp(const Vec3& a, const Vec3& b, float t) { return a + (b - a) * t; }
}  // namespace

struct InterpolationBuffer::Impl {
    InterpolationConfig config;
    std::deque<Timed> buffer;
    mutable bool starved = false;
};

InterpolationBuffer::InterpolationBuffer(const InterpolationConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}
InterpolationBuffer::~InterpolationBuffer() = default;

void InterpolationBuffer::Add(const Snapshot& s, float arrival_time) {
    Impl& im = *impl_;
    // OUT OF ORDER ARRIVALS ARE INSERTED, not appended. A snapshot that took a
    // slower route still describes a moment between two others, and dropping it
    // because it arrived late throws away data that is perfectly usable --
    // while appending it would leave the buffer unsorted and the search below
    // would straddle the wrong pair.
    if (!im.buffer.empty() && s.tick <= im.buffer.back().snapshot.tick) {
        for (const Timed& t : im.buffer)
            if (t.snapshot.tick == s.tick) return;  // a duplicate
        auto it = std::lower_bound(
            im.buffer.begin(), im.buffer.end(), s.tick,
            [](const Timed& t, std::uint32_t v) { return t.snapshot.tick < v; });
        im.buffer.insert(it, Timed{s, arrival_time});
    } else {
        im.buffer.push_back(Timed{s, arrival_time});
    }
    while (int(im.buffer.size()) > im.config.history) im.buffer.pop_front();
}

bool InterpolationBuffer::Sample(float now, Snapshot* out) const {
    const Impl& im = *impl_;
    if (!out || im.buffer.empty()) return false;
    im.starved = false;
    const float target = now - im.config.delay_seconds;

    // Before the oldest: nothing to interpolate from, so hold the oldest.
    if (target <= im.buffer.front().arrival) {
        *out = im.buffer.front().snapshot;
        im.starved = im.buffer.size() < 2;
        return true;
    }
    // After the newest: the buffer has run dry.
    if (target >= im.buffer.back().arrival) {
        const Timed& last = im.buffer.back();
        const float over = target - last.arrival;
        *out = last.snapshot;
        im.starved = true;
        // EXTRAPOLATE, briefly. Continuing indefinitely sends everything
        // sliding off in a straight line -- a player who stopped keeps walking
        // through a wall -- so past the limit the last state is held instead,
        // which reads as a freeze and is much easier to recognise as a network
        // problem.
        if (over <= im.config.max_extrapolation_seconds)
            for (EntityState& e : out->entities)
                e.position = e.position + e.velocity * over;
        return true;
    }

    // The pair straddling the target.
    std::size_t i = 0;
    while (i + 1 < im.buffer.size() && im.buffer[i + 1].arrival < target) ++i;
    const Timed& a = im.buffer[i];
    const Timed& b = im.buffer[i + 1];
    const float span = b.arrival - a.arrival;
    const float t = span > 1e-6f ? std::clamp((target - a.arrival) / span, 0.0f, 1.0f)
                                 : 0.0f;

    *out = a.snapshot;
    out->tick = t < 0.5f ? a.snapshot.tick : b.snapshot.tick;
    for (EntityState& e : out->entities) {
        const EntityState* next = b.snapshot.Find(e.id);
        // An entity present in one snapshot and not the other is spawning or
        // despawning. Holding its known state rather than interpolating toward
        // nothing is what stops a spawning object flying in from the origin.
        if (!next) continue;
        e.position = Lerp(e.position, next->position, t);
        e.velocity = Lerp(e.velocity, next->velocity, t);
        e.rotation = Normalize(Slerp(e.rotation, next->rotation, t));
    }
    return true;
}

int InterpolationBuffer::Count() const { return int(impl_->buffer.size()); }
std::uint32_t InterpolationBuffer::NewestTick() const {
    return impl_->buffer.empty() ? 0 : impl_->buffer.back().snapshot.tick;
}
bool InterpolationBuffer::Starved() const { return impl_->starved; }

// ------------------------------------------------------------------ prediction

struct Predictor::Impl {
    EntityState predicted;
    // The input AND the state it produced.
    //
    // Keeping the state is what makes the reported error mean anything. Without
    // it the only thing available to compare against the server is the CURRENT
    // prediction, which is legitimately ahead by every input the server has not
    // seen -- so a perfectly correct client reports an error equal to its own
    // input lag, and there is no way to tell that from a real disagreement.
    // Measured: 0.5 m of "error" on a client that had predicted every frame
    // exactly right.
    struct Step {
        InputCommand input;
        EntityState after;
    };
    std::deque<Step> pending;
    std::uint32_t last_acked = 0;
    int replays = 0;
};

Predictor::Predictor() : impl_(std::make_unique<Impl>()) {}
Predictor::~Predictor() = default;

void Predictor::Apply(const InputCommand& input, SimulateFn simulate) {
    if (!simulate) return;
    simulate(&impl_->predicted, input);
    impl_->pending.push_back(Impl::Step{input, impl_->predicted});
}

const EntityState& Predictor::Predicted() const { return impl_->predicted; }
void Predictor::SetPredicted(const EntityState& s) { impl_->predicted = s; }
int Predictor::PendingInputs() const { return int(impl_->pending.size()); }
std::uint32_t Predictor::LastAcked() const { return impl_->last_acked; }
int Predictor::ReplayCount() const { return impl_->replays; }

float Predictor::Reconcile(std::uint32_t acked_sequence,
                           const EntityState& server_state, SimulateFn simulate) {
    Impl& im = *impl_;
    if (!simulate) return 0.0f;
    im.last_acked = acked_sequence;

    // WHAT WE PREDICTED AT THAT SEQUENCE, which is the only thing comparable
    // with what the server sent. The current prediction includes inputs the
    // server has not seen and is supposed to differ.
    float error = 0.0f;
    bool found = false;
    for (const Impl::Step& s : im.pending)
        if (s.input.sequence == acked_sequence) {
            error = Length(s.after.position - server_state.position);
            found = true;
            break;
        }
    // No record: the acknowledgement is for an input from before the history,
    // which happens after a long stall. Nothing to compare, and reporting the
    // current lead as an error would be the bug this replaces.
    (void)found;

    // Everything the server has already seen is history.
    while (!im.pending.empty() && im.pending.front().input.sequence <= acked_sequence)
        im.pending.pop_front();

    // SNAP to the server, then REPLAY. The alternative -- smoothing toward the
    // server's answer without replaying -- leaves the pending inputs applied to
    // a state the server never had, so the error never actually goes away and
    // the client drifts further with every correction.
    im.predicted = server_state;
    for (Impl::Step& s : im.pending) {
        simulate(&im.predicted, s.input);
        // The replayed state replaces the one recorded when the input was
        // first applied, so a later reconciliation compares against what we
        // now believe rather than against a superseded guess.
        s.after = im.predicted;
        ++im.replays;
    }
    return error;
}

}  // namespace eng::net
