#include "engine/net/connection.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <map>

namespace eng::net {
namespace {

// The packet header, on the wire. Every field fixed-width and little-endian --
// which is what every machine this runs on already is, and the alternative is
// a byte-swap on every field for a portability nobody is asking for. Stated
// rather than assumed, because a protocol that silently depends on endianness
// is one that breaks on a platform port and not before.
struct PacketHeader {
    std::uint32_t protocol;
    std::uint16_t sequence;      // this packet's own number
    std::uint16_t ack;           // the newest sequence seen from the peer
    std::uint32_t ack_bits;      // and which of the previous 32 arrived
    std::uint16_t reliable_id;   // 0 when the packet carries no reliable message
    std::uint16_t payload_bytes;
};
static_assert(sizeof(PacketHeader) == 16, "the packet header must not drift");

constexpr std::size_t kHeaderBytes = sizeof(PacketHeader);
constexpr int kAckWindow = 32;

struct SentRecord {
    std::uint16_t sequence = 0;
    float sent_at = 0.0f;
    bool acked = false;
    bool counted = false;   // already tallied as acked or lost
    std::uint16_t reliable_id = 0;
};

// OutgoingReliable, not PendingReliable: Connection has a method of that name,
// and inside a member function the method wins -- so a bare `PendingReliable`
// there is a call to the accessor rather than a type.
struct OutgoingReliable {
    std::uint16_t id = 0;
    std::vector<std::uint8_t> payload;
    float last_sent = -1e9f;
    bool acked = false;
};

}  // namespace

struct Connection::Impl {
    Transport* transport = nullptr;
    Address peer;
    ConnectionConfig config;
    ConnectionStats stats;

    float now = 0.0f;
    float last_heard = 0.0f;
    bool heard_anything = false;

    std::uint16_t next_sequence = 1;
    // What we have received from the peer.
    std::uint16_t remote_newest = 0;
    std::uint32_t remote_bits = 0;
    bool have_remote = false;

    // Our own sent packets, for turning acks into round-trip samples.
    std::deque<SentRecord> sent;

    // Reliable, out.
    std::uint16_t next_reliable_id = 1;
    std::deque<OutgoingReliable> pending;

    // Reliable, in. Held until the gap before them is filled.
    std::uint16_t next_expected_reliable = 1;
    std::map<std::uint16_t, std::vector<std::uint8_t>> held;

    // The payloads handed out by the last Receive. Owned here so the spans in
    // Received stay valid until the next call -- which is the contract, and is
    // why Received holds a span rather than a vector: a caller that only wants
    // to look at a packet should not pay for a copy of it.
    std::vector<std::vector<std::uint8_t>> inbox;

    bool SendPacket(std::span<const std::uint8_t> payload, std::uint16_t reliable_id,
                    std::uint16_t* out_sequence);
    void OnAck(std::uint16_t ack, std::uint32_t bits);
    void RetireOldSent();
};

bool Connection::Impl::SendPacket(std::span<const std::uint8_t> payload,
                                  std::uint16_t reliable_id,
                                  std::uint16_t* out_sequence) {
    if (!transport) return false;
    if (kHeaderBytes + payload.size() > std::size_t(kMaxPacketBytes)) return false;

    PacketHeader h{};
    h.protocol = config.protocol_id;
    h.sequence = next_sequence++;
    // Sequence 0 is reserved as "no sequence", so Send can return it to mean
    // failure without it ever being a real packet.
    if (next_sequence == 0) next_sequence = 1;
    h.ack = remote_newest;
    h.ack_bits = remote_bits;
    h.reliable_id = reliable_id;
    h.payload_bytes = std::uint16_t(payload.size());

    std::uint8_t buffer[kMaxPacketBytes];
    std::memcpy(buffer, &h, kHeaderBytes);
    if (!payload.empty())
        std::memcpy(buffer + kHeaderBytes, payload.data(), payload.size());

    if (!transport->Send(peer, {buffer, kHeaderBytes + payload.size()})) return false;

    SentRecord r;
    r.sequence = h.sequence;
    r.sent_at = now;
    r.reliable_id = reliable_id;
    sent.push_back(r);
    ++stats.packets_sent;
    if (out_sequence) *out_sequence = h.sequence;
    return true;
}

void Connection::Impl::OnAck(std::uint16_t ack, std::uint32_t bits) {
    for (SentRecord& r : sent) {
        if (r.acked) continue;
        bool is_acked = r.sequence == ack;
        if (!is_acked && SequenceNewer(ack, r.sequence)) {
            const std::uint16_t distance = std::uint16_t(ack - r.sequence);
            if (distance >= 1 && distance <= kAckWindow)
                is_acked = (bits & (1u << (distance - 1))) != 0;
        }
        if (!is_acked) continue;
        r.acked = true;
        if (!r.counted) {
            r.counted = true;
            ++stats.packets_acked;
        }

        // ROUND TRIP, from this packet's own send time. Measured per packet
        // rather than by pinging: every packet is already a probe, and a
        // separate ping measures a different path through the send queue.
        const float sample = now - r.sent_at;
        if (stats.rtt_seconds <= 0.0f) {
            stats.rtt_seconds = sample;
        } else {
            const float k = config.rtt_smoothing;
            // Jitter BEFORE the mean is updated, so it measures the deviation
            // from the estimate the sample is being compared against.
            const float deviation = std::fabs(sample - stats.rtt_seconds);
            stats.jitter_seconds += (deviation - stats.jitter_seconds) * k;
            stats.rtt_seconds += (sample - stats.rtt_seconds) * k;
        }

        if (r.reliable_id != 0)
            for (OutgoingReliable& p : pending)
                if (p.id == r.reliable_id) p.acked = true;
    }
}

void Connection::Impl::RetireOldSent() {
    // Anything more than a window behind the newest can never be acknowledged
    // again -- the bitfield does not reach it -- so an unacked record that old
    // is a lost packet.
    while (sent.size() > std::size_t(kAckWindow * 4)) {
        const SentRecord& r = sent.front();
        if (!r.counted && !r.acked) ++stats.packets_lost;
        sent.pop_front();
    }
    while (!pending.empty() && pending.front().acked) pending.pop_front();
}

Connection::Connection(Transport& transport, const Address& peer,
                       const ConnectionConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->transport = &transport;
    impl_->peer = peer;
    impl_->config = config;
}
Connection::~Connection() = default;

std::uint16_t Connection::Send(std::span<const std::uint8_t> payload) {
    std::uint16_t sequence = 0;
    if (!impl_->SendPacket(payload, 0, &sequence)) return 0;
    return sequence;
}

bool Connection::SendReliable(std::span<const std::uint8_t> payload) {
    Impl& im = *impl_;
    if (int(im.pending.size()) >= im.config.max_pending_reliable) return false;
    if (kHeaderBytes + payload.size() > std::size_t(kMaxPacketBytes)) return false;
    OutgoingReliable p;
    p.id = im.next_reliable_id++;
    if (im.next_reliable_id == 0) im.next_reliable_id = 1;
    p.payload.assign(payload.begin(), payload.end());
    im.pending.push_back(std::move(p));
    ++im.stats.reliable_sent;
    return true;
}

int Connection::PendingReliable() const {
    int n = 0;
    for (const auto& p : impl_->pending)
        if (!p.acked) ++n;
    return n;
}

std::vector<Received> Connection::Receive(float dt) {
    Impl& im = *impl_;
    im.now += dt;
    std::vector<Received> out;
    im.inbox.clear();

    std::uint8_t buffer[kMaxPacketBytes];
    for (;;) {
        Address from;
        const int n = im.transport->Receive(&from, {buffer, sizeof(buffer)});
        if (n <= 0) break;
        if (std::size_t(n) < kHeaderBytes) continue;

        PacketHeader h{};
        std::memcpy(&h, buffer, kHeaderBytes);
        // NOT OUR PROTOCOL. A stray packet from something else on the same port
        // would otherwise be parsed as a game packet, and its random sequence
        // number would poison the ack window.
        if (h.protocol != im.config.protocol_id) continue;
        if (!(from == im.peer)) continue;
        if (kHeaderBytes + h.payload_bytes > std::size_t(n)) continue;

        ++im.stats.packets_received;
        im.last_heard = im.now;
        im.heard_anything = true;
        im.OnAck(h.ack, h.ack_bits);

        // --- fold this sequence into the ack window --------------------------
        if (!im.have_remote) {
            im.have_remote = true;
            im.remote_newest = h.sequence;
            im.remote_bits = 0;
        } else if (SequenceNewer(h.sequence, im.remote_newest)) {
            const std::uint16_t shift = std::uint16_t(h.sequence - im.remote_newest);
            // The old newest becomes bit 0 of the mask, and everything else
            // moves up. A shift of 32 or more clears the window entirely, which
            // is correct: nothing in it is within reach any more, and in C++ a
            // shift by the width is undefined rather than zero.
            im.remote_bits = shift >= 32 ? 0u : ((im.remote_bits << shift) | (1u << (shift - 1)));
            im.remote_newest = h.sequence;
        } else if (h.sequence == im.remote_newest) {
            // A DUPLICATE OF THE NEWEST, which is the commonest duplicate there
            // is -- a retransmit that crossed the original arrives immediately
            // after it.
            //
            // Handled explicitly because the general path below computes a
            // distance and shifts by `distance - 1`: at a distance of zero that
            // is a shift by 65535 of a 32-bit value, which is undefined
            // behaviour and in practice let the duplicate straight through.
            // Sixteen of twenty-nine duplicates were delivered twice before
            // this case existed.
            ++im.stats.duplicates_rejected;
            continue;
        } else {
            const std::uint16_t distance = std::uint16_t(im.remote_newest - h.sequence);
            if (distance > kAckWindow) {
                // Far older than anything we can record. Almost certainly a
                // straggler from before a stall; accepting it would deliver
                // state from seconds ago on top of the current state.
                ++im.stats.out_of_order_rejected;
                continue;
            }
            const std::uint32_t bit = 1u << (distance - 1);
            if (im.remote_bits & bit) {
                // ALREADY SEEN. A real network duplicates packets, and applying
                // an input twice is how a player teleports.
                ++im.stats.duplicates_rejected;
                continue;
            }
            im.remote_bits |= bit;
        }

        const std::span<const std::uint8_t> payload{buffer + kHeaderBytes,
                                                    h.payload_bytes};
        if (h.reliable_id == 0) {
            if (h.payload_bytes == 0) continue;  // an ack-only keepalive
            im.inbox.emplace_back(payload.begin(), payload.end());
            Received r;
            r.sequence = h.sequence;
            r.reliable = false;
            out.push_back(r);
            continue;
        }

        // RELIABLE, IN ORDER. A message that arrives early is held until the
        // gap before it is filled, so the application never sees message 7
        // before message 6 -- which for a spawn followed by a state update
        // would mean updating an entity that does not exist yet.
        if (SequenceNewer(im.next_expected_reliable, h.reliable_id) ||
            im.next_expected_reliable == std::uint16_t(h.reliable_id + 1)) {
            ++im.stats.duplicates_rejected;
            continue;
        }
        if (int(im.held.size()) >= im.config.max_held_reliable &&
            im.held.find(h.reliable_id) == im.held.end()) {
            // Full, and this is not one already held. See max_held_reliable:
            // the sender will resend it, and if the gap is never filled the
            // connection is dead anyway -- which is a better outcome than
            // holding everything it can invent.
            ++im.stats.held_dropped;
            continue;
        }
        im.held[h.reliable_id] = std::vector<std::uint8_t>(payload.begin(), payload.end());
    }

    // Drain whatever is now contiguous.
    for (;;) {
        auto it = im.held.find(im.next_expected_reliable);
        if (it == im.held.end()) break;
        im.inbox.push_back(std::move(it->second));
        Received r;
        r.reliable = true;
        out.push_back(r);
        im.held.erase(it);
        ++im.next_expected_reliable;
        if (im.next_expected_reliable == 0) im.next_expected_reliable = 1;
    }

    // The spans could not be taken while `inbox` was still growing -- a
    // reallocation would leave every earlier one dangling. Filled in now that
    // it is final.
    for (std::size_t i = 0; i < out.size() && i < im.inbox.size(); ++i)
        out[i].payload = {im.inbox[i].data(), im.inbox[i].size()};

    // --- resend anything unacknowledged ---------------------------------------
    //
    // A little over the measured round trip. Resending after a fixed short
    // interval floods a slow link with copies of a message that is merely still
    // in flight, and each copy makes the congestion that caused the delay
    // worse.
    const float wait = std::max(im.config.resend_min_seconds,
                                im.stats.rtt_seconds * 1.5f + im.stats.jitter_seconds * 2.0f);
    for (OutgoingReliable& p : im.pending) {
        if (p.acked) continue;
        if (im.now - p.last_sent < wait) continue;
        std::uint16_t sequence = 0;
        if (!im.SendPacket(p.payload, p.id, &sequence)) break;
        if (p.last_sent > -1e8f) ++im.stats.reliable_resent;
        p.last_sent = im.now;
    }
    im.RetireOldSent();
    return out;
}

const ConnectionStats& Connection::Stats() const { return impl_->stats; }
const Address& Connection::Peer() const { return impl_->peer; }

bool Connection::Connected() const {
    const Impl& im = *impl_;
    if (!im.heard_anything) return false;
    return im.now - im.last_heard < im.config.timeout_seconds;
}

float Connection::SilentFor() const {
    const Impl& im = *impl_;
    return im.heard_anything ? im.now - im.last_heard : im.now;
}

}  // namespace eng::net
