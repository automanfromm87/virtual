// A connection over UDP: sequence numbers, acknowledgements, round-trip
// estimation, and reliability for the messages that need it.
//
// WHAT THIS ADDS TO A SOCKET. A datagram arrives, or does not, possibly twice,
// possibly out of order, with nothing to say which one it was. Everything a
// game protocol does above that -- knowing which state the other end has,
// resending a chat message, measuring the round trip so prediction can be
// tuned -- needs those three facts, and they all come from one small header.
//
// THE ACK BITFIELD is the trick that makes it cheap. Each packet carries the
// sequence number of the newest packet received from the other end, plus a
// 32-bit mask of which of the previous 32 were also received. So every packet
// acknowledges up to 33 packets, and losing an ack costs nothing at all -- the
// next packet re-states it. A protocol that acknowledged one packet per ack
// would lose that information whenever the ack itself was dropped, which at 5%
// loss is 5% of everything.
//
// SEQUENCE NUMBERS WRAP, and the comparison has to know it. A 16-bit sequence
// wraps every eleven minutes at 100 Hz; at that moment a naive `a > b` says
// every new packet is ancient and the connection stops accepting anything. The
// comparison below is the standard serial-number arithmetic, and it is the
// single most commonly omitted line in a hand-rolled protocol.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "engine/net/transport.h"

namespace eng::net {

// Is `a` newer than `b`, allowing for wraparound? True when the forward
// distance from b to a is less than half the sequence space.
[[nodiscard]] inline bool SequenceNewer(std::uint16_t a, std::uint16_t b) {
    return (a != b) && (std::uint16_t(a - b) < 0x8000u);
}

struct ConnectionConfig {
    // Distinguishes this game's packets from anything else that arrives on the
    // port. Not security -- it stops a stray packet from another application
    // being parsed as a game packet, which otherwise reads as a protocol bug.
    std::uint32_t protocol_id = 0x56495254u;  // 'VIRT'
    // How long without hearing anything before the peer is considered gone.
    float timeout_seconds = 5.0f;
    // How long a reliable message waits for its ack before being sent again.
    // Too short and a slow link is flooded with duplicates; too long and a lost
    // message costs a visible pause. A little over one round trip is right,
    // which is why this is a floor and the real value tracks the measured RTT.
    float resend_min_seconds = 0.08f;
    // How much of the new sample each RTT measurement takes. 0.1 is the usual
    // smoothing: fast enough to follow a route change, slow enough that one
    // late packet does not move it.
    float rtt_smoothing = 0.1f;
    // The most reliable messages held at once. A sender that outruns this stops
    // accepting new ones rather than growing without bound -- an unbounded
    // queue on a dead connection is a memory leak that looks like lag.
    int max_pending_reliable = 256;
    // The most OUT-OF-ORDER reliable messages held while waiting for the gap
    // ahead of them to fill.
    //
    // The send side has had a bound since it was written; the receive side had
    // none, and the receive side is the one fed by the network. A peer that
    // sends ids scattered across the sequence space and never fills the gap
    // makes this map grow until it has half the sequence space in it -- 32768
    // messages of up to a packet each, which is not literally unbounded but is
    // nine figures of memory arriving from someone else's decision.
    //
    // Over the limit the newest are DROPPED rather than the oldest evicted:
    // reliable delivery is in order, so the oldest held message is the one
    // closest to being deliverable and throwing it away guarantees the stall
    // it is waiting on never resolves.
    int max_held_reliable = 512;
};

struct ConnectionStats {
    // Out-of-order reliable messages refused because the hold table was full.
    // Non-zero means either a badly reordering path or a peer feeding gaps it
    // never intends to fill.
    std::uint32_t held_dropped = 0;
    std::uint32_t packets_sent = 0;
    std::uint32_t packets_received = 0;
    std::uint32_t packets_acked = 0;
    // Counted from the acks: a sequence that was never acknowledged before it
    // fell off the end of the window. An estimate, and the only one available
    // without the other end reporting.
    std::uint32_t packets_lost = 0;
    std::uint32_t reliable_sent = 0;
    std::uint32_t reliable_resent = 0;
    std::uint32_t duplicates_rejected = 0;
    std::uint32_t out_of_order_rejected = 0;
    float rtt_seconds = 0.0f;
    float jitter_seconds = 0.0f;
    [[nodiscard]] float LossFraction() const {
        const std::uint32_t total = packets_acked + packets_lost;
        return total > 0 ? float(packets_lost) / float(total) : 0.0f;
    }
};

// One packet's worth of payload, as handed to the application.
struct Received {
    std::uint16_t sequence = 0;
    bool reliable = false;
    std::span<const std::uint8_t> payload;
};

class Connection {
  public:
    Connection(Transport&, const Address& peer, const ConnectionConfig& = {});
    ~Connection();
    Connection(const Connection&) = delete;
    Connection& operator=(const Connection&) = delete;

    // Sends `payload` in one packet, unreliably. Returns the sequence number it
    // went out as, or 0 if it could not be sent.
    //
    // UNRELIABLE IS THE DEFAULT, deliberately. Almost everything a game sends
    // is state that will be sent again next tick, and resending a stale
    // position is worse than useless -- it costs bandwidth to deliver something
    // the receiver will immediately overwrite.
    std::uint16_t Send(std::span<const std::uint8_t> payload);

    // Queued and resent until acknowledged, then delivered to the far end in
    // order. For the messages that genuinely cannot be dropped: a chat line, a
    // spawn, a level change.
    bool SendReliable(std::span<const std::uint8_t> payload);
    [[nodiscard]] int PendingReliable() const;

    // Reads everything waiting on the transport and returns the payloads, in
    // the order they should be processed. Reliable payloads are held back until
    // the gap before them is filled, so the caller sees them in the order they
    // were sent; unreliable ones are handed over immediately and only if they
    // are the newest so far.
    std::vector<Received> Receive(float dt);

    [[nodiscard]] const ConnectionStats& Stats() const;
    [[nodiscard]] bool Connected() const;
    [[nodiscard]] const Address& Peer() const;
    // Seconds since the last packet arrived from the peer.
    [[nodiscard]] float SilentFor() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eng::net
