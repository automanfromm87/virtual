// The wire: UDP, and a simulated network to test against.
//
// WHY UDP AND NOT TCP. TCP guarantees that every byte arrives in order, and it
// achieves that by stopping. A lost packet blocks everything behind it until it
// is retransmitted -- head-of-line blocking -- so one dropped datagram costs a
// round trip of latency on every update after it, including the ones that
// already arrived. For a game that is exactly backwards: a player's position
// from 100 ms ago is worthless, and holding up the current one to deliver it is
// the worst possible trade. A game wants the newest state, and wants
// reliability only for the few messages that genuinely need it -- which is what
// the layer above this provides, selectively.
//
// WHY A SIMULATED TRANSPORT EXISTS AT ALL. Because a network protocol's whole
// purpose is behaving correctly when the network does not, and a loopback
// socket never loses, reorders or duplicates anything. Every interesting case
// -- an ack lost on the way back, a duplicate arriving after the retransmit, a
// packet from two seconds ago turning up -- is unreachable through a real
// socket on one machine, which is why protocols shipped without one fail in the
// field and not in testing.
#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace eng::net {

// The largest datagram this layer will send.
//
// 1200 bytes, not the 1500 of an Ethernet frame. The path to a player goes
// through tunnels, VPNs and mobile links that each take their own bytes off the
// front, and a datagram that exceeds the smallest MTU on the path is fragmented
// by IP -- at which point losing any one fragment loses the whole packet, and
// the loss rate multiplies. 1200 is the number everyone converged on because it
// survives essentially every real path.
inline constexpr int kMaxPacketBytes = 1200;

struct Address {
    // IPv4, host byte order. IPv6 is a real gap and is deliberate: supporting
    // it properly means sockaddr_storage everywhere and a resolver, and the
    // engine's networking is worth having before it is worth having twice.
    std::uint32_t ip = 0;
    std::uint16_t port = 0;

    [[nodiscard]] bool operator==(const Address& o) const {
        return ip == o.ip && port == o.port;
    }
    [[nodiscard]] bool Valid() const { return port != 0; }
    [[nodiscard]] std::string ToString() const;

    [[nodiscard]] static Address Loopback(std::uint16_t port) {
        return Address{0x7F000001u, port};
    }
    // Dotted quad only, no DNS. A resolver is a blocking call and blocking in a
    // frame is what this whole module exists to avoid; a caller that needs one
    // should do it before the game loop starts.
    [[nodiscard]] static Address Parse(const std::string& text, bool* ok = nullptr);
};

// What both the real and simulated networks look like from above.
class Transport {
  public:
    virtual ~Transport() = default;
    // Never blocks. Returns false when the datagram could not be handed to the
    // system -- a full send buffer, an unreachable route -- which is NOT an
    // error a caller should retry: the protocol above already handles loss, and
    // a retry loop here would make a congested link worse.
    virtual bool Send(const Address& to, std::span<const std::uint8_t> data) = 0;
    // Returns the number of bytes read, or 0 when nothing is waiting. Never
    // blocks.
    virtual int Receive(Address* from, std::span<std::uint8_t> out) = 0;
    [[nodiscard]] virtual Address Local() const = 0;
    // Advances any internal timers. The real socket ignores it; the simulator
    // needs it to deliver delayed packets.
    virtual void Update(float dt) {}
};

// --- a real socket ------------------------------------------------------------

class UdpTransport : public Transport {
  public:
    // `port` of 0 asks the system for one, which is what a client wants.
    [[nodiscard]] static std::unique_ptr<UdpTransport> Open(std::uint16_t port,
                                                            std::string& error);
    ~UdpTransport() override;

    bool Send(const Address& to, std::span<const std::uint8_t> data) override;
    int Receive(Address* from, std::span<std::uint8_t> out) override;
    [[nodiscard]] Address Local() const override;

    [[nodiscard]] std::uint64_t BytesSent() const;
    [[nodiscard]] std::uint64_t BytesReceived() const;

  private:
    UdpTransport();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// --- a network that misbehaves on purpose ------------------------------------

struct NetworkConditions {
    // Fraction of packets dropped, each way.
    float loss = 0.0f;
    // One-way delay, and how much it varies. Latency is what makes prediction
    // necessary; JITTER is what makes interpolation necessary, and a simulator
    // without it lets a protocol that assumes even spacing pass.
    float latency_seconds = 0.0f;
    float jitter_seconds = 0.0f;
    // Fraction of packets delivered twice. Real networks do this -- a
    // retransmit that crossed the original -- and a protocol that applies an
    // input twice because of it is a protocol that teleports players.
    float duplicate = 0.0f;
    // Fraction delivered out of order, by delaying one packet an extra frame.
    float reorder = 0.0f;
    // Deterministic. A network test that fails one run in fifty is a test
    // nobody trusts and everybody re-runs.
    std::uint32_t seed = 12345;
};

// Two or more transports wired to each other in memory.
class LoopbackNetwork {
  public:
    explicit LoopbackNetwork(const NetworkConditions& = {});
    ~LoopbackNetwork();
    LoopbackNetwork(const LoopbackNetwork&) = delete;
    LoopbackNetwork& operator=(const LoopbackNetwork&) = delete;

    // Creates an endpoint at `port`. The returned transport is owned by the
    // network and stays valid until it is destroyed.
    [[nodiscard]] Transport* AddEndpoint(std::uint16_t port);
    void SetConditions(const NetworkConditions&);
    void Update(float dt);

    [[nodiscard]] int Sent() const;
    [[nodiscard]] int Delivered() const;
    [[nodiscard]] int Dropped() const;
    [[nodiscard]] int Duplicated() const;

    // Public because the endpoints the simulator hands out are defined in the
    // .cc and need to reach back into it. Opaque either way -- nothing outside
    // the implementation can do anything with an incomplete type.
    struct Impl;

  private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace eng::net
