#include "engine/net/transport.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <deque>
#include <unordered_map>

namespace eng::net {

std::string Address::ToString() const {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%u.%u.%u.%u:%u", (ip >> 24) & 0xFFu,
                  (ip >> 16) & 0xFFu, (ip >> 8) & 0xFFu, ip & 0xFFu, port);
    return buf;
}

Address Address::Parse(const std::string& text, bool* ok) {
    if (ok) *ok = false;
    Address a;
    const std::size_t colon = text.rfind(':');
    if (colon == std::string::npos) return a;
    const std::string host = text.substr(0, colon);
    in_addr addr{};
    if (inet_pton(AF_INET, host.c_str(), &addr) != 1) return a;
    a.ip = ntohl(addr.s_addr);
    a.port = std::uint16_t(std::strtoul(text.c_str() + colon + 1, nullptr, 10));
    if (ok) *ok = a.port != 0;
    return a;
}

// ------------------------------------------------------------------- real UDP

struct UdpTransport::Impl {
    int fd = -1;
    Address local;
    std::uint64_t sent = 0;
    std::uint64_t received = 0;
};

UdpTransport::UdpTransport() : impl_(std::make_unique<Impl>()) {}

UdpTransport::~UdpTransport() {
    if (impl_ && impl_->fd >= 0) ::close(impl_->fd);
}

std::unique_ptr<UdpTransport> UdpTransport::Open(std::uint16_t port,
                                                 std::string& error) {
    std::unique_ptr<UdpTransport> t(new UdpTransport());
    const int fd = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (fd < 0) {
        error = "net: could not create a UDP socket";
        return nullptr;
    }
    t->impl_->fd = fd;

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        error = "net: could not bind port " + std::to_string(port);
        return nullptr;
    }

    // NON-BLOCKING, and this is the entire reason a game can have networking at
    // all. A blocking recvfrom in the frame loop stops the game until a packet
    // arrives, which on a quiet connection is forever.
    const int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags < 0 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        error = "net: could not put the socket into non-blocking mode";
        return nullptr;
    }

    // Whatever the system actually gave us, which for a requested port of 0 is
    // the only way to find out.
    sockaddr_in bound{};
    socklen_t len = sizeof(bound);
    if (::getsockname(fd, reinterpret_cast<sockaddr*>(&bound), &len) == 0)
        t->impl_->local = Address{0x7F000001u, ntohs(bound.sin_port)};
    return t;
}

bool UdpTransport::Send(const Address& to, std::span<const std::uint8_t> data) {
    if (impl_->fd < 0 || data.empty() || data.size() > kMaxPacketBytes) return false;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(to.ip);
    addr.sin_port = htons(to.port);
    const ssize_t n = ::sendto(impl_->fd, data.data(), data.size(), 0,
                               reinterpret_cast<const sockaddr*>(&addr), sizeof(addr));
    if (n < 0) return false;
    impl_->sent += std::uint64_t(n);
    return std::size_t(n) == data.size();
}

int UdpTransport::Receive(Address* from, std::span<std::uint8_t> out) {
    if (impl_->fd < 0 || out.empty()) return 0;
    sockaddr_in addr{};
    socklen_t len = sizeof(addr);
    const ssize_t n = ::recvfrom(impl_->fd, out.data(), out.size(), 0,
                                 reinterpret_cast<sockaddr*>(&addr), &len);
    // EWOULDBLOCK is the normal answer on a quiet frame, not an error.
    if (n <= 0) return 0;
    if (from) *from = Address{ntohl(addr.sin_addr.s_addr), ntohs(addr.sin_port)};
    impl_->received += std::uint64_t(n);
    return int(n);
}

Address UdpTransport::Local() const { return impl_->local; }
std::uint64_t UdpTransport::BytesSent() const { return impl_->sent; }
std::uint64_t UdpTransport::BytesReceived() const { return impl_->received; }

// ------------------------------------------------------------ the simulator

namespace {

class LoopbackEndpoint;

struct InFlight {
    Address from;
    Address to;
    std::vector<std::uint8_t> data;
    float deliver_at = 0.0f;
};

// A small deterministic generator. Not std::mt19937 seeded from the clock: a
// network test that fails one run in fifty is a test nobody trusts.
struct Rng {
    std::uint32_t s = 1;
    std::uint32_t Next() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return s;
    }
    float Unit() { return float(Next() >> 8) / 16777216.0f; }
};

}  // namespace

struct LoopbackNetwork::Impl {
    NetworkConditions conditions;
    Rng rng;
    float now = 0.0f;
    std::vector<InFlight> in_flight;
    std::unordered_map<std::uint16_t, std::unique_ptr<LoopbackEndpoint>> endpoints;
    int sent = 0, delivered = 0, dropped = 0, duplicated = 0;

    void Enqueue(const Address& from, const Address& to,
                 std::span<const std::uint8_t> data);
};

namespace {

class LoopbackEndpoint : public Transport {
  public:
    LoopbackEndpoint(LoopbackNetwork::Impl* net, Address local)
        : net_(net), local_(local) {}

    bool Send(const Address& to, std::span<const std::uint8_t> data) override {
        if (data.empty() || data.size() > kMaxPacketBytes) return false;
        net_->Enqueue(local_, to, data);
        return true;
    }
    int Receive(Address* from, std::span<std::uint8_t> out) override {
        if (inbox_.empty() || out.empty()) return 0;
        InFlight p = std::move(inbox_.front());
        inbox_.pop_front();
        const std::size_t n = std::min(out.size(), p.data.size());
        std::memcpy(out.data(), p.data.data(), n);
        if (from) *from = p.from;
        return int(n);
    }
    [[nodiscard]] Address Local() const override { return local_; }

    void Deliver(InFlight p) { inbox_.push_back(std::move(p)); }

  private:
    LoopbackNetwork::Impl* net_ = nullptr;
    Address local_;
    std::deque<InFlight> inbox_;
};

}  // namespace

void LoopbackNetwork::Impl::Enqueue(const Address& from, const Address& to,
                                    std::span<const std::uint8_t> data) {
    ++sent;
    if (conditions.loss > 0.0f && rng.Unit() < conditions.loss) {
        ++dropped;
        return;
    }
    const auto delay = [&]() {
        float d = conditions.latency_seconds;
        if (conditions.jitter_seconds > 0.0f)
            d += (rng.Unit() * 2.0f - 1.0f) * conditions.jitter_seconds;
        return std::max(d, 0.0f);
    };

    InFlight p;
    p.from = from;
    p.to = to;
    p.data.assign(data.begin(), data.end());
    p.deliver_at = now + delay();
    // REORDERING, by giving one packet an extra helping of delay. Modelling it
    // as an independent extra delay rather than as a swap is what makes it
    // possible for a packet to arrive two or three places out of order, which
    // is what a real path does when a route changes.
    if (conditions.reorder > 0.0f && rng.Unit() < conditions.reorder)
        p.deliver_at += std::max(conditions.latency_seconds, 0.005f);

    if (conditions.duplicate > 0.0f && rng.Unit() < conditions.duplicate) {
        InFlight copy = p;
        copy.deliver_at = now + delay();
        in_flight.push_back(std::move(copy));
        ++duplicated;
    }
    in_flight.push_back(std::move(p));
}

LoopbackNetwork::LoopbackNetwork(const NetworkConditions& conditions)
    : impl_(std::make_unique<Impl>()) {
    impl_->conditions = conditions;
    impl_->rng.s = conditions.seed ? conditions.seed : 1u;
}
LoopbackNetwork::~LoopbackNetwork() = default;

Transport* LoopbackNetwork::AddEndpoint(std::uint16_t port) {
    auto it = impl_->endpoints.find(port);
    if (it != impl_->endpoints.end()) return it->second.get();
    auto ep = std::make_unique<LoopbackEndpoint>(impl_.get(),
                                                 Address::Loopback(port));
    LoopbackEndpoint* raw = ep.get();
    impl_->endpoints.emplace(port, std::move(ep));
    return raw;
}

void LoopbackNetwork::SetConditions(const NetworkConditions& c) {
    impl_->conditions = c;
}

void LoopbackNetwork::Update(float dt) {
    Impl& im = *impl_;
    im.now += dt;
    // Partitioned rather than erased one at a time: delivering in the middle of
    // a vector while iterating it is the classic way to skip an element, and a
    // dropped packet that was not supposed to be dropped is the hardest kind of
    // network bug to see.
    std::vector<InFlight> still_waiting;
    still_waiting.reserve(im.in_flight.size());
    for (InFlight& p : im.in_flight) {
        if (p.deliver_at > im.now) {
            still_waiting.push_back(std::move(p));
            continue;
        }
        auto it = im.endpoints.find(p.to.port);
        if (it == im.endpoints.end()) {
            // Nobody listening. A real UDP send to a closed port succeeds and
            // the packet vanishes, so this does too.
            ++im.dropped;
            continue;
        }
        ++im.delivered;
        it->second->Deliver(std::move(p));
    }
    im.in_flight = std::move(still_waiting);
}

int LoopbackNetwork::Sent() const { return impl_->sent; }
int LoopbackNetwork::Delivered() const { return impl_->delivered; }
int LoopbackNetwork::Dropped() const { return impl_->dropped; }
int LoopbackNetwork::Duplicated() const { return impl_->duplicated; }

}  // namespace eng::net
