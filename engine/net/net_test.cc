// Networking, against a network that misbehaves on purpose.
//
// A protocol tested on loopback is a protocol tested against a perfect link,
// and every bug it has is in the code that handles imperfection -- which never
// runs. So almost everything here runs over the simulator with loss, latency,
// jitter, duplication and reordering turned on, and the checks are about what
// survives.
//
// The simulator is deterministic, because a network test that fails one run in
// fifty is a test nobody trusts and everybody re-runs until it passes.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "engine/net/connection.h"
#include "engine/net/replication.h"
#include "engine/net/transport.h"

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
}

using eng::Vec3;
using namespace eng::net;

std::vector<std::uint8_t> Bytes(const std::string& s) {
    return {s.begin(), s.end()};
}
std::string Text(std::span<const std::uint8_t> b) {
    return std::string(b.begin(), b.end());
}

// The one simulation both ends run. Deliberately trivial and deliberately
// SHARED: the whole point of reconciliation is that the client and server agree,
// and two copies of "nearly the same" movement code is the commonest reason
// they do not.
void Move(EntityState* e, const InputCommand& in) {
    e->velocity = in.move * 5.0f;
    e->position = e->position + e->velocity * in.dt;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);

    {
        std::printf("sequence comparison survives the wraparound\n");
        Check(SequenceNewer(5, 4), "5 is newer than 4");
        Check(!SequenceNewer(4, 5), "and 4 is not newer than 5");
        // THE ONE THAT MATTERS. At 100 Hz a 16-bit sequence wraps every eleven
        // minutes, and a naive `a > b` says every packet after the wrap is
        // ancient -- so the connection silently stops accepting anything.
        Check(SequenceNewer(0, 65535), "0 is newer than 65535");
        Check(SequenceNewer(3, 65530), "and 3 is newer than 65530");
        Check(!SequenceNewer(65535, 0), "while 65535 is not newer than 0");
        Check(!SequenceNewer(7, 7), "and nothing is newer than itself");
    }

    {
        std::printf("\na clean link delivers everything\n");
        LoopbackNetwork net;
        Transport* a = net.AddEndpoint(1000);
        Transport* b = net.AddEndpoint(2000);
        Connection client(*a, Address::Loopback(2000));
        Connection server(*b, Address::Loopback(1000));

        bool all_sent = true;
        for (int i = 0; i < 20; ++i)
            if (client.Send(Bytes("hello " + std::to_string(i))) == 0) all_sent = false;
        Check(all_sent, "twenty packets go out");
        net.Update(0.016f);
        const auto got = server.Receive(0.016f);
        std::printf("    sent 20, received %zu\n", got.size());
        Check(got.size() == 20, "all twenty arrive");
        Check(Text(got[0].payload) == "hello 0", "and the first is intact");
        Check(Text(got[19].payload) == "hello 19", "and so is the last");
    }

    {
        std::printf("\nthe ack bitfield reports what arrived\n");
        LoopbackNetwork net;
        Transport* a = net.AddEndpoint(1001);
        Transport* b = net.AddEndpoint(2001);
        Connection client(*a, Address::Loopback(2001));
        Connection server(*b, Address::Loopback(1001));

        for (int i = 0; i < 40; ++i) {
            (void)client.Send(Bytes("x"));
            net.Update(0.016f);
            (void)server.Receive(0.016f);
            (void)server.Send(Bytes("y"));
            net.Update(0.016f);
            (void)client.Receive(0.016f);
        }
        std::printf("    client sent %u, acked %u, lost %u; rtt %.1f ms\n",
                    client.Stats().packets_sent, client.Stats().packets_acked,
                    client.Stats().packets_lost, client.Stats().rtt_seconds * 1000.0f);
        Check(client.Stats().packets_acked >= 38, "nearly every packet is acknowledged");
        Check(client.Stats().packets_lost == 0, "and none is counted as lost");
        Check(client.Connected() && server.Connected(), "both ends consider it up");
    }

    {
        std::printf("\nround-trip time is measured\n");
        NetworkConditions c;
        c.latency_seconds = 0.05f;  // 100 ms round trip
        LoopbackNetwork net(c);
        Transport* a = net.AddEndpoint(1002);
        Transport* b = net.AddEndpoint(2002);
        Connection client(*a, Address::Loopback(2002));
        Connection server(*b, Address::Loopback(1002));

        for (int i = 0; i < 200; ++i) {
            (void)client.Send(Bytes("ping"));
            (void)server.Send(Bytes("pong"));
            net.Update(0.01f);
            (void)client.Receive(0.01f);
            (void)server.Receive(0.01f);
        }
        const float rtt = client.Stats().rtt_seconds * 1000.0f;
        std::printf("    simulated 100 ms round trip, measured %.1f ms\n", rtt);
        Check(rtt > 80.0f && rtt < 130.0f, "the estimate is within 30% of the truth");
    }

    {
        std::printf("\nduplicates are rejected and stragglers are dropped\n");
        NetworkConditions c;
        c.duplicate = 0.5f;  // half of everything twice
        LoopbackNetwork net(c);
        Transport* a = net.AddEndpoint(1003);
        Transport* b = net.AddEndpoint(2003);
        Connection client(*a, Address::Loopback(2003));
        Connection server(*b, Address::Loopback(1003));

        int delivered = 0;
        for (int i = 0; i < 60; ++i) {
            (void)client.Send(Bytes("once"));
            net.Update(0.016f);
            delivered += int(server.Receive(0.016f).size());
        }
        std::printf("    60 sent, %d duplicated by the network, %d delivered, "
                    "%u rejected\n", net.Duplicated(), delivered,
                    server.Stats().duplicates_rejected);
        Check(net.Duplicated() > 15, "the simulator really did duplicate packets");
        // EXACTLY ONCE. A protocol that delivers a duplicate is one that applies
        // an input twice, and a player who fires twice for one trigger pull.
        Check(delivered == 60, "each packet is delivered exactly once");
        Check(server.Stats().duplicates_rejected == std::uint32_t(net.Duplicated()),
              "and every duplicate is accounted for");
    }

    {
        std::printf("\nreliable messages survive heavy loss, in order\n");
        NetworkConditions c;
        c.loss = 0.3f;              // brutal
        c.latency_seconds = 0.03f;
        c.jitter_seconds = 0.01f;
        c.reorder = 0.1f;
        LoopbackNetwork net(c);
        Transport* a = net.AddEndpoint(1004);
        Transport* b = net.AddEndpoint(2004);
        Connection client(*a, Address::Loopback(2004));
        Connection server(*b, Address::Loopback(1004));

        constexpr int kMessages = 30;
        bool all_queued = true;
        for (int i = 0; i < kMessages; ++i)
            if (!client.SendReliable(Bytes("msg" + std::to_string(i)))) all_queued = false;
        Check(all_queued, "thirty reliable messages are queued");

        std::vector<std::string> received;
        for (int frame = 0; frame < 600; ++frame) {
            net.Update(0.016f);
            for (const Received& r : server.Receive(0.016f))
                if (r.reliable) received.push_back(Text(r.payload));
            (void)server.Send(Bytes("ack"));  // carries the ack bitfield back
            net.Update(0.016f);
            (void)client.Receive(0.016f);
        }
        std::printf("    %d of %d arrived over a 30%% loss link; %u resends\n",
                    int(received.size()), kMessages, client.Stats().reliable_resent);
        Check(int(received.size()) == kMessages, "every reliable message arrives");

        bool ordered = true;
        for (int i = 0; i < int(received.size()); ++i)
            if (received[std::size_t(i)] != "msg" + std::to_string(i)) ordered = false;
        // IN ORDER. A spawn followed by a state update for the same entity has
        // to arrive that way round, or the update is applied to something that
        // does not exist.
        Check(ordered, "and in the order they were sent");
        Check(client.PendingReliable() == 0, "with nothing still outstanding");
        Check(client.Stats().reliable_resent > 0,
              "and the resend path was actually exercised");
    }

    {
        std::printf("\na timeout is noticed\n");
        LoopbackNetwork net;
        Transport* a = net.AddEndpoint(1005);
        Transport* b = net.AddEndpoint(2005);
        ConnectionConfig cfg;
        cfg.timeout_seconds = 1.0f;
        Connection client(*a, Address::Loopback(2005), cfg);
        Connection server(*b, Address::Loopback(1005), cfg);

        Check(!client.Connected(), "a connection that has heard nothing is not up");
        (void)server.Send(Bytes("hi"));
        net.Update(0.016f);
        (void)client.Receive(0.016f);
        Check(client.Connected(), "and is up once something arrives");

        for (int i = 0; i < 100; ++i) {
            net.Update(0.016f);
            (void)client.Receive(0.016f);
        }
        std::printf("    silent for %.2f s with a 1.00 s timeout\n",
                    client.SilentFor());
        Check(!client.Connected(), "and down again after the timeout");
    }

    // --- replication ---------------------------------------------------------

    {
        std::printf("\ndelta compression sends only what changed\n");
        Snapshot base;
        base.tick = 100;
        for (std::uint32_t i = 0; i < 50; ++i)
            base.entities.push_back(EntityState{i, Vec3{float(i), 0, 0}, {}, {}, 0});

        Snapshot next = base;
        next.tick = 101;
        next.entities[7].position.x += 1.0f;
        next.entities[23].flags = 5;

        const auto full = EncodeSnapshot(next, nullptr);
        const auto delta = EncodeSnapshot(next, &base);
        std::printf("    50 entities, 2 changed: %zu bytes full, %zu bytes delta\n",
                    full.size(), delta.size());
        Check(delta.size() < full.size() / 10,
              "a delta of two changes is a tenth the size");

        Snapshot decoded;
        Check(DecodeSnapshot(delta, &base, &decoded), "the delta decodes");
        Check(decoded.tick == 101, "with the right tick");
        Check(decoded.entities.size() == 50, "and every entity present");
        bool same = true;
        for (std::size_t i = 0; i < next.entities.size(); ++i)
            if (!(decoded.entities[i] == next.entities[i])) same = false;
        Check(same, "and identical to what was encoded");

        // A FULL snapshot decodes with no baseline at all, which is what a
        // client joining mid-game gets.
        Snapshot from_full;
        Check(DecodeSnapshot(full, nullptr, &from_full), "a full snapshot needs no baseline");
        Check(from_full.entities.size() == 50, "and reconstructs everything");
    }

    {
        std::printf("\nspawns and despawns replicate\n");
        Snapshot base;
        base.tick = 1;
        for (std::uint32_t i = 0; i < 5; ++i)
            base.entities.push_back(EntityState{i, Vec3{float(i), 0, 0}, {}, {}, 0});
        Snapshot next;
        next.tick = 2;
        // Remove 2, keep the rest, add 9.
        for (std::uint32_t i = 0; i < 5; ++i)
            if (i != 2)
                next.entities.push_back(EntityState{i, Vec3{float(i), 0, 0}, {}, {}, 0});
        next.entities.push_back(EntityState{9, Vec3{9, 0, 0}, {}, {}, 0});

        Snapshot decoded;
        Check(DecodeSnapshot(EncodeSnapshot(next, &base), &base, &decoded),
              "the delta decodes");
        Check(decoded.Find(2) == nullptr, "a removed entity is gone");
        Check(decoded.Find(9) != nullptr, "a new one is there");
        Check(decoded.entities.size() == 5, "and the count is right");
    }

    {
        std::printf("\na delta against the wrong baseline is refused\n");
        Snapshot a;
        a.tick = 10;
        a.entities.push_back(EntityState{1, Vec3{1, 0, 0}, {}, {}, 0});
        Snapshot b;
        b.tick = 11;
        b.entities.push_back(EntityState{1, Vec3{2, 0, 0}, {}, {}, 0});
        const auto delta = EncodeSnapshot(b, &a);

        Snapshot wrong;
        wrong.tick = 99;
        wrong.entities.push_back(EntityState{1, Vec3{50, 0, 0}, {}, {}, 0});
        Snapshot out;
        // APPLYING IT ANYWAY would produce a world that is plausible and wrong,
        // and every later delta would compound it -- so it is refused. This is
        // also why the server deltas against what the client CONFIRMED rather
        // than against its own last send.
        Check(!DecodeSnapshot(delta, &wrong, &out),
              "a delta cannot be applied to a different baseline");
        Check(!DecodeSnapshot(delta, nullptr, &out), "nor to no baseline at all");
        Check(DecodeSnapshot(delta, &a, &out), "and works against the right one");

        std::uint32_t tick = 0, baseline_tick = 0;
        Check(PeekSnapshotHeader(delta, &tick, &baseline_tick) && tick == 11 &&
                  baseline_tick == 10,
              "and the header can be read without decoding");
    }

    {
        std::printf("\nthe server deltas against what the client confirmed\n");
        SnapshotHistory history(16);
        for (std::uint32_t t = 1; t <= 5; ++t) {
            Snapshot s;
            s.tick = t;
            s.entities.push_back(EntityState{1, Vec3{float(t), 0, 0}, {}, {}, 0});
            history.Add(s);
        }
        Check(history.Count() == 5, "five snapshots are kept");
        Check(history.Newest()->tick == 5, "and the newest is the newest");

        // The client last confirmed tick 2, so the delta must be against 2 --
        // not against 4, which it never received.
        const auto encoded = history.EncodeFor(2);
        std::uint32_t baseline_tick = 0;
        (void)PeekSnapshotHeader(encoded, nullptr, &baseline_tick);
        Check(baseline_tick == 2, "the delta is against the confirmed tick");

        Snapshot out;
        Check(DecodeSnapshot(encoded, history.Get(2), &out) && out.tick == 5,
              "and the client can decode it from what it has");

        // A client with nothing gets a full snapshot.
        (void)PeekSnapshotHeader(history.EncodeFor(0), nullptr, &baseline_tick);
        Check(baseline_tick == 0, "a client with no baseline gets a full snapshot");
    }

    {
        std::printf("\ninterpolation smooths jittery arrivals\n");
        InterpolationConfig cfg;
        cfg.delay_seconds = 0.1f;
        InterpolationBuffer buffer(cfg);

        // An entity moving at a steady 10 m/s, in snapshots arriving unevenly.
        const float arrivals[] = {0.00f, 0.048f, 0.113f, 0.147f, 0.198f,
                                  0.251f, 0.312f, 0.348f, 0.401f, 0.452f};
        for (int i = 0; i < 10; ++i) {
            Snapshot s;
            s.tick = std::uint32_t(i + 1);
            // The server's own timing is even: every snapshot is 50 ms of
            // simulation apart however unevenly it arrives.
            s.entities.push_back(
                EntityState{1, Vec3{float(i) * 0.5f, 0, 0}, {}, Vec3{10, 0, 0}, 0});
            buffer.Add(s, arrivals[i]);
        }
        Check(buffer.Count() == 10, "ten snapshots buffered");

        // Sampled at even intervals, the position must advance evenly -- which
        // is the whole point. Drawing each snapshot as it lands would advance
        // in the uneven steps above.
        float previous = -1e9f;
        float min_step = 1e9f, max_step = -1e9f;
        for (float t = 0.16f; t <= 0.44f; t += 0.02f) {
            Snapshot out;
            if (!buffer.Sample(t, &out)) continue;
            const EntityState* e = out.Find(1);
            if (!e) continue;
            if (previous > -1e8f) {
                const float step = e->position.x - previous;
                min_step = std::min(min_step, step);
                max_step = std::max(max_step, step);
            }
            previous = e->position.x;
        }
        std::printf("    step per 20 ms: smallest %.4f, largest %.4f\n", min_step,
                    max_step);
        Check(min_step > 0.0f, "the entity always moves forward");
        // A ratio near one means the motion is smooth. Playing snapshots as
        // they arrive would give a ratio matching the arrival jitter, which
        // here is over three to one.
        Check(max_step / std::max(min_step, 1e-6f) < 2.0f,
              "and at a nearly constant rate despite the jitter");
        Check(!buffer.Starved(), "without the buffer running dry");
    }

    {
        std::printf("\nthe buffer says when it has run dry\n");
        InterpolationConfig cfg;
        cfg.delay_seconds = 0.1f;
        cfg.max_extrapolation_seconds = 0.2f;
        InterpolationBuffer buffer(cfg);
        Snapshot s;
        s.tick = 1;
        s.entities.push_back(EntityState{1, Vec3{0, 0, 0}, {}, Vec3{10, 0, 0}, 0});
        buffer.Add(s, 0.0f);

        Snapshot out;
        Check(buffer.Sample(0.15f, &out), "it still answers with one snapshot");
        Check(buffer.Starved(), "and says it is starved");
        const float extrapolated = out.Find(1)->position.x;
        std::printf("    50 ms past the last snapshot, extrapolated to %.3f m\n",
                    extrapolated);
        Check(extrapolated > 0.3f && extrapolated < 0.7f,
              "extrapolating briefly along the velocity");

        // PAST THE LIMIT it holds instead. Extrapolating indefinitely sends
        // everything sliding off in a straight line -- a player who stopped
        // keeps walking through a wall -- and a freeze is far easier to
        // recognise as a network problem.
        (void)buffer.Sample(2.0f, &out);
        std::printf("    two seconds later, held at %.3f m\n", out.Find(1)->position.x);
        Check(out.Find(1)->position.x < 3.0f, "and holding once past the limit");
    }

    {
        std::printf("\nout-of-order snapshots are inserted, not dropped\n");
        InterpolationBuffer buffer;
        Snapshot a, b, c;
        a.tick = 1;
        b.tick = 2;
        c.tick = 3;
        for (Snapshot* s : {&a, &b, &c})
            s->entities.push_back(
                EntityState{1, Vec3{float(s->tick), 0, 0}, {}, {}, 0});
        buffer.Add(a, 0.00f);
        buffer.Add(c, 0.10f);  // 3 arrives before 2
        buffer.Add(b, 0.05f);
        Check(buffer.Count() == 3, "the late snapshot is kept");
        buffer.Add(b, 0.05f);
        Check(buffer.Count() == 3, "and a duplicate is not");
    }

    {
        std::printf("\nprediction is corrected without drifting\n");
        Predictor predictor;
        EntityState start;
        start.id = 1;
        predictor.SetPredicted(start);

        // The client runs ahead; the server confirms with a lag.
        EntityState authoritative = start;
        std::uint32_t confirmed = 0;
        float worst_error = 0.0f;
        for (std::uint32_t frame = 1; frame <= 120; ++frame) {
            InputCommand in;
            in.sequence = frame;
            in.dt = 1.0f / 60.0f;
            in.move = Vec3{1.0f, 0.0f, 0.0f};
            predictor.Apply(in, Move);

            // The server processes inputs six frames behind.
            if (frame > 6) {
                InputCommand server_input = in;
                server_input.sequence = frame - 6;
                Move(&authoritative, server_input);
                confirmed = frame - 6;
                worst_error =
                    std::max(worst_error, predictor.Reconcile(confirmed, authoritative, Move));
            }
        }
        std::printf("    after 120 frames: predicted %.4f, server %.4f, worst "
                    "correction %.6f m, %d replays\n",
                    predictor.Predicted().position.x, authoritative.position.x,
                    worst_error, predictor.ReplayCount());
        Check(predictor.PendingInputs() == 6, "six inputs are still unconfirmed");
        // ZERO ERROR on a deterministic simulation with no loss. A value that is
        // never zero means the client and server are running different code,
        // which shows up in play as rubber banding on a perfect connection.
        Check(worst_error < 1e-4f, "the prediction is never wrong");
        // AND THE PREDICTION IS AHEAD, which is the point: six frames of input
        // have been applied that the server has not seen.
        Check(predictor.Predicted().position.x > authoritative.position.x + 0.4f,
              "and stays ahead of the server by the inputs in flight");
    }

    {
        std::printf("\na mispredicted client snaps back and replays\n");
        Predictor predictor;
        EntityState start;
        start.id = 1;
        predictor.SetPredicted(start);
        for (std::uint32_t f = 1; f <= 10; ++f) {
            InputCommand in;
            in.sequence = f;
            in.dt = 1.0f / 60.0f;
            in.move = Vec3{1, 0, 0};
            predictor.Apply(in, Move);
        }
        const float before = predictor.Predicted().position.x;

        // The server disagrees: the player was blocked by a wall it did not
        // know about, so it is two metres behind after input 4.
        EntityState server_state;
        server_state.id = 1;
        server_state.position = Vec3{before - 2.0f, 0, 0};
        const float error = predictor.Reconcile(4, server_state, Move);
        std::printf("    reported error %.3f m; position %.3f -> %.3f\n", error,
                    before, predictor.Predicted().position.x);
        Check(error > 1.5f, "the disagreement is reported");
        Check(predictor.PendingInputs() == 6, "the unconfirmed inputs are kept");
        // REPLAYED, not discarded. Snapping to the server and dropping the
        // pending inputs would throw away six frames of the player's movement,
        // which reads as the character stuttering backwards.
        Check(predictor.Predicted().position.x > server_state.position.x,
              "and the six unseen inputs are replayed on top of the correction");
    }

    {
        std::printf("\nend to end: a server and a client over a bad link\n");
        NetworkConditions c;
        c.loss = 0.12f;
        c.latency_seconds = 0.04f;
        c.jitter_seconds = 0.015f;
        c.reorder = 0.08f;
        c.duplicate = 0.05f;
        LoopbackNetwork net(c);
        Transport* server_t = net.AddEndpoint(7777);
        Transport* client_t = net.AddEndpoint(7778);
        Connection server(*server_t, Address::Loopback(7778));
        Connection client(*client_t, Address::Loopback(7777));

        SnapshotHistory history(64);
        // THE CLIENT KEEPS A HISTORY TOO, and that is not symmetry for its own
        // sake. The ack it sends takes a round trip to arrive, so by the time
        // the server builds a delta against tick N the client has usually
        // decoded several ticks past it -- and a client holding only its newest
        // snapshot no longer has N to apply the delta to.
        //
        // Measured with one baseline: 79 snapshots decoded and 100 undecodable.
        SnapshotHistory client_history(64);
        InterpolationBuffer buffer;
        Snapshot client_world;
        std::uint32_t client_acked = 0;
        float now = 0.0f;
        int decoded_ok = 0, decoded_failed = 0;

        for (std::uint32_t tick = 1; tick <= 200; ++tick) {
            now += 1.0f / 30.0f;
            // The server moves one entity in a circle and snapshots it.
            Snapshot s;
            s.tick = tick;
            const float a = float(tick) * 0.05f;
            s.entities.push_back(EntityState{
                1, Vec3{std::cos(a) * 5.0f, 0, std::sin(a) * 5.0f}, {}, {}, 0});
            history.Add(s);

            const auto payload = history.EncodeFor(client_acked);
            (void)server.Send(payload);
            net.Update(1.0f / 30.0f);

            for (const Received& r : client.Receive(1.0f / 30.0f)) {
                std::uint32_t t = 0, baseline_tick = 0;
                if (!PeekSnapshotHeader(r.payload, &t, &baseline_tick)) continue;
                const Snapshot* baseline =
                    baseline_tick != 0 ? client_history.Get(baseline_tick) : nullptr;
                Snapshot decoded;
                if (!DecodeSnapshot(r.payload, baseline, &decoded)) {
                    // UNDECODABLE, because the baseline it needs never arrived.
                    // The client says nothing and keeps its old ack, so the
                    // server falls back to a snapshot it can use -- which is
                    // exactly why acks drive the baseline.
                    ++decoded_failed;
                    continue;
                }
                ++decoded_ok;
                client_world = decoded;
                client_history.Add(decoded);
                buffer.Add(decoded, now);
            }
            // The client tells the server what it has, on its own packets.
            std::uint8_t ack_bytes[4];
            std::memcpy(ack_bytes, &client_world.tick, 4);
            (void)client.Send({ack_bytes, 4});
            net.Update(1.0f / 30.0f);
            for (const Received& r : server.Receive(1.0f / 30.0f))
                if (r.payload.size() == 4) std::memcpy(&client_acked, r.payload.data(), 4);
        }

        std::printf("    %d snapshots decoded, %d undecodable, client at tick %u of "
                    "200\n", decoded_ok, decoded_failed, client_world.tick);
        std::printf("    link: %d sent, %d delivered, %d dropped; connection rtt "
                    "%.0f ms, loss %.0f%%\n", net.Sent(), net.Delivered(),
                    net.Dropped(), client.Stats().rtt_seconds * 1000.0f,
                    client.Stats().LossFraction() * 100.0f);
        Check(decoded_ok > 120, "most snapshots get through and decode");
        // THE RECOVERY PROPERTY. An undecodable snapshot must not be fatal: the
        // client keeps its old baseline, the server sees the unchanged ack, and
        // the next delta is one the client can apply. A protocol without that
        // loses one packet and never recovers.
        Check(client_world.tick >= 190, "and the client ends up nearly current");
        Snapshot drawn;
        Check(buffer.Sample(now - 0.05f, &drawn) && drawn.Find(1) != nullptr,
              "and has something to draw");
        const Vec3 p = drawn.Find(1)->position;
        const float radius = std::sqrt(p.x * p.x + p.z * p.z);
        std::printf("    the drawn entity is %.3f m from the origin (should be 5)\n",
                    radius);
        Check(std::fabs(radius - 5.0f) < 0.3f,
              "in roughly the right place after all that");
    }

    {
        std::printf("\naddresses are parsed or refused, never guessed\n");
        // strtoul with no endptr and no range check converts things that are
        // not ports into ports nobody asked for. Connecting to a port the
        // caller did not name is worse than refusing the string.
        const auto port_of = [](const char* text, bool* ok) {
            return eng::net::Address::Parse(text, ok).port;
        };
        bool ok = false;
        Check(port_of("127.0.0.1:7777", &ok) == 7777 && ok, "a real one works");
        Check(port_of("127.0.0.1:65535", &ok) == 65535 && ok, "and so does the top of the range");

        struct { const char* text; const char* why; } bad[] = {
            {"127.0.0.1:99999", "out of range, and 99999 & 0xFFFF is 33983"},
            {"127.0.0.1:-1", "negative, and strtoul wraps it to 65535"},
            {"127.0.0.1:80xyz", "trailing junk, silently taken as 80"},
            {"127.0.0.1:0", "port zero is not a port"},
            {"127.0.0.1:", "empty"},
            {"127.0.0.1: 80", "a space is not a digit"},
            {"127.0.0.1:+80", "nor is a sign"},
        };
        for (const auto& b : bad) {
            ok = true;
            const std::uint16_t p = port_of(b.text, &ok);
            std::printf("    %-18s -> port %5u, ok %d   (%s)\n", b.text, p, int(ok), b.why);
            Check(!ok, b.why);
        }
    }

    std::printf(g_failures == 0 ? "\nnet_test: all checks passed\n"
                                : "\nnet_test: %d FAILED\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
