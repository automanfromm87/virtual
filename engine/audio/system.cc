#include "engine/audio/system.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <vector>

#include "engine/audio/queue.h"
#include "engine/platform/audio_out.h"

namespace eng::audio {
namespace {

enum class Op : std::uint8_t { Play, Stop, StopAll, Gain, Position, Listener,
                               Master };

// One command, trivially copyable, no owning pointers. Everything the audio
// thread needs must be IN it: a command holding a reference to something the
// game thread might change is a data race with no symptom until it has one.
struct Command {
    Op op = Op::Play;
    std::uint32_t id = 0;
    PlayDesc desc;
    Vec3 vec_a{0, 0, 0}, vec_b{0, 0, 0}, vec_c{0, 0, 0};
    float value = 0.0f;
};

}  // namespace

struct AudioSystem::Impl {
    std::unique_ptr<platform::AudioOut> device;
    std::unique_ptr<Mixer> mixer;
    SpscQueue<Command, 512> queue;
    std::atomic<int> dropped{0};
    std::atomic<int> active{0};
    std::atomic<float> peak{0.0f};
    std::uint32_t next_id = 1;

    // Game-thread ids to the mixer's voices. Owned by the AUDIO thread and
    // touched by nothing else, which is why it is a plain array and not
    // anything synchronised.
    struct Live {
        std::uint32_t id = 0;
        VoiceId voice;
    };
    std::vector<Live> live;

    // A PUBLISHED MIRROR of `live`, so the game thread can ask whether a handle
    // is still audible without touching the table the audio thread owns. Copied
    // out at the end of every Drain; relaxed on both sides, because a stale
    // answer here is a frame old and nothing is ordered against it.
    std::vector<std::atomic<std::uint32_t>> published;
    // The highest Play id the audio thread has actually seen. A command sits in
    // the queue for up to a block, and without this a sound triggered on the
    // game thread reads as "finished" for the ten milliseconds before the audio
    // thread gets to it -- which is the wrong answer at the only moment anyone
    // asks. Compared with wrapping arithmetic, so the 32-bit id rolling over
    // after four billion sounds is not a cliff.
    std::atomic<std::uint32_t> drained_plays{0};
    std::atomic<int> starved{0};

    void Drain() {
        // RETIRE FIRST, then the commands. The other order -- which this was --
        // loses handles: with the table full and one voice having ended during
        // the last Render, a Play succeeds in the mixer but finds no free Live
        // slot, because the slot it should have taken is not released until the
        // bottom of the function. The sound plays and the caller's handle is
        // bound to nothing, so Stop, SetGain and SetPosition on it are silent
        // no-ops for the life of the voice.
        for (Live& l : live)
            if (l.id != 0 && !mixer->Playing(l.voice)) l.id = 0;

        Command c;
        while (queue.Pop(&c)) {
            switch (c.op) {
                case Op::Play: {
                    // Recorded whether or not a voice was found, because the
                    // question it answers is "has the audio thread seen this
                    // yet", not "did it succeed".
                    drained_plays.store(c.id, std::memory_order_relaxed);
                    const VoiceId v = mixer->Play(c.desc);
                    if (!v.Valid()) break;
                    for (Live& l : live)
                        if (l.id == 0) { l.id = c.id; l.voice = v; break; }
                    break;
                }
                case Op::Stop:
                    for (Live& l : live)
                        if (l.id == c.id) { mixer->Stop(l.voice); l.id = 0; }
                    break;
                case Op::StopAll:
                    mixer->StopAll();
                    for (Live& l : live) l.id = 0;
                    break;
                case Op::Gain:
                    for (Live& l : live)
                        if (l.id == c.id) mixer->SetGain(l.voice, c.value);
                    break;
                case Op::Position:
                    for (Live& l : live)
                        if (l.id == c.id) mixer->SetPosition(l.voice, c.vec_a);
                    break;
                case Op::Listener:
                    mixer->SetListener(c.vec_a, c.vec_b, c.vec_c);
                    break;
                case Op::Master:
                    mixer->SetMasterGain(c.value);
                    break;
            }
        }

        for (std::size_t i = 0; i < live.size(); ++i)
            published[i].store(live[i].id, std::memory_order_relaxed);
        starved.store(mixer->Starved(), std::memory_order_relaxed);
    }

    void Render(float* out, int frames) {
        // COMMANDS FIRST, then the mix. The other order plays a sound one block
        // late, which at a 512-frame buffer is eleven milliseconds -- audible
        // as a footstep landing after the foot.
        Drain();
        mixer->Render(out, frames);
        active.store(mixer->ActiveVoices(), std::memory_order_relaxed);
        peak.store(mixer->LastPeak(), std::memory_order_relaxed);
    }

    bool Post(const Command& c) {
        if (queue.Push(c)) return true;
        dropped.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
};

namespace {
void RenderThunk(void* user, float* out, int frames) {
    static_cast<AudioSystem::Impl*>(user)->Render(out, frames);
}
}  // namespace

AudioSystem::AudioSystem() : impl_(std::make_unique<Impl>()) {}

AudioSystem::~AudioSystem() {
    // The device FIRST. Its callback reads the mixer, so tearing the mixer down
    // while the device can still call is a use-after-free on a thread that does
    // not appear in any stack trace the crash reports.
    if (impl_) impl_->device.reset();
}

std::unique_ptr<AudioSystem> AudioSystem::CreateSilent(int rate, int max_voices) {
    std::unique_ptr<AudioSystem> s(new AudioSystem());
    const std::size_t n = std::size_t(std::max(1, max_voices));
    s->impl_->mixer = std::make_unique<Mixer>(rate, max_voices);
    s->impl_->live.resize(n);
    s->impl_->published = std::vector<std::atomic<std::uint32_t>>(n);
    return s;
}

std::unique_ptr<AudioSystem> AudioSystem::Create(std::string& error,
                                                 int sample_rate,
                                                 int max_voices) {
    std::unique_ptr<AudioSystem> s(new AudioSystem());
    Impl& im = *s->impl_;
    const std::size_t n = std::size_t(std::max(1, max_voices));
    im.live.resize(n);
    im.published = std::vector<std::atomic<std::uint32_t>>(n);
    im.device = platform::AudioOut::Create(sample_rate, &RenderThunk, &im, error);
    if (!im.device) return nullptr;
    // Built at the rate the DEVICE settled on, not the one asked for. A mixer
    // running at a different rate plays everything slightly fast and drifts out
    // of sync with the game over minutes.
    im.mixer = std::make_unique<Mixer>(im.device->SampleRate(), max_voices);
    im.device->Start();
    // CHECKED. Start() reports failure only through Running(), and a system
    // that was created but never started is worse than one that failed: the
    // game takes the `if (audio)` branch forever, posts commands into a queue
    // nothing drains, and after a few hundred frames is silently dropping every
    // one of them with no error anywhere to say why.
    if (!im.device->Running()) {
        error = "audio: the output device was created but would not start";
        return nullptr;
    }
    return s;
}

Sound AudioSystem::Play(const PlayDesc& d) {
    if (!d.clip || !d.clip->Valid()) return {};
    Command c;
    c.op = Op::Play;
    c.id = impl_->next_id++;
    if (impl_->next_id == 0) impl_->next_id = 1;
    c.desc = d;
    if (!impl_->Post(c)) return {};
    return Sound{c.id};
}

void AudioSystem::Stop(Sound s) {
    if (!s.Valid()) return;
    Command c;
    c.op = Op::Stop;
    c.id = s.id;
    (void)impl_->Post(c);
}

void AudioSystem::StopAll() {
    Command c;
    c.op = Op::StopAll;
    (void)impl_->Post(c);
}

void AudioSystem::SetGain(Sound s, float g) {
    if (!s.Valid()) return;
    Command c;
    c.op = Op::Gain;
    c.id = s.id;
    c.value = g;
    (void)impl_->Post(c);
}

void AudioSystem::SetPosition(Sound s, Vec3 p) {
    if (!s.Valid()) return;
    Command c;
    c.op = Op::Position;
    c.id = s.id;
    c.vec_a = p;
    (void)impl_->Post(c);
}

void AudioSystem::SetListener(Vec3 position, Vec3 forward, Vec3 up) {
    Command c;
    c.op = Op::Listener;
    c.vec_a = position;
    c.vec_b = forward;
    c.vec_c = up;
    (void)impl_->Post(c);
}

void AudioSystem::SetMasterGain(float g) {
    Command c;
    c.op = Op::Master;
    c.value = g;
    (void)impl_->Post(c);
}

int AudioSystem::SampleRate() const {
    return impl_->mixer ? impl_->mixer->SampleRate() : 0;
}
int AudioSystem::ActiveVoices() const {
    return impl_->active.load(std::memory_order_relaxed);
}
int AudioSystem::DroppedCommands() const {
    return impl_->dropped.load(std::memory_order_relaxed);
}
int AudioSystem::StarvedVoices() const {
    return impl_->starved.load(std::memory_order_relaxed);
}
bool AudioSystem::Playing(Sound s) const {
    if (!s.Valid()) return false;
    // Still in the queue counts as playing. The cast to signed is SERIAL NUMBER
    // arithmetic, not a sloppy comparison: it answers "is s.id after drained"
    // correctly across the point where the 32-bit counter wraps, which a plain
    // `>` does not.
    const std::uint32_t drained = impl_->drained_plays.load(std::memory_order_relaxed);
    if (std::int32_t(s.id - drained) > 0) return true;
    for (const auto& p : impl_->published)
        if (p.load(std::memory_order_relaxed) == s.id) return true;
    return false;
}
float AudioSystem::LastPeak() const {
    return impl_->peak.load(std::memory_order_relaxed);
}

void AudioSystem::RenderForTest(float* out, int frames) {
    impl_->Render(out, frames);
}

}  // namespace eng::audio
