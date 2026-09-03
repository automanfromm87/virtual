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

    void Drain() {
        Command c;
        while (queue.Pop(&c)) {
            switch (c.op) {
                case Op::Play: {
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
        // Retire handles whose voice has finished, so the table does not fill
        // with the ids of sounds that ended minutes ago.
        for (Live& l : live)
            if (l.id != 0 && !mixer->Playing(l.voice)) l.id = 0;
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
    s->impl_->mixer = std::make_unique<Mixer>(rate, max_voices);
    s->impl_->live.resize(std::size_t(std::max(1, max_voices)));
    return s;
}

std::unique_ptr<AudioSystem> AudioSystem::Create(std::string& error,
                                                 int sample_rate,
                                                 int max_voices) {
    std::unique_ptr<AudioSystem> s(new AudioSystem());
    Impl& im = *s->impl_;
    im.live.resize(std::size_t(std::max(1, max_voices)));
    im.device = platform::AudioOut::Create(sample_rate, &RenderThunk, &im, error);
    if (!im.device) return nullptr;
    // Built at the rate the DEVICE settled on, not the one asked for. A mixer
    // running at a different rate plays everything slightly fast and drifts out
    // of sync with the game over minutes.
    im.mixer = std::make_unique<Mixer>(im.device->SampleRate(), max_voices);
    im.device->Start();
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
float AudioSystem::LastPeak() const {
    return impl_->peak.load(std::memory_order_relaxed);
}

void AudioSystem::RenderForTest(float* out, int frames) {
    impl_->Render(out, frames);
}

}  // namespace eng::audio
