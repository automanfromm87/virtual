#include "engine/platform/audio_out.h"

#import <AudioToolbox/AudioToolbox.h>

#include <algorithm>
#include <vector>

namespace eng::platform {

struct AudioOut::Impl {
    AudioUnit unit = nullptr;
    AudioOut::RenderFn render = nullptr;
    void* user = nullptr;
    int rate = 48000;
    bool running = false;
    // Scratch for the interleaved mix, sized once at startup. The callback must
    // not allocate, so this cannot be a vector that grows: it is sized for a
    // buffer far larger than any device will ask for, and a request past it is
    // truncated rather than served from a fresh allocation.
    std::vector<float> scratch;
};

namespace {

OSStatus Render(void* ref, AudioUnitRenderActionFlags*, const AudioTimeStamp*,
                UInt32, UInt32 frames, AudioBufferList* io) {
    auto* impl = static_cast<AudioOut::Impl*>(ref);
    // NON-INTERLEAVED output, which is what an audio unit asks for by default:
    // one buffer per channel rather than one buffer of pairs. The mixer works
    // interleaved because that is what every file format and every sample of
    // DSP literature uses, so the split happens here.
    float* left = io->mNumberBuffers > 0
                      ? static_cast<float*>(io->mBuffers[0].mData) : nullptr;
    float* right = io->mNumberBuffers > 1
                       ? static_cast<float*>(io->mBuffers[1].mData) : left;
    if (!left) return noErr;

    const int want = int(frames);
    const int can = std::min(want, int(impl->scratch.size() / 2));
    if (impl->render && can > 0) impl->render(impl->user, impl->scratch.data(), can);

    for (int i = 0; i < can; ++i) {
        left[i] = impl->scratch[std::size_t(i) * 2];
        right[i] = impl->scratch[std::size_t(i) * 2 + 1];
    }
    // Anything beyond the scratch is silence rather than stale samples. A
    // buffer left untouched repeats whatever it held, which is a loud buzz.
    for (int i = can; i < want; ++i) { left[i] = 0.0f; right[i] = 0.0f; }
    return noErr;
}

}  // namespace

AudioOut::AudioOut() : impl_(std::make_unique<Impl>()) {}

AudioOut::~AudioOut() {
    if (impl_ && impl_->unit) {
        // Stopped BEFORE uninitialised: tearing down a running unit leaves the
        // callback able to run against freed state for as long as the last
        // buffer takes to drain.
        AudioOutputUnitStop(impl_->unit);
        AudioUnitUninitialize(impl_->unit);
        AudioComponentInstanceDispose(impl_->unit);
    }
}

std::unique_ptr<AudioOut> AudioOut::Create(int sample_rate, RenderFn fn,
                                           void* user, std::string& error) {
    error.clear();
    std::unique_ptr<AudioOut> out(new AudioOut());
    Impl& im = *out->impl_;
    im.render = fn;
    im.user = user;
    im.rate = sample_rate > 0 ? sample_rate : 48000;
    // 8192 frames of headroom. A device asks for 256 to 1024; the margin costs
    // 64 KB and removes a whole class of "it only breaks on that interface".
    im.scratch.assign(8192 * 2, 0.0f);

    AudioComponentDescription desc = {};
    desc.componentType = kAudioUnitType_Output;
    desc.componentSubType = kAudioUnitSubType_DefaultOutput;
    desc.componentManufacturer = kAudioUnitManufacturer_Apple;
    AudioComponent comp = AudioComponentFindNext(nullptr, &desc);
    if (!comp) {
        error = "no default audio output component";
        return nullptr;
    }
    if (AudioComponentInstanceNew(comp, &im.unit) != noErr || !im.unit) {
        error = "could not create the audio unit";
        return nullptr;
    }

    AudioStreamBasicDescription fmt = {};
    fmt.mSampleRate = double(im.rate);
    fmt.mFormatID = kAudioFormatLinearPCM;
    // Non-interleaved float, which is CoreAudio's canonical format. Asking for
    // anything else makes the unit insert a converter.
    fmt.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked |
                       kAudioFormatFlagIsNonInterleaved;
    fmt.mChannelsPerFrame = 2;
    fmt.mBitsPerChannel = 32;
    fmt.mFramesPerPacket = 1;
    fmt.mBytesPerFrame = 4;
    fmt.mBytesPerPacket = 4;
    if (AudioUnitSetProperty(im.unit, kAudioUnitProperty_StreamFormat,
                             kAudioUnitScope_Input, 0, &fmt, sizeof(fmt)) != noErr) {
        error = "the audio unit refused the stream format";
        return nullptr;
    }

    AURenderCallbackStruct cb = {};
    cb.inputProc = &Render;
    cb.inputProcRefCon = &im;
    if (AudioUnitSetProperty(im.unit, kAudioUnitProperty_SetRenderCallback,
                             kAudioUnitScope_Input, 0, &cb, sizeof(cb)) != noErr) {
        error = "could not install the render callback";
        return nullptr;
    }
    if (AudioUnitInitialize(im.unit) != noErr) {
        error = "could not initialise the audio unit";
        return nullptr;
    }

    // What the device SETTLED on, which need not be what was asked for. A mixer
    // running at a different rate from the device is the classic cause of audio
    // that plays slightly fast and slowly drifts out of sync with the game.
    AudioStreamBasicDescription actual = {};
    UInt32 size = sizeof(actual);
    if (AudioUnitGetProperty(im.unit, kAudioUnitProperty_StreamFormat,
                             kAudioUnitScope_Input, 0, &actual, &size) == noErr &&
        actual.mSampleRate > 0)
        im.rate = int(actual.mSampleRate);
    return out;
}

int AudioOut::SampleRate() const { return impl_->rate; }
bool AudioOut::Running() const { return impl_->running; }

void AudioOut::Start() {
    if (impl_->unit && !impl_->running)
        impl_->running = AudioOutputUnitStart(impl_->unit) == noErr;
}

void AudioOut::Stop() {
    if (impl_->unit && impl_->running) {
        AudioOutputUnitStop(impl_->unit);
        impl_->running = false;
    }
}

}  // namespace eng::platform
