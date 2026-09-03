#include "engine/platform/audio_file.h"

#import <AudioToolbox/AudioToolbox.h>
#import <Foundation/Foundation.h>

#include <vector>

namespace eng::platform {

audio::Clip DecodeAudioFile(const std::string& path, std::string& error) {
    @autoreleasepool {
        audio::Clip clip;
        error.clear();

        // Through the FILE MANAGER, not stringWithUTF8String:. A path is a byte
        // string and the filesystem permits bytes that are not valid UTF-8;
        // stringWithUTF8String: answers nil for those, and
        // [NSURL fileURLWithPath:nil] RAISES rather than returning nil, so a
        // badly named file would kill the process instead of costing a sound.
        NSString* p = [[NSFileManager defaultManager]
            stringWithFileSystemRepresentation:path.c_str()
                                        length:path.size()];
        if (!p) {
            error = "audio: " + path + " is not a usable path";
            return clip;
        }
        NSURL* url = [NSURL fileURLWithPath:p];
        ExtAudioFileRef file = nullptr;
        if (ExtAudioFileOpenURL((__bridge CFURLRef)url, &file) != noErr || !file) {
            error = "audio: cannot open " + path;
            return clip;
        }

        // What the file actually is, so its own sample rate and channel count
        // survive. Forcing everything to 44.1 kHz stereo here would resample
        // twice -- once to that and once to the device's rate -- for no reason.
        AudioStreamBasicDescription src = {};
        UInt32 size = sizeof(src);
        if (ExtAudioFileGetProperty(file, kExtAudioFileProperty_FileDataFormat,
                                    &size, &src) != noErr) {
            error = "audio: cannot read the format of " + path;
            ExtAudioFileDispose(file);
            return clip;
        }

        const int channels = int(src.mChannelsPerFrame) > 0
                                 ? int(src.mChannelsPerFrame) : 2;
        AudioStreamBasicDescription want = {};
        want.mSampleRate = src.mSampleRate;
        want.mFormatID = kAudioFormatLinearPCM;
        // INTERLEAVED float, because that is what Clip holds and what every
        // file format uses; the non-interleaved split happens at the speaker.
        want.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked;
        want.mChannelsPerFrame = UInt32(channels);
        want.mBitsPerChannel = 32;
        want.mFramesPerPacket = 1;
        want.mBytesPerFrame = UInt32(4 * channels);
        want.mBytesPerPacket = want.mBytesPerFrame;
        if (ExtAudioFileSetProperty(file, kExtAudioFileProperty_ClientDataFormat,
                                    sizeof(want), &want) != noErr) {
            error = "audio: the decoder refused float output for " + path;
            ExtAudioFileDispose(file);
            return clip;
        }

        SInt64 frames = 0;
        size = sizeof(frames);
        ExtAudioFileGetProperty(file, kExtAudioFileProperty_FileLengthFrames, &size,
                                &frames);
        // Bounded in BYTES, not frames. The resize below is frames * channels *
        // 4, and channels is whatever the file's header claims -- so a frame
        // count that looks reasonable on its own is 6.4 GB at eight channels,
        // allocated up front, from a number read straight out of an untrusted
        // header before a single sample has been decoded.
        constexpr SInt64 kMaxBytes = 2ll << 30;  // 2 GiB of decoded float
        const SInt64 bytes = frames * SInt64(channels) * 4;
        if (frames <= 0 || bytes > kMaxBytes) {
            error = "audio: " + path + " reports " + std::to_string(frames) +
                    " frames of " + std::to_string(channels) +
                    " channels, which is nothing or far too much to hold";
            ExtAudioFileDispose(file);
            return clip;
        }

        clip.channels = channels;
        clip.rate = int(src.mSampleRate);
        clip.samples.resize(std::size_t(frames) * std::size_t(channels));

        // Read in chunks and stop when a read returns nothing. The frame count
        // from the header is a hint: some encoders round it, and a decoder that
        // trusts it exactly either truncates the end or leaves uninitialised
        // memory at it -- which is a burst of noise on the last frame.
        std::size_t filled = 0;
        constexpr UInt32 kChunk = 16384;
        while (filled < clip.samples.size()) {
            const UInt32 remaining =
                UInt32((clip.samples.size() - filled) / std::size_t(channels));
            UInt32 n = remaining < kChunk ? remaining : kChunk;
            AudioBufferList list = {};
            list.mNumberBuffers = 1;
            list.mBuffers[0].mNumberChannels = UInt32(channels);
            list.mBuffers[0].mDataByteSize = n * UInt32(4 * channels);
            list.mBuffers[0].mData = clip.samples.data() + filled;
            if (ExtAudioFileRead(file, &n, &list) != noErr) {
                error = "audio: decoding failed part way through " + path;
                ExtAudioFileDispose(file);
                return audio::Clip{};
            }
            if (n == 0) break;  // the real end, whatever the header claimed
            filled += std::size_t(n) * std::size_t(channels);
        }
        clip.samples.resize(filled);
        ExtAudioFileDispose(file);

        if (!clip.Valid()) {
            error = "audio: " + path + " decoded to nothing";
            return audio::Clip{};
        }
        return clip;
    }
}

}  // namespace eng::platform
