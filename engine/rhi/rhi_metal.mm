// Objective-C++. The Metal implementation of the RHI.
//
// THE ONLY FILE IN THE ENGINE THAT KNOWS METAL EXISTS. If a Metal type appears
// anywhere else, the RHI boundary has failed and porting to a second backend
// means rewriting the renderer.
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>

#include "engine/rhi/rhi.h"

#include <vector>

namespace eng::rhi {
namespace {

MTLPixelFormat ToMTL(Format f) {
    switch (f) {
        case Format::RGBA8Unorm: return MTLPixelFormatRGBA8Unorm;
        case Format::BGRA8Unorm: return MTLPixelFormatBGRA8Unorm;
        case Format::RGBA16Float: return MTLPixelFormatRGBA16Float;
        case Format::Depth32Float: return MTLPixelFormatDepth32Float;
    }
    return MTLPixelFormatInvalid;
}

MTLCompareFunction ToMTL(Compare c) {
    switch (c) {
        case Compare::Never: return MTLCompareFunctionNever;
        case Compare::Less: return MTLCompareFunctionLess;
        case Compare::Greater: return MTLCompareFunctionGreater;
        case Compare::Always: return MTLCompareFunctionAlways;
    }
    return MTLCompareFunctionAlways;
}

MTLCullMode ToMTL(Cull c) {
    switch (c) {
        case Cull::None: return MTLCullModeNone;
        case Cull::Back: return MTLCullModeBack;
        case Cull::Front: return MTLCullModeFront;
    }
    return MTLCullModeNone;
}

MTLWinding ToMTL(Winding w) {
    return w == Winding::CounterClockwise ? MTLWindingCounterClockwise
                                          : MTLWindingClockwise;
}

struct PipelineObj {
    id<MTLRenderPipelineState> pso = nil;
    id<MTLDepthStencilState> dss = nil;  // nil when the pipeline has no depth
};

// Slot 1 of the texture table is reserved for whichever drawable is current.
// Registering a fresh handle per frame would grow the table without bound.
constexpr std::uint32_t kDrawableSlot = 1;

}  // namespace

// ---------------------------------------------------------------- Swapchain --

struct Swapchain::Impl {
    CAMetalLayer* layer = nil;
    id<CAMetalDrawable> current = nil;
};

Swapchain::Swapchain() : impl_(std::make_unique<Impl>()) {}
Swapchain::~Swapchain() = default;

void* Swapchain::NativeLayer() const { return (__bridge void*)impl_->layer; }

void Swapchain::Resize(int width, int height) {
    if (width >= 1 && height >= 1)
        impl_->layer.drawableSize = CGSizeMake(width, height);
}

int Swapchain::Width() const { return int(impl_->layer.drawableSize.width); }
int Swapchain::Height() const { return int(impl_->layer.drawableSize.height); }

// ------------------------------------------------------------------- Device --

struct Device::Impl {
    id<MTLDevice> dev = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLCommandBuffer> cb = nil;
    id<MTLRenderCommandEncoder> enc = nil;
    bool pass_has_depth = false;

    // Throttles the CPU to kFramesInFlight outstanding frames. Signalled from
    // each command buffer's completion handler.
    dispatch_semaphore_t frame_sem = nil;
    std::uint64_t frame_index = 0;
    std::atomic<int> in_flight{0};
    std::atomic<int> peak_in_flight{0};

    // Index 0 is the null handle in every table.
    std::vector<id<MTLBuffer>> buffers{nil};
    std::vector<id<MTLTexture>> textures{nil, nil};  // slot 1 = current drawable
    // Slots freed by DestroyTexture, ready to be handed out again. Without
    // this the table grows by one entry per window resize forever.
    std::vector<std::uint32_t> free_textures;

    // Never recycles slot 0 (null) or kDrawableSlot.
    std::uint32_t AllocTextureSlot(id<MTLTexture> t) {
        if (!free_textures.empty()) {
            const std::uint32_t slot = free_textures.back();
            free_textures.pop_back();
            textures[slot] = t;
            return slot;
        }
        textures.push_back(t);
        return std::uint32_t(textures.size() - 1);
    }
    std::vector<PipelineObj> pipelines{PipelineObj{}};
    std::vector<id<MTLSamplerState>> samplers{nil};
};

Device::Device() : impl_(std::make_unique<Impl>()) {}

Device::~Device() {
    // The completion blocks captured &impl_->in_flight as a RAW POINTER, and
    // they run on a Metal callback thread. Destroying Impl with frames still
    // outstanding lets those blocks decrement freed memory. Taking every permit
    // proves nothing is in flight; hand them back so dispatch_semaphore_dispose
    // does not trip on a count below its initial value.
    if (impl_ && impl_->frame_sem) {
        for (int i = 0; i < kFramesInFlight; ++i)
            dispatch_semaphore_wait(impl_->frame_sem, DISPATCH_TIME_FOREVER);
        for (int i = 0; i < kFramesInFlight; ++i)
            dispatch_semaphore_signal(impl_->frame_sem);
    }
}

std::unique_ptr<Device> Device::Create(std::string& error) {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) {
        error = "MTLCreateSystemDefaultDevice returned nil";
        return nullptr;
    }
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    if (!queue) {
        error = "newCommandQueue returned nil";
        return nullptr;
    }
    std::unique_ptr<Device> d(new Device());
    d->impl_->dev = dev;
    d->impl_->queue = queue;
    d->impl_->frame_sem = dispatch_semaphore_create(kFramesInFlight);
    return d;
}

BufferId Device::CreateBuffer(const void* data, std::size_t bytes) {
    if (!data || bytes == 0) return {};
    id<MTLBuffer> b = [impl_->dev newBufferWithBytes:data
                                              length:bytes
                                             options:MTLResourceStorageModeShared];
    if (!b) return {};
    impl_->buffers.push_back(b);
    return BufferId{std::uint32_t(impl_->buffers.size() - 1)};
}

BufferId Device::CreateDynamicBuffer(std::size_t bytes) {
    if (bytes == 0) return {};
    // Shared storage keeps one copy visible to both CPU and GPU on unified
    // memory; the pointer stays valid for the buffer's whole lifetime.
    id<MTLBuffer> b = [impl_->dev newBufferWithLength:bytes
                                              options:MTLResourceStorageModeShared];
    if (!b) return {};
    impl_->buffers.push_back(b);
    return BufferId{std::uint32_t(impl_->buffers.size() - 1)};
}

void* Device::MapBuffer(BufferId b) {
    if (!Valid(b) || b.v >= impl_->buffers.size()) return nullptr;
    return [impl_->buffers[b.v] contents];
}

std::size_t Device::UniformAlignment() const {
    // Intel macs want 256; Apple silicon is happy with 32. Taking the larger
    // is correct everywhere and costs a few hundred wasted bytes per frame.
    return 256;
}

TextureId Device::CreateRenderTarget(int width, int height, Format format,
                                     bool cpu_readable, int samples) {
    if (width <= 0 || height <= 0) return {};
    MTLTextureDescriptor* td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:ToMTL(format)
                                                           width:NSUInteger(width)
                                                          height:NSUInteger(height)
                                                       mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    if (samples > 1) {
        td.textureType = MTLTextureType2DMultisample;
        td.sampleCount = NSUInteger(samples);
        // A multisample target is written and resolved, never sampled. Saying
        // so lets the driver keep it in tile memory on a TBDR part, which is
        // the difference between 4x MSAA being nearly free and costing four
        // times the bandwidth of the whole frame.
        td.usage = MTLTextureUsageRenderTarget;
        td.storageMode = MTLStorageModePrivate;
        if (@available(macOS 11.0, *)) {
            if ([impl_->dev supportsFamily:MTLGPUFamilyApple1])
                td.storageMode = MTLStorageModeMemoryless;
        }
        id<MTLTexture> ms = [impl_->dev newTextureWithDescriptor:td];
        if (!ms) return {};
        return TextureId{impl_->AllocTextureSlot(ms)};
    }
    // Shared only when the CPU will actually look at it. Forcing Shared on
    // every target — including ones that only feed the next pass — gives up
    // layouts the driver would otherwise be free to choose.
    td.storageMode = cpu_readable ? MTLStorageModeShared : MTLStorageModePrivate;
    id<MTLTexture> t = [impl_->dev newTextureWithDescriptor:td];
    if (!t) return {};
    return TextureId{impl_->AllocTextureSlot(t)};
}

TextureId Device::CreateDepthTarget(int width, int height, bool sampleable,
                                    int samples) {
    if (width <= 0 || height <= 0) return {};
    MTLTextureDescriptor* td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:ToMTL(Format::Depth32Float)
                                     width:NSUInteger(width)
                                    height:NSUInteger(height)
                                 mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget |
               (sampleable ? MTLTextureUsageShaderRead : MTLTextureUsage(0));
    if (samples > 1) {
        td.textureType = MTLTextureType2DMultisample;
        td.sampleCount = NSUInteger(samples);
    }
    // Depth here is normally transient: the RHI exposes no way to sample or read
    // back a depth target, and every pass uses storeAction DontCare. On a TBDR
    // Apple GPU that means it can live entirely in tile memory and never touch
    // DRAM. Falls back to Private on Intel, where memoryless does not exist.
    bool tbdr = false;
    if (!sampleable) {
        if (@available(macOS 10.15, *)) {
            tbdr = [impl_->dev supportsFamily:MTLGPUFamilyApple1];
        }
    }
    td.storageMode = tbdr ? MTLStorageModeMemoryless : MTLStorageModePrivate;
    id<MTLTexture> t = [impl_->dev newTextureWithDescriptor:td];
    if (!t && tbdr) {
        // Memoryless refused for some reason — take the allocation rather than
        // failing the frame.
        td.storageMode = MTLStorageModePrivate;
        t = [impl_->dev newTextureWithDescriptor:td];
    }
    if (!t) return {};
    return TextureId{impl_->AllocTextureSlot(t)};
}

TextureId Device::CreateTexture2D(int width, int height, const void* rgba8) {
    if (width <= 0 || height <= 0 || !rgba8) return {};
    MTLTextureDescriptor* td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                     width:NSUInteger(width)
                                    height:NSUInteger(height)
                                 mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    // Shared so the upload below is a plain memcpy. A Private texture would
    // need a staging buffer and a blit — worth it for large assets, pointless
    // for the procedural ones this engine has.
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> t = [impl_->dev newTextureWithDescriptor:td];
    if (!t) return {};
    [t replaceRegion:MTLRegionMake2D(0, 0, NSUInteger(width), NSUInteger(height))
         mipmapLevel:0
           withBytes:rgba8
         bytesPerRow:NSUInteger(width) * 4];
    return TextureId{impl_->AllocTextureSlot(t)};
}

SamplerId Device::CreateSampler(Filter filter, Wrap wrap) {
    MTLSamplerDescriptor* sd = [[MTLSamplerDescriptor alloc] init];
    const MTLSamplerMinMagFilter f = (filter == Filter::Linear)
                                         ? MTLSamplerMinMagFilterLinear
                                         : MTLSamplerMinMagFilterNearest;
    sd.minFilter = f;
    sd.magFilter = f;
    const MTLSamplerAddressMode a = (wrap == Wrap::Repeat)
                                        ? MTLSamplerAddressModeRepeat
                                        : MTLSamplerAddressModeClampToEdge;
    sd.sAddressMode = a;
    sd.tAddressMode = a;
    id<MTLSamplerState> ss = [impl_->dev newSamplerStateWithDescriptor:sd];
    if (!ss) return {};
    impl_->samplers.push_back(ss);
    return SamplerId{std::uint32_t(impl_->samplers.size() - 1)};
}

TextureId Device::CreateShadowMap(int size) {
    if (size <= 0) return {};
    MTLTextureDescriptor* td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:ToMTL(Format::Depth32Float)
                                     width:NSUInteger(size)
                                    height:NSUInteger(size)
                                 mipmapped:NO];
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    // NOT memoryless: another pass samples this one, so it has to survive past
    // the end of the pass that writes it.
    td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> t = [impl_->dev newTextureWithDescriptor:td];
    if (!t) return {};
    return TextureId{impl_->AllocTextureSlot(t)};
}

PipelineId Device::CreatePipeline(const PipelineDesc& desc, std::string& error) {
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:desc.source.c_str()];
    id<MTLLibrary> lib = [impl_->dev newLibraryWithSource:src
                                                  options:[MTLCompileOptions new]
                                                    error:&err];
    if (!lib) {
        error = std::string("shader compile failed: ") +
                (err ? err.localizedDescription.UTF8String : "unknown error");
        return {};
    }

    // newFunctionWithName returns nil for a missing entry point; assigning that
    // nil to the descriptor fails much later with a far worse message.
    id<MTLFunction> vs =
        [lib newFunctionWithName:[NSString stringWithUTF8String:desc.vertex_fn.c_str()]];
    // A depth-only pipeline usually has no fragment stage, but it MAY have a
    // void one — a shadow pass still has to be able to discard.
    id<MTLFunction> fs = nil;
    if (!desc.fragment_fn.empty()) {
        fs = [lib newFunctionWithName:
                      [NSString stringWithUTF8String:desc.fragment_fn.c_str()]];
    }
    if (!vs || (!desc.fragment_fn.empty() && !fs)) {
        error = "shader is missing " + desc.vertex_fn + " and/or " + desc.fragment_fn;
        return {};
    }

    MTLRenderPipelineDescriptor* pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = vs;
    pd.fragmentFunction = fs;
    // A depth-only pipeline has NO colour attachment. Declaring one that the
    // pass will not provide is a pipeline/pass mismatch.
    pd.colorAttachments[0].pixelFormat =
        desc.depth_only ? MTLPixelFormatInvalid : ToMTL(desc.color);
    // Without this the depth attachment is silently ignored at draw time.
    if (desc.depth) pd.depthAttachmentPixelFormat = ToMTL(Format::Depth32Float);
    // Has to match the attachments exactly. Metal rejects a mismatch here,
    // which is the one class of format error it does NOT let through silently.
    pd.rasterSampleCount = NSUInteger(desc.samples > 0 ? desc.samples : 1);
    if (desc.blend && !desc.depth_only) {
        MTLRenderPipelineColorAttachmentDescriptor* ca = pd.colorAttachments[0];
        ca.blendingEnabled = YES;
        ca.rgbBlendOperation = MTLBlendOperationAdd;
        ca.alphaBlendOperation = MTLBlendOperationAdd;
        ca.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        ca.sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
        ca.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        ca.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    }

    PipelineObj obj;
    obj.pso = [impl_->dev newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!obj.pso) {
        error = std::string("pipeline creation failed: ") +
                (err ? err.localizedDescription.UTF8String : "unknown error");
        return {};
    }
    if (desc.depth) {
        MTLDepthStencilDescriptor* dsd = [[MTLDepthStencilDescriptor alloc] init];
        dsd.depthCompareFunction = ToMTL(desc.depth_compare);
        dsd.depthWriteEnabled = desc.depth_write ? YES : NO;
        obj.dss = [impl_->dev newDepthStencilStateWithDescriptor:dsd];
        if (!obj.dss) {
            error = "newDepthStencilStateWithDescriptor returned nil";
            return {};
        }
    }
    impl_->pipelines.push_back(obj);
    return PipelineId{std::uint32_t(impl_->pipelines.size() - 1)};
}

void Device::DestroyTexture(TextureId t) {
    if (!Valid(t) || t.v >= impl_->textures.size()) return;
    if (t.v == kDrawableSlot) return;  // owned by the swapchain, not the caller
    if (!impl_->textures[t.v]) return;  // already destroyed; do not double-free
    impl_->textures[t.v] = nil;         // ARC releases the MTLTexture here
    impl_->free_textures.push_back(t.v);
}

std::unique_ptr<Swapchain> Device::CreateSwapchain(Format format, std::string& error) {
    CAMetalLayer* layer = [CAMetalLayer layer];
    if (!layer) {
        error = "CAMetalLayer creation failed";
        return nullptr;
    }
    layer.device = impl_->dev;
    layer.pixelFormat = ToMTL(format);
    layer.framebufferOnly = YES;
    std::unique_ptr<Swapchain> sc(new Swapchain());
    sc->impl_->layer = layer;
    return sc;
}

TextureId Device::AcquireDrawable(Swapchain& sc) { @autoreleasepool {
    // Blocks until a drawable frees up — this is what paces a windowed app to
    // vsync. nil means the frame was dropped.
    id<CAMetalDrawable> d = [sc.impl_->layer nextDrawable];
    if (!d) return {};
    sc.impl_->current = d;
    impl_->textures[kDrawableSlot] = d.texture;
    return TextureId{kDrawableSlot};
}}

void Device::BeginFrame() { @autoreleasepool {
    // Blocks when kFramesInFlight frames are already outstanding. This is the
    // whole reason a ring buffer is safe: by the time we reuse slot N, the
    // frame that last used it has provably finished on the GPU.
    dispatch_semaphore_wait(impl_->frame_sem, DISPATCH_TIME_FOREVER);

    const int now = impl_->in_flight.fetch_add(1) + 1;
    int prev = impl_->peak_in_flight.load();
    while (now > prev && !impl_->peak_in_flight.compare_exchange_weak(prev, now)) {
    }

    impl_->cb = [impl_->queue commandBuffer];
    ++impl_->frame_index;

    dispatch_semaphore_t sem = impl_->frame_sem;
    std::atomic<int>* counter = &impl_->in_flight;
    [impl_->cb addCompletedHandler:^(id<MTLCommandBuffer>) {
        counter->fetch_sub(1);
        dispatch_semaphore_signal(sem);
    }];
}}

int Device::FrameSlot() const {
    // frame_index is pre-incremented in BeginFrame, so subtract one. Guard the
    // pre-first-frame case explicitly: 0u - 1 would wrap to a huge value and
    // only happen to land on 0 by arithmetic accident.
    if (impl_->frame_index == 0) return 0;
    return int((impl_->frame_index - 1) % kFramesInFlight);
}

std::uint64_t Device::FrameIndex() const { return impl_->frame_index; }

int Device::PeakFramesInFlight() const { return impl_->peak_in_flight.load(); }

int Device::TextureSlotCount() const { return int(impl_->textures.size()); }

Encoder Device::BeginPass(const PassDesc& desc) { @autoreleasepool {
    // Descriptors are autoreleased. Without a pool here the engine leaks one
    // per pass per frame for the whole run — there is no Cocoa run loop to
    // drain them, because the app drives its own loop.
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    if (Valid(desc.color)) {
        rp.colorAttachments[0].texture = impl_->textures[desc.color.v];
        rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        // RESOLVE rather than store, when the caller supplied somewhere to
        // resolve into. Storing the samples themselves would write four times
        // the data and nothing can read it anyway.
        if (Valid(desc.resolve)) {
            rp.colorAttachments[0].resolveTexture = impl_->textures[desc.resolve.v];
            rp.colorAttachments[0].storeAction = MTLStoreActionMultisampleResolve;
        } else {
            rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        }
        rp.colorAttachments[0].clearColor =
            MTLClearColorMake(desc.clear_color[0], desc.clear_color[1],
                              desc.clear_color[2], desc.clear_color[3]);
    }
    if (Valid(desc.depth)) {
        rp.depthAttachment.texture = impl_->textures[desc.depth.v];
        rp.depthAttachment.loadAction = MTLLoadActionClear;
        rp.depthAttachment.storeAction =
            desc.keep_depth ? MTLStoreActionStore : MTLStoreActionDontCare;
        rp.depthAttachment.clearDepth = desc.clear_depth;
    }
    // The encoder is held by a strong ivar, so it outlives this pool.
    impl_->pass_has_depth = Valid(desc.depth);
    impl_->enc = [impl_->cb renderCommandEncoderWithDescriptor:rp];
    return Encoder(this);
}}

void Device::EndPass() {
    [impl_->enc endEncoding];
    impl_->enc = nil;
    impl_->pass_has_depth = false;
}

bool Device::CurrentPassHasDepth() const { return impl_->pass_has_depth; }

void Device::Present(Swapchain& sc) {
    if (sc.impl_->current) {
        [impl_->cb presentDrawable:sc.impl_->current];
        sc.impl_->current = nil;
    }
    // Release the drawable's texture. BeginPass already copied it into the pass
    // descriptor, so nothing still needs the handle — and holding it would keep
    // one drawable alive until the next AcquireDrawable, starving the
    // swapchain's small pool.
    impl_->textures[kDrawableSlot] = nil;
}

void Device::Commit() {
    [impl_->cb commit];
    impl_->cb = nil;
}

bool Device::CommitAndWait(std::string& error) {
    [impl_->cb commit];
    [impl_->cb waitUntilCompleted];  // MUST sync before reading pixels back
    const bool ok = (impl_->cb.error == nil);
    if (!ok) {
        error = std::string("GPU execution failed: ") +
                impl_->cb.error.localizedDescription.UTF8String;
    }
    impl_->cb = nil;
    return ok;
}

bool Device::ReadPixels(TextureId tex, int width, int height,
                        std::span<std::uint8_t> out) {
    if (!Valid(tex) || width <= 0 || height <= 0) return false;
    // A Private texture has no CPU-visible contents; getBytes on one is a
    // Metal validation error rather than a silent zero fill.
    if (impl_->textures[tex.v].storageMode != MTLStorageModeShared) return false;
    // Four bytes per pixel is baked into the stride below, so a half-float
    // target would be read at half its width and the caller would get a
    // plausible, wrong image rather than an error.
    const MTLPixelFormat pf = impl_->textures[tex.v].pixelFormat;
    if (pf != MTLPixelFormatRGBA8Unorm && pf != MTLPixelFormatBGRA8Unorm)
        return false;
    const std::size_t need = std::size_t(width) * std::size_t(height) * 4;
    if (out.size() < need) return false;
    [impl_->textures[tex.v]
             getBytes:out.data()
          bytesPerRow:std::size_t(width) * 4
           fromRegion:MTLRegionMake2D(0, 0, NSUInteger(width), NSUInteger(height))
          mipmapLevel:0];
    return true;
}

// ------------------------------------------------------------------ Encoder --

void Encoder::SetPipeline(PipelineId p) {
    const PipelineObj& obj = device_->impl_->pipelines[p.v];
    [device_->impl_->enc setRenderPipelineState:obj.pso];
    if (obj.dss) [device_->impl_->enc setDepthStencilState:obj.dss];
}

void Encoder::SetViewport(int x, int y, int width, int height) {
    const MTLViewport vp{double(x), double(y), double(width), double(height),
                         0.0, 1.0};
    [device_->impl_->enc setViewport:vp];
}

void Encoder::SetScissor(int x, int y, int width, int height) {
    const MTLScissorRect r{NSUInteger(x), NSUInteger(y), NSUInteger(width),
                           NSUInteger(height)};
    [device_->impl_->enc setScissorRect:r];
}

void Encoder::SetCull(Cull c, Winding w) {
    [device_->impl_->enc setFrontFacingWinding:ToMTL(w)];
    [device_->impl_->enc setCullMode:ToMTL(c)];
}

void Encoder::SetVertexBuffer(BufferId b, std::size_t offset, int slot) {
    [device_->impl_->enc setVertexBuffer:device_->impl_->buffers[b.v]
                                  offset:offset
                                 atIndex:NSUInteger(slot)];
}

void Encoder::SetFragmentBuffer(BufferId b, std::size_t offset, int slot) {
    [device_->impl_->enc setFragmentBuffer:device_->impl_->buffers[b.v]
                                    offset:offset
                                   atIndex:NSUInteger(slot)];
}

void Encoder::SetVertexBytes(const void* data, std::size_t bytes, int slot) {
    [device_->impl_->enc setVertexBytes:data
                                 length:bytes
                                atIndex:NSUInteger(slot)];
}

void Encoder::SetFragmentBytes(const void* data, std::size_t bytes, int slot) {
    [device_->impl_->enc setFragmentBytes:data
                                   length:bytes
                                  atIndex:NSUInteger(slot)];
}

void Encoder::SetFragmentTexture(TextureId t, int slot) {
    [device_->impl_->enc setFragmentTexture:device_->impl_->textures[t.v]
                                    atIndex:NSUInteger(slot)];
}

void Encoder::SetFragmentSampler(SamplerId s, int slot) {
    [device_->impl_->enc setFragmentSamplerState:device_->impl_->samplers[s.v]
                                         atIndex:NSUInteger(slot)];
}

void Encoder::Draw(std::size_t vertex_count) {
    [device_->impl_->enc drawPrimitives:MTLPrimitiveTypeTriangle
                            vertexStart:0
                            vertexCount:vertex_count];
}

void Encoder::DrawIndexedU16(BufferId indices, std::size_t index_count) {
    [device_->impl_->enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                    indexCount:index_count
                                     indexType:MTLIndexTypeUInt16
                                   indexBuffer:device_->impl_->buffers[indices.v]
                             indexBufferOffset:0];
}

}  // namespace eng::rhi
