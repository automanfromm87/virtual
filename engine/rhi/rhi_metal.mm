// Objective-C++. The Metal implementation of the RHI.
//
// THE ONLY FILE IN THE ENGINE THAT KNOWS METAL EXISTS. If a Metal type appears
// anywhere else, the RHI boundary has failed and porting to a second backend
// means rewriting the renderer.
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <AppKit/AppKit.h>
#include <dispatch/dispatch.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>

#include "engine/rhi/rhi.h"

#include <vector>

namespace eng::rhi {
namespace {

MTLPixelFormat ToMTL(Format f) {
    switch (f) {
        case Format::RGBA8Unorm: return MTLPixelFormatRGBA8Unorm;
        case Format::RGBA8Srgb: return MTLPixelFormatRGBA8Unorm_sRGB;
        case Format::BC1: return MTLPixelFormatBC1_RGBA;
        case Format::BC1Srgb: return MTLPixelFormatBC1_RGBA_sRGB;
        case Format::BC3: return MTLPixelFormatBC3_RGBA;
        case Format::BC3Srgb: return MTLPixelFormatBC3_RGBA_sRGB;
        case Format::BC5: return MTLPixelFormatBC5_RGUnorm;
        case Format::BGRA8Unorm: return MTLPixelFormatBGRA8Unorm;
        case Format::RGBA16Float: return MTLPixelFormatRGBA16Float;
        case Format::Depth32Float: return MTLPixelFormatDepth32Float;
        case Format::RG16Float: return MTLPixelFormatRG16Float;
        case Format::RGBA32Float: return MTLPixelFormatRGBA32Float;
    }
    return MTLPixelFormatInvalid;
}

MTLCompareFunction ToMTL(Compare c) {
    switch (c) {
        case Compare::Never: return MTLCompareFunctionNever;
        case Compare::Less: return MTLCompareFunctionLess;
        case Compare::Greater: return MTLCompareFunctionGreater;
        case Compare::GreaterEqual: return MTLCompareFunctionGreaterEqual;
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
    // The last buffer handed to the GPU, kept alive after `cb` is cleared so a
    // readback has something to wait on. See ReadPixels.
    id<MTLCommandBuffer> submitted = nil;
    id<MTLRenderCommandEncoder> enc = nil;
    bool pass_has_depth = false;
    // The current pass's view count. Applied when a pipeline is BOUND rather
    // than when the pass begins: setVertexAmplificationCount is validated
    // against the bound pipeline's maxVertexAmplificationCount, and at the top
    // of a pass there is no pipeline bound for it to be validated against, so
    // the call is accepted and then forgotten.
    int pass_views = 1;

    // --- GPU timing -----------------------------------------------------------
    //
    // Two slots per timed pass -- Metal writes a GPU tick into each at a stage
    // boundary and the difference is the pass's duration -- and ONE REGION PER
    // FRAME SLOT, which is the part this got wrong for its whole life.
    //
    // It was a single region of kMaxTimedPasses pairs, reused every frame, and
    // the comment defending that said the reuse was "safe for the same reason
    // the uniform ring is: the frame that last wrote these slots has provably
    // completed". That reason does not apply here. The uniform ring is indexed
    // by FrameSlot; this was not indexed at all. BeginFrame resets
    // timed_this_frame to zero, so frame N+1 began overwriting slot 0 while
    // frame N's samples were still unresolved -- with kFramesInFlight = 3,
    // three frames writing one region. The symptom was not silence, which is
    // why it survived: every pass reported a plausible number, drawn from
    // whichever frame happened to write last. A static camera produced a
    // bimodal shadow time (2.0 ms and 16.3 ms in the same run) and per-pass
    // totals two to three times the frame's own GPU time.
    bool timing_supported = false;
    id<MTLCounterSampleBuffer> counters = nil;
    static constexpr int kMaxTimedPasses = 32;
    // Labels recorded while encoding THIS frame, in the order the passes were
    // begun. Copied into the completion handler, which resolves them.
    std::vector<const char*> timing_labels;
    std::atomic<double> gpu_ms{0.0};
    int timed_this_frame = 0;

    // PUBLISHED UNDER A LOCK, and read out as a COPY.
    //
    // The reader used to be handed `{timings.data(), timings.size()}` -- a raw
    // span into a vector that a driver-thread completion block swapped, and
    // whose swapped-out buffer the NEXT completion destroyed. That is a
    // use-after-free with a two-frame fuse, and it stayed hidden only because
    // nothing outside a test ever called LastFrameTimings.
    //
    // A mutex rather than an atomic swap because there is more than one
    // producer: up to kFramesInFlight completion handlers can run at once, on
    // whatever threads Metal chooses. `timings_frame` is what keeps them in
    // order -- a handler that completes late must not overwrite a newer frame's
    // numbers with its own.
    std::mutex timings_mutex;
    std::vector<GpuTiming> timings;
    std::uint64_t timings_frame = 0;

    // --- GPU faults -------------------------------------------------------
    //
    // A command buffer that fails on the GPU sets `error` and produces NOTHING
    // -- no draws, no error return, no exception. Commit() never looked, so the
    // only symptom reaching a person was a black or stale frame, and three
    // recent commit messages say exactly that in three different
    // investigations. The buffer is created asking for
    // EncoderExecutionStatus, so the fault can name the encoder that faulted
    // rather than only the frame, which is the difference between a bug report
    // and a bisect.
    std::mutex fault_mutex;
    std::string last_fault;
    int fault_count = 0;

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

    // Where this frame's pairs start in the shared buffer. Frame slot N owns
    // pairs [N * kMaxTimedPasses, (N+1) * kMaxTimedPasses).
    [[nodiscard]] int TimingBase() const {
        const std::uint64_t f = frame_index == 0 ? 0 : frame_index - 1;
        return int(f % kFramesInFlight) * kMaxTimedPasses;
    }

    // Claims a pair of timestamp slots for a pass, or -1 when the pass asked
    // for no timing, the hardware cannot sample, or the frame has already used
    // all of them. Dropping the extras silently is deliberate: running out of
    // timer slots must degrade the profile, not the frame.
    int BeginTiming(const char* label) {
        if (!label || !timing_supported) return -1;
        if (timed_this_frame >= kMaxTimedPasses) return -1;
        const int slot = TimingBase() + timed_this_frame++;
        timing_labels.push_back(label);
        return slot;
    }

    // Never recycles slot 0 (null) or kDrawableSlot.
    //
    // LABELS EVERY TEXTURE ON THE WAY THROUGH, which is why the label is set
    // here and not at the twenty call sites: this is the one funnel they all
    // pass through, and the description is read off the texture itself rather
    // than passed in, so no creation signature has to grow a name. Without it
    // an Xcode capture of this engine is a list of numbered resources, and the
    // one thing a capture is for -- finding out which target a pass is reading
    // -- takes longer than reading the source.
    std::uint32_t AllocTextureSlot(id<MTLTexture> t) {
        std::uint32_t slot;
        if (!free_textures.empty()) {
            slot = free_textures.back();
            free_textures.pop_back();
            textures[slot] = t;
        } else {
            textures.push_back(t);
            slot = std::uint32_t(textures.size() - 1);
        }
        if (t.label == nil) {
            NSMutableString* n = [NSMutableString
                stringWithFormat:@"tex%u %lux%lu", slot,
                                 (unsigned long)t.width, (unsigned long)t.height];
            if (t.depth > 1) [n appendFormat:@"x%lu", (unsigned long)t.depth];
            if (t.arrayLength > 1) [n appendFormat:@"[%lu]", (unsigned long)t.arrayLength];
            if (t.mipmapLevelCount > 1) [n appendFormat:@" %lumip", (unsigned long)t.mipmapLevelCount];
            if (t.sampleCount > 1) [n appendFormat:@" %lux", (unsigned long)t.sampleCount];
            [n appendFormat:@" fmt%lu", (unsigned long)t.pixelFormat];
            t.label = n;
        }
        return slot;
    }
    // Same reason as AllocTextureSlot: the label is what makes a capture
    // readable, and there is no reason for a creation signature to grow a name
    // when the kind and the size already say which buffer this is.
    std::uint32_t AllocBufferSlot(id<MTLBuffer> b, const char* kind) {
        if (b.label == nil)
            b.label = [NSString stringWithFormat:@"%s%zu %zuB", kind,
                                                 buffers.size(),
                                                 std::size_t(b.length)];
        buffers.push_back(b);
        return std::uint32_t(buffers.size() - 1);
    }
    std::vector<PipelineObj> pipelines{PipelineObj{}};
    std::vector<id<MTLComputePipelineState>> compute_pipelines{nil};
    id<MTLComputeCommandEncoder> compute_enc = nil;
    // The threadgroup width the last-bound pipeline actually wants. Dispatch
    // needs it, and asking the pipeline rather than assuming 64 is what keeps a
    // kernel that declares a different size from silently running fewer threads
    // than the caller asked for.
    int compute_group_max = 64;
    std::vector<id<MTLSamplerState>> samplers{nil};

    // Acceleration structures, and for each one everything the GPU must have
    // resident to traverse it. A top-level structure REFERENCES its bottom-level
    // ones, and those reference the vertex and index buffers, and none of that
    // is visible to the driver's automatic residency tracking -- binding only
    // the top-level structure and tracing into it is a GPU fault, not a wrong
    // picture.
    std::vector<id<MTLAccelerationStructure>> accels{nil};
    std::vector<std::vector<id<MTLResource>>> accel_deps{{}};
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

bool Device::SupportsRaytracing() const {
    return impl_->dev != nil && impl_->dev.supportsRaytracing;
}

namespace {

// Builds one acceleration structure and waits for it. Synchronous on purpose:
// these are built at load time, and threading a build through the frame's
// command buffer would mean the first frame traces a structure that is not
// finished yet -- which reads as geometry missing from reflections for one
// frame and is very hard to attribute.
id<MTLAccelerationStructure> BuildAccel(id<MTLDevice> dev,
                                        id<MTLCommandQueue> queue,
                                        MTLAccelerationStructureDescriptor* desc,
                                        std::string& error) {
    const MTLAccelerationStructureSizes sizes =
        [dev accelerationStructureSizesWithDescriptor:desc];
    id<MTLAccelerationStructure> accel =
        [dev newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    if (!accel) {
        error = "could not allocate an acceleration structure";
        return nil;
    }
    id<MTLBuffer> scratch =
        [dev newBufferWithLength:std::max<NSUInteger>(sizes.buildScratchBufferSize, 16)
                         options:MTLResourceStorageModePrivate];
    if (!scratch) {
        error = "could not allocate acceleration structure scratch";
        return nil;
    }
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> enc =
        [cb accelerationStructureCommandEncoder];
    [enc buildAccelerationStructure:accel descriptor:desc scratchBuffer:scratch
               scratchBufferOffset:0];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        error = std::string("acceleration structure build failed: ") +
                cb.error.localizedDescription.UTF8String;
        return nil;
    }
    return accel;
}

}  // namespace

AccelId Device::CreateBlas(BufferId vb, int vertex_stride, BufferId ib,
                           int index_count, std::string& error) {
    if (!SupportsRaytracing()) {
        error = "this device has no hardware ray tracing";
        return {};
    }
    if (!Valid(vb) || vb.v >= impl_->buffers.size() || !Valid(ib) ||
        ib.v >= impl_->buffers.size() || vertex_stride <= 0 || index_count <= 0) {
        error = "CreateBlas: bad buffer or count";
        return {};
    }
    // Triangles, so the index count must divide by three. A partial triangle
    // would be read as whatever follows it in the buffer.
    if (index_count % 3 != 0) {
        error = "CreateBlas: index count is not a multiple of 3";
        return {};
    }

    MTLAccelerationStructureTriangleGeometryDescriptor* geo =
        [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
    geo.vertexBuffer = impl_->buffers[vb.v];
    geo.vertexBufferOffset = 0;
    geo.vertexStride = NSUInteger(vertex_stride);
    geo.indexBuffer = impl_->buffers[ib.v];
    geo.indexBufferOffset = 0;
    geo.indexType = MTLIndexTypeUInt32;
    geo.triangleCount = NSUInteger(index_count / 3);
    // OPAQUE. Without it every hit would invoke an intersection function to ask
    // whether it counts, which is both slower and pointless for geometry that
    // has no alpha cutout.
    geo.opaque = YES;

    MTLPrimitiveAccelerationStructureDescriptor* desc =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    desc.geometryDescriptors = @[geo];

    id<MTLAccelerationStructure> accel =
        BuildAccel(impl_->dev, impl_->queue, desc, error);
    if (!accel) return {};

    impl_->accels.push_back(accel);
    // The geometry buffers have to stay resident whenever this structure is
    // traversed: the BVH's leaves point into them.
    impl_->accel_deps.push_back({impl_->buffers[vb.v], impl_->buffers[ib.v]});
    return AccelId{std::uint32_t(impl_->accels.size() - 1)};
}

AccelId Device::CreateTlas(std::span<const AccelInstance> instances,
                           std::string& error) {
    if (!SupportsRaytracing()) {
        error = "this device has no hardware ray tracing";
        return {};
    }
    if (instances.empty()) {
        error = "CreateTlas: no instances";
        return {};
    }

    // The distinct bottom-level structures, and each instance's index into that
    // list. Deduplicating is the entire point of a two-level structure: a
    // thousand crates share one BVH and differ only by a transform.
    NSMutableArray<id<MTLAccelerationStructure>>* blas_list = [NSMutableArray array];
    std::vector<std::uint32_t> blas_of_instance;
    std::vector<std::uint32_t> unique;
    std::vector<id<MTLResource>> deps;
    for (const AccelInstance& inst : instances) {
        if (!Valid(inst.blas) || inst.blas.v >= impl_->accels.size()) {
            error = "CreateTlas: an instance names an unbuilt structure";
            return {};
        }
        std::uint32_t at = 0;
        bool found = false;
        for (std::uint32_t i = 0; i < unique.size(); ++i)
            if (unique[i] == inst.blas.v) { at = i; found = true; break; }
        if (!found) {
            at = std::uint32_t(unique.size());
            unique.push_back(inst.blas.v);
            [blas_list addObject:impl_->accels[inst.blas.v]];
            deps.push_back(impl_->accels[inst.blas.v]);
            for (id<MTLResource> r : impl_->accel_deps[inst.blas.v])
                deps.push_back(r);
        }
        blas_of_instance.push_back(at);
    }

    id<MTLBuffer> inst_buf = [impl_->dev
        newBufferWithLength:sizeof(MTLAccelerationStructureInstanceDescriptor) *
                            instances.size()
                    options:MTLResourceStorageModeShared];
    if (!inst_buf) {
        error = "CreateTlas: could not allocate the instance buffer";
        return {};
    }
    auto* d = static_cast<MTLAccelerationStructureInstanceDescriptor*>(
        inst_buf.contents);
    for (std::size_t i = 0; i < instances.size(); ++i) {
        d[i] = {};
        d[i].accelerationStructureIndex = blas_of_instance[i];
        d[i].options = MTLAccelerationStructureInstanceOptionOpaque;
        d[i].mask = 0xFF;
        d[i].intersectionFunctionTableOffset = 0;
        // MTLPackedFloat4x3 holds four COLUMNS of three components: the three
        // basis vectors and then the translation. That is the same order the
        // engine's column-major 4x4 already has, minus the bottom row -- so
        // this is a copy and not a transpose. Transposing it here is the
        // classic way to get a scene that renders correctly and whose shadows
        // are rotated.
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 3; ++r)
                d[i].transformationMatrix.columns[c][r] =
                    instances[i].transform[c * 4 + r];
    }

    MTLInstanceAccelerationStructureDescriptor* desc =
        [MTLInstanceAccelerationStructureDescriptor descriptor];
    desc.instancedAccelerationStructures = blas_list;
    desc.instanceCount = instances.size();
    desc.instanceDescriptorBuffer = inst_buf;

    id<MTLAccelerationStructure> accel =
        BuildAccel(impl_->dev, impl_->queue, desc, error);
    if (!accel) return {};

    impl_->accels.push_back(accel);
    impl_->accel_deps.push_back(std::move(deps));
    return AccelId{std::uint32_t(impl_->accels.size() - 1)};
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

    // --- GPU timestamps, if the hardware will do them --------------------------
    //
    // Two separate capabilities, and both are required. The device must expose
    // a TIMESTAMP counter set at all, and it must be able to sample AT STAGE
    // BOUNDARIES -- some hardware can only sample between whole command
    // buffers, which is a number the frame time already tells us.
    if ([dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
        id<MTLCounterSet> timestamps = nil;
        for (id<MTLCounterSet> set in dev.counterSets)
            if ([set.name isEqualToString:MTLCommonCounterSetTimestamp])
                timestamps = set;
        if (timestamps) {
            MTLCounterSampleBufferDescriptor* cd =
                [[MTLCounterSampleBufferDescriptor alloc] init];
            cd.counterSet = timestamps;
            // ONE REGION PER FRAME SLOT. Three frames are in flight and each
            // writes its own; sharing one region is what made every per-pass
            // number a mixture of up to three frames.
            cd.sampleCount =
                NSUInteger(Device::Impl::kMaxTimedPasses * 2 * kFramesInFlight);
            // SHARED, so resolveCounterRange can read it without a blit. The
            // buffer is a few hundred bytes and is read once a frame.
            cd.storageMode = MTLStorageModeShared;
            NSError* err = nil;
            d->impl_->counters = [dev newCounterSampleBufferWithDescriptor:cd
                                                                     error:&err];
            // Failure here is NOT an error for the caller. Timing is a
            // diagnostic; a device that will not give us one still renders.
            d->impl_->timing_supported = (d->impl_->counters != nil);
        }
    }
    return d;
}

BufferId Device::CreateBuffer(const void* data, std::size_t bytes) {
    if (!data || bytes == 0) return {};
    id<MTLBuffer> b = [impl_->dev newBufferWithBytes:data
                                              length:bytes
                                             options:MTLResourceStorageModeShared];
    if (!b) return {};
    return BufferId{impl_->AllocBufferSlot(b, "static")};
}

BufferId Device::CreateStorageBuffer(std::size_t bytes) {
    if (bytes == 0) return {};
    // Shared, so MapBuffer works on it. See the header: on unified memory this
    // costs nothing, and a posed mesh is wanted on the CPU for collision as
    // much as on the GPU for ray tracing.
    id<MTLBuffer> b = [impl_->dev newBufferWithLength:bytes
                                              options:MTLResourceStorageModeShared];
    if (!b) return {};
    return BufferId{impl_->AllocBufferSlot(b, "storage")};
}

ComputePipelineId Device::CreateComputePipeline(const std::string& source,
                                                const std::string& fn,
                                                std::string& error) {
    @autoreleasepool {
        NSError* err = nil;
        id<MTLLibrary> lib =
            [impl_->dev newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                     options:nil
                                       error:&err];
        if (!lib) {
            error = std::string("compute shader compilation failed: ") +
                    (err ? err.localizedDescription.UTF8String : "unknown error");
            return {};
        }
        id<MTLFunction> kernel =
            [lib newFunctionWithName:[NSString stringWithUTF8String:fn.c_str()]];
        if (!kernel) {
            error = "compute kernel '" + fn + "' not found in the source";
            return {};
        }
        // A DESCRIPTOR, only so the pipeline can carry a label -- the state
        // object's own label is read-only, and an unnamed compute pipeline in a
        // capture is indistinguishable from every other one.
        MTLComputePipelineDescriptor* cpd =
            [[MTLComputePipelineDescriptor alloc] init];
        cpd.label = [NSString stringWithUTF8String:fn.c_str()];
        cpd.computeFunction = kernel;
        id<MTLComputePipelineState> pso =
            [impl_->dev newComputePipelineStateWithDescriptor:cpd
                                                      options:MTLPipelineOptionNone
                                                   reflection:nil
                                                        error:&err];
        if (!pso) {
            error = std::string("compute pipeline creation failed: ") +
                    (err ? err.localizedDescription.UTF8String : "unknown error");
            return {};
        }
        impl_->compute_pipelines.push_back(pso);
        return ComputePipelineId{
            std::uint32_t(impl_->compute_pipelines.size() - 1)};
    }
}

ComputeEncoder Device::BeginCompute(const char* timer, const char* label) { @autoreleasepool {
    const int slot = impl_->BeginTiming(timer);
    if (slot < 0) {
        impl_->compute_enc = [impl_->cb computeCommandEncoder];
    } else {
        // A compute pass has only one stage, so the pair is dispatch-start and
        // dispatch-end rather than vertex-start and fragment-end.
        MTLComputePassDescriptor* cp = [MTLComputePassDescriptor computePassDescriptor];
        cp.sampleBufferAttachments[0].sampleBuffer = impl_->counters;
        cp.sampleBufferAttachments[0].startOfEncoderSampleIndex = NSUInteger(slot * 2);
        cp.sampleBufferAttachments[0].endOfEncoderSampleIndex = NSUInteger(slot * 2 + 1);
        impl_->compute_enc = [impl_->cb computeCommandEncoderWithDescriptor:cp];
    }
    const char* name = label ? label : timer;
    if (name) impl_->compute_enc.label = [NSString stringWithUTF8String:name];
    impl_->compute_group_max = 64;
    return ComputeEncoder(this);
}}

void Device::EndCompute() {
    [impl_->compute_enc endEncoding];
    impl_->compute_enc = nil;
}

void ComputeEncoder::SetPipeline(ComputePipelineId p) {
    if (!device_ || !Valid(p) || p.v >= device_->impl_->compute_pipelines.size())
        return;
    id<MTLComputePipelineState> pso = device_->impl_->compute_pipelines[p.v];
    [device_->impl_->compute_enc setComputePipelineState:pso];
    // What the pipeline can actually run, not what the caller guessed. A kernel
    // whose threadgroup exceeds this is rejected at dispatch, and clamping here
    // turns that into a smaller group rather than a dropped dispatch.
    device_->impl_->compute_group_max =
        int(pso.maxTotalThreadsPerThreadgroup);
}

void ComputeEncoder::SetBuffer(BufferId b, std::size_t offset, int slot) {
    if (!device_ || !Valid(b) || b.v >= device_->impl_->buffers.size()) return;
    [device_->impl_->compute_enc setBuffer:device_->impl_->buffers[b.v]
                                    offset:offset
                                   atIndex:NSUInteger(slot)];
}

void ComputeEncoder::SetTexture(TextureId t, int slot) {
    if (!device_ || !Valid(t) || t.v >= device_->impl_->textures.size()) return;
    [device_->impl_->compute_enc setTexture:device_->impl_->textures[t.v]
                                    atIndex:NSUInteger(slot)];
}

void ComputeEncoder::SetSampler(SamplerId sampler, int slot) {
    if (!device_ || !Valid(sampler) || sampler.v >= device_->impl_->samplers.size())
        return;
    [device_->impl_->compute_enc setSamplerState:device_->impl_->samplers[sampler.v]
                                         atIndex:NSUInteger(slot)];
}

void ComputeEncoder::SetBytes(const void* data, std::size_t bytes, int slot) {
    if (!device_ || !data || bytes == 0) return;
    [device_->impl_->compute_enc setBytes:data
                                   length:bytes
                                  atIndex:NSUInteger(slot)];
}

void ComputeEncoder::Dispatch2D(int width, int height, int gx, int gy) {
    if (!device_ || width <= 0 || height <= 0) return;
    // The product must fit the pipeline's own maximum, which is why the clamp
    // is on gx*gy and not on each separately: 32x32 is a legal-looking pair
    // that asks for 1024 threads, and a pipeline that wants at most 512 would
    // have the dispatch rejected at runtime rather than at the call site.
    int ax = std::max(1, gx), ay = std::max(1, gy);
    while (ax * ay > device_->impl_->compute_group_max && ay > 1) ay /= 2;
    while (ax * ay > device_->impl_->compute_group_max && ax > 1) ax /= 2;
    [device_->impl_->compute_enc
        dispatchThreadgroups:MTLSizeMake(NSUInteger((width + ax - 1) / ax),
                                         NSUInteger((height + ay - 1) / ay), 1)
       threadsPerThreadgroup:MTLSizeMake(NSUInteger(ax), NSUInteger(ay), 1)];
}

void ComputeEncoder::Dispatch3D(int width, int height, int depth, int gx, int gy,
                                int gz) {
    if (!device_ || width <= 0 || height <= 0 || depth <= 0) return;
    int ax = std::max(1, gx), ay = std::max(1, gy), az = std::max(1, gz);
    while (ax * ay * az > device_->impl_->compute_group_max && ay > 1) ay /= 2;
    while (ax * ay * az > device_->impl_->compute_group_max && ax > 1) ax /= 2;
    while (ax * ay * az > device_->impl_->compute_group_max && az > 1) az /= 2;
    [device_->impl_->compute_enc
        dispatchThreadgroups:MTLSizeMake(NSUInteger((width + ax - 1) / ax),
                                         NSUInteger((height + ay - 1) / ay),
                                         NSUInteger((depth + az - 1) / az))
       threadsPerThreadgroup:MTLSizeMake(NSUInteger(ax), NSUInteger(ay),
                                         NSUInteger(az))];
}

void ComputeEncoder::Dispatch(int count, int group) {
    if (!device_ || count <= 0) return;
    const int g = std::clamp(group, 1, device_->impl_->compute_group_max);
    // Rounded UP, so the last partial group still runs. The threads past the
    // end are the shader's problem, and its guard is the only thing that stops
    // them -- which is why the header says so rather than leaving it implied.
    const NSUInteger groups = NSUInteger((count + g - 1) / g);
    [device_->impl_->compute_enc
        dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(NSUInteger(g), 1, 1)];
}

BufferId Device::CreateDynamicBuffer(std::size_t bytes) {
    if (bytes == 0) return {};
    // Shared storage keeps one copy visible to both CPU and GPU on unified
    // memory; the pointer stays valid for the buffer's whole lifetime.
    id<MTLBuffer> b = [impl_->dev newBufferWithLength:bytes
                                              options:MTLResourceStorageModeShared];
    if (!b) return {};
    return BufferId{impl_->AllocBufferSlot(b, "dynamic")};
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
                                    int samples, bool cpu_readable) {
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
    // Depth here is normally transient: nothing samples it and every pass uses
    // storeAction DontCare, so on a TBDR Apple GPU it can live entirely in tile
    // memory and never touch DRAM. Falls back to Private on Intel, where
    // memoryless does not exist -- and to Shared when the CPU has to read it,
    // which is a test asking what a depth resolve produced.
    bool tbdr = false;
    if (!sampleable && !cpu_readable) {
        if (@available(macOS 10.15, *)) {
            tbdr = [impl_->dev supportsFamily:MTLGPUFamilyApple1];
        }
    }
    td.storageMode = cpu_readable ? MTLStorageModeShared
                     : tbdr       ? MTLStorageModeMemoryless
                                  : MTLStorageModePrivate;
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

TextureId Device::CreateTexture2DFloat(int width, int height,
                                       const float* rgba32f) {
    if (width <= 0 || height <= 0 || !rgba32f) return {};
    MTLTextureDescriptor* td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                     width:NSUInteger(width)
                                    height:NSUInteger(height)
                                 mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> t = [impl_->dev newTextureWithDescriptor:td];
    if (!t) return {};
    [t replaceRegion:MTLRegionMake2D(0, 0, NSUInteger(width), NSUInteger(height))
         mipmapLevel:0
           withBytes:rgba32f
         bytesPerRow:NSUInteger(width) * 16];
    return TextureId{impl_->AllocTextureSlot(t)};
}

TextureId Device::CreateCubemap(int size, Format format, int mip_levels) {
    if (size <= 0) return {};
    MTLTextureDescriptor* td =
        [MTLTextureDescriptor textureCubeDescriptorWithPixelFormat:ToMTL(format)
                                                              size:NSUInteger(size)
                                                         mipmapped:mip_levels != 1];
    if (mip_levels > 0) td.mipmapLevelCount = NSUInteger(mip_levels);
    // READ and WRITE and RENDER, plus PIXEL FORMAT VIEW. The last is the one
    // that is easy to miss and fails late: without it newTextureViewWithFormat
    // returns nil, and the mip views this class hands out for the prefilter
    // chain are exactly that call.
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
               MTLTextureUsageRenderTarget | MTLTextureUsagePixelFormatView;
    td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> t = [impl_->dev newTextureWithDescriptor:td];
    if (!t) return {};
    return TextureId{impl_->AllocTextureSlot(t)};
}

TextureId Device::CreateStorageTexture3D(int width, int height, int depth,
                                         Format format) {
    if (width <= 0 || height <= 0 || depth <= 0) return {};
    MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
    td.textureType = MTLTextureType3D;
    td.pixelFormat = ToMTL(format);
    td.width = NSUInteger(width);
    td.height = NSUInteger(height);
    td.depth = NSUInteger(depth);
    td.mipmapLevelCount = 1;
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> t = [impl_->dev newTextureWithDescriptor:td];
    if (!t) return {};
    return TextureId{impl_->AllocTextureSlot(t)};
}

TextureId Device::CreateStorageTexture2D(int width, int height, Format format,
                                         int mip_levels) {
    if (width <= 0 || height <= 0) return {};
    MTLTextureDescriptor* td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:ToMTL(format)
                                                           width:NSUInteger(width)
                                                          height:NSUInteger(height)
                                                       mipmapped:mip_levels != 1];
    if (mip_levels > 0) td.mipmapLevelCount = NSUInteger(mip_levels);
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite |
               MTLTextureUsagePixelFormatView;
    td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> t = [impl_->dev newTextureWithDescriptor:td];
    if (!t) return {};
    return TextureId{impl_->AllocTextureSlot(t)};
}

TextureId Device::CreateMipView(TextureId src, int mip) {
    if (!Valid(src) || src.v >= impl_->textures.size()) return {};
    id<MTLTexture> t = impl_->textures[src.v];
    if (!t || mip < 0 || NSUInteger(mip) >= t.mipmapLevelCount) return {};
    // A CUBE becomes a 2D ARRAY of six slices. A compute kernel cannot write to
    // a texturecube -- the type has no write access qualifier in MSL -- and it
    // does not need to: the six faces are six array slices in memory, and the
    // cube-ness is a sampling-time interpretation. So the view keeps the
    // storage and changes only how the shader is allowed to address it.
    const MTLTextureType type =
        (t.textureType == MTLTextureTypeCube) ? MTLTextureType2DArray
                                              : MTLTextureType2D;
    const NSUInteger slices = (type == MTLTextureType2DArray) ? 6 : 1;
    id<MTLTexture> view =
        [t newTextureViewWithPixelFormat:t.pixelFormat
                             textureType:type
                                  levels:NSMakeRange(NSUInteger(mip), 1)
                                  slices:NSMakeRange(0, slices)];
    if (!view) return {};
    return TextureId{impl_->AllocTextureSlot(view)};
}

int Device::TextureWidth(TextureId h) const {
    if (!Valid(h) || h.v >= impl_->textures.size() || !impl_->textures[h.v]) return 0;
    return int(impl_->textures[h.v].width);
}
int Device::TextureHeight(TextureId h) const {
    if (!Valid(h) || h.v >= impl_->textures.size() || !impl_->textures[h.v]) return 0;
    return int(impl_->textures[h.v].height);
}
int Device::TextureMipLevels(TextureId h) const {
    if (!Valid(h) || h.v >= impl_->textures.size() || !impl_->textures[h.v]) return 0;
    return int(impl_->textures[h.v].mipmapLevelCount);
}

TextureId Device::CreateTexture2D(int width, int height, const void* rgba8,
                                  bool mips, bool srgb) {
    if (width <= 0 || height <= 0 || !rgba8) return {};
    // A 1x1 texture has exactly one level however you ask, and asking Metal to
    // generate mipmaps for it is an error rather than a no-op. The engine's
    // white and black placeholders are 1x1, so this is the common case.
    const bool want_mips = mips && (width > 1 || height > 1);
    MTLTextureDescriptor* td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:(srgb ? MTLPixelFormatRGBA8Unorm_sRGB
                                                 : MTLPixelFormatRGBA8Unorm)
                                     width:NSUInteger(width)
                                    height:NSUInteger(height)
                                 mipmapped:(want_mips ? YES : NO)];
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

    if (want_mips) {
        // Its OWN command buffer, committed and waited on here. Texture upload
        // is a load-time operation and there may be no frame in flight to
        // append to -- and appending to one would make the texture unusable
        // until that frame completed, which the caller has no way to know.
        id<MTLCommandBuffer> cb = [impl_->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit generateMipmapsForTexture:t];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
    return TextureId{impl_->AllocTextureSlot(t)};
}

TextureId Device::CreateRenderTargetArray(int width, int height, int layers,
                                          Format format, bool cpu_readable) {
    if (width <= 0 || height <= 0 || layers <= 0) return {};
    MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
    td.textureType = MTLTextureType2DArray;
    td.pixelFormat = ToMTL(format);
    td.width = NSUInteger(width);
    td.height = NSUInteger(height);
    td.arrayLength = NSUInteger(layers);
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead |
               MTLTextureUsagePixelFormatView;
    td.storageMode = cpu_readable ? MTLStorageModeShared : MTLStorageModePrivate;
    id<MTLTexture> t = [impl_->dev newTextureWithDescriptor:td];
    if (!t) return {};
    return TextureId{impl_->AllocTextureSlot(t)};
}

TextureId Device::CreateDepthTargetArray(int width, int height, int layers) {
    if (width <= 0 || height <= 0 || layers <= 0) return {};
    MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
    td.textureType = MTLTextureType2DArray;
    td.pixelFormat = MTLPixelFormatDepth32Float;
    td.width = NSUInteger(width);
    td.height = NSUInteger(height);
    td.arrayLength = NSUInteger(layers);
    td.usage = MTLTextureUsageRenderTarget;
    // NOT memoryless. A layered depth buffer is written by the amplified pass
    // and may be sampled afterwards; memoryless would discard it at end of
    // pass and the second eye's SSAO would read nothing.
    td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> t = [impl_->dev newTextureWithDescriptor:td];
    if (!t) return {};
    return TextureId{impl_->AllocTextureSlot(t)};
}

TextureId Device::CreateArraySlice(TextureId array, int slice) {
    if (!Valid(array) || array.v >= impl_->textures.size() || slice < 0) return {};
    id<MTLTexture> src = impl_->textures[array.v];
    if (!src || slice >= int(src.arrayLength)) return {};
    // A VIEW, not a copy: it shares the array's storage, so reading it reads
    // whatever the pass wrote. Creating one per readback would leak a handle
    // per frame -- make them once alongside the array.
    id<MTLTexture> view = [src newTextureViewWithPixelFormat:src.pixelFormat
                                                 textureType:MTLTextureType2D
                                                      levels:NSMakeRange(0, 1)
                                                      slices:NSMakeRange(NSUInteger(slice), 1)];
    if (!view) return {};
    return TextureId{impl_->AllocTextureSlot(view)};
}

TextureId Device::CreateTexture3DFloat(int width, int height, int depth,
                                       const float* rgba32f) {
    if (width <= 0 || height <= 0 || depth <= 0 || !rgba32f) return {};
    MTLTextureDescriptor* td = [[MTLTextureDescriptor alloc] init];
    td.textureType = MTLTextureType3D;
    td.pixelFormat = MTLPixelFormatRGBA16Float;
    td.width = NSUInteger(width);
    td.height = NSUInteger(height);
    td.depth = NSUInteger(depth);
    td.mipmapLevelCount = 1;
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> t = [impl_->dev newTextureWithDescriptor:td];
    if (!t) return {};

    // Converted to half on the CPU. replaceRegion does no format conversion --
    // handing it float data for a half texture writes the raw bits and the
    // result is not "slightly wrong colours", it is noise.
    std::vector<std::uint16_t> half(std::size_t(width) * height * depth * 4);
    for (std::size_t i = 0; i < half.size(); ++i) {
        const float f = rgba32f[i];
        // __fp16 is the compiler's own conversion, including the round-to-
        // nearest-even and the overflow-to-infinity that a hand-rolled bit
        // twiddle gets wrong at the edges.
        const __fp16 h = __fp16(f);
        std::uint16_t bits;
        std::memcpy(&bits, &h, sizeof(bits));
        half[i] = bits;
    }
    [t replaceRegion:MTLRegionMake3D(0, 0, 0, NSUInteger(width),
                                     NSUInteger(height), NSUInteger(depth))
         mipmapLevel:0
               slice:0
           withBytes:half.data()
         bytesPerRow:NSUInteger(width) * 8
       bytesPerImage:NSUInteger(width) * NSUInteger(height) * 8];
    return TextureId{impl_->AllocTextureSlot(t)};
}

namespace {

int BlockBytesOf(Format f) {
    switch (f) {
        case Format::BC1:
        case Format::BC1Srgb: return 8;
        case Format::BC3:
        case Format::BC3Srgb:
        case Format::BC5: return 16;
        default: return 0;
    }
}

}  // namespace

TextureId Device::CreateTexture2DCompressed(
    int width, int height, Format fmt, std::span<const std::uint8_t> data,
    std::span<const std::size_t> level_offsets) {
    const int block = BlockBytesOf(fmt);
    if (width <= 0 || height <= 0 || block == 0 || level_offsets.empty())
        return {};

    MTLTextureDescriptor* td = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:ToMTL(fmt)
                                     width:NSUInteger(width)
                                    height:NSUInteger(height)
                                 mipmapped:(level_offsets.size() > 1 ? YES : NO)];
    td.mipmapLevelCount = NSUInteger(level_offsets.size());
    td.usage = MTLTextureUsageShaderRead;
    td.storageMode = MTLStorageModeShared;
    id<MTLTexture> t = [impl_->dev newTextureWithDescriptor:td];
    if (!t) return {};

    for (std::size_t level = 0; level < level_offsets.size(); ++level) {
        const int lw = std::max(1, width >> level);
        const int lh = std::max(1, height >> level);
        const int bx = (lw + 3) / 4, by = (lh + 3) / 4;
        const std::size_t bytes = std::size_t(bx) * by * block;
        const std::size_t off = level_offsets[level];
        // Every level bounds-checked against the blob. A short buffer here is
        // not a validation error in Metal -- replaceRegion reads whatever
        // follows the allocation, and the texture comes out with garbage in its
        // small mips, which only shows at a distance.
        if (off + bytes > data.size()) return {};
        [t replaceRegion:MTLRegionMake2D(0, 0, NSUInteger(lw), NSUInteger(lh))
             mipmapLevel:level
               withBytes:data.data() + off
             // bytesPerRow is per BLOCK ROW for a compressed format, not per
             // texel row. Passing the texel figure is a four-times overread and
             // a texture that decodes as diagonal streaks.
             bytesPerRow:NSUInteger(bx) * NSUInteger(block)];
    }
    return TextureId{impl_->AllocTextureSlot(t)};
}

SamplerId Device::CreateSampler(Filter filter, Wrap wrap, int max_anisotropy) {
    MTLSamplerDescriptor* sd = [[MTLSamplerDescriptor alloc] init];
    const MTLSamplerMinMagFilter f = (filter == Filter::Linear)
                                         ? MTLSamplerMinMagFilterLinear
                                         : MTLSamplerMinMagFilterNearest;
    sd.minFilter = f;
    sd.magFilter = f;
    // TRILINEAR whenever the filter is linear. A texture with no mip chain has
    // one level and this costs nothing; one with a chain would otherwise snap
    // between levels at a visible line across the floor.
    if (filter == Filter::Linear) sd.mipFilter = MTLSamplerMipFilterLinear;
    // Metal requires 1..16. Clamping rather than rejecting: an out-of-range
    // value here is a caller asking for "as much as possible", and refusing to
    // build the sampler would take down the whole renderer for a quality hint.
    sd.maxAnisotropy = NSUInteger(max_anisotropy < 1    ? 1
                                  : max_anisotropy > 16 ? 16
                                                        : max_anisotropy);
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

SamplerId Device::CreateMipSampler(Wrap wrap) {
    MTLSamplerDescriptor* sd = [[MTLSamplerDescriptor alloc] init];
    sd.minFilter = MTLSamplerMinMagFilterLinear;
    sd.magFilter = MTLSamplerMinMagFilterLinear;
    // LINEAR between levels, which is the whole reason this exists. Nearest
    // would snap to the closest mip and put a visible ring on every surface
    // whose roughness crosses a level boundary.
    sd.mipFilter = MTLSamplerMipFilterLinear;
    const MTLSamplerAddressMode a = (wrap == Wrap::Repeat)
                                        ? MTLSamplerAddressModeRepeat
                                        : MTLSamplerAddressModeClampToEdge;
    sd.sAddressMode = a;
    sd.tAddressMode = a;
    sd.rAddressMode = a;
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
    // Named by its entry points, which is how a capture identifies a draw.
    pd.label = [NSString stringWithFormat:@"%s/%s", desc.vertex_fn.c_str(),
                                          desc.depth_only ? "depth-only"
                                                          : desc.fragment_fn.c_str()];
    pd.vertexFunction = vs;
    pd.fragmentFunction = fs;
    // A depth-only pipeline has NO colour attachment. Declaring one that the
    // pass will not provide is a pipeline/pass mismatch.
    pd.colorAttachments[0].pixelFormat =
        desc.depth_only ? MTLPixelFormatInvalid : ToMTL(desc.color);
    if (!desc.depth_only)
        for (std::size_t i = 0; i < desc.extra_colors.size(); ++i)
            pd.colorAttachments[i + 1].pixelFormat = ToMTL(desc.extra_colors[i]);
    // Without this the depth attachment is silently ignored at draw time.
    if (desc.depth) pd.depthAttachmentPixelFormat = ToMTL(Format::Depth32Float);
    // Has to match the attachments exactly. Metal rejects a mismatch here,
    // which is the one class of format error it does NOT let through silently.
    pd.rasterSampleCount = NSUInteger(desc.samples > 0 ? desc.samples : 1);
    // ALWAYS, not only for amplified pipelines. Metal requires it of any
    // pipeline whose vertex stage writes render_target_array_index -- and once
    // that member is in the shared vertex output struct, every pipeline built
    // from that shader writes it, amplified or not. The error is explicit
    // ("Vertex shader writes render_target_array_index but
    // inputPrimitiveTopology is not specified") and it takes down every
    // material in the engine at once.
    //
    // Triangle is right for everything here; there is no line or point
    // rendering in this renderer.
    pd.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    if (desc.amplification > 1)
        pd.maxVertexAmplificationCount = NSUInteger(desc.amplification);
    if (desc.blend == Blend::OitAccumulate && !desc.depth_only) {
        // TWO attachments with DIFFERENT blend functions, which is the whole
        // trick. Accumulation sums weighted premultiplied colour; revealage
        // multiplies down what visibility is left. Both operations commute, so
        // the result is the same whatever order the surfaces arrive in.
        MTLRenderPipelineColorAttachmentDescriptor* acc = pd.colorAttachments[0];
        acc.blendingEnabled = YES;
        acc.rgbBlendOperation = MTLBlendOperationAdd;
        acc.alphaBlendOperation = MTLBlendOperationAdd;
        acc.sourceRGBBlendFactor = MTLBlendFactorOne;
        acc.sourceAlphaBlendFactor = MTLBlendFactorOne;
        acc.destinationRGBBlendFactor = MTLBlendFactorOne;
        acc.destinationAlphaBlendFactor = MTLBlendFactorOne;

        MTLRenderPipelineColorAttachmentDescriptor* rev = pd.colorAttachments[1];
        rev.blendingEnabled = YES;
        rev.rgbBlendOperation = MTLBlendOperationAdd;
        rev.alphaBlendOperation = MTLBlendOperationAdd;
        // dst * (1 - src). MTLBlendFactorZero on the source and
        // OneMinusSourceColor on the destination is a MULTIPLY, expressed in
        // the only vocabulary a fixed-function blender has.
        rev.sourceRGBBlendFactor = MTLBlendFactorZero;
        rev.sourceAlphaBlendFactor = MTLBlendFactorZero;
        rev.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceColor;
        rev.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    } else if (desc.blend != Blend::None && !desc.depth_only) {
        MTLRenderPipelineColorAttachmentDescriptor* ca = pd.colorAttachments[0];
        ca.blendingEnabled = YES;
        ca.rgbBlendOperation = MTLBlendOperationAdd;
        ca.alphaBlendOperation = MTLBlendOperationAdd;
        ca.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
        ca.sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
        if (desc.blend == Blend::Additive) {
            // ONE, so the destination is never scaled down. That is what makes
            // it order-independent: addition commutes, so ten thousand sparks
            // need no sort. It also means it can only brighten -- an additive
            // surface cannot darken what is behind it, which is exactly right
            // for fire and exactly wrong for smoke.
            ca.destinationRGBBlendFactor = MTLBlendFactorOne;
            ca.destinationAlphaBlendFactor = MTLBlendFactorOne;
        } else {
            ca.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
            ca.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        }
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

bool Device::SupportsMeshShaders() const {
    return [impl_->dev supportsFamily:MTLGPUFamilyApple7] ||
           [impl_->dev supportsFamily:MTLGPUFamilyMetal3];
}

PipelineId Device::CreateMeshPipeline(const MeshPipelineDesc& desc,
                                      std::string& error) { @autoreleasepool {
    if (!SupportsMeshShaders()) {
        error = "this GPU has no mesh shader stage";
        return {};
    }
    NSError* err = nil;
    id<MTLLibrary> lib = [impl_->dev
        newLibraryWithSource:[NSString stringWithUTF8String:desc.source.c_str()]
                     options:nil
                       error:&err];
    if (!lib) {
        error = std::string("shader compile failed: ") +
                err.localizedDescription.UTF8String;
        return {};
    }
    MTLMeshRenderPipelineDescriptor* pd =
        [[MTLMeshRenderPipelineDescriptor alloc] init];
    pd.label = [NSString stringWithFormat:@"%s/%s/%s", desc.object_fn.c_str(),
                                          desc.mesh_fn.c_str(),
                                          desc.fragment_fn.c_str()];
    pd.objectFunction =
        [lib newFunctionWithName:[NSString stringWithUTF8String:desc.object_fn.c_str()]];
    pd.meshFunction =
        [lib newFunctionWithName:[NSString stringWithUTF8String:desc.mesh_fn.c_str()]];
    pd.fragmentFunction =
        [lib newFunctionWithName:[NSString stringWithUTF8String:desc.fragment_fn.c_str()]];
    if (!pd.meshFunction || !pd.fragmentFunction) {
        error = "mesh or fragment function not found in the shader source";
        return {};
    }
    pd.colorAttachments[0].pixelFormat = ToMTL(desc.color);
    for (std::size_t i = 0; i < desc.extra_colors.size(); ++i)
        pd.colorAttachments[i + 1].pixelFormat = ToMTL(desc.extra_colors[i]);
    if (desc.depth) pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    pd.rasterSampleCount = NSUInteger(desc.samples > 0 ? desc.samples : 1);
    pd.payloadMemoryLength = NSUInteger(desc.payload_bytes);
    pd.maxTotalThreadsPerObjectThreadgroup = NSUInteger(desc.object_threads);
    pd.maxTotalThreadsPerMeshThreadgroup = NSUInteger(desc.mesh_threads);

    PipelineObj obj;
    obj.pso = [impl_->dev newRenderPipelineStateWithMeshDescriptor:pd
                                                           options:MTLPipelineOptionNone
                                                        reflection:nil
                                                             error:&err];
    if (!obj.pso) {
        error = std::string("mesh pipeline creation failed: ") +
                err.localizedDescription.UTF8String;
        return {};
    }
    if (desc.depth) {
        MTLDepthStencilDescriptor* dd = [[MTLDepthStencilDescriptor alloc] init];
        dd.depthCompareFunction = ToMTL(desc.depth_compare);
        dd.depthWriteEnabled = desc.depth_write;
        obj.dss = [impl_->dev newDepthStencilStateWithDescriptor:dd];
    }
    impl_->pipelines.push_back(obj);
    return PipelineId{std::uint32_t(impl_->pipelines.size() - 1)};
}}

std::unique_ptr<Swapchain> Device::CreateSwapchain(Format format,
                                                   std::string& error, bool hdr) {
    if (hdr && format != Format::RGBA16Float) {
        error = "an HDR swapchain needs RGBA16Float; an 8-bit layer has nowhere "
                "to put a value above one whatever colour space it is in";
        return nullptr;
    }
    CAMetalLayer* layer = [CAMetalLayer layer];
    if (!layer) {
        error = "CAMetalLayer creation failed";
        return nullptr;
    }
    layer.device = impl_->dev;
    layer.pixelFormat = ToMTL(format);
    layer.framebufferOnly = YES;
    if (hdr) {
        layer.wantsExtendedDynamicRangeContent = YES;
        // EXTENDED LINEAR sRGB: the same primaries as everything else in this
        // engine, with values allowed past 1. Display P3 would be a wider
        // gamut and would need every colour in the pipeline reinterpreted --
        // the same numbers would mean more saturated colours, so a scene tuned
        // on an SDR display would shift the moment HDR was switched on.
        CGColorSpaceRef cs =
            CGColorSpaceCreateWithName(kCGColorSpaceExtendedLinearSRGB);
        layer.colorspace = cs;
        CGColorSpaceRelease(cs);
    }
    std::unique_ptr<Swapchain> sc(new Swapchain());
    sc->impl_->layer = layer;
    return sc;
}

float Device::DisplayHeadroom(const Swapchain& sc) const {
    if (!sc.impl_->layer.wantsExtendedDynamicRangeContent) return 1.0f;
    NSScreen* screen = sc.impl_->layer.contentsScale > 0 ? [NSScreen mainScreen] : nil;
    if (!screen) return 1.0f;
    const float ratio = float(screen.maximumExtendedDynamicRangeColorComponentValue);
    // Below 1 is meaningless and 1 means no headroom; both come back as "SDR".
    return ratio > 1.0f ? ratio : 1.0f;
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

    // A DESCRIPTOR, for errorOptions. Without EncoderExecutionStatus a GPU
    // fault reports only that the buffer failed; with it, done.error carries an
    // MTLCommandBufferEncoderInfo per encoder saying which one was affected and
    // whether it completed, started, or never ran.
    MTLCommandBufferDescriptor* cbd = [[MTLCommandBufferDescriptor alloc] init];
    cbd.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
    impl_->cb = [impl_->queue commandBufferWithDescriptor:cbd];
    ++impl_->frame_index;
    impl_->cb.label = [NSString stringWithFormat:@"frame %llu",
                                                 (unsigned long long)impl_->frame_index];
    impl_->timed_this_frame = 0;
    impl_->timing_labels.clear();

}}

// Attached at COMMIT, not at BeginFrame.
//
// It used to be attached at BeginFrame, three lines after timing_labels was
// cleared, so the block captured an EMPTY label list and `timed` was always
// zero. The resolve below never ran and LastFrameTimings always returned
// nothing -- the whole GPU timing feature reported silence. Nothing in the
// engine consumed it, so nothing noticed until a test asked for a number.
//
// The labels do genuinely have to be copied: the vector is cleared at the top
// of the next BeginFrame and this block runs on a driver thread at an
// unpredictable time, so capturing the member would be a data race whose
// symptom is a garbage label, or a crash if it reallocated.
void Device::InstallFrameCompletion() { @autoreleasepool {
    dispatch_semaphore_t sem = impl_->frame_sem;
    std::atomic<int>* counter = &impl_->in_flight;
    Impl* impl = impl_.get();
    // The labels are COPIED into the block. The vector is cleared at the top of
    // the next BeginFrame, and the completion handler runs on a driver thread
    // at an unpredictable time -- capturing the member would be a data race
    // whose symptom is a garbage label, or a crash if it reallocated.
    std::vector<const char*> labels = impl_->timing_labels;
    const int timed = impl_->timed_this_frame;
    // WHERE this frame's pairs live, captured with them. Resolving from zero
    // would read whichever frame owns slot 0 rather than this one.
    const int base = impl_->TimingBase();
    const std::uint64_t frame = impl_->frame_index;
    id<MTLCounterSampleBuffer> counters = impl_->counters;
    [impl_->cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
        // GPUEndTime and GPUStartTime are seconds, and available whether or not
        // stage-boundary sampling is. This is the number to trust: the per-pass
        // times below do not sum to it, because a tiler overlaps the vertex
        // work of one pass with the fragment work of the last.
        impl->gpu_ms.store((done.GPUEndTime - done.GPUStartTime) * 1000.0,
                           std::memory_order_relaxed);
        if (done.error) {
            // Name the encoders, not just the frame. Metal reports one info
            // record per encoder; the ones that did not complete are the
            // interesting ones and are listed first in the message.
            std::string msg = std::string("frame ") + std::to_string(frame) +
                              ": " + done.error.localizedDescription.UTF8String;
            NSArray<id<MTLCommandBufferEncoderInfo>>* infos =
                done.error.userInfo[MTLCommandBufferEncoderInfoErrorKey];
            for (id<MTLCommandBufferEncoderInfo> info in infos) {
                if (info.errorState == MTLCommandEncoderErrorStateCompleted)
                    continue;
                const char* state =
                    info.errorState == MTLCommandEncoderErrorStateAffected  ? "affected"
                    : info.errorState == MTLCommandEncoderErrorStateFaulted ? "FAULTED"
                    : info.errorState == MTLCommandEncoderErrorStatePending ? "pending"
                                                                            : "unknown";
                msg += std::string("\n    encoder '") +
                       (info.label ? info.label.UTF8String : "(unnamed)") + "' " +
                       state;
            }
            std::lock_guard<std::mutex> guard(impl->fault_mutex);
            impl->last_fault = std::move(msg);
            ++impl->fault_count;
        }
        if (timed > 0 && counters) {
            NSData* data = [counters
                resolveCounterRange:NSMakeRange(NSUInteger(base * 2),
                                                NSUInteger(timed * 2))];
            if (data && data.length >= sizeof(MTLCounterResultTimestamp) * std::size_t(timed * 2)) {
                const auto* t = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
                std::vector<GpuTiming> out;
                out.reserve(std::size_t(timed));
                // THE FRAME'S OWN ORIGIN, taken from the samples rather than
                // from GPUStartTime. The two are not the same clock -- one is
                // seconds on the CPU timebase and the other is GPU ticks -- and
                // correlating them buys nothing here, because begin and end are
                // only ever compared with each other.
                MTLTimestamp origin = ~MTLTimestamp{0};
                for (int i = 0; i < timed * 2; ++i)
                    if (t[i].timestamp != MTLCounterErrorValue &&
                        t[i].timestamp < origin)
                        origin = t[i].timestamp;
                if (origin == ~MTLTimestamp{0}) origin = 0;
                for (int i = 0; i < timed; ++i) {
                    const MTLTimestamp a = t[i * 2].timestamp;
                    const MTLTimestamp b = t[i * 2 + 1].timestamp;
                    // MTLCounterErrorValue marks a sample the GPU did not take
                    // -- a pass that was culled, or one whose encoder produced
                    // no work. Reporting the subtraction of two error values as
                    // a duration would put an enormous number in the profile.
                    const bool bad = a == MTLCounterErrorValue ||
                                     b == MTLCounterErrorValue || b < a;
                    GpuTiming g;
                    g.label = labels[std::size_t(i)];
                    // Ticks are NANOSECONDS on Apple Silicon: the GPU and CPU
                    // share a timebase, which is exactly why no correlation
                    // call is needed here.
                    g.milliseconds = bad ? 0.0 : double(b - a) * 1e-6;
                    g.begin_ms = bad ? 0.0 : double(a - origin) * 1e-6;
                    g.end_ms = bad ? 0.0 : double(b - origin) * 1e-6;
                    out.push_back(g);
                }
                // IN ORDER, and under the lock. Completion handlers run on
                // driver threads and there can be kFramesInFlight of them at
                // once, so "last to finish" is not "newest" -- without the
                // frame check a handler that came back late would replace a
                // newer frame's numbers with its own.
                std::lock_guard<std::mutex> guard(impl->timings_mutex);
                if (frame >= impl->timings_frame) {
                    impl->timings = std::move(out);
                    impl->timings_frame = frame;
                }
            }
        }
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

bool Device::SupportsGpuTiming() const { return impl_->timing_supported; }

std::vector<GpuTiming> Device::LastFrameTimings() const {
    std::lock_guard<std::mutex> guard(impl_->timings_mutex);
    return impl_->timings;
}

std::string Device::TakeGpuFault() {
    std::lock_guard<std::mutex> guard(impl_->fault_mutex);
    std::string out;
    out.swap(impl_->last_fault);
    return out;
}

int Device::GpuFaultCount() const {
    std::lock_guard<std::mutex> guard(impl_->fault_mutex);
    return impl_->fault_count;
}

double Device::LastFrameGpuMilliseconds() const {
    return impl_->gpu_ms.load(std::memory_order_relaxed);
}

Encoder Device::BeginPass(const PassDesc& desc) { @autoreleasepool {
    // Descriptors are autoreleased. Without a pool here the engine leaks one
    // per pass per frame for the whole run — there is no Cocoa run loop to
    // drain them, because the app drives its own loop.
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    // A LAYERED pass renders every slice at once. renderTargetArrayLength is
    // what tells Metal that; without it the pass writes slice 0 only and the
    // second eye is simply never drawn -- no error, just a black eye.
    if (desc.views > 1) rp.renderTargetArrayLength = NSUInteger(desc.views);
    if (Valid(desc.color)) {
        rp.colorAttachments[0].texture = impl_->textures[desc.color.v];
        rp.colorAttachments[0].loadAction =
            desc.load ? MTLLoadActionLoad : MTLLoadActionClear;
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
    for (std::size_t i = 0; i < desc.extra_colors.size(); ++i) {
        if (!Valid(desc.extra_colors[i])) continue;
        const NSUInteger at = i + 1;
        rp.colorAttachments[at].texture = impl_->textures[desc.extra_colors[i].v];
        rp.colorAttachments[at].loadAction =
            desc.load ? MTLLoadActionLoad : MTLLoadActionClear;
        rp.colorAttachments[at].storeAction = MTLStoreActionStore;
        rp.colorAttachments[at].clearColor =
            MTLClearColorMake(desc.clear_color[0], desc.clear_color[1],
                              desc.clear_color[2], desc.clear_color[3]);
    }
    if (Valid(desc.depth)) {
        rp.depthAttachment.texture = impl_->textures[desc.depth.v];
        rp.depthAttachment.loadAction =
            desc.load_depth ? MTLLoadActionLoad : MTLLoadActionClear;
        rp.depthAttachment.storeAction =
            desc.keep_depth ? MTLStoreActionStore : MTLStoreActionDontCare;
        rp.depthAttachment.clearDepth = desc.clear_depth;
        // RESOLVED like a colour attachment, and for the same reason: the
        // samples only exist inside tile memory, so this is the only way to get
        // the frame's depth out as something a later pass can sample.
        //
        // MAX, which under this engine's reversed-Z is the NEAREST sample. Not
        // a preference -- Metal offers sample-zero, min and max, and of the
        // three only max returns a depth that some real surface actually had at
        // that pixel, on the near side of a silhouette where it matters.
        if (Valid(desc.depth_resolve)) {
            rp.depthAttachment.resolveTexture = impl_->textures[desc.depth_resolve.v];
            rp.depthAttachment.depthResolveFilter =
                MTLMultisampleDepthResolveFilterMax;
            rp.depthAttachment.storeAction =
                desc.keep_depth ? MTLStoreActionStoreAndMultisampleResolve
                                : MTLStoreActionMultisampleResolve;
        }
    }
    const int slot = impl_->BeginTiming(desc.timer);
    if (slot >= 0) {
        // START OF VERTEX to END OF FRAGMENT, which brackets the whole pass.
        // Sampling at the encoder boundaries instead would include the driver's
        // setup, and on a tiler it would include the wait for the previous
        // pass's tiles to flush -- time that belongs to the other pass.
        rp.sampleBufferAttachments[0].sampleBuffer = impl_->counters;
        rp.sampleBufferAttachments[0].startOfVertexSampleIndex = NSUInteger(slot * 2);
        rp.sampleBufferAttachments[0].endOfFragmentSampleIndex = NSUInteger(slot * 2 + 1);
    }
    // The encoder is held by a strong ivar, so it outlives this pool.
    impl_->pass_has_depth = Valid(desc.depth);
    impl_->enc = [impl_->cb renderCommandEncoderWithDescriptor:rp];
    // NAMED, which PassDesc::timer's comment has always claimed and the code
    // never did. A capture of this engine was fourteen numbered encoders.
    const char* name = desc.label ? desc.label : desc.timer;
    if (name) impl_->enc.label = [NSString stringWithUTF8String:name];
    impl_->pass_views = desc.views > 8 ? 8 : (desc.views < 1 ? 1 : desc.views);
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
    InstallFrameCompletion();
    [impl_->cb commit];
    impl_->submitted = impl_->cb;
    impl_->cb = nil;
}

bool Device::CommitAndWait(std::string& error) {
    InstallFrameCompletion();
    [impl_->cb commit];
    impl_->submitted = impl_->cb;
    [impl_->cb waitUntilCompleted];  // MUST sync before reading pixels back
    const bool ok = (impl_->cb.error == nil);
    if (!ok) {
        error = std::string("GPU execution failed: ") +
                impl_->cb.error.localizedDescription.UTF8String;
    }
    impl_->cb = nil;
    return ok;
}

void Device::WaitForGpu() {
    if (!impl_->submitted) return;  // nothing in flight, or already waited
    [impl_->submitted waitUntilCompleted];
    impl_->submitted = nil;
}

bool Device::ReadPixels(TextureId tex, int width, int height,
                        std::span<std::uint8_t> out) {
    if (!Valid(tex) || width <= 0 || height <= 0) return false;

    // WAIT FOR THE GPU FIRST. getBytes does not synchronise; it copies whatever
    // is in the texture at the instant it is called, and on a tile-based GPU
    // that is a partially finished image -- the tiles that happen to have
    // retired. It showed up as a band of garbage across the middle of a capture
    // from the interactive viewer, which commits with Commit() and does not
    // wait, while apps/shot uses CommitAndWait and was always correct. The
    // invariant was already written down one function up; it just was not
    // enforced where it mattered.
    //
    WaitForGpu();

    // A Private texture has no CPU-visible contents; getBytes on one is a
    // Metal validation error rather than a silent zero fill.
    if (impl_->textures[tex.v].storageMode != MTLStorageModeShared) return false;
    // The stride comes from the FORMAT. It used to be a hard-coded four bytes,
    // which meant a half-float target could not be read back at all -- and a
    // half-float target is exactly what an HDR composite writes, so there was
    // no way to measure the one output where the values above 1 are the point.
    //
    // The caller still gets raw bytes and still has to know how to read them;
    // what this fixes is the size, not the interpretation.
    const MTLPixelFormat pf = impl_->textures[tex.v].pixelFormat;
    std::size_t bpp = 0;
    switch (pf) {
        case MTLPixelFormatRGBA8Unorm:
        case MTLPixelFormatRGBA8Unorm_sRGB:
        case MTLPixelFormatBGRA8Unorm:
            bpp = 4;
            break;
        case MTLPixelFormatRGBA16Float:
            bpp = 8;
            break;
        case MTLPixelFormatRGBA32Float:
            bpp = 16;
            break;
        // DEPTH, four bytes of float, so a test can assert what a depth resolve
        // actually produced. Sampleable depth targets are Shared for exactly
        // this reason and the check above already refuses the private ones.
        case MTLPixelFormatDepth32Float:
            bpp = 4;
            break;
        default:
            // Refused rather than guessed. A wrong stride does not fail, it
            // returns a plausible image at the wrong width.
            return false;
    }
    const std::size_t need = std::size_t(width) * std::size_t(height) * bpp;
    if (out.size() < need) return false;
    [impl_->textures[tex.v]
             getBytes:out.data()
          bytesPerRow:std::size_t(width) * bpp
           fromRegion:MTLRegionMake2D(0, 0, NSUInteger(width), NSUInteger(height))
          mipmapLevel:0];
    return true;
}

// ------------------------------------------------------------------ Encoder --

void Encoder::SetPipeline(PipelineId p) {
    const PipelineObj& obj = device_->impl_->pipelines[p.v];
    [device_->impl_->enc setRenderPipelineState:obj.pso];
    if (obj.dss) [device_->impl_->enc setDepthStencilState:obj.dss];
    if (device_->impl_->pass_views > 1) {
        // The identity mapping: amplification index i renders to array slice i,
        // with no viewport offset. Anything else would be a foveated or
        // multi-viewport arrangement this does not do.
        //
        // AFTER the pipeline, every time. Set once at the top of the pass it is
        // accepted and has no effect -- both eyes land in slice 0 and the
        // second one is black, with no error anywhere. That is not a
        // documented rule so much as what the driver does, and it is exactly
        // the failure the stereo test exists to catch.
        MTLVertexAmplificationViewMapping maps[8] = {};
        for (int i = 0; i < device_->impl_->pass_views; ++i) {
            maps[i].renderTargetArrayIndexOffset = uint32_t(i);
            maps[i].viewportArrayIndexOffset = 0;
        }
        [device_->impl_->enc
            setVertexAmplificationCount:NSUInteger(device_->impl_->pass_views)
                           viewMappings:maps];
    }
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

void Encoder::SetFragmentAccel(AccelId a, int slot) {
    if (!device_ || !Valid(a) || a.v >= device_->impl_->accels.size()) return;
    id<MTLRenderCommandEncoder> enc = device_->impl_->enc;
    // Residency FIRST, then the binding. Everything the structure reaches --
    // the bottom-level structures and the vertex and index buffers under them
    // -- is referenced by GPU address rather than by a binding, so no automatic
    // tracking sees it.
    //
    // Honest note: removing these calls does NOT fault on this driver, with
    // Metal API validation and GPU validation both on. The tests cannot
    // demonstrate the difference, so they do not claim to. The calls stay
    // because the contract requires them and the failure they prevent is a
    // fault under memory pressure or on another device -- the kind that
    // reproduces on someone else's machine and not on this one.
    for (id<MTLResource> r : device_->impl_->accel_deps[a.v])
        [enc useResource:r usage:MTLResourceUsageRead stages:MTLRenderStageFragment];
    [enc useResource:device_->impl_->accels[a.v]
               usage:MTLResourceUsageRead
              stages:MTLRenderStageFragment];
    [enc setFragmentAccelerationStructure:device_->impl_->accels[a.v]
                            atBufferIndex:NSUInteger(slot)];
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

void Encoder::DrawInstanced(std::size_t vertex_count, std::size_t instance_count) {
    if (!device_ || vertex_count == 0 || instance_count == 0) return;
    [device_->impl_->enc drawPrimitives:MTLPrimitiveTypeTriangle
                           vertexStart:0
                           vertexCount:vertex_count
                         instanceCount:instance_count];
}

void Encoder::SetObjectBuffer(BufferId b, std::size_t offset, int slot) {
    [device_->impl_->enc setObjectBuffer:device_->impl_->buffers[b.v]
                                 offset:offset
                                atIndex:NSUInteger(slot)];
}

void Encoder::SetMeshBuffer(BufferId b, std::size_t offset, int slot) {
    [device_->impl_->enc setMeshBuffer:device_->impl_->buffers[b.v]
                               offset:offset
                              atIndex:NSUInteger(slot)];
}

void Encoder::DrawMeshThreadgroups(int count, int object_threads,
                                   int mesh_threads) {
    if (count <= 0) return;
    [device_->impl_->enc
        drawMeshThreadgroups:MTLSizeMake(NSUInteger(count), 1, 1)
        threadsPerObjectThreadgroup:MTLSizeMake(NSUInteger(object_threads), 1, 1)
          threadsPerMeshThreadgroup:MTLSizeMake(NSUInteger(mesh_threads), 1, 1)];
}

void Encoder::DrawIndexedU32(BufferId indices, std::size_t index_count) {
    [device_->impl_->enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                    indexCount:index_count
                                     indexType:MTLIndexTypeUInt32
                                   indexBuffer:device_->impl_->buffers[indices.v]
                             indexBufferOffset:0];
}

void Encoder::DrawIndexedInstancedU32(BufferId indices, std::size_t index_count,
                                      std::size_t instance_count) {
    if (instance_count == 0) return;  // Metal rejects zero outright
    [device_->impl_->enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                    indexCount:index_count
                                     indexType:MTLIndexTypeUInt32
                                   indexBuffer:device_->impl_->buffers[indices.v]
                             indexBufferOffset:0
                                 instanceCount:instance_count];
}

void Encoder::DrawIndexedIndirectU32(BufferId indices, BufferId args,
                                     std::size_t offset) {
    if (!Valid(indices) || !Valid(args)) return;
    [device_->impl_->enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                     indexType:MTLIndexTypeUInt32
                                   indexBuffer:device_->impl_->buffers[indices.v]
                             indexBufferOffset:0
                                indirectBuffer:device_->impl_->buffers[args.v]
                          indirectBufferOffset:offset];
}

}  // namespace eng::rhi
