/**
 *
 *  Copyright 2016-2026 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"
#include "log.h"

// Generated from the compiled kernels (vmaf.metallib) by xxd.
extern unsigned char vmaf_metallib[];
extern unsigned int vmaf_metallib_len;

struct VmafMetalPipeline {
    char *name;
    uint32_t variant;
    void *pipeline; ///< id<MTLComputePipelineState>, retained
    VmafMetalPipeline *next;
};

static int state_open(VmafMetalState *s, id<MTLDevice> device, bool fast_math)
{
    if (!device) device = MTLCreateSystemDefaultDevice();
    if (!device) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "no Metal device available\n");
        return -ENODEV;
    }

    dispatch_data_t data =
        dispatch_data_create(vmaf_metallib, vmaf_metallib_len, NULL,
                             DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithData:data error:&error];
    if (!library) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR, "problem loading Metal library: %s\n",
                 error.localizedDescription.UTF8String);
        return -EINVAL;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) return -ENOMEM;

    s->device = (void *)CFBridgingRetain(device);
    s->queue = (void *)CFBridgingRetain(queue);
    s->library = (void *)CFBridgingRetain(library);
    s->fast_math = fast_math;
    s->pipelines = NULL;
    pthread_mutex_init(&s->lock, NULL);
    pthread_mutex_init(&s->work_lock, NULL);
    pthread_cond_init(&s->work_done, NULL);
    s->work_pending = 0;
    s->work_err = 0;
    return 0;
}

int vmaf_metal_state_init(VmafMetalState **metal_state,
                          VmafMetalConfiguration cfg)
{
    if (!metal_state) return -EINVAL;

    VmafMetalState *s = calloc(1, sizeof(*s));
    if (!s) return -ENOMEM;

    @autoreleasepool {
        int err = state_open(s, (__bridge id<MTLDevice>)cfg.device,
                             cfg.fast_math);
        if (err) {
            free(s);
            return err;
        }

        // The kernels need 32-wide SIMD groups, which every Apple GPU has.
        // Intel and AMD GPUs (Intel Macs) may use other widths.
        if (!vmaf_metal_pipeline(s, "vmaf_metal_selftest", 0)) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "Metal device %s is not supported\n",
                     vmaf_metal_device(s).name.UTF8String);
            vmaf_metal_state_free(s);
            return -ENOTSUP;
        }
    }

    *metal_state = s;
    return 0;
}

void vmaf_metal_state_free(VmafMetalState *s)
{
    if (!s) return;

    vmaf_metal_sync(s);
    VmafMetalPipeline *p = s->pipelines;
    while (p) {
        VmafMetalPipeline *next = p->next;
        CFRelease(p->pipeline);
        free(p->name);
        free(p);
        p = next;
    }
    if (s->library) CFRelease(s->library);
    if (s->queue) CFRelease(s->queue);
    if (s->device) CFRelease(s->device);
    pthread_mutex_destroy(&s->lock);
    pthread_mutex_destroy(&s->work_lock);
    pthread_cond_destroy(&s->work_done);
    free(s);
}

static VmafMetalState *default_state;
static pthread_once_t default_state_once = PTHREAD_ONCE_INIT;

static void default_state_open(void)
{
    VmafMetalState *s = NULL;
    VmafMetalConfiguration cfg = { 0 };
    if (!vmaf_metal_state_init(&s, cfg))
        default_state = s;
}

VmafMetalState *vmaf_metal_state_default(void)
{
    pthread_once(&default_state_once, default_state_open);
    return default_state;
}

VmafMetalState *vmaf_metal_state_default_peek(void)
{
    return __atomic_load_n(&default_state, __ATOMIC_ACQUIRE);
}

int vmaf_metal_sync(VmafMetalState *s)
{
    if (!s) return 0;
    pthread_mutex_lock(&s->work_lock);
    while (s->work_pending)
        pthread_cond_wait(&s->work_done, &s->work_lock);
    const int err = s->work_err;
    pthread_mutex_unlock(&s->work_lock);
    return err;
}

int vmaf_metal_frames_init(VmafMetalFrames *f, VmafMetalState *ms)
{
    f->ms = ms;
    f->next = 0;
    dispatch_queue_t queue =
        dispatch_queue_create("org.vmaf.metal.frames", DISPATCH_QUEUE_SERIAL);
    if (!queue) return -ENOMEM;
    f->queue = (void *)CFBridgingRetain(queue);
    for (unsigned i = 0; i < VMAF_METAL_FRAMES; i++) {
        dispatch_semaphore_t sem = dispatch_semaphore_create(1);
        if (!sem) return -ENOMEM;
        f->slot_free[i] = (void *)CFBridgingRetain(sem);
    }
    return 0;
}

static inline dispatch_semaphore_t slot_sem(VmafMetalFrames *f, unsigned slot)
{
    return (__bridge dispatch_semaphore_t)f->slot_free[slot];
}

unsigned vmaf_metal_frames_acquire(VmafMetalFrames *f)
{
    const unsigned slot = f->next;
    f->next = (f->next + 1) % VMAF_METAL_FRAMES;
    dispatch_semaphore_wait(slot_sem(f, slot), DISPATCH_TIME_FOREVER);
    return slot;
}

void vmaf_metal_frames_cancel(VmafMetalFrames *f, unsigned slot)
{
    dispatch_semaphore_signal(slot_sem(f, slot));
}

void vmaf_metal_frames_commit(VmafMetalFrames *f, unsigned slot,
                              id<MTLCommandBuffer> cb, int (^done)(int err))
{
    VmafMetalState *s = f->ms;
    dispatch_semaphore_t sem = slot_sem(f, slot);
    dispatch_queue_t queue = (__bridge dispatch_queue_t)f->queue;

    pthread_mutex_lock(&s->work_lock);
    s->work_pending++;
    pthread_mutex_unlock(&s->work_lock);

    // Metal runs the completion handlers of a queue one at a time, so `done`
    // runs on the context's own serial queue: the contexts of a thread pool
    // compute their scores in parallel, each in frame order.
    [cb addCompletedHandler:^(id<MTLCommandBuffer> c) { dispatch_async(queue, ^{
        // Everything the committing thread wrote before the commit (state,
        // input buffers) is visible here, through the driver. Make that
        // ordering explicit, also for ThreadSanitizer: the committing thread
        // released work_lock after counting this frame.
        pthread_mutex_lock(&s->work_lock);
        pthread_mutex_unlock(&s->work_lock);

        int err = c.status == MTLCommandBufferStatusCompleted ? 0 : -EIO;
        if (err) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "Metal command buffer failed: %s\n",
                     c.error.localizedDescription.UTF8String);
        }
        err = done(err);
        dispatch_semaphore_signal(sem);

        pthread_mutex_lock(&s->work_lock);
        if (err && !s->work_err) s->work_err = err;
        if (!--s->work_pending) pthread_cond_broadcast(&s->work_done);
        pthread_mutex_unlock(&s->work_lock);
    }); }];
    [cb commit];
}

void vmaf_metal_frames_close(VmafMetalFrames *f)
{
    for (unsigned i = 0; i < VMAF_METAL_FRAMES; i++) {
        if (!f->slot_free[i]) continue;
        dispatch_semaphore_t sem = slot_sem(f, i);
        dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
        // a semaphore must be back at its initial value when released
        dispatch_semaphore_signal(sem);
        CFRelease(f->slot_free[i]);
        f->slot_free[i] = NULL;
    }
    // pending blocks retain the queue
    if (f->queue) CFRelease(f->queue);
    f->queue = NULL;
}

id<MTLComputePipelineState> vmaf_metal_pipeline(VmafMetalState *s,
                                                const char *name,
                                                uint32_t variant)
{
    id<MTLComputePipelineState> pipeline = nil;

    pthread_mutex_lock(&s->lock);
    for (VmafMetalPipeline *p = s->pipelines; p; p = p->next) {
        if (p->variant == variant && !strcmp(p->name, name)) {
            pipeline = (__bridge id<MTLComputePipelineState>)p->pipeline;
            goto unlock;
        }
    }

    @autoreleasepool {
        MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new];
        bool fast = s->fast_math;
        [cv setConstantValue:&fast type:MTLDataTypeBool atIndex:0];
        [cv setConstantValue:&variant type:MTLDataTypeUInt atIndex:1];

        NSError *error = nil;
        id<MTLLibrary> library = (__bridge id<MTLLibrary>)s->library;
        id<MTLFunction> fn =
            [library newFunctionWithName:[NSString stringWithUTF8String:name]
                          constantValues:cv
                                   error:&error];
        if (!fn) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "problem loading Metal kernel %s: %s\n", name,
                     error.localizedDescription.UTF8String);
            goto unlock;
        }
        pipeline = [vmaf_metal_device(s) newComputePipelineStateWithFunction:fn
                                                                       error:&error];
        if (!pipeline) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "problem creating Metal pipeline %s: %s\n", name,
                     error.localizedDescription.UTF8String);
            goto unlock;
        }
        if (pipeline.threadExecutionWidth != VMAF_METAL_SIMD_WIDTH) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR,
                     "Metal pipeline %s has SIMD width %lu, need %d\n", name,
                     (unsigned long)pipeline.threadExecutionWidth,
                     VMAF_METAL_SIMD_WIDTH);
            pipeline = nil;
            goto unlock;
        }
    }

    VmafMetalPipeline *p = malloc(sizeof(*p));
    if (!p) {
        pipeline = nil;
        goto unlock;
    }
    p->name = strdup(name);
    p->variant = variant;
    p->pipeline = (void *)CFBridgingRetain(pipeline);
    p->next = s->pipelines;
    s->pipelines = p;

unlock:
    pthread_mutex_unlock(&s->lock);
    return pipeline;
}

id<MTLBuffer> vmaf_metal_buffer(VmafMetalState *s, size_t size)
{
    return [vmaf_metal_device(s) newBufferWithLength:size ? size : 1
                                             options:MTLResourceStorageModeShared];
}

void vmaf_metal_dispatch_2d(id<MTLComputeCommandEncoder> enc,
                            id<MTLComputePipelineState> pipeline,
                            unsigned w, unsigned h)
{
    const NSUInteger tw = pipeline.threadExecutionWidth;
    const NSUInteger th =
        MAX(1, pipeline.maxTotalThreadsPerThreadgroup / tw / 4);
    [enc setComputePipelineState:pipeline];
    [enc dispatchThreads:MTLSizeMake(w, h, 1)
        threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
}

void vmaf_metal_copy_plane(void *dst, size_t dst_stride, const VmafPicture *pic,
                           unsigned plane)
{
    const size_t row = pic->w[plane] * (pic->bpc > 8 ? 2 : 1);
    const uint8_t *src = pic->data[plane];
    uint8_t *d = dst;

    if (row == dst_stride && (size_t)pic->stride[plane] == dst_stride) {
        memcpy(d, src, row * pic->h[plane]);
        return;
    }
    for (unsigned i = 0; i < pic->h[plane]; i++) {
        memcpy(d, src, row);
        d += dst_stride;
        src += pic->stride[plane];
    }
}

int vmaf_metal_selftest(VmafMetalState *s)
{
    if (!s) return -EINVAL;

    @autoreleasepool {
        const unsigned n = 1024;
        id<MTLComputePipelineState> pipeline =
            vmaf_metal_pipeline(s, "vmaf_metal_selftest", 3);
        if (!pipeline) return -EINVAL;

        id<MTLBuffer> buf = vmaf_metal_buffer(s, n * sizeof(uint32_t));
        if (!buf) return -ENOMEM;
        uint32_t *data = buf.contents;
        for (unsigned i = 0; i < n; i++) data[i] = i;

        id<MTLCommandBuffer> cb = [vmaf_metal_queue(s) commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setBuffer:buf offset:0 atIndex:0];
        vmaf_metal_dispatch_2d(enc, pipeline, n, 1);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return -EIO;

        // kernel computes data[i] = data[i] * 2 + kVariant (+1 if kFast)
        for (unsigned i = 0; i < n; i++) {
            if (data[i] != i * 2 + 3 + s->fast_math) return -EIO;
        }
    }

    return 0;
}

int vmaf_metal_soft_f64_eval(VmafMetalState *s, enum VmafMetalSoftF64Op op,
                             const uint64_t *a, const uint64_t *b,
                             const uint64_t *c, uint64_t *out, unsigned n)
{
    if (!s || !a || !b || !c || !out || !n) return -EINVAL;

    @autoreleasepool {
        id<MTLComputePipelineState> pipeline =
            vmaf_metal_pipeline(s, "vmaf_metal_soft_f64_eval", op);
        if (!pipeline) return -EINVAL;

        const size_t sz = n * sizeof(uint64_t);
        id<MTLDevice> dev = vmaf_metal_device(s);
        const MTLResourceOptions opt = MTLResourceStorageModeShared;
        id<MTLBuffer> ba = [dev newBufferWithBytes:a length:sz options:opt];
        id<MTLBuffer> bb = [dev newBufferWithBytes:b length:sz options:opt];
        id<MTLBuffer> bc = [dev newBufferWithBytes:c length:sz options:opt];
        id<MTLBuffer> bo = vmaf_metal_buffer(s, sz);
        if (!ba || !bb || !bc || !bo) return -ENOMEM;

        id<MTLCommandBuffer> cb = [vmaf_metal_queue(s) commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setBuffer:ba offset:0 atIndex:0];
        [enc setBuffer:bb offset:0 atIndex:1];
        [enc setBuffer:bc offset:0 atIndex:2];
        [enc setBuffer:bo offset:0 atIndex:3];
        vmaf_metal_dispatch_2d(enc, pipeline, n, 1);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) return -EIO;
        memcpy(out, bo.contents, sz);
    }

    return 0;
}
