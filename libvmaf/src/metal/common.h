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

#ifndef __VMAF_SRC_METAL_COMMON_H__
#define __VMAF_SRC_METAL_COMMON_H__

#include <pthread.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "libvmaf/libvmaf_metal.h"
#include "libvmaf/picture.h"

typedef struct VmafMetalPipeline VmafMetalPipeline;

struct VmafMetalState {
    void *device;  ///< id<MTLDevice>, retained
    void *queue;   ///< id<MTLCommandQueue>, retained
    void *library; ///< id<MTLLibrary>, retained
    bool fast_math;
    pthread_mutex_t lock; ///< guards the pipeline cache
    VmafMetalPipeline *pipelines;

    // in-flight frames, see vmaf_metal_frames_commit() and vmaf_metal_sync()
    pthread_mutex_t work_lock;
    pthread_cond_t work_done;
    unsigned work_pending;
    int work_err;
};

/**
 * Release a VmafMetalState created by `vmaf_metal_state_init()`.
 */
void vmaf_metal_state_free(VmafMetalState *s);

/**
 * Process-wide default state (system default device, exact mode). Used by
 * Metal feature extractors when no state has been imported into the
 * VmafContext, e.g. when a Metal extractor is requested explicitly by name.
 *
 * @return NULL if no Metal device is available.
 */
VmafMetalState *vmaf_metal_state_default(void);

/**
 * Wait until all frames committed with vmaf_metal_frames_commit() on state
 * `s` have completed, including their completion blocks, which write the
 * scores.
 *
 * @return the first error of any completed frame, or 0.
 */
int vmaf_metal_sync(VmafMetalState *s);

/**
 * The default state if it has been created, else NULL.
 */
VmafMetalState *vmaf_metal_state_default_peek(void);

/**
 * Dispatch the self-test kernel, used to verify that the embedded metallib
 * loads and executes.
 *
 * @return 0 on success.
 */
int vmaf_metal_selftest(VmafMetalState *s);

enum VmafMetalSoftF64Op {
    VMAF_METAL_SOFT_F64_ADD = 0,
    VMAF_METAL_SOFT_F64_SUB,
    VMAF_METAL_SOFT_F64_MUL,
    VMAF_METAL_SOFT_F64_DIV,
    VMAF_METAL_SOFT_F64_FMA,
    VMAF_METAL_SOFT_F64_TRUNC_I64,
    VMAF_METAL_SOFT_F64_FROM_I64,
    VMAF_METAL_SOFT_F64_LT,
};

/**
 * Evaluate soft_f64.h operation `op` on the GPU for n inputs (IEEE-754 bit
 * patterns). For testing.
 *
 * @return 0 on success.
 */
int vmaf_metal_soft_f64_eval(VmafMetalState *s, enum VmafMetalSoftF64Op op,
                             const uint64_t *a, const uint64_t *b,
                             const uint64_t *c, uint64_t *out, unsigned n);

/**
 * Copy one plane of `pic` into `dst`, tightly packed with `dst_stride` bytes
 * per row.
 */
void vmaf_metal_copy_plane(void *dst, size_t dst_stride, const VmafPicture *pic,
                           unsigned plane);

#ifdef __OBJC__
#import <Metal/Metal.h>

static inline id<MTLDevice> vmaf_metal_device(VmafMetalState *s)
{
    return (__bridge id<MTLDevice>)s->device;
}

static inline id<MTLCommandQueue> vmaf_metal_queue(VmafMetalState *s)
{
    return (__bridge id<MTLCommandQueue>)s->queue;
}

/**
 * Fetch (and lazily create) a compute pipeline for kernel `name`.
 * Kernels are specialized with function constants:
 *   index 0 (bool kFast)   : s->fast_math
 *   index 1 (uint kVariant): `variant`, kernel specific
 * Thread-safe; pipelines live as long as the state.
 *
 * @return nil on error.
 */
id<MTLComputePipelineState> vmaf_metal_pipeline(VmafMetalState *s,
                                                const char *name,
                                                uint32_t variant);

/**
 * Allocate a buffer in shared (CPU + GPU visible) storage.
 */
id<MTLBuffer> vmaf_metal_buffer(VmafMetalState *s, size_t size);

/**
 * Number of frames an extractor context can have in flight.
 */
#define VMAF_METAL_FRAMES 3

/**
 * Asynchronous frames of one extractor context.
 *
 * extract() acquires a slot, fills the slot's input buffers, encodes and
 * commits a command buffer, and returns without waiting. A completion block
 * reads the slot's output buffers and writes the scores. Buffers that the CPU
 * writes before the commit or reads after the completion are per slot; GPU
 * intermediates can be shared, since command buffers of one queue are
 * ordered by Metal's hazard tracking.
 */
typedef struct VmafMetalFrames {
    VmafMetalState *ms;
    unsigned next;
    void *slot_free[VMAF_METAL_FRAMES]; ///< dispatch_semaphore_t, retained
} VmafMetalFrames;

int vmaf_metal_frames_init(VmafMetalFrames *f, VmafMetalState *ms);

/**
 * Wait until the next slot is free.
 *
 * @return slot index, in [0, VMAF_METAL_FRAMES).
 */
unsigned vmaf_metal_frames_acquire(VmafMetalFrames *f);

/**
 * Release a slot without committing, e.g. on an encoding error.
 */
void vmaf_metal_frames_cancel(VmafMetalFrames *f, unsigned slot);

/**
 * Commit `cb` for `slot`. When the GPU has completed it, `done` is called
 * with 0, or -EIO if the command buffer failed, on a Metal thread. Its return
 * value is recorded for vmaf_metal_sync(). The slot is free again after
 * `done` returns.
 */
void vmaf_metal_frames_commit(VmafMetalFrames *f, unsigned slot,
                              id<MTLCommandBuffer> cb, int (^done)(int err));

/**
 * Wait until all slots are free, and release them. Safe to call on a
 * zero-initialized VmafMetalFrames.
 */
void vmaf_metal_frames_close(VmafMetalFrames *f);

/**
 * Dispatch a 2D grid of w x h threads using a threadgroup size that suits
 * `pipeline`.
 */
void vmaf_metal_dispatch_2d(id<MTLComputeCommandEncoder> enc,
                            id<MTLComputePipelineState> pipeline,
                            unsigned w, unsigned h);
#endif

#endif /* __VMAF_SRC_METAL_COMMON_H__ */
