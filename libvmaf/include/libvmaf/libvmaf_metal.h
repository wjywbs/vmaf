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

#ifndef __VMAF_METAL_H__
#define __VMAF_METAL_H__

#include "libvmaf/libvmaf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VmafMetalState VmafMetalState;

typedef struct VmafMetalConfiguration {
    void *device; ///< id<MTLDevice>, optional. System default device if NULL.
    /**
     * Precision mode of Metal feature extractors.
     * 0 (default): bit-exact with the CPU feature extractors. Double precision
     *              operations are emulated in software on the GPU.
     * 1:           fast mode. Double precision operations are replaced with
     *              single precision, scores may differ slightly from the CPU.
     */
    int fast_math;
} VmafMetalConfiguration;

/**
 * Initialize VmafMetalState.
 * VmafMetalState can optionally be configured with VmafMetalConfiguration.
 *
 * @param metal_state The Metal state to open.
 *
 * @param cfg         Optional configuration parameters.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_metal_state_init(VmafMetalState **metal_state,
                          VmafMetalConfiguration cfg);

/**
 * Import VmafMetalState for use during Metal feature extraction.
 * Ownership of `metal_state` is transferred to the VmafContext, it is
 * released during `vmaf_close()`.
 *
 * @param vmaf        VMAF context allocated with `vmaf_init()`.
 *
 * @param metal_state Metal state allocated with `vmaf_metal_state_init()`.
 *
 *
 * @return 0 on success, or < 0 (a negative errno code) on error.
 */
int vmaf_metal_import_state(VmafContext *vmaf, VmafMetalState *metal_state);

#ifdef __cplusplus
}
#endif

#endif /* __VMAF_METAL_H__ */
