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

/**
 * Integer motion definitions shared between the Metal kernel
 * (integer_motion.metal) and the host (integer_motion_metal.m).
 */

#ifndef __VMAF_FEATURE_METAL_INTEGER_MOTION_METAL_H__
#define __VMAF_FEATURE_METAL_INTEGER_MOTION_METAL_H__

#define MOTION_METAL_MAX_TG_W 256

typedef struct MotionMetalParams {
    int w, h;
    int prev_stride, cur_stride; // pixels
} MotionMetalParams;

#endif /* __VMAF_FEATURE_METAL_INTEGER_MOTION_METAL_H__ */
