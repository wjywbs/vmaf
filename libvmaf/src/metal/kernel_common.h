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

// Shared prelude for all libvmaf Metal kernels.

#ifndef __VMAF_SRC_METAL_KERNEL_COMMON_H__
#define __VMAF_SRC_METAL_KERNEL_COMMON_H__

#include <metal_stdlib>
using namespace metal;

// The CPU feature extractors are the reference. Never fuse a * b + c into an
// fma unless the CPU code does so explicitly.
#pragma METAL fp contract(off)

// Function constants, set by vmaf_metal_pipeline() (src/metal/common.m).
constant bool kFast [[function_constant(0)]];    // fast (non bit-exact) mode
constant uint kVariant [[function_constant(1)]]; // kernel specific

#endif /* __VMAF_SRC_METAL_KERNEL_COMMON_H__ */
