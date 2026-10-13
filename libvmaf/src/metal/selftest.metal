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

#include "kernel_common.h"

kernel void vmaf_metal_selftest(device uint *data [[buffer(0)]],
                                uint2 pos [[thread_position_in_grid]])
{
    data[pos.x] = data[pos.x] * 2 + kVariant + (kFast ? 1 : 0);
}

#include "soft_f64.h"

/**
 * Evaluate soft_f64 operations on the GPU, used to verify the Metal
 * compilation of soft_f64.h against hardware doubles. kVariant selects the
 * operation, see enum VmafMetalSoftF64Op.
 */
kernel void vmaf_metal_soft_f64_eval(device const ulong *a [[buffer(0)]],
                                     device const ulong *b [[buffer(1)]],
                                     device const ulong *c [[buffer(2)]],
                                     device ulong *out [[buffer(3)]],
                                     uint2 pos [[thread_position_in_grid]])
{
    const uint i = pos.x;
    const sf64 x = sf_from_bits(a[i]), y = sf_from_bits(b[i]),
               z = sf_from_bits(c[i]);
    switch (kVariant) {
    case 0: out[i] = sf_to_bits(sf_add(x, y)); break;
    case 1: out[i] = sf_to_bits(sf_sub(x, y)); break;
    case 2: out[i] = sf_to_bits(sf_mul(x, y)); break;
    case 3: out[i] = sf_to_bits(sf_div(x, y)); break;
    case 4: out[i] = sf_to_bits(sf_fma(x, y, z)); break;
    case 5: out[i] = (ulong)sf_trunc_i64(x); break;
    case 6: out[i] = sf_to_bits(sf_from_i64((long)a[i])); break;
    case 7: out[i] = sf_lt(x, y) ? 1 : 0; break;
    }
}
