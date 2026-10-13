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
 * Integer VIF definitions shared between the Metal kernels
 * (integer_vif.metal), the host (integer_vif_metal.m) and the unit tests.
 */

#ifndef __VMAF_FEATURE_METAL_INTEGER_VIF_METAL_H__
#define __VMAF_FEATURE_METAL_INTEGER_VIF_METAL_H__

#include "soft_f64.h"

#define VIF_METAL_TG_W 32
#define VIF_METAL_TG_H 8
#define VIF_METAL_N_ACCUM 4 // num_log, den_log, num_non_log, den_non_log

typedef struct VifMetalParams {
    unsigned w, h;        // dimensions of the current scale
    unsigned src_stride;  // in pixels
    unsigned dst_stride;  // in pixels (subsample output)
    unsigned bpc;
    unsigned scale;
    unsigned partial_offset; // first partial sum slot of this scale
    unsigned tg_cols;        // threadgroups per row
    unsigned noclamp_end;    // sigma2_sq is not clamped to >= 0 in columns
                             // [0, noclamp_end), as vif_statistic_8_neon()
    float gain_limit_f;      // vif_enhn_gain_limit, fast mode
    sf_u64 gain_limit;       // vif_enhn_gain_limit, IEEE-754 bits
} VifMetalParams;

typedef struct VifMetalG {
    sf_i32 sv_sq;
    sf_i64 numer1_tmp; // without numer1
} VifMetalG;

/**
 * Bit-exact equivalent of the double precision part of vif_statistic_*():
 *
 *     const double eps = 65536 * 1.0e-10;
 *     double g = sigma12 / (sigma1_sq + eps);
 *     int32_t sv_sq = sigma2_sq - g * sigma12;   // compiled to fmsub
 *     sv_sq = (uint32_t)(MAX(sv_sq, 0));
 *     g = MIN(g, vif_enhn_gain_limit);
 *     (int64_t)((g * g * sigma1_sq))
 */
static inline VifMetalG vif_metal_g_exact(sf_i32 sigma1_sq, sf_i32 sigma2_sq,
                                          sf_i32 sigma12, sf_u64 gain_limit)
{
    const sf64 eps = sf_from_bits(SF_U64_C(0x3EDB7CDFD9D7BDBB)); // 65536 * 1.0e-10
    const sf64 s1 = sf_from_i64(sigma1_sq);
    const sf64 s12 = sf_from_i64(sigma12);
    sf64 g = sf_div(s12, sf_add(s1, eps));
    const sf64 sv = sf_fma(sf_neg(g), s12, sf_from_i64(sigma2_sq));
    const sf64 limit = sf_from_bits(gain_limit);
    if (!sf_lt(g, limit)) g = limit;

    VifMetalG r;
    r.sv_sq = sv.s ? 0 : (sf_i32)sf_trunc_i64(sv);
    r.numer1_tmp = sf_trunc_i64(sf_mul(sf_mul(g, g), s1));
    return r;
}

/**
 * Single precision approximation of vif_metal_g_exact(), fast mode.
 */
static inline VifMetalG vif_metal_g_fast(sf_i32 sigma1_sq, sf_i32 sigma2_sq,
                                         sf_i32 sigma12, float gain_limit)
{
    float g = (float)sigma12 / (float)sigma1_sq;
    const float sv = (float)sigma2_sq - g * (float)sigma12;
    g = g < gain_limit ? g : gain_limit;

    VifMetalG r;
    r.sv_sq = sv > 0.0f ? (sf_i32)sv : 0;
    r.numer1_tmp = (sf_i64)(g * g * (float)sigma1_sq);
    return r;
}

#endif /* __VMAF_FEATURE_METAL_INTEGER_VIF_METAL_H__ */
