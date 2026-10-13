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
 * Integer ADM definitions shared between the Metal kernels
 * (integer_adm.metal), the host (integer_adm_metal.m) and the unit tests.
 */

#ifndef __VMAF_FEATURE_METAL_INTEGER_ADM_METAL_H__
#define __VMAF_FEATURE_METAL_INTEGER_ADM_METAL_H__

#include "soft_f64.h"

#define ADM_METAL_MAX_TG_W 256
#define ADM_METAL_N_DEN 3 // h, v, d
#define ADM_METAL_N_CM 6  // DLM h, v, d, AIM h, v, d

// kVariant bits
#define ADM_METAL_VARIANT_LO_ONLY (1u << 16) // dwt: low-pass band only
#define ADM_METAL_VARIANT_SKIP_AIM (1u << 17) // cm: DLM only

typedef struct AdmMetalParams {
    sf_u64 gain;          // adm_enhn_gain_limit, IEEE-754 bits
    sf_u64 cos_1deg_sq;   // (double)(float)cos(1 deg)^2, IEEE-754 bits
    float gain_f;         // adm_enhn_gain_limit, fast mode
    float cos_1deg_sq_f;  // fast mode
    int gain_int;         // adm_enhn_gain_limit if it is an integer <= 2^20,
                          // else 0
    int scale;
    int bpc;              // scale 0 input
    int src_w, src_h;     // DWT input
    int w, h;             // DWT output (bands)
    int src_stride;       // pixels
    int stride;           // pixels, bands
    unsigned plane;       // elements per decouple plane
    // decouple region
    int dec_left, dec_top, dec_right, dec_bottom;
    // csf_den / cm region
    int left, top, right, bottom;
    // csf_den rounding
    unsigned den_add_sq, den_shift_sq;   // scales 1-3
    unsigned den_add_cub, den_shift_cub; // scales 1-3
    unsigned den_add_row, den_shift_row;
    // cm
    unsigned i_rfactor[3];
    unsigned cm_add_cub[3], cm_shift_cub[3];
    unsigned cm_add_row, cm_shift_row;
    unsigned den_row_offset; // first row slot of this scale
    unsigned cm_row_offset;
} AdmMetalParams;

/**
 * (float)x rounded to nearest even, as an sf64. x >= 0.
 */
static inline sf64 adm_metal_rn24(sf_u64 x)
{
    if (!(x >> 24)) return sf_from_mag(x, 0);
    const sf_i32 sh = 63 - sf_clz64(x) - 23;
    sf_u64 q = x >> sh;
    const sf_u64 rem = x & ((((sf_u64)1) << sh) - 1);
    const sf_u64 hlf = ((sf_u64)1) << (sh - 1);
    if (rem > hlf || (rem == hlf && (q & 1))) q++;
    sf64 r = sf_from_mag(q, 0);
    r.e += sh;
    return r;
}

/**
 * Bit-exact equivalent of the angle test of adm_decouple*():
 *
 *     ((float)ot_dp / 4096.0) >= 0.0f &&
 *     ((float)ot_dp / 4096.0) * ((float)ot_dp / 4096.0) >=
 *         cos_1deg_sq * ((float)o_mag_sq / 4096.0) * ((float)t_mag_sq / 4096.0)
 *
 * The powers of two cancel out. (float)ot_dp squared is exact in double.
 */
static inline bool adm_metal_angle_exact(sf_i64 ot_dp, sf_i64 o_mag_sq,
                                         sf_i64 t_mag_sq, sf_u64 cos_1deg_sq)
{
    if (ot_dp < 0) return false;
    const sf64 d = adm_metal_rn24((sf_u64)ot_dp);
    const sf64 rhs = sf_mul(sf_mul(sf_from_bits(cos_1deg_sq),
                                   adm_metal_rn24((sf_u64)o_mag_sq)),
                            adm_metal_rn24((sf_u64)t_mag_sq));
    return !sf_lt(sf_mul(d, d), rhs);
}

/**
 * Bit-exact equivalent of the enhancement gain limit of adm_decouple*():
 *
 *     if (rst_f > 0.) rst = MIN(rst * adm_enhn_gain_limit, t);
 *     if (rst_f < 0.) rst = MAX(rst * adm_enhn_gain_limit, t);
 *
 * with the double result converted back to an integer. `positive` is
 * rst_f > 0.
 */
static inline sf_i64 adm_metal_gain_exact(sf_i64 rst, sf_i64 t, bool positive,
                                          sf_u64 gain, int gain_int)
{
    if (gain_int) {
        // rst * gain is exact in double
        const sf_i64 x = rst * gain_int;
        return positive ? (x < t ? x : t) : (x > t ? x : t);
    }
    const sf64 x = sf_mul(sf_from_i64(rst), sf_from_bits(gain));
    const sf64 tt = sf_from_i64(t);
    const bool take_x = positive ? sf_lt(x, tt) : sf_lt(tt, x);
    return take_x ? sf_trunc_i64(x) : t;
}

#endif /* __VMAF_FEATURE_METAL_INTEGER_ADM_METAL_H__ */
