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
 * SSIM and MS-SSIM definitions shared between the Metal kernels (ssim.metal)
 * and the host (ssim_metal.m), and the exact summation that the kernels and
 * the unit test (test_metal_ssim.c) share.
 */

#ifndef __VMAF_FEATURE_METAL_SSIM_METAL_H__
#define __VMAF_FEATURE_METAL_SSIM_METAL_H__

#include "soft_f64.h"

#ifdef __METAL_VERSION__
#define SSIM_THREAD thread
static inline unsigned ssim_float_bits(float f) { return as_type<uint>(f); }
static inline float ssim_bits_float(unsigned u) { return as_type<float>(u); }
static inline int ssim_clz64(ulong x) { return (int)clz(x); }
#else
#include <stdint.h>
#include <string.h>
#define SSIM_THREAD
static inline unsigned ssim_float_bits(float f)
{
    unsigned u;
    memcpy(&u, &f, sizeof(u));
    return u;
}
static inline float ssim_bits_float(unsigned u)
{
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}
static inline int ssim_clz64(uint64_t x) { return x ? __builtin_clzll(x) : 64; }
#endif

#define SSIM_METAL_GAUSSIAN_LEN 11
#define SSIM_METAL_MAPS 5   // mu1, mu2, ref², cmp², ref·cmp
#define SSIM_METAL_SCALES 5 // MS-SSIM
#define SSIM_METAL_MAX_TAPS 100 // decimation kernel, up to 10x10

/** picture_copy() */
typedef struct SsimMetalLoadParams {
    int w, h;
    int in_stride; // pixels
    float scaler;  // divisor for bpc > 8
} SsimMetalLoadParams;

/** _iqa_decimate() with _iqa_filter_pixel(), of the 2 planes */
typedef struct SsimMetalDecParams {
    int w, h;         // input
    int out_w, out_h; // w / factor + (w & 1)
    int factor;
    int kw, kh;       // kernel size
} SsimMetalDecParams;

/** _iqa_convolve() of the 5 maps, then the per pixel part of _iqa_ssim() */
typedef struct SsimMetalConvParams {
    int w, h;         // input
    int dst_w, dst_h; // w - 10, h - 10
    float c1, c2, c3;
    unsigned out_offset; // pixels, of this scale in the output buffers
    unsigned part_offset; // fast mode: threadgroups, of this scale
} SsimMetalConvParams;

/**
 * Exact emulation of the CPU's
 *
 *     double sum = 0.0;
 *     for (...) sum += (float)(x * k);
 *     return (float)sum;
 *
 * for n float terms given in the CPU's order, as a filter with a fallback:
 *
 * SsimSum adds the terms in float-float (hi + lo, with TwoSum), and their
 * magnitudes in a. Its value differs from the exact sum S by at most
 * n^2 * 2^-48 * a (the rounding of lo), and the CPU's double sum D from S by
 * at most n * 2^-53 * a. If the interval of that size around hi + lo lies
 * within the rounding interval of one float r, (float)D is r.
 *
 * At a power of 2, the rounding interval is narrower below r than above;
 * the check then uses the narrower side for both.
 *
 * Otherwise (`ok` false: cancellation, or D close to the midpoint of two
 * floats), the caller adds the terms again with ssim_sum_slow(), which
 * rounds every partial sum to double with soft_f64.h.
 */
typedef struct SsimSum {
    float hi, lo; // float-float sum
    float a;      // sum of the magnitudes
} SsimSum;

static inline void ssim_sum_init(SSIM_THREAD SsimSum *s)
{
    s->hi = 0.0f;
    s->lo = 0.0f;
    s->a = 0.0f;
}

/** TwoSum: x + y == *r + return value, exactly */
static inline float ssim_two_sum(float x, float y, SSIM_THREAD float *r)
{
    const float sum = x + y;
    const float yy = sum - x;
    *r = sum;
    return (x - (sum - yy)) + (y - yy);
}

static inline void ssim_sum_add(SSIM_THREAD SsimSum *s, float t)
{
    const float e = ssim_two_sum(s->hi, t, &s->hi);
    s->lo = s->lo + e;
    s->a = s->a + (t < 0.0f ? -t : t);
}

/**
 * (float)D, if `*ok`. `eps_k` bounds the error relative to a:
 * ssim_sum_eps(n).
 */
static inline float ssim_sum_result(SSIM_THREAD SsimSum *s, float eps_k,
                                    SSIM_THREAD bool *ok)
{
    *ok = true;
    if (s->a == 0.0f) return 0.0f; // all terms +-0: the CPU's sum is +0
    float r;
    const float rem = ssim_two_sum(s->hi, s->lo, &r);
    // the distance from r to the midpoints on either side: half the unit
    // in the last place of r, or a quarter below a power of 2
    const unsigned bits = ssim_float_bits(r);
    const unsigned e = (bits >> 23) & 0xff;
    if (e < 26 || e == 0xff) {
        *ok = false;
        return 0.0f;
    }
    const bool pow2 = !(bits & 0x7fffff);
    const float half_ulp = ssim_bits_float((pow2 ? e - 25 : e - 24) << 23);
    const float eps = s->a * eps_k;
    const float d = (rem < 0.0f ? -rem : rem) + eps;
    *ok = d < half_ulp;
    return r;
}

/** eps_k for n terms: 4 * n^2 * 2^-48, covers both error bounds twice. */
static inline float ssim_sum_eps(int n)
{
    return (float)(n * n) * 1.4210855e-14f; // 2^-46
}

/**
 * (float)(mag * 2^e2), negative if `neg`, rounded to nearest even, with
 * subnormal results.
 */
static inline float ssim_round_float(sf_u64 mag, int e2, bool neg)
{
    if (!mag) return 0.0f; // the CPU's sum is +0 then
    const int p = 63 - ssim_clz64(mag);
    int biased = p + e2 + 127; // of the leading bit
    // bits to drop: keep 24 bits, or fewer for a subnormal result
    const int sh = biased >= 1 ? p - 23 : p - 23 + (1 - biased);
    sf_u64 q;
    if (sh <= 0) {
        q = mag << -sh;
    } else if (sh > 63) {
        q = 0;
    } else {
        q = mag >> sh;
        const bool guard = ((mag >> (sh - 1)) & 1) != 0;
        const bool rest = (mag & ((((sf_u64)1) << (sh - 1)) - 1)) != 0;
        if (guard && (rest || (q & 1))) q++;
    }
    unsigned bits;
    if (biased >= 1) {
        if (q == ((sf_u64)1 << 24)) { // carry
            q >>= 1;
            biased++;
        }
        bits = biased >= 255 ? 0x7f800000u :
               ((unsigned)biased << 23) | ((unsigned)q & 0x7fffff);
    } else {
        bits = (unsigned)q; // 2^23 is the smallest normal number
    }
    return ssim_bits_float(bits | (neg ? 0x80000000u : 0));
}

/** A float as a soft double. */
static inline sf64 ssim_sf_from_float(float t)
{
    const unsigned u = ssim_float_bits(t);
    const int e = (int)((u >> 23) & 0xff);
    sf64 r;
    r.s = (int)(u >> 31);
    sf_u64 m = u & 0x7fffff;
    if (e) {
        r.m = (m | 0x800000) << 29;
        r.e = e - 150 - 29;
        return r;
    }
    if (!m) return sf_zero();
    // subnormal: normalize to 2^52 <= m < 2^53
    const int sh = 52 - (63 - ssim_clz64(m));
    r.m = m << sh;
    r.e = -149 - sh;
    return r;
}

/** One term of the slow path: sum += (double)t */
static inline sf64 ssim_sum_slow(sf64 sum, float t)
{
    return sf_add(sum, ssim_sf_from_float(t));
}

/** (float) of a soft double */
static inline float ssim_sf_to_float(sf64 a)
{
    return ssim_round_float(a.m, a.e, a.s != 0);
}

#endif /* __VMAF_FEATURE_METAL_SSIM_METAL_H__ */
