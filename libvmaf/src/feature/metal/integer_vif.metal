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
 * Integer VIF, bit-exact port of integer_vif.c.
 *
 * Per scale:
 *   vif_metal_vertical:  vertical filter of ref, dis, ref^2, dis^2, ref*dis
 *   vif_metal_stat:      horizontal filter + vif statistic, reduced to int64
 *                        partial sums per threadgroup
 *   vif_metal_subsample: filter + decimate into the next scale
 *
 * kVariant = scale | (bpc << 8)
 */

#include "kernel_common.h"
#include "integer_vif_metal.h"

constant uint kScale = kVariant & 0xff;
constant uint kBpc = kVariant >> 8;
// input samples are 16 bit for bpc > 8 and for all scales > 0
constant bool kHbd = kScale > 0 || kBpc > 8;

constant ushort vif_filter[4][17] = {
    { 489, 935, 1640, 2640, 3896, 5274, 6547, 7455, 7784, 7455, 6547, 5274, 3896, 2640, 1640, 935, 489 },
    { 1244, 3663, 7925, 12590, 14692, 12590, 7925, 3663, 1244 },
    { 3571, 16004, 26386, 16004, 3571 },
    { 10904, 43728, 10904 },
};

constant int vif_filter_width[4] = { 17, 9, 5, 3 };

constant int sigma_nsq = 65536 << 1;

// mirror padding, as pad_top_and_bottom() and PADDING_SQ_DATA()
static inline int reflect(int i, int n)
{
    if (i < 0) return -i;
    if (i >= n) return 2 * (n - 1) - i;
    return i;
}

static inline uint load(device const uchar *src, uint idx)
{
    return kHbd ? ((device const ushort *)src)[idx] : src[idx];
}

static inline int log2_32(device const ushort *log2_table, uint temp)
{
    const int k = 16 - (int)clz(temp);
    return log2_table[temp >> k] + 2048 * k;
}

static inline int log2_64(device const ushort *log2_table, ulong temp)
{
    const int k = 48 - (int)clz(temp);
    return log2_table[temp >> k] + 2048 * k;
}

kernel void vif_metal_vertical(constant VifMetalParams &p [[buffer(0)]],
                               device const uchar *ref [[buffer(1)]],
                               device const uchar *dis [[buffer(2)]],
                               device ushort *mu1 [[buffer(3)]],
                               device ushort *mu2 [[buffer(4)]],
                               device uint *xx [[buffer(5)]],
                               device uint *yy [[buffer(6)]],
                               device uint *xy [[buffer(7)]],
                               uint2 pos [[thread_position_in_grid]])
{
    if (pos.x >= p.w || pos.y >= p.h) return;

    const int fw = vif_filter_width[kScale];
    uint shift_vp, add_vp, shift_sq;
    ulong add_sq;
    if (kScale == 0) {
        shift_vp = kBpc;
        add_vp = 1u << (kBpc - 1);
        shift_sq = (kBpc - 8) * 2;
        add_sq = kBpc == 8 ? 0 : 1ul << (shift_sq - 1);
    } else {
        shift_vp = 16;
        add_vp = 32768;
        shift_sq = 16;
        add_sq = 32768;
    }

    /**
     * The taps of every filter sum to 65536, so sum(fc * x) fits 32 bits for
     * any 16 bit x. Products of two samples are split into 16 bit halves and
     * accumulated in 32 bits, which avoids 64 bit arithmetic in the loop.
     */
    uint a_mu1 = 0, a_mu2 = 0;
    uint xx_lo = 0, yy_lo = 0, xy_lo = 0, xx_hi = 0, yy_hi = 0, xy_hi = 0;
    for (int fi = 0; fi < fw; fi++) {
        const int row = reflect((int)pos.y - fw / 2 + fi, p.h);
        const uint idx = row * p.src_stride + pos.x;
        const uint r = load(ref, idx);
        const uint d = load(dis, idx);
        const uint fc = vif_filter[kScale][fi];
        a_mu1 += fc * r;
        a_mu2 += fc * d;
        const uint rr = r * r, dd = d * d, rd = r * d;
        if (kHbd) {
            xx_lo += fc * (rr & 0xffff);
            yy_lo += fc * (dd & 0xffff);
            xy_lo += fc * (rd & 0xffff);
            xx_hi += fc * (rr >> 16);
            yy_hi += fc * (dd >> 16);
            xy_hi += fc * (rd >> 16);
        } else {
            // 8 bit: 65536 * 255 * 255 < 2^32
            xx_lo += fc * rr;
            yy_lo += fc * dd;
            xy_lo += fc * rd;
        }
    }
    const ulong a_xx = ((ulong)xx_hi << 16) + xx_lo;
    const ulong a_yy = ((ulong)yy_hi << 16) + yy_lo;
    const ulong a_xy = ((ulong)xy_hi << 16) + xy_lo;

    const uint o = pos.y * p.w + pos.x;
    mu1[o] = (ushort)((a_mu1 + add_vp) >> shift_vp);
    mu2[o] = (ushort)((a_mu2 + add_vp) >> shift_vp);
    xx[o] = (uint)((a_xx + add_sq) >> shift_sq);
    yy[o] = (uint)((a_yy + add_sq) >> shift_sq);
    xy[o] = (uint)((a_xy + add_sq) >> shift_sq);
}

kernel void vif_metal_stat(constant VifMetalParams &p [[buffer(0)]],
                           device const ushort *mu1 [[buffer(1)]],
                           device const ushort *mu2 [[buffer(2)]],
                           device const uint *xx [[buffer(3)]],
                           device const uint *yy [[buffer(4)]],
                           device const uint *xy [[buffer(5)]],
                           device const ushort *log2_table [[buffer(6)]],
                           device long *partial [[buffer(7)]],
                           uint2 pos [[thread_position_in_grid]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint lid [[thread_index_in_threadgroup]],
                           uint simd_lane [[thread_index_in_simdgroup]],
                           uint simd_group [[simdgroup_index_in_threadgroup]],
                           uint n_simd_groups [[simdgroups_per_threadgroup]])
{
    long num_log = 0, den_log = 0, num_non_log = 0, den_non_log = 0;

    if (pos.x < p.w && pos.y < p.h) {
        const int fw = vif_filter_width[kScale];
        const uint row = pos.y * p.w;
        // 32 bit accumulation as in vif_metal_vertical()
        uint a_mu1 = 0, a_mu2 = 0;
        uint xx_lo = 0, yy_lo = 0, xy_lo = 0, xx_hi = 0, yy_hi = 0, xy_hi = 0;
        for (int fj = 0; fj < fw; fj++) {
            const uint idx = row + reflect((int)pos.x - fw / 2 + fj, p.w);
            const uint fc = vif_filter[kScale][fj];
            a_mu1 += fc * (uint)mu1[idx];
            a_mu2 += fc * (uint)mu2[idx];
            const uint vxx = xx[idx], vyy = yy[idx], vxy = xy[idx];
            xx_lo += fc * (vxx & 0xffff);
            yy_lo += fc * (vyy & 0xffff);
            xy_lo += fc * (vxy & 0xffff);
            xx_hi += fc * (vxx >> 16);
            yy_hi += fc * (vyy >> 16);
            xy_hi += fc * (vxy >> 16);
        }

        // ((ulong)a * b + 2^31) >> 32
        const uint mu1_sq = mulhi(a_mu1, a_mu1) + ((a_mu1 * a_mu1) >> 31);
        const uint mu2_sq = mulhi(a_mu2, a_mu2) + ((a_mu2 * a_mu2) >> 31);
        const uint mu1_mu2 = mulhi(a_mu1, a_mu2) + ((a_mu1 * a_mu2) >> 31);
        // (((ulong)hi << 16) + lo + 32768) >> 16
        const uint xx_filt = xx_hi + (uint)(((ulong)xx_lo + 32768) >> 16);
        const uint yy_filt = yy_hi + (uint)(((ulong)yy_lo + 32768) >> 16);
        const uint xy_filt = xy_hi + (uint)(((ulong)xy_lo + 32768) >> 16);

        const int sigma1_sq = as_type<int>(xx_filt - mu1_sq);
        // only feeds num_non_log when negative, the log branch needs > 0
        const int sigma2_raw = as_type<int>(yy_filt - mu2_sq);
        const int sigma2_sq =
            pos.x < p.noclamp_end ? sigma2_raw : max(sigma2_raw, 0);
        const int sigma12 = as_type<int>(xy_filt - mu1_mu2);

        if (sigma1_sq >= sigma_nsq) {
            den_log = log2_32(log2_table, (uint)sigma_nsq + (uint)sigma1_sq) - 2048 * 17;
            if (sigma12 > 0 && sigma2_sq > 0) {
                const VifMetalG g = kFast ?
                    vif_metal_g_fast(sigma1_sq, sigma2_sq, sigma12, p.gain_limit_f) :
                    vif_metal_g_exact(sigma1_sq, sigma2_sq, sigma12, p.gain_limit);
                const uint numer1 = (uint)g.sv_sq + sigma_nsq;
                const long numer1_tmp = g.numer1_tmp + numer1;
                num_log = log2_64(log2_table, (ulong)numer1_tmp) -
                          log2_64(log2_table, numer1);
            }
        } else {
            num_non_log = sigma2_sq;
            den_non_log = 1;
        }
    }

    /**
     * Per threadgroup sums. Per pixel, num_log and den_log are below 2^17 and
     * den_non_log is 0 or 1, so 32 bit sums over a threadgroup are exact.
     * num_non_log (|sigma2_sq| < 2^31) is summed in 16 bit halves.
     */
    threadgroup int sh[VIF_METAL_N_ACCUM + 1][VIF_METAL_TG_W * VIF_METAL_TG_H / 32];
    const int v[VIF_METAL_N_ACCUM + 1] = {
        (int)num_log, (int)den_log, (int)(num_non_log >> 16),
        (int)(num_non_log & 0xffff), (int)den_non_log,
    };
    for (uint k = 0; k < VIF_METAL_N_ACCUM + 1; k++) {
        const int sum = simd_sum(v[k]);
        if (simd_lane == 0) sh[k][simd_group] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid < VIF_METAL_N_ACCUM) {
        long t[VIF_METAL_N_ACCUM + 1] = { 0, 0, 0, 0, 0 };
        for (uint k = 0; k < VIF_METAL_N_ACCUM + 1; k++)
            for (uint g = 0; g < n_simd_groups; g++)
                t[k] += sh[k][g];
        const long out[VIF_METAL_N_ACCUM] = {
            t[0], t[1], (t[2] << 16) + t[3], t[4],
        };
        const uint slot = p.partial_offset + tg.y * p.tg_cols + tg.x;
        partial[slot * VIF_METAL_N_ACCUM + lid] = out[lid];
    }
}

/**
 * Filter scale kScale with the filter of scale kScale + 1 and decimate by 2,
 * as subsample_rd_8() / subsample_rd_16() + decimate_and_pad().
 * One thread per output sample of the next scale.
 */
kernel void vif_metal_subsample(constant VifMetalParams &p [[buffer(0)]],
                                device const uchar *ref [[buffer(1)]],
                                device const uchar *dis [[buffer(2)]],
                                device ushort *ref_out [[buffer(3)]],
                                device ushort *dis_out [[buffer(4)]],
                                uint2 pos [[thread_position_in_grid]])
{
    if (pos.x >= p.w / 2 || pos.y >= p.h / 2) return;

    const int fw = vif_filter_width[kScale + 1];
    const uint shift_vp = kScale == 0 ? kBpc : 16;
    const uint add_vp = 1u << (shift_vp - 1);
    const int x = 2 * pos.x, y = 2 * pos.y;

    uint a_ref = 0, a_dis = 0;
    for (int fj = 0; fj < fw; fj++) {
        const int col = reflect(x - fw / 2 + fj, p.w);
        uint v_ref = 0, v_dis = 0;
        for (int fi = 0; fi < fw; fi++) {
            const int row = reflect(y - fw / 2 + fi, p.h);
            const uint idx = row * p.src_stride + col;
            const uint fc = vif_filter[kScale + 1][fi];
            v_ref += fc * load(ref, idx);
            v_dis += fc * load(dis, idx);
        }
        const uint fc = vif_filter[kScale + 1][fj];
        a_ref += fc * (uint)(ushort)((v_ref + add_vp) >> shift_vp);
        a_dis += fc * (uint)(ushort)((v_dis + add_vp) >> shift_vp);
    }

    const uint o = pos.y * p.dst_stride + pos.x;
    ref_out[o] = (ushort)((a_ref + 32768) >> 16);
    dis_out[o] = (ushort)((a_dis + 32768) >> 16);
}
