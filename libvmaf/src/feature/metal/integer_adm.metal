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
 * Integer ADM, bit-exact port of integer_adm.c.
 *
 * Per scale:
 *   adm_metal_dwt: DWT of ref and dis (adm_dwt2_8/16, adm_dwt2_s123_combined),
 *                  then per band coefficient:
 *                  - csf_den of ref (adm_csf_den_scale, adm_csf_den_s123),
 *                    summed per row
 *                  - decouple (adm_decouple, adm_decouple_s123), written to
 *                    the six planes r_h, r_v, r_d, a_h, a_v, a_d
 *                  - the low-pass bands, the input of the next scale
 *   adm_metal_cm:  csf (adm_csf, i4_adm_csf) and contrast masking (adm_cm,
 *                  i4_adm_cm) of DLM and AIM, summed per row
 *
 * One threadgroup computes one row, so the per-row rounding of the CPU is
 * applied here. The host sums the rows and computes the scores.
 *
 * kVariant = scale | (bpc << 8) | ADM_METAL_VARIANT_*
 */

#include "kernel_common.h"
#include "integer_adm_metal.h"

constant uint kScale = kVariant & 0xff;
constant uint kBpc = (kVariant >> 8) & 0xff;
constant bool kS0 = kScale == 0;
constant bool kLoOnly = (kVariant & ADM_METAL_VARIANT_LO_ONLY) != 0;
constant bool kSkipAim = (kVariant & ADM_METAL_VARIANT_SKIP_AIM) != 0;

constant int dwt_lo[4] = { 15826, 27411, 7345, -4240 };
constant int dwt_hi[4] = { -4240, -7345, 27411, -15826 };
constant int dwt_lo_sum = 46342;

// adm_dwt2_s123_combined() rounding, by scale - 1
constant int s123_add_vp[3] = { 0, 32768, 32768 };
constant int s123_shift_vp[3] = { 0, 16, 16 };
constant int s123_add_hp[3] = { 16384, 32768, 16384 };
constant int s123_shift_hp[3] = { 15, 16, 15 };

// C abs(), INT32_MIN wraps
static inline int iabs(int x) { return x < 0 ? -x : x; }
#define INT32_MIN_ (-2147483647 - 1)

/** Sum over a threadgroup of 32 * n_simd threads, valid in thread 0. */
template <typename T, int N>
static inline void tg_sum(thread T *v, threadgroup T *sh, uint lid,
                          uint simd_lane, uint simd_group, uint n_simd_groups)
{
    for (int k = 0; k < N; k++) {
        for (ushort o = 16; o > 0; o >>= 1)
            v[k] += as_type<T>(simd_shuffle_down(as_type<uint2>(v[k]), o));
        if (simd_lane == 0) sh[simd_group * N + k] = v[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        for (uint g = 1; g < n_simd_groups; g++) {
            for (int k = 0; k < N; k++) v[k] += sh[g * N + k];
        }
    }
}

/** dwt2_src_indices_filt() */
static inline int dwt_index(int i, int k, int n)
{
    const int v = 2 * i - 1 + k;
    return v < 0 ? -v : (v >= n ? 2 * n - v - 1 : v);
}

struct Bands {
    int a, v, h, d;
};

static inline int load_src(device const uchar *src, int idx)
{
    if (kS0 && kBpc == 8) return src[idx];
    if (kS0) return ((device const ushort *)src)[idx];
    return ((device const int *)src)[idx];
}

/** Vertical pass of one column: low-pass and high-pass. */
static inline int2 dwt_vertical(constant AdmMetalParams &p,
                                device const uchar *src, thread const int *row,
                                int col)
{
    int s[4];
    for (int k = 0; k < 4; k++) s[k] = load_src(src, row[k] * p.src_stride + col);

    if (kS0) {
        const int add = 1 << (kBpc - 1);
        int lo = 0, hi = 0;
        for (int k = 0; k < 4; k++) {
            lo += dwt_lo[k] * s[k];
            hi += dwt_hi[k] * s[k];
        }
        lo -= dwt_lo_sum * add;
        return int2((short)((lo + add) >> kBpc), (short)((hi + add) >> kBpc));
    }

    long lo = 0, hi = 0;
    for (int k = 0; k < 4; k++) {
        lo += (long)dwt_lo[k] * s[k];
        hi += (long)dwt_hi[k] * s[k];
    }
    const int add = s123_add_vp[kScale - 1], shift = s123_shift_vp[kScale - 1];
    return int2((int)((lo + add) >> shift), (int)((hi + add) >> shift));
}

static inline int dwt_horizontal(thread const int *s, constant int *f)
{
    if (kS0) {
        int acc = 0;
        for (int k = 0; k < 4; k++) acc += f[k] * s[k];
        return (short)((acc + 32768) >> 16);
    }
    long acc = 0;
    for (int k = 0; k < 4; k++) acc += (long)f[k] * s[k];
    return (int)((acc + s123_add_hp[kScale - 1]) >> s123_shift_hp[kScale - 1]);
}

static inline Bands dwt(constant AdmMetalParams &p, device const uchar *src,
                        thread const int *row, int j)
{
    int lo[4], hi[4];
    for (int k = 0; k < 4; k++) {
        const int2 v = dwt_vertical(p, src, row, dwt_index(j, k, p.src_w));
        lo[k] = v.x;
        hi[k] = v.y;
    }
    Bands b;
    b.a = dwt_horizontal(lo, dwt_lo);
    if (!kLoOnly) {
        b.v = dwt_horizontal(lo, dwt_hi);
        b.h = dwt_horizontal(hi, dwt_lo);
        b.d = dwt_horizontal(hi, dwt_hi);
    }
    return b;
}

static inline ulong den_cube(constant AdmMetalParams &p, int x)
{
    if (kS0) {
        const uint u = (ushort)iabs(x);
        return (ulong)(u * u) * u;
    }
    const uint u = (uint)iabs(x);
    return (((((ulong)u * u) + p.den_add_sq) >> p.den_shift_sq) * u +
            p.den_add_cub) >> p.den_shift_cub;
}

/** k of adm_decouple*(), the clamped ratio t / o in Q15. */
static inline int decouple_k(int o, int t)
{
    if (o == 0) return 32768;
    long k;
    if (kS0) {
        // div_lookup[o + 32768]
        const int recip = (int)(0x40000000u / (uint)iabs(o));
        k = (int)((((long)(o < 0 ? -recip : recip) * t) + 16384) >> 15);
    } else {
        const uint abs_o = (uint)iabs(o);
        int shift = 0;
        uint msb = abs_o;
        if (abs_o >= 32768) {
            // get_best15_from32()
            shift = 17 - (int)clz(abs_o);
            msb = (ushort)((abs_o + (1u << (shift - 1))) >> shift);
        }
        const long recip = (int)(0x40000000u / msb);
        // (1 << 31) is INT32_MIN on the CPU
        k = ((recip * t) * (o < 0 ? -1 : 1) + (long)(int)(1u << (14 + shift))) >>
            (15 + shift);
    }
    return (int)(k < 0 ? 0 : (k > 32768 ? 32768 : k));
}

static inline int decouple_rst(constant AdmMetalParams &p, int o, int t,
                               bool angle_flag)
{
    const int k = decouple_k(o, t);
    int rst = kS0 ? (int)(short)((k * o + 16384) >> 15)
                  : (int)(((long)k * o + 16384) >> 15);
    // rst_f = (k / 32768) * (o / 64) is nonzero exactly when k and o are
    if (angle_flag && k > 0 && o != 0) {
        const bool positive = o > 0;
        if (kFast) {
            const float x = (float)rst * p.gain_f;
            const float tf = (float)t;
            rst = (int)(positive ? (x < tf ? x : tf) : (x > tf ? x : tf));
        } else {
            rst = (int)adm_metal_gain_exact(rst, t, positive, p.gain, p.gain_int);
        }
        if (kS0) rst = (short)rst;
    }
    return rst;
}

static inline bool decouple_angle(constant AdmMetalParams &p, Bands o, Bands t)
{
    const long ot_dp = (long)o.h * t.h + (long)o.v * t.v;
    const long o_mag_sq = (long)o.h * o.h + (long)o.v * o.v;
    const long t_mag_sq = (long)t.h * t.h + (long)t.v * t.v;
    if (kFast) {
        const float d = (float)ot_dp;
        return d >= 0.0f &&
               d * d >= p.cos_1deg_sq_f * (float)o_mag_sq * (float)t_mag_sq;
    }
    return adm_metal_angle_exact(ot_dp, o_mag_sq, t_mag_sq, p.cos_1deg_sq);
}

static inline void store_band(device uchar *dec, uint idx, int v)
{
    if (kS0) ((device short *)dec)[idx] = (short)v;
    else ((device int *)dec)[idx] = v;
}

kernel void adm_metal_dwt(constant AdmMetalParams &p [[buffer(0)]],
                          device const uchar *ref [[buffer(1)]],
                          device const uchar *dis [[buffer(2)]],
                          device int *ref_a [[buffer(3)]],
                          device int *dis_a [[buffer(4)]],
                          device uchar *dec [[buffer(5)]],
                          device ulong *den_rows [[buffer(6)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint lid [[thread_index_in_threadgroup]],
                          uint tg_w [[threads_per_threadgroup]],
                          uint simd_lane [[thread_index_in_simdgroup]],
                          uint simd_group [[simdgroup_index_in_threadgroup]],
                          uint n_simd_groups [[simdgroups_per_threadgroup]])
{
    const int i = tg;
    int row[4];
    for (int k = 0; k < 4; k++) row[k] = dwt_index(i, k, p.src_h);

    const bool den_row = i >= p.top && i < p.bottom;
    const bool dec_row = i >= p.dec_top && i < p.dec_bottom;
    ulong den[ADM_METAL_N_DEN] = { 0, 0, 0 };

    for (int j = lid; j < p.w; j += tg_w) {
        const Bands o = dwt(p, ref, row, j);
        const Bands t = dwt(p, dis, row, j);
        const uint idx = i * p.stride + j;
        ref_a[idx] = o.a;
        dis_a[idx] = t.a;
        if (kLoOnly) continue;

        if (den_row && j >= p.left && j < p.right) {
            den[0] += den_cube(p, o.h);
            den[1] += den_cube(p, o.v);
            den[2] += den_cube(p, o.d);
        }

        if (dec_row && j >= p.dec_left && j < p.dec_right) {
            const bool angle_flag = decouple_angle(p, o, t);
            const int rst_h = decouple_rst(p, o.h, t.h, angle_flag);
            const int rst_v = decouple_rst(p, o.v, t.v, angle_flag);
            const int rst_d = decouple_rst(p, o.d, t.d, angle_flag);
            store_band(dec, 0 * p.plane + idx, rst_h);
            store_band(dec, 1 * p.plane + idx, rst_v);
            store_band(dec, 2 * p.plane + idx, rst_d);
            store_band(dec, 3 * p.plane + idx, t.h - rst_h);
            store_band(dec, 4 * p.plane + idx, t.v - rst_v);
            store_band(dec, 5 * p.plane + idx, t.d - rst_d);
        }
    }
    if (kLoOnly) return;

    threadgroup ulong sh[ADM_METAL_MAX_TG_W / 32 * ADM_METAL_N_DEN];
    tg_sum<ulong, ADM_METAL_N_DEN>(den, sh, lid, simd_lane, simd_group,
                                   n_simd_groups);
    if (lid == 0) {
        for (int k = 0; k < ADM_METAL_N_DEN; k++) {
            den_rows[(p.den_row_offset + i) * ADM_METAL_N_DEN + k] = den_row ?
                (den[k] + p.den_add_row) >> p.den_shift_row : 0;
        }
    }
}

static inline int load_band(device const uchar *dec, uint idx)
{
    if (kS0) return ((device const short *)dec)[idx];
    return ((device const int *)dec)[idx];
}

/** adm_csf(), i4_adm_csf(): csf_a */
static inline int csf_dst(constant AdmMetalParams &p, int v, int theta)
{
    if (kS0) {
        const int add = theta == 2 ? 65535 : 16384;
        const int shift = theta == 2 ? 17 : 15;
        return (short)(((int)p.i_rfactor[theta] * v + add) >> shift);
    }
    return (int)(((long)p.i_rfactor[theta] * v + (1 << 27)) >> 28);
}

/** adm_csf(), i4_adm_csf(): csf_f */
static inline int csf_flt(int dst)
{
    if (kS0) return (short)((4369 * iabs(dst) + 2048) >> 12);
    return (int)(((long)143165577 * iabs(dst) + (long)INT32_MIN_) >> 32);
}

/** center term of ADM_CM_THRESH_* */
static inline int csf_center(int dst)
{
    if (kS0) return (short)((8738 * iabs(dst) + 2048) >> 12);
    return (int)(((long)286331153 * iabs(dst) + (long)INT32_MIN_) >> 32);
}

/**
 * Masking threshold at (i, j) from the 3x3 neighborhood of the three
 * orientations of plane `first` to `first + 2`, ADM_CM_THRESH_*.
 */
static inline int cm_threshold(constant AdmMetalParams &p,
                               device const uchar *dec, uint first,
                               thread const int *rows, thread const int *cols)
{
    int thr = 0;
    for (int theta = 0; theta < 3; theta++) {
        const uint base = (first + theta) * p.plane;
        int sum = 0;
        for (int y = 0; y < 3; y++) {
            for (int x = 0; x < 3; x++) {
                const int v = load_band(dec, base + rows[y] * p.stride + cols[x]);
                const int dst = csf_dst(p, v, theta);
                sum += (y == 1 && x == 1) ? csf_center(dst) : csf_flt(dst);
            }
        }
        thr += sum;
    }
    return thr;
}

/** ADM_CM_ACCUM_ROUND, I4_ADM_CM_ACCUM_ROUND */
static inline long cm_value(constant AdmMetalParams &p, int src, int thr,
                            int theta)
{
    int x;
    long x_sq;
    if (kS0) {
        x = src * (int)p.i_rfactor[theta];
        x = iabs(x) - (thr << (theta == 2 ? 12 : 10));
        x = max(x, 0);
        x_sq = (int)((((long)x * x) + (theta == 2 ? 536870912 : 268435456)) >>
                     (theta == 2 ? 30 : 29));
    } else {
        x = (int)((((long)src * p.i_rfactor[theta]) + (1 << 27)) >> 28);
        x = iabs(x) - thr;
        x = max(x, 0);
        x_sq = (int)((((long)x * x) + 536870912) >> 30);
    }
    return ((x_sq * x) + p.cm_add_cub[theta]) >> p.cm_shift_cub[theta];
}

kernel void adm_metal_cm(constant AdmMetalParams &p [[buffer(0)]],
                         device const uchar *dec [[buffer(1)]],
                         device long *cm_rows [[buffer(2)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint lid [[thread_index_in_threadgroup]],
                         uint tg_w [[threads_per_threadgroup]],
                         uint simd_lane [[thread_index_in_simdgroup]],
                         uint simd_group [[simdgroup_index_in_threadgroup]],
                         uint n_simd_groups [[simdgroups_per_threadgroup]])
{
    const int i = p.top + tg;
    const int rows[3] = { i == 0 ? 1 : i - 1, i, i == p.h - 1 ? p.h - 1 : i + 1 };
    long acc[ADM_METAL_N_CM] = { 0, 0, 0, 0, 0, 0 };

    for (int j = p.left + lid; j < p.right; j += tg_w) {
        const int cols[3] = { j == 0 ? 1 : j - 1, j, j == p.w - 1 ? p.w - 1 : j + 1 };
        const uint idx = i * p.stride + j;

        // DLM: restored image masked by the additive impairment
        const int thr_dlm = cm_threshold(p, dec, 3, rows, cols);
        for (int theta = 0; theta < 3; theta++) {
            const int r = load_band(dec, theta * p.plane + idx);
            acc[theta] += cm_value(p, r, thr_dlm, theta);
        }
        if (kSkipAim) continue;

        // AIM: additive impairment masked by the restored image
        const int thr_aim = cm_threshold(p, dec, 0, rows, cols);
        for (int theta = 0; theta < 3; theta++) {
            const int a = load_band(dec, (3 + theta) * p.plane + idx);
            acc[3 + theta] += cm_value(p, a, thr_aim, theta);
        }
    }

    threadgroup long sh[ADM_METAL_MAX_TG_W / 32 * ADM_METAL_N_CM];
    tg_sum<long, ADM_METAL_N_CM>(acc, sh, lid, simd_lane, simd_group,
                                 n_simd_groups);
    if (lid == 0) {
        for (int k = 0; k < ADM_METAL_N_CM; k++) {
            cm_rows[(p.cm_row_offset + i) * ADM_METAL_N_CM + k] =
                (acc[k] + p.cm_add_row) >> p.cm_shift_row;
        }
    }
}
