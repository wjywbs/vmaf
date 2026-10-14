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
 * CAMBI, bit-exact port of cambi_score() in cambi.c.
 *
 * The CPU code works in place on one picture and updates histograms as it
 * slides down the rows. Here each step is a pure function of the previous
 * one, so it can run per pixel:
 *
 *   cambi_metal_preprocess  cambi_preprocessing(): convert to 10 bits,
 *                           resample, anti-dithering filter
 *   cambi_metal_mask_h/v    get_spatial_mask(): 7x7 sum of the zero
 *                           derivative, compared with the mask index
 *   cambi_metal_mode        decimate(), then filter_mode(), per scale
 *   cambi_metal_cvalues     calculate_c_values(), per scale
 *   cambi_metal_sel_*       spatial_pooling(): the sum of the top k c-values
 *
 * All of it is integer, except one float product per c-value, which the GPU
 * rounds as the CPU. The kernels ignore kFast; fast mode only skips a rare
 * host fallback of the pooling (see pool() in cambi_metal.m).
 */

#include "kernel_common.h"
#include "cambi_metal.h"

static inline ushort mode3(ushort a, ushort b, ushort c)
{
    if (a == b || a == c) return a;
    if (b == c) return b;
    return min(a, min(b, c));
}

/* -------------------------------------------------------------------------
 * Preprocessing. kVariant = bpc of the input.
 */

constant uint kBpc = kVariant & 0xff;

static inline uint load_in(device const uchar *in, int idx)
{
    if (kBpc <= 8) return in[idx];
    return ((device const ushort *)in)[idx];
}

/** decimate_generic_*_and_convert_to_10b() for one output pixel */
static inline uint convert(device const uchar *in,
                           constant CambiMetalPreParams &p,
                           device const int *ori_x, device const int *ori_y,
                           int y, int x)
{
    const int sy = p.resample ? ori_y[y] : y;
    const int sx = p.resample ? ori_x[x] : x;
    const uint v = load_in(in, sy * p.in_stride + sx);
    if (kBpc <= 8) return (v << (10 - kBpc)) & 0xffff;
    if (kBpc == 9) return (v << 1) & 0xffff;
    const uint shift = kBpc - 10;
    const uint rounding = shift == 0 ? 0 : 1u << (shift - 1);
    return ((v + rounding) >> shift) & 0xffff;
}

kernel void cambi_metal_preprocess(constant CambiMetalPreParams &p [[buffer(0)]],
                                   device const uchar *in [[buffer(1)]],
                                   device const int *ori_x [[buffer(2)]],
                                   device const int *ori_y [[buffer(3)]],
                                   device ushort *out [[buffer(4)]],
                                   uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y;
    if (x >= p.w || y >= p.h) return;

    uint v = convert(in, p, ori_x, ori_y, y, x);
    if (p.anti_dither) {
        // anti_dithering_filter(): in place, but every output only reads
        // pixels that have not been written yet
        const bool last_col = x == p.w - 1, last_row = y == p.h - 1;
        if (!last_row && !last_col) {
            v = (v + convert(in, p, ori_x, ori_y, y, x + 1) +
                 convert(in, p, ori_x, ori_y, y + 1, x) +
                 convert(in, p, ori_x, ori_y, y + 1, x + 1)) >> 2;
        } else if (!last_row) {
            v = (v + convert(in, p, ori_x, ori_y, y + 1, x)) >> 1;
        } else if (!last_col) {
            v = (v + convert(in, p, ori_x, ori_y, y, x + 1)) >> 1;
        }
    }
    out[y * p.w + x] = v;
}

/** validate_image(): flag values above the bit depth */
kernel void cambi_metal_validate(constant CambiMetalValidateParams &p [[buffer(0)]],
                                 device const uchar *in [[buffer(1)]],
                                 device atomic_uint *invalid [[buffer(2)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y;
    bool bad = false;
    if (x < p.w && y < p.h) {
        if (kBpc <= 8) bad = in[y * p.stride + x] > (uchar)p.max_val;
        else bad = load_in(in, y * p.stride + x) > (uint)p.max_val;
    }
    if (simd_any(bad) && simd_is_first())
        atomic_store_explicit(invalid, 1u, memory_order_relaxed);
}

/* -------------------------------------------------------------------------
 * Spatial mask
 */

static inline uint zero_deriv(device const ushort *img, int w, int h,
                              int y, int x)
{
    const ushort v = img[y * w + x];
    const bool horizontal = x == w - 1 || v == img[y * w + x + 1];
    const bool vertical = y == h - 1 || v == img[(y + 1) * w + x];
    return horizontal && vertical;
}

/** Horizontal 7-sum of the zero derivative, columns past the edges are 0. */
kernel void cambi_metal_mask_h(constant CambiMetalMaskParams &p [[buffer(0)]],
                               device const ushort *img [[buffer(1)]],
                               device uchar *hsum [[buffer(2)]],
                               uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y;
    if (x >= p.w || y >= p.h) return;
    uint sum = 0;
    for (int c = max(x - 3, 0); c <= min(x + 3, p.w - 1); c++)
        sum += zero_deriv(img, p.w, p.h, y, c);
    hsum[y * p.w + x] = sum;
}

/** Vertical 7-sum, compared with the mask index. */
kernel void cambi_metal_mask_v(constant CambiMetalMaskParams &p [[buffer(0)]],
                               device const uchar *hsum [[buffer(1)]],
                               device uchar *mask [[buffer(2)]],
                               uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y;
    if (x >= p.w || y >= p.h) return;
    int sum = 0;
    for (int r = max(y - 3, 0); r <= min(y + 3, p.h - 1); r++)
        sum += hsum[r * p.w + x];
    mask[y * p.w + x] = sum > p.mask_index;
}

/* -------------------------------------------------------------------------
 * decimate() and filter_mode()
 */

/**
 * filter_mode() of the source picture, decimated by src_step. The CPU
 * filters in place, but only rows that it no longer reads: the first and
 * last rows keep their values, the other rows are the vertical mode of the
 * horizontal modes, where the first and last columns keep their values.
 */
kernel void cambi_metal_mode(constant CambiMetalModeParams &p [[buffer(0)]],
                             device const ushort *src [[buffer(1)]],
                             device ushort *dst [[buffer(2)]],
                             uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y;
    if (x >= p.w || y >= p.h) return;

    device const ushort *col = src + x * p.src_step;
    const int row = p.src_stride * p.src_step;
    if (y == 0 || y == p.h - 1) {
        dst[y * p.w + x] = col[y * row];
        return;
    }

    ushort hm[3];
    for (int k = 0; k < 3; k++) {
        device const ushort *s = col + (y - 1 + k) * row;
        hm[k] = x == 0 || x == p.w - 1
              ? s[0]
              : mode3(s[-p.src_step], s[0], s[p.src_step]);
    }
    dst[y * p.w + x] = mode3(hm[0], hm[1], hm[2]);
}

/* -------------------------------------------------------------------------
 * c-values
 *
 * The CPU keeps, for every column, a histogram of the values of the masked
 * pixels in the window around it, and slides it down the rows. Here one
 * thread does the same for one column and a chunk of rows, with the
 * histogram in device memory: column c of hist[value][column] of its chunk.
 * Each step adds the row entering the window and subtracts the row leaving
 * it. With `clean`, it subtracts the last window at the end, so the
 * histograms are zero again for the next scale or frame. Otherwise, the host
 * zeroes them before the next use. Pixels that are not counted have the key
 * CAMBI_METAL_KEY_NONE:
 *
 *   key = value if the pixel is masked and its value in the histogram band
 */

kernel void cambi_metal_keys(constant CambiMetalCValueParams &p [[buffer(0)]],
                             device const ushort *img [[buffer(1)]],
                             device const uchar *mask [[buffer(2)]],
                             device ushort *keys [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y;
    if (x >= p.w || y >= p.h) return;
    ushort key = CAMBI_METAL_KEY_NONE;
    if (mask[(y << p.mask_shift) * p.mask_stride + (x << p.mask_shift)]) {
        const ushort v = img[y * p.w + x];
        if ((ushort)(v - p.v_band_base) < p.v_band_size) key = v;
    }
    keys[y * p.w + x] = key;
}

kernel void cambi_metal_zero(device uint4 *buf [[buffer(0)]],
                             constant uint &n [[buffer(1)]],
                             uint gid [[thread_position_in_grid]])
{
    if (gid < n) buf[gid] = 0;
}

/**
 * Add `delta` times the keys of columns jlo..jhi of `row` to the histogram
 * of one column. Runs of equal keys are added at once.
 */
static inline void add_row(device const ushort *row, int jlo, int jhi,
                           device ushort *hc, int w, int base, int delta)
{
    int cur = row[jlo], n = 1;
    for (int j = jlo + 1; j <= jhi; j++) {
        const int k = row[j];
        if (k == cur) {
            n++;
            continue;
        }
        if (cur != CAMBI_METAL_KEY_NONE) hc[(cur - base) * w] += delta * n;
        cur = k;
        n = 1;
    }
    if (cur != CAMBI_METAL_KEY_NONE) hc[(cur - base) * w] += delta * n;
}

kernel void cambi_metal_cvalues(constant CambiMetalCValueParams &p [[buffer(0)]],
                                device const ushort *keys [[buffer(1)]],
                                device ushort *hist [[buffer(2)]],
                                constant float *reciprocal_lut [[buffer(3)]],
                                device float *c_values [[buffer(4)]],
                                uint2 gid [[thread_position_in_grid]])
{
    const int c = gid.x, w = p.w, h = p.h, pad = p.pad;
    const int r0 = gid.y * p.chunk_rows;
    if (c >= w || r0 >= h) return;
    const int r1 = min(r0 + p.chunk_rows, h);
    const int base = p.v_band_base;
    const int nd = p.num_diffs;
    device ushort *hc = hist + gid.y * p.v_band_size * w + c;
    const int jlo = max(c - pad, 0), jhi = min(c + pad, w - 1);

    for (int r = max(r0 - pad, 0); r <= min(r0 + pad, h - 1); r++)
        add_row(keys + r * w, jlo, jhi, hc, w, base, 1);

    for (int i = r0; i < r1; i++) {
        const int key = keys[i * w + c];
        const int value = key + nd;
        float c_value = 0.0f;

        if (key != CAMBI_METAL_KEY_NONE) {
            // c_value_pixel()
            const uint p_0 = hc[(key - base) * w];
            for (int d = 0; d < nd; d++) {
                if (value <= p.tvi[d] && value + d + 1 > p.vlt_luma) {
                    const uint p_1 = hc[(key + d + 1 - base) * w];
                    const int idx2 = key - d - 1 - base;
                    const uint p_2 = idx2 >= 0 ? hc[idx2 * w] : 0;
                    float val;
                    if (p_1 > p_2) {
                        val = (float)(p.weights[d] * p_0 * p_1) *
                              reciprocal_lut[p_1 + p_0];
                    } else {
                        val = (float)(p.weights[d] * p_0 * p_2) *
                              reciprocal_lut[p_2 + p_0];
                    }
                    if (val > c_value) c_value = val;
                }
            }
        }
        c_values[p.c_offset + i * w + c] = c_value;

        if (i + 1 < r1) {
            if (i - pad >= 0)
                add_row(keys + (i - pad) * w, jlo, jhi, hc, w, base, -1);
            if (i + pad + 1 < h)
                add_row(keys + (i + pad + 1) * w, jlo, jhi, hc, w, base, 1);
        }
    }

    if (!p.clean) return;
    for (int r = max(r1 - 1 - pad, 0); r <= min(r1 - 1 + pad, h - 1); r++)
        add_row(keys + r * w, jlo, jhi, hc, w, base, -1);
}

/* -------------------------------------------------------------------------
 * Top k selection
 *
 * spatial_pooling() averages the k largest c-values. c-values are 0 or at
 * least 0.5, so their float bits order them. A radix select finds the k-th
 * largest value t in 3 levels:
 *   1: bin of (bits - 0.5f) >> 16, plus bin 0 for 0
 *   2: bits 8..15
 *   3: bits 0..7
 * Then the sum of the top k is the sum of the values > t, plus t for the
 * remaining ranks. Segments (one scale of one picture) are independent:
 * grid y, or one threadgroup each.
 */

static inline uint l1_bin(uint bits)
{
    if (bits == 0) return 0;
    return min(((bits - CAMBI_METAL_L1_BASE) >> 16) + 1,
               (uint)CAMBI_METAL_L1_BINS - 1);
}

/**
 * Add 1 to bin `bin` for the active lanes. Most c-values fall in a few bins,
 * so the lanes of a SIMD group with the same bin add their count at once.
 */
static inline void hist_add(threadgroup atomic_uint *local, uint bin,
                            bool active)
{
    while (simd_any(active)) {
        const uint lb = simd_min(active ? bin : 0xffffffffu);
        const bool match = active && bin == lb;
        const uint n = simd_sum((uint)match);
        if (match && simd_prefix_exclusive_sum((uint)match) == 0)
            atomic_fetch_add_explicit(&local[lb], n, memory_order_relaxed);
        active = active && !match;
    }
}

kernel void cambi_metal_sel_hist(constant CambiMetalSelParams &p [[buffer(0)]],
                                 constant CambiMetalSeg *segs [[buffer(1)]],
                                 device const uint *c_bits [[buffer(2)]],
                                 device const CambiMetalSel *sel [[buffer(3)]],
                                 device atomic_uint *hist [[buffer(4)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint lid [[thread_index_in_threadgroup]])
{
    threadgroup atomic_uint local[CAMBI_METAL_L1_BINS];
    const uint seg = tg.y;
    const CambiMetalSel st = sel[seg];
    if (p.level > 1 && st.done) return;

    const uint n_bins = p.level == 1 ? CAMBI_METAL_L1_BINS : 256;
    for (uint b = lid; b < n_bins; b += CAMBI_METAL_SEL_THREADS)
        atomic_store_explicit(&local[b], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const CambiMetalSeg s = segs[seg];
    device const uint *v = c_bits + s.offset;
    for (uint i = tg.x * CAMBI_METAL_SEL_THREADS + lid; i < s.n;
         i += CAMBI_METAL_SEL_TGS * CAMBI_METAL_SEL_THREADS) {
        const uint bits = v[i];
        uint bin;
        bool active;
        if (p.level == 1) {
            bin = l1_bin(bits);
            active = true;
        } else if (p.level == 2) {
            bin = (bits >> 8) & 0xff;
            active = l1_bin(bits) == st.bin1;
        } else {
            bin = bits & 0xff;
            active = bits >> 8 == st.prefix;
        }
        hist_add(local, bin, active);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device atomic_uint *h = hist + seg * CAMBI_METAL_L1_BINS;
    for (uint b = lid; b < n_bins; b += CAMBI_METAL_SEL_THREADS) {
        const uint c = atomic_load_explicit(&local[b], memory_order_relaxed);
        if (c) atomic_fetch_add_explicit(&h[b], c, memory_order_relaxed);
    }
}

/** Find the bin of the k-th largest value, one threadgroup per segment. */
kernel void cambi_metal_sel_find(constant CambiMetalSelParams &p [[buffer(0)]],
                                 constant CambiMetalSeg *segs [[buffer(1)]],
                                 device const uint *hist [[buffer(2)]],
                                 device CambiMetalSel *sel [[buffer(3)]],
                                 uint seg [[threadgroup_position_in_grid]],
                                 uint lid [[thread_index_in_threadgroup]])
{
    threadgroup uint chunk[CAMBI_METAL_SEL_THREADS];
    device CambiMetalSel &st = sel[seg];
    if (p.level > 1 && st.done) return;

    const uint n_bins = p.level == 1 ? CAMBI_METAL_L1_BINS : 256;
    const uint per = n_bins / CAMBI_METAL_SEL_THREADS;
    device const uint *h = hist + seg * CAMBI_METAL_L1_BINS;

    // thread 0 has the highest bins
    const uint top = n_bins - lid * per; // exclusive
    uint sum = 0;
    for (uint b = top - per; b < top; b++) sum += h[b];
    chunk[lid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid) return;

    const uint kk = p.level == 1 ? segs[seg].k : st.kk;
    uint above = 0, t = 0;
    while (t < CAMBI_METAL_SEL_THREADS - 1 && above + chunk[t] < kk)
        above += chunk[t++];
    uint b = n_bins - t * per - 1;
    for (uint i = 0; i < per - 1 && above + h[b] < kk; i++)
        above += h[b--];

    st.kk = kk - above;
    if (p.level == 1) {
        st.bin1 = b;
        st.done = b == 0;
        st.t_bits = 0;
    } else if (p.level == 2) {
        st.prefix = ((CAMBI_METAL_L1_BASE >> 8) + ((st.bin1 - 1) << 8)) | b;
    } else {
        st.t_bits = st.prefix << 8 | b;
        st.done = 1;
    }
}

/** c-values above the threshold: count, and sum in units of 2^-24. */
kernel void cambi_metal_sel_sum(constant CambiMetalSeg *segs [[buffer(1)]],
                                device const uint *c_bits [[buffer(2)]],
                                device const CambiMetalSel *sel [[buffer(3)]],
                                device cambi_metal_u64 *sums [[buffer(4)]],
                                device uint *cnts [[buffer(5)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint lid [[thread_index_in_threadgroup]])
{
    threadgroup ulong ts[CAMBI_METAL_SEL_THREADS];
    threadgroup uint tc[CAMBI_METAL_SEL_THREADS];
    const uint seg = tg.y;
    const uint t_bits = sel[seg].t_bits;
    const CambiMetalSeg s = segs[seg];
    device const uint *v = c_bits + s.offset;

    ulong sum = 0;
    uint cnt = 0;
    for (uint i = tg.x * CAMBI_METAL_SEL_THREADS + lid; i < s.n;
         i += CAMBI_METAL_SEL_TGS * CAMBI_METAL_SEL_THREADS) {
        const uint bits = v[i];
        if (bits > t_bits) {
            // bits > 0, so the value is >= 0.5: exponent >= -1
            const ulong m = (bits & 0x7fffff) | 0x800000;
            const int e = (int)(bits >> 23) - 127;
            sum += m << (e + 1);
            cnt++;
        }
    }
    ts[lid] = sum;
    tc[lid] = cnt;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint n = CAMBI_METAL_SEL_THREADS / 2; n > 0; n >>= 1) {
        if (lid < n) {
            ts[lid] += ts[lid + n];
            tc[lid] += tc[lid + n];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        sums[seg * CAMBI_METAL_SEL_TGS + tg.x] = ts[0];
        cnts[seg * CAMBI_METAL_SEL_TGS + tg.x] = tc[0];
    }
}
