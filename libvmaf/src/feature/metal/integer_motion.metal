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
 * Integer motion, bit-exact port of motion_score_pipeline_8/16() in
 * integer_motion.c: the SAD of the 5-tap blurred difference prev - cur.
 *
 * One threadgroup computes one row. The CPU sums each row in uint32_t, which
 * is reproduced here; the host adds the row sums in uint64_t. The row is
 * processed in chunks of the threadgroup width: the vertical filter of the
 * difference goes into threadgroup memory, with a 2 column halo, then the
 * horizontal filter reads it.
 *
 * kVariant = bpc. Integer only, kFast is ignored.
 */

#include "kernel_common.h"
#include "integer_motion_metal.h"

constant uint kBpc = kVariant & 0xff;

constant int motion_filter[5] = { 3571, 16004, 26386, 16004, 3571 };

static inline int mirror(int idx, int size)
{
    if (idx < 0) return -idx;
    if (idx >= size) return 2 * size - idx - 2;
    return idx;
}

static inline int load(device const uchar *src, int idx)
{
    if (kBpc == 8) return src[idx];
    return ((device const ushort *)src)[idx];
}

/** Vertical filter of the difference at column j, rounded as the CPU. */
static inline int vertical(device const uchar *prev, device const uchar *cur,
                           thread const int *prev_row, thread const int *cur_row,
                           int j)
{
    if (kBpc <= 15) {
        // |diff| < 2^15, the taps sum to 65536: fits int32
        int accum = 0;
        for (int k = 0; k < 5; k++)
            accum += motion_filter[k] * (load(prev, prev_row[k] + j) - load(cur, cur_row[k] + j));
        const int shift = kBpc == 8 ? 8 : kBpc;
        return (accum + (1 << (shift - 1))) >> shift;
    }
    long accum = 0;
    for (int k = 0; k < 5; k++)
        accum += (long)motion_filter[k] * (load(prev, prev_row[k] + j) - load(cur, cur_row[k] + j));
    return (int)((accum + (1 << (kBpc - 1))) >> kBpc);
}

kernel void motion_metal_sad(constant MotionMetalParams &p [[buffer(0)]],
                             device const uchar *prev [[buffer(1)]],
                             device const uchar *cur [[buffer(2)]],
                             device uint *row_sad [[buffer(3)]],
                             uint i [[threadgroup_position_in_grid]],
                             uint lid [[thread_position_in_threadgroup]],
                             uint tg_w [[threads_per_threadgroup]],
                             uint sid [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]])
{
    threadgroup int y_row[MOTION_METAL_MAX_TG_W + 4];
    threadgroup uint sh[MOTION_METAL_MAX_TG_W / 32];

    int prev_row[5], cur_row[5];
    for (int k = 0; k < 5; k++) {
        const int row = mirror((int)i - 2 + k, p.h);
        prev_row[k] = row * p.prev_stride;
        cur_row[k] = row * p.cur_stride;
    }

    uint sad = 0; // wraps as the CPU's uint32_t row_sad
    for (int c0 = 0; c0 < p.w; c0 += tg_w) {
        for (uint t = lid; t < tg_w + 4; t += tg_w) {
            const int col = c0 - 2 + (int)t;
            // columns past w + 1 are not read by any output column
            if (col <= p.w + 1)
                y_row[t] = vertical(prev, cur, prev_row, cur_row, mirror(col, p.w));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (c0 + (int)lid < p.w) {
            long accum = 0;
            for (int k = 0; k < 5; k++)
                accum += (long)motion_filter[k] * y_row[lid + k];
            sad += (uint)abs((int)((accum + (1 << 15)) >> 16));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    sad = simd_sum(sad);
    if (lane == 0) sh[sid] = sad;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        uint total = 0;
        for (uint s = 0; s < (tg_w + 31) / 32; s++) total += sh[s];
        row_sad[i] = total;
    }
}
