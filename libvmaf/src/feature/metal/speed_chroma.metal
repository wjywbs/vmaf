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
 * SpEED chroma: the full resolution part of filter_and_downscale() in
 * speed.c, for the 4 chroma planes of a frame.
 *
 * picture_copy() converts a plane to float, minus 128. Then
 * vif_filter1d_dec16_scalar_s() filters it with the anti-aliasing filter,
 * at every 16th row and column only. One thread computes one output pixel:
 * the vertical filter at the fwidth columns that the horizontal filter
 * reads, then the horizontal filter. The CPU build contracts
 * `accum += f * x` into a fused multiply-add, which fma() reproduces.
 *
 * With a prescale, speed_chroma_metal_scale first converts and scales the
 * planes to float, as vif_scale_frame_s() does, and the filter reads them.
 *
 * kVariant = bpc (8, 10, 12 or 16) of the input, or 32 for float input.
 * The rest of the computation runs on the host, on 1/256 of the pixels.
 * kFast is ignored.
 */

#include "kernel_common.h"
#include "speed_chroma_metal.h"

constant uint kBpc = kVariant;

/** picture_copy() of one pixel */
static inline float load_px(device const uchar *in, constant SpeedMetalParams &p,
                            uint idx)
{
    if (kBpc == 32) return ((device const float *)in)[idx];
    if (kBpc == 8) return (float)in[idx] + (float)-128;
    const float v = (float)((device const ushort *)in)[idx];
    return v / p.scaler + (float)-128;
}

static inline int mirror(int i, int n)
{
    return i < 0 ? -i : (i >= n ? 2 * n - i - 2 : i);
}

/**
 * vif_scale_frame_nearest_s() and vif_scale_frame_bilinear_apply_s(): the
 * host computes the source rows and columns and the weights, with the CPU
 * functions. One thread per output pixel, p.w x p.h per plane.
 *
 * xi: first and second source column (2 x p.w), xf: weight of the second
 * yi, yf: the same for the rows (2 x p.h). nearest only uses xi and yi.
 */
kernel void speed_chroma_metal_scale(constant SpeedMetalParams &p [[buffer(0)]],
                                     device const uchar *in [[buffer(1)]],
                                     device const int *xi [[buffer(2)]],
                                     device const float *xf [[buffer(3)]],
                                     device const int *yi [[buffer(4)]],
                                     device const float *yf [[buffer(5)]],
                                     device float *out [[buffer(6)]],
                                     uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x;
    const int plane = gid.y / p.h, y = gid.y % p.h;
    if (x >= p.w || plane >= SPEED_METAL_PLANES) return;

    const uint base = plane * p.in_plane;
    const uint r1 = base + yi[y] * p.in_stride;
    float v;
    if (p.method == SPEED_METAL_SCALE_NEAREST) {
        v = load_px(in, p, r1 + xi[x]);
    } else {
        // the CPU build's contraction of the 4 products
        const uint r2 = base + yi[p.h + y] * p.in_stride;
        const int x1 = xi[x], x2 = xi[p.w + x];
        const float dx = xf[x], dy = yf[y];
        float acc = (1 - dy) * dx * load_px(in, p, r1 + x2);
        acc = fma((1 - dy) * (1 - dx), load_px(in, p, r1 + x1), acc);
        acc = fma(dy * (1 - dx), load_px(in, p, r2 + x1), acc);
        v = fma(dy * dx, load_px(in, p, r2 + x2), acc);
    }
    out[(plane * p.h + y) * p.w + x] = v;
}

kernel void speed_chroma_metal_dec16(constant SpeedMetalParams &p [[buffer(0)]],
                                     device const uchar *in [[buffer(1)]],
                                     constant float *f [[buffer(2)]],
                                     device float *out [[buffer(3)]],
                                     uint2 gid [[thread_position_in_grid]])
{
    const int j = gid.x;
    const int plane = gid.y / p.out_h, i = gid.y % p.out_h;
    if (j >= p.out_w || plane >= SPEED_METAL_PLANES) return;

    const uint base = plane * p.in_plane;
    const int r = p.fwidth / 2;
    float accum = 0;
    for (int fj = 0; fj < p.fwidth; fj++) {
        const int jj = mirror(j * 16 - r + fj, p.w);
        float tmp = 0;
        for (int fi = 0; fi < p.fwidth; fi++) {
            const int ii = mirror(i * 16 - r + fi, p.h);
            tmp = fma(f[fi], load_px(in, p, base + ii * p.in_stride + jj), tmp);
        }
        accum = fma(f[fj], tmp, accum);
    }
    out[(plane * p.out_h + i) * p.out_w + j] = accum;
}
