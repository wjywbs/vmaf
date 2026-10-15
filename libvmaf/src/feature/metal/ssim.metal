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
 * float_ssim and float_ms_ssim: the image processing of compute_ssim() and
 * compute_ms_ssim() (ssim.c, ms_ssim.c and iqa/), per scale:
 *
 * ssim_metal_load       picture_copy() of ref and dist
 * ssim_metal_decimate   _iqa_decimate(): SSIM's box filter, MS-SSIM's 9x9
 *                       low-pass filter
 * ssim_metal_conv_h     _iqa_convolve(), horizontal pass, of the 5 maps
 * ssim_metal_conv_v     vertical pass, then the per pixel part of
 *                       _iqa_ssim()
 *
 * Exact mode (kFast = false): the CPU sums the products of every filter in
 * double and rounds the sum to float; SsimSum (ssim_metal.h) reproduces
 * this, mostly in float-float. The per pixel l and c are double quotients,
 * computed with soft_f64.h, and written out with s; the host adds them up
 * in the CPU's order. Fast mode filters in float and adds l, c and s up per threadgroup.
 *
 * kVariant = bpc of ssim_metal_load (8, 10, 12 or 16).
 */

#include "kernel_common.h"
#include "ssim_metal.h"

constant uint kBpc = kVariant;

/** KBND_SYMMETRIC() */
static inline int mirror(int i, int n)
{
    return i < 0 ? -1 - i : (i >= n ? 2 * n - i - 1 : i);
}

kernel void ssim_metal_load(constant SsimMetalLoadParams &p [[buffer(0)]],
                            device const uchar *in [[buffer(1)]],
                            device float *out [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, plane = gid.y / p.h, y = gid.y % p.h;
    if (x >= p.w || plane >= 2) return;
    const uint idx = (plane * p.h + y) * p.in_stride + x;
    float v;
    if (kBpc == 8) v = (float)in[idx];
    else v = (float)((device const ushort *)in)[idx] / p.scaler;
    out[(plane * p.h + y) * p.w + x] = v;
}

kernel void ssim_metal_decimate(constant SsimMetalDecParams &p [[buffer(0)]],
                                device const float *in [[buffer(1)]],
                                constant float *k [[buffer(2)]],
                                device float *out [[buffer(3)]],
                                uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, plane = gid.y / p.out_h, y = gid.y % p.out_h;
    if (x >= p.out_w || plane >= 2) return;
    device const float *img = in + plane * p.w * p.h;
    const int cx = x * p.factor, cy = y * p.factor;
    const int uc = p.kw / 2, vc = p.kh / 2;
    const int u1 = uc - !(p.kw & 1), v1 = vc - !(p.kh & 1);

    float r;
    if (kFast) {
        r = 0.0f;
        int ko = 0;
        for (int v = -vc; v <= v1; v++) {
            device const float *row = img + mirror(cy + v, p.h) * p.w;
            for (int u = -uc; u <= u1; u++, ko++)
                r = fma(row[mirror(cx + u, p.w)], k[ko], r);
        }
    } else {
        SsimSum s;
        ssim_sum_init(&s);
        int ko = 0;
        for (int v = -vc; v <= v1; v++) {
            device const float *row = img + mirror(cy + v, p.h) * p.w;
            for (int u = -uc; u <= u1; u++, ko++)
                ssim_sum_add(&s, row[mirror(cx + u, p.w)] * k[ko]);
        }
        bool ok;
        r = ssim_sum_result(&s, ssim_sum_eps(p.kw * p.kh), &ok);
        if (!ok) {
            sf64 sum = sf_zero();
            ko = 0;
            for (int v = -vc; v <= v1; v++) {
                device const float *row = img + mirror(cy + v, p.h) * p.w;
                for (int u = -uc; u <= u1; u++, ko++)
                    sum = ssim_sum_slow(sum, row[mirror(cx + u, p.w)] * k[ko]);
            }
            r = ssim_sf_to_float(sum);
        }
    }
    out[(plane * p.out_h + y) * p.out_w + x] = r;
}

/** The 5 products of the maps, as the CPU computes them. */
static inline void map_terms(float a, float b, float k, thread float *t)
{
    t[0] = a * k;
    t[1] = b * k;
    t[2] = (a * a) * k;
    t[3] = (b * b) * k;
    t[4] = (a * b) * k;
}

kernel void ssim_metal_conv_h(constant SsimMetalConvParams &p [[buffer(0)]],
                              device const float *in [[buffer(1)]],
                              constant float *k [[buffer(2)]],
                              device float *out [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x, y = gid.y;
    if (x >= p.dst_w || y >= p.h) return;
    device const float *ra = in + y * p.w + x;
    device const float *rb = ra + p.w * p.h;
    const uint plane = p.dst_w * p.h;
    device float *o = out + y * p.dst_w + x;

    float t[SSIM_METAL_MAPS];
    if (kFast) {
        float r[SSIM_METAL_MAPS] = { 0 };
        for (int u = 0; u < SSIM_METAL_GAUSSIAN_LEN; u++) {
            const float a = ra[u], b = rb[u];
            r[0] = fma(a, k[u], r[0]);
            r[1] = fma(b, k[u], r[1]);
            r[2] = fma(a * a, k[u], r[2]);
            r[3] = fma(b * b, k[u], r[3]);
            r[4] = fma(a * b, k[u], r[4]);
        }
        for (int m = 0; m < SSIM_METAL_MAPS; m++) o[m * plane] = r[m];
        return;
    }

    SsimSum s[SSIM_METAL_MAPS];
    for (int m = 0; m < SSIM_METAL_MAPS; m++) ssim_sum_init(&s[m]);
    for (int u = 0; u < SSIM_METAL_GAUSSIAN_LEN; u++) {
        map_terms(ra[u], rb[u], k[u], t);
        for (int m = 0; m < SSIM_METAL_MAPS; m++) ssim_sum_add(&s[m], t[m]);
    }
    const float eps_k = ssim_sum_eps(SSIM_METAL_GAUSSIAN_LEN);
    for (int m = 0; m < SSIM_METAL_MAPS; m++) {
        bool ok;
        float r = ssim_sum_result(&s[m], eps_k, &ok);
        if (!ok) {
            sf64 sum = sf_zero();
            for (int u = 0; u < SSIM_METAL_GAUSSIAN_LEN; u++) {
                map_terms(ra[u], rb[u], k[u], t);
                sum = ssim_sum_slow(sum, t[m]);
            }
            r = ssim_sf_to_float(sum);
        }
        o[m * plane] = r;
    }
}

kernel void ssim_metal_conv_v(constant SsimMetalConvParams &p [[buffer(0)]],
                              device const float *in [[buffer(1)]],
                              constant float *k [[buffer(2)]],
                              device ulong *out_l [[buffer(3)]],
                              device ulong *out_c [[buffer(4)]],
                              device float *out_s [[buffer(5)]],
                              device float4 *part [[buffer(6)]],
                              uint2 gid [[thread_position_in_grid]],
                              uint2 tgid [[threadgroup_position_in_grid]],
                              uint2 tgs [[threadgroups_per_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint2 tg_size [[threads_per_threadgroup]],
                              uint sid [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]])
{
    const int x = gid.x, y = gid.y;
    const bool valid = x < p.dst_w && y < p.dst_h;
    const uint plane = p.dst_w * p.h;
    device const float *col = in + y * p.dst_w + x;

    float r[SSIM_METAL_MAPS] = { 0 };
    if (valid && kFast) {
        for (int v = 0; v < SSIM_METAL_GAUSSIAN_LEN; v++) {
            for (int m = 0; m < SSIM_METAL_MAPS; m++)
                r[m] = fma(col[m * plane + v * p.dst_w], k[v], r[m]);
        }
    } else if (valid) {
        SsimSum s[SSIM_METAL_MAPS];
        for (int m = 0; m < SSIM_METAL_MAPS; m++) ssim_sum_init(&s[m]);
        for (int v = 0; v < SSIM_METAL_GAUSSIAN_LEN; v++) {
            for (int m = 0; m < SSIM_METAL_MAPS; m++)
                ssim_sum_add(&s[m], col[m * plane + v * p.dst_w] * k[v]);
        }
        const float eps_k = ssim_sum_eps(SSIM_METAL_GAUSSIAN_LEN);
        for (int m = 0; m < SSIM_METAL_MAPS; m++) {
            bool ok;
            r[m] = ssim_sum_result(&s[m], eps_k, &ok);
            if (ok) continue;
            sf64 sum = sf_zero();
            for (int v = 0; v < SSIM_METAL_GAUSSIAN_LEN; v++)
                sum = ssim_sum_slow(sum, col[m * plane + v * p.dst_w] * k[v]);
            r[m] = ssim_sf_to_float(sum);
        }
    }

    // _iqa_ssim(): sigmas, then l, c and s. The CPU build fuses the
    // subtractions.
    const float mu1 = r[0], mu2 = r[1];
    float s1 = fma(-mu1, mu1, r[2]);
    float s2 = fma(-mu2, mu2, r[3]);
    s1 = s1 < 0.0f ? 0.0f : s1;
    s2 = s2 < 0.0f ? 0.0f : s2;
    const float s12 = fma(-mu1, mu2, r[4]);

    const float srsc = sqrt(s1 * s2);
    const float den_l = fma(mu1, mu1, mu2 * mu2) + p.c1;
    const float den_c = (s1 + s2) + p.c2;
    const float clamped = s12 < 0.0f && srsc <= 0.0f ? 0.0f : s12;
    const float s = (clamped + p.c3) / (srsc + p.c3);

    if (!kFast) {
        if (!valid) return;
        // l = (2.0 * mu1 * mu2 + C1) / den_l: the product is exact
        const sf64 num_l = sf_add(sf_mul(ssim_sf_from_float(2.0f * mu1),
                                         ssim_sf_from_float(mu2)),
                                  ssim_sf_from_float(p.c1));
        // c = (2.0 * srsc + C2) / den_c
        const sf64 num_c = sf_add(ssim_sf_from_float(2.0f * srsc),
                                  ssim_sf_from_float(p.c2));
        const uint i = p.out_offset + y * p.dst_w + x;
        out_l[i] = sf_to_bits(sf_div(num_l, ssim_sf_from_float(den_l)));
        out_c[i] = sf_to_bits(sf_div(num_c, ssim_sf_from_float(den_c)));
        out_s[i] = s;
        return;
    }

    // fast mode: sums of l * c * s, l, c and s per threadgroup
    threadgroup float4 sh[32];
    float4 v = 0.0f;
    if (valid) {
        const float l = (2.0f * mu1 * mu2 + p.c1) / den_l;
        const float c = (2.0f * srsc + p.c2) / den_c;
        v = float4(l * c * s, l, c, s);
    }
    v = simd_sum(v);
    if (lane == 0) sh[sid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float4 t = 0.0f;
        for (uint i = 0; i < (tg_size.x * tg_size.y + 31) / 32; i++) t += sh[i];
        part[p.part_offset + tgid.y * tgs.x + tgid.x] = t;
    }
}
