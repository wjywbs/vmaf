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

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_metal.h"
#include "metal/soft_f64.h"
#include "feature/metal/integer_vif_metal.h"

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static uint64_t rng(void)
{
    // xorshift64*
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 0x2545F4914F6CDD1Dull;
}

static uint64_t bits(double d)
{
    uint64_t u;
    memcpy(&u, &d, sizeof(u));
    return u;
}

/**
 * vif_metal_g_exact() compiled as C vs the CPU expression with hardware
 * doubles, over the input domain of vif_statistic_*():
 * sigma1_sq >= sigma_nsq, sigma2_sq > 0, sigma12 > 0.
 */
static char *test_vif_g_exact()
{
    const double gain_limits[] = { 100.0, 1.0, 1.5 };
    unsigned failures = 0;

    for (unsigned l = 0; l < 3; l++) {
        const double vif_enhn_gain_limit = gain_limits[l];
        for (unsigned i = 0; i < 2000000; i++) {
            // mix full range and small values, where g is large
            const unsigned shift1 = rng() % 15;
            const unsigned shift2 = rng() % 31;
            const int32_t sigma1_sq =
                (65536 << 1) + (int32_t)((rng() & 0x7FFFFFFF) >> (shift1 + 1)) %
                (0x7FFFFFFF - (65536 << 1));
            const int32_t sigma2_sq = 1 + (int32_t)((rng() & 0x7FFFFFFE) >> shift2);
            const int32_t sigma12 = 1 + (int32_t)((rng() & 0x7FFFFFFE) >> shift2);

            const double eps = 65536 * 1.0e-10;
            double g = sigma12 / (sigma1_sq + eps);
            int32_t sv_sq = (int32_t)fma(-g, sigma12, sigma2_sq);
            sv_sq = sv_sq > 0 ? sv_sq : 0;
            g = g < vif_enhn_gain_limit ? g : vif_enhn_gain_limit;
            const int64_t numer1_tmp = (int64_t)(g * g * sigma1_sq);

            const VifMetalG r = vif_metal_g_exact(sigma1_sq, sigma2_sq, sigma12,
                                                  bits(vif_enhn_gain_limit));
            if (r.sv_sq != sv_sq || r.numer1_tmp != numer1_tmp) {
                if (failures++ < 5) {
                    fprintf(stderr, "g_exact(%d, %d, %d, %g): "
                            "sv_sq %d vs %d, numer1_tmp %lld vs %lld\n",
                            sigma1_sq, sigma2_sq, sigma12, vif_enhn_gain_limit,
                            r.sv_sq, sv_sq, (long long)r.numer1_tmp,
                            (long long)numer1_tmp);
                }
            }
        }
    }
    mu_assert("vif_metal_g_exact() differs from hardware doubles", !failures);
    return NULL;
}

static void fill(VmafPicture *ref, VmafPicture *dis, unsigned frame)
{
    const unsigned w = ref->w[0], h = ref->h[0], bpc = ref->bpc;
    const int max = (1 << bpc) - 1;
    const double scale = (double)(1 << (bpc - 8));

    for (unsigned i = 0; i < h; i++) {
        for (unsigned j = 0; j < w; j++) {
            // texture + gradient + noise
            const double t = 128.0 + 60.0 * sin(0.21 * j + 0.05 * frame) *
                             cos(0.13 * i) + 0.1 * (i + j) +
                             (double)(rng() % 31) - 15.0;
            int r = (int)(t * scale);
            int d;
            // regions: flat distorted, noisy distorted, inverted, identical,
            // and both nearly flat, where sigma2_sq can round below 0 in the
            // non-log branch
            switch (((i / 41) + (j / 53) + frame) % 5) {
            case 0: d = (int)((128 + (i / 8) % 3) * scale); break;
            case 1: d = r + (int)((double)((int)(rng() % 41) - 20) * scale); break;
            case 2: d = max - r; break;
            case 3:
                r = (int)((100 + (rng() % 8 == 0)) * scale);
                d = (int)((37 + 29 * frame % 200 + (rng() % 16 == 0)) * scale) +
                    (int)(rng() % (1u << (bpc - 8)));
                break;
            default: d = r; break;
            }
            r = r < 0 ? 0 : r > max ? max : r;
            d = d < 0 ? 0 : d > max ? max : d;
            if (bpc == 8) {
                ((uint8_t *)ref->data[0])[i * ref->stride[0] + j] = r;
                ((uint8_t *)dis->data[0])[i * dis->stride[0] + j] = d;
            } else {
                ((uint16_t *)ref->data[0])[i * ref->stride[0] / 2 + j] = r;
                ((uint16_t *)dis->data[0])[i * dis->stride[0] / 2 + j] = d;
            }
        }
    }
}

static const char *vif_features[] = {
    "VMAF_integer_feature_vif_scale0_score",
    "VMAF_integer_feature_vif_scale1_score",
    "VMAF_integer_feature_vif_scale2_score",
    "VMAF_integer_feature_vif_scale3_score", "integer_vif", "integer_vif_num", "integer_vif_den",
    "integer_vif_num_scale0", "integer_vif_den_scale0",
    "integer_vif_num_scale1", "integer_vif_den_scale1",
    "integer_vif_num_scale2", "integer_vif_den_scale2",
    "integer_vif_num_scale3", "integer_vif_den_scale3",
    NULL,
};

/**
 * Run "vif" and "vif_metal" on the same synthetic frames, return the number
 * of scores that are not bit-identical.
 */
static int compare_vif(unsigned w, unsigned h, unsigned bpc, uint64_t cpumask,
                       unsigned *mismatches)
{
    const unsigned n_frames = 3;
    VmafContext *cpu, *gpu;
    VmafConfiguration cfg = { .log_level = VMAF_LOG_LEVEL_NONE,
                              .cpumask = cpumask };
    int err = vmaf_init(&cpu, cfg);
    err |= vmaf_init(&gpu, cfg);
    if (err) return err;

    VmafFeatureDictionary *d_cpu = NULL, *d_gpu = NULL;
    err |= vmaf_feature_dictionary_set(&d_cpu, "debug", "true");
    err |= vmaf_feature_dictionary_set(&d_gpu, "debug", "true");
    err |= vmaf_use_feature(cpu, "vif", d_cpu);
    err |= vmaf_use_feature(gpu, "vif_metal", d_gpu);
    if (err) return err;

    for (unsigned f = 0; f < n_frames; f++) {
        VmafPicture ref, dis, ref2, dis2;
        err |= vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, bpc, w, h);
        err |= vmaf_picture_alloc(&dis, VMAF_PIX_FMT_YUV420P, bpc, w, h);
        err |= vmaf_picture_alloc(&ref2, VMAF_PIX_FMT_YUV420P, bpc, w, h);
        err |= vmaf_picture_alloc(&dis2, VMAF_PIX_FMT_YUV420P, bpc, w, h);
        if (err) return err;
        fill(&ref, &dis, f);
        const size_t sz = (size_t)ref.stride[0] * h;
        memcpy(ref2.data[0], ref.data[0], sz);
        memcpy(dis2.data[0], dis.data[0], sz);
        err |= vmaf_read_pictures(cpu, &ref, &dis, f);
        err |= vmaf_read_pictures(gpu, &ref2, &dis2, f);
        if (err) return err;
    }
    err |= vmaf_read_pictures(cpu, NULL, NULL, 0);
    err |= vmaf_read_pictures(gpu, NULL, NULL, 0);
    if (err) return err;

    *mismatches = 0;
    for (unsigned k = 0; vif_features[k]; k++) {
        for (unsigned f = 0; f < n_frames; f++) {
            double a, b;
            err |= vmaf_feature_score_at_index(cpu, vif_features[k], &a, f);
            err |= vmaf_feature_score_at_index(gpu, vif_features[k], &b, f);
            if (err) return err;
            if (memcmp(&a, &b, sizeof(a))) {
                if ((*mismatches)++ < 5) {
                    fprintf(stderr, "%ux%u %u-bit cpumask %llu: %s frame %u: "
                            "%.17g vs %.17g\n", w, h, bpc,
                            (unsigned long long)cpumask, vif_features[k], f,
                            a, b);
                }
            }
        }
    }

    vmaf_close(cpu);
    vmaf_close(gpu);
    return 0;
}

static char *test_vif_metal_vs_cpu()
{
    static const struct { unsigned w, h; } sizes[] = {
        { 176, 144 }, // multiple of 8 and of 16
        { 203, 117 }, // odd, NEON residual columns at every scale
        { 64, 72 },
    };
    // NEON (default) and scalar C on arm64, which differ in clamping
    static const uint64_t cpumasks[] = { 0, 1 };

    for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        for (unsigned bpc = 8; bpc <= 12; bpc += 2) {
            for (unsigned m = 0; m < 2; m++) {
                unsigned mismatches = 0;
                int err = compare_vif(sizes[s].w, sizes[s].h, bpc,
                                      cpumasks[m], &mismatches);
                mu_assert("problem running vif / vif_metal", !err);
                mu_assert("vif_metal differs from vif", !mismatches);
            }
        }
    }
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_vif_g_exact);
    mu_run_test(test_vif_metal_vs_cpu);
    return NULL;
}
