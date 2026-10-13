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
#include "feature/feature_extractor.h"
#include "feature/feature_name.h"

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static uint64_t rng(void)
{
    // xorshift64*
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 0x2545F4914F6CDD1Dull;
}

/**
 * Moving texture with noise. Some frames repeat the previous one (zero
 * motion), some jump to full-range noise (large SAD).
 */
static void fill(VmafPicture *pic, unsigned frame)
{
    const unsigned w = pic->w[0], h = pic->h[0], bpc = pic->bpc;
    const int max = (1 << bpc) - 1;
    const double scale = (double)(1 << (bpc - 8));
    const unsigned f = frame % 4 == 3 ? frame - 1 : frame;
    uint64_t state = 0x1234567 + f;

    for (unsigned i = 0; i < h; i++) {
        for (unsigned j = 0; j < w; j++) {
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            int v;
            if (f % 7 == 5) {
                v = (int)(state >> 33) & max;
            } else {
                const double t = 128.0 + 90.0 * sin(0.17 * (j + 3.0 * f)) *
                                 cos(0.11 * (i - 2.0 * f)) +
                                 (double)((state >> 40) % 9) - 4.0;
                v = (int)(t * scale);
            }
            v = v < 0 ? 0 : v > max ? max : v;
            if (bpc == 8)
                ((uint8_t *)pic->data[0])[i * pic->stride[0] + j] = v;
            else
                ((uint16_t *)pic->data[0])[i * pic->stride[0] / 2 + j] = v;
        }
    }
}

static const char *motion_features[] = {
    "VMAF_integer_feature_motion_sad_score",
    "VMAF_integer_feature_motion_score",
    "VMAF_integer_feature_motion2_score",
    "VMAF_integer_feature_motion3_score",
    NULL,
};

/**
 * Run "motion" and "motion_metal" with the option `key` = `val` on the same
 * synthetic frames, return the number of scores that are not bit-identical.
 */
static int compare_motion(unsigned w, unsigned h, unsigned bpc,
                          uint64_t cpumask, unsigned n_threads,
                          const char *key, const char *val,
                          unsigned *mismatches)
{
    const unsigned n_frames = 12;
    VmafContext *cpu, *gpu;
    VmafConfiguration cfg = { .log_level = VMAF_LOG_LEVEL_NONE,
                              .cpumask = cpumask, .n_threads = n_threads };
    int err = vmaf_init(&cpu, cfg);
    err |= vmaf_init(&gpu, cfg);
    if (err) return err;

    VmafFeatureDictionary *d_cpu = NULL, *d_gpu = NULL, *d_name = NULL;
    err |= vmaf_feature_dictionary_set(&d_cpu, "debug", "true");
    err |= vmaf_feature_dictionary_set(&d_gpu, "debug", "true");
    err |= vmaf_feature_dictionary_set(&d_name, "debug", "true");
    if (key) {
        err |= vmaf_feature_dictionary_set(&d_cpu, key, val);
        err |= vmaf_feature_dictionary_set(&d_gpu, key, val);
        err |= vmaf_feature_dictionary_set(&d_name, key, val);
    }
    if (err) return err;

    // options add a suffix to the output names, e.g. "_mffw"
    char *names[8] = { 0 };
    unsigned n_names = 0;
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("motion");
    VmafFeatureExtractorContext *fex_ctx = NULL;
    err = vmaf_feature_extractor_context_create(&fex_ctx, fex,
                                                (VmafDictionary *)d_name);
    if (err) return err;
    for (unsigned k = 0; motion_features[k]; k++) {
        names[n_names++] = vmaf_feature_name_from_options(motion_features[k],
                fex->options, fex_ctx->fex->priv);
    }
    vmaf_feature_extractor_context_destroy(fex_ctx);

    err |= vmaf_use_feature(cpu, "motion", d_cpu);
    err |= vmaf_use_feature(gpu, "motion_metal", d_gpu);
    if (err) return err;

    for (unsigned f = 0; f < n_frames; f++) {
        VmafPicture ref, dis, ref2, dis2;
        err |= vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, bpc, w, h);
        err |= vmaf_picture_alloc(&dis, VMAF_PIX_FMT_YUV420P, bpc, w, h);
        err |= vmaf_picture_alloc(&ref2, VMAF_PIX_FMT_YUV420P, bpc, w, h);
        err |= vmaf_picture_alloc(&dis2, VMAF_PIX_FMT_YUV420P, bpc, w, h);
        if (err) return err;
        fill(&ref, f);
        const size_t sz = (size_t)ref.stride[0] * h;
        memcpy(dis.data[0], ref.data[0], sz);
        memcpy(ref2.data[0], ref.data[0], sz);
        memcpy(dis2.data[0], ref.data[0], sz);
        err |= vmaf_read_pictures(cpu, &ref, &dis, f);
        err |= vmaf_read_pictures(gpu, &ref2, &dis2, f);
        if (err) return err;
    }
    err |= vmaf_read_pictures(cpu, NULL, NULL, 0);
    err |= vmaf_read_pictures(gpu, NULL, NULL, 0);
    if (err) return err;

    *mismatches = 0;
    unsigned nonzero = 0;
    for (unsigned k = 0; k < n_names; k++) {
        if (!names[k]) return -1;
        for (unsigned f = 0; f < n_frames; f++) {
            double a, b;
            err |= vmaf_feature_score_at_index(cpu, names[k], &a, f);
            err |= vmaf_feature_score_at_index(gpu, names[k], &b, f);
            if (err) return err;
            nonzero += a != 0.0;
            if (memcmp(&a, &b, sizeof(a))) {
                if ((*mismatches)++ < 5) {
                    fprintf(stderr, "%ux%u %u-bit cpumask %llu: %s "
                            "frame %u: %.17g vs %.17g\n", w, h, bpc,
                            (unsigned long long)cpumask, names[k], f, a, b);
                }
            }
        }
        free(names[k]);
    }

    vmaf_close(cpu);
    vmaf_close(gpu);
    // all zero would not test anything, unless forced
    return nonzero || (key && !strcmp(key, "motion_force_zero")) ? 0 : -1;
}

static char *test_motion_metal_vs_cpu()
{
    static const struct { unsigned w, h; } sizes[] = {
        { 176, 144 },
        { 203, 117 }, // odd, NEON residual columns
        { 600, 40 },  // wider than a threadgroup: halo between chunks
        { 513, 9 },   // one column past two full chunks
        { 8, 6 },
    };

    for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        for (unsigned bpc = 8; bpc <= 16; bpc += 2) {
            for (uint64_t m = 0; m < 2; m++) {
                unsigned mismatches = 0;
                int err = compare_motion(sizes[s].w, sizes[s].h, bpc, m, 0,
                                         NULL, NULL, &mismatches);
                mu_assert("problem running motion / motion_metal", !err);
                mu_assert("motion_metal differs from motion", !mismatches);
            }
        }
    }
    return NULL;
}

static char *test_motion_metal_options()
{
    static const char *opts[][2] = {
        { "motion_five_frame_window", "true" },
        { "motion_moving_average", "true" },
        { "motion_blend_factor", "0.5" },
        { "motion_blend_offset", "1.5" },
        { "motion_fps_weight", "2.5" },
        { "motion_max_val", "3.0" },
        { "motion_force_zero", "true" },
    };

    for (unsigned o = 0; o < sizeof(opts) / sizeof(opts[0]); o++) {
        for (unsigned bpc = 8; bpc <= 10; bpc += 2) {
            unsigned mismatches = 0;
            int err = compare_motion(176, 144, bpc, 0, 0, opts[o][0],
                                     opts[o][1], &mismatches);
            mu_assert("problem running motion / motion_metal", !err);
            mu_assert("motion_metal differs from motion", !mismatches);
        }
    }
    return NULL;
}

static char *test_motion_metal_threads()
{
    // PREV_REF extractors run on the thread pool
    unsigned mismatches = 0;
    int err = compare_motion(176, 144, 8, 0, 4, NULL, NULL, &mismatches);
    mu_assert("problem running motion / motion_metal", !err);
    mu_assert("motion_metal differs from motion", !mismatches);
    err = compare_motion(176, 144, 10, 0, 4, "motion_five_frame_window",
                         "true", &mismatches);
    mu_assert("problem running motion / motion_metal", !err);
    mu_assert("motion_metal differs from motion", !mismatches);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_motion_metal_vs_cpu);
    mu_run_test(test_motion_metal_options);
    mu_run_test(test_motion_metal_threads);
    return NULL;
}
