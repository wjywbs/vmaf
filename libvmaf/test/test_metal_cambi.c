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

/**
 * Slow gradients (dist): bands of one 8-bit code value, or on odd frames,
 * dark bands of one code value, a few to tens of pixels wide, which CAMBI
 * detects. ref is the same with noise. Some frames are full-range noise,
 * some are partly flat.
 */
static void fill(VmafPicture *ref, VmafPicture *dis, unsigned frame)
{
    const unsigned w = ref->w[0], h = ref->h[0], bpc = ref->bpc;
    const int max = (1 << bpc) - 1;
    const int shift = bpc - 8;
    const double slope = 8.0 + 12.0 * (frame % 3);
    uint64_t state = 0x1234567 + frame;

    for (unsigned i = 0; i < h; i++) {
        for (unsigned j = 0; j < w; j++) {
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            int v, d;
            if (frame % 5 == 4) {
                v = d = (int)(state >> 33) & max;
            } else if (frame % 5 == 2 && i < h / 2) {
                v = d = 100 << shift;
            } else {
                const double t = (frame % 2 ? 12.0 : 70.0) + slope * j / w + 6.0 * i / h +
                                 3.0 * sin(0.02 * (j + 5.0 * frame)) *
                                 cos(0.015 * i);
                d = frame % 2 ? (int)(t * (1 << shift)) : (int)t << shift;
                v = (int)(t * (1 << shift)) + (int)((state >> 40) % 5) - 2;
                v = v < 0 ? 0 : v > max ? max : v;
            }
            if (bpc == 8) {
                ((uint8_t *)ref->data[0])[i * ref->stride[0] + j] = v;
                ((uint8_t *)dis->data[0])[i * dis->stride[0] + j] = d;
            } else {
                ((uint16_t *)ref->data[0])[i * ref->stride[0] / 2 + j] = v;
                ((uint16_t *)dis->data[0])[i * dis->stride[0] / 2 + j] = d;
            }
        }
    }
}

static void copy_plane(VmafPicture *dst, const VmafPicture *src)
{
    memcpy(dst->data[0], src->data[0], (size_t)src->stride[0] * src->h[0]);
}

/**
 * Run "cambi" and "cambi_metal" with the options `opts` ("key=value:..."
 * or NULL) on the same synthetic frames. Add the number of scores that are
 * not bit-identical to `mismatches`, and of nonzero scores to `nonzero`.
 */
static int compare_cambi(unsigned w, unsigned h, unsigned bpc,
                         unsigned n_threads, const char *opts,
                         unsigned *mismatches, unsigned *nonzero)
{
    const unsigned n_frames = 10;
    VmafContext *cpu, *gpu;
    VmafConfiguration cfg = { .log_level = VMAF_LOG_LEVEL_NONE,
                              .n_threads = n_threads };
    int err = vmaf_init(&cpu, cfg);
    err |= vmaf_init(&gpu, cfg);
    if (err) return err;

    VmafFeatureDictionary *d_cpu = NULL, *d_gpu = NULL, *d_name = NULL;
    bool full_ref = false;
    if (opts) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s", opts);
        for (char *kv = strtok(buf, ":"); kv; kv = strtok(NULL, ":")) {
            char *val = strchr(kv, '=');
            if (!val) return -1;
            *val++ = '\0';
            err |= vmaf_feature_dictionary_set(&d_cpu, kv, val);
            err |= vmaf_feature_dictionary_set(&d_gpu, kv, val);
            err |= vmaf_feature_dictionary_set(&d_name, kv, val);
            full_ref |= !strcmp(kv, "full_ref") && !strcmp(val, "true");
        }
        if (err) return err;
    }

    // options add a suffix to the output name, e.g. "_mlc_1"
    char *names[3] = { 0 };
    unsigned n_names = 0;
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("cambi");
    VmafFeatureExtractorContext *fex_ctx = NULL;
    err = vmaf_feature_extractor_context_create(&fex_ctx, fex,
                                                (VmafDictionary *)d_name);
    if (err) return err;
    names[n_names++] = vmaf_feature_name_from_options(
            "Cambi_feature_cambi_score", fex->options, fex_ctx->fex->priv);
    vmaf_feature_extractor_context_destroy(fex_ctx);
    if (full_ref) {
        names[n_names++] = strdup("cambi_source");
        names[n_names++] = strdup("cambi_full_reference");
    }

    err |= vmaf_use_feature(cpu, "cambi", d_cpu);
    err |= vmaf_use_feature(gpu, "cambi_metal", d_gpu);
    if (err) return err;

    for (unsigned f = 0; f < n_frames; f++) {
        VmafPicture ref, dis, ref2, dis2;
        err |= vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, bpc, w, h);
        err |= vmaf_picture_alloc(&dis, VMAF_PIX_FMT_YUV420P, bpc, w, h);
        err |= vmaf_picture_alloc(&ref2, VMAF_PIX_FMT_YUV420P, bpc, w, h);
        err |= vmaf_picture_alloc(&dis2, VMAF_PIX_FMT_YUV420P, bpc, w, h);
        if (err) return err;
        fill(&ref, &dis, f);
        copy_plane(&ref2, &ref);
        copy_plane(&dis2, &dis);
        err |= vmaf_read_pictures(cpu, &ref, &dis, f);
        err |= vmaf_read_pictures(gpu, &ref2, &dis2, f);
        if (err) return err;
    }
    err |= vmaf_read_pictures(cpu, NULL, NULL, 0);
    err |= vmaf_read_pictures(gpu, NULL, NULL, 0);
    if (err) return err;

    for (unsigned k = 0; k < n_names; k++) {
        if (!names[k]) return -1;
        for (unsigned f = 0; f < n_frames; f++) {
            double a, b;
            err |= vmaf_feature_score_at_index(cpu, names[k], &a, f);
            err |= vmaf_feature_score_at_index(gpu, names[k], &b, f);
            if (err) return err;
            *nonzero += a != 0.0;
            if (memcmp(&a, &b, sizeof(a))) {
                if ((*mismatches)++ < 5) {
                    fprintf(stderr, "%ux%u %u-bit %s: %s frame %u: "
                            "%.17g vs %.17g\n", w, h, bpc,
                            opts ? opts : "", names[k], f, a, b);
                }
            }
        }
        free(names[k]);
    }

    vmaf_close(cpu);
    vmaf_close(gpu);
    return 0;
}

static char *test_cambi_metal_vs_cpu()
{
    static const struct { unsigned w, h; } sizes[] = {
        { 320, 240 },
        { 257, 233 }, // odd at every scale
        { 216, 216 }, // smallest size
        { 640, 100 },
        { 100, 600 },
        { 1000, 24 }, // pad > height at the last scales: runs on the CPU
    };

    for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        for (unsigned bpc = 8; bpc <= 16; bpc += 2) {
            if (bpc == 14) continue;
            unsigned mismatches = 0, nonzero = 0;
            int err = compare_cambi(sizes[s].w, sizes[s].h, bpc, 0, NULL,
                                    &mismatches, &nonzero);
            mu_assert("problem running cambi / cambi_metal", !err);
            mu_assert("cambi_metal differs from cambi", !mismatches);
            mu_assert("all cambi scores are 0", nonzero);
        }
    }
    return NULL;
}

static char *test_cambi_metal_options()
{
    static const char *opts[] = {
        "full_ref=true",
        "enc_width=240:enc_height=180",           // resampled
        "enc_bitdepth=8",                         // anti-dithering
        "enc_width=288:enc_height=216:enc_bitdepth=9:full_ref=true",
        "src_width=320:src_height=200:full_ref=true", // resampled source
        "cambi_vis_lum_threshold=0.06",
        "window_size=100",
        "topk=1",
        "topk=0.001",
        "tvi_threshold=0.01",
        "max_log_contrast=0",                     // 10-bit only
        "max_log_contrast=1",
        "max_log_contrast=3",                     // runs on the CPU
        "eotf=pq",
        "cambi_max_val=0.5",
    };

    for (unsigned o = 0; o < sizeof(opts) / sizeof(opts[0]); o++) {
        unsigned nonzero = 0;
        for (unsigned bpc = 8; bpc <= 10; bpc += 2) {
            unsigned mismatches = 0;
            int err = compare_cambi(320, 240, bpc, 0, opts[o], &mismatches,
                                    &nonzero);
            if (err || mismatches)
                fprintf(stderr, "%s, %u-bit: error %d\n", opts[o], bpc, err);
            mu_assert("problem running cambi / cambi_metal", !err);
            mu_assert("cambi_metal differs from cambi", !mismatches);
        }
        if (!nonzero) fprintf(stderr, "%s: all scores are 0\n", opts[o]);
        mu_assert("all cambi scores are 0", nonzero);
    }
    return NULL;
}

static char *test_cambi_metal_1080p()
{
    // the options of the v1.0.16 models, the speedup is only active from
    // about 1080p
    unsigned mismatches = 0, nonzero = 0;
    int err = compare_cambi(1920, 1080, 8, 0, "cambi_high_res_speedup=1080:"
                            "cambi_vis_lum_threshold=0.06", &mismatches,
                            &nonzero);
    mu_assert("problem running cambi / cambi_metal", !err);
    // windows of 33 (histograms zeroed before each scale) and 21 (left zero
    // by the kernel)
    err = compare_cambi(1920, 1080, 10, 0, "full_ref=true:src_width=1280:"
                        "src_height=720", &mismatches, &nonzero);
    mu_assert("problem running cambi / cambi_metal", !err);
    mu_assert("cambi_metal differs from cambi", !mismatches);
    mu_assert("all cambi scores are 0", nonzero);
    return NULL;
}

static char *test_cambi_metal_threads()
{
    unsigned mismatches = 0, nonzero = 0;
    int err = compare_cambi(320, 240, 8, 4, NULL, &mismatches, &nonzero);
    mu_assert("problem running cambi / cambi_metal", !err);
    err = compare_cambi(320, 240, 10, 4, "full_ref=true", &mismatches,
                        &nonzero);
    mu_assert("problem running cambi / cambi_metal", !err);
    mu_assert("cambi_metal differs from cambi", !mismatches);
    mu_assert("all cambi scores are 0", nonzero);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_cambi_metal_vs_cpu);
    mu_run_test(test_cambi_metal_options);
    mu_run_test(test_cambi_metal_1080p);
    mu_run_test(test_cambi_metal_threads);
    return NULL;
}
