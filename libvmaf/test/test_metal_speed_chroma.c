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

static void put(VmafPicture *pic, unsigned p, unsigned i, unsigned j, int v)
{
    const int max = (1 << pic->bpc) - 1;
    v = v < 0 ? 0 : v > max ? max : v;
    if (pic->bpc == 8)
        ((uint8_t *)pic->data[p])[i * pic->stride[p] + j] = v;
    else
        ((uint16_t *)pic->data[p])[i * pic->stride[p] / 2 + j] = v;
}

/**
 * Chroma textures with noise (ref), and the same with more noise and
 * coarser values (dist). Some frames have flat planes, where the covariance
 * matrix is singular: both U planes, or only the distorted V plane.
 */
static void fill(VmafPicture *ref, VmafPicture *dis, unsigned frame)
{
    const double scale = (double)(1 << (ref->bpc - 8));
    uint64_t state = 0x1234567 + frame;

    for (unsigned p = 0; p < 3; p++) {
        const bool flat_ref = p == 1 && frame % 5 == 2;
        const bool flat_dis = flat_ref || (p == 2 && frame % 5 == 3);
        for (unsigned i = 0; i < ref->h[p]; i++) {
            for (unsigned j = 0; j < ref->w[p]; j++) {
                state = state * 6364136223846793005ull +
                        1442695040888963407ull;
                const double t = 128.0 +
                    40.0 * sin(0.05 * (j + 3.0 * frame) * (p + 1)) *
                    cos(0.03 * i * (3 - p)) +
                    (double)((state >> 40) % 9) - 4.0;
                const double d = t + (double)((state >> 50) % 17) - 8.0;
                const int q = 1 + frame % 4;
                put(ref, p, i, j, flat_ref ? (int)(100 * scale) :
                                             (int)(t * scale));
                put(dis, p, i, j, flat_dis ? (int)(90 * scale) :
                                  (int)((int)d / q * q * scale));
            }
        }
    }
}

static int copy_picture(VmafPicture *dst, const VmafPicture *src)
{
    int err = vmaf_picture_alloc(dst, src->pix_fmt, src->bpc, src->w[0],
                                 src->h[0]);
    if (err) return err;
    for (unsigned p = 0; p < 3; p++)
        memcpy(dst->data[p], src->data[p], (size_t)src->stride[p] * src->h[p]);
    return 0;
}

static const char *speed_features[] = {
    "Speed_chroma_feature_speed_chroma_u_score",
    "Speed_chroma_feature_speed_chroma_v_score",
    "Speed_chroma_feature_speed_chroma_uv_score",
    NULL,
};

/**
 * Run "speed_chroma" and "speed_chroma_metal" with the options `opts`
 * ("key=value:..." or NULL) on the same synthetic frames. Add the number of
 * scores that are not bit-identical to `mismatches`, and of nonzero scores
 * to `nonzero`.
 */
static int compare_speed(unsigned w, unsigned h, enum VmafPixelFormat fmt,
                         unsigned bpc, unsigned n_threads, const char *opts,
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
        }
        if (err) return err;
    }

    // options add a suffix to the output names, e.g. "_wvm_5"
    char *names[3] = { 0 };
    unsigned n_names = 0;
    VmafFeatureExtractor *fex =
        vmaf_get_feature_extractor_by_name("speed_chroma");
    VmafFeatureExtractorContext *fex_ctx = NULL;
    err = vmaf_feature_extractor_context_create(&fex_ctx, fex,
                                                (VmafDictionary *)d_name);
    if (err) return err;
    for (unsigned k = 0; speed_features[k]; k++) {
        names[n_names++] = vmaf_feature_name_from_options(speed_features[k],
                fex->options, fex_ctx->fex->priv);
    }
    vmaf_feature_extractor_context_destroy(fex_ctx);

    err |= vmaf_use_feature(cpu, "speed_chroma", d_cpu);
    err |= vmaf_use_feature(gpu, "speed_chroma_metal", d_gpu);
    if (err) return err;

    for (unsigned f = 0; f < n_frames; f++) {
        VmafPicture ref, dis, ref2, dis2;
        err |= vmaf_picture_alloc(&ref, fmt, bpc, w, h);
        err |= vmaf_picture_alloc(&dis, fmt, bpc, w, h);
        if (err) return err;
        fill(&ref, &dis, f);
        err |= copy_picture(&ref2, &ref);
        err |= copy_picture(&dis2, &dis);
        if (err) return err;
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
                    fprintf(stderr, "%ux%u fmt %d %u-bit %s: %s frame %u: "
                            "%.17g vs %.17g\n", w, h, fmt, bpc,
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

static char *test_speed_chroma_metal_vs_cpu()
{
    static const struct {
        unsigned w, h;
        enum VmafPixelFormat fmt;
    } sizes[] = {
        { 640, 480, VMAF_PIX_FMT_YUV420P },
        { 834, 586, VMAF_PIX_FMT_YUV420P },  // not a multiple of 16 or 5
        { 1920, 1080, VMAF_PIX_FMT_YUV420P },
        { 640, 360, VMAF_PIX_FMT_YUV422P },
        { 400, 300, VMAF_PIX_FMT_YUV444P },
    };

    for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        for (unsigned bpc = 8; bpc <= 16; bpc += 2) {
            if (bpc == 14) continue;
            unsigned mismatches = 0, nonzero = 0;
            int err = compare_speed(sizes[s].w, sizes[s].h, sizes[s].fmt, bpc,
                                    0, NULL, &mismatches, &nonzero);
            mu_assert("problem running speed_chroma / speed_chroma_metal",
                      !err);
            mu_assert("speed_chroma_metal differs from speed_chroma",
                      !mismatches);
            mu_assert("all speed_chroma scores are 0", nonzero);
        }
    }
    return NULL;
}

static char *test_speed_chroma_metal_options()
{
    static const char *opts[] = {
        "speed_kernelscale=1.0:speed_nn_floor=0.1:speed_sigma_nn=0.19:speed_weight_var_mode=5:speed_max_val=45.0", // vmaf_v1.0.16
        "speed_kernelscale=0.5",
        "speed_kernelscale=2.0",
        "speed_kernelscale=3.7113402",            // 360 / 97, widest filter
        "speed_sigma_nn=1.5",
        "speed_nn_floor=0.5",
        "speed_weight_var_mode=1",
        "speed_weight_var_mode=3",
        "speed_weight_var_mode=6",
        "speed_max_val=0.5",
        "speed_prescale=2.0",                     // nearest
        "speed_prescale=1.5:speed_prescale_method=bilinear",
        "speed_prescale=1.0011",                  // same size: copied
        "speed_prescale=1.5:speed_prescale_method=bicubic", // runs on the CPU
    };

    for (unsigned o = 0; o < sizeof(opts) / sizeof(opts[0]); o++) {
        unsigned nonzero = 0;
        for (unsigned bpc = 8; bpc <= 10; bpc += 2) {
            unsigned mismatches = 0;
            int err = compare_speed(640, 480, VMAF_PIX_FMT_YUV420P, bpc, 0,
                                    opts[o], &mismatches, &nonzero);
            if (err || mismatches)
                fprintf(stderr, "%s, %u-bit: error %d\n", opts[o], bpc, err);
            mu_assert("problem running speed_chroma / speed_chroma_metal",
                      !err);
            mu_assert("speed_chroma_metal differs from speed_chroma",
                      !mismatches);
        }
        if (!nonzero) fprintf(stderr, "%s: all scores are 0\n", opts[o]);
        mu_assert("all speed_chroma scores are 0", nonzero);
    }
    return NULL;
}

static char *test_speed_chroma_metal_prescale()
{
    // the vmaf_v1.0.16 5d0h and 3d0h_2160 models: bilinear 0.6 and 0.5
    static const char *opts[] = {
        "speed_prescale=0.6:speed_prescale_method=bilinear:speed_nn_floor=0.1:speed_sigma_nn=0.19:speed_weight_var_mode=5",
        "speed_prescale=0.5:speed_prescale_method=bilinear",
        "speed_prescale=0.5",
        "speed_prescale=0.7:speed_prescale_method=lanczos4", // runs on the CPU
    };

    for (unsigned o = 0; o < sizeof(opts) / sizeof(opts[0]); o++) {
        unsigned nonzero = 0;
        for (unsigned bpc = 8; bpc <= 10; bpc += 2) {
            unsigned mismatches = 0;
            int err = compare_speed(1920, 1080, VMAF_PIX_FMT_YUV420P, bpc, 0,
                                    opts[o], &mismatches, &nonzero);
            if (err || mismatches)
                fprintf(stderr, "%s, %u-bit: error %d\n", opts[o], bpc, err);
            mu_assert("problem running speed_chroma / speed_chroma_metal",
                      !err);
            mu_assert("speed_chroma_metal differs from speed_chroma",
                      !mismatches);
        }
        if (!nonzero) fprintf(stderr, "%s: all scores are 0\n", opts[o]);
        mu_assert("all speed_chroma scores are 0", nonzero);
    }
    // odd sizes, 4:4:4
    unsigned mismatches = 0, nonzero = 0;
    int err = compare_speed(834, 586, VMAF_PIX_FMT_YUV444P, 12, 0,
                            "speed_prescale=0.7:speed_prescale_method=bilinear",
                            &mismatches, &nonzero);
    mu_assert("problem running speed_chroma / speed_chroma_metal", !err);
    mu_assert("speed_chroma_metal differs from speed_chroma", !mismatches);
    mu_assert("all speed_chroma scores are 0", nonzero);
    return NULL;
}

static char *test_speed_chroma_metal_threads()
{
    unsigned mismatches = 0, nonzero = 0;
    int err = compare_speed(640, 480, VMAF_PIX_FMT_YUV420P, 8, 4, NULL,
                            &mismatches, &nonzero);
    mu_assert("problem running speed_chroma / speed_chroma_metal", !err);
    err = compare_speed(640, 480, VMAF_PIX_FMT_YUV420P, 10, 4,
                        "speed_weight_var_mode=5", &mismatches, &nonzero);
    mu_assert("problem running speed_chroma / speed_chroma_metal", !err);
    mu_assert("speed_chroma_metal differs from speed_chroma", !mismatches);
    mu_assert("all speed_chroma scores are 0", nonzero);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_speed_chroma_metal_vs_cpu);
    mu_run_test(test_speed_chroma_metal_options);
    mu_run_test(test_speed_chroma_metal_prescale);
    mu_run_test(test_speed_chroma_metal_threads);
    return NULL;
}
