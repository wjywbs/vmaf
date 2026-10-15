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
#include "feature/metal/ssim_metal.h"

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;

static uint64_t rng(void)
{
    rng_state = rng_state * 6364136223846793005ull + 1442695040888963407ull;
    return rng_state >> 11;
}

/** uniform in [0, 1) */
static double rng_unit(void)
{
    return (double)(rng() >> 10) * 0x1p-43;
}

/**
 * The exact sum of ssim_metal.h for `n` terms, as the kernels use it:
 * the filter, or the slow path. `*slow` counts the fallbacks.
 */
static float exact_sum(const float *t, int n, unsigned *slow)
{
    SsimSum s;
    ssim_sum_init(&s);
    for (int i = 0; i < n; i++) ssim_sum_add(&s, t[i]);
    bool ok;
    const float r = ssim_sum_result(&s, ssim_sum_eps(n), &ok);
    if (ok) return r;
    (*slow)++;
    sf64 sum = sf_zero();
    for (int i = 0; i < n; i++) sum = ssim_sum_slow(sum, t[i]);
    return ssim_sf_to_float(sum);
}

/** The CPU's sum */
static float cpu_sum(const float *t, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += t[i];
    return (float)sum;
}

/** The slow path only */
static float slow_sum(const float *t, int n)
{
    sf64 sum = sf_zero();
    for (int i = 0; i < n; i++) sum = ssim_sum_slow(sum, t[i]);
    return ssim_sf_to_float(sum);
}

static bool same(float a, float b)
{
    return !memcmp(&a, &b, sizeof(a));
}

static char *test_ssim_sum_random()
{
    static const int ns[] = { 1, 2, 11, 25, 81, 100 };
    float t[SSIM_METAL_MAX_TAPS];
    unsigned mismatches = 0, slow_mismatches = 0;
    unsigned slow[6] = { 0 }, total[6] = { 0 };

    for (unsigned it = 0; it < 2000000; it++) {
        const int n = ns[it % 6];
        const unsigned kind = (it / 6) % 6;
        for (int i = 0; i < n; i++) {
            double v;
            switch (kind) {
            case 0: // filter products: positive, similar magnitudes
                v = rng_unit() * 255.0 * 0.1;
                break;
            case 1: // wide spread of magnitudes and signs
                v = ldexp(rng_unit() + 0.5, (int)(rng() % 80) - 40);
                if (rng() & 1) v = -v;
                break;
            case 2: // cancellation: pairs of nearly opposite terms
                v = (i & 1) ? -(double)t[i - 1] *
                              (1.0 + ldexp((double)(rng() % 64) - 32.0, -23))
                            : rng_unit() * 1000.0;
                break;
            case 3: // small integers and powers of 2: exact ties
                v = (rng() & 1) ? (double)(rng() % 256) :
                                  ldexp(1.0, (int)(rng() % 60) - 30);
                if (rng() % 4 == 0) v = -v;
                break;
            case 4: // subnormal and tiny terms
                v = ldexp(rng_unit(), -126 - (int)(rng() % 20));
                if (rng() & 1) v = -v;
                break;
            default: // all +-0 or exact zero sums
                v = (rng() & 1) ? -0.0 : 0.0;
                if (rng() % 4 == 0) v = (double)(int)(rng() % 3) - 1.0;
                break;
            }
            t[i] = (float)v;
        }
        const float ref = cpu_sum(t, n);
        if (!same(exact_sum(t, n, &slow[kind]), ref) && mismatches++ < 5) {
            fprintf(stderr, "kind %u n %d: %a vs %a\n", kind, n,
                    exact_sum(t, n, &slow[kind]), ref);
        }
        if (!same(slow_sum(t, n), ref)) slow_mismatches++;
        total[kind]++;
    }
    for (unsigned k = 0; k < 6; k++) {
        fprintf(stderr, "ssim_sum kind %u: %u of %u sums took the slow path\n",
                k, slow[k], total[k]);
    }
    mu_assert("SsimSum differs from the CPU's double sum", !mismatches);
    mu_assert("the slow path differs from the CPU's double sum",
              !slow_mismatches);
    mu_assert("the slow path is never taken", slow[2] && slow[4]);
    mu_assert("the filter is rarely accepted for filter products",
              slow[0] < total[0] / 10);
    return NULL;
}

/**
 * Sums whose float-float value and double value round to different floats:
 * the filter must reject them.
 */
static char *test_ssim_sum_ties()
{
    unsigned slow = 0;
    // 1 + 2^-24 is a tie, rounded to even (1); with 2^-60, lost in the
    // double sum, the exact sum rounds up
    const float a[] = { 1.0f, 0x1p-24f, 0x1p-60f };
    mu_assert("tie with a tiny term", same(exact_sum(a, 3, &slow), 1.0f));
    mu_assert("tie with a tiny term", same(cpu_sum(a, 3), 1.0f));
    // the same below a power of 2: 2 - 2^-24 rounds to even (2)
    const float b[] = { 2.0f, -0x1p-24f, -0x1p-60f };
    mu_assert("tie below a power of 2", same(exact_sum(b, 3, &slow),
                                             cpu_sum(b, 3)));
    // terms that cancel to 0 but not in double: 2^60 + 1 - 2^60
    const float c[] = { 0x1p60f, 1.0f, -0x1p60f };
    mu_assert("cancellation", same(exact_sum(c, 3, &slow), cpu_sum(c, 3)));
    mu_assert("cancellation", same(cpu_sum(c, 3), 0.0f));
    // -0 + -0 is -0, the CPU's sum starts at +0
    const float d[] = { -0.0f, -0.0f };
    mu_assert("negative zeros", same(exact_sum(d, 2, &slow), 0.0f));
    mu_assert("the filter accepts a tie", slow == 3);
    return NULL;
}

static void put(VmafPicture *pic, unsigned i, unsigned j, int v)
{
    const int max = (1 << pic->bpc) - 1;
    v = v < 0 ? 0 : v > max ? max : v;
    if (pic->bpc == 8)
        ((uint8_t *)pic->data[0])[i * pic->stride[0] + j] = v;
    else
        ((uint16_t *)pic->data[0])[i * pic->stride[0] / 2 + j] = v;
}

/**
 * Textures with noise (ref), and the same with more noise and coarser
 * values (dist). Some frames are flat, or identical.
 */
static void fill(VmafPicture *ref, VmafPicture *dis, unsigned frame)
{
    const double scale = (double)(1 << (ref->bpc - 8));
    const bool flat = frame % 5 == 2, identical = frame % 5 == 3;
    uint64_t state = 0x1234567 + frame;

    for (unsigned i = 0; i < ref->h[0]; i++) {
        for (unsigned j = 0; j < ref->w[0]; j++) {
            state = state * 6364136223846793005ull + 1442695040888963407ull;
            const double t = 128.0 +
                100.0 * sin(0.05 * (j + 3.0 * frame)) * cos(0.03 * i) +
                (double)((state >> 40) % 9) - 4.0;
            const double d = t + (double)((state >> 50) % 17) - 8.0;
            const int q = 1 + frame % 4;
            const double r = flat ? 100.0 : t;
            put(ref, i, j, (int)(r * scale));
            put(dis, i, j, identical ? (int)(r * scale) :
                           (int)((int)d / q * q * scale));
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

static const char *ssim_features[] = {
    "float_ssim", "float_ssim_l", "float_ssim_c", "float_ssim_s", NULL,
};

static const char *ms_ssim_features[] = {
    "float_ms_ssim",
    "float_ms_ssim_l_scale0", "float_ms_ssim_l_scale1",
    "float_ms_ssim_l_scale2", "float_ms_ssim_l_scale3",
    "float_ms_ssim_l_scale4",
    "float_ms_ssim_c_scale0", "float_ms_ssim_c_scale1",
    "float_ms_ssim_c_scale2", "float_ms_ssim_c_scale3",
    "float_ms_ssim_c_scale4",
    "float_ms_ssim_s_scale0", "float_ms_ssim_s_scale1",
    "float_ms_ssim_s_scale2", "float_ms_ssim_s_scale3",
    "float_ms_ssim_s_scale4",
    NULL,
};

/**
 * Run `name` ("float_ssim" or "float_ms_ssim") and its Metal version with
 * the options `opts` ("key=value:..." or NULL) on the same synthetic frames.
 * Add the number of scores that are not bit-identical to `mismatches`, and
 * of nonzero scores to `nonzero`. Return 1 if both fail on the frame size.
 */
static int compare_ssim(const char *name, unsigned w, unsigned h,
                        unsigned bpc, unsigned n_threads, const char *opts,
                        unsigned *mismatches, unsigned *nonzero)
{
    const unsigned n_frames = 6;
    VmafContext *cpu, *gpu;
    VmafConfiguration cfg = { .log_level = VMAF_LOG_LEVEL_NONE,
                              .n_threads = n_threads };
    int err = vmaf_init(&cpu, cfg);
    err |= vmaf_init(&gpu, cfg);
    if (err) return err;

    VmafFeatureDictionary *d_cpu = NULL, *d_gpu = NULL;
    if (opts) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s", opts);
        for (char *kv = strtok(buf, ":"); kv; kv = strtok(NULL, ":")) {
            char *val = strchr(kv, '=');
            if (!val) return -1;
            *val++ = '\0';
            err |= vmaf_feature_dictionary_set(&d_cpu, kv, val);
            err |= vmaf_feature_dictionary_set(&d_gpu, kv, val);
        }
        if (err) return err;
    }

    char metal_name[64];
    snprintf(metal_name, sizeof(metal_name), "%s_metal", name);
    err |= vmaf_use_feature(cpu, name, d_cpu);
    err |= vmaf_use_feature(gpu, metal_name, d_gpu);
    if (err) return err;

    for (unsigned f = 0; f < n_frames; f++) {
        VmafPicture ref, dis, ref2, dis2;
        err |= vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, bpc, w, h);
        err |= vmaf_picture_alloc(&dis, VMAF_PIX_FMT_YUV420P, bpc, w, h);
        if (err) return err;
        fill(&ref, &dis, f);
        err |= copy_picture(&ref2, &ref);
        err |= copy_picture(&dis2, &dis);
        if (err) return err;
        const int err_cpu = vmaf_read_pictures(cpu, &ref, &dis, f);
        const int err_gpu = vmaf_read_pictures(gpu, &ref2, &dis2, f);
        if (err_cpu || err_gpu) {
            // e.g. "scale below 1x1": both must fail
            vmaf_close(cpu);
            vmaf_close(gpu);
            return err_cpu && err_gpu ? 1 : -1;
        }
    }
    err |= vmaf_read_pictures(cpu, NULL, NULL, 0);
    err |= vmaf_read_pictures(gpu, NULL, NULL, 0);
    if (err) return err;

    const char **names = strcmp(name, "float_ssim") ? ms_ssim_features :
                                                      ssim_features;
    for (unsigned k = 0; names[k]; k++) {
        for (unsigned f = 0; f < n_frames; f++) {
            double a, b;
            const int err_a = vmaf_feature_score_at_index(cpu, names[k], &a, f);
            const int err_b = vmaf_feature_score_at_index(gpu, names[k], &b, f);
            if (!err_a != !err_b) return -1; // provided by one only
            if (err_a) break; // not enabled
            *nonzero += a != 0.0;
            if (memcmp(&a, &b, sizeof(a))) {
                if ((*mismatches)++ < 5) {
                    fprintf(stderr, "%ux%u %u-bit %s: %s frame %u: "
                            "%.17g vs %.17g\n", w, h, bpc,
                            opts ? opts : "", names[k], f, a, b);
                }
            }
        }
    }

    vmaf_close(cpu);
    vmaf_close(gpu);
    return 0;
}

static char *run_sizes(const char *name)
{
    static const struct { unsigned w, h; bool ms_fails; } sizes[] = {
        { 640, 480, false },
        { 835, 587, false }, // odd: decimated sizes round up, w % 8 != 0
        { 1920, 1080, false },
        { 1283, 723, false }, // odd, SSIM scale 3
        { 200, 100, true },  // MS-SSIM: "scale below 1x1"
        { 64, 21, true },
    };

    for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        for (unsigned bpc = 8; bpc <= 16; bpc++) {
            if (bpc != 8 && bpc != 9 && bpc % 2) continue;
            if (bpc == 14) continue;
            unsigned mismatches = 0, nonzero = 0;
            int err = compare_ssim(name, sizes[s].w, sizes[s].h, bpc, 0,
                                   "enable_lcs=true", &mismatches, &nonzero);
            if (err == 1 && sizes[s].ms_fails && strcmp(name, "float_ssim"))
                continue;
            if (err || mismatches)
                fprintf(stderr, "%s %ux%u %u-bit: error %d\n", name,
                        sizes[s].w, sizes[s].h, bpc, err);
            mu_assert("problem running the CPU / Metal SSIM", !err);
            mu_assert("Metal SSIM differs from the CPU", !mismatches);
            mu_assert("all SSIM scores are 0", nonzero);
        }
    }
    return NULL;
}

static char *test_ssim_metal_vs_cpu()
{
    return run_sizes("float_ssim");
}

static char *test_ms_ssim_metal_vs_cpu()
{
    return run_sizes("float_ms_ssim");
}

static char *test_ssim_metal_options()
{
    static const struct { const char *name, *opts; } cases[] = {
        { "float_ssim", "enable_db=true" },
        { "float_ssim", "enable_db=true:clip_db=true" },
        { "float_ssim", "scale=1:enable_lcs=true" },
        { "float_ssim", "scale=2" },
        { "float_ssim", "scale=3:enable_lcs=true" },
        { "float_ssim", "scale=7" },
        { "float_ssim", "scale=10" },
        { "float_ms_ssim", "enable_db=true" },
        { "float_ms_ssim", "enable_db=true:clip_db=true" },
    };

    for (unsigned c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        unsigned mismatches = 0, nonzero = 0;
        int err = compare_ssim(cases[c].name, 1283, 723, 8, 0, cases[c].opts,
                               &mismatches, &nonzero);
        err |= compare_ssim(cases[c].name, 640, 480, 10, 0, cases[c].opts,
                            &mismatches, &nonzero);
        if (err || mismatches)
            fprintf(stderr, "%s %s: error %d\n", cases[c].name,
                    cases[c].opts, err);
        mu_assert("problem running the CPU / Metal SSIM", !err);
        mu_assert("Metal SSIM differs from the CPU", !mismatches);
        mu_assert("all SSIM scores are 0", nonzero);
    }

    // decimated to 23x12, on the GPU, and to 11x10, on the CPU
    unsigned mismatches = 0, nonzero = 0;
    int err = compare_ssim("float_ssim", 225, 115, 8, 0, "scale=10",
                           &mismatches, &nonzero);
    mu_assert("problem running the CPU / Metal SSIM", !err);
    err = compare_ssim("float_ssim", 109, 99, 10, 0, "scale=10",
                       &mismatches, &nonzero);
    mu_assert("problem running the CPU / Metal SSIM", !err || err == 1);
    mu_assert("Metal SSIM differs from the CPU", !mismatches);
    mu_assert("all SSIM scores are 0", nonzero);
    return NULL;
}

static char *test_ssim_metal_threads()
{
    unsigned mismatches = 0, nonzero = 0;
    int err = compare_ssim("float_ssim", 835, 587, 8, 4, "enable_lcs=true",
                           &mismatches, &nonzero);
    err |= compare_ssim("float_ms_ssim", 835, 587, 10, 4, "enable_lcs=true",
                        &mismatches, &nonzero);
    mu_assert("problem running the CPU / Metal SSIM", !err);
    mu_assert("Metal SSIM differs from the CPU", !mismatches);
    mu_assert("all SSIM scores are 0", nonzero);
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_ssim_sum_random);
    mu_run_test(test_ssim_sum_ties);
    mu_run_test(test_ssim_metal_vs_cpu);
    mu_run_test(test_ms_ssim_metal_vs_cpu);
    mu_run_test(test_ssim_metal_options);
    mu_run_test(test_ssim_metal_threads);
    return NULL;
}
