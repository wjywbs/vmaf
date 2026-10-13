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
#include "feature/feature_extractor.h"
#include "feature/feature_name.h"
#include "feature/metal/integer_adm_metal.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

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

/** A band value with a random magnitude of up to `max_bits` bits. */
static int32_t band_value(unsigned max_bits)
{
    const unsigned b = rng() % (max_bits + 1);
    const int32_t v = b ? (int32_t)(rng() & ((1u << b) - 1)) : 0;
    return rng() & 1 ? -v : v;
}

/**
 * adm_metal_angle_exact() compiled as C vs the angle test of adm_decouple*()
 * with hardware doubles. Nearly parallel vectors make the test close.
 */
static char *test_adm_angle_exact()
{
    const float cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);
    unsigned failures = 0, taken = 0;
    const unsigned n = 4000000;

    for (unsigned i = 0; i < n; i++) {
        // int16 bands at scale 0, wider at scales 1-3
        const unsigned max_bits = i & 1 ? 16 : 28;
        const int32_t oh = band_value(max_bits), ov = band_value(max_bits);
        int32_t th, tv;
        if (i % 4 < 2) {
            // about 1 degree apart
            const double a = (rng() % 4001) / 2000.0 * M_PI / 180.0;
            const double s = 0.5 + (rng() % 1000) / 1000.0;
            th = (int32_t)(s * (oh * cos(a) - ov * sin(a)));
            tv = (int32_t)(s * (oh * sin(a) + ov * cos(a)));
        } else {
            th = band_value(max_bits);
            tv = band_value(max_bits);
        }

        const int64_t ot_dp = (int64_t)oh * th + (int64_t)ov * tv;
        const int64_t o_mag_sq = (int64_t)oh * oh + (int64_t)ov * ov;
        const int64_t t_mag_sq = (int64_t)th * th + (int64_t)tv * tv;
        const int angle_flag = (((float)ot_dp / 4096.0) >= 0.0f) &&
            (((float)ot_dp / 4096.0) * ((float)ot_dp / 4096.0) >=
                cos_1deg_sq * ((float)o_mag_sq / 4096.0) * ((float)t_mag_sq / 4096.0));
        taken += angle_flag;

        const bool r = adm_metal_angle_exact(ot_dp, o_mag_sq, t_mag_sq,
                                             bits(cos_1deg_sq));
        if (r != !!angle_flag && failures++ < 5) {
            fprintf(stderr, "angle_exact(%lld, %lld, %lld): %d vs %d\n",
                    (long long)ot_dp, (long long)o_mag_sq, (long long)t_mag_sq,
                    r, angle_flag);
        }
    }
    mu_assert("adm_metal_angle_exact() differs from hardware doubles", !failures);
    mu_assert("angle test inputs are one-sided", taken > n / 8 && taken < n * 7 / 8);
    return NULL;
}

/**
 * adm_metal_gain_exact() compiled as C vs the enhancement gain limit of
 * adm_decouple*() with hardware doubles.
 */
static char *test_adm_gain_exact()
{
    const double gain_limits[] = { 100.0, 1.0, 1.2, 1.5, 2.75, 1.0 + 1e-9 };
    unsigned failures = 0;

    for (unsigned l = 0; l < sizeof(gain_limits) / sizeof(gain_limits[0]); l++) {
        const double adm_enhn_gain_limit = gain_limits[l];
        const int gain_int = adm_enhn_gain_limit == floor(adm_enhn_gain_limit) ?
                             (int)adm_enhn_gain_limit : 0;
        for (unsigned i = 0; i < 1000000; i++) {
            const unsigned max_bits = i & 1 ? 16 : 30;
            const int32_t rst = band_value(max_bits);
            // t near rst * gain, or random
            const int32_t t = i % 3 ?
                (int32_t)(rst * adm_enhn_gain_limit) + (int32_t)(rng() % 5) - 2 :
                band_value(max_bits);
            const bool positive = rst > 0;
            if (!rst) continue;

            int32_t ref = rst;
            if (positive) ref = MIN(rst * adm_enhn_gain_limit, t);
            else ref = MAX(rst * adm_enhn_gain_limit, t);

            const int64_t r = adm_metal_gain_exact(rst, t, positive,
                                                   bits(adm_enhn_gain_limit),
                                                   gain_int);
            if (r != ref && failures++ < 5) {
                fprintf(stderr, "gain_exact(%d, %d, %g): %lld vs %d\n", rst, t,
                        adm_enhn_gain_limit, (long long)r, ref);
            }
        }
    }
    mu_assert("adm_metal_gain_exact() differs from hardware doubles", !failures);
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
            // enhanced and attenuated contrast (decouple gain limit and
            // angle test), black and white (clipping)
            switch (((i / 23) + (j / 37) + frame) % 8) {
            case 0: d = (int)((128 + (i / 8) % 3) * scale); break;
            case 1: d = r + (int)((double)((int)(rng() % 41) - 20) * scale); break;
            case 2: d = max - r; break;
            case 3: d = (int)(128 * scale + (r - 128 * scale) * 1.3); break;
            case 4: d = (int)(128 * scale + (r - 128 * scale) * 0.7) +
                        (int)(rng() % 3) - 1; break;
            case 5: r = 0; d = (int)(rng() % 3); break;
            case 6: r = max; d = max - (int)(rng() % 2); break;
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

static const char *adm_features[] = {
    "VMAF_integer_feature_adm2_score", "VMAF_integer_feature_aim_score",
    "VMAF_integer_feature_adm3_score",
    "integer_adm_scale0", "integer_adm_scale1", "integer_adm_scale2",
    "integer_adm_scale3",
    "integer_adm", "integer_adm_num", "integer_adm_den",
    "integer_adm_num_scale0", "integer_adm_den_scale0",
    "integer_adm_num_scale1", "integer_adm_den_scale1",
    "integer_adm_num_scale2", "integer_adm_den_scale2",
    "integer_adm_num_scale3", "integer_adm_den_scale3",
    NULL,
};

/**
 * Run "adm" and "adm_metal" with the option `key` = `val` on the same
 * synthetic frames, return the number of scores that are not bit-identical.
 */
static int compare_adm(unsigned w, unsigned h, unsigned bpc, uint64_t cpumask,
                       const char *key, const char *val, unsigned *mismatches)
{
    const unsigned n_frames = 3;
    VmafContext *cpu, *gpu;
    VmafConfiguration cfg = { .log_level = VMAF_LOG_LEVEL_NONE,
                              .cpumask = cpumask };
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

    // options add a suffix to the output names, e.g. "_egl_1.2"
    char *names[32] = { 0 };
    unsigned n_names = 0;
    VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name("adm");
    VmafFeatureExtractorContext *fex_ctx = NULL;
    err = vmaf_feature_extractor_context_create(&fex_ctx, fex,
                                                (VmafDictionary *)d_name);
    if (err) return err;
    for (unsigned k = 0; adm_features[k]; k++) {
        names[n_names++] = vmaf_feature_name_from_options(adm_features[k],
                fex->options, fex_ctx->fex->priv);
    }
    vmaf_feature_extractor_context_destroy(fex_ctx);

    err |= vmaf_use_feature(cpu, "adm", d_cpu);
    err |= vmaf_use_feature(gpu, "adm_metal", d_gpu);
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
    for (unsigned k = 0; k < n_names; k++) {
        if (!names[k]) return -1;
        for (unsigned f = 0; f < n_frames; f++) {
            double a, b;
            err |= vmaf_feature_score_at_index(cpu, names[k], &a, f);
            err |= vmaf_feature_score_at_index(gpu, names[k], &b, f);
            if (err) return err;
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
    return 0;
}

static char *test_adm_metal_vs_cpu()
{
    static const struct { unsigned w, h; uint64_t cpumask; } sizes[] = {
        { 176, 144, 0 }, // multiple of 16, NEON DWT at scale 0
        { 203, 117, 0 }, // odd at every scale
        { 144, 98, 0 },  // odd band sizes at scales 1-3
        { 64, 72, 0 },
        // w % 16 == 8: adm_dwt2_8_neon() is wrong, compare to scalar C only
        { 168, 96, 1 },
        { 120, 66, 1 },
    };

    for (unsigned s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        for (unsigned bpc = 8; bpc <= 12; bpc += 2) {
            for (uint64_t m = sizes[s].cpumask; m < 2; m++) {
                unsigned mismatches = 0;
                int err = compare_adm(sizes[s].w, sizes[s].h, bpc, m,
                                      NULL, NULL, &mismatches);
                mu_assert("problem running adm / adm_metal", !err);
                mu_assert("adm_metal differs from adm", !mismatches);
            }
        }
    }
    return NULL;
}

static char *test_adm_metal_options()
{
    static const char *opts[][2] = {
        { "adm_enhn_gain_limit", "1.0" },
        { "adm_enhn_gain_limit", "1.2" },
        { "adm_csf_mode", "1" },
        { "adm_csf_mode", "2" },
        { "adm_csf_mode", "3" },
        { "adm_norm_view_dist", "4.5" },
        { "adm_ref_display_height", "2160" },
        { "adm_csf_scale", "2.0" },
        { "adm_csf_diag_scale", "0.5" },
        { "adm_noise_weight", "0.5" },
        { "adm_skip_scale0", "true" },
        { "adm_skip_aim", "true" },
        { "adm_dlm_weight", "0.8" },
        { "adm_min_val", "0.9" },
    };

    for (unsigned o = 0; o < sizeof(opts) / sizeof(opts[0]); o++) {
        for (unsigned bpc = 8; bpc <= 10; bpc += 2) {
            unsigned mismatches = 0;
            int err = compare_adm(176, 144, bpc, 0, opts[o][0], opts[o][1],
                                  &mismatches);
            mu_assert("problem running adm / adm_metal", !err);
            mu_assert("adm_metal differs from adm", !mismatches);
        }
    }
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_adm_angle_exact);
    mu_run_test(test_adm_gain_exact);
    mu_run_test(test_adm_metal_vs_cpu);
    mu_run_test(test_adm_metal_options);
    return NULL;
}
