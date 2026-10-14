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
#include <stdlib.h>
#include <string.h>

#include "test.h"
#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_metal.h"
#include "metal/common.h"

static char *test_metal_state()
{
    VmafMetalState *s = NULL;
    VmafMetalConfiguration cfg = { 0 };
    int err = vmaf_metal_state_init(&s, cfg);
    mu_assert("problem during vmaf_metal_state_init", !err && s);
    mu_assert("exact mode by default", !s->fast_math);

    err = vmaf_metal_selftest(s);
    mu_assert("selftest failed in exact mode", !err);
    // second run hits the pipeline cache
    err = vmaf_metal_selftest(s);
    mu_assert("selftest failed on cached pipeline", !err);
    vmaf_metal_state_free(s);

    cfg.fast_math = 1;
    err = vmaf_metal_state_init(&s, cfg);
    mu_assert("problem during vmaf_metal_state_init", !err && s);
    err = vmaf_metal_selftest(s);
    mu_assert("selftest failed in fast mode", !err);
    vmaf_metal_state_free(s);

    return NULL;
}

static char *test_metal_default_state()
{
    VmafMetalState *s = vmaf_metal_state_default();
    mu_assert("no default state", s);
    mu_assert("default state is not a singleton", s == vmaf_metal_state_default());
    mu_assert("selftest failed on default state", !vmaf_metal_selftest(s));
    return NULL;
}

static char *test_metal_import_state()
{
    VmafContext *vmaf;
    VmafConfiguration vmaf_cfg = { 0 };
    int err = vmaf_init(&vmaf, vmaf_cfg);
    mu_assert("problem during vmaf_init", !err);

    VmafMetalState *s = NULL;
    VmafMetalConfiguration cfg = { 0 };
    err = vmaf_metal_state_init(&s, cfg);
    mu_assert("problem during vmaf_metal_state_init", !err);
    err = vmaf_metal_import_state(vmaf, s);
    mu_assert("problem during vmaf_metal_import_state", !err);
    err = vmaf_metal_import_state(vmaf, s);
    mu_assert("double import should fail", err);

    err = vmaf_close(vmaf);
    mu_assert("problem during vmaf_close", !err);
    return NULL;
}

static uint64_t rng_state = 0x2545F4914F6CDD1Dull;

static uint64_t rng(void)
{
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return rng_state = x;
}

static int read_frame(VmafContext *vmaf, unsigned w, unsigned h, unsigned index)
{
    VmafPicture ref, dist;
    int err = vmaf_picture_alloc(&ref, VMAF_PIX_FMT_YUV420P, 8, w, h);
    err |= vmaf_picture_alloc(&dist, VMAF_PIX_FMT_YUV420P, 8, w, h);
    if (err) return err;
    for (unsigned i = 0; i < h; i++) {
        uint8_t *r = (uint8_t *)ref.data[0] + i * ref.stride[0];
        uint8_t *d = (uint8_t *)dist.data[0] + i * dist.stride[0];
        for (unsigned j = 0; j < w; j++) {
            r[j] = (uint8_t)(64 + (i * 3 + j * 5 + index * 7) % 128);
            d[j] = (uint8_t)(r[j] + rng() % 9 - 4);
        }
    }
    return vmaf_read_pictures(vmaf, &ref, &dist, index);
}

static int use_metal_features(VmafContext *vmaf)
{
    int err = vmaf_use_feature(vmaf, "vif_metal", NULL);
    err |= vmaf_use_feature(vmaf, "adm_metal", NULL);
    err |= vmaf_use_feature(vmaf, "motion_metal", NULL);
    return err;
}

/**
 * Metal extractors write their scores asynchronously. All scores must be
 * available after the flush, and closing a context with frames in flight
 * must be safe.
 */
static char *test_metal_async()
{
    static const char *names[] = {
        "VMAF_integer_feature_vif_scale0_score",
        "VMAF_integer_feature_adm2_score",
        "VMAF_integer_feature_motion_sad_score",
    };
    const unsigned w = 160, h = 96, n = 8;

    for (unsigned n_threads = 0; n_threads <= 2; n_threads += 2) {
        VmafContext *vmaf;
        VmafConfiguration cfg = { .log_level = VMAF_LOG_LEVEL_NONE,
                                  .n_threads = n_threads };
        int err = vmaf_init(&vmaf, cfg);
        mu_assert("problem during vmaf_init", !err);
        VmafMetalState *s = NULL;
        VmafMetalConfiguration metal_cfg = { 0 };
        err = vmaf_metal_state_init(&s, metal_cfg);
        err |= vmaf_metal_import_state(vmaf, s);
        err |= use_metal_features(vmaf);
        mu_assert("problem setting up Metal features", !err);

        for (unsigned i = 0; i < n; i++) {
            err = read_frame(vmaf, w, h, i);
            mu_assert("problem during vmaf_read_pictures", !err);
        }
        err = vmaf_read_pictures(vmaf, NULL, NULL, 0);
        mu_assert("problem flushing", !err);
        for (unsigned i = 0; i < n; i++) {
            for (unsigned k = 0; k < 3; k++) {
                double score;
                err = vmaf_feature_score_at_index(vmaf, names[k], &score, i);
                mu_assert("score not available after flush", !err);
            }
        }
        err = vmaf_close(vmaf);
        mu_assert("problem during vmaf_close", !err);
    }

    // close with frames in flight, without flushing
    for (unsigned n_threads = 0; n_threads <= 2; n_threads += 2) {
        VmafContext *vmaf;
        VmafConfiguration cfg = { .log_level = VMAF_LOG_LEVEL_NONE,
                                  .n_threads = n_threads };
        int err = vmaf_init(&vmaf, cfg);
        err |= use_metal_features(vmaf); // default Metal state
        mu_assert("problem setting up Metal features", !err);
        for (unsigned i = 0; i < n; i++) {
            err = read_frame(vmaf, 1920, 1080, i);
            mu_assert("problem during vmaf_read_pictures", !err);
        }
        err = vmaf_close(vmaf);
        mu_assert("problem during vmaf_close", !err);
    }
    return NULL;
}

static uint64_t to_bits(double d)
{
    uint64_t u;
    memcpy(&u, &d, sizeof(u));
    return u;
}

static double from_bits(uint64_t u)
{
    double d;
    memcpy(&d, &u, sizeof(d));
    return d;
}

// see test_soft_f64.c
static double rand_double(int range)
{
    uint64_t m = rng() & ((1ull << 52) - 1);
    switch (rng() % 4) {
    case 0: m &= ~((1ull << (rng() % 52)) - 1); break;
    case 1: m |= (1ull << (rng() % 52)) - 1; break;
    default: break;
    }
    const int e = (int)(rng() % (2 * range + 1)) - range;
    const uint64_t sign = rng() & 1;
    return from_bits((sign << 63) | ((uint64_t)(e + 1023) << 52) | m);
}

static uint64_t reference(enum VmafMetalSoftF64Op op, double a, double b,
                          double c, uint64_t a_bits)
{
    volatile double r;
    switch (op) {
    case VMAF_METAL_SOFT_F64_ADD: r = a + b; return to_bits(r);
    case VMAF_METAL_SOFT_F64_SUB: r = a - b; return to_bits(r);
    case VMAF_METAL_SOFT_F64_MUL: r = a * b; return to_bits(r);
    case VMAF_METAL_SOFT_F64_DIV: r = a / b; return to_bits(r);
    case VMAF_METAL_SOFT_F64_FMA: return to_bits(fma(a, b, c));
    case VMAF_METAL_SOFT_F64_TRUNC_I64: return (uint64_t)(int64_t)a;
    case VMAF_METAL_SOFT_F64_FROM_I64: r = (double)(int64_t)a_bits; return to_bits(r);
    case VMAF_METAL_SOFT_F64_LT: return a < b;
    }
    return 0;
}

static char *test_metal_soft_f64()
{
    VmafMetalState *s = vmaf_metal_state_default();
    mu_assert("no default state", s);

    const unsigned n = 1 << 20;
    uint64_t *a = malloc(n * sizeof(*a)), *b = malloc(n * sizeof(*b));
    uint64_t *c = malloc(n * sizeof(*c)), *out = malloc(n * sizeof(*out));
    mu_assert("malloc failed", a && b && c && out);

    char *msg = NULL;
    for (int op = VMAF_METAL_SOFT_F64_ADD; op <= VMAF_METAL_SOFT_F64_LT; op++) {
        for (unsigned i = 0; i < n; i++) {
            const int range = op == VMAF_METAL_SOFT_F64_TRUNC_I64 ? 62 : 60;
            double x = rand_double(range), y = rand_double(range),
                   z = rand_double(2 * range);
            if (op == VMAF_METAL_SOFT_F64_ADD && i % 3 == 0)
                y = -x * (1.0 + (double)(rng() % 64) * 0x1p-52);
            if (op == VMAF_METAL_SOFT_F64_FMA && i % 2 == 0)
                z = -(x * y);
            if (op == VMAF_METAL_SOFT_F64_DIV && i % 3 == 0) {
                // exact quotients and quotients close to 1, 2, 4
                x = i % 2 ? y * (double)(rng() % 4096 + 1) :
                    y * (double)(1 << (rng() % 3)) *
                    (1.0 + (double)((int)(rng() % 64) - 32) * 0x1p-52);
            }
            a[i] = op == VMAF_METAL_SOFT_F64_FROM_I64 ?
                   (uint64_t)((int64_t)rng() >> (rng() % 64)) : to_bits(x);
            b[i] = to_bits(y);
            c[i] = to_bits(z);
        }
        if (vmaf_metal_soft_f64_eval(s, op, a, b, c, out, n)) {
            msg = "vmaf_metal_soft_f64_eval failed";
            goto end;
        }
        for (unsigned i = 0; i < n; i++) {
            const uint64_t expect = reference(op, from_bits(a[i]),
                                              from_bits(b[i]), from_bits(c[i]),
                                              a[i]);
            if (out[i] != expect) {
                fprintf(stderr, "\nop %d (%a, %a, %a): expected %llx got %llx\n",
                        op, from_bits(a[i]), from_bits(b[i]), from_bits(c[i]),
                        (unsigned long long)expect, (unsigned long long)out[i]);
                msg = "GPU soft_f64 mismatch";
                goto end;
            }
        }
    }

end:
    free(a);
    free(b);
    free(c);
    free(out);
    return msg;
}

char *run_tests()
{
    mu_run_test(test_metal_state);
    mu_run_test(test_metal_default_state);
    mu_run_test(test_metal_import_state);
    mu_run_test(test_metal_soft_f64);
    mu_run_test(test_metal_async);
    return NULL;
}
