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
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "test.h"
#include "metal/soft_f64.h"

#define N_ITER 2000000

static uint64_t rng_state = 0x9E3779B97F4A7C15ull;

static uint64_t rng(void)
{
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return rng_state = x;
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

static sf64 sf(double d)
{
    return sf_from_bits(to_bits(d));
}

static double df(sf64 a)
{
    return from_bits(sf_to_bits(a));
}

// Random normal double with exponent in [-range, range]. Mantissas are biased
// towards patterns that produce rounding ties and cancellation.
static double rand_double(int range)
{
    uint64_t m = rng() & ((1ull << 52) - 1);
    switch (rng() % 4) {
    case 0: m &= ~((1ull << (rng() % 52)) - 1); break; // trailing zeros
    case 1: m |= (1ull << (rng() % 52)) - 1; break;    // trailing ones
    default: break;
    }
    const int e = (int)(rng() % (2 * range + 1)) - range;
    const uint64_t sign = rng() & 1;
    return from_bits((sign << 63) | ((uint64_t)(e + 1023) << 52) | m);
}

static int mismatch(const char *op, double a, double b, double c,
                    double expect, double got)
{
    fprintf(stderr, "\n%s(%a, %a, %a): expected %a got %a\n", op, a, b, c,
            expect, got);
    return 1;
}

static char *test_round_trip()
{
    for (unsigned i = 0; i < N_ITER; i++) {
        const double a = rand_double(1000);
        if (to_bits(df(sf(a))) != to_bits(a)) {
            mismatch("round_trip", a, 0, 0, a, df(sf(a)));
            return "bits round trip mismatch";
        }
    }
    mu_assert("zero", sf_to_bits(sf(0.0)) == 0);
    return NULL;
}

static char *test_add_sub()
{
    for (unsigned i = 0; i < N_ITER; i++) {
        const double a = rand_double(80);
        double b = rand_double(80);
        // near cancellation
        if (i % 3 == 0) b = -a * (1.0 + (double)(rng() % 64) * 0x1p-52);
        const volatile double add = a + b, sub = a - b;
        if (to_bits(df(sf_add(sf(a), sf(b)))) != to_bits(add)) {
            mismatch("add", a, b, 0, add, df(sf_add(sf(a), sf(b))));
            return "sf_add mismatch";
        }
        if (to_bits(df(sf_sub(sf(a), sf(b)))) != to_bits(sub)) {
            mismatch("sub", a, b, 0, sub, df(sf_sub(sf(a), sf(b))));
            return "sf_sub mismatch";
        }
    }
    return NULL;
}

static char *test_mul_div()
{
    for (unsigned i = 0; i < N_ITER; i++) {
        const double a = rand_double(200), b = rand_double(200);
        const volatile double mul = a * b, div = a / b;
        if (to_bits(df(sf_mul(sf(a), sf(b)))) != to_bits(mul)) {
            mismatch("mul", a, b, 0, mul, df(sf_mul(sf(a), sf(b))));
            return "sf_mul mismatch";
        }
        if (to_bits(df(sf_div(sf(a), sf(b)))) != to_bits(div)) {
            mismatch("div", a, b, 0, div, df(sf_div(sf(a), sf(b))));
            return "sf_div mismatch";
        }
    }
    return NULL;
}

static char *test_fma()
{
    for (unsigned i = 0; i < N_ITER; i++) {
        const double a = rand_double(60), b = rand_double(60);
        double c = rand_double(120);
        switch (i % 4) {
        case 0: c = -(a * b); break; // exact product residual
        case 1: c = -(a * b) * (1.0 + (double)(rng() % 16) * 0x1p-52); break;
        default: break;
        }
        const double expect = fma(a, b, c);
        const double got = df(sf_fma(sf(a), sf(b), sf(c)));
        if (to_bits(got) != to_bits(expect)) {
            mismatch("fma", a, b, c, expect, got);
            return "sf_fma mismatch";
        }
    }
    return NULL;
}

static char *test_int_conversion()
{
    for (unsigned i = 0; i < N_ITER; i++) {
        const int64_t x = (int64_t)rng() >> (rng() % 64);
        const volatile double d = (double)x;
        if (to_bits(df(sf_from_i64(x))) != to_bits(d)) {
            mismatch("from_i64", (double)x, 0, 0, d, df(sf_from_i64(x)));
            return "sf_from_i64 mismatch";
        }
        const uint64_t u = rng() >> (rng() % 64);
        const volatile double du = (double)u;
        if (to_bits(df(sf_from_u64(u))) != to_bits(du)) {
            mismatch("from_u64", (double)u, 0, 0, du, df(sf_from_u64(u)));
            return "sf_from_u64 mismatch";
        }
        const double a = rand_double(62);
        if (sf_trunc_i64(sf(a)) != (int64_t)a) {
            mismatch("trunc_i64", a, 0, 0, (double)(int64_t)a,
                     (double)sf_trunc_i64(sf(a)));
            return "sf_trunc_i64 mismatch";
        }
    }
    return NULL;
}

static char *test_lt()
{
    for (unsigned i = 0; i < N_ITER; i++) {
        const double a = rand_double(4);
        const double b = i % 5 ? rand_double(4) : (i % 2 ? a : -a);
        if (sf_lt(sf(a), sf(b)) != (a < b)) {
            mismatch("lt", a, b, 0, a < b, sf_lt(sf(a), sf(b)));
            return "sf_lt mismatch";
        }
    }
    mu_assert("0 < 0", !sf_lt(sf(0.0), sf(0.0)));
    mu_assert("0 < 1", sf_lt(sf(0.0), sf(1.0)));
    mu_assert("-1 < 0", sf_lt(sf(-1.0), sf(0.0)));
    return NULL;
}

char *run_tests()
{
    mu_run_test(test_round_trip);
    mu_run_test(test_add_sub);
    mu_run_test(test_mul_div);
    mu_run_test(test_fma);
    mu_run_test(test_int_conversion);
    mu_run_test(test_lt);
    return NULL;
}
