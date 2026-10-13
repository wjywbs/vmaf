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
 * Software IEEE-754 binary64 arithmetic, bit-exact with hardware doubles
 * (round to nearest, ties to even). Metal has no fp64; Metal kernels use this
 * where the CPU feature extractors compute with doubles.
 *
 * This header is shared between Metal kernels and C (unit tests). Values are
 * restricted to zero and finite normal numbers: no subnormals, infinities or
 * NaNs, which is sufficient for the feature extractors.
 */

#ifndef __VMAF_SRC_METAL_SOFT_F64_H__
#define __VMAF_SRC_METAL_SOFT_F64_H__

#ifdef __METAL_VERSION__
typedef ulong sf_u64;
typedef long sf_i64;
typedef int sf_i32;
#define SF_U64_C(x) x##ul
static inline sf_u64 sf_mulhi(sf_u64 a, sf_u64 b) { return mulhi(a, b); }
static inline sf_i32 sf_clz64(sf_u64 x) { return (sf_i32)clz(x); }
#else
#include <stdbool.h>
#include <stdint.h>
typedef uint64_t sf_u64;
typedef int64_t sf_i64;
typedef int32_t sf_i32;
#define SF_U64_C(x) x##ull
static inline sf_u64 sf_mulhi(sf_u64 a, sf_u64 b)
{
    return (sf_u64)(((unsigned __int128)a * b) >> 64);
}
static inline sf_i32 sf_clz64(sf_u64 x) { return x ? __builtin_clzll(x) : 64; }
#endif

/** value = (-1)^s * m * 2^e, m == 0 or 2^52 <= m < 2^53 */
typedef struct sf64 {
    sf_u64 m;
    sf_i32 e;
    sf_i32 s;
} sf64;

typedef struct sf_u128 {
    sf_u64 hi, lo;
} sf_u128;

static inline sf_u128 sf_u128_make(sf_u64 hi, sf_u64 lo)
{
    sf_u128 r;
    r.hi = hi;
    r.lo = lo;
    return r;
}

static inline sf_i32 sf_u128_clz(sf_u128 a)
{
    return a.hi ? sf_clz64(a.hi) : 64 + sf_clz64(a.lo);
}

static inline sf_u128 sf_u128_shl(sf_u128 a, sf_i32 n)
{
    if (n <= 0) return a;
    if (n >= 128) return sf_u128_make(0, 0);
    if (n >= 64) return sf_u128_make(a.lo << (n - 64), 0);
    return sf_u128_make((a.hi << n) | (a.lo >> (64 - n)), a.lo << n);
}

static inline sf_u128 sf_u128_shr(sf_u128 a, sf_i32 n)
{
    if (n <= 0) return a;
    if (n >= 128) return sf_u128_make(0, 0);
    if (n >= 64) return sf_u128_make(0, a.hi >> (n - 64));
    return sf_u128_make(a.hi >> n, (a.lo >> n) | (a.hi << (64 - n)));
}

/** true if any of the low n bits of a are set */
static inline bool sf_u128_low_bits(sf_u128 a, sf_i32 n)
{
    if (n <= 0) return false;
    if (n >= 128) return (a.hi | a.lo) != 0;
    if (n >= 64) {
        const sf_u64 mask = n == 64 ? 0 : (~(sf_u64)0 >> (128 - n));
        return a.lo != 0 || (a.hi & mask) != 0;
    }
    return (a.lo & (~(sf_u64)0 >> (64 - n))) != 0;
}

static inline sf_u128 sf_u128_add(sf_u128 a, sf_u128 b)
{
    const sf_u64 lo = a.lo + b.lo;
    return sf_u128_make(a.hi + b.hi + (lo < a.lo ? 1 : 0), lo);
}

/** a - b, requires a >= b */
static inline sf_u128 sf_u128_sub(sf_u128 a, sf_u128 b)
{
    return sf_u128_make(a.hi - b.hi - (a.lo < b.lo ? 1 : 0), a.lo - b.lo);
}

static inline bool sf_u128_lt(sf_u128 a, sf_u128 b)
{
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}

static inline sf_u128 sf_mul64(sf_u64 a, sf_u64 b)
{
    return sf_u128_make(sf_mulhi(a, b), a * b);
}

static inline sf64 sf_zero(void)
{
    sf64 r;
    r.m = 0;
    r.e = 0;
    r.s = 0;
    return r;
}

/**
 * Round (-1)^s * (M + sticky) * 2^E to 53 bits. `sticky` signals nonzero bits
 * below the least significant bit of M, it requires M to carry at least two
 * bits more than the target precision.
 */
/** sf_round() of a 64 bit M */
static inline sf64 sf_round64(sf_u64 M, sf_i32 E, bool sticky, sf_i32 s)
{
    if (!M) return sf_zero();

    const sf_i32 p = 63 - sf_clz64(M);
    sf64 r;
    r.s = s;
    if (p <= 52) {
        r.m = M << (52 - p);
        r.e = E - (52 - p);
        return r;
    }

    const sf_i32 sh = p - 52;
    sf_u64 q = M >> sh;
    const bool guard = ((M >> (sh - 1)) & 1) != 0;
    const bool rest = sticky || (M & ((((sf_u64)1) << (sh - 1)) - 1)) != 0;
    sf_i32 e = E + sh;
    if (guard && (rest || (q & 1))) {
        q++;
        if (q == ((sf_u64)1 << 53)) {
            q >>= 1;
            e++;
        }
    }
    r.m = q;
    r.e = e;
    return r;
}

static inline sf64 sf_round(sf_u128 M, sf_i32 E, bool sticky, sf_i32 s)
{
    if (!M.hi) return sf_round64(M.lo, E, sticky, s);

    const sf_i32 p = 127 - sf_u128_clz(M);
    sf64 r;
    r.s = s;
    if (p <= 52) {
        r.m = M.lo << (52 - p);
        r.e = E - (52 - p);
        return r;
    }

    const sf_i32 sh = p - 52;
    sf_u64 q = sf_u128_shr(M, sh).lo;
    const bool guard = (sf_u128_shr(M, sh - 1).lo & 1) != 0;
    const bool rest = sticky || sf_u128_low_bits(M, sh - 1);
    sf_i32 e = E + sh;
    if (guard && (rest || (q & 1))) {
        q++;
        if (q == ((sf_u64)1 << 53)) {
            q >>= 1;
            e++;
        }
    }
    r.m = q;
    r.e = e;
    return r;
}

static inline sf64 sf_from_bits(sf_u64 bits)
{
    const sf_i32 exp = (sf_i32)((bits >> 52) & 0x7ff);
    if (!exp) return sf_zero();
    sf64 r;
    r.m = (bits & (((sf_u64)1 << 52) - 1)) | ((sf_u64)1 << 52);
    r.e = exp - 1075;
    r.s = (sf_i32)(bits >> 63);
    return r;
}

static inline sf_u64 sf_to_bits(sf64 a)
{
    const sf_u64 sign = (sf_u64)a.s << 63;
    if (!a.m) return sign;
    return sign | ((sf_u64)(a.e + 1075) << 52) |
           (a.m & (((sf_u64)1 << 52) - 1));
}

static inline sf64 sf_from_mag(sf_u64 x, sf_i32 s)
{
    if (x >> 53) return sf_round(sf_u128_make(0, x), 0, false, s);
    if (!x) return sf_zero();
    // exact, no rounding
    const sf_i32 sh = sf_clz64(x) - 11;
    sf64 r;
    r.m = x << sh;
    r.e = -sh;
    r.s = s;
    return r;
}

static inline sf64 sf_from_u64(sf_u64 x)
{
    return sf_from_mag(x, 0);
}

static inline sf64 sf_from_i64(sf_i64 x)
{
    if (x < 0) return sf_from_mag((sf_u64)0 - (sf_u64)x, 1);
    return sf_from_mag((sf_u64)x, 0);
}

static inline sf64 sf_neg(sf64 a)
{
    a.s ^= 1;
    return a;
}

/**
 * Round (-1)^sa * A * 2^Ea + (-1)^sb * B * 2^Eb, exact inputs of up to 126
 * bits.
 */
static inline sf64 sf_add_exact(sf_u128 A, sf_i32 Ea, sf_i32 sa,
                                sf_u128 B, sf_i32 Eb, sf_i32 sb)
{
    const bool a0 = !A.hi && !A.lo;
    const bool b0 = !B.hi && !B.lo;
    if (a0 && b0) return sf_zero();
    if (b0) return sf_round(A, Ea, false, sa);
    if (a0) return sf_round(B, Eb, false, sb);

    // X: operand with the larger leading bit position, normalized such that
    // its leading bit is bit 126. Y: aligned to X, with sticky.
    const sf_i32 la = 127 - sf_u128_clz(A), lb = 127 - sf_u128_clz(B);
    sf_u128 X, Y;
    sf_i32 Ex, Ey, sx, sy, lx;
    if (la + Ea >= lb + Eb) {
        X = A; Ex = Ea; sx = sa; lx = la;
        Y = B; Ey = Eb; sy = sb;
    } else {
        X = B; Ex = Eb; sx = sb; lx = lb;
        Y = A; Ey = Ea; sy = sa;
    }
    const sf_i32 hx = 126 - lx;
    X = sf_u128_shl(X, hx);
    const sf_i32 E = Ex - hx;
    const sf_i32 d = Ey - E;
    bool sticky = false;
    if (d >= 0) {
        Y = sf_u128_shl(Y, d);
    } else {
        sticky = sf_u128_low_bits(Y, -d);
        Y = sf_u128_shr(Y, -d);
    }

    if (sx == sy) return sf_round(sf_u128_add(X, Y), E, sticky, sx);

    if (sf_u128_lt(X, Y)) { // only possible without sticky
        const sf_u128 t = X;
        X = Y;
        Y = t;
        sx = sy;
    }
    sf_u128 M = sf_u128_sub(X, Y);
    if (sticky) M = sf_u128_sub(M, sf_u128_make(0, 1));
    return sf_round(M, E, sticky, sx);
}

static inline sf64 sf_add(sf64 a, sf64 b)
{
    if (!b.m) return a.m ? a : sf_zero();
    if (!a.m) return b;
    // |a| >= |b|
    if (a.e < b.e || (a.e == b.e && a.m < b.m)) {
        const sf64 t = a;
        a = b;
        b = t;
    }
    // 10 guard bits: X < 2^63, Y aligned to X with sticky
    const sf_i32 d = a.e - b.e;
    const sf_u64 X = a.m << 10;
    const sf_u64 B = b.m << 10;
    sf_u64 Y = 0;
    bool sticky = true;
    if (d < 64) {
        Y = B >> d;
        sticky = d > 0 && (B & ((((sf_u64)1) << d) - 1)) != 0;
    }
    if (a.s == b.s) return sf_round64(X + Y, a.e - 10, sticky, a.s);
    // with sticky, d > 10 and X - Y > 2^61: enough bits for rounding
    return sf_round64(X - Y - (sticky ? 1 : 0), a.e - 10, sticky, a.s);
}

static inline sf64 sf_sub(sf64 a, sf64 b)
{
    return sf_add(a, sf_neg(b));
}

static inline sf64 sf_mul(sf64 a, sf64 b)
{
    if (!a.m || !b.m) return sf_zero();
    // product in [2^104, 2^106): round its top 56 bits plus sticky
    const sf_u64 hi = sf_mulhi(a.m, b.m), lo = a.m * b.m;
    const sf_u64 M = (hi << 14) | (lo >> 50);
    const bool sticky = (lo & ((((sf_u64)1) << 50) - 1)) != 0;
    return sf_round64(M, a.e + b.e + 50, sticky, a.s ^ b.s);
}

/** fma(a, b, c) = a * b + c with a single rounding */
static inline sf64 sf_fma(sf64 a, sf64 b, sf64 c)
{
    if (!a.m || !b.m) return c;
    return sf_add_exact(sf_mul64(a.m, b.m), a.e + b.e, a.s ^ b.s,
                        sf_u128_make(0, c.m), c.e, c.s);
}

static inline sf64 sf_div(sf64 a, sf64 b)
{
    if (!a.m) return sf_zero();
    /**
     * Long division q = floor(a.m * 2^55 / b.m), 55 or 56 bits: one leading
     * bit, then five 11 bit digits. The remainder stays below b.m < 2^53, so
     * remainder << 11 fits 64 bits. Each digit is estimated with a float
     * reciprocal: the estimate of r / b.m < 2^11 has a relative error below
     * 2^-20, so lowering it by 2^-8 yields the exact digit or one less, fixed
     * by one correction step.
     */
    const sf_u64 d = b.m;
    const float inv = 1.0f / (float)d;
    sf_u64 r = a.m, q = 0;
    if (r >= d) {
        r -= d;
        q = 1;
    }
    for (int i = 0; i < 5; i++) {
        r <<= 11;
        const float est = (float)r * inv - 0.00390625f;
        sf_u64 digit = est > 0.0f ? (sf_u64)est : 0;
        r -= digit * d;
        if (r >= d) {
            r -= d;
            digit++;
        }
        q = (q << 11) | digit;
    }
    return sf_round(sf_u128_make(0, q), a.e - b.e - 55, r != 0, a.s ^ b.s);
}

/** a < b */
static inline bool sf_lt(sf64 a, sf64 b)
{
    if (!a.m && !b.m) return false;
    if (a.s != b.s) return a.s != 0;
    const bool mag_lt = !a.m || (b.m && (a.e < b.e || (a.e == b.e && a.m < b.m)));
    const bool mag_eq = a.m == b.m && (a.e == b.e || !a.m);
    if (mag_eq) return false;
    return a.s ? !mag_lt : mag_lt;
}

/** conversion to integer, rounding toward zero (C cast semantics) */
static inline sf_i64 sf_trunc_i64(sf64 a)
{
    if (!a.m || a.e <= -53) return 0;
    const sf_u64 v = a.e >= 0 ? a.m << a.e : a.m >> -a.e;
    return a.s ? -(sf_i64)v : (sf_i64)v;
}

#endif /* __VMAF_SRC_METAL_SOFT_F64_H__ */
