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
 * SpEED chroma: the full resolution part of filter_and_downscale() in
 * speed.c, for the 4 chroma planes of a frame.
 *
 * picture_copy() converts a plane to float, minus 128. Then
 * vif_filter1d_dec16_scalar_s() filters it with the anti-aliasing filter,
 * at every 16th row and column only. One thread computes one output pixel:
 * the vertical filter at the fwidth columns that the horizontal filter
 * reads, then the horizontal filter. The CPU build contracts
 * `accum += f * x` into a fused multiply-add, which fma() reproduces.
 *
 * With a prescale, speed_chroma_metal_scale first converts and scales the
 * planes to float, as vif_scale_frame_s() does, and the filter reads them.
 *
 * kVariant = bpc (8, 10, 12 or 16) of the input, or 32 for float input.
 * In exact mode, the rest of the computation runs on the host, on 1/256 of
 * the pixels. In fast mode, the kernels at the end of this file run it in
 * float: the second filter, the covariance matrices, their eigenvalues and
 * the linear systems.
 */

#include "kernel_common.h"
#include "speed_chroma_metal.h"

constant uint kBpc = kVariant;

/** picture_copy() of one pixel */
static inline float load_px(device const uchar *in, constant SpeedMetalParams &p,
                            uint idx)
{
    if (kBpc == 32) return ((device const float *)in)[idx];
    if (kBpc == 8) return (float)in[idx] + (float)-128;
    const float v = (float)((device const ushort *)in)[idx];
    return v / p.scaler + (float)-128;
}

static inline int mirror(int i, int n)
{
    return i < 0 ? -i : (i >= n ? 2 * n - i - 2 : i);
}

/**
 * vif_scale_frame_nearest_s() and vif_scale_frame_bilinear_apply_s(): the
 * host computes the source rows and columns and the weights, with the CPU
 * functions. One thread per output pixel, p.w x p.h per plane.
 *
 * xi: first and second source column (2 x p.w), xf: weight of the second
 * yi, yf: the same for the rows (2 x p.h). nearest only uses xi and yi.
 */
kernel void speed_chroma_metal_scale(constant SpeedMetalParams &p [[buffer(0)]],
                                     device const uchar *in [[buffer(1)]],
                                     device const int *xi [[buffer(2)]],
                                     device const float *xf [[buffer(3)]],
                                     device const int *yi [[buffer(4)]],
                                     device const float *yf [[buffer(5)]],
                                     device float *out [[buffer(6)]],
                                     uint2 gid [[thread_position_in_grid]])
{
    const int x = gid.x;
    const int plane = gid.y / p.h, y = gid.y % p.h;
    if (x >= p.w || plane >= SPEED_METAL_PLANES) return;

    const uint base = plane * p.in_plane;
    const uint r1 = base + yi[y] * p.in_stride;
    float v;
    if (p.method == SPEED_METAL_SCALE_NEAREST) {
        v = load_px(in, p, r1 + xi[x]);
    } else {
        // the CPU build's contraction of the 4 products
        const uint r2 = base + yi[p.h + y] * p.in_stride;
        const int x1 = xi[x], x2 = xi[p.w + x];
        const float dx = xf[x], dy = yf[y];
        float acc = (1 - dy) * dx * load_px(in, p, r1 + x2);
        acc = fma((1 - dy) * (1 - dx), load_px(in, p, r1 + x1), acc);
        acc = fma(dy * (1 - dx), load_px(in, p, r2 + x1), acc);
        v = fma(dy * dx, load_px(in, p, r2 + x2), acc);
    }
    out[(plane * p.h + y) * p.w + x] = v;
}

kernel void speed_chroma_metal_dec16(constant SpeedMetalParams &p [[buffer(0)]],
                                     device const uchar *in [[buffer(1)]],
                                     constant float *f [[buffer(2)]],
                                     device float *out [[buffer(3)]],
                                     uint2 gid [[thread_position_in_grid]])
{
    const int j = gid.x;
    const int plane = gid.y / p.out_h, i = gid.y % p.out_h;
    if (j >= p.out_w || plane >= SPEED_METAL_PLANES) return;

    const uint base = plane * p.in_plane;
    const int r = p.fwidth / 2;
    float accum = 0;
    for (int fj = 0; fj < p.fwidth; fj++) {
        const int jj = mirror(j * 16 - r + fj, p.w);
        float tmp = 0;
        for (int fi = 0; fi < p.fwidth; fi++) {
            const int ii = mirror(i * 16 - r + fi, p.h);
            tmp = fma(f[fi], load_px(in, p, base + ii * p.in_stride + jj), tmp);
        }
        accum = fma(f[fj], tmp, accum);
    }
    out[(plane * p.out_h + i) * p.out_w + j] = accum;
}

/* -------------------------------------------------------------------------
 * Fast mode: the rest of filter_and_downscale() and est_params() on the GPU,
 * in float. The host only runs get_speed_score() on the results.
 */

/**
 * The second filter, vif_filter1d_s(), and subtract_image(): each pixel
 * minus its local mean, of the 4 planes filtered down by 16.
 */
kernel void speed_chroma_metal_local(constant SpeedMetalEstParams &p [[buffer(0)]],
                                     device const float *in [[buffer(1)]],
                                     constant float *f [[buffer(2)]],
                                     device float *out [[buffer(3)]],
                                     uint2 gid [[thread_position_in_grid]])
{
    const int j = gid.x;
    const int plane = gid.y / p.h, i = gid.y % p.h;
    if (j >= p.w || plane >= SPEED_METAL_PLANES) return;

    device const float *img = in + plane * p.w * p.h;
    const int r = p.fwidth / 2;
    float accum = 0;
    for (int fj = 0; fj < p.fwidth; fj++) {
        const int jj = mirror(j - r + fj, p.w);
        float tmp = 0;
        for (int fi = 0; fi < p.fwidth; fi++)
            tmp = fma(f[fi], img[mirror(i - r + fi, p.h) * p.w + jj], tmp);
        accum = fma(f[fj], tmp, accum);
    }
    out[(plane * p.h + i) * p.w + j] = img[i * p.w + j] - accum;
}

/**
 * The 25 means of compute_covariance_matrix(), of the submatrices at the
 * offsets of a block: one threadgroup per offset and plane.
 */
kernel void speed_chroma_metal_means(constant SpeedMetalEstParams &p [[buffer(0)]],
                                     device const float *loc [[buffer(1)]],
                                     device float *means [[buffer(2)]],
                                     uint2 tgid [[threadgroup_position_in_grid]],
                                     uint2 tg_size [[threads_per_threadgroup]],
                                     uint2 tpos [[thread_position_in_threadgroup]],
                                     uint sid [[simdgroup_index_in_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float sh[32];
    const int e = tgid.x, plane = tgid.y;
    device const float *d = loc + plane * p.w * p.h +
                            (e / SPEED_METAL_BS) * p.w + e % SPEED_METAL_BS;
    float a = 0;
    for (int i = tpos.y; i < p.sub_h; i += tg_size.y) {
        for (int j = tpos.x; j < p.sub_w; j += tg_size.x)
            a += d[i * p.w + j];
    }
    a = simd_sum(a);
    if (lane == 0) sh[sid] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tpos.x == 0 && tpos.y == 0) {
        float t = 0;
        for (uint k = 0; k < (tg_size.x * tg_size.y + 31) / 32; k++) t += sh[k];
        means[plane * SPEED_METAL_E + e] = t / (p.sub_w * p.sub_h);
    }
}

/**
 * compute_covariance_matrix(): one threadgroup per row x of the matrix and
 * plane, for the entries y <= x. Each thread adds up its pixels for all of
 * them, then the threadgroup adds up the threads.
 */
kernel void speed_chroma_metal_cov(constant SpeedMetalEstParams &p [[buffer(0)]],
                                   device const float *loc [[buffer(1)]],
                                   device const float *means [[buffer(2)]],
                                   device float *cov [[buffer(3)]],
                                   uint2 tgid [[threadgroup_position_in_grid]],
                                   uint2 tg_size [[threads_per_threadgroup]],
                                   uint2 tpos [[thread_position_in_threadgroup]],
                                   uint sid [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float sh[32][SPEED_METAL_E];
    const int xi = tgid.x, plane = tgid.y;
    device const float *data = loc + plane * p.w * p.h;
    device const float *m = means + plane * SPEED_METAL_E;
    device const float *dx = data + (xi / SPEED_METAL_BS) * p.w +
                             xi % SPEED_METAL_BS;
    const float mx = m[xi];

    float c[SPEED_METAL_E];
#pragma unroll
    for (int y = 0; y < SPEED_METAL_E; y++) c[y] = 0;
    for (int i = tpos.y; i < p.sub_h; i += tg_size.y) {
        for (int j = tpos.x; j < p.sub_w; j += tg_size.x) {
            const float a = dx[i * p.w + j] - mx;
            device const float *row = data + i * p.w + j;
            // fixed bounds: c stays in registers
#pragma unroll
            for (int y = 0; y < SPEED_METAL_E; y++) {
                if (y <= xi) {
                    const float b = row[(y / SPEED_METAL_BS) * p.w +
                                        y % SPEED_METAL_BS] - m[y];
                    c[y] = fma(a, b, c[y]);
                }
            }
        }
    }
#pragma unroll
    for (int y = 0; y < SPEED_METAL_E; y++) {
        const float v = simd_sum(c[y]);
        if (lane == 0) sh[sid][y] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint tid = tpos.y * tg_size.x + tpos.x;
    const uint n_sg = (tg_size.x * tg_size.y + 31) / 32;
    if ((int)tid <= xi) {
        float t = 0;
        for (uint k = 0; k < n_sg; k++) t += sh[k][tid];
        t /= p.sub_w * p.sub_h;
        device float *mat = cov + plane * SPEED_METAL_E * SPEED_METAL_E;
        mat[xi * SPEED_METAL_E + tid] = t;
        mat[tid * SPEED_METAL_E + xi] = t;
    }
}

/*
 * The eigenvalues of the covariance matrix: compute_eigenvalues() of
 * speed.c (adapted from GSL), in float, on threadgroup memory. The
 * reduction to tridiagonal form runs on the 32 lanes of a simdgroup, one
 * row per lane; the QR iterations on one thread.
 */

#define TG threadgroup
#define EIG_N SPEED_METAL_E
#define EIG_EPS 1e-6f
#define EIG_MAX_ITERS 500

static inline int get_sign(float x)
{
    return x >= 0 ? 1 : -1;
}

static inline float pythagoras(float x, float y)
{
    return sqrt(x * x + y * y);
}

static inline void simd_sync()
{
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

/** convert_to_tridiagonal(), with compute_householder_transform() */
static void convert_to_tridiagonal(TG float *A, TG float *d, TG float *sd,
                                   TG float *v, TG float *x, int r)
{
    for (int i = 0; i < EIG_N - 2; i++) {
        const int s = i + 1; // the rows and columns [s, N)
        // the Householder transform that annihilates A[i+2:N, i]
        float xn = r > s && r < EIG_N ? A[r * EIG_N + i] : 0.0f;
        const float xnorm = sqrt(simd_sum(xn * xn));
        const float alpha = A[s * EIG_N + i];
        float tau = 0.0f;
        if (xnorm != 0.0f) {
            const float beta = -get_sign(alpha) * pythagoras(alpha, xnorm);
            tau = (beta - alpha) / beta;
            const float sc = alpha - beta;
            simd_sync();
            if (sc != 0.0f && r >= s && r < EIG_N)
                A[r * EIG_N + i] = r == s ? beta : A[r * EIG_N + i] / sc;
        }
        simd_sync();
        if (r >= s && r < EIG_N) v[r] = A[r * EIG_N + i];
        simd_sync();
        if (tau == 0.0f) continue;
        if (r == s) v[s] = 1.0f;
        simd_sync();
        // x = tau A v
        float xr = 0.0f;
        if (r >= s && r < EIG_N) {
            for (int c = s; c < EIG_N; c++) xr += tau * A[r * EIG_N + c] * v[c];
        }
        // x -= 0.5 tau (x . v) v
        const float xv = simd_sum(r >= s && r < EIG_N ? xr * v[r] : 0.0f);
        const float a2 = -0.5f * tau * xv;
        if (r >= s && r < EIG_N) x[r] = xr + a2 * v[r];
        simd_sync();
        // A -= x v' + v x'
        if (r >= s && r < EIG_N) {
            for (int c = s; c < EIG_N; c++)
                A[r * EIG_N + c] -= x[r] * v[c] + v[r] * x[c];
        }
        simd_sync();
    }
    if (r < EIG_N) d[r] = A[r * EIG_N + r];
    if (r < EIG_N - 1) sd[r] = A[(r + 1) * EIG_N + r];
    simd_sync();
}

static void chop_small_elements(TG float *d, TG float *sd, int n)
{
    for (int i = 0; i < n - 1; i++) {
        if (fabs(sd[i]) < EIG_EPS * (fabs(d[i]) + fabs(d[i + 1])))
            sd[i] = 0.0f;
    }
}

static float trailing_eigenvalue(TG float *d, TG float *sd, int n)
{
    const float ta = d[n - 2], tb = d[n - 1], tab = sd[n - 2];
    const float dt = (ta - tb) / 2.0f;
    if (dt > 0) return tb - tab * (tab / (dt + pythagoras(dt, tab)));
    if (dt == 0) return tb - fabs(tab);
    return tb + tab * (tab / (-dt + pythagoras(dt, tab)));
}

static void create_givens(float a, float b, thread float &c, thread float &s)
{
    if (b == 0) {
        c = 1;
        s = 0;
    } else if (fabs(b) > fabs(a)) {
        const float t = -a / b;
        const float s1 = 1.0f / sqrt(1 + t * t);
        s = s1;
        c = s1 * t;
    } else {
        const float t = -b / a;
        const float c1 = 1.0f / sqrt(1 + t * t);
        c = c1;
        s = c1 * t;
    }
}

static void qr_step(TG float *d, TG float *sd, int n)
{
    float mu = trailing_eigenvalue(d, sd, n);
    if (EIG_EPS * fabs(mu) > fabs(d[0]) + fabs(sd[0])) mu = 0;

    float x = d[0] - mu, z = sd[0];
    float ak = 0, bk = 0, zk = 0;
    float ap = d[0], bp = sd[0], aq = d[1];

    if (n == 2) {
        float c, s;
        create_givens(x, z, c, s);
        d[0] = c * (c * ap - s * bp) + s * (s * aq - c * bp);
        sd[0] = c * (s * ap + c * bp) - s * (s * bp + c * aq);
        d[1] = s * (s * ap + c * bp) + c * (s * bp + c * aq);
        return;
    }

    float bq = sd[1];
    for (int k = 0; k < n - 1; k++) {
        float c, s;
        create_givens(x, z, c, s);

        const float bk1 = c * bk - s * zk;
        const float ap1 = c * (c * ap - s * bp) + s * (s * aq - c * bp);
        const float bp1 = c * (s * ap + c * bp) - s * (s * bp + c * aq);
        const float zp1 = -s * bq;
        const float aq1 = s * (s * ap + c * bp) + c * (s * bp + c * aq);
        const float bq1 = c * bq;

        ak = ap1;
        bk = bp1;
        zk = zp1;
        ap = aq1;
        bp = bq1;

        if (k < n - 2) aq = d[k + 2];
        if (k < n - 3) bq = sd[k + 2];
        d[k] = ak;
        if (k > 0) sd[k - 1] = bk1;
        if (k < n - 2) sd[k + 1] = bp;
        x = bk;
        z = zk;
    }
    d[n - 1] = ap;
    sd[n - 2] = bk;
}

static void eigenvalues_tridiagonal(TG float *d, TG float *sd)
{
    chop_small_elements(d, sd, EIG_N);
    int b = EIG_N - 1, iter = 0;
    while (b > 0 && iter < EIG_MAX_ITERS) {
        if (sd[b - 1] == 0.0f) {
            b--;
            continue;
        }
        int a = b - 1;
        while (a > 0 && sd[a - 1] != 0.0f) a--;
        qr_step(d + a, sd + a, b - a + 1);
        chop_small_elements(d + a, sd + a, b - a + 1);
        iter++;
    }
}

/**
 * est_params() of one plane per threadgroup, after the covariance matrix:
 * the eigenvalues (simdgroup 0), the Cholesky factor of the covariance
 * matrix (simdgroup 1, in parallel), then per block: the solution of the
 * linear system, the variance S and the entropy E.
 *
 * The CPU solves the system with a QR decomposition, if all eigenvalues
 * are at least 1e-6. The matrix is then positive definite, and Cholesky
 * solves the same system with less work.
 *
 * out per plane: S (nb), E (nb), then 1 if the matrix is singular.
 */
kernel void speed_chroma_metal_est(constant SpeedMetalEstParams &p [[buffer(0)]],
                                   device const float *loc [[buffer(1)]],
                                   device const float *cov [[buffer(2)]],
                                   device float *out [[buffer(3)]],
                                   uint2 tgid [[threadgroup_position_in_grid]],
                                   uint2 tg_size [[threads_per_threadgroup]],
                                   uint2 tpos [[thread_position_in_threadgroup]],
                                   uint sid [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float A[EIG_N * EIG_N], L[EIG_N * EIG_N];
    threadgroup float d[EIG_N], sd[EIG_N], v[EIG_N], x[EIG_N];
    threadgroup bool chol_ok[1]; // written by simdgroup 1
    const uint tid = tpos.x, n = tg_size.x;
    const int plane = tgid.x;
    device const float *K = cov + plane * EIG_N * EIG_N;
    device const float *data = loc + plane * p.w * p.h;

    for (uint k = tid; k < EIG_N * EIG_N; k += n) A[k] = K[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sid == 0) {
        convert_to_tridiagonal(A, d, sd, v, x, lane);
        if (lane == 0) eigenvalues_tridiagonal(d, sd);
    } else if (sid == 1) {
        // Cholesky, lane i computes row i of each column
        const int i = lane;
        bool ok = true;
        for (int j = 0; j < EIG_N; j++) {
            float t = i < EIG_N ? K[i * EIG_N + j] : 0.0f;
            for (int k = 0; k < j; k++) {
                if (i < EIG_N) t -= L[i * EIG_N + k] * L[j * EIG_N + k];
            }
            const float djj = simd_shuffle(t, j); // row j: the pivot
            if (!(djj > 0.0f)) {
                ok = false;
                break;
            }
            const float ljj = sqrt(djj);
            if (i == j) L[j * EIG_N + j] = ljj;
            else if (i > j && i < EIG_N) L[i * EIG_N + j] = t / ljj;
            simd_sync();
        }
        if (lane == 0) chol_ok[0] = ok;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    bool singular = !chol_ok[0];
    for (int k = 0; k < EIG_N; k++) singular |= d[k] < EIG_EPS;

    device float *S_out = out + plane * (2 * p.nb + 1);
    device float *E_out = S_out + p.nb;
    for (int b = tid; b < p.nb; b += n) {
        // compute_independent_term(): the block's 25 pixels
        float y[EIG_N], z[EIG_N];
        device const float *blk = data + (b / p.nbh) * SPEED_METAL_BS * p.w +
                                  (b % p.nbh) * SPEED_METAL_BS;
        for (int i = 0; i < EIG_N; i++)
            y[i] = blk[(i / SPEED_METAL_BS) * p.w + i % SPEED_METAL_BS];

        float S = 0;
        if (!singular) {
            // L L' z = y
            for (int i = 0; i < EIG_N; i++) {
                float t = y[i];
                for (int k = 0; k < i; k++) t -= L[i * EIG_N + k] * z[k];
                z[i] = t / L[i * EIG_N + i];
            }
            for (int i = EIG_N - 1; i >= 0; i--) {
                float t = z[i];
                for (int k = i + 1; k < EIG_N; k++) t -= L[k * EIG_N + i] * z[k];
                z[i] = t / L[i * EIG_N + i];
            }
            // compute_pointwise_product_and_division(), sum_columns()
            for (int i = 0; i < EIG_N; i++) S += z[i] * y[i] / EIG_N;
        }
        // update_entropy() for each eigenvalue
        float E = 0;
        for (int k = 0; k < EIG_N; k++) {
            const float ev = d[k] < 0 ? 0 : d[k];
            E += log2(ev * S + p.sigma_nn) + p.log2_2pie;
        }
        S_out[b] = S;
        E_out[b] = E;
    }
    if (tid == 0) E_out[p.nb] = singular ? 1.0f : 0.0f;
}
