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
 * SpEED chroma (speed_chroma in speed.c) with the full resolution filtering
 * on Metal, see speed_chroma.metal.
 *
 * The GPU converts the 4 chroma planes of a frame to float and filters
 * them down by 16. When the frame completes, the host runs the rest of
 * speed_extract_score() on the small planes, with the CPU code: the second
 * Gaussian filter, the local mean subtraction, and est_params() (the
 * covariance matrix, its eigenvalues and the linear system of each plane).
 * Every step is the CPU's, so the scores are bit-identical.
 *
 * Fast mode runs these steps on the GPU too, in float (speed_chroma.metal,
 * encode_est()). The host only runs get_speed_score().
 */

#import <Metal/Metal.h>

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "cpu.h"
#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "mem.h"
#include "metal/common.h"
#include "opt.h"
#include "speed.h"
#include "vif_tools.h"
#if ARCH_AARCH64
#include "arm64/speed_neon.h"
#endif
#if ARCH_X86
#include "x86/cpu.h"
#include "x86/speed_avx2.h"
#if HAVE_AVX512
#include "x86/speed_avx512.h"
#endif
#endif

#include "speed_chroma_metal.h"

// Foundation's, replaced by speed.c's below
#undef MIN
#undef MAX

/* -------------------------------------------------------------------------
 * Copied from speed.c: the types, est_params() with its helpers,
 * get_speed_score(), subtract_image() and speed_init_dimensions().
 */

typedef double (*compute_cov_kernel_fn)(const float *data_x, const float *data_y,
                                        size_t stride_px, size_t height,
                                        size_t width, double mean_x,
                                        double mean_y);

typedef struct SpeedDimensions {
    size_t original_height;
    size_t original_width;
    size_t scaled_height;
    size_t scaled_width;
    size_t alloc_height;
    size_t alloc_width;
    size_t operating_height;
    size_t operating_width;
    size_t block_size;
    size_t truncated_width;
    size_t truncated_height;
    size_t num_blocks_horizontal;
    size_t num_blocks_vertical;
    size_t num_blocks;
    size_t elements_in_block;
    size_t submatrix_width;
    size_t submatrix_height;
} SpeedDimensions;

typedef struct SpeedResultBuffers {
    float *entropies;
    float *variances;
} SpeedResultBuffers;

typedef struct SpeedBuffers {
    float *independent_term;
    float *linear_system_sol;
    float *cov_mat;
    float *eigenvalues;
    float *tmp_buffer;
    // Bilinear column table for the (fixed, resolution-derived) prescale of
    // this feature extractor instance. Populated once in speed_init() and
    // reused for every frame instead of being recomputed per call. NULL
    // when the configured scaling method isn't bilinear.
    int *bilinear_x1a;
    int *bilinear_x2a;
    float *bilinear_dxa;
} SpeedBuffers;

// Everything that is passed in as a feature option and is needed for
// SpEED computation
typedef struct SpeedOptions {
    double speed_kernelscale;
    double speed_prescale;
    char *speed_prescale_method;
    double speed_sigma_nn;
    double speed_nn_floor;
    int speed_weight_var_mode;
} SpeedOptions;

// Everything that is needed to compute SpEED given a pair of float buffers
// (ref, dis), except for what is provided in SpeedOptions
typedef struct SpeedState {
    SpeedDimensions dimensions;
    SpeedResultBuffers ref_results;
    SpeedResultBuffers dis_results;
    SpeedBuffers buffers;
    size_t float_stride;
    compute_cov_kernel_fn compute_cov_kernel;
} SpeedState;

#define DEFAULT_BLOCK_SIZE (5)
#define NUM_SQUARE_BUFFERS (5)
#define NUM_RECT_BUFFERS (1)
#define NUM_FRAME_BUFFERS (2)
#define NUM_SCALES (4)
#define EIGENVALUE_EPS (1e-6)
#define EIGENVALUE_MAX_ITERS (500)
#define ALMOST_EQUAL(x,c) (fabs((x) - (c)) < 1.0e-3)
#define MAX(x, y) ((x) > (y) ? (x) : (y))

typedef struct Matrix {
    int rows;
    int cols;
    float *data;
} Matrix;

static void matrix_init(Matrix *mat, int rows, int cols, float *buffer)
{
    mat->data = buffer;
    mat->rows = rows;
    mat->cols = cols;
}

static void matrix_transpose(Matrix *m)
{
    // only works for square matrices due to being an in-place operation
    assert(m->rows == m->cols);
    int size = m->rows;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < i; j++) {
            float temp = m->data[i * size + j];
            m->data[i * size + j] = m->data[j * size + i];
            m->data[j * size + i] = temp;
        }
    }
}

static void matrix_zero(Matrix *m)
{
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->data[i * m->cols + j] = 0;
        }
    }
}

static void matrix_identity(Matrix *m)
{
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            m->data[i * m->cols + j] = (i == j ? 1 : 0);
        }
    }
}

static void matrix_copy(Matrix *dst, const Matrix *src)
{
    dst->rows = src->rows;
    dst->cols = src->cols;
    for (int i = 0; i < dst->rows; i++) {
        for (int j = 0; j < dst->cols; j++) {
            dst->data[i * dst->cols + j] = src->data[i * src->cols + j];
        }
    }
}

static void matrix_mul(Matrix *dst, const Matrix *x, const Matrix *y)
{
    assert(x->cols == y->rows);
    matrix_zero(dst);
    for (int i = 0; i < x->rows; i++) {
        for (int k = 0; k < x->cols; k++) {
            for (int j = 0; j < y->cols; j++) {
                dst->data[i * dst->cols + j] +=
                    x->data[i * x->cols + k] * y->data[k * y->cols + j];
            }
        }
    }
}

static void matrix_minor(Matrix *mat, int d)
{
    // only works for square matrices
    assert(mat->rows == mat->cols);
    int size = mat->rows;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i < d || j < d) {
                if (i == j)
                    mat->data[i * size + j] = 1;
                else
                    mat->data[i * size + j] = 0;
            }
        }
    }
}

static void matrix_identity_minus_v_vt(Matrix *dst, float *v)
{
    // dst = I - v v^T
    // only works for square matrices
    assert(dst->rows == dst->cols);
    int size = dst->rows;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            dst->data[i * size + j] = -2 * v[i] * v[j];
        }
    }

    for (int i = 0; i < size; i++)
        dst->data[i * size + i] += 1;
}

static float vector_norm(const float *x, int n)
{
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += x[i] * x[i];
    return sqrt(sum);
}

static void vector_div(const float *x, float d, float *y, int n)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] / d;
}

static void matrix_take_kth_column(const Matrix *mat, float *v, int k)
{
    //take k-th column of m, put in v
    for (int i = 0; i < mat->rows; i++)
        v[i] = mat->data[i * mat->cols + k];
}

static int get_sign(float x)
{
    return (x >= 0 ? 1 : -1);
}

static float pythagoras(float x, float y)
{
    //hypotenuse length of the right triangle with sides x and y
    return sqrt(x * x + y * y);
}

static float compute_column_norm(float *A, int col, int start_row, int size)
{
    // euclidean norm of the column vector A[start_row:size, col]
    float norm = 0;
    for (int i = start_row; i < size; i++)
        norm += A[i * size + col] * A[i * size + col];
    return sqrt(norm);
}

// Compute a householder transformation (tau,v) of a vector
// x so that P x = [ I - tau*v*v' ] x annihilates x(1:n-1)
//
// On output, v is normalized so that v[0] = 1. The 1 is
// not actually stored; instead v[0] = -sign(x[0])*||x|| so
// that:
//              P x = v[0] * e_1

static float compute_householder_transform(float *A, int col, int start_row,
                                           int size)
{
    if (size - start_row == 1)
        return 0.0f;
    float xnorm = compute_column_norm(A, col, start_row + 1, size);
    if (xnorm == 0.0f)
        return 0.0f;
    float alpha = A[start_row * size + col];
    float beta = -get_sign(alpha) * pythagoras(alpha, xnorm);
    float tau = (beta - alpha) / beta;
    float s = alpha - beta;
    if (s != 0.0f) {
        for (int i = start_row; i < size; i++)
            A[i * size + col] /= s;
        A[start_row * size + col] = beta;
    }
    return tau;
}

// x = tau_i + A * v
// NOTE: only the rows and columns [start:size] of the involved matrices and
// vectors are used
static void tridiagonal_multiply(float *A, float *v, float *x, float tau_i,
                                 int start, int size)
{
    for (int i = start; i < size; i++) {
        x[i] = 0.0f;
        for (int j = start; j < size; j++)
            x[i] += tau_i * A[i * size + j] * v[j];
    }
}

// returns dot(x, v)
// NOTE: only the rows and columns [start:size] of the involved matrices and
// vectors are used
static float tridiagonal_dot_product(float *x, float *v, int start, int size)
{
    float res = 0;
    for (int i = start; i < size; i++)
        res += x[i] * v[i];
    return res;
}

// x += alpha * v
// NOTE: only the rows and columns [start:size] of the involved matrices and
// vectors are used
static void tridiagonal_axpy(float *x, float *v, float alpha, int start, int size)
{
    for (int i = start; i < size; i++)
        x[i] += alpha * v[i];
}

// A -= x * v' + v * x'
// NOTE: only the rows and columns [start:size] of the involved matrices and
// vectors are used
static void tridiagonal_syr2(float *A, float *x, float *v, int start, int size)
{
    for (int i = start; i < size; i++) {
        for (int j = start; j < size; j++) {
            A[i * size + j] -= x[i] * v[j] + v[i] * x[j];
        }
    }
}

// Adapted gsl_linalg_symmtd_decomp:
// https://github.com/ampl/gsl/blob/master/linalg/symmtd.c
static void convert_to_tridiagonal(float *A, int size, float *d, float *sd,
                                   float *buffer)
{
    float *v = buffer; buffer += size;
    float *x = buffer; buffer += size;

    // We apply N-2 Householder transformations to zero out the elements
    // outside of the diagonal or subdiagonal of each of the N-2 first columns
    for (int i = 0; i < size - 2; i++) {
        // Compute the vector v of the Householder transform that
        // annihilates A[i+2:size, i]
        float tau_i = compute_householder_transform(A, i, i + 1, size);

        // Copy i'th subcolumn of a into v
        for (int j = i + 1; j < size; j++)
            v[j] = A[j * size + i];

        if (tau_i != 0.0f) {
            // Set the first element of the returned vector to 1,
            // since compute_householder_transform does this implicitly
            A[(i + 1) * size + i] = v[i + 1];
            v[i + 1] = 1.0f;
            // All operations described here are applied only to the rows and
            // columns in [i+1:size]
            // x = tau_i * A * v
            tridiagonal_multiply(A, v, x, tau_i, i + 1, size);
            // x -= 0.5 * tau_i * dot(x, v) * v]
            float xv = tridiagonal_dot_product(x, v, i + 1, size);
            float alpha = -0.5 * tau_i * xv;
            tridiagonal_axpy(x, v, alpha, i + 1, size);
            // A = A - v * x' - x * v'
            tridiagonal_syr2(A, x, v, i + 1, size);
        }
    }

    // Copy the diagonal and subdiagonal elements into d and sd, respectively
    for (int i = 0; i < size; i++)
        d[i] = A[i * size + i];
    for (int i = 0; i < size - 1; i++)
        sd[i] = A[(i + 1) * size + i];
}

static void chop_small_elements(float *d, float *sd, int size)
{
    for (int i = 0; i < size - 1; i++) {
        if (fabsf(sd[i]) < EIGENVALUE_EPS * (fabsf(d[i]) + fabsf(d[i + 1])))
            sd[i] = 0.0f;
    }
}

static float trailing_eigenvalue(float *d, float *sd, int n)
{
    float ta = d[n - 2];
    float tb = d[n - 1];
    float tab = sd[n - 2];
    float dt = (ta - tb) / 2.0;

    if (dt > 0)
        return tb - tab * (tab / (dt + pythagoras(dt, tab)));
    else if (dt == 0)
        return tb - fabsf(tab);
    else
        return tb + tab * (tab / (-dt + pythagoras(dt, tab)));
}

static void create_givens(const float a, const float b, float *c, float *s)
{
    if (b == 0) {
        *c = 1;
        *s = 0;
    } else if (fabsf(b) > fabsf(a)) {
        float t = -a / b;
        float s1 = 1.0 / sqrt(1 + t * t);
        *s = s1;
        *c = s1 * t;
    } else {
        float t = -b / a;
        float c1 = 1.0 / sqrt(1 + t * t);
        *c = c1;
        *s = c1 * t;
    }
}

static void qr_step(float *d, float *sd, int n)
{
    float mu = trailing_eigenvalue(d, sd, n);
    if (EIGENVALUE_EPS * fabsf(mu) > fabsf(d[0]) + fabsf(sd[0]))
        mu = 0;

    float x = d[0] - mu;
    float z = sd[0];

    float ak = 0;
    float bk = 0;
    float zk = 0;

    float ap = d[0];
    float bp = sd[0];

    float aq = d[1];

    if (n == 2) {
        float c, s;
        create_givens(x, z, &c, &s);

        float ap1 = c * (c * ap - s * bp) + s * (s * aq - c * bp);
        float bp1 = c * (s * ap + c * bp) - s * (s * bp + c * aq);
        float aq1 = s * (s * ap + c * bp) + c * (s * bp + c * aq);

        ak = ap1;
        bk = bp1;

        ap = aq1;

        d[0] = ak;
        sd[0] = bk;
        d[1] = ap;

        return;
    }

    float bq = sd[1];

    for (int k = 0; k < n - 1; k++) {
        float c, s;
        create_givens(x, z, &c, &s);

        /* compute G' T G */
        float bk1 = c * bk - s * zk;

        float ap1 = c * (c * ap - s * bp) + s * (s * aq - c * bp);
        float bp1 = c * (s * ap + c * bp) - s * (s * bp + c * aq);
        float zp1 = -s * bq;

        float aq1 = s * (s * ap + c * bp) + c * (s * bp + c * aq);
        float bq1 = c * bq;

        ak = ap1;
        bk = bp1;
        zk = zp1;

        ap = aq1;
        bp = bq1;

        if (k < n - 2)
            aq = d[k + 2];

        if (k < n - 3)
            bq = sd[k + 2];

        d[k] = ak;

        if (k > 0)
            sd[k - 1] = bk1;

        if (k < n - 2) {
            sd[k + 1] = bp;
        }

        x = bk;
        z = zk;
    }

    d[n - 1] = ap;
    sd[n - 2] = bk;
}

// Adapted gsl_eigen_symm: https://github.com/ampl/gsl/blob/master/eigen/symm.c
static void compute_eigenvalues_tridiagonal(float *d, float *sd,
                                            float *eigenvalues, int size)
{
    // Initial pass to remove subdiagonal elements which are effectively zero
    chop_small_elements(d, sd, size);

    // Progressively reduce the matrix until it is diagonal
    int b = size - 1;
    int iter = 0;
    while (b > 0 && iter < EIGENVALUE_MAX_ITERS) {
        if (sd[b - 1] == 0.0f) {
            b--;
            continue;
        }
        // Find the largest unreduced block (a, b) starting from b and working
        // backwards
        int a = b - 1;
        while (a > 0) {
            if (sd[a - 1] == 0.0f) {
                break;
            }
            a--;
        }

        const int n_block = b - a + 1;
        float *d_block = d + a;
        float *sd_block = sd + a;
        qr_step(d_block, sd_block, n_block);
        chop_small_elements(d_block, sd_block, n_block);
        iter++;
    }

    if (iter == EIGENVALUE_MAX_ITERS) {
        vmaf_log(VMAF_LOG_LEVEL_WARNING,
                 "compute_eigenvalues_tridiagonal: max iterations reached, "
                 "possible non-convergence\n");
    }

    for (int i = 0; i < size; i++)
        eigenvalues[i] = d[i];
}

static void compute_eigenvalues(float *A_immutable, float *eigenvalues,
                                int size, float *buffer)
{
    float *A = buffer; buffer += size * size;
    float *d = buffer; buffer += size;
    float *sd = buffer; buffer += size;
    float *tmp = buffer; buffer += 2 * size;
    // Operate on a copy of the matrix
    memcpy(A, A_immutable, size * size * sizeof(float));
    // Handle special case
    if (size == 1) {
        eigenvalues[0] = A[0 * size + 0];
        return;
    }
    convert_to_tridiagonal(A, size, d, sd, tmp);
    compute_eigenvalues_tridiagonal(d, sd, eigenvalues, size);
}

// Implementation of the QR decomposition algorithm with Householder
// reflections for an arbitrary square matrix
// https://www.cs.utexas.edu/users/flame/Notes/NotesOnHouseholderQR.pdf
static void matrix_qr_decomposition(Matrix *A, Matrix *Q, Matrix *R,
                                    Matrix *tmp_q, Matrix *tmp_z)
{
    assert(A->rows == A->cols);
    int size = A->rows;

    // We need 3 temporary matrices. We can use R as a temporary matrix during
    // the process since it's only used at the end. We can also use R for the
    // temporary vector, since both uses are disjoint in time
    Matrix *tmp_mul = R;
    float *vec = R->data;

    // Initially, tmp_z = A
    matrix_copy(tmp_z, A);

    // Q starts as the identity, and is left-multiplied by tmp_q at each iteration
    matrix_identity(Q);

    for (int k = 0; k < size - 1; k++) {
        matrix_minor(tmp_z, k);
        matrix_take_kth_column(tmp_z, vec, k);

        float norm = vector_norm(vec, size);
        int sign = get_sign(A->data[k * size + k]);
        vec[k] += sign * norm;

        vector_div(vec, vector_norm(vec, size), vec, size);
        matrix_identity_minus_v_vt(tmp_q, vec);

        matrix_mul(tmp_mul, tmp_q, tmp_z);
        matrix_copy(tmp_z, tmp_mul);
        matrix_mul(tmp_mul, tmp_q, Q);
        matrix_copy(Q, tmp_mul);
    }

    matrix_mul(R, Q, A);
    matrix_transpose(Q);
}

// Solves RX = B, where R is square, upper triangular and invertible
// Uses backward substitution algorithm
static int solve_triangular_system(const Matrix *R, Matrix *X, const Matrix *B)
{
    assert(R->rows == R->cols);
    assert(R->rows == X->rows);
    assert(R->rows == B->rows);
    assert(X->cols == B->cols);

    for (int i = R->rows - 1; i >= 0; i--) {
        float denominator = R->data[i * R->cols + i];
        if (fabsf(denominator) < EIGENVALUE_EPS) {
            return -EINVAL;
        }
        for (int j = 0; j < X->cols; j++) {
            float independent_term = B->data[i * B->cols + j];
            for (int k = i + 1; k < R->rows; k++) {
                independent_term -=
                    (X->data[k * X->cols + j] * R->data[i * R->cols + k]);
            }
            X->data[i * X->cols + j] = independent_term / denominator;
        }
    }
    return 0;
}

// Solves the linear system A X = B, using the QR decomposition of A
static int solve_linear_system(float *A_data, int A_size, float *B_data,
                               int B_cols, float *output_data,
                               float *tmp_buffer)
{
    float *A_buffer = tmp_buffer; tmp_buffer += A_size * A_size;
    float *Q_buffer = tmp_buffer; tmp_buffer += A_size * A_size;
    float *R_buffer = tmp_buffer; tmp_buffer += A_size * A_size;
    float *tmp1_buffer = tmp_buffer; tmp_buffer += A_size * A_size;
    float *tmp2_buffer = tmp_buffer; tmp_buffer += A_size * A_size;
    float *tmp_rect_buffer = tmp_buffer; tmp_buffer += A_size * B_cols;

    Matrix A_immutable; matrix_init(&A_immutable, A_size, A_size, A_data);
    Matrix A; matrix_init(&A, A_size, A_size, A_buffer);
    matrix_copy(&A, &A_immutable);

    Matrix Q; matrix_init(&Q, A_size, A_size, Q_buffer);
    Matrix R; matrix_init(&R, A_size, A_size, R_buffer);
    Matrix tmp1; matrix_init(&tmp1, A_size, A_size, tmp1_buffer);
    Matrix tmp2; matrix_init(&tmp2, A_size, A_size, tmp2_buffer);

    Matrix B_immutable; matrix_init(&B_immutable, A_size, B_cols, B_data);

    Matrix tmp_rect; matrix_init(&tmp_rect, A_size, B_cols, tmp_rect_buffer);
    Matrix X; matrix_init(&X, A_size, B_cols, output_data);

    // A = Q R, where Q^{-1} = Q^T and R is upper triangular
    // A X = B  ==>  Q R X = B  ==>  R X = Q^T B
    matrix_qr_decomposition(&A, &Q, &R, &tmp1, &tmp2);
    matrix_transpose(&Q);
    matrix_mul(&tmp_rect, &Q, &B_immutable);
    int err = solve_triangular_system(&R, &X, &tmp_rect);
    return err;
}

static float compute_mean(SpeedDimensions dim, const float *data,
                          size_t stride_px, int start_row, int start_col)
{
    float result = 0;
    for (size_t i = 0; i < dim.submatrix_height; i++) {
        for (size_t j = 0; j < dim.submatrix_width; j++)
            result += data[(start_row + i) * stride_px + (start_col + j)];
    }

    return result / (dim.submatrix_width * dim.submatrix_height);
}

// Computes the covariance between two arrays of the same given size
// The arrays are stored as submatrices of the input data, starting at
// (start_row_x, start_col_x) and (start_row_y, start_col_y)
// and having dimensions (submatrix_height, submatrix_width)
static float compute_covariance(SpeedDimensions dim, const float *data,
                                const float *means, size_t stride_px,
                                int start_row_x, int start_col_x,
                                int start_row_y, int start_col_y,
                                compute_cov_kernel_fn kernel)
{
    double mean_x = means[start_row_x * dim.block_size + start_col_x];
    double mean_y = means[start_row_y * dim.block_size + start_col_y];
    const float *data_x = data + start_row_x * stride_px + start_col_x;
    const float *data_y = data + start_row_y * stride_px + start_col_y;
    double result = kernel(data_x, data_y, stride_px,
                           dim.submatrix_height, dim.submatrix_width,
                           mean_x, mean_y);
    return result / (dim.submatrix_width * dim.submatrix_height);
}

static void compute_covariance_matrix(SpeedDimensions dim, const float *data,
                                      float *cov_mat, float *means,
                                      size_t stride_px,
                                      compute_cov_kernel_fn kernel)
{
    for (size_t start_row = 0; start_row < dim.block_size; start_row++) {
        for (size_t start_col = 0; start_col < dim.block_size; start_col++) {
            means[start_row * dim.block_size + start_col] =
                compute_mean(dim, data, stride_px, start_row, start_col);
        }
    }
    size_t elements_in_block = dim.block_size * dim.block_size;

    for (size_t x_index = 0; x_index < dim.elements_in_block; x_index++) {
        for (size_t y_index = 0; y_index <= x_index; y_index++) {
            size_t start_row_x = x_index / dim.block_size;
            size_t start_col_x = x_index % dim.block_size;
            size_t start_row_y = y_index / dim.block_size;
            size_t start_col_y = y_index % dim.block_size;
            float covariance =
                compute_covariance(dim, data, means, stride_px, start_row_x,
                                   start_col_x, start_row_y, start_col_y,
                                   kernel);
            cov_mat[x_index * elements_in_block + y_index] = covariance;
            cov_mat[y_index * elements_in_block + x_index] = covariance;
        }
    }
}

static void compute_independent_term(SpeedDimensions dim, const float *data,
                                     float *independent_term, size_t stride_px)
{
    for (size_t start_i = 0; start_i < dim.block_size; start_i++) {
        for (size_t start_j = 0; start_j < dim.block_size; start_j++) {
            for (size_t i = start_i; i < dim.truncated_height; i += dim.block_size) {
                for (size_t j = start_j; j < dim.truncated_width; j += dim.block_size) {
                    size_t out_row = start_i * dim.block_size + start_j;
                    size_t out_col =
                        (((i - start_i) / dim.block_size) *
                            dim.num_blocks_horizontal) + ((j - start_j) / dim.block_size);
                    independent_term[out_row * dim.num_blocks + out_col] =
                        data[i * stride_px + j];
                }
            }
        }
    }
}

static void compute_pointwise_product_and_division(SpeedDimensions dim,
                                                   float *X, const float *Y,
                                                   float denominator)
{
    for (size_t i = 0; i < dim.elements_in_block; i++) {
        for (size_t j = 0; j < dim.num_blocks; j++) {
            X[i * dim.num_blocks + j] =
                (X[i * dim.num_blocks + j] * Y[i * dim.num_blocks + j]) / denominator;
        }
    }
}

static void sum_columns(SpeedDimensions dim, float *X)
{
    for (size_t i = 1; i < dim.elements_in_block; i++) {
        for (size_t j = 0; j < dim.num_blocks; j++) {
            X[0 * dim.num_blocks + j] += X[i * dim.num_blocks + j];
        }
    }
}

static void update_entropy(SpeedDimensions dim, float *entropy, const float *S,
                           float L, float sigma_nn)
{
    for (size_t i = 0; i < dim.num_blocks_vertical; i++) {
        for (size_t j = 0; j < dim.num_blocks_horizontal; j++) {
            entropy[i * dim.num_blocks_horizontal + j] +=
                log2(L * S[i * dim.num_blocks_horizontal + j] + sigma_nn) +
                log2(2 * M_PI * M_E);
        }
    }
}

static bool is_matrix_regular(SpeedDimensions dim, const float *eigenvalues)
{
    for (size_t i = 0; i < dim.elements_in_block; i++) {
        if (eigenvalues[i] < EIGENVALUE_EPS) {
            return false;
        }
    }
    return true;
}

static int est_params(SpeedState *s, const float *data, float sigma_nn,
                      SpeedResultBuffers *output)
{
    SpeedDimensions dim = s->dimensions;
    size_t stride_px = s->float_stride / sizeof(float);

    // Step 1: Compute the covariance matrix K
    // We use the eigenvalues array as a temporary array to store the means
    // needed for the covariance.
    // Default to the scalar kernel if speed_init() wasn't called (e.g. unit
    // tests that construct SpeedState via struct literal).
    compute_cov_kernel_fn kernel = s->compute_cov_kernel
        ? s->compute_cov_kernel : compute_cov_kernel_scalar;
    compute_covariance_matrix(dim, data, s->buffers.cov_mat,
                              s->buffers.eigenvalues, stride_px, kernel);

    // Step 2: Compute the eigenvalues of the covariance matrix
    compute_eigenvalues(s->buffers.cov_mat, s->buffers.eigenvalues,
                        dim.elements_in_block, s->buffers.tmp_buffer);

    // Step 3: Compute independent term for the linear system
    compute_independent_term(dim, data, s->buffers.independent_term, stride_px);

    // Step 4: Solve the linear equation KX = Y, where K is the covariance
    // matrix and Y is the independent term matrix above
    int err = 0;
    bool regular = is_matrix_regular(dim, s->buffers.eigenvalues);
    if (regular) {
        err = solve_linear_system(s->buffers.cov_mat, dim.elements_in_block,
                                  s->buffers.independent_term,
                                  dim.num_blocks, s->buffers.linear_system_sol,
                                  s->buffers.tmp_buffer);
    }

    bool cannot_invert = !regular || err;
    if (cannot_invert) {
        vmaf_log(VMAF_LOG_LEVEL_WARNING,
                 "est_params: covariance matrix is singular\n");
        memset(s->buffers.linear_system_sol, 0,
               dim.elements_in_block * dim.num_blocks * sizeof(float));
    }

    // Step 5: Compute the pointwise product Z = (X * Y)/B^2, where X and Y are
    // from the linear system above, and B is the block size.
    // Store the results in s->linear_system_sol
    compute_pointwise_product_and_division(dim, s->buffers.linear_system_sol,
                                           s->buffers.independent_term,
                                           dim.elements_in_block);

    // Step 6: Sum each column in Z into an array S.
    // Store the results in the first row of s->linear_system_sol
    sum_columns(dim, s->buffers.linear_system_sol);

    // Step 7: Reshape S into a matrix of dimensions (H/B, W/B)
    // This is a no-op, we just need to index it with the right dimensions

    // Step 8: Construct a zeroed-out matrix E of size (H/B, W/B)
    memset(output->entropies, 0,
           dim.num_blocks_horizontal * dim.num_blocks_vertical * sizeof(float));

    // Step 9: For each eigenvalue L, update the entropy matrix
    for (size_t k = 0; k < dim.elements_in_block; k++) {
        float L = s->buffers.eigenvalues[k] < 0 ? 0 : s->buffers.eigenvalues[k];
        update_entropy(dim, output->entropies, s->buffers.linear_system_sol, L,
                       sigma_nn);
    }

    // Step 10: Return S, E
    memcpy(output->variances, s->buffers.linear_system_sol,
           dim.num_blocks * sizeof(float));

    return cannot_invert ? -EINVAL : 0;
}

static float get_speed_score(SpeedDimensions dim, SpeedResultBuffers ref_results,
                             SpeedResultBuffers dis_results, float sigma_nn,
                             float nn_floor, int speed_weight_var_mode)
{
    float score = 0;
    float base_entropy = dim.elements_in_block * (log2((1 + nn_floor) * sigma_nn) + log2(2 * M_PI * M_E));
    for (size_t i = 0; i < dim.num_blocks; i++) {
        if ((ref_results.entropies[i] < base_entropy) &&
            (dis_results.entropies[i] < base_entropy))
        {
            // If both entropies are below the base_entropy,
            // there is no visible difference
            score += 0;
        } else {
            float spatial_ref = 0.0;
            float spatial_dis = 0.0;
            if (speed_weight_var_mode == 0) {
                spatial_ref = ref_results.entropies[i] * log2(1 + ref_results.variances[i]);
                spatial_dis = dis_results.entropies[i] * log2(1 + dis_results.variances[i]);
            } else if (speed_weight_var_mode == 1) {
                spatial_ref = ref_results.entropies[i] * log2(1 + ref_results.variances[i]);
                spatial_dis = dis_results.entropies[i] * log2(1 + ref_results.variances[i]);
            } else if (speed_weight_var_mode == 2) {
                spatial_ref = ref_results.entropies[i] * log2(1 + dis_results.variances[i]);
                spatial_dis = dis_results.entropies[i] * log2(1 + dis_results.variances[i]);
            } else if (speed_weight_var_mode == 3) {
                spatial_ref = ref_results.entropies[i] * log2(1 + (ref_results.variances[i] + dis_results.variances[i]) / 2.0);
                spatial_dis = dis_results.entropies[i] * log2(1 + (ref_results.variances[i] + dis_results.variances[i]) / 2.0);
            } else if (speed_weight_var_mode == 4) {
                spatial_ref = ref_results.entropies[i] * log2(1 + ref_results.variances[i]);
                spatial_dis = dis_results.entropies[i] * log2(1 + (ref_results.variances[i] + dis_results.variances[i]) / 2.0);
            } else if (speed_weight_var_mode == 5) {
                spatial_ref = ref_results.entropies[i] * log2(1 + ref_results.variances[i]);
                spatial_dis = dis_results.entropies[i] * log2(1 + (0.75 * ref_results.variances[i] + 0.25 * dis_results.variances[i]));
            } else if (speed_weight_var_mode == 6) {
                spatial_ref = ref_results.entropies[i] * log2(1 + ref_results.variances[i]);
                spatial_dis = dis_results.entropies[i] * log2(1 + (0.25 * ref_results.variances[i] + 0.75 * dis_results.variances[i]));
            } else {
                return -EINVAL;
            }
            score += fabsf(spatial_ref - spatial_dis);
        }

    }

    return score / dim.num_blocks;
}

static void subtract_image(float *im1, float *im2, int w, int h, size_t stride) {
    size_t stride_px = stride / sizeof(float);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            im1[i * stride_px + j] -= im2[i * stride_px + j];
        }
    }
}

static int speed_init_dimensions(SpeedDimensions *dim, int w, int h,
                                 double speed_prescale)
{
    dim->original_height = h;
    dim->original_width = w;
    dim->scaled_height = (int)(dim->original_height * speed_prescale + 0.5);
    dim->scaled_width = (int)(dim->original_width * speed_prescale + 0.5);
    dim->alloc_height = MAX(dim->original_height, dim->scaled_height);
    dim->alloc_width = MAX(dim->original_width, dim->scaled_width);
    dim->operating_height = dim->scaled_height >> NUM_SCALES;
    dim->operating_width = dim->scaled_width >> NUM_SCALES;
    dim->block_size = DEFAULT_BLOCK_SIZE;
    dim->truncated_width =
        (dim->operating_width / dim->block_size) * dim->block_size;
    dim->truncated_height =
        (dim->operating_height / dim->block_size * dim->block_size);
    dim->num_blocks_horizontal = dim->truncated_width / dim->block_size;
    dim->num_blocks_vertical = dim->truncated_height / dim->block_size;
    dim->num_blocks = dim->num_blocks_horizontal * dim->num_blocks_vertical;
    dim->elements_in_block = dim->block_size * dim->block_size;
    dim->submatrix_width = dim->truncated_width - dim->block_size + 1;
    dim->submatrix_height = dim->truncated_height - dim->block_size + 1;

    if (dim->truncated_height == 0 || dim->truncated_width == 0) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
                 "SpEED: image too small, operating width or height is 0");
        return -EINVAL;
    }
    return 0;
}

/* -------------------------------------------------------------------------
 * Options, as in speed.c
 */

#define DEFAULT_SPEED_SIGMA_NN (0.29)
#define DEFAULT_SPEED_MAX_VAL (1000.0)
#define DEFAULT_SPEED_NN_FLOOR (0.0)
#define DEFAULT_SPEED_KERNELSCALE (1.0)
#define DEFAULT_SPEED_PRESCALE (1.0)
#define DEFAULT_SPEED_PRESCALE_METHOD ("nearest")
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

/** Host buffers of one frame slot: est_params() of the small planes. */
typedef struct SpeedSlot {
    SpeedState st;
    float *frame[2]; // ref, dis: filtered down by 16
    float *curr;     // vif_filter1d_s() output
    float *tmpbuf;
} SpeedSlot;

typedef struct SpeedChromaStateMetal {
    // options, see speed.c
    double speed_chroma_kernelscale;
    double speed_chroma_prescale;
    char *speed_chroma_prescale_method;
    double speed_chroma_sigma_nn;
    double speed_chroma_nn_floor;
    double speed_chroma_max_val;
    int speed_weight_var_mode;

    SpeedOptions speed_options;
    VmafDictionary *feature_name_dict;
    VmafFeatureExtractor *cpu; // delegate, see cpu_init()

    VmafMetalState *ms;
    VmafMetalFrames frames;
    unsigned bpc;
    SpeedMetalParams params;
    float filter[SPEED_METAL_MAX_FWIDTH]; // second filter, at 1/16
    int filter_width;
    void *in[VMAF_METAL_FRAMES];  // id<MTLBuffer>, retained: 4 planes
    void *out[VMAF_METAL_FRAMES]; // 4 planes filtered down by 16
    void *filter_aa;              // anti-aliasing filter
    // prescale: the planes scaled to float, shared by the slots, and the
    // source rows, columns and weights
    bool scale;
    SpeedMetalParams scale_params;
    void *scaled, *scale_xi, *scale_xf, *scale_yi, *scale_yf;
    SpeedSlot slot[VMAF_METAL_FRAMES];
    // fast mode: est_params() on the GPU. Shared: the planes minus their
    // local means, the covariance matrices; per slot: S, E and the
    // singular flag of the 4 planes.
    bool fast;
    SpeedMetalEstParams est_params;
    void *filter_buf, *local, *means, *cov;
    void *est_out[VMAF_METAL_FRAMES];
} SpeedChromaStateMetal;

static const VmafOption options[] = {
    {
        .name = "speed_kernelscale",
        .help = "scaling factor for the gaussian kernel (2.0 means "
                "multiplying the standard deviation by 2 and enlarge "
                "the kernel size accordingly",
        .offset = offsetof(SpeedChromaStateMetal, speed_chroma_kernelscale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_KERNELSCALE,
        .min = 0.1,
        .max = 4.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ks",
    },
    {
        .name = "speed_prescale",
        .help = "scaling factor for the frame (2.0 means "
                "making the image twice as large on each dimension)",
        .offset = offsetof(SpeedChromaStateMetal, speed_chroma_prescale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_PRESCALE,
        .min = 0.1,
        .max = 4.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ps",
    },
    {
        .name = "speed_prescale_method",
        .help = "scaling method for the frame, supported options: "
                 "[nearest, bilinear, bicubic, lanczos4]",
        .offset = offsetof(SpeedChromaStateMetal, speed_chroma_prescale_method),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = DEFAULT_SPEED_PRESCALE_METHOD,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "psm",
    },
    {
        .name = "speed_sigma_nn",
        .help = "standard deviation of neural noise",
        .offset = offsetof(SpeedChromaStateMetal, speed_chroma_sigma_nn),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_SIGMA_NN,
        .min = 0.1,
        .max = 2.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "snn",
    },
    {
        .name = "speed_nn_floor",
        .help = "neural noise floor, expressed in percentage of sigma_nn",
        .offset = offsetof(SpeedChromaStateMetal, speed_chroma_nn_floor),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_NN_FLOOR,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "nnf",
    },
    {
        .name = "speed_max_val",
        .help = "maximum value allowed; "
                "larger values will be clipped to this value",
        .offset = offsetof(SpeedChromaStateMetal, speed_chroma_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_SPEED_MAX_VAL,
        .min = 0.0,
        .max = 1000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "mxv",
    },
    {
        .name = "speed_weight_var_mode",
        .help = "different approaches to perform variance-absed weighting",
        .offset = offsetof(SpeedChromaStateMetal, speed_weight_var_mode),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.d = 0,
        .min = 0,
        .max = 6,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "wvm",
    },
    { 0 }
};

/* -------------------------------------------------------------------------
 * Buffers
 */

static void *buffer_new(VmafMetalState *ms, size_t size)
{
    id<MTLBuffer> buf = vmaf_metal_buffer(ms, size ? size : 16);
    return buf ? (void *)CFBridgingRetain(buf) : NULL;
}

static inline id<MTLBuffer> B(void *buf)
{
    return (__bridge id<MTLBuffer>)buf;
}

static void release(void **buf)
{
    if (*buf) CFRelease(*buf);
    *buf = NULL;
}

static void slot_free(SpeedSlot *k)
{
    float **bufs[] = {
        &k->st.buffers.independent_term, &k->st.buffers.linear_system_sol,
        &k->st.buffers.cov_mat, &k->st.buffers.eigenvalues,
        &k->st.buffers.tmp_buffer, &k->st.ref_results.entropies,
        &k->st.ref_results.variances, &k->st.dis_results.entropies,
        &k->st.dis_results.variances, &k->frame[0], &k->frame[1], &k->curr,
        &k->tmpbuf,
    };
    for (unsigned i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) {
        if (*bufs[i]) aligned_free(*bufs[i]);
        *bufs[i] = NULL;
    }
}

/** The parts of speed_init() that est_params() uses, at 1/16. */
static int slot_init(SpeedSlot *k, const SpeedDimensions *dim)
{
    k->st.dimensions = *dim;
    const size_t e = dim->elements_in_block, n = dim->num_blocks;
    const size_t w = dim->operating_width, h = dim->operating_height;
    k->st.float_stride = ALIGN_CEIL(w * sizeof(float));
    const size_t frame = k->st.float_stride * h;

    k->st.buffers.independent_term = aligned_malloc(sizeof(float) * n * e, 32);
    k->st.buffers.linear_system_sol = aligned_malloc(sizeof(float) * n * e, 32);
    k->st.buffers.cov_mat = aligned_malloc(sizeof(float) * e * e, 32);
    k->st.buffers.eigenvalues = aligned_malloc(sizeof(float) * e, 32);
    k->st.buffers.tmp_buffer = aligned_malloc(sizeof(float) *
            (NUM_SQUARE_BUFFERS * e * e + NUM_RECT_BUFFERS * e * n), 32);
    k->st.ref_results.entropies = aligned_malloc(sizeof(float) * n, 32);
    k->st.ref_results.variances = aligned_malloc(sizeof(float) * n, 32);
    k->st.dis_results.entropies = aligned_malloc(sizeof(float) * n, 32);
    k->st.dis_results.variances = aligned_malloc(sizeof(float) * n, 32);
    for (unsigned i = 0; i < 2; i++)
        k->frame[i] = aligned_malloc(frame, 32);
    k->curr = aligned_malloc(frame, 32);
    k->tmpbuf = aligned_malloc(frame, 32);
    if (!k->st.buffers.independent_term || !k->st.buffers.linear_system_sol ||
        !k->st.buffers.cov_mat || !k->st.buffers.eigenvalues ||
        !k->st.buffers.tmp_buffer || !k->st.ref_results.entropies ||
        !k->st.ref_results.variances || !k->st.dis_results.entropies ||
        !k->st.dis_results.variances || !k->frame[0] || !k->frame[1] ||
        !k->curr || !k->tmpbuf)
        return -ENOMEM;

    k->st.compute_cov_kernel = compute_cov_kernel_scalar;
#if ARCH_AARCH64
    if (vmaf_get_cpu_flags() & VMAF_ARM_CPU_FLAG_NEON)
        k->st.compute_cov_kernel = compute_cov_kernel_neon;
#endif
#if ARCH_X86
    unsigned flags = vmaf_get_cpu_flags();
    if (flags & VMAF_X86_CPU_FLAG_AVX2) {
        k->st.compute_cov_kernel = compute_cov_kernel_avx2;
    }
#if HAVE_AVX512
    if (flags & VMAF_X86_CPU_FLAG_AVX512) {
        k->st.compute_cov_kernel = compute_cov_kernel_avx512;
    }
#endif
#endif
    return 0;
}

static void buffers_free(SpeedChromaStateMetal *s)
{
    vmaf_metal_frames_close(&s->frames);
    for (unsigned k = 0; k < VMAF_METAL_FRAMES; k++) {
        release(&s->in[k]);
        release(&s->out[k]);
        release(&s->est_out[k]);
        slot_free(&s->slot[k]);
    }
    release(&s->filter_buf);
    release(&s->local);
    release(&s->means);
    release(&s->cov);
    release(&s->filter_aa);
    release(&s->scaled);
    release(&s->scale_xi);
    release(&s->scale_xf);
    release(&s->scale_yi);
    release(&s->scale_yf);
}

/**
 * The source indices (2 x n) and weights (n) of vif_scale_frame_s(), for n
 * output rows or columns out of src. Bilinear uses the CPU function for
 * both, as the CPU's row loop computes the rows in the same way.
 */
static void scale_table(enum vif_scaling_method method, int src, int n,
                        int *idx, float *frac)
{
    if (method == vif_scale_bilinear) {
        vif_scale_frame_bilinear_precompute_columns_s(src, n, idx, idx + n,
                                                      frac);
        return;
    }
    // vif_scale_frame_nearest_s()
    float ratio = (float)src / n;
    for (int x = 0; x < n; x++) {
        idx[x] = idx[n + x] = (int)(x * ratio);
        frac[x] = 0;
    }
}

static int scale_init(SpeedChromaStateMetal *s, enum vif_scaling_method method,
                      const SpeedDimensions *dim)
{
    const int w = dim->scaled_width, h = dim->scaled_height;
    s->scaled = buffer_new(s->ms, (size_t)SPEED_METAL_PLANES * w * h *
                           sizeof(float));
    s->scale_xi = buffer_new(s->ms, 2 * w * sizeof(int));
    s->scale_xf = buffer_new(s->ms, w * sizeof(float));
    s->scale_yi = buffer_new(s->ms, 2 * h * sizeof(int));
    s->scale_yf = buffer_new(s->ms, h * sizeof(float));
    if (!s->scaled || !s->scale_xi || !s->scale_xf || !s->scale_yi ||
        !s->scale_yf)
        return -ENOMEM;
    scale_table(method, dim->original_width, w, B(s->scale_xi).contents,
                B(s->scale_xf).contents);
    scale_table(method, dim->original_height, h, B(s->scale_yi).contents,
                B(s->scale_yf).contents);
    return 0;
}

/* -------------------------------------------------------------------------
 * Initialization
 */

/**
 * Configurations that the kernels do not cover run on the CPU extractor:
 * the bicubic and lanczos4 prescale methods, bit depths other than 8, 10,
 * 12 and 16 (picture_copy() handles only those), and the cases where
 * init_chroma() fails or leaves the state unusable.
 */
static int cpu_init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                    unsigned bpc, unsigned w, unsigned h)
{
    SpeedChromaStateMetal *s = fex->priv;
    extern VmafFeatureExtractor vmaf_fex_speed_chroma;

    VmafFeatureExtractor *cpu = malloc(sizeof(*cpu));
    if (!cpu) return -ENOMEM;
    memcpy(cpu, &vmaf_fex_speed_chroma, sizeof(*cpu));
    cpu->priv = calloc(1, cpu->priv_size);
    if (!cpu->priv) {
        free(cpu);
        return -ENOMEM;
    }

    // the option values as set by the framework, by name
    for (const VmafOption *o = cpu->options; o->name; o++) {
        const VmafOption *m = options;
        while (m->name && strcmp(m->name, o->name)) m++;
        if (!m->name || m->type != o->type) continue;
        const uint8_t *src = (const uint8_t *)s + m->offset;
        uint8_t *dst = (uint8_t *)cpu->priv + o->offset;
        switch (o->type) {
        case VMAF_OPT_TYPE_BOOL: memcpy(dst, src, sizeof(bool)); break;
        case VMAF_OPT_TYPE_INT: memcpy(dst, src, sizeof(int)); break;
        case VMAF_OPT_TYPE_DOUBLE: memcpy(dst, src, sizeof(double)); break;
        case VMAF_OPT_TYPE_STRING: memcpy(dst, src, sizeof(char *)); break;
        default: break;
        }
    }

    s->cpu = cpu;
    return cpu->init(cpu, pix_fmt, bpc, w, h);
}

static int init_fex(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                    unsigned bpc, unsigned w, unsigned h)
{
    SpeedChromaStateMetal *s = fex->priv;
    const unsigned luma_w = w, luma_h = h;

    // init_chroma()
    switch (pix_fmt) {
        case VMAF_PIX_FMT_UNKNOWN:
        case VMAF_PIX_FMT_YUV400P:
            return cpu_init(fex, pix_fmt, bpc, w, h);
        case VMAF_PIX_FMT_YUV420P:
            w /= 2;
            h /= 2;
            break;
        case VMAF_PIX_FMT_YUV422P:
            w /= 2;
            break;
        case VMAF_PIX_FMT_YUV444P:
            break;
    }

    s->speed_options = (SpeedOptions) {
        .speed_kernelscale = s->speed_chroma_kernelscale,
        .speed_prescale = s->speed_chroma_prescale,
        .speed_prescale_method = s->speed_chroma_prescale_method,
        .speed_sigma_nn = s->speed_chroma_sigma_nn,
        .speed_nn_floor = s->speed_chroma_nn_floor,
        .speed_weight_var_mode = s->speed_weight_var_mode,
    };

    // speed_init()
    SpeedDimensions dim;
    enum vif_scaling_method scaling_method;
    bool cpu =
        (bpc != 8 && bpc != 10 && bpc != 12 && bpc != 16) ||
        speed_init_dimensions(&dim, w, h, s->speed_options.speed_prescale) ||
        !vif_validate_kernelscale(s->speed_options.speed_kernelscale) ||
        vif_get_scaling_method(s->speed_options.speed_prescale_method,
                               &scaling_method);
    // filter_and_downscale(): the CPU copies the planes if the size does
    // not change
    s->scale = !cpu && !ALMOST_EQUAL(s->speed_options.speed_prescale, 1.0) &&
               (dim.scaled_width != dim.original_width ||
                dim.scaled_height != dim.original_height);
    cpu |= s->scale && scaling_method != vif_scale_nearest &&
           scaling_method != vif_scale_bilinear;
    if (cpu) return cpu_init(fex, pix_fmt, bpc, luma_w, luma_h);

    s->ms = fex->metal_state ? fex->metal_state : vmaf_metal_state_default();
    if (!s->ms) return -ENODEV;
    s->bpc = bpc;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                                                      fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;

    // filter_and_downscale()
    float filter_aa[SPEED_METAL_MAX_FWIDTH];
    const int fwidth_aa =
        vif_get_filter_size(1, s->speed_options.speed_kernelscale);
    speed_get_antialias_filter(filter_aa, NUM_SCALES,
                               s->speed_options.speed_kernelscale);
    s->filter_width =
        vif_get_filter_size(NUM_SCALES, s->speed_options.speed_kernelscale);
    vif_get_filter(s->filter, NUM_SCALES, s->speed_options.speed_kernelscale);

    const float scaler = bpc == 10 ? 4.0f : bpc == 12 ? 16.0f : 256.0f;
    s->scale_params = (SpeedMetalParams) {
        .w = dim.scaled_width, .h = dim.scaled_height,
        .in_stride = w,
        .in_plane = w * h,
        .method = scaling_method == vif_scale_bilinear ?
                  SPEED_METAL_SCALE_BILINEAR : SPEED_METAL_SCALE_NEAREST,
        .scaler = scaler,
    };
    // the filter reads the scaled float planes, or the input
    const int fw = s->scale ? dim.scaled_width : w;
    const int fh = s->scale ? dim.scaled_height : h;
    s->params = (SpeedMetalParams) {
        .w = fw, .h = fh,
        .in_stride = fw,
        .in_plane = fw * fh,
        .out_w = dim.scaled_width >> NUM_SCALES,
        .out_h = dim.scaled_height >> NUM_SCALES,
        .fwidth = fwidth_aa,
        .scaler = scaler,
    };

    if (vmaf_metal_frames_init(&s->frames, s->ms)) goto fail;
    if (s->scale && scale_init(s, scaling_method, &dim)) goto fail;
    s->filter_aa = buffer_new(s->ms, fwidth_aa * sizeof(float));
    if (!s->filter_aa) goto fail;
    memcpy(B(s->filter_aa).contents, filter_aa, fwidth_aa * sizeof(float));
    s->fast = s->ms->fast_math;
    const size_t n_out = (size_t)s->params.out_w * s->params.out_h;
    if (s->fast) {
        s->est_params = (SpeedMetalEstParams) {
            .w = s->params.out_w, .h = s->params.out_h,
            .fwidth = s->filter_width,
            .sub_w = dim.submatrix_width, .sub_h = dim.submatrix_height,
            .nbh = dim.num_blocks_horizontal, .nb = dim.num_blocks,
            .sigma_nn = s->speed_options.speed_sigma_nn,
            .log2_2pie = log2(2 * M_PI * M_E),
        };
        s->filter_buf = buffer_new(s->ms, s->filter_width * sizeof(float));
        s->local = buffer_new(s->ms, SPEED_METAL_PLANES * n_out * sizeof(float));
        s->cov = buffer_new(s->ms, SPEED_METAL_PLANES * SPEED_METAL_E *
                            SPEED_METAL_E * sizeof(float));
        s->means = buffer_new(s->ms, SPEED_METAL_PLANES * SPEED_METAL_E *
                              sizeof(float));
        if (!s->filter_buf || !s->local || !s->means || !s->cov) goto fail;
        memcpy(B(s->filter_buf).contents, s->filter,
               s->filter_width * sizeof(float));
    }
    const size_t bpp = bpc > 8 ? 2 : 1;
    for (unsigned k = 0; k < VMAF_METAL_FRAMES; k++) {
        if (s->fast) {
            s->est_out[k] = buffer_new(s->ms, SPEED_METAL_PLANES *
                    (2 * dim.num_blocks + 1) * sizeof(float));
            if (!s->est_out[k]) goto fail;
        }
        s->in[k] = buffer_new(s->ms, (size_t)SPEED_METAL_PLANES * w * h * bpp);
        if (!s->in[k]) goto fail;
        s->out[k] = buffer_new(s->ms, (size_t)SPEED_METAL_PLANES *
                s->params.out_w * s->params.out_h * sizeof(float));
        if (!s->out[k]) goto fail;
        if (slot_init(&s->slot[k], &dim)) goto fail;
    }
    return 0;

fail:
    buffers_free(s);
    return -ENOMEM;
}

/* -------------------------------------------------------------------------
 * Extraction
 */

/**
 * filter_and_downscale() after the anti-aliasing filter, then est_params(),
 * of one plane filtered down by 16 (`d`, out_w x out_h).
 */
static int est_plane(SpeedChromaStateMetal *s, SpeedSlot *k, const float *d,
                     float *frame_buffer, SpeedResultBuffers *results)
{
    const SpeedDimensions dim = k->st.dimensions;
    const size_t float_stride = k->st.float_stride;
    const size_t stride_px = float_stride / sizeof(float);
    const size_t downscaled_w = dim.scaled_width >> NUM_SCALES;
    const size_t downscaled_h = dim.scaled_height >> NUM_SCALES;

    for (size_t i = 0; i < downscaled_h; i++)
        memcpy(frame_buffer + i * stride_px, d + i * downscaled_w,
               downscaled_w * sizeof(float));
    vif_filter1d_s(s->filter, frame_buffer, k->curr, k->tmpbuf, downscaled_w,
                   downscaled_h, float_stride, float_stride, s->filter_width);
    subtract_image(frame_buffer, k->curr, downscaled_w, downscaled_h,
                   float_stride);

    return est_params(&k->st, frame_buffer, s->speed_options.speed_sigma_nn,
                      results);
}

/**
 * Fast mode: the results of est_params() of `plane`, from the GPU
 * (speed_chroma_metal_est).
 */
static int est_plane_fast(SpeedChromaStateMetal *s, unsigned slot,
                          unsigned plane, SpeedResultBuffers *results)
{
    const int nb = s->est_params.nb;
    float *o = (float *)B(s->est_out[slot]).contents + plane * (2 * nb + 1);
    results->variances = o;
    results->entropies = o + nb;
    if (o[2 * nb] != 0.0f) {
        vmaf_log(VMAF_LOG_LEVEL_WARNING,
                 "est_params: covariance matrix is singular\n");
        return -EINVAL;
    }
    return 0;
}

/** speed_extract_score() of one channel: planes `ref` and `ref` + 2. */
static int channel_score(SpeedChromaStateMetal *s, unsigned slot,
                         unsigned ref, float *score)
{
    SpeedSlot *k = &s->slot[slot];
    SpeedResultBuffers ref_results = k->st.ref_results;
    SpeedResultBuffers dis_results = k->st.dis_results;
    int err_ref, err_dis;
    if (s->fast) {
        err_ref = est_plane_fast(s, slot, ref, &ref_results);
        err_dis = est_plane_fast(s, slot, ref + 2, &dis_results);
    } else {
        const float *out = B(s->out[slot]).contents;
        const size_t n = (size_t)s->params.out_w * s->params.out_h;
        err_ref = est_plane(s, k, out + ref * n, k->frame[0], &ref_results);
        err_dis = est_plane(s, k, out + (ref + 2) * n, k->frame[1],
                            &dis_results);
    }

    if ((err_ref && !err_dis) || (!err_ref && err_dis)) {
        *score = 0.0f;
    } else {
        *score = get_speed_score(k->st.dimensions, ref_results, dis_results,
                                 s->speed_options.speed_sigma_nn,
                                 s->speed_options.speed_nn_floor,
                                 s->speed_options.speed_weight_var_mode);
    }
    return err_ref || err_dis;
}

/** The scores of extract_chroma() in speed.c */
static int write_scores(SpeedChromaStateMetal *s, unsigned slot,
                        unsigned index, VmafFeatureCollector *feature_collector)
{
    float score_u, score_v;
    int err_u = channel_score(s, slot, 0, &score_u);
    int err_v = channel_score(s, slot, 1, &score_v);

    float score_uv;
    if (err_u && !err_v) {
        score_uv = score_v;
    } else if (err_v && !err_u) {
        score_uv = score_u; }
    else {
        score_uv = (score_u + score_v) / 2.0;
    }

    int err = 0;

    err |=
        vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "Speed_chroma_feature_speed_chroma_u_score",
             MIN(score_u, s->speed_chroma_max_val), index);
    err |=
        vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "Speed_chroma_feature_speed_chroma_v_score",
            MIN(score_v, s->speed_chroma_max_val), index);
    err |=
        vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict, "Speed_chroma_feature_speed_chroma_uv_score",
            MIN(score_uv, s->speed_chroma_max_val), index);
    return err;
}

/**
 * Fast mode: the rest of filter_and_downscale() and est_params() of the 4
 * planes in s->out[slot], see speed_chroma.metal.
 */
static void encode_est(SpeedChromaStateMetal *s,
                       id<MTLComputeCommandEncoder> enc, unsigned slot)
{
    const SpeedMetalEstParams *p = &s->est_params;
    id<MTLComputePipelineState> pl_local =
        vmaf_metal_pipeline(s->ms, "speed_chroma_metal_local", 0);
    id<MTLComputePipelineState> pl_means =
        vmaf_metal_pipeline(s->ms, "speed_chroma_metal_means", 0);
    id<MTLComputePipelineState> pl_cov =
        vmaf_metal_pipeline(s->ms, "speed_chroma_metal_cov", 0);
    id<MTLComputePipelineState> pl_est =
        vmaf_metal_pipeline(s->ms, "speed_chroma_metal_est", 0);

    [enc setBytes:p length:sizeof(*p) atIndex:0];
    [enc setBuffer:B(s->out[slot]) offset:0 atIndex:1];
    [enc setBuffer:B(s->filter_buf) offset:0 atIndex:2];
    [enc setBuffer:B(s->local) offset:0 atIndex:3];
    vmaf_metal_dispatch_2d(enc, pl_local, p->w, SPEED_METAL_PLANES * p->h);

    // means: one threadgroup per offset in the block and plane;
    // covariance: one per row of the matrix and plane
    const MTLSize tg = MTLSizeMake(32, 8, 1);
    const MTLSize grid = MTLSizeMake(SPEED_METAL_E, SPEED_METAL_PLANES, 1);
    [enc setComputePipelineState:pl_means];
    [enc setBuffer:B(s->local) offset:0 atIndex:1];
    [enc setBuffer:B(s->means) offset:0 atIndex:2];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    [enc setComputePipelineState:pl_cov];
    [enc setBuffer:B(s->means) offset:0 atIndex:2];
    [enc setBuffer:B(s->cov) offset:0 atIndex:3];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];

    // one threadgroup per plane, of at least 2 simdgroups
    const NSUInteger n_est = MIN(256, pl_est.maxTotalThreadsPerThreadgroup);
    assert(n_est >= 64);
    [enc setComputePipelineState:pl_est];
    [enc setBuffer:B(s->local) offset:0 atIndex:1];
    [enc setBuffer:B(s->cov) offset:0 atIndex:2];
    [enc setBuffer:B(s->est_out[slot]) offset:0 atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(SPEED_METAL_PLANES, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(n_est, 1, 1)];
}

static int extract_fex(VmafFeatureExtractor *fex,
                       VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                       VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                       unsigned index, VmafFeatureCollector *feature_collector)
{
    SpeedChromaStateMetal *s = fex->priv;
    if (s->cpu) {
        return s->cpu->extract(s->cpu, ref_pic, ref_pic_90, dist_pic,
                               dist_pic_90, index, feature_collector);
    }

    id<MTLComputePipelineState> pl =
        vmaf_metal_pipeline(s->ms, "speed_chroma_metal_dec16",
                            s->scale ? 32 : s->bpc);
    id<MTLComputePipelineState> pl_scale = s->scale ?
        vmaf_metal_pipeline(s->ms, "speed_chroma_metal_scale", s->bpc) : nil;
    if (!pl || (s->scale && !pl_scale)) return -EINVAL;
    if (s->fast && (!vmaf_metal_pipeline(s->ms, "speed_chroma_metal_local", 0) ||
                    !vmaf_metal_pipeline(s->ms, "speed_chroma_metal_means", 0) ||
                    !vmaf_metal_pipeline(s->ms, "speed_chroma_metal_cov", 0) ||
                    !vmaf_metal_pipeline(s->ms, "speed_chroma_metal_est", 0)))
        return -EINVAL;

    // the input planes, at the size of init_chroma()
    const SpeedMetalParams *in_p = s->scale ? &s->scale_params : &s->params;
    const unsigned slot = vmaf_metal_frames_acquire(&s->frames);
    const size_t bpp = s->bpc > 8 ? 2 : 1;
    const size_t row = (size_t)in_p->in_stride * bpp;
    uint8_t *in = B(s->in[slot]).contents;
    const size_t plane = (size_t)in_p->in_plane * bpp;
    vmaf_metal_copy_plane(in + 0 * plane, row, ref_pic, 1);
    vmaf_metal_copy_plane(in + 1 * plane, row, ref_pic, 2);
    vmaf_metal_copy_plane(in + 2 * plane, row, dist_pic, 1);
    vmaf_metal_copy_plane(in + 3 * plane, row, dist_pic, 2);

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [vmaf_metal_queue(s->ms) commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (s->scale) {
            [enc setBytes:&s->scale_params length:sizeof(s->scale_params)
                  atIndex:0];
            [enc setBuffer:B(s->in[slot]) offset:0 atIndex:1];
            [enc setBuffer:B(s->scale_xi) offset:0 atIndex:2];
            [enc setBuffer:B(s->scale_xf) offset:0 atIndex:3];
            [enc setBuffer:B(s->scale_yi) offset:0 atIndex:4];
            [enc setBuffer:B(s->scale_yf) offset:0 atIndex:5];
            [enc setBuffer:B(s->scaled) offset:0 atIndex:6];
            vmaf_metal_dispatch_2d(enc, pl_scale, s->scale_params.w,
                                   SPEED_METAL_PLANES * s->scale_params.h);
        }
        [enc setBytes:&s->params length:sizeof(s->params) atIndex:0];
        [enc setBuffer:B(s->scale ? s->scaled : s->in[slot]) offset:0
                atIndex:1];
        [enc setBuffer:B(s->filter_aa) offset:0 atIndex:2];
        [enc setBuffer:B(s->out[slot]) offset:0 atIndex:3];
        vmaf_metal_dispatch_2d(enc, pl, s->params.out_w,
                               SPEED_METAL_PLANES * s->params.out_h);
        if (s->fast) encode_est(s, enc, slot);
        [enc endEncoding];

        vmaf_metal_frames_commit(&s->frames, slot, cb, ^int(int err) {
            if (err) return err;
            return write_scores(s, slot, index, feature_collector);
        });
    }
    return 0;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    SpeedChromaStateMetal *s = fex->priv;
    int err = 0;
    if (s->cpu) {
        err = s->cpu->close(s->cpu);
        free(s->cpu->priv);
        free(s->cpu);
        s->cpu = NULL;
    }
    buffers_free(s);
    if (s->feature_name_dict)
        err |= vmaf_dictionary_free(&s->feature_name_dict);
    return err;
}

static const char *provided_features[] = {
    "Speed_chroma_feature_speed_chroma_u_score",
    "Speed_chroma_feature_speed_chroma_v_score",
    "Speed_chroma_feature_speed_chroma_uv_score",
    NULL
};

VmafFeatureExtractor vmaf_fex_speed_chroma_metal = {
    .name = "speed_chroma_metal",
    .init = init_fex,
    .extract = extract_fex,
    .close = close_fex,
    .options = options,
    .priv_size = sizeof(SpeedChromaStateMetal),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_METAL,
};
