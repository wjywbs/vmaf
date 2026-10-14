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
 * CAMBI definitions shared between the Metal kernels (cambi.metal) and the
 * host (cambi_metal.m).
 */

#ifndef __VMAF_FEATURE_METAL_CAMBI_METAL_H__
#define __VMAF_FEATURE_METAL_CAMBI_METAL_H__

#ifdef __METAL_VERSION__
typedef ulong cambi_metal_u64;
#else
#include <stdint.h>
typedef uint64_t cambi_metal_u64;
#endif

#define CAMBI_METAL_NUM_SCALES 5
#define CAMBI_METAL_MAX_SEGS (2 * CAMBI_METAL_NUM_SCALES) // dist + source
#define CAMBI_METAL_MAX_DIFFS 4 // max_log_contrast <= 2

// c-values: one thread per column and chunk of rows, about this many threads
#define CAMBI_METAL_HIST_THREADS 16384
#define CAMBI_METAL_KEY_NONE 0xffff // pixel not counted in any histogram

// top-k selection: radix select on the float bits of the c-values
#define CAMBI_METAL_SEL_TGS 32 // threadgroups per segment
#define CAMBI_METAL_SEL_THREADS 256
#define CAMBI_METAL_L1_BINS 2048
#define CAMBI_METAL_L1_BASE 0x3f000000u // bits of 0.5f, smallest c-value > 0

/** cambi_preprocessing() */
typedef struct CambiMetalPreParams {
    int in_stride; // pixels
    int w, h;      // output, the encode or source size
    int resample;  // input and output sizes differ
    int anti_dither;
} CambiMetalPreParams;

typedef struct CambiMetalValidateParams {
    int w, h, stride; // pixels
    int max_val;
} CambiMetalValidateParams;

/** get_spatial_mask() */
typedef struct CambiMetalMaskParams {
    int w, h;
    int mask_index;
} CambiMetalMaskParams;

/** decimate() then filter_mode() */
typedef struct CambiMetalModeParams {
    int w, h;
    int src_stride;
    int src_step; // 2 if the source is decimated
} CambiMetalModeParams;

/** calculate_c_values() */
typedef struct CambiMetalCValueParams {
    int w, h;
    int mask_stride, mask_shift; // the mask is decimated by mask_shift
    int pad;
    int chunk_rows;
    int clean; // leave the histograms zero
    int v_band_base, v_band_size;
    int num_diffs, vlt_luma;
    int tvi[CAMBI_METAL_MAX_DIFFS];
    int weights[CAMBI_METAL_MAX_DIFFS];
    unsigned c_offset; // in the c-values buffer
} CambiMetalCValueParams;

/** The c-values of one scale of one picture. */
typedef struct CambiMetalSeg {
    unsigned offset, n;
    unsigned k; // number of top c-values to average
} CambiMetalSeg;

typedef struct CambiMetalSelParams {
    unsigned level; // 1, 2 or 3
} CambiMetalSelParams;

/** Radix select state of one segment. */
typedef struct CambiMetalSel {
    unsigned done;   // threshold found
    unsigned kk;     // rank of the threshold within the current bin
    unsigned bin1;   // level 1 bin
    unsigned prefix; // bits >> 8 of the threshold, after level 2
    unsigned t_bits; // threshold: the k-th largest c-value
    unsigned pad[3];
} CambiMetalSel;

/** Per frame output. */
typedef struct CambiMetalOut {
    CambiMetalSel sel[CAMBI_METAL_MAX_SEGS];
    unsigned invalid; // input has values above the bit depth
    unsigned pad[3];
    // c-values > threshold: sum (in units of 2^-24) and count
    cambi_metal_u64 sum[CAMBI_METAL_MAX_SEGS][CAMBI_METAL_SEL_TGS];
    unsigned cnt[CAMBI_METAL_MAX_SEGS][CAMBI_METAL_SEL_TGS];
} CambiMetalOut;

#endif /* __VMAF_FEATURE_METAL_CAMBI_METAL_H__ */
