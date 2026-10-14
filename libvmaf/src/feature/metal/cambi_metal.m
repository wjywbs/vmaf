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

// first: defines MIN and MAX, which Foundation then leaves alone
#include "cambi.h"

#import <Metal/Metal.h>

#include <errno.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "luminance_tools.h"
#include "metal/common.h"
#include "opt.h"

#include "cambi_metal.h"

/* Defaults and limits, see cambi.c */
#define DEFAULT_CAMBI_TOPK_POOLING (0.6)
#define DEFAULT_CAMBI_WINDOW_SIZE (65)
#define DEFAULT_CAMBI_TVI (0.019)
#define DEFAULT_CAMBI_VLT (0.0)
#define DEFAULT_CAMBI_MAX_LOG_CONTRAST (2)
#define DEFAULT_CAMBI_FULL_REF_FLAG (false)
#define DEFAULT_CAMBI_EOTF ("bt1886")
#define DEFAULT_CAMBI_HIGH_RES_SPEEDUP (0)
#define CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1080p (1920 * 1080)
#define CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1440p (2560 * 1440)
#define CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_2160p (3840 * 2160)
#define CAMBI_MIN_WIDTH_HEIGHT (216)
#define CAMBI_4K_WIDTH (3840)
#define CAMBI_4K_HEIGHT (2160)
#define DEFAULT_CAMBI_MAX_VAL (1000.0)
#define NUM_SCALES CAMBI_METAL_NUM_SCALES
#define MASK_FILTER_SIZE 7

static const int g_scale_weights[NUM_SCALES] = {16, 8, 4, 2, 1};
static const int g_contrast_weights[32] = {1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8,
                                           8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9};

/** One picture to score: the distorted picture, or the source. */
typedef struct CambiPass {
    unsigned w, h; // encode or source size
    uint16_t window_size;
    int resample;
    unsigned scale_w[NUM_SCALES], scale_h[NUM_SCALES];
    unsigned filt_offset[NUM_SCALES]; // in the filtered pictures buffer
    unsigned seg; // first segment
    uint16_t mask_index;
    unsigned chunks[NUM_SCALES], chunk_rows[NUM_SCALES]; // c-values threads
    void *ori_x, *ori_y; // id<MTLBuffer>, retained: resampling indices
} CambiPass;

typedef struct CambiStateMetal {
    // options, see cambi.c
    int enc_width;
    int enc_height;
    int enc_bitdepth;
    int src_width;
    int src_height;
    int window_size;
    double topk;
    double cambi_topk;
    double tvi_threshold;
    double cambi_max_val;
    double cambi_vis_lum_threshold;
    int max_log_contrast;
    char *heatmaps_path;
    char *eotf;
    char *cambi_eotf;
    bool full_ref;
    int cambi_high_res_speedup;

    // configurations that run on the CPU extractor
    VmafFeatureExtractor *cpu;

    VmafDictionary *feature_name_dict;
    VmafMetalState *ms;
    VmafMetalFrames frames;
    unsigned w, h, bpc;
    double pool_topk;
    CambiMetalCValueParams cv; // fields common to all scales
    CambiPass pass[2];
    unsigned n_pass;
    CambiMetalSeg segs[CAMBI_METAL_MAX_SEGS];
    unsigned n_seg;

    // id<MTLBuffer>, retained
    void *in[2][VMAF_METAL_FRAMES]; // distorted, source pictures
    void *c_values[VMAF_METAL_FRAMES];
    void *out[VMAF_METAL_FRAMES]; // CambiMetalOut
    void *pre, *hsum, *mask, *filt, *keys, *chist, *hist, *segs_buf, *recip;
    bool chist_dirty; // chist is not zero, see cambi_metal_cvalues
} CambiStateMetal;

static const VmafOption options[] = {
    {
        .name = "cambi_max_val",
        .help = "maximum value allowed; larger values will be clipped to this value",
        .offset = offsetof(CambiStateMetal, cambi_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_MAX_VAL,
        .min = 0.0,
        .max = 1000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "cmxv",
    },
    {
        .name = "enc_width",
        .help = "Encoding width",
        .offset = offsetof(CambiStateMetal, enc_width),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 144,
        .max = 7680,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "encw",
    },
    {
        .name = "enc_height",
        .help = "Encoding height",
        .offset = offsetof(CambiStateMetal, enc_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 144,
        .max = 7680,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ench",
    },
    {
        .name = "enc_bitdepth",
        .help = "Encoding bitdepth",
        .offset = offsetof(CambiStateMetal, enc_bitdepth),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 6,
        .max = 16,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "encbd",
    },
    {
        .name = "src_width",
        .help = "Source width. Only used when full_ref=true.",
        .offset = offsetof(CambiStateMetal, src_width),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 320,
        .max = 7680,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "srcw",
    },
    {
        .name = "src_height",
        .help = "Source height. Only used when full_ref=true.",
        .offset = offsetof(CambiStateMetal, src_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 200,
        .max = 4320,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "srch",
    },
    {
        .name = "window_size",
        .help = "Window size to compute CAMBI: 65 corresponds to ~1 degree at 4k",
        .offset = offsetof(CambiStateMetal, window_size),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_CAMBI_WINDOW_SIZE,
        .min = 15,
        .max = 127,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ws",
    },
    {
        .name = "topk",
        .help = "Ratio of pixels for the spatial pooling computation, must be 0 < topk <= 1.0",
        .offset = offsetof(CambiStateMetal, topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_TOPK_POOLING,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "cambi_topk",
        .help = "Ratio of pixels for the spatial pooling computation, must be 0 < cambi_topk <= 1.0",
        .offset = offsetof(CambiStateMetal, cambi_topk),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_TOPK_POOLING,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ctpk",
    },
    {
        .name = "tvi_threshold",
        .help = "Visibilty threshold for luminance ΔL < tvi_threshold*L_mean",
        .offset = offsetof(CambiStateMetal, tvi_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_TVI,
        .min = 0.0001,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "tvit",
    },
    {
        .name = "cambi_vis_lum_threshold",
        .help = "Luminance value below which we assume any banding is not visible",
        .offset = offsetof(CambiStateMetal, cambi_vis_lum_threshold),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_CAMBI_VLT,
        .min = 0.0,
        .max = 300.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "vlt",
    },
    {
        .name = "max_log_contrast",
        .help = "Maximum contrast in log luma level (2^max_log_contrast) at 10-bits, "
                "e.g., 2 is equivalent to 4 luma levels at 10-bit and 1 luma level at 8-bit. "
                "From 0 to 5: default 2 is recommended for banding from compression.",
        .offset = offsetof(CambiStateMetal, max_log_contrast),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_CAMBI_MAX_LOG_CONTRAST,
        .min = 0,
        .max = 5,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "mlc",
    },
    {
        .name = "heatmaps_path",
        .help = "Path where heatmaps will be dumped.",
        .offset = offsetof(CambiStateMetal, heatmaps_path),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = NULL,
    },
    {
        .name = "full_ref",
        .help = "If true, CAMBI will be run in full-reference mode and will be computed on both the reference and distorted inputs",
        .offset = offsetof(CambiStateMetal, full_ref),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = DEFAULT_CAMBI_FULL_REF_FLAG,
    },
    {
        .name = "eotf",
        .help = "Determines the EOTF used to compute the visibility thresholds. Possible values: ['bt1886', 'pq']. Default: 'bt1886'",
        .offset = offsetof(CambiStateMetal, eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = DEFAULT_CAMBI_EOTF,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "cambi_eotf",
        .help = "Determines the EOTF used to compute the visibility thresholds. Possible values: ['bt1886', 'pq']. Default: 'bt1886'. If both eotf and cambi_eotf are set, cambi_eotf takes precedence.",
        .offset = offsetof(CambiStateMetal, cambi_eotf),
        .type = VMAF_OPT_TYPE_STRING,
        .default_val.s = DEFAULT_CAMBI_EOTF,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "ceot",
    },
    {
        .name = "cambi_high_res_speedup",
        .help = "Speed up the processing by downsampling post spatial mask for resolutions >= 1080p. "
                "Min speed-up resolution possible values: [1080, 1440, 2160, 0]. Default: 0 (not applied)"
                "Note some loss of accuracy is expected with this speedup.",
        .offset = offsetof(CambiStateMetal, cambi_high_res_speedup),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_CAMBI_HIGH_RES_SPEEDUP,
        .min = 0,
        .max = CAMBI_4K_HEIGHT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
        .alias = "hrs",
    },
    { 0 }
};

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

static void buffers_free(CambiStateMetal *s)
{
    vmaf_metal_frames_close(&s->frames);
    for (unsigned k = 0; k < VMAF_METAL_FRAMES; k++) {
        release(&s->in[0][k]);
        release(&s->in[1][k]);
        release(&s->c_values[k]);
        release(&s->out[k]);
    }
    for (unsigned p = 0; p < 2; p++) {
        release(&s->pass[p].ori_x);
        release(&s->pass[p].ori_y);
    }
    void **bufs[] = { &s->pre, &s->hsum, &s->mask, &s->filt, &s->keys,
                      &s->chist, &s->hist, &s->segs_buf, &s->recip };
    for (unsigned i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++)
        release(bufs[i]);
}

/* -------------------------------------------------------------------------
 * Copies of the static helpers of cambi.c
 */

static void adjust_window_size(uint16_t *window_size, unsigned input_width,
                               unsigned input_height, bool cambi_high_res_speedup)
{
    // Adjustment weight: (input_width + input_height) / (CAMBI_4K_WIDTH + CAMBI_4K_HEIGHT)
    (*window_size) = (((*window_size) * (input_width+input_height)) / 375) >> 4;
    if (cambi_high_res_speedup) {
        (*window_size) = ((*window_size) + 1) >> 1;
    }
    // round up to odd
    *window_size |= 1;
}

static uint16_t ceil_log2(uint32_t num)
{
    if (num==0)
        return 0;

    uint32_t tmp = num - 1;
    uint16_t shift = 0;
    while (tmp>0) {
        tmp >>= 1;
        shift += 1;
    }
    return shift;
}

static uint16_t get_mask_index(unsigned input_width, unsigned input_height,
                               uint16_t filter_size)
{
    uint32_t shifted_wh = (input_width >> 6) * (input_height >> 6);
    return (filter_size * filter_size + 3 * (ceil_log2(shifted_wh) - 11) - 1)>>1;
}

#define SWAP_FLOATS(x, y) \
    {                     \
        float temp = x;   \
        x = y;            \
        y = temp;         \
    }

static double average_topk_elements(const float *arr, int topk_elements)
{
    double sum = 0;
    for (int i = 0; i < topk_elements; i++)
        sum += arr[i];

    return (double)sum / topk_elements;
}

static void quick_select(float *arr, int n, int k)
{
    if (n == k) return;
    int left = 0;
    int right = n - 1;
    while (left < right) {
        float pivot = arr[k];
        int i = left;
        int j = right;
        do {
            while (arr[i] > pivot) {
                i++;
            }
            while (arr[j] < pivot) {
                j--;
            }
            if (i <= j) {
                SWAP_FLOATS(arr[i], arr[j]);
                i++;
                j--;
            }
        } while (i <= j);
        if (j < k) {
            left = i;
        }
        if (k < i) {
            right = j;
        }
    }
}

static uint16_t get_pixels_in_window(uint16_t window_length)
{
    uint16_t odd_length = 2 * (window_length >> 1) + 1;
    return odd_length * odd_length;
}

static double weight_scores_per_scale(double *scores_per_scale, uint16_t normalization)
{
    double score = 0.0;
    for (unsigned scale = 0; scale < NUM_SCALES; scale++)
        score += (scores_per_scale[scale] * g_scale_weights[scale]);

    return score / normalization;
}

/** The resampling of decimate_generic_*_and_convert_to_10b(). */
static void resample_indices(int *ori, unsigned in_n, unsigned out_n)
{
    float ratio = (float)in_n / out_n;
    float start = ratio / 2 - 0.5;
    float x = start;
    for (unsigned j = 0; j < out_n; j++) {
        ori[j] = (int)(x + 0.5);
        x += ratio;
    }
}

/* -------------------------------------------------------------------------
 * Initialization
 */

/**
 * Configurations that the kernels do not cover run on the CPU extractor:
 * heatmaps, max_log_contrast > 2 (more than 4 contrast levels), and scales
 * smaller than half the window, where calculate_c_values() reads outside
 * the picture.
 */
static int cpu_init(VmafFeatureExtractor *fex, const void *opts,
                    enum VmafPixelFormat pix_fmt, unsigned bpc,
                    unsigned w, unsigned h)
{
    CambiStateMetal *s = fex->priv;
    extern VmafFeatureExtractor vmaf_fex_cambi;

    VmafFeatureExtractor *cpu = malloc(sizeof(*cpu));
    if (!cpu) return -ENOMEM;
    memcpy(cpu, &vmaf_fex_cambi, sizeof(*cpu));
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
        const uint8_t *src = (const uint8_t *)opts + m->offset;
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

static int pass_init(CambiStateMetal *s, CambiPass *p, unsigned w, unsigned h,
                     uint16_t window_size, bool speedup, unsigned *filt_size,
                     unsigned *c_size)
{
    p->w = w;
    p->h = h;
    p->window_size = window_size;
    p->resample = w != s->w || h != s->h;
    p->mask_index = get_mask_index(w, h, MASK_FILTER_SIZE);

    const unsigned pad = window_size >> 1;
    unsigned sw = w, sh = h;
    *filt_size = 0;
    p->seg = s->n_seg;
    for (unsigned scale = 0; scale < NUM_SCALES; scale++) {
        if (scale > 0 || speedup) {
            sw = (sw + 1) >> 1;
            sh = (sh + 1) >> 1;
        }
        if (sw < pad || sh < pad) return 1; // CPU
        p->scale_w[scale] = sw;
        p->scale_h[scale] = sh;
        p->filt_offset[scale] = *filt_size;
        *filt_size += sw * sh;

        CambiMetalSeg *seg = &s->segs[s->n_seg++];
        const int n = sw * sh;
        int k = s->pool_topk * n;
        seg->offset = *c_size;
        seg->n = n;
        seg->k = k < 1 ? 1 : k > n ? n : k;
        *c_size += n;
    }
    return 0;
}

static int pass_tables(CambiStateMetal *s, CambiPass *p)
{
    p->ori_x = buffer_new(s->ms, p->w * sizeof(int));
    p->ori_y = buffer_new(s->ms, p->h * sizeof(int));
    if (!p->ori_x || !p->ori_y) return -ENOMEM;
    if (p->resample) {
        resample_indices(B(p->ori_x).contents, s->w, p->w);
        resample_indices(B(p->ori_y).contents, s->h, p->h);
    }
    return 0;
}

static int init_fex(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                    unsigned bpc, unsigned w, unsigned h)
{
    CambiStateMetal *s = fex->priv;
    CambiStateMetal opts = *s;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features, fex->options, s);
    if (!s->feature_name_dict)
        return -ENOMEM;

    // init() of cambi.c
    if (s->enc_bitdepth == 0) {
        s->enc_bitdepth = bpc;
    }
    if (s->enc_width == 0 || s->enc_height == 0) {
        s->enc_width = w;
        s->enc_height = h;
    }
    if (s->src_width == 0 || s->src_height == 0) {
        s->src_width = w;
        s->src_height = h;
    }

    // if the encode had been downscaled, there is no need to upscale it back to the encoding resolution
    if ((unsigned)s->enc_height > h || (unsigned)s->enc_width > w) {
        s->enc_width = w;
        s->enc_height = h;
    }

    if (s->enc_width < CAMBI_MIN_WIDTH_HEIGHT && s->enc_height < CAMBI_MIN_WIDTH_HEIGHT) {
        return -EINVAL;
    }
    if (s->src_width < CAMBI_MIN_WIDTH_HEIGHT && s->src_height < CAMBI_MIN_WIDTH_HEIGHT) {
        return -EINVAL;
    }
    if (s->src_width > s->enc_width && s->src_height < s->enc_height) {
        return -EINVAL;
    }
    if (s->src_width < s->enc_width && s->src_height > s->enc_height) {
        return -EINVAL;
    }

    int enc_pix = s->enc_width * s->enc_height;
    switch (s->cambi_high_res_speedup) {
        case 1080:
            if (enc_pix < CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1080p) {
                s->cambi_high_res_speedup = 0;
            }
            break;
        case 1440:
            if (enc_pix < CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_1440p) {
                s->cambi_high_res_speedup = 0;
            }
            break;
        case 2160:
            if (enc_pix < CAMBI_HIGH_RES_SPEEDUP_THRESHOLD_2160p) {
                s->cambi_high_res_speedup = 0;
            }
            break;
        default:
            s->cambi_high_res_speedup = 0;
    }
    const bool speedup = s->cambi_high_res_speedup;

    const int num_diffs = 1 << s->max_log_contrast;

    VmafLumaRange luma_range;
    int err = vmaf_luminance_init_luma_range(&luma_range, 10, VMAF_PIXEL_RANGE_LIMITED);
    if (err) return err;

    /* use cambi_eotf if it has a non-default value, else use eotf */
    const char *effective_eotf;
    if (strcmp(s->cambi_eotf, DEFAULT_CAMBI_EOTF) != 0) {
        effective_eotf = s->cambi_eotf;
    } else {
        effective_eotf = s->eotf;
    }

    VmafEOTF eotf;
    err = vmaf_luminance_init_eotf(&eotf, effective_eotf);
    if (err) return err;

    uint16_t window_size = s->window_size;
    uint16_t src_window_size = s->window_size;
    adjust_window_size(&window_size, s->enc_width, s->enc_height, speedup);
    adjust_window_size(&src_window_size, s->src_width, s->src_height, speedup);

    int max_window = MAX(window_size, src_window_size);
    if (max_window * max_window >= CAMBI_RECIPROCAL_LUT_SIZE) {
        vmaf_log(VMAF_LOG_LEVEL_ERROR,
            "cambi: window_size %d too large for reciprocal LUT\n", max_window);
        return -EINVAL;
    }

    /* use the original topk setting if it has a non-default value, else use the cambi_topk one */
    if (s->topk != DEFAULT_CAMBI_TOPK_POOLING) {
        s->pool_topk = s->topk;
    } else {
        s->pool_topk = s->cambi_topk;
    }

    s->w = w;
    s->h = h;
    s->bpc = bpc;
    s->n_pass = s->full_ref ? 2 : 1;
    s->n_seg = 0;

    unsigned filt_size[2] = { 0 }, c_size = 0;
    bool on_cpu = s->heatmaps_path || num_diffs > CAMBI_METAL_MAX_DIFFS;
    on_cpu |= pass_init(s, &s->pass[0], s->enc_width, s->enc_height,
                        window_size, speedup, &filt_size[0], &c_size);
    if (s->full_ref) {
        on_cpu |= pass_init(s, &s->pass[1], s->src_width, s->src_height,
                            src_window_size, speedup, &filt_size[1], &c_size);
    }
    if (on_cpu) {
        vmaf_dictionary_free(&s->feature_name_dict);
        return cpu_init(fex, &opts, pix_fmt, bpc, w, h);
    }

    // c_value_pixel() parameters
    CambiMetalCValueParams *cv = &s->cv;
    memset(cv, 0, sizeof(*cv));
    cv->num_diffs = num_diffs;
    cv->vlt_luma = get_vlt_luma(s->cambi_vis_lum_threshold, luma_range, eotf);
    for (int d = 0; d < num_diffs; d++) {
        cv->tvi[d] = get_tvi_for_diff(d + 1, s->tvi_threshold, 10, luma_range, eotf);
        cv->tvi[d] += num_diffs;
        cv->weights[d] = g_contrast_weights[d];
    }
    int v_lo_signed = cv->vlt_luma - 3 * num_diffs + 1;
    uint16_t v_band_base = v_lo_signed > 0 ? (uint16_t)v_lo_signed : 0;
    cv->v_band_base = v_band_base;
    cv->v_band_size = (uint16_t)(cv->tvi[num_diffs - 1] + 1 - v_band_base);

    // Metal
    s->ms = fex->metal_state ? fex->metal_state : vmaf_metal_state_default();
    if (!s->ms) return -ENODEV;

    // c-values: one thread per column and chunk of rows, each chunk with its
    // own histograms. They are cleared here, then by the kernel or before
    // the next use (chist_dirty).
    unsigned max_pix = 0;
    size_t chist_size = 0;
    for (unsigned p = 0; p < s->n_pass; p++) {
        CambiPass *ps = &s->pass[p];
        max_pix = MAX(max_pix, ps->w * ps->h);
        for (unsigned scale = 0; scale < NUM_SCALES; scale++) {
            const unsigned sw = ps->scale_w[scale], sh = ps->scale_h[scale];
            const unsigned n = MAX(MIN(CAMBI_METAL_HIST_THREADS / sw, sh), 1);
            ps->chunk_rows[scale] = (sh + n - 1) / n;
            ps->chunks[scale] = (sh + ps->chunk_rows[scale] - 1) / ps->chunk_rows[scale];
            chist_size = MAX(chist_size, (size_t)ps->chunks[scale] *
                             cv->v_band_size * sw * sizeof(uint16_t));
        }
    }
    chist_size = (chist_size + 15) & ~(size_t)15;

    if (vmaf_metal_frames_init(&s->frames, s->ms)) goto fail;
    const size_t bpp = bpc > 8 ? 2 : 1;
    for (unsigned k = 0; k < VMAF_METAL_FRAMES; k++) {
        for (unsigned p = 0; p < s->n_pass; p++) {
            s->in[p][k] = buffer_new(s->ms, (size_t)w * h * bpp);
            if (!s->in[p][k]) goto fail;
        }
        s->c_values[k] = buffer_new(s->ms, (size_t)c_size * sizeof(float));
        s->out[k] = buffer_new(s->ms, sizeof(CambiMetalOut));
        if (!s->c_values[k] || !s->out[k]) goto fail;
    }
    s->pre = buffer_new(s->ms, (size_t)max_pix * sizeof(uint16_t));
    s->hsum = buffer_new(s->ms, max_pix);
    s->mask = buffer_new(s->ms, max_pix);
    s->filt = buffer_new(s->ms,
            (size_t)MAX(filt_size[0], filt_size[1]) * sizeof(uint16_t));
    s->keys = buffer_new(s->ms, (size_t)max_pix * sizeof(uint16_t));
    s->chist = buffer_new(s->ms, chist_size);
    s->hist = buffer_new(s->ms,
            3 * CAMBI_METAL_MAX_SEGS * CAMBI_METAL_L1_BINS * sizeof(uint32_t));
    s->segs_buf = buffer_new(s->ms, sizeof(s->segs));
    s->recip = buffer_new(s->ms, sizeof(reciprocal_lut));
    if (!s->pre || !s->hsum || !s->mask || !s->filt || !s->keys ||
        !s->chist || !s->hist ||
        !s->segs_buf || !s->recip)
        goto fail;
    memset(B(s->chist).contents, 0, chist_size);
    memcpy(B(s->segs_buf).contents, s->segs, sizeof(s->segs));
    memcpy(B(s->recip).contents, reciprocal_lut, sizeof(reciprocal_lut));
    for (unsigned p = 0; p < s->n_pass; p++) {
        if (pass_tables(s, &s->pass[p])) goto fail;
    }

    return 0;

fail:
    buffers_free(s);
    return -ENOMEM;
}

/* -------------------------------------------------------------------------
 * Extraction
 */

static void encode_pass(CambiStateMetal *s, id<MTLComputeCommandEncoder> enc,
                        unsigned slot, unsigned pass,
                        id<MTLComputePipelineState> const *pl)
{
    enum { VALIDATE, PREPROCESS, MASK_H, MASK_V, MODE, KEYS, ZERO, CVALUES };
    const CambiPass *p = &s->pass[pass];
    const bool speedup = s->cambi_high_res_speedup;
    id<MTLBuffer> in = B(s->in[pass][slot]);

    if (s->bpc != 8 && s->bpc != 16) {
        // validate_image()
        const CambiMetalValidateParams vp = {
            .w = s->w, .h = s->h, .stride = s->w,
            .max_val = (1 << s->bpc) - 1,
        };
        [enc setBytes:&vp length:sizeof(vp) atIndex:0];
        [enc setBuffer:in offset:0 atIndex:1];
        [enc setBuffer:B(s->out[slot]) offset:offsetof(CambiMetalOut, invalid)
               atIndex:2];
        vmaf_metal_dispatch_2d(enc, pl[VALIDATE], s->w, s->h);
    }

    const CambiMetalPreParams pp = {
        .in_stride = s->w,
        .w = p->w, .h = p->h,
        .resample = p->resample,
        .anti_dither = s->enc_bitdepth < 10,
    };
    [enc setBytes:&pp length:sizeof(pp) atIndex:0];
    [enc setBuffer:in offset:0 atIndex:1];
    [enc setBuffer:B(p->ori_x) offset:0 atIndex:2];
    [enc setBuffer:B(p->ori_y) offset:0 atIndex:3];
    [enc setBuffer:B(s->pre) offset:0 atIndex:4];
    vmaf_metal_dispatch_2d(enc, pl[PREPROCESS], p->w, p->h);

    const CambiMetalMaskParams mp = {
        .w = p->w, .h = p->h, .mask_index = p->mask_index,
    };
    [enc setBytes:&mp length:sizeof(mp) atIndex:0];
    [enc setBuffer:B(s->pre) offset:0 atIndex:1];
    [enc setBuffer:B(s->hsum) offset:0 atIndex:2];
    vmaf_metal_dispatch_2d(enc, pl[MASK_H], p->w, p->h);
    [enc setBuffer:B(s->hsum) offset:0 atIndex:1];
    [enc setBuffer:B(s->mask) offset:0 atIndex:2];
    vmaf_metal_dispatch_2d(enc, pl[MASK_V], p->w, p->h);

    const unsigned pad = p->window_size >> 1;
    for (unsigned scale = 0; scale < NUM_SCALES; scale++) {
        const unsigned sw = p->scale_w[scale], sh = p->scale_h[scale];
        const CambiMetalModeParams mo = {
            .w = sw, .h = sh,
            .src_stride = scale ? p->scale_w[scale - 1] : p->w,
            .src_step = scale > 0 || speedup ? 2 : 1,
        };
        [enc setBytes:&mo length:sizeof(mo) atIndex:0];
        if (scale) {
            [enc setBuffer:B(s->filt)
                    offset:p->filt_offset[scale - 1] * sizeof(uint16_t)
                   atIndex:1];
        } else {
            [enc setBuffer:B(s->pre) offset:0 atIndex:1];
        }
        [enc setBuffer:B(s->filt)
                offset:p->filt_offset[scale] * sizeof(uint16_t) atIndex:2];
        vmaf_metal_dispatch_2d(enc, pl[MODE], sw, sh);

        CambiMetalCValueParams cv = s->cv;
        cv.w = sw;
        cv.h = sh;
        cv.mask_stride = p->w;
        cv.mask_shift = scale + speedup;
        cv.pad = pad;
        cv.chunk_rows = p->chunk_rows[scale];
        // Leaving the histograms zero reads the (2 pad + 1)^2 keys of the
        // last window per thread, zeroing them writes v_band_size entries.
        cv.clean = (2 * pad + 1) * (2 * pad + 1) < (unsigned)cv.v_band_size;
        cv.c_offset = s->segs[p->seg + scale].offset;
        [enc setBytes:&cv length:sizeof(cv) atIndex:0];
        [enc setBuffer:B(s->filt)
                offset:p->filt_offset[scale] * sizeof(uint16_t) atIndex:1];
        [enc setBuffer:B(s->mask) offset:0 atIndex:2];
        [enc setBuffer:B(s->keys) offset:0 atIndex:3];
        vmaf_metal_dispatch_2d(enc, pl[KEYS], sw, sh);

        if (s->chist_dirty) {
            const unsigned n16 = (p->chunks[scale] * cv.v_band_size * sw *
                                  sizeof(uint16_t) + 15) / 16;
            [enc setBuffer:B(s->chist) offset:0 atIndex:0];
            [enc setBytes:&n16 length:sizeof(n16) atIndex:1];
            vmaf_metal_dispatch_2d(enc, pl[ZERO], n16, 1);
        }
        s->chist_dirty = !cv.clean;

        [enc setBytes:&cv length:sizeof(cv) atIndex:0];
        [enc setBuffer:B(s->keys) offset:0 atIndex:1];
        [enc setBuffer:B(s->chist) offset:0 atIndex:2];
        [enc setBuffer:B(s->recip) offset:0 atIndex:3];
        [enc setBuffer:B(s->c_values[slot]) offset:0 atIndex:4];
        vmaf_metal_dispatch_2d(enc, pl[CVALUES], sw, p->chunks[scale]);
    }
}

static void encode_select(CambiStateMetal *s, id<MTLComputeCommandEncoder> enc,
                          unsigned slot, id<MTLComputePipelineState> hist,
                          id<MTLComputePipelineState> find,
                          id<MTLComputePipelineState> sum)
{
    id<MTLBuffer> out = B(s->out[slot]);
    const size_t level_bytes =
        CAMBI_METAL_MAX_SEGS * CAMBI_METAL_L1_BINS * sizeof(uint32_t);
    const MTLSize threads = MTLSizeMake(CAMBI_METAL_SEL_THREADS, 1, 1);

    [enc setBuffer:B(s->segs_buf) offset:0 atIndex:1];
    [enc setBuffer:B(s->c_values[slot]) offset:0 atIndex:2];
    [enc setBuffer:out offset:offsetof(CambiMetalOut, sel) atIndex:3];
    for (unsigned level = 1; level <= 3; level++) {
        const CambiMetalSelParams sp = { .level = level };
        [enc setBytes:&sp length:sizeof(sp) atIndex:0];
        [enc setBuffer:B(s->hist) offset:(level - 1) * level_bytes atIndex:4];
        [enc setComputePipelineState:hist];
        [enc dispatchThreadgroups:MTLSizeMake(CAMBI_METAL_SEL_TGS, s->n_seg, 1)
            threadsPerThreadgroup:threads];
        [enc setComputePipelineState:find];
        [enc setBuffer:B(s->hist) offset:(level - 1) * level_bytes atIndex:2];
        [enc dispatchThreadgroups:MTLSizeMake(s->n_seg, 1, 1)
            threadsPerThreadgroup:threads];
        [enc setBuffer:B(s->c_values[slot]) offset:0 atIndex:2];
    }
    [enc setComputePipelineState:sum];
    [enc setBuffer:out offset:offsetof(CambiMetalOut, sum) atIndex:4];
    [enc setBuffer:out offset:offsetof(CambiMetalOut, cnt) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(CAMBI_METAL_SEL_TGS, s->n_seg, 1)
        threadsPerThreadgroup:threads];
}

/** The k-th largest c-value, as a multiple of 2^-24. */
static uint64_t fixed_24(uint32_t bits)
{
    if (!bits) return 0;
    const uint64_t m = (bits & 0x7fffff) | 0x800000;
    const int e = (int)(bits >> 23) - 127;
    return m << (e + 1);
}

/**
 * spatial_pooling() of one segment. The CPU sums the top k c-values in
 * double, in the order that quick_select() leaves them. c-values are
 * multiples of 2^-24, so while the sum is below 2^29 every partial sum is
 * exact, and the result does not depend on the order. Above, the exact mode
 * runs quick_select() on the c-values.
 */
static double pool(CambiStateMetal *s, unsigned slot, unsigned seg)
{
    const CambiMetalOut *out = B(s->out[slot]).contents;
    const CambiMetalSeg *g = &s->segs[seg];
    uint64_t sum = 0;
    uint32_t cnt = 0;
    for (unsigned t = 0; t < CAMBI_METAL_SEL_TGS; t++) {
        sum += out->sum[seg][t];
        cnt += out->cnt[seg][t];
    }
    sum += (uint64_t)(g->k - cnt) * fixed_24(out->sel[seg].t_bits);

    if (sum < (1ull << 53) || s->ms->fast_math)
        return (double)sum * 0x1p-24 / g->k;

    float *c = malloc((size_t)g->n * sizeof(float));
    if (!c) return NAN;
    memcpy(c, (const float *)B(s->c_values[slot]).contents + g->offset,
           (size_t)g->n * sizeof(float));
    quick_select(c, g->n, g->k);
    const double avg = average_topk_elements(c, g->k);
    free(c);
    return avg;
}

static double pass_score(CambiStateMetal *s, unsigned slot, unsigned pass)
{
    const CambiPass *p = &s->pass[pass];
    double scores_per_scale[NUM_SCALES];
    for (unsigned scale = 0; scale < NUM_SCALES; scale++)
        scores_per_scale[scale] = pool(s, slot, p->seg + scale);
    uint16_t pixels_in_window = get_pixels_in_window(p->window_size);
    return weight_scores_per_scale(scores_per_scale, pixels_in_window);
}

static double combine_dist_src_scores(double dist_score, double src_score) {
    return MAX(0, dist_score - src_score);
}

/** The scores of extract() of cambi.c */
static int write_scores(CambiStateMetal *s, unsigned slot, unsigned index,
                        VmafFeatureCollector *feature_collector)
{
    const CambiMetalOut *out = B(s->out[slot]).contents;
    if (out->invalid) {
        vmaf_log(
            VMAF_LOG_LEVEL_ERROR,
            "Invalid input. The input contains values greater than %d, which exceeds the maximum value for a %d-bit depth format.",
            (1 << s->bpc) - 1, s->bpc
        );
        return -EINVAL;
    }

    double dist_score = pass_score(s, slot, 0);
    int err = vmaf_feature_collector_append_with_dict(
        feature_collector, s->feature_name_dict, "Cambi_feature_cambi_score", MIN(dist_score, s->cambi_max_val), index
    );
    if (err) return err;

    if (s->full_ref) {
        double src_score = pass_score(s, slot, 1);
        err = vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, "cambi_source", MIN(src_score, s->cambi_max_val), index
        );
        if (err) return err;

        double combined_score = combine_dist_src_scores(dist_score, src_score);
        err = vmaf_feature_collector_append_with_dict(
            feature_collector, s->feature_name_dict, "cambi_full_reference", MIN(combined_score, s->cambi_max_val), index
        );
        if (err) return err;
    }

    return 0;
}

static int extract_fex(VmafFeatureExtractor *fex,
                       VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                       VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                       unsigned index, VmafFeatureCollector *feature_collector)
{
    CambiStateMetal *s = fex->priv;
    if (s->cpu) {
        return s->cpu->extract(s->cpu, ref_pic, ref_pic_90, dist_pic,
                               dist_pic_90, index, feature_collector);
    }

    static const char *const names[] = {
        "cambi_metal_validate", "cambi_metal_preprocess", "cambi_metal_mask_h",
        "cambi_metal_mask_v", "cambi_metal_mode", "cambi_metal_keys",
        "cambi_metal_zero", "cambi_metal_cvalues",
    };
    id<MTLComputePipelineState> pl[8];
    for (unsigned i = 0; i < 8; i++) {
        pl[i] = vmaf_metal_pipeline(s->ms, names[i], i < 2 ? s->bpc : 0);
        if (!pl[i]) return -EINVAL;
    }
    id<MTLComputePipelineState> hist =
        vmaf_metal_pipeline(s->ms, "cambi_metal_sel_hist", 0);
    id<MTLComputePipelineState> find =
        vmaf_metal_pipeline(s->ms, "cambi_metal_sel_find", 0);
    id<MTLComputePipelineState> sum =
        vmaf_metal_pipeline(s->ms, "cambi_metal_sel_sum", 0);
    if (!hist || !find || !sum) return -EINVAL;

    const unsigned slot = vmaf_metal_frames_acquire(&s->frames);
    const size_t bpp = s->bpc > 8 ? 2 : 1;
    vmaf_metal_copy_plane(B(s->in[0][slot]).contents, s->w * bpp, dist_pic, 0);
    if (s->full_ref)
        vmaf_metal_copy_plane(B(s->in[1][slot]).contents, s->w * bpp, ref_pic, 0);
    ((CambiMetalOut *)B(s->out[slot]).contents)->invalid = 0;

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [vmaf_metal_queue(s->ms) commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit fillBuffer:B(s->hist) range:NSMakeRange(0, B(s->hist).length)
                   value:0];
        [blit endEncoding];

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        for (unsigned p = 0; p < s->n_pass; p++)
            encode_pass(s, enc, slot, p, pl);
        encode_select(s, enc, slot, hist, find, sum);
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
    CambiStateMetal *s = fex->priv;
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
    "Cambi_feature_cambi_score",
    NULL
};

VmafFeatureExtractor vmaf_fex_cambi_metal = {
    .name = "cambi_metal",
    .init = init_fex,
    .extract = extract_fex,
    .options = options,
    .close = close_fex,
    .priv_size = sizeof(CambiStateMetal),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_METAL,
};
