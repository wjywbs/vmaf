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

#import <Metal/Metal.h>

#include <errno.h>
#include <math.h>
#include <string.h>

#include "barten_csf_tools.h"
#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "integer_adm.h"
#include "log.h"
#include "metal/common.h"
#include "opt.h"

#include "integer_adm_metal.h"

typedef struct AdmStateMetal {
    // options, see integer_adm.c
    bool debug;
    bool adm_skip_aim;
    bool adm_skip_scale0;
    double adm_csf_diag_scale;
    double adm_csf_scale;
    double adm_dlm_weight;
    double adm_enhn_gain_limit;
    double adm_norm_view_dist;
    double adm_noise_weight;
    double adm_min_val;
    int adm_ref_display_height;
    int adm_csf_mode;

    VmafDictionary *feature_name_dict;
    VmafMetalState *ms;
    VmafMetalFrames frames;
    unsigned w, h, bpc;
    AdmMetalParams params[4];
    float rfactor[4][3];
    unsigned den_rows, cm_rows;

    // id<MTLBuffer>, retained
    void *src_ref[VMAF_METAL_FRAMES], *src_dis[VMAF_METAL_FRAMES];
    void *ref_a[2], *dis_a[2]; // low-pass bands, scale s writes [s & 1]
    void *dec;                 // decouple planes r_h, r_v, r_d, a_h, a_v, a_d
    void *den_out[VMAF_METAL_FRAMES], *cm_out[VMAF_METAL_FRAMES]; // per row sums
} AdmStateMetal;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(AdmStateMetal, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "adm_csf_scale",
        .alias = "scf",
        .help = "scale coefficient for the horizontal & vertical direction terms of CSF",
        .offset = offsetof(AdmStateMetal, adm_csf_scale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_CSF_SCALE,
        .min = 0.0,
        .max = 50.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_csf_diag_scale",
        .alias = "scfd",
        .help = "scale coefficient for the diagonal direction term of CSF",
        .offset = offsetof(AdmStateMetal, adm_csf_diag_scale),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_CSF_DIAG_SCALE,
        .min = 0.0,
        .max = 50.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_dlm_weight",
        .alias = "dlmw",
        .help = "linear weighting between DLM and AIM; 1 corresponds to DLM-only",
        .offset = offsetof(AdmStateMetal, adm_dlm_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 0.5,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_enhn_gain_limit",
        .alias = "egl",
        .help = "enhancement gain imposed on adm, must be >= 1.0, "
                "where 1.0 means the gain is completely disabled",
        .offset = offsetof(AdmStateMetal, adm_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .min = 1.0,
        .max = DEFAULT_ADM_ENHN_GAIN_LIMIT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_norm_view_dist",
        .alias = "nvd",
        .help = "normalized viewing distance = viewing distance / ref display's physical height",
        .offset = offsetof(AdmStateMetal, adm_norm_view_dist),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NORM_VIEW_DIST,
        .min = 0.75,
        .max = 24.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_ref_display_height",
        .alias = "rdh",
        .help = "reference display height in pixels",
        .offset = offsetof(AdmStateMetal, adm_ref_display_height),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_ADM_REF_DISPLAY_HEIGHT,
        .min = 1,
        .max = 4320,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_csf_mode",
        .alias = "csf",
        .help = "contrast sensitivity function",
        .offset = offsetof(AdmStateMetal, adm_csf_mode),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = DEFAULT_ADM_CSF_MODE,
        .min = 0,
        .max = 3,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_noise_weight",
        .alias = "nw",
        .help = "noise weight",
        .offset = offsetof(AdmStateMetal, adm_noise_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_NOISE_WEIGHT,
        .min = 0.0,
        .max = 1500.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_skip_aim",
        .help = "skip the calculation of AIM",
        .offset = offsetof(AdmStateMetal, adm_skip_aim),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "adm_skip_scale0",
        .alias = "ssz",
        .help = "skip the calculation of scale 0",
        .offset = offsetof(AdmStateMetal, adm_skip_scale0),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "adm_min_val",
        .alias = "min",
        .help = "minimum value allowed; lower values will be clipped to this value",
        .offset = offsetof(AdmStateMetal, adm_min_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_ADM_MIN_VAL,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    { 0 }
};

/** integer_adm.c */
static float dwt_quant_step(const struct dwt_model_params *params, int lambda,
                            int theta, double adm_norm_view_dist,
                            int adm_ref_display_height)
{
    float r = adm_norm_view_dist * adm_ref_display_height * M_PI / 180.0;
    float temp = log10(pow(2.0, lambda + 1)*params->f0*params->g[theta] / r);
    float Q = 2.0*params->a*pow(10.0, params->k*temp*temp) /
        dwt_7_9_basis_function_amplitudes[lambda][theta];
    return Q;
}

/** factor1, factor2 of adm_csf(), adm_cm() and friends */
static void csf_rfactor(const AdmStateMetal *s, int scale, float rfactor[3])
{
    const double nvd = s->adm_norm_view_dist;
    const int rdh = s->adm_ref_display_height;
    float factor1, factor2;
    if (s->adm_csf_mode == ADM_CSF_MODE_BARTEN) {
        factor1 = barten_csf(scale, nvd, rdh, DEFAULT_ADM_CSF_LUM, s->adm_csf_scale);
        factor2 = barten_csf(scale, nvd, rdh, DEFAULT_ADM_CSF_LUM, s->adm_csf_diag_scale);
    } else if (s->adm_csf_mode == ADM_CSF_MODE_BARTEN_WATSON_BLEND) {
        factor1 = barten_watson_blend_csf(scale, 0, nvd, rdh);
        factor2 = barten_watson_blend_csf(scale, 1, nvd, rdh);
    } else if (s->adm_csf_mode == ADM_CSF_MODE_BARTEN_WATSON_BLEND_MAE) {
        factor1 = barten_watson_blend_csf_mae(scale, 0, nvd, rdh);
        factor2 = barten_watson_blend_csf_mae(scale, 1, nvd, rdh);
    } else {
        factor1 = 1.0f / dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 1, nvd, rdh);
        factor2 = 1.0f / dwt_quant_step(&dwt_7_9_YCbCr_threshold[0], scale, 2, nvd, rdh);
    }
    rfactor[0] = factor1;
    rfactor[1] = factor1;
    rfactor[2] = factor2;
}

/**
 * Rounding parameters of one scale, with the expressions (and integer types)
 * of integer_adm.c.
 */
static void scale_params(AdmStateMetal *s, int scale, int w, int h,
                         AdmMetalParams *p)
{
    p->w = w;
    p->h = h;
    p->stride = w;
    p->plane = w * h;

    // adm_decouple(), adm_csf()
    int left = w * ADM_BORDER_FACTOR - 0.5 - 1;
    int top = h * ADM_BORDER_FACTOR - 0.5 - 1;
    int right = w - left + 2;
    int bottom = h - top + 2;
    p->dec_left = left < 0 ? 0 : left;
    p->dec_top = top < 0 ? 0 : top;
    p->dec_right = right > w ? w : right;
    p->dec_bottom = bottom > h ? h : bottom;

    // adm_csf_den_*(), adm_cm()
    p->left = w * ADM_BORDER_FACTOR - 0.5;
    p->top = h * ADM_BORDER_FACTOR - 0.5;
    p->right = w - p->left;
    p->bottom = h - p->top;

    float *rfactor = s->rfactor[scale];
    csf_rfactor(s, scale, rfactor);

    if (scale == 0) {
        int32_t shift_accum = (int32_t)ceil(log2((p->bottom - p->top) *
                                                 (p->right - p->left)) - 20);
        shift_accum = shift_accum > 0 ? shift_accum : 0;
        p->den_shift_row = shift_accum;
        p->den_add_row = shift_accum > 0 ? (1 << (shift_accum - 1)) : 0;

        uint16_t i_rfactor[3];
        if (fabs(s->adm_norm_view_dist * s->adm_ref_display_height -
                 DEFAULT_ADM_NORM_VIEW_DIST * DEFAULT_ADM_REF_DISPLAY_HEIGHT) < 1.0e-8 &&
            s->adm_csf_mode == ADM_CSF_MODE_WATSON97) {
            i_rfactor[0] = 36453;
            i_rfactor[1] = 36453;
            i_rfactor[2] = 49417;
        } else {
            const double pow2_21 = pow(2, 21);
            const double pow2_23 = pow(2, 23);
            i_rfactor[0] = (uint16_t)(rfactor[0] * pow2_21);
            i_rfactor[1] = (uint16_t)(rfactor[1] * pow2_21);
            i_rfactor[2] = (uint16_t)(rfactor[2] * pow2_23);
        }
        for (int k = 0; k < 3; k++) p->i_rfactor[k] = i_rfactor[k];

        const uint32_t shift_xhcub = (uint32_t)ceil(log2(w) - 4);
        const uint32_t shift_xdcub = (uint32_t)ceil(log2(w) - 3);
        p->cm_shift_cub[0] = p->cm_shift_cub[1] = shift_xhcub;
        p->cm_add_cub[0] = p->cm_add_cub[1] = (uint32_t)pow(2, (shift_xhcub - 1));
        p->cm_shift_cub[2] = shift_xdcub;
        p->cm_add_cub[2] = (uint32_t)pow(2, (shift_xdcub - 1));
    } else {
        const uint32_t shift_sq[3] = { 31, 30, 31 };
        p->den_shift_sq = shift_sq[scale - 1];
        p->den_add_sq = 1u << shift_sq[scale - 1];
        const uint32_t shift_cub = (uint32_t)ceil(log2(p->right - p->left));
        p->den_shift_cub = shift_cub;
        p->den_add_cub = (uint32_t)pow(2, (shift_cub - 1));
        const uint32_t shift_accum = (uint32_t)ceil(log2(p->bottom - p->top));
        p->den_shift_row = shift_accum;
        p->den_add_row = (uint32_t)pow(2, (shift_accum - 1));

        const double pow2_32 = pow(2, 32);
        for (int k = 0; k < 3; k++)
            p->i_rfactor[k] = (uint32_t)(rfactor[k] * pow2_32);

        const uint32_t cm_shift_cub = (uint32_t)ceil(log2(w));
        for (int k = 0; k < 3; k++) {
            p->cm_shift_cub[k] = cm_shift_cub;
            p->cm_add_cub[k] = (uint32_t)pow(2, (cm_shift_cub - 1));
        }
    }
    const uint32_t shift_inner_accum = (uint32_t)ceil(log2(h));
    p->cm_shift_row = shift_inner_accum;
    p->cm_add_row = (uint32_t)pow(2, (shift_inner_accum - 1));
}

/** Final part of adm_csf_den_scale(), adm_csf_den_s123(). */
static float den_score(const AdmStateMetal *s, int scale, const uint64_t *rows)
{
    const AdmMetalParams *p = &s->params[scale];
    uint64_t accum_h = 0, accum_v = 0, accum_d = 0;
    for (int i = p->top; i < p->bottom; i++) {
        const uint64_t *r = rows + (p->den_row_offset + i) * ADM_METAL_N_DEN;
        accum_h += r[0];
        accum_v += r[1];
        accum_d += r[2];
    }

    double shift_csf;
    if (scale == 0) {
        shift_csf = pow(2, (18 - (int32_t)p->den_shift_row));
    } else {
        const uint32_t accum_convert_float[3] = { 32, 27, 23 };
        shift_csf = pow(2, (accum_convert_float[scale - 1] - p->den_shift_row -
                            p->den_shift_cub));
    }
    const float *rfactor = s->rfactor[scale];
    double csf_h = (double)(accum_h / shift_csf) * pow(rfactor[0], 3);
    double csf_v = (double)(accum_v / shift_csf) * pow(rfactor[1], 3);
    double csf_d = (double)(accum_d / shift_csf) * pow(rfactor[2], 3);

    float powf_add = powf((p->bottom - p->top) * (p->right - p->left) *
                          s->adm_noise_weight, 1.0f / 3.0f);
    float den_scale_h = powf(csf_h, 1.0f / 3.0f) + powf_add;
    float den_scale_v = powf(csf_v, 1.0f / 3.0f) + powf_add;
    float den_scale_d = powf(csf_d, 1.0f / 3.0f) + powf_add;

    return (den_scale_h + den_scale_v + den_scale_d);
}

/** Final part of adm_cm(), i4_adm_cm(). `first`: 0 DLM, 3 AIM. */
static float cm_score(const AdmStateMetal *s, int scale, const int64_t *rows,
                      int first, double adm_noise_weight)
{
    const AdmMetalParams *p = &s->params[scale];
    int64_t accum_h = 0, accum_v = 0, accum_d = 0;
    for (int i = p->top; i < p->bottom; i++) {
        const int64_t *r = rows + (p->cm_row_offset + i) * ADM_METAL_N_CM + first;
        accum_h += r[0];
        accum_v += r[1];
        accum_d += r[2];
    }

    const uint32_t shift_inner_accum = p->cm_shift_row;
    float f_accum_h, f_accum_v, f_accum_d;
    if (scale == 0) {
        const uint32_t shift_xhcub = p->cm_shift_cub[0];
        const uint32_t shift_xvcub = p->cm_shift_cub[1];
        const uint32_t shift_xdcub = p->cm_shift_cub[2];
        f_accum_h = (float)(accum_h / pow(2, (52 - shift_xhcub - shift_inner_accum)));
        f_accum_v = (float)(accum_v / pow(2, (52 - shift_xvcub - shift_inner_accum)));
        f_accum_d = (float)(accum_d / pow(2, (57 - shift_xdcub - shift_inner_accum)));
    } else {
        const uint32_t shift_cub = p->cm_shift_cub[0];
        float final_shift[3] = { pow(2,(45 - shift_cub - shift_inner_accum)),
                                 pow(2,(39 - shift_cub - shift_inner_accum)),
                                 pow(2,(36 - shift_cub - shift_inner_accum)) };
        f_accum_h = (float)(accum_h / final_shift[scale - 1]);
        f_accum_v = (float)(accum_v / final_shift[scale - 1]);
        f_accum_d = (float)(accum_d / final_shift[scale - 1]);
    }

    const int area = (p->bottom - p->top) * (p->right - p->left);
    float num_scale_h = powf(f_accum_h, 1.0f / 3.0f) + powf(area * adm_noise_weight, 1.0f / 3.0f);
    float num_scale_v = powf(f_accum_v, 1.0f / 3.0f) + powf(area * adm_noise_weight, 1.0f / 3.0f);
    float num_scale_d = powf(f_accum_d, 1.0f / 3.0f) + powf(area * adm_noise_weight, 1.0f / 3.0f);

    return (num_scale_h + num_scale_v + num_scale_d);
}

static void *buffer_new(VmafMetalState *ms, size_t size)
{
    id<MTLBuffer> buf = vmaf_metal_buffer(ms, size);
    return buf ? (void *)CFBridgingRetain(buf) : NULL;
}

static inline id<MTLBuffer> B(void *buf)
{
    return (__bridge id<MTLBuffer>)buf;
}

static void buffers_free(AdmStateMetal *s)
{
    vmaf_metal_frames_close(&s->frames);
    void **bufs[] = {
        &s->src_ref[0], &s->src_ref[1], &s->src_ref[2],
        &s->src_dis[0], &s->src_dis[1], &s->src_dis[2],
        &s->den_out[0], &s->den_out[1], &s->den_out[2],
        &s->cm_out[0], &s->cm_out[1], &s->cm_out[2],
        &s->ref_a[0], &s->ref_a[1], &s->dis_a[0], &s->dis_a[1], &s->dec,
    };
    _Static_assert(VMAF_METAL_FRAMES == 3, "update bufs[]");
    for (unsigned i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) {
        if (*bufs[i]) CFRelease(*bufs[i]);
        *bufs[i] = NULL;
    }
}

static uint64_t double_bits(double d)
{
    uint64_t u;
    memcpy(&u, &d, sizeof(u));
    return u;
}

static int init_fex(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                    unsigned bpc, unsigned w, unsigned h)
{
    AdmStateMetal *s = fex->priv;
    (void)pix_fmt;

    s->ms = fex->metal_state ? fex->metal_state : vmaf_metal_state_default();
    if (!s->ms) return -ENODEV;
    s->w = w;
    s->h = h;
    s->bpc = bpc;

    // adm_decouple*()
    const float cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);
    const double gain = s->adm_enhn_gain_limit;

    int sw = w, sh = h;
    s->den_rows = s->cm_rows = 0;
    for (int scale = 0; scale < 4; scale++) {
        AdmMetalParams *p = &s->params[scale];
        memset(p, 0, sizeof(*p));
        p->gain = double_bits(gain);
        p->gain_f = gain;
        p->gain_int = gain == floor(gain) && gain <= (1 << 20) ? (int)gain : 0;
        p->cos_1deg_sq = double_bits(cos_1deg_sq);
        p->cos_1deg_sq_f = cos_1deg_sq;
        p->scale = scale;
        p->bpc = bpc;
        p->src_w = sw;
        p->src_h = sh;
        p->src_stride = scale == 0 ? (int)w : s->params[scale - 1].stride;
        scale_params(s, scale, (sw + 1) / 2, (sh + 1) / 2, p);
        p->den_row_offset = s->den_rows;
        p->cm_row_offset = s->cm_rows;
        s->den_rows += p->h;
        s->cm_rows += p->h;
        sw = p->w;
        sh = p->h;
    }

    const size_t bpp = bpc > 8 ? 2 : 1;
    const size_t band = (size_t)s->params[0].w * s->params[0].h;
    if (vmaf_metal_frames_init(&s->frames, s->ms)) goto fail;
    bool ok = true;
    for (int k = 0; k < VMAF_METAL_FRAMES; k++) {
        s->src_ref[k] = buffer_new(s->ms, (size_t)w * h * bpp);
        s->src_dis[k] = buffer_new(s->ms, (size_t)w * h * bpp);
        s->den_out[k] = buffer_new(s->ms, s->den_rows * ADM_METAL_N_DEN * sizeof(uint64_t));
        s->cm_out[k] = buffer_new(s->ms, s->cm_rows * ADM_METAL_N_CM * sizeof(int64_t));
        ok = ok && s->src_ref[k] && s->src_dis[k] && s->den_out[k] && s->cm_out[k];
    }
    for (int k = 0; k < 2; k++) {
        s->ref_a[k] = buffer_new(s->ms, band * sizeof(int32_t));
        s->dis_a[k] = buffer_new(s->ms, band * sizeof(int32_t));
    }
    s->dec = buffer_new(s->ms, 6 * band * sizeof(int32_t));
    if (!ok || !s->ref_a[0] || !s->ref_a[1] || !s->dis_a[0] || !s->dis_a[1] ||
        !s->dec)
        goto fail;

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) goto fail;

    return 0;

fail:
    buffers_free(s);
    return -ENOMEM;
}

/** One row per threadgroup, threads loop over the n columns. */
static void dispatch_rows(id<MTLComputeCommandEncoder> enc,
                          id<MTLComputePipelineState> pipeline,
                          unsigned rows, unsigned n)
{
    NSUInteger tw = MIN((n + 31) / 32 * 32, ADM_METAL_MAX_TG_W);
    tw = MIN(tw, pipeline.maxTotalThreadsPerThreadgroup / 32 * 32);
    [enc setComputePipelineState:pipeline];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
}

typedef struct {
    double score, num, den, aim;
    double scores[8];
} AdmScore;

static int write_scores(const AdmStateMetal *s, const AdmScore *out,
                        unsigned index, VmafFeatureCollector *feature_collector)
{
    VmafDictionary *dict = s->feature_name_dict;
    int err = 0;

    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            dict, "VMAF_integer_feature_adm2_score", out->score, index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            dict, "VMAF_integer_feature_aim_score", out->aim, index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            dict, "VMAF_integer_feature_adm3_score",
            MAX(out->score * s->adm_dlm_weight + (1 - out->aim) * (1 - s->adm_dlm_weight), s->adm_min_val),
            index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            dict, "integer_adm_scale0", out->scores[0] / out->scores[1], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            dict, "integer_adm_scale1", out->scores[2] / out->scores[3], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            dict, "integer_adm_scale2", out->scores[4] / out->scores[5], index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            dict, "integer_adm_scale3", out->scores[6] / out->scores[7], index);

    if (!s->debug) return err;

    static const char *debug_names[8] = {
        "integer_adm_num_scale0", "integer_adm_den_scale0",
        "integer_adm_num_scale1", "integer_adm_den_scale1",
        "integer_adm_num_scale2", "integer_adm_den_scale2",
        "integer_adm_num_scale3", "integer_adm_den_scale3",
    };
    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            dict, "integer_adm", out->score, index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            dict, "integer_adm_num", out->num, index);
    err |= vmaf_feature_collector_append_with_dict(feature_collector,
            dict, "integer_adm_den", out->den, index);
    for (int k = 0; k < 8; k++) {
        err |= vmaf_feature_collector_append_with_dict(feature_collector,
                dict, debug_names[k], out->scores[k], index);
    }
    return err;
}

/** integer_compute_adm(), from the row sums of frame slot `slot`. */
static int compute_scores(const AdmStateMetal *s, unsigned slot,
                          unsigned index, VmafFeatureCollector *feature_collector)
{
    const uint64_t *den_rows = B(s->den_out[slot]).contents;
    const int64_t *cm_rows = B(s->cm_out[slot]).contents;
    double num = 0, den = 0, aim_num = 0;
    AdmScore out = { 0 };
    for (int scale = 0; scale < 4; scale++) {
        float num_scale = 0.0;
        float den_scale = 0.0;
        float aim_num_scale = 0.0;

        if (scale == 0 && s->adm_skip_scale0) {
            den_scale = 1e-10;  // avoid divide by zero
        } else {
            den_scale = den_score(s, scale, den_rows);
            num_scale = cm_score(s, scale, cm_rows, 0, s->adm_noise_weight);
            if (!s->adm_skip_aim)
                aim_num_scale = cm_score(s, scale, cm_rows, 3, 0.0);
        }

        num += num_scale;
        den += den_scale;
        aim_num += aim_num_scale;

        out.scores[2 * scale + 0] = num_scale;
        out.scores[2 * scale + 1] = den_scale;
    }

    const double numden_limit = 1e-10 * (s->w * s->h) / (1920.0 * 1080.0);
    double num_v = num < numden_limit ? 0 : num;
    double den_v = den < numden_limit ? 0 : den;
    if (den_v == 0.0) {
        out.score = 1.0f;
    } else {
        // normalize AIM score by the DLM denominator
        out.aim = aim_num / den_v;
        out.score = num_v / den_v;
    }
    out.num = num_v;
    out.den = den_v;

    return write_scores(s, &out, index, feature_collector);
}

static int extract_fex(VmafFeatureExtractor *fex,
                       VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                       VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                       unsigned index, VmafFeatureCollector *feature_collector)
{
    AdmStateMetal *s = fex->priv;
    (void)ref_pic_90;
    (void)dist_pic_90;

    // current implementation is limited by the 16-bit data pipeline, thus
    // cannot handle an angular frequency smaller than 1080p * 3H
    if (s->adm_norm_view_dist * s->adm_ref_display_height <
        DEFAULT_ADM_NORM_VIEW_DIST * DEFAULT_ADM_REF_DISPLAY_HEIGHT)
        return -EINVAL;

    id<MTLComputePipelineState> dwt[4], cm[4];
    for (int scale = 0; scale < 4; scale++) {
        const bool lo_only = scale == 0 && s->adm_skip_scale0;
        const uint32_t variant = scale | (scale == 0 ? s->bpc << 8 : 0);
        dwt[scale] = vmaf_metal_pipeline(s->ms, "adm_metal_dwt",
            variant | (lo_only ? ADM_METAL_VARIANT_LO_ONLY : 0));
        cm[scale] = vmaf_metal_pipeline(s->ms, "adm_metal_cm",
            variant | (s->adm_skip_aim ? ADM_METAL_VARIANT_SKIP_AIM : 0));
        if (!dwt[scale] || !cm[scale]) return -EINVAL;
    }

    const unsigned slot = vmaf_metal_frames_acquire(&s->frames);
    const size_t bpp = s->bpc > 8 ? 2 : 1;
    vmaf_metal_copy_plane(B(s->src_ref[slot]).contents, s->w * bpp, ref_pic, 0);
    vmaf_metal_copy_plane(B(s->src_dis[slot]).contents, s->w * bpp, dist_pic, 0);

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [vmaf_metal_queue(s->ms) commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        for (int scale = 0; scale < 4; scale++) {
            const AdmMetalParams *p = &s->params[scale];
            const bool lo_only = scale == 0 && s->adm_skip_scale0;

            [enc setBytes:p length:sizeof(*p) atIndex:0];
            [enc setBuffer:B(scale ? s->ref_a[(scale - 1) & 1] : s->src_ref[slot]) offset:0 atIndex:1];
            [enc setBuffer:B(scale ? s->dis_a[(scale - 1) & 1] : s->src_dis[slot]) offset:0 atIndex:2];
            [enc setBuffer:B(s->ref_a[scale & 1]) offset:0 atIndex:3];
            [enc setBuffer:B(s->dis_a[scale & 1]) offset:0 atIndex:4];
            [enc setBuffer:B(s->dec) offset:0 atIndex:5];
            [enc setBuffer:B(s->den_out[slot]) offset:0 atIndex:6];
            dispatch_rows(enc, dwt[scale], p->h, p->w);
            if (lo_only) continue;

            [enc setBuffer:B(s->dec) offset:0 atIndex:1];
            [enc setBuffer:B(s->cm_out[slot]) offset:0 atIndex:2];
            dispatch_rows(enc, cm[scale], p->bottom - p->top, p->right - p->left);
        }

        [enc endEncoding];
        vmaf_metal_frames_commit(&s->frames, slot, cb, ^int(int err) {
            if (err) return err;
            return compute_scores(s, slot, index, feature_collector);
        });
    }
    return 0;
}

static int close_fex(VmafFeatureExtractor *fex)
{
    AdmStateMetal *s = fex->priv;
    buffers_free(s);
    vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features[] = {
    "VMAF_integer_feature_adm2_score", "VMAF_integer_feature_aim_score", "VMAF_integer_feature_adm3_score",
    "integer_adm_scale0", "integer_adm_scale1", "integer_adm_scale2", "integer_adm_scale3",
    "integer_adm", "integer_adm_num", "integer_adm_den",
    "integer_adm_num_scale0", "integer_adm_den_scale0", "integer_adm_num_scale1",
    "integer_adm_den_scale1", "integer_adm_num_scale2", "integer_adm_den_scale2",
    "integer_adm_num_scale3", "integer_adm_den_scale3",
    NULL
};

VmafFeatureExtractor vmaf_fex_integer_adm_metal = {
    .name = "adm_metal",
    .init = init_fex,
    .extract = extract_fex,
    .options = options,
    .close = close_fex,
    .priv_size = sizeof(AdmStateMetal),
    .flags = VMAF_FEATURE_EXTRACTOR_METAL,
    .provided_features = provided_features,
};
