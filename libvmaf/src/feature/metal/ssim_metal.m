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
 * float_ssim_metal and float_ms_ssim_metal: float_ssim.c and float_ms_ssim.c
 * on Metal. The kernels (ssim.metal) filter the pictures and compute l, c
 * and s per pixel. The host adds them up per scale, in exact mode in the
 * CPU's order, and computes the scores as _iqa_ssim(), compute_ssim() and
 * compute_ms_ssim() do.
 */

#import <Metal/Metal.h>

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "iqa/math_utils.h"
#include "iqa/ssim_tools.h"
#include "metal/common.h"
#include "opt.h"
#include "ssim_metal.h"

typedef struct SsimScale {
    int w, h;         // input of the scale
    int dst_w, dst_h; // after the 11x11 filter
    unsigned out_offset;  // pixels, in out_l, out_c, out_s
    unsigned part_offset; // threadgroups, in part
    unsigned n_parts;
} SsimScale;

typedef struct SsimStateMetal {
    // options, see float_ssim.c and float_ms_ssim.c
    bool enable_lcs;
    bool enable_db;
    bool clip_db;
    int scale; // float_ssim only
    double max_db;

    bool ms; // MS-SSIM
    VmafFeatureExtractor *cpu; // delegate, see cpu_init()

    VmafMetalState *mtl;
    VmafMetalFrames frames;
    unsigned bpc;
    int w, h;
    int factor; // float_ssim: decimation of the picture, 1 = none
    int n_scales;
    SsimScale sc[SSIM_METAL_SCALES];
    float c1, c2, c3;
    SsimMetalLoadParams load;

    // id<MTLBuffer>, retained. Per slot: the input, and the output of
    // conv_v. Shared: the intermediates.
    void *in[VMAF_METAL_FRAMES];
    void *out_l[VMAF_METAL_FRAMES], *out_c[VMAF_METAL_FRAMES];
    void *out_s[VMAF_METAL_FRAMES], *part[VMAF_METAL_FRAMES];
    void *full;                       // picture_copy() of ref and dist
    void *planes[SSIM_METAL_SCALES];  // input of each scale
    void *maps;                       // conv_h output
    void *dec_kernel;
} SsimStateMetal;

static const VmafOption options_ssim[] = {
    {
        .name = "enable_lcs",
        .help = "enable luminance, contrast and structure intermediate output",
        .offset = offsetof(SsimStateMetal, enable_lcs),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "enable_db",
        .help = "write SSIM values as dB",
        .offset = offsetof(SsimStateMetal, enable_db),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "clip_db",
        .help = "clip dB scores",
        .offset = offsetof(SsimStateMetal, clip_db),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "scale",
        .help = "decimation scale factor (0=auto, 1=no downscaling, 2-10=explicit)",
        .offset = offsetof(SsimStateMetal, scale),
        .type = VMAF_OPT_TYPE_INT,
        .default_val.i = 0,
        .min = 0,
        .max = 10,
    },
    { 0 }
};

static const VmafOption options_ms_ssim[] = {
    {
        .name = "enable_lcs",
        .help = "enable luminance, contrast and structure intermediate output",
        .offset = offsetof(SsimStateMetal, enable_lcs),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "enable_db",
        .help = "write MS-SSIM values as dB",
        .offset = offsetof(SsimStateMetal, enable_db),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "clip_db",
        .help = "clip dB scores",
        .offset = offsetof(SsimStateMetal, clip_db),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    { 0 }
};

/* ms_ssim.c: low-pass filter for down-sampling, and the exponents */
#define LPF_LEN 9
static const float g_lpf[LPF_LEN][LPF_LEN] = {
   { 0.000714f,-0.000450f,-0.002090f, 0.007132f, 0.016114f, 0.007132f,-0.002090f,-0.000450f, 0.000714f},
   {-0.000450f, 0.000283f, 0.001316f,-0.004490f,-0.010146f,-0.004490f, 0.001316f, 0.000283f,-0.000450f},
   {-0.002090f, 0.001316f, 0.006115f,-0.020867f,-0.047149f,-0.020867f, 0.006115f, 0.001316f,-0.002090f},
   { 0.007132f,-0.004490f,-0.020867f, 0.071207f, 0.160885f, 0.071207f,-0.020867f,-0.004490f, 0.007132f},
   { 0.016114f,-0.010146f,-0.047149f, 0.160885f, 0.363505f, 0.160885f,-0.047149f,-0.010146f, 0.016114f},
   { 0.007132f,-0.004490f,-0.020867f, 0.071207f, 0.160885f, 0.071207f,-0.020867f,-0.004490f, 0.007132f},
   {-0.002090f, 0.001316f, 0.006115f,-0.020867f,-0.047149f,-0.020867f, 0.006115f, 0.001316f,-0.002090f},
   {-0.000450f, 0.000283f, 0.001316f,-0.004490f,-0.010146f,-0.004490f, 0.001316f, 0.000283f,-0.000450f},
   { 0.000714f,-0.000450f,-0.002090f, 0.007132f, 0.016114f, 0.007132f,-0.002090f,-0.000450f, 0.000714f},
};
static float g_alphas[] = { 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.1333f };
static float g_betas[]  = { 0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f };
static float g_gammas[] = { 0.0448f, 0.2856f, 0.3001f, 0.2363f, 0.1333f };

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

static void buffers_free(SsimStateMetal *s)
{
    vmaf_metal_frames_close(&s->frames);
    for (unsigned k = 0; k < VMAF_METAL_FRAMES; k++) {
        release(&s->in[k]);
        release(&s->out_l[k]);
        release(&s->out_c[k]);
        release(&s->out_s[k]);
        release(&s->part[k]);
    }
    for (unsigned i = 0; i < SSIM_METAL_SCALES; i++) release(&s->planes[i]);
    release(&s->full);
    release(&s->maps);
    release(&s->dec_kernel);
}

/* -------------------------------------------------------------------------
 * Scores
 */

/**
 * The sums of _iqa_ssim() over l, c and s of one scale, in the CPU's order.
 * The arm64 CPU build vectorizes the loop by 8 pixels: there, it rounds
 * l * c * s before adding it; in the remaining columns of each row it fuses
 * the last multiplication and the addition.
 */
static void scale_sums(const double *L, const double *C, const float *S,
                       int w, int h, float *res, float *l_mean,
                       float *c_mean, float *s_mean)
{
    double ssim_sum = 0.0, l_sum = 0.0, c_sum = 0.0, s_sum = 0.0;
    const int w8 = w & ~7;
    for (int y = 0; y < h; ++y) {
        const double *lr = L + y * w, *cr = C + y * w;
        const float *sr = S + y * w;
        int x = 0;
        for (; x < w8; ++x) {
            const double l = lr[x], c = cr[x], s = sr[x];
            double lcs = l * c;
            lcs = lcs * s;
            ssim_sum += lcs;
            l_sum += l;
            c_sum += c;
            s_sum += s;
        }
        for (; x < w; ++x) {
            const double l = lr[x], c = cr[x], s = sr[x];
#if ARCH_AARCH64
            ssim_sum = fma(l * c, s, ssim_sum);
#else
            double lcs = l * c;
            lcs = lcs * s;
            ssim_sum += lcs;
#endif
            l_sum += l;
            c_sum += c;
            s_sum += s;
        }
    }
    *l_mean = (float)(l_sum / (double)(w*h));
    *c_mean = (float)(c_sum / (double)(w*h));
    *s_mean = (float)(s_sum / (double)(w*h));
    *res = (float)(ssim_sum / (double)(w*h));
}

/** Fast mode: the sums of the threadgroups. */
static void scale_sums_fast(const float *part, unsigned n, int w, int h,
                            float *res, float *l_mean, float *c_mean,
                            float *s_mean)
{
    double sum[4] = { 0 };
    for (unsigned i = 0; i < n; i++) {
        for (unsigned j = 0; j < 4; j++) sum[j] += part[4 * i + j];
    }
    *res = (float)(sum[0] / (double)(w*h));
    *l_mean = (float)(sum[1] / (double)(w*h));
    *c_mean = (float)(sum[2] / (double)(w*h));
    *s_mean = (float)(sum[3] / (double)(w*h));
}

#undef MIN
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

static double convert_to_db(double score, double max_db)
{
    return MIN(-10. * log10(1 - score), max_db);
}

static int write_scores(SsimStateMetal *s, unsigned slot, unsigned index,
                        VmafFeatureCollector *fc)
{
    float res[SSIM_METAL_SCALES], l[SSIM_METAL_SCALES], c[SSIM_METAL_SCALES];
    float sc[SSIM_METAL_SCALES];
    for (int i = 0; i < s->n_scales; i++) {
        const SsimScale *k = &s->sc[i];
        if (s->mtl->fast_math) {
            scale_sums_fast((const float *)B(s->part[slot]).contents +
                            4 * k->part_offset, k->n_parts, k->dst_w,
                            k->dst_h, &res[i], &l[i], &c[i], &sc[i]);
        } else {
            scale_sums((const double *)B(s->out_l[slot]).contents +
                       k->out_offset,
                       (const double *)B(s->out_c[slot]).contents +
                       k->out_offset,
                       (const float *)B(s->out_s[slot]).contents +
                       k->out_offset,
                       k->dst_w, k->dst_h, &res[i], &l[i], &c[i], &sc[i]);
        }
    }

    int err = 0;
    if (!s->ms) {
        // compute_ssim() and float_ssim.c
        double score = (double)res[0];
        if (s->enable_db) score = convert_to_db(score, s->max_db);
        err = vmaf_feature_collector_append(fc, "float_ssim", score, index);
        if (s->enable_lcs) {
            err |= vmaf_feature_collector_append(fc, "float_ssim_l",
                                                 (double)l[0], index);
            err |= vmaf_feature_collector_append(fc, "float_ssim_c",
                                                 (double)c[0], index);
            err |= vmaf_feature_collector_append(fc, "float_ssim_s",
                                                 (double)sc[0], index);
        }
        return err;
    }

    // compute_ms_ssim() and float_ms_ssim.c
    double msssim = 1.0;
    for (int idx = 0; idx < SSIM_METAL_SCALES; ++idx) {
        msssim *= pow(l[idx], g_alphas[idx]) * pow(c[idx], g_betas[idx]) *
                  pow(sc[idx], g_gammas[idx]);
        if (msssim == INFINITY) return -EINVAL;
    }
    double score = msssim;
    if (s->enable_db) score = convert_to_db(score, s->max_db);
    err = vmaf_feature_collector_append(fc, "float_ms_ssim", score, index);
    if (s->enable_lcs) {
        static const char *names[3][SSIM_METAL_SCALES] = {
            { "float_ms_ssim_l_scale0", "float_ms_ssim_l_scale1",
              "float_ms_ssim_l_scale2", "float_ms_ssim_l_scale3",
              "float_ms_ssim_l_scale4" },
            { "float_ms_ssim_c_scale0", "float_ms_ssim_c_scale1",
              "float_ms_ssim_c_scale2", "float_ms_ssim_c_scale3",
              "float_ms_ssim_c_scale4" },
            { "float_ms_ssim_s_scale0", "float_ms_ssim_s_scale1",
              "float_ms_ssim_s_scale2", "float_ms_ssim_s_scale3",
              "float_ms_ssim_s_scale4" },
        };
        const float *v[3] = { l, c, sc };
        for (int j = 0; j < 3; j++) {
            for (int i = 0; i < SSIM_METAL_SCALES; i++) {
                err |= vmaf_feature_collector_append(fc, names[j][i],
                                                     (double)v[j][i], index);
            }
        }
    }
    return err;
}

/* -------------------------------------------------------------------------
 * Initialization
 */

/**
 * Configurations that the kernels do not cover run on the CPU extractor:
 * bit depths other than 8, 10, 12 and 16 (picture_copy() handles only
 * those), and pictures too small for the 11x11 window at some scale, where
 * the CPU fails or reads outside the picture.
 */
static int cpu_init(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                    unsigned bpc, unsigned w, unsigned h)
{
    SsimStateMetal *s = fex->priv;
    extern VmafFeatureExtractor vmaf_fex_float_ssim, vmaf_fex_float_ms_ssim;
    const VmafFeatureExtractor *src =
        s->ms ? &vmaf_fex_float_ms_ssim : &vmaf_fex_float_ssim;

    VmafFeatureExtractor *cpu = malloc(sizeof(*cpu));
    if (!cpu) return -ENOMEM;
    memcpy(cpu, src, sizeof(*cpu));
    cpu->priv = calloc(1, cpu->priv_size);
    if (!cpu->priv) {
        free(cpu);
        return -ENOMEM;
    }

    // the option values as set by the framework, by name
    for (const VmafOption *o = cpu->options; o->name; o++) {
        const VmafOption *m = fex->options;
        while (m->name && strcmp(m->name, o->name)) m++;
        if (!m->name || m->type != o->type) continue;
        const uint8_t *src_opt = (const uint8_t *)s + m->offset;
        uint8_t *dst = (uint8_t *)cpu->priv + o->offset;
        switch (o->type) {
        case VMAF_OPT_TYPE_BOOL: memcpy(dst, src_opt, sizeof(bool)); break;
        case VMAF_OPT_TYPE_INT: memcpy(dst, src_opt, sizeof(int)); break;
        default: break;
        }
    }

    s->cpu = cpu;
    return cpu->init(cpu, pix_fmt, bpc, w, h);
}

/** Pixels and threadgroups of the outputs of each scale. */
static int scales_init(SsimStateMetal *s)
{
    id<MTLComputePipelineState> pl =
        vmaf_metal_pipeline(s->mtl, "ssim_metal_conv_v", 0);
    if (!pl) return -EINVAL;
    const unsigned tw = (unsigned)pl.threadExecutionWidth;
    const unsigned th =
        MAX(1, (unsigned)pl.maxTotalThreadsPerThreadgroup / tw / 4);

    unsigned out = 0, part = 0;
    for (int i = 0; i < s->n_scales; i++) {
        SsimScale *k = &s->sc[i];
        k->dst_w = k->w - SSIM_METAL_GAUSSIAN_LEN + 1;
        k->dst_h = k->h - SSIM_METAL_GAUSSIAN_LEN + 1;
        k->out_offset = out;
        k->part_offset = part;
        k->n_parts = ((k->dst_w + tw - 1) / tw) * ((k->dst_h + th - 1) / th);
        out += k->dst_w * k->dst_h;
        part += k->n_parts;
    }

    const bool fast = s->mtl->fast_math;
    for (unsigned k = 0; k < VMAF_METAL_FRAMES; k++) {
        s->out_l[k] = buffer_new(s->mtl, fast ? 0 : out * sizeof(uint64_t));
        s->out_c[k] = buffer_new(s->mtl, fast ? 0 : out * sizeof(uint64_t));
        s->out_s[k] = buffer_new(s->mtl, fast ? 0 : out * sizeof(float));
        s->part[k] = buffer_new(s->mtl, fast ? part * 4 * sizeof(float) : 0);
        if (!s->out_l[k] || !s->out_c[k] || !s->out_s[k] || !s->part[k])
            return -ENOMEM;
    }
    return 0;
}

static int init_fex(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                    unsigned bpc, unsigned w, unsigned h)
{
    SsimStateMetal *s = fex->priv;
    s->ms = !strcmp(fex->name, "float_ms_ssim_metal");

    const unsigned peak = (1 << bpc) - 1;
    if (s->clip_db) {
        const double mse = 0.5 / (w * h);
        s->max_db = ceil(10. * log10(peak * peak / mse));
    } else {
        s->max_db = INFINITY;
    }

    bool cpu = bpc != 8 && bpc != 10 && bpc != 12 && bpc != 16;
    s->factor = 1;
    if (s->ms) {
        // compute_ms_ssim(): "scale below 1x1"
        int cw = w, ch = h;
        for (int i = 0; i < SSIM_METAL_SCALES; i++) {
            cpu |= cw < SSIM_METAL_GAUSSIAN_LEN || ch < SSIM_METAL_GAUSSIAN_LEN;
            cw /= 2;
            ch /= 2;
        }
        s->n_scales = SSIM_METAL_SCALES;
        cw = w, ch = h;
        for (int i = 0; i < SSIM_METAL_SCALES; i++) {
            s->sc[i].w = cw;
            s->sc[i].h = ch;
            cw = cw / 2 + (cw & 1);
            ch = ch / 2 + (ch & 1);
        }
    } else {
        // compute_ssim()
        int scale;
        if (s->scale > 0) scale = s->scale;
        else scale = _max(1, _round((float)_min(w, h) / 256.0f));
        s->factor = scale;
        s->n_scales = 1;
        s->sc[0].w = scale > 1 ? (int)w / scale + (w & 1) : (int)w;
        s->sc[0].h = scale > 1 ? (int)h / scale + (h & 1) : (int)h;
        cpu |= s->sc[0].w < SSIM_METAL_GAUSSIAN_LEN ||
               s->sc[0].h < SSIM_METAL_GAUSSIAN_LEN;
    }
    if (cpu) return cpu_init(fex, pix_fmt, bpc, w, h);

    s->mtl = fex->metal_state ? fex->metal_state : vmaf_metal_state_default();
    if (!s->mtl) return -ENODEV;
    s->bpc = bpc;
    s->w = w;
    s->h = h;

    // _iqa_ssim()
    const float K1 = 0.01f, K2 = 0.03f;
    const int L = 255;
    s->c1 = (K1*L)*(K1*L);
    s->c2 = (K2*L)*(K2*L);
    s->c3 = s->c2 / 2.0f;

    s->load = (SsimMetalLoadParams) {
        .w = w, .h = h, .in_stride = w,
        .scaler = bpc == 10 ? 4.0f : bpc == 12 ? 16.0f : 256.0f,
    };

    if (vmaf_metal_frames_init(&s->frames, s->mtl)) goto fail;
    const size_t bpp = bpc > 8 ? 2 : 1;
    for (unsigned k = 0; k < VMAF_METAL_FRAMES; k++) {
        s->in[k] = buffer_new(s->mtl, 2 * (size_t)w * h * bpp);
        if (!s->in[k]) goto fail;
    }
    s->full = buffer_new(s->mtl, 2 * (size_t)w * h * sizeof(float));
    if (!s->full) goto fail;
    for (int i = 0; i < s->n_scales; i++) {
        if (i == 0 && s->factor == 1) continue; // the full picture
        s->planes[i] = buffer_new(s->mtl, 2 * (size_t)s->sc[i].w *
                                  s->sc[i].h * sizeof(float));
        if (!s->planes[i]) goto fail;
    }
    s->maps = buffer_new(s->mtl, SSIM_METAL_MAPS * sizeof(float) *
                         (size_t)(s->sc[0].w - SSIM_METAL_GAUSSIAN_LEN + 1) *
                         s->sc[0].h);
    if (!s->maps) goto fail;

    // decimation kernel: compute_ssim()'s low-pass filter, or g_lpf
    if (s->ms) {
        s->dec_kernel = buffer_new(s->mtl, sizeof(g_lpf));
        if (!s->dec_kernel) goto fail;
        memcpy(B(s->dec_kernel).contents, g_lpf, sizeof(g_lpf));
    } else if (s->factor > 1) {
        const int n = s->factor * s->factor;
        s->dec_kernel = buffer_new(s->mtl, n * sizeof(float));
        if (!s->dec_kernel) goto fail;
        float *k = B(s->dec_kernel).contents;
        for (int i = 0; i < n; i++) k[i] = 1.0f/(s->factor*s->factor);
    }

    if (scales_init(s)) goto fail;
    return 0;

fail:
    buffers_free(s);
    return -ENOMEM;
}

/* -------------------------------------------------------------------------
 * Extraction
 */

static void encode_decimate(SsimStateMetal *s, id<MTLComputeCommandEncoder> enc,
                            id<MTLComputePipelineState> pl, void *src,
                            int w, int h, void *dst, int factor, int kw)
{
    const SsimMetalDecParams p = {
        .w = w, .h = h,
        .out_w = w / factor + (w & 1), .out_h = h / factor + (h & 1),
        .factor = factor, .kw = kw, .kh = kw,
    };
    [enc setBytes:&p length:sizeof(p) atIndex:0];
    [enc setBuffer:B(src) offset:0 atIndex:1];
    [enc setBuffer:B(s->dec_kernel) offset:0 atIndex:2];
    [enc setBuffer:B(dst) offset:0 atIndex:3];
    vmaf_metal_dispatch_2d(enc, pl, p.out_w, 2 * p.out_h);
}

static int extract_fex(VmafFeatureExtractor *fex,
                       VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                       VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                       unsigned index, VmafFeatureCollector *feature_collector)
{
    SsimStateMetal *s = fex->priv;
    if (s->cpu) {
        return s->cpu->extract(s->cpu, ref_pic, ref_pic_90, dist_pic,
                               dist_pic_90, index, feature_collector);
    }

    id<MTLComputePipelineState> pl_load =
        vmaf_metal_pipeline(s->mtl, "ssim_metal_load", s->bpc);
    id<MTLComputePipelineState> pl_dec =
        vmaf_metal_pipeline(s->mtl, "ssim_metal_decimate", 0);
    id<MTLComputePipelineState> pl_h =
        vmaf_metal_pipeline(s->mtl, "ssim_metal_conv_h", 0);
    id<MTLComputePipelineState> pl_v =
        vmaf_metal_pipeline(s->mtl, "ssim_metal_conv_v", 0);
    if (!pl_load || !pl_dec || !pl_h || !pl_v) return -EINVAL;

    const unsigned slot = vmaf_metal_frames_acquire(&s->frames);
    const size_t row = (size_t)s->w * (s->bpc > 8 ? 2 : 1);
    uint8_t *in = B(s->in[slot]).contents;
    vmaf_metal_copy_plane(in, row, ref_pic, 0);
    vmaf_metal_copy_plane(in + row * s->h, row, dist_pic, 0);

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [vmaf_metal_queue(s->mtl) commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        [enc setBytes:&s->load length:sizeof(s->load) atIndex:0];
        [enc setBuffer:B(s->in[slot]) offset:0 atIndex:1];
        [enc setBuffer:B(s->full) offset:0 atIndex:2];
        vmaf_metal_dispatch_2d(enc, pl_load, s->w, 2 * s->h);

        // the input of each scale
        void *planes[SSIM_METAL_SCALES];
        planes[0] = s->factor > 1 ? s->planes[0] : s->full;
        if (s->factor > 1) {
            encode_decimate(s, enc, pl_dec, s->full, s->w, s->h, planes[0],
                            s->factor, s->factor);
        }
        for (int i = 1; i < s->n_scales; i++) {
            planes[i] = s->planes[i];
            encode_decimate(s, enc, pl_dec, planes[i - 1], s->sc[i - 1].w,
                            s->sc[i - 1].h, planes[i], 2, LPF_LEN);
        }

        for (int i = 0; i < s->n_scales; i++) {
            const SsimScale *k = &s->sc[i];
            const SsimMetalConvParams p = {
                .w = k->w, .h = k->h, .dst_w = k->dst_w, .dst_h = k->dst_h,
                .c1 = s->c1, .c2 = s->c2, .c3 = s->c3,
                .out_offset = k->out_offset, .part_offset = k->part_offset,
            };
            [enc setBytes:&p length:sizeof(p) atIndex:0];
            [enc setBuffer:B(planes[i]) offset:0 atIndex:1];
            [enc setBytes:g_gaussian_window_h
                   length:sizeof(g_gaussian_window_h) atIndex:2];
            [enc setBuffer:B(s->maps) offset:0 atIndex:3];
            vmaf_metal_dispatch_2d(enc, pl_h, k->dst_w, k->h);

            [enc setBuffer:B(s->maps) offset:0 atIndex:1];
            [enc setBuffer:B(s->out_l[slot]) offset:0 atIndex:3];
            [enc setBuffer:B(s->out_c[slot]) offset:0 atIndex:4];
            [enc setBuffer:B(s->out_s[slot]) offset:0 atIndex:5];
            [enc setBuffer:B(s->part[slot]) offset:0 atIndex:6];
            vmaf_metal_dispatch_2d(enc, pl_v, k->dst_w, k->dst_h);
        }
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
    SsimStateMetal *s = fex->priv;
    int err = 0;
    if (s->cpu) {
        err = s->cpu->close(s->cpu);
        free(s->cpu->priv);
        free(s->cpu);
        s->cpu = NULL;
    }
    buffers_free(s);
    return err;
}

static const char *provided_features_ssim[] = {
    "float_ssim",
    NULL
};

VmafFeatureExtractor vmaf_fex_float_ssim_metal = {
    .name = "float_ssim_metal",
    .init = init_fex,
    .extract = extract_fex,
    .close = close_fex,
    .options = options_ssim,
    .priv_size = sizeof(SsimStateMetal),
    .provided_features = provided_features_ssim,
    .flags = VMAF_FEATURE_EXTRACTOR_METAL,
};

static const char *provided_features_ms_ssim[] = {
    "float_ms_ssim",
    NULL
};

VmafFeatureExtractor vmaf_fex_float_ms_ssim_metal = {
    .name = "float_ms_ssim_metal",
    .init = init_fex,
    .extract = extract_fex,
    .close = close_fex,
    .options = options_ms_ssim,
    .priv_size = sizeof(SsimStateMetal),
    .provided_features = provided_features_ms_ssim,
    .flags = VMAF_FEATURE_EXTRACTOR_METAL,
};
