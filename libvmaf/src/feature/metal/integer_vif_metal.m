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
#include <string.h>

#include "config.h"
#include "cpu.h"
#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "integer_vif.h"
#include "log.h"
#include "metal/common.h"
#include "opt.h"

#include "integer_vif_metal.h"

typedef struct VifStateMetal {
    // options, see integer_vif.c
    bool debug;
    double vif_enhn_gain_limit;
    bool vif_skip_scale0;

    VmafDictionary *feature_name_dict;
    VmafMetalState *ms;
    unsigned w, h, bpc;
    unsigned tg_cnt[4]; // stat threadgroups per scale
    unsigned partial_offset[4];
    unsigned partial_cnt;
    unsigned noclamp_end;
    uint16_t log2_table[65537];

    // id<MTLBuffer>, retained
    void *src_ref[4], *src_dis[4]; // scale 0: input bpc, scales > 0: uint16
    void *mu1, *mu2, *xx, *yy, *xy;
    void *log2;
    void *partial;
} VifStateMetal;

static const VmafOption options[] = {
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(VifStateMetal, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
    },
    {
        .name = "vif_enhn_gain_limit",
        .alias = "egl",
        .help = "enhancement gain imposed on vif, must be >= 1.0, "
                "where 1.0 means the gain is completely disabled",
        .offset = offsetof(VifStateMetal, vif_enhn_gain_limit),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_VIF_ENHN_GAIN_LIMIT,
        .min = 1.0,
        .max = DEFAULT_VIF_ENHN_GAIN_LIMIT,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "vif_skip_scale0",
        .alias = "ssclz",
        .help = "when set, skip scale 0 calculations",
        .offset = offsetof(VifStateMetal, vif_skip_scale0),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    { 0 }
};

static void *buffer_new(VmafMetalState *ms, size_t size)
{
    id<MTLBuffer> buf = vmaf_metal_buffer(ms, size);
    return buf ? (void *)CFBridgingRetain(buf) : NULL;
}

static inline id<MTLBuffer> B(void *buf)
{
    return (__bridge id<MTLBuffer>)buf;
}

static void buffers_free(VifStateMetal *s)
{
    void **bufs[] = {
        &s->src_ref[0], &s->src_ref[1], &s->src_ref[2], &s->src_ref[3],
        &s->src_dis[0], &s->src_dis[1], &s->src_dis[2], &s->src_dis[3],
        &s->mu1, &s->mu2, &s->xx, &s->yy, &s->xy, &s->log2, &s->partial,
    };
    for (unsigned i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) {
        if (*bufs[i]) CFRelease(*bufs[i]);
        *bufs[i] = NULL;
    }
}

static int init_fex(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                unsigned bpc, unsigned w, unsigned h)
{
    VifStateMetal *s = fex->priv;
    (void)pix_fmt;

    s->ms = fex->metal_state ? fex->metal_state : vmaf_metal_state_default();
    if (!s->ms) return -ENODEV;
    s->w = w;
    s->h = h;
    s->bpc = bpc;

    log_generate(s->log2_table);

    /**
     * vif_statistic_8_neon() does not clamp sigma2_sq to >= 0 in its vector
     * columns, unlike every other implementation. Match the CPU extractor of
     * this machine.
     */
    s->noclamp_end = 0;
#if ARCH_AARCH64
    if (bpc == 8 && (vmaf_get_cpu_flags() & VMAF_ARM_CPU_FLAG_NEON))
        s->noclamp_end = w & ~7u;
#endif

    const size_t px = (size_t)w * h;
    const size_t bpp = bpc > 8 ? 2 : 1;
    s->src_ref[0] = buffer_new(s->ms, px * bpp);
    s->src_dis[0] = buffer_new(s->ms, px * bpp);
    for (unsigned scale = 1; scale < 4; scale++) {
        const size_t spx = (size_t)(w >> scale) * (h >> scale);
        s->src_ref[scale] = buffer_new(s->ms, spx * sizeof(uint16_t));
        s->src_dis[scale] = buffer_new(s->ms, spx * sizeof(uint16_t));
    }
    s->mu1 = buffer_new(s->ms, px * sizeof(uint16_t));
    s->mu2 = buffer_new(s->ms, px * sizeof(uint16_t));
    s->xx = buffer_new(s->ms, px * sizeof(uint32_t));
    s->yy = buffer_new(s->ms, px * sizeof(uint32_t));
    s->xy = buffer_new(s->ms, px * sizeof(uint32_t));
    s->log2 = buffer_new(s->ms, sizeof(s->log2_table));

    s->partial_cnt = 0;
    for (unsigned scale = 0; scale < 4; scale++) {
        const unsigned sw = w >> scale, sh = h >> scale;
        s->tg_cnt[scale] = ((sw + VIF_METAL_TG_W - 1) / VIF_METAL_TG_W) *
                           ((sh + VIF_METAL_TG_H - 1) / VIF_METAL_TG_H);
        s->partial_offset[scale] = s->partial_cnt;
        s->partial_cnt += s->tg_cnt[scale];
    }
    s->partial = buffer_new(s->ms, s->partial_cnt * VIF_METAL_N_ACCUM *
                                   sizeof(int64_t));

    bool ok = s->src_ref[0] && s->src_dis[0] && s->mu1 && s->mu2 && s->xx &&
              s->yy && s->xy && s->log2 && s->partial;
    for (unsigned scale = 1; scale < 4; scale++)
        ok = ok && s->src_ref[scale] && s->src_dis[scale];
    if (!ok) goto fail;

    memcpy(B(s->log2).contents, s->log2_table, sizeof(s->log2_table));

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) goto fail;

    return 0;

fail:
    buffers_free(s);
    return -ENOMEM;
}

static uint64_t double_bits(double d)
{
    uint64_t u;
    memcpy(&u, &d, sizeof(u));
    return u;
}

static int extract_fex(VmafFeatureExtractor *fex,
                   VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                   VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                   unsigned index, VmafFeatureCollector *feature_collector)
{
    VifStateMetal *s = fex->priv;
    (void)ref_pic_90;
    (void)dist_pic_90;

    const size_t bpp = s->bpc > 8 ? 2 : 1;
    vmaf_metal_copy_plane(B(s->src_ref[0]).contents, s->w * bpp, ref_pic, 0);
    vmaf_metal_copy_plane(B(s->src_dis[0]).contents, s->w * bpp, dist_pic, 0);

    const unsigned scale_start = s->vif_skip_scale0 ? 1 : 0;

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [vmaf_metal_queue(s->ms) commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        for (unsigned scale = 0; scale < 4; scale++) {
            const unsigned variant = scale | (s->bpc << 8);
            VifMetalParams p = {
                .w = s->w >> scale,
                .h = s->h >> scale,
                .src_stride = s->w >> scale,
                .bpc = s->bpc,
                .scale = scale,
                .partial_offset = s->partial_offset[scale],
                .tg_cols = ((s->w >> scale) + VIF_METAL_TG_W - 1) / VIF_METAL_TG_W,
                .noclamp_end = scale == 0 ? s->noclamp_end : 0,
                .gain_limit_f = s->vif_enhn_gain_limit,
                .gain_limit = double_bits(s->vif_enhn_gain_limit),
            };

            if (scale >= scale_start) {
                id<MTLComputePipelineState> vertical =
                    vmaf_metal_pipeline(s->ms, "vif_metal_vertical", variant);
                id<MTLComputePipelineState> stat =
                    vmaf_metal_pipeline(s->ms, "vif_metal_stat", variant);
                if (!vertical || !stat) {
                    [enc endEncoding];
                    return -EINVAL;
                }

                [enc setBytes:&p length:sizeof(p) atIndex:0];
                [enc setBuffer:B(s->src_ref[scale]) offset:0 atIndex:1];
                [enc setBuffer:B(s->src_dis[scale]) offset:0 atIndex:2];
                [enc setBuffer:B(s->mu1) offset:0 atIndex:3];
                [enc setBuffer:B(s->mu2) offset:0 atIndex:4];
                [enc setBuffer:B(s->xx) offset:0 atIndex:5];
                [enc setBuffer:B(s->yy) offset:0 atIndex:6];
                [enc setBuffer:B(s->xy) offset:0 atIndex:7];
                vmaf_metal_dispatch_2d(enc, vertical, p.w, p.h);

                [enc setComputePipelineState:stat];
                [enc setBuffer:B(s->mu1) offset:0 atIndex:1];
                [enc setBuffer:B(s->mu2) offset:0 atIndex:2];
                [enc setBuffer:B(s->xx) offset:0 atIndex:3];
                [enc setBuffer:B(s->yy) offset:0 atIndex:4];
                [enc setBuffer:B(s->xy) offset:0 atIndex:5];
                [enc setBuffer:B(s->log2) offset:0 atIndex:6];
                [enc setBuffer:B(s->partial) offset:0 atIndex:7];
                [enc dispatchThreadgroups:MTLSizeMake(p.tg_cols,
                        s->tg_cnt[scale] / p.tg_cols, 1)
                    threadsPerThreadgroup:MTLSizeMake(VIF_METAL_TG_W,
                        VIF_METAL_TG_H, 1)];
            }

            if (scale < 3) {
                id<MTLComputePipelineState> subsample =
                    vmaf_metal_pipeline(s->ms, "vif_metal_subsample", variant);
                if (!subsample) {
                    [enc endEncoding];
                    return -EINVAL;
                }
                p.dst_stride = s->w >> (scale + 1);
                [enc setBytes:&p length:sizeof(p) atIndex:0];
                [enc setBuffer:B(s->src_ref[scale]) offset:0 atIndex:1];
                [enc setBuffer:B(s->src_dis[scale]) offset:0 atIndex:2];
                [enc setBuffer:B(s->src_ref[scale + 1]) offset:0 atIndex:3];
                [enc setBuffer:B(s->src_dis[scale + 1]) offset:0 atIndex:4];
                vmaf_metal_dispatch_2d(enc, subsample, p.w / 2, p.h / 2);
            }
        }

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) {
            vmaf_log(VMAF_LOG_LEVEL_ERROR, "vif_metal: command buffer failed\n");
            return -EIO;
        }
    }

    VifScore vif_score = { 0 };
    const int64_t *partial = B(s->partial).contents;
    for (unsigned scale = scale_start; scale < 4; scale++) {
        VifResiduals r = { 0 };
        const int64_t *p = partial + s->partial_offset[scale] * VIF_METAL_N_ACCUM;
        for (unsigned i = 0; i < s->tg_cnt[scale]; i++) {
            r.accum_num_log += p[i * VIF_METAL_N_ACCUM + 0];
            r.accum_den_log += p[i * VIF_METAL_N_ACCUM + 1];
            r.accum_num_non_log += p[i * VIF_METAL_N_ACCUM + 2];
            r.accum_den_non_log += p[i * VIF_METAL_N_ACCUM + 3];
        }
        vif_residuals_to_score(r, &vif_score.scale[scale].num,
                               &vif_score.scale[scale].den);
    }

    return vif_write_scores(feature_collector, s->feature_name_dict, index,
                            vif_score, s->vif_skip_scale0, s->debug);
}

static int close_fex(VmafFeatureExtractor *fex)
{
    VifStateMetal *s = fex->priv;
    buffers_free(s);
    vmaf_dictionary_free(&s->feature_name_dict);
    return 0;
}

static const char *provided_features[] = {
    "VMAF_integer_feature_vif_scale0_score", "VMAF_integer_feature_vif_scale1_score",
    "VMAF_integer_feature_vif_scale2_score", "VMAF_integer_feature_vif_scale3_score",
    "integer_vif", "integer_vif_num", "integer_vif_den", "integer_vif_num_scale0",
    "integer_vif_den_scale0", "integer_vif_num_scale1", "integer_vif_den_scale1",
    "integer_vif_num_scale2", "integer_vif_den_scale2", "integer_vif_num_scale3",
    "integer_vif_den_scale3",
    NULL
};

VmafFeatureExtractor vmaf_fex_integer_vif_metal = {
    .name = "vif_metal",
    .init = init_fex,
    .extract = extract_fex,
    .options = options,
    .close = close_fex,
    .priv_size = sizeof(VifStateMetal),
    .flags = VMAF_FEATURE_EXTRACTOR_METAL,
    .provided_features = provided_features,
};
