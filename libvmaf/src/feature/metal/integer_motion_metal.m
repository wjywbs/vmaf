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

// first: defines MIN, which Foundation then leaves alone
#include "motion_blend_tools.h"

#import <Metal/Metal.h>

#include <errno.h>
#include <string.h>

#include "dict.h"
#include "feature_collector.h"
#include "feature_extractor.h"
#include "feature_name.h"
#include "log.h"
#include "metal/common.h"
#include "opt.h"

#include "integer_motion_metal.h"

/* Default maximum value allowed for motion */
#define DEFAULT_MOTION_MAX_VAL (10000.0)

typedef struct MotionStateMetal {
    // options, see integer_motion.c
    double motion_max_val;
    double motion_blend_factor;
    double motion_blend_offset;
    double motion_fps_weight;
    bool motion_five_frame_window;
    bool motion_moving_average;
    bool motion_force_zero;
    bool debug;

    VmafDictionary *feature_name_dict;
    VmafMetalState *ms;
    VmafMetalFrames frames;
    unsigned w, h, bpc;

    // id<MTLBuffer>, retained, per frame slot
    void *prev[VMAF_METAL_FRAMES], *cur[VMAF_METAL_FRAMES];
    void *row_sad[VMAF_METAL_FRAMES];
} MotionStateMetal;

static const VmafOption options[] = {
    {
        .name = "motion_force_zero",
        .alias = "force_0",
        .help = "forcing motion score to zero",
        .offset = offsetof(MotionStateMetal, motion_force_zero),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_blend_factor",
        .alias = "mbf",
        .help = "blend motion score given an offset",
        .offset = offsetof(MotionStateMetal, motion_blend_factor),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 1.0,
        .min = 0.0,
        .max = 1.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_blend_offset",
        .alias = "mbo",
        .help = "blend motion score starting from this offset",
        .offset = offsetof(MotionStateMetal, motion_blend_offset),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 40.0,
        .min = 0.0,
        .max = 1000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_fps_weight",
        .alias = "mfw",
        .help = "fps-aware multiplicative weight/correction",
        .offset = offsetof(MotionStateMetal, motion_fps_weight),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = 1.0,
        .min = 0.0,
        .max = 5.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_max_val",
        .alias = "mmxv",
        .help = "maximum value allowed; larger values will be clipped to this value",
        .offset = offsetof(MotionStateMetal, motion_max_val),
        .type = VMAF_OPT_TYPE_DOUBLE,
        .default_val.d = DEFAULT_MOTION_MAX_VAL,
        .min = 0.0,
        .max = 10000.0,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_five_frame_window",
        .alias = "mffw",
        .help = "use five-frame temporal window",
        .offset = offsetof(MotionStateMetal, motion_five_frame_window),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "motion_moving_average",
        .alias = "mma",
        .help = "use moving average for motion scores after first frame",
        .offset = offsetof(MotionStateMetal, motion_moving_average),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
        .flags = VMAF_OPT_FLAG_FEATURE_PARAM,
    },
    {
        .name = "debug",
        .help = "debug mode: enable additional output",
        .offset = offsetof(MotionStateMetal, debug),
        .type = VMAF_OPT_TYPE_BOOL,
        .default_val.b = false,
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

static void buffers_free(MotionStateMetal *s)
{
    vmaf_metal_frames_close(&s->frames);
    for (unsigned k = 0; k < VMAF_METAL_FRAMES; k++) {
        void **bufs[] = { &s->prev[k], &s->cur[k], &s->row_sad[k] };
        for (unsigned i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) {
            if (*bufs[i]) CFRelease(*bufs[i]);
            *bufs[i] = NULL;
        }
    }
}

static int init_fex(VmafFeatureExtractor *fex, enum VmafPixelFormat pix_fmt,
                    unsigned bpc, unsigned w, unsigned h)
{
    MotionStateMetal *s = fex->priv;
    (void)pix_fmt;

    s->ms = fex->metal_state ? fex->metal_state : vmaf_metal_state_default();
    if (!s->ms) return -ENODEV;
    s->w = w;
    s->h = h;
    s->bpc = bpc;

    if (vmaf_metal_frames_init(&s->frames, s->ms)) goto fail;
    const size_t bpp = bpc > 8 ? 2 : 1;
    for (unsigned k = 0; k < VMAF_METAL_FRAMES; k++) {
        s->prev[k] = buffer_new(s->ms, (size_t)w * h * bpp);
        s->cur[k] = buffer_new(s->ms, (size_t)w * h * bpp);
        s->row_sad[k] = buffer_new(s->ms, (size_t)h * sizeof(uint32_t));
        if (!s->prev[k] || !s->cur[k] || !s->row_sad[k]) goto fail;
    }

    s->feature_name_dict =
        vmaf_feature_name_dict_from_provided_features(fex->provided_features,
                fex->options, s);
    if (!s->feature_name_dict) goto fail;

    return 0;

fail:
    buffers_free(s);
    return -ENOMEM;
}

static int write_scores(MotionStateMetal *s, double score, unsigned index,
                        VmafFeatureCollector *feature_collector)
{
    int err = vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict,
            "VMAF_integer_feature_motion_sad_score", score, index);
    if (err) return err;

    if (s->debug) {
        return vmaf_feature_collector_append_with_dict(feature_collector,
                s->feature_name_dict,
                "VMAF_integer_feature_motion_score", score, index);
    }

    return 0;
}

/**
 * motion_score_pipeline_8/16() of integer_motion.c on the GPU, then the
 * score of extract(), written when the frame completes.
 */
static int motion_sad(MotionStateMetal *s, const VmafPicture *prev,
                      const VmafPicture *cur, unsigned index,
                      VmafFeatureCollector *feature_collector)
{
    id<MTLComputePipelineState> pipeline =
        vmaf_metal_pipeline(s->ms, "motion_metal_sad", s->bpc);
    if (!pipeline) return -EINVAL;

    const unsigned slot = vmaf_metal_frames_acquire(&s->frames);
    const size_t bpp = s->bpc > 8 ? 2 : 1;
    vmaf_metal_copy_plane(B(s->prev[slot]).contents, s->w * bpp, prev, 0);
    vmaf_metal_copy_plane(B(s->cur[slot]).contents, s->w * bpp, cur, 0);

    const MotionMetalParams p = {
        .w = s->w,
        .h = s->h,
        .prev_stride = s->w,
        .cur_stride = s->w,
    };

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [vmaf_metal_queue(s->ms) commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        NSUInteger tw = MIN((s->w + 31) / 32 * 32, MOTION_METAL_MAX_TG_W);
        tw = MIN(tw, pipeline.maxTotalThreadsPerThreadgroup / 32 * 32);
        [enc setComputePipelineState:pipeline];
        [enc setBytes:&p length:sizeof(p) atIndex:0];
        [enc setBuffer:B(s->prev[slot]) offset:0 atIndex:1];
        [enc setBuffer:B(s->cur[slot]) offset:0 atIndex:2];
        [enc setBuffer:B(s->row_sad[slot]) offset:0 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(s->h, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        [enc endEncoding];

        vmaf_metal_frames_commit(&s->frames, slot, cb, ^int(int err) {
            if (err) return err;
            const uint32_t *row_sad = B(s->row_sad[slot]).contents;
            uint64_t sad = 0;
            for (unsigned i = 0; i < s->h; i++) sad += row_sad[i];

            const unsigned w = s->w;
            const unsigned h = s->h;
            const double score =
                MIN((double)sad / 256. / (w * h) * s->motion_fps_weight,
                    s->motion_max_val);
            return write_scores(s, score, index, feature_collector);
        });
    }
    return 0;
}

/** extract() of integer_motion.c */
static int extract_fex(VmafFeatureExtractor *fex,
                       VmafPicture *ref_pic, VmafPicture *ref_pic_90,
                       VmafPicture *dist_pic, VmafPicture *dist_pic_90,
                       unsigned index, VmafFeatureCollector *feature_collector)
{
    MotionStateMetal *s = fex->priv;

    (void) dist_pic;
    (void) ref_pic_90;
    (void) dist_pic_90;

    if (s->motion_force_zero)
        return write_scores(s, 0., index, feature_collector);

    const unsigned min_idx = s->motion_five_frame_window ? 2 : 1;
    if (index < min_idx)
        return write_scores(s, 0., index, feature_collector);

    const VmafPicture *prev = s->motion_five_frame_window
        ? &fex->prev_prev_ref
        : &fex->prev_ref;
    if (!prev->ref)
        return -EINVAL;

    return motion_sad(s, prev, ref_pic, index, feature_collector);
}

static int close_fex(VmafFeatureExtractor *fex)
{
    MotionStateMetal *s = fex->priv;
    buffers_free(s);
    return vmaf_dictionary_free(&s->feature_name_dict);
}

/** flush() of integer_motion.c */
static int flush_fex(VmafFeatureExtractor *fex,
                     VmafFeatureCollector *feature_collector)
{
    MotionStateMetal *s = fex->priv;

    if (!s->feature_name_dict) {
        s->feature_name_dict = vmaf_feature_name_dict_from_provided_features(
                fex->provided_features, fex->options, s);
        if (!s->feature_name_dict) return -ENOMEM;
    }

    const VmafDictionaryEntry *sad_entry = vmaf_dictionary_get(
            &s->feature_name_dict, "VMAF_integer_feature_motion_sad_score", 0);
    if (!sad_entry) return -EINVAL;
    const char *sad_name = sad_entry->val;

    unsigned n = 0;
    double score;
    while (!vmaf_feature_collector_get_score(feature_collector, sad_name, &score, n))
        n++;
    const unsigned stride = s->motion_five_frame_window ? 2 : 1;
    const unsigned min_idx = s->motion_five_frame_window ? 2 : 1;
    if (!n) {
        vmaf_dictionary_free(&s->feature_name_dict);
        return 1;
    }

    double stamp_value = 0.;
    if (n > min_idx) {
        double sad_at_min_idx;
        if (!vmaf_feature_collector_get_score(feature_collector, sad_name,
                                              &sad_at_min_idx, min_idx)) {
            stamp_value = MIN(motion_blend(sad_at_min_idx,
                                           s->motion_blend_factor,
                                           s->motion_blend_offset),
                              s->motion_max_val);
        }
    }

    double prev_processed = 0.;
    for (unsigned i = 0; i < n; i++) {
        double sad_i;
        vmaf_feature_collector_get_score(feature_collector, sad_name, &sad_i, i);

        double motion2;

        if (i < min_idx) {
            motion2 = 0.;
        } else {
            const int lo_idx = (int)i - (int)(stride - 1);
            const int hi_idx = (int)i + 1;
            double hi;
            const bool has_hi = !vmaf_feature_collector_get_score(
                    feature_collector, sad_name, &hi, hi_idx);
            if (!has_hi) {
                motion2 = sad_i;
            } else if (lo_idx >= (int)min_idx) {
                double lo;
                vmaf_feature_collector_get_score(
                        feature_collector, sad_name, &lo, lo_idx);
                motion2 = lo < hi ? lo : hi;
            } else {
                motion2 = hi;
            }
        }

        vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict,
            "VMAF_integer_feature_motion2_score", motion2, i);

        double motion3;
        if (i < min_idx) {
            motion3 = stamp_value;
            prev_processed = stamp_value;
        } else {
            double processed = MIN(motion_blend(motion2,
                                                s->motion_blend_factor,
                                                s->motion_blend_offset),
                                   s->motion_max_val);
            motion3 = s->motion_moving_average
                    ? (processed + prev_processed) / 2.0
                    : processed;
            prev_processed = processed;
        }

        vmaf_feature_collector_append_with_dict(feature_collector,
            s->feature_name_dict,
            "VMAF_integer_feature_motion3_score", motion3, i);
    }

    vmaf_dictionary_free(&s->feature_name_dict);
    return 1;
}

static const char *provided_features[] = {
    "VMAF_integer_feature_motion_sad_score",
    "VMAF_integer_feature_motion_score",
    "VMAF_integer_feature_motion2_score",
    "VMAF_integer_feature_motion3_score",
    NULL
};

VmafFeatureExtractor vmaf_fex_integer_motion_metal = {
    .name = "motion_metal",
    .options = options,
    .init = init_fex,
    .extract = extract_fex,
    .flush = flush_fex,
    .close = close_fex,
    .priv_size = sizeof(MotionStateMetal),
    .provided_features = provided_features,
    .flags = VMAF_FEATURE_EXTRACTOR_PREV_REF | VMAF_FEATURE_EXTRACTOR_METAL,
};
