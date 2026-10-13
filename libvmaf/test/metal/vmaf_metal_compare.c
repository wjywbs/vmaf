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
 * Compare Metal feature extraction against the CPU, bit for bit.
 *
 * Runs a CPU VmafContext and a Metal VmafContext side by side on the same
 * pictures and compares every feature score (as doubles, not the 6 digit
 * output) at every frame.
 *
 * usage: vmaf_metal_compare -r ref.y4m -d dist.y4m
 *            [--feature name]... [--model version] [--frames n]
 *            [--threads n] [--fast] [--tol x]
 *
 *   --feature vif   compares CPU "vif" with "vif_metal", options are
 *                   passed as --feature vif:debug=true:vif_enhn_gain_limit=1.0
 *   --model v       compares the features (and score) of built-in model v,
 *                   the Metal context prefers Metal feature extractors
 *   --fast          Metal fast mode, report differences, fail above --tol
 *
 * Exit status is 0 when all scores match (exact mode) or are within --tol
 * (fast mode).
 */

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "libvmaf/libvmaf.h"
#include "libvmaf/libvmaf_metal.h"
#include "libvmaf/model.h"
#include "dict.h"
#include "feature/feature_extractor.h"
#include "feature/feature_name.h"

#define MAX_FEATURES 32
#define MAX_NAMES 1024

typedef struct {
    FILE *f;
    unsigned w, h, bpc;
    enum VmafPixelFormat pix_fmt;
} Y4m;

static int y4m_open(Y4m *y, const char *path)
{
    y->f = fopen(path, "rb");
    if (!y->f) return -errno;

    char line[1024];
    if (!fgets(line, sizeof(line), y->f)) return -EINVAL;
    if (strncmp(line, "YUV4MPEG2 ", 10)) return -EINVAL;

    y->bpc = 8;
    y->pix_fmt = VMAF_PIX_FMT_YUV420P;
    for (char *tok = strtok(line + 10, " \n"); tok; tok = strtok(NULL, " \n")) {
        switch (tok[0]) {
        case 'W': y->w = atoi(tok + 1); break;
        case 'H': y->h = atoi(tok + 1); break;
        case 'C':
            if (!strncmp(tok + 1, "422", 3)) y->pix_fmt = VMAF_PIX_FMT_YUV422P;
            if (!strncmp(tok + 1, "444", 3)) y->pix_fmt = VMAF_PIX_FMT_YUV444P;
            // e.g. C420p10, but not C420jpeg
            if (tok[4] == 'p' && tok[5] >= '0' && tok[5] <= '9')
                y->bpc = atoi(tok + 5);
            break;
        }
    }
    return y->w && y->h ? 0 : -EINVAL;
}

static int y4m_read(Y4m *y, VmafPicture *pic)
{
    char line[256];
    if (!fgets(line, sizeof(line), y->f)) return 1;
    if (strncmp(line, "FRAME", 5)) return -EINVAL;

    int err = vmaf_picture_alloc(pic, y->pix_fmt, y->bpc, y->w, y->h);
    if (err) return err;

    const size_t bytes = y->bpc > 8 ? 2 : 1;
    for (unsigned p = 0; p < 3; p++) {
        uint8_t *data = pic->data[p];
        for (unsigned i = 0; i < pic->h[p]; i++) {
            if (fread(data, bytes, pic->w[p], y->f) != pic->w[p]) {
                vmaf_picture_unref(pic);
                return -EIO;
            }
            data += pic->stride[p];
        }
    }
    return 0;
}

static int copy_picture(VmafPicture *dst, VmafPicture *src)
{
    int err = vmaf_picture_alloc(dst, src->pix_fmt, src->bpc, src->w[0],
                                 src->h[0]);
    if (err) return err;
    const size_t bytes = src->bpc > 8 ? 2 : 1;
    for (unsigned p = 0; p < 3; p++) {
        for (unsigned i = 0; i < src->h[p]; i++) {
            memcpy((uint8_t *)dst->data[p] + i * dst->stride[p],
                   (uint8_t *)src->data[p] + i * src->stride[p],
                   src->w[p] * bytes);
        }
    }
    return 0;
}

static void usage(const char *app)
{
    fprintf(stderr,
            "usage: %s -r ref.y4m -d dist.y4m [--feature name]... "
            "[--model version] [--frames n] [--threads n] [--fast] "
            "[--tol x]\n", app);
    exit(1);
}

int main(int argc, char **argv)
{
    const char *ref_path = NULL, *dist_path = NULL, *model_version = NULL;
    const char *features[MAX_FEATURES];
    unsigned n_features = 0, max_frames = UINT32_MAX, n_threads = 0;
    bool fast = false;
    double tol = 1e-4;

    for (int i = 1; i < argc; i++) {
        const bool has_val = i + 1 < argc;
        if (!strcmp(argv[i], "-r") && has_val) ref_path = argv[++i];
        else if (!strcmp(argv[i], "-d") && has_val) dist_path = argv[++i];
        else if (!strcmp(argv[i], "--feature") && has_val &&
                 n_features < MAX_FEATURES)
            features[n_features++] = argv[++i];
        else if (!strcmp(argv[i], "--model") && has_val)
            model_version = argv[++i];
        else if (!strcmp(argv[i], "--frames") && has_val)
            max_frames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && has_val)
            n_threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tol") && has_val)
            tol = atof(argv[++i]);
        else if (!strcmp(argv[i], "--fast")) fast = true;
        else usage(argv[0]);
    }
    if (!ref_path || !dist_path || (!n_features && !model_version))
        usage(argv[0]);

    Y4m ref_in, dist_in;
    if (y4m_open(&ref_in, ref_path) || y4m_open(&dist_in, dist_path)) {
        fprintf(stderr, "problem opening inputs\n");
        return 1;
    }

    VmafConfiguration cfg = {
        .log_level = VMAF_LOG_LEVEL_WARNING,
        .n_threads = n_threads,
    };
    VmafContext *cpu, *gpu;
    int err = vmaf_init(&cpu, cfg) | vmaf_init(&gpu, cfg);
    if (err) return 1;

    VmafMetalState *metal_state;
    VmafMetalConfiguration metal_cfg = { .fast_math = fast };
    err = vmaf_metal_state_init(&metal_state, metal_cfg);
    err |= vmaf_metal_import_state(gpu, metal_state);
    if (err) {
        fprintf(stderr, "problem initializing Metal\n");
        return 1;
    }

    // Candidate feature names: everything the CPU extractors can provide.
    const char *names[MAX_NAMES];
    unsigned n_names = 0;
    static const char *fex_names[] = {
        "vif", "adm", "motion", "cambi", "float_ssim", "float_ms_ssim",
        "psnr", "ciede", NULL,
    };
    for (unsigned i = 0; fex_names[i]; i++) {
        VmafFeatureExtractor *fex =
            vmaf_get_feature_extractor_by_name(fex_names[i]);
        if (!fex || !fex->provided_features) continue;
        for (unsigned j = 0; fex->provided_features[j]; j++) {
            if (n_names < MAX_NAMES)
                names[n_names++] = fex->provided_features[j];
        }
    }

    for (unsigned i = 0; i < n_features; i++) {
        // name[:key=value]...
        char spec[256], metal_name[256];
        snprintf(spec, sizeof(spec), "%s", features[i]);
        char *opts = strchr(spec, ':');
        if (opts) *opts++ = '\0';
        snprintf(metal_name, sizeof(metal_name), "%s_metal", spec);

        VmafFeatureDictionary *d_cpu = NULL, *d_gpu = NULL, *d_name = NULL;
        for (char *kv = opts ? strtok(opts, ":") : NULL; kv;
             kv = strtok(NULL, ":")) {
            char *val = strchr(kv, '=');
            if (!val) usage(argv[0]);
            *val++ = '\0';
            err |= vmaf_feature_dictionary_set(&d_cpu, kv, val);
            err |= vmaf_feature_dictionary_set(&d_gpu, kv, val);
            err |= vmaf_feature_dictionary_set(&d_name, kv, val);
        }
        // Options can add a suffix to the output names (e.g. "_egl_1"),
        // derive them as the extractor does.
        VmafFeatureExtractor *fex = vmaf_get_feature_extractor_by_name(spec);
        if (fex && fex->provided_features) {
            VmafFeatureExtractorContext *fex_ctx = NULL;
            err |= vmaf_feature_extractor_context_create(&fex_ctx, fex,
                                                    (VmafDictionary *)d_name);
            if (!err && fex_ctx->fex->priv) {
                for (unsigned j = 0; fex->provided_features[j]; j++) {
                    char *fn = vmaf_feature_name_from_options(
                        fex->provided_features[j], fex->options,
                        fex_ctx->fex->priv);
                    if (fn && n_names < MAX_NAMES) names[n_names++] = fn;
                }
            }
            if (fex_ctx) vmaf_feature_extractor_context_destroy(fex_ctx);
        } else if (d_name) {
            vmaf_feature_dictionary_free(&d_name);
        }
        err |= vmaf_use_feature(cpu, spec, d_cpu);
        err |= vmaf_use_feature(gpu, metal_name, d_gpu);
        if (err) {
            fprintf(stderr, "problem using feature %s / %s\n", features[i],
                    metal_name);
            return 1;
        }
    }

    VmafModel *model_cpu = NULL, *model_gpu = NULL;
    if (model_version) {
        VmafModelConfig model_cfg = { .name = "vmaf" };
        err = vmaf_model_load(&model_cpu, &model_cfg, model_version);
        err |= vmaf_model_load(&model_gpu, &model_cfg, model_version);
        err |= vmaf_use_features_from_model(cpu, model_cpu);
        err |= vmaf_use_features_from_model(gpu, model_gpu);
        if (err) {
            fprintf(stderr, "problem loading model %s\n", model_version);
            return 1;
        }
    }

    unsigned n_frames = 0;
    while (n_frames < max_frames) {
        VmafPicture ref, dist, ref2, dist2;
        int r0 = y4m_read(&ref_in, &ref);
        int r1 = r0 ? r0 : y4m_read(&dist_in, &dist);
        if (r0 || r1) {
            if (!r0) vmaf_picture_unref(&ref);
            if (r0 < 0 || r1 < 0) {
                fprintf(stderr, "problem reading frame %u\n", n_frames);
                return 1;
            }
            break;
        }
        err = copy_picture(&ref2, &ref) | copy_picture(&dist2, &dist);
        err |= vmaf_read_pictures(cpu, &ref, &dist, n_frames);
        err |= vmaf_read_pictures(gpu, &ref2, &dist2, n_frames);
        if (err) {
            fprintf(stderr, "problem extracting frame %u\n", n_frames);
            return 1;
        }
        n_frames++;
    }
    err = vmaf_read_pictures(cpu, NULL, NULL, 0);
    err |= vmaf_read_pictures(gpu, NULL, NULL, 0);
    if (err) return 1;

    unsigned n_compared = 0, n_failed = 0;
    for (unsigned k = 0; k <= n_names; k++) {
        const bool is_model = k == n_names;
        if (is_model && !model_cpu) break;
        const char *name = is_model ? "vmaf" : names[k];

        unsigned present = 0, mismatches = 0, first_mismatch = 0;
        double max_diff = 0.0;
        for (unsigned i = 0; i < n_frames; i++) {
            double a, b;
            int ea, eb;
            if (is_model) {
                ea = vmaf_score_at_index(cpu, model_cpu, &a, i);
                eb = vmaf_score_at_index(gpu, model_gpu, &b, i);
            } else {
                ea = vmaf_feature_score_at_index(cpu, name, &a, i);
                eb = vmaf_feature_score_at_index(gpu, name, &b, i);
            }
            if (ea && eb) continue;
            if (ea || eb) {
                fprintf(stderr, "%s: frame %u only present on %s\n", name, i,
                        ea ? "metal" : "cpu");
                n_failed++;
                break;
            }
            present++;
            if (memcmp(&a, &b, sizeof(a))) {
                if (!mismatches) first_mismatch = i;
                mismatches++;
                max_diff = fmax(max_diff, fabs(a - b));
            }
        }
        if (!present) continue;

        n_compared++;
        const bool fail = fast ? max_diff > tol : mismatches > 0;
        if (fail) n_failed++;
        printf("%-45s frames %4u  mismatches %4u  max_abs_diff %.3e%s",
               name, present, mismatches, max_diff, fail ? "  FAIL" : "");
        if (mismatches) printf("  (first at frame %u)", first_mismatch);
        printf("\n");
    }

    printf("%u frames, %u features compared, %s mode: %s\n", n_frames,
           n_compared, fast ? "fast" : "exact",
           n_failed || !n_compared ? "FAIL" : "PASS");

    if (model_cpu) vmaf_model_destroy(model_cpu);
    if (model_gpu) vmaf_model_destroy(model_gpu);
    vmaf_close(cpu);
    vmaf_close(gpu);
    fclose(ref_in.f);
    fclose(dist_in.f);
    return n_failed || !n_compared;
}
