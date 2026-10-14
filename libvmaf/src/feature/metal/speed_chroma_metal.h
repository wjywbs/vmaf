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
 * SpEED chroma definitions shared between the Metal kernel
 * (speed_chroma.metal) and the host (speed_chroma_metal.m).
 */

#ifndef __VMAF_FEATURE_METAL_SPEED_CHROMA_METAL_H__
#define __VMAF_FEATURE_METAL_SPEED_CHROMA_METAL_H__

#define SPEED_METAL_PLANES 4 // ref U, ref V, dis U, dis V
#define SPEED_METAL_MAX_FWIDTH 128

// speed_prescale_method, on the GPU
#define SPEED_METAL_SCALE_NEAREST 0
#define SPEED_METAL_SCALE_BILINEAR 1

typedef struct SpeedMetalParams {
    int w, h;          // output of the scaling, input of the filter
    int in_stride;     // pixels
    unsigned in_plane; // pixels between the input planes
    int out_w, out_h;  // w / 16, h / 16
    int fwidth;
    int method;        // SPEED_METAL_SCALE_*
    float scaler;      // picture_copy(): divisor for bpc > 8
} SpeedMetalParams;

#endif /* __VMAF_FEATURE_METAL_SPEED_CHROMA_METAL_H__ */
