# Metal backend, step 2: integer ADM

This step ports the integer ADM feature extractor (`adm`, `integer_adm.c`) to
Metal as `adm_metal`. With VIF on Metal since step 1, ADM was the largest
CPU cost left in the default model. The CPU code is not changed in this step.

## Result

* Exact mode (the default) is bit-identical to the CPU extractor for every
  feature, frame and option tested, on 8-bit and 10-bit input. This includes
  all four CSF modes.
* On the full 375-frame 1080p clip:
  * adm alone runs 2.6x faster at 1 thread and 1.4x faster at 10 threads.
  * The default VMAF model, now with vif and adm on Metal, runs 9.1x faster
    at 1 thread and 2.3x faster at 10 threads. After step 1 it ran 3.2x and
    2.4x faster.

## What was ported

`adm_metal` provides the same features as `adm`:

* `VMAF_integer_feature_adm2_score`, `_aim_score`, `_adm3_score`
* `integer_adm_scale0` to `integer_adm_scale3`
* the debug features `integer_adm`, `integer_adm_num`, `integer_adm_den`,
  and `integer_adm_num_scaleN`, `integer_adm_den_scaleN`

It accepts the same options, with the same defaults, ranges and output name
suffixes:

* `debug`
* `adm_csf_scale`, `adm_csf_diag_scale`
* `adm_dlm_weight`
* `adm_enhn_gain_limit`
* `adm_norm_view_dist`, `adm_ref_display_height`
* `adm_csf_mode` (0 to 3)
* `adm_noise_weight`
* `adm_skip_aim`, `adm_skip_scale0`
* `adm_min_val`

**Not supported:** `adm_norm_view_dist_extra`, the second viewing distance,
and the merge of its scores. No built-in model uses it.

The model path selects `adm_metal` automatically when Metal is enabled. Motion
still runs on the CPU in this step.

### Kernels (`src/feature/metal/integer_adm.metal`)

Each scale runs two kernels. All four scales of a frame go into one command
buffer.

| kernel | work |
|---|---|
| `adm_metal_dwt` | DWT of ref and dis, then per band coefficient: the CSF denominator of ref, decouple into restored (`r`) and additive impairment (`a`) bands, and the low-pass band that is the next scale's input |
| `adm_metal_cm` | CSF and contrast masking for DLM (`r` masked by `a`) and AIM (`a` masked by `r`) |

* **Fused DWT, CSF denominator and decouple.** The v, h and d bands of ref
  and dis are never stored. Only the six decoupled planes and the two
  low-pass bands go to memory.
* **CSF computed inside the CM kernel.** The CPU writes the CSF output and
  its 1/15-filtered absolute value to memory, then reads back the 3x3
  neighbourhood in `adm_cm`. `adm_metal_cm` instead computes the CSF of the
  3x3 neighbourhood on the fly from the decoupled planes.
* **One threadgroup per row.** The CPU rounds and shifts each row's sum
  (`accum_inner`) before adding it to the frame total. Each threadgroup
  computes exactly one row's sum: 64-bit, reduced with SIMD shuffles of
  `uint2` halves, because MSL `simd_sum` has no 64-bit type. It then applies
  the CPU's row rounding.
* **Host-side tail.** The host sums the rows and computes the
  single-precision `powf` tails. It copies the CPU's expressions and integer
  types into `integer_adm_metal.m`. This includes:
  * the `(uint32_t)pow(2, shift - 1)` rounding constants
  * `i_rfactor`
  * `dwt_quant_step()`, which is static in `integer_adm.c`
* **Specialization.** Kernels are specialized with function constants:
  * `kVariant = scale | bpc << 8 | flags` selects:
    * 8-bit or 16-bit loads
    * int16 storage and rounding at scale 0, or int32 at scales 1 to 3
    * the "low-pass only" variant for `adm_skip_scale0`
    * the "DLM only" variant for `adm_skip_aim`
  * `kFast` selects the precision mode.

The CPU code has several details that a straightforward port gets wrong. The
kernels reproduce each of them:

* **Scale-dependent truncation.** Bands are truncated to int16 at scale 0
  and to int32 at scales 1 to 3.
* **Rounding constants that are not half.**
  * The scale 1–3 decouple rounding constant `1 << (14 + shift)` wraps to
    `INT32_MIN` when `shift` is 17.
  * The scale 1–3 CSF filter and centre adds are `INT32_MIN`.
  * The scale 1–3 denominator square uses `1u << shift_sq`, not half of it.
* **Two regions per scale, with different borders:**
  * decouple and CSF: `left = w * 0.1 - 1.5`, `right = w - left + 2`
  * CSF denominator and CM: `left = w * 0.1 - 0.5`, `right = w - left`
* **CM neighbours at the image edges.** The 3x3 neighbourhood mirrors
  without repeating the edge: row 0 uses row 1 as its upper neighbour, and
  the last row uses itself as its lower neighbour. Columns work the same way.

### The double-precision part

Decouple has two double-precision steps per coefficient:

```c
int angle_flag = (((float)ot_dp / 4096.0) >= 0.0f) &&
    (((float)ot_dp / 4096.0) * ((float)ot_dp / 4096.0) >=
        cos_1deg_sq * ((float)o_mag_sq / 4096.0) * ((float)t_mag_sq / 4096.0));
...
if (angle_flag && (rst_h_f > 0.)) rst_h = MIN((rst_h * adm_enhn_gain_limit), th);
```

Exact mode evaluates both with `soft_f64.h`. The helpers are in
`integer_adm_metal.h`, which both the kernel and the unit test include:

* **`adm_metal_angle_exact()`**
  * The powers of 4096 cancel. `(float)x` is computed as an exact
    round-to-nearest-even to 24 bits.
  * `(float)ot_dp` squared is exact in double, so only the right-hand side
    needs two `sf_mul`.
* **`adm_metal_gain_exact()`**
  * When the gain limit is an integer, as with the default 100 or with 1,
    `rst * gain` is exact in integer arithmetic and no soft double is needed.
  * Otherwise it uses `sf_mul`, a soft compare, and truncation.

Fast mode evaluates both in single precision.

## CPU inconsistencies

### `adm_dwt2_8_neon()` is wrong when `w % 16 == 8`

On arm64, `integer_adm.c` selects `adm_dwt2_8_neon()` for 8-bit input when
`w % 8 == 0`. When `w % 16 == 8`, its last 8 columns read stale temporary
columns. The scalar C path (`--cpumask 1`) gives a different, correct
result.

For example, 120x66 8-bit synthetic frames give `adm2` 0.6352 with NEON and
0.6902 with scalar C. `adm_metal` matches scalar C. It does not mirror this
bug: the bug is a wrong DWT, unlike the clamp difference that `vif_metal`
mirrors in step 1. 1080p and other common widths (multiples of 16) are not
affected.

### Barten CSF at scale 0 converts out of range

At scale 0, `i_rfactor = (uint16_t)(rfactor * pow(2, 21))`. With
`adm_csf_mode=1` (Barten), the value is about 2.5 million, outside the
`uint16_t` range, so the conversion is undefined behavior in C. On this
machine, clang keeps the full value. `adm_metal` matches the CPU because its
host computes `i_rfactor` with the same C expression and the same compiler.
A different compiler or build flags could change the CPU result. The other
CSF modes are in range.

## Verification

Bit-exactness on the test clip, compared as doubles with
`test/vmaf_metal_compare` (from `libvmaf/build-metal`):

| run | frames | features | result |
|---|---:|---:|---|
| `--feature adm:debug=true` 8-bit, 10 threads | 375 | 18 | exact |
| `--feature adm:debug=true` 8-bit, 1 thread | 375 | 18 | exact |
| `--feature adm:debug=true` 10-bit | 30 | 18 | exact |
| `adm_enhn_gain_limit` = 1.0 and 1.2 | 60 | 18 | exact |
| `adm_csf_mode` = 1, 2, 3 | 60 | 18 | exact |
| `adm_norm_view_dist=4`, `adm_ref_display_height=2160` | 60 | 18 | exact |
| `adm_csf_scale=2:adm_csf_diag_scale=0.5`, `adm_noise_weight=0.5` | 60 | 18 | exact |
| `adm_skip_scale0=true`, `adm_skip_aim=true` | 60 | 18 | exact |
| `--model vmaf_v0.6.1` 8-bit (vif, adm on Metal) | 375 | 15 incl. VMAF | exact |
| `--model vmaf_v0.6.1` 10-bit | 30 | 15 incl. VMAF | exact |

Fast mode, maximum absolute difference to the CPU:

| score | 8-bit (375 frames) | 10-bit (30 frames) |
|---|---:|---:|
| `VMAF_integer_feature_adm2_score` | 3.6e-5 | 2.5e-5 |
| `VMAF_integer_feature_aim_score` | 6.9e-6 | 3.7e-7 |
| `VMAF_integer_feature_adm3_score` | 1.8e-5 | 1.3e-5 |
| `integer_adm_scale0` | 8.8e-6 | 0 |
| `integer_adm_scale1` | 2.7e-5 | 0 |
| `integer_adm_scale2` | 1.0e-4 | 1.0e-4 |
| `integer_adm_scale3` | 8.4e-5 | 1.4e-7 |
| VMAF (vmaf_v0.6.1, vif and adm fast) | 7.6e-3 | 5.3e-3 |

Fast mode changes the VMAF score more than VIF's fast mode did, where the
difference was 7.8e-5. A likely cause is the angle test: in single precision,
coefficients near the 1° threshold can switch between the restored and the
impairment band.

Tests (`ninja test`): 30/30 pass in the Metal build, 26/26 in a build without
Metal. The new test **`test/test_metal_adm.c`** covers:

* **The exact helpers, compiled as C, against hardware doubles:**
  * `adm_metal_angle_exact()`: 4M inputs, half of them vector pairs about
    1° apart, so the test is close to its threshold.
  * `adm_metal_gain_exact()`: 6M inputs, six gain limits including
    non-integer ones.
* **`adm_metal` against `adm` with `debug=true` on synthetic frames:**
  * sizes:
    * 176x144
    * 203x117, odd at every scale
    * 144x98, odd band sizes at scales 1 to 3
    * 64x72
    * 168x96 and 120x66, which have `w % 16 == 8` and are compared to scalar
      C only
  * 8, 10 and 12 bit
  * NEON and scalar CPU paths
  * frames with flat, noisy, inverted and identical regions, contrast
    enhancement and attenuation (gain limit, angle test), and black and white
    regions
* **Each option above**, at 8 and 10 bit.

## Speed

Hardware: Apple M3 Max (10 performance cores, 30-core GPU), macOS 26.
Full 375-frame clip, wall time of the `vmaf` CLI, best of 3, including y4m
reading (I/O floor 0.19 s). Command: `RUNS=3 test/metal/bench.sh adm|model`.

adm only (`--feature adm` on CPU vs `--feature adm_metal`, `--no_prediction`):

| config | threads | time (s) | fps | speedup vs CPU |
|---|---:|---:|---:|---:|
| CPU | 1 | 2.34 | 160.3 | 1.00x |
| Metal exact | 1 | 0.91 | 412.1 | 2.57x |
| Metal fast | 1 | 0.84 | 446.4 | 2.79x |
| CPU | 10 | 0.46 | 815.2 | 1.00x |
| Metal exact | 10 | 0.33 | 1136.4 | 1.39x |
| Metal fast | 10 | 0.33 | 1136.4 | 1.39x |

Default model (vmaf_v0.6.1; vif and adm on Metal, motion still on CPU):

| config | threads | time (s) | fps | speedup vs CPU | step 1 |
|---|---:|---:|---:|---:|---:|
| CPU | 1 | 12.33 | 30.4 | 1.00x | |
| Metal exact | 1 | 1.36 | 275.7 | 9.07x | 3.15x |
| Metal fast | 1 | 1.15 | 326.1 | 10.72x | 3.27x |
| CPU | 10 | 1.38 | 271.7 | 1.00x | |
| Metal exact | 10 | 0.61 | 614.8 | 2.26x | 2.37x |
| Metal fast | 10 | 0.47 | 797.9 | 2.94x | 2.37x |

### Exact vs fast

ADM's soft double work is small. The angle test needs two `sf_mul` per
coefficient. The gain limit needs none at the default integer gain of 100.
GPU time per 1080p frame, from command buffer timestamps on the full clip:

| | exact | fast |
|---|---:|---:|
| GPU time per frame, 1 thread | 0.94–0.97 ms | 0.89–0.91 ms |

At 1 thread, the GPU idles between frames. The timestamps depend on its clock
state and varied up to 1.6x between runs. Exact mode costs about 5% more than
fast. The DWT and decouple kernel takes about 70% of the time, and CM the
rest.

At 10 threads:

* **adm alone** is bound by the host, not the GPU. Exact and fast take the
  same time, 0.14 s above the I/O floor.
* **Default model, exact mode:** stays at 0.61 s (0.59 s in step 1). The GPU
  is the bottleneck, and VIF's exact statistic dominates it.
* **Default model, fast mode:** improves from 0.59 s to 0.47 s, because ADM
  no longer competes for CPU cores.

## Possible follow-ups

* **Share vertical DWT results across threads.** Each output coefficient
  recomputes four vertical filter columns, and neighbouring outputs share two
  of them. Staging the vertical results in threadgroup memory would halve
  the DWT loads.
* **Upload each frame once.** Shared frame upload across vif and adm is
  step 4.
