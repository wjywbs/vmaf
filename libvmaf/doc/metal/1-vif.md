# Metal backend, step 1: integer VIF

This step ports the integer VIF feature extractor (`vif`, `integer_vif.c`) to
Metal as `vif_metal`. VIF accounts for about 80% of the default model's CPU
time (see [0-infrastructure.md](0-infrastructure.md)).

## Result

* Exact mode (the default) is bit-identical to the CPU extractor for every
  feature, frame and option tested, on 8-bit and 10-bit input.
* On the full 375-frame 1080p clip:
  * vif alone runs 12.5x faster at 1 thread and 2.1x faster at 10 threads.
  * The default VMAF model runs 3.2x faster at 1 thread and 2.4x faster at
    10 threads.

## What was ported

`vif_metal` provides the same features as `vif`, including the debug features
(`integer_vif_num_scale0`, ...). It accepts the same options:

* `debug`
* `vif_enhn_gain_limit` (alias `egl`)
* `vif_skip_scale0` (alias `ssclz`)

Output names, including option suffixes such as `integer_vif_scale0_egl_1`,
are the same as on the CPU. The final score math matches
`integer_vif.c`:

* `vif_residuals_to_score()` (new, in `integer_vif.h`) turns the int64
  accumulators into num/den with the same expression as the CPU statistic
  functions.
* `vif_write_scores()`, which writes the per-scale and debug features, is now
  non-static.

The model path selects `vif_metal` automatically when Metal is enabled. ADM
and motion still run on the CPU in this step.

### Kernels (`src/feature/metal/integer_vif.metal`)

Per scale, one command buffer per frame runs:

| kernel | work |
|---|---|
| `vif_metal_vertical` | vertical filter of ref, dis, ref², dis², ref·dis with the CPU's fixed-point rounding |
| `vif_metal_stat` | horizontal filter, then the VIF statistic per pixel, then a threadgroup reduction to int64 partial sums |
| `vif_metal_subsample` | filters the current scale with the next scale's filter, then decimates by 2 (the next scale's input) |

The host adds the partial sums in int64, so the result does not depend on
summation order. It then applies the CPU's num/den formula.

* **Specialization.** Kernels are specialized with function constants:
  * `kVariant = scale | bpc << 8` selects 8-bit or 16-bit loads, the rounding
    shifts, and the filter.
  * `kFast` selects the precision mode.
* **32-bit accumulation.** The taps of every VIF filter sum to 65536. So
  `sum(fc * x)` fits in 32 bits for any 16-bit `x`. Sample products (up to
  32 bits) are split into 16-bit halves and accumulated in two 32-bit sums,
  then combined into 64 bits once per pixel. This avoids emulated 64-bit
  arithmetic in the filter loops. Partial sums are reduced with `simd_sum` on
  32-bit values. num_non_log is split into 16-bit halves for this.

### The double-precision part

The CPU computes, per pixel:

```c
double g = sigma12 / (sigma1_sq + eps);
int32_t sv_sq = sigma2_sq - g * sigma12;  // fused multiply-subtract on arm64
g = MIN(g, vif_enhn_gain_limit);
int64_t numer1_tmp = (int64_t)(g * g * sigma1_sq);
```

Exact mode evaluates this with `soft_f64.h`. The implementation is
`vif_metal_g_exact()` in `integer_vif_metal.h`, which both the kernel and the
unit test include. Fast mode (`vif_metal_g_fast()`) uses single precision.

As part of this step, `soft_f64.h` got faster:

* `sf_div` now does long division in 11-bit digits, each estimated with a
  float reciprocal and corrected with one integer step. It used to be a
  56-iteration restoring loop.
* `sf_add` and `sf_mul` use 64-bit fast paths instead of 128-bit alignment.
* Integer-to-double conversion is exact below 2^53 and skips rounding.

## A CPU inconsistency that the port reproduces

The first full-clip comparison differed on 5 of 375 frames: `num_scale0` was
off by 0.25. The cause is `vif_statistic_8_neon()`, the CPU path used for
8-bit input at scale 0 on arm64:
* It does not clamp `sigma2_sq` to `>= 0` in its vector columns
  `[0, w & ~7)`.
* Every other implementation clamps: the scalar C code, the NEON residual
  columns, `vif_statistic_16_neon()`, AVX2, AVX-512 and CUDA.

A negative `sigma2_sq` only reaches `num_non_log`, in flat regions where
rounding makes `yy - mu2²` negative.

To be bit-identical with the CPU on the same machine, `vif_metal` mirrors the
extractor that the CPU would select. When the CPU would run
`vif_statistic_8_neon()`, it skips the clamp for columns `< w & ~7` at scale 0
(`noclamp_end` in `VifMetalParams`). With `--cpumask 1` (NEON disabled) the
CPU runs scalar C, and `vif_metal` clamps everywhere. The unit test covers both
cases.

## Verification

Bit-exactness on the test clip, compared as doubles with
`test/metal/vmaf_metal_compare` (from `libvmaf/build`):

| run | frames | features | result |
|---|---:|---:|---|
| `--feature vif:debug=true` 8-bit, 10 threads | 375 | 30 | exact |
| `--feature vif:debug=true` 8-bit, 1 thread | 375 | 30 | exact |
| `--feature vif:debug=true` 10-bit | 30 | 30 | exact |
| `--feature vif:vif_enhn_gain_limit=1.0` | 375 | 4 | exact |
| `--feature vif:vif_skip_scale0=true` | 375 | 4 | exact |
| `--model vmaf_v0.6.1` 8-bit (vif on Metal, adm and motion on CPU) | 375 | 15 incl. VMAF | exact |
| `--model vmaf_v0.6.1` 10-bit | 30 | 15 incl. VMAF | exact |

Fast mode, maximum absolute difference to the CPU:

| score | 8-bit (375 frames) | 10-bit (30 frames) |
|---|---:|---:|
| `VMAF_integer_feature_vif_scale0_score` | 2.4e-7 | 1.8e-7 |
| `VMAF_integer_feature_vif_scale1_score` | 6.0e-7 | 4.8e-7 |
| `VMAF_integer_feature_vif_scale2_score` | 6.0e-7 | 5.4e-7 |
| `VMAF_integer_feature_vif_scale3_score` | 7.2e-7 | 6.0e-7 |
| VMAF (vmaf_v0.6.1) | 7.8e-5 | 7.1e-5 |

Tests (`ninja test`): 29/29 pass in the Metal build, 26/26 in a build without
Metal. New and extended tests:

* **`test/test_metal_vif.c`**:
  * `vif_metal_g_exact()` compiled as C against hardware doubles: 6M inputs
    over the statistic's domain, three gain limits.
  * `vif_metal` against `vif` with `debug=true` on synthetic frames:
    * sizes 176x144, 203x117 (odd sizes exercise the NEON residual columns)
      and 64x72
    * 8, 10 and 12 bit
    * NEON and scalar CPU paths (`cpumask` 0 and 1)
    * frames with flat, noisy, inverted and identical regions, plus nearly
      flat regions that produce negative `sigma2_sq`

    Removing the NEON clamp emulation makes this test fail.
* **`test/test_soft_f64.c` and `test/test_metal.c`**: division cases with
  exact quotients, quotients near 1, 2 and 4, and extreme mantissas, checked
  on both CPU and GPU.
* **`vmaf_metal_compare`**: accepts `--feature name:key=value:...`. It derives
  the option-suffixed output names the same way the extractor does.

## Speed

Hardware: Apple M3 Max (10 performance cores, 30-core GPU), macOS 26.
Full 375-frame clip, wall time of the `vmaf` CLI, best of 3, including y4m
reading (I/O floor 0.19 s). Command: `RUNS=3 test/metal/bench.sh vif|model`.

vif only (`--feature vif` on CPU vs `--feature vif_metal`, `--no_prediction`):

| config | threads | time (s) | fps | speedup vs CPU |
|---|---:|---:|---:|---:|
| CPU | 1 | 9.63 | 38.9 | 1.00x |
| Metal exact | 1 | 0.77 | 487.0 | 12.51x |
| Metal fast | 1 | 0.61 | 614.8 | 15.79x |
| CPU | 10 | 1.06 | 353.8 | 1.00x |
| Metal exact | 10 | 0.50 | 750.0 | 2.12x |
| Metal fast | 10 | 0.41 | 914.6 | 2.59x |

Default model (vmaf_v0.6.1; vif on Metal, adm and motion still on CPU):

| config | threads | time (s) | fps | speedup vs CPU |
|---|---:|---:|---:|---:|
| CPU | 1 | 12.34 | 30.4 | 1.00x |
| Metal exact | 1 | 3.92 | 95.7 | 3.15x |
| Metal fast | 1 | 3.77 | 99.5 | 3.27x |
| CPU | 10 | 1.40 | 267.9 | 1.00x |
| Metal exact | 10 | 0.59 | 635.6 | 2.37x |
| Metal fast | 10 | 0.59 | 635.6 | 2.37x |

### Exact vs fast

GPU time per 1080p frame, summed from command buffer timestamps at 1 thread:

| | exact | fast |
|---|---:|---:|
| GPU time per frame | 1.44 ms | 0.75 ms |

The exact mode overhead is the software double math. It is evaluated only
for pixels with `sigma1_sq >= sigma_nsq`, `sigma12 > 0` and `sigma2_sq > 0`,
but a SIMD group pays for it if any of its lanes needs it. At 10 threads,
exact mode is GPU bound. Fast mode is bound by the host: y4m reading and the
copy of each frame into a Metal buffer. Step 4 (shared upload and pipelining)
targets this host overhead.

With the default model, the remaining CPU extractors (ADM, motion) dominate.
That is why exact and fast take the same time there.

### Optimization history (vif only, exact mode, 10 threads)

| version | time (s) |
|---|---:|
| first bit-exact version | 0.72 |
| `sf_div`: 11-bit digits with float reciprocal estimate, replacing the 56-step restoring loop | 0.57 |
| 32-bit filter accumulation, `simd_sum` reduction | 0.55 |
| 64-bit fast paths in `sf_add`, `sf_mul`, rounding | 0.50 |

At 1 thread, fast mode went from 0.98 s to 0.61 s with the 32-bit
accumulation.

## Possible follow-ups

* **Filter approach for exact mode.** Compute the statistic in float-float
  with a rigorous error bound, and fall back to `soft_f64` only when a
  truncation boundary falls inside the error interval. This could bring exact
  mode close to fast mode.
* **Fused tiles.** Fuse `vif_metal_vertical` and `vif_metal_stat` through
  threadgroup memory. This would avoid writing the five intermediate planes,
  16 bytes per pixel at scale 0.
