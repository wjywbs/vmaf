# Metal backend, step 7: float_ssim and float_ms_ssim

This step ports float_ssim (`float_ssim.c`, `compute_ssim()`) and
float_ms_ssim (`float_ms_ssim.c`, `compute_ms_ssim()`) to Metal, as
`float_ssim_metal` and `float_ms_ssim_metal`. No built-in model uses them.
They are the heaviest optional features left: MS-SSIM takes 57 s on the
CPU for the 375-frame test clip at `--threads 0`.

## Result

* All scores are bit-identical to the CPU in exact mode (the default):
  `float_ssim`, `float_ms_ssim`, and the `enable_lcs` outputs
  (`float_ssim_l/c/s`, `float_ms_ssim_{l,c,s}_scale0..4`). This holds at
  0, 1 and 10 threads, at 8 to 16 bits, and with all options.
* Fast mode differs by at most 2.5e-6 (`float_ms_ssim_s_scale0`), and by
  4.8e-7 on `float_ssim` and 2.1e-7 on `float_ms_ssim`.
* Full 375-frame 1080p clip, see [Speed](#speed):

  | feature | threads | CPU | Metal exact | Metal fast |
  |---|---:|---:|---:|---:|
  | float_ssim | 0 | 2.95 s | 0.26 s (11x) | 0.25 s (12x) |
  | float_ms_ssim | 0 | 56.65 s | 1.25 s (45x) | 0.40 s (142x) |
  | float_ms_ssim | 10 | 6.42 s | 1.29 s (5.0x) | 0.42 s (15x) |

## Design

Files:
* `src/feature/metal/ssim.metal`: the kernels
* `src/feature/metal/ssim_metal.m`: the host code, for both extractors
* `src/feature/metal/ssim_metal.h`: the parameter structs, and the exact
  summation, shared by the kernels, the host and the unit test

The CPU code is unchanged. `ssim_metal.m` contains copies of the option
tables, MS-SSIM's `g_lpf` filter and exponents, and the score code of
`float_ssim.c`, `float_ms_ssim.c`, `compute_ssim()` and
`compute_ms_ssim()`. It uses the exported `g_gaussian_window_h` and the
macros of `iqa/math_utils.h`.

### What the CPU computes

Both features convert the luma planes to float (`picture_copy()`). SSIM
then optionally decimates both planes by an integer factor (`scale`
option, by default `round(min(w, h) / 256)`, 4 at 1080p) with a box filter.
MS-SSIM uses 5 scales, each decimated by 2 with a 9x9 low-pass filter.

At every scale, `_iqa_ssim()` filters 5 maps with an 11-tap Gaussian, in
two passes (`_iqa_convolve()`): ref, dist, ref², dist² and ref·dist. It
then computes l, c and s per pixel, and adds them up over the picture in
double.

### Kernels

All of a frame's work is one command buffer:

| kernel | CPU function | grid |
|---|---|---|
| `ssim_metal_load` | `picture_copy()` | both planes |
| `ssim_metal_decimate` | `_iqa_decimate()` with `_iqa_filter_pixel()` | both planes, output size; SSIM with a factor > 1, MS-SSIM scales 1 to 4 |
| `ssim_metal_conv_h` | `_iqa_convolve()`, horizontal pass | the 5 maps at once, valid columns × all rows |
| `ssim_metal_conv_v` | vertical pass, and the per pixel part of `_iqa_ssim()` | valid columns × valid rows |

`kVariant` is the bit depth for `ssim_metal_load`.

* **Exact mode:** `conv_v` writes l and c (double, as 64-bit patterns) and
  s (float) per pixel. When the frame completes, the host adds them up in
  the CPU's order. These outputs are 20 bytes per pixel.
* **Fast mode:** `conv_v` computes l, c and s in float, and adds up
  `l·c·s`, l, c and s per threadgroup (`simd_sum`, then threadgroup
  memory). The host adds up the partial sums in double.

The intermediates (the converted planes, the decimated planes and the
`conv_h` maps) are shared by the 3 frame slots. The command queue runs
the frames in order. The per pixel outputs are per slot, because the host
reads them after the frame completes, while the next frames run.

### Exactness

The CPU code does three things in double that the GPU cannot do natively:

1. **Filter sums.** `_iqa_filter_pixel()` and both passes of
   `_iqa_convolve()` add float products in a double and round the result
   to float:

   ```c
   double sum = 0.0;
   for (...) sum += img[...] * k[...]; // the float product is rounded
   return (float)sum;
   ```

   A float sum gives a different result. A software double (`soft_f64.h`,
   step 1) per term is exact, but slow. `ssim_metal.h` uses a filter with
   a fallback:
   * `SsimSum` adds the terms in float-float (TwoSum: `hi` + `lo`), and
     their magnitudes in `a`. Its value differs from the exact sum by at
     most n²·2⁻⁴⁸·a. The CPU's double sum differs from the exact sum by at
     most n·2⁻⁵³·a.
   * `ssim_sum_result()` rounds `hi + lo` to the float r. If the distance
     to r plus the error bound n²·2⁻⁴⁶·a (both bounds, with a margin) is
     less than half a unit in the last place of r (a quarter below a power
     of 2, where the rounding interval is narrower below), the CPU's
     result is also r.
   * Otherwise (cancellation, ties, sums near a midpoint between two
     floats, subnormals), the kernel adds the terms again with
     `soft_f64.h`, rounding after every term as the CPU does, and rounds
     the result to float (`ssim_round_float()`, with subnormals).

   The first version used int64 fixed-point sums, which took 3.5 s for
   MS-SSIM at 10 threads (1.3 s of that in `conv_h`). The float-float
   filter takes 1.29 s.

2. **l and c.** `_iqa_ssim()` computes, with float mu and sigma:

   ```c
   l = (2.0 * mu1 * mu2 + C1) / (double)(mu1 * mu1 + mu2 * mu2 + C1);
   c = (2.0 * srsc + C2) / (double)((sigma1_sq + sigma2_sq) + C2);
   ```

   The numerators and quotients are double. `conv_v` computes them with
   `soft_f64.h` (`2.0 * mu1 * mu2` is exact in double). The float parts
   follow the arm64 build's contraction: the sigmas and the denominator of
   l use `fma()`, as the CPU build fuses them.

3. **The sums over the picture.** The CPU adds `l * c * s`, l, c and s
   over all pixels in 4 double chains, which depend on the order. The
   host does this with the per pixel outputs, in the CPU's order. The
   arm64 CPU build vectorizes the loop by 8 pixels: there it rounds
   `l * c` and `l * c * s` before adding; in the last `w % 8` columns of
   each row it fuses the last multiplication and the addition. The host
   does the same.

   Rounding the double sum to float hides almost every difference here:
   adding all pixels without the fusion also gives the CPU's scores on the
   test clip and in the unit test. The host follows the CPU anyway, as
   its double sums are then identical.

At t0, the host sums take about 0.75 s of the 1.25 s of MS-SSIM: about
2.7 million pixels × 4 chains per frame. Each chain is a sequence of
dependent double additions. Splitting the row loop at `w & ~7`, without a
branch per pixel, made them 2.2x faster (1.64 s to 0.75 s).

`picture_copy()` divides by 4, 16 or 256 above 8 bits, which the kernel
does too. The arithmetic is otherwise the CPU's, in the same order, with
`fp contract(off)`.

### Fast mode

The filters use `fma()` in float; l, c and s are float; the sums are per
threadgroup. The pipeline is otherwise the same. The code of both modes
is in the same kernels, selected with `kFast`.

### CPU delegation

`init()` delegates to the CPU extractor (`vmaf_fex_float_ssim` or
`vmaf_fex_float_ms_ssim`) when:
* the bit depth is not 8, 10, 12 or 16, which `picture_copy()` does not
  handle
* MS-SSIM: the picture is too small for 5 scales. The CPU fails with
  "scale below 1x1", and so does the delegate.
* SSIM: the decimated picture is smaller than the 11x11 filter

The delegate gets a copy of the options, by name.

### Memory

Per context at 1080p:
* MS-SSIM exact: about 240 MB. The per pixel outputs of the 5 scales are
  54 MB per slot, 162 MB for 3 slots; the `conv_h` maps 41 MB, the
  converted and decimated planes 22 MB, the input 12 MB.
* MS-SSIM fast: about 75 MB, as the per pixel outputs are replaced by
  partial sums.
* SSIM (factor 4): about 40 MB, mostly the input and the converted
  planes.

With `--threads 10`, MS-SSIM exact peaks at 1.84 GB resident, against
1.11 GB on the CPU and 0.22 GB in fast mode.

## Verification

Bit-exactness on the test clip with `test/metal/vmaf_metal_compare`:

| run | frames | result |
|---|---:|---|
| `--feature float_ssim:enable_lcs=true --feature float_ms_ssim:enable_lcs=true`, `--threads` 0, 1, 10 | 375 | exact, 20 outputs |
| `float_ssim:enable_db=true:scale=3`, `float_ms_ssim:enable_db=true:clip_db=true`, `--threads 0` | 40 | exact |
| identical ref and dist (foreman_cif), `enable_db=true:clip_db=true` | 20 | exact |
| 10-bit, `enable_lcs=true` | 30 | exact |
| `--model vmaf_v0.6.1` | 50 | exact (unchanged) |
| `--fast`, 8-bit | 375 | max diff 2.5e-6 |
| `--fast`, 10-bit | 30 | max diff 1.4e-6 |

Tests (`ninja test`): 34/34 pass in the Metal build, 26/26 in a build
without Metal. The Metal tests pass with no reports in a
`-Db_sanitize=thread` build.

**New: `test/test_metal_ssim.c`.**
* **Summation:** 2 million random sums of 1 to 100 terms, against the
  CPU's double sum: filter products, wide magnitude and sign spreads,
  cancelling pairs, small integers and powers of 2, subnormals, and zeros.
  The filter with its fallback, and the fallback alone, must both match.
  The filter accepts about 94 % of the filter-product sums. The rest are
  mostly exact ties of a few terms.
* **Ties:** sums that the float-float value and the double sum round
  differently (a tie with a term lost in the double sum, below a power of
  2, cancellation, -0). The filter must reject them.
* **CPU vs Metal:** float_ssim and float_ms_ssim with `enable_lcs` on
  synthetic frames (textures with noise, quantized dist, flat and
  identical frames), bit-exact:
  * sizes 640x480, 835x587 and 1283x723 (odd: the decimated sizes round
    up, `w % 8 != 0`), 1920x1080, 200x100 and 64x21 (MS-SSIM fails on both)
  * 8, 9 (delegated), 10, 12 and 16 bits
  * options: `enable_db`, `clip_db`, SSIM `scale` 1, 2, 3, 7, 10, and
    `scale=10` at 225x115 (23x12, on the GPU) and 109x99 (delegated)
  * 4 threads
  * each test checks that some scores are nonzero

Setting the error bound to 0 makes the test fail. Removing the fused tail
of the sums does not, see [Exactness](#exactness).

## Speed

Hardware: Apple M3 Max (10 performance cores, 30-core GPU), macOS 26.
Full 375-frame clip, wall time of the `vmaf` CLI, including y4m reading
(I/O floor 0.19 s). Command:
`THREAD_LIST="0 1 10" RUNS=3 test/metal/bench.sh <feature>` (MS-SSIM: best
of 2).

| feature | threads | CPU | Metal exact | Metal fast |
|---|---:|---:|---:|---:|
| float_ssim | 0 | 2.95 | 0.26 (11.4x) | 0.25 (11.8x) |
| float_ssim | 1 | 2.89 | 0.26 (11.1x) | 0.25 (11.6x) |
| float_ssim | 10 | 0.39 | 0.30 (1.3x) | 0.30 (1.3x) |
| float_ms_ssim | 0 | 56.65 | 1.25 (45.3x) | 0.40 (141.6x) |
| float_ms_ssim | 1 | 56.70 | 1.25 (45.4x) | 0.40 (141.8x) |
| float_ms_ssim | 10 | 6.42 | 1.29 (5.0x) | 0.42 (15.3x) |

* **float_ssim** decimates by 4 at 1080p before the filters, so most of
  its work is the decimation and the input. It is close to the I/O floor
  in both modes.
* **float_ms_ssim exact** is bound by the exact work: the filter sums and
  their fallbacks on the GPU, the soft double divisions of l and c, and
  the host sums. More threads do not help: the GPU is busy, and the
  host sums of the slots overlap the GPU work.
* **float_ms_ssim fast** is 3x faster than exact, and close to the I/O
  floor.

## Not done

* **Smaller exact outputs.** The per pixel outputs make MS-SSIM exact use
  more memory than the CPU at 10 threads, see [Memory](#memory). Fewer
  frame slots, or one set of outputs shared by the slots, would reduce
  it, at some speed cost.
* **x86.** The kernels and the host sums reproduce the arm64 build's
  contraction and vectorization. An x86 build rounds differently. Metal
  on Intel Macs is not tested.
* **Other features.** `psnr`, `float_psnr`, `psnr_hvs`, `ciede` and the
  other speed feature (`speed_temporal`) are cheap or not used by the
  built-in models, and are not ported. Neither are the float versions of
  VIF, ADM and motion (`float_vif`, `float_adm`, `float_motion`),
  `float_ansnr` and `float_moment`.
