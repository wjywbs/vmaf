# Metal backend, step 5: CAMBI

This step ports CAMBI, the banding detector (`cambi.c`), to Metal as
`cambi_metal`. CAMBI is not in the default model, but the vmaf_v1.0.16
models use it (`Cambi_feature_cambi_score`, with
`cambi_high_res_speedup=1080:cambi_vis_lum_threshold=0.06`). Since step 4,
they ran their ADM and motion features on Metal, and CAMBI and speed_chroma
on the CPU.

## Result

* All CAMBI scores are bit-identical to the CPU: `Cambi_feature_cambi_score`,
  `cambi_source` and `cambi_full_reference`. This holds in exact and fast
  mode, at 0, 1 and 10 threads, 8- and 10-bit, with all options.
* CAMBI alone, full 375-frame 1080p clip, `--threads 0`:
  * default options: 5.30 s on the CPU, 1.37 s on Metal (3.9x)
  * the models' options: 2.50 s on the CPU, 0.41 s on Metal (6.1x)
* vmaf_v1.0.16_3d0h and vmaf_v1.0.16_hfr_3d0h:
  * 0 or 1 thread: 3.3–3.6x faster than before this step, and 5.6–6.2x
    faster than the CPU
  * 10 threads: 0.44 s, against 0.46 s before this step and 0.68 s on the
    CPU

  See [Speed](#speed).
* Configurations that the kernels do not cover run the CPU extractor. See
  [CPU delegation](#cpu-delegation).

## Design

Files:
* `src/feature/metal/cambi.metal`: the kernels
* `src/feature/metal/cambi_metal.m`: the host code
* `src/feature/metal/cambi_metal.h`: structs shared by both

The CPU code is unchanged. The option table and the static helpers that
the host needs are copied from `cambi.c` into `cambi_metal.m`:
* `adjust_window_size`, `get_mask_index`
* `quick_select`, `average_topk_elements`
* `weight_scores_per_scale`
* the resampling index loop

`get_tvi_for_diff()` and `get_vlt_luma()` are already exported by `cambi.h`
and `luminance_tools.h`.

The CPU works in place on one picture and slides histograms down the rows.
On the GPU, each stage is a pure function of the previous one. Per frame,
one command buffer runs, for the distorted picture and, with
`full_ref=true`, for the source:

| kernel | CPU function | work |
|---|---|---|
| `cambi_metal_validate` | `validate_image()` | only for bpc other than 8 and 16: flags values above the bit depth |
| `cambi_metal_preprocess` | `cambi_preprocessing()` | converts to 10 bits, resamples to the encode or source size, anti-dithers if `enc_bitdepth` < 10 |
| `cambi_metal_mask_h`, `_v` | `get_spatial_mask()` | 7x7 sum of the zero derivative, as two 1D sums, compared with the mask index |
| `cambi_metal_mode` | `decimate()`, `filter_mode()` | per scale |
| `cambi_metal_keys`, `cambi_metal_cvalues` | `calculate_c_values()` | per scale |
| `cambi_metal_sel_*` | `spatial_pooling()` | the average of the top k c-values of every scale |

The host then weights the scales (`weight_scores_per_scale()`), and writes
the scores when the frame completes, as in step 4.

### Exactness

* **Integer stages.** Every stage before the c-values is integer:
  * The resampling indices are computed on the host with the CPU's float
    loop.
  * The spatial mask is computed once at full size, as on the CPU. A scale
    reads it at `(y << shift, x << shift)`, which is what `decimate()` does
    to the CPU's mask.
  * `filter_mode()` filters in place, but only rows that it no longer reads.
    The kernel reproduces this: the first and last rows and columns keep
    their values.
* **c-values.** A c-value is `(float)(w * p0 * p1) * reciprocal_lut[p0 + p1]`.
  The product is one correctly rounded float multiplication, so the GPU
  rounds it the same way.
* **Pooling.** The CPU averages the top k c-values in double, in the order
  that `quick_select()` leaves them:
  * Nonzero c-values are at least 0.5 and are multiples of 2^-24. The GPU
    sums them exactly as integers in units of 2^-24.
  * While that sum is below 2^53, every partial sum of the CPU is exact too,
    so the CPU's result does not depend on the order. The host then computes
    `(double)sum * 2^-24 / k`.
  * Above, exact mode runs the copied `quick_select()` and
    `average_topk_elements()` on the frame's c-values on the host. On the
    test clip, this never happens; the unit test reaches it 4 times.
  * Fast mode skips that fallback. This is the only difference between the
    modes, so the kernels ignore `kFast`.

### c-values (`cambi_metal_cvalues`)

The CPU keeps, for every column, a histogram of the values of the masked
pixels in the window around it, and updates all of them as it moves to the
next row. On the GPU:

* **One thread per column and chunk of rows.** Each thread keeps the
  histogram of its column in device memory: `hist[chunk][value][column]`,
  for the values in the CPU's band (`v_band_base`, `v_band_size`). The
  chunks give about 16384 threads per scale (`CAMBI_METAL_HIST_THREADS`).
  Each thread builds the window of its first row, then subtracts the row
  that leaves the window and adds the row that enters it.
* **Keys.** `cambi_metal_keys` first writes one 16-bit key per pixel: the
  value if the pixel is masked and its value is in the band, otherwise
  "none". The row updates read the keys and add runs of equal keys at once,
  which helps in flat areas, where banding is.
* **Clearing the histograms.** The histograms must be zero before each scale.
  There are two ways to do this, and the host picks per scale:
  * The kernel subtracts its last window at the end, which reads
    (2 pad + 1)² keys per thread.
  * Otherwise, a zero kernel writes `v_band_size` entries per thread before
    the next use.

  The kernel cleans up when (2 pad + 1)² < `v_band_size`. At 1080p, that is
  the models' options: window 17, band 497. The default options zero
  instead: window 33, band 564. The host tracks whether the buffer is dirty;
  it is cleared once at init.

The first version counted each window directly, about 1000 reads per pixel.
That alone took 3.6 s of the 3.9 s total.

### Top-k selection (`cambi_metal_sel_hist`, `_find`, `_sum`)

`spatial_pooling()` needs the sum of the k = `topk` × n largest c-values of
each scale, where n is the number of pixels in the scale. A full sort is not
needed:

* **Radix select.** Nonzero c-values are at least 0.5, so their float bits
  order them. A radix select finds the k-th largest value t in 3 levels:
  1. 2048 bins: one for 0, then bits 31–16 relative to 0.5f
  2. bits 15–8
  3. bits 7–0

  If t is 0, the select ends after level 1.
* **Histogram kernel.** It counts in threadgroup memory. Lanes of a SIMD
  group with the same bin add their count with one atomic. c-values fall in
  few bins, so plain atomics serialized: selection took 0.5 s of the 1.67 s
  total with default options, and 0.13 s after this change.
* **Sum.** `_sum` adds the values above t as 64-bit integers, in units of
  2^-24. The host adds (k − count) × t.
* **Segments.** All scales of all pictures of a frame are selected together,
  one segment each.

### CPU delegation

`init()` delegates to the CPU extractor, `vmaf_fex_cambi`, when:
* `heatmaps_path` is set
* `max_log_contrast` > 2, which needs more than 4 contrast levels
* a scale is smaller than half the window in width or height. There,
  `calculate_c_values()` reads outside the picture.

The delegate gets a copy of the options, by name. Its scores are the CPU's.

### Memory

Per context at 1080p:
* default options: about 75 MB
* the models' options: about 45 MB

Each of the 3 frame slots has its own input planes, c-values and output.
The intermediates and the c-value histograms (16–18 MB) are shared by the
slots. A frame's c-values stay in its slot until it completes, for the
fallback.

## Verification

Bit-exactness on the test clip with `test/metal/vmaf_metal_compare`:

| run | frames | result |
|---|---:|---|
| `--feature cambi`, `--threads` 0, 1, 10 | 375 | exact |
| `--feature cambi:cambi_high_res_speedup=1080:cambi_vis_lum_threshold=0.06`, `--threads` 0, 1, 10 | 375 | exact |
| `--model vmaf_v1.0.16_3d0h`, `vmaf_v1.0.16_hfr_3d0h`, `--threads` 0, 10 | 375 | exact |
| 10-bit: both CAMBI option sets, `--model vmaf_v1.0.16_3d0h` | 30 | exact |
| `--fast`: both CAMBI option sets, `--model vmaf_v1.0.16_3d0h` | 375 | exact (max diff 0) |
| `full_ref=true` (with `src_width=1280:src_height=720`) | 375 | exact |
| `enc_width/height`, `enc_bitdepth` 8 and 9, `max_log_contrast` 0, 1, 3, `window_size`, `topk`, `tvi_threshold`, `cambi_max_val`, `eotf=pq` | 10 | exact |

`full_ref` with a resampled source and `heatmaps_path` (delegated) were also
compared through the CLI's JSON output. All three scores match on all 375
frames.

Tests (`ninja test`): 32/32 pass in the Metal build, 26/26 in a build without
Metal. The new tests pass with no reports in a `-Db_sanitize=thread` build.

**New: `test/test_metal_cambi.c`.** It compares cambi and cambi_metal
bit-exactly on synthetic frames: slow gradients with bands of one code
value, noise, and flat areas.
* **Sizes:** 320x240, 257x233, 216x216, 640x100, 100x600, and 1000x24
  (delegated), at 8, 10, 12 and 16 bits.
* **Options:** 15 option sets at 8 and 10 bits.
* **1080p:** the models' options, and default options with a resampled
  source. These cover both ways of clearing the histograms.
* **Threads:** 4 threads.
* **Nonzero scores:** each test checks that some scores are nonzero.

Disabling the zero kernel, or starting a chunk's window one row late, makes
the test fail.

## Speed

Hardware: Apple M3 Max (10 performance cores, 30-core GPU), macOS 26.
Full 375-frame clip, wall time of the `vmaf` CLI, best of 3, including y4m
reading (I/O floor 0.19 s). `bench.sh` now takes feature options
(`bench.sh cambi=key=value:...`) and a model (`MODEL=version=...`).

CAMBI alone (`--no_prediction`), command:
`THREAD_LIST="0 1 10" RUNS=3 test/metal/bench.sh cambi[=options]`.

| options | threads | CPU | Metal exact | Metal fast |
|---|---:|---:|---:|---:|
| default | 0 | 5.30 | 1.37 (3.9x) | 1.36 (3.9x) |
| default | 1 | 5.17 | 1.35 (3.8x) | 1.36 (3.8x) |
| default | 10 | 0.60 | 0.88 (0.68x) | 0.88 (0.68x) |
| models' | 0 | 2.50 | 0.41 (6.1x) | 0.41 (6.1x) |
| models' | 1 | 2.39 | 0.41 (5.8x) | 0.41 (5.8x) |
| models' | 10 | 0.35 | 0.31 (1.13x) | 0.31 (1.13x) |

"Models' options" is `cambi_high_res_speedup=1080:cambi_vis_lum_threshold=0.06`.
They run CAMBI at half resolution with a window of 17, which is why they are
cheaper.

vmaf_v1.0.16 models (`MODEL=version=<model> test/metal/bench.sh model`).
"Metal, step 4" is the Metal build before this step, which ran CAMBI on the
CPU. Speedups are relative to the CPU build:

| model | threads | CPU | Metal, step 4 | Metal exact | Metal fast |
|---|---:|---:|---:|---:|---:|
| vmaf_v1.0.16_3d0h | 0 | 5.50 | 3.28 (1.7x) | 0.99 (5.6x) | 0.99 (5.6x) |
| vmaf_v1.0.16_3d0h | 1 | 5.37 | 3.15 (1.7x) | 0.87 (6.2x) | 0.87 (6.2x) |
| vmaf_v1.0.16_3d0h | 10 | 0.68 | 0.46 (1.5x) | 0.44 (1.5x) | 0.44 (1.5x) |
| vmaf_v1.0.16_hfr_3d0h | 0 | 5.46 | 3.28 (1.7x) | 0.99 (5.5x) | 0.98 (5.6x) |
| vmaf_v1.0.16_hfr_3d0h | 1 | 5.36 | 3.15 (1.7x) | 0.86 (6.2x) | 0.87 (6.2x) |
| vmaf_v1.0.16_hfr_3d0h | 10 | 0.69 | 0.46 (1.5x) | 0.44 (1.6x) | 0.43 (1.6x) |

* **0 and 1 thread.** Before this step, the models were bound by CAMBI on
  the calling thread.
* **1 thread is faster than 0.** With 1 thread, the extractors run on a
  worker thread, including speed_chroma, which is still on the CPU. The
  calling thread reads the next frames in parallel.
* **10 threads.** The models are bound by the GPU. Before this step, CAMBI
  ran on the CPU cores in parallel with ADM and motion on the GPU. Now
  CAMBI's GPU time adds to theirs, and the gain is small.
* **Fast mode.** It is the same speed as exact mode, because exact mode only
  differs by the pooling fallback, which this clip never needs.

## Not done

* **CAMBI alone with default options at 10 threads.** It is slower than the
  CPU: 0.88 s against 0.60 s. The c-value kernel does O(window) work per
  pixel and per row, as the CPU does. Most of the GPU time is in this
  kernel. Possible improvements:
  * running the 5 scales of a frame as one dispatch, as they are
    independent after the mode filter. This would help most at 0 and 1
    thread, where the dependent dispatches of one frame leave the GPU
    partly idle: 1.37 s at 0 threads against 0.88 s at 10.
  * a different counting scheme with less work per pixel
* **speed_chroma.** The vmaf_v1.0.16 models still run it on the CPU. It is
  the next step.
* **Heatmaps and `max_log_contrast` > 2** run on the CPU, see
  [CPU delegation](#cpu-delegation).
