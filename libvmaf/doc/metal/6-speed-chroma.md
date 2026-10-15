# Metal backend, step 6: speed_chroma

This step ports speed_chroma, SpEED on the chroma planes (`speed.c`), to
Metal as `speed_chroma_metal`. The vmaf_v1.0.16 models use it
(`speed_chroma_uv`). After step 5 it was the last feature of these models
that ran on the CPU. Now all of their features run on Metal.

## Result

* All speed_chroma scores are bit-identical to the CPU:
  `speed_chroma_u_score`, `_v_score` and `_uv_score`. This holds in exact
  and fast mode, at 0, 1 and 10 threads, at 8 to 16 bits, for 4:2:0, 4:2:2
  and 4:4:4, and with all options.
* All 8 vmaf_v1.0.16 models give bit-identical VMAF scores to the CPU.
* speed_chroma alone, full 375-frame 1080p clip:
  * `--threads 0`: 0.80 s on the CPU, 0.43 s on Metal (1.9x)
  * `--threads 10`: 0.28 s on the CPU, 0.23 s on Metal (1.2x)
* vmaf_v1.0.16_3d0h, `--threads 0`: 0.52 s, against 0.99 s after step 5 and
  5.50 s on the CPU (10.6x). See [Speed](#speed).
* Completion handlers now run on a serial dispatch queue per context. Before
  this change, the score computations of all contexts were serialized. See
  [Completion handlers](#completion-handlers).

## Design

Files:
* `src/feature/metal/speed_chroma.metal`: the kernels
* `src/feature/metal/speed_chroma_metal.m`: the host code
* `src/feature/metal/speed_chroma_metal.h`: the struct shared by both

The CPU code is unchanged. `speed_chroma_metal.m` contains a verbatim copy
of the static parts of `speed.c` that the host needs: the types, the
matrix code, `est_params()` and the functions it calls, `get_speed_score()`,
`subtract_image()`, `speed_init_dimensions()`, and the option table.
`vif_filter1d_s()`, `vif_get_filter()`, the bilinear column table, and the
covariance kernels (`compute_cov_kernel_neon()` and the others) are
already exported.

### Where the time goes

`filter_and_downscale()` converts each chroma plane to float and filters it
with an anti-aliasing filter at every 16th row and column
(`vif_filter1d_dec16_s()`). The rest runs at 1/16 of the width and height:
a second filter, the local mean subtraction, and `est_params()`. The
covariance matrix and the eigenvalue solve of `est_params()` are in double.

A CPU profile of speed_chroma at 1080p:

| part | share |
|---|---:|
| `vif_filter1d_dec16_s()` | ~33 % |
| covariance (NEON, double) | ~24 % |
| eigenvalues and linear solve | ~38 % |

### Split between the GPU and the host

The GPU does the full-resolution part, on all 4 planes of a frame
(ref U and V, dist U and V) in one command buffer:

| kernel | CPU function | work |
|---|---|---|
| `speed_chroma_metal_scale` | `picture_copy()`, `vif_scale_frame_s()` | only with a prescale: converts to float and scales |
| `speed_chroma_metal_dec16` | `picture_copy()`, `vif_filter1d_dec16_s()` | one thread per output pixel: the vertical filter at the columns that the horizontal filter reads, then the horizontal filter |

The output is 1/256 of the pixels. When the frame completes, the host
runs the rest of `filter_and_downscale()` and `speed_extract_score()` on it,
with the CPU's code: the second filter, the mean subtraction,
`est_params()`, and `get_speed_score()`.

The covariance and the eigenvalue solve stay on the host:
* **Precision.** They are in double, and Metal has no double. The
  covariance has to be bit-exact, as the eigenvalues amplify any
  difference. Step 1's `soft_f64.h` would make it exact, but slow.
* **Size.** They work on 1/256 of the pixels, and with 3 frame slots they
  run in parallel with the GPU work of the next frames.

### Exactness

* **Conversion.** `picture_copy()` computes `(float)v + -128` for 8 bits,
  and `v / scaler + -128` above. The kernel does the same operations in
  float.
* **Filters.** The arm64 CPU build contracts `accum += f * x` into a fused
  multiply-add. MSL kernels are compiled with `fp contract(off)`, so the
  kernel calls `fma()` explicitly, in the CPU's order: the vertical sum
  over the rows first, for each column, then the horizontal sum.
* **Prescale.** `vif_scale_frame_s()` computes the source rows, columns and
  weights in double, then rounds them to float. The host computes them
  with the CPU's code at init:
  * bilinear: `vif_scale_frame_bilinear_precompute_columns_s()`, for the
    columns and for the rows, which the CPU computes in the same way
  * nearest: `(int)(x * ratio)`

  The kernel blends the 4 source pixels with the CPU build's contraction:
  the second product is rounded, then the other three are added with
  `fma()`. If the scaled size equals the input size, the CPU copies the
  planes, and so does the GPU path: it skips the scaling.
* **Host.** The rest is the CPU's code, with the same covariance kernel
  as the CPU extractor.

The kernels ignore `kFast`; fast mode gives the same scores.

### CPU delegation

`init()` delegates to the CPU extractor, `vmaf_fex_speed_chroma`, when:
* `speed_prescale_method` is bicubic or lanczos4 (with a prescale other
  than 1)
* the bit depth is not 8, 10, 12 or 16, which `picture_copy()` does not
  handle
* the format is 4:0:0, or the CPU extractor's `init()` would fail: the
  picture is too small, or the kernel scale or prescale method is invalid

The delegate gets a copy of the options, by name. Its scores are the CPU's.

### Completion handlers

Metal runs the completion handlers of a command queue one at a time, on
one thread. All extractor contexts share the queue, so with `--threads 10`
the host work of all contexts was serialized. This did not matter for
VIF, ADM, motion and CAMBI, whose host work is small. speed_chroma's
`est_params()` is not small: at 10 threads it took 0.43 s, slower than the
CPU extractor (0.28 s).

`vmaf_metal_frames_commit()` now only dispatches `done` from the
completion handler to a serial dispatch queue of the `VmafMetalFrames`.
Each context still processes its frames in commit order. Different
contexts now run in parallel. speed_chroma at 10 threads went from 0.43 s
to 0.23 s. The other extractors are not slower.

### Memory

Per context at 1080p 4:2:0, about 7 MB: each of the 3 frame slots has the
4 input planes and its host buffers at 1/16. A prescale adds one buffer of
4 scaled float planes, shared by the slots: 3 MB for a prescale of 0.6.

## Verification

Bit-exactness on the test clip with `test/metal/vmaf_metal_compare`:

| run | frames | result |
|---|---:|---|
| `--feature speed_chroma`, `--threads` 0, 1, 10 | 375 | exact |
| `--feature speed_chroma` with the 3d0h options, `--threads` 0, 1, 10 | 375 | exact |
| `--feature speed_chroma` with prescale 0.6 bilinear and 0.5 nearest | 375 | exact |
| `--model` with each of the 8 vmaf_v1.0.16 models, `--threads` 0, 10 | 375 | exact |
| `--model vmaf_v0.6.1`, `--threads` 0, 1, 10 (completion handler change) | 375 | exact |
| 10-bit: `--feature speed_chroma`, models 3d0h, 5d0h, 3d0h_2160 | 30 | exact |
| `--fast`: `--feature speed_chroma` | 375 | exact |
| `--fast`: models | 375 | VMAF max diff 1.1e-5, from ADM |

Tests (`ninja test`): 33/33 pass in the Metal build, 26/26 in a build
without Metal. The Metal tests pass with no reports in a
`-Db_sanitize=thread` build.

**New: `test/test_metal_speed_chroma.c`.** It compares speed_chroma and
speed_chroma_metal bit-exactly on synthetic frames: textured chroma with
noise, a noisier and quantized distorted picture, and flat planes on some
frames. The flat planes make the covariance matrix singular, which the CPU
reports and handles.
* **Sizes:** 640x480, 834x586 and 1920x1080 in 4:2:0, 640x360 in 4:2:2,
  400x300 in 4:4:4, at 8, 10, 12 and 16 bits.
* **Options:** 14 option sets at 8 and 10 bits: the kernel scales,
  including the widest filter, `sigma_nn`, `nn_floor`, the variance
  weighting modes, `max_val`, prescales 2.0 (nearest), 1.5 (bilinear and
  bicubic, delegated) and 1.0011 (same size).
* **Prescale:** at 1080p, the options of the 5d0h and 3d0h_2160 models,
  nearest 0.5, and lanczos4 (delegated); bilinear 0.7 at 834x586 4:4:4,
  12 bits.
* **Threads:** 4 threads.
* **Nonzero scores:** each test checks that some scores are nonzero.

Removing the `fma()` of the filter, or replacing the first `fma()` of the
bilinear blend with a multiplication and an addition, makes the test fail.

## Speed

Hardware: Apple M3 Max (10 performance cores, 30-core GPU), macOS 26.
Full 375-frame clip, wall time of the `vmaf` CLI, best of 3, including y4m
reading (I/O floor 0.19 s).

speed_chroma alone (`--no_prediction`), command:
`THREAD_LIST="0 1 10" RUNS=3 test/metal/bench.sh speed_chroma[=options]`.

| options | threads | CPU | Metal exact | Metal fast |
|---|---:|---:|---:|---:|
| default | 0 | 0.80 | 0.43 (1.9x) | 0.43 (1.9x) |
| default | 1 | 0.69 | 0.43 (1.6x) | 0.43 (1.6x) |
| default | 10 | 0.28 | 0.23 (1.2x) | 0.23 (1.2x) |
| 3d0h | 0 | 0.79 | 0.43 (1.8x) | 0.43 (1.8x) |
| 3d0h | 1 | 0.71 | 0.43 (1.7x) | 0.43 (1.7x) |
| 3d0h | 10 | 0.28 | 0.23 (1.2x) | 0.23 (1.2x) |
| 5d0h | 0 | 0.78 | 0.30 (2.6x) | 0.30 (2.6x) |
| 5d0h | 1 | 0.71 | 0.30 (2.4x) | 0.30 (2.4x) |
| 5d0h | 10 | 0.31 | 0.25 (1.2x) | 0.25 (1.2x) |

* **3d0h** is the options of vmaf_v1.0.16_3d0h:
  `speed_kernelscale=1.0:speed_nn_floor=0.1:speed_sigma_nn=0.19:speed_weight_var_mode=5:speed_max_val=45.0`.
* **5d0h** is the options of vmaf_v1.0.16_5d0h: the same with
  `speed_prescale=0.6:speed_prescale_method=bilinear`.
* **At 0 and 1 thread,** the host part bounds speed_chroma alone. With a
  prescale of 0.6, the host part is 0.36x as large.
* **At 10 threads,** the host parts of the contexts run in parallel, and
  the I/O floor dominates.

vmaf_v1.0.16 models (`MODEL=version=<model> test/metal/bench.sh model`).
"Metal, step 5" is the Metal build before this step, which ran
speed_chroma on the CPU. Speedups are relative to the CPU build:

| model | threads | CPU | Metal, step 5 | Metal exact | Metal fast |
|---|---:|---:|---:|---:|---:|
| vmaf_v1.0.16_3d0h | 0 | 5.50 | 0.99 (5.6x) | 0.52 (10.6x) | 0.48 (11.5x) |
| vmaf_v1.0.16_3d0h | 1 | 5.37 | 0.87 (6.2x) | 0.52 (10.3x) | 0.50 (10.7x) |
| vmaf_v1.0.16_3d0h | 10 | 0.68 | 0.44 (1.5x) | 0.42 (1.6x) | 0.41 (1.7x) |
| vmaf_v1.0.16_hfr_3d0h | 0 | 5.45 | 0.99 (5.5x) | 0.52 (10.5x) | 0.50 (10.9x) |
| vmaf_v1.0.16_hfr_3d0h | 1 | 5.39 | 0.87 (6.2x) | 0.52 (10.4x) | 0.50 (10.8x) |
| vmaf_v1.0.16_hfr_3d0h | 10 | 0.69 | 0.43 (1.6x) | 0.42 (1.6x) | 0.41 (1.7x) |
| vmaf_v1.0.16_5d0h | 0 | 5.49 | 1.00 (5.5x) | 0.51 (10.8x) | 0.50 (11.0x) |
| vmaf_v1.0.16_5d0h | 1 | 5.37 | 0.87 (6.2x) | 0.51 (10.5x) | 0.50 (10.7x) |
| vmaf_v1.0.16_5d0h | 10 | 0.70 | 0.44 (1.6x) | 0.43 (1.6x) | 0.42 (1.7x) |
| vmaf_v1.0.16_3d0h_2160 | 0 | 5.37 | 0.87 (6.2x) | 0.50 (10.7x) | 0.49 (11.0x) |
| vmaf_v1.0.16_3d0h_2160 | 1 | 5.25 | 0.75 (7.0x) | 0.50 (10.5x) | 0.49 (10.7x) |
| vmaf_v1.0.16_3d0h_2160 | 10 | 0.70 | 0.45 (1.6x) | 0.42 (1.7x) | 0.41 (1.7x) |

* **0 and 1 thread.** Before this step, the models were bound by
  speed_chroma on the CPU. They now take about as long as the default
  model (0.63 s exact, 0.41 s fast).
* **10 threads.** The models are bound by the GPU, as after step 5, and the
  gain is small.
* **Fast mode.** speed_chroma is the same speed in both modes. The models'
  fast mode is slightly faster through ADM.
* **Default model.** It is unchanged by the completion handler change:
  0.63 s at 0 threads and 0.64 s at 10, against 0.62 s and 0.64 s after
  step 5.

Does the result mean vmaf_v0.6.1 is slower to compute than vmaf_v1.0.16, and vmaf_v1.0.16 has lower speedup on GPU?

Yes to both. The table below comes from the benchmarks in this session: full 375-frame 1080p clip, wall time in seconds, including about 0.19 s of y4m reading.

| model | threads | CPU | Metal exact | Metal fast |
|---|---:|---:|---:|---:|
| vmaf_v0.6.1 (default) | 0 | 12.42 | 0.63 (19.7x) | 0.41 (30.3x) |
| vmaf_v0.6.1 (default) | 10 | 1.42 | 0.64 (2.2x) | 0.43 (3.3x) |
| vmaf_v1.0.16_3d0h | 0 | 5.50 | 0.52 (10.6x) | 0.48 (11.5x) |
| vmaf_v1.0.16_3d0h | 10 | 0.68 | 0.42 (1.6x) | 0.41 (1.7x) |

**On the CPU, vmaf_v0.6.1 is about 2.3x slower to compute.** It uses integer VIF, which was about 75% of its CPU time in the step 0 baseline. vmaf_v1.0.16_3d0h doesn't use VIF. Its features are CAMBI, speed_chroma, adm3 and motion3, which are cheaper on the CPU.

**On Metal, the two models take about the same time.** In exact mode vmaf_v1.0.16 is a bit faster (0.52 s vs 0.63 s). In fast mode vmaf_v0.6.1 is a bit faster (0.41 s vs 0.48 s). VIF's exact mode uses the software double math from step 1, and that is where most of v0.6.1's exact-vs-fast gap comes from.

**vmaf_v1.0.16 has a lower GPU speedup for three reasons:**
- **Cheaper CPU baseline.** VIF is v0.6.1's most expensive CPU feature, and it also maps very well to the GPU. v1.0.16 has no VIF, so there is less CPU time to save.
- **Less GPU-friendly features.** CAMBI's sliding histograms and top-k selection map less well to the GPU. speed_chroma keeps its double-precision covariance and eigenvalue work on the CPU.
- **Fixed costs weigh more.** The 0.19 s of I/O and the GPU time per frame are a large share of a 0.4–0.5 s total. At 10 threads both models are limited by the GPU and I/O, which caps the speedup at 1.6–3.3x for both.

In short, the GPU makes both models cost about 0.4–0.6 s. The speedup ratio mostly shows how expensive each model was on the CPU.

## Not done

* **Covariance and eigenvalues on the GPU.** They are 60 % of the CPU time
  of speed_chroma, but in double. A software double FMA for the
  covariance would be exact. At 0 and 1 thread, the host part overlaps the
  GPU part, so the gain would be small.
  Step 8 ([8-speed-chroma-fast.md](8-speed-chroma-fast.md)) runs them on
  the GPU in fast mode.
* **x86.** The GPU filter reproduces the arm64 build's contraction. An x86
  build without FMA contraction rounds differently. Metal on Intel Macs is
  not tested.
* **Bicubic and lanczos4 prescale** run on the CPU, see
  [CPU delegation](#cpu-delegation). No built-in model uses them.
* **speed_temporal** is not ported. No built-in model uses it.
