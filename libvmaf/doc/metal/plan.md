# Apple Metal GPU acceleration for libvmaf

## Context
libvmaf has CPU (C, NEON, AVX) and CUDA backends but no GPU path on Apple Silicon (this machine: M3 Max, Xcode Metal toolchain present). Goal: add a Metal backend incrementally, one feature extractor per step, highest-impact first. Each step must (1) match the CPU baseline, (2) pass `ninja test`, (3) be benchmarked against the baseline and documented in `libvmaf/doc/metal/N-<feature>.md`, and (4) be committed on its own.

Baseline CPU cost: 60 frames of pedestrian_area_1080p25, `--threads 1`, NEON:

| feature | time | share of default model |
|---|---|---|
| vif (integer) | 1.52 s | ~75 % |
| cambi | 0.74 s | not in default model |
| adm (integer) | 0.48 s | ~24 % |
| float_ssim | 0.45 s | not in default model |
| motion (integer) | 0.07 s | ~3 % |
| full default model (vif + adm + motion) | 1.99 s | 100 % |

**Precision requirement (decided with user):** two modes behind one flag. **Exact** is the default and must be bit-identical to the CPU. **Fast** may drift slightly. One kernel source covers both modes, specialized with Metal function constants so code isn't duplicated. Each step measures the speed of both modes.

Why exact needs work: Metal has no fp64, but integer VIF computes a per-pixel `double g = sigma12/(sigma1_sq+eps)` and then truncates (`src/feature/integer_vif.c:567-577`). ADM's decouple step does a `double` angle test on int64 products (`cuda/integer_adm/adm_decouple.cu:71`). Exact mode uses a small software IEEE-754 double helper for these few operations; fast mode uses float or float-float math.

## Architecture (built in step 0, reused by every later step)

- **No changes to picture plumbing.** Unlike CUDA, which needs special buffer types, ring buffers and `translate_picture`, Metal extractors are ordinary extractors that take normal CPU `VmafPicture`s. Apple Silicon has unified memory, so `extract()` copies the luma plane into a `MTLStorageModeShared` buffer (about 2 MB per 1080p frame, roughly 0.1 ms). Consequences:
  - Metal extractors run in the existing threaded path (`threaded_extract_batch_func`, `src/libvmaf.c:520`).
  - `--threads N` gives N independent command streams, which keeps the GPU busy.
  - Temporal extractors such as motion stay serial, as they are today.
- **New flag `VMAF_FEATURE_EXTRACTOR_METAL = 1 << 4`** in `src/feature/feature_extractor.h`, plus a `VmafMetalState *metal_state` field under `#ifdef HAVE_METAL`. The framework sets the field the same way `set_fex_cuda_state` does (`src/libvmaf.c:275`). Per-thread worker contexts copy the fex struct, so they inherit it automatically.
- **Selection with CPU fallback.** When Metal is imported and `gpumask == 0`, `vmaf_use_features_from_model` (`src/libvmaf.c:433`) first looks up `vmaf_get_feature_extractor_by_feature_name(name, METAL)`. If nothing matches, it falls back to the CPU extractor. This is new behavior; CUDA currently fails on a miss. It lets the default model run at every step while only part of it is ported. Explicit names like `--feature vif_metal` also work.
- **Public API: `include/libvmaf/libvmaf_metal.h`**, mirroring `libvmaf_cuda.h`:
  - `VmafMetalConfiguration { void *device; int fast_math; }`
  - `vmaf_metal_state_init()`
  - `vmaf_metal_import_state()`
  - Released in `vmaf_close`.
  - The precision flag lives here (`fast_math`). The CLI exposes it as `--metal_fast`.
  - The CLI (`tools/vmaf.c:183`) initializes Metal the same way it initializes CUDA: on by default when built with Metal, disabled with `--gpumask`.
- **Host code in `src/metal/`** (Objective-C, `-fobjc-arc`, frameworks Metal + Foundation):
  - `common.m/.h`: device, command queue, and a pipeline cache keyed by (kernel, function constants). Pipelines are created once in the shared state and are thread-safe. Also helpers for buffer allocation and plane upload.
  - Per-extractor host files `src/feature/metal/integer_<feat>_metal.m`. Each holds its own per-context buffers.
- **Kernels** in `src/feature/metal/*.metal`:
  - Compiled at build time with `xcrun -sdk macosx metal -fno-fast-math` into `.air` files, linked into `vmaf.metallib`, then embedded as a byte array (`custom_target` + `xxd -i`). There are no runtime path dependencies.
  - Load with `newLibraryWithData:`.
  - Shared constant `[[function_constant(0)]] bool kFast` selects exact or fast code paths.
  - A `bpc > 8` constant selects the uint8 or uint16 loads.
- **`src/metal/soft_f64.h`**: an exact software double with round-to-nearest-even. It covers only the operations needed:
  - conversion from i32/i64
  - add, sub, mul, div
  - min, compare
  - truncate to i32/i64

  It is written in a C/MSL-common subset (macros for `ulong` and `inline`), so the same file is `#include`d by the `.metal` kernels and by a C unit test that checks it against native `double`.
- **Reductions.** Kernels write per-threadgroup int64 partial sums. The host does the final sum and the double-precision score math, reusing the CPU formulas such as the score code at `integer_vif.c:724-740`. This keeps the final arithmetic identical to the CPU.
- **Porting references.** The CUDA kernels already reproduce the CPU's border mirroring and fixed-point rounding:
  - `src/feature/cuda/integer_vif/filter1d.cu` and `vif_statistics.cuh`
  - `src/feature/cuda/integer_adm/*.cu`
  - `src/feature/cuda/integer_motion/motion_score.cu`

  The CPU C code (`integer_vif.c`, `integer_adm.c`, `integer_motion.c`) remains the reference of truth.

## Shared verification and benchmark tooling (step 0)

- **Test inputs:**
  - `ref`: `libvmaf/build/pedestrian_area_1080p25.y4m`
  - `dist`: decoded once with `dav1d -i libvmaf/build/svtav1_rtc0_p9_crf39.ivf -o libvmaf/build/svtav1_rtc0_p9_crf39.y4m` (the CLI reads only y4m/yuv)
  - 10-bit coverage: the first 30 frames of both, converted with `ffmpeg -pix_fmt yuv420p10le`
- **Build directory:** a separate `libvmaf/build-metal` (`meson setup build-metal --buildtype release -Denable_metal=true`), so the existing `build/` stays a pristine CPU baseline.
- **`libvmaf/test/metal/compare_scores.py`:** compares two `--json` outputs feature by feature and frame by frame, and prints max abs diff per feature.
  - `--exact` fails on any nonzero diff.
  - Without it, it reports and checks a tolerance (default 1e-4 per feature, 1e-3 on the VMAF score).
- **`libvmaf/test/metal/bench.sh <feature>`:** runs the full clip for each configuration and prints a markdown table of wall time, fps, and speedup.
  - CPU baseline: `build/tools/vmaf`, `--threads 1` and `--threads $(sysctl -n hw.perflevel0.physicalcpu)`.
  - Metal exact and Metal fast: same thread counts.
  - Per-feature mode uses `--feature X --no_prediction`. End-to-end mode uses the default model.
- **Unit tests** in `libvmaf/test/` under `if is_metal_enabled`:
  - `test_soft_f64.c`: about 10M random plus edge-case inputs against native double, bit-for-bit.
  - `test_metal_<feat>.c`: runs the CPU and Metal extractors through the fex API on synthetic and real frames, 8- and 10-bit, odd sizes, and asserts bit-exact results in exact mode.

## Per-step procedure (repeated for every step)
1. Implement the kernel(s) and host extractor. Register the extractor in `feature_extractor.c` and `src/meson.build`.
2. `ninja -C build-metal test`: all existing tests plus the new tests must pass.
3. `compare_scores.py --exact` CPU vs Metal on the full ref/AV1 clip (8-bit) and the 10-bit clip: diff must be 0. Fast mode: record the max diff.
4. `bench.sh`: per-feature and default-model timings at 1 thread and N threads.
5. Write `libvmaf/doc/metal/N-<feature>.md` covering:
   - what was ported and the kernel design
   - the verification commands and results (exact diff = 0, fast max diff)
   - the speed table (CPU / Metal exact / Metal fast, 1 and N threads, speedup)
   - hardware and OS
6. `git commit` (one commit per step, attribution trailer included).

## Steps (most impact first)

### Step 0: Metal infrastructure + tooling → `doc/metal/0-infrastructure.md`
- Meson option `enable_metal` (default false; errors on non-darwin). Sets `HAVE_METAL`, adds the objc language and frameworks, builds and embeds the metallib.
- Code from the Architecture section:
  - `src/metal/common.{h,m}`
  - `soft_f64.h`
  - `include/libvmaf/libvmaf_metal.h` (installed header)
  - the fex flag and field
  - selection with fallback
  - CLI `--metal_fast` and Metal init
- A trivial `null_metal`-style smoke kernel plus a test that dispatches it, which proves the build, embed, and load path works.
- `test_soft_f64.c`, `compare_scores.py`, `bench.sh`, and the decoded `dist` y4m.
- The doc records the CPU baseline numbers for every later step: per feature and default model, full clip, 1 and N threads.

### Step 1: Integer VIF (`vif_metal`) → `doc/metal/1-vif.md` (about 75 % of default-model time)
- Kernels:
  - One fused kernel per scale computes the vertical then horizontal 17/9/5/3-tap filters of mu1, mu2, ref², dis², and ref·dis with the CPU's fixed-point rounding, then the per-pixel statistic.
  - A separate decimate kernel produces the next scale.
- The statistic is the only exact/fast split:
  - `kFast=false`: the `g`, `sv_sq`, and `g*g*sigma1_sq` math runs on `soft_f64`.
  - `kFast=true`: float/float-float math.
- The log2 LUT is uploaded once.
- Per-threadgroup int64 accumulators feed the host final score.
- Supports the same options as the CPU extractor: `vif_enhn_gain_limit`, `vif_kernelscale` if it is used by the CPU fex. Unsupported option values delegate to the CPU callbacks.
- Provides the same `provided_features` as `integer_vif.c`.

### Step 2: Integer ADM (`adm_metal`) → `doc/metal/2-adm.md` (about 24 %)
- Port the DWT2, decouple, CSF, CSF-den, and CM kernels, following the CUDA kernels.
- The decouple angle test uses `soft_f64` in exact mode and float in fast mode.
- Single-precision ops use `precise::` functions with no FMA contraction, to match the CPU. Verify how the arm64 CPU build contracts and mirror it.
- Emits the same feature set as `integer_adm.c` for default options. Configurations not ported (AIM/adm3 extras, `adm_norm_view_dist_extra` merge, non-default CSF modes) delegate to the CPU `vmaf_fex_integer_adm` callbacks, so the output is always complete.

### Step 3: Integer motion (`motion_metal`) → `doc/metal/3-motion.md`
- Blur and SAD kernels. The previous blurred frame stays resident in GPU memory. Same motion2 and flush semantics as `integer_motion.c`.
- Integer only, so there is only an exact mode (the flag is a no-op).
- After this step the whole default model runs on Metal. The doc includes the end-to-end table.

### Step 4: Shared-frame upload + async pipelining → `doc/metal/4-pipeline.md`
- Upload each ref/dist frame once per frame and share it across vif, adm, and motion (cached in the state by picture pointer and index), instead of uploading once per extractor.
- Use `newBufferWithBytesNoCopy` when the plane is page-aligned (`vmaf_picture_wrap` callers).
- Encode all of a frame's kernels into one command buffer where possible.
- Measures the end-to-end gain at 1 and N threads.

### Step 5: CAMBI (`cambi_metal`) → `doc/metal/5-cambi.md` (0.74 s/60 frames; heaviest optional feature)
- Port the per-scale decimation, spatial mask, and sliding-window histogram/c-value computation from `cambi.c`. It is integer or float-deterministic, and exact match is required.
- The vmaf_v1.0.16 models (`vmaf_v1.0.16_3d0h`, `_hfr_3d0h`, ...) use CAMBI, so they are benchmarked too.

### Step 6: speed_chroma (`speed_chroma_metal`) → `doc/metal/6-speed-chroma.md`
- The other CPU feature of the vmaf_v1.0.16 models. Follow-up to step 5, benchmarked on the same models.
- The GPU converts, prescales (nearest, bilinear) and filters the full-resolution planes. The host keeps the double-precision covariance and eigenvalue solve on the 1/256 output, with the CPU's code. Bicubic and lanczos4 prescales delegate to the CPU.
- Completion handlers move to a serial dispatch queue per context, so the host work of the contexts runs in parallel.

### Step 7: float_ssim / float_ms_ssim → `doc/metal/7-ssim.md`
- Gaussian filtering and SSIM maps in float. Exact mode reproduces the CPU's float evaluation order; fast mode allows FMA and reordering.

PSNR, ciede, and the other speed features are cheap or niche and are left out of scope (noted in the step 7 doc).

## Critical files
- **Modify:**
  - `libvmaf/meson_options.txt`
  - `libvmaf/src/meson.build`
  - `libvmaf/include/libvmaf/meson.build`
  - `libvmaf/src/libvmaf.c` (state, selection fallback, close)
  - `libvmaf/src/feature/feature_extractor.{h,c}`
  - `libvmaf/tools/vmaf.c`
  - `libvmaf/tools/cli_parse.{c,h}`
  - `libvmaf/test/meson.build`
- **New:**
  - `libvmaf/include/libvmaf/libvmaf_metal.h`
  - `libvmaf/src/metal/{common.h,common.m,soft_f64.h}`
  - `libvmaf/src/feature/metal/{integer_vif,integer_adm,integer_motion,cambi,float_ssim}_metal.m` and the `.metal` kernels
  - `libvmaf/test/test_soft_f64.c`
  - `libvmaf/test/test_metal_*.c`
  - `libvmaf/test/metal/{compare_scores.py,bench.sh}`
  - `libvmaf/doc/metal/*.md`
- **Reuse:**
  - CPU option tables and score math in `integer_vif.c`, `integer_adm.c`, `integer_motion.c`. Expose the static helpers through their headers where needed rather than copying them.
  - `log_generate` (`integer_vif.c:226`)
  - CUDA kernels as the porting reference
  - `aligned_malloc` / `vmaf_picture_*`

## Risks
- **FP contraction and op ordering in single-precision paths (ADM, SSIM).** Mitigation:
  - Build with `-fno-fast-math`.
  - Use `precise::`.
  - Rely on the bit-exact unit tests to catch mismatches early.
- **Throughput at `--threads 1`.** A single command stream may underuse the GPU at 1 thread. Step 4 addresses this, and benchmarks always report N threads too.
- **Cost of `soft_f64`.** It is slower than native math. The exact-vs-fast speed difference is reported per step, as requested.
