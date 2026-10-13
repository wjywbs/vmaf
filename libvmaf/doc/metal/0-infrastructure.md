# Metal backend, step 0: infrastructure

This step adds the Apple Metal GPU backend scaffolding. It ports no feature
extractor yet. Later steps (`1-*.md`, `2-*.md`, ...) port one feature extractor
at a time, with the most impactful first. Each step is verified against the
CPU baseline and benchmarked.

## Build and usage

```sh
cd libvmaf
meson setup build-metal --buildtype release -Denable_metal=true
ninja -C build-metal
ninja -C build-metal test
```

Building requires macOS 14+ and Xcode with the Metal toolchain. If `metal` is
missing, install it with `xcodebuild -downloadComponent MetalToolchain`.

When libvmaf is built with Metal, the `vmaf` CLI uses Metal by default:

| flag | effect |
|---|---|
| (default) | Metal feature extractors are preferred over CPU ones. Features without a Metal port run on the CPU. |
| `--gpumask 1` | disable Metal feature selection (CPU only) |
| `--metal_fast` | fast mode: not bit-exact, see below |
| `--feature vif_metal` | request a Metal extractor explicitly |

Library users call `vmaf_metal_state_init()` and `vmaf_metal_import_state()`
(see `include/libvmaf/libvmaf_metal.h`) to get the same behaviour.

## Precision modes

| mode | `VmafMetalConfiguration.fast_math` | result |
|---|---|---|
| exact (default) | 0 | bit-identical to the CPU feature extractors |
| fast | 1 | double precision math replaced by single precision, small score differences |

Metal has no fp64. Some integer extractors still use doubles per pixel. For
example, integer VIF computes `g = sigma12 / (sigma1_sq + eps)` in double and
then truncates. Exact mode reproduces these operations bit for bit with a
software binary64 implementation, `src/metal/soft_f64.h`. Fast mode uses float
instead. A single kernel source serves both modes, selected through the
function constant `kFast`.

The CPU baseline was disassembled to pin down the exact sequence of double
operations it performs. For example, `sigma2_sq - g * sigma12` is compiled to
a fused `fmsub`, so the Metal port must use `sf_fma()` rather than
`sf_mul()` + `sf_sub()`. All Metal kernels are compiled with
`#pragma METAL fp contract(off)`, because the Metal compiler otherwise fuses
`a * b + c` into an fma even in safe math mode.

## Architecture

* **No changes to the picture pipeline.** Apple Silicon has unified memory, so
  Metal extractors are ordinary extractors that receive CPU `VmafPicture`s and
  copy the planes they need into shared `MTLBuffer`s (about 2 MB per 1080p
  luma plane). Unlike CUDA, there are no device picture types and no ring
  buffers.
  * Metal extractors run through the existing thread pool. With
    `--threads N`, N frames are in flight on the GPU at the same time.
  * Temporal extractors stay serial, as on the CPU.
* **Selection.** Metal extractors carry `VMAF_FEATURE_EXTRACTOR_METAL`.
  `vmaf_use_features_from_model()` looks for a Metal extractor for each model
  feature first and falls back to the CPU extractor. This way the full model
  runs at every step of the port.
* **State.** `VmafMetalState` (`src/metal/common.{h,m}`) holds:
  * the device, a command queue, and the embedded kernel library
  * a thread-safe cache of compute pipelines, specialized by the function
    constants `kFast` (index 0) and `kVariant` (index 1, kernel specific)

  The framework sets `VmafFeatureExtractor.metal_state`. When a Metal
  extractor runs without an imported state, it uses a process-wide default
  state (exact mode).
* **Kernels** (`src/metal/*.metal`, `src/feature/metal/*.metal`) are built
  with `xcrun metal -std=metal3.1 -fno-fast-math` and linked into
  `vmaf.metallib`. The metallib is embedded into libvmaf with `xxd`, so there
  is no runtime file dependency. Loading it at init takes about 30 ms.

## Verification tooling

| tool | purpose |
|---|---|
| `test/test_soft_f64.c` (`ninja test`) | `soft_f64.h` compiled as C vs hardware doubles: add, sub, mul, div, fma, int conversions, truncation, compare. 2M random operands per op, biased toward rounding ties and cancellation. |
| `test/test_metal.c` (`ninja test`) | state init and import, embedded library and pipeline cache, function constants, and `soft_f64.h` **compiled by the Metal compiler and run on the GPU** vs hardware doubles (8 ops × 1M operands). |
| `test/metal/vmaf_metal_compare` | runs a CPU and a Metal `VmafContext` side by side on real video and compares every feature score bit for bit (the CLI output keeps only 6 digits). |
| `test/metal/bench.sh` | wall-time table: CPU vs Metal exact vs Metal fast, at 1 thread and at N threads. |

```sh
# bit-exactness on the test clip
build-metal/test/vmaf_metal_compare -r build/pedestrian_area_1080p25.y4m \
    -d build/av1.y4m --model vmaf_v0.6.1 --threads 8
build-metal/test/vmaf_metal_compare ... --feature vif          # vif vs vif_metal
build-metal/test/vmaf_metal_compare ... --feature vif --fast   # report fast-mode diffs

# speed
test/metal/bench.sh vif      # per feature
test/metal/bench.sh model    # default model end to end
```

### Test inputs

* reference: `build/pedestrian_area_1080p25.y4m`: 1920x1080, 8-bit 4:2:0,
  375 frames
* distorted: `build/av1.y4m`, decoded from
  `av1.ivf` with `dav1d -i av1.ivf -o av1.y4m`
* 10-bit coverage: the first 30 frames of both, converted with
  `ffmpeg -i X.y4m -frames:v 30 -pix_fmt yuv420p10le -strict -1 X_10bit30.y4m`

### Step 0 results

* `ninja test`: 28/28 pass in the Metal build (`build-metal`) and 26/26 in a
  build without Metal.
* `vmaf_metal_compare --model vmaf_v0.6.1` passes on the 8-bit and 10-bit
  clips. At this step this is an end-to-end check of the fallback path, since
  no extractor is ported yet.

## CPU baseline

Hardware: Apple M3 Max (10 performance cores), macOS 26.
Build: `build/` from master, `--buildtype release`, NEON enabled. All numbers
are wall time of the `vmaf` CLI over the full 375-frame clip, best of 3. They
include y4m reading (I/O floor with the `null` extractor: 0.19 s).
Per-feature runs use `--no_prediction --feature X`.

| workload | 1 thread (s) | fps | 10 threads (s) | fps |
|---|---:|---:|---:|---:|
| default model (vmaf_v0.6.1: vif + adm + motion) | 12.45 | 30.1 | 1.39 | 269.8 |
| vif | 9.84 | 38.1 | 1.08 | 347.2 |
| adm | 2.47 | 151.8 | 0.46 | 815.2 |
| motion | 0.46 | 815.2 | 0.25 | 1500.0 |
| float_ms_ssim | 57.74 | 6.5 | 6.41 | 58.5 |
| cambi | 5.25 | 71.4 | 0.61 | 614.8 |
| float_ssim | 2.97 | 126.3 | 0.40 | 937.5 |

## Port order

Ordered by impact on the default VMAF model first, then by cost of the
optional features:

1. integer VIF: about 80% of the default model's time
2. integer ADM: about 20%
3. integer motion: completes the default model on the GPU
4. shared frame upload and pipelining across extractors
5. float_ssim / float_ms_ssim: `float_ms_ssim` is the most expensive extractor
   overall
6. CAMBI
