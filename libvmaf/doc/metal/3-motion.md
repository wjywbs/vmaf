# Metal backend, step 3: integer motion

This step ports the integer motion feature extractor (`motion`,
`integer_motion.c`) to Metal as `motion_metal`. With it, all three
extractors of the default model (vif, adm, motion) run on Metal. The CPU
code is not changed in this step.

## Result

* `motion_metal` is bit-identical to the CPU extractor for every feature,
  frame and option tested, on 8-bit and 10-bit input. Motion is integer
  only, so exact and fast mode compute the same thing.
* On the full 375-frame 1080p clip:
  * The default VMAF model, now entirely on Metal, runs 11.2x faster than
    the CPU at 1 thread (exact mode; 13.7x in fast mode). After step 2 it
    ran 9.1x faster. At 10 threads, the speedup stays at 2.2x (exact) and
    3.0x (fast).
  * Motion alone is cheap on the CPU. `motion_metal` is 1.4x faster at 1
    thread and 8% slower at 10 threads, where the per-frame upload and GPU
    round trip cost more than the CPU computation.

## What was ported

`motion_metal` provides the same features as `motion`:

* `VMAF_integer_feature_motion_sad_score`
* `VMAF_integer_feature_motion2_score`, `VMAF_integer_feature_motion3_score`
* the debug feature `VMAF_integer_feature_motion_score`

It accepts the same options, with the same defaults, ranges and output name
suffixes:

* `debug`
* `motion_force_zero`
* `motion_blend_factor`, `motion_blend_offset`
* `motion_fps_weight`
* `motion_max_val`
* `motion_five_frame_window`
* `motion_moving_average`

The model path selects `motion_metal` automatically when Metal is enabled.

### How the CPU extractor works today

The plan expected motion to keep the previous blurred frame in memory, so
that frames must be processed in order. The current `integer_motion.c` works
differently:

* It is a `PREV_REF` extractor. For each frame, the framework passes the
  previous reference picture (`fex->prev_ref`, or `fex->prev_prev_ref` with
  the five-frame window) along with the current one.
* It blurs the difference `prev - cur` directly with the 5-tap filter, so no
  state is kept between frames.
* motion2 and motion3 are computed in `flush()`, from the SAD scores in the
  feature collector.

So `PREV_REF` extractors run on the thread pool like any other extractor.
`motion_metal` uses the same flags and the same structure. Its `extract()`
and `flush()` are copies of the CPU functions. Only the SAD computation
(`motion_score_pipeline_8/16()`) runs on the GPU.

### Kernel (`src/feature/metal/integer_motion.metal`)

One kernel, `motion_metal_sad`, runs once per frame:

* **One threadgroup per row.** The CPU adds each row's SAD in a `uint32_t`,
  then adds the row sums in `uint64_t`. The kernel reduces each row in
  `uint32` arithmetic, which wraps the same way, and the host adds the row
  sums in `uint64_t`.
* **Chunks with a halo.** The threadgroup walks the row in chunks of its
  width (up to 256 columns). For each chunk, it computes the vertical filter
  of the difference for the chunk plus 2 columns on each side into
  threadgroup memory. The horizontal filter then reads 5 neighbours from
  threadgroup memory. Columns past the image edge use the CPU's mirroring.
  Any width works.
* **Rounding.** The vertical filter rounds and shifts by 8 for 8-bit input
  and by `bpc` otherwise, the horizontal filter by 16, as on the CPU.
  * The vertical sum fits in 32 bits up to 15-bit input. 16-bit input uses
    64 bits, as the CPU does.
  * The horizontal sum uses 64 bits.
* **Specialization.** `kVariant = bpc` selects 8-bit or 16-bit loads and
  the rounding shift. `kFast` is ignored.

The CPU skips the horizontal pass for rows where the vertical filter is
zero everywhere. Those rows add 0 either way, so the kernel does not need
this shortcut.

## Verification

Bit-exactness on the test clip, compared as doubles with
`test/vmaf_metal_compare` (from `libvmaf/build-metal`):

| run | frames | features | result |
|---|---:|---:|---|
| `--feature motion:debug=true` 8-bit, 10 threads | 375 | 4 | exact |
| `--feature motion:debug=true` 8-bit, 1 thread | 100 | 4 | exact |
| `--feature motion:debug=true` 10-bit | 30 | 4 | exact |
| `motion_five_frame_window=true` | 60 | 4 | exact |
| `motion_moving_average=true` | 60 | 4 | exact |
| `motion_blend_factor=0.5:motion_blend_offset=2` | 60 | 4 | exact |
| `motion_fps_weight=2.5`, `motion_max_val=3`, `motion_force_zero=true` | 60 | 4 | exact |
| `--model vmaf_v0.6.1` 8-bit, 1 and 10 threads (all on Metal) | 375 | 15 incl. VMAF | exact |
| `--model vmaf_v0.6.1` 10-bit | 30 | 15 incl. VMAF | exact |

In fast mode, the motion scores are still exact. The VMAF score differs from
the CPU by at most 7.6e-3 (8-bit) and 5.3e-3 (10-bit), the same as after
step 2. All of that difference comes from vif and adm.

Tests (`ninja test`): 31/31 pass in the Metal build, 26/26 in a build without
Metal. The new test **`test/test_metal_motion.c`** compares `motion_metal` to
`motion` with `debug=true` on 12 synthetic frames. The frames contain a
moving texture, repeated frames (zero motion) and full-range noise (large
SAD). It covers:

* sizes:
  * 176x144
  * 203x117, odd
  * 600x40, wider than a threadgroup, so the halo crosses chunks
  * 513x9, one column past two full chunks
  * 8x6
* 8, 10, 12, 14 and 16 bit
* NEON and scalar CPU paths (`cpumask` 0 and 1)
* each option above, at 8 and 10 bit
* the thread pool (4 threads), with and without the five-frame window

Replacing the mirroring at the right edge with clamping makes this test
fail.

## Speed

Hardware: Apple M3 Max (10 performance cores, 30-core GPU), macOS 26.
Full 375-frame clip, wall time of the `vmaf` CLI, best of 3, including y4m
reading (I/O floor 0.19 s). Command: `RUNS=3 test/metal/bench.sh motion|model`.

motion only (`--feature motion` on CPU vs `--feature motion_metal`,
`--no_prediction`):

| config | threads | time (s) | fps | speedup vs CPU |
|---|---:|---:|---:|---:|
| CPU | 1 | 0.42 | 892.9 | 1.00x |
| Metal exact | 1 | 0.30 | 1250.0 | 1.40x |
| Metal fast | 1 | 0.31 | 1209.7 | 1.35x |
| CPU | 10 | 0.24 | 1562.5 | 1.00x |
| Metal exact | 10 | 0.26 | 1442.3 | 0.92x |
| Metal fast | 10 | 0.26 | 1442.3 | 0.92x |

Exact and fast mode run the same kernel; the 1-thread difference is noise.

Each frame, `motion_metal` copies two planes (previous and current
reference, about 4 MB at 1080p) into Metal buffers and waits for one command
buffer. On the CPU, motion costs about 0.6 ms per frame at 1 thread. Spread
over 10 threads, that is less than the copies and the GPU round trip.

### Default model, end to end

vmaf_v0.6.1, all extractors on Metal:

| config | threads | time (s) | fps | speedup vs CPU | after step 2 | after step 1 |
|---|---:|---:|---:|---:|---:|---:|
| CPU | 1 | 12.36 | 30.3 | 1.00x | | |
| Metal exact | 1 | 1.10 | 340.9 | 11.24x | 9.07x | 3.15x |
| Metal fast | 1 | 0.90 | 416.7 | 13.73x | 10.72x | 3.27x |
| CPU | 10 | 1.39 | 269.8 | 1.00x | | |
| Metal exact | 10 | 0.63 | 595.2 | 2.21x | 2.26x | 2.37x |
| Metal fast | 10 | 0.47 | 797.9 | 2.96x | 2.94x | 2.37x |

* **1 thread.** Motion was the last CPU extractor. Moving it to the GPU
  saves 0.26 s.
* **10 threads.**
  * In exact mode, the GPU is the bottleneck, and VIF's exact statistic
    dominates its time. Moving motion to the GPU adds to that, so the time
    stays about the same.
  * In fast mode, the time stays at 0.47 s, 0.28 s above the I/O floor.
    Each frame is now copied into Metal buffers 6 times (vif and adm copy
    ref and dis, motion copies two refs), with three command buffers per
    frame.

## Possible follow-ups

Step 4 (shared upload and pipelining) targets the per-frame host cost that
now limits the multi-threaded runs:

* Upload each picture once and share the buffer between vif, adm and motion.
  Motion's previous reference was already uploaded as the current reference
  of the previous frame.
* Encode all of a frame's kernels into one command buffer, and wait once.
