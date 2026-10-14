# Metal backend, step 4: asynchronous frames

Until step 3, every Metal extractor uploaded its frame, committed a command
buffer and waited for it inside `extract()`. Profiling showed that with one
thread, the GPU was idle half of the time, while the CPU read the next frame,
copied it and computed the scores. This step makes the Metal extractors
asynchronous: `extract()` commits the frame and returns, and the scores are
written when the GPU completes it.

The plan for this step also listed sharing one upload of each frame between
the extractors, zero-copy uploads and one command buffer per frame. Profiling
after the asynchronous change showed that these could save at most 5–7%,
because the GPU is now busy almost all of the time. They were not
implemented; see [Not done](#not-done).

## Result

* All scores are still bit-identical to the CPU in exact mode, at 0, 1 and 10
  threads.
* The default VMAF model on the full 375-frame 1080p clip:

  | | before (step 3) | after |
  |---|---:|---:|
  | 1 thread, exact | 1.10 s (11.2x vs CPU) | 0.63 s (19.7x) |
  | 1 thread, fast | 0.90 s (13.7x) | 0.42 s (29.6x) |
  | 10 threads, exact | 0.63 s (2.2x) | 0.64 s (2.2x) |
  | 10 threads, fast | 0.47 s (3.0x) | 0.44 s (3.2x) |

  With `--threads 0`, the CLI default, the model runs in 0.62 s (exact) and
  0.42 s (fast), 20x and 30x faster than the CPU.
* One thread is now as fast as ten. Both are limited by GPU compute.

## Where the time went

GPU busy time is the union of the command buffers' GPU intervals (temporary
instrumentation, full clip, default model):

| config | GPU busy | GPU span | wall |
|---|---:|---:|---:|
| step 3, 1 thread, exact | 0.62 s | 1.07 s | 1.11 s |
| step 3, 1 thread, fast | 0.40 s | 0.85 s | 0.90 s |
| step 3, 10 threads, exact | 0.57 s | 0.57 s | 0.62 s |
| step 3, 10 threads, fast | 0.38 s | 0.43 s | 0.49 s |
| step 4, 0 threads, exact | 0.58 s | 0.58 s | 0.63 s |
| step 4, 0 threads, fast | 0.39 s | 0.41 s | 0.42 s |
| step 4, 10 threads, exact | 0.58 s | 0.58 s | 0.64 s |
| step 4, 10 threads, fast | 0.39 s | 0.39 s | 0.46 s |

* **Before:** with one thread, each extractor's upload, encoding and score
  math ran while the GPU sat idle, and the GPU ran while the CPU waited. Ten
  threads hid this by having ten frames in flight.
* **After:** the GPU is busy 93–100% of the time in every configuration. The
  remaining wall time above the GPU busy time is reading the y4m input
  before the first frame and finishing the last one.

## Design

### Frames in flight (`src/metal/common.{h,m}`)

`VmafMetalFrames` holds the asynchronous frames of one extractor context. It
has `VMAF_METAL_FRAMES` = 3 slots, each guarded by a dispatch semaphore. An
extractor's `extract()` now does this:

1. Fetches its pipelines. Errors are still returned synchronously.
2. `vmaf_metal_frames_acquire()`: waits until the next slot is free, which
   happens when the frame from three calls ago has completed.
3. Copies the input planes into the slot's input buffers.
4. Encodes the kernels.
5. `vmaf_metal_frames_commit()`: commits the command buffer with a
   completion block, and returns.

When the GPU completes the frame, the completion block runs on a Metal
thread. It reads the slot's output buffers (partial sums, row sums), computes
the scores with the same host code as before, writes them to the feature
collector, and frees the slot.

* **Per-slot and shared buffers.** Buffers that the CPU writes before the
  commit (inputs) or reads after completion (partial sums) exist once per
  slot. GPU-only intermediates are shared by the slots: command buffers on
  one queue that use the same tracked buffers are ordered by Metal's hazard
  tracking. Extra memory per context at 1080p:
  * vif: two more pairs of 8-bit input planes, about 8 MB
  * adm: the same, about 8 MB
  * motion: about 8 MB
* **Code changes in the extractors.** The extractor code barely changes: the
  score code moved into a `compute_scores()` function that the block calls.
  The kernels are unchanged.

### Waiting for scores (`src/libvmaf.c`)

The Metal state counts frames in flight. `vmaf_metal_sync()` waits until
that count is zero, including the completion blocks, and returns the first
error of any frame (for example, a failed command buffer). libvmaf calls it
in two places, both under `#ifdef HAVE_METAL`, like the CUDA
synchronization. Each call syncs the imported state and the default state,
if one was created:

* **Flush (`flush_context()`):** after waiting for the thread pool, if any,
  and before the extractors' `flush()` callbacks. Motion's `flush()` reads
  the SAD scores of all frames, so this ordering is required.
  `vmaf_read_pictures(NULL)` returns the error.
* **`vmaf_close()`:** before the feature collector is destroyed. The thread
  pool's extractor contexts are destroyed after the collector, so their
  in-flight frames must already have completed. This also covers a context
  closed without a flush.

Scores are therefore complete after the flush. `vmaf_read_pictures()` is
documented to queue pictures "for eventual feature extraction". With a
thread pool, or with CUDA, scores were already only complete after the
flush. Without a thread pool, a score of a Metal extractor may now be
missing if it is read before the flush.

* **Metadata callbacks.** Completion blocks write scores from Metal threads.
  The feature collector is already locked for the thread pool. A metadata
  callback registered with `vmaf_register_metadata_handler()` can now run on
  a Metal thread even without a thread pool. With a thread pool, it already
  ran on worker threads.
* **Memory ordering.** The completion block first locks and unlocks the
  state's mutex, which the committing thread released after counting the
  frame. This makes the ordering of the commit before the block explicit.
  Without it, ThreadSanitizer reports false races between `init()` or the
  input copies and the block, because it cannot see the driver's ordering.

## Verification

Bit-exactness on the test clip with `test/vmaf_metal_compare`:

| run | frames | result |
|---|---:|---|
| `--model vmaf_v0.6.1` 8-bit, `--threads` 0, 1, 10 | 375 | exact |
| `--model vmaf_v0.6.1` 10-bit, `--threads 0` | 30 | exact |
| `--feature vif:debug=true`, `adm:debug=true`, `motion:debug=true`, `--threads 0` | 375 | exact |
| `--feature motion:debug=true:motion_five_frame_window=true`, `--threads 0` | 375 | exact |

375 frames cycle through each of the 3 slots 125 times. In fast mode, the
VMAF difference to the CPU is unchanged at 7.6e-3.

Tests (`ninja test`): 31/31 pass in the Metal build, 26/26 in a build without
Metal.

* **New: `test_metal_async` in `test/test_metal.c`.** It runs vif_metal,
  adm_metal and motion_metal:
  * With 0 and 2 threads, it reads every score after the flush. Removing
    the sync from `flush_context()` makes this test fail.
  * It closes contexts with 1080p frames in flight, without flushing, using
    both an imported state and the default state.
* **Existing tests.** `test_metal_vif`, `test_metal_adm` and
  `test_metal_motion` now exercise the asynchronous path. `test_metal_motion`
  includes a 4-thread run.
* **ThreadSanitizer.** All four Metal tests pass with no reports in a
  `-Db_sanitize=thread` build.

## Speed

Hardware: Apple M3 Max (10 performance cores, 30-core GPU), macOS 26.
Full 375-frame clip, wall time of the `vmaf` CLI, best of 3, including y4m
reading (I/O floor 0.19 s). Command:
`THREAD_LIST="0 1 10" RUNS=3 test/metal/bench.sh vif|adm|motion|model`.
`bench.sh` now takes the thread counts from `THREAD_LIST`. 0 runs the
extractors on the calling thread, the CLI default.

Default model (vmaf_v0.6.1, all on Metal):

| config | threads | time (s) | fps | speedup vs CPU |
|---|---:|---:|---:|---:|
| CPU | 0 | 12.54 | 29.9 | 1.00x |
| Metal exact | 0 | 0.62 | 604.8 | 20.23x |
| Metal fast | 0 | 0.42 | 892.9 | 29.86x |
| CPU | 1 | 12.41 | 30.2 | 1.00x |
| Metal exact | 1 | 0.63 | 595.2 | 19.70x |
| Metal fast | 1 | 0.42 | 892.9 | 29.55x |
| CPU | 10 | 1.40 | 267.9 | 1.00x |
| Metal exact | 10 | 0.64 | 585.9 | 2.19x |
| Metal fast | 10 | 0.44 | 852.3 | 3.18x |

Single features (`--feature X` vs `--feature X_metal`, `--no_prediction`),
time in seconds, with the speedup vs the CPU at the same thread count:

| feature | threads | CPU | Metal exact | Metal fast | before (step 1–3), exact |
|---|---:|---:|---:|---:|---:|
| vif | 0 | 9.90 | 0.51 (19.4x) | 0.32 (30.9x) | |
| vif | 1 | 9.68 | 0.51 (19.0x) | 0.35 (27.7x) | 0.77 (12.5x) |
| vif | 10 | 1.06 | 0.51 (2.1x) | 0.37 (2.9x) | 0.50 (2.1x) |
| adm | 0 | 2.54 | 0.25 (10.2x) | 0.26 (9.8x) | |
| adm | 1 | 2.43 | 0.26 (9.4x) | 0.27 (9.0x) | 0.91 (2.6x) |
| adm | 10 | 0.45 | 0.28 (1.6x) | 0.29 (1.6x) | 0.33 (1.4x) |
| motion | 0 | 0.53 | 0.23 (2.3x) | 0.23 (2.3x) | |
| motion | 1 | 0.44 | 0.24 (1.8x) | 0.24 (1.8x) | 0.30 (1.4x) |
| motion | 10 | 0.25 | 0.25 (1.0x) | 0.24 (1.0x) | 0.26 (0.9x) |

adm and motion alone are now close to the I/O floor.

## Not done

Upper bounds were measured by temporarily skipping all plane copies (fast
mode, default model). This gives wrong scores, so it is for timing only:

| threads | with copies | without copies |
|---|---|---|
| 0 | 0.41 s wall, 0.43 s CPU | 0.39 s wall, 0.35 s CPU |
| 10 | 0.44 s wall, 0.68 s CPU | 0.41 s wall, 0.52 s CPU |

* **Sharing one upload per frame between vif, adm and motion.** This would
  remove 4 of the 6 plane copies per frame, about two thirds of the
  difference above: about 0.02 s, or 5%. It needs a cross-extractor cache in
  the Metal state that is safe with the thread pool and with asynchronous
  frames.
* **Zero-copy upload (`newBufferWithBytesNoCopy`).** This would remove all
  copies, at most 0.03 s, or 7%. But:
  * Pictures are only 32-byte aligned (`DATA_ALIGN` in `picture.c`), and
    Metal needs page-aligned memory, so every libvmaf user would get
    page-aligned allocations.
  * All kernels would need the picture stride.
  * Pictures would have to stay referenced until their frames complete.
* **One command buffer per frame across extractors.** The gaps between
  command buffers are at most 7% of the GPU span (fast, 0 threads; 0–1% in
  the other configurations). Merging command buffers needs the extractors to
  encode into a shared command buffer owned by the framework.

The remaining time is GPU compute. The largest part is VIF's exact
statistic: 0.51 s for vif alone in exact mode, against 0.32 s in fast mode.
GPU kernel work, such as a cheaper exact path for VIF (see
[1-vif.md](1-vif.md)) and sharing the vertical DWT results in ADM (see
[2-adm.md](2-adm.md)), would help more than further host-side work.
