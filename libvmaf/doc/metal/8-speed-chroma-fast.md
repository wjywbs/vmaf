# Metal backend, step 8: speed_chroma's covariance and eigenvalues on the GPU (fast mode)

Step 6 left the end of speed_chroma on the host, in both modes: the second
filter at 1/16 of the size, the 25x25 covariance matrix of each plane, its
eigenvalues, and the linear systems of `est_params()`. This step runs them
on the GPU in fast mode (`--metal_fast`), in float. Exact mode is
unchanged.

## Result

* **Exact mode** is unchanged and bit-identical to the CPU, alone and in
  all 8 vmaf_v1.0.16 models.
* **Fast mode** scores differ from the CPU by at most 1.8e-5. They are
  between 0.5 and 30, so the relative difference is about 1e-6. The VMAF
  scores of the vmaf_v1.0.16 models differ by at most 2.2e-5, against
  1.1e-5 (from ADM) before.
* **speed_chroma alone** in fast mode: 0.23 s instead of 0.44 s at
  `--threads 0` and 1, near the I/O floor of 0.19 s. The CPU takes 0.82 s.
  At `--threads 10` it is unchanged, at 0.23 s.
* **The vmaf_v1.0.16 models** do not gain: about the same at
  `--threads 0` and 1 (0.50 s), and 0.04 s slower at `--threads 10`
  (0.45 s instead of 0.41 s). See [Speed](#speed).

## Design

### Kernels

After `speed_chroma_metal_dec16`, fast mode encodes four more kernels into
the frame's command buffer:

| kernel | CPU function | grid |
|---|---|---|
| `speed_chroma_metal_local` | `vif_filter1d_s()` and `subtract_image()`: each pixel minus its local mean | one thread per pixel of the 4 planes |
| `speed_chroma_metal_means` | `compute_mean()`: the 25 means of the shifted submatrices | one threadgroup per offset and plane |
| `speed_chroma_metal_cov` | `compute_covariance_matrix()` | one threadgroup per matrix row and plane |
| `speed_chroma_metal_est` | `compute_eigenvalues()`, `solve_linear_system()`, `update_entropy()` and the steps between them | one threadgroup per plane |

`speed_chroma_metal_est` writes the variances S and entropies E of each
block, and a flag if the matrix is singular. The host passes them to the
CPU's `get_speed_score()`. A singular plane logs the CPU's warning and
gets the CPU's error handling.

Buffers: the outputs of `local`, `means` and `cov` are shared by the frame
slots. The command queue runs the frames in order. S, E and the flags are
per slot, about 2.3 KB per slot at 1080p.

### Covariance

Each threadgroup of `speed_chroma_metal_cov` handles one row x of a
matrix, and adds up the entries y <= x. Each thread adds its pixels for
all of them, in 25 registers, then the threadgroup adds up the threads.
The loops have fixed bounds, so the 25 sums stay in registers. With a
`break` at y > x they were spilled to memory, and the kernel was 4x
slower.

The CPU adds the products in double. A float-float version of the sums
gave the same scores, so the kernel adds in float.

### Eigenvalues

The CPU reduces the matrix to tridiagonal form with Householder
transforms, then runs implicit QR steps on it (adapted from GSL). The
kernel runs the same algorithm in float:
* **Tridiagonal reduction:** on the 32 lanes of simdgroup 0, one row of
  the matrix per lane.
* **QR steps:** on one lane. Each step is O(n) and depends on the
  previous one.

### Linear systems

The CPU solves K X = Y, with Y the 25 pixels of every block, by a QR
decomposition of K, if all eigenvalues of K are at least 1e-6. Then K is
positive definite, and the kernel solves with a Cholesky factor:
* simdgroup 1 computes the factor, in parallel with the eigenvalues, one
  row per lane
* each thread then solves the systems of its blocks, and computes S and E

The matrix is singular if an eigenvalue is below 1e-6, as on the CPU, or
if a Cholesky pivot is not positive. In the tests the two checks always
agree: removing the eigenvalue check fails no test.

### Accuracy

The scores follow the CPU in all the singular / regular decisions of the
test clip and the unit test. Where they differ, the cause is the float
solution of the linear systems, not the GPU:
* **Real video:** replacing only the CPU path's float QR solve with a
  float Cholesky solve, on the host, changes the scores by up to 1.3e-5.
  Fast mode differs by up to 1.8e-5.
* **Smooth synthetic frames:** the per pixel noise is averaged out by the
  filter down by 16, and the covariance matrices are nearly singular. The
  float solutions then differ by up to 5e-4 relative. Replacing the host
  solve with a Cholesky solve in float gives the same differences. Against
  a double-precision solve of the same matrices, fast mode is closer than
  the CPU in 3 of 4 frames. For example, frame 3 is off by 1.2e-2 on the
  CPU and by 1.3e-3 in fast mode.

A bug in an earlier version of the kernel shows why the clip comparison
matters: a mistyped log2(2πe) constant (4.0941296 instead of 4.0941911)
changed the entropies by a relative 2.7e-5. `vmaf_metal_compare --fast`
reported a 1.5e-4 difference on the clip, and the per block comparison
found the bug. The host now computes the constant and passes it to the
kernel.

## Verification

`test/metal/vmaf_metal_compare` on the test clip:

| run | frames | result |
|---|---:|---|
| `--feature speed_chroma`, `--threads` 0, 10 | 375 | exact |
| `--model` with each of the 8 vmaf_v1.0.16 models, `--threads 10` | 375 | exact |
| `--fast --feature speed_chroma`, `--threads` 0, 1, 10 | 375 | max diff 1.6e-5 (u 1.5e-5, v 1.6e-5, uv 1.2e-5) |
| `--fast`, 5d0h options (prescale 0.6 bilinear) | 375 | max diff 1.8e-5 |
| `--fast`, 10-bit | 30 | max diff 1.5e-5 |
| `--fast --model`, vmaf_v1.0.16 models | 375 | VMAF max diff 1.1e-5 (5d0h) to 2.2e-5 (3d0h_2160) |

Tests (`ninja test`): 34/34 pass in the Metal build, 26/26 in a build
without Metal. `test_metal_speed_chroma` passes with no reports in a
`-Db_sanitize=thread` build.

**New in `test/test_metal_speed_chroma.c`: `test_speed_chroma_metal_fast`.**
It compares fast mode with the CPU, with a relative tolerance of 2e-3.
The largest difference it sees is 7e-4.
* **Content:** the frames have texture at the 16-pixel scale (random 8x8
  tiles), which keeps the matrices well-conditioned, as in real video.
  They also include flat planes (singular, as in the existing tests) and
  horizontal stripes (rank deficient).
* **Sizes:** 640x480, 834x586, 1920x1080, 400x352 (2x2 blocks) in 4:2:0;
  640x360 in 4:2:2; 400x300 in 4:4:4.
* **Options:** the 3d0h and 5d0h options, prescale 2.0 (nearest), kernel
  scales 0.5 and 3.71, `sigma_nn` 1.5, variance mode 3, and bicubic
  (delegated to the CPU).
* **Bit depths and threads:** 8 and 10 bits; 4 threads on some cases.

Making the back substitution skip a term, or skipping the QR steps, makes
the test fail.

## Speed

Hardware: Apple M3 Max (10 performance cores, 30-core GPU), macOS 26.
Full 375-frame clip, wall time of the `vmaf` CLI, best of 5 (I/O floor
0.19 s).

speed_chroma alone, `THREAD_LIST="0 1 10" RUNS=5 test/metal/bench.sh speed_chroma[=options]`.
"Fast, step 6" is fast mode before this step, the same as exact mode.

| options | threads | CPU | Metal exact | Metal fast, step 6 | Metal fast |
|---|---:|---:|---:|---:|---:|
| default | 0 | 0.82 | 0.44 | 0.43 | 0.23 (3.6x) |
| default | 1 | 0.71 | 0.44 | 0.43 | 0.23 (3.1x) |
| default | 10 | 0.29 | 0.24 | 0.23 | 0.23 (1.3x) |
| 3d0h | 0 | 0.82 | 0.44 | 0.43 | 0.22 (3.7x) |
| 3d0h | 1 | 0.71 | 0.44 | 0.43 | 0.22 (3.2x) |
| 3d0h | 10 | 0.29 | 0.23 | 0.23 | 0.22 (1.3x) |
| 5d0h | 0 | 0.82 | 0.30 | 0.30 | 0.22 (3.7x) |
| 5d0h | 1 | 0.71 | 0.30 | 0.30 | 0.24 (3.0x) |
| 5d0h | 10 | 0.30 | 0.26 | 0.25 | 0.25 (1.2x) |

(5d0h: best of 3.)

vmaf_v1.0.16_3d0h (`MODEL=version=vmaf_v1.0.16_3d0h test/metal/bench.sh model`):

| threads | CPU | Metal exact | Metal fast, step 6 | Metal fast |
|---:|---:|---:|---:|---:|
| 0 | 5.55 | 0.52 | 0.48 | 0.50 |
| 1 | 5.42 | 0.52 | 0.50 | 0.49 |
| 10 | 0.69 | 0.42 | 0.41 | 0.45 |

The 5d0h and 3d0h_2160 models behave the same way: 0.50 s at 0 threads,
0.45 s at 10 threads in fast mode.

* **0 and 1 thread.** The host part bounded speed_chroma alone. It is
  gone now, so fast mode is nearly 2x faster than exact. The models were
  bound by VIF, ADM and CAMBI on the GPU, and the host part of
  speed_chroma overlapped them. They do not gain.
* **10 threads.** The models are bound by the GPU. Before, the host part
  ran on idle CPU cores, at no cost to the GPU. Now it adds GPU time. In
  an A/B run of fast mode with the work on the GPU or on the host, the
  model took 0.45 s against 0.41 s at 10 threads, and the same time at 0
  threads.

GPU time of speed_chroma fast mode, over the clip at `--threads 0`
(`GPUEndTime - GPUStartTime` of the command buffers). The GPU clock
varies at this low load, so these are approximate:

| part | GPU time |
|---|---:|
| `dec16` (all of exact mode) | ~0.01 s |
| `local` | ~0.01 s |
| `means` | <0.01 s |
| `cov` | 0.01-0.03 s (0.09 s before the fixed loop bounds) |
| `est` | 0.03-0.08 s |
| total | ~0.13 s |

## Power

Energy of a full run of the `vmaf` CLI on the 375-frame clip: CPU,
Metal exact and Metal fast. Measured on an Apple M3 Max with macOS 26,
on AC power with the display on.

### Method

The IOReport "Energy Model" counters are the same ones that
`powermetrics` reads, and reading them needs no root. A small tool
samples them, runs the command, samples again, and prints the energy
between the samples:
* **CPU:** `CPU Energy`, all CPU clusters
* **GPU:** `GPU Energy`
* **DRAM:** `DRAM0`

Total is CPU + GPU + DRAM. The display and the rest of the SoC are not
included.

The counters cover the whole machine, not only the process. Idle, with
the same background load, drew 1.16 W before the 3d0h runs (0.59 W CPU,
0.16 W GPU, 0.41 W DRAM) and 0.50 W before the vmaf_v0.6.1 runs (0.23 W
CPU, 0.03 W GPU, 0.24 W DRAM). That is 0.2-0.7 J over the Metal runs
below, and these numbers include it. Each configuration ran 3 times, and the table
shows the run with the median total. The input was in the page cache.

Energy is the energy of the whole run, in joules. Power is that energy
divided by the wall time, in watts. Energy per frame is total energy
divided by 375.

### vmaf_v0.6.1 (the default model)

`--model version=vmaf_v0.6.1 [--metal_fast]`: integer VIF, ADM and motion,
all on Metal since step 3. speed_chroma is not part of this model; the
table shows the modes on the default model for comparison.

| mode | threads | time (s) | CPU (J) | GPU (J) | DRAM (J) | total (J) | mJ / frame | CPU (W) | GPU (W) | total (W) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU | 0 | 12.35 | 71.80 | 0.24 | 10.78 | 82.82 | 220.8 | 5.8 | 0.0 | 6.7 |
| Metal exact | 0 | 0.63 | 1.15 | 22.29 | 2.72 | 26.16 | 69.8 | 1.8 | 35.5 | 41.7 |
| Metal fast | 0 | 0.42 | 1.89 | 15.21 | 2.74 | 19.84 | 52.9 | 4.5 | 36.4 | 47.5 |
| CPU | 10 | 1.39 | 68.99 | 0.02 | 4.55 | 73.56 | 196.2 | 49.7 | 0.0 | 53.0 |
| Metal exact | 10 | 0.64 | 2.38 | 23.19 | 3.39 | 28.97 | 77.2 | 3.7 | 36.1 | 45.1 |
| Metal fast | 10 | 0.44 | 2.16 | 15.31 | 3.08 | 20.56 | 54.8 | 4.9 | 35.0 | 47.0 |

* **Metal against the CPU.** Metal exact uses 32 % of the CPU build's
  energy at 0 threads (26.2 J against 82.8 J) and 39 % at 10 threads
  (29.0 J against 73.6 J). The CPU spends most of its 12.35 s at 0 threads
  in VIF.
* **Fast against exact.** Fast mode uses 24-29 % less energy (19.8 J
  against 26.2 J, 20.6 J against 29.0 J), and is also 1.5x faster. Unlike
  the 3d0h model, the saving is GPU energy (15.2 J against 22.3 J): exact
  mode computes VIF's per pixel statistic with software doubles
  (`soft_f64.h`), fast mode in float.
* **Power.** The GPU draws about 35 W in both modes, against 20-25 W with
  the 3d0h model below.

### vmaf_v1.0.16_3d0h

`--model version=vmaf_v1.0.16_3d0h [--metal_fast]`. Fast mode also
changes the other features of the model (VIF, ADM), so this compares the
modes as a whole, not only speed_chroma.

| mode | threads | time (s) | CPU (J) | GPU (J) | DRAM (J) | total (J) | mJ / frame | CPU (W) | GPU (W) | total (W) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU | 0 | 5.49 | 31.52 | 0.19 | 6.44 | 38.15 | 101.7 | 5.7 | 0.0 | 6.9 |
| Metal exact | 0 | 0.53 | 4.23 | 10.82 | 1.72 | 16.77 | 44.7 | 8.0 | 20.5 | 31.8 |
| Metal fast | 0 | 0.50 | 2.10 | 10.39 | 1.69 | 14.18 | 37.8 | 4.2 | 20.7 | 28.3 |
| CPU | 10 | 0.69 | 30.31 | 0.01 | 4.33 | 34.65 | 92.4 | 43.9 | 0.0 | 50.1 |
| Metal exact | 10 | 0.43 | 4.64 | 10.66 | 2.08 | 17.38 | 46.3 | 10.9 | 25.0 | 40.7 |
| Metal fast | 10 | 0.45 | 2.47 | 10.17 | 1.99 | 14.63 | 39.0 | 5.5 | 22.5 | 32.4 |

* **Metal against the CPU.** Metal exact uses 44-50 % of the CPU build's
  energy (16.8 J against 38.1 J at 0 threads, 17.4 J against 34.7 J at
  10 threads). The GPU draws 20-25 W while it runs, so the power is much
  higher than the single-threaded CPU run (31.8 W against 6.9 W). The run
  is 10x shorter.
* **Fast against exact.** Fast mode uses 15-16 % less energy (14.2 J
  against 16.8 J, 14.6 J against 17.4 J). Most of it is CPU energy
  (-2.1 J), mainly speed_chroma's host part, which is gone. This holds at
  10 threads too, where fast mode is 0.03 s slower (see [Speed](#speed)):
  it draws 32.4 W against 40.7 W.
* **GPU energy** is about the same in both modes (10.2-10.8 J). The GPU
  work of the model is dominated by VIF, ADM and CAMBI. The extra GPU work
  of speed_chroma's fast mode (about 0.5 J for speed_chroma alone) is
  offset by the cheaper float paths of VIF and ADM.

### Caveats

* The counters are system-wide. Background activity (the editor, the
  terminal) adds noise of a few tenths of a joule per run. The CPU run of
  the model is long enough to collect about 6 J of idle energy.
* DRAM energy includes the idle refresh of the total GB of memory.
* Power is averaged over the whole run, including the start-up (reading
  the model, creating the Metal pipelines), which is a larger share of
  the short Metal runs.

## Not done

* **Choosing the path by load.** Fast mode always runs these steps on the
  GPU. When the GPU is the bottleneck, as with the models at 10 threads,
  the host would be faster. An option, or a choice based on the thread
  count, could select the path.
* **A parallel eigenvalue solver.** The QR steps run on one lane. A
  parallel algorithm such as Jacobi rotations would shorten `est`.
* **Exact mode.** It keeps the host path. An exact covariance would need
  software double additions per product (`soft_f64.h`), and the
  eigenvalues and QR solve would have to follow the CPU's float rounding
  and contraction step by step.
