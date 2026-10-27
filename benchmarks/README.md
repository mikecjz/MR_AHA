# Benchmarks: Toeplitz vs. NUFFT $A^HA$

This folder benchmarks the speed and accuracy of the Toeplitz-embedded operator of MR_AHA
(`mr_aha.AhAPlan`, Python) against the "vanilla" operator that applies type-2 and type-1 cufinufft
transforms in every call (`mr_aha.AhANufftPlan` in Python, `AhA_cufinufft.m` in MATLAB). It also
documents the experiment behind the threshold of the automatic k-space mask (`calc_mask`).

> **Mask default changed after these runs.** `calc_mask` now defaults to the neighbourhood mask
> (`mask_method = 'neighbour'`, `mask_radius = 2`); see [neighbour_mask.md](neighbour_mask.md).
> Every "masked" result in this file used the kernel-threshold method at 0.05, which is still
> available as `mask_method = 'threshold'`.

Everything was run on two machines:

| machine | GPU | software |
|---|---|---|
| local workstation `pact400-recon3` | NVIDIA TITAN RTX, 24 GB (Turing, sm_75) | CUDA 12.6 (nvcc), CuPy 14.2, cufinufft 2.5.1 (pip) for Python, cufinufft 2.6-dev (built from the `finufft/` submodule) for MATLAB R2025b |
| HPC node `esplhpc-cp075` | NVIDIA L40S, 48 GB (Ada, sm_89) | CUDA 12.9 module, CuPy 14.2, cufinufft 2.5.1 (pip) for Python, the cufinufft MEX bundled in the `matlab_biri_v1.6` Singularity image for MATLAB R2025b |

Full per-machine tables, including every intermediate number, are in
[`results/pact400-recon3/summary.md`](results/pact400-recon3/summary.md) and
[`results/esplhpc-cp075/summary.md`](results/esplhpc-cp075/summary.md). The raw numbers are in the
JSON files next to them.

## Summary

* **Toeplitz is faster than the best-tuned NUFFT operator in every case.** One $A^HA$ application
  is **8–20×** faster in 2D, **1.8–2.3×** faster for the stack-of-stars data, and **2.5×** (L40S)
  to **5.1×** (TITAN RTX) faster for the 3D kooshball. The 20-iteration CG reconstructions speed up
  by the same factors, and they still win comfortably once the one-off `calc_toep` time is included.
* **Both forms compute the same operator.** On image-like inputs the Toeplitz and NUFFT operators
  agree to 1–4·10⁻⁶. The 20-iteration CG solutions agree to 3.4–3.6·10⁻⁶ in 2D and stack. The
  MATLAB and Python NUFFT solutions agree with each other to 2.7–3.0·10⁻⁶.
* **The 3D CG difference of 1.5–2.5·10⁻⁴ is a numerical floor, not a Toeplitz error.** The 3D
  problem is badly conditioned (15680 spokes for a 160³ matrix), so CG amplifies single-precision
  rounding. Two runs of the *same* NUFFT operator that differ only in cufinufft's spreading method
  differ by 3.6·10⁻⁶ per application and by **3.5·10⁻⁴** after 20 CG iterations. That is more than
  the Toeplitz-vs-NUFFT difference.
* **MATLAB and Python NUFFT run at about the same speed.** MATLAB is 5–15 % faster on stack and 3D;
  its cufinufft build is newer, since the Python wheel is 2.5.1. Python is 12–17 % faster in 2D.
  Selecting cufinufft's `gpu_method` per transform type matters more than the language: it is worth
  1.4–1.9× in 2D and stack. The default settings are always reported alongside the tuned ones.
* **Masking (`calc_mask`, default threshold 0.05) saves memory, not time.** It keeps 83 % (2D),
  81 % (stack) and 64 % (3D) of the upsampled grid. The kernel shrinks by 17–36 %, which is what
  lets the 3D problem run on the 24 GB TITAN at all: unmasked it needs 19 GB of kernel. The CG
  solution moves by only 2·10⁻⁵. One masked application is *not* faster, for reasons explained in
  [Masking](#masking-threshold-experiment).
* **For a sparse trajectory the mask saves far more.** With a two-arm spiral it keeps only
  6–11 % of the grid at N = 256–384, so the stack-of-spirals kernel (320² × 56, L = 12) shrinks
  **12×**, from 13.3 GB to 1.1 GB. The masked $A^HA$ is also 1.7× faster than the unmasked one. The
  catch is accuracy: at 0.05 the operator error is 10⁻³, not 10⁻⁶; threshold 0.01 brings it to
  10⁻⁴ and still saves 2.5×. See [Sparse trajectories](#sparse-trajectories-two-arm-spiral).

## Problems

| name | data | size | coils | L | samples | CG right-hand side |
|---|---|---|---|---|---|---|
| `2d` | `data/human_C-35_2D_radial.mat` | 160² | 16 | 6 | 377 spokes × 160 | stored `AhAx_ref` = $A^HA\,x$ (CPU finufft), so the truth is the stored `x` |
| `stack` | `data/AhAx_test_toeplitz.mat` | 320² × 56 | 12 | 12 | 280 spokes × 320, × 56 kz | stored `Ahb` |
| `3d` | `data/kooshball_3d.mat`, synthetic ([`make_kooshball.py`](make_kooshball.py)) | 160³ | 12 | 12 | 15680 spokes × 160 | $A^HA\,x_\text{true}$ for a random-ellipsoid `x_true` |

The synthetic 3D problem uses golden-means kooshball directions, smooth complex Gaussian coil
maps, and a smooth complex orthonormal temporal basis $U$. Each spoke is acquired at one time
point, so its Gram block is the rank-one $\overline{u_t}u_t^T$.

**Stack data convention.** `Phi2` is stored as `[L, L, Ntraj, Nz]`. Its trace over kz peaks at
indices 0 and 55 and is smallest around 28, so kz is in *natural* FFT order (DC first). The loaders
permute it to `[Ntraj, Nz, L, L]` and `fftshift` it along kz into the centred order the library
expects. Both operators use the same convention, so the comparison is unaffected either way, but
the reconstruction is only physically right with the shift.

## Method

* **NUFFT operator.** Per coil: sensitivity → type-2 cufinufft (sign −1) → one $L\times L$ block
  per readout → type-1 cufinufft (sign +1) → conjugate sensitivity. All `L` ranks (× `Nz` kz planes
  for the stack) go through one batched plan, and the plans and point sorting are built once. The
  settings are `eps = 1e-6`, `upsampfac = 2`, single precision.
  * For the stack, the centred z-FFT is a plain FFT applied to the image after modulating it by
    $e^{2\pi i c n/N_z}$, folded into the coil map. The $k$-side phases cancel because the Gram
    blocks are diagonal in kz, so no `fftshift` copies are made.
  * The block-diagonal product uses `pagemtimes` in MATLAB (4 ms vs 16 ms for a broadcast-and-sum
    on the 3D data) and `einsum` in CuPy.
* **Tuning.** "Tuned" picks the fastest (type-1, type-2) `gpu_method` pair out of {1, 2}² for each
  dataset and machine.
* **Toeplitz operator.** `mr_aha.calc_toep` once, then `AhAPlan.apply`. "Masked" uses
  `mr_aha.calc_mask` with the threshold method at 0.05, which was the default at the time.
* **Timing.** Wall time with device synchronisation, after a warm-up call. The median of 20 (2D)
  or 5 (stack, 3D) calls is reported.
* **CG.** Plain conjugate gradient from $u_0 = 0$, exactly 20 iterations, no stopping test. The
  algorithm is identical in Python (`common.cg`) and MATLAB (`bench_matlab.m`). All solutions are
  compared pairwise as $\|a-b\|/\|b\|$.

## One $A^HA$ application (median)

| problem | GPU | NUFFT Py default / tuned | NUFFT MATLAB default / tuned | Toeplitz | Toeplitz masked | Toeplitz vs best NUFFT |
|---|---|---|---|---|---|---|
| 2d | TITAN RTX | 35.6 / 19.0 ms | 42.8 / 23.0 ms | **2.3 ms** | 2.6 ms | **8.2×** |
| 2d | L40S | 25.4 / 13.4 ms | 29.6 / 15.3 ms | **0.67 ms** | 0.79 ms | **19.9×** |
| stack | TITAN RTX | 5.23 / 3.64 s | 4.36 / 3.16 s | **1.36 s** | 1.38 s | **2.3×** |
| stack | L40S | 2.88 / 1.84 s | 2.55 / 1.71 s | **0.94 s** | 1.00 s | **1.8×** |
| 3d | TITAN RTX | 11.85 / 9.45 s | 10.48 / 8.97 s | does not fit (19 GB kernel) | **1.74 s** | **5.1×** |
| 3d | L40S | 3.79 / 3.79 s | 3.94 / 3.39 s | **1.33 s** | 1.36 s | **2.5×** |

One-off pre-computation (`calc_toep`):

| problem | TITAN RTX | L40S | kernel size, unmasked → masked |
|---|---|---|---|
| 2d | 0.03 s | 0.17 s | 16 → 14 MB |
| stack | 2.9 s | 2.9 s | 13.3 → 10.8 GB |
| 3d | 20.3 s (masked only) | 7.6 s | 19.0 → 12.1 GB |

On the HPC node every first-time plan or allocation carries about 0.1–0.15 s of overhead, which
dominates its small 2D pre-computation times. Local plan creation takes milliseconds.

## CG, 20 iterations from zero

| problem | GPU | NUFFT Py | NUFFT MATLAB | Toeplitz (+ `calc_toep`) | Toeplitz masked (+ mask + `calc_toep`) |
|---|---|---|---|---|---|
| 2d | TITAN RTX | 0.40 s | 0.50 s | **0.05 s** (+0.03) | 0.06 s (+0.02) |
| 2d | L40S | 0.31 s | 0.51 s | **0.02 s** (+0.17) | 0.02 s (+0.28) |
| stack | TITAN RTX | 74.1 s | 64.2 s | **27.6 s** (+2.9) | 27.9 s (+3.2) |
| stack | L40S | 37.1 s | 34.4 s | **19.1 s** (+2.9) | 20.3 s (+2.7) |
| 3d | TITAN RTX | 184.5 s | 175.3 s | does not fit | **35.1 s** (+20.3) |
| 3d | L40S | 76.0 s | 68.6 s | **26.8 s** (+7.6) | 27.4 s (+7.1) |

Relative differences between the CG solutions; both machines agree to the digits shown:

| problem | Toeplitz vs NUFFT (Py) | Toeplitz vs NUFFT (MATLAB) | NUFFT Py vs MATLAB | masked vs unmasked Toeplitz | error vs truth (all operators) |
|---|---|---|---|---|---|
| 2d | 3.4·10⁻⁶ | 3.4–3.5·10⁻⁶ | 2.7–2.8·10⁻⁶ | 1.8·10⁻⁵ | 8.355·10⁻² |
| stack | 3.6·10⁻⁶ | 3.6·10⁻⁶ | 3.0·10⁻⁶ | 2.7·10⁻⁵ | – |
| 3d | 1.7·10⁻⁴ (TITAN, masked), 2.5·10⁻⁴ (L40S) | 2.5·10⁻⁴ (L40S) | 1.7·10⁻⁵ (TITAN), 7.7·10⁻⁵ (L40S) | 2.3·10⁻⁵ | 2.483·10⁻¹ |

The final relative residuals agree to three or more digits in every case (for example
1.768·10⁻² for all four stack solutions). The errors against the truth are identical, so the
choice of operator has no visible effect on reconstruction quality.

**The 3D floor.** At the operator level the 3D Toeplitz and NUFFT differ by 3.8·10⁻⁶, the same
order as in 2D. The CG difference is about 50× larger because the 20 iterations of a poorly
conditioned problem amplify any 10⁻⁶-level perturbation. To check this, CG was rerun with the NUFFT
operator using spreading methods (2,2) instead of (1,2). That solution differs from the (1,2)
solution by **3.5·10⁻⁴**, although the two operators differ by only 3.6·10⁻⁶. Raising the NUFFT
accuracy to its single-precision limit (`eps` = 1.2·10⁻⁷) moves the operator by another 2·10⁻⁶. The
Toeplitz solution therefore lies within the spread of the NUFFT solutions themselves.

## Masking threshold experiment

`calc_mask` evaluates the Toeplitz kernel of the trajectory with all weights set to one, max-filters
it over each location's 3×3 (3×3×3) neighbourhood, and keeps locations above a threshold. The study
behind the design ([`mask_threshold.py`](mask_threshold.py)) measured, for every threshold:

* the kept fraction;
* the error of one masked application against the NUFFT operator, for an image-like input and for
  white noise (the worst case);
* the difference between the 20-iteration CG solutions of the masked and NUFFT operators.

Results on the L40S, where every threshold fits, at the chosen default **0.05**:

| problem | kept | operator error, image | operator error, white noise | CG diff vs NUFFT | CG diff of unmasked Toeplitz |
|---|---|---|---|---|---|
| 2d | 83.1 % | 4.3·10⁻⁶ | 8.4·10⁻⁴ | 1.8·10⁻⁵ | 3.4·10⁻⁶ |
| stack | 81.2 % | 1.3·10⁻⁶ | 3.6·10⁻³ | 2.7·10⁻⁵ | 3.6·10⁻⁶ |
| 3d | 63.7 % | 3.9·10⁻⁶ | 7.3·10⁻⁴ | 1.7·10⁻⁴ | 1.5·10⁻⁴ |

How the design was arrived at:

1. **Plain thresholding of $|T|$ is not enough.** The unit-weight kernel is a sum of Dirichlet
   kernels, and it cancels to nearly zero at isolated points *inside* the sampled disc. The kernel
   of the real Gram blocks does not cancel there. Even a perfect disc mask ($|k| \le$ the sampled
   radius) leaves a 2·10⁻² white-noise error, because the exact kernel has slowly decaying tails.
   Max-filtering before thresholding fixed most of this: at the same kept fraction the image-input
   error fell about tenfold (2D, 80 % kept: 3.5·10⁻⁵ instead of 3.8·10⁻⁴). A 5×5 or 7×7 filter
   gained little more, so the 3×3 filter was kept.
2. **The threshold must be absolute, not relative to $\max|T|$.** For radial sampling the
   maximum sits at the k-space centre, where every spoke contributes. It is therefore the number of
   spokes (378, 280 and 15680 here) and has nothing to do with the level at the edge of k-space.
   With a relative threshold, 2D wanted about 10⁻⁴ and 3D about 10⁻⁵. Measured in samples (one
   isolated on-grid sample contributes exactly 1), the good operating points coincide at a few
   hundredths in all three problems.
3. **Choice.** Above 0.1 the CG difference grows quickly: τ = 1 gives 5·10⁻³ in 2D and 3·10⁻² for
   the stack. Below 0.03 the kept fraction rises steeply in 3D (91.7 % at 0.01) for no gain in 3D
   accuracy. 0.05 sits in between: image-input errors stay at a few 10⁻⁶ (at most 4× the unmasked
   operator's own error against the NUFFT), and CG solutions move by about 2·10⁻⁵.

**Why masking is not faster.** The masked execute path in `mr_aha.cu` gathers the retained
locations into a compact buffer, multiplies them, zeroes the *whole* padded grid with `cudaMemset`
and scatters the result back. That costs about as much as the dense in-place multiply it replaces,
and the two padded FFTs, which masking does not touch, dominate the operator anyway. A cheaper
masked path is possible; see the recommendations below.

## Sparse trajectories: two-arm spiral

Radial trajectories cover most of the 2×-upsampled grid, so the mask can only drop 20–35 % of it.
To see what a sparse trajectory gains, [`spiral_mask.py`](spiral_mask.py) repeats the experiment
with a **two-arm Archimedean spiral**: 2 arms × 1024 samples, one revolution, from
[`make_spiral.py`](make_spiral.py) (mrinufft's `initialize_2D_spiral`). Everything was measured on
the TITAN RTX only.

The coefficient images, coil maps and Gram blocks are synthetic. The images are random ellipses or
ellipsoids, the coil maps are 12 smooth complex Gaussians, each readout (and each kz for the stack)
gets a random rank-3 PSD block, and L = 12. The CG right-hand side is $A^HA\,x_\text{true}$
computed with the NUFFT operator.

**Kept fraction and kernel size** (`phi2_toep` plus the `int64` mask, L = 12; the stack has
Nz = 56):

| N | kept at 0.05 | kept at 0.01 | 2D kernel: unmasked → 0.05 / 0.01 | stack kernel: unmasked → 0.05 / 0.01 |
|---|---|---|---|---|
| 128 | 39.6 % | 82.3 % | 39 → 15.6 / 32.5 MB | 2.13 → 0.86 / 1.78 GB |
| 192 | 18.6 % | 68.8 % | 88 → 16.5 / 61.2 MB | 4.80 → 0.90 / 3.35 GB |
| 256 | 10.6 % | 50.0 % | 156 → 16.8 / 79.0 MB | 8.53 → 0.92 / 4.32 GB |
| 320 | 8.0 % | 38.6 % | 244 → 19.7 / 95.3 MB | 13.3 → 1.08 / 5.21 GB |
| 384 | 6.4 % | 30.9 % | 351 → 22.7 / 110 MB | 19.2 → 1.24 / 6.00 GB |

The same 2048 samples cover a smaller share of a bigger grid, so the saving grows with N. At 0.05
the kernel is 9–15× smaller for N ≥ 256.

**Accuracy and measured memory.** Errors are against the NUFFT operator. "CG diff" compares the
20-iteration CG solutions of the masked Toeplitz and NUFFT operators. The NUFFT CG solution itself
is 0.94 (2D) and 0.81 (stack) away from `x_true`: two arms are extremely undersampled.

| problem | threshold | kept | kernel | operator error, image | operator error, white noise | CG diff | one $A^HA$ |
|---|---|---|---|---|---|---|---|
| 2D 256² | unmasked | 100 % | 156 MB | 5.2·10⁻⁷ | 4.8·10⁻⁷ | 2.2·10⁻⁵ | 13.2 ms |
| | 0.05 | 10.6 % | 16.8 MB | 1.0·10⁻³ | 9.8·10⁻³ | 1.6·10⁻² | **7.6 ms** |
| | 0.01 | 50.0 % | 79.0 MB | 1.0·10⁻⁴ | 2.5·10⁻³ | 7.8·10⁻⁴ | 11.0 ms |
| | 0.003 | 83.3 % | 132 MB | 4.6·10⁻⁶ | 3.3·10⁻⁴ | 6.6·10⁻⁵ | 13.7 ms |
| stack 320² × 56 | unmasked | 100 % | 13.3 GB | does not fit with the CG vectors (~19.5 GB) | | | |
| | 0.05 | 8.0 % | 1.08 GB | 1.1·10⁻³ | 1.2·10⁻² | 9.6·10⁻³ | **0.88 s** |
| | 0.01 | 38.6 % | 5.21 GB | 1.1·10⁻⁴ | 3.6·10⁻³ | 7.9·10⁻⁴ | 1.12 s |

For comparison, one NUFFT $A^HA$ takes 14.2 ms (2D) and 1.46 s (stack). With only 2048 samples
per plane, the NUFFT is cheap and the unmasked Toeplitz operator barely beats it in 2D.

Findings:

* **The kernel is no longer the main allocation once masked.** For the stack at 0.05, the device
  memory of the operator is about 1.1 GB of kernel plus 2.8 GB measured while creating the
  `AhAPlan` and applying it once. That 2.8 GB is mostly the padded `[Mx·My·Mz, L]` work buffer
  (2.1 GB) plus the output; masking does not shrink it. Unmasked, the same operator would need about
  13.3 + 2.6 ≈ 16 GB, so masking cuts the total about 4×.
* **`calc_toep` uses whatever memory it is given.** Its chunk size scales with `mem_fraction`
  (default 0.5 of free memory), so the stack peaked at about 10 GB. With `mem_fraction = 0.02` the
  peak drops to 1.6 GB at 0.05, which is the kernel plus about 0.5 GB, and it was no slower (1.8 s
  either way). Lower `mem_fraction` when the kernel must coexist with other data.
* **Here masking is also faster.** When only 6–11 % of the grid is kept, the gather → multiply →
  scatter path touches far less kernel data than the dense multiply. The 2D operator gets 1.7×
  faster (7.6 vs 13.2 ms). With radial data (80 % kept) the same path is not faster.
* **The default 0.05 is coarser for sparse sampling.** The operator error is about 10⁻³, against
  4·10⁻⁶ for radial data at the same threshold. In the image domain the unit-weight kernel is the
  trajectory's point-spread function; in k-space it is each sample's Dirichlet kernel, whose tails
  decay slowly. With two arms, the tails between the arms carry a much larger share of the
  kernel's energy, and the threshold removes exactly those tails.
  * Threshold 0.01 restores 10⁻⁴ operator error and a 7.8·10⁻⁴ CG difference while keeping
    the kernel 2–3× smaller.
  * 0.003 matches the radial accuracy but keeps 83 % of the grid.
  * Whether a 10⁻² CG difference matters depends on the application. Here it is two orders of
    magnitude below the reconstruction's own error against the truth.

An alternative mask, keeping only grid locations within r cells of a sample, is compared with
this one in [neighbour_mask.md](neighbour_mask.md).

## Recommendations

* **Use the Toeplitz operator for iterative reconstruction.** It wins at every size tested, even
  counting `calc_toep`. Keep the NUFFT operators (`AhA_cufinufft`, `AhANufftPlan`) as an independent
  reference.
* **Use `is_masked = true` when memory is tight.** In 3D the kernel is the largest allocation. The
  threshold mask at 0.05 cuts it by about a third at negligible accuracy cost. The neighbourhood
  mask with r = 2, now the default, cuts it by almost half at the same accuracy.
* **Sparse trajectories (few spiral arms, few shots): pick the threshold for the accuracy you
  need.** 0.05 gives a 10–15× smaller kernel at about 10⁻³ operator error; 0.01 gives 2–3× at
  10⁻⁴. Also pass a small `mem_fraction` so that `calc_toep`'s transient memory stays close to the
  masked kernel's size.
* **A faster masked execute path (not implemented).** Replace gather / memset / scatter with one
  pass over the padded grid that uses a precomputed `int32` index map (−1 outside the mask). Masked
  locations would be multiplied in place and the rest zeroed. Kernel reads would then shrink with
  the kept fraction (78 complex values per location at L = 12, versus 24 for the data). That should
  turn the memory saving into a speed-up of the block-diagonal stage.
* **When using the NUFFT path, tune `gpu_method`.** Pass `type1_opts` / `type2_opts`; on these
  problems (1, 2) or (1, 1) was best except for 3D on the L40S, where (2, 2) won.

## Reproducing

```bash
# environment (both machines): conda env "mr_aha"
pip install numpy scipy h5py cupy-cuda12x cufinufft \
            "nvidia-cuda-nvrtc-cu12==12.9.*" "nvidia-cuda-cccl-cu12==12.9.*" \
            "nvidia-cuda-runtime-cu12==12.9.*"
# libcudart.so.12 must be on LD_LIBRARY_PATH for the cufinufft wheel
# (module load cuda12.9/toolkit on the HPC).

cmake -S . -B build -DMR_AHA_BUILD_PYTHON=ON -DMR_AHA_BUILD_MATLAB=ON \
      -DMR_AHA_BUILD_CUFINUFFT_MATLAB=ON && cmake --build build -j

python benchmarks/make_kooshball.py            # synthetic 3D data (~1.2 GB)
python benchmarks/mask_threshold.py            # threshold experiment
pip install mri-nufft                          # only for make_spiral.py
python benchmarks/make_spiral.py -o benchmarks/data/spiral_trajectory.mat
python benchmarks/spiral_mask.py               # sparse-trajectory mask experiment
python benchmarks/bench_python.py              # Toeplitz + NUFFT (Python)
matlab -batch "addpath('build/finufft/matlab'); cd benchmarks; bench_matlab"
python benchmarks/compare_cg.py                # -> results/<host>/summary.md
```

On the HPC, `sbatch benchmarks/hpc/run_bench_hpc.sh` (from the repository root) runs all of the
above on one L40S. It builds the library with the CUDA 12.9 module and runs MATLAB inside the
`matlab_biri_v1.6.sif` image with its bundled cufinufft MEX.

Notes:

* CuPy 14 with the system NVRTC 12.6 crashes while compiling kernels, so the env pins the pip
  NVRTC/CCCL 12.9.
* The existing `LDR` env mixes CUDA 13 wheels and segfaults for the same reason.
* `stage2.md` names the image-to-k-space step "type 1". In cufinufft's convention that direction
  is type 2 (uniform → non-uniform, sign −1) and the adjoint is type 1 (sign +1). This is the
  pairing that matches the Toeplitz kernel and the CPU-finufft reference of the 2D data, where the
  operator error is 3.6·10⁻⁶.
