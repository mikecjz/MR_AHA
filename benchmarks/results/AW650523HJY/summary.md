# MR_AHA benchmark results: Mac Studio `AW650523HJY` (Metal backend)

Apple M3 Ultra, 512 GB unified memory, macOS 26.7.2. Metal backend of MR_AHA (branch `mlx`), VkFFT
fork `mikecjz/VkFFT@2f9fb8f`, MLX 0.31.2, mlx-nufft 0.3.3, Python 3.12, NumPy 2.5 (Accelerate).
Produced by `benchmarks/bench_python.py` (same problems, operators and 20-iteration CG as the CUDA
runs in `../esplhpc-cp075` and `../pact400-recon3`), with
`MR_AHA_CG_REF=data/cg_ref_cuda` holding the TITAN RTX CG solutions for the cross-backend
comparison. Raw numbers: `python_{2d,stack,3d}.json`.

On Metal the NUFFT reference operator is MLX + mlx-nufft (no `gpu_method` to tune), `calc_toep`
runs its type-1 NUFFTs with mlx-nufft and finishes in the library, and the CG vector updates run
on the CPU in NumPy with dot products accumulated in double precision (see `dot64` in
`benchmarks/common.py`: Accelerate's single-precision dot product is off by 1.2 % on the
69-million-element stack vectors).

## One A^H A application (median)

| problem | M3 Ultra, Metal | L40S, CUDA | TITAN RTX, CUDA |
|---|---|---|---|
| 2d Toeplitz | 1.7 ms | 0.7 ms | 2.3 ms |
| 2d Toeplitz masked | 2.0 ms | 0.8 ms | 2.6 ms |
| 2d NUFFT | 78.9 ms | 13.4 ms (tuned) | 19.0 ms (tuned) |
| stack Toeplitz | **0.84 s** | 0.94 s | 1.36 s |
| stack Toeplitz masked | **0.89 s** | 1.00 s | 1.38 s |
| stack NUFFT | 6.15 s | 1.84 s (tuned) | 3.64 s (tuned) |
| 3d Toeplitz | **1.17 s** | 1.33 s | out of memory (24 GB) |
| 3d Toeplitz masked | **1.10 s** | 1.36 s | 1.74 s |
| 3d NUFFT | 19.9 s | 3.79 s (tuned) | 9.45 s (tuned) |

The Toeplitz operator on the M3 Ultra is 11-19 % faster than on the L40S for the two large
problems and 1.4-1.6x faster than on the TITAN RTX. The 2D problem is too small to occupy the GPU
(per-call overhead dominates). The NUFFT reference is 3-5x slower than cufinufft: mlx-nufft's
type-2 transforms one vector per call (672 calls per coil for the stack), and it is only the
validation baseline.

## Pre-computation (`calc_toep`, unmasked / masked)

| problem | M3 Ultra (mlx-nufft + library) | L40S | TITAN RTX |
|---|---|---|---|
| 2d | 0.03 / 0.01 s | 0.17 / 0.14 s | 0.03 / 0.01 s |
| stack | 4.9 / 3.6 s | 2.9 / 2.5 s | 2.9 / 3.2 s |
| 3d | 15.2 / 14.0 s | 7.6 / 6.7 s | – / 20.0 s |

## CG, 20 iterations from zero

Total time (s):

| problem | NUFFT | Toeplitz | Toeplitz masked |
|---|---|---|---|
| 2d | 1.57 | 0.04 | 0.05 |
| stack | 127.4 | 20.1 | 21.6 |
| 3d | 398.5 | 25.4 | 24.6 |

(L40S Toeplitz: stack 19.1 s, 3d 26.8 s. On Metal the CG vector updates run on the CPU.)

Relative differences of the CG solutions:

| | 2d | stack | 3d |
|---|---|---|---|
| Metal NUFFT vs Metal Toeplitz | 5.5e-6 | 5.8e-6 | 5.7e-5 |
| Metal Toeplitz vs CUDA Toeplitz | 5.1e-6 | 8.1e-6 | – (CUDA run out of memory) |
| Metal NUFFT vs CUDA NUFFT | 6.3e-6 | 9.5e-6 | 5.4e-5 |
| Metal masked vs CUDA masked | 2.9e-4 | 1.1e-4 | 1.8e-4 |
| any solution vs truth | 8.355e-2 (CUDA 8.36e-2) | – | 0.2484 (CUDA 0.248) |

The Metal and CUDA reconstructions agree to single-precision CG accuracy; the 3D values sit in
the 1.5-2.5e-4 CG floor of that ill-conditioned problem documented in `../../README.md`. The
masked rows differ more because the CUDA runs used the earlier threshold mask (0.05) and the
Metal runs the current default neighbourhood mask (radius 2).

## Single-application operator error (`op_error.py`)

Toeplitz vs NUFFT reference, one application on the benchmark input:

| | M3 Ultra | TITAN RTX |
|---|---|---|
| 2d / masked | 1.00e-6 / 5.29e-5 | 1.02e-6 / 5.29e-5 |
| stack / masked | 0.94e-6 / 1.77e-6 | 1.18e-6 / 1.90e-6 |
| 3d / masked | 1.44e-5 / 1.46e-5 | – / 1.92e-6 |

The 3D difference comes from mlx-nufft's single-precision type-1 transform of the strongly peaked
kooshball point-spread function (1.7e-5 against double-precision FINUFFT; cufinufft 7e-6).
`calc_toep(..., nufft='finufft')` brings it to 5.0e-6 at 67 s instead of 15 s.
