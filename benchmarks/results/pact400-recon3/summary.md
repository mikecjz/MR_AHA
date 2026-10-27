# MR_AHA benchmark results: `pact400-recon3`

## 2d: 2D radial 160x160, 16 coils, L=6, 377x160 samples

GPU: NVIDIA TITAN RTX (23.5 GB), CuPy 14.2.0, cufinufft (Python) 2.5.1. MATLAB 25.2.0.3042426 (R2025b) Update 1.

### One A^H A application

| operator | pre-compute | plan | A^H A (median) | speed-up vs NUFFT (Py, tuned) |
|---|---|---|---|---|
| NUFFT Python, default gpu_method | – | 74.8 ms | 35.6 ms | 0.5x |
| NUFFT Python, tuned gpu_method (1,2) | – | 3.0 ms | 19.0 ms | 1.0x |
| NUFFT MATLAB, default gpu_method | – | 64.8 ms | 42.8 ms | 0.4x |
| NUFFT MATLAB, tuned gpu_method (1,2) | – | 3.2 ms | 23.0 ms | 0.8x |
| Toeplitz (kernel 16 MB, kept 100.0%) | 25.1 ms | 0.6 ms | 2.3 ms | 8.2x |
| Toeplitz masked (kernel 14 MB, kept 83.1%) | 4.3 ms mask + 12.7 ms | 0.6 ms | 2.6 ms | 7.4x |

### CG, 20 iterations from zero

| operator | total time | final rel. residual | rel. diff vs NUFFT (Py) CG | rel. error vs truth |
|---|---|---|---|---|
| NUFFT (Python, cufinufft) | 400.2 ms | 1.453e-02 | – | 8.355e-02 |
| NUFFT (MATLAB, cufinufft) | 495.8 ms | 1.453e-02 | 2.71e-06 | 8.355e-02 |
| Toeplitz (Python, MR_AHA) | 50.9 ms | 1.453e-02 | 3.44e-06 | 8.355e-02 |
| Toeplitz masked (Python, MR_AHA) | 55.5 ms | 1.454e-02 | 1.85e-05 | 8.355e-02 |

Pairwise relative differences between CG solutions (`||a-b|| / ||b||`):

| a \ b | nufft | toeplitz | toeplitz_masked | matlab |
|---|---|---|---|---|
| nufft | – | 3.44e-06 | 1.85e-05 | 2.71e-06 |
| toeplitz | 3.44e-06 | – | 1.83e-05 | 3.51e-06 |
| toeplitz_masked | 1.85e-05 | 1.83e-05 | – | 1.85e-05 |
| matlab | 2.71e-06 | 3.51e-06 | 1.85e-05 | – |

### Mask threshold sweep (masked Toeplitz vs NUFFT)

| threshold | kept | err (image input) | err (white noise) | CG diff | calc_mask | A^H A |
|---|---|---|---|---|---|---|
| 1 | 75.8% | 7.81e-04 | 4.34e-02 | 5.03e-03 | 10.5 ms | 2.5 ms |
| 0.3 | 79.7% | 3.05e-05 | 3.14e-03 | 1.48e-04 | 2.2 ms | 2.6 ms |
| 0.1 | 81.5% | 9.36e-06 | 1.30e-03 | 4.10e-05 | 2.1 ms | 2.6 ms |
| 0.05 | 83.1% | 4.35e-06 | 8.44e-04 | 1.85e-05 | 2.1 ms | 2.6 ms |
| 0.03 | 84.6% | 2.96e-06 | 6.54e-04 | 1.20e-05 | 2.1 ms | 2.6 ms |
| 0.01 | 89.0% | 1.73e-06 | 2.84e-04 | 5.41e-06 | 2.2 ms | 2.6 ms |
| 0.003 | 92.1% | 1.47e-06 | 1.87e-04 | 4.16e-06 | 2.1 ms | 2.7 ms |
| 0 | 100.0% | 1.13e-06 | 1.20e-06 | 3.46e-06 | 2.0 ms | 2.7 ms |
| unmasked | 100.0% | 1.13e-06 | 1.19e-06 | 3.47e-06 | 0.0 ms | 2.3 ms |

## stack: stack-of-stars 320x320x56, 12 coils, L=12, 280x320 samples x 56 kz

GPU: NVIDIA TITAN RTX (23.5 GB), CuPy 14.2.0, cufinufft (Python) 2.5.1. MATLAB 25.2.0.3042426 (R2025b) Update 1.

### One A^H A application

| operator | pre-compute | plan | A^H A (median) | speed-up vs NUFFT (Py, tuned) |
|---|---|---|---|---|
| NUFFT Python, default gpu_method | – | 162.6 ms | 5.23 s | 0.7x |
| NUFFT Python, tuned gpu_method (1,2) | – | 3.8 ms | 3.64 s | 1.0x |
| NUFFT MATLAB, default gpu_method | – | 11.1 ms | 4.36 s | 0.8x |
| NUFFT MATLAB, tuned gpu_method (1,1) | – | 4.7 ms | 3.16 s | 1.2x |
| Toeplitz (kernel 13.3 GB, kept 100.0%) | 2.87 s | 2.1 ms | 1.36 s | 2.7x |
| Toeplitz masked (kernel 10.8 GB, kept 81.2%) | 13.6 ms mask + 3.22 s | 3.1 ms | 1.38 s | 2.6x |

### CG, 20 iterations from zero

| operator | total time | final rel. residual | rel. diff vs NUFFT (Py) CG | rel. error vs truth |
|---|---|---|---|---|
| NUFFT (Python, cufinufft) | 74.07 s | 1.768e-02 | – | – |
| NUFFT (MATLAB, cufinufft) | 64.22 s | 1.768e-02 | 3.00e-06 | – |
| Toeplitz (Python, MR_AHA) | 27.62 s | 1.768e-02 | 3.57e-06 | – |
| Toeplitz masked (Python, MR_AHA) | 27.91 s | 1.768e-02 | 2.73e-05 | – |

Pairwise relative differences between CG solutions (`||a-b|| / ||b||`):

| a \ b | nufft | toeplitz | toeplitz_masked | matlab |
|---|---|---|---|---|
| nufft | – | 3.57e-06 | 2.73e-05 | 3.00e-06 |
| toeplitz | 3.57e-06 | – | 2.72e-05 | 3.64e-06 |
| toeplitz_masked | 2.73e-05 | 2.72e-05 | – | 2.73e-05 |
| matlab | 3.00e-06 | 3.64e-06 | 2.73e-05 | – |

### Mask threshold sweep (masked Toeplitz vs NUFFT)

| threshold | kept | err (image input) | err (white noise) | CG diff | calc_mask | A^H A |
|---|---|---|---|---|---|---|
| 1 | 27.1% | 2.77e-04 | 4.71e-01 | 3.18e-02 | 6.8 ms | 1.03 s |
| 0.3 | 78.9% | 1.79e-06 | 8.42e-03 | 9.99e-05 | 11.3 ms | 1.37 s |
| 0.1 | 79.8% | 1.39e-06 | 5.28e-03 | 4.78e-05 | 10.9 ms | 1.38 s |
| 0.05 | 81.2% | 1.28e-06 | 3.55e-03 | 2.73e-05 | 17.3 ms | 1.38 s |
| 0.03 | 83.2% | 1.26e-06 | 2.94e-03 | 2.05e-05 | 11.4 ms | 1.40 s |
| 0.01 | 93.6% | skipped (insufficient device memory) | | | | |
| 0.003 | 99.5% | skipped (insufficient device memory) | | | | |
| 0 | 100.0% | skipped (insufficient device memory) | | | | |
| unmasked | 100.0% | skipped (insufficient device memory) | | | | |

## 3d: 3D kooshball 160^3, 12 coils, L=12, 15680x160 samples (synthetic)

GPU: NVIDIA TITAN RTX (23.5 GB), CuPy 14.2.0, cufinufft (Python) 2.5.1. MATLAB 25.2.0.3042426 (R2025b) Update 1.

### One A^H A application

| operator | pre-compute | plan | A^H A (median) | speed-up vs NUFFT (Py, tuned) |
|---|---|---|---|---|
| NUFFT Python, default gpu_method | – | 17.5 ms | 11.85 s | 0.8x |
| NUFFT Python, tuned gpu_method (1,2) | – | 16.2 ms | 9.45 s | 1.0x |
| NUFFT MATLAB, default gpu_method | – | 16.1 ms | 10.48 s | 0.9x |
| NUFFT MATLAB, tuned gpu_method (1,2) | – | 4.5 ms | 8.97 s | 1.1x |
| Toeplitz (kept 100.0%) | skipped: needs ~28.7 GB of device memory | | | |
| Toeplitz masked (kernel 12.1 GB, kept 63.7%) | 315.7 ms mask + 19.95 s | 3.2 ms | 1.74 s | 5.4x |

### CG, 20 iterations from zero

| operator | total time | final rel. residual | rel. diff vs NUFFT (Py) CG | rel. error vs truth |
|---|---|---|---|---|
| NUFFT (Python, cufinufft) | 184.54 s | 3.799e-03 | – | 2.484e-01 |
| NUFFT (MATLAB, cufinufft) | 175.30 s | 3.797e-03 | 1.72e-05 | 2.484e-01 |
| Toeplitz masked (Python, MR_AHA) | 35.08 s | 3.817e-03 | 1.67e-04 | 2.483e-01 |

Pairwise relative differences between CG solutions (`||a-b|| / ||b||`):

| a \ b | nufft | toeplitz_masked | matlab |
|---|---|---|---|
| nufft | – | 1.67e-04 | 1.72e-05 |
| toeplitz_masked | 1.67e-04 | – | 1.71e-04 |
| matlab | 1.72e-05 | 1.71e-04 | – |

### Mask threshold sweep (masked Toeplitz vs NUFFT)

| threshold | kept | err (image input) | err (white noise) | CG diff | calc_mask | A^H A |
|---|---|---|---|---|---|---|
| 1 | 30.5% | 1.02e-04 | 6.96e-02 | 1.23e-03 | 307.5 ms | 1.46 s |
| 0.3 | 52.1% | 6.17e-06 | 4.05e-03 | 1.56e-04 | 311.4 ms | 1.64 s |
| 0.1 | 56.0% | 4.08e-06 | 9.98e-04 | 1.63e-04 | 304.6 ms | 1.66 s |
| 0.05 | 63.7% | skipped (insufficient device memory) | | | | |
| 0.03 | 75.6% | skipped (insufficient device memory) | | | | |
| 0.01 | 91.7% | skipped (insufficient device memory) | | | | |
| 0.003 | 94.5% | skipped (insufficient device memory) | | | | |
| 0 | 100.0% | skipped (insufficient device memory) | | | | |
| unmasked | 100.0% | skipped (insufficient device memory) | | | | |
