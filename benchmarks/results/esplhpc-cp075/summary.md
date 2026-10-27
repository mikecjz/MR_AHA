# MR_AHA benchmark results: `esplhpc-cp075`

## 2d: 2D radial 160x160, 16 coils, L=6, 377x160 samples

GPU: NVIDIA L40S (44.4 GB), CuPy 14.2.0, cufinufft (Python) 2.5.1. MATLAB 25.2.0.3042426 (R2025b) Update 1.

### One A^H A application

| operator | pre-compute | plan | A^H A (median) | speed-up vs NUFFT (Py, tuned) |
|---|---|---|---|---|
| NUFFT Python, default gpu_method | – | 179.4 ms | 25.4 ms | 0.5x |
| NUFFT Python, tuned gpu_method (1,2) | – | 134.1 ms | 13.4 ms | 1.0x |
| NUFFT MATLAB, default gpu_method | – | 1.14 s | 29.6 ms | 0.5x |
| NUFFT MATLAB, tuned gpu_method (1,1) | – | 142.3 ms | 15.3 ms | 0.9x |
| Toeplitz (kernel 16 MB, kept 100.0%) | 168.8 ms | 65.5 ms | 0.7 ms | 19.9x |
| Toeplitz masked (kernel 14 MB, kept 83.1%) | 140.9 ms mask + 143.7 ms | 64.7 ms | 0.8 ms | 17.1x |

### CG, 20 iterations from zero

| operator | total time | final rel. residual | rel. diff vs NUFFT (Py) CG | rel. error vs truth |
|---|---|---|---|---|
| NUFFT (Python, cufinufft) | 307.8 ms | 1.454e-02 | – | 8.355e-02 |
| NUFFT (MATLAB, cufinufft) | 508.5 ms | 1.454e-02 | 2.79e-06 | 8.355e-02 |
| Toeplitz (Python, MR_AHA) | 18.2 ms | 1.453e-02 | 3.38e-06 | 8.355e-02 |
| Toeplitz masked (Python, MR_AHA) | 19.9 ms | 1.453e-02 | 1.85e-05 | 8.355e-02 |

Pairwise relative differences between CG solutions (`||a-b|| / ||b||`):

| a \ b | nufft | toeplitz | toeplitz_masked | matlab |
|---|---|---|---|---|
| nufft | – | 3.38e-06 | 1.85e-05 | 2.79e-06 |
| toeplitz | 3.38e-06 | – | 1.83e-05 | 3.39e-06 |
| toeplitz_masked | 1.85e-05 | 1.83e-05 | – | 1.85e-05 |
| matlab | 2.79e-06 | 3.39e-06 | 1.85e-05 | – |

### Mask threshold sweep (masked Toeplitz vs NUFFT)

| threshold | kept | err (image input) | err (white noise) | CG diff | calc_mask | A^H A |
|---|---|---|---|---|---|---|
| 1 | 75.8% | 7.81e-04 | 4.34e-02 | 5.03e-03 | 289.8 ms | 0.8 ms |
| 0.3 | 79.7% | 3.05e-05 | 3.14e-03 | 1.48e-04 | 137.6 ms | 0.8 ms |
| 0.1 | 81.5% | 9.34e-06 | 1.30e-03 | 4.09e-05 | 135.8 ms | 0.8 ms |
| 0.05 | 83.1% | 4.31e-06 | 8.44e-04 | 1.84e-05 | 136.0 ms | 0.8 ms |
| 0.03 | 84.6% | 2.92e-06 | 6.54e-04 | 1.20e-05 | 135.9 ms | 0.8 ms |
| 0.01 | 89.0% | 1.65e-06 | 2.84e-04 | 5.34e-06 | 136.0 ms | 0.8 ms |
| 0.003 | 92.1% | 1.34e-06 | 1.87e-04 | 4.09e-06 | 138.6 ms | 0.8 ms |
| 0 | 100.0% | 1.03e-06 | 1.21e-06 | 3.37e-06 | 137.1 ms | 0.8 ms |
| unmasked | 100.0% | 1.05e-06 | 1.21e-06 | 3.35e-06 | 0.0 ms | 0.7 ms |

## stack: stack-of-stars 320x320x56, 12 coils, L=12, 280x320 samples x 56 kz

GPU: NVIDIA L40S (44.4 GB), CuPy 14.2.0, cufinufft (Python) 2.5.1. MATLAB 25.2.0.3042426 (R2025b) Update 1.

### One A^H A application

| operator | pre-compute | plan | A^H A (median) | speed-up vs NUFFT (Py, tuned) |
|---|---|---|---|---|
| NUFFT Python, default gpu_method | – | 155.0 ms | 2.88 s | 0.6x |
| NUFFT Python, tuned gpu_method (1,2) | – | 154.6 ms | 1.84 s | 1.0x |
| NUFFT MATLAB, default gpu_method | – | 378.7 ms | 2.55 s | 0.7x |
| NUFFT MATLAB, tuned gpu_method (1,2) | – | 145.9 ms | 1.71 s | 1.1x |
| Toeplitz (kernel 13.3 GB, kept 100.0%) | 2.93 s | 99.7 ms | 940.8 ms | 2.0x |
| Toeplitz masked (kernel 10.8 GB, kept 81.2%) | 208.2 ms mask + 2.53 s | 98.0 ms | 999.8 ms | 1.8x |

### CG, 20 iterations from zero

| operator | total time | final rel. residual | rel. diff vs NUFFT (Py) CG | rel. error vs truth |
|---|---|---|---|---|
| NUFFT (Python, cufinufft) | 37.11 s | 1.768e-02 | – | – |
| NUFFT (MATLAB, cufinufft) | 34.43 s | 1.768e-02 | 3.02e-06 | – |
| Toeplitz (Python, MR_AHA) | 19.13 s | 1.768e-02 | 3.58e-06 | – |
| Toeplitz masked (Python, MR_AHA) | 20.30 s | 1.768e-02 | 2.73e-05 | – |

Pairwise relative differences between CG solutions (`||a-b|| / ||b||`):

| a \ b | nufft | toeplitz | toeplitz_masked | matlab |
|---|---|---|---|---|
| nufft | – | 3.58e-06 | 2.73e-05 | 3.02e-06 |
| toeplitz | 3.58e-06 | – | 2.72e-05 | 3.65e-06 |
| toeplitz_masked | 2.73e-05 | 2.72e-05 | – | 2.73e-05 |
| matlab | 3.02e-06 | 3.65e-06 | 2.73e-05 | – |

### Mask threshold sweep (masked Toeplitz vs NUFFT)

| threshold | kept | err (image input) | err (white noise) | CG diff | calc_mask | A^H A |
|---|---|---|---|---|---|---|
| 1 | 27.1% | 2.77e-04 | 4.71e-01 | 3.18e-02 | 221.3 ms | 817.8 ms |
| 0.3 | 78.9% | 1.77e-06 | 8.42e-03 | 9.99e-05 | 205.4 ms | 1.02 s |
| 0.1 | 79.8% | 1.33e-06 | 5.28e-03 | 4.78e-05 | 206.1 ms | 1.02 s |
| 0.05 | 81.2% | 1.25e-06 | 3.55e-03 | 2.73e-05 | 205.1 ms | 1.00 s |
| 0.03 | 83.2% | 1.23e-06 | 2.94e-03 | 2.05e-05 | 204.2 ms | 1.03 s |
| 0.01 | 93.6% | 1.20e-06 | 1.81e-03 | 1.11e-05 | 205.2 ms | 1.08 s |
| 0.003 | 99.5% | 1.14e-06 | 4.12e-04 | 4.22e-06 | 204.0 ms | 1.07 s |
| 0 | 100.0% | 1.17e-06 | 1.51e-06 | 3.58e-06 | 205.8 ms | 1.07 s |
| unmasked | 100.0% | 1.16e-06 | 1.51e-06 | 3.58e-06 | 0.0 ms | 940.7 ms |

## 3d: 3D kooshball 160^3, 12 coils, L=12, 15680x160 samples (synthetic)

GPU: NVIDIA L40S (44.4 GB), CuPy 14.2.0, cufinufft (Python) 2.5.1. MATLAB 25.2.0.3042426 (R2025b) Update 1.

### One A^H A application

| operator | pre-compute | plan | A^H A (median) | speed-up vs NUFFT (Py, tuned) |
|---|---|---|---|---|
| NUFFT Python, default gpu_method | – | 227.3 ms | 3.79 s | 1.0x |
| NUFFT Python, tuned gpu_method (2,2) | – | 225.8 ms | 3.79 s | 1.0x |
| NUFFT MATLAB, default gpu_method | – | 384.5 ms | 3.94 s | 1.0x |
| NUFFT MATLAB, tuned gpu_method (2,2) | – | 215.9 ms | 3.39 s | 1.1x |
| Toeplitz (kernel 19.0 GB, kept 100.0%) | 7.61 s | 106.4 ms | 1.33 s | 2.9x |
| Toeplitz masked (kernel 12.1 GB, kept 63.7%) | 362.9 ms mask + 6.72 s | 108.4 ms | 1.36 s | 2.8x |

### CG, 20 iterations from zero

| operator | total time | final rel. residual | rel. diff vs NUFFT (Py) CG | rel. error vs truth |
|---|---|---|---|---|
| NUFFT (Python, cufinufft) | 76.03 s | 3.876e-03 | – | 2.483e-01 |
| NUFFT (MATLAB, cufinufft) | 68.58 s | 3.879e-03 | 7.74e-05 | 2.483e-01 |
| Toeplitz (Python, MR_AHA) | 26.81 s | 3.824e-03 | 2.52e-04 | 2.483e-01 |
| Toeplitz masked (Python, MR_AHA) | 27.41 s | 3.821e-03 | 2.52e-04 | 2.483e-01 |

Pairwise relative differences between CG solutions (`||a-b|| / ||b||`):

| a \ b | nufft | toeplitz | toeplitz_masked | matlab |
|---|---|---|---|---|
| nufft | – | 2.52e-04 | 2.52e-04 | 7.74e-05 |
| toeplitz | 2.52e-04 | – | 2.29e-05 | 2.48e-04 |
| toeplitz_masked | 2.52e-04 | 2.29e-05 | – | 2.51e-04 |
| matlab | 7.74e-05 | 2.48e-04 | 2.51e-04 | – |

### Mask threshold sweep (masked Toeplitz vs NUFFT)

| threshold | kept | err (image input) | err (white noise) | CG diff | calc_mask | A^H A |
|---|---|---|---|---|---|---|
| 1 | 30.5% | 1.02e-04 | 6.96e-02 | 1.23e-03 | 384.1 ms | 1.19 s |
| 0.3 | 52.1% | 6.08e-06 | 4.05e-03 | 1.57e-04 | 370.2 ms | 1.29 s |
| 0.1 | 56.0% | 3.95e-06 | 9.98e-04 | 1.83e-04 | 365.3 ms | 1.31 s |
| 0.05 | 63.7% | 3.87e-06 | 7.31e-04 | 1.65e-04 | 366.8 ms | 1.36 s |
| 0.03 | 75.6% | 3.84e-06 | 5.01e-04 | 1.74e-04 | 365.0 ms | 1.44 s |
| 0.01 | 91.7% | 3.81e-06 | 8.44e-05 | 1.91e-04 | 366.1 ms | 1.52 s |
| 0.003 | 94.5% | 3.81e-06 | 5.82e-06 | 1.97e-04 | 484.3 ms | 1.54 s |
| 0 | 100.0% | 3.81e-06 | 2.18e-06 | 1.80e-04 | 364.8 ms | 1.51 s |
| unmasked | 100.0% | 3.81e-06 | 2.18e-06 | 1.54e-04 | 0.0 ms | 1.33 s |
