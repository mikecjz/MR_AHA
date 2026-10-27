# mr_aha (Python interface)

Python bindings for **MR_AHA**, a CUDA library that evaluates the forward
normal operator `AᴴA` for non-Cartesian MRI with low-rank
(subspace / MR-Multitasking) modelling, using a Toeplitz embedding.

## Install

Build the shared library first, from the repository root:

```bash
cmake -S . -B build -DMR_AHA_BUILD_PYTHON=ON
cmake --build build -j
```

Then install the package (editable is convenient during development):

```bash
pip install -e python
pip install cupy-cuda12x   # match your CUDA toolkit
```

The binding locates `libmr_aha_c.so` automatically in `build/lib`; set
`MR_AHA_LIBRARY` to an explicit path if it lives elsewhere.

## Array layout

The library documents arrays in **column-major** (MATLAB) order. NumPy and CuPy
are row-major, so every shape is reversed here — the memory layout is identical
and no data is ever moved.

| column-major (docs)          | row-major (this package)   |
| ---------------------------- | -------------------------- |
| `x[Nx, Ny, Nz, L]`           | `x[L, Nz, Ny, Nx]`         |
| `SEs[Nx, Ny, Nz, Nc]`        | `SEs[Nc, Nz, Ny, Nx]`      |
| `phi2_toep[Mx, My, Mz, T]`   | `phi2_toep[T, Mz, My, Mx]` |
| `phi2[Ntrajs, L, L]`         | `phi2[L, L, Ntrajs]`       |
| `phi2[Ntrajs, Nkz, L, L]`    | `phi2[L, L, Nkz, Ntrajs]`  |
| `traj[Ntrajs, Nkx, ndim]`    | `traj[ndim, Nkx, Ntrajs]`  |
| `mask_kernel[Mx, My, Mz]`    | `mask_kernel[Mz, My, Mx]`  |

## Usage

```python
import cupy as cp
import mr_aha

cfg = mr_aha.make_config("stack", (256, 256, 64), L=16, n_coils=8)

Om = mr_aha.calc_toep(cfg, phi2, traj)      # one-off pre-computation

with mr_aha.AhAPlan(cfg) as plan:           # reuse inside the CG loop
    for _ in range(n_iter):
        y = plan.apply(x, SEs, Om)
```

Masked operators keep `phi2_toep` only at selected upsampled k-space samples.
`calc_mask` derives them from the trajectory. By default it keeps every grid location within
`mask_radius=2` cells of a sample. With `mask_method="threshold"` it thresholds the unit-weight
Toeplitz kernel at `mask_threshold` (absolute, in samples, default 0.05) instead. Per call,
`calc_mask(cfg, traj, radius=...)` or `threshold=...` overrides the configured method:

```python
cfg  = mr_aha.make_config("stack", (256, 256, 64), L=16, n_coils=8, is_masked=True)
mask = mr_aha.calc_mask(cfg, traj)                  # or your own indices
Om   = mr_aha.calc_toep(cfg, phi2, traj, mask=mask) # fills cfg.num_masked
y    = mr_aha.aha_toeplitz(cfg, x, SEs, Om, mask=mask)
```

`mask` holds **0-based** linear indices into the centred upsampled grid.

## NUFFT reference

`mr_aha.AhANufftPlan` evaluates the same operator with type-2 / type-1
cufinufft transforms instead of the Toeplitz embedding (pure CuPy + the
`cufinufft` package, no MR_AHA C code). It needs `pip install cufinufft`, and
the cufinufft wheel expects `libcudart.so.12` on the library path (load your
CUDA module or install `nvidia-cuda-runtime-cu12`).

```python
with mr_aha.AhANufftPlan(cfg, traj, type1_opts={"gpu_method": 1},
                         type2_opts={"gpu_method": 2}) as ref:
    y_ref = ref.apply(x, SEs, phi2)    # agrees with aha_toeplitz to ~1e-6
```
