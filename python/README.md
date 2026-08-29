# mt_cuda (Python interface)

Python bindings for **MT_CUDA**, a CUDA library that evaluates the forward
normal operator `AᴴA` for non-Cartesian MRI with low-rank
(subspace / MR-Multitasking) modelling, using a Toeplitz embedding.

## Install

Build the shared library first, from the repository root:

```bash
cmake -S . -B build -DMT_CUDA_BUILD_PYTHON=ON
cmake --build build -j
```

Then install the package (editable is convenient during development):

```bash
pip install -e python
pip install cupy-cuda12x   # match your CUDA toolkit
```

The binding locates `libmt_cuda_c.so` automatically in `build/lib`; set
`MT_CUDA_LIBRARY` to an explicit path if it lives elsewhere.

## Array layout

The library documents arrays in **column-major** (MATLAB) order. NumPy and CuPy
are row-major, so every shape is reversed here — the memory layout is identical
and no data is ever moved.

| column-major (docs)          | row-major (this package)   |
| ---------------------------- | -------------------------- |
| `x[Nx, Ny, Nz, L]`           | `x[L, Nz, Ny, Nx]`         |
| `SEs[Nx, Ny, Nz, Nc]`        | `SEs[Nc, Nz, Ny, Nx]`      |
| `phi2_toep[Mx, My, Mz, T]`   | `phi2_toep[T, Mz, My, Mx]` |
| `phi2[Ntrajs, L*L]`          | `phi2[L*L, Ntrajs]`        |
| `traj[Ntrajs, Nkx, ndim]`    | `traj[ndim, Nkx, Ntrajs]`  |

## Usage

```python
import cupy as cp
import mt_cuda

cfg = mt_cuda.make_config("stack", (256, 256, 64), L=16, n_coils=8)

Om = mt_cuda.calc_toep(cfg, phi2, traj)      # one-off pre-computation

with mt_cuda.AhAPlan(cfg) as plan:           # reuse inside the CG loop
    for _ in range(n_iter):
        y = plan.apply(x, SEs, Om)
```

Masked operators keep `phi2_toep` only at selected upsampled k-space samples:

```python
cfg = mt_cuda.make_config("stack", (256, 256, 64), L=16, n_coils=8,
                          is_masked=True, num_masked=mask.size)
Om = mt_cuda.calc_toep(cfg, phi2, traj, mask=mask)
y  = mt_cuda.aha_toeplitz(cfg, x, SEs, Om, mask=mask)
```

`mask` holds **0-based** linear indices into the centred upsampled grid.
