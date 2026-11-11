# mr_aha (Python interface)

Python bindings for **MR_AHA**, a GPU library (CUDA or Apple-silicon Metal) that evaluates the
forward normal operator `AᴴA` for non-Cartesian MRI with low-rank (subspace / MR-Multitasking)
modelling, using a Toeplitz embedding.

Documentation: <https://mikecjz.github.io/MR_AHA_docs/>
(Python guide: <https://mikecjz.github.io/MR_AHA_docs/getting-started/python.html>).

## Install

Build the shared library first, from the repository root:

```bash
cmake -S . -B build -DMR_AHA_BUILD_PYTHON=ON
cmake --build build -j
```

Then install the package:

```bash
pip install -e python
pip install cupy-cuda12x   # CUDA: match your CUDA toolkit
# Apple silicon instead:
pip install numpy "mlx==0.31.2" -e third_party/mlx-nufft
```

The binding finds `libmr_aha_c.so` / `libmr_aha_c.dylib` in `build/lib`; set `MR_AHA_LIBRARY` to
an explicit path if it lives elsewhere.

## Usage

```python
import mr_aha

cfg = mr_aha.make_config("stack", (320, 320, 56), L=16, n_coils=12)

with mr_aha.create_op(cfg, SEs, phi2, traj) as op:   # Toeplitz kernel computed here
    for _ in range(n_iter):
        AhAx = op(x)
```

Arrays are CuPy arrays on the CUDA backend and NumPy arrays on Apple silicon
(`mr_aha.backend()` reports which), in row-major order: `x` is `(L, Nz, Ny, Nx)`, `SEs` is
`(Ncoils, Nz, Ny, Nx)`, `traj` is `(ndim, Nkx, Ntrajs)`. The documentation covers the masked
operator, the NUFFT method, several GPUs and the low-level primitives.
