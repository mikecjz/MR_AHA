# MR_AHA

GPU evaluation of the forward normal operator $A^H A$ for **non-Cartesian MRI with low-rank
(subspace / MR-Multitasking) modelling**, via a Toeplitz embedding.

The numerics live in a standalone CUDA/C++ library with a pure C surface. MATLAB, Python and plain
C/C++ all call the same code:

```
include/mr_aha/mr_aha.h   public C API  (extern "C")
src/core/                 the C++/CUDA implementation  ->  libmr_aha
matlab/                   MEX gateways + .m wrappers
python/                   ctypes binding + CuPy-friendly API
examples/                 standalone C++/CUDA example
legacy/                   the previous MATLAB-only implementation (reference only)
```

Everything runs on the GPU. All array arguments are device pointers (`gpuArray` in MATLAB,
`cupy.ndarray` in Python); the library never copies bulk data through the host.

---

## Contents

1. [The two operations](#the-two-operations)
2. [Conventions](#conventions)
3. [Building](#building)
4. [MATLAB](#matlab)
5. [Python](#python)
6. [C / C++](#c--c)
7. [Why Toeplitz and how does it work?](#why-toeplitz-and-how-does-it-work)
8. [Implementation notes](#implementation-notes)

---

## The two operations

### `calc_toep` — one-off pre-computation

```
phi2_toep = calc_toep(phi2, traj, config)
```

Turns the non-Cartesian Gram blocks $\Phi^H\Phi$, one $L \times L$ matrix per trajectory, into the
Cartesian block-diagonal multiplier used by the operator. Internally: pack each block into a
triangle, type-1 cufinufft onto the upsampled grid, centred inverse FFT, optional mask gather.

### `AhA_Toeplitz` — the hot loop

```
AhAx = AhA_Toeplitz(x, SEs, phi2_toep, config)
AhAx = AhA_Toeplitz(x, SEs, phi2_toep, config, mask)
```

For every coil: multiply by the sensitivity map, zero-pad onto the upsampled grid, FFT, multiply by
the Hermitian $L \times L$ block at each k-space location, inverse FFT, crop, multiply by the
adjoint sensitivity map, accumulate. No NUFFT is evaluated here, which is the whole point of the
Toeplitz embedding.

---

## Conventions

**Column-major.** Every array shape in this document and in the C header is column-major (MATLAB /
Fortran). Row-major callers (NumPy, CuPy) reverse the dimension order — the memory layout is
identical, so nothing ever has to be permuted.

**Centred arrays.** Image-space and k-space arrays have their origin in the *middle* of the array,
at index `floor(dim/2)`. The library performs the required shifts internally by index remapping, so
the convention is exact for even and odd matrix sizes alike.

**Single precision.** All complex data is `complexFloat` (`single` gpuArray, `complex64` CuPy).

### Array shapes

| Array | 2D | 3D and stack |
| ----- | -- | ------------ |
| `x` | `[Nx, Ny, L]` | `[Nx, Ny, Nz, L]` |
| `SEs` | `[Nx, Ny, Ncoils]` | `[Nx, Ny, Nz, Ncoils]` |
| `phi2_toep` (unmasked) | `[Mx, My, (L²+L)/2]` | `[Mx, My, Mz, (L²+L)/2]` |
| `phi2_toep` (masked) | `[num_masked, (L²+L)/2]` | `[num_masked, (L²+L)/2]` |
| `phi2` | `[Ntrajs, L*L]` | `[Ntrajs, L*L]`, or `[Ntrajs, Nkz, L*L]` for stack |
| `traj` | `[Ntrajs, Nkx, 2]` | `[Ntrajs, Nkx, 3]`, or `[Ntrajs, Nkx, 2]` for stack |
| `AhAx` | same as `x` | same as `x` |

There is one sensitivity map per coil; it is shared by every rank.

`traj` holds coordinates in **radians per unit of the un-padded image grid**, i.e. the usual
$[-\pi, \pi)$ range.

### `config`

| Field | Meaning |
| ----- | ------- |
| `traj_type` | `'2D'`, `'3D'` or `'stack'` |
| `n_dim` | `[Nx Ny]` or `[Nx Ny Nz]` |
| `L` | number of ranks |
| `triangulation` | `'upper'` or `'lower'` — which half of each Hermitian block is stored |
| `phi2_dim` | upsampled Toeplitz size `[Mx My]` or `[Mx My Mz]`, typically `2*n_dim` |
| `is_masked` | whether `phi2_toep` is stored only at selected upsampled samples |

Optional: `n_coils`, `num_masked`, `eps`, `mem_fraction`, `gpu_device_id`, `verbose`. The MATLAB
helper `mr_default_config` and the Python `mr_aha.make_config` fill in sensible defaults.

### Triangular packing

Only $(L^2+L)/2$ of the $L^2$ entries of each block are stored; the rest follows from
$A_{ij} = \overline{A_{ji}}$. Both packings are the column-major packing of the corresponding MATLAB
triangle, i.e. exactly `A(tril(true(L)))` / `A(triu(true(L)))`.

### Masking

`mask` holds linear indices into the *centred* upsampled grid `phi2_dim`, 1-based in MATLAB and
0-based in Python and C. Samples outside the mask contribute nothing. This is what makes a large 3D
operator fit in GPU memory when the upsampled k-space support is sparse.

---

## Building

Requirements: CUDA toolkit (11.x or 12.x), CMake ≥ 3.18, a C++17 compiler, and the bundled FINUFFT
submodule. Unlike the previous version, **stock upstream FINUFFT is enough** — no fork is needed.

```bash
git submodule update --init --recursive

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=75           \
      -DMR_AHA_BUILD_PYTHON=ON               \
      -DMR_AHA_BUILD_MATLAB=ON               \
      -DMR_AHA_BUILD_EXAMPLES=ON
cmake --build build -j
```

| Option | Default | Produces |
| ------ | ------- | -------- |
| — | | `build/libmr_aha.a` — the standalone library |
| `MR_AHA_BUILD_PYTHON` | `ON` | `build/lib/libmr_aha_c.so` — for the ctypes binding |
| `MR_AHA_BUILD_MATLAB` | `OFF` | `build/matlab/*.mexa64` — needs `MATLAB_ROOT` and the Parallel Computing Toolbox |
| `MR_AHA_BUILD_EXAMPLES` | `OFF` | `build/mr_example_aha` |

Set `MATLAB_ROOT` (or `-DMatlab_ROOT_DIR=...`) to your `matlabroot` before configuring with
`MR_AHA_BUILD_MATLAB=ON`.

---

## MATLAB

Put both `matlab/` and `build/matlab/` on the MATLAB path.

```matlab
config = mr_default_config('stack', [320 320 56], 16);

Om = calc_toep(phi2, traj, config);          % once
for it = 1:n_iter
    AhAx = AhA_Toeplitz(x, SEs, Om, config); % inside the CG loop
end
```

Masked operator:

```matlab
config = mr_default_config('stack', [320 320 56], 16, 'is_masked', true);
Om     = calc_toep(phi2, traj, config, mask);   % mask: 1-based linear indices
AhAx   = AhA_Toeplitz(x, SEs, Om, config, mask);
```

The `.m` wrappers move host arrays to the GPU, cast to complex `single`, derive `n_coils` from
`size(SEs)`, and convert the 1-based mask to the 0-based `int64` mask the gateway expects. Call
`AhA_Toeplitz_mex` / `calc_toep_mex` directly only if you want to skip that.

> ⓘ MATLAB must have initialised its GPU before the first MEX call. Running `gpuDevice(1)`, or
> simply creating any `gpuArray`, is enough; the wrappers do this implicitly by moving their inputs.

---

## Python

```bash
cmake -S . -B build -DMR_AHA_BUILD_PYTHON=ON && cmake --build build -j
pip install -e python
pip install cupy-cuda12x        # match your CUDA toolkit
```

```python
import mr_aha

cfg = mr_aha.make_config("stack", (320, 320, 56), L=16, n_coils=12)

Om = mr_aha.calc_toep(cfg, phi2, traj)
with mr_aha.AhAPlan(cfg) as plan:
    for _ in range(n_iter):
        y = plan.apply(x, SEs, Om)
```

Shapes are the reverse of the table above, because CuPy is row-major: `x` is `(L, Nz, Ny, Nx)`,
`SEs` is `(Ncoils, Nz, Ny, Nx)`, `traj` is `(ndim, Nkx, Ntrajs)`, and so on. See
[`python/README.md`](python/README.md).

---

## C / C++

```c
#include "mr_aha/mr_aha.h"

mr_config cfg;
mr_config_defaults(&cfg, MR_TRAJ_STACK);
cfg.n_dim[0] = 320; cfg.n_dim[1] = 320; cfg.n_dim[2] = 56;
cfg.phi2_dim[0] = 640; cfg.phi2_dim[1] = 640; cfg.phi2_dim[2] = 56;
cfg.L = 16; cfg.n_coils = 12;

mr_calc_toep(&cfg, d_phi2, d_traj, Ntrajs, Nkx, NULL, d_toep);

mr_aha_plan plan;
mr_aha_plan_create(&cfg, &plan);
for (int it = 0; it < n_iter; ++it)
    mr_aha_plan_execute(plan, d_x, d_SEs, d_toep, NULL, d_AhAx);
mr_aha_plan_destroy(plan);
```

Every entry point returns `MR_SUCCESS` (0) or a negative `mr_status`; `mr_status_string` gives the
category and `mr_last_error` the detailed, thread-local diagnostic. `mr_aha_toeplitz` is the
one-shot convenience form. Link against `mr_aha`, or `target_link_libraries(your_target mr_aha)`
after `add_subdirectory(MR_AHA)`.

[`examples/example_aha.cu`](examples/example_aha.cu) is a complete, self-contained program that
builds a synthetic stack-of-stars problem and checks the operator's Hermitian symmetry.

---

## Why Toeplitz and How Does It Work?

The AhA operator for MRI is [Toeplitz](https://en.wikipedia.org/wiki/Toeplitz_matrix) under
arbitrary non-Cartesian trajectories ([Baron et al., MRM
2018](https://onlinelibrary.wiley.com/doi/10.1002/mrm.26928)). A Toeplitz matrix can be padded to 2X
on its original dimensions to form a [Circulant
matrix](https://en.wikipedia.org/wiki/Circulant_matrix), and be [calculated via
FFTs](https://math.mit.edu/icg/resources/teaching/18.085-spring2015/toeplitz.pdf), requiring only
$O(N \log N)$ operations instead of $O(N^2)$ for general matrix-vector multiplications. In other
words, the Toeplitz operator can be diagonalized by FFTs.

$$Tx \implies F^H\Lambda F\hat{x}, $$

>* $T \in \mathbb{C}^{N \times N}$ is a symmetric Toeplitz operator
>* $x \in \mathbb{C}^{N}$ is a vector
>* $F \in \mathbb{C}^{2N \times 2N}$ is the 2D DFT operator
>* $\Lambda \in \mathbb{C}^{2N \times 2N}$ is a diagonal matrix
>* $\hat{x} \in \mathbb{C}^{2N}$ is the zero-padded vector of $x$

The diagonal elements of $\Lambda$ can be derived based on the non-Cartesian coordinates but we
usually use NUFFT packages to estimate them (see [Baron et al., MRM
2018](https://onlinelibrary.wiley.com/doi/10.1002/mrm.26928)).

Under MR-Multitasking, the AhA operator is block-Toeplitz, which can be diagonalized similarly, with
the "diagonal" aspect being block-diagonal, a.k.a. the gridded $\Phi\Phi^H$ operator — this is
`phi2_toep`.

The challenge for 3D stack-of-stars AhA is that there is usually not enough GPU memory to hold the
gridded $\Phi\Phi^H$ operator, which has a size of $2N_x \times 2N_y \times N_z \times L \times L$.
For a $320 \times 320 \times 56$ matrix with $L = 16$, that is about 26 GB in single precision and a
24 GB GPU cannot hold it.

Luckily for us, each individual $L \times L$ matrix of the gridded $\Phi\Phi^H$ operator is
[Hermitian](https://en.wikipedia.org/wiki/Hermitian_matrix), so the lower triangle is the complex
conjugate of the upper one. Storing a single triangle — which is what `config.triangulation`
selects — halves that to about 13 GB and makes the 3D operator feasible. When even that is too
much, `config.is_masked` keeps only the upsampled k-space samples that actually carry signal.

### A note on Hermitian symmetry and the old "deconvolve" hack

The previous implementation observed that the gridded $\Phi\Phi^H$ came out slightly non-Hermitian,
and blamed the deconvolution (transfer-function division) step of the type-1 NUFFT; it worked around
this with a [FINUFFT fork](https://github.com/mikecjz/finufft/tree/toeplitz) exposing the spreading
and deconvolution stages separately.

That workaround is gone. Hermitian symmetry is now enforced *structurally*: only one triangle is
ever computed and stored, and the mirrored half is produced by conjugation inside the
block-diagonal kernel. The gridded operator is therefore Hermitian by construction, to machine
precision, whatever the NUFFT does internally — which also means unmodified upstream FINUFFT works.
[`examples/example_aha.cu`](examples/example_aha.cu) checks
$\langle y, A^HAx\rangle = \langle A^HAy, x\rangle$ end to end.

---

## Implementation notes

**No scaling anywhere in the hot loop.** `calc_toep` evaluates the type-1 NUFFT with sign $-1$ and
then a *normalised* centred inverse FFT. Since the point-spread functions of the two signs are
spatial reflections of one another, $p^{-}[d] = p^{+}[-d]$, this equals
$\mathrm{DFT}(p^{+})/\prod M$ — exactly the multiplier needed between the two unnormalised cuFFTs of
`AhA_Toeplitz`. The $1/\prod M$ is therefore baked into `phi2_toep` and no scaling survives in the
iteration.

**Centring by index remap, not phase modulation.** The pad and crop kernels write and read at
$(x_d - \lfloor N_d/2 \rfloor) \bmod M_d$, and the block-diagonal kernel converts between natural
and centred order on the fly. This makes the pad/crop pair an exact adjoint pair (so the composed
operator is Hermitian whenever the multiplier is), removes two full array passes per FFT, and is
exact for odd matrix sizes, which a $(-1)^{x+y}$ modulation is not.

**Stack-of-stars.** `kz` is already Cartesian, so `Mz = Nz`: `calc_toep` runs a 2D type-1 per kz
encode with no transform and no shift along z, while `AhA_Toeplitz` still runs a full 3D FFT because
z participates in the image-domain transform. `phi2` therefore carries an extra `Nkz` axis.

**Memory.** `calc_toep` chunks its work over the $(L^2+L)/2$ triangular components and sizes the
chunk against `cudaMemGetInfo`, honouring `config.mem_fraction` (default: half the free memory); it
also bounds cufinufft's internal fine grid through `gpu_maxbatchsize`. An `mr_aha_plan` holds one
`[Mx, My, Mz, L]` working buffer for the lifetime of the plan, so the iteration allocates nothing.

**Re-entrancy.** No `__constant__` or other shared device state: kernel geometry travels by value.
Several plans, CUDA streams and host threads can be in flight at once, and `mr_last_error` is
thread-local. `config.stream` selects the stream; `config.gpu_device_id = -1` (the default for
MATLAB and Python) keeps whichever device the caller has selected.

---

## Development and debugging in Visual Studio Code

Set `includePath` in [.vscode/c_cpp_properties.json](.vscode/c_cpp_properties.json) to include
`include/`, `src/core/`, `finufft/include/` and the CUDA and MATLAB headers for IntelliSense.

To debug the MEX gateways: build with `-DCMAKE_BUILD_TYPE=Debug`, point `program` in
[.vscode/launch.json](.vscode/launch.json) at your MATLAB executable, and set breakpoints in
`matlab/*.cu` or `src/core/*.cu`. The [MATLAB GPU CUDA and MEX Programming
Manual](https://www.mathworks.com/help/parallel-computing/gpu-cuda-and-mex-programming.html) is the
reference for the MathWorks side.

## License

[PolyForm Noncommercial 1.0.0](LICENSE).
