# MR_AHA

GPU evaluation of the forward normal operator $A^H A$ for **non-Cartesian MRI with low-rank
(subspace / MR-Multitasking) modelling**, via a Toeplitz embedding.

The numerics live in a standalone C++ library with a pure C surface. MATLAB, Python and plain
C/C++ all call the same code. The library is built for one GPU backend, selected with the CMake
option `MR_AHA_BACKEND`: **CUDA** (NVIDIA GPUs) or **Metal** (Apple silicon, see
[Apple silicon](#apple-silicon-metal-backend)):

```
include/mr_aha/mr_aha.h   public C API  (extern "C")
src/common/               backend-independent configuration and error handling
src/cuda/                 the CUDA backend  ->  libmr_aha
src/metal/                the Metal backend (Apple silicon)  ->  libmr_aha
matlab/                   MEX gateways + .m wrappers
python/                   ctypes binding + CuPy / NumPy API
examples/                 standalone C++/CUDA example
benchmarks/               Toeplitz vs. NUFFT speed / CG benchmarks and results
tests/                    reference checks of a build against stored CUDA results
third_party/              VkFFT (Metal FFTs, fork with a Metal fix) and mlx-nufft
legacy/                   the previous MATLAB-only implementation (reference only)
```

Everything runs on the GPU. With CUDA all array arguments are device pointers (`gpuArray` in
MATLAB, `cupy.ndarray` in Python) and the library never copies bulk data through the host. With
Metal they are ordinary host arrays (MATLAB arrays, `numpy.ndarray`), which the Apple GPU reads in
place through unified memory.

---

## Contents

1. [The operations](#the-operations)
2. [Conventions](#conventions) (including the [Gram-block convention](#gram-blocks-mr-multitasking-convention))
3. [Building](#building)
4. [MATLAB](#matlab)
5. [Python](#python)
6. [C / C++](#c--c)
7. [Multi-GPU (CUDA)](#multi-gpu-cuda)
8. [Why Toeplitz and how does it work?](#why-toeplitz-and-how-does-it-work)
9. [Implementation notes](#implementation-notes)
10. [Apple silicon (Metal backend)](#apple-silicon-metal-backend)
11. [Benchmarks](#benchmarks)

---

## The operations

### `calc_toep` — one-off pre-computation

```
phi2_toep = calc_toep(phi2, traj, config)
```

Turns the non-Cartesian Gram blocks $\Phi_2 = \Phi\Phi^H$ (MR-Multitasking's
`Phi_rt(:,idx) * Phi_rt(:,idx)'`, see [Gram blocks](#gram-blocks-mr-multitasking-convention)), one
$L \times L$ matrix per trajectory, into the Cartesian block-diagonal multiplier used by the
operator. Internally: pack each block into a
triangle, type-1 cufinufft onto the upsampled grid, centred inverse FFT, optional mask gather.

### `AhA_Toeplitz` — the hot loop

```
AhAx = AhA_Toeplitz(x, SEs, phi2_toep, config)
AhAx = AhA_Toeplitz(x, SEs, phi2_toep, config, mask)
```

For every coil: multiply by the sensitivity map, zero-pad onto the upsampled grid, FFT, multiply the
rank coefficients at each k-space location, as a row vector from the right, by that location's
Hermitian $L \times L$ block ($v \leftarrow v\,B$, as MR-Multitasking does), inverse FFT, crop,
multiply by the adjoint sensitivity map, accumulate. No NUFFT is evaluated here, which is the whole point of the
Toeplitz embedding.

### `calc_mask` — automatic mask for the masked operator

```
mask = calc_mask(traj, config)
```

Derives the upsampled k-space locations worth keeping from the trajectory alone. By default it keeps
every grid location within `config.mask_radius` (default 2) cells of a sample. The alternative
`config.mask_method = 'threshold'` thresholds the trajectory's unit-weight Toeplitz kernel at
`config.mask_threshold`. See [Masking](#masking).

### `AhA_cufinufft` / `mr_aha.AhANufftPlan` — reference without Toeplitz

```
AhAx = AhA_cufinufft(x, SEs, phi2, traj, config[, plan])     % MATLAB
y    = mr_aha.AhANufftPlan(cfg, traj).apply(x, SEs, phi2)    # Python
```

The same operator evaluated the "vanilla" way, coil by coil: type-2 cufinufft, one $L\times L$ block
per readout, type-1 cufinufft. Pure MATLAB / CuPy on top of cufinufft (no MR_AHA C code), used to
validate the Toeplitz operator and as the benchmark baseline. Both agree with `AhA_Toeplitz` to
$\sim 10^{-6}$.

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
| `phi2` | `[Ntrajs, L, L]` | `[Ntrajs, L, L]`, or `[Ntrajs, Nkz, L, L]` for stack (centred kz order) |
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

Optional: `n_coils`, `num_masked`, `eps`, `mem_fraction`, `gpu_device_id`, `verbose`,
`mask_method` (`'neighbour'` or `'threshold'`), `mask_radius`, `mask_threshold`. The MATLAB
helper `mr_default_config` and the Python `mr_aha.make_config` fill in sensible defaults.

### Gram blocks (MR-Multitasking convention)

The library follows MR-Multitasking. The temporal basis $\Phi$ is **rank × time** (`Phi_rt`,
`[L, T]`), the spatial coefficients keep the rank as their **last** dimension, and the Gram block of
a readout (and kz encode) is

$$\Phi_2 = \sum_{t\,\in\,\text{readout}} \Phi(:,t)\,\Phi(:,t)^H, \qquad
\texttt{Phi2 = Phi\_rt(:,idx) * Phi\_rt(:,idx)'} .$$

Every block multiplication (by `phi2` in the NUFFT reference operators, by `phi2_toep` in the
Toeplitz operator) treats the $L$ rank coefficients of one k-space location as a **row vector
multiplied from the right**, $v \leftarrow v\,\Phi_2$, i.e. $v_l \leftarrow \sum_m v_m (\Phi_2)_{ml}$ —
exactly MR-Multitasking's `reshape(k, [], L) * Phi2`. Since MR-Multitasking forms $A^Hb$ as
`k.' * Phi_rt'`, this is its normal operator, and a MR-Multitasking `Phi2` (`[L, L, Ntrajs, Nz]`)
is passed unchanged apart from the dimension order: `permute(Phi2, [3 4 1 2])` in MATLAB, or the
row-major `P.transpose(2, 3, 0, 1)` of the h5py array in Python (plus an `fftshift` along kz if it is
in natural order).

> **Changed.** Earlier versions multiplied $\Phi_2 v$ (column vector from the left), i.e. they
> expected $\Phi^H\Phi$ of a time × rank basis — the transpose (for these Hermitian blocks, the
> complex conjugate) of the block above. Gram blocks prepared for those versions need their two
> $L$ dimensions swapped; `phi2_toep` computed by them corresponds to the *other* triangulation with
> its entries transposed, so recompute it with `calc_toep`.

### Triangular packing

Only $(L^2+L)/2$ of the $L^2$ entries of each block are stored; the rest follows from
$A_{ij} = \overline{A_{ji}}$. Both packings are the column-major packing of the corresponding MATLAB
triangle, i.e. exactly `A(tril(true(L)))` / `A(triu(true(L)))`.

### Masking

`mask` holds linear indices into the *centred* upsampled grid `phi2_dim`, 1-based in MATLAB and
0-based in Python and C. Samples outside the mask contribute nothing. This is what makes a large 3D
operator fit in GPU memory when the upsampled k-space support is sparse.

`calc_mask` derives the mask from the trajectory, using one of two methods (`mask_method`).

**`'neighbour'` (default).**

1. Map every sample to its nearest point on the upsampled grid: `round(k·M/2π) + M/2` per axis,
   modulo `M`.
2. Keep every location within `mask_radius` cells of such a point (a $(2r+1)^d$ box). The default
   radius is 2.

This needs no NUFFT, and the same trajectory always gives the same mask. On the densely sampled
test problems radius 2 is as accurate as the unmasked operator:

* 280-spoke stack-of-stars: 79 % of the grid kept, operator error 3·10⁻⁶;
* 3D kooshball: 54 % kept, operator error 4·10⁻⁶.

It is less accurate on the 377-spoke 160² radial data (80 % kept, error 5·10⁻⁵) and on sparse
trajectories (64 spokes: 5·10⁻⁴; two-arm spiral: 3·10⁻²). There, raise `mask_radius` or use the
threshold method.

**`'threshold'`.**

1. Evaluate the Toeplitz kernel $T(\mathbf{k})$ of the trajectory with every weight set to one
   (exactly `calc_toep` with $L = 1$ and $\Phi\Phi^H = 1$).
2. Max-filter $|T|$ over the $3\times3$ ($3\times3\times3$ in 3D) neighbourhood of every location.
3. Keep the locations where the result exceeds `mask_threshold`.

The threshold is **absolute, in samples**: one isolated sample lying on the upsampled grid
contributes exactly 1 to $T$, so the default 0.05 drops locations that receive less than 5 % of one
sample's worth. A threshold relative to $\max|T|$ does not transfer between trajectories, because
the radial maximum sits at the k-space centre and equals the number of spokes (378, 280 and 15680
for the three test trajectories). Step 2 fills isolated near-zeros where the unit-weight Dirichlet
sum cancels inside the sampled region but the kernel of the real Gram blocks does not; it lowers the
masking error about tenfold at equal kept fraction.

This method follows the kernel's slowly decaying tails along the kx and ky axes, so it is more
accurate per kept location on sparse trajectories. At 0.05 it keeps 81–83 % of the grid in 2D and
stack, with an operator error of a few $10^{-6}$; it keeps 64 % for the kooshball.

[`benchmarks/neighbour_mask.md`](benchmarks/neighbour_mask.md) compares the two methods, and
[`benchmarks/README.md`](benchmarks/README.md) documents the threshold study.

---

## Building

Requirements (CUDA backend): CUDA toolkit (11.x or 12.x), CMake ≥ 3.24, a C++17 compiler, and the
bundled FINUFFT submodule (`third_party/finufft`). Unlike the previous version, **stock upstream FINUFFT is enough** — no
fork is needed. For Apple silicon see [Apple silicon](#apple-silicon-metal-backend).

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
| `MR_AHA_BACKEND` | `METAL` on macOS, else `CUDA` | which GPU backend the library is built for |
| — | | `build/libmr_aha.a` — the standalone library |
| `MR_AHA_BUILD_PYTHON` | `ON` | `build/lib/libmr_aha_c.so` (`.dylib` on macOS) — for the ctypes binding |
| `MR_AHA_BUILD_MATLAB` | `OFF` | `build/matlab/*.mexa64` / `*.mexmaca64` — needs `MATLAB_ROOT`; the CUDA gateways also need the Parallel Computing Toolbox |
| `MR_AHA_METAL_PRECOMPILE` | `ON` | Metal only: precompile the shaders when `xcrun metal` works (they are always embedded as source too) |
| `MR_AHA_BUILD_EXAMPLES` | `OFF` | `build/mr_example_aha` |
| `MR_AHA_BUILD_CUFINUFFT_MATLAB` | `OFF` | `build/finufft/matlab/cufinufft.mexa64` + `.m` wrappers — finufft's own GPU MATLAB interface, needed by `AhA_cufinufft.m` |

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

Masked operator, with the mask derived from the trajectory:

```matlab
config = mr_default_config('stack', [320 320 56], 16, 'is_masked', true);
[Om, mask] = calc_toep(phi2, traj, config);     % calls calc_mask when no mask is given
AhAx   = AhA_Toeplitz(x, SEs, Om, config, mask);

mask   = calc_mask(traj, config, 0.05);         % or explicitly, 1-based int64 gpuArray
```

NUFFT reference (needs `-DMR_AHA_BUILD_CUFINUFFT_MATLAB=ON`, and `build/finufft/matlab` on the
path, or any other cufinufft MATLAB installation):

```matlab
plan = mr_nufft_plan(traj, config);             % once: plans + point sorting
AhAx = AhA_cufinufft(x, SEs, phi2, traj, config, plan);
```

The `.m` wrappers move host arrays to the GPU, cast to complex `single`, derive `n_coils` from
`size(SEs)`, and convert the 1-based mask to the 0-based `int64` mask the gateway expects. Call
`AhA_Toeplitz_mex` / `calc_toep_mex` directly only if you want to skip that.

> ⓘ MATLAB must have initialised its GPU before the first MEX call. Running `gpuDevice(1)`, or
> simply creating any `gpuArray`, is enough; the wrappers do this implicitly by moving their inputs.

On Apple silicon the same functions take ordinary MATLAB arrays (no Parallel Computing Toolbox);
`calc_toep`, the `'threshold'` mask and `AhA_cufinufft` run their NUFFTs through MATLAB's Python
interface. See [Apple silicon](#apple-silicon-metal-backend).

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

On Apple silicon the arrays are NumPy arrays instead of CuPy arrays (see
[Apple silicon](#apple-silicon-metal-backend)); everything else is the same. Shapes are the reverse
of the table above, because CuPy and NumPy are row-major: `x` is `(L, Nz, Ny, Nx)`,
`SEs` is `(Ncoils, Nz, Ny, Nx)`, `traj` is `(ndim, Nkx, Ntrajs)`, and so on. See
[`python/README.md`](python/README.md).

Masked operator and NUFFT reference (`pip install cufinufft`):

```python
cfg  = mr_aha.make_config("3D", (160, 160, 160), L=12, n_coils=12, is_masked=True)
mask = mr_aha.calc_mask(cfg, traj)
Om   = mr_aha.calc_toep(cfg, phi2, traj, mask=mask)      # also sets cfg.num_masked
y    = mr_aha.aha_toeplitz(cfg, x, SEs, Om, mask=mask)

with mr_aha.AhANufftPlan(cfg, traj) as ref:
    y_ref = ref.apply(x, SEs, phi2)
```

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

## Multi-GPU (CUDA)

$A^H A$ is a sum over coils, so one application can be spread over several GPUs of a node: each
GPU applies the operator for a share of the coils and the partial results are summed on the
caller's GPU (the *primary*). `phi2_toep` and the mask are copied to every GPU once, when the
operator is created, together with each GPU's own sensitivity maps; an application then only
broadcasts `x` and gathers the partial outputs. Multi-GPU makes the operator faster but does not
raise the largest problem size: the whole kernel must fit on every GPU.

```matlab
op   = mr_aha_op(SEs, Om, config, 'devices', 1:4);   % 1-based gpuDevice indices
AhAx = op.apply(x);                                  % or op * x
info = op.info();                                    % devices and coils used
delete(op);
```

```python
with mr_aha.MultiGPUPlan(cfg, SEs, Om, devices=[0, 1, 2, 3]) as op:
    y = op.apply(x)
    print(op.info)
```

```c
mr_mgpu_plan op;
mr_mgpu_plan_create(&cfg, devices, n_devices, d_SEs, d_toep, NULL, 0, &op);
mr_mgpu_plan_execute(op, d_x, d_AhAx);
mr_mgpu_plan_destroy(op);
```

* **Primary GPU.** Inputs and outputs live on the current GPU (MATLAB's `gpuDevice`, CuPy's
  current device, or `config.gpu_device_id`), which always takes part. The current device is
  unchanged on return. MATLAB uses the other GPUs only inside the library, so its "one GPU per
  session" rule is not violated.
* **How many GPUs.** With $C$ coils on $G$ GPUs the time is set by $\lceil C/G \rceil$ coils, so
  the smallest number of GPUs with the same $\lceil C/G \rceil$ is used (12 coils on 5 GPUs use
  4). Small problems (fewer than $2^{22}$ upsampled values per coil, e.g. the 2D benchmark) stay
  on the primary GPU; `MR_AHA_MGPU_MIN_WORK` changes the threshold.
* **Remainder split.** When the coils do not divide over the GPUs, the remaining coils can be split
  across all GPUs *by rank*, with an all-to-all exchange of k-space slabs around the block-diagonal
  stage. `MR_AHA_MGPU_SPLIT=auto` (default) enables it only when a cost model, fed with a one-coil
  timing and the measured GPU-to-GPU bandwidth, predicts a gain of at least 5 %, which in practice
  needs NVLink; `on` / `off` force it.
* **Memory.** Every GPU holds a copy of `phi2_toep`. `share_primary` (MATLAB, Python) /
  `MR_MGPU_SHARE_PRIMARY` (C) lets the primary GPU use the caller's arrays in place instead of a
  second copy; they must then stay unchanged while the operator exists. The primary also holds
  one `x`-sized staging buffer per other GPU.
* **MATLAB lifetime.** The MEX file stays locked while an operator exists, so `clear mex` cannot
  unload it under a live operator. After `reset(gpuDevice)` or selecting another GPU, `apply`
  raises an error instead of computing; delete the operator and create a new one.

`tests/check_mgpu.py` and `tests/check_mgpu_matlab.m` compare the multi-GPU and single-GPU
operators; `benchmarks/bench_mgpu.py` measures the scaling.

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
also bounds cufinufft's internal fine grid through `gpu_maxbatchsize`. An `mr_aha_plan` holds its
working buffers for the lifetime of the plan, so the iteration allocates nothing: the padded grids
of one coil group (`[Mx, My, Mz, L]` per coil, see below), the cuFFT work area and, when masked, an
`[Mx, My, Mz]` int32 index map.

**Block-diagonal stage (CUDA).** This stage, the per-location $L \times L$ Hermitian multiply, is
memory-bound on reading `phi2_toep`, the largest array of the operator. For $L \le 30$ it uses
kernels specialised on $L$: the rank vectors stay in registers, the stored triangle is walked once
so that each element is read once (not once per row), and the masked variant works in place on the
padded grid through a grid-to-mask index map instead of gathering into a compact buffer. Coils are
processed in **groups** (default 4): their padded grids sit side by side, one batched cuFFT
transforms them all, and one kernel pass reads `phi2_toep` once for the whole group instead of once
per coil. The group shrinks until its buffers fit `config.mem_fraction` of the free GPU memory
(a group of 1 is always tried). Environment variables, read when a plan is created:
`MR_AHA_CUDA_COIL_GROUP=n` (requested group size), `MR_AHA_CUDA_GENERIC_BD=1` (generic
shared-memory kernels, one coil at a time; always used for $L > 30$, where they are as fast); and at every application
`MR_AHA_CUDA_PROFILE=1` prints the GPU time of each stage. The results are bitwise identical to the
generic kernels (`tests/check_bd_kernels.py`).

**Re-entrancy.** No `__constant__` or other shared device state: kernel geometry travels by value.
Several plans, CUDA streams and host threads can be in flight at once, and `mr_last_error` is
thread-local. `config.stream` selects the stream; `config.gpu_device_id = -1` (the default for
MATLAB and Python) keeps whichever device the caller has selected.

---

## Development and debugging in Visual Studio Code

Set `includePath` in [.vscode/c_cpp_properties.json](.vscode/c_cpp_properties.json) to include
`include/`, `src/common/`, `src/cuda/`, `third_party/finufft/include/` and the CUDA and MATLAB headers for IntelliSense.

To debug the MEX gateways: build with `-DCMAKE_BUILD_TYPE=Debug`, point `program` in
[.vscode/launch.json](.vscode/launch.json) at your MATLAB executable, and set breakpoints in
`matlab/*.cu` or `src/cuda/*.cu`. The [MATLAB GPU CUDA and MEX Programming
Manual](https://www.mathworks.com/help/parallel-computing/gpu-cuda-and-mex-programming.html) is the
reference for the MathWorks side.

## Apple silicon (Metal backend)

The Metal backend runs the same operators on the GPU of Apple-silicon Macs (developed and tested on
an M3 Ultra). It is selected automatically on macOS (`MR_AHA_BACKEND=METAL`).

**How it maps onto the CUDA backend.**

| | CUDA | Metal |
|---|---|---|
| kernels | `src/cuda/*.cu` | `src/metal/mr_kernels.metal` (same indexing, ported kernel by kernel) |
| FFT | cuFFT | [VkFFT](https://github.com/DTolm/VkFFT)'s Metal backend (`third_party/VkFFT`) |
| type-1 NUFFT (`calc_toep`, threshold mask) | cufinufft, inside the library | [mlx-nufft](https://github.com/martinlachaine/mlx-nufft), in Python (`third_party/mlx-nufft`) |
| arrays | device pointers (gpuArray, CuPy) | host pointers in unified memory (MATLAB arrays, NumPy) |
| MATLAB | gpuArray MEX gateways | host-array MEX gateways (`matlab/metal/`) + MATLAB's Python interface |

There is no C/C++ NUFFT for Apple GPUs, so the Toeplitz pre-computation is split in two:
`mr_aha.calc_toep` (Python) evaluates the type-1 NUFFT of each triangular component with mlx-nufft,
then the library's `mr_calc_toep_from_psf` performs the inverse FFT, scaling and mask gather on the
GPU. The threshold mask works the same way through `mr_calc_mask_from_kernel`. The C function
`mr_calc_toep` returns `MR_ERR_UNSUPPORTED` on Metal with directions; the operator itself
(`mr_aha_plan_*`, `mr_aha_toeplitz`) and the neighbourhood mask (`mr_calc_mask`) are complete in C.
Both new functions also exist in the CUDA backend.

**Requirements.** macOS 13 or newer on Apple silicon, CMake ≥ 3.24, Apple clang (Xcode or its
Command Line Tools), Python 3.11/3.12 with `numpy`, `mlx==0.31.2` and mlx-nufft. Optional: Xcode's
Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`), which lets CMake precompile the
shaders; without it they are compiled by macOS's built-in Metal compiler on first use (~0.15 s,
cached by macOS afterwards).

```bash
git submodule update --init third_party/VkFFT third_party/mlx-nufft
cmake -S . -B build -DMR_AHA_BUILD_MATLAB=ON -DMatlab_ROOT_DIR=/Applications/MATLAB_R2024b.app
cmake --build build -j

uv venv --python 3.12 .venv                     # 3.12: newest Python MATLAB R2024b's pyenv accepts
uv pip install -p .venv/bin/python numpy scipy "mlx==0.31.2" -e third_party/mlx-nufft
```

Python: `import mr_aha` with `python/` on the path (or `pip install -e python`); arrays are NumPy.
MATLAB: put `matlab/` and `build/matlab/` on the path; `mr_python` loads `.venv/bin/python`
in-process (or the interpreter named by `MR_AHA_PYTHON`) the first time a NUFFT is needed.

**Validation.** `tests/check_reference.py` and `tests/check_reference_matlab.m` (via
`tests/run_matlab_tests.m`) compare every public function with CUDA reference outputs
(`benchmarks/make_reference.py`): 2D, 3D and stack-of-stars, odd sizes and a prime FFT length,
L = 1, 3, 4, both triangulations, unmasked and both masks. The operator matches CUDA to 2-3e-7; the
operator built by the mlx-nufft `calc_toep` matches the stored one to 0.8-1.9e-6; the neighbourhood
mask is identical.

**VkFFT fork.** `third_party/VkFFT` points to [mikecjz/VkFFT](https://github.com/mikecjz/VkFFT)
(branch `mr_aha`): upstream `develop` plus two Metal fixes, an autorelease double free
(DTolm/VkFFT#226) and push constants that were overwritten when several dispatches shared a
command buffer. Details in [`benchmarks/vkfft/README.md`](benchmarks/vkfft/README.md).

**Size limit.** One VkFFT transform on Metal must cover fewer than 2^31 complex values (VkFFT's
32-bit indexing; its 64-bit mode does not compile on Metal, and launch offsets cannot get around
it). The operator therefore requires `Mx·My·Mz·L < 2^31`, e.g. `L ≤ 65` for a 320^3 grid, and
returns `MR_ERR_UNSUPPORTED` beyond that; `calc_toep` chunks its components to stay below it.

**Accuracy of `calc_toep` on Metal.** mlx-nufft is a single-precision NUFFT, like the cufinufft
used on CUDA, and both are accurate to about 1e-6 to 3e-5 on large 3D grids, depending on the
input. On the 3D kooshball benchmark (smooth Gram blocks, a strongly peaked point-spread
function) the Metal operator differs from the NUFFT reference by 1.4e-5 per application, against
1.9e-6 for CUDA. Where that matters, `mr_aha.calc_toep(..., nufft='finufft')` computes the
point-spread functions with CPU FINUFFT in double precision: 5e-6, at about 4x the pre-compute
time (67 s instead of 15 s on an M3 Ultra). 2D and stack-of-stars are at CUDA's level (1e-6)
with the default.

**Performance tuning.** The block-diagonal stage, which is bound by reading `phi2_toep`, uses
kernels specialised on L (L ≤ 32; generic ones otherwise) and processes the coils in groups
(default 4, `MR_AHA_METAL_COIL_GROUP`), two at a time from registers, so `phi2_toep` is read far
fewer times than once per coil. Together they cut the GPU time of one A^H A by 16-23 % on the
benchmark problems; profile with `MR_AHA_METAL_PROFILE=1`.

**Environment variables** (testing and diagnostics): `MR_AHA_METAL_FROM_SOURCE=1` compiles the
embedded shader source even when a precompiled library is embedded; `MR_AHA_METAL_FORCE_COPY=1`
copies caller arrays instead of wrapping them in place; `MR_AHA_METAL_COIL_GROUP=<n>` sets the
coil group size; `MR_AHA_METAL_GENERIC_BD=1` uses the
generic block-diagonal kernels instead of the ones specialised on L; `MR_AHA_METAL_PROFILE=1`
prints the GPU time of each operator stage; `MR_AHA_PYTHON` selects MATLAB's Python.

**Distributing a prebuilt library.** No Xcode is needed at run time (shaders are precompiled or
compiled by macOS itself). A downloaded, unsigned `.dylib` / `.mexmaca64` is quarantined by
Gatekeeper: clear it with `xattr -dr com.apple.quarantine <folder>` or sign and notarise the binaries.

---

## Benchmarks

[`benchmarks/`](benchmarks/) compares the Toeplitz operator (Python) with the NUFFT reference
(Python and MATLAB) on 2D radial, stack-of-stars and synthetic 3D kooshball problems, including
20-iteration CG reconstructions, on a local TITAN RTX and an HPC L40S, and with the Metal backend
on an Apple M3 Ultra (faster than the L40S for the Toeplitz operator on the large problems,
identical reconstructions). Findings and how to reproduce them are in
[`benchmarks/README.md`](benchmarks/README.md).

---

## License

[PolyForm Noncommercial 1.0.0](LICENSE).
