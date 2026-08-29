# Legacy (pre-Stage-1) code

Everything in this directory is the original MATLAB-only implementation that
the C++/CUDA library in `src/core` replaces. It is kept for reference only: it
is **not** built by the top-level `CMakeLists.txt` and it is not maintained.

| File | What it was |
| ---- | ----------- |
| `AhA_cufinufft_toep.cu` | MEX gateway for the stack-of-stars Toeplitz `A'A`, hard-wired to MATLAB and to a `cufinufftf_deconvolve` entry point that exists only in a FINUFFT fork. |
| `AhA_ps_cufinufft.cu` | MEX gateway for the non-Toeplitz `A'A` (type-1/type-2 NUFFT per iteration). |
| `cufinufftf2d1.cu`, `cufinufftfspread2d1.cu`, `finufftf2d1.c` | Thin MEX wrappers used while benchmarking FINUFFT. |
| `AhA_ps3D_toep.m`, `compile_mex.m`, `build_mex_static.sh` | MATLAB-side driver and the old ad-hoc build scripts. |
| `cu_finufft_test.m`, `spread_test.m`, `toeplitz_test.m` | MATLAB scratch tests for the above. |
| `CPM.cmake` | CMake package manager used to fetch a FINUFFT fork; the submodule in `finufft/` replaces it. |

## What changed in the rewrite

* The numerics live in a standalone C library (`libmt_cuda`) with a pure C
  (`extern "C"`) surface, so MATLAB, Python and plain C/C++ all call the same
  code. Nothing in the core depends on MATLAB headers any more.
* The `cufinufftf_deconvolve` shortcut is gone. `calc_toep` now builds the
  Toeplitz kernel explicitly with a stock type-1 cufinufft plus a centred
  inverse FFT, so an unmodified upstream FINUFFT works.
* 2D, 3D and stack-of-stars trajectories are all supported, not just
  stack-of-stars, and the upsampled Toeplitz grid is an explicit parameter
  (`config.phi2_dim`) rather than being hard-coded to `2*N`.
* The Hermitian `L x L` blocks are stored in packed triangular form, with the
  masked variant available for sparse upsampled grids.
* Centring is realised by exact index remapping instead of phase modulation,
  which removes the half-sample bias for odd matrix sizes.
