/**
 * @file mr_aha.h
 * @brief Public C API of the MR_AHA library.
 *
 * MR_AHA evaluates the forward normal operator @f$A^H A@f$ for non-Cartesian
 * MRI with low-rank (subspace) modeling on the GPU, using a
 * Toeplitz embedding so that no non-uniform FFT is required inside the
 * iteration loop.
 *
 * The library is a standalone C++ library with a pure C interface
 * (`extern "C"`), so that it can be linked directly from C, C++, a MATLAB MEX
 * gateway or a Python `ctypes` binding. It is built for exactly one GPU
 * backend, chosen at configure time with the CMake option `MR_AHA_BACKEND`
 * and reported at run time by ::mr_backend:
 *
 * - **CUDA** (NVIDIA GPUs): CUDA kernels, cuFFT and cufinufft.
 * - **Metal** (Apple silicon): Metal kernels and VkFFT's Metal backend.
 *
 * @section mr_layout Array layout
 *
 * Every multi-dimensional array documented in this header uses **column-major**
 * order (the MATLAB / Fortran convention). A row-major caller (NumPy, CuPy)
 * must simply reverse the dimension order: a column-major `[Nx, Ny, Nz, L]`
 * array is a row-major `[L, Nz, Ny, Nx]` array with the *same* linear memory
 * layout, so no data movement is ever required.
 *
 * All image-space and k-space arrays are assumed to be **centered**, i.e. the
 * origin (DC / image center) sits in the middle of the array at index
 * `floor(dim/2)`, not in the corner. The library performs the required
 * (i)fftshifts internally, by index remapping rather than by phase modulation,
 * so the convention is exact for both even and odd dimensions.
 *
 * @section mr_gram Gram blocks
 *
 * The library uses one convention throughout. The temporal
 * basis is stored **rank x time**, @f$\Phi \in \mathbb{C}^{L \times T}@f$
 * (MATLAB `Phi_rt`, `[L, T]`), and the spatial coefficients keep the rank as
 * their **last** dimension (`[Nx, Ny, Nz, L]`). The Gram block of a readout
 * (and, for ::MR_TRAJ_STACK, of a kz encode) is
 * @f[
 *   \Phi_2 \;=\; \Phi_{\mathcal{T}}\,\Phi_{\mathcal{T}}^H
 *          \;=\; \sum_{t \in \mathcal{T}} \Phi(:,t)\,\Phi(:,t)^H ,
 * @f]
 * where @f$\mathcal{T}@f$ are the time points acquired on that readout, i.e.
 * MATLAB `Phi_rt(:, idx) * Phi_rt(:, idx)'`. It is Hermitian
 * @f$L \times L@f$.
 *
 * Every block-diagonal multiplication (the Toeplitz multiplier in
 * ::mr_aha_plan_execute and the Gram multiplication of the NUFFT reference
 * operators of the bindings) treats the length-@f$L@f$ rank vector @f$v@f$ of
 * one k-space location as a **row vector multiplied from the right**:
 * @f[
 *   v \;\leftarrow\; v\,\Phi_2, \qquad
 *   v_l \;\leftarrow\; \sum_m v_m\,(\Phi_2)_{ml},
 * @f]
 * in MATLAB `reshape(k, [], L) * Phi2`. For the data model
 * @f$k_t = \sum_m \Phi(m,t)\,(F S u_m)_t@f$ with adjoint
 * @f$(A^H k)_l = S^H F^H \sum_t \overline{\Phi(l,t)}\,k_t@f$ (in MATLAB
 * `k.' * Phi_rt'`), the coefficient coupling rank
 * @f$m@f$ into rank @f$l@f$ is
 * @f$\sum_t \overline{\Phi(l,t)}\,\Phi(m,t) = (\Phi_2)_{ml}@f$, so this is
 * the normal operator.
 *
 * @note Earlier versions of the library multiplied @f$\Phi_2 v@f$ (column
 *       vector from the left), i.e. they expected
 *       @f$\Phi^H\Phi@f$ with a time x rank @f$\Phi@f$, which is the
 *       transpose (equivalently, the complex conjugate) of the block above.
 *       Gram blocks prepared for those versions must have their two @f$L@f$
 *       axes swapped; `phi2_toep` arrays computed by them must be conjugated.
 *
 * @section mr_memory Memory ownership
 *
 * The caller allocates and frees every input and output array; the library
 * only allocates its own working buffers. What an array pointer is depends on
 * the backend:
 *
 * - **CUDA:** every array argument is a **device (GPU) pointer**. The library
 *   never touches host buffers for bulk data.
 * - **Metal:** every array argument is an ordinary **host pointer**. Apple
 *   silicon has unified memory, so the GPU reads and writes the caller's
 *   arrays directly (without a copy when they are page-aligned).
 *
 * The `d_` prefix of the array parameters below therefore means "device
 * pointer" for CUDA and "host pointer in unified memory" for Metal. Small
 * scalar configuration values are passed by value on the host through
 * ::mr_config.
 *
 * @section mr_precision Precision
 *
 * The library is single precision (`complexFloat` / ::mr_complex) throughout,
 * matching the `complexsingle` gpuArray type in MATLAB and `complex64` in
 * NumPy/CuPy.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#ifndef MR_AHA_MR_AHA_H
#define MR_AHA_MR_AHA_H

/* The header deliberately depends only on the C standard library: `int64_t`
 * (all element counts and mask indices are 64-bit so that grids above 2^31
 * voxels can be described) and `size_t`. No CUDA or Metal header is pulled in
 * unless the including translation unit asks for the CUDA complex type below. */
#include <stddef.h>
#include <stdint.h>

/* ------------------------------------------------------------------------ */
/* Export / alignment macros                                                */
/* ------------------------------------------------------------------------ */

/**
 * @def MR_EXPORT
 * @brief Marks a function as part of the library's exported ABI.
 *
 * - Windows: `__declspec(dllexport)` while the shared library itself is being
 *   built (the build defines `MR_AHA_BUILD_SHARED`), and empty otherwise, so
 *   static builds and consumers of the import library need no annotation.
 * - Other platforms: default ELF / Mach-O symbol visibility, so the symbols
 *   stay exported even when the library is compiled with
 *   `-fvisibility=hidden`.
 */
#ifdef _WIN32
#  ifdef MR_AHA_BUILD_SHARED
#    define MR_EXPORT __declspec(dllexport)
#  else
#    define MR_EXPORT
#  endif
#else
#  define MR_EXPORT __attribute__((visibility("default")))
#endif

/* Every declaration below has C linkage so that the symbol names are not
 * mangled: MATLAB MEX gateways, Python `ctypes` and plain C callers look them
 * up by their literal names. */
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @def MR_ALIGN8
 * @brief Forces 8-byte alignment of a type (MSVC and GCC/Clang spellings).
 *
 * Used on the fallback ::mr_complex so that its alignment matches CUDA's
 * `cuFloatComplex` (`float2`, 8-byte aligned); otherwise a host-compiled
 * struct of two floats would be only 4-byte aligned and arrays could not be
 * shared safely between the two definitions.
 */
#if defined(_MSC_VER)
#  define MR_ALIGN8 __declspec(align(8))
#else
#  define MR_ALIGN8 __attribute__((aligned(8)))
#endif

/* ------------------------------------------------------------------------ */
/* Complex scalar type                                                      */
/* ------------------------------------------------------------------------ */

/* Two definitions of the same 8-byte {float re, float im} layout: CUDA code
 * gets the native `cuFloatComplex` so kernels can call cuCmulf() & co. on
 * array elements without casts, everyone else gets a dependency-free struct.
 * Because the layouts are identical, a pointer produced by one side can be
 * passed to the other. */
#if defined(MR_AHA_CUDA_COMPLEX) || defined(__CUDACC__)
#  include <cuComplex.h>
/** @brief Single-precision complex scalar used by every MR_AHA array.
 *
 *  In the CUDA build (and in any translation unit compiled by `nvcc`) this is
 *  CUDA's `cuFloatComplex`, so that kernels can use the `cuCmulf` family
 *  directly. The CUDA backend defines `MR_AHA_CUDA_COMPLEX` for all its
 *  translation units so that they agree on the type. */
typedef cuFloatComplex mr_complex;
#else
/** @brief Single-precision complex scalar used by every MR_AHA array.
 *
 *  Interleaved real and imaginary parts, 8-byte aligned. Layout-identical to
 *  CUDA's `cuFloatComplex`, C99 `float _Complex`, C++ `std::complex<float>`,
 *  MATLAB's interleaved `single` complex and NumPy's `complex64`, so arrays of
 *  any of those can be passed by pointer. */
typedef struct MR_ALIGN8 mr_complex_s {
    float x; /**< Real part.      */
    float y; /**< Imaginary part. */
} mr_complex;
#endif

/* ------------------------------------------------------------------------ */
/* Library-wide constants                                                   */
/* ------------------------------------------------------------------------ */

/**
 * @brief Default neighborhood radius of ::mr_calc_mask, in upsampled grid cells.
 *
 * Installed into ::mr_config::mask_radius by ::mr_config_defaults.
 *
 * Measured in the documentation
 * (https://mikecjz.github.io/MR_AHA_docs/benchmarks/neighbour-mask.html):
 * with radius 2 the masked operator matches
 * the unmasked one to within the single-precision NUFFT tolerance on
 * image-like inputs for the 280-spoke radial (79 % of the grid kept) and 3D
 * kooshball (54 % kept) test problems. On the 377-spoke 160^2 radial data
 * the image-input error is 5e-5 (80 % kept), and on sparse trajectories
 * (64 radial spokes, two-arm spiral) it is 5e-4 to 3e-2; increase
 * ::mr_config::mask_radius there. Radius 0 or 1 leaves gaps between the
 * spokes near the edge of k-space.
 */
#define MR_DEFAULT_MASK_RADIUS 2

/* ------------------------------------------------------------------------ */
/* Status codes                                                             */
/* ------------------------------------------------------------------------ */

/**
 * @brief Return codes of every MR_AHA entry point.
 *
 * Every function returns ::MR_SUCCESS (== 0) on success and a negative value on
 * failure. Use ::mr_status_string to obtain a human readable description.
 *
 * The functions return a plain `int` (not ::mr_status) so that the ABI does
 * not depend on the compiler's choice of enum width; compare the result
 * against these enumerators. The codes are grouped by origin: argument and
 * configuration errors (-1, -2, -7) are detected on the host before any GPU
 * work is issued; -3 to -6 come from the CUDA backend and its libraries;
 * -9 and -10 only from the Metal backend. A failure is accompanied by a
 * detailed message in ::mr_last_error.
 */
typedef enum mr_status {
    MR_SUCCESS            =  0, /**< Operation completed successfully.        */
    MR_ERR_INVALID_ARG    = -1, /**< A NULL pointer or nonsensical size.      */
    MR_ERR_INVALID_CONFIG = -2, /**< ::mr_config failed validation.           */
    MR_ERR_ALLOC          = -3, /**< A GPU memory allocation failed.          */
    MR_ERR_CUDA           = -4, /**< A CUDA runtime call failed.              */
    MR_ERR_CUFFT          = -5, /**< A cuFFT call failed.                     */
    MR_ERR_CUFINUFFT      = -6, /**< A cufinufft call failed.                 */
    MR_ERR_UNSUPPORTED    = -7, /**< Valid but not implemented combination.   */
    MR_ERR_INTERNAL       = -8, /**< Should not happen; please report.        */
    MR_ERR_METAL          = -9, /**< A Metal call failed (Metal backend).     */
    MR_ERR_VKFFT          = -10 /**< A VkFFT call failed (Metal backend).     */
} mr_status;

/**
 * @brief Human readable description of an ::mr_status value.
 * @param[in] status Value previously returned by an MR_AHA function.
 * @return Pointer to a static, NUL-terminated string. Never NULL; values that
 *         are not ::mr_status enumerators map to `"unknown status"`. The
 *         string must not be freed or modified.
 */
MR_EXPORT const char *mr_status_string(int status);

/**
 * @brief Detailed message describing the most recent failure on this thread.
 *
 * The library records a short, human readable diagnostic (including the
 * offending CUDA / cuFFT / cufinufft error where applicable) in thread-local
 * storage whenever it returns a non-zero status. The buffer is overwritten by
 * the next failing call.
 *
 * Every status-returning entry point also clears the buffer when it starts
 * (and, on CUDA, resets the runtime's per-thread last error), so read the
 * message right after the failing call, before calling into the library
 * again. Because the storage is thread-local, concurrent callers on
 * different host threads never see each other's diagnostics.
 *
 * @return Pointer to a thread-local, NUL-terminated string. Never NULL; empty
 *         if no failure has been recorded on this thread.
 */
MR_EXPORT const char *mr_last_error(void);

/**
 * @brief Name of the GPU backend this library was built for.
 *
 * Lets a binding decide at run time how to pass arrays (device pointers for
 * CUDA, host pointers for Metal, see @ref mr_memory) and which features exist
 * (see @ref mr_op_backends).
 *
 * @return `"cuda"` or `"metal"`; a pointer to a static string, never NULL.
 */
MR_EXPORT const char *mr_backend(void);

/* ------------------------------------------------------------------------ */
/* Configuration                                                            */
/* ------------------------------------------------------------------------ */

/**
 * @brief Trajectory family, selecting how ::mr_calc_toep interprets its input.
 *
 * Determines the number of spatial dimensions (::mr_config::n_spatial), the
 * number of trajectory coordinates `ndim` (2 for ::MR_TRAJ_2D and
 * ::MR_TRAJ_STACK, 3 for ::MR_TRAJ_3D), the dimensionality of the type-1
 * NUFFT that builds the kernel, and whether the z axis is non-uniform (3D) or
 * already Cartesian (stack, where no upsampling or shift is applied along z).
 */
typedef enum mr_traj_type {
    MR_TRAJ_2D    = 0, /**< Planar 2D trajectory; image is `[Nx, Ny, L]`.      */
    MR_TRAJ_3D    = 1, /**< Fully 3D trajectory; image is `[Nx, Ny, Nz, L]`.   */
    MR_TRAJ_STACK = 2  /**< Stack of identical 2D trajectories along kz
                            (e.g. stack-of-stars). Image is `[Nx, Ny, Nz, L]`
                            and `Mz == Nz == Nkz`.                            */
} mr_traj_type;

/**
 * @brief Which triangle of the Hermitian @f$L \times L@f$ block is stored.
 *
 * `phi2_toep` holds only @f$(L^2+L)/2@f$ of the @f$L^2@f$ entries of each
 * Hermitian block. Both packings are **column-major** packings of the
 * corresponding MATLAB triangle, i.e. exactly the element order produced by
 * `A(tril(true(L)))` / `A(triu(true(L)))`:
 *
 * - ::MR_TRI_LOWER: element `(i, j)`, `i >= j`, lives at
 *   `t = j*L - j*(j-1)/2 + (i - j)`.
 * - ::MR_TRI_UPPER: element `(i, j)`, `i <= j`, lives at `t = j*(j+1)/2 + i`.
 *
 * The missing half is reconstructed on the fly from Hermitian symmetry,
 * @f$A_{ij} = \overline{A_{ji}}@f$.
 */
typedef enum mr_triangulation {
    MR_TRI_LOWER = 0, /**< Lower triangle, column-major packed. */
    MR_TRI_UPPER = 1  /**< Upper triangle, column-major packed. */
} mr_triangulation;

/**
 * @brief Complete description of one @f$A^H A@f$ operator.
 *
 * Fill with ::mr_config_defaults first, then override the fields you need. Use
 * ::mr_config_validate to check consistency before creating a plan (the plan
 * constructors validate as well).
 *
 * The struct is plain data passed by pointer; every function that keeps it
 * (plans, operators) stores its own copy, so the caller's instance may be
 * modified or freed afterwards. Fields fall into three groups: the problem
 * geometry (::traj_type to ::num_masked), the coil count (::n_coils) and the
 * execution environment (::eps to ::mask_radius).
 */
typedef struct mr_config {
    /** Trajectory family. See ::mr_traj_type. */
    mr_traj_type traj_type;

    /** Number of *spatial* dimensions of the image: 2 or 3. Derived from
     *  ::traj_type by ::mr_config_defaults but may be set explicitly:
     *  ::MR_TRAJ_2D uses 2, ::MR_TRAJ_3D and ::MR_TRAJ_STACK use 3.
     *  ::mr_config_validate rejects any other combination, so an explicit
     *  value must agree with ::traj_type. It is also the rank of the cuFFT
     *  over the upsampled grid. */
    int n_spatial;

    /** Image matrix size `[Nx, Ny, Nz]`. Entries beyond ::n_spatial must be 1.
     *  Every entry must be >= 1. Defines the shape of `x`, `AhAx` and the
     *  sensitivity maps; the image center is at `floor(N/2)` on each axis. */
    int n_dim[3];

    /** Upsampled (Toeplitz) matrix size `[Mx, My, Mz]`. Entries beyond
     *  ::n_spatial must be 1. Typically `M = 2*N`. For ::MR_TRAJ_STACK,
     *  `Mz` must equal `Nz` because kz is already Cartesian. `M >= N` is
     *  required on every axis. `M = 2*N` makes the circulant embedding of
     *  the Toeplitz operator exact; a smaller `M` wraps the point-spread
     *  function around and introduces aliasing in @f$A^H A@f$. */
    int phi2_dim[3];

    /** Number of ranks / subspace coefficients @f$L@f$. Must be >= 1. Sets
     *  the trailing extent of `x` / `AhAx`, the size of each Hermitian
     *  block and hence the `(L^2+L)/2` components of `phi2_toep`. */
    int L;

    /** Triangle stored in `phi2_toep`. See ::mr_triangulation. */
    mr_triangulation triangulation;

    /** Non-zero when `phi2_toep` is stored only on a subset of the upsampled
     *  Cartesian grid; then `phi2_toep` is `[num_masked, (L^2+L)/2]` and a
     *  `mask` of ::num_masked linear indices must be supplied. */
    int is_masked;

    /** Number of retained samples when ::is_masked is non-zero, else ignored.
     *  Must then satisfy `0 < num_masked <= Mx*My*Mz`. Typically the count
     *  returned by ::mr_calc_mask. */
    int64_t num_masked;

    /** Number of receive coils @f$N_{coils}@f$. Must be >= 1. Used only by
     *  ::mr_aha_toeplitz / ::mr_aha_plan_execute. */
    int n_coils;

    /** Requested cufinufft tolerance for ::mr_calc_toep (and for both
     *  transforms of an ::MR_OP_NUFFT operator). Default `1e-6`; must lie
     *  in the open interval (0, 1). */
    double eps;

    /** CUDA device to run on. Default 0. Set to -1 to keep the caller's
     *  current device (recommended when interoperating with MATLAB gpuArray or
     *  CuPy, which manage the current device themselves). Ignored by the
     *  Metal backend, which always uses the system's default GPU. */
    int gpu_device_id;

    /** CUDA stream (`cudaStream_t`) for all work, or NULL for the default
     *  stream. Declared `void *` to keep this header free of `cuda_runtime.h`.
     *  Ignored by the Metal backend, whose calls are synchronous. */
    void *stream;

    /** Fraction of the *free* device memory ::mr_calc_toep may use for its
     *  working buffers, and that ::mr_aha_plan_create may use for its coil
     *  groups (CUDA). Default 0.5. Clamped to (0, 0.95]: values above 0.95
     *  are reduced to 0.95 (headroom for the CUDA runtime and cuFFT), and
     *  non-positive or NaN values fall back to 0.5. The free memory is
     *  queried when the call starts, so buffers the caller has already
     *  allocated are excluded from it. */
    double mem_fraction;

    /** Non-zero to print per-stage timing to stdout. Default 0. In
     *  ::mr_aha_plan_execute this also synchronizes ::stream before
     *  printing, so leave it off in timing-critical loops. */
    int verbose;

    /** Neighborhood radius of ::mr_calc_mask, in upsampled grid cells: every
     *  location within this many cells of the grid point nearest to a sample
     *  is kept. Default ::MR_DEFAULT_MASK_RADIUS. Must be >= 0; 0 keeps only
     *  the grid point nearest to each sample. */
    int mask_radius;
} mr_config;

/**
 * @brief Populate @p cfg with library defaults for a given trajectory type.
 *
 * Sets `eps = 1e-6`, `gpu_device_id = 0`, `stream = NULL`, `n_coils = 1`,
 * `mem_fraction = 0.5`, `triangulation = MR_TRI_LOWER`, `is_masked = 0`,
 * `mask_radius = MR_DEFAULT_MASK_RADIUS`, all sizes to 1 and ::n_spatial
 * according to @p traj_type.
 *
 * The struct is zeroed first, so `num_masked = 0` and `verbose = 0` as well,
 * and `L = 1`. The sizes (::mr_config::n_dim, ::mr_config::phi2_dim,
 * ::mr_config::L, ::mr_config::n_coils) must then be set by the caller; the
 * result of this function alone describes a trivial 1-voxel problem.
 *
 * @param[out] cfg       Configuration to initialize. Must be non-NULL.
 * @param[in]  traj_type Trajectory family the operator will be used for.
 * @return ::MR_SUCCESS, or ::MR_ERR_INVALID_ARG if @p cfg is NULL.
 */
MR_EXPORT int mr_config_defaults(mr_config *cfg, mr_traj_type traj_type);

/**
 * @brief Check a configuration for internal consistency.
 *
 * Verifies dimensionality, positivity of all sizes, `Mz == Nz` for
 * ::MR_TRAJ_STACK, `M >= N` in every transformed direction, and that
 * ::mr_config::num_masked is positive and no larger than the upsampled grid
 * when ::mr_config::is_masked is set.
 *
 * In full, the checks are: ::mr_config::traj_type is a known enumerator;
 * ::mr_config::n_spatial equals 2 for ::MR_TRAJ_2D and 3 otherwise; every
 * `n_dim[d]` and `phi2_dim[d]` is >= 1, both are 1 for `d >= n_spatial`, and
 * `phi2_dim[d] >= n_dim[d]`; `phi2_dim[2] == n_dim[2]` for ::MR_TRAJ_STACK;
 * `L >= 1`; `n_coils >= 1`; ::mr_config::triangulation is a known
 * enumerator; `0 < eps < 1`; `mask_radius >= 0`; and, when masked,
 * `0 < num_masked <= Mx*My*Mz`. The execution fields (device, stream,
 * mem_fraction, verbose) are not checked. Purely host-side; touches no GPU.
 *
 * @param[in] cfg Configuration to check.
 * @return ::MR_SUCCESS, ::MR_ERR_INVALID_ARG if @p cfg is NULL, or
 *         ::MR_ERR_INVALID_CONFIG naming the first failed check (see
 *         ::mr_last_error).
 */
MR_EXPORT int mr_config_validate(const mr_config *cfg);

/* ------------------------------------------------------------------------ */
/* Derived-size helpers                                                     */
/* ------------------------------------------------------------------------ */

/* Element counts (not bytes) of the arrays the caller must allocate, so that
 * bindings need not re-derive the layouts above. Multiply by
 * `sizeof(mr_complex)` (8) or `sizeof(int64_t)` for bytes. All are pure
 * host-side arithmetic in 64 bits on an unvalidated @p cfg; they do not
 * clear or set ::mr_last_error. */

/**
 * @brief Number of stored entries of one packed Hermitian block, @f$(L^2+L)/2@f$.
 * @param[in] L Rank.
 * @return `(int64_t)L * (L + 1) / 2`, or 0 when `L <= 0`.
 */
MR_EXPORT int64_t mr_num_tri(int L);

/**
 * @brief Number of voxels of one image-space volume, `Nx*Ny*Nz`.
 * @param[in] cfg Configuration.
 * @return Voxel count, or 0 when @p cfg is NULL.
 */
MR_EXPORT int64_t mr_image_size(const mr_config *cfg);

/**
 * @brief Number of voxels of one upsampled (Toeplitz) volume, `Mx*My*Mz`.
 * @param[in] cfg Configuration.
 * @return Voxel count, or 0 when @p cfg is NULL.
 */
MR_EXPORT int64_t mr_pad_size(const mr_config *cfg);

/**
 * @brief Number of ::mr_complex elements the `phi2_toep` array must hold.
 *
 * `num_masked * (L^2+L)/2` when masked, else `Mx*My*Mz * (L^2+L)/2`.
 *
 * @param[in] cfg Configuration.
 * @return Element count, or 0 when @p cfg is NULL.
 */
MR_EXPORT int64_t mr_phi2_toep_size(const mr_config *cfg);

/**
 * @brief Number of ::mr_complex elements of the spatial-coefficient array `x`.
 *
 * Also the size of the output `AhAx`, which has the same shape.
 *
 * @param[in] cfg Configuration.
 * @return `Nx*Ny*Nz*L`, or 0 when @p cfg is NULL.
 */
MR_EXPORT int64_t mr_x_size(const mr_config *cfg);

/**
 * @brief Number of ::mr_complex elements of the sensitivity-map array `SEs`.
 *
 * One map per coil, shared by every rank, so the rank axis never appears.
 *
 * @param[in] cfg Configuration.
 * @return `Nx*Ny*Nz*Ncoils`, or 0 when @p cfg is NULL.
 */
MR_EXPORT int64_t mr_se_size(const mr_config *cfg);

/* ------------------------------------------------------------------------ */
/* Forward operator: AhA via Toeplitz embedding                             */
/* ------------------------------------------------------------------------ */

/* The stateless single-GPU building block: a plan holds only the work
 * buffers and FFT plans; the large inputs (maps, kernel, mask) are passed to
 * every execution. ::mr_op (below) wraps it into a stateful operator that
 * owns those inputs and can span several GPUs. */

/** @brief Opaque, reusable @f$A^H A@f$ plan. Create with ::mr_aha_plan_create.
 *
 *  A plan is bound to the device and stream of the configuration it was
 *  created with. It must not be executed concurrently from several host
 *  threads, because all executions share the plan's padded work buffers;
 *  separate plans are independent. */
typedef struct mr_aha_plan_s *mr_aha_plan;

/**
 * @brief Create a reusable @f$A^H A@f$ plan.
 *
 * The plan owns the cuFFT plans and the padded working buffers, so that an
 * iterative reconstruction can call ::mr_aha_plan_execute many times without
 * re-planning or re-allocating.
 *
 * CUDA backend: for @f$L \le 30@f$ the block-diagonal stage uses kernels
 * specialized on @f$L@f$, and coils are processed in groups whose padded
 * grids are transformed by one batched cuFFT and multiplied by one kernel
 * pass, which reads `phi2_toep` once per group instead of once per coil. The
 * plan holds `group * Mx*My*Mz*L` ::mr_complex of padded grids, the cuFFT work
 * area and, when masked, an `Mx*My*Mz` int32 index map (with the generic
 * kernels: a `num_masked*L` ::mr_complex buffer instead). The group size is 4
 * (environment variable `MR_AHA_CUDA_COIL_GROUP` overrides it), at most
 * ::mr_config::n_coils, and is reduced until these buffers fit in
 * ::mr_config::mem_fraction of the free device memory at plan creation; a
 * group of 1 (`Mx*My*Mz*L` values, the minimum) is always attempted. Raise
 * `mem_fraction` to allow larger groups on a GPU with little memory to spare.
 * `MR_AHA_CUDA_GENERIC_BD=1` selects the generic block-diagonal kernels (one
 * coil at a time; also used for @f$L > 512@f$, and for masked plans with
 * more than @f$2^{31}-1@f$ retained locations) and `MR_AHA_CUDA_PROFILE=1`
 * prints the GPU time of each stage at every execution. Ranks
 * @f$30 < L \le 512@f$ use tiled shared-memory kernels that keep the coil
 * groups and the read-once guarantee of the specialized ones.
 *
 * The call validates @p cfg, makes ::mr_config::gpu_device_id current for
 * its duration (restoring the caller's device on return), queries the free
 * memory once, and allocates everything up front. It issues no work on
 * ::mr_config::stream.
 *
 * @param[in]  cfg      Validated configuration. Copied into the plan; the
 *                      caller's struct need not outlive the plan.
 * @param[out] plan_out Receives the new plan. Set to NULL on failure.
 *                      Release it with ::mr_aha_plan_destroy.
 * @return ::MR_SUCCESS, ::MR_ERR_INVALID_ARG (NULL pointer),
 *         ::MR_ERR_INVALID_CONFIG, ::MR_ERR_ALLOC (even a group of one coil
 *         does not fit), ::MR_ERR_CUFFT, ::MR_ERR_CUDA or ::MR_ERR_UNSUPPORTED
 *         (generic kernels only: one thread's rank vector exceeds the shared
 *         memory per block).
 */
MR_EXPORT int mr_aha_plan_create(const mr_config *cfg, mr_aha_plan *plan_out);

/**
 * @brief Apply @f$A^H A@f$ to a set of spatial coefficients.
 *
 * Implements, looping over coils,
 * @f[
 *   A^H A x \;=\; \sum_c S_c^H \, \mathcal{P}^H \, F^{-1} \,
 *                 \hat{T} \, F \, \mathcal{P} \, S_c \, x ,
 * @f]
 * where @f$\mathcal{P}@f$ zero-pads the image onto the upsampled Toeplitz grid
 * (with the centering index remap folded in), @f$F@f$ is the centered DFT over
 * the upsampled grid, and @f$\hat{T}@f$ is the block-diagonal multiplication by
 * the Hermitian @f$L \times L@f$ block @f$B@f$ held in `phi2_toep` at every
 * upsampled k-space location, applied to the rank row
 * vector from the right, @f$v \leftarrow v B@f$ (see @ref mr_gram). The
 * scaling of `phi2_toep` produced by ::mr_calc_toep
 * already absorbs the @f$1/(M_xM_yM_z)@f$ of the unnormalized cuFFT pair, so no
 * further scaling is applied here.
 *
 * Per coil group the stages are: (1) multiply by the sensitivity maps and
 * scatter onto the zeroed padded grid in FFT order, (2) one batched forward
 * cuFFT over all ranks and coils of the group, (3) the block-diagonal
 * multiplication, (4) the batched inverse cuFFT, (5) crop, multiply by the
 * conjugate maps and sum over coils into @p d_AhAx.
 *
 * Asynchronous with respect to the host: all work is enqueued on the plan's
 * stream (::mr_config::stream) and the call returns before it completes
 * (except with ::mr_config::verbose, which synchronizes). The input arrays
 * must stay valid and unchanged until the stream has reached that point.
 * Different @p d_SEs, @p d_phi2_toep and @p d_mask may be passed on every
 * call as long as their sizes match the plan's configuration.
 *
 * @param[in]  plan        Plan from ::mr_aha_plan_create.
 * @param[in]  d_x         Device array of spatial coefficients,
 *                         `[Nx, Ny, L]` (2D) or `[Nx, Ny, Nz, L]` (3D/stack).
 * @param[in]  d_SEs       Device array of sensitivity maps,
 *                         `[Nx, Ny, Nz, Ncoils]` (`[Nx, Ny, Ncoils]` in 2D).
 *                         One map per coil; it is broadcast across all
 *                         @f$L@f$ ranks.
 * @param[in]  d_phi2_toep Device array holding the packed Hermitian blocks:
 *                         `[Mx, My, Mz, (L^2+L)/2]` when unmasked, or
 *                         `[num_masked, (L^2+L)/2]` when masked.
 * @param[in]  d_mask      Device array of ::mr_config::num_masked **0-based**
 *                         linear indices into the centered upsampled grid
 *                         `[Mx, My, Mz]`. Must be NULL when
 *                         ::mr_config::is_masked is zero, non-NULL otherwise.
 * @param[out] d_AhAx      Device output array, same shape and size as @p d_x.
 *                         Fully overwritten; need not be pre-zeroed.
 * @return ::MR_SUCCESS, ::MR_ERR_INVALID_ARG (NULL plan or array, aliasing,
 *         missing mask), ::MR_ERR_CUDA / ::MR_ERR_CUFFT for launch and
 *         cuFFT failures, or ::MR_ERR_UNSUPPORTED from the block-diagonal
 *         kernel launcher. Errors raised asynchronously by the kernels
 *         surface only at a later synchronization.
 *
 * @note @p d_AhAx must not alias @p d_x.
 */
MR_EXPORT int mr_aha_plan_execute(mr_aha_plan         plan,
                                  const mr_complex   *d_x,
                                  const mr_complex   *d_SEs,
                                  const mr_complex   *d_phi2_toep,
                                  const int64_t      *d_mask,
                                  mr_complex         *d_AhAx);

/**
 * @brief Release a plan and all device memory it owns.
 *
 * Frees the cuFFT plans, the padded grids, the cuFFT work area and the mask
 * buffers on the plan's device, then the plan itself. It does not
 * synchronize the stream: `cudaFree` already waits for outstanding work on
 * the device. The caller's input and output arrays are never touched.
 *
 * @param[in] plan Plan from ::mr_aha_plan_create, or NULL (no-op).
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
MR_EXPORT int mr_aha_plan_destroy(mr_aha_plan plan);

/**
 * @brief One-shot convenience wrapper: create, execute and destroy a plan.
 *
 * Equivalent to ::mr_aha_plan_create followed by ::mr_aha_plan_execute and
 * ::mr_aha_plan_destroy. Prefer the plan API inside an iterative loop:
 * every call here re-plans cuFFT and re-allocates the padded grids.
 *
 * @param[in]  cfg         Validated configuration.
 * @param[in]  d_x         See ::mr_aha_plan_execute.
 * @param[in]  d_SEs       See ::mr_aha_plan_execute.
 * @param[in]  d_phi2_toep See ::mr_aha_plan_execute.
 * @param[in]  d_mask      See ::mr_aha_plan_execute.
 * @param[out] d_AhAx      See ::mr_aha_plan_execute.
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
MR_EXPORT int mr_aha_toeplitz(const mr_config  *cfg,
                              const mr_complex *d_x,
                              const mr_complex *d_SEs,
                              const mr_complex *d_phi2_toep,
                              const int64_t    *d_mask,
                              mr_complex       *d_AhAx);

/* ------------------------------------------------------------------------ */
/* The A^H A operator object (single or multiple GPUs)                       */
/* ------------------------------------------------------------------------ */

/**
 * @defgroup mr_op The A^H A operator object
 * @brief One stateful operator, created from the Gram blocks, single- or
 *        multi-GPU on one node.
 *
 * ::mr_op_create builds the complete operator
 * @f$x \mapsto A^H A x@f$ from the trajectory, the Gram blocks and the
 * sensitivity maps, and ::mr_op_execute applies it. The object owns (or, with
 * ::MR_OP_SHARE_PRIMARY, references) everything it needs, so an application
 * only takes @f$x@f$. Two methods evaluate the same operator:
 *
 * - ::MR_OP_TOEPLITZ (default): the Toeplitz embedding. The object computes
 *   the kernel `phi2_toep` (::mr_calc_toep) at creation and applies it with
 *   ::mr_aha_plan_execute; no non-uniform FFT is needed per application.
 * - ::MR_OP_NUFFT: the direct route, per coil a type-2 NUFFT (image to
 *   trajectory), the block-diagonal Gram multiplication on the samples
 *   (see @ref mr_gram) and a type-1 NUFFT back,
 *   with cufinufft. No kernel; less memory, more time per application. This
 *   is the baseline the Toeplitz method is compared against.
 *
 * @section mr_op_devices Devices
 *
 * The *primary* device is ::mr_config::gpu_device_id, or the current device
 * when that is -1 (MATLAB gpuArray and CuPy callers). Every array argument of
 * ::mr_op_create and ::mr_op_execute lives there. ::mr_op_options::devices
 * selects further devices of the same node; the primary always takes part.
 * By default (no devices) the operator runs on the primary device only, with
 * no overhead over the single-GPU primitives: no worker streams, no peer
 * probes, no copy of @f$x@f$.
 *
 * On several devices, @f$A^H A x = \sum_c (\ldots)@f$ is split by **coils**:
 * each device applies its own contiguous range of whole coils and the partial
 * results are summed on the primary device. With @f$C@f$ coils and @f$G@f$
 * candidate devices the operator uses the smallest @f$G' \le G@f$ with the
 * same @f$\lceil C/G' \rceil@f$ coils per device (12 coils on 5 GPUs use 4),
 * and only the primary for problems too small to gain (fewer than
 * `MR_AHA_MGPU_MIN_WORK` upsampled values per coil, default @f$2^{22}@f$;
 * the environment variable overrides it, 0 disables the check).
 * ::mr_op_info reports the choice.
 *
 * Toeplitz on several devices: every device that applies the operator holds
 * the whole kernel. How the copies are produced is ::mr_op_options::toep_dist
 * (see ::mr_toep_dist). With the Toeplitz method the coils that do not divide
 * evenly over the devices can also be split by rank
 * (`MR_AHA_MGPU_SPLIT=auto|on|off`, default `auto`: only when a cost model
 * fed with the measured peer bandwidth predicts a gain, i.e. with NVLink).
 *
 * NUFFT on several devices: every device holds its own cufinufft plans, its
 * copy of the (reordered) Gram blocks and trajectory and its coils' maps; no
 * kernel, so the per-device memory is far below the Toeplitz method's.
 *
 * @section mr_op_backends Backends
 *
 * CUDA implements everything. The Metal backend implements ::MR_OP_TOEPLITZ
 * on a single device with a supplied kernel (::mr_op_inputs::d_phi2_toep):
 * its kernel pre-computation runs partly outside the library (see
 * ::mr_calc_toep_from_psf), so its Python binding computes the kernel and
 * passes it in. Anything else returns ::MR_ERR_UNSUPPORTED on Metal.
 * @{
 */

/* Device-selection limits and creation flags of ::mr_op_options. */

/** @brief Most devices one operator can use (also the length of the
 *         per-device arrays of ::mr_op_info_t). */
#define MR_OP_MAX_DEVICES 16

/** @brief ::mr_op_options::n_devices value selecting every visible device
 *         (with ::mr_op_options::devices NULL). */
#define MR_OP_ALL_DEVICES (-1)

/**
 * @brief ::mr_op_options::flags bit: on the primary device, use the caller's
 *        input arrays in place instead of copying them.
 *
 * The primary device then reads the caller's `d_SEs`, `d_mask` and, when
 * supplied, `d_phi2_toep` directly; the other devices always receive copies
 * (the data must be moved there anyway). This keeps the single-device memory
 * equal to the stateless primitives (no second copy of the maps). The
 * caller must keep these arrays alive and unchanged until ::mr_op_destroy.
 * Set by ::mr_op_options_defaults.
 *
 * Arrays the operator transforms at creation (the kernel it computes, the
 * NUFFT method's reordered Gram blocks and trajectory) always belong to the
 * operator, so `d_phi2` and `d_traj` may be released after ::mr_op_create
 * returns whatever this flag says.
 */
#define MR_OP_SHARE_PRIMARY 1

/** @brief How the operator evaluates @f$A^H A@f$. */
typedef enum mr_op_method {
    MR_OP_TOEPLITZ = 0, /**< Toeplitz embedding with a pre-computed kernel. */
    MR_OP_NUFFT    = 1  /**< Type-2 NUFFT, Gram multiplication, type-1 NUFFT
                             per coil (cufinufft); no kernel.               */
} mr_op_method;

/**
 * @brief How a multi-device Toeplitz operator puts the kernel on every device
 *        (::mr_op_options::toep_dist).
 *
 * The @f$(L^2+L)/2@f$ triangular components of `phi2_toep` are independent
 * (one type-1 NUFFT, inverse FFT and scaling each) and each fills a
 * contiguous slab, so they can be computed on any device and moved with one
 * copy per range. Without the operator's own choice, the environment
 * variable `MR_AHA_TOEP_SPLIT=auto|split|replicate|broadcast` overrides this
 * option (`off` is accepted as a synonym of `broadcast`); any other value is
 * an error. With one device all modes compute the kernel in place.
 */
typedef enum mr_toep_dist {
    /** Cost model: ::MR_TOEP_DIST_SPLIT or ::MR_TOEP_DIST_REPLICATE, fed with
     *  a timed first chunk on every device (useful work either way) and a
     *  concurrent bandwidth probe; roughly, the exchange wins when a device
     *  receives a component faster than it computes one. */
    MR_TOEP_DIST_AUTO      = 0,
    /** Every device computes a contiguous share of the components into its
     *  own copy, then the shares are exchanged (all-gather), overlapped chunk
     *  by chunk with the computation. Every component is computed once, so
     *  all copies are bit-identical. */
    MR_TOEP_DIST_SPLIT     = 1,
    /** Every device computes the whole kernel itself, without transfers
     *  (best when the devices cannot exchange data quickly). The copies agree
     *  to cufinufft's run-to-run noise (~1e-7). */
    MR_TOEP_DIST_REPLICATE = 2,
    /** The primary device computes the kernel alone and copies it to every
     *  other device (the slowest; kept for comparison). */
    MR_TOEP_DIST_BROADCAST = 3
} mr_toep_dist;

/**
 * @brief Inputs of ::mr_op_create. All arrays live on the primary device (see
 *        @ref mr_op_devices) and use the layouts of ::mr_calc_toep and
 *        ::mr_aha_plan_execute.
 */
typedef struct mr_op_inputs {
    /** Sensitivity maps `[Nx, Ny, Nz, Ncoils]` (`[Nx, Ny, Ncoils]` in 2D),
     *  ::mr_config::n_coils of them. Required. */
    const mr_complex *d_SEs;
    /** Gram blocks `[Ntrajs, L, L]` (`[Ntrajs, Nkz, L, L]` for
     *  ::MR_TRAJ_STACK), see ::mr_calc_toep. Required, except for
     *  ::MR_OP_TOEPLITZ with @ref d_phi2_toep given. */
    const mr_complex *d_phi2;
    /** Trajectory `[Ntrajs, Nkx, ndim]` in radians, see ::mr_calc_toep.
     *  Required whenever @ref d_phi2 is. */
    const float      *d_traj;
    /** Number of trajectories (readouts). */
    int64_t           Ntrajs;
    /** Samples per trajectory. */
    int64_t           Nkx;
    /** Mask of ::mr_config::num_masked 0-based indices into the centered
     *  upsampled grid when ::mr_config::is_masked (::MR_OP_TOEPLITZ only),
     *  else NULL. */
    const int64_t    *d_mask;
    /** Optional finished kernel (::mr_phi2_toep_size elements, as from
     *  ::mr_calc_toep) for ::MR_OP_TOEPLITZ; NULL to let the operator compute
     *  it from @ref d_phi2 and @ref d_traj. Meant for the Metal backend (whose
     *  kernel pre-computation runs partly outside the library) and for tests
     *  that inject a reference kernel; the language bindings do not expose it
     *  publicly. With ::MR_OP_SHARE_PRIMARY the primary device uses it in
     *  place; otherwise, and on every other device, it is copied
     *  (::mr_op_info_t::toep_source is ::MR_OP_TOEP_SUPPLIED). */
    const mr_complex *d_phi2_toep;
} mr_op_inputs;

/**
 * @brief Options of ::mr_op_create. Fill with ::mr_op_options_defaults, then
 *        override fields; new fields are added at the end with defaults that
 *        keep the previous behavior.
 */
typedef struct mr_op_options {
    /** Method, ::MR_OP_TOEPLITZ by default. */
    mr_op_method method;
    /** CUDA ordinals of further devices, or NULL. With NULL, @ref n_devices
     *  0 means the primary device only (default) and ::MR_OP_ALL_DEVICES
     *  every visible device (at most ::MR_OP_MAX_DEVICES). Duplicates and
     *  the primary itself are ignored; the primary always takes part. */
    const int   *devices;
    /** Length of @ref devices, at most ::MR_OP_MAX_DEVICES; see there. */
    int          n_devices;
    /** Kernel distribution of a multi-device ::MR_OP_TOEPLITZ operator,
     *  ::MR_TOEP_DIST_AUTO by default. See ::mr_toep_dist. */
    mr_toep_dist toep_dist;
    /** Bit set of ::MR_OP_SHARE_PRIMARY; the defaults set it. */
    int          flags;
    /** ::MR_OP_NUFFT: cufinufft upsampling factor, 0 for the default 2.0
     *  (the setting of the reference operators). The tolerance is
     *  ::mr_config::eps. */
    double       nufft_upsampfac;
    /** ::MR_OP_NUFFT: cufinufft `gpu_method` for both transforms (1:
     *  non-uniform points driven, 2: shared memory). 0 (default) selects the
     *  library default: method 1 for the type-1 transform (faster than
     *  cufinufft's own shared-memory choice in the benchmarks, at equal
     *  accuracy) and cufinufft's choice (method 1) for the type-2 one. */
    int          nufft_gpu_method;
} mr_op_options;

/** @brief Opaque @f$A^H A@f$ operator. Create with ::mr_op_create. */
typedef struct mr_op_s *mr_op;

/* Values of ::mr_op_info_t::toep_source, reporting where the kernel copies
 * of a Toeplitz operator came from. Plain macros (not an enum) because the
 * field is a fixed-width `int` in a struct read by foreign-language
 * bindings. */

/** @brief ::mr_op_info_t::toep_source: the caller supplied the kernel
 *         (::mr_op_inputs::d_phi2_toep). */
#define MR_OP_TOEP_SUPPLIED  0
/** @brief ::mr_op_info_t::toep_source: the primary device computed the kernel
 *         alone (and copied it to the other devices, if any). Always the
 *         case on one device. */
#define MR_OP_TOEP_BROADCAST 1
/** @brief ::mr_op_info_t::toep_source: every device computed a share of the
 *         components and the shares were exchanged (::MR_TOEP_DIST_SPLIT). */
#define MR_OP_TOEP_SPLIT     2
/** @brief ::mr_op_info_t::toep_source: every device computed the kernel
 *         itself (::MR_TOEP_DIST_REPLICATE). */
#define MR_OP_TOEP_REPLICATE 3
/** @brief ::mr_op_info_t::toep_source: no kernel (::MR_OP_NUFFT). */
#define MR_OP_TOEP_NONE      4

/**
 * @brief How an operator was built and how it distributes the work
 *        (::mr_op_info). Device 0 is the primary device: it holds the input
 *        and output arrays and reduces the partial results.
 */
typedef struct mr_op_info_t {
    /** ::mr_op_method of the operator. */
    int    method;
    /** Number of devices actually used (<= the number requested, see
     *  @ref mr_op_devices). */
    int    n_devices;
    /** CUDA ordinals of the devices used; `devices[0]` is the primary. */
    int    devices[MR_OP_MAX_DEVICES];
    /** First whole coil processed by each device. */
    int    coil_begin[MR_OP_MAX_DEVICES];
    /** Number of whole coils processed by each device. */
    int    coil_count[MR_OP_MAX_DEVICES];
    /** Coils per group of each device (::MR_OP_TOEPLITZ, see
     *  ::mr_aha_plan_create; 1 for ::MR_OP_NUFFT, which transforms one coil
     *  at a time). */
    int    group[MR_OP_MAX_DEVICES];
    /** Number of remainder coils split by rank across all devices
     *  (::MR_OP_TOEPLITZ only; 0 when the remainder split is off). */
    int    n_split_coils;
    /** Non-zero when direct peer access is enabled between the primary and
     *  every other device used (otherwise copies are staged through the
     *  host by the CUDA runtime). 0 on one device. */
    int    peer_access;
    /** Slowest measured primary <-> device copy bandwidth in GB/s, or 0 when
     *  it was not measured (it is measured only for the cost model of the
     *  remainder split, i.e. a Toeplitz operator with C % G != 0 and
     *  ``MR_AHA_MGPU_SPLIT`` unset or ``auto``). */
    double peer_bandwidth_gbs;
    /** How the kernel was obtained: ::MR_OP_TOEP_SUPPLIED,
     *  ::MR_OP_TOEP_BROADCAST, ::MR_OP_TOEP_SPLIT, ::MR_OP_TOEP_REPLICATE or
     *  ::MR_OP_TOEP_NONE. */
    int    toep_source;
    /** Devices that computed components of the kernel (0 when supplied or
     *  ::MR_OP_NUFFT). */
    int    toep_devices;
    /** Wall time of computing the kernel and, on several devices,
     *  distributing it, in ms (0 when supplied or ::MR_OP_NUFFT). */
    double toep_ms;
    /** GPU time per triangular component measured by the pilot chunks of
     *  ::MR_TOEP_DIST_AUTO / ::MR_TOEP_DIST_SPLIT / ::MR_TOEP_DIST_REPLICATE,
     *  in ms (0 when not measured). */
    double toep_component_ms;
    /** Concurrent receive bandwidth per device measured for the cost model
     *  of the kernel exchange, in GB/s (0 when not measured). */
    double toep_gather_gbs;
    /** Wall time of the whole ::mr_op_create, in ms. */
    double create_ms;
} mr_op_info_t;

/**
 * @brief Fill @p opt with the defaults: ::MR_OP_TOEPLITZ, the primary device
 *        only, ::MR_TOEP_DIST_AUTO, ::MR_OP_SHARE_PRIMARY, cufinufft's NUFFT
 *        defaults with an upsampling factor of 2.
 *
 * The struct is zeroed first, so `devices = NULL`, `n_devices = 0`,
 * `nufft_upsampfac = 0` (meaning 2.0) and `nufft_gpu_method = 0` (library
 * choice). Fields added in later versions are thereby given their
 * backwards-compatible defaults as well.
 *
 * @param[out] opt Options to initialize. Must be non-NULL.
 * @return ::MR_SUCCESS or ::MR_ERR_INVALID_ARG.
 */
MR_EXPORT int mr_op_options_defaults(mr_op_options *opt);

/**
 * @brief Create an @f$A^H A@f$ operator from the Gram blocks.
 *
 * Validates @p cfg, selects the devices (see @ref mr_op_devices) and builds,
 * on every device used, the constants and the per-device engine:
 *
 * - ::MR_OP_TOEPLITZ: the kernel `phi2_toep` (computed with ::mr_calc_toep,
 *   distributed per ::mr_op_options::toep_dist, or the supplied
 *   ::mr_op_inputs::d_phi2_toep), the device's maps and mask, and a
 *   ::mr_aha_plan for its coils (coil groups, cuFFT plans, specialized
 *   block-diagonal kernels). Memory per device: the kernel
 *   (::mr_phi2_toep_size values) plus the plan's buffers, sized within
 *   ::mr_config::mem_fraction; while it is computed, also the transients of
 *   ::mr_calc_toep.
 * - ::MR_OP_NUFFT: one type-2 (sign -1) and one type-1 (sign +1) cufinufft
 *   plan with ::mr_config::eps, centered modes `n_dim` and `L` transforms
 *   (`L*Nz` for ::MR_TRAJ_STACK, with a cuFFT along z), the points set once,
 *   the Gram blocks reordered for a batched GEMM (cuBLAS), and buffers of
 *   one coil: the image (`Nx*Ny*Nz*L` values) and twice the samples
 *   (`Ntrajs*Nkx*L*Nkz` values). Masked configurations are rejected
 *   (::MR_ERR_INVALID_ARG): the NUFFT route has no kernel to mask.
 *
 * The call is synchronous: it waits for ::mr_config::stream first (inputs
 * produced there are complete) and returns once the operator is ready. On
 * return the current device is the same as on entry.
 *
 * @param[in]  cfg    Validated configuration; copied. ::mr_config::stream
 *                    (on the primary device) is the stream ::mr_op_execute
 *                    is ordered on. ::mr_config::n_coils maps are used.
 * @param[in]  in     Input arrays, see ::mr_op_inputs.
 * @param[in]  opt    Options, or NULL for ::mr_op_options_defaults.
 * @param[out] op_out Receives the operator; NULL on failure.
 * @return ::MR_SUCCESS or a negative ::mr_status. A failure on another device
 *         names its ordinal in ::mr_last_error.
 */
MR_EXPORT int mr_op_create(const mr_config     *cfg,
                           const mr_op_inputs  *in,
                           const mr_op_options *opt,
                           mr_op               *op_out);

/**
 * @brief Apply the operator: @p d_AhAx @f$= A^H A@f$ @p d_x.
 *
 * Asynchronous with respect to the host, like ::mr_aha_plan_execute: the work
 * is ordered after everything previously issued on ::mr_config::stream (of
 * the primary device) and the result is complete once that stream reaches the
 * point of return. On several devices, @p d_x is copied to every other
 * device, each applies its coils, and the partial results are summed into
 * @p d_AhAx on the primary device.
 *
 * @param[in]  op     Operator from ::mr_op_create.
 * @param[in]  d_x    Spatial coefficients on the primary device,
 *                    `[Nx, Ny, L]` (2D) or `[Nx, Ny, Nz, L]` (3D/stack).
 * @param[out] d_AhAx Output on the primary device, same size as @p d_x;
 *                    fully overwritten. Must not alias @p d_x.
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
MR_EXPORT int mr_op_execute(mr_op op, const mr_complex *d_x, mr_complex *d_AhAx);

/**
 * @brief Describe how an operator was built and distributes the work.
 *
 * Copies the report recorded by ::mr_op_create; host-only, no GPU work and
 * no synchronization.
 *
 * @param[in]  op   Operator.
 * @param[out] info Receives the description (a full struct copy).
 * @return ::MR_SUCCESS or ::MR_ERR_INVALID_ARG (@p op or @p info NULL).
 */
MR_EXPORT int mr_op_info(mr_op op, mr_op_info_t *info);

/**
 * @brief The copy of `phi2_toep` a Toeplitz operator holds on one of its
 *        devices.
 *
 * This is how a caller reaches the kernel the operator computed (e.g. to
 * save it). The memory belongs to the operator (or is the caller's supplied
 * kernel shared on the primary): it stays valid, and must not be modified,
 * until ::mr_op_destroy.
 *
 * @param[in]  op          Operator.
 * @param[in]  k           Device index, `0 <= k < ` ::mr_op_info_t::n_devices
 *                         (0 is the primary; the ordinal is
 *                         ::mr_op_info_t::devices[k]).
 * @param[out] d_phi2_toep Receives the pointer (::mr_phi2_toep_size elements
 *                         on that device).
 * @return ::MR_SUCCESS, ::MR_ERR_INVALID_ARG (NULL argument or @p k out of
 *         range), or ::MR_ERR_UNSUPPORTED for an
 *         ::MR_OP_NUFFT operator (which has no kernel).
 */
MR_EXPORT int mr_op_kernel(mr_op op, int k, const mr_complex **d_phi2_toep);

/**
 * @brief Release an operator and everything it allocated on every device.
 *
 * Waits for work still in flight on the operator's streams (and on
 * ::mr_config::stream) first.
 *
 * @param[in] op Operator, or NULL (no-op).
 * @return ::MR_SUCCESS.
 */
MR_EXPORT int mr_op_destroy(mr_op op);

/**
 * @brief Release an operator whose primary device was reset.
 *
 * After `cudaDeviceReset` on the primary device (e.g. MATLAB's
 * `reset(gpuDevice)`), the operator's streams, events, plans and buffers there
 * no longer exist and ::mr_op_destroy would use dead handles. This variant
 * frees everything the operator holds on the *other* devices and skips the
 * primary, whose resources the reset has already released. Devices whose
 * context was re-created since (detected through the context ID) are skipped
 * as well by both functions.
 *
 * @param[in] op Operator, or NULL (no-op).
 * @return ::MR_SUCCESS.
 */
MR_EXPORT int mr_op_abandon(mr_op op);

/** @} */

/* ------------------------------------------------------------------------ */
/* Toeplitz kernel pre-computation                                          */
/* ------------------------------------------------------------------------ */

/**
 * @brief Pre-compute the block-diagonal Toeplitz operator `phi2_toep`.
 *
 * Given the non-Cartesian Gram blocks @f$\Phi_2 = \Phi\Phi^H@f$ (rank x time
 * @f$\Phi@f$, see @ref mr_gram) sampled along the trajectory, this routine
 *   1. packs each @f$L \times L@f$ block into the requested triangle,
 *   2. evaluates a type-1 cufinufft (sign @f$-1@f$) of the packed strengths
 *      onto the upsampled grid `phi2_dim`, giving the point-spread function,
 *   3. applies a centered **inverse** FFT (index remap + unnormalized cuFFT +
 *      @f$1/\prod M@f$), and
 *   4. optionally gathers the result at the masked locations.
 *
 * Step 2 with sign @f$-1@f$ followed by the normalized inverse FFT of step 3
 * is algebraically identical to the forward FFT of the sign @f$+1@f$
 * point-spread function divided by @f$\prod M@f$, which is exactly the
 * multiplier ::mr_aha_plan_execute needs between its unnormalized forward and
 * inverse cuFFTs. No further scaling is required anywhere.
 *
 * For ::MR_TRAJ_STACK the type-1 transform is two-dimensional and is repeated
 * for every kz encode; no transform and no shift is applied along z, because kz
 * is already Cartesian and `phi2` is supplied in centered kz order.
 *
 * Work is chunked over the @f$(L^2+L)/2@f$ triangular components so that the
 * transient buffers respect ::mr_config::mem_fraction of the free device
 * memory.
 *
 * @param[in]  cfg          Validated configuration. ::mr_config::phi2_dim gives
 *                          the upsampled grid the kernel is evaluated on.
 * @param[in]  d_phi2       Device array of Gram blocks,
 *                          `[Ntrajs, L, L]` for ::MR_TRAJ_2D / ::MR_TRAJ_3D or
 *                          `[Ntrajs, Nkz, L, L]` for ::MR_TRAJ_STACK, where the
 *                          trailing `[L, L]` axes hold the (column-major)
 *                          @f$L\times L@f$ Gram block
 *                          @f$\Phi_2 = \Phi\Phi^H@f$ of that trajectory
 *                          (MATLAB `Phi_rt(:,idx) * Phi_rt(:,idx)'`, see
 *                          @ref mr_gram) and `Nkz == Mz == Nz`. Gram
 *                          blocks stored `[L, L, Ntrajs, Nz]` are
 *                          `permute(Phi2, [3 4 1 2])` (plus a kz fftshift
 *                          if its kz order is natural).
 * @param[in]  d_traj       Device array of non-Cartesian coordinates,
 *                          `[Ntrajs, Nkx, ndim]`, in **radians per unit of the
 *                          un-padded image grid**, i.e. the usual
 *                          @f$[-\pi, \pi)@f$ range. `ndim` is 2 for
 *                          ::MR_TRAJ_2D and ::MR_TRAJ_STACK, 3 for
 *                          ::MR_TRAJ_3D.
 * @param[in]  Ntrajs       Number of non-Cartesian trajectories (e.g. spokes).
 *                          One Gram block is shared by the whole readout.
 * @param[in]  Nkx          Number of k-space samples per trajectory.
 * @param[in]  d_mask       Device array of ::mr_config::num_masked **0-based**
 *                          linear indices into the centered upsampled grid, or
 *                          NULL when ::mr_config::is_masked is zero.
 * @param[out] d_phi2_toep  Device output array of ::mr_phi2_toep_size elements:
 *                          `[Mx, My, Mz, (L^2+L)/2]` when unmasked, or
 *                          `[num_masked, (L^2+L)/2]` when masked.
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
MR_EXPORT int mr_calc_toep(const mr_config  *cfg,
                           const mr_complex *d_phi2,
                           const float      *d_traj,
                           int64_t           Ntrajs,
                           int64_t           Nkx,
                           const int64_t    *d_mask,
                           mr_complex       *d_phi2_toep);

/**
 * @brief Finish `phi2_toep` from externally computed point-spread functions.
 *
 * Steps 3 and 4 of ::mr_calc_toep (centered inverse FFT with the
 * @f$1/\prod M@f$ scaling, then the optional mask gather) for a batch of
 * triangular components whose type-1 NUFFT (step 2) was computed outside the
 * library. This is how the Metal backend builds `phi2_toep`: the type-1 NUFFT
 * runs in Python through mlx-nufft (`mr_aha.calc_toep`), and this function
 * does the rest on the GPU. Both backends implement it, so the two-stage path
 * can be validated against the one-shot ::mr_calc_toep on CUDA as well.
 *
 * Component @f$t@f$ of the point-spread function is the type-1 NUFFT, with
 * sign @f$-1@f$, centered (CMCL, `modeord = 0`) mode order and modes
 * `phi2_dim`, of the strengths obtained by broadcasting stored triangular
 * element @f$t@f$ (see ::mr_triangulation) of every trajectory's Gram block
 * along its readout:
 * @f[
 *   \mathrm{psf}_t[\mathbf{k}] = \sum_{m} c_{t,m}\, e^{-i \mathbf{k}\cdot
 *   \mathbf{x}_m}, \qquad \mathbf{k} \in \textstyle\prod_d
 *   [-\lfloor M_d/2 \rfloor, \lceil M_d/2 \rceil).
 * @f]
 * For ::MR_TRAJ_STACK the transform is two-dimensional and repeated for every
 * kz encode, giving one `[Mx, My]` plane per kz.
 *
 * @param[in]  cfg         Validated configuration of the full operator
 *                         (::mr_config::is_masked and ::mr_config::num_masked
 *                         describe @p d_phi2_toep).
 * @param[in]  d_psf       @p n_comp consecutive point-spread functions, each
 *                         `[Mx, My, Mz]` in centered order (`Mz` = kz planes for
 *                         a stack, 1 in 2D): `[Mx, My, Mz, n_comp]`. Not
 *                         modified.
 * @param[in]  t0          Index of the first component, `0 <= t0`.
 * @param[in]  n_comp      Number of components, `t0 + n_comp <= (L^2+L)/2`.
 * @param[in]  d_mask      Mask of ::mr_config::num_masked indices, or NULL
 *                         when ::mr_config::is_masked is zero.
 * @param[out] d_phi2_toep The **whole** output array of ::mr_calc_toep;
 *                         components `[t0, t0 + n_comp)` are written, the
 *                         others are left untouched, so the components can be
 *                         produced in any number of calls.
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
MR_EXPORT int mr_calc_toep_from_psf(const mr_config  *cfg,
                                    const mr_complex *d_psf,
                                    int64_t           t0,
                                    int64_t           n_comp,
                                    const int64_t    *d_mask,
                                    mr_complex       *d_phi2_toep);

/* ------------------------------------------------------------------------ */
/* Automatic k-space mask                                                   */
/* ------------------------------------------------------------------------ */

/**
 * @brief Derive the masked-operator sample set directly from the trajectory.
 *
 * Every sample @f$\mathbf{k}@f$ (radians) is mapped to its nearest point on
 * the upsampled grid, `rint(k * M / (2*pi)) + M/2` per transformed axis
 * (modulo `M`, the grid is periodic), those points are marked, and the marks
 * are dilated by a @f$(2r+1)^d@f$ box with @f$r@f$ =
 * ::mr_config::mask_radius. Every location within @f$r@f$ cells (Chebyshev
 * distance) of a sample's grid point is kept. No NUFFT is needed. Samples lie
 * between grid points and each one's Dirichlet kernel spreads over the
 * neighboring cells, so @f$r = 0@f$ is far too small; the default
 * @f$r = 2@f$ reaches the unmasked accuracy on densely sampled radial and 3D
 * trajectories while keeping roughly the sampled disk / ball of k-space.
 *
 * Locations far from every sample carry little of the Toeplitz kernel, so
 * dropping them from `phi2_toep` saves memory with an error controlled by the
 * radius.
 *
 * For ::MR_TRAJ_STACK the in-plane mask is replicated over all `Mz` kz
 * planes, because every kz encode uses the same 2D trajectory.
 *
 * The result is directly usable as the `d_mask` argument of ::mr_calc_toep and
 * ::mr_aha_plan_execute after setting ::mr_config::is_masked and
 * ::mr_config::num_masked.
 *
 * @param[in]  cfg            Validated configuration. Only
 *                            ::mr_config::traj_type, ::mr_config::n_dim,
 *                            ::mr_config::phi2_dim, ::mr_config::mask_radius
 *                            and the execution fields (device, stream,
 *                            verbose) are used; ::mr_config::L and
 *                            ::mr_config::is_masked are ignored.
 * @param[in]  d_traj         Device trajectory `[Ntrajs, Nkx, ndim]`, exactly
 *                            as for ::mr_calc_toep.
 * @param[in]  Ntrajs         Number of trajectories.
 * @param[in]  Nkx            Samples per trajectory.
 * @param[out] d_mask         Device array receiving the retained **0-based**
 *                            linear indices into the centered upsampled grid
 *                            `[Mx, My, Mz]`, in ascending order. Must have room
 *                            for ::mr_pad_size entries (the worst case). May be
 *                            NULL to only count the retained locations.
 * @param[out] num_masked_out Host pointer receiving the number of retained
 *                            locations. Must be non-NULL.
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
MR_EXPORT int mr_calc_mask(const mr_config *cfg,
                           const float     *d_traj,
                           int64_t          Ntrajs,
                           int64_t          Nkx,
                           int64_t         *d_mask,
                           int64_t         *num_masked_out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MR_AHA_MR_AHA_H */
