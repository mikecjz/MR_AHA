/**
 * @file mr_aha.h
 * @brief Public C API of the MR_AHA library.
 *
 * MR_AHA evaluates the forward normal operator @f$A^H A@f$ for non-Cartesian
 * MRI with low-rank (subspace / MR-Multitasking) modelling on the GPU, using a
 * Toeplitz embedding so that no non-uniform FFT is required inside the
 * iteration loop.
 *
 * The library is a standalone CUDA/C++ library with a pure C interface
 * (`extern "C"`), so that it can be linked directly from C, C++, a MATLAB MEX
 * gateway (MATLAB GPU C API) or a Python `ctypes` binding.
 *
 * @section mr_layout Array layout
 *
 * Every multi-dimensional array documented in this header uses **column-major**
 * order (the MATLAB / Fortran convention). A row-major caller (NumPy, CuPy)
 * must simply reverse the dimension order: a column-major `[Nx, Ny, Nz, L]`
 * array is a row-major `[L, Nz, Ny, Nx]` array with the *same* linear memory
 * layout, so no data movement is ever required.
 *
 * All image-space and k-space arrays are assumed to be **centred**, i.e. the
 * origin (DC / image centre) sits in the middle of the array at index
 * `floor(dim/2)`, not in the corner. The library performs the required
 * (i)fftshifts internally, by index remapping rather than by phase modulation,
 * so the convention is exact for both even and odd dimensions.
 *
 * @section mr_memory Memory ownership
 *
 * **All array arguments are device (GPU) pointers.** The library never
 * allocates or touches host buffers for bulk data; the caller is responsible
 * for allocating and freeing every input and output array on the device. Small
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

#include <cuComplex.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#  ifdef MR_AHA_BUILD_SHARED
#    define MR_EXPORT __declspec(dllexport)
#  else
#    define MR_EXPORT
#  endif
#else
#  define MR_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Single-precision complex scalar used by every MR_AHA array. */
typedef cuFloatComplex mr_complex;

/**
 * @brief Default neighbourhood radius of ::mr_calc_mask, in upsampled grid cells.
 *
 * Used by the default method ::MR_MASK_NEIGHBOUR. Measured in
 * `benchmarks/neighbour_mask.md`: with radius 2 the masked operator matches
 * the unmasked one to within the single-precision NUFFT tolerance on
 * image-like inputs for the 280-spoke radial (79 % of the grid kept) and 3D
 * kooshball (54 % kept) test problems. On the 377-spoke 160^2 radial data
 * the image-input error is 5e-5 (80 % kept), and on sparse trajectories
 * (64 radial spokes, two-arm spiral) it is 5e-4 to 3e-2; increase
 * ::mr_config::mask_radius or use ::MR_MASK_THRESHOLD there. Radius 0 or 1
 * leaves gaps between the spokes near the edge of k-space.
 */
#define MR_DEFAULT_MASK_RADIUS 2

/**
 * @brief Default (absolute) threshold of ::mr_calc_mask, in units of samples.
 *
 * Used only by ::MR_MASK_THRESHOLD. Chosen experimentally (see
 * `benchmarks/README.md`) on the 2D radial, stack-of-stars and 3D kooshball
 * test problems: at 0.05 the masked operator differs from the exact one by a
 * few times the single-precision NUFFT tolerance on image-like inputs, while
 * the empty corners of the upsampled k-space grid (about 16-20 % of it in 2D,
 * 36 % in 3D) are discarded.
 */
#define MR_DEFAULT_MASK_THRESHOLD 0.05

/* ------------------------------------------------------------------------ */
/* Status codes                                                             */
/* ------------------------------------------------------------------------ */

/**
 * @brief Return codes of every MR_AHA entry point.
 *
 * Every function returns ::MR_SUCCESS (== 0) on success and a negative value on
 * failure. Use ::mr_status_string to obtain a human readable description.
 */
typedef enum mr_status {
    MR_SUCCESS            =  0, /**< Operation completed successfully.        */
    MR_ERR_INVALID_ARG    = -1, /**< A NULL pointer or nonsensical size.      */
    MR_ERR_INVALID_CONFIG = -2, /**< ::mr_config failed validation.           */
    MR_ERR_ALLOC          = -3, /**< A device allocation failed.              */
    MR_ERR_CUDA           = -4, /**< A CUDA runtime call failed.              */
    MR_ERR_CUFFT          = -5, /**< A cuFFT call failed.                     */
    MR_ERR_CUFINUFFT      = -6, /**< A cufinufft call failed.                 */
    MR_ERR_UNSUPPORTED    = -7, /**< Valid but not implemented combination.   */
    MR_ERR_INTERNAL       = -8  /**< Should not happen; please report.        */
} mr_status;

/**
 * @brief Human readable description of an ::mr_status value.
 * @param[in] status Value previously returned by an MR_AHA function.
 * @return Pointer to a static, NUL-terminated string. Never NULL.
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
 * @return Pointer to a thread-local, NUL-terminated string. Never NULL; empty
 *         if no failure has been recorded on this thread.
 */
MR_EXPORT const char *mr_last_error(void);

/* ------------------------------------------------------------------------ */
/* Configuration                                                            */
/* ------------------------------------------------------------------------ */

/** @brief Trajectory family, selecting how ::mr_calc_toep interprets its input. */
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
 * @brief How ::mr_calc_mask chooses the retained upsampled k-space locations.
 */
typedef enum mr_mask_method {
    /** Keep every location within ::mr_config::mask_radius grid cells
     *  (Chebyshev distance, i.e. a `(2r+1)^d` box) of the grid point nearest
     *  to a sample. Cheap (no NUFFT) and the default. */
    MR_MASK_NEIGHBOUR = 0,
    /** Keep every location where the max-filtered unit-weight Toeplitz kernel
     *  exceeds ::mr_config::mask_threshold. Costs one @f$L = 1@f$
     *  ::mr_calc_toep; follows the kernel's axis-aligned tails, so it is the
     *  better choice for very sparse trajectories (e.g. few-arm spirals). */
    MR_MASK_THRESHOLD = 1
} mr_mask_method;

/**
 * @brief Complete description of one @f$A^H A@f$ operator.
 *
 * Fill with ::mr_config_defaults first, then override the fields you need. Use
 * ::mr_config_validate to check consistency before creating a plan (the plan
 * constructors validate as well).
 */
typedef struct mr_config {
    /** Trajectory family. See ::mr_traj_type. */
    mr_traj_type traj_type;

    /** Number of *spatial* dimensions of the image: 2 or 3. Derived from
     *  ::traj_type by ::mr_config_defaults but may be set explicitly:
     *  ::MR_TRAJ_2D uses 2, ::MR_TRAJ_3D and ::MR_TRAJ_STACK use 3. */
    int n_spatial;

    /** Image matrix size `[Nx, Ny, Nz]`. Entries beyond ::n_spatial must be 1. */
    int n_dim[3];

    /** Upsampled (Toeplitz) matrix size `[Mx, My, Mz]`. Entries beyond
     *  ::n_spatial must be 1. Typically `M = 2*N`. For ::MR_TRAJ_STACK,
     *  `Mz` must equal `Nz` because kz is already Cartesian. */
    int phi2_dim[3];

    /** Number of ranks / subspace coefficients @f$L@f$. Must be >= 1. */
    int L;

    /** Triangle stored in `phi2_toep`. See ::mr_triangulation. */
    mr_triangulation triangulation;

    /** Non-zero when `phi2_toep` is stored only on a subset of the upsampled
     *  Cartesian grid; then `phi2_toep` is `[num_masked, (L^2+L)/2]` and a
     *  `mask` of ::num_masked linear indices must be supplied. */
    int is_masked;

    /** Number of retained samples when ::is_masked is non-zero, else ignored. */
    int64_t num_masked;

    /** Number of receive coils @f$N_{coils}@f$. Must be >= 1. Used only by
     *  ::mr_aha_toeplitz / ::mr_aha_plan_execute. */
    int n_coils;

    /** Requested cufinufft tolerance for ::mr_calc_toep. Default `1e-6`. */
    double eps;

    /** CUDA device to run on. Default 0. Set to -1 to keep the caller's
     *  current device (recommended when interoperating with MATLAB gpuArray or
     *  CuPy, which manage the current device themselves). */
    int gpu_device_id;

    /** CUDA stream (`cudaStream_t`) for all work, or NULL for the default
     *  stream. Declared `void *` to keep this header free of `cuda_runtime.h`. */
    void *stream;

    /** Fraction of the *free* device memory ::mr_calc_toep may use for its
     *  working buffers. Default 0.5. Clamped to (0, 0.95]. */
    double mem_fraction;

    /** Non-zero to print per-stage timing to stdout. Default 0. */
    int verbose;

    /** Absolute threshold used by ::mr_calc_mask with ::MR_MASK_THRESHOLD,
     *  in units of samples: an upsampled k-space location is retained when
     *  the (locally max-filtered) magnitude of the unit-weight Toeplitz kernel
     *  there exceeds it. One isolated on-grid sample contributes exactly 1.
     *  Default ::MR_DEFAULT_MASK_THRESHOLD. Must be >= 0. */
    double mask_threshold;

    /** Mask construction used by ::mr_calc_mask. See ::mr_mask_method.
     *  Default ::MR_MASK_NEIGHBOUR. */
    mr_mask_method mask_method;

    /** Neighbourhood radius, in upsampled grid cells, used by ::mr_calc_mask
     *  with ::MR_MASK_NEIGHBOUR. Default ::MR_DEFAULT_MASK_RADIUS. Must be
     *  >= 0; 0 keeps only the grid point nearest to each sample. */
    int mask_radius;
} mr_config;

/**
 * @brief Populate @p cfg with library defaults for a given trajectory type.
 *
 * Sets `eps = 1e-6`, `gpu_device_id = 0`, `stream = NULL`, `n_coils = 1`,
 * `mem_fraction = 0.5`, `triangulation = MR_TRI_LOWER`, `is_masked = 0`,
 * `mask_method = MR_MASK_NEIGHBOUR`, `mask_radius = MR_DEFAULT_MASK_RADIUS`,
 * `mask_threshold = MR_DEFAULT_MASK_THRESHOLD`, all sizes to 1 and
 * ::n_spatial according to @p traj_type.
 *
 * @param[out] cfg       Configuration to initialise. Must be non-NULL.
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
 * @param[in] cfg Configuration to check.
 * @return ::MR_SUCCESS or ::MR_ERR_INVALID_CONFIG (see ::mr_last_error).
 */
MR_EXPORT int mr_config_validate(const mr_config *cfg);

/* ------------------------------------------------------------------------ */
/* Derived-size helpers                                                     */
/* ------------------------------------------------------------------------ */

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

/** @brief Opaque, reusable @f$A^H A@f$ plan. Create with ::mr_aha_plan_create. */
typedef struct mr_aha_plan_s *mr_aha_plan;

/**
 * @brief Create a reusable @f$A^H A@f$ plan.
 *
 * The plan owns the cuFFT plan and the padded working buffers, so that an
 * iterative reconstruction can call ::mr_aha_plan_execute many times without
 * re-planning or re-allocating. Device memory held by the plan is
 * `Mx*My*Mz*L` ::mr_complex (plus `num_masked*L` when masked).
 *
 * @param[in]  cfg      Validated configuration. Copied into the plan; the
 *                      caller's struct need not outlive the plan.
 * @param[out] plan_out Receives the new plan. Set to NULL on failure.
 * @return ::MR_SUCCESS or a negative ::mr_status.
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
 * (with the centring index remap folded in), @f$F@f$ is the centred DFT over
 * the upsampled grid, and @f$\hat{T}@f$ is the block-diagonal multiplication by
 * the Hermitian @f$L \times L@f$ block held in `phi2_toep` at every upsampled
 * k-space location. The scaling of `phi2_toep` produced by ::mr_calc_toep
 * already absorbs the @f$1/(M_xM_yM_z)@f$ of the unnormalised cuFFT pair, so no
 * further scaling is applied here.
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
 *                         linear indices into the centred upsampled grid
 *                         `[Mx, My, Mz]`. Must be NULL when
 *                         ::mr_config::is_masked is zero, non-NULL otherwise.
 * @param[out] d_AhAx      Device output array, same shape and size as @p d_x.
 *                         Fully overwritten; need not be pre-zeroed.
 * @return ::MR_SUCCESS or a negative ::mr_status.
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
 * @param[in] plan Plan from ::mr_aha_plan_create, or NULL (no-op).
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
MR_EXPORT int mr_aha_plan_destroy(mr_aha_plan plan);

/**
 * @brief One-shot convenience wrapper: create, execute and destroy a plan.
 *
 * Equivalent to ::mr_aha_plan_create followed by ::mr_aha_plan_execute and
 * ::mr_aha_plan_destroy. Prefer the plan API inside an iterative loop.
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
/* Toeplitz kernel pre-computation                                          */
/* ------------------------------------------------------------------------ */

/**
 * @brief Pre-compute the block-diagonal Toeplitz operator `phi2_toep`.
 *
 * Given the non-Cartesian Gram blocks @f$\Phi^H\Phi@f$ sampled along the
 * trajectory, this routine
 *   1. packs each @f$L \times L@f$ block into the requested triangle,
 *   2. evaluates a type-1 cufinufft (sign @f$-1@f$) of the packed strengths
 *      onto the upsampled grid `phi2_dim`, giving the point-spread function,
 *   3. applies a centred **inverse** FFT (index remap + unnormalised cuFFT +
 *      @f$1/\prod M@f$), and
 *   4. optionally gathers the result at the masked locations.
 *
 * Step 2 with sign @f$-1@f$ followed by the normalised inverse FFT of step 3
 * is algebraically identical to the forward FFT of the sign @f$+1@f$
 * point-spread function divided by @f$\prod M@f$, which is exactly the
 * multiplier ::mr_aha_plan_execute needs between its unnormalised forward and
 * inverse cuFFTs. No further scaling is required anywhere.
 *
 * For ::MR_TRAJ_STACK the type-1 transform is two-dimensional and is repeated
 * for every kz encode; no transform and no shift is applied along z, because kz
 * is already Cartesian and `phi2` is supplied in centred kz order.
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
 *                          @f$L\times L@f$ matrix @f$\Phi^H\Phi@f$ of that
 *                          trajectory and `Nkz == Mz == Nz`.
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
 *                          linear indices into the centred upsampled grid, or
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

/* ------------------------------------------------------------------------ */
/* Automatic k-space mask                                                   */
/* ------------------------------------------------------------------------ */

/**
 * @brief Number of elements of the unit-weight kernel of ::mr_calc_mask_kernel.
 *
 * The kernel depends only on the in-plane trajectory for ::MR_TRAJ_2D and
 * ::MR_TRAJ_STACK, so it is `Mx*My` for those and `Mx*My*Mz` for ::MR_TRAJ_3D.
 *
 * @param[in] cfg Configuration.
 * @return Element count, or 0 when @p cfg is NULL.
 */
MR_EXPORT int64_t mr_mask_kernel_size(const mr_config *cfg);

/**
 * @brief Magnitude of the unit-weight Toeplitz kernel of a trajectory.
 *
 * Evaluates exactly what ::mr_calc_toep would return for @f$L = 1@f$ and every
 * Gram block equal to one, i.e. the Toeplitz multiplier
 * @f$T(\mathbf{k})@f$ of the plain (non-subspace) normal operator
 * @f$F^H F@f$ of the trajectory, and stores @f$|T(\mathbf{k})|@f$.
 * @f$T@f$ is essentially the k-space sampling density blurred by the
 * Dirichlet kernel of the upsampled grid: it is large where the trajectory
 * samples and decays towards zero away from it (for instance in the corners
 * of the grid outside the sampled disc or ball of a radial trajectory). This
 * is the quantity thresholded by ::mr_calc_mask; it is exposed separately so
 * that the threshold can be studied.
 *
 * @param[in]  cfg      Validated configuration. Only ::mr_config::traj_type,
 *                      ::mr_config::n_dim, ::mr_config::phi2_dim and the
 *                      execution fields (eps, device, stream, mem_fraction)
 *                      are used; ::mr_config::L and ::mr_config::is_masked
 *                      are ignored.
 * @param[in]  d_traj   Device trajectory `[Ntrajs, Nkx, ndim]`, exactly as for
 *                      ::mr_calc_toep.
 * @param[in]  Ntrajs   Number of trajectories.
 * @param[in]  Nkx      Samples per trajectory.
 * @param[out] d_kernel Device array of ::mr_mask_kernel_size floats, centred
 *                      order: `[Mx, My]` (2D, stack) or `[Mx, My, Mz]` (3D).
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
MR_EXPORT int mr_calc_mask_kernel(const mr_config *cfg,
                                  const float     *d_traj,
                                  int64_t          Ntrajs,
                                  int64_t          Nkx,
                                  float           *d_kernel);

/**
 * @brief Derive the masked-operator sample set directly from the trajectory.
 *
 * Two constructions are available, selected by ::mr_config::mask_method:
 *
 * - **::MR_MASK_NEIGHBOUR (default).** Every sample @f$\mathbf{k}@f$
 *   (radians) is mapped to its nearest point on the upsampled grid,
 *   `rint(k * M / (2*pi)) + M/2` per transformed axis (modulo `M`, the grid
 *   is periodic), those points are marked, and the marks are dilated by a
 *   @f$(2r+1)^d@f$ box with @f$r@f$ = ::mr_config::mask_radius. Every
 *   location within @f$r@f$ cells (Chebyshev distance) of a sample's grid
 *   point is kept. No NUFFT is needed. Samples lie between grid points and
 *   each one's Dirichlet kernel spreads over the neighbouring cells, so
 *   @f$r = 0@f$ is far too small; the default @f$r = 2@f$ reaches the
 *   unmasked accuracy on densely sampled radial and 3D trajectories while
 *   keeping roughly the sampled disc / ball of k-space.
 *
 * - **::MR_MASK_THRESHOLD.** Computes the unit-weight kernel @f$|T|@f$ of
 *   ::mr_calc_mask_kernel, max-filters it over the immediate neighbourhood of
 *   every location (a @f$3\times3@f$ or @f$3\times3\times3@f$ box over the
 *   transformed axes), giving @f$\tilde{T}@f$, and keeps every location where
 *   @f$\tilde{T}(\mathbf{k}) > \tau@f$, with @f$\tau@f$ =
 *   ::mr_config::mask_threshold. The threshold is absolute, in units of
 *   samples: a single isolated sample on the grid contributes exactly 1 to the
 *   unit-weight kernel, so @f$\tau = 0.05@f$ drops locations that receive less
 *   than 5 % of one sample's worth. (A threshold relative to the maximum does
 *   not transfer between trajectories, because the maximum of a radial kernel
 *   is the number of spokes crossing the k-space centre.) The dilation matters
 *   because the unit-weight kernel, a sum of Dirichlet kernels, can cancel to
 *   almost zero at isolated points inside the sampled region where the kernel
 *   of the actual Gram blocks does not. Because it follows the kernel's
 *   slowly decaying, axis-aligned tails, this method is more accurate per
 *   retained location for very sparse trajectories such as few-arm spirals.
 *
 * Locations far from every sample carry little of the Toeplitz kernel, so
 * dropping them from `phi2_toep` saves memory with an error controlled by the
 * radius or threshold.
 *
 * For ::MR_TRAJ_STACK the in-plane mask is replicated over all `Mz` kz
 * planes, because every kz encode uses the same 2D trajectory.
 *
 * The result is directly usable as the `d_mask` argument of ::mr_calc_toep and
 * ::mr_aha_plan_execute after setting ::mr_config::is_masked and
 * ::mr_config::num_masked.
 *
 * @param[in]  cfg            Validated configuration (see
 *                            ::mr_calc_mask_kernel for the fields used, plus
 *                            ::mr_config::mask_method and
 *                            ::mr_config::mask_radius or
 *                            ::mr_config::mask_threshold).
 * @param[in]  d_traj         Device trajectory `[Ntrajs, Nkx, ndim]`.
 * @param[in]  Ntrajs         Number of trajectories.
 * @param[in]  Nkx            Samples per trajectory.
 * @param[out] d_mask         Device array receiving the retained **0-based**
 *                            linear indices into the centred upsampled grid
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
