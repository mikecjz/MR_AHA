/**
 * @file mt_cuda.h
 * @brief Public C API of the MT_CUDA library.
 *
 * MT_CUDA evaluates the forward normal operator @f$A^H A@f$ for non-Cartesian
 * MRI with low-rank (subspace / MR-Multitasking) modelling on the GPU, using a
 * Toeplitz embedding so that no non-uniform FFT is required inside the
 * iteration loop.
 *
 * The library is a standalone CUDA/C++ library with a pure C interface
 * (`extern "C"`), so that it can be linked directly from C, C++, a MATLAB MEX
 * gateway (MATLAB GPU C API) or a Python `ctypes` binding.
 *
 * @section mt_layout Array layout
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
 * @section mt_memory Memory ownership
 *
 * **All array arguments are device (GPU) pointers.** The library never
 * allocates or touches host buffers for bulk data; the caller is responsible
 * for allocating and freeing every input and output array on the device. Small
 * scalar configuration values are passed by value on the host through
 * ::mt_config.
 *
 * @section mt_precision Precision
 *
 * The library is single precision (`complexFloat` / ::mt_complex) throughout,
 * matching the `complexsingle` gpuArray type in MATLAB and `complex64` in
 * NumPy/CuPy.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#ifndef MT_CUDA_MT_CUDA_H
#define MT_CUDA_MT_CUDA_H

#include <cuComplex.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#  ifdef MT_CUDA_BUILD_SHARED
#    define MT_EXPORT __declspec(dllexport)
#  else
#    define MT_EXPORT
#  endif
#else
#  define MT_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Single-precision complex scalar used by every MT_CUDA array. */
typedef cuFloatComplex mt_complex;

/* ------------------------------------------------------------------------ */
/* Status codes                                                             */
/* ------------------------------------------------------------------------ */

/**
 * @brief Return codes of every MT_CUDA entry point.
 *
 * Every function returns ::MT_SUCCESS (== 0) on success and a negative value on
 * failure. Use ::mt_status_string to obtain a human readable description.
 */
typedef enum mt_status {
    MT_SUCCESS            =  0, /**< Operation completed successfully.        */
    MT_ERR_INVALID_ARG    = -1, /**< A NULL pointer or nonsensical size.      */
    MT_ERR_INVALID_CONFIG = -2, /**< ::mt_config failed validation.           */
    MT_ERR_ALLOC          = -3, /**< A device allocation failed.              */
    MT_ERR_CUDA           = -4, /**< A CUDA runtime call failed.              */
    MT_ERR_CUFFT          = -5, /**< A cuFFT call failed.                     */
    MT_ERR_CUFINUFFT      = -6, /**< A cufinufft call failed.                 */
    MT_ERR_UNSUPPORTED    = -7, /**< Valid but not implemented combination.   */
    MT_ERR_INTERNAL       = -8  /**< Should not happen; please report.        */
} mt_status;

/**
 * @brief Human readable description of an ::mt_status value.
 * @param[in] status Value previously returned by an MT_CUDA function.
 * @return Pointer to a static, NUL-terminated string. Never NULL.
 */
MT_EXPORT const char *mt_status_string(int status);

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
MT_EXPORT const char *mt_last_error(void);

/* ------------------------------------------------------------------------ */
/* Configuration                                                            */
/* ------------------------------------------------------------------------ */

/** @brief Trajectory family, selecting how ::mt_calc_toep interprets its input. */
typedef enum mt_traj_type {
    MT_TRAJ_2D    = 0, /**< Planar 2D trajectory; image is `[Nx, Ny, L]`.      */
    MT_TRAJ_3D    = 1, /**< Fully 3D trajectory; image is `[Nx, Ny, Nz, L]`.   */
    MT_TRAJ_STACK = 2  /**< Stack of identical 2D trajectories along kz
                            (e.g. stack-of-stars). Image is `[Nx, Ny, Nz, L]`
                            and `Mz == Nz == Nkz`.                            */
} mt_traj_type;

/**
 * @brief Which triangle of the Hermitian @f$L \times L@f$ block is stored.
 *
 * `phi2_toep` holds only @f$(L^2+L)/2@f$ of the @f$L^2@f$ entries of each
 * Hermitian block. Both packings are **column-major** packings of the
 * corresponding MATLAB triangle, i.e. exactly the element order produced by
 * `A(tril(true(L)))` / `A(triu(true(L)))`:
 *
 * - ::MT_TRI_LOWER: element `(i, j)`, `i >= j`, lives at
 *   `t = j*L - j*(j-1)/2 + (i - j)`.
 * - ::MT_TRI_UPPER: element `(i, j)`, `i <= j`, lives at `t = j*(j+1)/2 + i`.
 *
 * The missing half is reconstructed on the fly from Hermitian symmetry,
 * @f$A_{ij} = \overline{A_{ji}}@f$.
 */
typedef enum mt_triangulation {
    MT_TRI_LOWER = 0, /**< Lower triangle, column-major packed. */
    MT_TRI_UPPER = 1  /**< Upper triangle, column-major packed. */
} mt_triangulation;

/**
 * @brief Complete description of one @f$A^H A@f$ operator.
 *
 * Fill with ::mt_config_defaults first, then override the fields you need. Use
 * ::mt_config_validate to check consistency before creating a plan (the plan
 * constructors validate as well).
 */
typedef struct mt_config {
    /** Trajectory family. See ::mt_traj_type. */
    mt_traj_type traj_type;

    /** Number of *spatial* dimensions of the image: 2 or 3. Derived from
     *  ::traj_type by ::mt_config_defaults but may be set explicitly:
     *  ::MT_TRAJ_2D uses 2, ::MT_TRAJ_3D and ::MT_TRAJ_STACK use 3. */
    int n_spatial;

    /** Image matrix size `[Nx, Ny, Nz]`. Entries beyond ::n_spatial must be 1. */
    int n_dim[3];

    /** Upsampled (Toeplitz) matrix size `[Mx, My, Mz]`. Entries beyond
     *  ::n_spatial must be 1. Typically `M = 2*N`. For ::MT_TRAJ_STACK,
     *  `Mz` must equal `Nz` because kz is already Cartesian. */
    int phi2_dim[3];

    /** Number of ranks / subspace coefficients @f$L@f$. Must be >= 1. */
    int L;

    /** Triangle stored in `phi2_toep`. See ::mt_triangulation. */
    mt_triangulation triangulation;

    /** Non-zero when `phi2_toep` is stored only on a subset of the upsampled
     *  Cartesian grid; then `phi2_toep` is `[num_masked, (L^2+L)/2]` and a
     *  `mask` of ::num_masked linear indices must be supplied. */
    int is_masked;

    /** Number of retained samples when ::is_masked is non-zero, else ignored. */
    int64_t num_masked;

    /** Number of receive coils @f$N_{coils}@f$. Must be >= 1. Used only by
     *  ::mt_aha_toeplitz / ::mt_aha_plan_execute. */
    int n_coils;

    /** Requested cufinufft tolerance for ::mt_calc_toep. Default `1e-6`. */
    double eps;

    /** CUDA device to run on. Default 0. Set to -1 to keep the caller's
     *  current device (recommended when interoperating with MATLAB gpuArray or
     *  CuPy, which manage the current device themselves). */
    int gpu_device_id;

    /** CUDA stream (`cudaStream_t`) for all work, or NULL for the default
     *  stream. Declared `void *` to keep this header free of `cuda_runtime.h`. */
    void *stream;

    /** Fraction of the *free* device memory ::mt_calc_toep may use for its
     *  working buffers. Default 0.5. Clamped to (0, 0.95]. */
    double mem_fraction;

    /** Non-zero to print per-stage timing to stdout. Default 0. */
    int verbose;
} mt_config;

/**
 * @brief Populate @p cfg with library defaults for a given trajectory type.
 *
 * Sets `eps = 1e-6`, `gpu_device_id = 0`, `stream = NULL`, `n_coils = 1`,
 * `mem_fraction = 0.5`, `triangulation = MT_TRI_LOWER`, `is_masked = 0`, all
 * sizes to 1 and ::n_spatial according to @p traj_type.
 *
 * @param[out] cfg       Configuration to initialise. Must be non-NULL.
 * @param[in]  traj_type Trajectory family the operator will be used for.
 * @return ::MT_SUCCESS, or ::MT_ERR_INVALID_ARG if @p cfg is NULL.
 */
MT_EXPORT int mt_config_defaults(mt_config *cfg, mt_traj_type traj_type);

/**
 * @brief Check a configuration for internal consistency.
 *
 * Verifies dimensionality, positivity of all sizes, `Mz == Nz` for
 * ::MT_TRAJ_STACK, `M >= N` in every transformed direction, and that
 * ::mt_config::num_masked is positive and no larger than the upsampled grid
 * when ::mt_config::is_masked is set.
 *
 * @param[in] cfg Configuration to check.
 * @return ::MT_SUCCESS or ::MT_ERR_INVALID_CONFIG (see ::mt_last_error).
 */
MT_EXPORT int mt_config_validate(const mt_config *cfg);

/* ------------------------------------------------------------------------ */
/* Derived-size helpers                                                     */
/* ------------------------------------------------------------------------ */

/**
 * @brief Number of stored entries of one packed Hermitian block, @f$(L^2+L)/2@f$.
 * @param[in] L Rank.
 * @return `(int64_t)L * (L + 1) / 2`, or 0 when `L <= 0`.
 */
MT_EXPORT int64_t mt_num_tri(int L);

/**
 * @brief Number of voxels of one image-space volume, `Nx*Ny*Nz`.
 * @param[in] cfg Configuration.
 * @return Voxel count, or 0 when @p cfg is NULL.
 */
MT_EXPORT int64_t mt_image_size(const mt_config *cfg);

/**
 * @brief Number of voxels of one upsampled (Toeplitz) volume, `Mx*My*Mz`.
 * @param[in] cfg Configuration.
 * @return Voxel count, or 0 when @p cfg is NULL.
 */
MT_EXPORT int64_t mt_pad_size(const mt_config *cfg);

/**
 * @brief Number of ::mt_complex elements the `phi2_toep` array must hold.
 *
 * `num_masked * (L^2+L)/2` when masked, else `Mx*My*Mz * (L^2+L)/2`.
 *
 * @param[in] cfg Configuration.
 * @return Element count, or 0 when @p cfg is NULL.
 */
MT_EXPORT int64_t mt_phi2_toep_size(const mt_config *cfg);

/**
 * @brief Number of ::mt_complex elements of the spatial-coefficient array `x`.
 * @param[in] cfg Configuration.
 * @return `Nx*Ny*Nz*L`, or 0 when @p cfg is NULL.
 */
MT_EXPORT int64_t mt_x_size(const mt_config *cfg);

/**
 * @brief Number of ::mt_complex elements of the sensitivity-map array `SEs`.
 *
 * One map per coil, shared by every rank, so the rank axis never appears.
 *
 * @param[in] cfg Configuration.
 * @return `Nx*Ny*Nz*Ncoils`, or 0 when @p cfg is NULL.
 */
MT_EXPORT int64_t mt_se_size(const mt_config *cfg);

/* ------------------------------------------------------------------------ */
/* Forward operator: AhA via Toeplitz embedding                             */
/* ------------------------------------------------------------------------ */

/** @brief Opaque, reusable @f$A^H A@f$ plan. Create with ::mt_aha_plan_create. */
typedef struct mt_aha_plan_s *mt_aha_plan;

/**
 * @brief Create a reusable @f$A^H A@f$ plan.
 *
 * The plan owns the cuFFT plan and the padded working buffers, so that an
 * iterative reconstruction can call ::mt_aha_plan_execute many times without
 * re-planning or re-allocating. Device memory held by the plan is
 * `Mx*My*Mz*L` ::mt_complex (plus `num_masked*L` when masked).
 *
 * @param[in]  cfg      Validated configuration. Copied into the plan; the
 *                      caller's struct need not outlive the plan.
 * @param[out] plan_out Receives the new plan. Set to NULL on failure.
 * @return ::MT_SUCCESS or a negative ::mt_status.
 */
MT_EXPORT int mt_aha_plan_create(const mt_config *cfg, mt_aha_plan *plan_out);

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
 * k-space location. The scaling of `phi2_toep` produced by ::mt_calc_toep
 * already absorbs the @f$1/(M_xM_yM_z)@f$ of the unnormalised cuFFT pair, so no
 * further scaling is applied here.
 *
 * @param[in]  plan        Plan from ::mt_aha_plan_create.
 * @param[in]  d_x         Device array of spatial coefficients,
 *                         `[Nx, Ny, L]` (2D) or `[Nx, Ny, Nz, L]` (3D/stack).
 * @param[in]  d_SEs       Device array of sensitivity maps,
 *                         `[Nx, Ny, Nz, Ncoils]` (`[Nx, Ny, Ncoils]` in 2D).
 *                         One map per coil; it is broadcast across all
 *                         @f$L@f$ ranks.
 * @param[in]  d_phi2_toep Device array holding the packed Hermitian blocks:
 *                         `[Mx, My, Mz, (L^2+L)/2]` when unmasked, or
 *                         `[num_masked, (L^2+L)/2]` when masked.
 * @param[in]  d_mask      Device array of ::mt_config::num_masked **0-based**
 *                         linear indices into the centred upsampled grid
 *                         `[Mx, My, Mz]`. Must be NULL when
 *                         ::mt_config::is_masked is zero, non-NULL otherwise.
 * @param[out] d_AhAx      Device output array, same shape and size as @p d_x.
 *                         Fully overwritten; need not be pre-zeroed.
 * @return ::MT_SUCCESS or a negative ::mt_status.
 *
 * @note @p d_AhAx must not alias @p d_x.
 */
MT_EXPORT int mt_aha_plan_execute(mt_aha_plan         plan,
                                  const mt_complex   *d_x,
                                  const mt_complex   *d_SEs,
                                  const mt_complex   *d_phi2_toep,
                                  const int64_t      *d_mask,
                                  mt_complex         *d_AhAx);

/**
 * @brief Release a plan and all device memory it owns.
 * @param[in] plan Plan from ::mt_aha_plan_create, or NULL (no-op).
 * @return ::MT_SUCCESS or a negative ::mt_status.
 */
MT_EXPORT int mt_aha_plan_destroy(mt_aha_plan plan);

/**
 * @brief One-shot convenience wrapper: create, execute and destroy a plan.
 *
 * Equivalent to ::mt_aha_plan_create followed by ::mt_aha_plan_execute and
 * ::mt_aha_plan_destroy. Prefer the plan API inside an iterative loop.
 *
 * @param[in]  cfg         Validated configuration.
 * @param[in]  d_x         See ::mt_aha_plan_execute.
 * @param[in]  d_SEs       See ::mt_aha_plan_execute.
 * @param[in]  d_phi2_toep See ::mt_aha_plan_execute.
 * @param[in]  d_mask      See ::mt_aha_plan_execute.
 * @param[out] d_AhAx      See ::mt_aha_plan_execute.
 * @return ::MT_SUCCESS or a negative ::mt_status.
 */
MT_EXPORT int mt_aha_toeplitz(const mt_config  *cfg,
                              const mt_complex *d_x,
                              const mt_complex *d_SEs,
                              const mt_complex *d_phi2_toep,
                              const int64_t    *d_mask,
                              mt_complex       *d_AhAx);

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
 * multiplier ::mt_aha_plan_execute needs between its unnormalised forward and
 * inverse cuFFTs. No further scaling is required anywhere.
 *
 * For ::MT_TRAJ_STACK the type-1 transform is two-dimensional and is repeated
 * for every kz encode; no transform and no shift is applied along z, because kz
 * is already Cartesian and `phi2` is supplied in centred kz order.
 *
 * Work is chunked over the @f$(L^2+L)/2@f$ triangular components so that the
 * transient buffers respect ::mt_config::mem_fraction of the free device
 * memory.
 *
 * @param[in]  cfg          Validated configuration. ::mt_config::phi2_dim gives
 *                          the upsampled grid the kernel is evaluated on.
 * @param[in]  d_phi2       Device array of Gram blocks,
 *                          `[Ntrajs, L*L]` for ::MT_TRAJ_2D / ::MT_TRAJ_3D or
 *                          `[Ntrajs, Nkz, L*L]` for ::MT_TRAJ_STACK, where the
 *                          `L*L` axis is the column-major @f$L\times L@f$
 *                          matrix and `Nkz == Mz == Nz`.
 * @param[in]  d_traj       Device array of non-Cartesian coordinates,
 *                          `[Ntrajs, Nkx, ndim]`, in **radians per unit of the
 *                          un-padded image grid**, i.e. the usual
 *                          @f$[-\pi, \pi)@f$ range. `ndim` is 2 for
 *                          ::MT_TRAJ_2D and ::MT_TRAJ_STACK, 3 for
 *                          ::MT_TRAJ_3D.
 * @param[in]  Ntrajs       Number of non-Cartesian trajectories (e.g. spokes).
 *                          One Gram block is shared by the whole readout.
 * @param[in]  Nkx          Number of k-space samples per trajectory.
 * @param[in]  d_mask       Device array of ::mt_config::num_masked **0-based**
 *                          linear indices into the centred upsampled grid, or
 *                          NULL when ::mt_config::is_masked is zero.
 * @param[out] d_phi2_toep  Device output array of ::mt_phi2_toep_size elements:
 *                          `[Mx, My, Mz, (L^2+L)/2]` when unmasked, or
 *                          `[num_masked, (L^2+L)/2]` when masked.
 * @return ::MT_SUCCESS or a negative ::mt_status.
 */
MT_EXPORT int mt_calc_toep(const mt_config  *cfg,
                           const mt_complex *d_phi2,
                           const float      *d_traj,
                           int64_t           Ntrajs,
                           int64_t           Nkx,
                           const int64_t    *d_mask,
                           mt_complex       *d_phi2_toep);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MT_CUDA_MT_CUDA_H */
