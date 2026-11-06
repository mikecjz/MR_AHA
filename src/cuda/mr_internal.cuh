/**
 * @file mr_internal.cuh
 * @brief Internal helpers of the CUDA backend.
 *
 * Not part of the public API. Builds on the backend-independent
 * `src/common/mr_common.h` (error reporting, guard macros, ::mr_dims) and adds
 * thin RAII-free guard macros around the CUDA / cuFFT / cufinufft return codes
 * plus the `__host__ __device__` index helpers used by the kernels.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#ifndef MR_AHA_MR_INTERNAL_CUH
#define MR_AHA_MR_INTERNAL_CUH

#include "mr_common.h"

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdio>

/* ------------------------------------------------------------------------ */
/* CUDA / cuFFT / cufinufft error checks                                    */
/* ------------------------------------------------------------------------ */

/** @brief Evaluate a CUDA runtime call and bail out with ::MR_ERR_CUDA. */
#define MR_CUDA_CHECK(call)                                                    \
    do {                                                                       \
        cudaError_t mr_err_ = (call);                                          \
        if (mr_err_ != cudaSuccess)                                            \
            MR_FAIL(MR_ERR_CUDA, "%s failed at %s:%d: %s", #call, __FILE__,    \
                    __LINE__, cudaGetErrorString(mr_err_));                    \
    } while (0)

/** @brief Evaluate a cuFFT call and bail out with ::MR_ERR_CUFFT. */
#define MR_CUFFT_CHECK(call)                                                   \
    do {                                                                       \
        cufftResult mr_err_ = (call);                                          \
        if (mr_err_ != CUFFT_SUCCESS)                                          \
            MR_FAIL(MR_ERR_CUFFT, "%s failed at %s:%d: cuFFT code %d", #call,  \
                    __FILE__, __LINE__, (int)mr_err_);                         \
    } while (0)

/** @brief Evaluate a cufinufft call and bail out with ::MR_ERR_CUFINUFFT. */
#define MR_CUFINUFFT_CHECK(call)                                               \
    do {                                                                       \
        int mr_err_ = (call);                                                  \
        if (mr_err_ != 0)                                                      \
            MR_FAIL(MR_ERR_CUFINUFFT,                                          \
                    "%s failed at %s:%d: cufinufft code %d", #call, __FILE__,  \
                    __LINE__, mr_err_);                                        \
    } while (0)

/** @brief Check for a pending asynchronous kernel-launch error. */
#define MR_KERNEL_CHECK(name)                                                  \
    do {                                                                       \
        cudaError_t mr_err_ = cudaGetLastError();                              \
        if (mr_err_ != cudaSuccess)                                            \
            MR_FAIL(MR_ERR_CUDA, "kernel %s failed to launch at %s:%d: %s",    \
                    name, __FILE__, __LINE__, cudaGetErrorString(mr_err_));    \
    } while (0)

/* ------------------------------------------------------------------------ */
/* Small device-side utilities                                              */
/* ------------------------------------------------------------------------ */

/**
 * @brief Map a centred index to the corresponding natural (FFT-order) index.
 *
 * A centred index @p k in `[0, M)` denotes frequency/position `k - floor(M/2)`;
 * the natural index of that frequency is `(k - floor(M/2)) mod M`.
 *
 * @param[in] k Centred index.
 * @param[in] M Extent of the axis.
 * @param[in] c `floor(M/2)`, pre-computed by the host.
 * @return Natural index in `[0, M)`.
 */
__host__ __device__ __forceinline__ int mr_cent2nat(int k, int M, int c)
{
    int j = k - c;
    return (j < 0) ? (j + M) : j;
}

/**
 * @brief Map a natural (FFT-order) index to the corresponding centred index.
 * @param[in] j Natural index.
 * @param[in] M Extent of the axis.
 * @param[in] c `floor(M/2)`, pre-computed by the host.
 * @return Centred index in `[0, M)`.
 */
__host__ __device__ __forceinline__ int mr_nat2cent(int j, int M, int c)
{
    int k = j + c;
    return (k >= M) ? (k - M) : k;
}

/**
 * @brief Packed offset of a stored triangular element, plus a conjugate flag.
 *
 * Resolves @f$A_{ij}@f$ of the Hermitian block from the stored triangle. When
 * `(i, j)` lies outside the stored triangle, the transposed element is
 * addressed instead and @p conj_out is set, so the caller must conjugate.
 *
 * @param[in]  i         Row index, `0 <= i < L`.
 * @param[in]  j         Column index, `0 <= j < L`.
 * @param[in]  L         Rank.
 * @param[in]  tri_upper Non-zero when the upper triangle is stored.
 * @param[out] conj_out  Set to 1 when the caller must conjugate, else 0.
 * @return Offset along the `(L^2+L)/2` axis of `phi2_toep`.
 */
__host__ __device__ __forceinline__ int mr_tri_offset(int i, int j, int L,
                                                      int tri_upper,
                                                      int *conj_out)
{
    int r = i, c = j, cj = 0;
    if (tri_upper) {
        /* Stored elements satisfy r <= c. */
        if (r > c) { int t = r; r = c; c = t; cj = 1; }
        *conj_out = cj;
        return c * (c + 1) / 2 + r;
    }
    /* Lower triangle: stored elements satisfy r >= c. */
    if (r < c) { int t = r; r = c; c = t; cj = 1; }
    *conj_out = cj;
    return c * L - c * (c - 1) / 2 + (r - c);
}

/* ------------------------------------------------------------------------ */
/* Internal entry points shared between translation units                   */
/* ------------------------------------------------------------------------ */

/**
 * @brief Make @p cfg's device current, remembering the caller's device.
 *
 * A ::mr_config::gpu_device_id of -1 means "leave the current device alone",
 * which is what MATLAB gpuArray and CuPy callers want.
 *
 * @param[in]  cfg  Configuration carrying the requested device.
 * @param[out] prev Receives the previously current device, or -1 if unchanged.
 * @return ::MR_SUCCESS or ::MR_ERR_CUDA.
 */
int mr_toep_push_device(const mr_config *cfg, int *prev);

/**
 * @brief Body of ::mr_calc_toep without argument checking or device switching.
 *
 * Shared with ::mr_calc_mask_kernel, which evaluates the unit-weight kernel by
 * running the very same pipeline with @f$L = 1@f$. The caller must have
 * validated @p cfg and made the target device current.
 *
 * @param[in]  cfg         Validated configuration.
 * @param[in]  d_phi2      Gram blocks, see ::mr_calc_toep.
 * @param[in]  d_traj      Trajectory, see ::mr_calc_toep.
 * @param[in]  Ntrajs      Number of trajectories.
 * @param[in]  Nkx         Samples per trajectory.
 * @param[in]  d_mask      Mask, or NULL when not masked.
 * @param[out] d_phi2_toep Output, see ::mr_calc_toep.
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
int mr_calc_toep_impl(const mr_config *cfg, const mr_complex *d_phi2,
                      const float *d_traj, int64_t Ntrajs, int64_t Nkx,
                      const int64_t *d_mask, mr_complex *d_phi2_toep);

#endif /* MR_AHA_MR_INTERNAL_CUH */
