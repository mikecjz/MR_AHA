/**
 * @file mr_internal.cuh
 * @brief Internal helpers shared by the MR_AHA translation units.
 *
 * Not part of the public API. Provides the error-reporting plumbing behind
 * ::mr_last_error, thin RAII-free guard macros around the CUDA / cuFFT /
 * cufinufft return codes, and the small plain-old-data descriptor
 * (::mr_dims) that is passed by value into every kernel.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#ifndef MR_AHA_MR_INTERNAL_CUH
#define MR_AHA_MR_INTERNAL_CUH

#include "mr_aha/mr_aha.h"

#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdio>

/* ------------------------------------------------------------------------ */
/* Error reporting                                                          */
/* ------------------------------------------------------------------------ */

/**
 * @brief Record a formatted diagnostic in this thread's error buffer.
 *
 * The message is retrievable by the caller through ::mr_last_error until the
 * next failing library call on the same thread.
 *
 * @param[in] fmt `printf`-style format string, followed by its arguments.
 */
void mr_set_error(const char *fmt, ...);

/** @brief Clear this thread's error buffer. Called on entry to public API. */
void mr_clear_error(void);

/**
 * @brief Return @p status after recording a location-tagged diagnostic.
 * @param[in] status ::mr_status value to return.
 * @param[in] fmt    `printf`-style format string, followed by its arguments.
 */
#define MR_FAIL(status, ...)                                                   \
    do {                                                                       \
        mr_set_error(__VA_ARGS__);                                             \
        return (status);                                                       \
    } while (0)

/** @brief Bail out with ::MR_ERR_INVALID_ARG unless @p cond holds. */
#define MR_REQUIRE(cond, ...)                                                  \
    do {                                                                       \
        if (!(cond)) MR_FAIL(MR_ERR_INVALID_ARG, __VA_ARGS__);                 \
    } while (0)

/** @brief Bail out with ::MR_ERR_INVALID_CONFIG unless @p cond holds. */
#define MR_REQUIRE_CFG(cond, ...)                                              \
    do {                                                                       \
        if (!(cond)) MR_FAIL(MR_ERR_INVALID_CONFIG, __VA_ARGS__);              \
    } while (0)

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
/* Kernel-side geometry descriptor                                          */
/* ------------------------------------------------------------------------ */

/**
 * @brief Plain-old-data geometry descriptor passed by value into kernels.
 *
 * Passing the geometry as a kernel argument (rather than through
 * `__constant__` memory) keeps the library re-entrant: several plans, streams
 * or host threads can be in flight simultaneously without clobbering shared
 * device state.
 */
struct mr_dims {
    int n[3];            /**< Image size `[Nx, Ny, Nz]`, unused entries 1.     */
    int m[3];            /**< Upsampled size `[Mx, My, Mz]`, unused entries 1. */
    int c_n[3];          /**< Image centre offsets `floor(N_d/2)`.             */
    int c_m[3];          /**< Upsampled centre offsets `floor(M_d/2)`.         */
    int L;               /**< Rank.                                            */
    int tri_upper;       /**< 1 when the upper triangle is stored, else 0.     */
    long long image_size;/**< `Nx*Ny*Nz`.                                      */
    long long pad_size;  /**< `Mx*My*Mz`.                                      */
    long long se_coil;   /**< Coil stride into `SEs`, i.e. `image_size`. One
                              map per coil is shared by every rank.            */
};

/**
 * @brief Fill an ::mr_dims from a validated ::mr_config.
 * @param[in]  cfg Validated configuration.
 * @param[out] d   Descriptor to populate.
 */
void mr_dims_from_config(const mr_config *cfg, mr_dims *d);

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

/** @brief Integer ceiling division for launch-configuration arithmetic. */
__host__ __forceinline__ long long mr_ceil_div(long long a, long long b)
{
    return (a + b - 1) / b;
}

#endif /* MR_AHA_MR_INTERNAL_CUH */
