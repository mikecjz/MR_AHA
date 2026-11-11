/**
 * @file mr_common.h
 * @brief Backend-independent internal helpers shared by every MR_AHA backend.
 *
 * Not part of the public API. Provides the error-reporting plumbing behind
 * ::mr_last_error, the argument / configuration guard macros, and the small
 * plain-old-data geometry descriptor (::mr_dims) derived from an
 * ::mr_config. Everything here is plain C++ so that it compiles with the host
 * compiler for the Metal backend and with `nvcc` for the CUDA backend.
 *
 * Backend-specific helpers live next to their backend:
 * `src/cuda/mr_internal.cuh` (CUDA / cuFFT / cufinufft checks, device index
 * helpers) and `src/metal/` (Metal).
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#ifndef MR_AHA_MR_COMMON_H
#define MR_AHA_MR_COMMON_H

#include "mr_aha/mr_aha.h"

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

/**
 * @brief Clear this thread's error state. Called on entry to public API.
 *
 * Empties the diagnostic buffer and then lets the backend drop any stale
 * error state of its own through ::mr_backend_clear_error.
 */
void mr_clear_error(void);

/**
 * @brief Backend hook of ::mr_clear_error.
 *
 * Implemented once per backend. The CUDA backend resets the CUDA runtime's
 * per-thread "last error" (see `src/cuda/mr_backend_cuda.cu` for why); the
 * Metal backend has no such state and does nothing.
 */
void mr_backend_clear_error(void);

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

/* ------------------------------------------------------------------------ */
/* Kernel-side geometry descriptor                                          */
/* ------------------------------------------------------------------------ */

/**
 * @brief Plain-old-data geometry descriptor passed by value into kernels.
 *
 * Passing the geometry as a kernel argument (rather than through global
 * constant memory) keeps the library re-entrant: several plans, streams or
 * host threads can be in flight simultaneously without clobbering shared
 * device state. The Metal backend mirrors this struct field for field in its
 * shader source.
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

/** @brief Integer ceiling division for launch-configuration arithmetic. */
inline long long mr_ceil_div(long long a, long long b)
{
    return (a + b - 1) / b;
}

#endif /* MR_AHA_MR_COMMON_H */
