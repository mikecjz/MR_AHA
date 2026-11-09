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

/* ------------------------------------------------------------------------ */
/* calc_toep context: compute any range of components on one device         */
/* ------------------------------------------------------------------------ */

/**
 * @brief Opaque per-device state of the Toeplitz pre-computation.
 *
 * ::mr_calc_toep evaluates the @f$(L^2+L)/2@f$ packed triangular components
 * of `phi2_toep` independently of one another (one type-1 NUFFT, one centred
 * inverse FFT and one scaling each). A context holds everything that does not
 * depend on *which* components are evaluated (packed Gram blocks, chunk
 * buffers, cufinufft and cuFFT plans), so that ::mr_toep_ctx_run can evaluate
 * any range of components, any number of times, into any destination. The
 * multi-GPU pre-computation (`mr_toep_mgpu.cu`) gives each device its own
 * context and range; the single-GPU ::mr_calc_toep runs one context over all
 * components.
 */
struct mr_toep_ctx;

/**
 * @brief Callback of ::mr_toep_ctx_run, invoked on the host after the work of
 *        each chunk has been *issued* (not completed) on the context's stream.
 *
 * Work the callback enqueues behind an event recorded on that stream sees the
 * chunk's components in their final place, which is how the multi-GPU
 * pre-computation overlaps the transfer of finished chunks with the
 * computation of the next ones.
 *
 * @param[in] user Opaque pointer given to ::mr_toep_ctx_run.
 * @param[in] t0   First component of the chunk.
 * @param[in] n    Number of components of the chunk.
 * @return ::MR_SUCCESS, or a negative ::mr_status that aborts the run.
 */
typedef int (*mr_toep_chunk_fn)(void *user, int64_t t0, int64_t n);

/**
 * @brief Create a calc_toep context on the current device.
 *
 * Packs the Gram blocks of all components (asynchronously, on
 * ::mr_config::stream), sizes the chunks against ::mr_config::mem_fraction
 * of the device memory free *now* (allocate any large persistent output
 * first), allocates the chunk buffers and plans cufinufft (on the current
 * device, or ::mr_config::gpu_device_id when set) and cuFFT.
 *
 * @param[in]  cfg       Validated configuration; copied. Its stream and
 *                       device are used for all work of the context.
 * @param[in]  d_phi2    Gram blocks, see ::mr_calc_toep. Only read by work
 *                       issued here; may be released once the stream has
 *                       passed this call.
 * @param[in]  d_traj    Trajectory, see ::mr_calc_toep. cufinufft keeps the
 *                       pointer: it must stay valid until
 *                       ::mr_toep_ctx_destroy.
 * @param[in]  Ntrajs    Number of trajectories.
 * @param[in]  Nkx       Samples per trajectory.
 * @param[in]  max_chunk Upper bound on the components per chunk, or 0 for
 *                       none. More, smaller chunks let a caller start
 *                       transferring earlier.
 * @param[out] ctx_out   Receives the context; NULL on failure.
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
int mr_toep_ctx_create(const mr_config *cfg, const mr_complex *d_phi2,
                       const float *d_traj, int64_t Ntrajs, int64_t Nkx,
                       int64_t max_chunk, mr_toep_ctx **ctx_out);

/** @brief Components per chunk chosen by ::mr_toep_ctx_create. */
int mr_toep_ctx_chunk(const mr_toep_ctx *ctx);

/**
 * @brief Evaluate components `[t_begin, t_end)` of `phi2_toep`.
 *
 * Asynchronous on the context's stream unless @p ms_out is given. Component
 * @f$t@f$ is written to slab @f$t - t_{out0}@f$ of @p d_out, a slab being
 * `Mx*My*Mz` elements (unmasked) or ::mr_config::num_masked elements
 * (masked). With @p t_out0 = 0, @p d_out is the whole kernel; with
 * @p t_out0 = @p t_begin, a buffer holding only this range.
 *
 * @param[in,out] ctx      Context from ::mr_toep_ctx_create (current device
 *                         must be its device).
 * @param[in]     d_mask   Mask on the context's device, or NULL unmasked.
 * @param[in]     t_begin  First component.
 * @param[in]     t_end    One past the last component, `<= (L^2+L)/2`.
 * @param[out]    d_out    Destination, see above.
 * @param[in]     t_out0   Component held by slab 0 of @p d_out,
 *                         `<= t_begin`.
 * @param[in]     on_chunk Optional callback after each chunk, or NULL.
 * @param[in]     user     Passed to @p on_chunk.
 * @param[out]    ms_out   Optional: when non-NULL, the call waits for the
 *                         range to complete and stores its GPU time in
 *                         milliseconds (from the first to the last chunk,
 *                         i.e. without the planning of the context).
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
int mr_toep_ctx_run(mr_toep_ctx *ctx, const int64_t *d_mask, int64_t t_begin,
                    int64_t t_end, mr_complex *d_out, int64_t t_out0,
                    mr_toep_chunk_fn on_chunk, void *user, float *ms_out);

/**
 * @brief Wait for the context's stream, then free everything it holds.
 * @param[in] ctx Context, or NULL (no-op). Current device: its device.
 * @return ::MR_SUCCESS, or ::MR_ERR_CUDA when the synchronisation reported
 *         an error (the context is freed regardless).
 */
int mr_toep_ctx_destroy(mr_toep_ctx *ctx);

/* ------------------------------------------------------------------------ */
/* Multi-GPU helpers shared between translation units                       */
/* ------------------------------------------------------------------------ */

/**
 * @brief Enable direct peer access between devices @p a and @p b in both
 *        directions, when the hardware allows it. Changes the current device.
 * @return 1 when both directions are enabled, else 0 (copies between the two
 *         are then staged through the host by the CUDA runtime).
 */
int mr_enable_peer(int a, int b);

/** @brief How the multi-GPU pre-computation of `phi2_toep` distributes the
 *         components (see ::mr_toep_distribute). */
enum mr_toep_dist_mode {
    MR_TOEP_DIST_AUTO      = 0, /**< Cost model: exchange or self.          */
    MR_TOEP_DIST_EXCHANGE  = 1, /**< Every component computed once, then
                                     copied to every device that needs it. */
    MR_TOEP_DIST_SELF      = 2  /**< Every device that needs the whole
                                     kernel computes it itself; helpers
                                     drop out after the pilot.             */
};

/** @brief One device taking part in ::mr_toep_distribute. */
struct mr_toep_dist_dev {
    int         dev;    /**< CUDA ordinal.                                   */
    mr_complex *d_full; /**< Whole-kernel destination on this device (a
                             *receiver*), or NULL for a *helper*, which only
                             computes components for the receivers.          */
};

/** @brief What ::mr_toep_distribute did (for reports and the plan's info). */
struct mr_toep_dist_report {
    int    n_devices;     /**< Devices that computed components.            */
    int    mode;          /**< ::MR_TOEP_DIST_EXCHANGE or ::MR_TOEP_DIST_SELF
                               (what was run, after `auto`).               */
    double entry_ms;      /**< Measured GPU time of one component (slowest
                               device's pilot), ms.                        */
    double gather_gbs;    /**< Measured concurrent receive bandwidth per
                               receiver, GB/s (0: not measured).           */
    double t_exchange_ms; /**< Cost-model estimate of the exchange, ms.     */
    double t_self_ms;     /**< Cost-model estimate of self, ms.             */
    double total_ms;      /**< Wall time of ::mr_toep_distribute, ms.       */
    double probe_ms;      /**< Wall time of the bandwidth probe, ms.        */
    double phase1_ms;     /**< Wall time of setup + pilot, ms.              */
    double phase2_ms;     /**< Wall time of exchange / self, ms.            */
    double cleanup_ms;    /**< Wall time of freeing the transients, ms.     */
};

/**
 * @brief Compute `phi2_toep` with several devices (see `mr_toep_mgpu.cu`).
 *
 * The components are split into contiguous, nearly equal ranges, one per
 * device. Every device computes the first chunk of its range (the *pilot*,
 * timed); then the cost model (or @p mode) decides between
 *  - ::MR_TOEP_DIST_EXCHANGE: every device computes the rest of its range and
 *    copies each finished chunk to every receiver, so every receiver ends
 *    with the whole kernel and every component was computed exactly once;
 *  - ::MR_TOEP_DIST_SELF: every receiver computes all the components it does
 *    not have yet itself (no transfer); helpers stop after the pilot.
 *
 * Synchronous: returns when every receiver holds the whole kernel. One host
 * thread per device; the caller's thread only waits.
 *
 * @param[in]  cfg     Validated configuration. ::mr_config::stream (primary
 *                     device) is synchronised first, so inputs produced on it
 *                     are complete.
 * @param[in]  devs    Devices; `devs[0]` is the primary, which holds the
 *                     inputs, and must be a receiver. No duplicates.
 * @param[in]  G       Number of devices, `1 <= G <= MR_MGPU_MAX_DEVICES`.
 * @param[in]  d_phi2  Gram blocks on the primary device.
 * @param[in]  d_traj  Trajectory on the primary device.
 * @param[in]  Ntrajs  Number of trajectories.
 * @param[in]  Nkx     Samples per trajectory.
 * @param[in]  d_mask  Mask on the primary device, or NULL.
 * @param[in]  mode    ::mr_toep_dist_mode (`AUTO` for the cost model).
 * @param[out] report  Optional report, or NULL.
 * @return ::MR_SUCCESS or a negative ::mr_status; an error on another device
 *         names its ordinal in ::mr_last_error. The current device is
 *         changed.
 */
int mr_toep_distribute(const mr_config *cfg, const mr_toep_dist_dev *devs, int G,
                       const mr_complex *d_phi2, const float *d_traj,
                       int64_t Ntrajs, int64_t Nkx, const int64_t *d_mask,
                       int mode, mr_toep_dist_report *report);

/**
 * @brief Build the device list of a multi-GPU call: the primary device first
 *        (::mr_config::gpu_device_id, or the current device when -1), then
 *        @p devices (or every visible device when NULL) in order, without
 *        duplicates.
 * @param[in]  who         Function name for diagnostics.
 * @param[in]  cfg         Configuration.
 * @param[in]  devices     Requested CUDA ordinals, or NULL.
 * @param[in]  n_devices   Length of @p devices (0 with NULL).
 * @param[out] cand        Receives up to ::MR_MGPU_MAX_DEVICES ordinals.
 * @param[out] G_out       Number of entries of @p cand.
 * @param[out] primary_out The primary device.
 * @return ::MR_SUCCESS, ::MR_ERR_INVALID_ARG or ::MR_ERR_CUDA.
 */
int mr_mgpu_candidates(const char *who, const mr_config *cfg, const int *devices,
                       int n_devices, int *cand, int *G_out, int *primary_out);

/**
 * @brief Parse `MR_AHA_TOEP_SPLIT` (`off`, `split`, `replicate`, `auto`).
 * @return -1 for `off` (the primary computes the kernel alone), otherwise a
 *         ::mr_toep_dist_mode (`auto` when unset or unrecognised).
 */
int mr_toep_split_env(void);

/**
 * @brief Coils per group chosen by a single-GPU plan (see
 *        ::mr_aha_plan_create); used by the multi-GPU plan's report.
 * @param[in] plan A valid plan.
 * @return The group size (>= 1).
 */
int mr_aha_plan_group(mr_aha_plan plan);

/**
 * @brief Free only the host structure of a single-GPU plan whose device
 *        context was destroyed (its cuFFT plans and buffers went with the
 *        context and must not be touched); used by the multi-GPU plan.
 * @param[in] plan A plan, or NULL.
 */
void mr_aha_plan_forget(mr_aha_plan plan);

#endif /* MR_AHA_MR_INTERNAL_CUH */
