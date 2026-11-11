/**
 * @file mr_op.cu
 * @brief The @f$A^H A@f$ operator object (::mr_op_create): one or several
 *        GPUs of one node, Toeplitz or NUFFT method.
 *
 * @f$A^H A x = \sum_c S_c^H (\ldots) S_c x@f$ is a sum of independent
 * per-coil pipelines. An operator gives each of its devices a contiguous
 * range of whole coils and a per-device *engine* for them, chosen by the
 * method:
 *
 * - ::MR_OP_TOEPLITZ: an ordinary single-GPU ::mr_aha_plan (coil groups,
 *   specialised block-diagonal kernels, cuFFT plans) plus a device-resident
 *   copy of the kernel `phi2_toep` and the mask;
 * - ::MR_OP_NUFFT: a ::mr_nufft_eng (cufinufft type-2 / type-1 plans, the
 *   reordered Gram blocks and trajectory, cuBLAS; `mr_nufft_op.cu`).
 *
 * Every device also holds the sensitivity maps of its own coils. One
 * application with several devices then
 *  1. records an event on the caller's stream (primary device);
 *  2. on every other device: waits for that event, copies `x` from the primary
 *     device, applies its coils into a partial result and copies the partial
 *     result back into a staging buffer on the primary device;
 *  3. on the primary device: applies its own coils straight into the output;
 *  4. makes the caller's stream wait for all devices and adds the staged
 *     partial results into the output with one kernel.
 * A single host thread issues all of this asynchronously; nothing blocks the
 * host. Every entry point restores the caller's current device, so MATLAB's
 * and CuPy's device selection is left untouched.
 *
 * @section mr_op_direct One device
 *
 * When the operator ends up on one device (the default, or chosen by the
 * device rules), it takes a *direct* path that costs nothing over the
 * stateless primitives: the engine runs on the caller's stream, reads the
 * caller's `x` and writes the caller's output (no events, no copies, no
 * worker streams), the kernel is computed in place with ::mr_calc_toep, and
 * no bandwidth probe or cost model runs.
 *
 * @section mr_op_split Remainder split (Toeplitz only)
 *
 * When the coils do not divide evenly over the @f$G@f$ devices, the
 * @f$r = C \bmod G@f$ remaining coils can instead be split across all devices
 * by rank (`MR_AHA_MGPU_SPLIT`). Padding, FFT, inverse FFT and cropping are
 * independent per rank, but the block-diagonal stage couples all @f$L@f$
 * ranks of a k-space location. For each remaining coil:
 *  1. device @f$g@f$ pads and transforms its ranks @f$R_g@f$ (about
 *     @f$L/G@f$ of them) into `d_A` `[Mx*My*Mz, |R_g|]`;
 *  2. all-to-all: device @f$g@f$ sends the k-space slab @f$K_h@f$ (a
 *     contiguous range of about @f$M_xM_yM_z/G@f$ natural-order locations)
 *     of its ranks to device @f$h@f$, which collects all @f$L@f$ ranks of
 *     @f$K_h@f$ in `d_B` `[|K_h|, L]`;
 *  3. device @f$h@f$ applies the (replicated) kernel to @f$K_h@f$;
 *  4. all-to-all back, inverse FFT of @f$R_g@f$ and crop into ranks
 *     @f$R_g@f$ of the device's partial result.
 * Each device thus does @f$1/G@f$ of the work of a remaining coil, at the
 * price of moving its padded data twice between devices; worthwhile only
 * with fast links (NVLink). See ::mr_op_create_impl for the cost model.
 *
 * @section mr_op_kernel The Toeplitz kernel
 *
 * Unless the caller supplies it, the operator allocates its own copy of the
 * kernel on every device that applies it and fills them with
 * ::mr_toep_distribute (`mr_toep_mgpu.cu`): every device computes a share of
 * the triangular components into its own copy, then the shares are exchanged
 * (or, when the links are too slow, every device computes the rest itself),
 * per ::mr_op_options::toep_dist. With one device, or
 * ::MR_TOEP_DIST_BROADCAST, the primary computes it alone with
 * ::mr_calc_toep_impl. From then on the primary's copy plays the part of a
 * caller's kernel (see ::mr_op_compute_kernel).
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_kernels.cuh"

#include <cuda.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/** @brief Threads per block of the reduction kernel. */
#define MR_OP_BLOCK 256

/** @brief Default minimum number of upsampled values per coil
 *         (`Mx*My*Mz*L`) for a multi-GPU split to be worthwhile. Smaller
 *         problems (e.g. the 2D benchmark, launch-bound at < 1 ms) run on the
 *         primary device only. Overridden by `MR_AHA_MGPU_MIN_WORK`. */
#define MR_OP_MIN_WORK (1LL << 22)

/* ------------------------------------------------------------------------ */
/* Operator state                                                           */
/* ------------------------------------------------------------------------ */

/** @brief Per-device state of an operator. */
struct mr_op_dev {
    int          dev;        /**< CUDA ordinal.                                */
    cudaStream_t stream;     /**< Stream this device's work is issued on.      */
    int          owns_stream;/**< Non-zero when @ref stream was created here
                                  (not the caller's, see @ref mr_op_direct).  */
    cudaEvent_t  done;       /**< Recorded when the device's part is complete. */
    mr_aha_plan  plan;       /**< Toeplitz: single-GPU plan of its coils.      */
    mr_nufft_eng *nu;        /**< NUFFT: engine of its coils.                  */
    int          coil0;      /**< First whole coil of this device.             */
    int          ncoil;      /**< Number of whole coils of this device.        */
    mr_complex  *d_SEs;      /**< Maps of this device's coils `[.., ncoil]`.   */
    mr_complex  *d_toep;     /**< Toeplitz: replica of `phi2_toep`.            */
    int64_t     *d_mask;     /**< Toeplitz: replica of the mask, or NULL.      */
    int          owns_const; /**< Non-zero when `d_SEs` and `d_mask` are owned
                                  (freed on destroy).                          */
    int          owns_toep;  /**< Non-zero when `d_toep` is owned.             */
    mr_complex  *d_x;        /**< Copy of `x` (non-primary devices only).      */
    mr_complex  *d_part;     /**< Partial `AhAx` (non-primary devices only).   */
    mr_complex  *d_stage;    /**< Staging buffer for this device's partial
                                  result, on the primary device.               */

    /* Remainder split (Toeplitz); unused when the operator does not split. */
    int          r0;         /**< First rank of @f$R_g@f$.                     */
    int          nr;         /**< Number of ranks @f$|R_g|@f$ (may be 0).       */
    long long    k0;         /**< First location of the slab @f$K_g@f$.       */
    long long    nk;         /**< Number of locations @f$|K_g|@f$.            */
    cufftHandle  fft_r;      /**< cuFFT over `nr` volumes, or 0.               */
    void        *d_work_r;   /**< Its work area.                               */
    mr_complex  *d_A;        /**< Padded volumes of ranks @f$R_g@f$.           */
    mr_complex  *d_B;        /**< All ranks of the slab @f$K_g@f$.             */
    mr_complex  *d_SEs_rem;  /**< Maps of the remaining coils `[.., r]`.        */
    int         *d_inv;      /**< Grid-to-mask map (masked mode), see
                                  ::mr_k_mask_inverse.                         */
    cudaEvent_t  ev_fft;     /**< Ranks @f$R_g@f$ transformed (step 1).         */
    cudaEvent_t  ev_bd;      /**< Slab @f$K_g@f$ multiplied (step 3).           */

    unsigned long long ctx_id; /**< Unique ID of the device's primary CUDA
                                  context when the slot was set up (0: not
                                  set up); see ::mr_op_dev_alive.              */
};

/** @brief Internal state of an operator. */
struct mr_op_s {
    mr_config    cfg;        /**< Copy of the caller's configuration.          */
    mr_op_options opt;       /**< Copy of the options (`devices` cleared).     */
    int          primary;    /**< CUDA ordinal of the primary device.          */
    cudaStream_t stream;     /**< Caller's stream on the primary device.       */
    cudaEvent_t  start;      /**< Recorded on @ref stream at each execute
                                  (multi-device only).                         */
    int          direct;     /**< One device, no split: the direct path.       */
    int          ndev;       /**< Devices used; `dev[0]` is the primary.       */
    int          nsplit;     /**< Remaining coils split by rank (0: off).      */
    int          split0;     /**< First of the split coils.                    */
    mr_dims      dims;       /**< Geometry descriptor.                         */
    mr_bd_kernels bd;        /**< Specialised block-diagonal kernels.          */
    mr_op_dev    dev[MR_OP_MAX_DEVICES]; /**< Per-device state.                */
    mr_op_info_t info;       /**< What ::mr_op_info reports.                   */
    /* How phi2_toep was obtained; copied into @ref info at the end of the
     * creation (see the toep_* fields of ::mr_op_info_t). */
    int          toep_source;    /**< ::MR_OP_TOEP_SUPPLIED etc.               */
    int          toep_devices;   /**< Devices that computed components.        */
    double       toep_ms;        /**< Time of computing (and distributing).    */
    double       toep_comp_ms;   /**< Pilot time per component.                */
    double       toep_gbs;       /**< Probed concurrent receive bandwidth.     */
};

/** @brief Pointers to the staged partial results, passed by value. */
struct mr_op_parts {
    const mr_complex *p[MR_OP_MAX_DEVICES]; /**< Partial results.              */
    int               n;                    /**< Number of valid entries.      */
};

/* ------------------------------------------------------------------------ */
/* Kernels                                                                  */
/* ------------------------------------------------------------------------ */

/**
 * @brief Add the staged partial results into the output:
 *        `out[i] += sum_k parts.p[k][i]`.
 *
 * One pass over the output regardless of the number of devices.
 *
 * @param[in,out] out   Output `[Nx, Ny, Nz, L]`, holding the primary's part.
 * @param[in]     parts Staged partial results of the other devices.
 * @param[in]     n     Number of elements.
 */
__global__ void mr_k_op_reduce(mr_complex *out, mr_op_parts parts, long long n)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    mr_complex acc = out[i];
    for (int k = 0; k < parts.n; ++k) acc = cuCaddf(acc, parts.p[k][i]);
    out[i] = acc;
}

/* ------------------------------------------------------------------------ */
/* Helpers                                                                  */
/* ------------------------------------------------------------------------ */

/** @brief Read an integer environment variable, or @p def when unset. */
static long long mr_op_env(const char *name, long long def)
{
    const char *e = std::getenv(name);
    return (e && e[0]) ? atoll(e) : def;
}

/**
 * @brief Prefix the current diagnostic with the device ordinal.
 * @param[in] dev CUDA ordinal.
 * @param[in] rc  Status to pass through.
 * @return @p rc.
 */
static int mr_op_tag(int dev, int rc)
{
    if (rc != MR_SUCCESS) {
        char msg[1024];
        snprintf(msg, sizeof(msg), "%s", mr_last_error());
        mr_set_error("device %d: %s", dev, msg);
    }
    return rc;
}

/**
 * @brief Number of devices to use for @p C coils on @p G devices.
 *
 * The application time is set by the busiest device, i.e. by
 * @f$\lceil C/G \rceil@f$ coils; returns the smallest @f$G'@f$ with the same
 * @f$\lceil C/G' \rceil@f$, so that no device is added without a gain.
 */
static int mr_op_pick_count(int C, int G)
{
    if (G > C) G = C;
    if (G < 1) return 1;
    const int per = (C + G - 1) / G;
    int g = G;
    while (g > 1 && (C + (g - 1) - 1) / (g - 1) == per) --g;
    return g;
}

/**
 * @brief Unique ID of @p dev's primary context, creating it if needed.
 * @param[in]  dev CUDA ordinal; made current.
 * @param[out] id  Context ID (`cuCtxGetId`, unique per context instance for
 *                 the lifetime of the process).
 * @return ::MR_SUCCESS or ::MR_ERR_CUDA.
 */
static int mr_op_ctx_id(int dev, unsigned long long *id)
{
    MR_CUDA_CHECK(cudaSetDevice(dev));
    MR_CUDA_CHECK(cudaFree(0)); /* make sure the primary context exists */
    CUcontext ctx = NULL;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == NULL ||
        cuCtxGetId(ctx, id) != CUDA_SUCCESS)
        MR_FAIL(MR_ERR_CUDA, "could not identify the CUDA context of device %d", dev);
    return MR_SUCCESS;
}

/**
 * @brief Whether the resources of device slot @p d still exist.
 *
 * A caller can destroy a device's CUDA context under a live operator: MATLAB
 * resets a GPU on `reset(gpuDevice)` and whenever it is selected or
 * deselected with `gpuDevice(k)`, which can hit the primary *and* the other
 * devices of an operator. Using its stream, event or memory handles on such a
 * device afterwards crashes inside the CUDA driver, and freeing its old
 * pointers could free memory that a newer operator has since been given at
 * the same addresses. A slot is alive when the device's primary context is
 * active (checked through the driver API, which does not create one) and is
 * still the very context instance the slot was set up in (same `cuCtxGetId`;
 * a context re-created after a reset gets a new ID).
 *
 * @param[in] d Device slot; one that was never set up counts as alive (it
 *              holds nothing yet).
 * @return Non-zero when the slot's handles may be used.
 */
static int mr_op_dev_alive(const mr_op_dev *d)
{
    if (d->ctx_id == 0) return 1;
    CUdevice cud;
    unsigned int flags = 0;
    int active = 0;
    if (cuDeviceGet(&cud, d->dev) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxGetState(cud, &flags, &active) != CUDA_SUCCESS || !active)
        return 0;
    if (cudaSetDevice(d->dev) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    CUcontext ctx = NULL;
    unsigned long long id = 0;
    if (cuCtxGetCurrent(&ctx) != CUDA_SUCCESS || ctx == NULL ||
        cuCtxGetId(ctx, &id) != CUDA_SUCCESS)
        return 0;
    return id == d->ctx_id;
}

/**
 * @brief Release everything held by device slot @p d (best effort).
 *
 * @param[in]     p             Operator (primary device).
 * @param[in,out] d             Slot; zeroed on return.
 * @param[in]     alive         Whether the slot's device context still exists
 *                              (::mr_op_dev_alive); when not, its handles
 *                              are dropped without being used.
 * @param[in]     primary_alive Likewise for the primary device, which holds
 *                              the slot's staging buffer.
 */
static void mr_op_dev_release(mr_op p, mr_op_dev *d, int alive, int primary_alive)
{
    if (alive) {
        cudaSetDevice(d->dev);
        if (d->plan) mr_aha_plan_destroy(d->plan);
        if (d->nu) mr_nufft_eng_destroy(d->nu, 1);
        if (d->owns_const) {
            if (d->d_SEs) cudaFree(d->d_SEs);
            if (d->d_mask) cudaFree(d->d_mask);
        }
        if (d->owns_toep && d->d_toep) cudaFree(d->d_toep);
        if (d->d_x) cudaFree(d->d_x);
        if (d->d_part) cudaFree(d->d_part);
        if (d->fft_r) cufftDestroy(d->fft_r);
        if (d->d_work_r) cudaFree(d->d_work_r);
        if (d->d_A) cudaFree(d->d_A);
        if (d->d_B) cudaFree(d->d_B);
        if (d->d_SEs_rem) cudaFree(d->d_SEs_rem);
        if (d->d_inv) cudaFree(d->d_inv);
        if (d->ev_fft) cudaEventDestroy(d->ev_fft);
        if (d->ev_bd) cudaEventDestroy(d->ev_bd);
        if (d->done) cudaEventDestroy(d->done);
        if (d->stream && d->owns_stream) cudaStreamDestroy(d->stream);
    } else {
        /* The engines' host side only; their device handles are gone. */
        if (d->plan) mr_aha_plan_forget(d->plan);
        if (d->nu) mr_nufft_eng_destroy(d->nu, 0);
    }
    if (d->d_stage && primary_alive) {
        cudaSetDevice(p->primary);
        cudaFree(d->d_stage);
    }
    memset(d, 0, sizeof(*d));
}

/**
 * @brief Release a whole operator (best effort); the current device is
 *        changed.
 *
 * Devices whose context no longer exists are skipped (see
 * ::mr_op_dev_alive); with @p primary_lost the primary device is skipped
 * without being probed (::mr_op_abandon).
 */
static void mr_op_release(mr_op p, int primary_lost = 0)
{
    int alive[MR_OP_MAX_DEVICES];
    for (int k = 0; k < p->ndev; ++k)
        alive[k] = (k == 0 && primary_lost) ? 0 : mr_op_dev_alive(&p->dev[k]);
    const int primary_alive = (p->ndev > 0) ? alive[0] : !primary_lost;
    for (int k = 0; k < p->ndev; ++k)
        mr_op_dev_release(p, &p->dev[k], alive[k], primary_alive);
    if (p->start && primary_alive) {
        cudaSetDevice(p->primary);
        cudaEventDestroy(p->start);
    }
    cudaGetLastError();
    free(p);
}

/* Body of mr_enable_peer, documented in mr_internal.cuh. */
int mr_enable_peer(int a, int b)
{
    int ab = 0, ba = 0;
    if (cudaDeviceCanAccessPeer(&ab, a, b) != cudaSuccess) ab = 0;
    if (cudaDeviceCanAccessPeer(&ba, b, a) != cudaSuccess) ba = 0;
    if (!ab || !ba) {
        cudaGetLastError();
        return 0;
    }
    int ok = 1;
    if (cudaSetDevice(a) == cudaSuccess) {
        cudaError_t e = cudaDeviceEnablePeerAccess(b, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) ok = 0;
    }
    if (cudaSetDevice(b) == cudaSuccess) {
        cudaError_t e = cudaDeviceEnablePeerAccess(a, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) ok = 0;
    }
    cudaGetLastError(); /* "already enabled" is not an error for us */
    return ok;
}

/**
 * @brief Allocate the remainder-split resources of device slot @p d.
 *
 * Runs with @p d->dev current, after its constants are in place.
 *
 * @param[in,out] p     Operator (geometry, split coils).
 * @param[in,out] d     Slot with `r0`, `nr`, `k0`, `nk` filled.
 * @param[in]     d_SEs Caller's maps (primary device).
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
static int mr_op_split_setup(mr_op p, mr_op_dev *d, const mr_complex *d_SEs)
{
    const mr_config *cfg = &p->cfg;
    const size_t c8 = sizeof(mr_complex);
    const long long pad = p->dims.pad_size;
    const size_t se_bytes = (size_t)mr_image_size(cfg) * p->nsplit * c8;

    MR_CUDA_CHECK(cudaEventCreateWithFlags(&d->ev_fft, cudaEventDisableTiming));
    MR_CUDA_CHECK(cudaEventCreateWithFlags(&d->ev_bd, cudaEventDisableTiming));

    cudaError_t ce = cudaMalloc((void **)&d->d_SEs_rem, se_bytes);
    if (ce == cudaSuccess && d->nr > 0)
        ce = cudaMalloc((void **)&d->d_A, (size_t)pad * d->nr * c8);
    if (ce == cudaSuccess)
        ce = cudaMalloc((void **)&d->d_B, (size_t)d->nk * cfg->L * c8);
    if (ce == cudaSuccess && cfg->is_masked)
        ce = cudaMalloc((void **)&d->d_inv, (size_t)pad * sizeof(int));
    if (ce != cudaSuccess) {
        cudaGetLastError();
        MR_FAIL(MR_ERR_ALLOC, "failed to allocate the remainder-split buffers: %s",
                cudaGetErrorString(ce));
    }
    MR_CUDA_CHECK(cudaMemcpyPeerAsync(
        d->d_SEs_rem, d->dev, d_SEs + (size_t)mr_image_size(cfg) * p->split0,
        p->primary, se_bytes, d->stream));

    if (cfg->is_masked) {
        /* The mask is constant for the operator's lifetime: build the map
         * once. */
        const long long nm = (long long)cfg->num_masked;
        MR_CUDA_CHECK(cudaMemsetAsync(d->d_inv, 0xFF, (size_t)pad * sizeof(int),
                                      d->stream));
        mr_k_mask_inverse<<<(int)mr_ceil_div(nm, MR_OP_BLOCK), MR_OP_BLOCK, 0,
                            d->stream>>>(d->d_inv, d->d_mask, nm, p->dims);
        MR_KERNEL_CHECK("mr_k_mask_inverse");
    }

    if (d->nr > 0) {
        /* Batched cuFFT over this device's nr padded volumes. */
        const int rank = cfg->n_spatial;
        long long n[3];
        if (rank == 3) {
            n[0] = cfg->phi2_dim[2]; n[1] = cfg->phi2_dim[1]; n[2] = cfg->phi2_dim[0];
        } else {
            n[0] = cfg->phi2_dim[1]; n[1] = cfg->phi2_dim[0];
        }
        size_t work = 0;
        MR_CUFFT_CHECK(cufftCreate(&d->fft_r));
        MR_CUFFT_CHECK(cufftSetAutoAllocation(d->fft_r, 0));
        MR_CUFFT_CHECK(cufftMakePlanMany64(d->fft_r, rank, n, NULL, 1, pad, NULL,
                                           1, pad, CUFFT_C2C, (long long)d->nr,
                                           &work));
        if (work > 0) {
            ce = cudaMalloc(&d->d_work_r, work);
            if (ce != cudaSuccess) {
                cudaGetLastError();
                MR_FAIL(MR_ERR_ALLOC, "failed to allocate a cuFFT work area: %s",
                        cudaGetErrorString(ce));
            }
        }
        MR_CUFFT_CHECK(cufftSetWorkArea(d->fft_r, d->d_work_r));
        MR_CUFFT_CHECK(cufftSetStream(d->fft_r, d->stream));
    }
    MR_CUDA_CHECK(cudaStreamSynchronize(d->stream));
    return MR_SUCCESS;
}

/**
 * @brief Measure the copy bandwidth from the primary device to @p dev.
 *
 * Times a few 64 MB `cudaMemcpyPeerAsync` copies (direct over NVLink/PCIe
 * with peer access, staged through the host otherwise).
 *
 * @param[in]  primary Source device.
 * @param[in]  dev     Destination device.
 * @param[out] gbs     Bandwidth in GB/s.
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
static int mr_op_probe_bw(int primary, int dev, double *gbs)
{
    const size_t bytes = 64u << 20;
    void *src = NULL, *dst = NULL;
    cudaStream_t st = NULL;
    cudaEvent_t e0 = NULL, e1 = NULL;
    float ms = 0.0f;
    cudaError_t ce = cudaSetDevice(primary);
    if (ce == cudaSuccess) ce = cudaMalloc(&src, bytes);
    if (ce == cudaSuccess) ce = cudaSetDevice(dev);
    if (ce == cudaSuccess) ce = cudaMalloc(&dst, bytes);
    if (ce == cudaSuccess) ce = cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    if (ce == cudaSuccess) ce = cudaEventCreate(&e0);
    if (ce == cudaSuccess) ce = cudaEventCreate(&e1);
    if (ce == cudaSuccess) ce = cudaMemcpyPeerAsync(dst, dev, src, primary, bytes, st);
    if (ce == cudaSuccess) ce = cudaEventRecord(e0, st);
    for (int k = 0; k < 3 && ce == cudaSuccess; ++k)
        ce = cudaMemcpyPeerAsync(dst, dev, src, primary, bytes, st);
    if (ce == cudaSuccess) ce = cudaEventRecord(e1, st);
    if (ce == cudaSuccess) ce = cudaEventSynchronize(e1);
    if (ce == cudaSuccess) ce = cudaEventElapsedTime(&ms, e0, e1);
    const cudaError_t err = ce;
    if (e0) cudaEventDestroy(e0);
    if (e1) cudaEventDestroy(e1);
    if (st) cudaStreamDestroy(st);
    if (dst) cudaFree(dst);
    cudaSetDevice(primary);
    if (src) cudaFree(src);
    if (err != cudaSuccess) {
        cudaGetLastError();
        MR_FAIL(MR_ERR_CUDA, "peer bandwidth probe %d -> %d failed: %s", primary,
                dev, cudaGetErrorString(err));
    }
    *gbs = (ms > 0.0f) ? 3.0 * bytes / (ms * 1e-3) / 1e9 : 0.0;
    return MR_SUCCESS;
}

/**
 * @brief Time the single-GPU Toeplitz operator over @p ncoil coils on the
 *        primary device (for the remainder-split cost model).
 *
 * Builds a temporary plan for the first @p ncoil coils (with its coil groups,
 * as a device of the operator would), applies it once to warm up and once
 * timed, on zero input, using the caller's arrays.
 *
 * @param[in]  p      Operator (configuration, primary device current).
 * @param[in]  ncoil  Number of coils, `1 <= ncoil <= n_coils`.
 * @param[in]  d_SEs  Caller's maps.
 * @param[in]  d_toep Kernel on the primary.
 * @param[in]  d_mask Caller's mask or NULL.
 * @param[out] ms     Time of one application in milliseconds.
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
static int mr_op_probe_coils(mr_op p, int ncoil, const mr_complex *d_SEs,
                             const mr_complex *d_toep, const int64_t *d_mask,
                             double *ms)
{
    mr_config c1 = p->cfg;
    c1.n_coils = ncoil;
    c1.verbose = 0;
    const size_t x_bytes = (size_t)mr_x_size(&c1) * sizeof(mr_complex);
    mr_complex *x = NULL, *y = NULL;
    cudaStream_t st = NULL;
    cudaEvent_t e0 = NULL, e1 = NULL;
    mr_aha_plan tp = NULL;
    float t = 0.0f;
    int rc = MR_SUCCESS;
    cudaError_t ce = cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    if (ce == cudaSuccess) ce = cudaMalloc((void **)&x, x_bytes);
    if (ce == cudaSuccess) ce = cudaMalloc((void **)&y, x_bytes);
    if (ce == cudaSuccess) ce = cudaMemsetAsync(x, 0, x_bytes, st);
    if (ce == cudaSuccess) ce = cudaEventCreate(&e0);
    if (ce == cudaSuccess) ce = cudaEventCreate(&e1);
    if (ce == cudaSuccess) {
        c1.stream = (void *)st;
        rc = mr_aha_plan_create(&c1, &tp);
        if (rc == MR_SUCCESS)
            rc = mr_aha_plan_execute(tp, x, d_SEs, d_toep, d_mask, y);
        if (rc == MR_SUCCESS) ce = cudaEventRecord(e0, st);
        if (rc == MR_SUCCESS && ce == cudaSuccess)
            rc = mr_aha_plan_execute(tp, x, d_SEs, d_toep, d_mask, y);
        if (rc == MR_SUCCESS && ce == cudaSuccess) ce = cudaEventRecord(e1, st);
        if (rc == MR_SUCCESS && ce == cudaSuccess) ce = cudaEventSynchronize(e1);
        if (rc == MR_SUCCESS && ce == cudaSuccess) ce = cudaEventElapsedTime(&t, e0, e1);
    }
    if (tp) mr_aha_plan_destroy(tp);
    if (e0) cudaEventDestroy(e0);
    if (e1) cudaEventDestroy(e1);
    if (st) cudaStreamDestroy(st);
    if (x) cudaFree(x);
    if (y) cudaFree(y);
    if (rc != MR_SUCCESS) return rc;
    if (ce != cudaSuccess) {
        cudaGetLastError();
        MR_FAIL(MR_ERR_CUDA, "one-coil timing run failed: %s", cudaGetErrorString(ce));
    }
    *ms = t;
    return MR_SUCCESS;
}

/**
 * @brief Set up device slot @p d: stream, event, constants, buffers and the
 *        engine for its coils.
 *
 * Runs with any device current; leaves @p d->dev current.
 *
 * @param[in,out] p      Operator (geometry, method, primary device).
 * @param[in,out] d      Slot with `dev`, `coil0` and `ncoil` filled.
 * @param[in]     in     Caller's inputs (primary device).
 * @param[in]     d_toep Toeplitz: kernel to copy from (primary device), the
 *                       caller's or the operator's own copy on the primary.
 *                       Not used when the slot already holds a kernel
 *                       (`d->d_toep` set by ::mr_op_compute_kernel).
 * @param[in]     share  Use the caller's arrays in place (primary only).
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
static int mr_op_dev_setup(mr_op p, mr_op_dev *d, const mr_op_inputs *in,
                           const mr_complex *d_toep, int share)
{
    const mr_config *cfg = &p->cfg;
    const int toeplitz = (p->opt.method == MR_OP_TOEPLITZ);
    const size_t c8 = sizeof(mr_complex);
    const int is_primary = (d->dev == p->primary);
    const size_t se_bytes = (size_t)mr_image_size(cfg) * d->ncoil * c8;
    const size_t toep_bytes = toeplitz ? (size_t)mr_phi2_toep_size(cfg) * c8 : 0;
    const size_t mask_bytes =
        (toeplitz && cfg->is_masked) ? (size_t)cfg->num_masked * sizeof(int64_t) : 0;
    const size_t x_bytes = (size_t)mr_x_size(cfg) * c8;
    const mr_complex *se_src = in->d_SEs + (size_t)mr_image_size(cfg) * d->coil0;

    MR_CUDA_CHECK(cudaSetDevice(d->dev));
    if (p->direct) {
        /* One device: the engine runs on the caller's stream. */
        d->stream = p->stream;
        d->owns_stream = 0;
    } else {
        MR_CUDA_CHECK(cudaStreamCreateWithFlags(&d->stream, cudaStreamNonBlocking));
        d->owns_stream = 1;
        MR_CUDA_CHECK(cudaEventCreateWithFlags(&d->done, cudaEventDisableTiming));
    }
    {
        const int rc = mr_op_ctx_id(d->dev, &d->ctx_id);
        if (rc != MR_SUCCESS) return rc;
    }

    /* A kernel already in place (computed by mr_op_compute_kernel) is kept;
     * otherwise it is aliased (shared primary) or copied. */
    const int need_toep = toeplitz && (d->d_toep == NULL);
    if (is_primary && share) {
        /* Alias the caller's arrays: the maps of the primary's coils start at
         * coil0 of the caller's array. */
        d->owns_const = 0;
        d->d_SEs  = (mr_complex *)se_src;
        d->d_mask = mask_bytes ? (int64_t *)in->d_mask : NULL;
        if (need_toep) {
            d->owns_toep = 0;
            d->d_toep = (mr_complex *)d_toep;
        }
    } else {
        d->owns_const = 1;
        cudaError_t ce = cudaSuccess;
        if (need_toep) {
            ce = cudaMalloc((void **)&d->d_toep, toep_bytes);
            if (ce == cudaSuccess) d->owns_toep = 1;
        }
        if (ce == cudaSuccess && se_bytes) ce = cudaMalloc((void **)&d->d_SEs, se_bytes);
        if (ce == cudaSuccess && mask_bytes)
            ce = cudaMalloc((void **)&d->d_mask, mask_bytes);
        if (ce != cudaSuccess) {
            cudaGetLastError();
            MR_FAIL(MR_ERR_ALLOC,
                    "failed to allocate the %.2f GB copy of %sthe maps%s: %s",
                    ((need_toep ? toep_bytes : 0) + se_bytes + mask_bytes) / 1e9,
                    need_toep ? "phi2_toep, " : "", mask_bytes ? " and the mask" : "",
                    cudaGetErrorString(ce));
        }
        if (need_toep)
            MR_CUDA_CHECK(cudaMemcpyPeerAsync(d->d_toep, d->dev, d_toep, p->primary,
                                              toep_bytes, d->stream));
        if (se_bytes)
            MR_CUDA_CHECK(cudaMemcpyPeerAsync(d->d_SEs, d->dev, se_src, p->primary,
                                              se_bytes, d->stream));
        if (mask_bytes)
            MR_CUDA_CHECK(cudaMemcpyPeerAsync(d->d_mask, d->dev, in->d_mask,
                                              p->primary, mask_bytes, d->stream));
        MR_CUDA_CHECK(cudaStreamSynchronize(d->stream));
    }

    if (!is_primary) {
        cudaError_t ce = cudaMalloc((void **)&d->d_x, x_bytes);
        if (ce == cudaSuccess) ce = cudaMalloc((void **)&d->d_part, x_bytes);
        if (ce != cudaSuccess) {
            cudaGetLastError();
            MR_FAIL(MR_ERR_ALLOC, "failed to allocate the x / partial buffers: %s",
                    cudaGetErrorString(ce));
        }
        /* Staging buffer for the partial result, on the primary device. */
        MR_CUDA_CHECK(cudaSetDevice(p->primary));
        ce = cudaMalloc((void **)&d->d_stage, x_bytes);
        if (ce != cudaSuccess) {
            cudaGetLastError();
            MR_FAIL(MR_ERR_ALLOC,
                    "failed to allocate a %.2f GB staging buffer on the primary "
                    "device %d: %s",
                    x_bytes / 1e9, p->primary, cudaGetErrorString(ce));
        }
        MR_CUDA_CHECK(cudaSetDevice(d->dev));
    }

    if (p->nsplit > 0) {
        const int rc = mr_op_split_setup(p, d, in->d_SEs);
        if (rc != MR_SUCCESS) return rc;
    }

    /* The engine of this device's whole coils, on its stream. */
    if (d->ncoil == 0) return MR_SUCCESS;
    mr_config dc = *cfg;
    dc.n_coils = d->ncoil;
    dc.gpu_device_id = d->dev;
    dc.stream = (void *)d->stream;
    dc.verbose = 0;
    if (toeplitz) return mr_aha_plan_create(&dc, &d->plan);
    dc.verbose = cfg->verbose;
    return mr_nufft_eng_create(&dc, in->d_phi2, in->d_traj, p->primary, in->Ntrajs,
                               in->Nkx, p->opt.nufft_upsampfac,
                               p->opt.nufft_gpu_method, &d->nu);
}

/**
 * @brief Compute `phi2_toep` for a Toeplitz operator from the Gram blocks.
 *
 * Allocates the operator's own copy of the kernel on each of the first
 * @p nrecv candidates (the devices that certainly apply it: the @p G1
 * whole-coil devices, which the remainder split also uses, or all @p G
 * candidates when the remainder split is certain) and lets
 * ::mr_toep_distribute fill them, with every candidate computing a share
 * (::MR_TDIST_EXCHANGE) or every receiver computing the whole kernel
 * (::MR_TDIST_SELF), per @p mode. Devices the operator adds later (remainder
 * split chosen by the cost model) copy the kernel from the primary in
 * ::mr_op_dev_setup, as with a caller's kernel. With ::MR_TDIST_ALONE, or
 * when the operator uses one device (@p G1 == 1), the primary computes the
 * kernel alone with ::mr_calc_toep_impl (no threads, no probes) and
 * ::mr_op_dev_setup copies it to any other device.
 *
 * On return `p->ndev` covers the slots holding a kernel so that
 * ::mr_op_release frees them on a later failure.
 *
 * @param[in,out] p      Operator (configuration, primary device and stream).
 * @param[in]     cand   Candidate devices, primary first.
 * @param[in]     G      Number of candidates.
 * @param[in]     G1     Number of whole-coil devices (first @p G1 candidates).
 * @param[in]     nrecv  Number of devices that keep a copy of the kernel
 *                       (@p G1 or @p G; ignored when the primary computes
 *                       the kernel alone).
 * @param[in]     mode   ::mr_tdist_mode from ::mr_toep_dist_resolve.
 * @param[in]     in     Gram blocks, trajectory and mask (primary device).
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
static int mr_op_compute_kernel(mr_op p, const int *cand, int G, int G1, int nrecv,
                                int mode, const mr_op_inputs *in)
{
    const auto t0 = std::chrono::steady_clock::now();
    mr_config cfg = p->cfg; /* gpu_device_id = primary, stream = caller's */
    const size_t toep_bytes = (size_t)mr_phi2_toep_size(&cfg) * sizeof(mr_complex);
    const int alone = (mode == MR_TDIST_ALONE || G1 == 1 || G == 1);
    if (alone) nrecv = 1;

    /* The operator's copies, before any calc_toep transient (whose chunks are
     * sized against the memory left). */
    for (int k = 0; k < nrecv; ++k) {
        mr_op_dev *d = &p->dev[k];
        d->dev = cand[k];
        p->ndev = k + 1;
        MR_CUDA_CHECK(cudaSetDevice(d->dev));
        const cudaError_t ce = cudaMalloc((void **)&d->d_toep, toep_bytes);
        if (ce != cudaSuccess) {
            cudaGetLastError();
            d->d_toep = NULL;
            MR_FAIL(MR_ERR_ALLOC,
                    "device %d: failed to allocate the %.2f GB copy of phi2_toep: %s",
                    d->dev, toep_bytes / 1e9, cudaGetErrorString(ce));
        }
        d->owns_toep = 1;
    }

    int rc;
    if (alone) {
        MR_CUDA_CHECK(cudaSetDevice(p->primary));
        rc = mr_calc_toep_impl(&cfg, in->d_phi2, in->d_traj, in->Ntrajs, in->Nkx,
                               in->d_mask, p->dev[0].d_toep);
        /* Complete before the plan sizes its buffers and before the first
         * execution on a (possibly different) stream. */
        if (rc == MR_SUCCESS) MR_CUDA_CHECK(cudaStreamSynchronize(p->stream));
        p->toep_source = MR_OP_TOEP_BROADCAST;
        p->toep_devices = 1;
    } else {
        mr_toep_dist_dev devs[MR_OP_MAX_DEVICES];
        for (int k = 0; k < G; ++k) {
            devs[k].dev = cand[k];
            devs[k].d_full = (k < nrecv) ? p->dev[k].d_toep : NULL;
        }
        mr_toep_dist_report rep;
        memset(&rep, 0, sizeof(rep));
        rc = mr_toep_distribute(&cfg, devs, G, in->d_phi2, in->d_traj, in->Ntrajs,
                                in->Nkx, in->d_mask, mode, &rep);
        p->toep_source = (rep.mode == MR_TDIST_SELF) ? MR_OP_TOEP_REPLICATE
                                                     : MR_OP_TOEP_SPLIT;
        p->toep_devices = rep.n_devices;
        p->toep_comp_ms = rep.entry_ms;
        p->toep_gbs = rep.gather_gbs;
    }
    p->toep_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
    return rc;
}

/**
 * @brief Check the arguments of ::mr_op_create against the method.
 * @return ::MR_SUCCESS or ::MR_ERR_INVALID_ARG / ::MR_ERR_INVALID_CONFIG.
 */
static int mr_op_check(const mr_config *cfg, const mr_op_inputs *in,
                       const mr_op_options *o)
{
    int rc = mr_config_validate(cfg);
    if (rc != MR_SUCCESS) return rc;
    MR_REQUIRE(o->method == MR_OP_TOEPLITZ || o->method == MR_OP_NUFFT,
               "mr_op_create: method = %d is neither MR_OP_TOEPLITZ nor MR_OP_NUFFT",
               (int)o->method);
    MR_REQUIRE(in->d_SEs != NULL, "mr_op_create: d_SEs is NULL");
    const int need_gram = (o->method == MR_OP_NUFFT || in->d_phi2_toep == NULL);
    if (need_gram) {
        MR_REQUIRE(in->d_phi2 != NULL, "mr_op_create: d_phi2 is NULL");
        MR_REQUIRE(in->d_traj != NULL, "mr_op_create: d_traj is NULL");
        MR_REQUIRE(in->Ntrajs > 0 && in->Nkx > 0,
                   "mr_op_create: Ntrajs = %lld and Nkx = %lld must be > 0",
                   (long long)in->Ntrajs, (long long)in->Nkx);
    }
    if (o->method == MR_OP_NUFFT) {
        MR_REQUIRE(!cfg->is_masked,
                   "mr_op_create: the NUFFT method has no kernel to mask; set "
                   "is_masked = 0 (or use the Toeplitz method)");
        MR_REQUIRE(in->d_phi2_toep == NULL,
                   "mr_op_create: d_phi2_toep is only used by the Toeplitz method");
        MR_REQUIRE(o->nufft_upsampfac == 0.0 || o->nufft_upsampfac > 1.0,
                   "mr_op_create: nufft_upsampfac = %g must be > 1 (or 0 for the "
                   "default 2.0)", o->nufft_upsampfac);
    } else if (cfg->is_masked) {
        MR_REQUIRE(in->d_mask != NULL,
                   "mr_op_create: d_mask is required when is_masked is set");
    }
    return MR_SUCCESS;
}

/* ------------------------------------------------------------------------ */
/* Public API                                                               */
/* ------------------------------------------------------------------------ */

extern "C" int mr_op_options_defaults(mr_op_options *opt)
{
    mr_clear_error();
    MR_REQUIRE(opt != NULL, "mr_op_options_defaults: opt is NULL");
    memset(opt, 0, sizeof(*opt));
    opt->method = MR_OP_TOEPLITZ;
    opt->devices = NULL;
    opt->n_devices = 0;
    opt->toep_dist = MR_TOEP_DIST_AUTO;
    opt->flags = MR_OP_SHARE_PRIMARY;
    opt->nufft_upsampfac = 0.0;
    opt->nufft_gpu_method = 0;
    return MR_SUCCESS;
}

/**
 * @brief Body of ::mr_op_create; may change the current device.
 *
 * Device choice, kernel, remainder split and per-device setup, in this order.
 *
 * Remainder split (Toeplitz, see @ref mr_op_split): all G candidates take
 * q = floor(C/G) whole coils and the r = C % G remaining coils are split by
 * rank. A candidate when the coils do not divide over G, the specialised
 * block-diagonal kernels are available (they take a location range) and the
 * problem is not tiny. `auto` uses it when the cost model predicts a gain of
 * at least 5 %:
 *   t_whole = T(ceil(C/G1)),
 *   t_split = T(q) + r (T(1)/G + t_a2a),
 *   t_a2a   = 2 (G-1)/G^2 Mx*My*Mz*L*8 B / bw,
 * where T(n) is the measured time of the single-GPU operator over n coils on
 * the primary device (with its coil groups) and bw the slowest measured
 * primary -> device bandwidth. Probe failures (e.g. a busy GPU without 64 MB
 * to spare) only disable the split.
 */
static int mr_op_create_impl(const mr_config *cfg, const mr_op_inputs *in,
                             const mr_op_options *o, mr_op *op_out)
{
    const auto t_create = std::chrono::steady_clock::now();
    int rc = mr_op_check(cfg, in, o);
    if (rc != MR_SUCCESS) return rc;
    const int toeplitz = (o->method == MR_OP_TOEPLITZ);
    const int supplied = toeplitz && (in->d_phi2_toep != NULL);

    /* Kernel distribution (checked up front, also on one device, so that a
     * mistyped MR_AHA_TOEP_SPLIT is reported rather than ignored). */
    int tdist = MR_TDIST_AUTO;
    if (toeplitz && !supplied) {
        rc = mr_toep_dist_resolve((int)o->toep_dist, &tdist);
        if (rc != MR_SUCCESS) return rc;
    }

    /* Candidate list: the primary first, then the requested devices in order,
     * without duplicates. */
    int cand[MR_OP_MAX_DEVICES];
    int G = 0, primary = 0;
    rc = mr_op_candidates("mr_op_create", cfg, o->devices, o->n_devices, cand, &G,
                          &primary);
    if (rc != MR_SUCCESS) return rc;

    /* Number of devices for whole-coil assignment (see mr_op_pick_count);
     * tiny problems stay on the primary device. The work of one coil is
     * measured by the upsampled grid times L; a NUFFT operator also works on
     * every sample (spreading, interpolation, Gram multiplication), so it
     * takes the larger of the grid and the sample count. */
    const int C = cfg->n_coils;
    long long per_rank = mr_pad_size(cfg);
    if (!toeplitz) {
        const long long nz = (cfg->traj_type == MR_TRAJ_STACK) ? cfg->phi2_dim[2] : 1;
        const long long ns = (long long)in->Ntrajs * in->Nkx * nz;
        if (ns > per_rank) per_rank = ns;
    }
    const long long work = per_rank * (long long)cfg->L;
    const long long min_work = mr_op_env("MR_AHA_MGPU_MIN_WORK", MR_OP_MIN_WORK);
    const int small = (G > 1 && work < min_work);
    const int G1 = small ? 1 : mr_op_pick_count(C, G);

    mr_op p = (mr_op)calloc(1, sizeof(struct mr_op_s));
    if (p == NULL) MR_FAIL(MR_ERR_ALLOC, "failed to allocate the operator struct");
    p->cfg = *cfg;
    p->cfg.gpu_device_id = primary;
    p->opt = *o;
    p->opt.devices = NULL; /* the caller's array need not outlive the call */
    p->primary = primary;
    p->stream = (cudaStream_t)cfg->stream;
    mr_dims_from_config(cfg, &p->dims);
    const char *gen = std::getenv("MR_AHA_CUDA_GENERIC_BD");
    if (toeplitz && !(gen && gen[0] && atoi(gen) != 0)) p->bd = mr_bd_lookup(cfg->L);

    /* The caller's inputs may still be in flight on its stream. */
    if (cudaSetDevice(primary) != cudaSuccess ||
        cudaStreamSynchronize(p->stream) != cudaSuccess) {
        mr_set_error("mr_op_create: CUDA error on the primary device %d: %s",
                     primary, cudaGetErrorString(cudaGetLastError()));
        mr_op_release(p);
        return MR_ERR_CUDA;
    }

    /* Remainder split (Toeplitz only); see the function comment. */
    const char *se = std::getenv("MR_AHA_MGPU_SPLIT");
    const int mode = (se && !strcmp(se, "on")) ? 1 : (se && !strcmp(se, "off")) ? 0 : 2;
    const int can_split = (toeplitz && mode != 0 && G > 1 && C % G != 0 && !small &&
                           (cfg->is_masked ? p->bd.masked != NULL : p->bd.dense != NULL) &&
                           cfg->num_masked <= 2147483647LL);
    int split = (mode == 1 && can_split);

    /* Kernel from the Gram blocks: computed on the candidates into the
     * operator's own copies (on all G candidates when the split is certain,
     * so none of them computes a share only to receive a full copy later);
     * from here on it is used exactly like a caller's kernel on the
     * primary. */
    const mr_complex *d_toep = in->d_phi2_toep;
    if (toeplitz && !supplied) {
        rc = mr_op_compute_kernel(p, cand, G, G1, split ? G : G1, tdist, in);
        if (rc != MR_SUCCESS) {
            char msg[1024];
            snprintf(msg, sizeof(msg), "%s", mr_last_error());
            mr_op_release(p);
            mr_set_error("mr_op_create: %s", msg);
            return rc;
        }
        d_toep = p->dev[0].d_toep;
    } else {
        p->toep_source = toeplitz ? MR_OP_TOEP_SUPPLIED : MR_OP_TOEP_NONE;
    }

    double bw = 0.0, t_whole = 0.0, t_split = 0.0;
    if (can_split && mode == 2) {
        /* Bandwidth for the cost model (all G candidates). It is not
         * measured otherwise: a probe allocates and copies 2 x 64 MB per
         * device, which is not worth a report field
         * (mr_op_info_t::peer_bandwidth_gbs is then 0). */
        const int nprobe = G;
        double bmin = 1e30;
        int ok = 1;
        for (int k = 1; k < nprobe && ok; ++k) {
            double b = 0.0;
            mr_enable_peer(primary, cand[k]);
            if (mr_op_probe_bw(primary, cand[k], &b) != MR_SUCCESS) ok = 0;
            else if (b < bmin) bmin = b;
        }
        bw = (ok && nprobe > 1) ? bmin : 0.0;
        mr_clear_error(); /* a failed probe is not an error of the operator */
    }
    if (can_split && mode == 2 && bw > 0.0) {
        const int q = C / G, r = C % G, nw = (C + G1 - 1) / G1;
        double t1 = 0.0, tq = 0.0, tw = 0.0;
        int ok = (cudaSetDevice(primary) == cudaSuccess);
        if (ok) ok = (mr_op_probe_coils(p, 1, in->d_SEs, d_toep, in->d_mask, &t1) == MR_SUCCESS);
        if (ok) ok = (mr_op_probe_coils(p, nw, in->d_SEs, d_toep, in->d_mask, &tw) == MR_SUCCESS);
        if (ok && q > 0)
            ok = (mr_op_probe_coils(p, q, in->d_SEs, d_toep, in->d_mask, &tq) == MR_SUCCESS);
        if (ok) {
            const double bytes = 2.0 * (G - 1) / ((double)G * G) *
                                 (double)mr_pad_size(cfg) * cfg->L * sizeof(mr_complex);
            const double t_a2a = bytes / (bw * 1e9) * 1e3;
            t_whole = tw;
            t_split = tq + r * (t1 / G + t_a2a);
            split = (t_split < 0.95 * t_whole);
        }
        mr_clear_error(); /* a failed probe is not an error of the operator */
    }

    const int Gp = split ? G : G1;
    p->ndev = Gp;
    p->direct = (Gp == 1 && !split);
    if (split) {
        /* floor(C/G) whole coils each, then ranks and slabs of the split
         * coils as evenly as possible. */
        const int q = C / G;
        p->nsplit = C % G;
        p->split0 = q * G;
        const long long pad = p->dims.pad_size;
        for (int k = 0; k < Gp; ++k) {
            mr_op_dev *d = &p->dev[k];
            d->dev = cand[k];
            d->coil0 = k * q;
            d->ncoil = q;
            d->r0 = (int)((long long)cfg->L * k / G);
            d->nr = (int)((long long)cfg->L * (k + 1) / G) - d->r0;
            d->k0 = pad * k / G;
            d->nk = pad * (k + 1) / G - d->k0;
        }
    } else {
        /* Whole coils, as evenly as possible; the first C % G' devices take
         * one more. */
        const int q = C / Gp, r = C % Gp;
        int coil = 0;
        for (int k = 0; k < Gp; ++k) {
            p->dev[k].dev = cand[k];
            p->dev[k].coil0 = coil;
            p->dev[k].ncoil = q + (k < r ? 1 : 0);
            coil += p->dev[k].ncoil;
        }
    }

    /* Multi-device execution: the start event on the caller's stream. */
    if (!p->direct) {
        cudaError_t ce = cudaSetDevice(primary);
        if (ce == cudaSuccess)
            ce = cudaEventCreateWithFlags(&p->start, cudaEventDisableTiming);
        if (ce != cudaSuccess) {
            p->start = NULL;
            cudaGetLastError();
            mr_op_release(p);
            MR_FAIL(MR_ERR_CUDA, "mr_op_create: cudaEventCreate failed: %s",
                    cudaGetErrorString(ce));
        }
    }

    /* Peer access: primary <-> every device, and every pair for the split's
     * all-to-all exchange. */
    int peer_all = 1;
    for (int k = 1; k < Gp; ++k) peer_all &= mr_enable_peer(primary, cand[k]);
    if (split)
        for (int a = 1; a < Gp; ++a)
            for (int b = a + 1; b < Gp; ++b) mr_enable_peer(cand[a], cand[b]);

    /* Non-primary devices first: the primary then sizes its coil groups
     * knowing about the staging buffers it holds for the others. */
    for (int k = Gp - 1; k >= 0; --k) {
        rc = mr_op_dev_setup(p, &p->dev[k], in, d_toep, o->flags & MR_OP_SHARE_PRIMARY);
        if (rc != MR_SUCCESS) {
            mr_op_tag(p->dev[k].dev, rc);
            mr_op_release(p);
            return rc;
        }
    }

    mr_op_info_t *nf = &p->info;
    memset(nf, 0, sizeof(*nf));
    nf->method = (int)o->method;
    nf->n_devices = Gp;
    for (int k = 0; k < Gp; ++k) {
        nf->devices[k] = p->dev[k].dev;
        nf->coil_begin[k] = p->dev[k].coil0;
        nf->coil_count[k] = p->dev[k].ncoil;
        nf->group[k] = p->dev[k].plan ? mr_aha_plan_group(p->dev[k].plan)
                                      : (p->dev[k].nu ? 1 : 0);
    }
    nf->n_split_coils = p->nsplit;
    nf->peer_access = (Gp > 1) ? peer_all : 0;
    nf->peer_bandwidth_gbs = (bw < 1e29) ? bw : 0.0;
    nf->toep_source = p->toep_source;
    nf->toep_devices = p->toep_devices;
    nf->toep_ms = p->toep_ms;
    nf->toep_component_ms = p->toep_comp_ms;
    nf->toep_gather_gbs = p->toep_gbs;
    nf->create_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t_create)
                        .count();
    if (cfg->verbose && t_whole > 0.0)
        printf("[mr_aha] multi-GPU cost model: bw %.1f GB/s, whole coils on %d "
               "device(s) %.2f ms, split on %d %.2f ms -> %s\n",
               bw, G1, t_whole, G, t_split, split ? "split" : "whole");

    if (cfg->verbose) {
        printf("[mr_aha] %s operator: %d coil(s) on %d device(s) (primary %d)%s, "
               "created in %.0f ms\n",
               toeplitz ? "Toeplitz" : "NUFFT", cfg->n_coils, Gp, primary,
               (Gp > 1) ? (nf->peer_access ? ", peer access on" : ", peer access off")
                        : "",
               nf->create_ms);
        for (int k = 0; k < Gp; ++k)
            printf("[mr_aha]   device %d: %d whole coil(s) from %d, groups of %d\n",
                   nf->devices[k], nf->coil_count[k], nf->coil_begin[k],
                   nf->group[k]);
        if (p->nsplit)
            printf("[mr_aha]   %d coil(s) from %d split by rank over all devices\n",
                   p->nsplit, p->split0);
    }
    *op_out = p;
    return MR_SUCCESS;
}

extern "C" int mr_op_create(const mr_config *cfg, const mr_op_inputs *in,
                            const mr_op_options *opt, mr_op *op_out)
{
    mr_clear_error();
    MR_REQUIRE(op_out != NULL, "mr_op_create: op_out is NULL");
    *op_out = NULL;
    MR_REQUIRE(cfg != NULL, "mr_op_create: cfg is NULL");
    MR_REQUIRE(in != NULL, "mr_op_create: in is NULL");
    mr_op_options o;
    if (opt == NULL) mr_op_options_defaults(&o);
    else o = *opt;

    int prev = 0;
    MR_CUDA_CHECK(cudaGetDevice(&prev));
    const int rc = mr_op_create_impl(cfg, in, &o, op_out);
    cudaSetDevice(prev);
    return rc;
}

extern "C" int mr_op_kernel(mr_op op, int k, const mr_complex **d_phi2_toep)
{
    mr_clear_error();
    MR_REQUIRE(op != NULL, "mr_op_kernel: op is NULL");
    MR_REQUIRE(d_phi2_toep != NULL, "mr_op_kernel: d_phi2_toep is NULL");
    if (op->opt.method != MR_OP_TOEPLITZ)
        MR_FAIL(MR_ERR_UNSUPPORTED, "mr_op_kernel: a NUFFT operator has no kernel");
    MR_REQUIRE(k >= 0 && k < op->ndev,
               "mr_op_kernel: device index %d outside [0, %d)", k, op->ndev);
    *d_phi2_toep = op->dev[k].d_toep;
    return MR_SUCCESS;
}

/**
 * @brief Apply a device's engine to its whole coils.
 * @param[in]  d   Slot (its device current).
 * @param[in]  x   `x` on the slot's device.
 * @param[out] out Output on the slot's device.
 */
static int mr_op_engine_execute(mr_op_dev *d, const mr_complex *x, mr_complex *out)
{
    if (d->plan) return mr_aha_plan_execute(d->plan, x, d->d_SEs, d->d_toep, d->d_mask, out);
    return mr_nufft_eng_execute(d->nu, x, d->d_SEs, out);
}

/**
 * @brief Issue the remainder-split coils (see @ref mr_op_split).
 *
 * Appended to every device's stream after its whole coils; the results are
 * accumulated into ranks @f$R_g@f$ of each device's output (the caller's
 * output on the primary, the partial buffer elsewhere).
 *
 * @param[in] p    Operator.
 * @param[in] d_x  Caller's input (primary device).
 * @param[in] outs Output of each device.
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
static int mr_op_issue_split(mr_op p, const mr_complex *d_x, mr_complex *const *outs)
{
    const mr_config *cfg = &p->cfg;
    const mr_dims dims = p->dims;
    const long long pad = dims.pad_size, img = dims.image_size;
    const size_t c8 = sizeof(mr_complex);
    const int G = p->ndev;

    for (int j = 0; j < p->nsplit; ++j) {
        /* 1. Pad and transform ranks R_g of the remaining coil j. */
        for (int g = 0; g < G; ++g) {
            mr_op_dev *d = &p->dev[g];
            MR_CUDA_CHECK(cudaSetDevice(d->dev));
            if (d->nr > 0) {
                mr_dims dr = dims;
                dr.L = d->nr;
                const mr_complex *xg = (g == 0) ? d_x : d->d_x;
                mr_k_pad_mul_se<<<(int)mr_ceil_div(pad, MR_OP_BLOCK), MR_OP_BLOCK, 0,
                                  d->stream>>>(d->d_A, xg + img * d->r0, d->d_SEs_rem,
                                               j, dr);
                MR_KERNEL_CHECK("mr_k_pad_mul_se (split)");
                MR_CUFFT_CHECK(cufftExecC2C(d->fft_r, d->d_A, d->d_A, CUFFT_FORWARD));
            }
            MR_CUDA_CHECK(cudaEventRecord(d->ev_fft, d->stream));
        }

        /* 2.-3. Collect all ranks of slab K_h on device h, multiply. */
        for (int h = 0; h < G; ++h) {
            mr_op_dev *dh = &p->dev[h];
            MR_CUDA_CHECK(cudaSetDevice(dh->dev));
            for (int g = 0; g < G; ++g) {
                mr_op_dev *dg = &p->dev[g];
                if (dg->nr == 0) continue;
                MR_CUDA_CHECK(cudaStreamWaitEvent(dh->stream, dg->ev_fft, 0));
                /* One copy per rank: rank r0_g + l of slab K_h. (A single
                 * pitched cudaMemcpy3DPeerAsync would do, but its source
                 * pitch, the whole grid, exceeds cudaDevAttrMaxPitch (2^31
                 * bytes) for grids above 2^28 locations.) */
                for (int l = 0; l < dg->nr; ++l)
                    MR_CUDA_CHECK(cudaMemcpyPeerAsync(
                        dh->d_B + dh->nk * (dg->r0 + l), dh->dev,
                        dg->d_A + pad * l + dh->k0, dg->dev,
                        (size_t)dh->nk * c8, dh->stream));
            }
            const mr_bd_range rg = {dh->k0, dh->nk, dh->nk};
            const int grid = (int)mr_ceil_div(dh->nk, MR_BD_THREADS);
            if (cfg->is_masked) {
                p->bd.masked<<<grid, MR_BD_THREADS, 0, dh->stream>>>(
                    dh->d_B, dh->d_toep, dh->d_inv, (long long)cfg->num_masked, dims,
                    1, rg);
                MR_KERNEL_CHECK("mr_k_bd_masked_t (split)");
            } else {
                p->bd.dense<<<grid, MR_BD_THREADS, 0, dh->stream>>>(
                    dh->d_B, dh->d_toep, dims, 1, rg);
                MR_KERNEL_CHECK("mr_k_bd_dense_t (split)");
            }
            MR_CUDA_CHECK(cudaEventRecord(dh->ev_bd, dh->stream));
        }

        /* 4. Return the slabs, inverse transform and crop ranks R_g. */
        for (int g = 0; g < G; ++g) {
            mr_op_dev *dg = &p->dev[g];
            if (dg->nr == 0) continue;
            MR_CUDA_CHECK(cudaSetDevice(dg->dev));
            for (int h = 0; h < G; ++h) {
                mr_op_dev *dh = &p->dev[h];
                MR_CUDA_CHECK(cudaStreamWaitEvent(dg->stream, dh->ev_bd, 0));
                for (int l = 0; l < dg->nr; ++l) /* see step 2 */
                    MR_CUDA_CHECK(cudaMemcpyPeerAsync(
                        dg->d_A + pad * l + dh->k0, dg->dev,
                        dh->d_B + dh->nk * (dg->r0 + l), dh->dev,
                        (size_t)dh->nk * c8, dg->stream));
            }
            MR_CUFFT_CHECK(cufftExecC2C(dg->fft_r, dg->d_A, dg->d_A, CUFFT_INVERSE));
            mr_dims dr = dims;
            dr.L = dg->nr;
            mr_k_crop_mul_conj_se<<<(int)mr_ceil_div(img, MR_OP_BLOCK), MR_OP_BLOCK,
                                    0, dg->stream>>>(outs[g] + img * dg->r0, dg->d_A,
                                                     dg->d_SEs_rem, j, 1, dr);
            MR_KERNEL_CHECK("mr_k_crop_mul_conj_se (split)");
        }
    }
    return MR_SUCCESS;
}

/** @brief Body of ::mr_op_execute; may change the current device. */
static int mr_op_execute_impl(mr_op p, const mr_complex *d_x, mr_complex *d_AhAx)
{
    const mr_config *cfg = &p->cfg;
    const size_t x_bytes = (size_t)mr_x_size(cfg) * sizeof(mr_complex);
    mr_complex *outs[MR_OP_MAX_DEVICES];

    MR_CUDA_CHECK(cudaSetDevice(p->primary));
    if (p->direct) {
        /* One device: the engine on the caller's stream, arrays in place. */
        return mr_op_engine_execute(&p->dev[0], d_x, d_AhAx);
    }
    MR_CUDA_CHECK(cudaEventRecord(p->start, p->stream));

    /* Whole coils. Other devices first (their copies of x are the longest
     * chain), then the primary, whose engine writes straight into the output.
     * A device without whole coils zeroes its output instead, so that the
     * split coils can accumulate into it. */
    for (int k = p->ndev - 1; k >= 0; --k) {
        mr_op_dev *d = &p->dev[k];
        outs[k] = (k == 0) ? d_AhAx : d->d_part;
        const mr_complex *xk = (k == 0) ? d_x : d->d_x;
        MR_CUDA_CHECK(cudaSetDevice(d->dev));
        MR_CUDA_CHECK(cudaStreamWaitEvent(d->stream, p->start, 0));
        if (k > 0)
            MR_CUDA_CHECK(cudaMemcpyPeerAsync(d->d_x, d->dev, d_x, p->primary,
                                              x_bytes, d->stream));
        if (d->plan || d->nu) {
            const int rc = mr_op_engine_execute(d, xk, outs[k]);
            if (rc != MR_SUCCESS) return mr_op_tag(d->dev, rc);
        } else {
            MR_CUDA_CHECK(cudaMemsetAsync(outs[k], 0, x_bytes, d->stream));
        }
    }

    /* Remaining coils split by rank across all devices. */
    if (p->nsplit > 0) {
        const int rc = mr_op_issue_split(p, d_x, outs);
        if (rc != MR_SUCCESS) return rc;
    }

    /* Partial results back to the primary device. */
    for (int k = 0; k < p->ndev; ++k) {
        mr_op_dev *d = &p->dev[k];
        MR_CUDA_CHECK(cudaSetDevice(d->dev));
        if (k > 0)
            MR_CUDA_CHECK(cudaMemcpyPeerAsync(d->d_stage, p->primary, d->d_part,
                                              d->dev, x_bytes, d->stream));
        MR_CUDA_CHECK(cudaEventRecord(d->done, d->stream));
    }

    /* Reduce on the caller's stream once every device is done. */
    MR_CUDA_CHECK(cudaSetDevice(p->primary));
    mr_op_parts parts;
    memset(&parts, 0, sizeof(parts));
    for (int k = 0; k < p->ndev; ++k) {
        MR_CUDA_CHECK(cudaStreamWaitEvent(p->stream, p->dev[k].done, 0));
        if (k > 0) parts.p[parts.n++] = p->dev[k].d_stage;
    }
    if (parts.n > 0) {
        const long long n = mr_x_size(cfg);
        mr_k_op_reduce<<<(int)mr_ceil_div(n, MR_OP_BLOCK), MR_OP_BLOCK, 0, p->stream>>>(
            d_AhAx, parts, n);
        MR_KERNEL_CHECK("mr_k_op_reduce");
    }
    if (cfg->verbose) {
        MR_CUDA_CHECK(cudaStreamSynchronize(p->stream));
        printf("[mr_aha] AhA applied over %d coil(s) on %d device(s)\n", cfg->n_coils,
               p->ndev);
    }
    return MR_SUCCESS;
}

extern "C" int mr_op_execute(mr_op op, const mr_complex *d_x, mr_complex *d_AhAx)
{
    mr_clear_error();
    MR_REQUIRE(op != NULL, "mr_op_execute: op is NULL");
    MR_REQUIRE(d_x != NULL, "mr_op_execute: d_x is NULL");
    MR_REQUIRE(d_AhAx != NULL, "mr_op_execute: d_AhAx is NULL");
    MR_REQUIRE(d_AhAx != d_x, "mr_op_execute: d_AhAx must not alias d_x");

    int prev = 0;
    MR_CUDA_CHECK(cudaGetDevice(&prev));
    const int rc = mr_op_execute_impl(op, d_x, d_AhAx);
    if (rc != MR_SUCCESS && !op->direct) {
        /* Work issued before the failure may still read x or write AhAx and
         * is not ordered before the caller's stream: wait for it here. */
        for (int k = 0; k < op->ndev; ++k)
            if (cudaSetDevice(op->dev[k].dev) == cudaSuccess)
                cudaStreamSynchronize(op->dev[k].stream);
        cudaGetLastError();
    }
    cudaSetDevice(prev);
    return rc;
}

extern "C" int mr_op_info(mr_op op, mr_op_info_t *info)
{
    mr_clear_error();
    MR_REQUIRE(op != NULL, "mr_op_info: op is NULL");
    MR_REQUIRE(info != NULL, "mr_op_info: info is NULL");
    *info = op->info;
    return MR_SUCCESS;
}

extern "C" int mr_op_destroy(mr_op op)
{
    mr_clear_error();
    if (op == NULL) return MR_SUCCESS;
    int prev = 0;
    const cudaError_t ce = cudaGetDevice(&prev);
    /* Let in-flight work finish before freeing what it uses (on the devices
     * whose context still exists, see ::mr_op_dev_alive). */
    for (int k = 0; k < op->ndev; ++k) {
        if (!mr_op_dev_alive(&op->dev[k])) continue;
        if (op->dev[k].stream) cudaStreamSynchronize(op->dev[k].stream);
        if (k == 0) cudaStreamSynchronize(op->stream); /* NULL: legacy stream */
    }
    cudaGetLastError();
    mr_op_release(op);
    if (ce == cudaSuccess) cudaSetDevice(prev);
    return MR_SUCCESS;
}

extern "C" int mr_op_abandon(mr_op op)
{
    mr_clear_error();
    if (op == NULL) return MR_SUCCESS;
    int prev = 0;
    const cudaError_t ce = cudaGetDevice(&prev);
    /* Work still queued on the other devices may wait for events of the dead
     * primary; synchronising could hang, so their streams are destroyed
     * directly (cudaStreamDestroy returns at once and releases the stream
     * when its work is done). */
    mr_op_release(op, 1);
    if (ce == cudaSuccess) cudaSetDevice(prev);
    return MR_SUCCESS;
}
