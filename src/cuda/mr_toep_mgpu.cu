/**
 * @file mr_toep_mgpu.cu
 * @brief Multi-GPU pre-computation of the Toeplitz kernel `phi2_toep`.
 *
 * ::mr_calc_toep evaluates the @f$T = (L^2+L)/2@f$ packed triangular
 * components of `phi2_toep` independently of one another: each is one type-1
 * NUFFT, one centred inverse FFT and one scaling, and lands in its own
 * contiguous slab of the output (`Mx*My*Mz` elements unmasked,
 * `num_masked` elements masked). A range of components is therefore one
 * contiguous byte range, which can be computed on any device and moved with a
 * single `cudaMemcpyPeerAsync`. This file spreads the components over several
 * devices of one node.
 *
 * @section mr_toep_mgpu_roles Receivers and helpers
 *
 * A *receiver* needs the whole kernel in its own memory: the primary device
 * for ::mr_calc_toep_mgpu, every device of a multi-GPU @f$A^H A@f$ plan for
 * ::mr_mgpu_plan_create_gram (each of them applies the whole kernel to its
 * own coils). A *helper* only computes components for the receivers and
 * keeps nothing.
 *
 * @section mr_toep_mgpu_flow Flow of ::mr_toep_distribute
 *
 *  1. The components are split into contiguous, nearly equal ranges, one per
 *     device (the primary takes the first); chunks are capped at a quarter of
 *     a range so that transfers can start early.
 *  2. *Bandwidth probe* (mode `auto` only): every device copies 64 MB to
 *     every receiver, all devices at the same time and with the schedule of
 *     the exchange (step 5); a receiver's aggregate receive rate is the
 *     bandwidth of the cost model. Measuring all copies together matters on
 *     PCIe, where several GPUs may share links between CPU sockets.
 *  3. *Setup and pilot* (one host thread per device): copy the small inputs
 *     (`phi2`, trajectory, mask) from the primary, plan (::mr_toep_ctx_create),
 *     compute and time the first chunk of the own range. The pilot is useful
 *     work in every mode.
 *  4. *Decision* (caller's thread). With @f$t_e@f$ the slowest measured time
 *     per component, @f$c@f$ the components each device still has to compute,
 *     @f$R@f$ the time a receiver needs to receive all components it does not
 *     compute (at the probed bandwidth) and @f$k@f$ the chunks per device:
 *     @f[
 *       T_\mathrm{exchange} \approx \max(c\,t_e, R) + R/k, \qquad
 *       T_\mathrm{self} \approx (T - p)\,t_e,
 *     @f]
 *     where @f$p@f$ is what a receiver computed in the pilot. The exchange is
 *     chosen when it is faster; roughly, when a receiver can receive a
 *     component faster than it can compute one. `MR_AHA_TOEP_SPLIT` forces a
 *     mode.
 *  5. *Exchange* (one thread per device): compute the rest of the own range;
 *     after each chunk, record an event and copy the chunk to every other
 *     receiver, one copy at a time in rotated order on the device's copy
 *     stream (see ::mr_tpush_chunk), so transfers overlap with the next
 *     chunks and every link carries about one transfer at a time. Every component is computed exactly once, so all receivers
 *     end with bit-identical kernels.
 *     *Self*: every receiver computes the components it lacks; helpers stop.
 *     No transfer at all; the receivers' kernels differ at the level of
 *     cufinufft's run-to-run noise (~1e-7, atomic spreading).
 *  6. Wait for every stream, free everything but the receivers' kernels.
 *
 * Errors are thread-local (see ::mr_last_error), so each worker copies its
 * diagnostic into its own slot and the caller's thread reports the first
 * failure, prefixed with the device ordinal. Worker threads never call into
 * MATLAB or Python.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_kernels.cuh"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

/** @brief Bytes of each copy of the concurrent bandwidth probe. */
#define MR_TOEP_PROBE_BYTES ((size_t)64 << 20)

/** @brief Default minimum number of chunks a device's range is cut into, so
 *         that the exchange overlaps with the computation: the chunk size is
 *         capped at `ceil(range / chunks)`. `MR_AHA_TOEP_CHUNKS` overrides it
 *         (1: one chunk when memory allows, i.e. no overlap; for
 *         benchmarking). */
#define MR_TOEP_MIN_CHUNKS 4

/* ------------------------------------------------------------------------ */
/* State                                                                    */
/* ------------------------------------------------------------------------ */

/** @brief Per-device state of ::mr_toep_distribute. */
struct mr_tdev {
    int          dev;       /**< CUDA ordinal.                                */
    mr_complex  *d_full;    /**< Whole-kernel destination (receivers) or NULL. */
    mr_complex  *d_slice;   /**< Own range only (helpers), owned.             */
    mr_complex  *d_own;     /**< Where the own range is written.              */
    int64_t      own_t0;    /**< Component held by slab 0 of @ref d_own.      */
    int64_t      r0, r1;    /**< Own range `[r0, r1)`.                        */
    int64_t      done;      /**< Own components computed so far (pilot).      */
    cudaStream_t s;         /**< Compute stream.                              */
    cudaStream_t cs;        /**< The one stream of all outgoing copies.       */
    cudaEvent_t  ev;        /**< Recorded after each chunk on @ref s.         */
    mr_complex  *d_phi2;    /**< Inputs on this device (copies, except on     */
    float       *d_traj;    /**< the primary, where they are the caller's).   */
    int64_t     *d_mask;
    int          owns_in;   /**< Non-zero when the inputs above are copies.   */
    mr_toep_ctx *ctx;       /**< calc_toep context, or NULL.                  */
    float        pilot_ms;  /**< GPU time of the pilot chunk.                 */
    int          rc;        /**< Status of the last phase on this device.     */
    char         msg[512];  /**< Its diagnostic.                              */
};

/** @brief Everything the worker threads share (read-only during a phase). */
struct mr_tjob {
    mr_config         cfg;      /**< Configuration (primary's stream).        */
    mr_tdev          *d;        /**< Device slots, `d[0]` the primary.        */
    int               G;        /**< Number of slots.                         */
    int64_t           ntri;     /**< `(L^2+L)/2`.                             */
    int64_t           slab;     /**< Elements per component.                  */
    int64_t           Ntrajs;   /**< Inputs, see ::mr_calc_toep.               */
    int64_t           Nkx;
    const mr_complex *phi2;     /**< Inputs on the primary device.            */
    const float      *traj;
    const int64_t    *mask;
    int64_t           max_chunk;/**< Chunk cap (0: none).                     */
    int               mode;     /**< Phase 2: EXCHANGE or SELF.               */
};

/** @brief Argument of the chunk callback: one device of a job. */
struct mr_tpush {
    mr_tjob *job;               /**< The job.                                 */
    int      k;                 /**< Slot of the sending device.              */
};

/* ------------------------------------------------------------------------ */
/* Helpers                                                                  */
/* ------------------------------------------------------------------------ */

int mr_toep_split_env(void)
{
    const char *e = std::getenv("MR_AHA_TOEP_SPLIT");
    if (e == NULL || e[0] == '\0') return MR_TOEP_DIST_AUTO;
    if (!strcmp(e, "off")) return -1;
    if (!strcmp(e, "split")) return MR_TOEP_DIST_EXCHANGE;
    if (!strcmp(e, "replicate")) return MR_TOEP_DIST_SELF;
    return MR_TOEP_DIST_AUTO;
}

int mr_mgpu_candidates(const char *who, const mr_config *cfg, const int *devices,
                       int n_devices, int *cand, int *G_out, int *primary_out)
{
    MR_REQUIRE(n_devices >= 0 && n_devices <= MR_MGPU_MAX_DEVICES,
               "%s: n_devices must be in [0, %d]", who, MR_MGPU_MAX_DEVICES);
    int ndev_visible = 0;
    MR_CUDA_CHECK(cudaGetDeviceCount(&ndev_visible));
    int primary = cfg->gpu_device_id;
    if (primary < 0) MR_CUDA_CHECK(cudaGetDevice(&primary));
    MR_REQUIRE(primary < ndev_visible, "%s: primary device %d is not visible", who,
               primary);

    /* The primary first, then the requested (or all visible) devices in
     * order, without duplicates. */
    int G = 0;
    cand[G++] = primary;
    const int explicit_list = (devices != NULL && n_devices > 0);
    const int nreq = explicit_list ? n_devices : ndev_visible;
    for (int k = 0; k < nreq; ++k) {
        const int dv = explicit_list ? devices[k] : k;
        MR_REQUIRE(dv >= 0 && dv < ndev_visible,
                   "%s: device %d is not visible (%d visible)", who, dv,
                   ndev_visible);
        int dup = 0;
        for (int j = 0; j < G; ++j) dup |= (cand[j] == dv);
        if (dup) continue;
        if (!explicit_list) {
            if (G == MR_MGPU_MAX_DEVICES) break; /* all visible: use the first 16 */
        } else {
            MR_REQUIRE(G < MR_MGPU_MAX_DEVICES, "%s: at most %d devices are supported",
                       who, MR_MGPU_MAX_DEVICES);
        }
        cand[G++] = dv;
    }
    *G_out = G;
    *primary_out = primary;
    return MR_SUCCESS;
}

/** @brief Record a failure of slot @p d (status and this thread's message). */
static int mr_tfail(mr_tdev *d, int rc)
{
    d->rc = rc;
    if (rc != MR_SUCCESS) snprintf(d->msg, sizeof(d->msg), "%s", mr_last_error());
    return rc;
}

/**
 * @brief Run @p fn(job, k) for every slot k, one host thread each, and wait.
 *
 * Falls back to running a slot on the calling thread when a thread cannot be
 * started (the result is the same, only slower).
 */
static void mr_tparallel(mr_tjob *job, void (*fn)(mr_tjob *, int))
{
    std::thread th[MR_MGPU_MAX_DEVICES];
    int started[MR_MGPU_MAX_DEVICES] = {0};
    for (int k = 0; k < job->G; ++k) {
        try {
            th[k] = std::thread(fn, job, k);
            started[k] = 1;
        } catch (...) {
            started[k] = 0;
        }
    }
    for (int k = 0; k < job->G; ++k) {
        if (started[k]) th[k].join();
        else fn(job, k);
    }
}

/**
 * @brief Chunk callback of the exchange: copy components `[t0, t0+n)` of
 *        slot `k` to every other receiver, behind an event on its stream.
 *
 * Schedule. All copies of a slot go through its *one* copy stream, to the
 * receivers in rotated order `k+1, k+2, ... (mod G)`. Since the devices
 * compute at the same rate, at any time slot `k` is sending its `s`-th copy
 * to `k+s` while every other slot `i` sends to `i+s`: a permutation, so each
 * device has about one outgoing and one incoming transfer at a time, like a
 * ring all-gather but without forwarding. Copying to all receivers at once
 * (one stream per destination, @f$G(G-1)@f$ concurrent transfers) was
 * measured to collapse to 1-3 GB/s per device on PCIe nodes whose GPUs sit on
 * different NUMA nodes (4 L40S, `SYS` topology), several times slower than
 * the copies one at a time; one transfer per link is also what NVLink and
 * NVSwitch serve at full rate.
 */
static int mr_tpush_chunk(void *user, int64_t t0, int64_t n)
{
    mr_tpush *pu = (mr_tpush *)user;
    mr_tjob *job = pu->job;
    mr_tdev *d = &job->d[pu->k];
    if (n <= 0) return MR_SUCCESS;
    const size_t bytes = (size_t)n * (size_t)job->slab * sizeof(mr_complex);
    const mr_complex *src = d->d_own + (size_t)(t0 - d->own_t0) * job->slab;
    MR_CUDA_CHECK(cudaEventRecord(d->ev, d->s));
    MR_CUDA_CHECK(cudaStreamWaitEvent(d->cs, d->ev, 0));
    for (int s = 1; s < job->G; ++s) {
        mr_tdev *r = &job->d[(pu->k + s) % job->G];
        if (r->d_full == NULL) continue;
        MR_CUDA_CHECK(cudaMemcpyPeerAsync(r->d_full + (size_t)t0 * job->slab, r->dev,
                                          src, d->dev, bytes, d->cs));
    }
    return MR_SUCCESS;
}

/* ------------------------------------------------------------------------ */
/* Phases (one host thread per device)                                      */
/* ------------------------------------------------------------------------ */

/** @brief Body of phase 1 for slot @p k; returns a status. */
static int mr_tphase1_body(mr_tjob *job, int k)
{
    mr_tdev *d = &job->d[k];
    const mr_config *cfg = &job->cfg;
    const int primary = job->d[0].dev;

    MR_CUDA_CHECK(cudaSetDevice(d->dev));
    MR_CUDA_CHECK(cudaStreamCreateWithFlags(&d->s, cudaStreamNonBlocking));
    MR_CUDA_CHECK(cudaEventCreateWithFlags(&d->ev, cudaEventDisableTiming));
    MR_CUDA_CHECK(cudaStreamCreateWithFlags(&d->cs, cudaStreamNonBlocking));

    /* Inputs: the caller's on the primary, copies elsewhere. */
    if (k == 0) {
        d->d_phi2 = (mr_complex *)job->phi2;
        d->d_traj = (float *)job->traj;
        d->d_mask = (int64_t *)job->mask;
    } else {
        const int nkz = (cfg->traj_type == MR_TRAJ_STACK) ? cfg->phi2_dim[2] : 1;
        const int ndim = (cfg->traj_type == MR_TRAJ_3D) ? 3 : 2;
        const size_t phi2_bytes =
            (size_t)job->Ntrajs * nkz * cfg->L * cfg->L * sizeof(mr_complex);
        const size_t traj_bytes = (size_t)job->Ntrajs * job->Nkx * ndim * sizeof(float);
        const size_t mask_bytes =
            job->mask ? (size_t)cfg->num_masked * sizeof(int64_t) : 0;
        d->owns_in = 1;
        cudaError_t ce = cudaMalloc((void **)&d->d_phi2, phi2_bytes);
        if (ce == cudaSuccess) ce = cudaMalloc((void **)&d->d_traj, traj_bytes);
        if (ce == cudaSuccess && mask_bytes)
            ce = cudaMalloc((void **)&d->d_mask, mask_bytes);
        if (ce != cudaSuccess) {
            cudaGetLastError();
            MR_FAIL(MR_ERR_ALLOC, "failed to allocate the calc_toep inputs: %s",
                    cudaGetErrorString(ce));
        }
        MR_CUDA_CHECK(cudaMemcpyPeerAsync(d->d_phi2, d->dev, job->phi2, primary,
                                          phi2_bytes, d->s));
        MR_CUDA_CHECK(cudaMemcpyPeerAsync(d->d_traj, d->dev, job->traj, primary,
                                          traj_bytes, d->s));
        if (mask_bytes)
            MR_CUDA_CHECK(cudaMemcpyPeerAsync(d->d_mask, d->dev, job->mask, primary,
                                              mask_bytes, d->s));
    }

    /* Destination of the own range: the whole kernel (receiver) or a buffer
     * holding the range only (helper). Allocated before the context so that
     * the chunk sizing sees the memory left. */
    if (d->d_full != NULL) {
        d->d_own = d->d_full;
        d->own_t0 = 0;
    } else if (d->r1 > d->r0) {
        const size_t bytes =
            (size_t)(d->r1 - d->r0) * (size_t)job->slab * sizeof(mr_complex);
        if (cudaMalloc((void **)&d->d_slice, bytes) != cudaSuccess) {
            cudaGetLastError();
            MR_FAIL(MR_ERR_ALLOC,
                    "failed to allocate %.2f GB for components %lld-%lld",
                    bytes / 1e9, (long long)d->r0, (long long)(d->r1 - 1));
        }
        d->d_own = d->d_slice;
        d->own_t0 = d->r0;
    }
    if (d->r1 <= d->r0) return MR_SUCCESS; /* nothing to compute (yet) */

    mr_config dc = *cfg;
    dc.gpu_device_id = d->dev;
    dc.stream = (void *)d->s;
    dc.verbose = 0;
    int rc = mr_toep_ctx_create(&dc, d->d_phi2, d->d_traj, job->Ntrajs, job->Nkx,
                                job->max_chunk, &d->ctx);
    if (rc != MR_SUCCESS) return rc;

    /* Pilot: the first chunk of the own range, timed. */
    const int64_t n = d->r1 - d->r0;
    const int64_t p = (mr_toep_ctx_chunk(d->ctx) < n) ? mr_toep_ctx_chunk(d->ctx) : n;
    rc = mr_toep_ctx_run(d->ctx, d->d_mask, d->r0, d->r0 + p, d->d_own, d->own_t0,
                         NULL, NULL, &d->pilot_ms);
    if (rc != MR_SUCCESS) return rc;
    d->done = p;
    return MR_SUCCESS;
}

/** @brief Phase 1 (setup and pilot) of slot @p k. */
static void mr_tphase1(mr_tjob *job, int k)
{
    mr_clear_error();
    mr_tfail(&job->d[k], mr_tphase1_body(job, k));
}

/** @brief Body of phase 2 for slot @p k; returns a status. */
static int mr_tphase2_body(mr_tjob *job, int k)
{
    mr_tdev *d = &job->d[k];
    MR_CUDA_CHECK(cudaSetDevice(d->dev));

    if (job->mode == MR_TOEP_DIST_EXCHANGE) {
        if (d->r1 > d->r0) {
            mr_tpush pu = {job, k};
            int rc = mr_tpush_chunk(&pu, d->r0, d->done); /* the pilot */
            if (rc != MR_SUCCESS) return rc;
            rc = mr_toep_ctx_run(d->ctx, d->d_mask, d->r0 + d->done, d->r1, d->d_own,
                                 d->own_t0, mr_tpush_chunk, &pu, NULL);
            if (rc != MR_SUCCESS) return rc;
        }
    } else if (d->d_full != NULL) {
        /* Self: compute every component this receiver does not hold yet. */
        if (d->ctx == NULL) {
            mr_config dc = job->cfg;
            dc.gpu_device_id = d->dev;
            dc.stream = (void *)d->s;
            dc.verbose = 0;
            const int rc = mr_toep_ctx_create(&dc, d->d_phi2, d->d_traj, job->Ntrajs,
                                              job->Nkx, 0, &d->ctx);
            if (rc != MR_SUCCESS) return rc;
        }
        const int64_t a = (d->r1 > d->r0) ? d->r0 : 0;
        const int64_t b = (d->r1 > d->r0) ? d->r0 + d->done : 0;
        int rc = mr_toep_ctx_run(d->ctx, d->d_mask, 0, a, d->d_full, 0, NULL, NULL,
                                 NULL);
        if (rc == MR_SUCCESS)
            rc = mr_toep_ctx_run(d->ctx, d->d_mask, b, job->ntri, d->d_full, 0, NULL,
                                 NULL, NULL);
        if (rc != MR_SUCCESS) return rc;
    }

    /* Wait for the computation and every outgoing copy. */
    MR_CUDA_CHECK(cudaStreamSynchronize(d->s));
    MR_CUDA_CHECK(cudaStreamSynchronize(d->cs));
    return MR_SUCCESS;
}

/** @brief Phase 2 (exchange or self) of slot @p k. */
static void mr_tphase2(mr_tjob *job, int k)
{
    mr_clear_error();
    mr_tfail(&job->d[k], mr_tphase2_body(job, k));
}

/** @brief Release everything slot @p d holds except the receiver's kernel. */
static void mr_tcleanup(mr_tdev *d)
{
    if (cudaSetDevice(d->dev) != cudaSuccess) {
        cudaGetLastError();
        return;
    }
    if (d->ctx) mr_toep_ctx_destroy(d->ctx); /* synchronises its stream */
    d->ctx = NULL;
    if (d->cs) {
        cudaStreamSynchronize(d->cs);
        cudaStreamDestroy(d->cs);
        d->cs = NULL;
    }
    if (d->s) {
        cudaStreamSynchronize(d->s);
        cudaStreamDestroy(d->s);
    }
    if (d->ev) cudaEventDestroy(d->ev);
    if (d->d_slice) cudaFree(d->d_slice);
    if (d->owns_in) {
        if (d->d_phi2) cudaFree(d->d_phi2);
        if (d->d_traj) cudaFree(d->d_traj);
        if (d->d_mask) cudaFree(d->d_mask);
    }
    d->s = NULL;
    d->ev = NULL;
    d->d_slice = NULL;
    d->d_phi2 = NULL;
    d->d_traj = NULL;
    d->d_mask = NULL;
    cudaGetLastError();
}

/**
 * @brief Bandwidth probe with the schedule of the exchange: every slot copies
 *        ::MR_TOEP_PROBE_BYTES to every receiver, all slots at the same time,
 *        each through one stream in rotated order (see ::mr_tpush_chunk).
 *
 * Runs on the caller's thread before the workers start. Measuring all slots
 * together matters on PCIe, where several GPUs may share links between CPU
 * sockets or NUMA nodes.
 *
 * @param[in]  job Job (slots with `dev` and `d_full` set).
 * @param[out] gbs Aggregate receive rate of a receiver in GB/s (every
 *                 receiver receives the same amount; the slowest pace sets
 *                 the measured time).
 * @return ::MR_SUCCESS or a negative ::mr_status (e.g. no memory to spare);
 *         a failure only disables the cost model.
 */
static int mr_tprobe(mr_tjob *job, double *gbs)
{
    const int G = job->G;
    const size_t B = MR_TOEP_PROBE_BYTES;
    void *src[MR_MGPU_MAX_DEVICES] = {0};
    void *dst[MR_MGPU_MAX_DEVICES] = {0};
    cudaStream_t st[MR_MGPU_MAX_DEVICES] = {0};

    cudaError_t ce = cudaSuccess;
    for (int i = 0; i < G && ce == cudaSuccess; ++i) {
        ce = cudaSetDevice(job->d[i].dev);
        if (ce == cudaSuccess) ce = cudaMalloc(&src[i], B);
        /* One slot per sender on a receiver. */
        if (ce == cudaSuccess && job->d[i].d_full != NULL)
            ce = cudaMalloc(&dst[i], B * (size_t)G);
        if (ce == cudaSuccess) ce = cudaStreamCreateWithFlags(&st[i], cudaStreamNonBlocking);
    }
    double secs = 0.0;
    if (ce == cudaSuccess) {
        /* Round 0 warms up (peer mappings); rounds 1-2 are timed. */
        std::chrono::steady_clock::time_point t0;
        for (int round = 0; round < 3 && ce == cudaSuccess; ++round) {
            if (round == 1) t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < G && ce == cudaSuccess; ++i) {
                ce = cudaSetDevice(job->d[i].dev);
                for (int s = 1; s < G && ce == cudaSuccess; ++s) {
                    const int j = (i + s) % G;
                    if (job->d[j].d_full == NULL) continue;
                    ce = cudaMemcpyPeerAsync((char *)dst[j] + B * (size_t)i,
                                             job->d[j].dev, src[i], job->d[i].dev, B,
                                             st[i]);
                }
            }
            for (int i = 0; i < G && ce == cudaSuccess; ++i) {
                ce = cudaSetDevice(job->d[i].dev);
                if (ce == cudaSuccess) ce = cudaStreamSynchronize(st[i]);
            }
        }
        secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                   .count();
    }
    for (int i = 0; i < G; ++i) {
        if (cudaSetDevice(job->d[i].dev) != cudaSuccess) continue;
        if (st[i]) cudaStreamDestroy(st[i]);
        if (src[i]) cudaFree(src[i]);
        if (dst[i]) cudaFree(dst[i]);
    }
    if (ce != cudaSuccess) {
        cudaGetLastError();
        MR_FAIL(MR_ERR_CUDA, "concurrent bandwidth probe failed: %s",
                cudaGetErrorString(ce));
    }
    /* Every receiver receives from the G-1 other slots, twice. */
    *gbs = (secs > 0.0) ? 2.0 * (G - 1) * (double)B / secs / 1e9 : 0.0;
    return MR_SUCCESS;
}

/* ------------------------------------------------------------------------ */
/* Engine                                                                   */
/* ------------------------------------------------------------------------ */

int mr_toep_distribute(const mr_config *cfg, const mr_toep_dist_dev *devs, int G,
                       const mr_complex *d_phi2, const float *d_traj,
                       int64_t Ntrajs, int64_t Nkx, const int64_t *d_mask,
                       int mode, mr_toep_dist_report *report)
{
    const auto t_start = std::chrono::steady_clock::now();
    MR_REQUIRE(G >= 1 && G <= MR_MGPU_MAX_DEVICES && devs[0].d_full != NULL,
               "mr_toep_distribute: bad device list");
    const int64_t ntri = mr_num_tri(cfg->L);
    const int64_t slab = cfg->is_masked ? (int64_t)cfg->num_masked : mr_pad_size(cfg);

    /* Helpers beyond the number of components would have nothing to do. */
    mr_tdev *d = (mr_tdev *)calloc(MR_MGPU_MAX_DEVICES, sizeof(mr_tdev));
    if (d == NULL) MR_FAIL(MR_ERR_ALLOC, "failed to allocate the device slots");
    int n = 0;
    for (int k = 0; k < G; ++k) {
        if (devs[k].d_full == NULL && k >= ntri) continue;
        d[n].dev = devs[k].dev;
        d[n].d_full = devs[k].d_full;
        ++n;
    }
    /* Contiguous, nearly equal ranges; the primary takes the first. Slots
     * with index >= ntri (receivers only) get an empty range. */
    const int nc = (n < ntri) ? n : (int)ntri;
    for (int k = 0; k < n; ++k) {
        d[k].r0 = (k < nc) ? ntri * k / nc : ntri;
        d[k].r1 = (k < nc) ? ntri * (k + 1) / nc : ntri;
    }

    mr_tjob job;
    memset(&job, 0, sizeof(job));
    job.cfg = *cfg;
    job.d = d;
    job.G = n;
    job.ntri = ntri;
    job.slab = slab;
    job.Ntrajs = Ntrajs;
    job.Nkx = Nkx;
    job.phi2 = d_phi2;
    job.traj = d_traj;
    job.mask = d_mask;
    /* Several chunks per range let the exchange overlap with the computation;
     * pure self-computation needs no cap. */
    const int64_t per = (ntri + nc - 1) / nc;
    int64_t nchunks = MR_TOEP_MIN_CHUNKS;
    {
        const char *e = std::getenv("MR_AHA_TOEP_CHUNKS");
        if (e && e[0] && atoll(e) >= 1) nchunks = atoll(e);
    }
    job.max_chunk = (mode == MR_TOEP_DIST_SELF) ? 0 : (per + nchunks - 1) / nchunks;

    int rc = MR_SUCCESS;
    mr_toep_dist_report rep;
    memset(&rep, 0, sizeof(rep));
    rep.n_devices = nc;

    /* Inputs produced on the caller's stream must be complete. */
    {
        const cudaError_t ce = cudaSetDevice(d[0].dev);
        const cudaError_t ce2 = (ce == cudaSuccess)
                                    ? cudaStreamSynchronize((cudaStream_t)cfg->stream)
                                    : ce;
        if (ce2 != cudaSuccess) {
            free(d);
            MR_FAIL(MR_ERR_CUDA, "mr_toep_distribute: primary device %d: %s",
                    devs[0].dev, cudaGetErrorString(ce2));
        }
    }

    /* Peer access: every slot to every receiver (the primary is one, which
     * also covers the input copies). Without it, copies are staged through
     * the host. */
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (d[i].d_full != NULL || d[j].d_full != NULL) mr_enable_peer(d[i].dev, d[j].dev);

    /* Wall-clock milliseconds since t. */
    auto ms_since = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t)
            .count();
    };

    /* Bandwidth for the cost model. */
    auto t_ph = std::chrono::steady_clock::now();
    if (mode == MR_TOEP_DIST_AUTO && n > 1) {
        if (mr_tprobe(&job, &rep.gather_gbs) != MR_SUCCESS) rep.gather_gbs = 0.0;
        mr_clear_error(); /* a failed probe is not an error of the call */
    }
    rep.probe_ms = ms_since(t_ph);

    /* Phase 1: setup and pilot. */
    t_ph = std::chrono::steady_clock::now();
    mr_tparallel(&job, mr_tphase1);
    rep.phase1_ms = ms_since(t_ph);
    for (int k = 0; k < n && rc == MR_SUCCESS; ++k)
        if (d[k].rc != MR_SUCCESS) {
            rc = d[k].rc;
            mr_set_error("device %d: %s", d[k].dev, d[k].msg);
        }

    /* Decision. */
    if (rc == MR_SUCCESS) {
        double te = 0.0;
        int64_t c_rem = 0, k_chunks = 1;
        for (int k = 0; k < n; ++k) {
            if (d[k].done > 0) {
                const double t = (double)d[k].pilot_ms / (double)d[k].done;
                if (t > te) te = t;
                const int64_t rem = d[k].r1 - d[k].r0 - d[k].done;
                if (rem > c_rem) c_rem = rem;
                const int64_t ch = (d[k].r1 - d[k].r0 + d[k].done - 1) / d[k].done;
                if (ch > k_chunks) k_chunks = ch;
            }
        }
        double t_self = 0.0, recv = 0.0;
        for (int k = 0; k < n; ++k) {
            if (d[k].d_full == NULL) continue;
            const double s = (double)(ntri - d[k].done) * te;
            if (s > t_self) t_self = s;
            const double bytes =
                (double)(ntri - (d[k].r1 - d[k].r0)) * slab * sizeof(mr_complex);
            const double r = (rep.gather_gbs > 0.0) ? bytes / (rep.gather_gbs * 1e6) : 0.0;
            if (r > recv) recv = r;
        }
        const double comp = (double)c_rem * te;
        const double t_x = ((comp > recv) ? comp : recv) + recv / (double)k_chunks;
        rep.entry_ms = te;
        rep.t_exchange_ms = t_x;
        rep.t_self_ms = t_self;
        if (mode == MR_TOEP_DIST_AUTO)
            job.mode = (rep.gather_gbs <= 0.0 || t_x < t_self) ? MR_TOEP_DIST_EXCHANGE
                                                               : MR_TOEP_DIST_SELF;
        else
            job.mode = mode;
        rep.mode = job.mode;

        /* Phase 2: exchange or self. */
        t_ph = std::chrono::steady_clock::now();
        mr_tparallel(&job, mr_tphase2);
        rep.phase2_ms = ms_since(t_ph);
        for (int k = 0; k < n && rc == MR_SUCCESS; ++k)
            if (d[k].rc != MR_SUCCESS) {
                rc = d[k].rc;
                mr_set_error("device %d: %s", d[k].dev, d[k].msg);
            }
    }

    /* Keep the first diagnostic across the cleanup. */
    char msg[512];
    snprintf(msg, sizeof(msg), "%s", mr_last_error());
    t_ph = std::chrono::steady_clock::now();
    for (int k = 0; k < n; ++k) mr_tcleanup(&d[k]);
    rep.cleanup_ms = ms_since(t_ph);
    if (rc != MR_SUCCESS) mr_set_error("%s", msg);
    else mr_clear_error();

    rep.total_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - t_start)
                       .count();
    const char *env_rep = std::getenv("MR_AHA_TOEP_REPORT");
    if ((cfg->verbose || (env_rep && atoi(env_rep) != 0)) && rc == MR_SUCCESS) {
        printf("[mr_aha] calc_toep on %d device(s) (", nc);
        for (int k = 0; k < n; ++k)
            printf("%s%d%s", k ? " " : "", d[k].dev, d[k].d_full ? "" : "h");
        printf("): %s, chunk %lld, %.3f ms/component, receive %.1f GB/s, model "
               "exchange %.0f ms / self %.0f ms; probe %.0f, setup+pilot %.0f, "
               "%s %.0f, cleanup %.0f, total %.0f ms\n",
               rep.mode == MR_TOEP_DIST_EXCHANGE ? "split + exchange" : "replicate",
               (long long)(d[0].done), rep.entry_ms, rep.gather_gbs,
               rep.t_exchange_ms, rep.t_self_ms, rep.probe_ms, rep.phase1_ms,
               rep.mode == MR_TOEP_DIST_EXCHANGE ? "exchange" : "self",
               rep.phase2_ms, rep.cleanup_ms, rep.total_ms);
        fflush(stdout);
    }
    if (report) *report = rep;
    free(d);
    return rc;
}

/* ------------------------------------------------------------------------ */
/* Public API: approach D, the kernel on one device computed by several     */
/* ------------------------------------------------------------------------ */

/** @brief Body of ::mr_calc_toep_mgpu; may change the current device. */
static int mr_calc_toep_mgpu_impl(const mr_config *cfg, const int *devices,
                                  int n_devices, const mr_complex *d_phi2,
                                  const float *d_traj, int64_t Ntrajs, int64_t Nkx,
                                  const int64_t *d_mask, mr_complex *d_phi2_toep)
{
    int cand[MR_MGPU_MAX_DEVICES];
    int G = 0, primary = 0;
    int rc = mr_mgpu_candidates("mr_calc_toep_mgpu", cfg, devices, n_devices, cand,
                                &G, &primary);
    if (rc != MR_SUCCESS) return rc;

    mr_config pc = *cfg;
    pc.gpu_device_id = primary;
    const int mode = mr_toep_split_env();
    /* One device, `off`, or `replicate` (which for a single receiver means
     * the primary computes alone): the ordinary single-GPU path. */
    if (G == 1 || mode < 0 || mode == MR_TOEP_DIST_SELF) {
        MR_CUDA_CHECK(cudaSetDevice(primary));
        return mr_calc_toep_impl(&pc, d_phi2, d_traj, Ntrajs, Nkx, d_mask,
                                 d_phi2_toep);
    }
    mr_toep_dist_dev devs[MR_MGPU_MAX_DEVICES];
    for (int k = 0; k < G; ++k) {
        devs[k].dev = cand[k];
        devs[k].d_full = (k == 0) ? d_phi2_toep : NULL;
    }
    return mr_toep_distribute(&pc, devs, G, d_phi2, d_traj, Ntrajs, Nkx, d_mask,
                              mode, NULL);
}

extern "C" int mr_calc_toep_mgpu(const mr_config *cfg, const int *devices,
                                 int n_devices, const mr_complex *d_phi2,
                                 const float *d_traj, int64_t Ntrajs, int64_t Nkx,
                                 const int64_t *d_mask, mr_complex *d_phi2_toep)
{
    mr_clear_error();
    MR_REQUIRE(cfg != NULL, "mr_calc_toep_mgpu: cfg is NULL");
    MR_REQUIRE(d_phi2 != NULL, "mr_calc_toep_mgpu: d_phi2 is NULL");
    MR_REQUIRE(d_traj != NULL, "mr_calc_toep_mgpu: d_traj is NULL");
    MR_REQUIRE(d_phi2_toep != NULL, "mr_calc_toep_mgpu: d_phi2_toep is NULL");
    MR_REQUIRE(Ntrajs > 0, "mr_calc_toep_mgpu: Ntrajs = %lld must be > 0",
               (long long)Ntrajs);
    MR_REQUIRE(Nkx > 0, "mr_calc_toep_mgpu: Nkx = %lld must be > 0", (long long)Nkx);
    if (cfg->is_masked)
        MR_REQUIRE(d_mask != NULL,
                   "mr_calc_toep_mgpu: d_mask is required when is_masked is set");
    const int status = mr_config_validate(cfg);
    if (status != MR_SUCCESS) return status;

    int prev = 0;
    MR_CUDA_CHECK(cudaGetDevice(&prev));
    const int rc = mr_calc_toep_mgpu_impl(cfg, devices, n_devices, d_phi2, d_traj,
                                          Ntrajs, Nkx, d_mask, d_phi2_toep);
    cudaSetDevice(prev);
    return rc;
}
