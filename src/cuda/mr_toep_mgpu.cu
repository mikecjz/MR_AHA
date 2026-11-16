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
 * A *receiver* needs the whole kernel in its own memory: every device of a
 * multi-GPU Toeplitz operator (::mr_op_create) that applies the kernel to its
 * own coils. A *helper* only computes components for the receivers and keeps
 * nothing (a device the operator does not use for coils, e.g. when the coils
 * do not divide over all devices).
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
 *     component faster than it can compute one. ::mr_op_options::toep_dist
 *     (or `MR_AHA_TOEP_SPLIT`) forces a mode.
 *  5. *Exchange* (one thread per device): compute the rest of the own range;
 *     after each chunk, record an event and copy the chunk to every other
 *     receiver, one copy at a time in rotated order on the device's copy
 *     stream (see ::mr_tpush_chunk), so transfers overlap with the next
 *     chunks and every link carries about one transfer at a time. Every
 *     component is computed exactly once, so all receivers end with
 *     bit-identical kernels.
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

/**
 * @brief Per-device state of ::mr_toep_distribute.
 *
 * Slots live in one calloc'ed array, so every member starts zero / NULL and
 * ::mr_tcleanup can free whatever a (possibly failed) phase created. Each slot
 * is written only by its own worker thread during a phase and read by the
 * caller's thread between phases (after the join), so no locking is needed.
 * Ownership: @ref d_full belongs to the caller (the operator); @ref d_slice,
 * the streams, the event, the context and (when @ref owns_in) the input
 * copies belong to the slot.
 */
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

/**
 * @brief Everything the worker threads share (read-only during a phase).
 *
 * The only cross-slot access during a phase is the exchange callback reading
 * other slots' `dev` / `d_full`, which are set before any thread starts and
 * never change.
 */
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

/* Documented in mr_internal.cuh. Maps the public ::mr_toep_dist (possibly
 * overridden by MR_AHA_TOEP_SPLIT) to the internal ::mr_tdist_mode. */
int mr_toep_dist_resolve(int requested, int *mode_out)
{
    /* The environment variable overrides the option (benchmarks, A/B runs). */
    int m = requested;
    const char *e = std::getenv("MR_AHA_TOEP_SPLIT");
    if (e != NULL && e[0] != '\0') {
        if (!strcmp(e, "auto")) m = MR_TOEP_DIST_AUTO;
        else if (!strcmp(e, "split")) m = MR_TOEP_DIST_SPLIT;
        else if (!strcmp(e, "replicate")) m = MR_TOEP_DIST_REPLICATE;
        else if (!strcmp(e, "broadcast") || !strcmp(e, "off")) m = MR_TOEP_DIST_BROADCAST;
        else
            MR_FAIL(MR_ERR_INVALID_ARG,
                    "MR_AHA_TOEP_SPLIT='%s' is not one of auto, split, replicate, "
                    "broadcast (off)", e);
    }
    /* Public enum -> internal mode. An out-of-range option value (e.g. an
     * uninitialised field from a binding) is rejected rather than guessed. */
    switch (m) {
    case MR_TOEP_DIST_AUTO:      *mode_out = MR_TDIST_AUTO; break;
    case MR_TOEP_DIST_SPLIT:     *mode_out = MR_TDIST_EXCHANGE; break;
    case MR_TOEP_DIST_REPLICATE: *mode_out = MR_TDIST_SELF; break;
    case MR_TOEP_DIST_BROADCAST: *mode_out = MR_TDIST_ALONE; break;
    default:
        MR_FAIL(MR_ERR_INVALID_ARG, "toep_dist = %d is not a valid mr_toep_dist", m);
    }
    return MR_SUCCESS;
}

/* Documented in mr_internal.cuh. */
int mr_op_candidates(const char *who, const mr_config *cfg, const int *devices,
                     int n_devices, int *cand, int *G_out, int *primary_out)
{
    /* Argument checks: either the MR_OP_ALL_DEVICES sentinel (with devices
     * NULL) or an explicit list of at most MR_OP_MAX_DEVICES ordinals. */
    const int all = (devices == NULL && n_devices == MR_OP_ALL_DEVICES);
    MR_REQUIRE(all || (n_devices >= 0 && n_devices <= MR_OP_MAX_DEVICES),
               "%s: n_devices must be in [0, %d] or MR_OP_ALL_DEVICES (with "
               "devices NULL)", who, MR_OP_MAX_DEVICES);
    MR_REQUIRE(n_devices <= 0 || devices != NULL,
               "%s: devices is NULL but n_devices = %d", who, n_devices);
    /* Resolve the primary: the configured ordinal, or the caller's current
     * device for gpu_device_id = -1 (gpuArray / CuPy inputs live there). */
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
    const int nreq = all ? ndev_visible : n_devices;
    for (int k = 0; k < nreq; ++k) {
        const int dv = all ? k : devices[k];
        MR_REQUIRE(dv >= 0 && dv < ndev_visible,
                   "%s: device %d is not visible (%d visible)", who, dv,
                   ndev_visible);
        /* Duplicates (including the primary listed again) are dropped
         * silently, so `devices` may or may not repeat the primary. */
        int dup = 0;
        for (int j = 0; j < G; ++j) dup |= (cand[j] == dv);
        if (dup) continue;
        /* Capacity: with "all" the surplus devices are ignored; an explicit
         * list that would exceed the limit is an error. */
        if (all) {
            if (G == MR_OP_MAX_DEVICES) break; /* all visible: use the first 16 */
        } else {
            MR_REQUIRE(G < MR_OP_MAX_DEVICES, "%s: at most %d devices are supported",
                       who, MR_OP_MAX_DEVICES);
        }
        cand[G++] = dv;
    }
    *G_out = G;
    *primary_out = primary;
    return MR_SUCCESS;
}

/**
 * @brief Record a failure of slot @p d (status and this thread's message).
 *
 * ::mr_last_error is thread-local, so the worker must copy its diagnostic
 * into the slot before the thread ends; the caller's thread later re-raises
 * it with the device ordinal prefixed.
 */
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
    /* One thread per slot, because each slot's work is host-driven
     * (cufinufft / cuFFT calls, per-chunk callbacks) and a CUDA device is
     * selected per host thread; a single thread would serialise the issue of
     * all devices' work. std::thread throws std::system_error when it cannot
     * start a thread, hence the try/catch. */
    std::thread th[MR_OP_MAX_DEVICES];
    int started[MR_OP_MAX_DEVICES] = {0};
    for (int k = 0; k < job->G; ++k) {
        try {
            th[k] = std::thread(fn, job, k);
            started[k] = 1;
        } catch (...) {
            started[k] = 0;
        }
    }
    /* Join in slot order; a slot whose thread failed to start runs here,
     * after (or alongside) the started ones. The phase is over only when
     * every slot has returned. */
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
    /* Components [t0, t0+n) are one contiguous run of n slabs, both in the
     * sender's own buffer (offset relative to own_t0: 0 for a receiver's
     * whole kernel, r0 for a helper's slice) and in every receiver's whole
     * kernel (offset t0). */
    const size_t bytes = (size_t)n * (size_t)job->slab * sizeof(mr_complex);
    const mr_complex *src = d->d_own + (size_t)(t0 - d->own_t0) * job->slab;
    /* Order the copies after the chunk's computation without blocking the
     * host: the event marks "chunk issued so far" on the compute stream and
     * the copy stream waits for it on the GPU. The compute stream meanwhile
     * moves on to the next chunk. Re-recording the same event per chunk is
     * fine: cudaStreamWaitEvent captures the most recent record at call
     * time. */
    MR_CUDA_CHECK(cudaEventRecord(d->ev, d->s));
    MR_CUDA_CHECK(cudaStreamWaitEvent(d->cs, d->ev, 0));
    /* Rotated destinations k+1, k+2, ... (mod G); the sender itself (s = 0)
     * and helpers (no d_full) are skipped. The copies are serialised on cs,
     * which is what bounds each link to about one transfer at a time. */
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

    /* This thread's device and its private streams. Non-blocking streams do
     * not synchronise with the legacy default stream, so other work on the
     * device (e.g. the caller's) cannot stall the computation or the copies.
     * The event is timing-free: it only orders the copy stream after the
     * compute stream (see mr_tpush_chunk). */
    MR_CUDA_CHECK(cudaSetDevice(d->dev));
    MR_CUDA_CHECK(cudaStreamCreateWithFlags(&d->s, cudaStreamNonBlocking));
    MR_CUDA_CHECK(cudaEventCreateWithFlags(&d->ev, cudaEventDisableTiming));
    MR_CUDA_CHECK(cudaStreamCreateWithFlags(&d->cs, cudaStreamNonBlocking));

    /* Inputs: the caller's on the primary, copies elsewhere. The context
     * needs them on its own device (cufinufft keeps the trajectory pointer
     * and the packing kernel reads phi2 locally). They are small next to the
     * kernel, so each device takes a full copy rather than a share. The
     * primary's stream was synchronised by mr_toep_distribute, so the sources
     * are complete; the copies are queued on d->s, so the context created
     * below reads them only after they have landed. */
    if (k == 0) {
        d->d_phi2 = (mr_complex *)job->phi2;
        d->d_traj = (float *)job->traj;
        d->d_mask = (int64_t *)job->mask;
    } else {
        /* Sizes of the inputs as documented for mr_calc_toep: phi2
         * `[Ntrajs, Nkz, L, L]`, traj `[Ntrajs, Nkx, ndim]`, mask
         * `[num_masked]`. owns_in is set before allocating so that
         * mr_tcleanup frees whichever allocations succeeded. */
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

    /* Per-device copy of the configuration: this device, this thread's
     * compute stream, and no per-chunk printing from the worker threads (the
     * caller prints one summary). The chunk cap job->max_chunk keeps several
     * chunks per range so the exchange can overlap with the computation. */
    mr_config dc = *cfg;
    dc.gpu_device_id = d->dev;
    dc.stream = (void *)d->s;
    dc.verbose = 0;
    int rc = mr_toep_ctx_create(&dc, d->d_phi2, d->d_traj, job->Ntrajs, job->Nkx,
                                job->max_chunk, &d->ctx);
    if (rc != MR_SUCCESS) return rc;

    /* Pilot: the first chunk of the own range, timed. Its GPU time per
     * component feeds the cost model; the components it produced are kept
     * (d->done) and are not recomputed in phase 2 by this device. ms_out
     * makes the run synchronous, so the pilot is complete on return. */
    const int64_t n = d->r1 - d->r0;
    const int64_t p = (mr_toep_ctx_chunk(d->ctx) < n) ? mr_toep_ctx_chunk(d->ctx) : n;
    rc = mr_toep_ctx_run(d->ctx, d->d_mask, d->r0, d->r0 + p, d->d_own, d->own_t0,
                         NULL, NULL, &d->pilot_ms);
    if (rc != MR_SUCCESS) return rc;
    d->done = p;
    return MR_SUCCESS;
}

/** @brief Phase 1 (setup and pilot) of slot @p k. Thread entry point: the
 *         body's MR_FAIL / MR_CUDA_CHECK early returns all end up in the
 *         slot's status and message here. */
static void mr_tphase1(mr_tjob *job, int k)
{
    mr_clear_error();
    mr_tfail(&job->d[k], mr_tphase1_body(job, k));
}

/** @brief Body of phase 2 for slot @p k; returns a status. */
static int mr_tphase2_body(mr_tjob *job, int k)
{
    mr_tdev *d = &job->d[k];
    /* A new thread: select this slot's device again. */
    MR_CUDA_CHECK(cudaSetDevice(d->dev));

    if (job->mode == MR_TDIST_EXCHANGE) {
        /* Exchange: ship the pilot (already complete), then compute the rest
         * of the own range [r0 + done, r1) with mr_tpush_chunk as the chunk
         * callback, so each finished chunk is queued for every receiver
         * while the next one is computed. Slots with an empty range only
         * receive. pu lives on this frame, which outlives the run because
         * the callback is invoked synchronously from mr_toep_ctx_run. */
        if (d->r1 > d->r0) {
            mr_tpush pu = {job, k};
            int rc = mr_tpush_chunk(&pu, d->r0, d->done); /* the pilot */
            if (rc != MR_SUCCESS) return rc;
            rc = mr_toep_ctx_run(d->ctx, d->d_mask, d->r0 + d->done, d->r1, d->d_own,
                                 d->own_t0, mr_tpush_chunk, &pu, NULL);
            if (rc != MR_SUCCESS) return rc;
        }
    } else if (d->d_full != NULL) {
        /* Self: compute every component this receiver does not hold yet.
         * A receiver with an empty range (more devices than components) has
         * no context from phase 1, so it plans one here, uncapped. Helpers
         * fall through both branches: their pilot work is simply dropped. */
        if (d->ctx == NULL) {
            mr_config dc = job->cfg;
            dc.gpu_device_id = d->dev;
            dc.stream = (void *)d->s;
            dc.verbose = 0;
            const int rc = mr_toep_ctx_create(&dc, d->d_phi2, d->d_traj, job->Ntrajs,
                                              job->Nkx, 0, &d->ctx);
            if (rc != MR_SUCCESS) return rc;
        }
        /* The receiver already holds [r0, r0 + done) from its pilot, so it
         * computes the two pieces around it: [0, a) and [b, ntri). With an
         * empty range a = b = 0, i.e. everything. Both runs write straight
         * into the whole kernel (t_out0 = 0). */
        const int64_t a = (d->r1 > d->r0) ? d->r0 : 0;
        const int64_t b = (d->r1 > d->r0) ? d->r0 + d->done : 0;
        int rc = mr_toep_ctx_run(d->ctx, d->d_mask, 0, a, d->d_full, 0, NULL, NULL,
                                 NULL);
        if (rc == MR_SUCCESS)
            rc = mr_toep_ctx_run(d->ctx, d->d_mask, b, job->ntri, d->d_full, 0, NULL,
                                 NULL, NULL);
        if (rc != MR_SUCCESS) return rc;
    }

    /* Wait for the computation and every outgoing copy. Incoming copies are
     * issued on the senders' copy streams and waited for by their threads;
     * after mr_tparallel has joined all threads, every receiver is
     * complete. */
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

/**
 * @brief Release everything slot @p d holds except the receiver's kernel.
 *
 * Safe on a slot in any state (partially set up, failed, or complete): every
 * resource is checked before release. Streams are synchronised before they
 * and the buffers they use are destroyed, so a failed run cannot leave
 * in-flight work reading freed memory. Errors are swallowed (cleanup must
 * not mask the first diagnostic) and the last (non-sticky) error is cleared.
 */
static void mr_tcleanup(mr_tdev *d)
{
    if (cudaSetDevice(d->dev) != cudaSuccess) {
        cudaGetLastError();
        return;
    }
    if (d->ctx) mr_toep_ctx_destroy(d->ctx); /* synchronises its stream */
    d->ctx = NULL;
    /* The copy stream reads d_own (d_slice for a helper): drain it before
     * d_slice is freed below. */
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
    /* On the primary the inputs are the caller's and must not be freed. */
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
    void *src[MR_OP_MAX_DEVICES] = {0};
    void *dst[MR_OP_MAX_DEVICES] = {0};
    cudaStream_t st[MR_OP_MAX_DEVICES] = {0};

    /* Setup: per slot a B-byte source, a G*B-byte landing area on receivers
     * (slot i writes at offset i*B, so concurrent copies never overlap) and
     * one non-blocking stream, mirroring the one copy stream per sender of
     * the exchange. ce carries the first failure; later steps are skipped
     * but the release loop below always runs. */
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
            /* Issue every slot's copies before waiting on any, so that all
             * links are loaded at once (the point of the probe); same
             * rotated destination order as mr_tpush_chunk. */
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
            /* End of round: every stream drained (host wall clock, since the
             * copies span several devices and no single event covers them). */
            for (int i = 0; i < G && ce == cudaSuccess; ++i) {
                ce = cudaSetDevice(job->d[i].dev);
                if (ce == cudaSuccess) ce = cudaStreamSynchronize(st[i]);
            }
        }
        /* Elapsed time of rounds 1-2 (only meaningful when ce is success). */
        secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
                   .count();
    }
    /* Release on every path. After a failed round some copies may still be
     * pending; this relies on cudaStreamDestroy deferring the release of the
     * stream until its work is done and on cudaFree's implicit
     * synchronisation. */
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
    /* Every receiver receives from the G-1 other slots, twice. All receivers
     * receive the same amount, so this is also the per-receiver rate the
     * cost model divides its receive volume by. */
    *gbs = (secs > 0.0) ? 2.0 * (G - 1) * (double)B / secs / 1e9 : 0.0;
    return MR_SUCCESS;
}

/* ------------------------------------------------------------------------ */
/* Engine                                                                   */
/* ------------------------------------------------------------------------ */

/* Documented in mr_internal.cuh; the flow is in the file comment above. */
int mr_toep_distribute(const mr_config *cfg, const mr_toep_dist_dev *devs, int G,
                       const mr_complex *d_phi2, const float *d_traj,
                       int64_t Ntrajs, int64_t Nkx, const int64_t *d_mask,
                       int mode, mr_toep_dist_report *report)
{
    /* slab: elements per component in every buffer (the unit of all
     * offsets and transfers below). */
    const auto t_start = std::chrono::steady_clock::now();
    MR_REQUIRE(G >= 1 && G <= MR_OP_MAX_DEVICES && devs[0].d_full != NULL,
               "mr_toep_distribute: bad device list");
    const int64_t ntri = mr_num_tri(cfg->L);
    const int64_t slab = cfg->is_masked ? (int64_t)cfg->num_masked : mr_pad_size(cfg);

    /* Helpers beyond the number of components would have nothing to do.
     * Slots are zero-initialised (calloc), which mr_tcleanup relies on; n
     * counts the slots kept, and receivers are always kept because they need
     * the kernel even without a range. */
    mr_tdev *d = (mr_tdev *)calloc(MR_OP_MAX_DEVICES, sizeof(mr_tdev));
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
    /* nc: devices with a non-empty range. Slot k < nc gets
     * [floor(T*k/nc), floor(T*(k+1)/nc)), sizes differing by at most one;
     * the rest get the empty range [T, T). */
    const int nc = (n < ntri) ? n : (int)ntri;
    for (int k = 0; k < n; ++k) {
        d[k].r0 = (k < nc) ? ntri * k / nc : ntri;
        d[k].r1 = (k < nc) ? ntri * (k + 1) / nc : ntri;
    }

    /* Shared job description; job.mode is filled in after the decision. */
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
     * pure self-computation needs no cap. per is the largest range; the cap
     * ceil(per / nchunks) also sizes the pilot (one chunk). In `auto` mode the
     * cap applies even if self is chosen later, since the contexts are
     * planned before the decision. */
    const int64_t per = (ntri + nc - 1) / nc;
    int64_t nchunks = MR_TOEP_MIN_CHUNKS;
    {
        const char *e = std::getenv("MR_AHA_TOEP_CHUNKS");
        if (e && e[0] && atoll(e) >= 1) nchunks = atoll(e);
    }
    job.max_chunk = (mode == MR_TDIST_SELF) ? 0 : (per + nchunks - 1) / nchunks;

    int rc = MR_SUCCESS;
    mr_toep_dist_report rep;
    memset(&rep, 0, sizeof(rep));
    rep.n_devices = nc;

    /* Inputs produced on the caller's stream must be complete: the workers
     * read them from their own non-blocking streams, which do not order
     * against the caller's stream. */
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
     * the host. Helper-helper pairs never exchange data and are skipped.
     * Done here, on one thread, before any worker runs. */
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            if (d[i].d_full != NULL || d[j].d_full != NULL) mr_enable_peer(d[i].dev, d[j].dev);

    /* Wall-clock milliseconds since t. */
    auto ms_since = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t)
            .count();
    };

    /* Bandwidth for the cost model (step 2 of the file comment). Only `auto`
     * needs it, and only with more than one device. A failure leaves
     * gather_gbs = 0, which makes the decision below pick the exchange. */
    auto t_ph = std::chrono::steady_clock::now();
    if (mode == MR_TDIST_AUTO && n > 1) {
        if (mr_tprobe(&job, &rep.gather_gbs) != MR_SUCCESS) rep.gather_gbs = 0.0;
        mr_clear_error(); /* a failed probe is not an error of the call */
    }
    rep.probe_ms = ms_since(t_ph);

    /* Phase 1: setup and pilot. Afterwards, collect the first failing slot
     * (in slot order) and re-raise its message on this thread, prefixed with
     * its ordinal. */
    t_ph = std::chrono::steady_clock::now();
    mr_tparallel(&job, mr_tphase1);
    rep.phase1_ms = ms_since(t_ph);
    for (int k = 0; k < n && rc == MR_SUCCESS; ++k)
        if (d[k].rc != MR_SUCCESS) {
            rc = d[k].rc;
            mr_set_error("device %d: %s", d[k].dev, d[k].msg);
        }

    /* Decision (step 4 of the file comment). Inputs of the model, from the
     * slots that ran a pilot:
     *  - te:       slowest per-component time (ms), so the slowest device
     *              sets the pace;
     *  - c_rem:    most components any device still has to compute in its
     *              own range (c in the formula);
     *  - k_chunks: most chunks in any range (k), estimated as
     *              ceil(range / pilot size) since the pilot is one chunk. */
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
        /* Per receiver: t_self = (T - p) * te, the components it lacks after
         * its pilot; recv = R, the bytes of all components outside its own
         * range over the probed rate (GB/s * 1e6 = bytes/ms). The slowest
         * receiver of each kind decides. */
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
        /* T_exchange = max(c*te, R) + R/k: computation and transfer overlap
         * chunk by chunk, so the longer of the two dominates, plus the tail
         * transfer of the last chunk (about R/k) that cannot overlap. */
        const double comp = (double)c_rem * te;
        const double t_x = ((comp > recv) ? comp : recv) + recv / (double)k_chunks;
        rep.entry_ms = te;
        rep.t_exchange_ms = t_x;
        rep.t_self_ms = t_self;
        /* `auto`: exchange when the model says it is faster, or when no
         * bandwidth was measured (one device, or a failed probe), where
         * exchange is the safe choice (with one device it is just "compute
         * the rest"). A forced mode is taken as is. */
        if (mode == MR_TDIST_AUTO)
            job.mode = (rep.gather_gbs <= 0.0 || t_x < t_self) ? MR_TDIST_EXCHANGE
                                                               : MR_TDIST_SELF;
        else
            job.mode = mode;
        rep.mode = job.mode;

        /* Phase 2: exchange or self, then the same first-failure collection
         * as after phase 1. */
        t_ph = std::chrono::steady_clock::now();
        mr_tparallel(&job, mr_tphase2);
        rep.phase2_ms = ms_since(t_ph);
        for (int k = 0; k < n && rc == MR_SUCCESS; ++k)
            if (d[k].rc != MR_SUCCESS) {
                rc = d[k].rc;
                mr_set_error("device %d: %s", d[k].dev, d[k].msg);
            }
    }

    /* Keep the first diagnostic across the cleanup. Cleanup runs on every
     * path (also after a phase-1 failure, where phase 2 was skipped) and
     * leaves only the receivers' d_full, which the caller owns. */
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
    /* One-line summary (verbose or MR_AHA_TOEP_REPORT != 0, successful runs
     * only). Helpers are marked with an "h" after their ordinal; "chunk" is
     * the primary's pilot size. */
    const char *env_rep = std::getenv("MR_AHA_TOEP_REPORT");
    if ((cfg->verbose || (env_rep && atoi(env_rep) != 0)) && rc == MR_SUCCESS) {
        printf("[mr_aha] calc_toep on %d device(s) (", nc);
        for (int k = 0; k < n; ++k)
            printf("%s%d%s", k ? " " : "", d[k].dev, d[k].d_full ? "" : "h");
        printf("): %s, chunk %lld, %.3f ms/component, receive %.1f GB/s, model "
               "exchange %.0f ms / self %.0f ms; probe %.0f, setup+pilot %.0f, "
               "%s %.0f, cleanup %.0f, total %.0f ms\n",
               rep.mode == MR_TDIST_EXCHANGE ? "split + exchange" : "replicate",
               (long long)(d[0].done), rep.entry_ms, rep.gather_gbs,
               rep.t_exchange_ms, rep.t_self_ms, rep.probe_ms, rep.phase1_ms,
               rep.mode == MR_TDIST_EXCHANGE ? "exchange" : "self",
               rep.phase2_ms, rep.cleanup_ms, rep.total_ms);
        fflush(stdout);
    }
    if (report) *report = rep;
    free(d);
    return rc;
}
