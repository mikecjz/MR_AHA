/**
 * @file mr_calc_toep.cu
 * @brief Pre-computation of the block-diagonal Toeplitz operator `phi2_toep`.
 *
 * Given the Gram blocks @f$\Phi\Phi^H@f$ (rank x time @f$\Phi@f$,
 * see @ref mr_gram) sampled along the non-Cartesian
 * trajectory, this file evaluates the Toeplitz multiplier that
 * ::mr_aha_plan_execute applies between its two cuFFTs.
 *
 * @section mr_toep_sign Why a type-1 with sign -1 followed by an inverse FFT
 *
 * Write @f$p^{\pm}[d] = \sum_m w_m e^{\pm i k_m d}@f$ for the point-spread
 * function of the sampling pattern on the upsampled grid; a type-1 NUFFT with
 * `iflag = ±1` evaluates exactly that. Since @f$p^{-}[d] = p^{+}[-d]@f$, the
 * *normalized inverse* DFT of @f$p^{-}@f$ equals the forward DFT of
 * @f$p^{+}@f$ divided by @f$\prod M@f$:
 * @f[
 *   \mathrm{IDFT}(p^{-})[k] \;=\; \frac{1}{\prod M}\,\mathrm{DFT}(p^{+})[k].
 * @f]
 * The right-hand side is precisely the multiplier required between the
 * unnormalized forward and inverse cuFFTs of the @f$A^H A@f$ pipeline, so the
 * pseudo-code's "run ifft" both matches the literature convention and removes
 * every remaining scaling factor from the hot loop.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_kernels.cuh"

#include <cufinufft.h>

#include <cstdio>
#include <cstdlib>
#include <new>

/** @brief Default thread-block size for the elementwise kernels here. */
#define MR_BLOCK 256

/**
 * @brief Everything one ::mr_toep_ctx allocates, so that one cleanup path
 *        frees it.
 */
struct mr_toep_workspace {
    mr_complex     *d_tri;   /**< Packed Gram blocks `[nbatch, ntri]`.        */
    mr_complex     *d_c;     /**< cufinufft strengths for one chunk.          */
    mr_complex     *d_fk;    /**< cufinufft output slab, centered order.       */
    mr_complex     *d_work;  /**< Natural-order scratch for the inverse FFT.  */
    cufinufftf_plan nufft;   /**< Type-1 plan, reused across chunks.          */
    cufftHandle     fft;     /**< Batched inverse cuFFT over one chunk.       */
};

/**
 * @brief Release every buffer held by a workspace. Safe on partial failure.
 *
 * Every member starts zeroed (the owning ::mr_toep_ctx is value-initialized),
 * so a NULL / 0 member simply means "never created" and is skipped. Members
 * are reset after release, so a second call is a no-op. Does not synchronize:
 * the caller must make sure no work still reads these buffers.
 */
static void mr_toep_workspace_free(mr_toep_workspace *w)
{
    /* Plans first: the cufinufft plan owns internal device buffers (fine
     * grids, sort arrays) that are independent of ours. */
    if (w->nufft) { cufinufftf_destroy(w->nufft); w->nufft = NULL; }
    if (w->fft)   { cufftDestroy(w->fft);         w->fft = 0; }
    if (w->d_tri)  { cudaFree(w->d_tri);  w->d_tri = NULL; }
    if (w->d_c)    { cudaFree(w->d_c);    w->d_c = NULL; }
    if (w->d_fk)   { cudaFree(w->d_fk);   w->d_fk = NULL; }
    if (w->d_work) { cudaFree(w->d_work); w->d_work = NULL; }
}

/* Documented in mr_internal.cuh. The caller restores the previous device with
 * `if (prev >= 0) cudaSetDevice(prev);` once its work has been issued. */
int mr_toep_push_device(const mr_config *cfg, int *prev)
{
    /* -1: "nothing to restore". gpu_device_id < 0 keeps the caller's current
     * device, which is the right one for MATLAB gpuArray / CuPy inputs. */
    *prev = -1;
    if (cfg->gpu_device_id < 0) return MR_SUCCESS;
    int cur = 0;
    MR_CUDA_CHECK(cudaGetDevice(&cur));
    /* Only switch (and only report a device to restore) when it differs, so
     * the common single-GPU case costs one cudaGetDevice and nothing else. */
    if (cur != cfg->gpu_device_id) {
        MR_CUDA_CHECK(cudaSetDevice(cfg->gpu_device_id));
        *prev = cur;
    }
    return MR_SUCCESS;
}

/**
 * @brief Choose how many triangular components to process per chunk.
 *
 * The transient cost of one component is two upsampled slabs (the cufinufft
 * output and the natural-order scratch) plus its non-uniform strengths.
 * cufinufft additionally holds an internal fine grid of
 * @f$\sigma^{D}\prod M@f$ complex numbers per transform in flight, which is
 * bounded separately through `gpu_maxbatchsize`.
 *
 * @param[in]  cfg         Configuration (supplies ::mr_config::mem_fraction).
 * @param[in]  pad_size    `Mx*My*Mz`.
 * @param[in]  strengths   Non-uniform strengths per component,
 *                         `Ntrajs*Nkx*Nkz_eff`.
 * @param[in]  ntri        Total number of triangular components.
 * @param[in]  persistent  Bytes already committed to persistent buffers.
 * @param[out] chunk_out   Chosen number of components per chunk, >= 1.
 * @param[out] budget_out  Bytes left over for cufinufft's internal grids.
 * @return ::MR_SUCCESS or ::MR_ERR_CUDA.
 */
static int mr_toep_pick_chunk(const mr_config *cfg, int64_t pad_size,
                              int64_t strengths, int64_t ntri,
                              size_t persistent, int *chunk_out,
                              size_t *budget_out)
{
    /* Memory free on the current device right now; anything the caller
     * allocated before this call (e.g. the whole phi2_toep output) is already
     * excluded from it. */
    size_t free_bytes = 0, total_bytes = 0;
    MR_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    /* Clamp the user fraction: a non-positive (or NaN, hence the negated
     * comparison) value falls back to the documented default 0.5, and at
     * most 95 % is used so the CUDA runtime / cuFFT keep some headroom. */
    double frac = cfg->mem_fraction;
    if (!(frac > 0.0)) frac = 0.5;
    if (frac > 0.95) frac = 0.95;

    /* The packed Gram blocks are already allocated when this runs (the
     * caller passes their size as `persistent`), so they count against the
     * budget as well. */
    size_t budget = (size_t)(frac * (double)free_bytes);
    budget = (budget > persistent) ? (budget - persistent) : 0;

    /* Split the budget evenly between our slabs and cufinufft's fine grids. */
    const size_t ours = budget / 2;
    const size_t per_component =
        (size_t)(2 * pad_size + strengths) * sizeof(mr_complex);

    /* At least one component per chunk even when the budget is exhausted (the
     * subsequent cudaMalloc then reports the shortage), and never more than
     * there are components, so a small problem runs as a single chunk. */
    int64_t chunk = (per_component > 0) ? (int64_t)(ours / per_component) : ntri;
    if (chunk < 1) chunk = 1;
    if (chunk > ntri) chunk = ntri;

    *chunk_out  = (int)chunk;
    *budget_out = budget - ours;
    return MR_SUCCESS;
}

/**
 * @brief Geometry of the centered inverse FFT that turns point-spread functions
 *        into `phi2_toep` (steps 3-4 of ::mr_calc_toep).
 */
struct mr_toep_fft_geom {
    int   rank;   /**< 3 for a 3D trajectory, else 2.                         */
    int   n[3];   /**< cuFFT extents, slowest first.                          */
    int   dist;   /**< Elements between consecutive transforms.               */
    int   per_comp; /**< Transforms per triangular component (Nkz for a stack). */
    float scale;  /**< 1/prod(M) over the transformed axes.                   */
    int   s[3];   /**< Centering shifts, 0 on an axis that is not transformed. */
};

/**
 * @brief Fill an ::mr_toep_fft_geom from a validated configuration.
 *
 * A 3D trajectory transforms every axis of the upsampled grid. For 2D and
 * stack-of-stars, kz (if any) is already Cartesian, so it is merely another
 * batch axis and is neither transformed nor shifted.
 */
static void mr_toep_geom(const mr_config *cfg, mr_toep_fft_geom *g)
{
    /* Volumes are stored x-fastest (column-major `[Mx, My, Mz]`), while cuFFT
     * takes its extents slowest-first, hence the reversed order of n[]. The
     * shifts s[] are indexed by storage axis (x, y, z) because the remap
     * kernels (mr_k_to_natural & co.) are. */
    const int Mx = cfg->phi2_dim[0], My = cfg->phi2_dim[1], Mz = cfg->phi2_dim[2];
    if (cfg->traj_type == MR_TRAJ_3D) {
        /* One 3D transform per component over the whole upsampled volume;
         * every axis is centered and contributes to the 1/prod(M) scaling. */
        g->rank = 3;
        g->n[0] = Mz; g->n[1] = My; g->n[2] = Mx;
        g->dist = Mx * My * Mz;
        g->per_comp = 1;
        g->scale = 1.0f / (float)((double)Mx * My * Mz);
        g->s[0] = Mx / 2; g->s[1] = My / 2; g->s[2] = Mz / 2;
    } else {
        /* 2D / stack: one 2D transform per kz plane. The Mz planes of a
         * component are contiguous `[Mx, My]` slabs `Mx*My` apart, so they
         * form Mz consecutive batch entries of cufftPlanMany. Mz == 1 in 2D.
         * kz stays in centered order (s[2] = 0) and is not part of the
         * scaling. */
        g->rank = 2;
        g->n[0] = My; g->n[1] = Mx; g->n[2] = 1;
        g->dist = Mx * My;
        g->per_comp = Mz; /* Nkz for a stack, 1 in 2D */
        g->scale = 1.0f / (float)((double)Mx * My);
        g->s[0] = Mx / 2; g->s[1] = My / 2; g->s[2] = 0;
    }
}

/* ------------------------------------------------------------------------ */
/* Reusable per-device context: plan once, compute any range of components  */
/* ------------------------------------------------------------------------ */

/**
 * @brief State of one device's share of ::mr_calc_toep (see
 *        ::mr_toep_ctx_create in mr_internal.cuh).
 *
 * Holds everything that does not depend on which components are computed:
 * the packed Gram blocks of *all* components, the chunk buffers, the type-1
 * cufinufft plan (with the trajectory set) and the batched inverse cuFFT.
 * ::mr_toep_ctx_run then evaluates any range of components with them, as
 * often as needed, which is what the multi-GPU pre-computation
 * (`mr_toep_mgpu.cu`) relies on: a device first computes a short pilot range,
 * then the rest, without planning twice.
 */
struct mr_toep_ctx {
    mr_config          cfg;      /**< Copy of the configuration.               */
    mr_toep_workspace  w;        /**< Buffers and plans.                       */
    mr_toep_fft_geom   g;        /**< Inverse-FFT geometry and shifts.          */
    int64_t            Ntrajs;   /**< Number of trajectories.                  */
    int64_t            Nkx;      /**< Samples per trajectory.                  */
    int                Nkz_eff;  /**< kz encodes (stack) or 1.                 */
    int64_t            ntri;     /**< `(L^2+L)/2`.                             */
    int64_t            pad_size; /**< `Mx*My*Mz`.                              */
    int                chunk;    /**< Components per chunk.                    */
    cudaStream_t       s;        /**< Stream all work is issued on.            */
    cudaEvent_t        ev0;      /**< Timing events of ::mr_toep_ctx_run,      */
    cudaEvent_t        ev1;      /**< created on first use.                    */
};

/**
 * @brief Free a context's resources without synchronizing (error paths).
 *
 * Used directly by the failure paths of ::mr_toep_ctx_create, where at most
 * the Gram-packing kernel can be in flight, and by ::mr_toep_ctx_destroy after
 * it has synchronized. Events are created lazily by ::mr_toep_ctx_run, so
 * they may legitimately be NULL here.
 */
static void mr_toep_ctx_release(mr_toep_ctx *c)
{
    mr_toep_workspace_free(&c->w);
    if (c->ev0) cudaEventDestroy(c->ev0);
    if (c->ev1) cudaEventDestroy(c->ev1);
    delete c;
}

int mr_toep_ctx_create(const mr_config *cfg, const mr_complex *d_phi2,
                       const float *d_traj, int64_t Ntrajs, int64_t Nkx,
                       int64_t max_chunk, mr_toep_ctx **ctx_out)
{
    /* -- 0. Host-side state -------------------------------------------------- */
    /* `new T()` value-initializes the aggregate, so every pointer / handle in
     * the workspace and both events start NULL / 0; mr_toep_ctx_release
     * relies on that to free only what was created. The configuration is
     * copied so the context does not depend on the caller's struct. */
    *ctx_out = NULL;
    mr_toep_ctx *c = new (std::nothrow) mr_toep_ctx();
    if (c == NULL) MR_FAIL(MR_ERR_ALLOC, "failed to allocate the calc_toep context");
    c->cfg      = *cfg;
    c->Ntrajs   = Ntrajs;
    c->Nkx      = Nkx;
    c->Nkz_eff  = (cfg->traj_type == MR_TRAJ_STACK) ? cfg->phi2_dim[2] : 1;
    c->ntri     = mr_num_tri(cfg->L);
    c->pad_size = mr_pad_size(cfg);
    c->s        = (cudaStream_t)cfg->stream;

    /* Derived sizes:
     *  - nufft_dim: dimension of the type-1 NUFFT (a stack is a 2D NUFFT per
     *    kz plane, kz being Cartesian already);
     *  - M_nu:      non-uniform points of one transform (all readouts);
     *  - nbatch:    Gram blocks, one per readout (and per kz for a stack);
     *               also the leading extent of the packed d_tri. */
    const int      nufft_dim = (cfg->traj_type == MR_TRAJ_3D) ? 3 : 2;
    const int      Nkz_eff   = c->Nkz_eff;
    const int64_t  ntri      = c->ntri;
    const int64_t  pad_size  = c->pad_size;
    const int64_t  M_nu      = Ntrajs * Nkx;
    const int64_t  nbatch    = Ntrajs * (int64_t)Nkz_eff;
    cudaStream_t   s         = c->s;
    mr_toep_workspace &w     = c->w;

    const int Mx = cfg->phi2_dim[0], My = cfg->phi2_dim[1], Mz = cfg->phi2_dim[2];

    /* cufinufft runs on opts.gpu_device_id (default 0), not on the current
     * device, so it must always be told explicitly. */
    int dev = cfg->gpu_device_id;
    if (dev < 0 && cudaGetDevice(&dev) != cudaSuccess) {
        mr_toep_ctx_release(c);
        MR_FAIL(MR_ERR_CUDA, "cudaGetDevice failed: %s",
                cudaGetErrorString(cudaGetLastError()));
    }

    /* -- 1. Pack the dense Gram blocks into the requested triangle ---------- */
    /* All components are packed (nbatch * ntri elements, tens of MB at most)
     * so that any range can be computed afterwards. Packing halves the work:
     * a Gram block is Hermitian, so the type-1 NUFFT of the unstored half
     * would be the conjugate-mirrored PSF of the stored one and is never
     * computed. Layout `[nbatch, ntri]`: component t is a contiguous column
     * of nbatch strengths, which is what mr_k_fill_strengths broadcasts. */
    if (cudaMalloc((void **)&w.d_tri,
                   (size_t)nbatch * ntri * sizeof(mr_complex)) != cudaSuccess) {
        /* Clear the (non-sticky) allocation error so that a later
         * cudaGetLastError / MR_KERNEL_CHECK does not report it again. */
        cudaGetLastError();
        mr_toep_ctx_release(c);
        MR_FAIL(MR_ERR_ALLOC, "failed to allocate %lld packed Gram elements",
                (long long)(nbatch * ntri));
    }
    {
        /* One thread per element of the dense `[nbatch, L, L]` input; threads
         * on the unstored half return immediately. Asynchronous on s, so
         * d_phi2 only has to stay valid until the stream passes this kernel. */
        const long long total = nbatch * (long long)cfg->L * cfg->L;
        const int grid = (int)mr_ceil_div(total, MR_BLOCK);
        mr_k_extract_tri<<<grid, MR_BLOCK, 0, s>>>(w.d_tri, d_phi2, nbatch,
                                                   cfg->L, cfg->triangulation ==
                                                               MR_TRI_UPPER);
        if (cudaGetLastError() != cudaSuccess) {
            mr_toep_ctx_release(c);
            MR_FAIL(MR_ERR_CUDA, "mr_k_extract_tri failed to launch: %s",
                    cudaGetErrorString(cudaGetLastError()));
        }
    }

    /* -- 2. Size the chunking against the free device memory ---------------- */
    /* A chunk is the number of triangular components pushed through one
     * cufinufft execute + one batched cuFFT. Larger chunks amortize launch
     * and spreading overheads; the budget keeps the transient slabs inside
     * mem_fraction of what is free now. nufft_budget is the half of the
     * budget left for cufinufft's internal fine grids (used in step 3). */
    int    chunk  = 1;
    size_t nufft_budget = 0;
    {
        const int rc = mr_toep_pick_chunk(cfg, pad_size, M_nu * Nkz_eff, ntri,
                                          (size_t)nbatch * ntri * sizeof(mr_complex),
                                          &chunk, &nufft_budget);
        if (rc != MR_SUCCESS) { mr_toep_ctx_release(c); return rc; }
    }
    /* Caller cap (the multi-GPU path uses smaller chunks so that finished
     * components can be shipped to other devices earlier). */
    if (max_chunk > 0 && chunk > max_chunk) chunk = (int)max_chunk;
    {
        /* Tuning / benchmarking: an upper bound on the chunk size (smaller
         * chunks mean smaller buffers and cuFFT/cufinufft plans). */
        const char *e = std::getenv("MR_AHA_TOEP_MAX_CHUNK");
        if (e && e[0] && atoll(e) >= 1 && chunk > atoll(e)) chunk = (int)atoll(e);
    }
    c->chunk = chunk;

    /* Per-chunk buffer sizes. Every chunk has exactly `chunk` components (the
     * last one is zero-padded by mr_k_fill_strengths), so these buffers and
     * both plans are sized once and reused:
     *  - d_c:    `[M_nu, Nkz_eff, chunk]` strengths = ntr transforms of M_nu;
     *  - d_fk:   `[Mx, My, Mz, chunk]` cufinufft output, centered order;
     *  - d_work: same shape, natural order, transformed in place by cuFFT.
     * For a stack, the ntr = Nkz_eff*chunk 2D transforms of `[Mx, My]` modes
     * are laid out kz-major within a component, i.e. exactly `[Mx, My, Mz]`
     * per component, so d_fk needs no reshuffling. */
    const int64_t ntr        = (int64_t)Nkz_eff * chunk; /* transforms/chunk  */
    const int64_t slab_elems = pad_size * chunk;
    const int64_t c_elems    = M_nu * ntr;

    if (cudaMalloc((void **)&w.d_c, (size_t)c_elems * sizeof(mr_complex)) !=
            cudaSuccess ||
        cudaMalloc((void **)&w.d_fk, (size_t)slab_elems * sizeof(mr_complex)) !=
            cudaSuccess ||
        cudaMalloc((void **)&w.d_work, (size_t)slab_elems * sizeof(mr_complex)) !=
            cudaSuccess) {
        cudaGetLastError();
        mr_toep_ctx_release(c);
        MR_FAIL(MR_ERR_ALLOC,
                "failed to allocate the calc_toep chunk buffers (chunk = %d); "
                "lower mr_config.mem_fraction or the upsampled grid size",
                chunk);
    }

    /* -- 3. Plan the type-1 NUFFT once and reuse it for every chunk --------- */
    /* The plan evaluates, for each of the ntr transforms of a chunk,
     *   fk[k] = sum_m c[m] exp(-i k.x_m),   k in the centered phi2_dim grid,
     * i.e. the point-spread function of the sampling weighted by one Gram
     * component (sign -1, see the file comment for why). upsampfac 2.0 is
     * cufinufft's standard accuracy/size trade-off; the plan runs on our
     * stream so all steps of a chunk serialize without host syncs. */
    cufinufft_opts opts;
    cufinufft_default_opts(&opts);
    opts.upsampfac = 2.0;
    opts.modeord   = 0; /* CMCL / centered mode order, matching our convention */
    opts.gpu_stream = cfg->stream;
    opts.gpu_device_id = dev;
    opts.debug = cfg->verbose ? 1 : 0;
    {
        /* Tuning / benchmarking (experimental): cufinufft's internal
         * upsampling factor sigma (MR_AHA_TOEP_UPSAMPFAC, 2.0 or 1.25 for the
         * Horner kernel) and its type-1 spreading method
         * (MR_AHA_TOEP_GPU_METHOD, 0 = cufinufft's choice, 1 = GM-sort,
         * 2 = shared memory). sigma only sizes cufinufft's fine grid; the
         * Toeplitz grid phi2_dim is unaffected. */
        const char *e = std::getenv("MR_AHA_TOEP_UPSAMPFAC");
        if (e && e[0] && atof(e) > 1.0) opts.upsampfac = atof(e);
        e = std::getenv("MR_AHA_TOEP_GPU_METHOD");
        if (e && e[0] && atoi(e) >= 0) opts.gpu_method = atoi(e);
    }
    {
        /* Bound cufinufft's internal fine grid: sigma^D * prod(M) per transform.
         * cufinufft processes the ntr transforms in batches of
         * gpu_maxbatchsize, holding one fine grid per transform of a batch;
         * `fine` is the modes of ONE transform (Mx*My for 2D / stack, since a
         * stack's kz planes are separate transforms). mb is how many such
         * grids fit into the budget left over by mr_toep_pick_chunk. */
        const double sigma_d = (nufft_dim == 3)
            ? opts.upsampfac * opts.upsampfac * opts.upsampfac
            : opts.upsampfac * opts.upsampfac;
        const int64_t fine = (nufft_dim == 3) ? pad_size : (int64_t)Mx * My;
        const size_t per_tr = (size_t)(sigma_d * (double)fine) * sizeof(mr_complex);
        int64_t mb = (per_tr > 0) ? (int64_t)(nufft_budget / per_tr) : ntr;
        /* cufinufft addresses one batch of fine grids with 32-bit integers
         * (maxbatchsize * nf1 * nf2 [* nf3]), so a batch must hold fewer than
         * 2^31 fine-grid points whatever the memory budget. Exceeding it fails
         * in cufinufftf_execute with FINUFFT_ERR_CUDA_FAILURE; on an 80 GB GPU
         * this happens for the 640^2 x 56 stack, whose leftover budget allows
         * ~1400 transforms of 1280^2 points. The 1.25 factor covers cufinufft
         * rounding each fine dimension up to a 2-3-5-smooth size. */
        const double fine_pts = 1.25 * sigma_d * (double)fine;
        const int64_t mb_int32 = (int64_t)(2147483647.0 / fine_pts);
        /* Final clamp: int32 limit, at least one transform per batch, and no
         * more than the transforms of a chunk (a larger batch could never be
         * filled). */
        if (mb > mb_int32) mb = mb_int32;
        if (mb < 1) mb = 1;
        if (mb > ntr) mb = ntr;
        opts.gpu_maxbatchsize = (int)mb;
    }

    /* Type 1, iflag -1, ntr transforms per execute. For nufft_dim == 2
     * cufinufft only reads modes[0..1]; Mz then enters solely through ntr. */
    int64_t modes[3] = {Mx, My, Mz};
    {
        const int err = cufinufftf_makeplan(1, nufft_dim, modes, -1, (int)ntr,
                                            (float)cfg->eps, &w.nufft, &opts);
        if (err != 0) {
            w.nufft = NULL;
            mr_toep_ctx_release(c);
            MR_FAIL(MR_ERR_CUFINUFFT, "cufinufftf_makeplan failed with code %d",
                    err);
        }
    }
    {
        /* cufinufft keeps these pointers: d_traj must outlive the context.
         * d_traj is `[Ntrajs, Nkx, ndim]`, so each coordinate is a contiguous
         * run of M_nu values. The same M_nu points serve every transform of
         * the batch (all components and, for a stack, all kz planes); setpts
         * also bins/sorts them once for all later executes. */
        const float *xj = d_traj;
        const float *yj = d_traj + M_nu;
        const float *zj = (nufft_dim == 3) ? (d_traj + 2 * M_nu) : NULL;
        const int err = cufinufftf_setpts(w.nufft, M_nu, xj, yj, zj, 0, NULL,
                                          NULL, NULL);
        if (err != 0) {
            mr_toep_ctx_release(c);
            MR_FAIL(MR_ERR_CUFINUFFT, "cufinufftf_setpts failed with code %d", err);
        }
    }

    /* -- 4. Plan the centered inverse FFT over one chunk --------------------- */
    /* Contiguous batched C2C (stride 1, dist = one transform's volume) over
     * per_comp * chunk transforms, executed in place on d_work. The plan is
     * bound to the context's stream; with the legacy default stream (s ==
     * NULL) cuFFT already uses it, so no call is needed. */
    mr_toep_geom(cfg, &c->g);
    {
        mr_toep_fft_geom &g = c->g;
        const cufftResult fr =
            cufftPlanMany(&w.fft, g.rank, g.n, NULL, 1, g.dist, NULL, 1, g.dist,
                          CUFFT_C2C, (int)((int64_t)g.per_comp * chunk));
        if (fr != CUFFT_SUCCESS) {
            w.fft = 0;
            mr_toep_ctx_release(c);
            MR_FAIL(MR_ERR_CUFFT, "cufftPlanMany failed with code %d", (int)fr);
        }
        if (s != NULL) cufftSetStream(w.fft, s);
    }

    /* Success: ownership of every buffer and plan passes to the caller, who
     * releases them with mr_toep_ctx_destroy. The packing kernel of step 1
     * may still be running; work issued later on s is ordered after it. */
    *ctx_out = c;
    return MR_SUCCESS;
}

/* Documented in mr_internal.cuh. */
int mr_toep_ctx_chunk(const mr_toep_ctx *ctx) { return ctx->chunk; }

/* Documented in mr_internal.cuh. Issues steps 5a-5d of the pipeline for every
 * chunk of [t_begin, t_end) on the context's stream; the host only blocks when
 * ms_out asks for a timing. */
int mr_toep_ctx_run(mr_toep_ctx *ctx, const int64_t *d_mask, int64_t t_begin,
                    int64_t t_end, mr_complex *d_out, int64_t t_out0,
                    mr_toep_chunk_fn on_chunk, void *user, float *ms_out)
{
    /* Local aliases of the context. c_elems / slab_elems are the FULL chunk
     * sizes the buffers and plans were made for; the last chunk of a range
     * is still processed at full size (zero-padded) and only its `valid`
     * components are written out in 5d. */
    const mr_config *cfg = &ctx->cfg;
    mr_toep_workspace &w = ctx->w;
    const mr_toep_fft_geom &g = ctx->g;
    cudaStream_t s = ctx->s;
    const int     chunk      = ctx->chunk;
    const int64_t pad_size   = ctx->pad_size;
    const int64_t M_nu       = ctx->Ntrajs * ctx->Nkx;
    const int64_t c_elems    = M_nu * (int64_t)ctx->Nkz_eff * chunk;
    const int64_t slab_elems = pad_size * chunk;
    const float   scale      = g.scale;
    const int Mx = cfg->phi2_dim[0], My = cfg->phi2_dim[1], Mz = cfg->phi2_dim[2];
    const int s0 = g.s[0], s1 = g.s[1], s2 = g.s[2];

    MR_REQUIRE(t_begin >= 0 && t_begin <= t_end && t_end <= ctx->ntri &&
                   t_out0 <= t_begin,
               "mr_toep_ctx_run: bad component range [%lld, %lld) (output from %lld)",
               (long long)t_begin, (long long)t_end, (long long)t_out0);
    /* Optional timing: events are created on first use and kept in the
     * context for later runs; ev0 marks the start of the range on s. */
    if (ms_out) {
        *ms_out = 0.0f;
        if (!ctx->ev0) MR_CUDA_CHECK(cudaEventCreate(&ctx->ev0));
        if (!ctx->ev1) MR_CUDA_CHECK(cudaEventCreate(&ctx->ev1));
        MR_CUDA_CHECK(cudaEventRecord(ctx->ev0, s));
    }

    /* -- 5. Chunk loop ------------------------------------------------------ */
    /* Each iteration evaluates components [t0, t0 + valid). All four steps
     * reuse the same d_c / d_fk / d_work, which is safe without host syncs
     * because everything (kernels, cufinufft, cuFFT) is issued on stream s
     * and therefore executes in order. */
    for (int64_t t0 = t_begin; t0 < t_end; t0 += chunk) {
        const int64_t remaining = t_end - t0;
        const int     valid     = (int)((remaining < chunk) ? remaining : chunk);

        /* 5a. Broadcast the packed blocks onto the non-uniform samples.
         *     Components at or beyond t_end are zero padding of the batch.
         *     Every sample m of readout itraj receives that readout's Gram
         *     element (the block is constant along the readout), giving the
         *     `[M_nu, Nkz_eff, chunk]` strengths of ntr transforms. Passing
         *     t_end (not ntri) as the limit also zero-pads a range that ends
         *     before the last component. */
        {
            const int grid = (int)mr_ceil_div(c_elems, MR_BLOCK);
            mr_k_fill_strengths<<<grid, MR_BLOCK, 0, s>>>(
                w.d_c, w.d_tri, ctx->Ntrajs, ctx->Nkx, ctx->Nkz_eff, t0, chunk,
                t_end);
            if (cudaGetLastError() != cudaSuccess)
                MR_FAIL(MR_ERR_CUDA, "mr_k_fill_strengths failed to launch: %s",
                        cudaGetErrorString(cudaGetLastError()));
        }

        /* 5b. Type-1 NUFFT: the point-spread function on the upsampled grid.
         *     All ntr transforms of the chunk in one call (cufinufft splits
         *     them into gpu_maxbatchsize batches internally); output d_fk is
         *     in centered (CMCL) mode order. */
        {
            const int err = cufinufftf_execute(w.nufft, w.d_c, w.d_fk);
            if (err != 0)
                MR_FAIL(MR_ERR_CUFINUFFT, "cufinufftf_execute failed with code %d",
                        err);
        }

        /* 5c. Centered inverse FFT: shift to natural order, transform, then
         *     shift back while applying the 1/prod(M) normalization.
         *     The shift into d_work (out of place) is a pure index remap with
         *     the per-axis offsets g.s (0 on kz for a stack); the cuFFT then
         *     runs in place on d_work. The shift back and the scaling are
         *     fused into the output kernels of 5d. */
        {
            const int grid = (int)mr_ceil_div(slab_elems, MR_BLOCK);
            mr_k_to_natural<<<grid, MR_BLOCK, 0, s>>>(w.d_work, w.d_fk, Mx, My, Mz,
                                                      s0, s1, s2, chunk);
            if (cudaGetLastError() != cudaSuccess)
                MR_FAIL(MR_ERR_CUDA, "mr_k_to_natural failed to launch: %s",
                        cudaGetErrorString(cudaGetLastError()));
        }
        {
            const cufftResult fr =
                cufftExecC2C(w.fft, w.d_work, w.d_work, CUFFT_INVERSE);
            if (fr != CUFFT_SUCCESS)
                MR_FAIL(MR_ERR_CUFFT, "cufftExecC2C failed with code %d", (int)fr);
        }

        /* 5d. Write out the valid components of this chunk; component t goes
         *     to slab (t - t_out0) of d_out. Only `valid` volumes are read
         *     from d_work, so the zero-padded tail of the last chunk is
         *     dropped here. */
        if (cfg->is_masked) {
            /* Masked: gather num_masked centered locations per component into
             * column (t - t_out0) of the `[num_masked, ...]` output. One
             * thread per (location, component). */
            const long long total = (long long)cfg->num_masked * valid;
            const int grid = (int)mr_ceil_div(total, MR_BLOCK);
            mr_k_gather_masked_scale<<<grid, MR_BLOCK, 0, s>>>(
                d_out, w.d_work, d_mask, (long long)cfg->num_masked, Mx, My,
                Mz, s0, s1, s2, t0 - t_out0, valid, scale);
            if (cudaGetLastError() != cudaSuccess)
                MR_FAIL(MR_ERR_CUDA,
                        "mr_k_gather_masked_scale failed to launch: %s",
                        cudaGetErrorString(cudaGetLastError()));
        } else {
            /* Unmasked: whole `[Mx, My, Mz]` volumes back to centered order,
             * written at slab offset pad_size * (t0 - t_out0). */
            const long long total = pad_size * valid;
            const int grid = (int)mr_ceil_div(total, MR_BLOCK);
            mr_k_to_centered_scale<<<grid, MR_BLOCK, 0, s>>>(
                d_out + pad_size * (t0 - t_out0), w.d_work, Mx, My, Mz, s0, s1, s2,
                valid, scale);
            if (cudaGetLastError() != cudaSuccess)
                MR_FAIL(MR_ERR_CUDA, "mr_k_to_centered_scale failed to launch: %s",
                        cudaGetErrorString(cudaGetLastError()));
        }

        /* Hook for the multi-GPU path: called once the chunk's work is
         * ISSUED (not finished). The callback typically records an event on
         * s and enqueues peer copies behind it, overlapping the transfer of
         * this chunk with the computation of the next one. */
        if (on_chunk != NULL) {
            const int rc = on_chunk(user, t0, valid);
            if (rc != MR_SUCCESS) return rc;
        }

        if (cfg->verbose)
            printf("[mr_aha] calc_toep: components %lld-%lld of %lld issued\n",
                   (long long)t0, (long long)(t0 + valid - 1),
                   (long long)ctx->ntri);
    }

    /* Timed run: mark the end of the range and block until it completes, so
     * the elapsed GPU time (excluding planning) can be read back. */
    if (ms_out) {
        MR_CUDA_CHECK(cudaEventRecord(ctx->ev1, s));
        MR_CUDA_CHECK(cudaEventSynchronize(ctx->ev1));
        MR_CUDA_CHECK(cudaEventElapsedTime(ms_out, ctx->ev0, ctx->ev1));
    }
    return MR_SUCCESS;
}

/* Documented in mr_internal.cuh. */
int mr_toep_ctx_destroy(mr_toep_ctx *ctx)
{
    if (ctx == NULL) return MR_SUCCESS;
    /* Synchronize before freeing the scratch the asynchronous work reads.
     * With the legacy default stream (NULL) the whole device is synced. The
     * context is released even if the sync failed (its error is reported
     * afterwards), so destroy never leaks. */
    const cudaError_t ce = (ctx->s != NULL) ? cudaStreamSynchronize(ctx->s)
                                            : cudaDeviceSynchronize();
    mr_toep_ctx_release(ctx);
    if (ce != cudaSuccess) {
        cudaGetLastError();
        MR_FAIL(MR_ERR_CUDA, "synchronization after calc_toep failed: %s",
                cudaGetErrorString(ce));
    }
    return MR_SUCCESS;
}

/* Body of ::mr_calc_toep, run with the target device current. Documented in
 * mr_internal.cuh because the operator object (mr_op.cu) reuses it. */
int mr_calc_toep_impl(const mr_config *cfg, const mr_complex *d_phi2,
                      const float *d_traj, int64_t Ntrajs, int64_t Nkx,
                      const int64_t *d_mask, mr_complex *d_phi2_toep)
{
    /* Single-device path: one context (no chunk cap) evaluates every
     * component [0, ntri) straight into the caller's whole output (t_out0 =
     * 0), without callback or timing. */
    mr_toep_ctx *ctx = NULL;
    int rc = mr_toep_ctx_create(cfg, d_phi2, d_traj, Ntrajs, Nkx, 0, &ctx);
    if (rc != MR_SUCCESS) return rc;
    rc = mr_toep_ctx_run(ctx, d_mask, 0, mr_num_tri(cfg->L), d_phi2_toep, 0,
                         NULL, NULL, NULL);
    if (rc != MR_SUCCESS) {
        /* Keep the first diagnostic; still wait before freeing. */
        char msg[512];
        snprintf(msg, sizeof(msg), "%s", mr_last_error());
        mr_toep_ctx_destroy(ctx);
        mr_set_error("%s", msg);
        return rc;
    }
    /* Success: destroy waits for the stream, so mr_calc_toep is synchronous
     * and its return code includes any asynchronous execution error. */
    return mr_toep_ctx_destroy(ctx);
}

/* Public entry point (documented in mr_aha.h): argument checks, config
 * validation, device switch, then mr_calc_toep_impl. */
extern "C" int mr_calc_toep(const mr_config *cfg, const mr_complex *d_phi2,
                            const float *d_traj, int64_t Ntrajs, int64_t Nkx,
                            const int64_t *d_mask, mr_complex *d_phi2_toep)
{
    /* Pointer and size checks first (cheap, host-only); MR_REQUIRE returns
     * ::MR_ERR_INVALID_ARG with a message in mr_last_error. */
    mr_clear_error();
    MR_REQUIRE(cfg != NULL, "mr_calc_toep: cfg is NULL");
    MR_REQUIRE(d_phi2 != NULL, "mr_calc_toep: d_phi2 is NULL");
    MR_REQUIRE(d_traj != NULL, "mr_calc_toep: d_traj is NULL");
    MR_REQUIRE(d_phi2_toep != NULL, "mr_calc_toep: d_phi2_toep is NULL");
    MR_REQUIRE(Ntrajs > 0, "mr_calc_toep: Ntrajs = %lld must be > 0",
               (long long)Ntrajs);
    MR_REQUIRE(Nkx > 0, "mr_calc_toep: Nkx = %lld must be > 0", (long long)Nkx);
    if (cfg->is_masked)
        MR_REQUIRE(d_mask != NULL,
                   "mr_calc_toep: d_mask is required when is_masked is set");

    const int status = mr_config_validate(cfg);
    if (status != MR_SUCCESS) return status;

    /* Run on cfg->gpu_device_id and restore the caller's device on every
     * exit path, success or failure. */
    int prev = -1;
    int rc = mr_toep_push_device(cfg, &prev);
    if (rc == MR_SUCCESS)
        rc = mr_calc_toep_impl(cfg, d_phi2, d_traj, Ntrajs, Nkx, d_mask,
                               d_phi2_toep);
    if (prev >= 0) cudaSetDevice(prev);
    return rc;
}

/* ------------------------------------------------------------------------ */
/* Two-stage path: finish phi2_toep from external point-spread functions    */
/* ------------------------------------------------------------------------ */

/**
 * @brief Body of ::mr_calc_toep_from_psf, run with the target device current.
 *
 * Steps 5c-5d of ::mr_toep_ctx_run for @p n_comp externally computed PSFs, in
 * a single batch: no chunking, so the scratch and the cuFFT plan are sized for
 * all @p n_comp components at once (the caller bounds memory through
 * @p n_comp). Not
 * reusable: scratch and plan are created and destroyed per call, and the call
 * synchronizes the stream before returning.
 */
static int mr_calc_toep_from_psf_impl(const mr_config *cfg,
                                      const mr_complex *d_psf, int64_t t0,
                                      int64_t n_comp, const int64_t *d_mask,
                                      mr_complex *d_phi2_toep)
{
    /* slab: elements of the n_comp input volumes `[Mx, My, Mz, n_comp]`. */
    const int64_t pad_size = mr_pad_size(cfg);
    const int Mx = cfg->phi2_dim[0], My = cfg->phi2_dim[1], Mz = cfg->phi2_dim[2];
    cudaStream_t s = (cudaStream_t)cfg->stream;
    const long long slab = (long long)pad_size * n_comp;

    /* Same transform geometry as the one-shot path, so both produce the same
     * kernel for the same PSFs. */
    mr_toep_fft_geom g;
    mr_toep_geom(cfg, &g);

    /* Natural-order scratch, transformed in place: d_psf is const and must
     * not be modified, hence the copy. Plan the batched inverse cuFFT over
     * per_comp * n_comp transforms and bind it to the caller's stream. */
    mr_complex *d_work = NULL;
    if (cudaMalloc((void **)&d_work, (size_t)slab * sizeof(mr_complex)) !=
        cudaSuccess)
        MR_FAIL(MR_ERR_ALLOC, "failed to allocate %lld scratch elements for "
                "mr_calc_toep_from_psf", slab);
    cufftHandle fft = 0;
    if (cufftPlanMany(&fft, g.rank, g.n, NULL, 1, g.dist, NULL, 1, g.dist,
                      CUFFT_C2C, (int)((int64_t)g.per_comp * n_comp)) !=
        CUFFT_SUCCESS) {
        cudaFree(d_work);
        MR_FAIL(MR_ERR_CUFFT, "cufftPlanMany failed in mr_calc_toep_from_psf");
    }
    if (s != NULL) cufftSetStream(fft, s);

    /* From here on errors do not return early: rc records the first failure,
     * later steps are skipped, and the shared tail below always syncs and
     * frees the scratch and plan. */
    int rc = MR_SUCCESS;
    /* Centered -> natural order (5c of mr_toep_ctx_run). */
    mr_k_to_natural<<<(int)mr_ceil_div(slab, MR_BLOCK), MR_BLOCK, 0, s>>>(
        d_work, d_psf, Mx, My, Mz, g.s[0], g.s[1], g.s[2], n_comp);
    if (cudaGetLastError() != cudaSuccess) {
        rc = MR_ERR_CUDA;
        mr_set_error("mr_k_to_natural failed to launch in mr_calc_toep_from_psf");
    }
    /* Unnormalized inverse FFT, in place. */
    if (rc == MR_SUCCESS &&
        cufftExecC2C(fft, d_work, d_work, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        rc = MR_ERR_CUFFT;
        mr_set_error("cufftExecC2C failed in mr_calc_toep_from_psf");
    }
    /* Natural -> centered order with the 1/prod(M) scaling, written to
     * components [t0, t0 + n_comp) of the WHOLE output (masked: columns of
     * `[num_masked, ntri]`; unmasked: slabs at offset pad_size * t0). Other
     * components of d_phi2_toep are left untouched. */
    if (rc == MR_SUCCESS) {
        if (cfg->is_masked) {
            const long long total = (long long)cfg->num_masked * n_comp;
            mr_k_gather_masked_scale<<<(int)mr_ceil_div(total, MR_BLOCK), MR_BLOCK,
                                       0, s>>>(
                d_phi2_toep, d_work, d_mask, (long long)cfg->num_masked, Mx, My,
                Mz, g.s[0], g.s[1], g.s[2], t0, n_comp, g.scale);
        } else {
            mr_k_to_centered_scale<<<(int)mr_ceil_div(slab, MR_BLOCK), MR_BLOCK, 0,
                                     s>>>(d_phi2_toep + pad_size * t0, d_work, Mx,
                                          My, Mz, g.s[0], g.s[1], g.s[2], n_comp,
                                          g.scale);
        }
        if (cudaGetLastError() != cudaSuccess) {
            rc = MR_ERR_CUDA;
            mr_set_error("output kernel failed to launch in mr_calc_toep_from_psf");
        }
    }
    /* Shared tail: wait for the issued work (it reads d_work) before freeing
     * it; this also surfaces asynchronous execution errors. The first error
     * wins. */
    const cudaError_t ce =
        (s != NULL) ? cudaStreamSynchronize(s) : cudaDeviceSynchronize();
    if (rc == MR_SUCCESS && ce != cudaSuccess) {
        rc = MR_ERR_CUDA;
        mr_set_error("synchronization failed in mr_calc_toep_from_psf: %s",
                     cudaGetErrorString(ce));
    }
    cufftDestroy(fft);
    cudaFree(d_work);
    return rc;
}

/* Public entry point (documented in mr_aha.h): checks, then the impl on the
 * configured device, restoring the caller's device afterwards. */
extern "C" int mr_calc_toep_from_psf(const mr_config *cfg,
                                     const mr_complex *d_psf, int64_t t0,
                                     int64_t n_comp, const int64_t *d_mask,
                                     mr_complex *d_phi2_toep)
{
    /* The component range is checked against ntri only after the config is
     * validated, because ntri derives from cfg->L. */
    mr_clear_error();
    MR_REQUIRE(cfg != NULL, "mr_calc_toep_from_psf: cfg is NULL");
    MR_REQUIRE(d_psf != NULL, "mr_calc_toep_from_psf: d_psf is NULL");
    MR_REQUIRE(d_phi2_toep != NULL, "mr_calc_toep_from_psf: d_phi2_toep is NULL");
    const int status = mr_config_validate(cfg);
    if (status != MR_SUCCESS) return status;
    const int64_t ntri = mr_num_tri(cfg->L);
    MR_REQUIRE(t0 >= 0 && n_comp >= 1 && t0 + n_comp <= ntri,
               "mr_calc_toep_from_psf: components [%lld, %lld) outside [0, %lld)",
               (long long)t0, (long long)(t0 + n_comp), (long long)ntri);
    if (cfg->is_masked)
        MR_REQUIRE(d_mask != NULL, "mr_calc_toep_from_psf: d_mask is required "
                   "when is_masked is set");

    int prev = -1;
    int rc = mr_toep_push_device(cfg, &prev);
    if (rc == MR_SUCCESS)
        rc = mr_calc_toep_from_psf_impl(cfg, d_psf, t0, n_comp, d_mask,
                                        d_phi2_toep);
    if (prev >= 0) cudaSetDevice(prev);
    return rc;
}
