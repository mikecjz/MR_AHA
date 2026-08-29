/**
 * @file mt_calc_toep.cu
 * @brief Pre-computation of the block-diagonal Toeplitz operator `phi2_toep`.
 *
 * Given the Gram blocks @f$\Phi^H\Phi@f$ sampled along the non-Cartesian
 * trajectory, this file evaluates the Toeplitz multiplier that
 * ::mt_aha_plan_execute applies between its two cuFFTs.
 *
 * @section mt_toep_sign Why a type-1 with sign -1 followed by an inverse FFT
 *
 * Write @f$p^{\pm}[d] = \sum_m w_m e^{\pm i k_m d}@f$ for the point-spread
 * function of the sampling pattern on the upsampled grid; a type-1 NUFFT with
 * `iflag = ±1` evaluates exactly that. Since @f$p^{-}[d] = p^{+}[-d]@f$, the
 * *normalised inverse* DFT of @f$p^{-}@f$ equals the forward DFT of
 * @f$p^{+}@f$ divided by @f$\prod M@f$:
 * @f[
 *   \mathrm{IDFT}(p^{-})[k] \;=\; \frac{1}{\prod M}\,\mathrm{DFT}(p^{+})[k].
 * @f]
 * The right-hand side is precisely the multiplier required between the
 * unnormalised forward and inverse cuFFTs of the @f$A^H A@f$ pipeline, so the
 * pseudo-code's "run ifft" both matches the literature convention and removes
 * every remaining scaling factor from the hot loop.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mt_kernels.cuh"

#include <cufinufft.h>

#include <cstdio>
#include <cstdlib>

/** @brief Default thread-block size for the elementwise kernels here. */
#define MT_BLOCK 256

/**
 * @brief Everything ::mt_calc_toep allocates, so that one cleanup path frees it.
 */
struct mt_toep_workspace {
    mt_complex     *d_tri;   /**< Packed Gram blocks `[nbatch, ntri]`.        */
    mt_complex     *d_c;     /**< cufinufft strengths for one chunk.          */
    mt_complex     *d_fk;    /**< cufinufft output slab, centred order.       */
    mt_complex     *d_work;  /**< Natural-order scratch for the inverse FFT.  */
    cufinufftf_plan nufft;   /**< Type-1 plan, reused across chunks.          */
    cufftHandle     fft;     /**< Batched inverse cuFFT over one chunk.       */
};

/** @brief Release every buffer held by a workspace. Safe on partial failure. */
static void mt_toep_workspace_free(mt_toep_workspace *w)
{
    if (w->nufft) { cufinufftf_destroy(w->nufft); w->nufft = NULL; }
    if (w->fft)   { cufftDestroy(w->fft);         w->fft = 0; }
    if (w->d_tri)  { cudaFree(w->d_tri);  w->d_tri = NULL; }
    if (w->d_c)    { cudaFree(w->d_c);    w->d_c = NULL; }
    if (w->d_fk)   { cudaFree(w->d_fk);   w->d_fk = NULL; }
    if (w->d_work) { cudaFree(w->d_work); w->d_work = NULL; }
}

/**
 * @brief Make @p cfg's device current, remembering the caller's device.
 * @param[in]  cfg  Configuration carrying the requested device.
 * @param[out] prev Receives the previously current device, or -1 if unchanged.
 * @return ::MT_SUCCESS or ::MT_ERR_CUDA.
 */
static int mt_toep_push_device(const mt_config *cfg, int *prev)
{
    *prev = -1;
    if (cfg->gpu_device_id < 0) return MT_SUCCESS;
    int cur = 0;
    MT_CUDA_CHECK(cudaGetDevice(&cur));
    if (cur != cfg->gpu_device_id) {
        MT_CUDA_CHECK(cudaSetDevice(cfg->gpu_device_id));
        *prev = cur;
    }
    return MT_SUCCESS;
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
 * @param[in]  cfg         Configuration (supplies ::mt_config::mem_fraction).
 * @param[in]  pad_size    `Mx*My*Mz`.
 * @param[in]  strengths   Non-uniform strengths per component,
 *                         `Ntrajs*Nkx*Nkz_eff`.
 * @param[in]  ntri        Total number of triangular components.
 * @param[in]  persistent  Bytes already committed to persistent buffers.
 * @param[out] chunk_out   Chosen number of components per chunk, >= 1.
 * @param[out] budget_out  Bytes left over for cufinufft's internal grids.
 * @return ::MT_SUCCESS or ::MT_ERR_CUDA.
 */
static int mt_toep_pick_chunk(const mt_config *cfg, int64_t pad_size,
                              int64_t strengths, int64_t ntri,
                              size_t persistent, int *chunk_out,
                              size_t *budget_out)
{
    size_t free_bytes = 0, total_bytes = 0;
    MT_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    double frac = cfg->mem_fraction;
    if (!(frac > 0.0)) frac = 0.5;
    if (frac > 0.95) frac = 0.95;

    size_t budget = (size_t)(frac * (double)free_bytes);
    budget = (budget > persistent) ? (budget - persistent) : 0;

    /* Split the budget evenly between our slabs and cufinufft's fine grids. */
    const size_t ours = budget / 2;
    const size_t per_component =
        (size_t)(2 * pad_size + strengths) * sizeof(mt_complex);

    int64_t chunk = (per_component > 0) ? (int64_t)(ours / per_component) : ntri;
    if (chunk < 1) chunk = 1;
    if (chunk > ntri) chunk = ntri;

    *chunk_out  = (int)chunk;
    *budget_out = budget - ours;
    return MT_SUCCESS;
}

/** @brief Body of ::mt_calc_toep, run with the target device current. */
static int mt_calc_toep_impl(const mt_config *cfg, const mt_complex *d_phi2,
                             const float *d_traj, int64_t Ntrajs, int64_t Nkx,
                             const int64_t *d_mask, mt_complex *d_phi2_toep)
{
    const int      is_stack  = (cfg->traj_type == MT_TRAJ_STACK);
    const int      nufft_dim = (cfg->traj_type == MT_TRAJ_3D) ? 3 : 2;
    const int      Nkz_eff   = is_stack ? cfg->phi2_dim[2] : 1;
    const int64_t  ntri      = mt_num_tri(cfg->L);
    const int64_t  pad_size  = mt_pad_size(cfg);
    const int64_t  M_nu      = Ntrajs * Nkx;
    const int64_t  nbatch    = Ntrajs * (int64_t)Nkz_eff;
    cudaStream_t   s         = (cudaStream_t)cfg->stream;

    const int Mx = cfg->phi2_dim[0], My = cfg->phi2_dim[1], Mz = cfg->phi2_dim[2];

    mt_toep_workspace w = {};
    int rc = MT_SUCCESS;

    /* -- 1. Pack the dense Gram blocks into the requested triangle ---------- */
    if (cudaMalloc((void **)&w.d_tri,
                   (size_t)nbatch * ntri * sizeof(mt_complex)) != cudaSuccess) {
        mt_toep_workspace_free(&w);
        MT_FAIL(MT_ERR_ALLOC, "failed to allocate %lld packed Gram elements",
                (long long)(nbatch * ntri));
    }
    {
        const long long total = nbatch * (long long)cfg->L * cfg->L;
        const int grid = (int)mt_ceil_div(total, MT_BLOCK);
        mt_k_extract_tri<<<grid, MT_BLOCK, 0, s>>>(w.d_tri, d_phi2, nbatch,
                                                   cfg->L, cfg->triangulation ==
                                                               MT_TRI_UPPER);
        if (cudaGetLastError() != cudaSuccess) {
            mt_toep_workspace_free(&w);
            MT_FAIL(MT_ERR_CUDA, "mt_k_extract_tri failed to launch: %s",
                    cudaGetErrorString(cudaGetLastError()));
        }
    }

    /* -- 2. Size the chunking against the free device memory ---------------- */
    int    chunk  = 1;
    size_t nufft_budget = 0;
    rc = mt_toep_pick_chunk(cfg, pad_size, M_nu * Nkz_eff, ntri,
                            (size_t)nbatch * ntri * sizeof(mt_complex), &chunk,
                            &nufft_budget);
    if (rc != MT_SUCCESS) { mt_toep_workspace_free(&w); return rc; }

    const int64_t ntr        = (int64_t)Nkz_eff * chunk; /* transforms/chunk  */
    const int64_t slab_elems = pad_size * chunk;
    const int64_t c_elems    = M_nu * ntr;

    if (cudaMalloc((void **)&w.d_c, (size_t)c_elems * sizeof(mt_complex)) !=
            cudaSuccess ||
        cudaMalloc((void **)&w.d_fk, (size_t)slab_elems * sizeof(mt_complex)) !=
            cudaSuccess ||
        cudaMalloc((void **)&w.d_work, (size_t)slab_elems * sizeof(mt_complex)) !=
            cudaSuccess) {
        mt_toep_workspace_free(&w);
        MT_FAIL(MT_ERR_ALLOC,
                "failed to allocate the calc_toep chunk buffers (chunk = %d); "
                "lower mt_config.mem_fraction or the upsampled grid size",
                chunk);
    }

    /* -- 3. Plan the type-1 NUFFT once and reuse it for every chunk --------- */
    cufinufft_opts opts;
    cufinufft_default_opts(&opts);
    opts.upsampfac = 2.0;
    opts.modeord   = 0; /* CMCL / centred mode order, matching our convention */
    opts.gpu_stream = cfg->stream;
    if (cfg->gpu_device_id >= 0) opts.gpu_device_id = cfg->gpu_device_id;
    opts.debug = cfg->verbose ? 1 : 0;
    {
        /* Bound cufinufft's internal fine grid: sigma^D * prod(M) per transform. */
        const double sigma_d = (nufft_dim == 3) ? 8.0 : 4.0;
        const int64_t fine = (nufft_dim == 3) ? pad_size : (int64_t)Mx * My;
        const size_t per_tr = (size_t)(sigma_d * (double)fine) * sizeof(mt_complex);
        int64_t mb = (per_tr > 0) ? (int64_t)(nufft_budget / per_tr) : ntr;
        if (mb < 1) mb = 1;
        if (mb > ntr) mb = ntr;
        opts.gpu_maxbatchsize = (int)mb;
    }

    int64_t modes[3] = {Mx, My, Mz};
    {
        const int err = cufinufftf_makeplan(1, nufft_dim, modes, -1, (int)ntr,
                                            (float)cfg->eps, &w.nufft, &opts);
        if (err != 0) {
            mt_toep_workspace_free(&w);
            MT_FAIL(MT_ERR_CUFINUFFT, "cufinufftf_makeplan failed with code %d",
                    err);
        }
    }
    {
        const float *xj = d_traj;
        const float *yj = d_traj + M_nu;
        const float *zj = (nufft_dim == 3) ? (d_traj + 2 * M_nu) : NULL;
        const int err = cufinufftf_setpts(w.nufft, M_nu, xj, yj, zj, 0, NULL,
                                          NULL, NULL);
        if (err != 0) {
            mt_toep_workspace_free(&w);
            MT_FAIL(MT_ERR_CUFINUFFT, "cufinufftf_setpts failed with code %d", err);
        }
    }

    /* -- 4. Plan the centred inverse FFT over one chunk --------------------- */
    int fft_rank, fft_n[3], fft_batch, fft_dist;
    float scale;
    if (nufft_dim == 3) {
        /* Fully 3D: transform every axis of the upsampled grid. */
        fft_rank  = 3;
        fft_n[0]  = Mz; fft_n[1] = My; fft_n[2] = Mx;
        fft_batch = chunk;
        fft_dist  = (int)pad_size;
        scale     = 1.0f / (float)((double)Mx * My * Mz);
    } else {
        /* 2D and stack-of-stars: kz (if any) is already Cartesian, so it is
         * merely another batch axis and is neither transformed nor shifted. */
        fft_rank  = 2;
        fft_n[0]  = My; fft_n[1] = Mx;
        fft_batch = (int)((int64_t)Nkz_eff * chunk);
        fft_dist  = Mx * My;
        scale     = 1.0f / (float)((double)Mx * My);
    }
    {
        const cufftResult fr =
            cufftPlanMany(&w.fft, fft_rank, fft_n, NULL, 1, fft_dist, NULL, 1,
                          fft_dist, CUFFT_C2C, fft_batch);
        if (fr != CUFFT_SUCCESS) {
            mt_toep_workspace_free(&w);
            MT_FAIL(MT_ERR_CUFFT, "cufftPlanMany failed with code %d", (int)fr);
        }
        if (s != NULL) cufftSetStream(w.fft, s);
    }

    /* Shift amounts: zero on an axis that is not transformed. */
    const int s0 = Mx / 2;
    const int s1 = My / 2;
    const int s2 = (nufft_dim == 3) ? (Mz / 2) : 0;

    /* -- 5. Chunk loop ------------------------------------------------------ */
    for (int64_t t0 = 0; t0 < ntri; t0 += chunk) {
        const int64_t remaining = ntri - t0;
        const int     valid     = (int)((remaining < chunk) ? remaining : chunk);

        /* 5a. Broadcast the packed blocks onto the non-uniform samples. */
        {
            const int grid = (int)mt_ceil_div(c_elems, MT_BLOCK);
            mt_k_fill_strengths<<<grid, MT_BLOCK, 0, s>>>(
                w.d_c, w.d_tri, Ntrajs, Nkx, Nkz_eff, t0, chunk, ntri);
            if (cudaGetLastError() != cudaSuccess) {
                mt_toep_workspace_free(&w);
                MT_FAIL(MT_ERR_CUDA, "mt_k_fill_strengths failed to launch: %s",
                        cudaGetErrorString(cudaGetLastError()));
            }
        }

        /* 5b. Type-1 NUFFT: the point-spread function on the upsampled grid. */
        {
            const int err = cufinufftf_execute(w.nufft, w.d_c, w.d_fk);
            if (err != 0) {
                mt_toep_workspace_free(&w);
                MT_FAIL(MT_ERR_CUFINUFFT, "cufinufftf_execute failed with code %d",
                        err);
            }
        }

        /* 5c. Centred inverse FFT: shift to natural order, transform, then
         *     shift back while applying the 1/prod(M) normalisation. */
        {
            const int grid = (int)mt_ceil_div(slab_elems, MT_BLOCK);
            mt_k_to_natural<<<grid, MT_BLOCK, 0, s>>>(w.d_work, w.d_fk, Mx, My, Mz,
                                                      s0, s1, s2, chunk);
            if (cudaGetLastError() != cudaSuccess) {
                mt_toep_workspace_free(&w);
                MT_FAIL(MT_ERR_CUDA, "mt_k_to_natural failed to launch: %s",
                        cudaGetErrorString(cudaGetLastError()));
            }
        }
        {
            const cufftResult fr =
                cufftExecC2C(w.fft, w.d_work, w.d_work, CUFFT_INVERSE);
            if (fr != CUFFT_SUCCESS) {
                mt_toep_workspace_free(&w);
                MT_FAIL(MT_ERR_CUFFT, "cufftExecC2C failed with code %d", (int)fr);
            }
        }

        /* 5d. Write out the valid components of this chunk. */
        if (cfg->is_masked) {
            const long long total = (long long)cfg->num_masked * valid;
            const int grid = (int)mt_ceil_div(total, MT_BLOCK);
            mt_k_gather_masked_scale<<<grid, MT_BLOCK, 0, s>>>(
                d_phi2_toep, w.d_work, d_mask, (long long)cfg->num_masked, Mx, My,
                Mz, s0, s1, s2, t0, valid, scale);
            if (cudaGetLastError() != cudaSuccess) {
                mt_toep_workspace_free(&w);
                MT_FAIL(MT_ERR_CUDA,
                        "mt_k_gather_masked_scale failed to launch: %s",
                        cudaGetErrorString(cudaGetLastError()));
            }
        } else {
            const long long total = pad_size * valid;
            const int grid = (int)mt_ceil_div(total, MT_BLOCK);
            mt_k_to_centered_scale<<<grid, MT_BLOCK, 0, s>>>(
                d_phi2_toep + pad_size * t0, w.d_work, Mx, My, Mz, s0, s1, s2,
                valid, scale);
            if (cudaGetLastError() != cudaSuccess) {
                mt_toep_workspace_free(&w);
                MT_FAIL(MT_ERR_CUDA, "mt_k_to_centered_scale failed to launch: %s",
                        cudaGetErrorString(cudaGetLastError()));
            }
        }

        if (cfg->verbose)
            printf("[mt_cuda] calc_toep: components %lld-%lld of %lld done\n",
                   (long long)t0, (long long)(t0 + valid - 1), (long long)ntri);
    }

    /* Synchronise before freeing the scratch the asynchronous work reads. */
    {
        const cudaError_t ce = (s != NULL) ? cudaStreamSynchronize(s)
                                           : cudaDeviceSynchronize();
        if (ce != cudaSuccess) {
            mt_toep_workspace_free(&w);
            MT_FAIL(MT_ERR_CUDA, "synchronisation after calc_toep failed: %s",
                    cudaGetErrorString(ce));
        }
    }
    mt_toep_workspace_free(&w);
    return MT_SUCCESS;
}

extern "C" int mt_calc_toep(const mt_config *cfg, const mt_complex *d_phi2,
                            const float *d_traj, int64_t Ntrajs, int64_t Nkx,
                            const int64_t *d_mask, mt_complex *d_phi2_toep)
{
    mt_clear_error();
    MT_REQUIRE(cfg != NULL, "mt_calc_toep: cfg is NULL");
    MT_REQUIRE(d_phi2 != NULL, "mt_calc_toep: d_phi2 is NULL");
    MT_REQUIRE(d_traj != NULL, "mt_calc_toep: d_traj is NULL");
    MT_REQUIRE(d_phi2_toep != NULL, "mt_calc_toep: d_phi2_toep is NULL");
    MT_REQUIRE(Ntrajs > 0, "mt_calc_toep: Ntrajs = %lld must be > 0",
               (long long)Ntrajs);
    MT_REQUIRE(Nkx > 0, "mt_calc_toep: Nkx = %lld must be > 0", (long long)Nkx);
    if (cfg->is_masked)
        MT_REQUIRE(d_mask != NULL,
                   "mt_calc_toep: d_mask is required when is_masked is set");

    const int status = mt_config_validate(cfg);
    if (status != MT_SUCCESS) return status;

    int prev = -1;
    int rc = mt_toep_push_device(cfg, &prev);
    if (rc == MT_SUCCESS)
        rc = mt_calc_toep_impl(cfg, d_phi2, d_traj, Ntrajs, Nkx, d_mask,
                               d_phi2_toep);
    if (prev >= 0) cudaSetDevice(prev);
    return rc;
}
