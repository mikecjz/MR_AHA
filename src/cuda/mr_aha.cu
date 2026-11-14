/**
 * @file mr_aha.cu
 * @brief Implementation of the Toeplitz-embedded forward normal operator AhA.
 *
 * The operator is evaluated coil by coil as
 * @f[
 *   A^H A x = \sum_c S_c^H \, \mathcal{P}^H \, F^{-1} \hat{T} F \,
 *             \mathcal{P} \, S_c \, x ,
 * @f]
 * where @f$\mathcal{P}@f$ zero-pads onto the upsampled Toeplitz grid,
 * @f$F@f$ is an unnormalised cuFFT whose centring is realised by index remaps
 * folded into @f$\mathcal{P}@f$ and into the block-diagonal lookup, and
 * @f$\hat{T}@f$ is the per-voxel Hermitian @f$L\times L@f$ multiplication.
 *
 * The @f$1/\prod M@f$ that the unnormalised forward/inverse cuFFT pair would
 * otherwise require is already carried by `phi2_toep` as produced by
 * ::mr_calc_toep, so no scaling happens in this file.
 *
 * Performance notes (see also `mr_kernels_bd.cu`):
 *  - The block-diagonal stage is memory-bound on reading `phi2_toep`, the
 *    largest array of the operator. For @f$L \le@f$ ::MR_BD_MAX_L it uses
 *    kernels specialised on @f$L@f$ (register vectors, each stored element
 *    read once per triangle walk), and up to ::MR_BDT_MAX_L tiled kernels
 *    with the same guarantees (`mr_kernels_bdt.cu`).
 *  - Coils are processed in groups of up to ::mr_aha_plan_s::group: the padded
 *    grids of a group sit side by side, one batched cuFFT transforms them all,
 *    and one block-diagonal launch serves the whole group, reading
 *    `phi2_toep` from DRAM about once per group instead of once per coil. The
 *    group size is the default ::MR_CUDA_COIL_GROUP (or
 *    `MR_AHA_CUDA_COIL_GROUP`), reduced so that the plan's buffers fit
 *    ::mr_config::mem_fraction of the free device memory.
 *  - Environment variables, read at plan creation:
 *    `MR_AHA_CUDA_GENERIC_BD=1` forces the generic shared-memory kernels (one
 *    coil at a time; for testing and A/B timing), `MR_AHA_CUDA_TILED_BD=1`
 *    the tiled kernels (`mr_kernels_bdt.cu`, normally used above
 *    ::MR_BD_MAX_L) for every rank, `MR_AHA_CUDA_COIL_GROUP=n`
 *    sets the requested group size, `MR_AHA_CUDA_BD_THREADS=n` the threads
 *    per block of the specialised kernels (default ::MR_BD_THREADS, at most
 *    that). At execution, `MR_AHA_CUDA_PROFILE=1` prints the GPU time of each
 *    stage to `stderr`.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_kernels.cuh"

#include <cstdio>
#include <cstdlib>
#include <vector>

/** @brief Default thread-block size for the elementwise (non-shared) kernels. */
#define MR_BLOCK 256

/** @brief Default number of coils per group (see the file comment). */
#define MR_CUDA_COIL_GROUP 4

/**
 * @brief Internal state of a reusable @f$A^H A@f$ plan.
 *
 * Holds the cuFFT plans and every persistent device buffer, so that repeated
 * ::mr_aha_plan_execute calls inside a conjugate-gradient loop allocate
 * nothing.
 */
struct mr_aha_plan_s {
    mr_config     cfg;       /**< Copy of the caller's configuration.         */
    mr_dims       dims;      /**< Kernel geometry descriptor.                 */
    cudaStream_t  stream;    /**< Stream every operation is issued on.        */
    int           group;     /**< Coils per group (>= 1).                     */
    cufftHandle   fft;       /**< cuFFT over `group * L` padded volumes.      */
    cufftHandle   fft_rem;   /**< cuFFT over the last, smaller group's
                                  `(n_coils % group) * L` volumes, or 0 when
                                  the coils divide evenly into groups.        */
    void         *d_work;    /**< cuFFT work area shared by both plans.       */
    mr_complex   *d_pad;     /**< Padded grids `[Mx,My,Mz,L,group]`.          */
    mr_complex   *d_compact; /**< `[num_masked,L]`; masked mode with the
                                  generic kernels only.                       */
    int          *d_inv;     /**< `[Mx,My,Mz]` grid-to-mask index map; masked
                                  mode with the specialised kernels only.     */
    mr_bd_kernels bd;        /**< Specialised block-diagonal kernels, or NULL
                                  pointers for the generic ones.              */
    int           bd_block;  /**< Threads per block for the block-diag kernel.*/
    size_t        bd_shared; /**< Dynamic shared bytes for the generic kernel.*/
};

/* ------------------------------------------------------------------------ */
/* Small helpers                                                            */
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
static int mr_push_device(const mr_config *cfg, int *prev)
{
    *prev = -1;
    if (cfg->gpu_device_id < 0) return MR_SUCCESS;
    int cur = 0;
    MR_CUDA_CHECK(cudaGetDevice(&cur));
    if (cur != cfg->gpu_device_id) {
        MR_CUDA_CHECK(cudaSetDevice(cfg->gpu_device_id));
        *prev = cur;
    }
    return MR_SUCCESS;
}

/** @brief Restore the device saved by ::mr_push_device. Errors are ignored. */
static void mr_pop_device(int prev)
{
    if (prev >= 0) cudaSetDevice(prev);
}

/**
 * @brief Choose a launch configuration for the shared-memory block-diagonal
 *        kernel.
 *
 * Each thread stages its own length-@f$L@f$ rank vector in shared memory, so
 * the block size is bounded by the shared-memory budget. Blocks are halved from
 * 256 until they fit; if 96 KB-class opt-in shared memory is required, it is
 * requested explicitly on both block-diagonal kernels.
 *
 * @param[in]  L         Rank.
 * @param[out] block_out Chosen threads per block.
 * @param[out] shared_out Dynamic shared-memory bytes per block.
 * @return ::MR_SUCCESS, ::MR_ERR_CUDA or ::MR_ERR_UNSUPPORTED when even a
 *         single warp cannot be staged.
 */
static int mr_pick_blockdiag_launch(int L, int *block_out, size_t *shared_out)
{
    int dev = 0;
    MR_CUDA_CHECK(cudaGetDevice(&dev));

    int smem_default = 0, smem_optin = 0;
    MR_CUDA_CHECK(cudaDeviceGetAttribute(
        &smem_default, cudaDevAttrMaxSharedMemoryPerBlock, dev));
    MR_CUDA_CHECK(cudaDeviceGetAttribute(
        &smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    const size_t limit =
        (size_t)((smem_optin > smem_default) ? smem_optin : smem_default);

    const size_t per_thread = (size_t)L * sizeof(mr_complex);
    int block = MR_BLOCK;
    while (block > 32 && (size_t)block * per_thread > limit) block /= 2;

    const size_t needed = (size_t)block * per_thread;
    if (needed > limit)
        MR_FAIL(MR_ERR_UNSUPPORTED,
                "L = %d needs %zu bytes of shared memory for a single warp but "
                "the device offers at most %zu",
                L, needed, limit);

    if (needed > (size_t)smem_default) {
        /* Opt in to the larger dynamic shared-memory carve-out. */
        MR_CUDA_CHECK(cudaFuncSetAttribute(
            mr_k_blockdiag_dense, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)needed));
        MR_CUDA_CHECK(cudaFuncSetAttribute(
            mr_k_blockdiag_masked, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)needed));
    }

    *block_out  = block;
    *shared_out = needed;
    return MR_SUCCESS;
}

/* ------------------------------------------------------------------------ */
/* Plan life cycle                                                          */
/* ------------------------------------------------------------------------ */

/**
 * @brief Read a non-negative integer from the environment.
 * @param[in] name Variable name.
 * @param[in] def  Value returned when the variable is unset or empty.
 * @return The parsed value, or @p def.
 */
static int mr_env_int(const char *name, int def)
{
    const char *e = std::getenv(name);
    return (e && e[0]) ? atoi(e) : def;
}

/** @brief Release every resource of a (possibly partially built) plan's
 *         coil-group buffers, leaving the struct itself alive. */
static void mr_aha_plan_release(mr_aha_plan p)
{
    if (p->fft) cufftDestroy(p->fft);
    if (p->fft_rem) cufftDestroy(p->fft_rem);
    if (p->d_work) cudaFree(p->d_work);
    if (p->d_pad) cudaFree(p->d_pad);
    if (p->d_compact) cudaFree(p->d_compact);
    if (p->d_inv) cudaFree(p->d_inv);
    p->fft = p->fft_rem = 0;
    p->d_work = NULL;
    p->d_pad = p->d_compact = NULL;
    p->d_inv = NULL;
}

/**
 * @brief Make one batched cuFFT plan over @p nvol padded volumes without
 *        allocating its work area (the caller provides a shared one).
 *
 * The 64-bit planning interface is used because `group * L * Mx*My*Mz` may
 * exceed @f$2^{31}@f$ elements.
 *
 * @param[in]  p      Plan whose geometry and stream to use.
 * @param[in]  nvol   Number of volumes in the batch.
 * @param[out] h      Receives the cuFFT handle.
 * @param[out] work   Receives the required work-area bytes.
 * @return ::MR_SUCCESS or ::MR_ERR_CUFFT (with @p h destroyed and zeroed).
 */
static int mr_make_fft(mr_aha_plan p, int nvol, cufftHandle *h, size_t *work)
{
    /* cuFFT expects extents slowest-varying first; our arrays are column-major
     * [Mx, My, Mz], i.e. row-major [Mz, My, Mx]. */
    const mr_config *cfg = &p->cfg;
    const int rank = cfg->n_spatial;
    long long n[3];
    if (rank == 3) {
        n[0] = cfg->phi2_dim[2];
        n[1] = cfg->phi2_dim[1];
        n[2] = cfg->phi2_dim[0];
    } else {
        n[0] = cfg->phi2_dim[1];
        n[1] = cfg->phi2_dim[0];
    }
    const long long pad = p->dims.pad_size;

    *h = 0;
    cufftResult fr = cufftCreate(h);
    if (fr == CUFFT_SUCCESS) fr = cufftSetAutoAllocation(*h, 0);
    if (fr == CUFFT_SUCCESS)
        fr = cufftMakePlanMany64(*h, rank, n, NULL, 1, pad, NULL, 1, pad,
                                 CUFFT_C2C, (long long)nvol, work);
    if (fr == CUFFT_SUCCESS && p->stream != NULL)
        fr = cufftSetStream(*h, p->stream);
    if (fr != CUFFT_SUCCESS) {
        if (*h) cufftDestroy(*h);
        *h = 0;
        MR_FAIL(MR_ERR_CUFFT, "cuFFT planning over %d volumes failed with code %d",
                nvol, (int)fr);
    }
    return MR_SUCCESS;
}

/**
 * @brief Build the cuFFT plans and buffers for coil groups of size @p group.
 *
 * Unless @p force is set, gives up with ::MR_ERR_ALLOC (without allocating)
 * when the padded grids, the masked scratch and the cuFFT work area would
 * exceed @p budget bytes, so that the caller can retry with a smaller group.
 *
 * @param[in,out] p      Plan; on success its group fields are filled.
 * @param[in]     group  Coils per group.
 * @param[in]     budget Byte budget for this plan's buffers.
 * @param[in]     force  Non-zero to skip the budget check (group of 1).
 * @return ::MR_SUCCESS, ::MR_ERR_ALLOC or ::MR_ERR_CUFFT. On failure every
 *         group resource is released again.
 */
static int mr_aha_plan_build_group(mr_aha_plan p, int group, size_t budget,
                                   int force)
{
    const mr_config *cfg = &p->cfg;
    const size_t c8 = sizeof(mr_complex);
    const size_t pad_bytes = (size_t)p->dims.pad_size * cfg->L * group * c8;
    /* Masked: the generic kernels (one coil per group) need a compact
     * [num_masked, L] buffer, the specialised ones a [Mx,My,Mz] int32 map. */
    const size_t cmp_bytes = (cfg->is_masked && !mr_bd_available(p->bd, true))
                                 ? (size_t)cfg->num_masked * cfg->L * c8 : 0;
    const size_t inv_bytes = (cfg->is_masked && mr_bd_available(p->bd, true))
                                 ? (size_t)p->dims.pad_size * sizeof(int) : 0;

    size_t work = 0, work_rem = 0;
    int rc = mr_make_fft(p, cfg->L * group, &p->fft, &work);
    if (rc != MR_SUCCESS) return rc;
    const int rem = cfg->n_coils % group;
    if (rem != 0) {
        rc = mr_make_fft(p, cfg->L * rem, &p->fft_rem, &work_rem);
        if (rc != MR_SUCCESS) { mr_aha_plan_release(p); return rc; }
    }
    const size_t work_bytes = (work > work_rem) ? work : work_rem;

    const size_t total = pad_bytes + cmp_bytes + inv_bytes + work_bytes;
    if (!force && total > budget) {
        mr_aha_plan_release(p);
        MR_FAIL(MR_ERR_ALLOC, "a group of %d coils needs %zu bytes, budget %zu",
                group, total, budget);
    }

    cudaError_t ce = cudaMalloc((void **)&p->d_pad, pad_bytes);
    if (ce == cudaSuccess && cmp_bytes)
        ce = cudaMalloc((void **)&p->d_compact, cmp_bytes);
    if (ce == cudaSuccess && inv_bytes)
        ce = cudaMalloc((void **)&p->d_inv, inv_bytes);
    if (ce == cudaSuccess && work_bytes) ce = cudaMalloc(&p->d_work, work_bytes);
    if (ce != cudaSuccess) {
        cudaGetLastError(); /* clear the sticky allocation error */
        mr_aha_plan_release(p);
        MR_FAIL(MR_ERR_ALLOC,
                "failed to allocate %zu bytes of buffers for a group of %d "
                "coil(s): %s",
                total, group, cudaGetErrorString(ce));
    }

    cufftResult fr = cufftSetWorkArea(p->fft, p->d_work);
    if (fr == CUFFT_SUCCESS && p->fft_rem)
        fr = cufftSetWorkArea(p->fft_rem, p->d_work);
    if (fr != CUFFT_SUCCESS) {
        mr_aha_plan_release(p);
        MR_FAIL(MR_ERR_CUFFT, "cufftSetWorkArea failed with code %d", (int)fr);
    }
    p->group = group;
    return MR_SUCCESS;
}

/** @brief Body of ::mr_aha_plan_create, run with the target device current. */
static int mr_aha_plan_create_impl(const mr_config *cfg, mr_aha_plan *plan_out)
{
    const int status = mr_config_validate(cfg);
    if (status != MR_SUCCESS) return status;

    mr_aha_plan p = (mr_aha_plan)calloc(1, sizeof(struct mr_aha_plan_s));
    if (p == NULL) MR_FAIL(MR_ERR_ALLOC, "failed to allocate the plan struct");

    p->cfg    = *cfg;
    p->stream = (cudaStream_t)cfg->stream;
    mr_dims_from_config(cfg, &p->dims);

    /* Block-diagonal kernels: specialised on L (register vectors, each stored
     * element read once, coil groups) for L <= MR_BD_MAX_L, tiled (shared-
     * memory panels, same guarantees) up to MR_BDT_MAX_L; the generic
     * shared-memory kernels (one coil at a time) otherwise or when
     * MR_AHA_CUDA_GENERIC_BD=1. */
    if (mr_env_int("MR_AHA_CUDA_GENERIC_BD", 0) == 0) p->bd = mr_bd_lookup(cfg->L);
    if (cfg->is_masked && cfg->num_masked > 2147483647LL)
        p->bd = mr_bd_kernels{}; /* the int32 index map is too small */
    int rc;
    if (mr_bd_available(p->bd, false)) {
        int t = mr_env_int("MR_AHA_CUDA_BD_THREADS", MR_BD_THREADS);
        if (t < 32 || t > MR_BD_THREADS) t = MR_BD_THREADS;
        p->bd_block  = t;
        p->bd_shared = 0;
    } else {
        rc = mr_pick_blockdiag_launch(cfg->L, &p->bd_block, &p->bd_shared);
        if (rc != MR_SUCCESS) { free(p); return rc; }
    }

    /* Coil group size: requested (default or MR_AHA_CUDA_COIL_GROUP), at most
     * the number of coils, 1 for the generic kernels; then reduced until the
     * buffers fit mem_fraction of the free memory. A group of 1 (the plain
     * coil-by-coil operator) is always attempted. */
    int group = mr_env_int("MR_AHA_CUDA_COIL_GROUP", MR_CUDA_COIL_GROUP);
    if (!mr_bd_available(p->bd, false) || group < 1) group = 1;
    if (group > cfg->n_coils) group = cfg->n_coils;

    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t ce = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (ce != cudaSuccess) {
        free(p);
        MR_FAIL(MR_ERR_CUDA, "cudaMemGetInfo failed: %s", cudaGetErrorString(ce));
    }
    double frac = cfg->mem_fraction;
    if (!(frac > 0.0)) frac = 0.5;
    if (frac > 0.95) frac = 0.95;
    const size_t budget = (size_t)(frac * (double)free_bytes);

    /* cuFFT planning of a large batch can also fail for lack of memory
     * (reported as a cuFFT error), so both errors fall back to smaller
     * groups. */
    for (;; --group) {
        rc = mr_aha_plan_build_group(p, group, budget, group == 1);
        if (rc == MR_SUCCESS || group == 1 ||
            (rc != MR_ERR_ALLOC && rc != MR_ERR_CUFFT))
            break;
    }
    if (rc != MR_SUCCESS) {
        free(p);
        return rc;
    }
    mr_clear_error(); /* drop the diagnostics of the larger groups tried */

    if (cfg->verbose)
        printf("[mr_aha] plan: %s block-diagonal kernels, %d coil(s) per group\n",
               p->bd.tiled ? "tiled" : p->bd.dense ? "specialised" : "generic",
               p->group);

    *plan_out = p;
    return MR_SUCCESS;
}

extern "C" int mr_aha_plan_create(const mr_config *cfg, mr_aha_plan *plan_out)
{
    mr_clear_error();
    MR_REQUIRE(plan_out != NULL, "mr_aha_plan_create: plan_out is NULL");
    *plan_out = NULL;
    MR_REQUIRE(cfg != NULL, "mr_aha_plan_create: cfg is NULL");

    int prev = -1;
    int rc = mr_push_device(cfg, &prev);
    if (rc == MR_SUCCESS) rc = mr_aha_plan_create_impl(cfg, plan_out);
    mr_pop_device(prev);
    return rc;
}

int mr_aha_plan_group(mr_aha_plan plan) { return plan->group; }

void mr_aha_plan_forget(mr_aha_plan plan) { free(plan); }

extern "C" int mr_aha_plan_destroy(mr_aha_plan plan)
{
    mr_clear_error();
    if (plan == NULL) return MR_SUCCESS;

    int prev = -1;
    (void)mr_push_device(&plan->cfg, &prev);
    mr_aha_plan_release(plan);
    mr_pop_device(prev);

    free(plan);
    return MR_SUCCESS;
}

/* ------------------------------------------------------------------------ */
/* Optional per-stage profiling                                             */
/* ------------------------------------------------------------------------ */

/** @brief Number of profiled stages of one coil (group), see ::mr_stage_names. */
#define MR_N_STAGES 5

/** @brief Names of the profiled stages, in execution order. */
static const char *const mr_stage_names[MR_N_STAGES] = {
    "pad * S_c", "forward FFT", "block-diagonal", "inverse FFT",
    "crop * conj(S_c)"};

/**
 * @brief Optional per-stage GPU timing (`MR_AHA_CUDA_PROFILE=1`).
 *
 * When enabled, a CUDA event is recorded on the plan's stream after every
 * stage of every coil (group); ::mr_profile_end waits for the stream, sums the
 * elapsed times between consecutive events per stage and prints them to
 * `stderr`. Events do not serialise the stream, so the profiled run takes the
 * same time as a normal one. Disabled (the default), every call is a no-op.
 */
struct mr_profile {
    bool                     on = false;   /**< Profiling requested.          */
    cudaStream_t             stream = 0;   /**< Stream the stages run on.     */
    int                      next = 0;     /**< Stage of the next mark.       */
    std::vector<cudaEvent_t> ev;           /**< Start event, then one per mark.*/
    std::vector<int>         stage;        /**< Stage index of `ev[i + 1]`.   */

    ~mr_profile()
    {
        for (cudaEvent_t e : ev) cudaEventDestroy(e);
    }
};

/**
 * @brief Enable @p prof if `MR_AHA_CUDA_PROFILE` is set, recording the start
 *        event on @p s.
 * @return `cudaSuccess` or the first CUDA error.
 */
static cudaError_t mr_profile_begin(mr_profile *prof, cudaStream_t s)
{
    const char *e = std::getenv("MR_AHA_CUDA_PROFILE");
    prof->on = e && e[0] && e[0] != '0';
    if (!prof->on) return cudaSuccess;
    prof->stream = s;
    cudaEvent_t ev;
    cudaError_t ce = cudaEventCreate(&ev);
    if (ce != cudaSuccess) return ce;
    prof->ev.push_back(ev);
    return cudaEventRecord(ev, s);
}

/**
 * @brief Mark the end of stage @p k (which started at the previous mark).
 * @return `cudaSuccess` or the first CUDA error.
 */
static cudaError_t mr_profile_mark(mr_profile *prof, int k)
{
    if (!prof->on) return cudaSuccess;
    cudaEvent_t ev;
    cudaError_t ce = cudaEventCreate(&ev);
    if (ce != cudaSuccess) return ce;
    prof->ev.push_back(ev);
    prof->stage.push_back(k);
    return cudaEventRecord(ev, prof->stream);
}

/**
 * @brief Wait for the profiled work and print the per-stage totals.
 * @param[in] prof  Profile filled by ::mr_profile_mark.
 * @param[in] cfg   Configuration (for the header line).
 * @param[in] group Coils per group (for the header line).
 * @return `cudaSuccess` or the first CUDA error.
 */
static cudaError_t mr_profile_end(mr_profile *prof, const mr_config *cfg,
                                  int group)
{
    if (!prof->on) return cudaSuccess;
    cudaError_t ce = cudaEventSynchronize(prof->ev.back());
    if (ce != cudaSuccess) return ce;
    double ms[MR_N_STAGES] = {0};
    for (size_t i = 0; i < prof->stage.size(); ++i) {
        float t = 0.0f;
        ce = cudaEventElapsedTime(&t, prof->ev[i], prof->ev[i + 1]);
        if (ce != cudaSuccess) return ce;
        ms[prof->stage[i]] += t;
    }
    double total = 0.0;
    for (int k = 0; k < MR_N_STAGES; ++k) total += ms[k];
    fprintf(stderr, "[mr_aha] CUDA profile, %d coil(s) in groups of %d, %s:\n",
            cfg->n_coils, group, cfg->is_masked ? "masked" : "unmasked");
    for (int k = 0; k < MR_N_STAGES; ++k)
        fprintf(stderr, "[mr_aha]   %-18s %9.2f ms  (%4.1f%%)\n",
                mr_stage_names[k], ms[k], total > 0 ? 100.0 * ms[k] / total : 0.0);
    fprintf(stderr, "[mr_aha]   %-18s %9.2f ms\n", "total GPU", total);
    return cudaSuccess;
}

/* ------------------------------------------------------------------------ */
/* Execution                                                                */
/* ------------------------------------------------------------------------ */

/**
 * @brief Block-diagonal stage of one coil group (step 3 of the operator).
 *
 * @param[in] plan Plan.
 * @param[in] nc   Coils in this group; their grids fill the first @p nc slots
 *                 of the padded buffer.
 * @param[in] d_phi2_toep Packed blocks.
 * @param[in] d_mask      Mask, or NULL when unmasked.
 * @return ::MR_SUCCESS or ::MR_ERR_CUDA.
 */
static int mr_aha_blockdiag(mr_aha_plan plan, int nc,
                            const mr_complex *d_phi2_toep, const int64_t *d_mask)
{
    const mr_config *cfg = &plan->cfg;
    const mr_dims    dims = plan->dims;
    cudaStream_t     s = plan->stream;
    const long long  slot = dims.pad_size * dims.L;
    const long long  nm = (long long)cfg->num_masked;

    if (!cfg->is_masked) {
        if (mr_bd_available(plan->bd, false))
            return mr_bd_launch_dense(plan->bd, plan->bd_block, plan->d_pad,
                                      d_phi2_toep, dims, nc, mr_bd_whole(dims), s);
        /* Generic kernels: one coil per group. */
        const int grid = (int)mr_ceil_div(dims.pad_size, plan->bd_block);
        mr_k_blockdiag_dense<<<grid, plan->bd_block, plan->bd_shared, s>>>(
            plan->d_pad, d_phi2_toep, dims);
        MR_KERNEL_CHECK("mr_k_blockdiag_dense");
        return MR_SUCCESS;
    }

    if (mr_bd_available(plan->bd, true))
        /* In place over the whole grid, zeroing outside the mask. */
        return mr_bd_launch_masked(plan->bd, plan->bd_block, plan->d_pad, d_phi2_toep,
                                   plan->d_inv, nm, dims, nc, mr_bd_whole(dims), s);

    /* Generic kernels (one coil per group): gather into the compact buffer,
     * then zero the grid (samples outside the mask contribute nothing) and
     * scatter back. */
    const int grid = (int)mr_ceil_div(nm, plan->bd_block);
    mr_k_blockdiag_masked<<<grid, plan->bd_block, plan->bd_shared, s>>>(
        plan->d_compact, plan->d_pad, d_phi2_toep, d_mask, nm, dims);
    MR_KERNEL_CHECK("mr_k_blockdiag_masked");
    MR_CUDA_CHECK(cudaMemsetAsync(plan->d_pad, 0,
                                  (size_t)slot * sizeof(mr_complex), s));
    mr_k_scatter_masked<<<(int)mr_ceil_div(nm, MR_BLOCK), MR_BLOCK, 0, s>>>(
        plan->d_pad, plan->d_compact, d_mask, nm, dims);
    MR_KERNEL_CHECK("mr_k_scatter_masked");
    return MR_SUCCESS;
}

/** @brief Body of ::mr_aha_plan_execute, run with the target device current. */
static int mr_aha_execute_impl(mr_aha_plan plan, const mr_complex *d_x,
                               const mr_complex *d_SEs,
                               const mr_complex *d_phi2_toep,
                               const int64_t *d_mask, mr_complex *d_AhAx)
{
    const mr_config *cfg = &plan->cfg;
    const mr_dims    dims = plan->dims;
    cudaStream_t     s = plan->stream;

    const long long slot = dims.pad_size * dims.L;
    const int grid_pad   = (int)mr_ceil_div(dims.pad_size, MR_BLOCK);
    const int grid_image = (int)mr_ceil_div(dims.image_size, MR_BLOCK);

    mr_profile prof;
    MR_CUDA_CHECK(mr_profile_begin(&prof, s));

    if (plan->d_inv) {
        /* Grid-to-mask index map of the specialised masked kernels; rebuilt
         * on every call (a cheap pass over the mask) since the caller may
         * pass a different mask each time. Counted as part of stage 0. */
        const long long nm = (long long)cfg->num_masked;
        MR_CUDA_CHECK(cudaMemsetAsync(plan->d_inv, 0xFF,
                                      (size_t)dims.pad_size * sizeof(int), s));
        mr_k_mask_inverse<<<(int)mr_ceil_div(nm, MR_BLOCK), MR_BLOCK, 0, s>>>(
            plan->d_inv, d_mask, nm, dims);
        MR_KERNEL_CHECK("mr_k_mask_inverse");
    }

    for (int coil0 = 0; coil0 < cfg->n_coils; coil0 += plan->group) {
        const int nc = (cfg->n_coils - coil0 < plan->group)
                           ? cfg->n_coils - coil0 : plan->group;
        cufftHandle fft = (nc == plan->group) ? plan->fft : plan->fft_rem;

        /* 1. Zero-pad and apply each coil's sensitivity into its slot, with
         *    the centring shift folded into the scatter (this also zeroes the
         *    rest of the grid). */
        for (int c = 0; c < nc; ++c) {
            mr_k_pad_mul_se<<<grid_pad, MR_BLOCK, 0, s>>>(plan->d_pad + slot * c,
                                                          d_x, d_SEs, coil0 + c,
                                                          dims);
            MR_KERNEL_CHECK("mr_k_pad_mul_se");
        }
        MR_CUDA_CHECK(mr_profile_mark(&prof, 0));

        /* 2. Forward FFT of all ranks of all coils of the group onto the
         *    upsampled Cartesian k-space grid. */
        MR_CUFFT_CHECK(cufftExecC2C(fft, plan->d_pad, plan->d_pad, CUFFT_FORWARD));
        MR_CUDA_CHECK(mr_profile_mark(&prof, 1));

        /* 3. Hermitian block-diagonal multiplication at every retained sample,
         *    for the whole group. */
        int rc = mr_aha_blockdiag(plan, nc, d_phi2_toep, d_mask);
        if (rc != MR_SUCCESS) return rc;
        MR_CUDA_CHECK(mr_profile_mark(&prof, 2));

        /* 4. Inverse FFT back to the padded image domain. */
        MR_CUFFT_CHECK(cufftExecC2C(fft, plan->d_pad, plan->d_pad, CUFFT_INVERSE));
        MR_CUDA_CHECK(mr_profile_mark(&prof, 3));

        /* 5. Crop, apply the adjoint sensitivity and accumulate over coils.
         *    The first coil overwrites, which spares an extra memset. */
        for (int c = 0; c < nc; ++c) {
            const int coil = coil0 + c;
            mr_k_crop_mul_conj_se<<<grid_image, MR_BLOCK, 0, s>>>(
                d_AhAx, plan->d_pad + slot * c, d_SEs, coil, coil > 0, dims);
            MR_KERNEL_CHECK("mr_k_crop_mul_conj_se");
        }
        MR_CUDA_CHECK(mr_profile_mark(&prof, 4));
    }
    MR_CUDA_CHECK(mr_profile_end(&prof, cfg, plan->group));

    if (cfg->verbose) {
        MR_CUDA_CHECK(cudaStreamSynchronize(s));
        printf("[mr_aha] AhA applied over %d coil(s) in groups of %d, grid "
               "[%d %d %d] -> [%d %d %d], L = %d%s\n",
               cfg->n_coils, plan->group, cfg->n_dim[0], cfg->n_dim[1],
               cfg->n_dim[2], cfg->phi2_dim[0], cfg->phi2_dim[1],
               cfg->phi2_dim[2], cfg->L, cfg->is_masked ? ", masked" : "");
    }
    return MR_SUCCESS;
}

extern "C" int mr_aha_plan_execute(mr_aha_plan plan, const mr_complex *d_x,
                                   const mr_complex *d_SEs,
                                   const mr_complex *d_phi2_toep,
                                   const int64_t *d_mask, mr_complex *d_AhAx)
{
    mr_clear_error();
    MR_REQUIRE(plan != NULL, "mr_aha_plan_execute: plan is NULL");
    MR_REQUIRE(d_x != NULL, "mr_aha_plan_execute: d_x is NULL");
    MR_REQUIRE(d_SEs != NULL, "mr_aha_plan_execute: d_SEs is NULL");
    MR_REQUIRE(d_phi2_toep != NULL, "mr_aha_plan_execute: d_phi2_toep is NULL");
    MR_REQUIRE(d_AhAx != NULL, "mr_aha_plan_execute: d_AhAx is NULL");
    MR_REQUIRE(d_AhAx != d_x, "mr_aha_plan_execute: d_AhAx must not alias d_x");
    if (plan->cfg.is_masked)
        MR_REQUIRE(d_mask != NULL,
                   "mr_aha_plan_execute: d_mask is required when is_masked is set");

    int prev = -1;
    int rc = mr_push_device(&plan->cfg, &prev);
    if (rc == MR_SUCCESS)
        rc = mr_aha_execute_impl(plan, d_x, d_SEs, d_phi2_toep, d_mask, d_AhAx);
    mr_pop_device(prev);
    return rc;
}

extern "C" int mr_aha_toeplitz(const mr_config *cfg, const mr_complex *d_x,
                               const mr_complex *d_SEs,
                               const mr_complex *d_phi2_toep,
                               const int64_t *d_mask, mr_complex *d_AhAx)
{
    mr_aha_plan plan = NULL;
    int rc = mr_aha_plan_create(cfg, &plan);
    if (rc != MR_SUCCESS) return rc;
    rc = mr_aha_plan_execute(plan, d_x, d_SEs, d_phi2_toep, d_mask, d_AhAx);
    const int rc_destroy = mr_aha_plan_destroy(plan);
    return (rc != MR_SUCCESS) ? rc : rc_destroy;
}
