/**
 * @file mt_aha.cu
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
 * ::mt_calc_toep, so no scaling happens in this file.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mt_kernels.cuh"

#include <cstdio>
#include <cstdlib>

/** @brief Default thread-block size for the elementwise (non-shared) kernels. */
#define MT_BLOCK 256

/**
 * @brief Internal state of a reusable @f$A^H A@f$ plan.
 *
 * Holds the cuFFT plan and every persistent device buffer, so that repeated
 * ::mt_aha_plan_execute calls inside a conjugate-gradient loop allocate
 * nothing.
 */
struct mt_aha_plan_s {
    mt_config    cfg;        /**< Copy of the caller's configuration.         */
    mt_dims      dims;       /**< Kernel geometry descriptor.                 */
    cudaStream_t stream;     /**< Stream every operation is issued on.        */
    cufftHandle  fft;        /**< Batched cuFFT plan over the upsampled grid. */
    mt_complex  *d_pad;      /**< Padded working buffer `[Mx,My,Mz,L]`.       */
    mt_complex  *d_compact;  /**< `[num_masked, L]` scratch; masked mode only.*/
    int          bd_block;   /**< Threads per block for the block-diag kernel.*/
    size_t       bd_shared;  /**< Dynamic shared bytes for the same kernel.   */
};

/* ------------------------------------------------------------------------ */
/* Small helpers                                                            */
/* ------------------------------------------------------------------------ */

/**
 * @brief Make @p cfg's device current, remembering the caller's device.
 *
 * A ::mt_config::gpu_device_id of -1 means "leave the current device alone",
 * which is what MATLAB gpuArray and CuPy callers want.
 *
 * @param[in]  cfg  Configuration carrying the requested device.
 * @param[out] prev Receives the previously current device, or -1 if unchanged.
 * @return ::MT_SUCCESS or ::MT_ERR_CUDA.
 */
static int mt_push_device(const mt_config *cfg, int *prev)
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

/** @brief Restore the device saved by ::mt_push_device. Errors are ignored. */
static void mt_pop_device(int prev)
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
 * @return ::MT_SUCCESS, ::MT_ERR_CUDA or ::MT_ERR_UNSUPPORTED when even a
 *         single warp cannot be staged.
 */
static int mt_pick_blockdiag_launch(int L, int *block_out, size_t *shared_out)
{
    int dev = 0;
    MT_CUDA_CHECK(cudaGetDevice(&dev));

    int smem_default = 0, smem_optin = 0;
    MT_CUDA_CHECK(cudaDeviceGetAttribute(
        &smem_default, cudaDevAttrMaxSharedMemoryPerBlock, dev));
    MT_CUDA_CHECK(cudaDeviceGetAttribute(
        &smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    const size_t limit =
        (size_t)((smem_optin > smem_default) ? smem_optin : smem_default);

    const size_t per_thread = (size_t)L * sizeof(mt_complex);
    int block = MT_BLOCK;
    while (block > 32 && (size_t)block * per_thread > limit) block /= 2;

    const size_t needed = (size_t)block * per_thread;
    if (needed > limit)
        MT_FAIL(MT_ERR_UNSUPPORTED,
                "L = %d needs %zu bytes of shared memory for a single warp but "
                "the device offers at most %zu",
                L, needed, limit);

    if (needed > (size_t)smem_default) {
        /* Opt in to the larger dynamic shared-memory carve-out. */
        MT_CUDA_CHECK(cudaFuncSetAttribute(
            mt_k_blockdiag_dense, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)needed));
        MT_CUDA_CHECK(cudaFuncSetAttribute(
            mt_k_blockdiag_masked, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)needed));
    }

    *block_out  = block;
    *shared_out = needed;
    return MT_SUCCESS;
}

/* ------------------------------------------------------------------------ */
/* Plan life cycle                                                          */
/* ------------------------------------------------------------------------ */

/** @brief Body of ::mt_aha_plan_create, run with the target device current. */
static int mt_aha_plan_create_impl(const mt_config *cfg, mt_aha_plan *plan_out)
{
    const int status = mt_config_validate(cfg);
    if (status != MT_SUCCESS) return status;

    mt_aha_plan p = (mt_aha_plan)calloc(1, sizeof(struct mt_aha_plan_s));
    if (p == NULL) MT_FAIL(MT_ERR_ALLOC, "failed to allocate the plan struct");

    p->cfg    = *cfg;
    p->stream = (cudaStream_t)cfg->stream;
    p->fft    = 0;
    mt_dims_from_config(cfg, &p->dims);

    int rc = mt_pick_blockdiag_launch(cfg->L, &p->bd_block, &p->bd_shared);
    if (rc != MT_SUCCESS) { free(p); return rc; }

    /* cuFFT expects extents slowest-varying first; our arrays are column-major
     * [Mx, My, Mz], i.e. row-major [Mz, My, Mx]. */
    const int rank = cfg->n_spatial;
    int n[3];
    if (rank == 3) {
        n[0] = cfg->phi2_dim[2];
        n[1] = cfg->phi2_dim[1];
        n[2] = cfg->phi2_dim[0];
    } else {
        n[0] = cfg->phi2_dim[1];
        n[1] = cfg->phi2_dim[0];
    }
    const long long pad = p->dims.pad_size;

    cufftResult fr = cufftPlanMany(&p->fft, rank, n, NULL, 1, (int)pad, NULL, 1,
                                   (int)pad, CUFFT_C2C, cfg->L);
    if (fr != CUFFT_SUCCESS) {
        free(p);
        MT_FAIL(MT_ERR_CUFFT, "cufftPlanMany failed with code %d", (int)fr);
    }
    if (p->stream != NULL) {
        fr = cufftSetStream(p->fft, p->stream);
        if (fr != CUFFT_SUCCESS) {
            cufftDestroy(p->fft);
            free(p);
            MT_FAIL(MT_ERR_CUFFT, "cufftSetStream failed with code %d", (int)fr);
        }
    }

    cudaError_t ce =
        cudaMalloc((void **)&p->d_pad, (size_t)pad * cfg->L * sizeof(mt_complex));
    if (ce != cudaSuccess) {
        cufftDestroy(p->fft);
        free(p);
        MT_FAIL(MT_ERR_ALLOC, "failed to allocate the %lld-element padded buffer: %s",
                (long long)pad * cfg->L, cudaGetErrorString(ce));
    }

    if (cfg->is_masked) {
        ce = cudaMalloc((void **)&p->d_compact,
                        (size_t)cfg->num_masked * cfg->L * sizeof(mt_complex));
        if (ce != cudaSuccess) {
            cudaFree(p->d_pad);
            cufftDestroy(p->fft);
            free(p);
            MT_FAIL(MT_ERR_ALLOC,
                    "failed to allocate the %lld-element masked scratch: %s",
                    (long long)cfg->num_masked * cfg->L, cudaGetErrorString(ce));
        }
    }

    *plan_out = p;
    return MT_SUCCESS;
}

extern "C" int mt_aha_plan_create(const mt_config *cfg, mt_aha_plan *plan_out)
{
    mt_clear_error();
    MT_REQUIRE(plan_out != NULL, "mt_aha_plan_create: plan_out is NULL");
    *plan_out = NULL;
    MT_REQUIRE(cfg != NULL, "mt_aha_plan_create: cfg is NULL");

    int prev = -1;
    int rc = mt_push_device(cfg, &prev);
    if (rc == MT_SUCCESS) rc = mt_aha_plan_create_impl(cfg, plan_out);
    mt_pop_device(prev);
    return rc;
}

extern "C" int mt_aha_plan_destroy(mt_aha_plan plan)
{
    mt_clear_error();
    if (plan == NULL) return MT_SUCCESS;

    int prev = -1;
    (void)mt_push_device(&plan->cfg, &prev);
    if (plan->fft) cufftDestroy(plan->fft);
    if (plan->d_pad) cudaFree(plan->d_pad);
    if (plan->d_compact) cudaFree(plan->d_compact);
    mt_pop_device(prev);

    free(plan);
    return MT_SUCCESS;
}

/* ------------------------------------------------------------------------ */
/* Execution                                                                */
/* ------------------------------------------------------------------------ */

/** @brief Body of ::mt_aha_plan_execute, run with the target device current. */
static int mt_aha_execute_impl(mt_aha_plan plan, const mt_complex *d_x,
                               const mt_complex *d_SEs,
                               const mt_complex *d_phi2_toep,
                               const int64_t *d_mask, mt_complex *d_AhAx)
{
    const mt_config *cfg = &plan->cfg;
    const mt_dims    dims = plan->dims;
    cudaStream_t     s = plan->stream;

    const long long pad_total = dims.pad_size * dims.L;
    const int grid_pad   = (int)mt_ceil_div(dims.pad_size, MT_BLOCK);
    const int grid_image = (int)mt_ceil_div(dims.image_size, MT_BLOCK);

    for (int coil = 0; coil < cfg->n_coils; ++coil) {
        /* 1. Zero-pad and apply the coil sensitivity, with the centring shift
         *    folded into the scatter (this also zeroes the rest of the grid). */
        mt_k_pad_mul_se<<<grid_pad, MT_BLOCK, 0, s>>>(plan->d_pad, d_x, d_SEs,
                                                      coil, dims);
        MT_KERNEL_CHECK("mt_k_pad_mul_se");

        /* 2. Forward FFT onto the upsampled Cartesian k-space grid. */
        MT_CUFFT_CHECK(cufftExecC2C(plan->fft, plan->d_pad, plan->d_pad,
                                    CUFFT_FORWARD));

        /* 3. Hermitian block-diagonal multiplication at every retained sample. */
        if (cfg->is_masked) {
            const int grid_mask =
                (int)mt_ceil_div((long long)cfg->num_masked, plan->bd_block);
            mt_k_blockdiag_masked<<<grid_mask, plan->bd_block, plan->bd_shared, s>>>(
                plan->d_compact, plan->d_pad, d_phi2_toep, d_mask,
                (long long)cfg->num_masked, dims);
            MT_KERNEL_CHECK("mt_k_blockdiag_masked");

            /* Samples outside the mask contribute nothing, hence the memset. */
            MT_CUDA_CHECK(cudaMemsetAsync(
                plan->d_pad, 0, (size_t)pad_total * sizeof(mt_complex), s));
            mt_k_scatter_masked<<<grid_mask, plan->bd_block, 0, s>>>(
                plan->d_pad, plan->d_compact, d_mask, (long long)cfg->num_masked,
                dims);
            MT_KERNEL_CHECK("mt_k_scatter_masked");
        } else {
            const int grid_bd =
                (int)mt_ceil_div(dims.pad_size, plan->bd_block);
            mt_k_blockdiag_dense<<<grid_bd, plan->bd_block, plan->bd_shared, s>>>(
                plan->d_pad, d_phi2_toep, dims);
            MT_KERNEL_CHECK("mt_k_blockdiag_dense");
        }

        /* 4. Inverse FFT back to the padded image domain. */
        MT_CUFFT_CHECK(cufftExecC2C(plan->fft, plan->d_pad, plan->d_pad,
                                    CUFFT_INVERSE));

        /* 5. Crop, apply the adjoint sensitivity and accumulate over coils.
         *    The first coil overwrites, which spares an extra memset. */
        mt_k_crop_mul_conj_se<<<grid_image, MT_BLOCK, 0, s>>>(
            d_AhAx, plan->d_pad, d_SEs, coil, coil > 0, dims);
        MT_KERNEL_CHECK("mt_k_crop_mul_conj_se");
    }

    if (cfg->verbose) {
        MT_CUDA_CHECK(cudaStreamSynchronize(s));
        printf("[mt_cuda] AhA applied over %d coil(s), grid [%d %d %d] -> "
               "[%d %d %d], L = %d%s\n",
               cfg->n_coils, cfg->n_dim[0], cfg->n_dim[1], cfg->n_dim[2],
               cfg->phi2_dim[0], cfg->phi2_dim[1], cfg->phi2_dim[2], cfg->L,
               cfg->is_masked ? ", masked" : "");
    }
    return MT_SUCCESS;
}

extern "C" int mt_aha_plan_execute(mt_aha_plan plan, const mt_complex *d_x,
                                   const mt_complex *d_SEs,
                                   const mt_complex *d_phi2_toep,
                                   const int64_t *d_mask, mt_complex *d_AhAx)
{
    mt_clear_error();
    MT_REQUIRE(plan != NULL, "mt_aha_plan_execute: plan is NULL");
    MT_REQUIRE(d_x != NULL, "mt_aha_plan_execute: d_x is NULL");
    MT_REQUIRE(d_SEs != NULL, "mt_aha_plan_execute: d_SEs is NULL");
    MT_REQUIRE(d_phi2_toep != NULL, "mt_aha_plan_execute: d_phi2_toep is NULL");
    MT_REQUIRE(d_AhAx != NULL, "mt_aha_plan_execute: d_AhAx is NULL");
    MT_REQUIRE(d_AhAx != d_x, "mt_aha_plan_execute: d_AhAx must not alias d_x");
    if (plan->cfg.is_masked)
        MT_REQUIRE(d_mask != NULL,
                   "mt_aha_plan_execute: d_mask is required when is_masked is set");

    int prev = -1;
    int rc = mt_push_device(&plan->cfg, &prev);
    if (rc == MT_SUCCESS)
        rc = mt_aha_execute_impl(plan, d_x, d_SEs, d_phi2_toep, d_mask, d_AhAx);
    mt_pop_device(prev);
    return rc;
}

extern "C" int mt_aha_toeplitz(const mt_config *cfg, const mt_complex *d_x,
                               const mt_complex *d_SEs,
                               const mt_complex *d_phi2_toep,
                               const int64_t *d_mask, mt_complex *d_AhAx)
{
    mt_aha_plan plan = NULL;
    int rc = mt_aha_plan_create(cfg, &plan);
    if (rc != MT_SUCCESS) return rc;
    rc = mt_aha_plan_execute(plan, d_x, d_SEs, d_phi2_toep, d_mask, d_AhAx);
    const int rc_destroy = mt_aha_plan_destroy(plan);
    return (rc != MT_SUCCESS) ? rc : rc_destroy;
}
