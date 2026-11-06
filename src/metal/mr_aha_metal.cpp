/**
 * @file mr_aha_metal.cpp
 * @brief Toeplitz-embedded forward normal operator AhA on Metal.
 *
 * Same algorithm and same kernels as `src/cuda/mr_aha.cu`, evaluated coil by
 * coil as
 * @f[
 *   A^H A x = \sum_c S_c^H \, \mathcal{P}^H \, F^{-1} \hat{T} F \,
 *             \mathcal{P} \, S_c \, x ,
 * @f]
 * with the zero-padding @f$\mathcal{P}@f$ (and its centring remap) and the
 * block-diagonal @f$\hat{T}@f$ in `mr_kernels.metal`, and @f$F@f$ the batched
 * unnormalised VkFFT transform over the upsampled grid. `phi2_toep` already
 * carries the @f$1/\prod M@f$ of the unnormalised FFT pair.
 *
 * Differences from the CUDA backend:
 *  - Arrays are host pointers in unified memory, wrapped in place as Metal
 *    buffers (see ::mr_metal_view).
 *  - Each coil is one command buffer (pad, FFT, block-diagonal, inverse FFT,
 *    crop), all committed back to back to the same queue, which runs them in
 *    order; the call waits for the last one. Keeping each command buffer short
 *    avoids the GPU watchdog on very large problems.
 *  - The calls are synchronous: on return @p d_AhAx holds the result.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_metal_internal.h"

#include <cstdio>
#include <new>
#include <vector>

static_assert(sizeof(mr_dims) == 80,
              "mr_dims must match the struct in mr_kernels.metal");

/** @brief Default threads per threadgroup for the block-diagonal kernels. */
#define MR_METAL_BD_THREADS 256

/** @brief Internal state of a reusable Metal @f$A^H A@f$ plan. */
struct mr_aha_plan_s {
    mr_config           cfg;      /**< Copy of the caller's configuration.     */
    mr_dims             dims;     /**< Kernel geometry descriptor.             */
    const mr_metal_ctx *ctx;      /**< Shared device and queue.                */
    mr_metal_fft       *fft;      /**< Batched FFT over the upsampled grid.    */
    MTL::Buffer        *pad;      /**< Padded working buffer `[Mx,My,Mz,L]`.   */
    MTL::Buffer        *compact;  /**< `[num_masked, L]`; masked mode only.    */
    MTL::ComputePipelineState *k_pad;     /**< mr_k_pad_mul_se.               */
    MTL::ComputePipelineState *k_crop;    /**< mr_k_crop_mul_conj_se.         */
    MTL::ComputePipelineState *k_bd;      /**< Dense or masked block-diagonal. */
    MTL::ComputePipelineState *k_scatter; /**< mr_k_scatter_masked (masked).  */
    MTL::ComputePipelineState *k_zero;    /**< mr_k_zero (masked).            */
    NS::UInteger        bd_threads; /**< Threads per threadgroup, block-diag.  */
    NS::UInteger        bd_tgmem;   /**< Threadgroup bytes, block-diag.        */
};

/* ------------------------------------------------------------------------ */
/* Plan life cycle                                                          */
/* ------------------------------------------------------------------------ */

/** @brief Release everything a (possibly half-built) plan holds. */
static void mr_plan_free(mr_aha_plan p)
{
    if (!p) return;
    mr_metal_fft_destroy(p->fft);
    if (p->pad) p->pad->release();
    if (p->compact) p->compact->release();
    delete p;
}

/**
 * @brief Choose the threadgroup size of a block-diagonal kernel.
 *
 * Each thread stages its length-@f$L@f$ rank vector in threadgroup memory, so
 * the threadgroup is halved from 256 until it fits the device's threadgroup
 * memory (32 KB on Apple GPUs), as the CUDA backend does with shared memory.
 */
static int mr_pick_bd(const mr_metal_ctx *ctx, const MTL::ComputePipelineState *pso,
                      int L, NS::UInteger *threads, NS::UInteger *tgmem)
{
    const NS::UInteger limit = ctx->device->maxThreadgroupMemoryLength();
    const NS::UInteger per_thread = (NS::UInteger)L * sizeof(mr_complex);
    NS::UInteger t = MR_METAL_BD_THREADS;
    if (t > pso->maxTotalThreadsPerThreadgroup())
        t = pso->maxTotalThreadsPerThreadgroup();
    while (t > 32 && t * per_thread > limit) t /= 2;
    const NS::UInteger need = (t * per_thread + 15) & ~(NS::UInteger)15;
    if (need > limit)
        MR_FAIL(MR_ERR_UNSUPPORTED,
                "L = %d needs %lu bytes of threadgroup memory for 32 threads but "
                "the device offers %lu", L, (unsigned long)need,
                (unsigned long)limit);
    *threads = t;
    *tgmem = need;
    return MR_SUCCESS;
}

extern "C" int mr_aha_plan_create(const mr_config *cfg, mr_aha_plan *plan_out)
{
    mr_clear_error();
    MR_REQUIRE(plan_out != NULL, "mr_aha_plan_create: plan_out is NULL");
    *plan_out = NULL;
    int rc = mr_config_validate(cfg);
    if (rc != MR_SUCCESS) return rc;

    mr_metal_pool pool;
    const mr_metal_ctx *ctx = nullptr;
    rc = mr_metal_context(&ctx);
    if (rc != MR_SUCCESS) return rc;

    mr_aha_plan p = new (std::nothrow) mr_aha_plan_s();
    if (!p) MR_FAIL(MR_ERR_ALLOC, "failed to allocate the plan struct");
    p->cfg = *cfg;
    p->ctx = ctx;
    mr_dims_from_config(cfg, &p->dims);

    /* Thread indices are 32-bit; see mr_kernels.metal. */
    const long long vmax = 0xFFFFFFFFLL;
    if (p->dims.pad_size > vmax || p->dims.image_size > vmax ||
        (cfg->is_masked && cfg->num_masked > vmax)) {
        mr_plan_free(p);
        MR_FAIL(MR_ERR_UNSUPPORTED, "the Metal backend supports volumes of "
                "fewer than 2^32 voxels (upsampled grid has %lld)",
                p->dims.pad_size);
    }

    /* Working buffers: GPU-only, so private storage. */
    const NS::UInteger pad_bytes =
        (NS::UInteger)p->dims.pad_size * cfg->L * sizeof(mr_complex);
    p->pad = ctx->device->newBuffer(pad_bytes, MTL::ResourceStorageModePrivate);
    if (!p->pad) {
        mr_plan_free(p);
        MR_FAIL(MR_ERR_ALLOC, "failed to allocate the %lu-byte padded buffer",
                (unsigned long)pad_bytes);
    }
    if (cfg->is_masked) {
        const NS::UInteger cb =
            (NS::UInteger)cfg->num_masked * cfg->L * sizeof(mr_complex);
        p->compact = ctx->device->newBuffer(cb, MTL::ResourceStorageModePrivate);
        if (!p->compact) {
            mr_plan_free(p);
            MR_FAIL(MR_ERR_ALLOC, "failed to allocate the %lu-byte masked buffer",
                    (unsigned long)cb);
        }
    }

    /* Pipelines. */
    if ((rc = mr_metal_pipeline("mr_k_pad_mul_se", &p->k_pad)) != MR_SUCCESS ||
        (rc = mr_metal_pipeline("mr_k_crop_mul_conj_se", &p->k_crop)) != MR_SUCCESS ||
        (rc = mr_metal_pipeline(cfg->is_masked ? "mr_k_blockdiag_masked"
                                               : "mr_k_blockdiag_dense",
                                &p->k_bd)) != MR_SUCCESS ||
        (cfg->is_masked &&
         ((rc = mr_metal_pipeline("mr_k_scatter_masked", &p->k_scatter)) != MR_SUCCESS ||
          (rc = mr_metal_pipeline("mr_k_zero", &p->k_zero)) != MR_SUCCESS)) ||
        (rc = mr_pick_bd(ctx, p->k_bd, cfg->L, &p->bd_threads, &p->bd_tgmem)) !=
            MR_SUCCESS) {
        mr_plan_free(p);
        return rc;
    }

    /* Batched FFT over the upsampled grid, one volume per rank. */
    rc = mr_metal_fft_create(ctx, cfg->n_spatial, cfg->phi2_dim, cfg->L, &p->fft);
    if (rc != MR_SUCCESS) {
        mr_plan_free(p);
        return rc;
    }

    *plan_out = p;
    return MR_SUCCESS;
}

extern "C" int mr_aha_plan_destroy(mr_aha_plan plan)
{
    mr_clear_error();
    mr_metal_pool pool;
    mr_plan_free(plan);
    return MR_SUCCESS;
}

/* ------------------------------------------------------------------------ */
/* Execution                                                                */
/* ------------------------------------------------------------------------ */

/**
 * @brief Encode the whole operator for one coil into @p enc.
 *
 * Buffer indices follow the kernel signatures in `mr_kernels.metal`.
 */
static int mr_encode_coil(mr_aha_plan p, MTL::CommandBuffer *cb,
                          MTL::ComputeCommandEncoder *enc, int coil,
                          const mr_metal_view &x, const mr_metal_view &se,
                          const mr_metal_view &toep, const mr_metal_view &mask,
                          const mr_metal_view &out)
{
    const mr_dims &d = p->dims;
    const int accumulate = coil > 0 ? 1 : 0;

    /* 1. Zero-pad and apply the coil sensitivity (also clears the grid). */
    enc->setComputePipelineState(p->k_pad);
    enc->setBuffer(p->pad, 0, 0);
    mr_metal_bind(enc, x, 1);
    mr_metal_bind(enc, se, 2);
    enc->setBytes(&d, sizeof(d), 3);
    enc->setBytes(&coil, sizeof(coil), 4);
    mr_metal_dispatch_1d(enc, p->k_pad, (uint64_t)d.pad_size);

    /* 2. Forward FFT onto the upsampled Cartesian k-space grid. */
    int rc = mr_metal_fft_encode(p->fft, cb, enc, p->pad, false);
    if (rc != MR_SUCCESS) return rc;

    /* 3. Hermitian block-diagonal multiplication. */
    if (p->cfg.is_masked) {
        const long long nm = (long long)p->cfg.num_masked;
        enc->setComputePipelineState(p->k_bd);
        enc->setBuffer(p->compact, 0, 0);
        enc->setBuffer(p->pad, 0, 1);
        mr_metal_bind(enc, toep, 2);
        mr_metal_bind(enc, mask, 3);
        enc->setBytes(&nm, sizeof(nm), 4);
        enc->setBytes(&d, sizeof(d), 5);
        enc->setThreadgroupMemoryLength(p->bd_tgmem, 0);
        mr_metal_dispatch_1d(enc, p->k_bd, (uint64_t)nm, p->bd_threads);

        /* Samples outside the mask contribute nothing. */
        const long long n = d.pad_size;
        const int count = d.L;
        enc->setComputePipelineState(p->k_zero);
        enc->setBuffer(p->pad, 0, 0);
        enc->setBytes(&n, sizeof(n), 1);
        enc->setBytes(&count, sizeof(count), 2);
        mr_metal_dispatch_1d(enc, p->k_zero, (uint64_t)n);

        enc->setComputePipelineState(p->k_scatter);
        enc->setBuffer(p->pad, 0, 0);
        enc->setBuffer(p->compact, 0, 1);
        mr_metal_bind(enc, mask, 2);
        enc->setBytes(&nm, sizeof(nm), 3);
        enc->setBytes(&d, sizeof(d), 4);
        mr_metal_dispatch_1d(enc, p->k_scatter, (uint64_t)nm);
    } else {
        enc->setComputePipelineState(p->k_bd);
        enc->setBuffer(p->pad, 0, 0);
        mr_metal_bind(enc, toep, 1);
        enc->setBytes(&d, sizeof(d), 2);
        enc->setThreadgroupMemoryLength(p->bd_tgmem, 0);
        mr_metal_dispatch_1d(enc, p->k_bd, (uint64_t)d.pad_size, p->bd_threads);
    }

    /* 4. Inverse FFT back to the padded image domain. */
    rc = mr_metal_fft_encode(p->fft, cb, enc, p->pad, true);
    if (rc != MR_SUCCESS) return rc;

    /* 5. Crop, apply the adjoint sensitivity and accumulate over coils. */
    enc->setComputePipelineState(p->k_crop);
    mr_metal_bind(enc, out, 0);
    enc->setBuffer(p->pad, 0, 1);
    mr_metal_bind(enc, se, 2);
    enc->setBytes(&d, sizeof(d), 3);
    enc->setBytes(&coil, sizeof(coil), 4);
    enc->setBytes(&accumulate, sizeof(accumulate), 5);
    mr_metal_dispatch_1d(enc, p->k_crop, (uint64_t)d.image_size);
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
    const mr_config *cfg = &plan->cfg;
    if (cfg->is_masked)
        MR_REQUIRE(d_mask != NULL,
                   "mr_aha_plan_execute: d_mask is required when is_masked is set");

    mr_metal_pool pool;
    const mr_metal_ctx *ctx = plan->ctx;
    const size_t c8 = sizeof(mr_complex);

    mr_metal_view x, se, toep, mask, out;
    int rc;
    if ((rc = mr_metal_view_make(ctx, d_x, (size_t)mr_x_size(cfg) * c8, false, &x)) ||
        (rc = mr_metal_view_make(ctx, d_SEs, (size_t)mr_se_size(cfg) * c8, false, &se)) ||
        (rc = mr_metal_view_make(ctx, d_phi2_toep,
                                 (size_t)mr_phi2_toep_size(cfg) * c8, false, &toep)) ||
        (rc = mr_metal_view_make(ctx, cfg->is_masked ? d_mask : nullptr,
                                 cfg->is_masked ? (size_t)cfg->num_masked * 8 : 0,
                                 false, &mask)) ||
        (rc = mr_metal_view_make(ctx, d_AhAx, (size_t)mr_x_size(cfg) * c8, true, &out)))
        return rc;

    /* One command buffer per coil, committed in order; wait for all. */
    std::vector<MTL::CommandBuffer *> cbs;
    cbs.reserve((size_t)cfg->n_coils);
    for (int coil = 0; coil < cfg->n_coils; ++coil) {
        MTL::CommandBuffer *cb = ctx->queue->commandBuffer();
        MTL::ComputeCommandEncoder *enc = cb->computeCommandEncoder();
        rc = mr_encode_coil(plan, cb, enc, coil, x, se, toep, mask, out);
        enc->endEncoding();
        if (rc != MR_SUCCESS) break;
        cb->commit();
        cbs.push_back(cb);
    }
    for (MTL::CommandBuffer *cb : cbs) cb->waitUntilCompleted();
    if (rc != MR_SUCCESS) return rc;
    for (MTL::CommandBuffer *cb : cbs)
        if (cb->status() != MTL::CommandBufferStatusCompleted)
            MR_FAIL(MR_ERR_METAL, "mr_aha_plan_execute: GPU command buffer "
                    "failed: %s",
                    cb->error() && cb->error()->localizedDescription()
                        ? cb->error()->localizedDescription()->utf8String()
                        : "(no error description)");
    mr_metal_view_finish(&out);

    if (cfg->verbose)
        printf("[mr_aha] AhA applied over %d coil(s), grid [%d %d %d] -> "
               "[%d %d %d], L = %d%s (Metal)\n",
               cfg->n_coils, cfg->n_dim[0], cfg->n_dim[1], cfg->n_dim[2],
               cfg->phi2_dim[0], cfg->phi2_dim[1], cfg->phi2_dim[2], cfg->L,
               cfg->is_masked ? ", masked" : "");
    return MR_SUCCESS;
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
