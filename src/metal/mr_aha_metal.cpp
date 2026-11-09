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
 *  - Coils are processed in groups of up to ::mr_aha_plan_s::group: the
 *    padded grids of a group sit side by side, one batched FFT transforms them
 *    all, and the block-diagonal kernel applies `phi2_toep` to every coil of
 *    the group in one pass. The block-diagonal stage is memory-bound on
 *    reading `phi2_toep`, and a group reads it from DRAM about once instead of
 *    once per coil (the GPU's caches serve the other coils), which removes a
 *    large share of the operator's memory traffic.
 *  - Each coil group is one command buffer (pad, FFT, block-diagonal, inverse
 *    FFT, crop), all committed back to back to the same queue, which runs them
 *    in order; the call waits for the last one. Keeping each command buffer
 *    short avoids the GPU watchdog on very large problems.
 *  - The calls are synchronous: on return @p d_AhAx holds the result.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_metal_internal.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <new>
#include <vector>

static_assert(sizeof(mr_dims) == 80,
              "mr_dims must match the struct in mr_kernels.metal");

/** @brief Default threads per threadgroup for the block-diagonal kernels. */
#define MR_METAL_BD_THREADS 256

/** @brief Largest L with a specialised block-diagonal kernel (see
 *         `mr_k_bd_dense<L>` in mr_kernels.metal). */
#define MR_METAL_BD_MAX_L 32

/** @brief Default number of coils per group (see the file comment). Measured
 *         on the M3 Ultra; `MR_AHA_METAL_COIL_GROUP` overrides it. */
#define MR_METAL_COIL_GROUP 4

/** @brief Internal state of a reusable Metal @f$A^H A@f$ plan. */
struct mr_aha_plan_s {
    mr_config           cfg;      /**< Copy of the caller's configuration.     */
    mr_dims             dims;     /**< Kernel geometry descriptor.             */
    const mr_metal_ctx *ctx;      /**< Shared device and queue.                */
    int                 group;    /**< Coils per group (>= 1).                 */
    mr_metal_fft       *fft;      /**< FFT over `group*L` volumes.             */
    mr_metal_fft       *fft_rem;  /**< FFT over the last, smaller group, or
                                       NULL when `n_coils % group == 0`.      */
    MTL::Buffer        *pad;      /**< Padded grids `[Mx,My,Mz,L,group]`.      */
    MTL::Buffer        *compact;  /**< `[num_masked,L,group]`; masked only.    */
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
    mr_metal_fft_destroy(p->fft_rem);
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

    /* Pipelines. The block-diagonal stage uses the kernels specialised on L
     * (register vectors, each stored element read once, coil groups) for
     * L <= 32, and the generic threadgroup-memory kernels (one coil at a time)
     * otherwise or when MR_AHA_METAL_GENERIC_BD=1 (kept for comparison). */
    const char *gen = std::getenv("MR_AHA_METAL_GENERIC_BD");
    const bool specialised =
        cfg->L <= MR_METAL_BD_MAX_L && !(gen && gen[0] && gen[0] != '0');
    char bd_name[64];
    if (specialised)
        snprintf(bd_name, sizeof(bd_name), "%s%d",
                 cfg->is_masked ? "mr_k_bd_masked_" : "mr_k_bd_dense_", cfg->L);
    else
        snprintf(bd_name, sizeof(bd_name), "%s",
                 cfg->is_masked ? "mr_k_blockdiag_masked" : "mr_k_blockdiag_dense");
    if ((rc = mr_metal_pipeline("mr_k_pad_mul_se", &p->k_pad)) != MR_SUCCESS ||
        (rc = mr_metal_pipeline("mr_k_crop_mul_conj_se", &p->k_crop)) != MR_SUCCESS ||
        (rc = mr_metal_pipeline(bd_name, &p->k_bd)) != MR_SUCCESS ||
        (cfg->is_masked &&
         ((rc = mr_metal_pipeline("mr_k_scatter_masked", &p->k_scatter)) != MR_SUCCESS ||
          (rc = mr_metal_pipeline("mr_k_zero", &p->k_zero)) != MR_SUCCESS))) {
        mr_plan_free(p);
        return rc;
    }
    if (specialised) {
        p->bd_threads = 0; /* default threadgroup size, no threadgroup memory */
        p->bd_tgmem = 0;
    } else if ((rc = mr_pick_bd(ctx, p->k_bd, cfg->L, &p->bd_threads,
                                &p->bd_tgmem)) != MR_SUCCESS) {
        mr_plan_free(p);
        return rc;
    }

    /* Coil group size: the default (or MR_AHA_METAL_COIL_GROUP), at most the
     * number of coils, a group's FFT must stay below 2^31 complex values
     * (mr_metal_fft.cpp), and its grids within a quarter of the GPU's
     * recommended working set. The generic kernels handle one coil. */
    const uint64_t per_coil = (uint64_t)p->dims.pad_size * (uint64_t)cfg->L;
    int group = MR_METAL_COIL_GROUP;
    {
        const char *e = std::getenv("MR_AHA_METAL_COIL_GROUP");
        if (e && e[0]) group = atoi(e);
    }
    if (!specialised || group < 1) group = 1;
    if (group > cfg->n_coils) group = cfg->n_coils;
    while (group > 1 && (per_coil * (uint64_t)group > (1ull << 31) - 1 ||
                         per_coil * (uint64_t)group * sizeof(mr_complex) >
                             ctx->device->recommendedMaxWorkingSetSize() / 4))
        --group;
    p->group = group;

    /* Working buffers: GPU-only, so private storage. */
    const NS::UInteger pad_bytes =
        (NS::UInteger)(per_coil * (uint64_t)group * sizeof(mr_complex));
    p->pad = ctx->device->newBuffer(pad_bytes, MTL::ResourceStorageModePrivate);
    if (!p->pad) {
        mr_plan_free(p);
        MR_FAIL(MR_ERR_ALLOC, "failed to allocate the %lu-byte padded buffer",
                (unsigned long)pad_bytes);
    }
    if (cfg->is_masked) {
        const NS::UInteger cb = (NS::UInteger)cfg->num_masked * cfg->L * group *
                                sizeof(mr_complex);
        p->compact = ctx->device->newBuffer(cb, MTL::ResourceStorageModePrivate);
        if (!p->compact) {
            mr_plan_free(p);
            MR_FAIL(MR_ERR_ALLOC, "failed to allocate the %lu-byte masked buffer",
                    (unsigned long)cb);
        }
    }

    /* Batched FFTs over the upsampled grid: one volume per rank and coil of a
     * group, plus one for a smaller last group. */
    rc = mr_metal_fft_create(ctx, cfg->n_spatial, cfg->phi2_dim, cfg->L * group,
                             &p->fft);
    const int rem = cfg->n_coils % group;
    if (rc == MR_SUCCESS && rem)
        rc = mr_metal_fft_create(ctx, cfg->n_spatial, cfg->phi2_dim, cfg->L * rem,
                                 &p->fft_rem);
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

/** @brief Number of profiled stages of one coil group (see ::mr_stage_names). */
#define MR_N_STAGES 5

/** @brief Names of the profiled stages, in execution order. */
static const char *const mr_stage_names[MR_N_STAGES] = {
    "pad * S_c", "forward FFT", "block-diagonal", "inverse FFT",
    "crop * conj(S_c)"};

/**
 * @brief Optional per-stage GPU timing (`MR_AHA_METAL_PROFILE=1`).
 *
 * When enabled, every stage of every coil is closed into its own command
 * buffer, run to completion and timed from the command buffer's GPU
 * timestamps, which serialises the work (the totals are therefore slightly
 * above the normal run time). Disabled, it costs nothing.
 */
struct mr_profile {
    bool   on = false;               /**< Profiling requested.             */
    const mr_metal_ctx *ctx = nullptr; /**< For new command buffers.        */
    double ms[MR_N_STAGES] = {0};    /**< Accumulated GPU time per stage.   */
};

/**
 * @brief Close stage @p k when profiling: run it, time it, reopen @p cb/@p enc.
 * @return ::MR_SUCCESS or ::MR_ERR_METAL.
 */
static int mr_stage_end(mr_profile *prof, int k, MTL::CommandBuffer *&cb,
                        MTL::ComputeCommandEncoder *&enc)
{
    if (!prof || !prof->on) return MR_SUCCESS;
    enc->endEncoding();
    const int rc = mr_metal_wait(cb, "mr_aha_plan_execute (profiled)");
    if (rc != MR_SUCCESS) return rc;
    prof->ms[k] += (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3;
    cb = prof->ctx->queue->commandBuffer();
    enc = cb->computeCommandEncoder();
    return MR_SUCCESS;
}

/**
 * @brief Encode the whole operator for coils `[coil0, coil0 + nc)` into @p enc.
 *
 * Buffer indices follow the kernel signatures in `mr_kernels.metal`. Coil
 * `coil0 + s` uses slot `s` of the padded (and compact) buffers. With
 * profiling on, @p cb and @p enc are replaced at every stage boundary.
 */
static int mr_encode_group(mr_aha_plan p, MTL::CommandBuffer *&cb,
                           MTL::ComputeCommandEncoder *&enc, int coil0, int nc,
                           const mr_metal_view &x, const mr_metal_view &se,
                           const mr_metal_view &toep, const mr_metal_view &mask,
                           const mr_metal_view &out, mr_profile *prof)
{
    const mr_dims &d = p->dims;
    const NS::UInteger slot =
        (NS::UInteger)d.pad_size * (NS::UInteger)d.L * sizeof(mr_complex);
    mr_metal_fft *fft = (nc == p->group) ? p->fft : p->fft_rem;

    /* 1. Zero-pad and apply each coil's sensitivity (also clears the grid). */
    for (int s = 0; s < nc; ++s) {
        const int coil = coil0 + s;
        enc->setComputePipelineState(p->k_pad);
        enc->setBuffer(p->pad, slot * (NS::UInteger)s, 0);
        mr_metal_bind(enc, x, 1);
        mr_metal_bind(enc, se, 2);
        enc->setBytes(&d, sizeof(d), 3);
        enc->setBytes(&coil, sizeof(coil), 4);
        mr_metal_dispatch_1d(enc, p->k_pad, (uint64_t)d.pad_size);
    }
    int rc = mr_stage_end(prof, 0, cb, enc);
    if (rc != MR_SUCCESS) return rc;

    /* 2. Forward FFT of all ranks of all coils of the group. */
    rc = mr_metal_fft_encode(fft, cb, enc, p->pad, false);
    if (rc != MR_SUCCESS) return rc;
    if ((rc = mr_stage_end(prof, 1, cb, enc)) != MR_SUCCESS) return rc;

    /* 3. Hermitian block-diagonal multiplication (one pass over phi2_toep
     *    for the whole group with the specialised kernels). */
    if (p->cfg.is_masked) {
        const long long nm = (long long)p->cfg.num_masked;
        const NS::UInteger cslot =
            (NS::UInteger)nm * (NS::UInteger)d.L * sizeof(mr_complex);
        enc->setComputePipelineState(p->k_bd);
        enc->setBuffer(p->compact, 0, 0);
        enc->setBuffer(p->pad, 0, 1);
        mr_metal_bind(enc, toep, 2);
        mr_metal_bind(enc, mask, 3);
        enc->setBytes(&nm, sizeof(nm), 4);
        enc->setBytes(&d, sizeof(d), 5);
        enc->setBytes(&nc, sizeof(nc), 6);
        if (p->bd_tgmem) enc->setThreadgroupMemoryLength(p->bd_tgmem, 0);
        mr_metal_dispatch_1d(enc, p->k_bd, (uint64_t)nm, p->bd_threads);

        /* Samples outside the mask contribute nothing. */
        const long long n = d.pad_size;
        const int count = d.L * nc;
        enc->setComputePipelineState(p->k_zero);
        enc->setBuffer(p->pad, 0, 0);
        enc->setBytes(&n, sizeof(n), 1);
        enc->setBytes(&count, sizeof(count), 2);
        mr_metal_dispatch_1d(enc, p->k_zero, (uint64_t)n);

        for (int s = 0; s < nc; ++s) {
            enc->setComputePipelineState(p->k_scatter);
            enc->setBuffer(p->pad, slot * (NS::UInteger)s, 0);
            enc->setBuffer(p->compact, cslot * (NS::UInteger)s, 1);
            mr_metal_bind(enc, mask, 2);
            enc->setBytes(&nm, sizeof(nm), 3);
            enc->setBytes(&d, sizeof(d), 4);
            mr_metal_dispatch_1d(enc, p->k_scatter, (uint64_t)nm);
        }
    } else {
        enc->setComputePipelineState(p->k_bd);
        enc->setBuffer(p->pad, 0, 0);
        mr_metal_bind(enc, toep, 1);
        enc->setBytes(&d, sizeof(d), 2);
        enc->setBytes(&nc, sizeof(nc), 3);
        if (p->bd_tgmem) enc->setThreadgroupMemoryLength(p->bd_tgmem, 0);
        mr_metal_dispatch_1d(enc, p->k_bd, (uint64_t)d.pad_size, p->bd_threads);
    }
    if ((rc = mr_stage_end(prof, 2, cb, enc)) != MR_SUCCESS) return rc;

    /* 4. Inverse FFT back to the padded image domain. */
    rc = mr_metal_fft_encode(fft, cb, enc, p->pad, true);
    if (rc != MR_SUCCESS) return rc;
    if ((rc = mr_stage_end(prof, 3, cb, enc)) != MR_SUCCESS) return rc;

    /* 5. Crop, apply the adjoint sensitivity and accumulate over coils. */
    for (int s = 0; s < nc; ++s) {
        const int coil = coil0 + s;
        const int accumulate = coil > 0 ? 1 : 0;
        enc->setComputePipelineState(p->k_crop);
        mr_metal_bind(enc, out, 0);
        enc->setBuffer(p->pad, slot * (NS::UInteger)s, 1);
        mr_metal_bind(enc, se, 2);
        enc->setBytes(&d, sizeof(d), 3);
        enc->setBytes(&coil, sizeof(coil), 4);
        enc->setBytes(&accumulate, sizeof(accumulate), 5);
        mr_metal_dispatch_1d(enc, p->k_crop, (uint64_t)d.image_size);
    }
    return mr_stage_end(prof, 4, cb, enc);
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
    mr_profile prof;
    {
        const char *e = std::getenv("MR_AHA_METAL_PROFILE");
        prof.on = e && e[0] && e[0] != '0';
        prof.ctx = ctx;
    }
    std::vector<MTL::CommandBuffer *> cbs;
    cbs.reserve((size_t)cfg->n_coils);
    for (int coil0 = 0; coil0 < cfg->n_coils; coil0 += plan->group) {
        const int nc = (cfg->n_coils - coil0 < plan->group) ? cfg->n_coils - coil0
                                                            : plan->group;
        MTL::CommandBuffer *cb = ctx->queue->commandBuffer();
        MTL::ComputeCommandEncoder *enc = cb->computeCommandEncoder();
        rc = mr_encode_group(plan, cb, enc, coil0, nc, x, se, toep, mask, out,
                             &prof);
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

    if (prof.on) {
        double total = 0.0;
        for (int k = 0; k < MR_N_STAGES; ++k) total += prof.ms[k];
        fprintf(stderr, "[mr_aha] Metal profile, %d coil(s) in groups of %d, %s:\n",
                cfg->n_coils, plan->group, cfg->is_masked ? "masked" : "unmasked");
        for (int k = 0; k < MR_N_STAGES; ++k)
            fprintf(stderr, "[mr_aha]   %-18s %9.2f ms  (%4.1f%%)\n",
                    mr_stage_names[k], prof.ms[k], 100.0 * prof.ms[k] / total);
        fprintf(stderr, "[mr_aha]   %-18s %9.2f ms\n", "total GPU", total);
    }
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

/* ------------------------------------------------------------------------ */
/* The operator object (single device, Toeplitz, supplied kernel)           */
/* ------------------------------------------------------------------------ */

/**
 * @brief Metal state of an ::mr_op: a single-device Toeplitz operator around
 *        an ::mr_aha_plan.
 *
 * The Metal backend computes the kernel partly outside the library (the
 * type-1 NUFFT runs through mlx-nufft in Python, see ::mr_calc_toep_from_psf),
 * so ::mr_op_create here requires the finished kernel in
 * ::mr_op_inputs::d_phi2_toep and supports one device only; the Python
 * binding (`mr_aha.create_op`) computes the kernel and passes it in, and
 * implements the NUFFT method itself (MLX + mlx-nufft). With
 * ::MR_OP_SHARE_PRIMARY the caller's arrays are used in place (unified
 * memory); otherwise they are copied into host buffers owned by the operator.
 */
struct mr_op_s {
    mr_config               cfg;   /**< Copy of the configuration.            */
    mr_aha_plan             plan;  /**< The single-device plan.               */
    const mr_complex       *SEs;   /**< Maps used by every execution.         */
    const mr_complex       *toep;  /**< Kernel used by every execution.       */
    const int64_t          *mask;  /**< Mask, or NULL.                        */
    std::vector<mr_complex> SEs_own;  /**< Copies when not sharing.           */
    std::vector<mr_complex> toep_own;
    std::vector<int64_t>    mask_own;
    mr_op_info_t            info;  /**< What ::mr_op_info reports.            */
};

extern "C" int mr_op_options_defaults(mr_op_options *opt)
{
    mr_clear_error();
    MR_REQUIRE(opt != NULL, "mr_op_options_defaults: opt is NULL");
    memset(opt, 0, sizeof(*opt));
    opt->method = MR_OP_TOEPLITZ;
    opt->toep_dist = MR_TOEP_DIST_AUTO;
    opt->flags = MR_OP_SHARE_PRIMARY;
    return MR_SUCCESS;
}

extern "C" int mr_op_create(const mr_config *cfg, const mr_op_inputs *in,
                            const mr_op_options *opt, mr_op *op_out)
{
    mr_clear_error();
    MR_REQUIRE(op_out != NULL, "mr_op_create: op_out is NULL");
    *op_out = NULL;
    MR_REQUIRE(cfg != NULL && in != NULL, "mr_op_create: cfg or in is NULL");
    mr_op_options o;
    if (opt) o = *opt;
    else mr_op_options_defaults(&o);
    const int status = mr_config_validate(cfg);
    if (status != MR_SUCCESS) return status;
    if (o.method != MR_OP_TOEPLITZ)
        MR_FAIL(MR_ERR_UNSUPPORTED,
                "mr_op_create: the Metal backend implements the Toeplitz method in "
                "the library; the NUFFT method is provided by the Python binding");
    const int one_device =
        (o.devices == NULL && (o.n_devices == 0 || o.n_devices == MR_OP_ALL_DEVICES)) ||
        (o.devices != NULL && o.n_devices == 1 && o.devices[0] == 0);
    if (!one_device)
        MR_FAIL(MR_ERR_UNSUPPORTED,
                "mr_op_create: the Metal backend has a single GPU (device 0)");
    MR_REQUIRE(in->d_SEs != NULL, "mr_op_create: d_SEs is NULL");
    if (in->d_phi2_toep == NULL)
        MR_FAIL(MR_ERR_UNSUPPORTED,
                "mr_op_create: the Metal backend needs the finished kernel in "
                "d_phi2_toep (computed by mr_aha.calc_toep)");
    if (cfg->is_masked)
        MR_REQUIRE(in->d_mask != NULL,
                   "mr_op_create: d_mask is required when is_masked is set");

    const auto t0 = std::chrono::steady_clock::now();
    mr_op p = new (std::nothrow) mr_op_s();
    if (!p) MR_FAIL(MR_ERR_ALLOC, "failed to allocate the operator struct");
    p->cfg = *cfg;
    const int rc = mr_aha_plan_create(cfg, &p->plan);
    if (rc != MR_SUCCESS) {
        delete p;
        return rc;
    }
    if (o.flags & MR_OP_SHARE_PRIMARY) {
        p->SEs = in->d_SEs;
        p->toep = in->d_phi2_toep;
        p->mask = cfg->is_masked ? in->d_mask : NULL;
    } else {
        const size_t nse = (size_t)mr_se_size(cfg);
        const size_t ntoep = (size_t)mr_phi2_toep_size(cfg);
        try {
            p->SEs_own.assign(in->d_SEs, in->d_SEs + nse);
            p->toep_own.assign(in->d_phi2_toep, in->d_phi2_toep + ntoep);
            if (cfg->is_masked)
                p->mask_own.assign(in->d_mask, in->d_mask + cfg->num_masked);
        } catch (...) {
            mr_aha_plan_destroy(p->plan);
            delete p;
            MR_FAIL(MR_ERR_ALLOC, "failed to copy the operator's inputs");
        }
        p->SEs = p->SEs_own.data();
        p->toep = p->toep_own.data();
        p->mask = cfg->is_masked ? p->mask_own.data() : NULL;
    }
    memset(&p->info, 0, sizeof(p->info));
    p->info.method = MR_OP_TOEPLITZ;
    p->info.n_devices = 1;
    p->info.coil_count[0] = cfg->n_coils;
    p->info.group[0] = 1;
    p->info.toep_source = MR_OP_TOEP_SUPPLIED;
    p->info.create_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
    *op_out = p;
    return MR_SUCCESS;
}

extern "C" int mr_op_execute(mr_op op, const mr_complex *d_x, mr_complex *d_AhAx)
{
    mr_clear_error();
    MR_REQUIRE(op != NULL, "mr_op_execute: op is NULL");
    return mr_aha_plan_execute(op->plan, d_x, op->SEs, op->toep, op->mask, d_AhAx);
}

extern "C" int mr_op_info(mr_op op, mr_op_info_t *info)
{
    mr_clear_error();
    MR_REQUIRE(op != NULL && info != NULL, "mr_op_info: op or info is NULL");
    *info = op->info;
    return MR_SUCCESS;
}

extern "C" int mr_op_kernel(mr_op op, int k, const mr_complex **d_phi2_toep)
{
    mr_clear_error();
    MR_REQUIRE(op != NULL && d_phi2_toep != NULL, "mr_op_kernel: NULL argument");
    MR_REQUIRE(k == 0, "mr_op_kernel: device index %d outside [0, 1)", k);
    *d_phi2_toep = op->toep;
    return MR_SUCCESS;
}

extern "C" int mr_op_destroy(mr_op op)
{
    mr_clear_error();
    if (op == NULL) return MR_SUCCESS;
    mr_aha_plan_destroy(op->plan);
    delete op;
    return MR_SUCCESS;
}

extern "C" int mr_op_abandon(mr_op op) { return mr_op_destroy(op); }
