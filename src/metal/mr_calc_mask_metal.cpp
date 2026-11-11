/**
 * @file mr_calc_mask_metal.cpp
 * @brief Automatic k-space mask on Metal.
 *
 * Same constructions as `src/cuda/mr_calc_mask.cu` (see its file comment for
 * the rationale of both methods):
 *
 * - **Neighbourhood mask** (::MR_MASK_NEIGHBOUR, default): mark the grid point
 *   nearest to every sample and dilate by a @f$(2r+1)^d@f$ box. No NUFFT, so
 *   ::mr_calc_mask computes it entirely here; the marking uses the same
 *   rounding as CUDA and the mask is identical.
 * - **Threshold mask** (::MR_MASK_THRESHOLD): needs the unit-weight Toeplitz
 *   kernel, i.e. a NUFFT, which the Metal backend runs in Python (mlx-nufft).
 *   Python's `mr_aha.calc_mask` computes the kernel and calls
 *   ::mr_calc_mask_from_kernel, which dilates, thresholds and compacts it here.
 *   ::mr_calc_mask_kernel and the threshold method of ::mr_calc_mask therefore
 *   return ::MR_ERR_UNSUPPORTED with directions.
 *
 * The GPU builds the selection grid (marks or kernel, then the separable
 * dilation) in a shared buffer; the sorted index list is then compacted on the
 * CPU directly from unified memory, which is a single pass over at most a few
 * ten million floats and yields ascending indices by construction.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_metal_internal.h"

#include <cstring>

/** @brief Mirror of `mr_mark_params` in `mr_kernels.metal`. */
struct mr_mark_params {
    long long n_samples; /**< `Ntrajs * Nkx`.                   */
    int       ndim;      /**< Trajectory coordinates (2 or 3).  */
    int       m[3];      /**< Grid extents (unused axes 1).     */
    float     sc[3];     /**< Per-axis scale `M_a / (2*pi)`.    */
};
static_assert(sizeof(mr_mark_params) == 40,
              "mr_mark_params must match the struct in mr_kernels.metal");

namespace {

/**
 * @brief Argument checks shared by the mask entry points (masking off, since
 *        the number of masked locations is what is being computed).
 */
int mr_mask_check(const char *fn, const mr_config *cfg)
{
    MR_REQUIRE(cfg != NULL, "%s: cfg is NULL", fn);
    mr_config c = *cfg;
    c.is_masked = 0;
    c.num_masked = 0;
    return mr_config_validate(&c);
}

/**
 * @brief Build the selection grid on the GPU, then compact it on the CPU.
 *
 * @param cfg        Validated configuration.
 * @param traj       Trajectory (neighbourhood method), or NULL.
 * @param n_samples  `Ntrajs * Nkx` (neighbourhood method).
 * @param kernel     Unit-weight kernel (threshold method), or NULL.
 * @param d_mask     Output indices or NULL to count.
 * @param n_out      Number of retained locations.
 */
int mr_mask_metal(const mr_config *cfg, const float *traj, int64_t n_samples,
                  const float *kernel, int64_t *d_mask, int64_t *n_out)
{
    *n_out = 0;
    mr_metal_pool pool;
    const mr_metal_ctx *ctx = nullptr;
    int rc = mr_metal_context(&ctx);
    if (rc != MR_SUCCESS) return rc;

    const bool three = cfg->traj_type == MR_TRAJ_3D;
    const int m[3] = {cfg->phi2_dim[0], cfg->phi2_dim[1],
                      three ? cfg->phi2_dim[2] : 1};
    const int64_t n = mr_mask_kernel_size(cfg);
    if (n > 0xFFFFFFFFLL || n_samples > 0xFFFFFFFFLL)
        MR_FAIL(MR_ERR_UNSUPPORTED, "mask grid too large for the Metal backend");

    MTL::ComputePipelineState *k_mark = nullptr, *k_dil = nullptr;
    if ((rc = mr_metal_pipeline("mr_k_mark_samples", &k_mark)) != MR_SUCCESS ||
        (rc = mr_metal_pipeline("mr_k_dilate_axis", &k_dil)) != MR_SUCCESS)
        return rc;

    /* Shared storage: the CPU fills / reads it directly. */
    const NS::UInteger bytes = (NS::UInteger)n * sizeof(float);
    MTL::Buffer *a = ctx->device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    MTL::Buffer *b = ctx->device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (!a || !b) {
        if (a) a->release();
        if (b) b->release();
        MR_FAIL(MR_ERR_ALLOC, "failed to allocate the %lld-element mask grid",
                (long long)n);
    }

    MTL::CommandBuffer *cb = ctx->queue->commandBuffer();
    MTL::ComputeCommandEncoder *enc = cb->computeCommandEncoder();
    mr_metal_view tv;
    int radius;
    float thr;
    if (kernel) {
        std::memcpy(a->contents(), kernel, (size_t)bytes);
        radius = 1;
        thr = (float)cfg->mask_threshold;
    } else {
        std::memset(a->contents(), 0, (size_t)bytes);
        const int ndim = three ? 3 : 2;
        rc = mr_metal_view_make(ctx, traj,
                                (size_t)n_samples * ndim * sizeof(float), false, &tv);
        if (rc == MR_SUCCESS) {
            const double tau = 6.283185307179586; /* 2*pi */
            mr_mark_params p = {(long long)n_samples, ndim, {m[0], m[1], m[2]},
                                {(float)(m[0] / tau), (float)(m[1] / tau),
                                 (float)(m[2] / tau)}};
            enc->setComputePipelineState(k_mark);
            enc->setBuffer(a, 0, 0);
            mr_metal_bind(enc, tv, 1);
            enc->setBytes(&p, sizeof(p), 2);
            mr_metal_dispatch_1d(enc, k_mark, (uint64_t)n_samples);
        }
        radius = cfg->mask_radius;
        thr = 0.5f; /* marks are exactly 0 or 1 */
    }

    /* Separable periodic dilation, ping-ponging between a and b. */
    if (rc == MR_SUCCESS && radius > 0) {
        long long stride = 1;
        const long long nn = (long long)n;
        for (int ax = 0; ax < 3; ++ax) {
            if (m[ax] > 1) {
                enc->setComputePipelineState(k_dil);
                enc->setBuffer(b, 0, 0);
                enc->setBuffer(a, 0, 1);
                enc->setBytes(&nn, sizeof(nn), 2);
                enc->setBytes(&m[ax], sizeof(int), 3);
                enc->setBytes(&stride, sizeof(stride), 4);
                enc->setBytes(&radius, sizeof(radius), 5);
                mr_metal_dispatch_1d(enc, k_dil, (uint64_t)n);
                MTL::Buffer *t = a; a = b; b = t;
            }
            stride *= m[ax];
        }
    }
    enc->endEncoding();
    if (rc == MR_SUCCESS) rc = mr_metal_wait(cb, "mr_calc_mask");

    int64_t k = 0;
    if (rc == MR_SUCCESS) {
        /* Ascending compaction straight from unified memory. */
        const float *sel = static_cast<const float *>(a->contents());
        for (int64_t i = 0; i < n; ++i)
            if (sel[i] > thr) {
                if (d_mask) d_mask[k] = i;
                ++k;
            }
        /* Stack: every kz plane shares the in-plane mask. */
        if (cfg->traj_type == MR_TRAJ_STACK) {
            const int Mz = cfg->phi2_dim[2];
            const int64_t plane = (int64_t)cfg->phi2_dim[0] * cfg->phi2_dim[1];
            if (d_mask)
                for (int kz = 1; kz < Mz; ++kz)
                    for (int64_t i = 0; i < k; ++i)
                        d_mask[i + k * kz] = d_mask[i] + plane * kz;
            k *= Mz;
        }
    }
    a->release();
    b->release();
    if (rc != MR_SUCCESS) return rc;

    if (cfg->verbose)
        printf("[mr_aha] calc_mask: kept %lld of %lld upsampled locations "
               "(%.1f%%) (Metal)\n", (long long)k, (long long)mr_pad_size(cfg),
               100.0 * (double)k / (double)mr_pad_size(cfg));
    *n_out = k;
    return MR_SUCCESS;
}

}  // namespace

extern "C" int mr_calc_mask(const mr_config *cfg, const float *d_traj,
                            int64_t Ntrajs, int64_t Nkx, int64_t *d_mask,
                            int64_t *num_masked_out)
{
    mr_clear_error();
    MR_REQUIRE(num_masked_out != NULL, "mr_calc_mask: num_masked_out is NULL");
    *num_masked_out = 0;
    int rc = mr_mask_check("mr_calc_mask", cfg);
    if (rc != MR_SUCCESS) return rc;
    MR_REQUIRE(d_traj != NULL, "mr_calc_mask: d_traj is NULL");
    MR_REQUIRE(Ntrajs > 0 && Nkx > 0, "mr_calc_mask: Ntrajs = %lld, Nkx = %lld",
               (long long)Ntrajs, (long long)Nkx);
    if (cfg->mask_method == MR_MASK_THRESHOLD)
        MR_FAIL(MR_ERR_UNSUPPORTED, "the threshold mask needs a NUFFT, which the "
                "Metal backend runs outside the library: compute the kernel "
                "(Python's mr_aha.calc_mask does it with mlx-nufft) and call "
                "mr_calc_mask_from_kernel");
    return mr_mask_metal(cfg, d_traj, Ntrajs * Nkx, nullptr, d_mask,
                         num_masked_out);
}

extern "C" int mr_calc_mask_from_kernel(const mr_config *cfg,
                                        const float *d_kernel, int64_t *d_mask,
                                        int64_t *num_masked_out)
{
    mr_clear_error();
    MR_REQUIRE(num_masked_out != NULL,
               "mr_calc_mask_from_kernel: num_masked_out is NULL");
    *num_masked_out = 0;
    int rc = mr_mask_check("mr_calc_mask_from_kernel", cfg);
    if (rc != MR_SUCCESS) return rc;
    MR_REQUIRE(d_kernel != NULL, "mr_calc_mask_from_kernel: d_kernel is NULL");
    return mr_mask_metal(cfg, nullptr, 0, d_kernel, d_mask, num_masked_out);
}

extern "C" int mr_calc_mask_kernel(const mr_config *cfg, const float *d_traj,
                                   int64_t Ntrajs, int64_t Nkx, float *d_kernel)
{
    (void)d_traj; (void)Ntrajs; (void)Nkx; (void)d_kernel;
    mr_clear_error();
    const int rc = mr_mask_check("mr_calc_mask_kernel", cfg);
    if (rc != MR_SUCCESS) return rc;
    MR_FAIL(MR_ERR_UNSUPPORTED, "mr_calc_mask_kernel needs a NUFFT, which the "
            "Metal backend runs outside the library (Python's "
            "mr_aha.calc_mask_kernel does it with mlx-nufft)");
}
