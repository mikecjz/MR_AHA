/**
 * @file mr_calc_mask_metal.cpp
 * @brief Automatic k-space mask on Metal.
 *
 * Same construction as `src/cuda/mr_calc_mask.cu` (see its file comment for
 * the rationale): mark the grid point nearest to every sample and dilate the
 * marks by a @f$(2r+1)^d@f$ box, @f$r@f$ = ::mr_config::mask_radius. No NUFFT
 * is involved, so ::mr_calc_mask runs entirely here; the marking uses the
 * same rounding as CUDA and the mask is identical.
 *
 * The GPU builds the grid of marks (the marking, then the separable dilation)
 * in a shared buffer; the sorted index list is then compacted on the CPU
 * directly from unified memory, which is a single pass over at most a few
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
 * @brief Argument checks of ::mr_calc_mask (masking off, since the number of
 *        masked locations is what is being computed).
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
 * @brief Build the grid of dilated sample marks on the GPU, then compact it
 *        on the CPU.
 *
 * @param cfg        Validated configuration.
 * @param traj       Trajectory `[Ntrajs, Nkx, ndim]`.
 * @param n_samples  `Ntrajs * Nkx`.
 * @param d_mask     Output indices or NULL to count.
 * @param n_out      Number of retained locations.
 */
int mr_mask_metal(const mr_config *cfg, const float *traj, int64_t n_samples,
                  int64_t *d_mask, int64_t *n_out)
{
    /* Autorelease pool for the whole call (see mr_metal_internal.h) and the
     * shared device / queue. */
    *n_out = 0;
    mr_metal_pool pool;
    const mr_metal_ctx *ctx = nullptr;
    int rc = mr_metal_context(&ctx);
    if (rc != MR_SUCCESS) return rc;

    /* Grid of marks. For 2D and stack-of-stars trajectories only the [Mx, My]
     * plane is built (third extent 1); for 3D the full [Mx, My, Mz] grid.
     * Kernel thread indices are 32-bit (mr_kernels.metal), hence the limit
     * on both the grid and the number of samples. */
    const bool three = cfg->traj_type == MR_TRAJ_3D;
    const int m[3] = {cfg->phi2_dim[0], cfg->phi2_dim[1],
                      three ? cfg->phi2_dim[2] : 1};
    /* Transformed axes only: a stack's kz is handled by replication. */
    const int64_t n = (int64_t)m[0] * m[1] * m[2];
    if (n > 0xFFFFFFFFLL || n_samples > 0xFFFFFFFFLL)
        MR_FAIL(MR_ERR_UNSUPPORTED, "mask grid too large for the Metal backend");

    /* Cached pipelines of the two kernels (ports of the CUDA kernels of the
     * same names in mr_calc_mask.cu). */
    MTL::ComputePipelineState *k_mark = nullptr, *k_dil = nullptr;
    if ((rc = mr_metal_pipeline("mr_k_mark_samples", &k_mark)) != MR_SUCCESS ||
        (rc = mr_metal_pipeline("mr_k_dilate_axis", &k_dil)) != MR_SUCCESS)
        return rc;

    /* Shared storage: the CPU fills / reads it directly. Two grids of float
     * marks (0 or 1) are needed because each dilation pass reads one and
     * writes the other; if either allocation fails, the other is released
     * before returning. */
    const NS::UInteger bytes = (NS::UInteger)n * sizeof(float);
    MTL::Buffer *a = ctx->device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    MTL::Buffer *b = ctx->device->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (!a || !b) {
        if (a) a->release();
        if (b) b->release();
        MR_FAIL(MR_ERR_ALLOC, "failed to allocate the %lld-element mask grid",
                (long long)n);
    }

    /* One command buffer and one compute encoder hold the whole GPU part
     * (mark, then up to three dilation passes); Metal orders dispatches
     * within an encoder, like kernels on one CUDA stream. The grid `a` is
     * cleared by the CPU (it is shared memory, so a memset replaces
     * cudaMemset), and the trajectory is wrapped as a read-only view. If
     * wrapping fails, nothing is encoded and the error falls through to the
     * common cleanup below. */
    MTL::CommandBuffer *cb = ctx->queue->commandBuffer();
    MTL::ComputeCommandEncoder *enc = cb->computeCommandEncoder();
    mr_metal_view tv;
    const int radius = cfg->mask_radius;
    std::memset(a->contents(), 0, (size_t)bytes);
    const int ndim = three ? 3 : 2;
    rc = mr_metal_view_make(ctx, traj,
                            (size_t)n_samples * ndim * sizeof(float), false, &tv);
    /* Marking: one thread per sample sets the nearest grid point of `a` to 1.
     * The trajectory is read as [n_samples, ndim] (coordinate slowest), in
     * k in radians, and scaled by M_a / (2*pi) to grid cells. */
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

    /* Separable periodic dilation, ping-ponging between a and b. A box
     * dilation is the product of 1-D dilations along each axis, so each
     * transformed axis (extent > 1) gets one pass reading `a` and writing
     * `b`, after which the two are swapped; `stride` is the element stride
     * of the current axis in the column-major grid. After the loop the
     * final marks are always in `a`. Radius 0 keeps the bare marks. */
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
    /* The encoder is always ended (an open encoder may not be abandoned);
     * the command buffer is only submitted when everything was encoded. */
    enc->endEncoding();
    if (rc == MR_SUCCESS) rc = mr_metal_wait(cb, "mr_calc_mask");

    int64_t k = 0;
    if (rc == MR_SUCCESS) {
        /* Ascending compaction straight from unified memory. This replaces
         * the thrust::count_if / copy_if of the CUDA backend: the CPU scans
         * the marks in index order, so the indices come out sorted. With
         * d_mask NULL only the count is produced (the sizing call). */
        const float *marks = static_cast<const float *>(a->contents());
        for (int64_t i = 0; i < n; ++i)
            if (marks[i] > 0.5f) { /* marks are exactly 0 or 1 */
                if (d_mask) d_mask[k] = i;
                ++k;
            }
        /* Stack: every kz plane shares the in-plane mask. kz is Cartesian,
         * so the k in-plane indices of plane 0 are repeated for planes
         * 1 .. Mz-1, offset by kz * Mx * My (CPU version of the CUDA
         * mr_k_replicate_planes). The result stays sorted. */
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
    /* Common cleanup for success and failure. The trajectory view releases
     * its own buffer when it goes out of scope. */
    a->release();
    b->release();
    if (rc != MR_SUCCESS) return rc;

    /* Optional report against the full upsampled grid (Mx*My*Mz). */
    if (cfg->verbose)
        printf("[mr_aha] calc_mask: kept %lld of %lld upsampled locations "
               "(%.1f%%) within radius %d of a sample (Metal)\n", (long long)k,
               (long long)mr_pad_size(cfg),
               100.0 * (double)k / (double)mr_pad_size(cfg), cfg->mask_radius);
    *n_out = k;
    return MR_SUCCESS;
}

}  // namespace

extern "C" int mr_calc_mask(const mr_config *cfg, const float *d_traj,
                            int64_t Ntrajs, int64_t Nkx, int64_t *d_mask,
                            int64_t *num_masked_out)
{
    /* Public entry point: reset the error state, validate the configuration
     * (with masking forced off) and the trajectory extents, then do the work
     * in mr_mask_metal. d_traj and d_mask are host pointers here (unified
     * memory), not device pointers as in the CUDA build. d_mask may be NULL
     * to query the count; otherwise it must hold up to mr_pad_size(cfg)
     * entries. */
    mr_clear_error();
    MR_REQUIRE(num_masked_out != NULL, "mr_calc_mask: num_masked_out is NULL");
    *num_masked_out = 0;
    int rc = mr_mask_check("mr_calc_mask", cfg);
    if (rc != MR_SUCCESS) return rc;
    MR_REQUIRE(d_traj != NULL, "mr_calc_mask: d_traj is NULL");
    MR_REQUIRE(Ntrajs > 0 && Nkx > 0, "mr_calc_mask: Ntrajs = %lld, Nkx = %lld",
               (long long)Ntrajs, (long long)Nkx);
    return mr_mask_metal(cfg, d_traj, Ntrajs * Nkx, d_mask, num_masked_out);
}
