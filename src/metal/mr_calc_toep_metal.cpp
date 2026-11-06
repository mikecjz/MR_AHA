/**
 * @file mr_calc_toep_metal.cpp
 * @brief Second stage of the Toeplitz pre-computation on Metal.
 *
 * On the Metal backend `phi2_toep` is built in two stages:
 *
 *  1. **Type-1 NUFFT (Python, mlx-nufft).** `mr_aha.calc_toep` packs the Gram
 *     blocks into the stored triangle, broadcasts them along each readout and
 *     evaluates the point-spread function of every triangular component with
 *     mlx-nufft (sign -1, centred modes `phi2_dim`, upsampling factor 2). There
 *     is no C/C++ NUFFT for Apple GPUs that this library could call.
 *  2. **This file, ::mr_calc_toep_from_psf.** Centred inverse FFT (VkFFT),
 *     @f$1/\prod M@f$ scaling and the optional mask gather, i.e. steps 3-4 of
 *     the CUDA ::mr_calc_toep (`src/cuda/mr_calc_toep.cu`), with the same
 *     kernels ported to `mr_kernels.metal`.
 *
 * See `src/cuda/mr_calc_toep.cu` for why a sign -1 type-1 followed by a
 * normalised inverse FFT is exactly the multiplier ::mr_aha_plan_execute
 * needs between its unnormalised FFTs.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_metal_internal.h"

/** @brief Mirror of `mr_reorder` in `mr_kernels.metal`. */
struct mr_reorder {
    int m[3]; /**< Axis extents.                                  */
    int s[3]; /**< Centring shifts, 0 on an untransformed axis.   */
};

extern "C" int mr_calc_toep_from_psf(const mr_config *cfg,
                                     const mr_complex *d_psf, int64_t t0,
                                     int64_t n_comp, const int64_t *d_mask,
                                     mr_complex *d_phi2_toep)
{
    mr_clear_error();
    MR_REQUIRE(cfg != NULL, "mr_calc_toep_from_psf: cfg is NULL");
    MR_REQUIRE(d_psf != NULL, "mr_calc_toep_from_psf: d_psf is NULL");
    MR_REQUIRE(d_phi2_toep != NULL, "mr_calc_toep_from_psf: d_phi2_toep is NULL");
    int rc = mr_config_validate(cfg);
    if (rc != MR_SUCCESS) return rc;
    const int64_t ntri = mr_num_tri(cfg->L);
    MR_REQUIRE(t0 >= 0 && n_comp >= 1 && t0 + n_comp <= ntri,
               "mr_calc_toep_from_psf: components [%lld, %lld) outside [0, %lld)",
               (long long)t0, (long long)(t0 + n_comp), (long long)ntri);
    if (cfg->is_masked)
        MR_REQUIRE(d_mask != NULL, "mr_calc_toep_from_psf: d_mask is required "
                   "when is_masked is set");

    const int64_t pad = mr_pad_size(cfg);
    if (pad > 0xFFFFFFFFLL || n_comp > 0xFFFFFFFFLL ||
        (cfg->is_masked && cfg->num_masked > 0xFFFFFFFFLL))
        MR_FAIL(MR_ERR_UNSUPPORTED, "the Metal backend supports volumes of "
                "fewer than 2^32 voxels (upsampled grid has %lld)", (long long)pad);

    mr_metal_pool pool;
    const mr_metal_ctx *ctx = nullptr;
    if ((rc = mr_metal_context(&ctx)) != MR_SUCCESS) return rc;

    MTL::ComputePipelineState *k_nat = nullptr, *k_out = nullptr;
    if ((rc = mr_metal_pipeline("mr_k_to_natural", &k_nat)) != MR_SUCCESS ||
        (rc = mr_metal_pipeline(cfg->is_masked ? "mr_k_gather_masked_scale"
                                               : "mr_k_to_centered_scale",
                                &k_out)) != MR_SUCCESS)
        return rc;

    /* Geometry of the centred inverse FFT, as in mr_toep_geom (CUDA). */
    const int Mx = cfg->phi2_dim[0], My = cfg->phi2_dim[1], Mz = cfg->phi2_dim[2];
    const bool is3d = cfg->traj_type == MR_TRAJ_3D;
    mr_reorder r = {{Mx, My, Mz}, {Mx / 2, My / 2, is3d ? Mz / 2 : 0}};
    const int fft_rank = is3d ? 3 : 2;
    const int fft_dims[3] = {Mx, My, is3d ? Mz : 1};
    const int64_t fft_batch = is3d ? n_comp : (int64_t)Mz * n_comp;
    const float scale =
        1.0f / (float)(is3d ? (double)Mx * My * Mz : (double)Mx * My);

    /* Caller arrays. The output view covers the whole phi2_toep. */
    const size_t c8 = sizeof(mr_complex);
    mr_metal_view psf, mask, out;
    if ((rc = mr_metal_view_make(ctx, d_psf, (size_t)(pad * n_comp) * c8, false,
                                 &psf)) ||
        (rc = mr_metal_view_make(ctx, cfg->is_masked ? d_mask : nullptr,
                                 cfg->is_masked ? (size_t)cfg->num_masked * 8 : 0,
                                 false, &mask)) ||
        (rc = mr_metal_view_make(ctx, d_phi2_toep,
                                 (size_t)mr_phi2_toep_size(cfg) * c8, true, &out)))
        return rc;

    /* Natural-order scratch, transformed in place by VkFFT. */
    const NS::UInteger work_bytes = (NS::UInteger)(pad * n_comp) * c8;
    MTL::Buffer *work =
        ctx->device->newBuffer(work_bytes, MTL::ResourceStorageModePrivate);
    if (!work)
        MR_FAIL(MR_ERR_ALLOC, "failed to allocate the %lu-byte calc_toep scratch",
                (unsigned long)work_bytes);
    mr_metal_fft *fft = nullptr;
    rc = mr_metal_fft_create(ctx, fft_rank, fft_dims, (int)fft_batch, &fft);
    if (rc != MR_SUCCESS) {
        work->release();
        return rc;
    }

    MTL::CommandBuffer *cb = ctx->queue->commandBuffer();
    MTL::ComputeCommandEncoder *enc = cb->computeCommandEncoder();

    /* Centred -> natural order. */
    enc->setComputePipelineState(k_nat);
    enc->setBuffer(work, 0, 0);
    mr_metal_bind(enc, psf, 1);
    enc->setBytes(&r, sizeof(r), 2);
    mr_metal_dispatch_2d(enc, k_nat, (uint64_t)pad, (uint64_t)n_comp);

    /* Unnormalised inverse FFT over the transformed axes. */
    rc = mr_metal_fft_encode(fft, cb, enc, work, true);

    /* Natural -> centred order with 1/prod(M), or the mask gather. */
    if (rc == MR_SUCCESS) {
        enc->setComputePipelineState(k_out);
        if (cfg->is_masked) {
            const long long nm = (long long)cfg->num_masked;
            const long long t0ll = (long long)t0;
            mr_metal_bind(enc, out, 0);
            enc->setBuffer(work, 0, 1);
            mr_metal_bind(enc, mask, 2);
            enc->setBytes(&nm, sizeof(nm), 3);
            enc->setBytes(&r, sizeof(r), 4);
            enc->setBytes(&t0ll, sizeof(t0ll), 5);
            enc->setBytes(&scale, sizeof(scale), 6);
            mr_metal_dispatch_2d(enc, k_out, (uint64_t)nm, (uint64_t)n_comp);
        } else {
            /* Components [t0, t0 + n_comp) of the [pad, ntri] output. */
            enc->setBuffer(out.buf, out.offset + (NS::UInteger)(pad * t0) * c8, 0);
            enc->setBuffer(work, 0, 1);
            enc->setBytes(&r, sizeof(r), 2);
            enc->setBytes(&scale, sizeof(scale), 3);
            mr_metal_dispatch_2d(enc, k_out, (uint64_t)pad, (uint64_t)n_comp);
        }
    }
    enc->endEncoding();
    if (rc == MR_SUCCESS) rc = mr_metal_wait(cb, "mr_calc_toep_from_psf");

    mr_metal_fft_destroy(fft);
    work->release();
    if (rc == MR_SUCCESS) mr_metal_view_finish(&out);
    if (rc == MR_SUCCESS && cfg->verbose)
        printf("[mr_aha] calc_toep_from_psf: components %lld-%lld of %lld done "
               "(Metal)\n", (long long)t0, (long long)(t0 + n_comp - 1),
               (long long)ntri);
    return rc;
}
