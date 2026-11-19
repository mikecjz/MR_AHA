/**
 * @file mr_calc_toep_metal.cpp
 * @brief Second stage of the Toeplitz pre-computation on Metal.
 *
 * On the Metal backend `phi2_toep` is built in two stages:
 *
 *  1. **Type-1 NUFFT (Python, mlx-nufft).** `mr_aha.calc_toep` packs the Gram
 *     blocks into the stored triangle, broadcasts them along each readout and
 *     evaluates the point-spread function of every triangular component with
 *     mlx-nufft (sign -1, centered modes `phi2_dim`, upsampling factor 2). There
 *     is no C/C++ NUFFT for Apple GPUs that this library could call.
 *  2. **This file, ::mr_calc_toep_from_psf.** Centered inverse FFT (VkFFT),
 *     @f$1/\prod M@f$ scaling and the optional mask gather, i.e. steps 3-4 of
 *     the CUDA ::mr_calc_toep (`src/cuda/mr_calc_toep.cu`), with the same
 *     kernels ported to `mr_kernels.metal`.
 *
 * ::mr_calc_toep itself (which would need the NUFFT inside the library) returns
 * ::MR_ERR_UNSUPPORTED with directions.
 *
 * See `src/cuda/mr_calc_toep.cu` for why a sign -1 type-1 followed by a
 * normalized inverse FFT is exactly the multiplier ::mr_aha_plan_execute
 * needs between its unnormalized FFTs.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_metal_internal.h"

/** @brief Mirror of `mr_reorder` in `mr_kernels.metal`. */
struct mr_reorder {
    int m[3]; /**< Axis extents.                                  */
    int s[3]; /**< Centering shifts, 0 on an untransformed axis.   */
};

extern "C" int mr_calc_toep_from_psf(const mr_config *cfg,
                                     const mr_complex *d_psf, int64_t t0,
                                     int64_t n_comp, const int64_t *d_mask,
                                     mr_complex *d_phi2_toep)
{
    /* Argument checks. The call finishes the triangular components
     * [t0, t0 + n_comp) of the ntri = L(L+1)/2 stored components, so that
     * the Python caller can process the kernel in chunks that respect the
     * VkFFT size limit and memory. */
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

    /* 32-bit thread indices in mr_kernels.metal: one volume, the component
     * batch (grid y) and the masked count must each fit. */
    const int64_t pad = mr_pad_size(cfg);
    if (pad > 0xFFFFFFFFLL || n_comp > 0xFFFFFFFFLL ||
        (cfg->is_masked && cfg->num_masked > 0xFFFFFFFFLL))
        MR_FAIL(MR_ERR_UNSUPPORTED, "the Metal backend supports volumes of "
                "fewer than 2^32 voxels (upsampled grid has %lld)", (long long)pad);

    /* Autorelease pool for the call (also required by VkFFT), shared context,
     * and the two reorder pipelines: centered -> natural before the FFT, and
     * natural -> centered with scaling (or the masked gather) after it. */
    mr_metal_pool pool;
    const mr_metal_ctx *ctx = nullptr;
    if ((rc = mr_metal_context(&ctx)) != MR_SUCCESS) return rc;

    MTL::ComputePipelineState *k_nat = nullptr, *k_out = nullptr;
    if ((rc = mr_metal_pipeline("mr_k_to_natural", &k_nat)) != MR_SUCCESS ||
        (rc = mr_metal_pipeline(cfg->is_masked ? "mr_k_gather_masked_scale"
                                               : "mr_k_to_centered_scale",
                                &k_out)) != MR_SUCCESS)
        return rc;

    /* Geometry of the centered inverse FFT, as in mr_toep_geom (CUDA).
     * 3D: one rank-3 transform per component, all three axes recentered.
     * 2D / stack: rank-2 transforms over [Mx, My], one per kz plane and
     * component (batch Mz * n_comp; Mz = 1 in 2D). A stack's kz axis is
     * already Cartesian, so it is neither transformed nor recentered (shift
     * 0). The scale 1/prod(M) covers only the transformed axes and makes the
     * unnormalized VkFFT inverse a normalized one. */
    const int Mx = cfg->phi2_dim[0], My = cfg->phi2_dim[1], Mz = cfg->phi2_dim[2];
    const bool is3d = cfg->traj_type == MR_TRAJ_3D;
    mr_reorder r = {{Mx, My, Mz}, {Mx / 2, My / 2, is3d ? Mz / 2 : 0}};
    const int fft_rank = is3d ? 3 : 2;
    const int fft_dims[3] = {Mx, My, is3d ? Mz : 1};
    const int64_t fft_batch = is3d ? n_comp : (int64_t)Mz * n_comp;
    const float scale =
        1.0f / (float)(is3d ? (double)Mx * My * Mz : (double)Mx * My);

    /* Caller arrays. The output view covers the whole phi2_toep. The PSF
     * holds n_comp centered volumes of pad values; the mask view is empty
     * when unmasked. Only the output is a writeback view. Views release
     * their buffers on return, so early returns need no cleanup. */
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

    /* Natural-order scratch, transformed in place by VkFFT. Private storage:
     * only the GPU touches it. The FFT plan is created per call (VkFFT
     * compiles its kernels for a new size; see mr_metal_fft.cpp) and
     * rejects batches of 2^31 values or more. */
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

    /* All three steps go into one command buffer and one compute encoder;
     * VkFFT appends its kernels to the same encoder, so the order is kept
     * without extra synchronization (as on the single CUDA stream). */
    MTL::CommandBuffer *cb = ctx->queue->commandBuffer();
    MTL::ComputeCommandEncoder *enc = cb->computeCommandEncoder();

    /* Centered -> natural order: the PSF's DC sits at floor(M/2), while the
     * FFT expects it at index 0. 2-D grid of (pad, n_comp) threads. */
    enc->setComputePipelineState(k_nat);
    enc->setBuffer(work, 0, 0);
    mr_metal_bind(enc, psf, 1);
    enc->setBytes(&r, sizeof(r), 2);
    mr_metal_dispatch_2d(enc, k_nat, (uint64_t)pad, (uint64_t)n_comp);

    /* Unnormalized inverse FFT over the transformed axes. */
    rc = mr_metal_fft_encode(fft, cb, enc, work, true);

    /* Natural -> centered order with 1/prod(M), or the mask gather. Masked:
     * only the num_masked retained locations are read (back in centered
     * order) and written to columns [t0, t0 + n_comp) of the
     * [num_masked, ntri] output; the kernel takes t0 and binds the whole
     * output. Unmasked: the output buffer is bound at the byte offset of
     * component t0 instead. Skipped if the FFT could not be encoded. */
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
    /* Close the encoder in every case; run and wait only if all encoded. */
    enc->endEncoding();
    if (rc == MR_SUCCESS) rc = mr_metal_wait(cb, "mr_calc_toep_from_psf");

    /* Cleanup for success and failure; the GPU is idle at this point (the
     * command buffer either completed or was never committed). The output
     * is copied back only if the view had to fall back to a copy. */
    mr_metal_fft_destroy(fft);
    work->release();
    if (rc == MR_SUCCESS) mr_metal_view_finish(&out);
    if (rc == MR_SUCCESS && cfg->verbose)
        printf("[mr_aha] calc_toep_from_psf: components %lld-%lld of %lld done "
               "(Metal)\n", (long long)t0, (long long)(t0 + n_comp - 1),
               (long long)ntri);
    return rc;
}

extern "C" int mr_calc_toep(const mr_config *cfg, const mr_complex *d_phi2,
                            const float *d_traj, int64_t Ntrajs, int64_t Nkx,
                            const int64_t *d_mask, mr_complex *d_phi2_toep)
{
    (void)d_phi2; (void)d_traj; (void)Ntrajs; (void)Nkx; (void)d_mask;
    (void)d_phi2_toep;
    /* Kept so that the Metal library exports the full C API (bindings link
     * against the same symbols for both backends). The configuration is
     * still validated so that an invalid one reports its real problem. */
    mr_clear_error();
    const int status = mr_config_validate(cfg);
    if (status != MR_SUCCESS) return status;
    /* There is no C/C++ NUFFT for Apple GPUs to call here: the type-1 NUFFT
     * runs in Python (mlx-nufft) and mr_calc_toep_from_psf does the rest. */
    MR_FAIL(MR_ERR_UNSUPPORTED, "mr_calc_toep is not available on the Metal "
            "backend: compute the type-1 NUFFT outside the library (Python's "
            "mr_aha.calc_toep does it with mlx-nufft) and finish with "
            "mr_calc_toep_from_psf");
}
