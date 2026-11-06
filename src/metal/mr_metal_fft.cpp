/**
 * @file mr_metal_fft.cpp
 * @brief Batched in-place C2C FFTs on Metal through VkFFT.
 *
 * The only translation unit that includes `vkFFT.h` (a large header-only
 * library), so that the rest of the backend compiles quickly and does not see
 * VkFFT's macros. VkFFT comes from the `third_party/VkFFT` submodule, which
 * points to the `mikecjz/VkFFT` fork carrying the Metal autorelease fix
 * (DTolm/VkFFT#226); with that fix VkFFT must be called inside an autorelease
 * pool, which every public entry point of the backend provides.
 *
 * This is also the one translation unit that instantiates metal-cpp:
 * `vkFFT.h` (`vkFFT_Structs.h`) defines `NS_/CA_/MTL_PRIVATE_IMPLEMENTATION`
 * itself before including the metal-cpp headers, so it must be included first
 * here, before any other header pulls metal-cpp in without those macros, and
 * no other translation unit may include it.
 *
 * Settings, from `benchmarks/vkfft/README.md` (M3 Ultra): single precision,
 * `coalescedMemory = 32` bytes (1.35x faster than VkFFT's default on the
 * stack-of-stars grid, neutral on 3D). Forward error 3e-7 against an fp64
 * reference, on par with cuFFT.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

/* First: instantiates metal-cpp (see above). */
#include "vkFFT.h"

#include "mr_metal_internal.h"

#include <cstdlib>
#include <new>

/**
 * @brief Largest buffer one VkFFT application may address, in bytes.
 *
 * From 2^34 bytes (16 GiB) on, VkFFT switches its generated kernels to 64-bit
 * addressing (`useUint64`), and that Metal code fails to compile
 * (`VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM`, seen with a 20 GB batch of 320^3
 * volumes). Larger batches are therefore split into sub-batches below this
 * size, launched at byte offsets. `MR_AHA_METAL_FFT_MAX_BYTES` lowers the
 * limit so that the splitting can be tested on small problems.
 */
static uint64_t mr_fft_max_bytes()
{
    const uint64_t hard = (1ull << 34) - 1;
    const char *e = std::getenv("MR_AHA_METAL_FFT_MAX_BYTES");
    if (e && e[0]) {
        const unsigned long long v = std::strtoull(e, nullptr, 10);
        if (v > 0 && v < hard) return (uint64_t)v;
    }
    return hard;
}

/**
 * @brief A batched FFT, possibly split into sub-batches.
 *
 * `app[0]` transforms `nb[0]` volumes and is launched `reps` times at
 * consecutive offsets; `app[1]`, if `napps == 2`, transforms the remaining
 * `nb[1]` volumes once at the end.
 */
struct mr_metal_fft {
    VkFFTApplication app[2];   /**< Full sub-batch and remainder plans.        */
    uint64_t         bytes[2]; /**< Their buffer sizes; VkFFT keeps pointers.  */
    int              nb[2];    /**< Volumes per plan.                          */
    int              napps;    /**< 1 or 2.                                    */
    int              reps;     /**< Launches of app[0].                        */
    uint64_t         vol_bytes;/**< Bytes per volume.                          */
    bool             offsets;  /**< Whether launches carry offsets.            */
};

/** @brief Initialise one VkFFT application for @p nb volumes. */
static int mr_fft_init_app(const mr_metal_ctx *ctx, int rank, const int dims[3],
                           int nb, bool offsets, VkFFTApplication *app,
                           uint64_t *bytes, uint64_t vol_bytes)
{
    *bytes = vol_bytes * (uint64_t)nb;
    VkFFTConfiguration cfg = {};
    cfg.FFTdim = (pfUINT)rank;
    for (int d = 0; d < rank; ++d) cfg.size[d] = (pfUINT)dims[d];
    cfg.numberBatches = (pfUINT)nb;
    cfg.device = ctx->device;
    cfg.queue = ctx->queue;
    cfg.bufferSize = bytes;         /* VkFFT keeps this pointer. */
    cfg.normalize = 0;              /* unnormalised inverse, like cuFFT */
    cfg.coalescedMemory = 32;
    cfg.specifyOffsetsAtLaunch = offsets ? 1 : 0;
    const VkFFTResult r = initializeVkFFT(app, cfg);
    if (r != VKFFT_SUCCESS)
        MR_FAIL(MR_ERR_VKFFT, "initializeVkFFT failed with code %d for a "
                "rank-%d transform of %d x %d x %d, batch %d",
                (int)r, rank, dims[0], rank > 1 ? dims[1] : 1,
                rank > 2 ? dims[2] : 1, nb);
    return MR_SUCCESS;
}

int mr_metal_fft_create(const mr_metal_ctx *ctx, int rank, const int dims[3],
                        int batch, mr_metal_fft **out)
{
    *out = nullptr;
    MR_REQUIRE(rank >= 1 && rank <= 3, "mr_metal_fft_create: rank %d", rank);
    MR_REQUIRE(batch >= 1, "mr_metal_fft_create: batch %d", batch);

    uint64_t volume = 1;
    for (int d = 0; d < rank; ++d) volume *= (uint64_t)dims[d];
    const uint64_t vol_bytes = volume * sizeof(mr_complex);
    const uint64_t max_bytes = mr_fft_max_bytes();
    if (vol_bytes > (1ull << 34) - 1)
        MR_FAIL(MR_ERR_UNSUPPORTED, "one FFT volume of %llu bytes exceeds the "
                "16 GiB VkFFT-Metal limit", (unsigned long long)vol_bytes);

    uint64_t sub = max_bytes / vol_bytes;
    if (sub < 1) sub = 1;
    if (sub > (uint64_t)batch) sub = (uint64_t)batch;
    const bool split = sub < (uint64_t)batch;
    /* Launch offsets are added in VkFFT's 32-bit index arithmetic. */
    if (split && volume * (uint64_t)batch > 0xFFFFFFFFull)
        MR_FAIL(MR_ERR_UNSUPPORTED, "a batched FFT of %llu complex values is too "
                "large for VkFFT-Metal (limit 2^32 values when split)",
                (unsigned long long)(volume * (uint64_t)batch));

    mr_metal_fft *f = new (std::nothrow) mr_metal_fft();
    if (!f) MR_FAIL(MR_ERR_ALLOC, "failed to allocate an FFT plan");
    f->vol_bytes = vol_bytes;
    f->offsets = split;
    f->nb[0] = (int)sub;
    f->reps = (int)((uint64_t)batch / sub);
    f->nb[1] = (int)((uint64_t)batch % sub);
    f->napps = f->nb[1] > 0 ? 2 : 1;

    int rc = mr_fft_init_app(ctx, rank, dims, f->nb[0], split, &f->app[0],
                             &f->bytes[0], vol_bytes);
    if (rc == MR_SUCCESS && f->napps == 2) {
        rc = mr_fft_init_app(ctx, rank, dims, f->nb[1], split, &f->app[1],
                             &f->bytes[1], vol_bytes);
        if (rc != MR_SUCCESS) deleteVkFFT(&f->app[0]);
    }
    if (rc != MR_SUCCESS) {
        delete f;
        return rc;
    }
    *out = f;
    return MR_SUCCESS;
}

int mr_metal_fft_encode(mr_metal_fft *fft, MTL::CommandBuffer *cb,
                        MTL::ComputeCommandEncoder *enc, MTL::Buffer *buf,
                        bool inverse)
{
    /* VkFFT direction: -1 forward (exp(-i)), +1 inverse (exp(+i)). */
    const int dir = inverse ? 1 : -1;
    const int launches = fft->reps + (fft->napps == 2 ? 1 : 0);
    for (int k = 0; k < launches; ++k) {
        const int a = (k < fft->reps) ? 0 : 1;
        VkFFTLaunchParams lp = {};
        lp.buffer = &buf;
        lp.commandBuffer = cb;
        lp.commandEncoder = enc;
        if (fft->offsets)
            lp.bufferOffset = (pfUINT)((uint64_t)k * fft->nb[0] * fft->vol_bytes);
        const VkFFTResult r = VkFFTAppend(&fft->app[a], dir, &lp);
        if (r != VKFFT_SUCCESS)
            MR_FAIL(MR_ERR_VKFFT, "VkFFTAppend (%s, launch %d of %d) failed with "
                    "code %d", inverse ? "inverse" : "forward", k + 1, launches,
                    (int)r);
    }
    return MR_SUCCESS;
}

void mr_metal_fft_destroy(mr_metal_fft *fft)
{
    if (!fft) return;
    for (int a = 0; a < fft->napps; ++a) deleteVkFFT(&fft->app[a]);
    delete fft;
}
