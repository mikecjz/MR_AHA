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

#include <new>

/**
 * @brief Largest number of complex values one VkFFT-Metal transform may cover.
 *
 * VkFFT indexes its buffers with 32-bit signed integers unless it switches to
 * 64-bit addressing (`useUint64`, automatic from 2^34-byte buffers), and the
 * Metal code of that mode does not compile (`VKFFT_ERROR_FAILED_TO_COMPILE_
 * PROGRAM`). Launch-time buffer offsets do not help either: they enter the same
 * 32-bit index arithmetic, and a 3D batch split at 2^31 values came out wrong
 * beyond that point. Batches of 2^31 values or more are therefore rejected;
 * the callers keep below the limit (Python's `calc_toep` chunks its components
 * accordingly).
 */
static const uint64_t MR_FFT_MAX_VALUES = (1ull << 31) - 1;

/** @brief A VkFFT application plus the storage its configuration points to. */
struct mr_metal_fft {
    VkFFTApplication app;   /**< The VkFFT plan (kernels for both directions). */
    uint64_t         bytes; /**< Buffer size; VkFFT keeps a pointer to it.     */
};

int mr_metal_fft_create(const mr_metal_ctx *ctx, int rank, const int dims[3],
                        int batch, mr_metal_fft **out,
                        const mr_metal_zeropad *zeropad)
{
    *out = nullptr;
    MR_REQUIRE(rank >= 1 && rank <= 3, "mr_metal_fft_create: rank %d", rank);
    MR_REQUIRE(batch >= 1, "mr_metal_fft_create: batch %d", batch);

    uint64_t volume = 1;
    for (int d = 0; d < rank; ++d) volume *= (uint64_t)dims[d];
    if (volume * (uint64_t)batch > MR_FFT_MAX_VALUES)
        MR_FAIL(MR_ERR_UNSUPPORTED, "a batched FFT of %llu complex values (%d x "
                "%llu) exceeds the 2^31-value limit of VkFFT on Metal; use fewer "
                "ranks / components per call",
                (unsigned long long)(volume * (uint64_t)batch), batch,
                (unsigned long long)volume);

    mr_metal_fft *f = new (std::nothrow) mr_metal_fft();
    if (!f) MR_FAIL(MR_ERR_ALLOC, "failed to allocate an FFT plan");
    f->bytes = volume * (uint64_t)batch * sizeof(mr_complex);

    VkFFTConfiguration cfg = {};
    cfg.FFTdim = (pfUINT)rank;
    for (int d = 0; d < rank; ++d) cfg.size[d] = (pfUINT)dims[d];
    cfg.numberBatches = (pfUINT)batch;
    cfg.device = ctx->device;
    cfg.queue = ctx->queue;
    cfg.bufferSize = &f->bytes;     /* VkFFT keeps this pointer. */
    cfg.normalize = 0;              /* unnormalised inverse, like cuFFT */
    cfg.coalescedMemory = 32;
    if (zeropad) {
        /* Spatial zero padding: skipped on the forward input and the inverse
         * output. VkFFT's block is [left, right). */
        for (int d = 0; d < rank; ++d) {
            if (zeropad->left[d] < zeropad->right[d]) {
                cfg.performZeropadding[d] = 1;
                cfg.fft_zeropad_left[d] = (pfUINT)zeropad->left[d];
                cfg.fft_zeropad_right[d] = (pfUINT)zeropad->right[d];
            }
        }
        cfg.frequencyZeroPadding = 0;
    }

    const VkFFTResult r = initializeVkFFT(&f->app, cfg);
    if (r != VKFFT_SUCCESS) {
        delete f;
        MR_FAIL(MR_ERR_VKFFT, "initializeVkFFT failed with code %d for a "
                "rank-%d transform of %d x %d x %d, batch %d",
                (int)r, rank, dims[0], rank > 1 ? dims[1] : 1,
                rank > 2 ? dims[2] : 1, batch);
    }
    *out = f;
    return MR_SUCCESS;
}

int mr_metal_fft_encode(mr_metal_fft *fft, MTL::CommandBuffer *cb,
                        MTL::ComputeCommandEncoder *enc, MTL::Buffer *buf,
                        bool inverse)
{
    VkFFTLaunchParams lp = {};
    lp.buffer = &buf;
    lp.commandBuffer = cb;
    lp.commandEncoder = enc;
    /* VkFFT direction: -1 forward (exp(-i)), +1 inverse (exp(+i)). */
    const VkFFTResult r = VkFFTAppend(&fft->app, inverse ? 1 : -1, &lp);
    if (r != VKFFT_SUCCESS)
        MR_FAIL(MR_ERR_VKFFT, "VkFFTAppend (%s) failed with code %d",
                inverse ? "inverse" : "forward", (int)r);
    return MR_SUCCESS;
}

void mr_metal_fft_destroy(mr_metal_fft *fft)
{
    if (!fft) return;
    deleteVkFFT(&fft->app);
    delete fft;
}
