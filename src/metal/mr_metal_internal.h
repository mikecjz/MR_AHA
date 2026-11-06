/**
 * @file mr_metal_internal.h
 * @brief Internal helpers of the Metal backend.
 *
 * Not part of the public API. metal-cpp itself is instantiated (its
 * `*_PRIVATE_IMPLEMENTATION` symbols emitted) in `mr_metal_fft.cpp`, through
 * `vkFFT.h`. Builds on the backend-independent
 * `src/common/mr_common.h` and adds the process-wide Metal context (device and
 * command queue) shared by every plan, plus an RAII autorelease-pool guard.
 *
 * @section mr_metal_pools Autorelease pools
 *
 * metal-cpp follows Cocoa's ownership rules: objects returned by `new*`,
 * `alloc`/`init` and `copy` are owned by the caller and must be released;
 * everything else (`commandBuffer()`, `computeCommandEncoder()`,
 * `NS::String::string()`, ...) is autoreleased and is freed when the
 * innermost autorelease pool drains. MATLAB and Python call into the library
 * from threads whose pools drain at unpredictable times, and VkFFT (in the
 * fixed fork used here, see `benchmarks/vkfft/README.md`) relies on the caller
 * to provide a pool. Every public entry point of the Metal backend therefore
 * opens an ::mr_metal_pool for its whole duration.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#ifndef MR_AHA_MR_METAL_INTERNAL_H
#define MR_AHA_MR_METAL_INTERNAL_H

#include "mr_common.h"

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"

/**
 * @brief Process-wide Metal objects shared by every plan.
 *
 * Created on first use by ::mr_metal_context and kept for the lifetime of the
 * process (a MEX file or a Python extension can be unloaded, but the system
 * device and its queue are cheap to keep and safe to leak at exit).
 */
struct mr_metal_ctx {
    MTL::Device       *device; /**< System default GPU.                  */
    MTL::CommandQueue *queue;  /**< Queue every command buffer goes to.  */
};

/**
 * @brief Return the process-wide Metal context, creating it on first use.
 *
 * Thread-safe. Fails only when the machine has no Metal device (e.g. a
 * virtual machine without GPU passthrough).
 *
 * @param[out] ctx_out Receives a pointer to the shared context.
 * @return ::MR_SUCCESS or ::MR_ERR_METAL (see ::mr_last_error).
 */
int mr_metal_context(const mr_metal_ctx **ctx_out);

/* ------------------------------------------------------------------------ */
/* Compute pipelines                                                        */
/* ------------------------------------------------------------------------ */

/**
 * @brief Return the compute pipeline of kernel @p name from `mr_kernels.metal`.
 *
 * The shader library is loaded once per process: from the `.metallib`
 * embedded at build time when the build had the Metal toolchain, otherwise
 * by compiling the embedded source with the system's Metal compiler (Metal
 * 3.0, fast math off). Setting the environment variable
 * `MR_AHA_METAL_FROM_SOURCE=1` forces the source path, so that both paths can
 * be tested from one build. Pipelines are cached by name; thread-safe.
 *
 * @param[in]  name Kernel function name, e.g. `"mr_k_pad_mul_se"`.
 * @param[out] out  Receives the pipeline (owned by the cache, never release).
 * @return ::MR_SUCCESS or ::MR_ERR_METAL (see ::mr_last_error).
 */
int mr_metal_pipeline(const char *name, MTL::ComputePipelineState **out);

/**
 * @brief Encode a 1-D dispatch of @p n threads of the current pipeline.
 *
 * Uses non-uniform threadgroups (`dispatchThreads`), so the kernels see
 * exactly @p n threads; they still bound-check for safety.
 *
 * @param enc Encoder with the pipeline and arguments already set.
 * @param pso The pipeline, for its thread-execution limits.
 * @param n   Number of threads (< 2^32).
 * @param tg  Threads per threadgroup, or 0 for the default (256, capped by
 *            the pipeline's limit).
 */
void mr_metal_dispatch_1d(MTL::ComputeCommandEncoder *enc,
                          const MTL::ComputePipelineState *pso, uint64_t n,
                          NS::UInteger tg = 0);

/**
 * @brief Encode a 2-D dispatch of `nx x ny` threads (`ny` is a batch axis).
 *
 * @param enc Encoder with the pipeline and arguments already set.
 * @param pso The pipeline, for its thread-execution limits.
 * @param nx  Threads along x (< 2^32), 256 per threadgroup.
 * @param ny  Threads along y (< 2^32), one per threadgroup row.
 */
void mr_metal_dispatch_2d(MTL::ComputeCommandEncoder *enc,
                          const MTL::ComputePipelineState *pso, uint64_t nx,
                          uint64_t ny);

/* ------------------------------------------------------------------------ */
/* Caller arrays as Metal buffers                                           */
/* ------------------------------------------------------------------------ */

/**
 * @brief A caller-owned host array made visible to the GPU.
 *
 * Apple silicon has unified memory, so a caller's array is normally wrapped
 * in place (`newBuffer(bytesNoCopy)`) over the page-aligned span that
 * contains it, and bound at @ref offset. Only if Metal refuses the wrap is
 * the array copied into a shared buffer (and, for outputs, copied back by
 * ::mr_metal_view_finish). The environment variable
 * `MR_AHA_METAL_FORCE_COPY=1` forces the copy path, for testing.
 */
struct mr_metal_view {
    MTL::Buffer  *buf       = nullptr; /**< Buffer to bind (owned).            */
    NS::UInteger  offset    = 0;       /**< Byte offset of the array in buf.   */
    void         *host      = nullptr; /**< The caller's pointer.              */
    size_t        bytes     = 0;       /**< Size of the caller's array.        */
    bool          copied    = false;   /**< True when buf is a private copy.   */
    bool          writeback = false;   /**< Copy back on finish (outputs).     */

    mr_metal_view() = default;
    mr_metal_view(const mr_metal_view &) = delete;
    mr_metal_view &operator=(const mr_metal_view &) = delete;
    /** @brief Releases the buffer (without copying back; see finish). */
    ~mr_metal_view();
};

/**
 * @brief Make a caller array of @p bytes bytes at @p ptr usable by the GPU.
 *
 * @param[in]  ctx       Metal context.
 * @param[in]  ptr       Caller's array (host pointer); may be NULL only when
 *                       @p bytes is 0, giving an empty view.
 * @param[in]  bytes     Size of the array in bytes.
 * @param[in]  writeback True for outputs: copy back if a copy was needed.
 * @param[out] v         View to populate.
 * @return ::MR_SUCCESS or ::MR_ERR_ALLOC.
 */
int mr_metal_view_make(const mr_metal_ctx *ctx, const void *ptr, size_t bytes,
                       bool writeback, mr_metal_view *v);

/**
 * @brief Copy an output view back to the caller's array if it was copied.
 *        Call after the GPU work has completed.
 */
void mr_metal_view_finish(mr_metal_view *v);

/** @brief Bind a view as buffer argument @p index of @p enc. */
inline void mr_metal_bind(MTL::ComputeCommandEncoder *enc,
                          const mr_metal_view &v, NS::UInteger index)
{
    enc->setBuffer(v.buf, v.offset, index);
}

/**
 * @brief Commit @p cb, wait for it, and report a GPU-side failure.
 * @param cb   Command buffer to run.
 * @param what Name of the operation, for the diagnostic.
 * @return ::MR_SUCCESS or ::MR_ERR_METAL.
 */
int mr_metal_wait(MTL::CommandBuffer *cb, const char *what);

/* ------------------------------------------------------------------------ */
/* Batched FFTs (VkFFT)                                                     */
/* ------------------------------------------------------------------------ */

/** @brief Opaque batched in-place C2C FFT plan (wraps a VkFFT application). */
struct mr_metal_fft;

/**
 * @brief Create a batched, in-place, unnormalised complex FFT plan.
 *
 * Transforms @p batch consecutive column-major volumes of extents
 * `dims[0] x ... x dims[rank-1]` (`dims[0]` fastest), with a batch stride of
 * their product, in both directions without normalisation, exactly like the
 * `cufftPlanMany` plans of the CUDA backend. VkFFT generates and compiles its
 * kernels here, which takes 0.2-0.3 s for a new size (cached by macOS for
 * later processes). Batches of 16 GiB or more are split into sub-batches
 * launched at offsets (VkFFT's 64-bit addressing mode does not compile on
 * Metal); a split batch is limited to 2^32 complex values.
 *
 * @param[in]  ctx   Metal context.
 * @param[in]  rank  Number of transformed axes, 1-3.
 * @param[in]  dims  Extents, fastest first.
 * @param[in]  batch Number of volumes.
 * @param[out] out   Receives the plan.
 * @return ::MR_SUCCESS, ::MR_ERR_VKFFT or ::MR_ERR_ALLOC.
 */
int mr_metal_fft_create(const mr_metal_ctx *ctx, int rank, const int dims[3],
                        int batch, mr_metal_fft **out);

/**
 * @brief Append one transform of @p buf to an open compute encoder.
 * @param fft     Plan.
 * @param cb      Command buffer that owns @p enc.
 * @param enc     Open compute encoder; the FFT kernels are encoded into it.
 * @param buf     Buffer holding the volumes at offset 0, transformed in place.
 * @param inverse False for the forward (`exp(-i...)`) transform, true for the
 *                unnormalised inverse.
 * @return ::MR_SUCCESS or ::MR_ERR_VKFFT.
 */
int mr_metal_fft_encode(mr_metal_fft *fft, MTL::CommandBuffer *cb,
                        MTL::ComputeCommandEncoder *enc, MTL::Buffer *buf,
                        bool inverse);

/** @brief Destroy a plan from ::mr_metal_fft_create (NULL is a no-op). */
void mr_metal_fft_destroy(mr_metal_fft *fft);

/**
 * @brief RAII guard holding an autorelease pool open for one API call.
 *
 * See @ref mr_metal_pools. Construct one at the top of every public entry
 * point; it drains the pool, and with it every autoreleased Metal object the
 * call created, when it goes out of scope.
 */
class mr_metal_pool {
public:
    mr_metal_pool() : pool_(NS::AutoreleasePool::alloc()->init()) {}
    ~mr_metal_pool() { pool_->release(); }
    mr_metal_pool(const mr_metal_pool &) = delete;
    mr_metal_pool &operator=(const mr_metal_pool &) = delete;

private:
    NS::AutoreleasePool *pool_; /**< The pool, drained by the destructor. */
};

#endif /* MR_AHA_MR_METAL_INTERNAL_H */
