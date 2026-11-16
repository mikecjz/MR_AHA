/**
 * @file mr_backend_metal.cpp
 * @brief Backend identification, error-state hook and the shared Metal context.
 *
 * Metal counterpart of `src/cuda/mr_backend_cuda.cu`. It holds the three
 * pieces the rest of the Metal backend relies on:
 *  - ::mr_backend, which lets the bindings (MATLAB `mr_backend.m`, Python)
 *    ask at run time which backend the library was built for;
 *  - ::mr_backend_clear_error, the backend hook that ::mr_clear_error calls
 *    (`src/common/mr_config.cpp`); the CUDA backend uses it to discard a
 *    pending `cudaGetLastError`, the Metal backend has nothing to discard;
 *  - ::mr_metal_context, the lazily created, process-wide device and command
 *    queue. Unlike CUDA there is no device ordinal or stream to choose: the
 *    Metal backend always uses the system default GPU and one shared queue.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_metal_internal.h"

#include <mutex>

/** @brief The Metal backend keeps no error state outside ::mr_last_error.
 *
 * Every Metal entry point waits for its own command buffers and checks their
 * status before returning (::mr_metal_wait), so, unlike an asynchronous CUDA
 * error, no failure can be left pending for a later call to pick up. */
void mr_backend_clear_error(void) {}

/* Backend name reported to the bindings ("cuda" in the CUDA build). */
extern "C" const char *mr_backend(void) { return "metal"; }

int mr_metal_context(const mr_metal_ctx **ctx_out)
{
    /* Function-local statics: the context lives for the whole process and is
     * initialised exactly once, even when MATLAB or Python threads call in
     * concurrently (std::call_once is the thread-safe one-time guard). */
    static mr_metal_ctx ctx = {nullptr, nullptr};
    static std::once_flag once;
    std::call_once(once, [] {
        /* CreateSystemDefaultDevice and newCommandQueue return owned (+1)
         * objects; they are kept for the lifetime of the process. */
        ctx.device = MTL::CreateSystemDefaultDevice();
        if (ctx.device) ctx.queue = ctx.device->newCommandQueue();
    });

    /* Checked on every call, outside call_once: if creation failed the first
     * time, every later call reports the same error rather than handing out
     * a context with NULL members. Creation is not retried. */
    if (!ctx.device)
        MR_FAIL(MR_ERR_METAL, "no Metal device is available on this machine");
    if (!ctx.queue)
        MR_FAIL(MR_ERR_METAL, "failed to create a Metal command queue on %s",
                ctx.device->name()->utf8String());
    *ctx_out = &ctx;
    return MR_SUCCESS;
}
