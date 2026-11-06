/**
 * @file mr_metal_internal.h
 * @brief Internal helpers of the Metal backend.
 *
 * Not part of the public API. Builds on the backend-independent
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
