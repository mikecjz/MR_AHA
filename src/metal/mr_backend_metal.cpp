/**
 * @file mr_backend_metal.cpp
 * @brief Backend identification, error-state hook and the shared Metal context.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_metal_internal.h"

#include <mutex>

/** @brief The Metal backend keeps no error state outside ::mr_last_error. */
void mr_backend_clear_error(void) {}

extern "C" const char *mr_backend(void) { return "metal"; }

int mr_metal_context(const mr_metal_ctx **ctx_out)
{
    static mr_metal_ctx ctx = {nullptr, nullptr};
    static std::once_flag once;
    std::call_once(once, [] {
        /* CreateSystemDefaultDevice and newCommandQueue return owned (+1)
         * objects; they are kept for the lifetime of the process. */
        ctx.device = MTL::CreateSystemDefaultDevice();
        if (ctx.device) ctx.queue = ctx.device->newCommandQueue();
    });
    if (!ctx.device)
        MR_FAIL(MR_ERR_METAL, "no Metal device is available on this machine");
    if (!ctx.queue)
        MR_FAIL(MR_ERR_METAL, "failed to create a Metal command queue on %s",
                ctx.device->name()->utf8String());
    *ctx_out = &ctx;
    return MR_SUCCESS;
}
