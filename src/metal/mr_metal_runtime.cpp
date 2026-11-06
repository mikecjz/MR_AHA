/**
 * @file mr_metal_runtime.cpp
 * @brief Shader library loading, pipeline cache, caller-array views and
 *        command-buffer helpers of the Metal backend.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_metal_internal.h"

#include <dispatch/dispatch.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

/* ------------------------------------------------------------------------ */
/* Embedded shaders                                                         */
/* ------------------------------------------------------------------------ */

/* Generated at build time by cmake/mr_embed.cmake from mr_kernels.metal:
 * the source is always embedded; the precompiled library only when the build
 * machine had the Metal toolchain (otherwise its size is 0). */
extern "C" const unsigned char mr_metal_source_data[];
extern "C" const unsigned long long mr_metal_source_size;
extern "C" const unsigned char mr_metal_metallib_data[];
extern "C" const unsigned long long mr_metal_metallib_size;

namespace {

/** @brief `MTLLanguageVersion3_0`, which the bundled metal-cpp predates. */
constexpr NS::UInteger MR_METAL_LANGUAGE_VERSION_3_0 = (3u << 16) | 0u;

/** @brief Process-wide shader library and pipeline cache. */
struct mr_metal_shaders {
    std::mutex mutex;                 /**< Guards everything below.        */
    MTL::Library *library = nullptr;  /**< Loaded once, kept forever.      */
    bool from_source = false;         /**< How the library was obtained.   */
    std::unordered_map<std::string, MTL::ComputePipelineState *> pipelines;
};

mr_metal_shaders &mr_shaders()
{
    static mr_metal_shaders s;
    return s;
}

/** @brief Text of an NS::Error, or a placeholder when there is none. */
const char *mr_ns_error_text(NS::Error *err)
{
    if (err && err->localizedDescription())
        return err->localizedDescription()->utf8String();
    return "(no error description)";
}

/**
 * @brief Load the shader library (caller holds the mutex).
 * @return ::MR_SUCCESS or ::MR_ERR_METAL.
 */
int mr_load_library(const mr_metal_ctx *ctx, mr_metal_shaders &s)
{
    const char *force = std::getenv("MR_AHA_METAL_FROM_SOURCE");
    const bool want_source =
        mr_metal_metallib_size == 0 || (force && force[0] && force[0] != '0');
    NS::Error *err = nullptr;

    if (!want_source) {
        /* DISPATCH_DATA_DESTRUCTOR_DEFAULT copies the bytes. */
        dispatch_data_t data = dispatch_data_create(
            mr_metal_metallib_data, (size_t)mr_metal_metallib_size, nullptr,
            DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        s.library = ctx->device->newLibrary(data, &err);
        dispatch_release(data);
        if (!s.library)
            MR_FAIL(MR_ERR_METAL, "failed to load the embedded metallib: %s",
                    mr_ns_error_text(err));
        s.from_source = false;
        return MR_SUCCESS;
    }

    NS::String *src = NS::String::alloc()->init(
        (void *)mr_metal_source_data, (NS::UInteger)mr_metal_source_size,
        NS::UTF8StringEncoding, false);
    MTL::CompileOptions *opt = MTL::CompileOptions::alloc()->init();
    opt->setFastMathEnabled(false);
    opt->setLanguageVersion((MTL::LanguageVersion)MR_METAL_LANGUAGE_VERSION_3_0);
    s.library = ctx->device->newLibrary(src, opt, &err);
    opt->release();
    src->release();
    if (!s.library)
        MR_FAIL(MR_ERR_METAL, "failed to compile the Metal kernels: %s",
                mr_ns_error_text(err));
    s.from_source = true;
    return MR_SUCCESS;
}

}  // namespace

int mr_metal_pipeline(const char *name, MTL::ComputePipelineState **out)
{
    const mr_metal_ctx *ctx = nullptr;
    int rc = mr_metal_context(&ctx);
    if (rc != MR_SUCCESS) return rc;

    mr_metal_shaders &s = mr_shaders();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.pipelines.find(name);
    if (it != s.pipelines.end()) {
        *out = it->second;
        return MR_SUCCESS;
    }
    if (!s.library) {
        rc = mr_load_library(ctx, s);
        if (rc != MR_SUCCESS) return rc;
    }

    NS::String *fname = NS::String::alloc()->init(name, NS::UTF8StringEncoding);
    MTL::Function *fn = s.library->newFunction(fname);
    fname->release();
    if (!fn)
        MR_FAIL(MR_ERR_INTERNAL, "kernel %s not found in the Metal library", name);
    NS::Error *err = nullptr;
    MTL::ComputePipelineState *pso = ctx->device->newComputePipelineState(fn, &err);
    fn->release();
    if (!pso)
        MR_FAIL(MR_ERR_METAL, "failed to create the pipeline of %s: %s", name,
                mr_ns_error_text(err));
    s.pipelines.emplace(name, pso);
    *out = pso;
    return MR_SUCCESS;
}

void mr_metal_dispatch_1d(MTL::ComputeCommandEncoder *enc,
                          const MTL::ComputePipelineState *pso, uint64_t n,
                          NS::UInteger tg)
{
    if (n == 0) return;
    const NS::UInteger limit = pso->maxTotalThreadsPerThreadgroup();
    if (tg == 0) tg = 256;
    if (tg > limit) tg = limit;
    enc->dispatchThreads(MTL::Size((NS::UInteger)n, 1, 1), MTL::Size(tg, 1, 1));
}

/* ------------------------------------------------------------------------ */
/* Caller-array views                                                       */
/* ------------------------------------------------------------------------ */

mr_metal_view::~mr_metal_view()
{
    if (buf) buf->release();
}

int mr_metal_view_make(const mr_metal_ctx *ctx, const void *ptr, size_t bytes,
                       bool writeback, mr_metal_view *v)
{
    v->host = const_cast<void *>(ptr);
    v->bytes = bytes;
    v->writeback = writeback;
    if (bytes == 0 || ptr == nullptr) return MR_SUCCESS;

    /* Wrap the page-aligned span containing the array: Metal requires a
     * page-aligned pointer and length for no-copy buffers, and every page
     * overlapping the caller's array is mapped. MR_AHA_METAL_FORCE_COPY=1
     * skips the wrap so that the copy fallback below can be tested. */
    static const bool force_copy = [] {
        const char *e = std::getenv("MR_AHA_METAL_FORCE_COPY");
        return e && e[0] && e[0] != '0';
    }();
    if (!force_copy) {
        const uintptr_t page = (uintptr_t)getpagesize();
        const uintptr_t addr = (uintptr_t)ptr;
        const uintptr_t base = addr & ~(page - 1);
        const uintptr_t end = (addr + bytes + page - 1) & ~(page - 1);
        v->buf = ctx->device->newBuffer((const void *)base,
                                        (NS::UInteger)(end - base),
                                        MTL::ResourceStorageModeShared, nullptr);
        if (v->buf) {
            v->offset = (NS::UInteger)(addr - base);
            return MR_SUCCESS;
        }
    }

    /* Fallback: a shared copy. */
    v->buf = ctx->device->newBuffer((NS::UInteger)bytes,
                                    MTL::ResourceStorageModeShared);
    if (!v->buf)
        MR_FAIL(MR_ERR_ALLOC, "failed to allocate a %zu-byte Metal buffer", bytes);
    std::memcpy(v->buf->contents(), ptr, bytes);
    v->offset = 0;
    v->copied = true;
    return MR_SUCCESS;
}

void mr_metal_view_finish(mr_metal_view *v)
{
    if (v->buf && v->copied && v->writeback)
        std::memcpy(v->host, v->buf->contents(), v->bytes);
}

/* ------------------------------------------------------------------------ */
/* Command buffers                                                          */
/* ------------------------------------------------------------------------ */

int mr_metal_wait(MTL::CommandBuffer *cb, const char *what)
{
    cb->commit();
    cb->waitUntilCompleted();
    if (cb->status() != MTL::CommandBufferStatusCompleted)
        MR_FAIL(MR_ERR_METAL, "%s: GPU command buffer failed: %s", what,
                mr_ns_error_text(cb->error()));
    return MR_SUCCESS;
}
