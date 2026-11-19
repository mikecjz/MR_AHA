/**
 * @file mr_metal_runtime.cpp
 * @brief Shader library loading, pipeline cache, caller-array views and
 *        command-buffer helpers of the Metal backend.
 *
 * This file replaces, for Metal, what the CUDA backend gets from the CUDA
 * runtime and `nvcc` for free:
 *  - **Kernel loading.** CUDA kernels are linked into the binary; Metal
 *    kernels live in a shader library that has to be loaded (or compiled)
 *    at run time, and each kernel needs a compute pipeline state object
 *    before it can be dispatched. ::mr_metal_pipeline does both, once.
 *  - **Launch geometry.** The `<<<grid, block>>>` launches of `src/cuda/`
 *    become ::mr_metal_dispatch_1d / ::mr_metal_dispatch_2d.
 *  - **Caller arrays.** The CUDA API takes device pointers; the Metal API
 *    takes host pointers in unified memory, which must be wrapped in
 *    `MTL::Buffer` objects to be bound to a kernel (::mr_metal_view_make).
 *  - **Synchronization.** `cudaStreamSynchronize` + `cudaGetLastError`
 *    become ::mr_metal_wait.
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

/** @brief The singleton ::mr_metal_shaders (function-local static, so it is
 *         constructed thread-safely on first use and never destroyed before
 *         the code that uses it). */
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
    /* Pick the loading path: the precompiled metallib when the build embedded
     * one (fast, no compiler needed), the embedded source otherwise or when
     * MR_AHA_METAL_FROM_SOURCE is set to anything but "" or "0". */
    const char *force = std::getenv("MR_AHA_METAL_FROM_SOURCE");
    const bool want_source =
        mr_metal_metallib_size == 0 || (force && force[0] && force[0] != '0');
    NS::Error *err = nullptr;

    /* Path 1: precompiled metallib. Metal loads a library from memory only
     * through a dispatch_data_t, so the embedded bytes are wrapped in one
     * (and released again once the library is created). */
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

    /* Path 2: compile the embedded mr_kernels.metal source with the system's
     * Metal compiler (available on every macOS with Metal, no Xcode needed).
     * Fast math off and Metal 3.0 mirror the flags the build passes to
     * `xcrun metal` for the metallib (-std=metal3.0 -fno-fast-math in
     * CMakeLists.txt), so both paths compile the same kernels. The
     * NS::String does not take ownership of the bytes (freeWhenDone false);
     * the string and the options are owned (alloc/init) and released here. */
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

    /* Fast path: the pipeline was already built. The whole lookup-or-build
     * runs under the mutex, so two threads asking for the same kernel never
     * build it twice and the library is loaded exactly once. */
    mr_metal_shaders &s = mr_shaders();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.pipelines.find(name);
    if (it != s.pipelines.end()) {
        *out = it->second;
        return MR_SUCCESS;
    }
    /* First pipeline of the process: load the shader library. If loading
     * fails, s.library stays NULL and the next call tries again. */
    if (!s.library) {
        rc = mr_load_library(ctx, s);
        if (rc != MR_SUCCESS) return rc;
    }

    /* Look the kernel up by its host name (for the L-specialized templates
     * this is the [[host_name]] given in mr_kernels.metal, e.g.
     * "mr_k_bd_dense_8"). A missing name is a library bug, hence
     * MR_ERR_INTERNAL rather than MR_ERR_METAL. */
    NS::String *fname = NS::String::alloc()->init(name, NS::UTF8StringEncoding);
    MTL::Function *fn = s.library->newFunction(fname);
    fname->release();
    if (!fn)
        MR_FAIL(MR_ERR_INTERNAL, "kernel %s not found in the Metal library", name);
    /* Build the compute pipeline (the GPU-specific compilation step) and
     * cache it. The function object is no longer needed once the pipeline
     * exists; the pipeline itself is owned by the cache and never released,
     * so callers may keep the pointer for the lifetime of the process. */
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
    /* Zero threads would be an invalid dispatch; encode nothing instead.
     * The threadgroup size defaults to 256 (MR_BLOCK-sized blocks in the CUDA
     * backend) and is clamped to what the pipeline allows, which can be lower
     * for kernels with heavy register or threadgroup-memory use.
     * dispatchThreads launches exactly n threads with a partial last
     * threadgroup, unlike CUDA's whole blocks of mr_ceil_div(n, MR_BLOCK). */
    if (n == 0) return;
    const NS::UInteger limit = pso->maxTotalThreadsPerThreadgroup();
    if (tg == 0) tg = 256;
    if (tg > limit) tg = limit;
    enc->dispatchThreads(MTL::Size((NS::UInteger)n, 1, 1), MTL::Size(tg, 1, 1));
}

void mr_metal_dispatch_2d(MTL::ComputeCommandEncoder *enc,
                          const MTL::ComputePipelineState *pso, uint64_t nx,
                          uint64_t ny)
{
    /* Grid of nx x ny threads with threadgroups of up to 256 x 1: each
     * threadgroup covers a run of one row, so consecutive threads touch
     * consecutive elements of one volume (coalesced) and y indexes the batch
     * (e.g. the triangular component in the calc_toep reorder kernels). */
    if (nx == 0 || ny == 0) return;
    NS::UInteger tg = 256;
    if (tg > pso->maxTotalThreadsPerThreadgroup())
        tg = pso->maxTotalThreadsPerThreadgroup();
    enc->dispatchThreads(MTL::Size((NS::UInteger)nx, (NS::UInteger)ny, 1),
                         MTL::Size(tg, 1, 1));
}

/* ------------------------------------------------------------------------ */
/* Caller-array views                                                       */
/* ------------------------------------------------------------------------ */

mr_metal_view::~mr_metal_view()
{
    /* Releasing a no-copy buffer only drops Metal's mapping: it was created
     * without a deallocator, so the caller's memory itself is untouched. */
    if (buf) buf->release();
}

int mr_metal_view_make(const mr_metal_ctx *ctx, const void *ptr, size_t bytes,
                       bool writeback, mr_metal_view *v)
{
    /* Record the caller's array for mr_metal_view_finish. An empty request
     * (e.g. the mask of an unmasked operator) leaves buf NULL; binding such a
     * view binds no buffer, which is fine because the kernels that would
     * read it are not dispatched. */
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
    /* Round the start down and the end up to page boundaries, wrap that span
     * with no deallocator (the caller keeps ownership), and remember where
     * the array starts inside it; kernels see the array at v->offset. A NULL
     * result (e.g. memory Metal cannot map) falls through to the copy. */
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

    /* Fallback: a shared copy. The caller's data is copied in for inputs and
     * outputs alike (so an output starts from the caller's contents, as with
     * the in-place wrap); outputs are copied back by mr_metal_view_finish. */
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
    /* Only the copy fallback needs work: a wrapped array already holds the
     * GPU's writes, since the buffer is the caller's memory. */
    if (v->buf && v->copied && v->writeback)
        std::memcpy(v->host, v->buf->contents(), v->bytes);
}

/* ------------------------------------------------------------------------ */
/* Command buffers                                                          */
/* ------------------------------------------------------------------------ */

int mr_metal_wait(MTL::CommandBuffer *cb, const char *what)
{
    /* Submit and block until the GPU is done (the analog of
     * cudaStreamSynchronize). Errors raised while the GPU executes (page
     * faults, watchdog timeouts, ...) only appear in the command buffer's
     * final status, so it is checked explicitly. The caller must already
     * have ended its encoder. */
    cb->commit();
    cb->waitUntilCompleted();
    if (cb->status() != MTL::CommandBufferStatusCompleted)
        MR_FAIL(MR_ERR_METAL, "%s: GPU command buffer failed: %s", what,
                mr_ns_error_text(cb->error()));
    return MR_SUCCESS;
}
