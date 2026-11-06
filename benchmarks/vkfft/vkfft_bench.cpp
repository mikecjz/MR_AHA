/**
 * @file vkfft_bench.cpp
 * @brief Feasibility benchmark of VkFFT's Metal backend for the MR_AHA FFTs.
 *
 * The Toeplitz operator applies, per coil, one forward and one inverse
 * batched complex-to-complex FFT to the zero-padded grid. In column-major
 * notation the grid is `[Mx, My, (Mz), L]`: `x` varies fastest and the `L`
 * ranks are the batch, with a batch stride of `Mx*My*Mz` elements. Both
 * directions are unnormalised, exactly like `cufftExecC2C` in
 * `src/core/mr_aha.cu`.
 *
 * This program reproduces that transform with VkFFT on an Apple GPU and
 * reports
 *   - the plan creation time (VkFFT generates and compiles its Metal kernels
 *     at run time),
 *   - the GPU time of one forward, one inverse and one forward+inverse pair
 *     (median over the iterations, from the command buffer's GPU timestamps),
 *   - the round-trip error `|| ifft(fft(x)) / N - x || / || x ||`.
 *
 * With `--in`/`--out` it also reads the input from a raw complex64 file and
 * writes the forward transform to another one, so that `check_vkfft.py` can
 * compare it against NumPy / SciPy and against MLX's `mx.fft`.
 *
 * Usage:
 * @code
 *   vkfft_bench --dims Mx My [Mz] --batch L [--iters N] [--coalesced B]
 *               [--in input.c64 --out forward.c64]
 * @endcode
 *
 * Built by `build.sh` next to this file (header-only VkFFT and metal-cpp from
 * the `third_party/VkFFT` submodule, `VKFFT_BACKEND=5`).
 */

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"
#include "QuartzCore/QuartzCore.hpp"
#include "vkFFT.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

/** @brief Command-line options. */
struct Options {
    std::vector<uint64_t> dims;   /**< FFT extents, fastest-varying first.   */
    uint64_t batch = 1;           /**< Number of batched transforms (L).    */
    int iters = 20;               /**< Timed repetitions per measurement.   */
    uint64_t coalesced = 0;       /**< VkFFT coalescedMemory (0 = default). */
    std::string in_path;          /**< Optional raw complex64 input.        */
    std::string out_path;         /**< Optional raw complex64 forward output. */
};

/** @brief Print usage and exit with an error. */
[[noreturn]] void usage(const char *argv0)
{
    std::fprintf(stderr,
                 "usage: %s --dims Mx My [Mz] --batch L [--iters N] "
                 "[--coalesced B] [--in in.c64 --out out.c64]\n",
                 argv0);
    std::exit(2);
}

/** @brief Parse the command line into an ::Options. */
Options parse(int argc, char **argv)
{
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dims") {
            while (i + 1 < argc && argv[i + 1][0] != '-')
                o.dims.push_back(std::strtoull(argv[++i], nullptr, 10));
        } else if (a == "--batch" && i + 1 < argc) {
            o.batch = std::strtoull(argv[++i], nullptr, 10);
        } else if (a == "--iters" && i + 1 < argc) {
            o.iters = std::atoi(argv[++i]);
        } else if (a == "--coalesced" && i + 1 < argc) {
            o.coalesced = std::strtoull(argv[++i], nullptr, 10);
        } else if (a == "--in" && i + 1 < argc) {
            o.in_path = argv[++i];
        } else if (a == "--out" && i + 1 < argc) {
            o.out_path = argv[++i];
        } else {
            usage(argv[0]);
        }
    }
    if (o.dims.size() < 1 || o.dims.size() > 3 || o.batch < 1 || o.iters < 1)
        usage(argv[0]);
    return o;
}

/** @brief Median of a vector of timings (the vector is reordered). */
double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/**
 * @brief Encode one or more VkFFT transforms into a fresh command buffer, run
 *        it to completion and return its GPU execution time in milliseconds.
 *
 * @param app        Initialised VkFFT application.
 * @param queue      Command queue the application was created with.
 * @param buf        Data buffer, transformed in place.
 * @param directions Sequence of VkFFT directions (-1 forward, +1 inverse),
 *                   all encoded into the same compute encoder.
 * @return GPU time in ms, or a negative value on a VkFFT error.
 */
double run(VkFFTApplication *app, MTL::CommandQueue *queue, MTL::Buffer *buf,
           const std::vector<int> &directions)
{
    /* The command buffer and encoder are autoreleased: drain them per run so
     * that they neither pile up over the iterations nor outlive the queue. */
    struct Pool {
        NS::AutoreleasePool *p = NS::AutoreleasePool::alloc()->init();
        ~Pool() { p->release(); }
    } pool;
    MTL::CommandBuffer *cb = queue->commandBuffer();
    MTL::ComputeCommandEncoder *enc = cb->computeCommandEncoder();
    VkFFTLaunchParams lp = {};
    lp.buffer = &buf;
    lp.commandBuffer = cb;
    lp.commandEncoder = enc;
    for (int dir : directions) {
        const VkFFTResult r = VkFFTAppend(app, dir, &lp);
        if (r != VKFFT_SUCCESS) {
            std::fprintf(stderr, "VkFFTAppend failed: %d\n", (int)r);
            enc->endEncoding();
            return -1.0;
        }
    }
    enc->endEncoding();
    cb->commit();
    cb->waitUntilCompleted();
    if (cb->status() != MTL::CommandBufferStatusCompleted) {
        std::fprintf(stderr, "command buffer failed (status %d)\n",
                     (int)cb->status());
        return -1.0;
    }
    return (cb->GPUEndTime() - cb->GPUStartTime()) * 1e3;
}

}  // namespace

int main(int argc, char **argv)
{
    /* Unbuffered, so progress survives a crash when stdout is a pipe. */
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const Options opt = parse(argc, argv);
    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();

    uint64_t grid = 1;
    for (uint64_t d : opt.dims) grid *= d;
    const uint64_t n_elem = grid * opt.batch;
    uint64_t bytes = n_elem * sizeof(std::complex<float>);

    MTL::Device *dev = MTL::CreateSystemDefaultDevice();
    MTL::CommandQueue *queue = dev->newCommandQueue();
    std::printf("device %s, dims", dev->name()->utf8String());
    for (uint64_t d : opt.dims) std::printf(" %llu", (unsigned long long)d);
    std::printf(", batch %llu, %.3f GB\n", (unsigned long long)opt.batch,
                bytes / 1e9);

    /* Shared storage: on Apple silicon this is the unified memory the CPU and
     * GPU both see, so the input is written and the output read in place. */
    MTL::Buffer *buf = dev->newBuffer(bytes, MTL::ResourceStorageModeShared);
    if (!buf) {
        std::fprintf(stderr, "failed to allocate %llu bytes\n",
                     (unsigned long long)bytes);
        return 1;
    }
    auto *data = static_cast<std::complex<float> *>(buf->contents());

    /* Input: either from file or a fixed-seed Gaussian. */
    std::vector<std::complex<float>> x0(n_elem);
    if (!opt.in_path.empty()) {
        FILE *f = std::fopen(opt.in_path.c_str(), "rb");
        if (!f || std::fread(x0.data(), sizeof(x0[0]), n_elem, f) != n_elem) {
            std::fprintf(stderr, "failed to read %llu elements from %s\n",
                         (unsigned long long)n_elem, opt.in_path.c_str());
            return 1;
        }
        std::fclose(f);
    } else {
        std::mt19937 rng(1234);
        std::normal_distribution<float> g;
        for (auto &v : x0) v = {g(rng), g(rng)};
    }

    /* ---- Plan ---------------------------------------------------------- */
    VkFFTConfiguration cfg = {};
    cfg.FFTdim = opt.dims.size();
    for (size_t i = 0; i < opt.dims.size(); ++i) cfg.size[i] = opt.dims[i];
    cfg.numberBatches = opt.batch;
    cfg.device = dev;
    cfg.queue = queue;
    cfg.bufferSize = &bytes;
    cfg.normalize = 0;  /* unnormalised inverse, like cuFFT */
    if (opt.coalesced) cfg.coalescedMemory = opt.coalesced;

    VkFFTApplication app = {};
    const auto t0 = std::chrono::steady_clock::now();
    const VkFFTResult ir = initializeVkFFT(&app, cfg);
    const auto t1 = std::chrono::steady_clock::now();
    if (ir != VKFFT_SUCCESS) {
        std::fprintf(stderr, "initializeVkFFT failed: %d\n", (int)ir);
        return 1;
    }
    std::printf("plan creation %.1f ms\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count());

    /* ---- Correctness: forward (optionally written out), then round trip. */
    std::memcpy(data, x0.data(), bytes);
    if (run(&app, queue, buf, {-1}) < 0) return 1;
    if (!opt.out_path.empty()) {
        FILE *f = std::fopen(opt.out_path.c_str(), "wb");
        if (!f || std::fwrite(data, sizeof(data[0]), n_elem, f) != n_elem) {
            std::fprintf(stderr, "failed to write %s\n", opt.out_path.c_str());
            return 1;
        }
        std::fclose(f);
    }
    if (run(&app, queue, buf, {1}) < 0) return 1;
    double num = 0.0, den = 0.0;
    for (uint64_t i = 0; i < n_elem; ++i) {
        const std::complex<double> r =
            std::complex<double>(data[i]) / double(grid) -
            std::complex<double>(x0[i]);
        num += std::norm(r);
        den += std::norm(std::complex<double>(x0[i]));
    }
    std::printf("round-trip rel. error %.3e\n", std::sqrt(num / den));

    /* ---- Timing: warm up, then medians of the GPU time. ---------------- */
    for (int i = 0; i < 3; ++i) run(&app, queue, buf, {-1, 1});
    std::vector<double> tf, ti, tfi;
    for (int i = 0; i < opt.iters; ++i) {
        tf.push_back(run(&app, queue, buf, {-1}));
        ti.push_back(run(&app, queue, buf, {1}));
        tfi.push_back(run(&app, queue, buf, {-1, 1}));
    }
    std::printf("GPU median ms: forward %.3f  inverse %.3f  fwd+inv %.3f\n",
                median(tf), median(ti), median(tfi));

    deleteVkFFT(&app);
    buf->release();
    queue->release();
    dev->release();
    /* The outer pool is deliberately NOT drained. VkFFT v1.3.4's Metal
     * backend calls release() on autoreleased objects it creates in
     * initializeVkFFT (NS::String::string, commandBuffer(), encoders), so
     * draining the pool that was active during initialisation double-frees
     * them (upstream issue DTolm/VkFFT#226, unmerged fix PR #227). The
     * process exits right after, so leaking the pool is harmless here. */
    (void)pool;
    return 0;
}
