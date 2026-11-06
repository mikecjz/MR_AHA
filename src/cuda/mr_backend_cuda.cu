/**
 * @file mr_backend_cuda.cu
 * @brief Backend identification and error-state hook of the CUDA backend.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_internal.cuh"

/**
 * @brief Reset the CUDA runtime's per-thread "last error".
 *
 * The library checks kernel launches with `cudaGetLastError`, which returns
 * the most recent error of *any* runtime call on the thread. A non-sticky
 * error that the caller already handled (for example an out-of-memory
 * `cudaMalloc` that CuPy's or MATLAB's allocator recovered from by freeing its
 * cache and retrying) would otherwise be reported as a launch failure of the
 * next library kernel. Sticky (context-corrupting) errors are unaffected: they
 * are reported again by the next runtime call.
 */
void mr_backend_clear_error(void) { (void)cudaGetLastError(); }

extern "C" const char *mr_backend(void) { return "cuda"; }
