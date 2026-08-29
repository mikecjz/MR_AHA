/**
 * @file mt_shared_stub.cu
 * @brief Anchor translation unit for the `mt_cuda_c` shared library.
 *
 * CMake requires at least one source file per target. The public entry points
 * themselves come from the `mt_cuda` static archive, which is linked with
 * whole-archive semantics so that nothing is stripped.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mt_cuda/mt_cuda.h"

extern "C" {

/**
 * @brief Version of the MT_CUDA shared library, as `major*10000 + minor*100 + patch`.
 * @return The packed version number.
 */
MT_EXPORT int mt_version(void) { return 1 * 10000 + 0 * 100 + 0; }

} /* extern "C" */
