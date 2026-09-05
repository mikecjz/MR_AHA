/**
 * @file mr_shared_stub.cu
 * @brief Anchor translation unit for the `mr_aha_c` shared library.
 *
 * CMake requires at least one source file per target. The public entry points
 * themselves come from the `mr_aha` static archive, which is linked with
 * whole-archive semantics so that nothing is stripped.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_aha/mr_aha.h"

extern "C" {

/**
 * @brief Version of the MR_AHA shared library, as `major*10000 + minor*100 + patch`.
 * @return The packed version number.
 */
MR_EXPORT int mr_version(void) { return 1 * 10000 + 0 * 100 + 0; }

} /* extern "C" */
