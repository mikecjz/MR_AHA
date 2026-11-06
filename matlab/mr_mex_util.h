/**
 * @file mr_mex_util.h
 * @brief gpuArray helpers of the MR_AHA MATLAB MEX gateways (CUDA backend).
 *
 * Adds the MATLAB GPU C API (`mxGPUArray`) checks to the backend-independent
 * config parsing of `mr_mex_config.h`.
 *
 * MATLAB stores arrays in column-major order, which is the layout MR_AHA
 * documents throughout, so no permutation is ever required.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#ifndef MR_AHA_MR_MEX_UTIL_H
#define MR_AHA_MR_MEX_UTIL_H

#include "mr_mex_config.h"
#include "gpu/mxGPUArray.h"

/**
 * @brief Verify that a gpuArray is single precision and complex.
 * @param[in] a    The array to check.
 * @param[in] name Name used in the diagnostic.
 * @param[in] id   Error identifier.
 */
static void mr_mex_check_complex_single(const mxGPUArray *a, const char *name,
                                        const char *id)
{
    if (mxGPUGetClassID(a) != mxSINGLE_CLASS)
        mexErrMsgIdAndTxt(id, "%s must be a single-precision gpuArray.", name);
    if (mxGPUGetComplexity(a) != mxCOMPLEX)
        mexErrMsgIdAndTxt(id,
                          "%s must be complex. Use complex(%s) in MATLAB if it "
                          "happens to be real-valued.",
                          name, name);
}

/**
 * @brief Verify a gpuArray's element count.
 * @param[in] a        The array to check.
 * @param[in] expected Expected number of elements.
 * @param[in] name     Name used in the diagnostic.
 * @param[in] id       Error identifier.
 */
static void mr_mex_check_numel(const mxGPUArray *a, int64_t expected,
                               const char *name, const char *id)
{
    const mwSize n = mxGPUGetNumberOfElements(a);
    if ((int64_t)n != expected)
        mexErrMsgIdAndTxt(id, "%s has %lld elements but %lld were expected.",
                          name, (long long)n, (long long)expected);
}

#endif /* MR_AHA_MR_MEX_UTIL_H */
