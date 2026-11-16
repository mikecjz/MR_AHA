/**
 * @file mr_backend_mex.c
 * @brief MATLAB MEX gateway reporting the library's GPU backend.
 *
 * @code
 *   b = mr_backend_mex()   % 'cuda' or 'metal'
 * @endcode
 *
 * Built for both backends; the `.m` wrappers call it (through `mr_backend.m`)
 * to choose between gpuArray data (CUDA) and ordinary host arrays (Metal,
 * Apple silicon unified memory).
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mex.h"
#include "mr_aha/mr_aha.h"

/**
 * @brief MEX entry point: `plhs[0] = mr_backend()` as a char row vector.
 * @param[in]  nlhs Number of outputs (0 or 1).
 * @param[out] plhs Receives the backend name.
 * @param[in]  nrhs Must be 0.
 * @param[in]  prhs Unused.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* No inputs; at most one output. prhs is unused. */
    (void)prhs;
    if (nrhs != 0) mexErrMsgIdAndTxt("MR_AHA:backend", "No inputs expected.");
    if (nlhs > 1) mexErrMsgIdAndTxt("MR_AHA:backend", "Too many outputs.");
    /* mr_backend() returns a static string naming the backend the linked
     * library was built for; copy it into a MATLAB char array. */
    plhs[0] = mxCreateString(mr_backend());
}
