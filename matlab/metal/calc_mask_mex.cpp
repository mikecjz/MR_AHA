/**
 * @file calc_mask_mex.cpp
 * @brief MATLAB MEX gateway for the automatic k-space mask (Metal).
 *
 * Usage from MATLAB (host arrays):
 * @code
 *   mask0 = calc_mask_mex(traj, config)
 * @endcode
 *
 * Runs ::mr_calc_mask, which needs no NUFFT and therefore runs entirely in
 * the library on Metal.
 * The result is an `int64` column of sorted **0-based** indices; `calc_mask.m`
 * adds one.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "../mr_mex_config.h"

#include <string.h>

/** @brief MATLAB error identifier used throughout this gateway. */
#define MR_ID "MR_AHA:calc_mask"

/**
 * @brief MEX entry point.
 *
 * @param[in]  nlhs Number of requested outputs (0 or 1).
 * @param[out] plhs `plhs[0]` receives the 0-based `int64` mask column.
 * @param[in]  nrhs 2.
 * @param[in]  prhs `{traj, config}`.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 2)
        mexErrMsgIdAndTxt(MR_ID, "Usage: mask0 = calc_mask_mex(traj, config).");
    if (nlhs > 1) mexErrMsgIdAndTxt(MR_ID, "Too many output arguments.");

    mr_config cfg;
    mr_mex_parse_config(prhs[1], &cfg, MR_ID);
    cfg.is_masked = 0;
    cfg.num_masked = 0;
    const int status = mr_config_validate(&cfg);
    if (status != MR_SUCCESS) mr_mex_fail(MR_ID, status);

    if (!mxIsSingle(prhs[0]) || mxIsComplex(prhs[0]))
        mexErrMsgIdAndTxt(MR_ID, "traj must be a real single array.");
    const int ndim = (cfg.traj_type == MR_TRAJ_3D) ? 3 : 2;
    const mwSize nd = mxGetNumberOfDimensions(prhs[0]);
    const mwSize *dt = mxGetDimensions(prhs[0]);
    if ((int)nd != 3 || (int)dt[2] != ndim)
        mexErrMsgIdAndTxt(MR_ID, "traj must be [Ntrajs, Nkx, %d].", ndim);

    /* Worst case: every location kept. */
    const int64_t cap = mr_pad_size(&cfg);
    int64_t *buf = (int64_t *)mxMalloc((size_t)cap * sizeof(int64_t));
    int64_t n_kept = 0;
    const int rc = mr_calc_mask(&cfg, mxGetSingles(prhs[0]), (int64_t)dt[0],
                                (int64_t)dt[1], buf, &n_kept);
    if (rc != MR_SUCCESS) {
        mxFree(buf);
        mr_mex_fail(MR_ID, rc);
    }

    mxArray *out = mxCreateNumericMatrix((mwSize)n_kept, 1, mxINT64_CLASS, mxREAL);
    memcpy(mxGetInt64s(out), buf, (size_t)n_kept * sizeof(int64_t));
    mxFree(buf);
    plhs[0] = out;
}
