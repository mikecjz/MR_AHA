/**
 * @file calc_mask_mex.cpp
 * @brief MATLAB MEX gateway for the automatic k-space mask (Metal).
 *
 * Usage from MATLAB (host arrays):
 * @code
 *   mask0 = calc_mask_mex(traj, config)              % neighbourhood method
 *   mask0 = calc_mask_mex(kernel, config, 'kernel')  % threshold method
 * @endcode
 *
 * The first form runs ::mr_calc_mask, which on Metal supports the
 * neighbourhood method (no NUFFT). The second form runs
 * ::mr_calc_mask_from_kernel on a unit-weight kernel that `calc_mask.m`
 * computes through Python (mlx-nufft), and implements the threshold method.
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
 * @param[in]  nrhs 2 or 3.
 * @param[in]  prhs `{traj, config}` or `{kernel, config, 'kernel'}`.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 2 && nrhs != 3)
        mexErrMsgIdAndTxt(MR_ID, "Usage: mask0 = calc_mask_mex(traj, config) or "
                                 "calc_mask_mex(kernel, config, 'kernel').");
    if (nlhs > 1) mexErrMsgIdAndTxt(MR_ID, "Too many output arguments.");
    bool from_kernel = false;
    if (nrhs == 3) {
        char mode[16];
        if (!mxIsChar(prhs[2]) || mxGetString(prhs[2], mode, sizeof(mode)) != 0 ||
            strcmp(mode, "kernel") != 0)
            mexErrMsgIdAndTxt(MR_ID, "The third argument must be 'kernel'.");
        from_kernel = true;
    }

    mr_config cfg;
    mr_mex_parse_config(prhs[1], &cfg, MR_ID);
    cfg.is_masked = 0;
    cfg.num_masked = 0;
    const int status = mr_config_validate(&cfg);
    if (status != MR_SUCCESS) mr_mex_fail(MR_ID, status);

    if (!mxIsSingle(prhs[0]) || mxIsComplex(prhs[0]))
        mexErrMsgIdAndTxt(MR_ID, "%s must be a real single array.",
                          from_kernel ? "kernel" : "traj");

    /* Worst case: every location kept. */
    const int64_t cap = mr_pad_size(&cfg);
    int64_t *buf = (int64_t *)mxMalloc((size_t)cap * sizeof(int64_t));
    int64_t n_kept = 0;
    int rc;
    if (from_kernel) {
        if ((int64_t)mxGetNumberOfElements(prhs[0]) != mr_mask_kernel_size(&cfg))
            mexErrMsgIdAndTxt(MR_ID, "kernel has %lld elements but %lld were "
                              "expected.", (long long)mxGetNumberOfElements(prhs[0]),
                              (long long)mr_mask_kernel_size(&cfg));
        rc = mr_calc_mask_from_kernel(&cfg, mxGetSingles(prhs[0]), buf, &n_kept);
    } else {
        const int ndim = (cfg.traj_type == MR_TRAJ_3D) ? 3 : 2;
        const mwSize nd = mxGetNumberOfDimensions(prhs[0]);
        const mwSize *dt = mxGetDimensions(prhs[0]);
        if ((int)nd != 3 || (int)dt[2] != ndim)
            mexErrMsgIdAndTxt(MR_ID, "traj must be [Ntrajs, Nkx, %d].", ndim);
        rc = mr_calc_mask(&cfg, mxGetSingles(prhs[0]), (int64_t)dt[0],
                          (int64_t)dt[1], buf, &n_kept);
    }
    if (rc != MR_SUCCESS) {
        mxFree(buf);
        mr_mex_fail(MR_ID, rc);
    }

    mxArray *out = mxCreateNumericMatrix((mwSize)n_kept, 1, mxINT64_CLASS, mxREAL);
    memcpy(mxGetInt64s(out), buf, (size_t)n_kept * sizeof(int64_t));
    mxFree(buf);
    plhs[0] = out;
}
