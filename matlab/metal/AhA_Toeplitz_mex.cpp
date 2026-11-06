/**
 * @file AhA_Toeplitz_mex.cpp
 * @brief MATLAB MEX gateway for the Toeplitz forward normal operator (Metal).
 *
 * Usage from MATLAB (all arrays ordinary, i.e. host, `single` complex):
 * @code
 *   AhAx = AhA_Toeplitz_mex(x, SEs, phi2_toep, config)
 *   AhAx = AhA_Toeplitz_mex(x, SEs, phi2_toep, config, mask)
 * @endcode
 *
 * Prefer the `AhA_Toeplitz.m` wrapper, which fills in the derived `config`
 * fields and converts a 1-based MATLAB index mask to the 0-based `int64` mask
 * this gateway expects.
 *
 * Apple silicon has unified memory, so there is no gpuArray: the GPU reads
 * MATLAB's own arrays in place (the library wraps them as Metal buffers
 * without copying) and writes the result straight into the output mxArray.
 * The Parallel Computing Toolbox is not needed. Arrays use MATLAB's
 * interleaved complex storage (R2018a API), which is the layout of
 * ::mr_complex.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "../mr_mex_config.h"

/** @brief MATLAB error identifier used throughout this gateway. */
#define MR_ID "MR_AHA:AhA_Toeplitz"

/**
 * @brief Verify that an array is complex single and has @p expected elements.
 * @param[in] a        The array.
 * @param[in] expected Expected number of elements, or -1 to skip the check.
 * @param[in] name     Name used in the diagnostic.
 */
static void mr_check_cs(const mxArray *a, int64_t expected, const char *name)
{
    if (!mxIsSingle(a) || !mxIsComplex(a))
        mexErrMsgIdAndTxt(MR_ID, "%s must be a complex single array.", name);
    if (expected >= 0 && (int64_t)mxGetNumberOfElements(a) != expected)
        mexErrMsgIdAndTxt(MR_ID, "%s has %lld elements but %lld were expected.",
                          name, (long long)mxGetNumberOfElements(a),
                          (long long)expected);
}

/**
 * @brief MEX entry point.
 *
 * @param[in]  nlhs Number of requested outputs (0 or 1).
 * @param[out] plhs `plhs[0]` receives `AhAx`, shaped like `x`.
 * @param[in]  nrhs Number of inputs (4 or 5).
 * @param[in]  prhs `{x, SEs, phi2_toep, config[, mask]}`.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 4 && nrhs != 5)
        mexErrMsgIdAndTxt(MR_ID, "Usage: AhAx = AhA_Toeplitz_mex(x, SEs, "
                                 "phi2_toep, config[, mask]).");
    if (nlhs > 1) mexErrMsgIdAndTxt(MR_ID, "Too many output arguments.");

    mr_config cfg;
    mr_mex_parse_config(prhs[3], &cfg, MR_ID);
    if (mr_mex_field(prhs[3], "n_coils") == NULL) {
        /* One map per coil: the coil count is SEs' trailing dimension. */
        const mwSize nd = mxGetNumberOfDimensions(prhs[1]);
        const mwSize *dims = mxGetDimensions(prhs[1]);
        cfg.n_coils = ((int)nd > cfg.n_spatial) ? (int)dims[cfg.n_spatial] : 1;
    }

    const int64_t *mask = NULL;
    if (cfg.is_masked) {
        if (nrhs != 5)
            mexErrMsgIdAndTxt(MR_ID, "config.is_masked is true, so a mask must "
                                     "be supplied as the fifth argument.");
        if (!mxIsInt64(prhs[4]) || mxIsComplex(prhs[4]))
            mexErrMsgIdAndTxt(MR_ID, "mask must be an int64 array of 0-based "
                                     "linear indices; AhA_Toeplitz.m converts.");
        const int64_t nm = (int64_t)mxGetNumberOfElements(prhs[4]);
        if (cfg.num_masked == 0) cfg.num_masked = nm;
        if (nm != cfg.num_masked)
            mexErrMsgIdAndTxt(MR_ID, "mask has %lld elements but "
                              "config.num_masked is %lld.", (long long)nm,
                              (long long)cfg.num_masked);
        mask = (const int64_t *)mxGetInt64s(prhs[4]);
    }

    const int status = mr_config_validate(&cfg);
    if (status != MR_SUCCESS) mr_mex_fail(MR_ID, status);

    mr_check_cs(prhs[0], mr_x_size(&cfg), "x");
    mr_check_cs(prhs[1], mr_se_size(&cfg), "SEs");
    mr_check_cs(prhs[2], mr_phi2_toep_size(&cfg), "phi2_toep");

    mxArray *out = mxCreateNumericArray(mxGetNumberOfDimensions(prhs[0]),
                                        mxGetDimensions(prhs[0]), mxSINGLE_CLASS,
                                        mxCOMPLEX);
    const int rc = mr_aha_toeplitz(
        &cfg, (const mr_complex *)mxGetComplexSingles(prhs[0]),
        (const mr_complex *)mxGetComplexSingles(prhs[1]),
        (const mr_complex *)mxGetComplexSingles(prhs[2]), mask,
        (mr_complex *)mxGetComplexSingles(out));
    if (rc != MR_SUCCESS) {
        mxDestroyArray(out);
        mr_mex_fail(MR_ID, rc);
    }
    plhs[0] = out;
}
