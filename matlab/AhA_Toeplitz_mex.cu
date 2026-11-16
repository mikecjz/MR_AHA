/**
 * @file AhA_Toeplitz_mex.cu
 * @brief MATLAB MEX gateway for the Toeplitz-embedded forward normal operator.
 *
 * Usage from MATLAB (all arrays are `gpuArray` of class `single`, complex):
 * @code
 *   AhAx = AhA_Toeplitz_mex(x, SEs, phi2_toep, config)
 *   AhAx = AhA_Toeplitz_mex(x, SEs, phi2_toep, config, mask)
 * @endcode
 *
 * Prefer the `AhA_Toeplitz.m` wrapper, which fills in the derived `config`
 * fields and converts a 1-based MATLAB index mask to the 0-based `int64`
 * mask this gateway expects.
 *
 * All arrays stay resident on the GPU: inputs are borrowed read-only from
 * MATLAB's gpuArray storage and the output is a freshly created gpuArray, so
 * no host round-trip ever happens.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_mex_util.h"

/** @brief MATLAB error identifier used throughout this gateway. */
#define MR_ID "MR_AHA:AhA_Toeplitz"

/**
 * @brief Deduce the coil count from the trailing dimension of `SEs`.
 *
 * The maps are `[Nx, Ny, Nz, Ncoils]` (`[Nx, Ny, Ncoils]` in 2D): one map per
 * coil, shared by every rank. MATLAB drops trailing singleton dimensions, so a
 * single-coil array simply has no coil axis. `config.n_coils` overrides this.
 *
 * @param[in]     SEs        The sensitivity-map gpuArray.
 * @param[in,out] cfg        Configuration; `n_coils` is filled in unless the
 *                           caller already supplied it.
 * @param[in]     user_coils Non-zero when the caller set `n_coils`.
 */
static void mr_mex_infer_n_coils(const mxGPUArray *SEs, mr_config *cfg,
                                 int user_coils)
{
    /* An explicit config.n_coils wins; it is checked later against numel(SEs). */
    if (user_coils) return;

    /* The coil axis follows the n_spatial spatial axes. If SEs has no axis past
     * them, MATLAB dropped a trailing singleton, i.e. there is one coil. The
     * dimension vector is mxMalloc'ed by the GPU API and must be freed. */
    const mwSize  nd   = mxGPUGetNumberOfDimensions(SEs);
    const mwSize *dims = mxGPUGetDimensions(SEs);
    const int     nsp  = cfg->n_spatial;

    cfg->n_coils = ((int)nd > nsp) ? (int)dims[nsp] : 1;
    mxFree((void *)dims);
}

/**
 * @brief MEX entry point.
 *
 * @param[in]  nlhs Number of requested outputs (must be 0 or 1).
 * @param[out] plhs `plhs[0]` receives `AhAx`, a gpuArray shaped like `x`.
 * @param[in]  nrhs Number of inputs (4 or 5).
 * @param[in]  prhs `{x, SEs, phi2_toep, config[, mask]}`.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* ---- Argument count ------------------------------------------------
     * 4 inputs for an unmasked kernel, 5 when a mask accompanies a masked
     * kernel; at most one output. */
    if (nrhs != 4 && nrhs != 5)
        mexErrMsgIdAndTxt(MR_ID,
                          "Usage: AhAx = AhA_Toeplitz_mex(x, SEs, phi2_toep, "
                          "config[, mask]).");
    if (nlhs > 1)
        mexErrMsgIdAndTxt(MR_ID, "Too many output arguments.");

    /* mxInitGPU must run before any other MathWorks GPU API call. */
    mxInitGPU();

    /* ---- Configuration ------------------------------------------------- */
    /* Translate the MATLAB config struct into an mr_config (library defaults
     * for the trajectory type, then the struct's fields). Whether n_coils was
     * given explicitly is recorded so it is not overwritten by inference. */
    mr_config cfg;
    mr_mex_parse_config(prhs[3], &cfg, MR_ID);
    const int user_coils = (mr_mex_field(prhs[3], "n_coils") != NULL);

    /* ---- Inputs --------------------------------------------------------- */
    /* Wrap the inputs as read-only mxGPUArrays. For gpuArray inputs this
     * borrows MATLAB's device memory without a copy; a host array would be
     * uploaded. Every wrapper must be released with mxGPUDestroyGPUArray. */
    const mxGPUArray *g_x    = mxGPUCreateFromMxArray(prhs[0]);
    const mxGPUArray *g_SEs  = mxGPUCreateFromMxArray(prhs[1]);
    const mxGPUArray *g_toep = mxGPUCreateFromMxArray(prhs[2]);
    const mxGPUArray *g_mask = NULL;

    /* The library is single precision complex throughout (mr_complex), so
     * every data array must be complex single; real arrays are rejected
     * rather than silently promoted. */
    mr_mex_check_complex_single(g_x,    "x",         MR_ID);
    mr_mex_check_complex_single(g_SEs,  "SEs",       MR_ID);
    mr_mex_check_complex_single(g_toep, "phi2_toep", MR_ID);

    /* Coil count from the trailing axis of SEs unless config.n_coils is set. */
    mr_mex_infer_n_coils(g_SEs, &cfg, user_coils);

    /* ---- Mask -----------------------------------------------------------
     * A masked phi2_toep is stored only at the retained grid locations, so the
     * operator needs their 0-based linear indices (int64). num_masked defaults
     * to numel(mask) and must agree with it when given. A mask passed with
     * is_masked false is ignored with a warning rather than an error. */
    if (cfg.is_masked) {
        if (nrhs != 5)
            mexErrMsgIdAndTxt(MR_ID,
                              "config.is_masked is true, so a mask must be "
                              "supplied as the fifth argument.");
        g_mask = mxGPUCreateFromMxArray(prhs[4]);
        if (mxGPUGetClassID(g_mask) != mxINT64_CLASS)
            mexErrMsgIdAndTxt(MR_ID,
                              "mask must be an int64 gpuArray of 0-based linear "
                              "indices; AhA_Toeplitz.m performs the conversion.");
        if (cfg.num_masked == 0)
            cfg.num_masked = (int64_t)mxGPUGetNumberOfElements(g_mask);
        if ((int64_t)mxGPUGetNumberOfElements(g_mask) != cfg.num_masked)
            mexErrMsgIdAndTxt(MR_ID,
                              "mask has %lld elements but config.num_masked is "
                              "%lld.",
                              (long long)mxGPUGetNumberOfElements(g_mask),
                              (long long)cfg.num_masked);
    } else if (nrhs == 5 && !mxIsEmpty(prhs[4])) {
        mexWarnMsgIdAndTxt(MR_ID ":unusedMask",
                           "A mask was supplied but config.is_masked is false; "
                           "the mask is ignored.");
    }

    /* ---- Validation -----------------------------------------------------
     * Check the completed config for internal consistency, then verify that
     * each array holds exactly the number of elements the config implies
     * (mr_x_size, mr_se_size, mr_phi2_toep_size). Only numel is checked, so
     * any reshaping with the same linear layout is accepted. */
    const int status = mr_config_validate(&cfg);
    if (status != MR_SUCCESS) mr_mex_fail(MR_ID, status);

    mr_mex_check_numel(g_x,    mr_x_size(&cfg),          "x",         MR_ID);
    mr_mex_check_numel(g_SEs,  mr_se_size(&cfg),         "SEs",       MR_ID);
    mr_mex_check_numel(g_toep, mr_phi2_toep_size(&cfg),  "phi2_toep", MR_ID);

    /* ---- Output: same shape as x --------------------------------------- */
    /* Allocate the result with x's dimensions. It is left uninitialised
     * because the library fully overwrites the output. */
    const mwSize  nd_x   = mxGPUGetNumberOfDimensions(g_x);
    const mwSize *dims_x = mxGPUGetDimensions(g_x);
    mxGPUArray *g_out = mxGPUCreateGPUArray(nd_x, dims_x, mxSINGLE_CLASS,
                                            mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    mxFree((void *)dims_x);

    /* ---- Run ------------------------------------------------------------ */
    /* One-shot call: create a plan, apply A^H A to x, destroy the plan. Raw
     * device pointers are passed straight through; the mask pointer is NULL
     * for an unmasked kernel. */
    const int rc = mr_aha_toeplitz(
        &cfg,
        (const mr_complex *)mxGPUGetDataReadOnly(g_x),
        (const mr_complex *)mxGPUGetDataReadOnly(g_SEs),
        (const mr_complex *)mxGPUGetDataReadOnly(g_toep),
        g_mask ? (const int64_t *)mxGPUGetDataReadOnly(g_mask) : NULL,
        (mr_complex *)mxGPUGetData(g_out));

    /* On failure release every wrapper before mr_mex_fail, which raises a
     * MATLAB error (longjmp) and never returns. */
    if (rc != MR_SUCCESS) {
        mxGPUDestroyGPUArray(g_out);
        mxGPUDestroyGPUArray(g_x);
        mxGPUDestroyGPUArray(g_SEs);
        mxGPUDestroyGPUArray(g_toep);
        if (g_mask) mxGPUDestroyGPUArray(g_mask);
        mr_mex_fail(MR_ID, rc);
    }

    /* Hand the output to MATLAB as a gpuArray (no copy), then release the C
     * wrappers; the gpuArray keeps the underlying device memory alive. */
    plhs[0] = mxGPUCreateMxArrayOnGPU(g_out);

    mxGPUDestroyGPUArray(g_out);
    mxGPUDestroyGPUArray(g_x);
    mxGPUDestroyGPUArray(g_SEs);
    mxGPUDestroyGPUArray(g_toep);
    if (g_mask) mxGPUDestroyGPUArray(g_mask);
}
