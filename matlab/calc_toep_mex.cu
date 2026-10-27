/**
 * @file calc_toep_mex.cu
 * @brief MATLAB MEX gateway for the Toeplitz-kernel pre-computation.
 *
 * Usage from MATLAB:
 * @code
 *   phi2_toep = calc_toep_mex(phi2, traj, config)
 *   phi2_toep = calc_toep_mex(phi2, traj, config, mask)
 * @endcode
 *
 * `phi2` is a complex single gpuArray of size `[Ntrajs, L, L]` (2D / 3D) or
 * `[Ntrajs, Nkz, L, L]` (stack), `traj` a real single gpuArray of size
 * `[Ntrajs, Nkx, ndim]` holding radians in @f$[-\pi, \pi)@f$, and `mask` an
 * `int64` gpuArray of 0-based linear indices.
 *
 * Prefer the `calc_toep.m` wrapper, which performs the class and index-base
 * conversions.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_mex_util.h"

/** @brief MATLAB error identifier used throughout this gateway. */
#define MR_ID "MR_AHA:calc_toep"

/**
 * @brief MEX entry point.
 *
 * @param[in]  nlhs Number of requested outputs (must be 0 or 1).
 * @param[out] plhs `plhs[0]` receives `phi2_toep` as a gpuArray shaped
 *                  `[Mx, My, Mz, (L^2+L)/2]` or `[num_masked, (L^2+L)/2]`.
 * @param[in]  nrhs Number of inputs (3 or 4).
 * @param[in]  prhs `{phi2, traj, config[, mask]}`.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 3 && nrhs != 4)
        mexErrMsgIdAndTxt(MR_ID,
                          "Usage: phi2_toep = calc_toep_mex(phi2, traj, "
                          "config[, mask]).");
    if (nlhs > 1)
        mexErrMsgIdAndTxt(MR_ID, "Too many output arguments.");

    mxInitGPU();

    /* ---- Configuration -------------------------------------------------- */
    mr_config cfg;
    mr_mex_parse_config(prhs[2], &cfg, MR_ID);

    /* ---- Inputs --------------------------------------------------------- */
    const mxGPUArray *g_phi2 = mxGPUCreateFromMxArray(prhs[0]);
    const mxGPUArray *g_traj = mxGPUCreateFromMxArray(prhs[1]);
    const mxGPUArray *g_mask = NULL;

    mr_mex_check_complex_single(g_phi2, "phi2", MR_ID);
    if (mxGPUGetClassID(g_traj) != mxSINGLE_CLASS ||
        mxGPUGetComplexity(g_traj) != mxREAL)
        mexErrMsgIdAndTxt(MR_ID, "traj must be a real single-precision gpuArray.");

    /* traj is [Ntrajs, Nkx, ndim]; ndim is fixed by the trajectory type. */
    const int      ndim_traj = (cfg.traj_type == MR_TRAJ_3D) ? 3 : 2;
    const mwSize   nd_traj   = mxGPUGetNumberOfDimensions(g_traj);
    const mwSize  *dt        = mxGPUGetDimensions(g_traj);
    if ((int)nd_traj != 3)
        mexErrMsgIdAndTxt(MR_ID,
                          "traj must be a 3-D array [Ntrajs, Nkx, %d].",
                          ndim_traj);
    const int64_t Ntrajs = (int64_t)dt[0];
    const int64_t Nkx    = (int64_t)dt[1];
    const int     ndim_in = (int)dt[2];
    mxFree((void *)dt);
    if (ndim_in != ndim_traj)
        mexErrMsgIdAndTxt(MR_ID,
                          "traj's third dimension is %d but %d coordinates are "
                          "required for config.traj_type.",
                          ndim_in, ndim_traj);

    if (cfg.is_masked) {
        if (nrhs != 4)
            mexErrMsgIdAndTxt(MR_ID,
                              "config.is_masked is true, so a mask must be "
                              "supplied as the fourth argument.");
        g_mask = mxGPUCreateFromMxArray(prhs[3]);
        if (mxGPUGetClassID(g_mask) != mxINT64_CLASS)
            mexErrMsgIdAndTxt(MR_ID,
                              "mask must be an int64 gpuArray of 0-based linear "
                              "indices; calc_toep.m performs the conversion.");
        if (cfg.num_masked == 0)
            cfg.num_masked = (int64_t)mxGPUGetNumberOfElements(g_mask);
    }

    const int status = mr_config_validate(&cfg);
    if (status != MR_SUCCESS) mr_mex_fail(MR_ID, status);

    /* phi2 holds one dense LxL block per trajectory (per kz for a stack). */
    const int64_t Nkz_eff = (cfg.traj_type == MR_TRAJ_STACK) ? cfg.phi2_dim[2] : 1;
    mr_mex_check_numel(g_phi2, Ntrajs * Nkz_eff * cfg.L * cfg.L, "phi2", MR_ID);

    /* ---- Output --------------------------------------------------------- */
    const int64_t ntri = mr_num_tri(cfg.L);
    mwSize  dims_out[4];
    mwSize  nd_out;
    if (cfg.is_masked) {
        nd_out      = 2;
        dims_out[0] = (mwSize)cfg.num_masked;
        dims_out[1] = (mwSize)ntri;
    } else {
        nd_out      = (mwSize)(cfg.n_spatial + 1);
        dims_out[0] = (mwSize)cfg.phi2_dim[0];
        dims_out[1] = (mwSize)cfg.phi2_dim[1];
        if (cfg.n_spatial == 3) {
            dims_out[2] = (mwSize)cfg.phi2_dim[2];
            dims_out[3] = (mwSize)ntri;
        } else {
            dims_out[2] = (mwSize)ntri;
        }
    }
    mxGPUArray *g_out = mxGPUCreateGPUArray(nd_out, dims_out, mxSINGLE_CLASS,
                                            mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);

    /* ---- Run ------------------------------------------------------------ */
    const int rc = mr_calc_toep(
        &cfg,
        (const mr_complex *)mxGPUGetDataReadOnly(g_phi2),
        (const float *)mxGPUGetDataReadOnly(g_traj),
        Ntrajs, Nkx,
        g_mask ? (const int64_t *)mxGPUGetDataReadOnly(g_mask) : NULL,
        (mr_complex *)mxGPUGetData(g_out));

    if (rc != MR_SUCCESS) {
        mxGPUDestroyGPUArray(g_out);
        mxGPUDestroyGPUArray(g_phi2);
        mxGPUDestroyGPUArray(g_traj);
        if (g_mask) mxGPUDestroyGPUArray(g_mask);
        mr_mex_fail(MR_ID, rc);
    }

    plhs[0] = mxGPUCreateMxArrayOnGPU(g_out);

    mxGPUDestroyGPUArray(g_out);
    mxGPUDestroyGPUArray(g_phi2);
    mxGPUDestroyGPUArray(g_traj);
    if (g_mask) mxGPUDestroyGPUArray(g_mask);
}
