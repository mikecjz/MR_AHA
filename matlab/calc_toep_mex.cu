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
 * `int64` gpuArray of 0-based linear indices (`[]` when unmasked). The
 * kernel is computed with ::mr_calc_toep on MATLAB's current GPU. (To build
 * an operator on several GPUs, `mr_aha_op.create_op` computes the kernel on
 * those GPUs itself.)
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
    if (nrhs < 3 || nrhs > 4)
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
    int64_t Ntrajs = 0, Nkx = 0;
    mr_mex_check_traj(g_traj, &cfg, &Ntrajs, &Nkx, MR_ID);

    if (cfg.is_masked) {
        if (nrhs < 4 || mxIsEmpty(prhs[3]))
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

    /* ---- Output ------------------------------------------------------- */
    mwSize dims_out[4];
    const mwSize nd_out = mr_mex_toep_dims(&cfg, dims_out);
    mxGPUArray *g_out = mxGPUCreateGPUArray(nd_out, dims_out, mxSINGLE_CLASS,
                                            mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);

    /* ---- Run ------------------------------------------------------------ */
    const mr_complex *p_phi2 = (const mr_complex *)mxGPUGetDataReadOnly(g_phi2);
    const float      *p_traj = (const float *)mxGPUGetDataReadOnly(g_traj);
    const int64_t    *p_mask =
        g_mask ? (const int64_t *)mxGPUGetDataReadOnly(g_mask) : NULL;
    mr_complex       *p_out  = (mr_complex *)mxGPUGetData(g_out);
    const int rc = mr_calc_toep(&cfg, p_phi2, p_traj, Ntrajs, Nkx, p_mask, p_out);

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
