/**
 * @file calc_mask_mex.cu
 * @brief MATLAB MEX gateway for the automatic k-space mask.
 *
 * Usage from MATLAB:
 * @code
 *   mask = calc_mask_mex(traj, config)            % int64, 0-based
 * @endcode
 *
 * `traj` is a real single gpuArray `[Ntrajs, Nkx, ndim]` in radians. The
 * result holds the sorted 0-based linear indices of the retained upsampled
 * k-space locations (see ::mr_calc_mask).
 *
 * Prefer the `calc_mask.m` wrapper, which converts to MATLAB's 1-based
 * indices.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_mex_util.h"

#include <cuda_runtime.h>

/** @brief MATLAB error identifier used throughout this gateway. */
#define MR_ID "MR_AHA:calc_mask"

/**
 * @brief MEX entry point.
 *
 * @param[in]  nlhs Number of requested outputs (0 or 1).
 * @param[out] plhs `plhs[0]` receives the mask as a gpuArray.
 * @param[in]  nrhs Number of inputs (2).
 * @param[in]  prhs `{traj, config}`.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* ---- Argument count: exactly (traj, config), at most one output. ---- */
    if (nrhs != 2)
        mexErrMsgIdAndTxt(MR_ID, "Usage: mask = calc_mask_mex(traj, config).");
    if (nlhs > 1) mexErrMsgIdAndTxt(MR_ID, "Too many output arguments.");

    /* Initialize the MathWorks GPU API before any mxGPU* call. */
    mxInitGPU();

    /* ---- Configuration -------------------------------------------------- */
    /* Only traj_type, n_dim, phi2_dim, mask_radius and the execution fields
     * (device, stream, verbose) matter to mr_calc_mask. */
    mr_config cfg;
    mr_mex_parse_config(prhs[1], &cfg, MR_ID);
    /* The mask is what is being computed, so masking is irrelevant here. */
    cfg.is_masked  = 0;
    cfg.num_masked = 0;
    const int status = mr_config_validate(&cfg);
    if (status != MR_SUCCESS) mr_mex_fail(MR_ID, status);

    /* ---- Trajectory ----------------------------------------------------- */
    /* traj must be real single [Ntrajs, Nkx, ndim] with ndim = 3 for a 3D
     * trajectory and 2 for 2D and stack (kz of a stack is Cartesian). The
     * extents are read defensively (nd may be < 3) and validated afterwards. */
    const mxGPUArray *g_traj = mxGPUCreateFromMxArray(prhs[0]);
    if (mxGPUGetClassID(g_traj) != mxSINGLE_CLASS ||
        mxGPUGetComplexity(g_traj) != mxREAL)
        mexErrMsgIdAndTxt(MR_ID, "traj must be a real single-precision gpuArray.");

    const int      ndim_traj = (cfg.traj_type == MR_TRAJ_3D) ? 3 : 2;
    const mwSize  *dt        = mxGPUGetDimensions(g_traj);
    const mwSize   nd        = mxGPUGetNumberOfDimensions(g_traj);
    const int64_t  Ntrajs    = (int64_t)dt[0];
    const int64_t  Nkx       = (nd >= 2) ? (int64_t)dt[1] : 1;
    const int      ndim_in   = (nd >= 3) ? (int)dt[2] : 1;
    mxFree((void *)dt);
    if ((int)nd != 3 || ndim_in != ndim_traj)
        mexErrMsgIdAndTxt(MR_ID, "traj must be a 3-D array [Ntrajs, Nkx, %d].",
                          ndim_traj);
    const float *d_traj = (const float *)mxGPUGetDataReadOnly(g_traj);

    /* ---- Mask: fill a worst-case buffer, then shrink to fit ------------- */
    /* The number of retained locations is only known after the call, so the
     * library writes into a buffer of the worst-case size Mx*My*Mz and
     * reports the count. The trajectory wrapper is no longer needed after
     * the call and is released immediately. */
    mwSize cap = (mwSize)mr_pad_size(&cfg);
    mxGPUArray *g_buf = mxGPUCreateGPUArray(1, &cap, mxINT64_CLASS, mxREAL,
                                            MX_GPU_DO_NOT_INITIALIZE);
    int64_t n_kept = 0;
    int rc = mr_calc_mask(&cfg, d_traj, Ntrajs, Nkx,
                          (int64_t *)mxGPUGetData(g_buf), &n_kept);
    mxGPUDestroyGPUArray(g_traj);
    if (rc != MR_SUCCESS) {
        mxGPUDestroyGPUArray(g_buf);
        mr_mex_fail(MR_ID, rc);
    }

    /* Copy the n_kept sorted 0-based indices into a right-sized [n_kept, 1]
     * int64 gpuArray (device-to-device), so MATLAB does not hold on to the
     * worst-case buffer. With n_kept == 0 the result is an empty column. */
    mwSize dims_out[2] = {(mwSize)n_kept, 1};
    mxGPUArray *g_out = mxGPUCreateGPUArray(2, dims_out, mxINT64_CLASS, mxREAL,
                                            MX_GPU_DO_NOT_INITIALIZE);
    if (n_kept > 0) {
        const cudaError_t ce =
            cudaMemcpy(mxGPUGetData(g_out), mxGPUGetDataReadOnly(g_buf),
                       (size_t)n_kept * sizeof(int64_t), cudaMemcpyDeviceToDevice);
        if (ce != cudaSuccess) {
            mxGPUDestroyGPUArray(g_out);
            mxGPUDestroyGPUArray(g_buf);
            mexErrMsgIdAndTxt(MR_ID, "copying the mask failed: %s",
                              cudaGetErrorString(ce));
        }
    }
    mxGPUDestroyGPUArray(g_buf); /* release the worst-case buffer promptly */
    plhs[0] = mxGPUCreateMxArrayOnGPU(g_out);
    mxGPUDestroyGPUArray(g_out);
}
