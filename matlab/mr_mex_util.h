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
    /* MR_AHA is single precision throughout (::mr_complex is two floats), so a
     * double gpuArray would be misread; reject it rather than convert. */
    if (mxGPUGetClassID(a) != mxSINGLE_CLASS)
        mexErrMsgIdAndTxt(id, "%s must be a single-precision gpuArray.", name);
    /* A real gpuArray stores only one float per element; the library reads
     * interleaved (re, im) pairs, so it must be explicitly complex. */
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
    /* Only the total count is checked: the library indexes flat column-major
     * memory, so any reshape with the right number of elements is accepted. */
    const mwSize n = mxGPUGetNumberOfElements(a);
    if ((int64_t)n != expected)
        mexErrMsgIdAndTxt(id, "%s has %lld elements but %lld were expected.",
                          name, (long long)n, (long long)expected);
}

/**
 * @brief Verify the trajectory of ::mr_calc_toep and read its sizes.
 *
 * `traj` must be a real single gpuArray `[Ntrajs, Nkx, ndim]` with `ndim` = 3
 * for ::MR_TRAJ_3D and 2 otherwise.
 *
 * @param[in]  g_traj The trajectory.
 * @param[in]  cfg    Parsed configuration (selects `ndim`).
 * @param[out] Ntrajs Number of trajectories.
 * @param[out] Nkx    Samples per trajectory.
 * @param[in]  id     Error identifier.
 */
static void mr_mex_check_traj(const mxGPUArray *g_traj, const mr_config *cfg,
                              int64_t *Ntrajs, int64_t *Nkx, const char *id)
{
    /* Class: k-space coordinates are real floats (cufinufft single precision). */
    if (mxGPUGetClassID(g_traj) != mxSINGLE_CLASS ||
        mxGPUGetComplexity(g_traj) != mxREAL)
        mexErrMsgIdAndTxt(id, "traj must be a real single-precision gpuArray.");
    /* Shape: exactly three dimensions. 2D and stack trajectories carry
     * (kx, ky) only, since kz of a stack is Cartesian and set by Nz. The
     * dimension vector returned by mxGPUGetDimensions is mxMalloc'ed and must be
     * freed on every path, including before raising an error. */
    const int     ndim_traj = (cfg->traj_type == MR_TRAJ_3D) ? 3 : 2;
    const mwSize  nd        = mxGPUGetNumberOfDimensions(g_traj);
    const mwSize *dt        = mxGPUGetDimensions(g_traj);
    if ((int)nd != 3) {
        mxFree((void *)dt);
        mexErrMsgIdAndTxt(id, "traj must be a 3-D array [Ntrajs, Nkx, %d].",
                          ndim_traj);
    }
    /* Copy the extents out before freeing the dimension vector. */
    *Ntrajs = (int64_t)dt[0];
    *Nkx    = (int64_t)dt[1];
    const int ndim_in = (int)dt[2];
    mxFree((void *)dt);
    /* The coordinate count must match config.traj_type. */
    if (ndim_in != ndim_traj)
        mexErrMsgIdAndTxt(id,
                          "traj's third dimension is %d but %d coordinates are "
                          "required for config.traj_type.",
                          ndim_in, ndim_traj);
}

/**
 * @brief Read the device list of `mr_aha_mex('create', ...)` for
 *        ::mr_op_options::devices.
 *
 * Accepts an `int32` vector of **CUDA ordinals** (0-based; `mr_aha.m`
 * converts the 1-based `gpuDevice` indices of its `'devices'` argument), the
 * character vector `'all'` (every visible device) or `[]` (the primary
 * device only).
 *
 * @param[in]  mx  The MATLAB array.
 * @param[out] out Receives up to ::MR_OP_MAX_DEVICES ordinals.
 * @param[in]  id  Error identifier.
 * @return Number of ordinals for ::mr_op_options::n_devices: 0 for `[]`,
 *         ::MR_OP_ALL_DEVICES for `'all'` (with nothing written to @p out).
 */
static int mr_mex_devices(const mxArray *mx, int *out, const char *id)
{
    /* []: n_devices = 0 means "primary device only" in ::mr_op_options. */
    if (mxIsEmpty(mx)) return 0;
    /* 'all' (or 'ALL'; other casings are rejected): every visible device. The
     * small buffer also makes mxGetString fail for any longer string. */
    if (mxIsChar(mx)) {
        char buf[8];
        if (mxGetString(mx, buf, sizeof(buf)) != 0 ||
            !(strcmp(buf, "all") == 0 || strcmp(buf, "ALL") == 0))
            mexErrMsgIdAndTxt(id, "devices must be 'all', [] or an int32 vector "
                                  "of CUDA ordinals.");
        return MR_OP_ALL_DEVICES;
    }
    /* Numeric list: must already be int32 CUDA ordinals. The .m layer did the
     * 1-based -> 0-based conversion; no range check here, the library
     * validates the ordinals in ::mr_op_create. */
    if (!mxIsInt32(mx) || mxIsComplex(mx))
        mexErrMsgIdAndTxt(id, "devices must be an int32 vector of CUDA ordinals "
                              "(the .m wrappers convert gpuDevice indices).");
    const mwSize n = mxGetNumberOfElements(mx);
    if (n > MR_OP_MAX_DEVICES)
        mexErrMsgIdAndTxt(id, "at most %d devices are supported.",
                          MR_OP_MAX_DEVICES);
    const int32_t *d = mxGetInt32s(mx);
    for (mwSize k = 0; k < n; ++k) out[k] = (int)d[k];
    return (int)n;
}

/**
 * @brief MATLAB shape of `phi2_toep` (see ::mr_calc_toep):
 *        `[Mx, My, T]` (2D), `[Mx, My, Mz, T]` (3D / stack) or
 *        `[num_masked, T]` (masked), `T = (L^2+L)/2`.
 * @param[in]  cfg  Validated configuration.
 * @param[out] dims Receives up to 4 extents.
 * @return Number of dimensions.
 */
static mwSize mr_mex_toep_dims(const mr_config *cfg, mwSize dims[4])
{
    /* The last axis is always the packed triangle of the L x L Gram block. */
    const int64_t ntri = mr_num_tri(cfg->L);
    /* Masked: one row per retained grid location, `[num_masked, T]`. */
    if (cfg->is_masked) {
        dims[0] = (mwSize)cfg->num_masked;
        dims[1] = (mwSize)ntri;
        return 2;
    }
    /* Unmasked: the full upsampled grid `[Mx, My(, Mz)]` followed by T. */
    dims[0] = (mwSize)cfg->phi2_dim[0];
    dims[1] = (mwSize)cfg->phi2_dim[1];
    if (cfg->n_spatial == 3) {
        dims[2] = (mwSize)cfg->phi2_dim[2];
        dims[3] = (mwSize)ntri;
        return 4;
    }
    dims[2] = (mwSize)ntri;
    return 3;
}

#endif /* MR_AHA_MR_MEX_UTIL_H */
