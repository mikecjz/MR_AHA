/**
 * @file mt_mex_util.h
 * @brief Shared helpers for the MT_CUDA MATLAB MEX gateways.
 *
 * Header-only glue between the MATLAB GPU C API (`mxGPUArray`) and the plain-C
 * MT_CUDA library. It converts a MATLAB `config` struct into an ::mt_config,
 * validates gpuArray classes, and turns library status codes into MATLAB
 * errors.
 *
 * MATLAB stores arrays in column-major order, which is the layout MT_CUDA
 * documents throughout, so no permutation is ever required.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#ifndef MT_CUDA_MT_MEX_UTIL_H
#define MT_CUDA_MT_MEX_UTIL_H

#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "mt_cuda/mt_cuda.h"

#include <string.h>

/**
 * @brief Raise a MATLAB error carrying the library's own diagnostic.
 * @param[in] id     MATLAB error identifier, e.g. `"MT_CUDA:calcToep"`.
 * @param[in] status Status code returned by an MT_CUDA function.
 */
static void mt_mex_fail(const char *id, int status)
{
    mexErrMsgIdAndTxt(id, "%s (%d): %s", mt_status_string(status), status,
                      mt_last_error());
}

/**
 * @brief Fetch a field of a MATLAB struct, or NULL when it is absent/empty.
 * @param[in] s    Struct array (element 0 is used).
 * @param[in] name Field name.
 * @return The field value, or NULL.
 */
static const mxArray *mt_mex_field(const mxArray *s, const char *name)
{
    if (!mxIsStruct(s)) return NULL;
    const mxArray *f = mxGetField(s, 0, name);
    if (f == NULL || mxIsEmpty(f)) return NULL;
    return f;
}

/**
 * @brief Read a required numeric scalar field.
 * @param[in] s    Config struct.
 * @param[in] name Field name.
 * @param[in] id   Error identifier used when the field is missing.
 * @return The field value as a double.
 */
static double mt_mex_scalar(const mxArray *s, const char *name, const char *id)
{
    const mxArray *f = mt_mex_field(s, name);
    if (f == NULL)
        mexErrMsgIdAndTxt(id, "config.%s is required.", name);
    if (!mxIsNumeric(f) && !mxIsLogical(f))
        mexErrMsgIdAndTxt(id, "config.%s must be numeric or logical.", name);
    return mxGetScalar(f);
}

/**
 * @brief Read an optional numeric scalar field.
 * @param[in] s        Config struct.
 * @param[in] name     Field name.
 * @param[in] fallback Value returned when the field is absent.
 * @return The field value, or @p fallback.
 */
static double mt_mex_scalar_opt(const mxArray *s, const char *name,
                                double fallback)
{
    const mxArray *f = mt_mex_field(s, name);
    if (f == NULL) return fallback;
    return mxGetScalar(f);
}

/**
 * @brief Read a required list-of-dimensions field into a length-3 array.
 *
 * Missing trailing entries are set to 1, matching MATLAB's habit of dropping
 * trailing singleton dimensions.
 *
 * @param[in]  s        Config struct.
 * @param[in]  name     Field name, e.g. `"n_dim"`.
 * @param[in]  expected Number of entries the field must contain.
 * @param[out] out      Receives `[d0, d1, d2]`.
 * @param[in]  id       Error identifier.
 */
static void mt_mex_dims(const mxArray *s, const char *name, int expected,
                        int out[3], const char *id)
{
    const mxArray *f = mt_mex_field(s, name);
    if (f == NULL) mexErrMsgIdAndTxt(id, "config.%s is required.", name);
    if (!mxIsNumeric(f))
        mexErrMsgIdAndTxt(id, "config.%s must be numeric.", name);
    const mwSize n = mxGetNumberOfElements(f);
    if ((int)n != expected)
        mexErrMsgIdAndTxt(id, "config.%s must have %d entries for this "
                              "trajectory type, but has %d.",
                          name, expected, (int)n);
    out[0] = out[1] = out[2] = 1;
    /* Read element-wise via a double so that any numeric class is accepted. */
    for (mwSize i = 0; i < n; ++i) {
        double v;
        if (mxIsDouble(f))       v = mxGetDoubles(f)[i];
        else if (mxIsSingle(f))  v = (double)mxGetSingles(f)[i];
        else if (mxIsInt32(f))   v = (double)mxGetInt32s(f)[i];
        else if (mxIsInt64(f))   v = (double)mxGetInt64s(f)[i];
        else if (mxIsUint32(f))  v = (double)mxGetUint32s(f)[i];
        else if (mxIsUint64(f))  v = (double)mxGetUint64s(f)[i];
        else {
            mexErrMsgIdAndTxt(id, "config.%s has an unsupported numeric class.",
                              name);
            return;
        }
        out[i] = (int)v;
    }
}

/**
 * @brief Read a required char field and compare it case-insensitively.
 * @param[in]  s    Config struct.
 * @param[in]  name Field name.
 * @param[out] buf  Receives the lower-cased string.
 * @param[in]  cap  Capacity of @p buf.
 * @param[in]  id   Error identifier.
 */
static void mt_mex_string(const mxArray *s, const char *name, char *buf,
                          int cap, const char *id)
{
    const mxArray *f = mt_mex_field(s, name);
    if (f == NULL) mexErrMsgIdAndTxt(id, "config.%s is required.", name);
    if (!mxIsChar(f))
        mexErrMsgIdAndTxt(id, "config.%s must be a character vector.", name);
    if (mxGetString(f, buf, cap) != 0)
        mexErrMsgIdAndTxt(id, "config.%s is too long.", name);
    for (int i = 0; buf[i]; ++i)
        if (buf[i] >= 'A' && buf[i] <= 'Z') buf[i] = (char)(buf[i] - 'A' + 'a');
}

/**
 * @brief Translate the MATLAB `config` struct into an ::mt_config.
 *
 * Recognised fields (see the repository documentation for the full contract):
 * `traj_type`, `n_dim`, `L`, `triangulation`, `phi2_dim`, `is_masked`, and the
 * optional `num_masked`, `n_coils`, `eps`, `mem_fraction`, `gpu_device_id`
 * and `verbose`.
 *
 * `gpu_device_id` defaults to -1, i.e. "use whichever device MATLAB has
 * selected", which is what a gpuArray workflow needs.
 *
 * @param[in]  mx_cfg MATLAB struct.
 * @param[out] cfg    Receives the translated configuration.
 * @param[in]  id     Error identifier used for every diagnostic.
 */
static void mt_mex_parse_config(const mxArray *mx_cfg, mt_config *cfg,
                                const char *id)
{
    if (!mxIsStruct(mx_cfg))
        mexErrMsgIdAndTxt(id, "config must be a scalar struct.");

    char traj[32];
    mt_mex_string(mx_cfg, "traj_type", traj, sizeof(traj), id);

    mt_traj_type tt;
    if (strcmp(traj, "2d") == 0)          tt = MT_TRAJ_2D;
    else if (strcmp(traj, "3d") == 0)     tt = MT_TRAJ_3D;
    else if (strcmp(traj, "stack") == 0)  tt = MT_TRAJ_STACK;
    else {
        mexErrMsgIdAndTxt(id, "config.traj_type must be '2D', '3D' or 'stack'.");
        return;
    }

    const int status = mt_config_defaults(cfg, tt);
    if (status != MT_SUCCESS) mt_mex_fail(id, status);

    const int nsp = cfg->n_spatial;
    mt_mex_dims(mx_cfg, "n_dim",    nsp, cfg->n_dim,    id);
    mt_mex_dims(mx_cfg, "phi2_dim", nsp, cfg->phi2_dim, id);

    cfg->L = (int)mt_mex_scalar(mx_cfg, "L", id);

    char tri[32];
    mt_mex_string(mx_cfg, "triangulation", tri, sizeof(tri), id);
    if (strcmp(tri, "lower") == 0)      cfg->triangulation = MT_TRI_LOWER;
    else if (strcmp(tri, "upper") == 0) cfg->triangulation = MT_TRI_UPPER;
    else mexErrMsgIdAndTxt(id, "config.triangulation must be 'upper' or 'lower'.");

    cfg->is_masked     = mt_mex_scalar(mx_cfg, "is_masked", id) != 0.0;
    cfg->num_masked    = (int64_t)mt_mex_scalar_opt(mx_cfg, "num_masked", 0);
    cfg->n_coils       = (int)mt_mex_scalar_opt(mx_cfg, "n_coils", 1);
    cfg->eps           = mt_mex_scalar_opt(mx_cfg, "eps", 1e-6);
    cfg->mem_fraction  = mt_mex_scalar_opt(mx_cfg, "mem_fraction", 0.5);
    cfg->verbose       = (int)mt_mex_scalar_opt(mx_cfg, "verbose", 0);
    /* -1 keeps MATLAB's currently selected GPU, which gpuArray data lives on. */
    cfg->gpu_device_id = (int)mt_mex_scalar_opt(mx_cfg, "gpu_device_id", -1);
    cfg->stream        = NULL;
}

/**
 * @brief Verify that a gpuArray is single precision and complex.
 * @param[in] a    The array to check.
 * @param[in] name Name used in the diagnostic.
 * @param[in] id   Error identifier.
 */
static void mt_mex_check_complex_single(const mxGPUArray *a, const char *name,
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
static void mt_mex_check_numel(const mxGPUArray *a, int64_t expected,
                               const char *name, const char *id)
{
    const mwSize n = mxGPUGetNumberOfElements(a);
    if ((int64_t)n != expected)
        mexErrMsgIdAndTxt(id, "%s has %lld elements but %lld were expected.",
                          name, (long long)n, (long long)expected);
}

#endif /* MT_CUDA_MT_MEX_UTIL_H */
