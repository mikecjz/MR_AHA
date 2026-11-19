/**
 * @file mr_mex_config.h
 * @brief Backend-independent helpers of the MR_AHA MATLAB MEX gateways.
 *
 * Header-only glue between MATLAB's C API and the plain-C MR_AHA library: it
 * converts a MATLAB `config` struct into an ::mr_config and turns library
 * status codes into MATLAB errors. Used by the CUDA gateways (through
 * `mr_mex_util.h`, which adds the gpuArray checks) and by the Metal gateways
 * in `matlab/metal/`, which work on ordinary host arrays.
 *
 * MATLAB stores arrays in column-major order, which is the layout MR_AHA
 * documents throughout, so no permutation is ever required.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#ifndef MR_AHA_MR_MEX_CONFIG_H
#define MR_AHA_MR_MEX_CONFIG_H

#include "mex.h"
#include "mr_aha/mr_aha.h"

#include <string.h>

/**
 * @brief Raise a MATLAB error carrying the library's own diagnostic.
 * @param[in] id     MATLAB error identifier, e.g. `"MR_AHA:calcToep"`.
 * @param[in] status Status code returned by an MR_AHA function.
 */
static void mr_mex_fail(const char *id, int status)
{
    /* Message = generic status text, numeric code, and the thread-local detail
     * recorded by the library for the failing call. Does not return. */
    mexErrMsgIdAndTxt(id, "%s (%d): %s", mr_status_string(status), status,
                      mr_last_error());
}

/**
 * @brief Fetch a field of a MATLAB struct, or NULL when it is absent/empty.
 * @param[in] s    Struct array (element 0 is used).
 * @param[in] name Field name.
 * @return The field value, or NULL.
 */
static const mxArray *mr_mex_field(const mxArray *s, const char *name)
{
    /* Absent and empty fields are treated alike, so callers can pass `[]` in
     * MATLAB to mean "use the default". */
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
static double mr_mex_scalar(const mxArray *s, const char *name, const char *id)
{
    /* Required: missing or empty is an error; logical is accepted so that
     * flags such as `is_masked = true` work. Only element 0 is read. */
    const mxArray *f = mr_mex_field(s, name);
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
static double mr_mex_scalar_opt(const mxArray *s, const char *name,
                                double fallback)
{
    /* Optional: no class check; mxGetScalar converts any numeric/logical
     * class to double (element 0 only). */
    const mxArray *f = mr_mex_field(s, name);
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
static void mr_mex_dims(const mxArray *s, const char *name, int expected,
                        int out[3], const char *id)
{
    /* Presence, class and length checks: the entry count must equal the
     * number of spatial dimensions of the trajectory type (2 or 3). */
    const mxArray *f = mr_mex_field(s, name);
    if (f == NULL) mexErrMsgIdAndTxt(id, "config.%s is required.", name);
    if (!mxIsNumeric(f))
        mexErrMsgIdAndTxt(id, "config.%s must be numeric.", name);
    const mwSize n = mxGetNumberOfElements(f);
    if ((int)n != expected)
        mexErrMsgIdAndTxt(id, "config.%s must have %d entries for this "
                              "trajectory type, but has %d.",
                          name, expected, (int)n);
    /* Unused trailing extents (e.g. Nz for 2D) default to 1, as ::mr_config
     * requires. */
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
static void mr_mex_string(const mxArray *s, const char *name, char *buf,
                          int cap, const char *id)
{
    /* Copy into @p buf; mxGetString fails when the value does not fit. */
    const mxArray *f = mr_mex_field(s, name);
    if (f == NULL) mexErrMsgIdAndTxt(id, "config.%s is required.", name);
    if (!mxIsChar(f))
        mexErrMsgIdAndTxt(id, "config.%s must be a character vector.", name);
    if (mxGetString(f, buf, cap) != 0)
        mexErrMsgIdAndTxt(id, "config.%s is too long.", name);
    /* ASCII lower-casing so '2D' / '2d', 'Lower' / 'lower' compare equal. */
    for (int i = 0; buf[i]; ++i)
        if (buf[i] >= 'A' && buf[i] <= 'Z') buf[i] = (char)(buf[i] - 'A' + 'a');
}

/**
 * @brief Translate the MATLAB `config` struct into an ::mr_config.
 *
 * Recognized fields (see the repository documentation for the full contract):
 * `traj_type`, `n_dim`, `L`, `triangulation`, `phi2_dim`, `is_masked`, and the
 * optional `num_masked`, `n_coils`, `eps`, `mem_fraction`, `gpu_device_id`,
 * `verbose` and `mask_radius`.
 *
 * `gpu_device_id` defaults to -1, i.e. "use whichever device MATLAB has
 * selected", which is what a gpuArray workflow needs (ignored on Metal).
 *
 * @param[in]  mx_cfg MATLAB struct.
 * @param[out] cfg    Receives the translated configuration.
 * @param[in]  id     Error identifier used for every diagnostic.
 */
static void mr_mex_parse_config(const mxArray *mx_cfg, mr_config *cfg,
                                const char *id)
{
    /* ---- Trajectory type: selects n_spatial and the library defaults ---- */
    if (!mxIsStruct(mx_cfg))
        mexErrMsgIdAndTxt(id, "config must be a scalar struct.");

    char traj[32];
    mr_mex_string(mx_cfg, "traj_type", traj, sizeof(traj), id);

    mr_traj_type tt;
    if (strcmp(traj, "2d") == 0)          tt = MR_TRAJ_2D;
    else if (strcmp(traj, "3d") == 0)     tt = MR_TRAJ_3D;
    else if (strcmp(traj, "stack") == 0)  tt = MR_TRAJ_STACK;
    else {
        mexErrMsgIdAndTxt(id, "config.traj_type must be '2D', '3D' or 'stack'.");
        return;
    }

    /* Start from the library defaults so that every field not exposed to
     * MATLAB (and every optional one left out) has a defined value. */
    const int status = mr_config_defaults(cfg, tt);
    if (status != MR_SUCCESS) mr_mex_fail(id, status);

    /* ---- Required geometry: image size N and upsampled Toeplitz size M --- */
    const int nsp = cfg->n_spatial;
    mr_mex_dims(mx_cfg, "n_dim",    nsp, cfg->n_dim,    id);
    mr_mex_dims(mx_cfg, "phi2_dim", nsp, cfg->phi2_dim, id);

    /* Subspace rank. */
    cfg->L = (int)mr_mex_scalar(mx_cfg, "L", id);

    /* Which packed triangle of each Hermitian L x L block phi2_toep holds. */
    char tri[32];
    mr_mex_string(mx_cfg, "triangulation", tri, sizeof(tri), id);
    if (strcmp(tri, "lower") == 0)      cfg->triangulation = MR_TRI_LOWER;
    else if (strcmp(tri, "upper") == 0) cfg->triangulation = MR_TRI_UPPER;
    else mexErrMsgIdAndTxt(id, "config.triangulation must be 'upper' or 'lower'.");

    /* ---- Masking flag (required) and optional tuning fields --------------
     * num_masked = 0 lets the gateways infer it from the mask length;
     * n_coils = 1 is overridden by gateways that infer it from SEs. */
    cfg->is_masked     = mr_mex_scalar(mx_cfg, "is_masked", id) != 0.0;
    cfg->num_masked    = (int64_t)mr_mex_scalar_opt(mx_cfg, "num_masked", 0);
    cfg->n_coils       = (int)mr_mex_scalar_opt(mx_cfg, "n_coils", 1);
    cfg->eps           = mr_mex_scalar_opt(mx_cfg, "eps", 1e-6);
    cfg->mem_fraction  = mr_mex_scalar_opt(mx_cfg, "mem_fraction", 0.5);
    cfg->verbose       = (int)mr_mex_scalar_opt(mx_cfg, "verbose", 0);
    /* -1 keeps MATLAB's currently selected GPU, which gpuArray data lives on. */
    cfg->gpu_device_id = (int)mr_mex_scalar_opt(mx_cfg, "gpu_device_id", -1);
    cfg->mask_radius =
        (int)mr_mex_scalar_opt(mx_cfg, "mask_radius", MR_DEFAULT_MASK_RADIUS);
    /* Legacy default stream: ordered with MATLAB's own GPU work. */
    cfg->stream        = NULL;
}

#endif /* MR_AHA_MR_MEX_CONFIG_H */
