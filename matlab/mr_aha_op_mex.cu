/**
 * @file mr_aha_op_mex.cu
 * @brief MATLAB MEX gateway of the multi-GPU @f$A^H A@f$ operator class
 *        `mr_aha_op` (CUDA backend).
 *
 * This gateway owns the multi-GPU plans (::mr_mgpu_plan) behind the
 * `mr_aha_op` handle class and is driven by a command string:
 * @code
 *   h    = mr_aha_op_mex('create',  SEs, phi2_toep, config, mask, devices, share)
 *   AhAx = mr_aha_op_mex('apply',   h, x)
 *   info = mr_aha_op_mex('info',    h)
 *          mr_aha_op_mex('destroy', h)
 *   n    = mr_aha_op_mex('count')
 * @endcode
 * Prefer the `mr_aha_op.m` class, which fills in the derived `config` fields,
 * converts a 1-based MATLAB mask and 1-based `gpuDevice` indices, and destroys
 * the plan when the object is deleted or cleared.
 *
 * - `create`: all arrays are complex single gpuArrays on MATLAB's current
 *   device (the *primary* device), except `mask`: an `int64` gpuArray of
 *   0-based linear indices, or `[]` when `config.is_masked` is false.
 *   `devices` is an `int32` vector of **CUDA ordinals** (0-based) or `[]` for
 *   every visible device; the primary device always takes part. `share` is a
 *   logical scalar selecting ::MR_MGPU_SHARE_PRIMARY. Returns the plan handle
 *   as a `uint64` scalar.
 * - `apply`: `x` is a complex single gpuArray of ::mr_x_size elements;
 *   returns a new gpuArray of the same shape.
 * - `info`: returns a struct with the work distribution (CUDA ordinals,
 *   0-based coil indices; the class converts both to 1-based).
 * - `destroy`: releases a plan. Unknown or already destroyed handles are
 *   ignored, so the call is idempotent.
 * - `count`: number of live plans (used by the tests).
 *
 * ## Lifetime and safety
 *
 * - **Registry.** Every live plan is recorded in a registry kept by this MEX
 *   file. A handle that is not in the registry (destroyed, never created, or
 *   forged) is rejected by `apply` / `info` with a clean MATLAB error instead
 *   of being dereferenced.
 * - **mexLock.** While at least one plan is alive the MEX file is locked, so
 *   `clear mex` / `clear functions` / `clear all` cannot unload the code and
 *   the registry that the live plans and `mr_aha_op` objects depend on. Once
 *   the last plan is destroyed the MEX file is unlocked again. `clear all`
 *   also clears the variables, which deletes the `mr_aha_op` objects (and
 *   hence their plans) first.
 * - **mexAtExit.** When MATLAB exits (or unloads the unlocked MEX file), any
 *   plan still alive is destroyed.
 * - **Stale plans.** A plan whose primary device was reset is never passed to
 *   ::mr_mgpu_plan_destroy (its stream handles on that device are dead and
 *   the CUDA driver crashes on them); `destroy` releases it with
 *   ::mr_mgpu_plan_abandon, which frees its allocations on the *other*
 *   devices only, and warns (see ::mr_op_release).
 * - **Device changes.** A plan is bound to the device that was current when
 *   it was created. MATLAB invalidates all gpuArrays (and destroys the CUDA
 *   context of the device, which frees the plan's allocations there) on
 *   `reset(gpuDevice)` or when another device is selected with
 *   `gpuDevice(k)`. To detect this, every registry entry holds a one-element
 *   *sentinel* gpuArray created on the primary device; `apply` refuses to run
 *   when the current device is not the primary device or when
 *   `mxGPUIsValidGPUData` reports the sentinel as invalid, so a stale plan
 *   raises a MATLAB error instead of touching freed device memory.
 * - **Shared arrays.** With `share` true (::MR_MGPU_SHARE_PRIMARY) the plan
 *   reads `SEs`, `phi2_toep` and `mask` in place on the primary device. The
 *   registry entry then holds `mxGPUArray` references to these arrays until
 *   the plan is destroyed, so MATLAB cannot free (or recycle) their storage
 *   even when the caller clears its variables or passed temporaries (e.g. the
 *   converted mask). gpuArrays are copy-on-write, so later assignments to
 *   the caller's variables never modify the shared storage.
 *
 * ## Streams
 *
 * Like `AhA_Toeplitz_mex`, the configuration's ::mr_config::stream is left
 * NULL, i.e. the legacy default stream of the primary device, which is
 * ordered with MATLAB's own GPU work (the contract of MathWorks' gpuArray MEX
 * API). ::mr_mgpu_plan_execute is asynchronous: it orders the other devices
 * after the legacy stream with an event and makes the legacy stream wait for
 * their partial results, so the returned gpuArray is safe to use from MATLAB
 * without a host synchronisation here. The library saves and restores the
 * current CUDA device in every call, so MATLAB's device selection is never
 * changed by this gateway.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_mex_util.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <string.h>
#include <vector>

/** @brief MATLAB error identifier used throughout this gateway. */
#define MR_ID "MR_AHA:mr_aha_op"

/* ------------------------------------------------------------------------ */
/* Registry of live plans                                                   */
/* ------------------------------------------------------------------------ */

/**
 * @brief One live multi-GPU plan and the MATLAB-side state it depends on.
 */
struct mr_op_entry {
    mr_mgpu_plan      plan;     /**< The library plan (never NULL).            */
    mr_config         cfg;      /**< Configuration the plan was created with;
                                     gives the expected size of `x`.           */
    int               primary;  /**< CUDA ordinal of the primary device.       */
    int               shared;   /**< Non-zero with ::MR_MGPU_SHARE_PRIMARY.     */
    mxArray          *sentinel; /**< Persistent one-element gpuArray on the
                                     primary device; becomes invalid when
                                     MATLAB resets or deselects that device.   */
    const mxGPUArray *keep[3];  /**< With @ref shared: references to `SEs`,
                                     `phi2_toep` and `mask` (or NULL) that keep
                                     their storage alive until destroy.        */
};

/** @brief All live plans of this MEX file. */
static std::vector<mr_op_entry *> g_entries;

/** @brief Non-zero once ::mr_op_at_exit has been registered. */
static int g_atexit_registered = 0;

/**
 * @brief Find the registry entry of a plan handle.
 * @param[in] h Handle as returned by `create` (the plan pointer).
 * @return Index into ::g_entries, or -1 when @p h is not a live plan.
 */
static int mr_op_find(uint64_t h)
{
    for (size_t i = 0; i < g_entries.size(); ++i)
        if ((uint64_t)(uintptr_t)g_entries[i]->plan == h) return (int)i;
    return -1;
}

/**
 * @brief Non-zero when the primary device of @p e is still the one the plan
 *        was created on, i.e. MATLAB has neither reset nor deselected it.
 */
static int mr_op_device_valid(const mr_op_entry *e)
{
    return e->sentinel != NULL && mxGPUIsValidGPUData(e->sentinel);
}

/**
 * @brief Release one registry entry: the plan, the kept arrays and the
 *        sentinel. Does not touch ::g_entries or the MEX lock.
 *
 * When the primary device was reset or deselected, MATLAB has already
 * destroyed its CUDA context and with it everything the plan had allocated
 * there, including the plan's stream and events on that device. Passing such a
 * plan to ::mr_mgpu_plan_destroy is not safe: it synchronises those stale
 * stream handles, which crashes inside the CUDA driver (observed with
 * R2022b: segmentation violation in `cuStreamSynchronize`). A stale plan is
 * therefore released with ::mr_mgpu_plan_abandon, which frees what it holds on
 * the other devices (the `phi2_toep` / mask replicas, their maps and buffers)
 * and skips the primary device; a warning says that the device was reset. The
 * MATLAB-side references (kept arrays, sentinel) are always released.
 *
 * @param[in] e Entry to release; freed on return.
 * @return 1 when the plan was stale (primary device reset), else 0.
 */
static int mr_op_release(mr_op_entry *e)
{
    const int stale = !mr_op_device_valid(e);
    if (stale)
        mr_mgpu_plan_abandon(e->plan);
    else
        mr_mgpu_plan_destroy(e->plan);
    for (int k = 0; k < 3; ++k)
        if (e->keep[k]) mxGPUDestroyGPUArray(e->keep[k]);
    if (e->sentinel) mxDestroyArray(e->sentinel);
    mxFree(e);
    return stale;
}

/**
 * @brief Remove entry @p i from the registry, release it and unlock the MEX
 *        file when it was the last one.
 * @param[in] i Index into ::g_entries.
 */
static void mr_op_remove(int i)
{
    mr_op_entry *e = g_entries[(size_t)i];
    const int primary = e->primary;
    g_entries.erase(g_entries.begin() + i);
    const int stale = mr_op_release(e);
    if (g_entries.empty() && mexIsLocked()) mexUnlock();
    if (stale)
        mexWarnMsgIdAndTxt(MR_ID ":staleDestroy",
                           "gpuDevice(%d) was reset or deselected while this "
                           "mr_aha_op existed; the operator had become unusable "
                           "and has been released.", primary + 1);
}

/**
 * @brief mexAtExit handler: destroy every plan still alive when MATLAB exits
 *        or the MEX file is unloaded.
 */
static void mr_op_at_exit(void)
{
    while (!g_entries.empty()) {
        mr_op_entry *e = g_entries.back();
        g_entries.pop_back();
        (void)mr_op_release(e); /* stale plans are skipped (see above) */
    }
    if (mexIsLocked()) mexUnlock();
}

/**
 * @brief Read a `uint64` plan handle argument and look it up.
 * @param[in] a   The MATLAB argument.
 * @param[in] cmd Command name used in diagnostics.
 * @return The registry entry; raises a MATLAB error when the handle is not a
 *         live plan.
 */
static mr_op_entry *mr_op_lookup(const mxArray *a, const char *cmd)
{
    if (!mxIsUint64(a) || mxGetNumberOfElements(a) != 1 || mxIsComplex(a))
        mexErrMsgIdAndTxt(MR_ID ":badHandle",
                          "%s: the plan handle must be a uint64 scalar.", cmd);
    const uint64_t h = *mxGetUint64s(a);
    const int i = mr_op_find(h);
    if (i < 0)
        mexErrMsgIdAndTxt(MR_ID ":invalidHandle",
                          "%s: this mr_aha_op has been destroyed (or the "
                          "handle is not a live plan).", cmd);
    return g_entries[(size_t)i];
}

/**
 * @brief Deduce the coil count from the trailing dimension of `SEs` unless
 *        `config.n_coils` was given (same rule as `AhA_Toeplitz_mex`).
 * @param[in]     SEs        Sensitivity maps.
 * @param[in,out] cfg        Configuration; `n_coils` is filled in.
 * @param[in]     user_coils Non-zero when the caller set `n_coils`.
 */
static void mr_op_infer_n_coils(const mxGPUArray *SEs, mr_config *cfg,
                                int user_coils)
{
    if (user_coils) return;
    const mwSize  nd   = mxGPUGetNumberOfDimensions(SEs);
    const mwSize *dims = mxGPUGetDimensions(SEs);
    const int     nsp  = cfg->n_spatial;
    cfg->n_coils = ((int)nd > nsp) ? (int)dims[nsp] : 1;
    mxFree((void *)dims);
}

/* ------------------------------------------------------------------------ */
/* Commands                                                                 */
/* ------------------------------------------------------------------------ */

/**
 * @brief `h = mr_aha_op_mex('create', SEs, phi2_toep, config, mask, devices,
 *        share)`.
 *
 * Validates the inputs exactly like `AhA_Toeplitz_mex`, creates the plan on
 * the current (primary) device, registers it, locks the MEX file and returns
 * the handle.
 */
static void mr_op_create(int nlhs, mxArray *plhs[], int nrhs,
                         const mxArray *prhs[])
{
    if (nrhs != 7)
        mexErrMsgIdAndTxt(MR_ID ":usage",
                          "Usage: h = mr_aha_op_mex('create', SEs, phi2_toep, "
                          "config, mask, devices, share_primary).");
    if (nlhs > 1) mexErrMsgIdAndTxt(MR_ID ":usage", "Too many outputs.");

    /* ---- Configuration ------------------------------------------------- */
    mr_config cfg;
    mr_mex_parse_config(prhs[3], &cfg, MR_ID);
    const int user_coils = (mr_mex_field(prhs[3], "n_coils") != NULL);
    /* The plan always runs on MATLAB's device: gpuArrays live there. */
    cfg.gpu_device_id = -1;

    /* ---- Devices (CUDA ordinals) and flags ------------------------------ */
    const mxArray *mx_dev = prhs[5];
    std::vector<int> devices;
    if (!mxIsEmpty(mx_dev)) {
        if (!mxIsInt32(mx_dev) || mxIsComplex(mx_dev))
            mexErrMsgIdAndTxt(MR_ID ":devices",
                              "devices must be an int32 vector of CUDA "
                              "ordinals (mr_aha_op.m converts gpuDevice "
                              "indices).");
        const mwSize n = mxGetNumberOfElements(mx_dev);
        if (n > MR_MGPU_MAX_DEVICES)
            mexErrMsgIdAndTxt(MR_ID ":devices",
                              "at most %d devices are supported.",
                              MR_MGPU_MAX_DEVICES);
        const int32_t *d = mxGetInt32s(mx_dev);
        devices.assign(d, d + n);
    }
    if (mxGetNumberOfElements(prhs[6]) != 1 ||
        !(mxIsLogical(prhs[6]) || mxIsNumeric(prhs[6])))
        mexErrMsgIdAndTxt(MR_ID ":share", "share_primary must be a scalar.");
    const int share = mxGetScalar(prhs[6]) != 0.0;

    /* ---- Arrays --------------------------------------------------------- */
    const mxGPUArray *g_SEs  = mxGPUCreateFromMxArray(prhs[1]);
    const mxGPUArray *g_toep = mxGPUCreateFromMxArray(prhs[2]);
    const mxGPUArray *g_mask = NULL;

    mr_mex_check_complex_single(g_SEs,  "SEs",       MR_ID);
    mr_mex_check_complex_single(g_toep, "phi2_toep", MR_ID);
    mr_op_infer_n_coils(g_SEs, &cfg, user_coils);

    if (cfg.is_masked) {
        if (mxIsEmpty(prhs[4]))
            mexErrMsgIdAndTxt(MR_ID ":missingMask",
                              "config.is_masked is true, so a mask must be "
                              "supplied.");
        g_mask = mxGPUCreateFromMxArray(prhs[4]);
        if (mxGPUGetClassID(g_mask) != mxINT64_CLASS)
            mexErrMsgIdAndTxt(MR_ID ":mask",
                              "mask must be an int64 gpuArray of 0-based linear "
                              "indices; mr_aha_op.m performs the conversion.");
        if (cfg.num_masked == 0)
            cfg.num_masked = (int64_t)mxGPUGetNumberOfElements(g_mask);
        if ((int64_t)mxGPUGetNumberOfElements(g_mask) != cfg.num_masked)
            mexErrMsgIdAndTxt(MR_ID ":mask",
                              "mask has %lld elements but config.num_masked is "
                              "%lld.",
                              (long long)mxGPUGetNumberOfElements(g_mask),
                              (long long)cfg.num_masked);
    } else if (!mxIsEmpty(prhs[4])) {
        mexWarnMsgIdAndTxt(MR_ID ":unusedMask",
                           "A mask was supplied but config.is_masked is false; "
                           "the mask is ignored.");
    }

    int rc = mr_config_validate(&cfg);
    if (rc != MR_SUCCESS) mr_mex_fail(MR_ID, rc);
    mr_mex_check_numel(g_SEs,  mr_se_size(&cfg),        "SEs",       MR_ID);
    mr_mex_check_numel(g_toep, mr_phi2_toep_size(&cfg), "phi2_toep", MR_ID);

    int primary = -1;
    if (cudaGetDevice(&primary) != cudaSuccess)
        mexErrMsgIdAndTxt(MR_ID ":cuda", "cudaGetDevice failed: %s",
                          cudaGetErrorString(cudaGetLastError()));

    /* ---- Plan ----------------------------------------------------------- */
    mr_mgpu_plan plan = NULL;
    rc = mr_mgpu_plan_create(
        &cfg, devices.empty() ? NULL : devices.data(), (int)devices.size(),
        (const mr_complex *)mxGPUGetDataReadOnly(g_SEs),
        (const mr_complex *)mxGPUGetDataReadOnly(g_toep),
        g_mask ? (const int64_t *)mxGPUGetDataReadOnly(g_mask) : NULL,
        share ? MR_MGPU_SHARE_PRIMARY : 0, &plan);
    if (rc != MR_SUCCESS) {
        mxGPUDestroyGPUArray(g_SEs);
        mxGPUDestroyGPUArray(g_toep);
        if (g_mask) mxGPUDestroyGPUArray(g_mask);
        mr_mex_fail(MR_ID, rc);
    }

    /* ---- Registry entry ------------------------------------------------- */
    mr_op_entry *e = (mr_op_entry *)mxCalloc(1, sizeof(mr_op_entry));
    mexMakeMemoryPersistent(e);
    e->plan    = plan;
    e->cfg     = cfg;
    e->primary = primary;
    e->shared  = share;

    /* Sentinel: a one-element gpuArray on the primary device, kept as a
     * persistent mxArray so that its validity can be queried later. */
    const mwSize one = 1;
    mxGPUArray *s = mxGPUCreateGPUArray(1, &one, mxUINT8_CLASS, mxREAL,
                                        MX_GPU_INITIALIZE_VALUES);
    e->sentinel = mxGPUCreateMxArrayOnGPU(s);
    mxGPUDestroyGPUArray(s);
    mexMakeArrayPersistent(e->sentinel);

    if (share) {
        /* The plan reads these in place: keep them alive until destroy. */
        e->keep[0] = g_SEs;
        e->keep[1] = g_toep;
        e->keep[2] = g_mask;
    } else {
        /* The plan holds its own copies; the references can go. */
        mxGPUDestroyGPUArray(g_SEs);
        mxGPUDestroyGPUArray(g_toep);
        if (g_mask) mxGPUDestroyGPUArray(g_mask);
    }

    if (!g_atexit_registered) {
        mexAtExit(mr_op_at_exit);
        g_atexit_registered = 1;
    }
    if (g_entries.empty() && !mexIsLocked()) mexLock();
    g_entries.push_back(e);

    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *mxGetUint64s(plhs[0]) = (uint64_t)(uintptr_t)plan;
}

/**
 * @brief `AhAx = mr_aha_op_mex('apply', h, x)`.
 *
 * Refuses to run when the plan is not registered, when MATLAB's current
 * device is not the plan's primary device, or when that device has been reset
 * or deselected since the plan was created.
 */
static void mr_op_apply(int nlhs, mxArray *plhs[], int nrhs,
                        const mxArray *prhs[])
{
    if (nrhs != 3)
        mexErrMsgIdAndTxt(MR_ID ":usage",
                          "Usage: AhAx = mr_aha_op_mex('apply', h, x).");
    if (nlhs > 1) mexErrMsgIdAndTxt(MR_ID ":usage", "Too many outputs.");

    mr_op_entry *e = mr_op_lookup(prhs[1], "apply");

    int cur = -1;
    if (cudaGetDevice(&cur) != cudaSuccess)
        mexErrMsgIdAndTxt(MR_ID ":cuda", "cudaGetDevice failed: %s",
                          cudaGetErrorString(cudaGetLastError()));
    if (cur != e->primary)
        mexErrMsgIdAndTxt(MR_ID ":deviceChanged",
                          "This mr_aha_op was created on gpuDevice(%d), but the "
                          "current device is gpuDevice(%d). Select gpuDevice(%d) "
                          "again (which resets it, see below) or delete the "
                          "operator and create a new one.",
                          e->primary + 1, cur + 1, e->primary + 1);
    if (!mr_op_device_valid(e))
        mexErrMsgIdAndTxt(MR_ID ":deviceReset",
                          "gpuDevice(%d) has been reset or deselected since "
                          "this mr_aha_op was created, which freed its GPU "
                          "data. Delete the operator and create a new one.",
                          e->primary + 1);

    const mxGPUArray *g_x = mxGPUCreateFromMxArray(prhs[2]);
    mr_mex_check_complex_single(g_x, "x", MR_ID);
    mr_mex_check_numel(g_x, mr_x_size(&e->cfg), "x", MR_ID);

    const mwSize  nd   = mxGPUGetNumberOfDimensions(g_x);
    const mwSize *dims = mxGPUGetDimensions(g_x);
    mxGPUArray *g_out = mxGPUCreateGPUArray(nd, dims, mxSINGLE_CLASS, mxCOMPLEX,
                                            MX_GPU_DO_NOT_INITIALIZE);
    mxFree((void *)dims);

    const int rc = mr_mgpu_plan_execute(
        e->plan, (const mr_complex *)mxGPUGetDataReadOnly(g_x),
        (mr_complex *)mxGPUGetData(g_out));
    if (rc != MR_SUCCESS) {
        mxGPUDestroyGPUArray(g_out);
        mxGPUDestroyGPUArray(g_x);
        mr_mex_fail(MR_ID, rc);
    }

    plhs[0] = mxGPUCreateMxArrayOnGPU(g_out);
    mxGPUDestroyGPUArray(g_out);
    mxGPUDestroyGPUArray(g_x);
}

/**
 * @brief `info = mr_aha_op_mex('info', h)`: the work distribution as a
 *        struct (0-based CUDA ordinals and coil indices, as the library
 *        reports them), plus `primary`, `share_primary` and `valid` (false
 *        once the primary device was reset or deselected).
 */
static void mr_op_info(int nlhs, mxArray *plhs[], int nrhs,
                       const mxArray *prhs[])
{
    if (nrhs != 2)
        mexErrMsgIdAndTxt(MR_ID ":usage",
                          "Usage: info = mr_aha_op_mex('info', h).");
    if (nlhs > 1) mexErrMsgIdAndTxt(MR_ID ":usage", "Too many outputs.");

    mr_op_entry *e = mr_op_lookup(prhs[1], "info");
    mr_mgpu_info in;
    const int rc = mr_mgpu_plan_info(e->plan, &in);
    if (rc != MR_SUCCESS) mr_mex_fail(MR_ID, rc);

    static const char *names[] = {
        "n_devices", "devices", "coil_begin", "coil_count", "group",
        "n_split_coils", "peer_access", "peer_bandwidth_gbs", "primary",
        "n_coils", "share_primary", "valid"};
    mxArray *s = mxCreateStructMatrix(1, 1, sizeof(names) / sizeof(names[0]),
                                      names);
    const int n = in.n_devices;
    /** Copy the first @p n entries of an int array into a double row. */
    auto row = [n](const int *v) {
        mxArray *a = mxCreateDoubleMatrix(1, (mwSize)n, mxREAL);
        double *p = mxGetDoubles(a);
        for (int k = 0; k < n; ++k) p[k] = (double)v[k];
        return a;
    };
    mxSetField(s, 0, "n_devices",  mxCreateDoubleScalar((double)n));
    mxSetField(s, 0, "devices",    row(in.devices));
    mxSetField(s, 0, "coil_begin", row(in.coil_begin));
    mxSetField(s, 0, "coil_count", row(in.coil_count));
    mxSetField(s, 0, "group",      row(in.group));
    mxSetField(s, 0, "n_split_coils",
               mxCreateDoubleScalar((double)in.n_split_coils));
    mxSetField(s, 0, "peer_access",
               mxCreateLogicalScalar(in.peer_access != 0));
    mxSetField(s, 0, "peer_bandwidth_gbs",
               mxCreateDoubleScalar(in.peer_bandwidth_gbs));
    mxSetField(s, 0, "primary", mxCreateDoubleScalar((double)e->primary));
    mxSetField(s, 0, "n_coils", mxCreateDoubleScalar((double)e->cfg.n_coils));
    mxSetField(s, 0, "share_primary", mxCreateLogicalScalar(e->shared != 0));
    mxSetField(s, 0, "valid", mxCreateLogicalScalar(mr_op_device_valid(e)));
    plhs[0] = s;
}

/**
 * @brief `mr_aha_op_mex('destroy', h)`: release a plan; a handle that is not
 *        (or no longer) registered is ignored.
 */
static void mr_op_destroy(int nlhs, mxArray *plhs[], int nrhs,
                          const mxArray *prhs[])
{
    (void)plhs;
    if (nrhs != 2)
        mexErrMsgIdAndTxt(MR_ID ":usage",
                          "Usage: mr_aha_op_mex('destroy', h).");
    if (nlhs > 0) mexErrMsgIdAndTxt(MR_ID ":usage", "destroy has no outputs.");
    if (!mxIsUint64(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 1)
        mexErrMsgIdAndTxt(MR_ID ":badHandle",
                          "destroy: the plan handle must be a uint64 scalar.");
    const int i = mr_op_find(*mxGetUint64s(prhs[1]));
    if (i >= 0) mr_op_remove(i);
}

/**
 * @brief MEX entry point: dispatch on the command string `prhs[0]`.
 *
 * @param[in]  nlhs Number of requested outputs.
 * @param[out] plhs Outputs of the command.
 * @param[in]  nrhs Number of inputs, including the command.
 * @param[in]  prhs `{command, ...}`, see the file documentation.
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1 || !mxIsChar(prhs[0]))
        mexErrMsgIdAndTxt(MR_ID ":usage",
                          "The first argument must be a command: 'create', "
                          "'apply', 'info', 'destroy' or 'count'.");
    char cmd[16];
    if (mxGetString(prhs[0], cmd, sizeof(cmd)) != 0)
        mexErrMsgIdAndTxt(MR_ID ":usage", "Unknown command.");

    if (strcmp(cmd, "count") == 0) {
        plhs[0] = mxCreateDoubleScalar((double)g_entries.size());
        return;
    }

    /* mxInitGPU must run before any other MathWorks GPU API call. */
    mxInitGPU();

    if (strcmp(cmd, "create") == 0)       mr_op_create(nlhs, plhs, nrhs, prhs);
    else if (strcmp(cmd, "apply") == 0)   mr_op_apply(nlhs, plhs, nrhs, prhs);
    else if (strcmp(cmd, "info") == 0)    mr_op_info(nlhs, plhs, nrhs, prhs);
    else if (strcmp(cmd, "destroy") == 0) mr_op_destroy(nlhs, plhs, nrhs, prhs);
    else
        mexErrMsgIdAndTxt(MR_ID ":usage",
                          "Unknown command '%s': expected 'create', 'apply', "
                          "'info', 'destroy' or 'count'.", cmd);
}
