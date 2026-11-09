/**
 * @file mr_aha_op_mex.cu
 * @brief MATLAB MEX gateway of the multi-GPU @f$A^H A@f$ operator class
 *        `mr_aha_op` (CUDA backend).
 *
 * This gateway owns the multi-GPU plans (::mr_mgpu_plan) behind the
 * `mr_aha_op` handle class and is driven by a command string:
 * @code
 *   h    = mr_aha_op_mex('create',  SEs, phi2_toep, config, mask, devices, share)
 *   h    = mr_aha_op_mex('create_gram', SEs, phi2, traj, config, mask, devices, share)
 *   AhAx = mr_aha_op_mex('apply',   h, x)
 *   info = mr_aha_op_mex('info',    h)
 *   Om   = mr_aha_op_mex('kernel',  h)
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
 * - `create_gram`: the same plan built from the Gram blocks `phi2` and the
 *   trajectory `traj` (as for `calc_toep_mex`) with
 *   ::mr_mgpu_plan_create_gram: every GPU computes a share of `phi2_toep`
 *   and the shares are exchanged between the GPUs. The GPUs are driven by
 *   library threads that never call the MATLAB API. With `share` only `SEs`
 *   and `mask` are used in place; the kernel belongs to the plan.
 * - `kernel`: a copy of the plan's kernel on the primary device as a new
 *   gpuArray, shaped like `calc_toep`'s output.
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
                                     `phi2_toep` (NULL for `create_gram`) and
                                     `mask` (or NULL) that keep their storage
                                     alive until destroy.                      */
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
 *        share)` and `h = mr_aha_op_mex('create_gram', SEs, phi2, traj,
 *        config, mask, devices, share)`.
 *
 * Validates the inputs exactly like `AhA_Toeplitz_mex` (and, for
 * `create_gram`, like `calc_toep_mex`), creates the plan on the current
 * (primary) device, registers it, locks the MEX file and returns the handle.
 *
 * @param[in] gram Non-zero for `create_gram`: the arguments after `SEs` are
 *                 `phi2, traj` instead of `phi2_toep`, shifting the rest by
 *                 one.
 */
static void mr_op_create(int nlhs, mxArray *plhs[], int nrhs,
                         const mxArray *prhs[], int gram)
{
    const int o = gram ? 1 : 0; /* offset of config, mask, devices, share */
    if (nrhs != 7 + o)
        mexErrMsgIdAndTxt(MR_ID ":usage",
                          gram ? "Usage: h = mr_aha_op_mex('create_gram', SEs, "
                                 "phi2, traj, config, mask, devices, "
                                 "share_primary)."
                               : "Usage: h = mr_aha_op_mex('create', SEs, "
                                 "phi2_toep, config, mask, devices, "
                                 "share_primary).");
    if (nlhs > 1) mexErrMsgIdAndTxt(MR_ID ":usage", "Too many outputs.");
    const mxArray *mx_cfg = prhs[3 + o], *mx_mask = prhs[4 + o];
    const mxArray *mx_dev = prhs[5 + o], *mx_share = prhs[6 + o];

    /* ---- Configuration ------------------------------------------------- */
    mr_config cfg;
    mr_mex_parse_config(mx_cfg, &cfg, MR_ID);
    const int user_coils = (mr_mex_field(mx_cfg, "n_coils") != NULL);
    /* The plan always runs on MATLAB's device: gpuArrays live there. */
    cfg.gpu_device_id = -1;

    /* ---- Devices (CUDA ordinals) and flags ------------------------------ */
    int dev_buf[MR_MGPU_MAX_DEVICES];
    const int n_dev = mr_mex_devices(mx_dev, dev_buf, MR_ID ":devices");
    std::vector<int> devices(dev_buf, dev_buf + n_dev);
    if (mxGetNumberOfElements(mx_share) != 1 ||
        !(mxIsLogical(mx_share) || mxIsNumeric(mx_share)))
        mexErrMsgIdAndTxt(MR_ID ":share", "share_primary must be a scalar.");
    const int share = mxGetScalar(mx_share) != 0.0;

    /* ---- Arrays --------------------------------------------------------- */
    const mxGPUArray *g_SEs  = mxGPUCreateFromMxArray(prhs[1]);
    const mxGPUArray *g_toep = NULL;       /* create      */
    const mxGPUArray *g_phi2 = NULL;       /* create_gram */
    const mxGPUArray *g_traj = NULL;       /* create_gram */
    const mxGPUArray *g_mask = NULL;
    int64_t Ntrajs = 0, Nkx = 0;

    mr_mex_check_complex_single(g_SEs,  "SEs",       MR_ID);
    if (gram) {
        g_phi2 = mxGPUCreateFromMxArray(prhs[2]);
        g_traj = mxGPUCreateFromMxArray(prhs[3]);
        mr_mex_check_complex_single(g_phi2, "phi2", MR_ID);
        mr_mex_check_traj(g_traj, &cfg, &Ntrajs, &Nkx, MR_ID ":traj");
    } else {
        g_toep = mxGPUCreateFromMxArray(prhs[2]);
        mr_mex_check_complex_single(g_toep, "phi2_toep", MR_ID);
    }
    mr_op_infer_n_coils(g_SEs, &cfg, user_coils);

    if (cfg.is_masked) {
        if (mxIsEmpty(mx_mask))
            mexErrMsgIdAndTxt(MR_ID ":missingMask",
                              "config.is_masked is true, so a mask must be "
                              "supplied.");
        g_mask = mxGPUCreateFromMxArray(mx_mask);
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
    } else if (!mxIsEmpty(mx_mask)) {
        mexWarnMsgIdAndTxt(MR_ID ":unusedMask",
                           "A mask was supplied but config.is_masked is false; "
                           "the mask is ignored.");
    }

    int rc = mr_config_validate(&cfg);
    if (rc != MR_SUCCESS) mr_mex_fail(MR_ID, rc);
    mr_mex_check_numel(g_SEs,  mr_se_size(&cfg),        "SEs",       MR_ID);
    if (gram) {
        const int64_t nkz = (cfg.traj_type == MR_TRAJ_STACK) ? cfg.phi2_dim[2] : 1;
        mr_mex_check_numel(g_phi2, Ntrajs * nkz * cfg.L * cfg.L, "phi2", MR_ID);
    } else {
        mr_mex_check_numel(g_toep, mr_phi2_toep_size(&cfg), "phi2_toep", MR_ID);
    }

    int primary = -1;
    if (cudaGetDevice(&primary) != cudaSuccess)
        mexErrMsgIdAndTxt(MR_ID ":cuda", "cudaGetDevice failed: %s",
                          cudaGetErrorString(cudaGetLastError()));

    /* ---- Plan ----------------------------------------------------------- */
    mr_mgpu_plan plan = NULL;
    const int *dev_p = devices.empty() ? NULL : devices.data();
    const int64_t *mask_p =
        g_mask ? (const int64_t *)mxGPUGetDataReadOnly(g_mask) : NULL;
    if (gram)
        rc = mr_mgpu_plan_create_gram(
            &cfg, dev_p, (int)devices.size(),
            (const mr_complex *)mxGPUGetDataReadOnly(g_SEs),
            (const mr_complex *)mxGPUGetDataReadOnly(g_phi2),
            (const float *)mxGPUGetDataReadOnly(g_traj), Ntrajs, Nkx, mask_p,
            share ? MR_MGPU_SHARE_PRIMARY : 0, &plan);
    else
        rc = mr_mgpu_plan_create(
            &cfg, dev_p, (int)devices.size(),
            (const mr_complex *)mxGPUGetDataReadOnly(g_SEs),
            (const mr_complex *)mxGPUGetDataReadOnly(g_toep), mask_p,
            share ? MR_MGPU_SHARE_PRIMARY : 0, &plan);
    /* The Gram blocks and the trajectory are only needed during creation. */
    if (g_phi2) mxGPUDestroyGPUArray(g_phi2);
    if (g_traj) mxGPUDestroyGPUArray(g_traj);
    if (rc != MR_SUCCESS) {
        mxGPUDestroyGPUArray(g_SEs);
        if (g_toep) mxGPUDestroyGPUArray(g_toep);
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
        if (g_toep) mxGPUDestroyGPUArray(g_toep);
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
        "n_coils", "share_primary", "valid", "toep_source", "toep_devices",
        "toep_ms", "toep_component_ms", "toep_gather_gbs"};
    static const char *sources[] = {"supplied", "broadcast", "split",
                                    "replicate"};
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
    mxSetField(s, 0, "toep_source",
               mxCreateString((in.toep_source >= 0 && in.toep_source <= 3)
                                  ? sources[in.toep_source]
                                  : "unknown"));
    mxSetField(s, 0, "toep_devices", mxCreateDoubleScalar((double)in.toep_devices));
    mxSetField(s, 0, "toep_ms", mxCreateDoubleScalar(in.toep_ms));
    mxSetField(s, 0, "toep_component_ms",
               mxCreateDoubleScalar(in.toep_component_ms));
    mxSetField(s, 0, "toep_gather_gbs", mxCreateDoubleScalar(in.toep_gather_gbs));
    plhs[0] = s;
}

/**
 * @brief `Om = mr_aha_op_mex('kernel', h)`: a copy of the plan's
 *        `phi2_toep` on the primary device, shaped like `calc_toep`'s output.
 *
 * A copy (not a view): MATLAB cannot wrap memory it does not own in a
 * gpuArray. Refuses to run when the primary device was reset or deselected.
 */
static void mr_op_kernel(int nlhs, mxArray *plhs[], int nrhs,
                         const mxArray *prhs[])
{
    if (nrhs != 2)
        mexErrMsgIdAndTxt(MR_ID ":usage", "Usage: Om = mr_aha_op_mex('kernel', h).");
    if (nlhs > 1) mexErrMsgIdAndTxt(MR_ID ":usage", "Too many outputs.");
    mr_op_entry *e = mr_op_lookup(prhs[1], "kernel");
    int cur = -1;
    if (cudaGetDevice(&cur) != cudaSuccess || cur != e->primary ||
        !mr_op_device_valid(e))
        mexErrMsgIdAndTxt(MR_ID ":deviceReset",
                          "kernel: gpuDevice(%d), the operator's primary GPU, is "
                          "not selected or has been reset since the operator "
                          "was created.",
                          e->primary + 1);
    const mr_complex *src = NULL;
    const int rc = mr_mgpu_plan_kernel(e->plan, 0, &src);
    if (rc != MR_SUCCESS) mr_mex_fail(MR_ID, rc);
    mwSize dims[4];
    const mwSize nd = mr_mex_toep_dims(&e->cfg, dims);
    mxGPUArray *g = mxGPUCreateGPUArray(nd, dims, mxSINGLE_CLASS, mxCOMPLEX,
                                        MX_GPU_DO_NOT_INITIALIZE);
    const size_t bytes = (size_t)mr_phi2_toep_size(&e->cfg) * sizeof(mr_complex);
    /* Legacy default stream: ordered after MATLAB's work and the plan's. */
    const cudaError_t ce = cudaMemcpy(mxGPUGetData(g), src, bytes,
                                      cudaMemcpyDeviceToDevice);
    if (ce != cudaSuccess) {
        mxGPUDestroyGPUArray(g);
        mexErrMsgIdAndTxt(MR_ID ":cuda", "kernel: copy failed: %s",
                          cudaGetErrorString(ce));
    }
    plhs[0] = mxGPUCreateMxArrayOnGPU(g);
    mxGPUDestroyGPUArray(g);
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
                          "'create_gram', 'apply', 'info', 'kernel', "
                          "'destroy' or 'count'.");
    char cmd[16];
    if (mxGetString(prhs[0], cmd, sizeof(cmd)) != 0)
        mexErrMsgIdAndTxt(MR_ID ":usage", "Unknown command.");

    if (strcmp(cmd, "count") == 0) {
        plhs[0] = mxCreateDoubleScalar((double)g_entries.size());
        return;
    }

    /* mxInitGPU must run before any other MathWorks GPU API call. */
    mxInitGPU();

    if (strcmp(cmd, "create") == 0)       mr_op_create(nlhs, plhs, nrhs, prhs, 0);
    else if (strcmp(cmd, "create_gram") == 0)
        mr_op_create(nlhs, plhs, nrhs, prhs, 1);
    else if (strcmp(cmd, "kernel") == 0)  mr_op_kernel(nlhs, plhs, nrhs, prhs);
    else if (strcmp(cmd, "apply") == 0)   mr_op_apply(nlhs, plhs, nrhs, prhs);
    else if (strcmp(cmd, "info") == 0)    mr_op_info(nlhs, plhs, nrhs, prhs);
    else if (strcmp(cmd, "destroy") == 0) mr_op_destroy(nlhs, plhs, nrhs, prhs);
    else
        mexErrMsgIdAndTxt(MR_ID ":usage",
                          "Unknown command '%s': expected 'create', "
                          "'create_gram', 'apply', 'info', 'kernel', "
                          "'destroy' or 'count'.", cmd);
}
