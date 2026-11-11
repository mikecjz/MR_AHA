/**
 * @file mr_aha_op_mex.cu
 * @brief MATLAB MEX gateway of the @f$A^H A@f$ operator class `mr_aha_op`
 *        (CUDA backend).
 *
 * This gateway owns the library operators (::mr_op) behind the `mr_aha_op`
 * handle class and is driven by a command string:
 * @code
 *   h    = mr_aha_op_mex('create',  SEs, phi2, traj, config, mask, opts)
 *   AhAx = mr_aha_op_mex('apply',   h, x)
 *   info = mr_aha_op_mex('info',    h)
 *   Om   = mr_aha_op_mex('kernel',  h, k)
 *          mr_aha_op_mex('destroy', h)
 *   n    = mr_aha_op_mex('count')
 * @endcode
 * Use the `mr_aha_op.create_op` method of the class, which validates the
 * name-value arguments, fills in the derived `config` fields, converts a
 * 1-based MATLAB mask and 1-based `gpuDevice` indices, and destroys the
 * operator when the object is deleted or cleared.
 *
 * - `create`: builds the operator with ::mr_op_create on MATLAB's current
 *   device (the *primary* device). `SEs` and `phi2` are complex single
 *   gpuArrays and `traj` a real single gpuArray there (layouts of
 *   ::mr_calc_toep); `phi2` and `traj` may be `[]` when a finished kernel is
 *   supplied (`opts.phi2_toep`). `mask` is an `int64` gpuArray of 0-based
 *   linear indices, or `[]` when `config.is_masked` is false. `opts` is a
 *   scalar struct whose fields are all optional (absent or empty: default):
 *   - `method`: `'toeplitz'` (default, ::MR_OP_TOEPLITZ) or `'nufft'`
 *     (::MR_OP_NUFFT);
 *   - `devices`: an `int32` vector of **CUDA ordinals** (0-based), `'all'`
 *     (::MR_OP_ALL_DEVICES) or `[]` (the primary device only); the primary
 *     device always takes part;
 *   - `toep_dist`: `'auto'` (default), `'split'`, `'replicate'` or
 *     `'broadcast'` (::mr_toep_dist);
 *   - `share_primary`: logical scalar selecting ::MR_OP_SHARE_PRIMARY
 *     (default true);
 *   - `nufft_upsampfac`, `nufft_gpu_method`: numeric scalars
 *     (::mr_op_options::nufft_upsampfac, ::mr_op_options::nufft_gpu_method;
 *     0 for the defaults);
 *   - `phi2_toep`: a finished kernel (complex single gpuArray shaped like
 *     `calc_toep`'s output) used instead of computing one
 *     (::mr_op_inputs::d_phi2_toep). Private: for tests that inject a
 *     reference kernel.
 *
 *   Returns the operator handle as a `uint64` scalar. The GPUs other than the
 *   primary are driven by library threads that never call the MATLAB API.
 * - `apply`: `x` is a complex single gpuArray of ::mr_x_size elements;
 *   returns a new gpuArray of the same shape.
 * - `info`: returns a struct describing the operator (CUDA ordinals,
 *   0-based coil indices; the class converts both to 1-based).
 * - `kernel`: a copy of the Toeplitz kernel the operator holds on its
 *   `k`-th device (0-based position in `info.devices`, 0 is the primary),
 *   returned as a new gpuArray on the primary device, shaped like
 *   `calc_toep`'s output. NUFFT operators have no kernel
 *   (`MR_AHA:mr_aha_op:noKernel`).
 * - `destroy`: releases an operator. Unknown or already destroyed handles are
 *   ignored, so the call is idempotent.
 * - `count`: number of live operators (used by the tests).
 *
 * ## Lifetime and safety
 *
 * - **Registry.** Every live operator is recorded in a registry kept by this
 *   MEX file. A handle that is not in the registry (destroyed, never created,
 *   or forged) is rejected by `apply` / `info` / `kernel` with a clean MATLAB
 *   error instead of being dereferenced.
 * - **mexLock.** While at least one operator is alive the MEX file is
 *   locked, so `clear mex` / `clear functions` / `clear all` cannot unload
 *   the code and the registry that the live operators and `mr_aha_op`
 *   objects depend on. Once the last operator is destroyed the MEX file is
 *   unlocked again. `clear all` also clears the variables, which deletes the
 *   `mr_aha_op` objects (and hence their operators) first.
 * - **mexAtExit.** When MATLAB exits (or unloads the unlocked MEX file), any
 *   operator still alive is destroyed.
 * - **Stale operators.** An operator whose primary device was reset is never
 *   passed to ::mr_op_destroy (its stream handles on that device are dead
 *   and the CUDA driver crashes on them); `destroy` releases it with
 *   ::mr_op_abandon, which frees its allocations on the *other* devices
 *   only, and warns (see ::mex_op_release).
 * - **Device changes.** An operator is bound to the device that was current
 *   when it was created. MATLAB invalidates all gpuArrays (and destroys the
 *   CUDA context of the device, which frees the operator's allocations
 *   there) on `reset(gpuDevice)` or when another device is selected with
 *   `gpuDevice(k)`. To detect this, every registry entry holds a one-element
 *   *sentinel* gpuArray created on the primary device; `apply` and `kernel`
 *   refuse to run when the current device is not the primary device or when
 *   `mxGPUIsValidGPUData` reports the sentinel as invalid, so a stale
 *   operator raises a MATLAB error instead of touching freed device memory.
 * - **Shared arrays.** With `share_primary` (::MR_OP_SHARE_PRIMARY) the
 *   operator reads `SEs`, `mask` and a supplied `phi2_toep` in place on the
 *   primary device. The registry entry then holds `mxGPUArray` references to
 *   these arrays until the operator is destroyed, so MATLAB cannot free (or
 *   recycle) their storage even when the caller clears its variables or
 *   passed temporaries (e.g. the converted mask). gpuArrays are
 *   copy-on-write, so later assignments to the caller's variables never
 *   modify the shared storage. `phi2` and `traj` are never kept: the
 *   operator transforms them into its own arrays at creation.
 *
 * ## Streams
 *
 * Like `AhA_Toeplitz_mex`, the configuration's ::mr_config::stream is left
 * NULL, i.e. the legacy default stream of the primary device, which is
 * ordered with MATLAB's own GPU work (the contract of MathWorks' gpuArray MEX
 * API). ::mr_op_execute is asynchronous: on several devices it orders the
 * other devices after the legacy stream with an event and makes the legacy
 * stream wait for their partial results, so the returned gpuArray is safe to
 * use from MATLAB without a host synchronisation here. The library saves and
 * restores the current CUDA device in every call, so MATLAB's device
 * selection is never changed by this gateway.
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
/* Registry of live operators                                               */
/* ------------------------------------------------------------------------ */

/**
 * @brief One live operator and the MATLAB-side state it depends on.
 */
struct mex_op_entry {
    mr_op             op;       /**< The library operator (never NULL).        */
    mr_config         cfg;      /**< Configuration the operator was created
                                     with; gives the expected size of `x` and
                                     the shape of the kernel.                  */
    int               primary;  /**< CUDA ordinal of the primary device.       */
    int               shared;   /**< Non-zero with ::MR_OP_SHARE_PRIMARY.      */
    mxArray          *sentinel; /**< Persistent one-element gpuArray on the
                                     primary device; becomes invalid when
                                     MATLAB resets or deselects that device.   */
    const mxGPUArray *keep[3];  /**< With @ref shared: references to `SEs`,
                                     the supplied `phi2_toep` (or NULL) and
                                     `mask` (or NULL) that keep their storage
                                     alive until destroy.                      */
};

/** @brief All live operators of this MEX file. */
static std::vector<mex_op_entry *> g_entries;

/** @brief Non-zero once ::mex_op_at_exit has been registered. */
static int g_atexit_registered = 0;

/**
 * @brief Find the registry entry of an operator handle.
 * @param[in] h Handle as returned by `create` (the operator pointer).
 * @return Index into ::g_entries, or -1 when @p h is not a live operator.
 */
static int mex_op_find(uint64_t h)
{
    for (size_t i = 0; i < g_entries.size(); ++i)
        if ((uint64_t)(uintptr_t)g_entries[i]->op == h) return (int)i;
    return -1;
}

/**
 * @brief Non-zero when the primary device of @p e is still the one the
 *        operator was created on, i.e. MATLAB has neither reset nor
 *        deselected it.
 */
static int mex_op_device_valid(const mex_op_entry *e)
{
    return e->sentinel != NULL && mxGPUIsValidGPUData(e->sentinel);
}

/**
 * @brief Release one registry entry: the operator, the kept arrays and the
 *        sentinel. Does not touch ::g_entries or the MEX lock.
 *
 * When the primary device was reset or deselected, MATLAB has already
 * destroyed its CUDA context and with it everything the operator had
 * allocated there, including its streams, events, cuFFT / cufinufft plans
 * and buffers on that device. Passing such an operator to ::mr_op_destroy is
 * not safe: it synchronises those stale stream handles, which crashes inside
 * the CUDA driver (observed with R2022b: segmentation violation in
 * `cuStreamSynchronize`). A stale operator is therefore released with
 * ::mr_op_abandon, which frees what it holds on the other devices (kernel
 * copies, maps, plans, buffers) and skips the primary device; a warning says
 * that the device was reset. The MATLAB-side references (kept arrays,
 * sentinel) are always released.
 *
 * @param[in] e Entry to release; freed on return.
 * @return 1 when the operator was stale (primary device reset), else 0.
 */
static int mex_op_release(mex_op_entry *e)
{
    const int stale = !mex_op_device_valid(e);
    if (stale)
        mr_op_abandon(e->op);
    else
        mr_op_destroy(e->op);
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
static void mex_op_remove(int i)
{
    mex_op_entry *e = g_entries[(size_t)i];
    const int primary = e->primary;
    g_entries.erase(g_entries.begin() + i);
    const int stale = mex_op_release(e);
    if (g_entries.empty() && mexIsLocked()) mexUnlock();
    if (stale)
        mexWarnMsgIdAndTxt(MR_ID ":staleDestroy",
                           "gpuDevice(%d) was reset or deselected while this "
                           "mr_aha_op existed; the operator had become unusable "
                           "and has been released.", primary + 1);
}

/**
 * @brief mexAtExit handler: destroy every operator still alive when MATLAB
 *        exits or the MEX file is unloaded.
 */
static void mex_op_at_exit(void)
{
    while (!g_entries.empty()) {
        mex_op_entry *e = g_entries.back();
        g_entries.pop_back();
        (void)mex_op_release(e); /* stale operators are abandoned (see above) */
    }
    if (mexIsLocked()) mexUnlock();
}

/**
 * @brief Read a `uint64` operator handle argument and look it up.
 * @param[in] a   The MATLAB argument.
 * @param[in] cmd Command name used in diagnostics.
 * @return The registry entry; raises a MATLAB error when the handle is not a
 *         live operator.
 */
static mex_op_entry *mex_op_lookup(const mxArray *a, const char *cmd)
{
    if (!mxIsUint64(a) || mxGetNumberOfElements(a) != 1 || mxIsComplex(a))
        mexErrMsgIdAndTxt(MR_ID ":badHandle",
                          "%s: the operator handle must be a uint64 scalar.", cmd);
    const uint64_t h = *mxGetUint64s(a);
    const int i = mex_op_find(h);
    if (i < 0)
        mexErrMsgIdAndTxt(MR_ID ":invalidHandle",
                          "%s: this mr_aha_op has been destroyed (or the "
                          "handle is not a live operator).", cmd);
    return g_entries[(size_t)i];
}

/**
 * @brief Raise a MATLAB error unless MATLAB's current device is still the
 *        operator's primary device, unreset since creation.
 * @param[in] e   Registry entry.
 * @param[in] cmd Command name used in diagnostics.
 */
static void mex_op_require_device(const mex_op_entry *e, const char *cmd)
{
    int cur = -1;
    if (cudaGetDevice(&cur) != cudaSuccess)
        mexErrMsgIdAndTxt(MR_ID ":cuda", "%s: cudaGetDevice failed: %s", cmd,
                          cudaGetErrorString(cudaGetLastError()));
    if (cur != e->primary)
        mexErrMsgIdAndTxt(MR_ID ":deviceChanged",
                          "%s: this mr_aha_op was created on gpuDevice(%d), but "
                          "the current device is gpuDevice(%d). Select "
                          "gpuDevice(%d) again (which resets it, see below) or "
                          "delete the operator and create a new one.",
                          cmd, e->primary + 1, cur + 1, e->primary + 1);
    if (!mex_op_device_valid(e))
        mexErrMsgIdAndTxt(MR_ID ":deviceReset",
                          "%s: gpuDevice(%d) has been reset or deselected since "
                          "this mr_aha_op was created, which freed its GPU "
                          "data. Delete the operator and create a new one.",
                          cmd, e->primary + 1);
}

/**
 * @brief Deduce the coil count from the trailing dimension of `SEs` unless
 *        `config.n_coils` was given (same rule as `AhA_Toeplitz_mex`).
 * @param[in]     SEs        Sensitivity maps.
 * @param[in,out] cfg        Configuration; `n_coils` is filled in.
 * @param[in]     user_coils Non-zero when the caller set `n_coils`.
 */
static void mex_op_infer_n_coils(const mxGPUArray *SEs, mr_config *cfg,
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
/* Options                                                                  */
/* ------------------------------------------------------------------------ */

/**
 * @brief Read an optional character-vector field of the `opts` struct.
 * @param[in]  opts The struct.
 * @param[in]  name Field name.
 * @param[out] buf  Receives the lower-case value, or @p def when the field is
 *                  absent or empty.
 * @param[in]  len  Size of @p buf.
 * @param[in]  def  Default value.
 */
static void mex_op_opt_string(const mxArray *opts, const char *name, char *buf,
                              size_t len, const char *def)
{
    const mxArray *f = mr_mex_field(opts, name);
    if (f == NULL) {
        snprintf(buf, len, "%s", def);
        return;
    }
    if (!mxIsChar(f) || mxGetString(f, buf, len) != 0)
        mexErrMsgIdAndTxt(MR_ID ":usage", "opts.%s must be a short character "
                                          "vector.", name);
    for (char *c = buf; *c; ++c)
        if (*c >= 'A' && *c <= 'Z') *c = (char)(*c - 'A' + 'a');
}

/**
 * @brief Read an optional numeric or logical scalar field of the `opts`
 *        struct.
 * @param[in] opts The struct.
 * @param[in] name Field name.
 * @param[in] def  Value when the field is absent or empty.
 * @return The value.
 */
static double mex_op_opt_scalar(const mxArray *opts, const char *name, double def)
{
    const mxArray *f = mr_mex_field(opts, name);
    if (f == NULL) return def;
    if (mxGetNumberOfElements(f) != 1 || mxIsComplex(f) ||
        !(mxIsNumeric(f) || mxIsLogical(f)))
        mexErrMsgIdAndTxt(MR_ID ":usage", "opts.%s must be a real scalar.", name);
    return mxGetScalar(f);
}

/**
 * @brief Translate the `opts` struct of `create` into ::mr_op_options.
 *
 * @param[in]  opts    The struct (or `[]` for all defaults).
 * @param[out] o       Options, starting from ::mr_op_options_defaults.
 * @param[out] devices Storage for the device list @p o points to.
 */
static void mex_op_parse_options(const mxArray *opts, mr_op_options *o,
                                 int devices[MR_OP_MAX_DEVICES])
{
    if (!mxIsEmpty(opts) && !(mxIsStruct(opts) && mxGetNumberOfElements(opts) == 1))
        mexErrMsgIdAndTxt(MR_ID ":usage", "opts must be a scalar struct.");
    mr_op_options_defaults(o);

    char s[32];
    mex_op_opt_string(opts, "method", s, sizeof(s), "toeplitz");
    if (strcmp(s, "toeplitz") == 0)   o->method = MR_OP_TOEPLITZ;
    else if (strcmp(s, "nufft") == 0) o->method = MR_OP_NUFFT;
    else mexErrMsgIdAndTxt(MR_ID ":method",
                           "method must be 'toeplitz' or 'nufft', not '%s'.", s);

    mex_op_opt_string(opts, "toep_dist", s, sizeof(s), "auto");
    if (strcmp(s, "auto") == 0)           o->toep_dist = MR_TOEP_DIST_AUTO;
    else if (strcmp(s, "split") == 0)     o->toep_dist = MR_TOEP_DIST_SPLIT;
    else if (strcmp(s, "replicate") == 0) o->toep_dist = MR_TOEP_DIST_REPLICATE;
    else if (strcmp(s, "broadcast") == 0) o->toep_dist = MR_TOEP_DIST_BROADCAST;
    else mexErrMsgIdAndTxt(MR_ID ":toepDist",
                           "toep_dist must be 'auto', 'split', 'replicate' or "
                           "'broadcast', not '%s'.", s);

    const mxArray *d = mr_mex_field(opts, "devices");
    o->n_devices = d ? mr_mex_devices(d, devices, MR_ID ":devices") : 0;
    o->devices = (o->n_devices > 0) ? devices : NULL;

    o->flags = (mex_op_opt_scalar(opts, "share_primary", 1.0) != 0.0)
                   ? MR_OP_SHARE_PRIMARY
                   : 0;
    o->nufft_upsampfac = mex_op_opt_scalar(opts, "nufft_upsampfac", 0.0);
    o->nufft_gpu_method = (int)mex_op_opt_scalar(opts, "nufft_gpu_method", 0.0);
}

/* ------------------------------------------------------------------------ */
/* Commands                                                                 */
/* ------------------------------------------------------------------------ */

/**
 * @brief `h = mr_aha_op_mex('create', SEs, phi2, traj, config, mask, opts)`.
 *
 * Validates the inputs exactly like `AhA_Toeplitz_mex` and `calc_toep_mex`,
 * creates the operator with ::mr_op_create on the current (primary) device,
 * registers it, locks the MEX file and returns the handle. See the file
 * documentation for the arguments.
 */
static void mex_op_create(int nlhs, mxArray *plhs[], int nrhs,
                          const mxArray *prhs[])
{
    if (nrhs != 7)
        mexErrMsgIdAndTxt(MR_ID ":usage",
                          "Usage: h = mr_aha_op_mex('create', SEs, phi2, traj, "
                          "config, mask, opts).");
    if (nlhs > 1) mexErrMsgIdAndTxt(MR_ID ":usage", "Too many outputs.");
    const mxArray *mx_cfg = prhs[4], *mx_mask = prhs[5], *mx_opts = prhs[6];

    /* ---- Configuration and options -------------------------------------- */
    mr_config cfg;
    mr_mex_parse_config(mx_cfg, &cfg, MR_ID);
    const int user_coils = (mr_mex_field(mx_cfg, "n_coils") != NULL);
    /* The operator always runs on MATLAB's device: gpuArrays live there. */
    cfg.gpu_device_id = -1;

    mr_op_options opt;
    int dev_buf[MR_OP_MAX_DEVICES];
    mex_op_parse_options(mx_opts, &opt, dev_buf);
    const int share = (opt.flags & MR_OP_SHARE_PRIMARY) != 0;
    const mxArray *mx_toep = mr_mex_field(mx_opts, "phi2_toep");

    /* ---- Arrays --------------------------------------------------------- */
    /* phi2 and traj are read unless a finished kernel is supplied and both
     * are empty (they are not needed then). */
    const mxGPUArray *g_SEs  = mxGPUCreateFromMxArray(prhs[1]);
    mr_mex_check_complex_single(g_SEs, "SEs", MR_ID);
    mex_op_infer_n_coils(g_SEs, &cfg, user_coils);

    const mxGPUArray *g_phi2 = NULL, *g_traj = NULL, *g_toep = NULL;
    const mxGPUArray *g_mask = NULL;
    int64_t Ntrajs = 0, Nkx = 0;
    /* Error exit for the checks below: mexErrMsgIdAndTxt does not return, and
     * MATLAB does not destroy mxGPUArray wrappers by itself, so every wrapper
     * created so far is destroyed first. */
    auto destroy_arrays = [&]() {
        if (g_SEs)  mxGPUDestroyGPUArray(g_SEs);
        if (g_phi2) mxGPUDestroyGPUArray(g_phi2);
        if (g_traj) mxGPUDestroyGPUArray(g_traj);
        if (g_toep) mxGPUDestroyGPUArray(g_toep);
        if (g_mask) mxGPUDestroyGPUArray(g_mask);
    };
#define MR_OP_CREATE_FAIL(...)                                                 \
    do {                                                                       \
        destroy_arrays();                                                      \
        mexErrMsgIdAndTxt(__VA_ARGS__);                                        \
    } while (0)
    if (!mxIsEmpty(prhs[2]) || !mxIsEmpty(prhs[3]) || mx_toep == NULL) {
        g_phi2 = mxGPUCreateFromMxArray(prhs[2]);
        g_traj = mxGPUCreateFromMxArray(prhs[3]);
        mr_mex_check_complex_single(g_phi2, "phi2", MR_ID);
        mr_mex_check_traj(g_traj, &cfg, &Ntrajs, &Nkx, MR_ID ":traj");
    }
    if (mx_toep != NULL) {
        if (opt.method != MR_OP_TOEPLITZ)
            MR_OP_CREATE_FAIL(MR_ID ":phi2Toep",
                              "a finished kernel is only used by the Toeplitz "
                              "method.");
        g_toep = mxGPUCreateFromMxArray(mx_toep);
        mr_mex_check_complex_single(g_toep, "phi2_toep", MR_ID);
    }

    if (cfg.is_masked) {
        if (opt.method == MR_OP_NUFFT)
            MR_OP_CREATE_FAIL(MR_ID ":maskedNufft",
                              "the NUFFT method has no kernel to mask; set "
                              "config.is_masked = false (or use the Toeplitz "
                              "method).");
        if (mxIsEmpty(mx_mask))
            MR_OP_CREATE_FAIL(MR_ID ":missingMask",
                              "config.is_masked is true, so a mask must be "
                              "supplied.");
        g_mask = mxGPUCreateFromMxArray(mx_mask);
        if (mxGPUGetClassID(g_mask) != mxINT64_CLASS)
            MR_OP_CREATE_FAIL(MR_ID ":mask",
                              "mask must be an int64 gpuArray of 0-based linear "
                              "indices; mr_aha_op.create_op performs the "
                              "conversion.");
        if (cfg.num_masked == 0)
            cfg.num_masked = (int64_t)mxGPUGetNumberOfElements(g_mask);
        if ((int64_t)mxGPUGetNumberOfElements(g_mask) != cfg.num_masked)
            MR_OP_CREATE_FAIL(MR_ID ":mask",
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
    if (rc != MR_SUCCESS) {
        destroy_arrays();
        mr_mex_fail(MR_ID, rc);
    }
    mr_mex_check_numel(g_SEs, mr_se_size(&cfg), "SEs", MR_ID);
    if (g_phi2) {
        const int64_t nkz = (cfg.traj_type == MR_TRAJ_STACK) ? cfg.phi2_dim[2] : 1;
        mr_mex_check_numel(g_phi2, Ntrajs * nkz * cfg.L * cfg.L, "phi2", MR_ID);
    }
    if (g_toep)
        mr_mex_check_numel(g_toep, mr_phi2_toep_size(&cfg), "phi2_toep", MR_ID);

    int primary = -1;
    if (cudaGetDevice(&primary) != cudaSuccess)
        MR_OP_CREATE_FAIL(MR_ID ":cuda", "cudaGetDevice failed: %s",
                          cudaGetErrorString(cudaGetLastError()));
#undef MR_OP_CREATE_FAIL

    /* ---- Operator ------------------------------------------------------- */
    mr_op_inputs in;
    memset(&in, 0, sizeof(in));
    in.d_SEs  = (const mr_complex *)mxGPUGetDataReadOnly(g_SEs);
    in.d_phi2 = g_phi2 ? (const mr_complex *)mxGPUGetDataReadOnly(g_phi2) : NULL;
    in.d_traj = g_traj ? (const float *)mxGPUGetDataReadOnly(g_traj) : NULL;
    in.Ntrajs = Ntrajs;
    in.Nkx    = Nkx;
    in.d_mask = g_mask ? (const int64_t *)mxGPUGetDataReadOnly(g_mask) : NULL;
    in.d_phi2_toep =
        g_toep ? (const mr_complex *)mxGPUGetDataReadOnly(g_toep) : NULL;

    mr_op op = NULL;
    rc = mr_op_create(&cfg, &in, &opt, &op);
    /* The Gram blocks and the trajectory are only needed during creation:
     * the operator keeps its own (transformed) copies. */
    if (g_phi2) mxGPUDestroyGPUArray(g_phi2);
    if (g_traj) mxGPUDestroyGPUArray(g_traj);
    if (rc != MR_SUCCESS) {
        mxGPUDestroyGPUArray(g_SEs);
        if (g_toep) mxGPUDestroyGPUArray(g_toep);
        if (g_mask) mxGPUDestroyGPUArray(g_mask);
        mr_mex_fail(MR_ID, rc);
    }

    /* ---- Registry entry ------------------------------------------------- */
    mex_op_entry *e = (mex_op_entry *)mxCalloc(1, sizeof(mex_op_entry));
    mexMakeMemoryPersistent(e);
    e->op      = op;
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
        /* The operator reads these in place: keep them alive until destroy. */
        e->keep[0] = g_SEs;
        e->keep[1] = g_toep;
        e->keep[2] = g_mask;
    } else {
        /* The operator holds its own copies; the references can go. */
        mxGPUDestroyGPUArray(g_SEs);
        if (g_toep) mxGPUDestroyGPUArray(g_toep);
        if (g_mask) mxGPUDestroyGPUArray(g_mask);
    }

    if (!g_atexit_registered) {
        mexAtExit(mex_op_at_exit);
        g_atexit_registered = 1;
    }
    if (g_entries.empty() && !mexIsLocked()) mexLock();
    g_entries.push_back(e);

    plhs[0] = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *mxGetUint64s(plhs[0]) = (uint64_t)(uintptr_t)op;
}

/**
 * @brief `AhAx = mr_aha_op_mex('apply', h, x)`.
 *
 * Refuses to run when the operator is not registered, when MATLAB's current
 * device is not the operator's primary device, or when that device has been
 * reset or deselected since the operator was created.
 */
static void mex_op_apply(int nlhs, mxArray *plhs[], int nrhs,
                         const mxArray *prhs[])
{
    if (nrhs != 3)
        mexErrMsgIdAndTxt(MR_ID ":usage",
                          "Usage: AhAx = mr_aha_op_mex('apply', h, x).");
    if (nlhs > 1) mexErrMsgIdAndTxt(MR_ID ":usage", "Too many outputs.");

    mex_op_entry *e = mex_op_lookup(prhs[1], "apply");
    mex_op_require_device(e, "apply");

    const mxGPUArray *g_x = mxGPUCreateFromMxArray(prhs[2]);
    mr_mex_check_complex_single(g_x, "x", MR_ID);
    mr_mex_check_numel(g_x, mr_x_size(&e->cfg), "x", MR_ID);

    const mwSize  nd   = mxGPUGetNumberOfDimensions(g_x);
    const mwSize *dims = mxGPUGetDimensions(g_x);
    mxGPUArray *g_out = mxGPUCreateGPUArray(nd, dims, mxSINGLE_CLASS, mxCOMPLEX,
                                            MX_GPU_DO_NOT_INITIALIZE);
    mxFree((void *)dims);

    const int rc = mr_op_execute(e->op,
                                 (const mr_complex *)mxGPUGetDataReadOnly(g_x),
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
 * @brief `info = mr_aha_op_mex('info', h)`: the description of
 *        ::mr_op_info as a struct (0-based CUDA ordinals and coil indices, as
 *        the library reports them), plus `primary`, `n_coils`,
 *        `share_primary` and `valid` (false once the primary device was reset
 *        or deselected). `method` and `toep_source` are strings.
 */
static void mex_op_info(int nlhs, mxArray *plhs[], int nrhs,
                        const mxArray *prhs[])
{
    if (nrhs != 2)
        mexErrMsgIdAndTxt(MR_ID ":usage",
                          "Usage: info = mr_aha_op_mex('info', h).");
    if (nlhs > 1) mexErrMsgIdAndTxt(MR_ID ":usage", "Too many outputs.");

    mex_op_entry *e = mex_op_lookup(prhs[1], "info");
    mr_op_info_t in;
    const int rc = mr_op_info(e->op, &in);
    if (rc != MR_SUCCESS) mr_mex_fail(MR_ID, rc);

    static const char *names[] = {
        "method", "n_devices", "devices", "coil_begin", "coil_count", "group",
        "n_split_coils", "peer_access", "peer_bandwidth_gbs", "primary",
        "n_coils", "share_primary", "valid", "toep_source", "toep_devices",
        "toep_ms", "toep_component_ms", "toep_gather_gbs", "create_ms"};
    /* Indexed by MR_OP_TOEP_SUPPLIED .. MR_OP_TOEP_NONE. */
    static const char *sources[] = {"supplied", "broadcast", "split",
                                    "replicate", "none"};
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
    mxSetField(s, 0, "method",
               mxCreateString(in.method == MR_OP_NUFFT ? "nufft" : "toeplitz"));
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
    mxSetField(s, 0, "valid", mxCreateLogicalScalar(mex_op_device_valid(e)));
    mxSetField(s, 0, "toep_source",
               mxCreateString((in.toep_source >= 0 && in.toep_source <= 4)
                                  ? sources[in.toep_source]
                                  : "unknown"));
    mxSetField(s, 0, "toep_devices", mxCreateDoubleScalar((double)in.toep_devices));
    mxSetField(s, 0, "toep_ms", mxCreateDoubleScalar(in.toep_ms));
    mxSetField(s, 0, "toep_component_ms",
               mxCreateDoubleScalar(in.toep_component_ms));
    mxSetField(s, 0, "toep_gather_gbs", mxCreateDoubleScalar(in.toep_gather_gbs));
    mxSetField(s, 0, "create_ms", mxCreateDoubleScalar(in.create_ms));
    plhs[0] = s;
}

/**
 * @brief `Om = mr_aha_op_mex('kernel', h, k)`: a copy of the `phi2_toep` the
 *        operator holds on its device `k` (0-based position in
 *        `info.devices`; 0 is the primary), on the primary device, shaped
 *        like `calc_toep`'s output.
 *
 * A copy (not a view): MATLAB cannot wrap memory it does not own in a
 * gpuArray, and a gpuArray can only live on the current device. Refuses to
 * run when the primary device was reset or deselected, and raises
 * `MR_AHA:mr_aha_op:noKernel` for a NUFFT operator.
 */
static void mex_op_kernel(int nlhs, mxArray *plhs[], int nrhs,
                          const mxArray *prhs[])
{
    if (nrhs != 3)
        mexErrMsgIdAndTxt(MR_ID ":usage",
                          "Usage: Om = mr_aha_op_mex('kernel', h, k).");
    if (nlhs > 1) mexErrMsgIdAndTxt(MR_ID ":usage", "Too many outputs.");
    mex_op_entry *e = mex_op_lookup(prhs[1], "kernel");
    mex_op_require_device(e, "kernel");
    if (!(mxIsNumeric(prhs[2]) && mxGetNumberOfElements(prhs[2]) == 1))
        mexErrMsgIdAndTxt(MR_ID ":usage", "kernel: k must be a numeric scalar.");
    const int k = (int)mxGetScalar(prhs[2]);

    mr_op_info_t in;
    int rc = mr_op_info(e->op, &in);
    if (rc != MR_SUCCESS) mr_mex_fail(MR_ID, rc);
    if (in.method != MR_OP_TOEPLITZ)
        mexErrMsgIdAndTxt(MR_ID ":noKernel",
                          "kernel: a NUFFT operator has no Toeplitz kernel.");
    if (k < 0 || k >= in.n_devices)
        mexErrMsgIdAndTxt(MR_ID ":usage",
                          "kernel: the operator uses %d device(s); k = %d is out "
                          "of range.", in.n_devices, k + 1);
    const mr_complex *src = NULL;
    rc = mr_op_kernel(e->op, k, &src);
    if (rc != MR_SUCCESS) mr_mex_fail(MR_ID, rc);

    mwSize dims[4];
    const mwSize nd = mr_mex_toep_dims(&e->cfg, dims);
    mxGPUArray *g = mxGPUCreateGPUArray(nd, dims, mxSINGLE_CLASS, mxCOMPLEX,
                                        MX_GPU_DO_NOT_INITIALIZE);
    const size_t bytes = (size_t)mr_phi2_toep_size(&e->cfg) * sizeof(mr_complex);
    /* Synchronous with respect to all work on both devices (legacy
     * semantics), so it is ordered after MATLAB's work and the operator's. */
    const cudaError_t ce = cudaMemcpyPeer(mxGPUGetData(g), e->primary, src,
                                          in.devices[k], bytes);
    if (ce != cudaSuccess) {
        mxGPUDestroyGPUArray(g);
        mexErrMsgIdAndTxt(MR_ID ":cuda", "kernel: copy failed: %s",
                          cudaGetErrorString(ce));
    }
    plhs[0] = mxGPUCreateMxArrayOnGPU(g);
    mxGPUDestroyGPUArray(g);
}

/**
 * @brief `mr_aha_op_mex('destroy', h)`: release an operator; a handle that is
 *        not (or no longer) registered is ignored.
 */
static void mex_op_destroy(int nlhs, mxArray *plhs[], int nrhs,
                           const mxArray *prhs[])
{
    (void)plhs;
    if (nrhs != 2)
        mexErrMsgIdAndTxt(MR_ID ":usage",
                          "Usage: mr_aha_op_mex('destroy', h).");
    if (nlhs > 0) mexErrMsgIdAndTxt(MR_ID ":usage", "destroy has no outputs.");
    if (!mxIsUint64(prhs[1]) || mxGetNumberOfElements(prhs[1]) != 1)
        mexErrMsgIdAndTxt(MR_ID ":badHandle",
                          "destroy: the operator handle must be a uint64 scalar.");
    const int i = mex_op_find(*mxGetUint64s(prhs[1]));
    if (i >= 0) mex_op_remove(i);
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
                          "'apply', 'info', 'kernel', 'destroy' or 'count'.");
    char cmd[16];
    if (mxGetString(prhs[0], cmd, sizeof(cmd)) != 0)
        mexErrMsgIdAndTxt(MR_ID ":usage", "Unknown command.");

    if (strcmp(cmd, "count") == 0) {
        plhs[0] = mxCreateDoubleScalar((double)g_entries.size());
        return;
    }

    /* mxInitGPU must run before any other MathWorks GPU API call. */
    mxInitGPU();

    if (strcmp(cmd, "create") == 0)       mex_op_create(nlhs, plhs, nrhs, prhs);
    else if (strcmp(cmd, "apply") == 0)   mex_op_apply(nlhs, plhs, nrhs, prhs);
    else if (strcmp(cmd, "info") == 0)    mex_op_info(nlhs, plhs, nrhs, prhs);
    else if (strcmp(cmd, "kernel") == 0)  mex_op_kernel(nlhs, plhs, nrhs, prhs);
    else if (strcmp(cmd, "destroy") == 0) mex_op_destroy(nlhs, plhs, nrhs, prhs);
    else
        mexErrMsgIdAndTxt(MR_ID ":usage",
                          "Unknown command '%s': expected 'create', 'apply', "
                          "'info', 'kernel', 'destroy' or 'count'.", cmd);
}
