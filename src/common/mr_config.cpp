/**
 * @file mr_config.cpp
 * @brief Configuration defaults, validation, derived sizes and error reporting.
 *
 * Backend-independent: compiled into both the CUDA and the Metal build.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_common.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>

/* ------------------------------------------------------------------------ */
/* Error reporting                                                          */
/* ------------------------------------------------------------------------ */

/** @brief Capacity of the thread-local diagnostic buffer, in bytes. */
#define MR_ERROR_BUFFER_SIZE 512

/**
 * @brief Thread-local diagnostic buffer backing ::mr_last_error.
 *
 * Thread-local so that concurrent host threads driving independent plans never
 * overwrite one another's diagnostics.
 */
static thread_local char mr_error_buffer[MR_ERROR_BUFFER_SIZE] = {0};

void mr_set_error(const char *fmt, ...)
{
    /* vsnprintf truncates to the buffer and always NUL-terminates, so an
     * over-long diagnostic is cut short rather than overflowing. A later
     * failure on the same thread overwrites the message. */
    va_list args;
    va_start(args, fmt);
    vsnprintf(mr_error_buffer, MR_ERROR_BUFFER_SIZE, fmt, args);
    va_end(args);
}

void mr_clear_error(void)
{
    mr_error_buffer[0] = '\0';
    mr_backend_clear_error();
}

/* Returns this thread's buffer: valid until the next library call on the same
 * thread that clears or sets the error; empty ("") when nothing failed. */
extern "C" const char *mr_last_error(void) { return mr_error_buffer; }

extern "C" const char *mr_status_string(int status)
{
    switch (status) {
    case MR_SUCCESS:            return "success";
    case MR_ERR_INVALID_ARG:    return "invalid argument";
    case MR_ERR_INVALID_CONFIG: return "invalid configuration";
    case MR_ERR_ALLOC:          return "GPU memory allocation failed";
    case MR_ERR_CUDA:           return "CUDA runtime error";
    case MR_ERR_CUFFT:          return "cuFFT error";
    case MR_ERR_CUFINUFFT:      return "cufinufft error";
    case MR_ERR_UNSUPPORTED:    return "unsupported configuration";
    case MR_ERR_INTERNAL:       return "internal error";
    case MR_ERR_METAL:          return "Metal error";
    case MR_ERR_VKFFT:          return "VkFFT error";
    default:                    return "unknown status";
    }
}

/* ------------------------------------------------------------------------ */
/* Configuration                                                            */
/* ------------------------------------------------------------------------ */

extern "C" int mr_config_defaults(mr_config *cfg, mr_traj_type traj_type)
{
    mr_clear_error();
    MR_REQUIRE(cfg != NULL, "mr_config_defaults: cfg is NULL");

    /* Zero everything first so that fields not listed below (and any padding)
     * have a defined value, then set the documented defaults. Sizes default
     * to 1 so that unused axes are already valid; the caller fills in the
     * real extents. */
    memset(cfg, 0, sizeof(*cfg));
    cfg->traj_type       = traj_type;
    cfg->n_spatial       = (traj_type == MR_TRAJ_2D) ? 2 : 3;
    cfg->n_dim[0]        = 1;
    cfg->n_dim[1]        = 1;
    cfg->n_dim[2]        = 1;
    cfg->phi2_dim[0]     = 1;
    cfg->phi2_dim[1]     = 1;
    cfg->phi2_dim[2]     = 1;
    cfg->L               = 1;
    cfg->triangulation   = MR_TRI_LOWER;
    cfg->is_masked       = 0;
    cfg->num_masked      = 0;
    cfg->n_coils         = 1;
    cfg->eps             = 1e-6;
    cfg->gpu_device_id   = 0;
    cfg->stream          = NULL;
    cfg->mem_fraction    = 0.5;
    cfg->verbose         = 0;
    cfg->mask_radius     = MR_DEFAULT_MASK_RADIUS;
    return MR_SUCCESS;
}

extern "C" int mr_config_validate(const mr_config *cfg)
{
    mr_clear_error();
    MR_REQUIRE(cfg != NULL, "mr_config_validate: cfg is NULL");

    MR_REQUIRE_CFG(cfg->traj_type == MR_TRAJ_2D || cfg->traj_type == MR_TRAJ_3D ||
                       cfg->traj_type == MR_TRAJ_STACK,
                   "traj_type must be one of MR_TRAJ_2D, MR_TRAJ_3D, MR_TRAJ_STACK");

    /* 2D images have two spatial axes; 3D and stack-of-X images have three. */
    const int expect_spatial = (cfg->traj_type == MR_TRAJ_2D) ? 2 : 3;
    MR_REQUIRE_CFG(cfg->n_spatial == expect_spatial,
                   "n_spatial is %d but trajectory type requires %d",
                   cfg->n_spatial, expect_spatial);

    /* Per-axis geometry: positive extents, unused trailing axes exactly 1,
     * and an upsampled grid at least as large as the image (the pad/crop
     * index map of the kernels is only injective for N <= M). */
    for (int d = 0; d < 3; ++d) {
        MR_REQUIRE_CFG(cfg->n_dim[d] >= 1, "n_dim[%d] = %d must be >= 1", d,
                       cfg->n_dim[d]);
        MR_REQUIRE_CFG(cfg->phi2_dim[d] >= 1, "phi2_dim[%d] = %d must be >= 1", d,
                       cfg->phi2_dim[d]);
        if (d >= cfg->n_spatial) {
            MR_REQUIRE_CFG(cfg->n_dim[d] == 1 && cfg->phi2_dim[d] == 1,
                           "n_dim[%d] and phi2_dim[%d] must be 1 for a %dD "
                           "trajectory",
                           d, d, cfg->n_spatial);
        }
        MR_REQUIRE_CFG(cfg->phi2_dim[d] >= cfg->n_dim[d],
                       "phi2_dim[%d] = %d must not be smaller than n_dim[%d] = %d",
                       d, cfg->phi2_dim[d], d, cfg->n_dim[d]);
    }

    /* kz of a stack trajectory is already Cartesian, so it is never upsampled. */
    if (cfg->traj_type == MR_TRAJ_STACK)
        MR_REQUIRE_CFG(cfg->phi2_dim[2] == cfg->n_dim[2],
                       "for MR_TRAJ_STACK, phi2_dim[2] (Mz = %d) must equal "
                       "n_dim[2] (Nz = %d)",
                       cfg->phi2_dim[2], cfg->n_dim[2]);

    /* Scalar parameters. eps is the requested cufinufft tolerance (of
     * mr_calc_toep and of the NUFFT operator), a relative accuracy in (0, 1). */
    MR_REQUIRE_CFG(cfg->L >= 1, "L = %d must be >= 1", cfg->L);
    MR_REQUIRE_CFG(cfg->n_coils >= 1, "n_coils = %d must be >= 1", cfg->n_coils);
    MR_REQUIRE_CFG(cfg->triangulation == MR_TRI_LOWER ||
                       cfg->triangulation == MR_TRI_UPPER,
                   "triangulation must be MR_TRI_LOWER or MR_TRI_UPPER");
    MR_REQUIRE_CFG(cfg->eps > 0.0 && cfg->eps < 1.0,
                   "eps = %g must lie in (0, 1)", cfg->eps);
    MR_REQUIRE_CFG(cfg->mask_radius >= 0, "mask_radius = %d must be >= 0",
                   cfg->mask_radius);

    /* A masked kernel keeps between 1 and all of the upsampled locations.
     * The mask contents themselves (range, sortedness) are device data and
     * are not checked here. */
    if (cfg->is_masked) {
        const int64_t pad = mr_pad_size(cfg);
        MR_REQUIRE_CFG(cfg->num_masked > 0,
                       "num_masked = %lld must be > 0 when is_masked is set",
                       (long long)cfg->num_masked);
        MR_REQUIRE_CFG(cfg->num_masked <= pad,
                       "num_masked = %lld exceeds the upsampled grid size %lld",
                       (long long)cfg->num_masked, (long long)pad);
    }

    return MR_SUCCESS;
}

/* ------------------------------------------------------------------------ */
/* Derived sizes                                                            */
/* ------------------------------------------------------------------------ */

extern "C" int64_t mr_num_tri(int L)
{
    if (L <= 0) return 0;
    return (int64_t)L * (L + 1) / 2;
}

extern "C" int64_t mr_image_size(const mr_config *cfg)
{
    if (cfg == NULL) return 0;
    return (int64_t)cfg->n_dim[0] * cfg->n_dim[1] * cfg->n_dim[2];
}

extern "C" int64_t mr_pad_size(const mr_config *cfg)
{
    if (cfg == NULL) return 0;
    return (int64_t)cfg->phi2_dim[0] * cfg->phi2_dim[1] * cfg->phi2_dim[2];
}

extern "C" int64_t mr_phi2_toep_size(const mr_config *cfg)
{
    if (cfg == NULL) return 0;
    const int64_t ntri = mr_num_tri(cfg->L);
    return (cfg->is_masked ? cfg->num_masked : mr_pad_size(cfg)) * ntri;
}

extern "C" int64_t mr_x_size(const mr_config *cfg)
{
    if (cfg == NULL) return 0;
    return mr_image_size(cfg) * cfg->L;
}

extern "C" int64_t mr_se_size(const mr_config *cfg)
{
    if (cfg == NULL) return 0;
    /* One map per coil, shared by every rank. */
    return mr_image_size(cfg) * cfg->n_coils;
}

/* ------------------------------------------------------------------------ */
/* Kernel geometry descriptor                                               */
/* ------------------------------------------------------------------------ */

void mr_dims_from_config(const mr_config *cfg, mr_dims *d)
{
    /* Copy the extents and pre-compute the center offsets floor(N/2) and
     * floor(M/2) used by the centered <-> natural index maps of the kernels
     * (integer division of non-negative values is floor). */
    memset(d, 0, sizeof(*d));
    for (int i = 0; i < 3; ++i) {
        d->n[i]   = cfg->n_dim[i];
        d->m[i]   = cfg->phi2_dim[i];
        d->c_n[i] = cfg->n_dim[i] / 2;
        d->c_m[i] = cfg->phi2_dim[i] / 2;
    }
    d->L          = cfg->L;
    d->tri_upper  = (cfg->triangulation == MR_TRI_UPPER) ? 1 : 0;
    d->image_size = (long long)mr_image_size(cfg);
    d->pad_size   = (long long)mr_pad_size(cfg);
    d->se_coil    = d->image_size;
}
