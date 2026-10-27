/**
 * @file mr_config.cu
 * @brief Configuration defaults, validation, derived sizes and error reporting.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_internal.cuh"

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
    va_list args;
    va_start(args, fmt);
    vsnprintf(mr_error_buffer, MR_ERROR_BUFFER_SIZE, fmt, args);
    va_end(args);
}

void mr_clear_error(void) { mr_error_buffer[0] = '\0'; }

extern "C" const char *mr_last_error(void) { return mr_error_buffer; }

extern "C" const char *mr_status_string(int status)
{
    switch (status) {
    case MR_SUCCESS:            return "success";
    case MR_ERR_INVALID_ARG:    return "invalid argument";
    case MR_ERR_INVALID_CONFIG: return "invalid configuration";
    case MR_ERR_ALLOC:          return "device allocation failed";
    case MR_ERR_CUDA:           return "CUDA runtime error";
    case MR_ERR_CUFFT:          return "cuFFT error";
    case MR_ERR_CUFINUFFT:      return "cufinufft error";
    case MR_ERR_UNSUPPORTED:    return "unsupported configuration";
    case MR_ERR_INTERNAL:       return "internal error";
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
    cfg->mask_threshold  = MR_DEFAULT_MASK_THRESHOLD;
    cfg->mask_method     = MR_MASK_NEIGHBOUR;
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

    const int expect_spatial = (cfg->traj_type == MR_TRAJ_2D) ? 2 : 3;
    MR_REQUIRE_CFG(cfg->n_spatial == expect_spatial,
                   "n_spatial is %d but trajectory type requires %d",
                   cfg->n_spatial, expect_spatial);

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

    MR_REQUIRE_CFG(cfg->L >= 1, "L = %d must be >= 1", cfg->L);
    MR_REQUIRE_CFG(cfg->n_coils >= 1, "n_coils = %d must be >= 1", cfg->n_coils);
    MR_REQUIRE_CFG(cfg->triangulation == MR_TRI_LOWER ||
                       cfg->triangulation == MR_TRI_UPPER,
                   "triangulation must be MR_TRI_LOWER or MR_TRI_UPPER");
    MR_REQUIRE_CFG(cfg->eps > 0.0 && cfg->eps < 1.0,
                   "eps = %g must lie in (0, 1)", cfg->eps);
    MR_REQUIRE_CFG(cfg->mask_threshold >= 0.0,
                   "mask_threshold = %g must be >= 0", cfg->mask_threshold);
    MR_REQUIRE_CFG(cfg->mask_method == MR_MASK_NEIGHBOUR ||
                       cfg->mask_method == MR_MASK_THRESHOLD,
                   "mask_method must be MR_MASK_NEIGHBOUR or MR_MASK_THRESHOLD");
    MR_REQUIRE_CFG(cfg->mask_radius >= 0, "mask_radius = %d must be >= 0",
                   cfg->mask_radius);

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

extern "C" int64_t mr_mask_kernel_size(const mr_config *cfg)
{
    if (cfg == NULL) return 0;
    /* 2D and stack kernels depend on the in-plane trajectory only. */
    const int64_t plane = (int64_t)cfg->phi2_dim[0] * cfg->phi2_dim[1];
    return (cfg->traj_type == MR_TRAJ_3D) ? plane * cfg->phi2_dim[2] : plane;
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
