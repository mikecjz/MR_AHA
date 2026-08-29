/**
 * @file mt_config.cu
 * @brief Configuration defaults, validation, derived sizes and error reporting.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mt_internal.cuh"

#include <cstdarg>
#include <cstdio>
#include <cstring>

/* ------------------------------------------------------------------------ */
/* Error reporting                                                          */
/* ------------------------------------------------------------------------ */

/** @brief Capacity of the thread-local diagnostic buffer, in bytes. */
#define MT_ERROR_BUFFER_SIZE 512

/**
 * @brief Thread-local diagnostic buffer backing ::mt_last_error.
 *
 * Thread-local so that concurrent host threads driving independent plans never
 * overwrite one another's diagnostics.
 */
static thread_local char mt_error_buffer[MT_ERROR_BUFFER_SIZE] = {0};

void mt_set_error(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vsnprintf(mt_error_buffer, MT_ERROR_BUFFER_SIZE, fmt, args);
    va_end(args);
}

void mt_clear_error(void) { mt_error_buffer[0] = '\0'; }

extern "C" const char *mt_last_error(void) { return mt_error_buffer; }

extern "C" const char *mt_status_string(int status)
{
    switch (status) {
    case MT_SUCCESS:            return "success";
    case MT_ERR_INVALID_ARG:    return "invalid argument";
    case MT_ERR_INVALID_CONFIG: return "invalid configuration";
    case MT_ERR_ALLOC:          return "device allocation failed";
    case MT_ERR_CUDA:           return "CUDA runtime error";
    case MT_ERR_CUFFT:          return "cuFFT error";
    case MT_ERR_CUFINUFFT:      return "cufinufft error";
    case MT_ERR_UNSUPPORTED:    return "unsupported configuration";
    case MT_ERR_INTERNAL:       return "internal error";
    default:                    return "unknown status";
    }
}

/* ------------------------------------------------------------------------ */
/* Configuration                                                            */
/* ------------------------------------------------------------------------ */

extern "C" int mt_config_defaults(mt_config *cfg, mt_traj_type traj_type)
{
    mt_clear_error();
    MT_REQUIRE(cfg != NULL, "mt_config_defaults: cfg is NULL");

    memset(cfg, 0, sizeof(*cfg));
    cfg->traj_type       = traj_type;
    cfg->n_spatial       = (traj_type == MT_TRAJ_2D) ? 2 : 3;
    cfg->n_dim[0]        = 1;
    cfg->n_dim[1]        = 1;
    cfg->n_dim[2]        = 1;
    cfg->phi2_dim[0]     = 1;
    cfg->phi2_dim[1]     = 1;
    cfg->phi2_dim[2]     = 1;
    cfg->L               = 1;
    cfg->triangulation   = MT_TRI_LOWER;
    cfg->is_masked       = 0;
    cfg->num_masked      = 0;
    cfg->n_coils         = 1;
    cfg->eps             = 1e-6;
    cfg->gpu_device_id   = 0;
    cfg->stream          = NULL;
    cfg->mem_fraction    = 0.5;
    cfg->verbose         = 0;
    return MT_SUCCESS;
}

extern "C" int mt_config_validate(const mt_config *cfg)
{
    mt_clear_error();
    MT_REQUIRE(cfg != NULL, "mt_config_validate: cfg is NULL");

    MT_REQUIRE_CFG(cfg->traj_type == MT_TRAJ_2D || cfg->traj_type == MT_TRAJ_3D ||
                       cfg->traj_type == MT_TRAJ_STACK,
                   "traj_type must be one of MT_TRAJ_2D, MT_TRAJ_3D, MT_TRAJ_STACK");

    const int expect_spatial = (cfg->traj_type == MT_TRAJ_2D) ? 2 : 3;
    MT_REQUIRE_CFG(cfg->n_spatial == expect_spatial,
                   "n_spatial is %d but trajectory type requires %d",
                   cfg->n_spatial, expect_spatial);

    for (int d = 0; d < 3; ++d) {
        MT_REQUIRE_CFG(cfg->n_dim[d] >= 1, "n_dim[%d] = %d must be >= 1", d,
                       cfg->n_dim[d]);
        MT_REQUIRE_CFG(cfg->phi2_dim[d] >= 1, "phi2_dim[%d] = %d must be >= 1", d,
                       cfg->phi2_dim[d]);
        if (d >= cfg->n_spatial) {
            MT_REQUIRE_CFG(cfg->n_dim[d] == 1 && cfg->phi2_dim[d] == 1,
                           "n_dim[%d] and phi2_dim[%d] must be 1 for a %dD "
                           "trajectory",
                           d, d, cfg->n_spatial);
        }
        MT_REQUIRE_CFG(cfg->phi2_dim[d] >= cfg->n_dim[d],
                       "phi2_dim[%d] = %d must not be smaller than n_dim[%d] = %d",
                       d, cfg->phi2_dim[d], d, cfg->n_dim[d]);
    }

    /* kz of a stack trajectory is already Cartesian, so it is never upsampled. */
    if (cfg->traj_type == MT_TRAJ_STACK)
        MT_REQUIRE_CFG(cfg->phi2_dim[2] == cfg->n_dim[2],
                       "for MT_TRAJ_STACK, phi2_dim[2] (Mz = %d) must equal "
                       "n_dim[2] (Nz = %d)",
                       cfg->phi2_dim[2], cfg->n_dim[2]);

    MT_REQUIRE_CFG(cfg->L >= 1, "L = %d must be >= 1", cfg->L);
    MT_REQUIRE_CFG(cfg->n_coils >= 1, "n_coils = %d must be >= 1", cfg->n_coils);
    MT_REQUIRE_CFG(cfg->triangulation == MT_TRI_LOWER ||
                       cfg->triangulation == MT_TRI_UPPER,
                   "triangulation must be MT_TRI_LOWER or MT_TRI_UPPER");
    MT_REQUIRE_CFG(cfg->eps > 0.0 && cfg->eps < 1.0,
                   "eps = %g must lie in (0, 1)", cfg->eps);

    if (cfg->is_masked) {
        const int64_t pad = mt_pad_size(cfg);
        MT_REQUIRE_CFG(cfg->num_masked > 0,
                       "num_masked = %lld must be > 0 when is_masked is set",
                       (long long)cfg->num_masked);
        MT_REQUIRE_CFG(cfg->num_masked <= pad,
                       "num_masked = %lld exceeds the upsampled grid size %lld",
                       (long long)cfg->num_masked, (long long)pad);
    }

    return MT_SUCCESS;
}

/* ------------------------------------------------------------------------ */
/* Derived sizes                                                            */
/* ------------------------------------------------------------------------ */

extern "C" int64_t mt_num_tri(int L)
{
    if (L <= 0) return 0;
    return (int64_t)L * (L + 1) / 2;
}

extern "C" int64_t mt_image_size(const mt_config *cfg)
{
    if (cfg == NULL) return 0;
    return (int64_t)cfg->n_dim[0] * cfg->n_dim[1] * cfg->n_dim[2];
}

extern "C" int64_t mt_pad_size(const mt_config *cfg)
{
    if (cfg == NULL) return 0;
    return (int64_t)cfg->phi2_dim[0] * cfg->phi2_dim[1] * cfg->phi2_dim[2];
}

extern "C" int64_t mt_phi2_toep_size(const mt_config *cfg)
{
    if (cfg == NULL) return 0;
    const int64_t ntri = mt_num_tri(cfg->L);
    return (cfg->is_masked ? cfg->num_masked : mt_pad_size(cfg)) * ntri;
}

extern "C" int64_t mt_x_size(const mt_config *cfg)
{
    if (cfg == NULL) return 0;
    return mt_image_size(cfg) * cfg->L;
}

extern "C" int64_t mt_se_size(const mt_config *cfg)
{
    if (cfg == NULL) return 0;
    /* One map per coil, shared by every rank. */
    return mt_image_size(cfg) * cfg->n_coils;
}

/* ------------------------------------------------------------------------ */
/* Kernel geometry descriptor                                               */
/* ------------------------------------------------------------------------ */

void mt_dims_from_config(const mt_config *cfg, mt_dims *d)
{
    memset(d, 0, sizeof(*d));
    for (int i = 0; i < 3; ++i) {
        d->n[i]   = cfg->n_dim[i];
        d->m[i]   = cfg->phi2_dim[i];
        d->c_n[i] = cfg->n_dim[i] / 2;
        d->c_m[i] = cfg->phi2_dim[i] / 2;
    }
    d->L          = cfg->L;
    d->tri_upper  = (cfg->triangulation == MT_TRI_UPPER) ? 1 : 0;
    d->image_size = (long long)mt_image_size(cfg);
    d->pad_size   = (long long)mt_pad_size(cfg);
    d->se_coil    = d->image_size;
}
