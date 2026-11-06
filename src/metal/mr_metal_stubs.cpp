/**
 * @file mr_metal_stubs.cpp
 * @brief Placeholders for the Metal entry points that are not ported yet.
 *
 * Each function validates its arguments like the CUDA backend, confirms that a
 * Metal device is available, and returns ::MR_ERR_UNSUPPORTED. They are
 * replaced one by one as the Metal port progresses (see `docs/mlx_plan.md`):
 * the operator (::mr_aha_plan_create and friends) first, then ::mr_calc_toep
 * and the mask functions.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_metal_internal.h"

namespace {

/**
 * @brief Shared body of every stub: check the device, then report the gap.
 * @param[in] what Name of the entry point, for the diagnostic.
 * @return ::MR_ERR_METAL when there is no device, else ::MR_ERR_UNSUPPORTED.
 */
int mr_metal_not_yet(const char *what)
{
    mr_metal_pool pool;
    const mr_metal_ctx *ctx = nullptr;
    const int rc = mr_metal_context(&ctx);
    if (rc != MR_SUCCESS) return rc;
    MR_FAIL(MR_ERR_UNSUPPORTED, "%s is not implemented yet on the Metal backend "
            "(device: %s)", what, ctx->device->name()->utf8String());
}

}  // namespace

extern "C" int mr_aha_plan_create(const mr_config *cfg, mr_aha_plan *plan_out)
{
    mr_clear_error();
    MR_REQUIRE(plan_out != NULL, "mr_aha_plan_create: plan_out is NULL");
    *plan_out = NULL;
    const int status = mr_config_validate(cfg);
    if (status != MR_SUCCESS) return status;
    return mr_metal_not_yet("mr_aha_plan_create");
}

extern "C" int mr_aha_plan_execute(mr_aha_plan plan, const mr_complex *d_x,
                                   const mr_complex *d_SEs,
                                   const mr_complex *d_phi2_toep,
                                   const int64_t *d_mask, mr_complex *d_AhAx)
{
    (void)d_x; (void)d_SEs; (void)d_phi2_toep; (void)d_mask; (void)d_AhAx;
    mr_clear_error();
    MR_REQUIRE(plan != NULL, "mr_aha_plan_execute: plan is NULL");
    return mr_metal_not_yet("mr_aha_plan_execute");
}

extern "C" int mr_aha_plan_destroy(mr_aha_plan plan)
{
    mr_clear_error();
    /* No plan can exist yet, so only NULL can arrive here. */
    return plan == NULL ? MR_SUCCESS : mr_metal_not_yet("mr_aha_plan_destroy");
}

extern "C" int mr_aha_toeplitz(const mr_config *cfg, const mr_complex *d_x,
                               const mr_complex *d_SEs,
                               const mr_complex *d_phi2_toep,
                               const int64_t *d_mask, mr_complex *d_AhAx)
{
    (void)d_x; (void)d_SEs; (void)d_phi2_toep; (void)d_mask; (void)d_AhAx;
    mr_clear_error();
    const int status = mr_config_validate(cfg);
    if (status != MR_SUCCESS) return status;
    return mr_metal_not_yet("mr_aha_toeplitz");
}

extern "C" int mr_calc_toep(const mr_config *cfg, const mr_complex *d_phi2,
                            const float *d_traj, int64_t Ntrajs, int64_t Nkx,
                            const int64_t *d_mask, mr_complex *d_phi2_toep)
{
    (void)d_phi2; (void)d_traj; (void)Ntrajs; (void)Nkx; (void)d_mask;
    (void)d_phi2_toep;
    mr_clear_error();
    const int status = mr_config_validate(cfg);
    if (status != MR_SUCCESS) return status;
    return mr_metal_not_yet("mr_calc_toep");
}

extern "C" int mr_calc_mask_kernel(const mr_config *cfg, const float *d_traj,
                                   int64_t Ntrajs, int64_t Nkx, float *d_kernel)
{
    (void)d_traj; (void)Ntrajs; (void)Nkx; (void)d_kernel;
    mr_clear_error();
    MR_REQUIRE(cfg != NULL, "mr_calc_mask_kernel: cfg is NULL");
    return mr_metal_not_yet("mr_calc_mask_kernel");
}

extern "C" int mr_calc_mask(const mr_config *cfg, const float *d_traj,
                            int64_t Ntrajs, int64_t Nkx, int64_t *d_mask,
                            int64_t *num_masked_out)
{
    (void)d_traj; (void)Ntrajs; (void)Nkx; (void)d_mask;
    mr_clear_error();
    MR_REQUIRE(cfg != NULL, "mr_calc_mask: cfg is NULL");
    MR_REQUIRE(num_masked_out != NULL, "mr_calc_mask: num_masked_out is NULL");
    return mr_metal_not_yet("mr_calc_mask");
}
