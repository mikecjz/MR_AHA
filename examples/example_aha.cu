/**
 * @file example_aha.cu
 * @brief Minimal standalone C++/CUDA use of the MT_CUDA library.
 *
 * Demonstrates the whole pipeline without MATLAB or Python:
 *   1. build a synthetic stack-of-stars trajectory and Gram blocks,
 *   2. pre-compute the Toeplitz operator with ::mt_calc_toep,
 *   3. apply @f$A^H A@f$ with an ::mt_aha_plan reused across iterations.
 *
 * The example also performs the standard Hermitian check
 * @f$\langle y, A^H A x\rangle = \langle A^H A y, x\rangle@f$, which is the
 * cheapest end-to-end sanity test of the operator.
 *
 * Build with `-DMT_CUDA_BUILD_EXAMPLES=ON`.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mt_cuda/mt_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

/** @brief Abort with a message when an MT_CUDA call fails. */
#define CHECK(call)                                                            \
    do {                                                                       \
        int rc_ = (call);                                                      \
        if (rc_ != MT_SUCCESS) {                                               \
            std::fprintf(stderr, "%s failed: %s: %s\n", #call,                 \
                         mt_status_string(rc_), mt_last_error());              \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    } while (0)

/** @brief Deterministic pseudo-random number in [-1, 1), so runs are repeatable. */
static float frand(unsigned &state)
{
    state = state * 1664525u + 1013904223u;
    return (float)((state >> 8) & 0xFFFFFF) / 8388608.0f - 1.0f;
}

/**
 * @brief Entry point.
 * @return `EXIT_SUCCESS` on success, `EXIT_FAILURE` otherwise.
 */
int main(void)
{
    /* ---- A small stack-of-stars problem -------------------------------- */
    const int Nx = 32, Ny = 32, Nz = 4, L = 3, Ncoils = 2;
    const int64_t Ntrajs = 24, Nkx = 64;

    mt_config cfg;
    CHECK(mt_config_defaults(&cfg, MT_TRAJ_STACK));
    cfg.n_dim[0] = Nx;      cfg.n_dim[1] = Ny;      cfg.n_dim[2] = Nz;
    cfg.phi2_dim[0] = 2*Nx; cfg.phi2_dim[1] = 2*Ny; cfg.phi2_dim[2] = Nz;
    cfg.L = L;
    cfg.n_coils = Ncoils;
    cfg.triangulation = MT_TRI_LOWER;
    cfg.gpu_device_id = 0;
    CHECK(mt_config_validate(&cfg));

    const int64_t ntri     = mt_num_tri(L);
    const int64_t x_elems  = mt_x_size(&cfg);
    const int64_t se_elems = mt_se_size(&cfg);
    const int64_t M_nu     = Ntrajs * Nkx;

    /* ---- Host-side synthetic data -------------------------------------- */
    unsigned seed = 12345u;
    std::vector<float2> h_x(x_elems), h_y(x_elems), h_se(se_elems);
    for (auto &v : h_x)  v = {frand(seed), frand(seed)};
    for (auto &v : h_y)  v = {frand(seed), frand(seed)};
    for (auto &v : h_se) v = {frand(seed), frand(seed)};

    /* Radial spokes, in radians per unit of the un-padded image grid. */
    std::vector<float> h_traj(2 * M_nu);
    for (int64_t s = 0; s < Ntrajs; ++s) {
        const double ang = M_PI * (double)s / (double)Ntrajs;
        for (int64_t r = 0; r < Nkx; ++r) {
            const double kr = M_PI * (2.0 * (double)r / (double)(Nkx - 1) - 1.0);
            const int64_t m = s + Ntrajs * r;           /* column-major */
            h_traj[m]        = (float)(kr * std::cos(ang));
            h_traj[m + M_nu] = (float)(kr * std::sin(ang));
        }
    }

    /* Hermitian positive-semidefinite Gram blocks Phi^H Phi = B^H B. */
    std::vector<float2> h_phi2((size_t)Ntrajs * Nz * L * L);
    for (int64_t s = 0; s < Ntrajs; ++s) {
        for (int64_t z = 0; z < Nz; ++z) {
            std::vector<float2> B((size_t)L * L);
            for (auto &v : B) v = {frand(seed), frand(seed)};
            for (int i = 0; i < L; ++i) {
                for (int j = 0; j < L; ++j) {
                    float re = 0.0f, im = 0.0f;
                    for (int k = 0; k < L; ++k) {
                        /* (B^H B)_{ij} = sum_k conj(B_{ki}) B_{kj} */
                        const float2 a = B[k + (size_t)L * i];
                        const float2 b = B[k + (size_t)L * j];
                        re += a.x * b.x + a.y * b.y;
                        im += a.x * b.y - a.y * b.x;
                    }
                    const size_t idx = (size_t)(s + Ntrajs * (z + Nz * (i + L * j)));
                    h_phi2[idx] = {re, im};
                }
            }
        }
    }

    /* ---- Device buffers ------------------------------------------------- */
    float2 *d_x = NULL, *d_y = NULL, *d_se = NULL, *d_phi2 = NULL;
    float2 *d_toep = NULL, *d_ax = NULL, *d_ay = NULL;
    float  *d_traj = NULL;
    cudaMalloc(&d_x,    x_elems * sizeof(float2));
    cudaMalloc(&d_y,    x_elems * sizeof(float2));
    cudaMalloc(&d_ax,   x_elems * sizeof(float2));
    cudaMalloc(&d_ay,   x_elems * sizeof(float2));
    cudaMalloc(&d_se,   se_elems * sizeof(float2));
    cudaMalloc(&d_phi2, h_phi2.size() * sizeof(float2));
    cudaMalloc(&d_traj, h_traj.size() * sizeof(float));
    cudaMalloc(&d_toep, (size_t)mt_pad_size(&cfg) * ntri * sizeof(float2));
    cudaMemcpy(d_x, h_x.data(), x_elems * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), x_elems * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_se, h_se.data(), se_elems * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi2, h_phi2.data(), h_phi2.size() * sizeof(float2),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_traj, h_traj.data(), h_traj.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    /* ---- 1. Pre-compute the Toeplitz operator --------------------------- */
    CHECK(mt_calc_toep(&cfg, (const mt_complex *)d_phi2, d_traj, Ntrajs, Nkx,
                       NULL, (mt_complex *)d_toep));
    std::printf("calc_toep done: %lld x %lld Toeplitz coefficients\n",
                (long long)mt_pad_size(&cfg), (long long)ntri);

    /* ---- 2. Apply AhA twice through a reusable plan --------------------- */
    mt_aha_plan plan = NULL;
    CHECK(mt_aha_plan_create(&cfg, &plan));
    CHECK(mt_aha_plan_execute(plan, (const mt_complex *)d_x,
                              (const mt_complex *)d_se,
                              (const mt_complex *)d_toep, NULL,
                              (mt_complex *)d_ax));
    CHECK(mt_aha_plan_execute(plan, (const mt_complex *)d_y,
                              (const mt_complex *)d_se,
                              (const mt_complex *)d_toep, NULL,
                              (mt_complex *)d_ay));
    CHECK(mt_aha_plan_destroy(plan));

    /* ---- 3. Hermitian check: <y, AhA x> == <AhA y, x> ------------------- */
    std::vector<float2> h_ax(x_elems), h_ay(x_elems);
    cudaMemcpy(h_ax.data(), d_ax, x_elems * sizeof(float2), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ay.data(), d_ay, x_elems * sizeof(float2), cudaMemcpyDeviceToHost);

    double lre = 0.0, lim = 0.0, rre = 0.0, rim = 0.0;
    for (int64_t i = 0; i < x_elems; ++i) {
        lre += (double)h_y[i].x * h_ax[i].x + (double)h_y[i].y * h_ax[i].y;
        lim += (double)h_y[i].x * h_ax[i].y - (double)h_y[i].y * h_ax[i].x;
        rre += (double)h_ay[i].x * h_x[i].x + (double)h_ay[i].y * h_x[i].y;
        rim += (double)h_ay[i].x * h_x[i].y - (double)h_ay[i].y * h_x[i].x;
    }
    const double denom = std::sqrt(lre * lre + lim * lim) + 1e-30;
    const double rel = std::sqrt((lre - rre) * (lre - rre) +
                                 (lim - rim) * (lim - rim)) / denom;
    std::printf("<y, AhA x>   = %.6e %+.6ei\n", lre, lim);
    std::printf("<AhA y, x>   = %.6e %+.6ei\n", rre, rim);
    std::printf("relative difference = %.3e (Hermitian symmetry)\n", rel);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_ax); cudaFree(d_ay);
    cudaFree(d_se); cudaFree(d_phi2); cudaFree(d_traj); cudaFree(d_toep);
    return (rel < 1e-4) ? EXIT_SUCCESS : EXIT_FAILURE;
}
