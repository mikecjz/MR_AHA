/**
 * @file example_aha.cu
 * @brief Minimal standalone C++/CUDA use of the MR_AHA library.
 *
 * Demonstrates the whole pipeline without MATLAB or Python:
 *   1. build a synthetic stack-of-stars trajectory and Gram blocks,
 *   2. create the @f$A^H A@f$ operator with ::mr_op_create (Toeplitz method:
 *      the kernel is computed inside), on the current GPU,
 *   3. apply it several times with ::mr_op_execute,
 *   4. create the same operator with the NUFFT method and compare.
 *
 * The example also performs the standard Hermitian check
 * @f$\langle y, A^H A x\rangle = \langle A^H A y, x\rangle@f$, which is the
 * cheapest end-to-end sanity test of the operator. To use several GPUs, set
 * ::mr_op_options::devices (or `n_devices = MR_OP_ALL_DEVICES`).
 *
 * Build with `-DMR_AHA_BUILD_EXAMPLES=ON`.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_aha/mr_aha.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

/** @brief Abort with a message when an MR_AHA call fails. */
#define CHECK(call)                                                            \
    do {                                                                       \
        int rc_ = (call);                                                      \
        if (rc_ != MR_SUCCESS) {                                               \
            std::fprintf(stderr, "%s failed: %s: %s\n", #call,                 \
                         mr_status_string(rc_), mr_last_error());              \
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

    mr_config cfg;
    CHECK(mr_config_defaults(&cfg, MR_TRAJ_STACK));
    cfg.n_dim[0] = Nx;      cfg.n_dim[1] = Ny;      cfg.n_dim[2] = Nz;
    cfg.phi2_dim[0] = 2*Nx; cfg.phi2_dim[1] = 2*Ny; cfg.phi2_dim[2] = Nz;
    cfg.L = L;
    cfg.n_coils = Ncoils;
    cfg.triangulation = MR_TRI_LOWER;
    cfg.gpu_device_id = 0;
    CHECK(mr_config_validate(&cfg));

    const int64_t ntri     = mr_num_tri(L);
    const int64_t x_elems  = mr_x_size(&cfg);
    const int64_t se_elems = mr_se_size(&cfg);
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

    /* Hermitian positive-semidefinite Gram blocks in the MR-Multitasking
     * convention: Phi2 = Phi Phi^H for a random rank x time basis Phi = B
     * (L x L here). The library applies them as v <- v * Phi2. */
    std::vector<float2> h_phi2((size_t)Ntrajs * Nz * L * L);
    for (int64_t s = 0; s < Ntrajs; ++s) {
        for (int64_t z = 0; z < Nz; ++z) {
            std::vector<float2> B((size_t)L * L);
            for (auto &v : B) v = {frand(seed), frand(seed)};
            for (int i = 0; i < L; ++i) {
                for (int j = 0; j < L; ++j) {
                    float re = 0.0f, im = 0.0f;
                    for (int k = 0; k < L; ++k) {
                        /* (B B^H)_{ij} = sum_k B_{ik} conj(B_{jk}) */
                        const float2 a = B[i + (size_t)L * k];
                        const float2 b = B[j + (size_t)L * k];
                        re += a.x * b.x + a.y * b.y;
                        im += a.y * b.x - a.x * b.y;
                    }
                    const size_t idx = (size_t)(s + Ntrajs * (z + Nz * (i + L * j)));
                    h_phi2[idx] = {re, im};
                }
            }
        }
    }

    /* ---- Device buffers ------------------------------------------------- */
    float2 *d_x = NULL, *d_y = NULL, *d_se = NULL, *d_phi2 = NULL;
    float2 *d_ax = NULL, *d_ay = NULL;
    float  *d_traj = NULL;
    cudaMalloc(&d_x,    x_elems * sizeof(float2));
    cudaMalloc(&d_y,    x_elems * sizeof(float2));
    cudaMalloc(&d_ax,   x_elems * sizeof(float2));
    cudaMalloc(&d_ay,   x_elems * sizeof(float2));
    cudaMalloc(&d_se,   se_elems * sizeof(float2));
    cudaMalloc(&d_phi2, h_phi2.size() * sizeof(float2));
    cudaMalloc(&d_traj, h_traj.size() * sizeof(float));
    cudaMemcpy(d_x, h_x.data(), x_elems * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), x_elems * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_se, h_se.data(), se_elems * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi2, h_phi2.data(), h_phi2.size() * sizeof(float2),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_traj, h_traj.data(), h_traj.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    /* ---- 1. Create the operator (Toeplitz: kernel computed inside) ----- */
    mr_op_inputs in = {};
    in.d_SEs  = (const mr_complex *)d_se;
    in.d_phi2 = (const mr_complex *)d_phi2;
    in.d_traj = d_traj;
    in.Ntrajs = Ntrajs;
    in.Nkx    = Nkx;
    mr_op_options opt;
    CHECK(mr_op_options_defaults(&opt)); /* Toeplitz, current GPU only */
    mr_op op = NULL;
    CHECK(mr_op_create(&cfg, &in, &opt, &op));
    mr_op_info_t info;
    CHECK(mr_op_info(op, &info));
    std::printf("operator created in %.0f ms (kernel %.0f ms, %lld x %lld "
                "Toeplitz coefficients) on %d GPU(s)\n",
                info.create_ms, info.toep_ms, (long long)mr_pad_size(&cfg),
                (long long)ntri, info.n_devices);

    /* ---- 2. Apply AhA twice; the phi2/traj buffers are no longer needed - */
    CHECK(mr_op_execute(op, (const mr_complex *)d_x, (mr_complex *)d_ax));
    CHECK(mr_op_execute(op, (const mr_complex *)d_y, (mr_complex *)d_ay));
    cudaDeviceSynchronize();
    CHECK(mr_op_destroy(op));

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

    /* ---- 4. The same operator with the NUFFT method --------------------- */
    opt.method = MR_OP_NUFFT;
    CHECK(mr_op_create(&cfg, &in, &opt, &op));
    CHECK(mr_op_execute(op, (const mr_complex *)d_x, (mr_complex *)d_ay));
    cudaDeviceSynchronize();
    CHECK(mr_op_destroy(op));
    std::vector<float2> h_nu(x_elems);
    cudaMemcpy(h_nu.data(), d_ay, x_elems * sizeof(float2), cudaMemcpyDeviceToHost);
    double num = 0.0, den = 0.0;
    for (int64_t i = 0; i < x_elems; ++i) {
        const double dr = (double)h_nu[i].x - h_ax[i].x, di = (double)h_nu[i].y - h_ax[i].y;
        num += dr * dr + di * di;
        den += (double)h_ax[i].x * h_ax[i].x + (double)h_ax[i].y * h_ax[i].y;
    }
    const double rel_nu = std::sqrt(num / (den + 1e-30));
    std::printf("NUFFT vs Toeplitz method: relative difference = %.3e\n", rel_nu);

    cudaFree(d_x); cudaFree(d_y); cudaFree(d_ax); cudaFree(d_ay);
    cudaFree(d_se); cudaFree(d_phi2); cudaFree(d_traj);
    return (rel < 1e-4 && rel_nu < 1e-4) ? EXIT_SUCCESS : EXIT_FAILURE;
}
