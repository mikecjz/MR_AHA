/**
 * @file mr_nufft_op.cu
 * @brief NUFFT engine of the @f$A^H A@f$ operator (::MR_OP_NUFFT): type-2
 *        NUFFT, block-diagonal Gram multiplication, type-1 NUFFT per coil.
 *
 * The engine evaluates, on one device and for a contiguous range of coils,
 * @f[
 *   A^H A x \;=\; \sum_c S_c^H \, F_{+}\, \mathcal{B}\, F_{-}\, S_c\, x ,
 * @f]
 * where @f$F_{-}@f$ is a type-2 NUFFT (uniform image to the non-uniform
 * samples, sign @f$-1@f$), @f$F_{+} = F_{-}^H@f$ the matching type-1 NUFFT
 * (sign @f$+1@f$) and @f$\mathcal{B}@f$ multiplies the length-@f$L@f$ rank
 * vector of every sample, as a row vector from the right (MR-Multitasking,
 * see @ref mr_gram), by the Gram block of its readout:
 * @f$v \leftarrow v\,\Phi_2@f$. Both NUFFTs are unnormalised, which is the
 * convention the Toeplitz kernel of ::mr_calc_toep reproduces, so the two
 * methods of ::mr_op_create agree to within the NUFFT tolerance.
 *
 * It mirrors, step by step, the independent Python reference
 * `mr_aha.nufft.AhANufftPlan` (CuPy + the cufinufft Python package) and the
 * MATLAB reference `AhA_cufinufft.m`, which the tests compare it against:
 *
 *  1. `im = x * (S_c * zmod)` for every rank (zmod: stack only, see below);
 *  2. stack only: unnormalised forward FFT along z (cuFFT);
 *  3. type-2 NUFFT of all `L` ranks (`L*Nz` planes for a stack) in one
 *     batched cufinufft call;
 *  4. Gram multiplication, one batched GEMM (cuBLAS) over all readouts (and
 *     kz encodes);
 *  5. type-1 NUFFT back, one batched call;
 *  6. stack only: unnormalised inverse FFT along z;
 *  7. `out (+)= im * conj(S_c * zmod)`.
 *
 * @section mr_nufft_order Sample order
 *
 * The caller's trajectory `[Ntrajs, Nkx, ndim]` (column-major) numbers the
 * samples `m = t + Ntrajs*kx`, readout index fastest. The engine hands
 * cufinufft a reordered copy with `m' = kx + Nkx*t`, so that the samples of
 * one readout are contiguous. The non-uniform data of one rank (or rank and
 * kz plane) is then, for readout @f$t@f$, a contiguous run of `Nkx` values,
 * and the Gram multiplication of all samples of a readout is one small GEMM
 * on strided columns,
 * @f[
 *   K'_t \;=\; K_t\,\Phi_{2,t}, \qquad
 *   K_t \in \mathbb{C}^{N_{kx} \times L}, \;
 *   (K_t)_{k,j} = k_j[\,t N_{kx} + k\,],
 * @f]
 * with leading dimension `Ntrajs*Nkx` (`Nz*Ntrajs*Nkx` for a stack), batched
 * over the readouts by `cublasCgemmStridedBatched` (or, for a stack, over
 * readouts and kz encodes by `cublasCgemmBatched` with pre-computed pointer
 * arrays). No transpose of the data is needed. The order of the samples does
 * not change the transforms: cufinufft sorts the points into bins anyway, so
 * the result differs from the reference only by floating-point summation
 * order.
 *
 * @section mr_nufft_stack Stack of stars
 *
 * kz is Cartesian: the image is Fourier transformed along z (centred,
 * unnormalised), every kz plane goes through the same 2D NUFFT pair, and the
 * Gram block depends on the readout and the kz encode (`phi2` is supplied in
 * centred kz order). With @f$c = \lfloor N_z/2 \rfloor@f$, the centred DFT
 * is a plain FFT of the image modulated by
 * @f$\mathrm{zmod}(z) = e^{2\pi i c z/N_z}@f$, followed by a phase that
 * depends only on the kz index. Because the Gram multiplication is diagonal
 * in kz, the kz-dependent phases of the forward and adjoint transforms cancel
 * exactly, so only the image-domain modulation remains; it is folded into the
 * coil sensitivity (steps 1 and 7), exactly as in the Python reference.
 *
 * @section mr_nufft_dev Devices and lifetime
 *
 * One engine serves one device. Its inputs (`phi2`, the trajectory) may sit
 * on another device (the operator's primary): they are copied in and
 * reordered at creation, after which the engine owns everything it uses.
 * cufinufft is told the device explicitly (`gpu_device_id`), since it does
 * not follow the current device.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_kernels.cuh"

#include <cublas_v2.h>
#include <cufinufft.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/** @brief Threads per block of the elementwise kernels of this file. */
#define MR_NU_BLOCK 256

/** @brief Evaluate a cuBLAS call and bail out with ::MR_ERR_CUDA. */
#define MR_CUBLAS_CHECK(call)                                                  \
    do {                                                                       \
        cublasStatus_t mr_err_ = (call);                                       \
        if (mr_err_ != CUBLAS_STATUS_SUCCESS)                                  \
            MR_FAIL(MR_ERR_CUDA, "%s failed at %s:%d: cuBLAS status %d", #call,\
                    __FILE__, __LINE__, (int)mr_err_);                         \
    } while (0)

/* ------------------------------------------------------------------------ */
/* State                                                                    */
/* ------------------------------------------------------------------------ */

/** @brief Per-device state of the NUFFT engine (see the file comment). */
struct mr_nufft_eng {
    mr_config       cfg;      /**< Copy of the configuration (this device's
                                   coils, device and stream).               */
    cudaStream_t    s;        /**< Stream of all work.                      */
    int             stack;    /**< Non-zero for ::MR_TRAJ_STACK.            */
    int64_t         Ntrajs;   /**< Readouts.                                */
    int64_t         Nkx;      /**< Samples per readout.                     */
    int64_t         M;        /**< Samples per plane, `Ntrajs*Nkx`.         */
    int64_t         nkz;      /**< kz planes (`Nz` for a stack, else 1).    */
    int64_t         ntr;      /**< Transforms per NUFFT, `L*nkz`.           */
    int64_t         img;      /**< Voxels per rank, `Nx*Ny*Nz`.             */
    int64_t         plane;    /**< Voxels per z plane, `Nx*Ny`.             */
    cufinufftf_plan t2;       /**< Type-2 plan (sign -1).                   */
    cufinufftf_plan t1;       /**< Type-1 plan (sign +1).                   */
    cublasHandle_t  blas;     /**< cuBLAS handle on @ref s.                 */
    cufftHandle     zfft;     /**< Stack: FFT along z of one rank, or 0.    */
    float          *d_traj;   /**< Reordered trajectory `[M, ndim]` (sample
                                   `kx + Nkx*t`), kept by cufinufft.         */
    mr_complex     *d_gram;   /**< Gram blocks `[L, L, Ntrajs*nkz]`, one
                                   column-major block per (readout, kz).     */
    mr_complex     *d_zmod;   /**< Stack: `zmod[z]`, else NULL.             */
    mr_complex     *d_im;     /**< Image buffer `[Nx, Ny, Nz, L]`.          */
    mr_complex     *d_k;      /**< Type-2 output `[M, ntr]`.                */
    mr_complex     *d_k2;     /**< Gram-multiplied samples `[M, ntr]`.      */
    const mr_complex **d_ptrs;/**< Stack: GEMM pointer arrays A, B, C.      */
};

/* ------------------------------------------------------------------------ */
/* Kernels                                                                  */
/* ------------------------------------------------------------------------ */

/**
 * @brief Reorder the trajectory so that the samples of a readout are
 *        contiguous: `dst[d*M + kx + Nkx*t] = src[d*M + t + Ntrajs*kx]`.
 * @param[out] dst    Reordered coordinates `[M, ndim]`.
 * @param[in]  src    Caller's trajectory `[Ntrajs, Nkx, ndim]`.
 * @param[in]  Ntrajs Readouts.
 * @param[in]  Nkx    Samples per readout.
 * @param[in]  total  `Ntrajs*Nkx*ndim`.
 */
__global__ void mr_k_nu_perm_traj(float *dst, const float *src, long long Ntrajs,
                                  long long Nkx, long long total)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const long long M = Ntrajs * Nkx;
    const long long d = i / M, m = i - d * M; /* m = t + Ntrajs*kx (source) */
    const long long kx = m / Ntrajs, t = m - kx * Ntrajs;
    dst[d * M + kx + Nkx * t] = src[i];
}

/**
 * @brief Reorder the Gram blocks into one contiguous column-major
 *        @f$L\times L@f$ block per (readout, kz) batch entry:
 *        `dst[b*L*L + i + L*j] = src[b + nb*(i + L*j)]` with
 *        `b = t + Ntrajs*z` and `nb = Ntrajs*nkz`.
 * @param[out] dst   Reordered blocks `[L, L, nb]`.
 * @param[in]  src   Caller's blocks `[Ntrajs, (Nkz,) L, L]`.
 * @param[in]  nb    Number of blocks.
 * @param[in]  L     Rank.
 * @param[in]  total `nb*L*L`.
 */
__global__ void mr_k_nu_reorder_gram(mr_complex *dst, const mr_complex *src,
                                     long long nb, int L, long long total)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const long long b = i % nb, ij = i / nb; /* ij = i + L*j of the block */
    dst[b * (long long)L * L + ij] = src[i];
}

/**
 * @brief Step 1: `im[l][r] = x[l][r] * (S[r] * zmod[z(r)])` over all ranks.
 *
 * The product is formed in the same order as the reference
 * (`se = S * zmod; im = x * se`).
 *
 * @param[out] im    Image buffer `[img, L]`.
 * @param[in]  x     Spatial coefficients `[img, L]`.
 * @param[in]  se    Map of the coil `[img]`.
 * @param[in]  zmod  Stack modulation `[Nz]`, or NULL.
 * @param[in]  img   Voxels per rank.
 * @param[in]  plane Voxels per z plane (`z = r / plane`).
 * @param[in]  total `img*L`.
 */
__global__ void mr_k_nu_mul_se(mr_complex *im, const mr_complex *x,
                               const mr_complex *se, const mr_complex *zmod,
                               long long img, long long plane, long long total)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const long long r = i % img;
    mr_complex s = se[r];
    if (zmod) s = cuCmulf(s, zmod[r / plane]);
    im[i] = cuCmulf(x[i], s);
}

/**
 * @brief Step 7: `out[l][r] (+)= im[l][r] * conj(S[r] * zmod[z(r)])`.
 * @param[in,out] out        Output `[img, L]`.
 * @param[in]     im         Image buffer `[img, L]`.
 * @param[in]     se         Map of the coil.
 * @param[in]     zmod       Stack modulation, or NULL.
 * @param[in]     img        Voxels per rank.
 * @param[in]     plane      Voxels per z plane.
 * @param[in]     total      `img*L`.
 * @param[in]     accumulate Zero for the first coil (overwrite).
 */
__global__ void mr_k_nu_acc_conj_se(mr_complex *out, const mr_complex *im,
                                    const mr_complex *se, const mr_complex *zmod,
                                    long long img, long long plane, long long total,
                                    int accumulate)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const long long r = i % img;
    mr_complex s = se[r];
    if (zmod) s = cuCmulf(s, zmod[r / plane]);
    const mr_complex v = cuCmulf(im[i], cuConjf(s));
    out[i] = accumulate ? cuCaddf(out[i], v) : v;
}

/* ------------------------------------------------------------------------ */
/* Life cycle                                                               */
/* ------------------------------------------------------------------------ */

void mr_nufft_eng_destroy(mr_nufft_eng *e, int alive)
{
    if (e == NULL) return;
    if (alive) {
        /* cufinufft switches to its own device internally. */
        if (e->t2) cufinufftf_destroy(e->t2);
        if (e->t1) cufinufftf_destroy(e->t1);
        if (e->blas) cublasDestroy(e->blas);
        if (e->zfft) cufftDestroy(e->zfft);
        if (e->d_traj) cudaFree(e->d_traj);
        if (e->d_gram) cudaFree(e->d_gram);
        if (e->d_zmod) cudaFree(e->d_zmod);
        if (e->d_im) cudaFree(e->d_im);
        if (e->d_k) cudaFree(e->d_k);
        if (e->d_k2) cudaFree(e->d_k2);
        if (e->d_ptrs) cudaFree((void *)e->d_ptrs);
        cudaGetLastError();
    }
    /* Otherwise the device context is gone together with every handle and
     * buffer above; cufinufft's and cuBLAS's small host structures are
     * leaked rather than touched (their destructors would call into the dead
     * context). */
    free(e);
}

/**
 * @brief Copy @p bytes from (@p src, @p src_dev) to a new buffer on the
 *        current device @p dev, or alias @p src when it already lives there.
 * @param[out] tmp     Receives the new buffer (to be freed by the caller), or
 *                     NULL when aliased.
 * @param[out] view    Receives the pointer to read from: @p src itself or
 *                     @p *tmp.
 * @param[in]  src     Source buffer (device memory on @p src_dev).
 * @param[in]  src_dev CUDA ordinal of the device holding @p src.
 * @param[in]  dev     CUDA ordinal of the current device (the engine's).
 * @param[in]  bytes   Number of bytes to copy.
 * @param[in]  s       Stream on @p dev for the peer copy (asynchronous; the
 *                     caller synchronises before freeing the source).
 * @return ::MR_SUCCESS or a negative ::mr_status (::MR_ERR_ALLOC,
 *         ::MR_ERR_CUDA); @p *tmp is NULL after a failed allocation.
 */
static int mr_nu_fetch(void **tmp, const void **view, const void *src, int src_dev,
                       int dev, size_t bytes, cudaStream_t s)
{
    *tmp = NULL;
    *view = src;
    if (src_dev == dev) return MR_SUCCESS;
    if (cudaMalloc(tmp, bytes) != cudaSuccess) {
        cudaGetLastError();
        *tmp = NULL;
        MR_FAIL(MR_ERR_ALLOC, "failed to allocate %.1f MB for an input copy",
                bytes / 1e6);
    }
    MR_CUDA_CHECK(cudaMemcpyPeerAsync(*tmp, dev, src, src_dev, bytes, s));
    *view = *tmp;
    return MR_SUCCESS;
}

/**
 * @brief Body of ::mr_nufft_eng_create; @p e is released by the caller on
 *        failure. Runs with the engine's device current.
 *
 * Copies the trajectory and the Gram blocks to the engine's device (when they
 * live elsewhere), reorders them by readout (sample m' = kx + Nkx * t), and
 * makes the cufinufft type-2 / type-1 plans, the stack's z FFT and the cuBLAS
 * handle.
 *
 * @param[in,out] e          Engine with `cfg`, `Ntrajs`, `Nkx`, `nkz`, `M`,
 *                           `img` and the stream already set.
 * @param[in]     d_phi2     Gram blocks [Ntrajs, (Nkz,) L, L] on @p src_dev.
 * @param[in]     d_traj     Trajectory [Ntrajs, Nkx, n_dim] on @p src_dev.
 * @param[in]     src_dev    CUDA ordinal of the device holding the inputs.
 * @param[in]     upsampfac  cufinufft upsampling factor (0: 2.0).
 * @param[in]     gpu_method cufinufft spreading method (0: cufinufft's
 *                           choice).
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
static int mr_nufft_eng_build(mr_nufft_eng *e, const mr_complex *d_phi2,
                              const float *d_traj, int src_dev, double upsampfac,
                              int gpu_method)
{
    const mr_config *cfg = &e->cfg;
    const int dev = cfg->gpu_device_id;
    const int L = cfg->L;
    const int ndim = (cfg->traj_type == MR_TRAJ_3D) ? 3 : 2;
    const size_t c8 = sizeof(mr_complex);
    const int64_t M = e->M, nb = e->Ntrajs * e->nkz;
    cudaStream_t s = e->s;

    MR_REQUIRE(e->nkz * M <= 2147483647LL,
               "NUFFT operator: Nkz*Ntrajs*Nkx = %lld exceeds the 32-bit leading "
               "dimension of cuBLAS", (long long)(e->nkz * M));

    /* -- Inputs: copy from the primary if needed, then reorder ------------ */
    {
        const size_t traj_bytes = (size_t)M * ndim * sizeof(float);
        const size_t gram_bytes = (size_t)nb * L * L * c8;
        void *tmp_t = NULL, *tmp_g = NULL;
        const void *vt = NULL, *vg = NULL;
        int rc = mr_nu_fetch(&tmp_t, &vt, d_traj, src_dev, dev, traj_bytes, s);
        if (rc == MR_SUCCESS)
            rc = mr_nu_fetch(&tmp_g, &vg, d_phi2, src_dev, dev, gram_bytes, s);
        cudaError_t ce = cudaSuccess;
        if (rc == MR_SUCCESS) {
            ce = cudaMalloc((void **)&e->d_traj, traj_bytes);
            if (ce == cudaSuccess) ce = cudaMalloc((void **)&e->d_gram, gram_bytes);
            if (ce != cudaSuccess) {
                cudaGetLastError();
                mr_set_error("failed to allocate the reordered trajectory and Gram "
                             "blocks: %s", cudaGetErrorString(ce));
                rc = MR_ERR_ALLOC;
            }
        }
        if (rc == MR_SUCCESS) {
            const long long nt = (long long)M * ndim;
            mr_k_nu_perm_traj<<<(int)mr_ceil_div(nt, MR_NU_BLOCK), MR_NU_BLOCK, 0, s>>>(
                e->d_traj, (const float *)vt, e->Ntrajs, e->Nkx, nt);
            const long long ng = (long long)nb * L * L;
            mr_k_nu_reorder_gram<<<(int)mr_ceil_div(ng, MR_NU_BLOCK), MR_NU_BLOCK, 0, s>>>(
                e->d_gram, (const mr_complex *)vg, nb, L, ng);
            ce = cudaGetLastError();
            if (ce == cudaSuccess) ce = cudaStreamSynchronize(s);
            if (ce != cudaSuccess) {
                mr_set_error("reordering the NUFFT inputs failed: %s",
                             cudaGetErrorString(ce));
                rc = MR_ERR_CUDA;
            }
        }
        /* The temporaries are only read by the work synchronised above. */
        if (tmp_t) cudaFree(tmp_t);
        if (tmp_g) cudaFree(tmp_g);
        if (rc != MR_SUCCESS) return rc;
    }

    /* -- Work buffers of one coil ----------------------------------------- */
    {
        const size_t im_bytes = (size_t)e->img * L * c8;
        const size_t k_bytes = (size_t)M * e->ntr * c8;
        cudaError_t ce = cudaMalloc((void **)&e->d_im, im_bytes);
        if (ce == cudaSuccess) ce = cudaMalloc((void **)&e->d_k, k_bytes);
        if (ce == cudaSuccess) ce = cudaMalloc((void **)&e->d_k2, k_bytes);
        if (ce != cudaSuccess) {
            cudaGetLastError();
            MR_FAIL(MR_ERR_ALLOC,
                    "failed to allocate the NUFFT work buffers (%.2f GB): %s",
                    (im_bytes + 2.0 * k_bytes) / 1e9, cudaGetErrorString(ce));
        }
    }

    /* -- Stack: z modulation and the FFT along z --------------------------- */
    if (e->stack) {
        const int nz = cfg->n_dim[2];
        std::vector<mr_complex> zm((size_t)nz);
        for (int z = 0; z < nz; ++z) {
            /* Same as the reference: phase in double, rounded to complex64. */
            const double ph = 2.0 * M_PI * (double)(nz / 2) * (double)z / (double)nz;
            zm[(size_t)z] = make_cuFloatComplex((float)std::cos(ph), (float)std::sin(ph));
        }
        MR_CUDA_CHECK(cudaMalloc((void **)&e->d_zmod, (size_t)nz * c8));
        MR_CUDA_CHECK(cudaMemcpy(e->d_zmod, zm.data(), (size_t)nz * c8,
                                 cudaMemcpyHostToDevice));
        /* One rank: Nx*Ny transforms of length Nz, stride Nx*Ny, distance 1. */
        int n[1] = {nz};
        int emb[1] = {nz};
        const int plane = (int)e->plane;
        MR_CUFFT_CHECK(cufftPlanMany(&e->zfft, 1, n, emb, plane, 1, emb, plane, 1,
                                     CUFFT_C2C, plane));
        MR_CUFFT_CHECK(cufftSetStream(e->zfft, s));
    }

    /* -- Gram multiplication: cuBLAS, plus pointer arrays for a stack ------ */
    MR_CUBLAS_CHECK(cublasCreate(&e->blas));
    MR_CUBLAS_CHECK(cublasSetStream(e->blas, s));
    if (e->stack) {
        /* Batch entry b = t + Ntrajs*z: column l of K at plane l*nkz + z. */
        std::vector<const mr_complex *> h((size_t)3 * nb);
        for (int64_t z = 0; z < e->nkz; ++z)
            for (int64_t t = 0; t < e->Ntrajs; ++t) {
                const int64_t b = t + e->Ntrajs * z;
                const size_t off = (size_t)(z * M + t * e->Nkx);
                h[(size_t)b] = e->d_k + off;
                h[(size_t)(nb + b)] = e->d_gram + (size_t)b * L * L;
                h[(size_t)(2 * nb + b)] = e->d_k2 + off;
            }
        MR_CUDA_CHECK(cudaMalloc((void **)&e->d_ptrs, h.size() * sizeof(void *)));
        MR_CUDA_CHECK(cudaMemcpy((void *)e->d_ptrs, h.data(), h.size() * sizeof(void *),
                                 cudaMemcpyHostToDevice));
    }

    /* -- cufinufft plans: same settings as the reference operators --------- */
    cufinufft_opts opts;
    cufinufft_default_opts(&opts);
    opts.upsampfac = (upsampfac > 0.0) ? upsampfac : 2.0;
    opts.modeord = 0;          /* centred (CMCL) modes, like the library     */
    opts.gpu_device_id = dev;  /* cufinufft ignores the current device       */
    opts.gpu_stream = (void *)s;
    if (gpu_method > 0) opts.gpu_method = gpu_method;
    opts.debug = 0;
    const int nufft_dim = (cfg->traj_type == MR_TRAJ_3D) ? 3 : 2;
    int64_t modes[3] = {cfg->n_dim[0], cfg->n_dim[1],
                        (nufft_dim == 3) ? cfg->n_dim[2] : 1};
    const float eps = (float)cfg->eps;
    int err = cufinufftf_makeplan(2, nufft_dim, modes, -1, (int)e->ntr, eps, &e->t2,
                                  &opts);
    if (err != 0) {
        e->t2 = NULL;
        MR_FAIL(MR_ERR_CUFINUFFT, "cufinufftf_makeplan (type 2) failed with code %d",
                err);
    }
    err = cufinufftf_makeplan(1, nufft_dim, modes, +1, (int)e->ntr, eps, &e->t1, &opts);
    if (err != 0) {
        e->t1 = NULL;
        MR_FAIL(MR_ERR_CUFINUFFT, "cufinufftf_makeplan (type 1) failed with code %d",
                err);
    }
    const float *xj = e->d_traj;
    const float *yj = e->d_traj + M;
    const float *zj = (nufft_dim == 3) ? e->d_traj + 2 * M : NULL;
    err = cufinufftf_setpts(e->t2, M, xj, yj, zj, 0, NULL, NULL, NULL);
    if (err == 0) err = cufinufftf_setpts(e->t1, M, xj, yj, zj, 0, NULL, NULL, NULL);
    if (err != 0)
        MR_FAIL(MR_ERR_CUFINUFFT, "cufinufftf_setpts failed with code %d", err);
    MR_CUDA_CHECK(cudaStreamSynchronize(s));
    return MR_SUCCESS;
}

int mr_nufft_eng_create(const mr_config *cfg, const mr_complex *d_phi2,
                        const float *d_traj, int src_dev, int64_t Ntrajs, int64_t Nkx,
                        double upsampfac, int gpu_method, mr_nufft_eng **out)
{
    *out = NULL;
    MR_REQUIRE(!cfg->is_masked,
               "the NUFFT method has no kernel to mask; set is_masked = 0");
    mr_nufft_eng *e = (mr_nufft_eng *)calloc(1, sizeof(mr_nufft_eng));
    if (e == NULL) MR_FAIL(MR_ERR_ALLOC, "failed to allocate the NUFFT engine");
    e->cfg = *cfg;
    e->s = (cudaStream_t)cfg->stream;
    e->stack = (cfg->traj_type == MR_TRAJ_STACK);
    e->Ntrajs = Ntrajs;
    e->Nkx = Nkx;
    e->M = Ntrajs * Nkx;
    e->nkz = e->stack ? cfg->n_dim[2] : 1;
    e->ntr = (int64_t)cfg->L * e->nkz;
    e->img = mr_image_size(cfg);
    e->plane = (int64_t)cfg->n_dim[0] * cfg->n_dim[1];
    const int rc = mr_nufft_eng_build(e, d_phi2, d_traj, src_dev, upsampfac, gpu_method);
    if (rc != MR_SUCCESS) {
        mr_nufft_eng_destroy(e, 1);
        return rc;
    }
    if (cfg->verbose)
        printf("[mr_aha] NUFFT engine on device %d: %lld transforms of %lld samples, "
               "%d coil(s)\n",
               cfg->gpu_device_id, (long long)e->ntr, (long long)e->M, cfg->n_coils);
    *out = e;
    return MR_SUCCESS;
}

/* ------------------------------------------------------------------------ */
/* Execution                                                                */
/* ------------------------------------------------------------------------ */

/**
 * @brief Stack only: unnormalised FFT along z of all ranks of the image
 *        buffer, in place (one cuFFT call per rank).
 * @param[in,out] e   Engine; its `d_im` buffer ([Nx, Ny, Nz] per rank, L
 *                    ranks) is transformed in place on the engine's stream.
 * @param[in]     dir `CUFFT_FORWARD` or `CUFFT_INVERSE`.
 * @return ::MR_SUCCESS or ::MR_ERR_CUFFT.
 */
static int mr_nu_zfft(mr_nufft_eng *e, int dir)
{
    for (int l = 0; l < e->cfg.L; ++l) {
        mr_complex *p = e->d_im + (size_t)l * e->img;
        MR_CUFFT_CHECK(cufftExecC2C(e->zfft, p, p, dir));
    }
    return MR_SUCCESS;
}

int mr_nufft_eng_execute(mr_nufft_eng *e, const mr_complex *d_x,
                         const mr_complex *d_SEs, mr_complex *d_out)
{
    const mr_config *cfg = &e->cfg;
    cudaStream_t s = e->s;
    const int L = cfg->L;
    const long long total = e->img * (long long)L;
    const int grid = (int)mr_ceil_div(total, MR_NU_BLOCK);
    const cuComplex one = make_cuFloatComplex(1.0f, 0.0f);
    const cuComplex zero = make_cuFloatComplex(0.0f, 0.0f);

    for (int c = 0; c < cfg->n_coils; ++c) {
        const mr_complex *se = d_SEs + (size_t)c * e->img;

        /* 1. Coil sensitivity (and stack z modulation), all ranks. */
        mr_k_nu_mul_se<<<grid, MR_NU_BLOCK, 0, s>>>(e->d_im, d_x, se, e->d_zmod,
                                                     e->img, e->plane, total);
        MR_KERNEL_CHECK("mr_k_nu_mul_se");

        /* 2. Stack: unnormalised FFT along z. */
        if (e->stack) {
            const int rc = mr_nu_zfft(e, CUFFT_FORWARD);
            if (rc != MR_SUCCESS) return rc;
        }

        /* 3. Type-2 NUFFT, all ranks (and kz planes) at once. */
        int err = cufinufftf_execute(e->t2, e->d_k, e->d_im);
        if (err != 0)
            MR_FAIL(MR_ERR_CUFINUFFT, "cufinufftf_execute (type 2) failed with code %d",
                    err);

        /* 4. Gram multiplication K'_b = K_b * Phi2_b over all batch entries. */
        if (e->stack) {
            const int64_t nb = e->Ntrajs * e->nkz;
            MR_CUBLAS_CHECK(cublasCgemmBatched(
                e->blas, CUBLAS_OP_N, CUBLAS_OP_N, (int)e->Nkx, L, L, &one,
                (const cuComplex *const *)e->d_ptrs, (int)(e->nkz * e->M),
                (const cuComplex *const *)(e->d_ptrs + nb), L, &zero,
                (cuComplex *const *)(e->d_ptrs + 2 * nb), (int)(e->nkz * e->M),
                (int)nb));
        } else {
            MR_CUBLAS_CHECK(cublasCgemmStridedBatched(
                e->blas, CUBLAS_OP_N, CUBLAS_OP_N, (int)e->Nkx, L, L, &one,
                e->d_k, (int)e->M, (long long)e->Nkx, e->d_gram, L,
                (long long)L * L, &zero, e->d_k2, (int)e->M, (long long)e->Nkx,
                (int)e->Ntrajs));
        }

        /* 5. Type-1 NUFFT back onto the image grid. */
        err = cufinufftf_execute(e->t1, e->d_k2, e->d_im);
        if (err != 0)
            MR_FAIL(MR_ERR_CUFINUFFT, "cufinufftf_execute (type 1) failed with code %d",
                    err);

        /* 6. Stack: unnormalised inverse FFT along z. */
        if (e->stack) {
            const int rc = mr_nu_zfft(e, CUFFT_INVERSE);
            if (rc != MR_SUCCESS) return rc;
        }

        /* 7. Adjoint coil sensitivity, accumulated (the first coil writes). */
        mr_k_nu_acc_conj_se<<<grid, MR_NU_BLOCK, 0, s>>>(d_out, e->d_im, se, e->d_zmod,
                                                          e->img, e->plane, total,
                                                          c > 0);
        MR_KERNEL_CHECK("mr_k_nu_acc_conj_se");
    }
    if (cfg->n_coils == 0)
        MR_CUDA_CHECK(cudaMemsetAsync(d_out, 0, (size_t)total * sizeof(mr_complex), s));
    return MR_SUCCESS;
}
