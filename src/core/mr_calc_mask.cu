/**
 * @file mr_calc_mask.cu
 * @brief Automatic derivation of the masked-operator sample set.
 *
 * A masked @f$A^H A@f$ stores `phi2_toep` only on a subset of the upsampled
 * Cartesian k-space grid. This file derives that subset from the trajectory
 * alone, without any knowledge of the temporal basis. Two constructions are
 * implemented, selected by ::mr_config::mask_method.
 *
 * **Neighbourhood mask (::MR_MASK_NEIGHBOUR, default).**
 *
 *   1. Every sample is mapped to its nearest point on the upsampled grid and
 *      that point is marked, see ::mr_k_mark_samples.
 *   2. The marks are dilated by a @f$(2r+1)^d@f$ box, @f$r@f$ =
 *      ::mr_config::mask_radius, see ::mr_k_dilate_axis.
 *   3. Every marked location is kept.
 *
 * **Kernel-threshold mask (::MR_MASK_THRESHOLD).**
 *
 *   1. Every Gram block is replaced by the scalar one (@f$L = 1@f$), which
 *      turns ::mr_calc_toep into the Toeplitz multiplier
 *      @f$T(\mathbf{k})@f$ of the plain normal operator @f$F^H F@f$ of the
 *      trajectory ("Toeplitz weights with all weights set to one").
 *   2. @f$|T|@f$ is max-filtered over the immediate (3^d) neighbourhood of
 *      every location (the same box dilation with @f$r = 1@f$).
 *   3. The dilated kernel is compared with the absolute threshold
 *      ::mr_config::mask_threshold; locations above it are kept.
 *
 * The threshold is absolute because the unit-weight kernel has a natural
 * unit: a single isolated sample lying on the upsampled grid contributes
 * exactly 1 at its own location (a sample between grid points spreads over
 * its neighbours through the Dirichlet kernel). A threshold of 0.05 therefore
 * means "less than 5 % of one sample's worth". A threshold relative to the
 * maximum would not transfer between trajectories: for radial sampling the
 * maximum sits at the centre of k-space, where every spoke contributes, so it
 * grows with the number of spokes while the outer k-space level does not
 * (378 vs. 280 vs. 15680 for the 2D, stack and 3D test trajectories).
 *
 * Why both are sound: for arbitrary Gram blocks the Toeplitz multiplier is
 * @f$\hat{T}(\mathbf{k}) = \sum_m \Phi_m^H\Phi_m\,
 * D(\mathbf{k}-\mathbf{k}_m)@f$, where @f$D@f$ is the Dirichlet kernel of the
 * upsampled grid. Its main lobe around each sample spans a few cells, which
 * the neighbourhood mask keeps; its tails decay slowly, which the threshold
 * mask follows. The unit-weight kernel is the same sum with every block set
 * to one, so wherever @f$|T|@f$ is negligible no sample lies nearby and
 * @f$\hat{T}@f$ is negligible as well, whatever the subspace. Measurements of
 * both are in `benchmarks/neighbour_mask.md`.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_internal.cuh"

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system_error.h>

#include <cstdio>

/** @brief Default thread-block size for the elementwise kernels here. */
#define MR_BLOCK 256

/* ------------------------------------------------------------------------ */
/* Kernels                                                                  */
/* ------------------------------------------------------------------------ */

/**
 * @brief Fill a complex array with the value one.
 * @param[out] d_out Array to fill.
 * @param[in]  n     Number of elements.
 */
__global__ void mr_k_fill_ones(mr_complex *d_out, long long n)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_out[tid] = make_cuFloatComplex(1.0f, 0.0f);
}

/**
 * @brief Elementwise magnitude of a complex array.
 * @param[out] d_abs Real output, @f$|z|@f$.
 * @param[in]  d_z   Complex input.
 * @param[in]  n     Number of elements.
 */
__global__ void mr_k_abs(float *d_abs, const mr_complex *d_z, long long n)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) d_abs[tid] = cuCabsf(d_z[tid]);
}

/**
 * @brief Mark the upsampled grid point nearest to every sample.
 *
 * Sample @f$\mathbf{k}@f$ (radians) lands on index
 * `rint(k_a * M_a / (2*pi)) + M_a/2` along transformed axis `a`, wrapped
 * into `[0, M_a)` because the grid is periodic (a sample at exactly
 * @f$+\pi@f$ therefore marks index 0, the same location as @f$-\pi@f$). The
 * marked location is set to 1; concurrent writes of the same value are
 * benign. The buffer must be zeroed beforehand.
 *
 * @param[out] d_marks   Marks `[m0, m1, m2]`, centred order, pre-zeroed.
 * @param[in]  d_traj    Trajectory `[n_samples, ndim]` (i.e. `[Ntrajs, Nkx,
 *                       ndim]` with the first two axes flattened).
 * @param[in]  n_samples `Ntrajs * Nkx`.
 * @param[in]  ndim      Number of trajectory coordinates (2 or 3).
 * @param[in]  m0,m1,m2  Grid extents (unused axes are 1).
 * @param[in]  s0,s1,s2  Per-axis scale `M_a / (2*pi)`.
 */
__global__ void mr_k_mark_samples(float *d_marks, const float *d_traj,
                                  long long n_samples, int ndim, int m0, int m1,
                                  int m2, float s0, float s1, float s2)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_samples) return;
    const int   m[3]  = {m0, m1, m2};
    const float sc[3] = {s0, s1, s2};
    long long idx = 0, stride = 1;
    for (int a = 0; a < ndim; ++a) {
        long long i = (long long)rintf(d_traj[tid + n_samples * a] * sc[a]) +
                      m[a] / 2;
        i %= m[a];
        if (i < 0) i += m[a];
        idx += i * stride;
        stride *= m[a];
    }
    d_marks[idx] = 1.0f;
}

/**
 * @brief One axis of a separable, periodic max-filter (grey-scale dilation).
 *
 * Each output location receives the largest input value among the locations
 * up to @p r cells away along one axis, wrapping periodically like the DFT
 * grid. Applying it along every axis gives the max over a @f$(2r+1)^d@f$ box,
 * i.e. over all locations within Chebyshev distance @p r.
 *
 * It serves both mask methods: with 0/1 marks it grows each sample's point
 * into a box of radius ::mr_config::mask_radius; with the unit-weight kernel
 * (@f$r = 1@f$) it fills the isolated pockets where the Dirichlet kernels of
 * unit weights cancel inside the sampled region but those of real Gram blocks
 * do not, which lowers the thresholding error by about an order of magnitude
 * at the same number of retained locations.
 *
 * @param[out] d_out  Filtered array, same shape as @p d_in.
 * @param[in]  d_in   Input array of @p n elements.
 * @param[in]  n      Total number of elements.
 * @param[in]  m      Extent of the filtered axis.
 * @param[in]  stride Element stride of the filtered axis.
 * @param[in]  r      Radius in cells, >= 1.
 */
__global__ void mr_k_dilate_axis(float *d_out, const float *d_in, long long n,
                                 int m, long long stride, int r)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    const int       i    = (int)((tid / stride) % m);
    const long long base = tid - (long long)i * stride;
    float v = d_in[tid];
    for (int d = -r; d <= r; ++d) {
        int j = (i + d) % m;
        if (j < 0) j += m;
        v = fmaxf(v, d_in[base + (long long)j * stride]);
    }
    d_out[tid] = v;
}

/**
 * @brief Replicate an in-plane mask over the kz planes of a stack grid.
 *
 * The first @p n_plane entries of @p d_mask hold the in-plane indices (into
 * the `[Mx, My]` plane, which coincide with the indices of plane `kz = 0` of
 * the `[Mx, My, Mz]` grid). Plane `kz` receives the same indices offset by
 * `kz * Mx * My`, so the full mask stays sorted.
 *
 * @param[in,out] d_mask     Mask buffer of `n_plane * Mz` entries.
 * @param[in]     n_plane    Number of retained in-plane locations.
 * @param[in]     plane_size `Mx * My`.
 * @param[in]     Mz         Number of kz planes.
 */
__global__ void mr_k_replicate_planes(int64_t *d_mask, long long n_plane,
                                      long long plane_size, int Mz)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n_plane * (long long)(Mz - 1);
    if (tid >= total) return;
    long long i  = tid % n_plane;
    long long kz = tid / n_plane + 1; /* plane 0 is already in place */
    d_mask[i + n_plane * kz] = d_mask[i] + plane_size * kz;
}

/** @brief Thrust predicate: is the value at linear index @p i above threshold? */
struct mr_above_threshold {
    const float *kernel; /**< Kernel magnitudes or 0/1 marks, centred order. */
    float        thr;    /**< Absolute threshold.                            */

    /** @brief Evaluate the predicate at linear index @p i. */
    __host__ __device__ bool operator()(int64_t i) const { return kernel[i] > thr; }
};

/* ------------------------------------------------------------------------ */
/* Implementation                                                           */
/* ------------------------------------------------------------------------ */

/**
 * @brief Build the configuration of the unit-weight problem.
 *
 * A stack trajectory is reduced to its 2D in-plane problem, since every kz
 * encode shares the same in-plane kernel; the rank becomes one and masking is
 * switched off so the full grid is produced.
 *
 * @param[in]  cfg Caller's configuration.
 * @param[out] k   Configuration for the unit-weight kernel.
 */
static void mr_unit_config(const mr_config *cfg, mr_config *k)
{
    *k = *cfg;
    k->L          = 1;
    k->is_masked  = 0;
    k->num_masked = 0;
    k->n_coils    = 1;
    if (cfg->traj_type == MR_TRAJ_STACK) {
        k->traj_type   = MR_TRAJ_2D;
        k->n_spatial   = 2;
        k->n_dim[2]    = 1;
        k->phi2_dim[2] = 1;
    }
}

/**
 * @brief Compute @f$|T|@f$ with the target device already current.
 * @param[in]  cfg      Validated caller configuration.
 * @param[in]  d_traj   Trajectory `[Ntrajs, Nkx, ndim]`.
 * @param[in]  Ntrajs   Number of trajectories.
 * @param[in]  Nkx      Samples per trajectory.
 * @param[out] d_kernel Output of ::mr_mask_kernel_size floats.
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
static int mr_calc_mask_kernel_impl(const mr_config *cfg, const float *d_traj,
                                    int64_t Ntrajs, int64_t Nkx, float *d_kernel)
{
    mr_config kcfg;
    mr_unit_config(cfg, &kcfg);
    const int status = mr_config_validate(&kcfg);
    if (status != MR_SUCCESS) return status;

    cudaStream_t  s = (cudaStream_t)cfg->stream;
    const int64_t n = mr_pad_size(&kcfg);

    mr_complex *d_ones = NULL, *d_T = NULL;
    if (cudaMalloc((void **)&d_ones, (size_t)Ntrajs * sizeof(mr_complex)) !=
            cudaSuccess ||
        cudaMalloc((void **)&d_T, (size_t)n * sizeof(mr_complex)) != cudaSuccess) {
        cudaFree(d_ones);
        cudaFree(d_T);
        MR_FAIL(MR_ERR_ALLOC, "failed to allocate the unit-weight kernel buffers");
    }

    mr_k_fill_ones<<<(int)mr_ceil_div(Ntrajs, MR_BLOCK), MR_BLOCK, 0, s>>>(
        d_ones, (long long)Ntrajs);
    int rc = MR_SUCCESS;
    if (cudaGetLastError() != cudaSuccess) {
        mr_set_error("mr_k_fill_ones failed to launch");
        rc = MR_ERR_CUDA;
    }

    /* The unit-weight Toeplitz multiplier, scaled exactly like phi2_toep. */
    if (rc == MR_SUCCESS)
        rc = mr_calc_toep_impl(&kcfg, d_ones, d_traj, Ntrajs, Nkx, NULL, d_T);
    cudaFree(d_ones); /* mr_calc_toep_impl synchronises before returning */

    if (rc == MR_SUCCESS) {
        mr_k_abs<<<(int)mr_ceil_div(n, MR_BLOCK), MR_BLOCK, 0, s>>>(d_kernel, d_T,
                                                                    (long long)n);
        const cudaError_t ce = cudaGetLastError();
        if (ce != cudaSuccess) {
            mr_set_error("mr_k_abs failed to launch: %s", cudaGetErrorString(ce));
            rc = MR_ERR_CUDA;
        }
    }
    if (rc == MR_SUCCESS) {
        /* d_T is freed next, so wait for the magnitude pass to read it. */
        const cudaError_t ce = (s != NULL) ? cudaStreamSynchronize(s)
                                           : cudaDeviceSynchronize();
        if (ce != cudaSuccess) {
            mr_set_error("synchronisation after mr_k_abs failed: %s",
                         cudaGetErrorString(ce));
            rc = MR_ERR_CUDA;
        }
    }
    cudaFree(d_T);
    return rc;
}

/**
 * @brief Dilate a grid in place by a periodic @f$(2r+1)^d@f$ box.
 *
 * Applies ::mr_k_dilate_axis along every axis of extent > 1, ping-ponging
 * between the two buffers; on return @p *d_a holds the result and @p *d_b
 * the scratch (the pointers may have been swapped).
 *
 * @param[in,out] d_a Input on entry, result on return.
 * @param[in,out] d_b Scratch buffer of the same size.
 * @param[in]     m   Grid extents `[m0, m1, m2]` (unused axes are 1).
 * @param[in]     r   Radius in cells; 0 leaves the grid unchanged.
 * @param[in]     s   Stream to launch on.
 * @return ::MR_SUCCESS or ::MR_ERR_CUDA.
 */
static int mr_dilate_box(float **d_a, float **d_b, const int m[3], int r,
                         cudaStream_t s)
{
    if (r <= 0) return MR_SUCCESS;
    const long long n = (long long)m[0] * m[1] * m[2];
    long long stride = 1;
    for (int ax = 0; ax < 3; ++ax) {
        if (m[ax] > 1) {
            mr_k_dilate_axis<<<(int)mr_ceil_div(n, MR_BLOCK), MR_BLOCK, 0, s>>>(
                *d_b, *d_a, n, m[ax], stride, r);
            MR_KERNEL_CHECK("mr_k_dilate_axis");
            float *t = *d_a; *d_a = *d_b; *d_b = t;
        }
        stride *= m[ax];
    }
    return MR_SUCCESS;
}

/**
 * @brief Build the selection grid of the chosen method, device already current.
 *
 * On success @p *d_sel holds ::mr_mask_kernel_size floats (owned by the
 * caller) and @p *thr the value a location must exceed to be kept: the
 * dilated unit-weight kernel and ::mr_config::mask_threshold for
 * ::MR_MASK_THRESHOLD, the dilated 0/1 sample marks and 0.5 for
 * ::MR_MASK_NEIGHBOUR.
 *
 * @param[in]  cfg    Validated caller configuration.
 * @param[in]  d_traj Trajectory `[Ntrajs, Nkx, ndim]`.
 * @param[in]  Ntrajs Number of trajectories.
 * @param[in]  Nkx    Samples per trajectory.
 * @param[out] d_sel  Receives the selection grid.
 * @param[out] thr    Receives the selection threshold.
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
static int mr_mask_selection(const mr_config *cfg, const float *d_traj,
                             int64_t Ntrajs, int64_t Nkx, float **d_sel,
                             float *thr)
{
    const int64_t n     = mr_mask_kernel_size(cfg);
    const int     three = (cfg->traj_type == MR_TRAJ_3D);
    /* Transformed axes only: a stack's kz is handled by replication. */
    const int m[3] = {cfg->phi2_dim[0], cfg->phi2_dim[1],
                      three ? cfg->phi2_dim[2] : 1};
    cudaStream_t s = (cudaStream_t)cfg->stream;

    float *d_a = NULL, *d_b = NULL;
    if (cudaMalloc((void **)&d_a, (size_t)n * sizeof(float)) != cudaSuccess)
        MR_FAIL(MR_ERR_ALLOC, "failed to allocate the %lld-element mask grid",
                (long long)n);

    int rc = MR_SUCCESS;
    int radius;
    if (cfg->mask_method == MR_MASK_THRESHOLD) {
        rc     = mr_calc_mask_kernel_impl(cfg, d_traj, Ntrajs, Nkx, d_a);
        radius = 1;
        /* Absolute threshold: the unit-weight kernel is measured in samples
         * (one isolated on-grid sample contributes exactly 1). */
        *thr   = (float)cfg->mask_threshold;
    } else {
        const long long ns  = (long long)Ntrajs * Nkx;
        const double    tau = 6.283185307179586; /* 2*pi */
        if (cudaMemsetAsync(d_a, 0, (size_t)n * sizeof(float), s) != cudaSuccess) {
            mr_set_error("clearing the mask grid failed");
            rc = MR_ERR_CUDA;
        } else {
            mr_k_mark_samples<<<(int)mr_ceil_div(ns, MR_BLOCK), MR_BLOCK, 0, s>>>(
                d_a, d_traj, ns, three ? 3 : 2, m[0], m[1], m[2],
                (float)(m[0] / tau), (float)(m[1] / tau), (float)(m[2] / tau));
            const cudaError_t ce = cudaGetLastError();
            if (ce != cudaSuccess) {
                mr_set_error("mr_k_mark_samples failed to launch: %s",
                             cudaGetErrorString(ce));
                rc = MR_ERR_CUDA;
            }
        }
        radius = cfg->mask_radius;
        *thr   = 0.5f; /* marks are exactly 0 or 1 */
    }
    if (rc != MR_SUCCESS) { cudaFree(d_a); return rc; }

    /* The scratch grid is allocated only now, after the threshold method's
     * NUFFT buffers are gone, to keep the peak memory down. */
    if (radius > 0) {
        if (cudaMalloc((void **)&d_b, (size_t)n * sizeof(float)) != cudaSuccess) {
            cudaFree(d_a);
            MR_FAIL(MR_ERR_ALLOC, "failed to allocate the dilation scratch grid");
        }
        rc = mr_dilate_box(&d_a, &d_b, m, radius, s);
        if (rc == MR_SUCCESS) {
            /* d_b is freed next, so wait for the passes that read it. */
            const cudaError_t se = (s != NULL) ? cudaStreamSynchronize(s)
                                               : cudaDeviceSynchronize();
            if (se != cudaSuccess) {
                mr_set_error("dilating the mask grid failed: %s",
                             cudaGetErrorString(se));
                rc = MR_ERR_CUDA;
            }
        }
        cudaFree(d_b);
        if (rc != MR_SUCCESS) { cudaFree(d_a); return rc; }
    }
    *d_sel = d_a;
    return MR_SUCCESS;
}

/**
 * @brief Build the selection grid and emit the sorted mask, device already current.
 * @param[in]  cfg            Validated caller configuration.
 * @param[in]  d_traj         Trajectory `[Ntrajs, Nkx, ndim]`.
 * @param[in]  Ntrajs         Number of trajectories.
 * @param[in]  Nkx            Samples per trajectory.
 * @param[out] d_mask         Output indices, capacity ::mr_pad_size, or NULL.
 * @param[out] num_masked_out Number of retained locations.
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
static int mr_calc_mask_impl(const mr_config *cfg, const float *d_traj,
                             int64_t Ntrajs, int64_t Nkx, int64_t *d_mask,
                             int64_t *num_masked_out)
{
    const int64_t n = mr_mask_kernel_size(cfg);
    float *d_kernel = NULL;
    float  thr      = 0.0f;
    int rc = mr_mask_selection(cfg, d_traj, Ntrajs, Nkx, &d_kernel, &thr);
    if (rc != MR_SUCCESS) return rc;

    cudaStream_t s = (cudaStream_t)cfg->stream;

    int64_t n_kept = 0;
    try {
        /* thrust reports failures by exception, which must not cross the C
         * boundary, hence the try block. */
        auto policy = thrust::cuda::par.on(s);
        const mr_above_threshold pred = {d_kernel, thr};
        thrust::counting_iterator<int64_t> first(0), last(n);
        if (d_mask != NULL) {
            /* copy_if preserves order, so the indices come out ascending. */
            thrust::device_ptr<int64_t> out(d_mask);
            n_kept = (int64_t)(thrust::copy_if(policy, first, last, out, pred) - out);
        } else {
            n_kept = (int64_t)thrust::count_if(policy, first, last, pred);
        }
    } catch (const thrust::system_error &e) {
        cudaFree(d_kernel);
        MR_FAIL(MR_ERR_CUDA, "compacting the mask failed: %s", e.what());
    } catch (...) {
        cudaFree(d_kernel);
        MR_FAIL(MR_ERR_INTERNAL, "compacting the mask failed");
    }
    cudaFree(d_kernel);

    /* Stack: every kz plane shares the in-plane mask. */
    if (cfg->traj_type == MR_TRAJ_STACK) {
        const int Mz = cfg->phi2_dim[2];
        if (d_mask != NULL && Mz > 1 && n_kept > 0) {
            const long long total = (long long)n_kept * (Mz - 1);
            mr_k_replicate_planes<<<(int)mr_ceil_div(total, MR_BLOCK), MR_BLOCK, 0,
                                    s>>>(d_mask, (long long)n_kept,
                                         (long long)cfg->phi2_dim[0] *
                                             cfg->phi2_dim[1],
                                         Mz);
            MR_KERNEL_CHECK("mr_k_replicate_planes");
        }
        n_kept *= Mz;
    }

    if (d_mask != NULL) {
        const cudaError_t ce = (s != NULL) ? cudaStreamSynchronize(s)
                                           : cudaDeviceSynchronize();
        if (ce != cudaSuccess)
            MR_FAIL(MR_ERR_CUDA, "synchronisation after mr_calc_mask failed: %s",
                    cudaGetErrorString(ce));
    }

    if (cfg->verbose) {
        if (cfg->mask_method == MR_MASK_THRESHOLD)
            printf("[mr_aha] calc_mask: kept %lld of %lld upsampled locations "
                   "(%.1f%%) at threshold %g\n",
                   (long long)n_kept, (long long)mr_pad_size(cfg),
                   100.0 * (double)n_kept / (double)mr_pad_size(cfg),
                   cfg->mask_threshold);
        else
            printf("[mr_aha] calc_mask: kept %lld of %lld upsampled locations "
                   "(%.1f%%) within radius %d of a sample\n",
                   (long long)n_kept, (long long)mr_pad_size(cfg),
                   100.0 * (double)n_kept / (double)mr_pad_size(cfg),
                   cfg->mask_radius);
    }

    *num_masked_out = n_kept;
    return MR_SUCCESS;
}

/**
 * @brief Argument checks shared by the two public entry points.
 *
 * Validates a copy of @p cfg with masking switched off, because the number of
 * masked locations is exactly what is being computed.
 *
 * @param[in] fn     Name of the calling function, for diagnostics.
 * @param[in] cfg    Caller's configuration.
 * @param[in] d_traj Trajectory pointer.
 * @param[in] Ntrajs Number of trajectories.
 * @param[in] Nkx    Samples per trajectory.
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
static int mr_mask_check_args(const char *fn, const mr_config *cfg,
                              const float *d_traj, int64_t Ntrajs, int64_t Nkx)
{
    MR_REQUIRE(cfg != NULL, "%s: cfg is NULL", fn);
    MR_REQUIRE(d_traj != NULL, "%s: d_traj is NULL", fn);
    MR_REQUIRE(Ntrajs > 0, "%s: Ntrajs = %lld must be > 0", fn, (long long)Ntrajs);
    MR_REQUIRE(Nkx > 0, "%s: Nkx = %lld must be > 0", fn, (long long)Nkx);
    mr_config c = *cfg;
    c.is_masked  = 0;
    c.num_masked = 0;
    return mr_config_validate(&c);
}

extern "C" int mr_calc_mask_kernel(const mr_config *cfg, const float *d_traj,
                                   int64_t Ntrajs, int64_t Nkx, float *d_kernel)
{
    mr_clear_error();
    MR_REQUIRE(d_kernel != NULL, "mr_calc_mask_kernel: d_kernel is NULL");
    int rc = mr_mask_check_args("mr_calc_mask_kernel", cfg, d_traj, Ntrajs, Nkx);
    if (rc != MR_SUCCESS) return rc;

    int prev = -1;
    rc = mr_toep_push_device(cfg, &prev);
    if (rc == MR_SUCCESS)
        rc = mr_calc_mask_kernel_impl(cfg, d_traj, Ntrajs, Nkx, d_kernel);
    if (prev >= 0) cudaSetDevice(prev);
    return rc;
}

extern "C" int mr_calc_mask(const mr_config *cfg, const float *d_traj,
                            int64_t Ntrajs, int64_t Nkx, int64_t *d_mask,
                            int64_t *num_masked_out)
{
    mr_clear_error();
    MR_REQUIRE(num_masked_out != NULL, "mr_calc_mask: num_masked_out is NULL");
    *num_masked_out = 0;
    int rc = mr_mask_check_args("mr_calc_mask", cfg, d_traj, Ntrajs, Nkx);
    if (rc != MR_SUCCESS) return rc;

    int prev = -1;
    rc = mr_toep_push_device(cfg, &prev);
    if (rc == MR_SUCCESS)
        rc = mr_calc_mask_impl(cfg, d_traj, Ntrajs, Nkx, d_mask, num_masked_out);
    if (prev >= 0) cudaSetDevice(prev);
    return rc;
}
