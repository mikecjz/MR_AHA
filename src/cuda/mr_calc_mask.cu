/**
 * @file mr_calc_mask.cu
 * @brief Automatic derivation of the masked-operator sample set.
 *
 * A masked @f$A^H A@f$ stores `phi2_toep` only on a subset of the upsampled
 * Cartesian k-space grid. This file derives that subset from the trajectory
 * alone, without any knowledge of the temporal basis:
 *
 *   1. Every sample is mapped to its nearest point on the upsampled grid and
 *      that point is marked, see ::mr_k_mark_samples.
 *   2. The marks are dilated by a @f$(2r+1)^d@f$ box, @f$r@f$ =
 *      ::mr_config::mask_radius, see ::mr_k_dilate_axis.
 *   3. Every marked location is kept.
 *
 * Why this is sound: for arbitrary Gram blocks the Toeplitz multiplier is
 * @f$\hat{T}(\mathbf{k}) = \sum_m (\Phi\Phi^H)_m\,
 * D(\mathbf{k}-\mathbf{k}_m)@f$, where @f$D@f$ is the Dirichlet kernel of the
 * upsampled grid. Its main lobe around each sample spans a few cells, which
 * the neighbourhood of radius @f$r@f$ keeps; far from every sample only the
 * slowly decaying tails remain, and @f$\hat{T}@f$ is small there whatever the
 * subspace. Measurements are in the documentation
 * (https://mikecjz.github.io/MR_AHA_docs/benchmarks/neighbour-mask.html).
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
 * Applied to the 0/1 sample marks it grows each sample's point into a box of
 * radius ::mr_config::mask_radius.
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

/** @brief Thrust predicate: is the location at linear index @p i marked? */
struct mr_is_marked {
    const float *marks; /**< Dilated 0/1 sample marks, centred order. */

    /** @brief Evaluate the predicate at linear index @p i. */
    __host__ __device__ bool operator()(int64_t i) const { return marks[i] > 0.5f; }
};

/* ------------------------------------------------------------------------ */
/* Implementation                                                           */
/* ------------------------------------------------------------------------ */

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
 * @brief Number of elements of the grid the samples are marked on.
 *
 * Only the transformed axes count: `Mx*My*Mz` for ::MR_TRAJ_3D and `Mx*My`
 * for ::MR_TRAJ_2D and ::MR_TRAJ_STACK (a stack's kz planes share the
 * in-plane mask, which is replicated afterwards).
 *
 * @param[in] cfg Validated configuration.
 * @return Element count.
 */
static int64_t mr_mask_grid_size(const mr_config *cfg)
{
    const int64_t plane = (int64_t)cfg->phi2_dim[0] * cfg->phi2_dim[1];
    return (cfg->traj_type == MR_TRAJ_3D) ? plane * cfg->phi2_dim[2] : plane;
}

/**
 * @brief Build the grid of dilated sample marks, device already current.
 *
 * On success @p *d_marks holds ::mr_mask_grid_size floats (owned by the
 * caller): 1 at every location within ::mr_config::mask_radius cells of the
 * grid point nearest to a sample, 0 elsewhere.
 *
 * @param[in]  cfg     Validated caller configuration.
 * @param[in]  d_traj  Trajectory `[Ntrajs, Nkx, ndim]`.
 * @param[in]  Ntrajs  Number of trajectories.
 * @param[in]  Nkx     Samples per trajectory.
 * @param[out] d_marks Receives the grid of marks.
 * @return ::MR_SUCCESS or a negative ::mr_status.
 */
static int mr_mask_marks(const mr_config *cfg, const float *d_traj,
                         int64_t Ntrajs, int64_t Nkx, float **d_marks)
{
    const int64_t n     = mr_mask_grid_size(cfg);
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
    if (rc != MR_SUCCESS) { cudaFree(d_a); return rc; }

    const int radius = cfg->mask_radius;
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
    *d_marks = d_a;
    return MR_SUCCESS;
}

/**
 * @brief Build the marks and emit the sorted mask, device already current.
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
    const int64_t n = mr_mask_grid_size(cfg);
    float *d_marks = NULL;
    int rc = mr_mask_marks(cfg, d_traj, Ntrajs, Nkx, &d_marks);
    if (rc != MR_SUCCESS) return rc;

    cudaStream_t s = (cudaStream_t)cfg->stream;

    int64_t n_kept = 0;
    try {
        /* thrust reports failures by exception, which must not cross the C
         * boundary, hence the try block. */
        auto policy = thrust::cuda::par.on(s);
        const mr_is_marked pred = {d_marks};
        thrust::counting_iterator<int64_t> first(0), last(n);
        if (d_mask != NULL) {
            /* copy_if preserves order, so the indices come out ascending. */
            thrust::device_ptr<int64_t> out(d_mask);
            n_kept = (int64_t)(thrust::copy_if(policy, first, last, out, pred) - out);
        } else {
            n_kept = (int64_t)thrust::count_if(policy, first, last, pred);
        }
    } catch (const thrust::system_error &e) {
        cudaFree(d_marks);
        MR_FAIL(MR_ERR_CUDA, "compacting the mask failed: %s", e.what());
    } catch (...) {
        cudaFree(d_marks);
        MR_FAIL(MR_ERR_INTERNAL, "compacting the mask failed");
    }
    cudaFree(d_marks);

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

    if (cfg->verbose)
        printf("[mr_aha] calc_mask: kept %lld of %lld upsampled locations "
               "(%.1f%%) within radius %d of a sample\n",
               (long long)n_kept, (long long)mr_pad_size(cfg),
               100.0 * (double)n_kept / (double)mr_pad_size(cfg),
               cfg->mask_radius);

    *num_masked_out = n_kept;
    return MR_SUCCESS;
}

/**
 * @brief Argument checks of ::mr_calc_mask.
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
