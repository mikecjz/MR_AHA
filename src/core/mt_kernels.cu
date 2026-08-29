/**
 * @file mt_kernels.cu
 * @brief CUDA kernel implementations for the MT_CUDA operators.
 *
 * See mt_kernels.cuh for the interface documentation of every kernel; the
 * comments here describe the indexing arithmetic.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mt_kernels.cuh"

/* ------------------------------------------------------------------------ */
/* Device-local index helpers                                               */
/* ------------------------------------------------------------------------ */

/**
 * @brief Decompose a column-major linear index into 3-D coordinates.
 * @param[in]  idx Linear index into a `[d0, d1, d2]` array.
 * @param[in]  d0  Fastest-varying extent.
 * @param[in]  d1  Middle extent.
 * @param[out] c   Receives `[i0, i1, i2]`.
 */
__device__ __forceinline__ void mt_decompose3(long long idx, int d0, int d1,
                                              int c[3])
{
    c[0] = (int)(idx % d0);
    idx /= d0;
    c[1] = (int)(idx % d1);
    c[2] = (int)(idx / d1);
}

/**
 * @brief Apply a Hermitian packed block to a staged rank vector.
 *
 * Computes @f$y_i = \sum_j A_{ij} v_j@f$ where the stored triangle lives at
 * `base[t * stride]` for packed offset `t`, and the unstored half follows from
 * @f$A_{ij} = \overline{A_{ji}}@f$.
 *
 * @param[in]  v         Staged input vector of length @p L (shared memory).
 * @param[in]  base      Pointer to packed component 0 of this location.
 * @param[in]  stride    Element stride between consecutive packed components.
 * @param[in]  L         Rank.
 * @param[in]  tri_upper Non-zero when the upper triangle is stored.
 * @param[in]  i         Output row to evaluate.
 * @return Row @p i of the matrix-vector product.
 */
__device__ __forceinline__ mt_complex mt_apply_block_row(const mt_complex *v,
                                                         const mt_complex *base,
                                                         long long stride, int L,
                                                         int tri_upper, int i)
{
    mt_complex acc = make_cuFloatComplex(0.0f, 0.0f);
    for (int j = 0; j < L; ++j) {
        int do_conj = 0;
        int t = mt_tri_offset(i, j, L, tri_upper, &do_conj);
        mt_complex a = base[(long long)t * stride];
        if (do_conj) a = cuConjf(a);
        acc = cuCaddf(acc, cuCmulf(a, v[j]));
    }
    return acc;
}

/* ------------------------------------------------------------------------ */
/* AhA kernels                                                              */
/* ------------------------------------------------------------------------ */

__global__ void mt_k_pad_mul_se(mt_complex *d_pad, const mt_complex *d_x,
                                const mt_complex *d_SEs, int coil, mt_dims dims)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dims.pad_size) return;

    int jc[3];
    mt_decompose3(tid, dims.m[0], dims.m[1], jc);

    /* Invert the scatter j = (x - floor(N/2)) mod M performed by the forward
     * mapping. The candidate lies inside the image exactly when it is < N,
     * because that map is injective for N <= M. */
    int xc[3];
    int inside = 1;
    for (int d = 0; d < 3; ++d) {
        int cand = jc[d] + dims.c_n[d];
        if (cand >= dims.m[d]) cand -= dims.m[d];
        xc[d] = cand;
        if (cand >= dims.n[d]) inside = 0;
    }

    if (!inside) {
        /* Zero fill: this pass replaces the memset of the whole padded buffer. */
        for (int l = 0; l < dims.L; ++l)
            d_pad[tid + dims.pad_size * l] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }

    long long ii = (long long)xc[0] +
                   (long long)dims.n[0] * (xc[1] + (long long)dims.n[1] * xc[2]);
    /* One map per coil, broadcast across every rank. */
    const mt_complex s = d_SEs[dims.se_coil * coil + ii];

    for (int l = 0; l < dims.L; ++l)
        d_pad[tid + dims.pad_size * l] =
            cuCmulf(s, d_x[ii + dims.image_size * l]);
}

__global__ void mt_k_crop_mul_conj_se(mt_complex *d_out, const mt_complex *d_pad,
                                      const mt_complex *d_SEs, int coil,
                                      int accumulate, mt_dims dims)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dims.image_size) return;

    int xc[3];
    mt_decompose3(tid, dims.n[0], dims.n[1], xc);

    /* Same remap as the scatter, so this kernel is its exact adjoint. */
    long long jj = 0, stride = 1;
    for (int d = 0; d < 3; ++d) {
        int j = xc[d] - dims.c_n[d];
        if (j < 0) j += dims.m[d];
        jj += (long long)j * stride;
        stride *= dims.m[d];
    }

    const mt_complex se_conj = cuConjf(d_SEs[dims.se_coil * coil + tid]);
    for (int l = 0; l < dims.L; ++l) {
        mt_complex v = cuCmulf(se_conj, d_pad[jj + dims.pad_size * l]);
        long long o = tid + dims.image_size * l;
        d_out[o] = accumulate ? cuCaddf(d_out[o], v) : v;
    }
}

__global__ void mt_k_blockdiag_dense(mt_complex *d_pad,
                                     const mt_complex *d_phi2_toep, mt_dims dims)
{
    extern __shared__ mt_complex mt_shared[];
    mt_complex *v = mt_shared + (long long)threadIdx.x * dims.L;

    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dims.pad_size) return;

    /* The working buffer is in natural FFT order; phi2_toep is centred. */
    int jc[3];
    mt_decompose3(tid, dims.m[0], dims.m[1], jc);
    long long pc = 0, stride = 1;
    for (int d = 0; d < 3; ++d) {
        int k = mt_nat2cent(jc[d], dims.m[d], dims.c_m[d]);
        pc += (long long)k * stride;
        stride *= dims.m[d];
    }

    for (int l = 0; l < dims.L; ++l) v[l] = d_pad[tid + dims.pad_size * l];

    const mt_complex *base = d_phi2_toep + pc;
    for (int i = 0; i < dims.L; ++i) {
        mt_complex y = mt_apply_block_row(v, base, dims.pad_size, dims.L,
                                          dims.tri_upper, i);
        d_pad[tid + dims.pad_size * i] = y;
    }
}

__global__ void mt_k_blockdiag_masked(mt_complex *d_compact,
                                      const mt_complex *d_pad,
                                      const mt_complex *d_phi2_toep,
                                      const int64_t *d_mask,
                                      long long num_masked, mt_dims dims)
{
    extern __shared__ mt_complex mt_shared[];
    mt_complex *v = mt_shared + (long long)threadIdx.x * dims.L;

    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_masked) return;

    /* The mask stores centred linear indices; convert to natural order. */
    int kc[3];
    mt_decompose3((long long)d_mask[tid], dims.m[0], dims.m[1], kc);
    long long jj = 0, stride = 1;
    for (int d = 0; d < 3; ++d) {
        int j = mt_cent2nat(kc[d], dims.m[d], dims.c_m[d]);
        jj += (long long)j * stride;
        stride *= dims.m[d];
    }

    for (int l = 0; l < dims.L; ++l) v[l] = d_pad[jj + dims.pad_size * l];

    const mt_complex *base = d_phi2_toep + tid;
    for (int i = 0; i < dims.L; ++i)
        d_compact[tid + num_masked * i] =
            mt_apply_block_row(v, base, num_masked, dims.L, dims.tri_upper, i);
}

__global__ void mt_k_scatter_masked(mt_complex *d_pad, const mt_complex *d_compact,
                                    const int64_t *d_mask, long long num_masked,
                                    mt_dims dims)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_masked) return;

    int kc[3];
    mt_decompose3((long long)d_mask[tid], dims.m[0], dims.m[1], kc);
    long long jj = 0, stride = 1;
    for (int d = 0; d < 3; ++d) {
        int j = mt_cent2nat(kc[d], dims.m[d], dims.c_m[d]);
        jj += (long long)j * stride;
        stride *= dims.m[d];
    }

    for (int l = 0; l < dims.L; ++l)
        d_pad[jj + dims.pad_size * l] = d_compact[tid + num_masked * l];
}

/* ------------------------------------------------------------------------ */
/* calc_toep kernels                                                        */
/* ------------------------------------------------------------------------ */

__global__ void mt_k_extract_tri(mt_complex *d_tri, const mt_complex *d_phi2,
                                 long long nbatch, int L, int tri_upper)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = nbatch * (long long)L * L;
    if (tid >= total) return;

    long long b = tid % nbatch;
    long long rem = tid / nbatch;
    int i = (int)(rem % L);
    int j = (int)(rem / L);

    /* Only the stored half is copied; the rest follows by Hermitian symmetry. */
    int keep = tri_upper ? (i <= j) : (i >= j);
    if (!keep) return;

    int do_conj = 0;
    int t = mt_tri_offset(i, j, L, tri_upper, &do_conj);
    d_tri[b + nbatch * t] = d_phi2[b + nbatch * (i + (long long)L * j)];
}

__global__ void mt_k_fill_strengths(mt_complex *d_c, const mt_complex *d_tri,
                                    long long Ntrajs, long long Nkx,
                                    int Nkz_eff, long long t0, int chunk,
                                    long long ntri)
{
    long long M_nu = Ntrajs * Nkx;
    long long total = M_nu * Nkz_eff * chunk;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    long long m = tid % M_nu;             /* non-uniform sample index          */
    long long rest = tid / M_nu;
    long long kz = rest % Nkz_eff;        /* kz encode (stack only)            */
    long long cl = rest / Nkz_eff;        /* component within this chunk       */

    const long long t = t0 + cl;
    if (t >= ntri) {                      /* zero padding of the final chunk   */
        d_c[tid] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }
    long long itraj = m % Ntrajs;         /* the block is shared by a readout  */
    d_c[tid] = d_tri[itraj + Ntrajs * (kz + Nkz_eff * t)];
}

__global__ void mt_k_to_natural(mt_complex *dst, const mt_complex *src,
                                int m0, int m1, int m2, int s0, int s1, int s2,
                                long long batch)
{
    long long vol = (long long)m0 * m1 * m2;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= vol * batch) return;

    long long b = tid / vol;
    long long idx = tid - b * vol;

    int jc[3];
    mt_decompose3(idx, m0, m1, jc);
    const int mm[3] = {m0, m1, m2};
    const int ss[3] = {s0, s1, s2};

    long long src_idx = 0, stride = 1;
    for (int d = 0; d < 3; ++d) {
        int k = jc[d] + ss[d];
        if (k >= mm[d]) k -= mm[d];
        src_idx += (long long)k * stride;
        stride *= mm[d];
    }
    dst[tid] = src[b * vol + src_idx];
}

__global__ void mt_k_to_centered_scale(mt_complex *dst, const mt_complex *src,
                                       int m0, int m1, int m2,
                                       int s0, int s1, int s2,
                                       long long batch, float scale)
{
    long long vol = (long long)m0 * m1 * m2;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= vol * batch) return;

    long long b = tid / vol;
    long long idx = tid - b * vol;

    int kc[3];
    mt_decompose3(idx, m0, m1, kc);
    const int mm[3] = {m0, m1, m2};
    const int ss[3] = {s0, s1, s2};

    long long src_idx = 0, stride = 1;
    for (int d = 0; d < 3; ++d) {
        int j = kc[d] - ss[d];
        if (j < 0) j += mm[d];
        src_idx += (long long)j * stride;
        stride *= mm[d];
    }
    mt_complex v = src[b * vol + src_idx];
    dst[tid] = make_cuFloatComplex(v.x * scale, v.y * scale);
}

__global__ void mt_k_gather_masked_scale(mt_complex *dst, const mt_complex *src,
                                         const int64_t *d_mask,
                                         long long num_masked,
                                         int m0, int m1, int m2,
                                         int s0, int s1, int s2,
                                         long long t0, long long batch,
                                         float scale)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_masked * batch) return;

    long long i = tid % num_masked;   /* masked location                       */
    long long b = tid / num_masked;   /* component within this chunk           */

    int kc[3];
    mt_decompose3((long long)d_mask[i], m0, m1, kc);
    const int mm[3] = {m0, m1, m2};
    const int ss[3] = {s0, s1, s2};

    long long vol = (long long)m0 * m1 * m2;
    long long src_idx = 0, stride = 1;
    for (int d = 0; d < 3; ++d) {
        int j = kc[d] - ss[d];
        if (j < 0) j += mm[d];
        src_idx += (long long)j * stride;
        stride *= mm[d];
    }
    mt_complex v = src[b * vol + src_idx];
    dst[i + num_masked * (t0 + b)] =
        make_cuFloatComplex(v.x * scale, v.y * scale);
}
