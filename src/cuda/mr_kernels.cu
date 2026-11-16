/**
 * @file mr_kernels.cu
 * @brief CUDA kernel implementations for the MR_AHA operators.
 *
 * See mr_kernels.cuh for the interface documentation of every kernel; the
 * comments here describe the indexing arithmetic.
 *
 * Two families live here:
 *  - the *apply* kernels of @f$A^H A@f$ for one coil @f$c@f$,
 *    @f$\mathcal{P}^H S_c^H F^{-1} B F \mathcal{P} S_c@f$, where
 *    ::mr_k_pad_mul_se is @f$\mathcal{P} S_c@f$ (pad), ::mr_k_crop_mul_conj_se
 *    is @f$S_c^H \mathcal{P}^H@f$ (crop), and the block-diagonal multiplier
 *    @f$B@f$ is ::mr_k_blockdiag_dense or ::mr_k_blockdiag_masked (+
 *    ::mr_k_scatter_masked). The FFTs in between are cuFFT calls on the host
 *    side. These generic block-diagonal kernels are the fallback used when
 *    ::mr_bd_lookup offers neither a specialised nor a tiled kernel;
 *  - the *calc_toep* kernels that build `phi2_toep`: pack the Gram blocks
 *    (::mr_k_extract_tri), broadcast them as NUFFT strengths
 *    (::mr_k_fill_strengths), and reorder / scale / gather the inverse-FFT
 *    output (::mr_k_to_natural, ::mr_k_to_centered_scale,
 *    ::mr_k_gather_masked_scale).
 *
 * Common conventions: all arrays are column-major (first index fastest);
 * every kernel is a flat 1-D launch with one thread per output element (or per
 * grid location), indexed by a 64-bit `tid` so that grids above 2^31 elements
 * work, and threads past the end return at once. None of these kernels uses
 * `__syncthreads`, so an early return is always safe.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_kernels.cuh"

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
__device__ __forceinline__ void mr_decompose3(long long idx, int d0, int d1,
                                              int c[3])
{
    /* idx = i0 + d0 * (i1 + d1 * i2): peel off the fastest axis first. The
     * slowest extent d2 is not needed (whatever remains is i2). */
    c[0] = (int)(idx % d0);
    idx /= d0;
    c[1] = (int)(idx % d1);
    c[2] = (int)(idx / d1);
}

/**
 * @brief Apply a Hermitian packed block to a staged rank vector.
 *
 * Gram-block convention: the rank vector is a *row* vector multiplied from
 * the right, @f$y = v A@f$, i.e.
 * @f$y_i = \sum_j v_j A_{ji}@f$ (the sum runs over the first index of the
 * block). The stored triangle lives at `base[t * stride]` for packed offset
 * `t`, and the unstored half follows from @f$A_{ji} = \overline{A_{ij}}@f$.
 *
 * @param[in]  v         Staged input vector of length @p L (shared memory).
 * @param[in]  base      Pointer to packed component 0 of this location.
 * @param[in]  stride    Element stride between consecutive packed components.
 * @param[in]  L         Rank.
 * @param[in]  tri_upper Non-zero when the upper triangle is stored.
 * @param[in]  i         Output component (column of the block) to evaluate.
 * @return Component @p i of the vector-matrix product @f$v A@f$.
 */
__device__ __forceinline__ mr_complex mr_apply_block_row(const mr_complex *v,
                                                         const mr_complex *base,
                                                         long long stride, int L,
                                                         int tri_upper, int i)
{
    /* Dot product of v with column i of the full Hermitian block. Each call
     * rebuilds the column from the triangle, so a caller evaluating all L
     * outputs reads every off-diagonal stored element twice (once directly,
     * once conjugated); the specialised / tiled kernels avoid this. */
    mr_complex acc = make_cuFloatComplex(0.0f, 0.0f);
    for (int j = 0; j < L; ++j) {
        int do_conj = 0;
        /* Element A_{ji}: row j, column i (v * A). */
        int t = mr_tri_offset(j, i, L, tri_upper, &do_conj);
        mr_complex a = base[(long long)t * stride];
        if (do_conj) a = cuConjf(a);
        acc = cuCaddf(acc, cuCmulf(a, v[j]));
    }
    return acc;
}

/* ------------------------------------------------------------------------ */
/* AhA kernels                                                              */
/* ------------------------------------------------------------------------ */

__global__ void mr_k_pad_mul_se(mr_complex *d_pad, const mr_complex *d_x,
                                const mr_complex *d_SEs, int coil, mr_dims dims)
{
    /* One thread per location of the padded grid (not of the image), so that
     * every location of every rank plane is written exactly once, including
     * the zeros: this is what lets the kernel replace a memset. */
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dims.pad_size) return;

    /* Natural-order grid coordinates (j0, j1, j2) of this thread. */
    int jc[3];
    mr_decompose3(tid, dims.m[0], dims.m[1], jc);

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

    /* Linear image index of (x0, x1, x2) in the [Nx, Ny, Nz] volume; the same
     * offset addresses the coil map and every rank plane of d_x. */
    long long ii = (long long)xc[0] +
                   (long long)dims.n[0] * (xc[1] + (long long)dims.n[1] * xc[2]);
    /* One map per coil, broadcast across every rank. */
    const mr_complex s = d_SEs[dims.se_coil * coil + ii];

    /* d_pad[:, l] = S_c .* x[:, l] for each rank plane l (planes pad_size
     * apart in d_pad, image_size apart in d_x). */
    for (int l = 0; l < dims.L; ++l)
        d_pad[tid + dims.pad_size * l] =
            cuCmulf(s, d_x[ii + dims.image_size * l]);
}

__global__ void mr_k_crop_mul_conj_se(mr_complex *d_out, const mr_complex *d_pad,
                                      const mr_complex *d_SEs, int coil,
                                      int accumulate, mr_dims dims)
{
    /* One thread per *image* location: the crop only reads the part of the
     * padded grid that the image covers; everything else is discarded. Each
     * thread owns its output element, so the accumulation needs no atomics
     * (coils are serialised by the caller, one launch per coil). */
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dims.image_size) return;

    /* Image coordinates (x0, x1, x2) of this thread. */
    int xc[3];
    mr_decompose3(tid, dims.n[0], dims.n[1], xc);

    /* Same remap as the scatter, so this kernel is its exact adjoint. */
    long long jj = 0, stride = 1;
    for (int d = 0; d < 3; ++d) {
        int j = xc[d] - dims.c_n[d];
        if (j < 0) j += dims.m[d];
        jj += (long long)j * stride;
        stride *= dims.m[d];
    }

    /* out[:, l] (+)= conj(S_c) .* pad[jj, l] for each rank plane. The first
     * coil overwrites (accumulate == 0), so d_out need not be pre-zeroed. */
    const mr_complex se_conj = cuConjf(d_SEs[dims.se_coil * coil + tid]);
    for (int l = 0; l < dims.L; ++l) {
        mr_complex v = cuCmulf(se_conj, d_pad[jj + dims.pad_size * l]);
        long long o = tid + dims.image_size * l;
        d_out[o] = accumulate ? cuCaddf(d_out[o], v) : v;
    }
}

__global__ void mr_k_blockdiag_dense(mr_complex *d_pad,
                                     const mr_complex *d_phi2_toep, mr_dims dims)
{
    /* Each thread owns a private slice of L complex values in dynamic shared
     * memory (blockDim.x * L in total, sized by the launcher). The rank is a
     * run-time value, so the vector cannot live in a register array; it must
     * be staged because the multiply is in place: every output needs every
     * input, and the outputs overwrite the inputs in d_pad. The slices are
     * private, so no __syncthreads is needed (and the early return below is
     * safe). */
    extern __shared__ mr_complex mr_shared[];
    mr_complex *v = mr_shared + (long long)threadIdx.x * dims.L;

    /* One thread per upsampled k-space location (natural order). */
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dims.pad_size) return;

    /* The working buffer is in natural FFT order; phi2_toep is centred. */
    int jc[3];
    mr_decompose3(tid, dims.m[0], dims.m[1], jc);
    long long pc = 0, stride = 1;
    for (int d = 0; d < 3; ++d) {
        int k = mr_nat2cent(jc[d], dims.m[d], dims.c_m[d]);
        pc += (long long)k * stride;
        stride *= dims.m[d];
    }

    /* Stage the input rank vector v_l = d_pad[tid, l] before any output is
     * written back over it. */
    for (int l = 0; l < dims.L; ++l) v[l] = d_pad[tid + dims.pad_size * l];

    /* Packed component t of this location's block is at base[t * pad_size]
     * (phi2_toep is [Mx, My, Mz, ntri], location fastest, so neighbouring
     * threads read neighbouring addresses for a given t). Evaluate y = v A
     * one output at a time and write it in place. */
    const mr_complex *base = d_phi2_toep + pc;
    for (int i = 0; i < dims.L; ++i) {
        mr_complex y = mr_apply_block_row(v, base, dims.pad_size, dims.L,
                                          dims.tri_upper, i);
        d_pad[tid + dims.pad_size * i] = y;
    }
}

__global__ void mr_k_blockdiag_masked(mr_complex *d_compact,
                                      const mr_complex *d_pad,
                                      const mr_complex *d_phi2_toep,
                                      const int64_t *d_mask,
                                      long long num_masked, mr_dims dims)
{
    /* Private per-thread staging slice, as in mr_k_blockdiag_dense. (Here the
     * output goes to a separate compact buffer, but staging still avoids
     * re-reading the scattered d_pad values once per output.) */
    extern __shared__ mr_complex mr_shared[];
    mr_complex *v = mr_shared + (long long)threadIdx.x * dims.L;

    /* One thread per retained (masked) location, i.e. per mask entry. */
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_masked) return;

    /* The mask stores centred linear indices; convert to natural order. */
    int kc[3];
    mr_decompose3((long long)d_mask[tid], dims.m[0], dims.m[1], kc);
    long long jj = 0, stride = 1;
    for (int d = 0; d < 3; ++d) {
        int j = mr_cent2nat(kc[d], dims.m[d], dims.c_m[d]);
        jj += (long long)j * stride;
        stride *= dims.m[d];
    }

    /* Gather the rank vector of grid location jj (a scattered read). */
    for (int l = 0; l < dims.L; ++l) v[l] = d_pad[jj + dims.pad_size * l];

    /* The masked kernel is [num_masked, ntri]: the block of mask entry tid is
     * column-strided by num_masked. Results go to the compact [num_masked, L]
     * buffer, which the caller scatters back with mr_k_scatter_masked. */
    const mr_complex *base = d_phi2_toep + tid;
    for (int i = 0; i < dims.L; ++i)
        d_compact[tid + num_masked * i] =
            mr_apply_block_row(v, base, num_masked, dims.L, dims.tri_upper, i);
}

__global__ void mr_k_mask_inverse(int *d_inv, const int64_t *d_mask,
                                  long long num_masked, mr_dims dims)
{
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_masked) return;

    /* Centred linear index -> natural (FFT-order) linear index. */
    int kc[3];
    mr_decompose3((long long)d_mask[tid], dims.m[0], dims.m[1], kc);
    long long jj = 0, stride = 1;
    for (int d = 0; d < 3; ++d) {
        int j = mr_cent2nat(kc[d], dims.m[d], dims.c_m[d]);
        jj += (long long)j * stride;
        stride *= dims.m[d];
    }
    /* Mask entries are distinct locations, so no two threads write the same
     * jj and no atomics are needed. Locations no thread writes keep the -1 of
     * the caller's 0xFF memset. */
    d_inv[jj] = (int)tid;
}

__global__ void mr_k_scatter_masked(mr_complex *d_pad, const mr_complex *d_compact,
                                    const int64_t *d_mask, long long num_masked,
                                    mr_dims dims)
{
    /* One thread per mask entry: the inverse of the gather in
     * mr_k_blockdiag_masked (same centred -> natural conversion). */
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_masked) return;

    int kc[3];
    mr_decompose3((long long)d_mask[tid], dims.m[0], dims.m[1], kc);
    long long jj = 0, stride = 1;
    for (int d = 0; d < 3; ++d) {
        int j = mr_cent2nat(kc[d], dims.m[d], dims.c_m[d]);
        jj += (long long)j * stride;
        stride *= dims.m[d];
    }

    /* Copy every rank of compact row tid to grid location jj. */
    for (int l = 0; l < dims.L; ++l)
        d_pad[jj + dims.pad_size * l] = d_compact[tid + num_masked * l];
}

/* ------------------------------------------------------------------------ */
/* calc_toep kernels                                                        */
/* ------------------------------------------------------------------------ */

__global__ void mr_k_extract_tri(mr_complex *d_tri, const mr_complex *d_phi2,
                                 long long nbatch, int L, int tri_upper)
{
    /* One thread per element of the *dense* input [nbatch, L, L]; threads on
     * the unstored half exit below. tid follows the input layout (batch
     * fastest), so the reads are coalesced; the writes are too, since
     * consecutive b land on consecutive addresses of a packed component. */
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = nbatch * (long long)L * L;
    if (tid >= total) return;

    /* tid = b + nbatch * (i + L * j): batch b, block row i, block column j. */
    long long b = tid % nbatch;
    long long rem = tid / nbatch;
    int i = (int)(rem % L);
    int j = (int)(rem / L);

    /* Only the stored half is copied; the rest follows by Hermitian symmetry. */
    int keep = tri_upper ? (i <= j) : (i >= j);
    if (!keep) return;

    /* (i, j) is inside the stored triangle here, so do_conj is always 0 and
     * the element is copied as is to packed offset t. */
    int do_conj = 0;
    int t = mr_tri_offset(i, j, L, tri_upper, &do_conj);
    d_tri[b + nbatch * t] = d_phi2[b + nbatch * (i + (long long)L * j)];
}

__global__ void mr_k_fill_strengths(mr_complex *d_c, const mr_complex *d_tri,
                                    long long Ntrajs, long long Nkx,
                                    int Nkz_eff, long long t0, int chunk,
                                    long long ntri)
{
    /* One thread per strength of the chunk: d_c is [M_nu, Nkz_eff, chunk]
     * with the non-uniform sample fastest, i.e. `Nkz_eff * chunk` stacked
     * strength vectors, one per cufinufft transform of the batch. */
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
    /* The trajectory is [Ntrajs, Nkx] column-major, so sample m belongs to
     * trajectory m % Ntrajs (readout position m / Ntrajs is irrelevant). */
    long long itraj = m % Ntrajs;         /* the block is shared by a readout  */
    d_c[tid] = d_tri[itraj + Ntrajs * (kz + Nkz_eff * t)];
}

__global__ void mr_k_to_natural(mr_complex *dst, const mr_complex *src,
                                int m0, int m1, int m2, int s0, int s1, int s2,
                                long long batch)
{
    /* One thread per *destination* element, so the writes are coalesced and
     * the reads are a per-axis cyclic shift of them (contiguous runs). */
    long long vol = (long long)m0 * m1 * m2;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= vol * batch) return;

    /* Split into volume b and in-volume linear index idx. */
    long long b = tid / vol;
    long long idx = tid - b * vol;

    /* Natural index j of the destination holds centred index
     * k = (j + s) mod m of the source (mr_nat2cent); s = 0 is the identity. */
    int jc[3];
    mr_decompose3(idx, m0, m1, jc);
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

__global__ void mr_k_to_centered_scale(mr_complex *dst, const mr_complex *src,
                                       int m0, int m1, int m2,
                                       int s0, int s1, int s2,
                                       long long batch, float scale)
{
    /* One thread per destination element, as in mr_k_to_natural. */
    long long vol = (long long)m0 * m1 * m2;
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= vol * batch) return;

    long long b = tid / vol;
    long long idx = tid - b * vol;

    /* Centred index k of the destination comes from natural index
     * j = (k - s) mod m of the source (mr_cent2nat); s = 0 is the identity.
     * The scale is folded into the same pass to save a read/write sweep. */
    int kc[3];
    mr_decompose3(idx, m0, m1, kc);
    const int mm[3] = {m0, m1, m2};
    const int ss[3] = {s0, s1, s2};

    long long src_idx = 0, stride = 1;
    for (int d = 0; d < 3; ++d) {
        int j = kc[d] - ss[d];
        if (j < 0) j += mm[d];
        src_idx += (long long)j * stride;
        stride *= mm[d];
    }
    mr_complex v = src[b * vol + src_idx];
    dst[tid] = make_cuFloatComplex(v.x * scale, v.y * scale);
}

__global__ void mr_k_gather_masked_scale(mr_complex *dst, const mr_complex *src,
                                         const int64_t *d_mask,
                                         long long num_masked,
                                         int m0, int m1, int m2,
                                         int s0, int s1, int s2,
                                         long long t0, long long batch,
                                         float scale)
{
    /* One thread per (mask entry, component of the chunk) pair, mask entry
     * fastest, so the writes into column t0 + b of the [num_masked, ntri]
     * output are coalesced; the reads follow the mask and are scattered. */
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_masked * batch) return;

    long long i = tid % num_masked;   /* masked location                       */
    long long b = tid / num_masked;   /* component within this chunk           */

    /* The mask holds centred linear indices; find the matching natural-order
     * element of volume b, exactly as in mr_k_to_centered_scale. */
    int kc[3];
    mr_decompose3((long long)d_mask[i], m0, m1, kc);
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
    mr_complex v = src[b * vol + src_idx];
    dst[i + num_masked * (t0 + b)] =
        make_cuFloatComplex(v.x * scale, v.y * scale);
}
