/**
 * @file mr_kernels.metal
 * @brief Metal Shading Language kernels of the Metal backend.
 *
 * Port of `src/cuda/mr_kernels.cu`, one kernel per CUDA kernel and with the
 * same indexing arithmetic; see that file and `src/cuda/mr_kernels.cuh` for the
 * derivations. Complex numbers are `float2` (`.x` real, `.y` imaginary), which
 * is layout-identical to ::mr_complex.
 *
 * The file is either compiled to a `.metallib` at build time (when the Metal
 * toolchain is installed) or embedded as source and compiled by the system's
 * Metal compiler on first use; see `src/metal/mr_metal_runtime.cpp`. Both
 * paths compile it as Metal 3.0 with fast math disabled.
 *
 * Index widths: a thread index (`uint`) addresses one voxel of one volume, so
 * the host rejects volumes of 2^32 voxels or more; offsets that include the
 * rank / component axis are formed in 64-bit (`ulong`).
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include <metal_stdlib>
using namespace metal;

/**
 * @brief Geometry descriptor, mirroring ::mr_dims in `src/common/mr_common.h`
 *        field for field (80 bytes; the host static_asserts the layout).
 */
struct mr_dims {
    int  n[3];       /**< Image size `[Nx, Ny, Nz]`, unused entries 1.     */
    int  m[3];       /**< Upsampled size `[Mx, My, Mz]`, unused entries 1. */
    int  c_n[3];     /**< Image centre offsets `floor(N_d/2)`.             */
    int  c_m[3];     /**< Upsampled centre offsets `floor(M_d/2)`.         */
    int  L;          /**< Rank.                                            */
    int  tri_upper;  /**< 1 when the upper triangle is stored, else 0.     */
    long image_size; /**< `Nx*Ny*Nz`.                                      */
    long pad_size;   /**< `Mx*My*Mz`.                                      */
    long se_coil;    /**< Coil stride into `SEs` (= image_size).           */
};

/* ------------------------------------------------------------------------ */
/* Helpers                                                                  */
/* ------------------------------------------------------------------------ */

/** @brief Complex product `a * b`. */
static inline float2 mr_cmul(float2 a, float2 b)
{
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

/** @brief Complex product `conj(a) * b`. */
static inline float2 mr_cmul_conj(float2 a, float2 b)
{
    return float2(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);
}

/** @brief Decompose a column-major linear index into `[i0, i1, i2]`. */
static inline uint3 mr_decompose3(uint idx, uint d0, uint d1)
{
    const uint i0 = idx % d0;
    idx /= d0;
    return uint3(i0, idx % d1, idx / d1);
}

/** @brief Centred index -> natural (FFT-order) index, see `mr_cent2nat`. */
static inline int mr_cent2nat(int k, int M, int c)
{
    const int j = k - c;
    return (j < 0) ? (j + M) : j;
}

/** @brief Natural (FFT-order) index -> centred index, see `mr_nat2cent`. */
static inline int mr_nat2cent(int j, int M, int c)
{
    const int k = j + c;
    return (k >= M) ? (k - M) : k;
}

/**
 * @brief Packed offset of element `(i, j)` of the stored Hermitian triangle.
 *
 * Same packing as `mr_tri_offset` in `src/cuda/mr_internal.cuh`; @p conj is
 * set when the transposed element is addressed and must be conjugated.
 */
static inline int mr_tri_offset(int i, int j, int L, int tri_upper,
                                thread bool &conj)
{
    int r = i, c = j;
    conj = false;
    if (tri_upper) {
        if (r > c) { const int t = r; r = c; c = t; conj = true; }
        return c * (c + 1) / 2 + r;
    }
    if (r < c) { const int t = r; r = c; c = t; conj = true; }
    return c * L - c * (c - 1) / 2 + (r - c);
}

/**
 * @brief Row @p i of the Hermitian block times the staged rank vector @p v.
 *
 * @param v         Staged input vector of length @p L (threadgroup memory).
 * @param base      Packed component 0 of this location.
 * @param stride    Element stride between consecutive packed components.
 * @param L         Rank.
 * @param tri_upper Non-zero when the upper triangle is stored.
 * @param i         Output row.
 */
static inline float2 mr_apply_block_row(threadgroup const float2 *v,
                                        device const float2 *base, ulong stride,
                                        int L, int tri_upper, int i)
{
    float2 acc = float2(0.0f);
    for (int j = 0; j < L; ++j) {
        bool conj;
        const int t = mr_tri_offset(i, j, L, tri_upper, conj);
        const float2 a = base[(ulong)t * stride];
        acc += conj ? mr_cmul_conj(a, v[j]) : mr_cmul(a, v[j]);
    }
    return acc;
}

/** @brief Natural-order linear index of a centred linear mask index. */
static inline ulong mr_mask_to_natural(long kidx, constant mr_dims &d)
{
    const uint3 kc = mr_decompose3((uint)kidx, (uint)d.m[0], (uint)d.m[1]);
    const ulong j0 = (ulong)mr_cent2nat((int)kc.x, d.m[0], d.c_m[0]);
    const ulong j1 = (ulong)mr_cent2nat((int)kc.y, d.m[1], d.c_m[1]);
    const ulong j2 = (ulong)mr_cent2nat((int)kc.z, d.m[2], d.c_m[2]);
    return j0 + (ulong)d.m[0] * (j1 + (ulong)d.m[1] * j2);
}

/* ------------------------------------------------------------------------ */
/* AhA kernels                                                              */
/* ------------------------------------------------------------------------ */

/**
 * @brief Zero-pad `S_c x` onto the upsampled grid in natural FFT order.
 *
 * One thread per upsampled voxel; writes all `L` ranks and zero-fills voxels
 * outside the image, so the padded buffer needs no separate clear. Port of
 * `mr_k_pad_mul_se`.
 */
kernel void mr_k_pad_mul_se(device float2       *pad  [[buffer(0)]],
                            device const float2 *x    [[buffer(1)]],
                            device const float2 *SEs  [[buffer(2)]],
                            constant mr_dims    &d    [[buffer(3)]],
                            constant int        &coil [[buffer(4)]],
                            uint tid [[thread_position_in_grid]])
{
    if ((long)tid >= d.pad_size) return;
    const uint3 jc = mr_decompose3(tid, (uint)d.m[0], (uint)d.m[1]);

    /* Invert the scatter j = (x - floor(N/2)) mod M; inside iff cand < N. */
    int xc[3];
    bool inside = true;
    for (int a = 0; a < 3; ++a) {
        int cand = (int)jc[a] + d.c_n[a];
        if (cand >= d.m[a]) cand -= d.m[a];
        xc[a] = cand;
        if (cand >= d.n[a]) inside = false;
    }

    const ulong padn = (ulong)d.pad_size;
    if (!inside) {
        for (int l = 0; l < d.L; ++l) pad[(ulong)tid + padn * l] = float2(0.0f);
        return;
    }

    const ulong ii = (ulong)xc[0] +
                     (ulong)d.n[0] * ((ulong)xc[1] + (ulong)d.n[1] * (ulong)xc[2]);
    const float2 s = SEs[(ulong)d.se_coil * (ulong)coil + ii];
    const ulong img = (ulong)d.image_size;
    for (int l = 0; l < d.L; ++l)
        pad[(ulong)tid + padn * l] = mr_cmul(s, x[ii + img * l]);
}

/**
 * @brief Crop back to the image, multiply by `conj(S_c)` and (optionally)
 *        accumulate into the output. Exact adjoint of ::mr_k_pad_mul_se.
 *        Port of `mr_k_crop_mul_conj_se`.
 */
kernel void mr_k_crop_mul_conj_se(device float2       *out        [[buffer(0)]],
                                  device const float2 *pad        [[buffer(1)]],
                                  device const float2 *SEs        [[buffer(2)]],
                                  constant mr_dims    &d          [[buffer(3)]],
                                  constant int        &coil       [[buffer(4)]],
                                  constant int        &accumulate [[buffer(5)]],
                                  uint tid [[thread_position_in_grid]])
{
    if ((long)tid >= d.image_size) return;
    const uint3 xc = mr_decompose3(tid, (uint)d.n[0], (uint)d.n[1]);

    ulong jj = 0, stride = 1;
    for (int a = 0; a < 3; ++a) {
        int j = (int)xc[a] - d.c_n[a];
        if (j < 0) j += d.m[a];
        jj += (ulong)j * stride;
        stride *= (ulong)d.m[a];
    }

    const float2 s = SEs[(ulong)d.se_coil * (ulong)coil + tid];
    const ulong img = (ulong)d.image_size, padn = (ulong)d.pad_size;
    for (int l = 0; l < d.L; ++l) {
        const float2 v = mr_cmul_conj(s, pad[jj + padn * l]);
        const ulong o = (ulong)tid + img * l;
        out[o] = accumulate ? out[o] + v : v;
    }
}

/**
 * @brief In-place Hermitian block-diagonal multiplication over the whole grid.
 *
 * Each thread stages its own rank vector in threadgroup memory (`L` float2
 * per thread, length set by the host) before overwriting it row by row.
 * Port of `mr_k_blockdiag_dense`.
 */
kernel void mr_k_blockdiag_dense(device float2         *pad  [[buffer(0)]],
                                 device const float2   *toep [[buffer(1)]],
                                 constant mr_dims      &d    [[buffer(2)]],
                                 threadgroup float2    *sh   [[threadgroup(0)]],
                                 uint tid [[thread_position_in_grid]],
                                 uint lid [[thread_position_in_threadgroup]])
{
    if ((long)tid >= d.pad_size) return;
    threadgroup float2 *v = sh + lid * (uint)d.L;

    /* The working buffer is in natural FFT order; phi2_toep is centred. */
    const uint3 jc = mr_decompose3(tid, (uint)d.m[0], (uint)d.m[1]);
    const ulong k0 = (ulong)mr_nat2cent((int)jc.x, d.m[0], d.c_m[0]);
    const ulong k1 = (ulong)mr_nat2cent((int)jc.y, d.m[1], d.c_m[1]);
    const ulong k2 = (ulong)mr_nat2cent((int)jc.z, d.m[2], d.c_m[2]);
    const ulong pc = k0 + (ulong)d.m[0] * (k1 + (ulong)d.m[1] * k2);

    const ulong padn = (ulong)d.pad_size;
    for (int l = 0; l < d.L; ++l) v[l] = pad[(ulong)tid + padn * l];

    device const float2 *base = toep + pc;
    for (int i = 0; i < d.L; ++i)
        pad[(ulong)tid + padn * i] =
            mr_apply_block_row(v, base, padn, d.L, d.tri_upper, i);
}

/**
 * @brief Block-diagonal multiplication at the masked locations only, written
 *        to the compact `[num_masked, L]` buffer. Port of
 *        `mr_k_blockdiag_masked`.
 */
kernel void mr_k_blockdiag_masked(device float2       *compact    [[buffer(0)]],
                                  device const float2 *pad        [[buffer(1)]],
                                  device const float2 *toep       [[buffer(2)]],
                                  device const long   *mask       [[buffer(3)]],
                                  constant long       &num_masked [[buffer(4)]],
                                  constant mr_dims    &d          [[buffer(5)]],
                                  threadgroup float2  *sh         [[threadgroup(0)]],
                                  uint tid [[thread_position_in_grid]],
                                  uint lid [[thread_position_in_threadgroup]])
{
    if ((long)tid >= num_masked) return;
    threadgroup float2 *v = sh + lid * (uint)d.L;

    const ulong jj = mr_mask_to_natural(mask[tid], d);
    const ulong padn = (ulong)d.pad_size, nm = (ulong)num_masked;
    for (int l = 0; l < d.L; ++l) v[l] = pad[jj + padn * l];

    device const float2 *base = toep + tid;
    for (int i = 0; i < d.L; ++i)
        compact[(ulong)tid + nm * i] =
            mr_apply_block_row(v, base, nm, d.L, d.tri_upper, i);
}

/* ------------------------------------------------------------------------ */
/* Block-diagonal kernels specialised on L                                  */
/* ------------------------------------------------------------------------ */

/**
 * @brief Hermitian block times vector, reading each stored element once.
 *
 * Walks the stored triangle in packed order (::mr_triangulation): element
 * `t` is entry `(r, c)` of the block, and it contributes `A_rc v_c` to `y_r`
 * and, off the diagonal, `conj(A_rc) v_r` to `y_c` (Hermitian symmetry). The
 * generic ::mr_apply_block_row instead evaluates every row separately and so
 * fetches each off-diagonal element twice; this halves the dominant memory
 * traffic of the block-diagonal stage. With @p L a compile-time constant the
 * vectors live in registers instead of threadgroup memory.
 *
 * @tparam L        Rank.
 * @param  v        Input vector.
 * @param  y        Output vector (overwritten).
 * @param  base     Packed component 0 of this location.
 * @param  stride   Element stride between consecutive packed components.
 * @param  upper    Whether the upper triangle is stored.
 */
template <int L>
static inline void mr_block_apply(thread const float2 (&v)[L], thread float2 (&y)[L],
                                  device const float2 *base, ulong stride,
                                  bool upper)
{
    for (int i = 0; i < L; ++i) y[i] = float2(0.0f);
    ulong off = 0;
    for (int c = 0; c < L; ++c) {
        const int r0 = upper ? 0 : c;
        const int r1 = upper ? c : L - 1;
        for (int r = r0; r <= r1; ++r, off += stride) {
            const float2 a = base[off];
            y[r] += mr_cmul(a, v[c]);
            if (r != c) y[c] += mr_cmul_conj(a, v[r]);
        }
    }
}

/**
 * @brief In-place block-diagonal multiplication over the whole grid,
 *        specialised on L (host names `mr_k_bd_dense_<L>`, L = 1..32).
 *
 * Same contract as ::mr_k_blockdiag_dense, without threadgroup memory.
 */
template <int L>
kernel void mr_k_bd_dense(device float2       *pad  [[buffer(0)]],
                          device const float2 *toep [[buffer(1)]],
                          constant mr_dims    &d    [[buffer(2)]],
                          uint tid [[thread_position_in_grid]])
{
    if ((long)tid >= d.pad_size) return;
    const uint3 jc = mr_decompose3(tid, (uint)d.m[0], (uint)d.m[1]);
    const ulong k0 = (ulong)mr_nat2cent((int)jc.x, d.m[0], d.c_m[0]);
    const ulong k1 = (ulong)mr_nat2cent((int)jc.y, d.m[1], d.c_m[1]);
    const ulong k2 = (ulong)mr_nat2cent((int)jc.z, d.m[2], d.c_m[2]);
    const ulong pc = k0 + (ulong)d.m[0] * (k1 + (ulong)d.m[1] * k2);
    const ulong padn = (ulong)d.pad_size;

    float2 v[L], y[L];
    for (int l = 0; l < L; ++l) v[l] = pad[(ulong)tid + padn * l];
    mr_block_apply<L>(v, y, toep + pc, padn, d.tri_upper != 0);
    for (int l = 0; l < L; ++l) pad[(ulong)tid + padn * l] = y[l];
}

/**
 * @brief Masked block-diagonal multiplication specialised on L (host names
 *        `mr_k_bd_masked_<L>`). Same contract as ::mr_k_blockdiag_masked.
 */
template <int L>
kernel void mr_k_bd_masked(device float2       *compact    [[buffer(0)]],
                           device const float2 *pad        [[buffer(1)]],
                           device const float2 *toep       [[buffer(2)]],
                           device const long   *mask       [[buffer(3)]],
                           constant long       &num_masked [[buffer(4)]],
                           constant mr_dims    &d          [[buffer(5)]],
                           uint tid [[thread_position_in_grid]])
{
    if ((long)tid >= num_masked) return;
    const ulong jj = mr_mask_to_natural(mask[tid], d);
    const ulong padn = (ulong)d.pad_size, nm = (ulong)num_masked;

    float2 v[L], y[L];
    for (int l = 0; l < L; ++l) v[l] = pad[jj + padn * l];
    mr_block_apply<L>(v, y, toep + tid, nm, d.tri_upper != 0);
    for (int l = 0; l < L; ++l) compact[(ulong)tid + nm * l] = y[l];
}

/* Explicit instantiations, L = 1..32. Larger ranks use the generic
 * threadgroup-memory kernels above. */
#define MR_BD_INSTANTIATE(L)                                                     \
    template [[host_name("mr_k_bd_dense_" #L)]] [[kernel]] void                 \
    mr_k_bd_dense<L>(device float2 *pad [[buffer(0)]],                           \
                     device const float2 *toep [[buffer(1)]],                    \
                     constant mr_dims &d [[buffer(2)]],                          \
                     uint tid [[thread_position_in_grid]]);                      \
    template [[host_name("mr_k_bd_masked_" #L)]] [[kernel]] void                \
    mr_k_bd_masked<L>(device float2 *compact [[buffer(0)]],                      \
                      device const float2 *pad [[buffer(1)]],                    \
                      device const float2 *toep [[buffer(2)]],                   \
                      device const long *mask [[buffer(3)]],                     \
                      constant long &num_masked [[buffer(4)]],                   \
                      constant mr_dims &d [[buffer(5)]],                         \
                      uint tid [[thread_position_in_grid]]);
MR_BD_INSTANTIATE(1)  MR_BD_INSTANTIATE(2)  MR_BD_INSTANTIATE(3)  MR_BD_INSTANTIATE(4)
MR_BD_INSTANTIATE(5)  MR_BD_INSTANTIATE(6)  MR_BD_INSTANTIATE(7)  MR_BD_INSTANTIATE(8)
MR_BD_INSTANTIATE(9)  MR_BD_INSTANTIATE(10) MR_BD_INSTANTIATE(11) MR_BD_INSTANTIATE(12)
MR_BD_INSTANTIATE(13) MR_BD_INSTANTIATE(14) MR_BD_INSTANTIATE(15) MR_BD_INSTANTIATE(16)
MR_BD_INSTANTIATE(17) MR_BD_INSTANTIATE(18) MR_BD_INSTANTIATE(19) MR_BD_INSTANTIATE(20)
MR_BD_INSTANTIATE(21) MR_BD_INSTANTIATE(22) MR_BD_INSTANTIATE(23) MR_BD_INSTANTIATE(24)
MR_BD_INSTANTIATE(25) MR_BD_INSTANTIATE(26) MR_BD_INSTANTIATE(27) MR_BD_INSTANTIATE(28)
MR_BD_INSTANTIATE(29) MR_BD_INSTANTIATE(30) MR_BD_INSTANTIATE(31) MR_BD_INSTANTIATE(32)
#undef MR_BD_INSTANTIATE

/**
 * @brief Scatter the compact masked result back into the (zeroed) padded
 *        grid. Port of `mr_k_scatter_masked`.
 */
kernel void mr_k_scatter_masked(device float2       *pad        [[buffer(0)]],
                                device const float2 *compact    [[buffer(1)]],
                                device const long   *mask       [[buffer(2)]],
                                constant long       &num_masked [[buffer(3)]],
                                constant mr_dims    &d          [[buffer(4)]],
                                uint tid [[thread_position_in_grid]])
{
    if ((long)tid >= num_masked) return;
    const ulong jj = mr_mask_to_natural(mask[tid], d);
    const ulong padn = (ulong)d.pad_size, nm = (ulong)num_masked;
    for (int l = 0; l < d.L; ++l)
        pad[jj + padn * l] = compact[(ulong)tid + nm * l];
}

/**
 * @brief Zero @p count consecutive volumes of @p n elements each (one thread
 *        per element of a volume). Replaces `cudaMemsetAsync` in the masked
 *        operator, so the whole operator stays in one compute encoder.
 */
kernel void mr_k_zero(device float2 *p     [[buffer(0)]],
                      constant long &n     [[buffer(1)]],
                      constant int  &count [[buffer(2)]],
                      uint tid [[thread_position_in_grid]])
{
    if ((long)tid >= n) return;
    for (int l = 0; l < count; ++l) p[(ulong)tid + (ulong)n * l] = float2(0.0f);
}

/* ------------------------------------------------------------------------ */
/* calc_toep kernels (stage 2: point-spread function -> phi2_toep)          */
/* ------------------------------------------------------------------------ */

/**
 * @brief Extents and per-axis centring shifts of a reorder kernel.
 *
 * A shift of 0 leaves that axis untouched, which is how the kz axis of a
 * stack-of-stars kernel is carried through without a transform.
 */
struct mr_reorder {
    int m[3]; /**< Axis extents `[m0, m1, m2]`.                         */
    int s[3]; /**< Centre offsets `floor(m/2)`, or 0 to skip that axis. */
};

/**
 * @brief Batch of volumes from centred to natural (FFT) order.
 *
 * Grid: `(m0*m1*m2, batch)`, one thread per element. Port of
 * `mr_k_to_natural`.
 */
kernel void mr_k_to_natural(device float2       *dst [[buffer(0)]],
                            device const float2 *src [[buffer(1)]],
                            constant mr_reorder &r   [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]])
{
    const uint vol = (uint)r.m[0] * (uint)r.m[1] * (uint)r.m[2];
    if (gid.x >= vol) return;
    const uint3 jc = mr_decompose3(gid.x, (uint)r.m[0], (uint)r.m[1]);
    ulong src_idx = 0, stride = 1;
    for (int a = 0; a < 3; ++a) {
        int k = (int)jc[a] + r.s[a];
        if (k >= r.m[a]) k -= r.m[a];
        src_idx += (ulong)k * stride;
        stride *= (ulong)r.m[a];
    }
    const ulong base = (ulong)gid.y * vol;
    dst[base + gid.x] = src[base + src_idx];
}

/**
 * @brief Batch of volumes from natural to centred order, scaled.
 *
 * Grid: `(m0*m1*m2, batch)`. Port of `mr_k_to_centered_scale`.
 */
kernel void mr_k_to_centered_scale(device float2       *dst   [[buffer(0)]],
                                   device const float2 *src   [[buffer(1)]],
                                   constant mr_reorder &r     [[buffer(2)]],
                                   constant float      &scale [[buffer(3)]],
                                   uint2 gid [[thread_position_in_grid]])
{
    const uint vol = (uint)r.m[0] * (uint)r.m[1] * (uint)r.m[2];
    if (gid.x >= vol) return;
    const uint3 kc = mr_decompose3(gid.x, (uint)r.m[0], (uint)r.m[1]);
    ulong src_idx = 0, stride = 1;
    for (int a = 0; a < 3; ++a) {
        int j = (int)kc[a] - r.s[a];
        if (j < 0) j += r.m[a];
        src_idx += (ulong)j * stride;
        stride *= (ulong)r.m[a];
    }
    const ulong base = (ulong)gid.y * vol;
    dst[base + gid.x] = src[base + src_idx] * scale;
}

/**
 * @brief Gather the masked locations of a natural-order batch, scaled, into
 *        columns `[t0, t0 + batch)` of the `[num_masked, ntri]` output.
 *
 * Grid: `(num_masked, batch)`. Port of `mr_k_gather_masked_scale`.
 */
kernel void mr_k_gather_masked_scale(device float2       *dst        [[buffer(0)]],
                                     device const float2 *src        [[buffer(1)]],
                                     device const long   *mask       [[buffer(2)]],
                                     constant long       &num_masked [[buffer(3)]],
                                     constant mr_reorder &r          [[buffer(4)]],
                                     constant long       &t0         [[buffer(5)]],
                                     constant float      &scale      [[buffer(6)]],
                                     uint2 gid [[thread_position_in_grid]])
{
    if ((long)gid.x >= num_masked) return;
    const uint3 kc = mr_decompose3((uint)mask[gid.x], (uint)r.m[0], (uint)r.m[1]);
    ulong src_idx = 0, stride = 1;
    for (int a = 0; a < 3; ++a) {
        int j = (int)kc[a] - r.s[a];
        if (j < 0) j += r.m[a];
        src_idx += (ulong)j * stride;
        stride *= (ulong)r.m[a];
    }
    const ulong vol = stride;
    dst[(ulong)gid.x + (ulong)num_masked * ((ulong)t0 + gid.y)] =
        src[(ulong)gid.y * vol + src_idx] * scale;
}

/* ------------------------------------------------------------------------ */
/* calc_mask kernels                                                        */
/* ------------------------------------------------------------------------ */

/**
 * @brief Parameters of ::mr_k_mark_samples.
 */
struct mr_mark_params {
    long  n_samples; /**< `Ntrajs * Nkx`.                               */
    int   ndim;      /**< Trajectory coordinates (2 or 3).              */
    int   m[3];      /**< Grid extents (unused axes are 1).             */
    float sc[3];     /**< Per-axis scale `M_a / (2*pi)`.                */
};

/**
 * @brief Mark the upsampled grid point nearest to every sample.
 *
 * Sample @f$\mathbf{k}@f$ lands on `rint(k_a * M_a / (2*pi)) + M_a/2` along
 * axis `a`, wrapped periodically; that location is set to 1 (concurrent
 * writes of the same value are benign). Same arithmetic as the CUDA
 * `mr_k_mark_samples` (`rint` rounds half to even, like `rintf`), so the
 * neighbourhood mask is bit-identical across backends.
 */
kernel void mr_k_mark_samples(device float             *marks [[buffer(0)]],
                              device const float       *traj  [[buffer(1)]],
                              constant mr_mark_params  &p     [[buffer(2)]],
                              uint tid [[thread_position_in_grid]])
{
    if ((long)tid >= p.n_samples) return;
    long idx = 0, stride = 1;
    for (int a = 0; a < p.ndim; ++a) {
        long i = (long)rint(traj[(ulong)tid + (ulong)p.n_samples * a] * p.sc[a]) +
                 p.m[a] / 2;
        i %= p.m[a];
        if (i < 0) i += p.m[a];
        idx += i * stride;
        stride *= p.m[a];
    }
    marks[idx] = 1.0f;
}

/**
 * @brief One axis of a separable, periodic max-filter (grey-scale dilation).
 *
 * Port of `mr_k_dilate_axis`: each location receives the maximum of the
 * input up to @p r cells away along one axis, wrapping periodically.
 */
kernel void mr_k_dilate_axis(device float       *out    [[buffer(0)]],
                             device const float *in     [[buffer(1)]],
                             constant long      &n      [[buffer(2)]],
                             constant int       &m      [[buffer(3)]],
                             constant long      &stride [[buffer(4)]],
                             constant int       &r      [[buffer(5)]],
                             uint tid [[thread_position_in_grid]])
{
    if ((long)tid >= n) return;
    const int  i    = (int)(((long)tid / stride) % m);
    const long base = (long)tid - (long)i * stride;
    float v = in[tid];
    for (int d = -r; d <= r; ++d) {
        int j = (i + d) % m;
        if (j < 0) j += m;
        v = fmax(v, in[base + (long)j * stride]);
    }
    out[tid] = v;
}
