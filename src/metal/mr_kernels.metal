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
