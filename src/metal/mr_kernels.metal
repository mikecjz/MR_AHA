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
    int  c_n[3];     /**< Image center offsets `floor(N_d/2)`.             */
    int  c_m[3];     /**< Upsampled center offsets `floor(M_d/2)`.         */
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

/** @brief Centered index -> natural (FFT-order) index, see `mr_cent2nat`. */
static inline int mr_cent2nat(int k, int M, int c)
{
    const int j = k - c;
    return (j < 0) ? (j + M) : j;
}

/** @brief Natural (FFT-order) index -> centered index, see `mr_nat2cent`. */
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
 * @brief Component @p i of the staged rank row vector @p v times the
 *        Hermitian block (Gram-block convention `y = v A`), i.e.
 *        `y_i = sum_j v_j A_ji`.
 *
 * @param v         Staged input vector of length @p L (threadgroup memory).
 * @param base      Packed component 0 of this location.
 * @param stride    Element stride between consecutive packed components.
 * @param L         Rank.
 * @param tri_upper Non-zero when the upper triangle is stored.
 * @param i         Output component (column of the block).
 */
static inline float2 mr_apply_block_row(threadgroup const float2 *v,
                                        device const float2 *base, ulong stride,
                                        int L, int tri_upper, int i)
{
    float2 acc = float2(0.0f);
    for (int j = 0; j < L; ++j) {
        bool conj;
        /* Element A_{ji}: row j, column i (v * A). */
        const int t = mr_tri_offset(j, i, L, tri_upper, conj);
        const float2 a = base[(ulong)t * stride];
        acc += conj ? mr_cmul_conj(a, v[j]) : mr_cmul(a, v[j]);
    }
    return acc;
}

/** @brief Natural-order linear index of a centered linear mask index. */
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
 * per thread, length set by the host) before overwriting it component by
 * component with `v A` (Gram-block convention).
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

    /* The working buffer is in natural FFT order; phi2_toep is centered. */
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
/* Block-diagonal kernels specialized on L                                  */
/* ------------------------------------------------------------------------ */

/**
 * @brief Row vector times Hermitian block, reading each stored element once.
 *
 * Gram-block convention: the rank vector is a row vector multiplied from
 * the right, `y = v A`, i.e. `y_c = sum_r v_r A_rc`. Walks the stored
 * triangle in packed order (::mr_triangulation): element `t` is entry
 * `(r, c)` of the block, and it contributes `v_r A_rc` to `y_c` and, off the
 * diagonal, `v_c A_cr = v_c conj(A_rc)` to `y_r` (Hermitian symmetry). The
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
            y[c] += mr_cmul(a, v[r]);
            if (r != c) y[r] += mr_cmul_conj(a, v[c]);
        }
    }
}

/**
 * @brief ::mr_block_apply for two coils at once: every stored element is
 *        loaded once and applied to both coils' vectors, so a coil pair costs
 *        one read of `phi2_toep` from memory (or cache) instead of two.
 */
template <int L>
static inline void mr_block_apply2(thread const float2 (&v0)[L],
                                   thread const float2 (&v1)[L],
                                   thread float2 (&y0)[L], thread float2 (&y1)[L],
                                   device const float2 *base, ulong stride,
                                   bool upper)
{
    for (int i = 0; i < L; ++i) { y0[i] = float2(0.0f); y1[i] = float2(0.0f); }
    ulong off = 0;
    for (int c = 0; c < L; ++c) {
        const int r0 = upper ? 0 : c;
        const int r1 = upper ? c : L - 1;
        for (int r = r0; r <= r1; ++r, off += stride) {
            const float2 a = base[off];
            y0[c] += mr_cmul(a, v0[r]);
            y1[c] += mr_cmul(a, v1[r]);
            if (r != c) {
                y0[r] += mr_cmul_conj(a, v0[c]);
                y1[r] += mr_cmul_conj(a, v1[c]);
            }
        }
    }
}

/**
 * @brief In-place block-diagonal multiplication over the whole grid,
 *        specialized on L (host names `mr_k_bd_dense_<L>`, L = 1..5).
 *
 * Same contract as ::mr_k_blockdiag_dense, without threadgroup memory, and
 * applied to @p ncoil coils whose `[Mx, My, Mz, L]` grids follow one another
 * in @p pad (`slot` = `L * Mx*My*Mz` elements apart). The coils of a group
 * are processed by the same thread, two at a time (::mr_block_apply2 loads
 * each `phi2_toep` value once for a coil pair), and the values it reads for
 * one pair are still in the GPU's caches for the next: `phi2_toep`, the
 * dominant memory traffic, is fetched from DRAM about once per group instead
 * of once per coil.
 */
template <int L>
kernel void mr_k_bd_dense(device float2       *pad   [[buffer(0)]],
                          device const float2 *toep  [[buffer(1)]],
                          constant mr_dims    &d     [[buffer(2)]],
                          constant int        &ncoil [[buffer(3)]],
                          uint tid [[thread_position_in_grid]])
{
    if ((long)tid >= d.pad_size) return;
    const uint3 jc = mr_decompose3(tid, (uint)d.m[0], (uint)d.m[1]);
    const ulong k0 = (ulong)mr_nat2cent((int)jc.x, d.m[0], d.c_m[0]);
    const ulong k1 = (ulong)mr_nat2cent((int)jc.y, d.m[1], d.c_m[1]);
    const ulong k2 = (ulong)mr_nat2cent((int)jc.z, d.m[2], d.c_m[2]);
    const ulong pc = k0 + (ulong)d.m[0] * (k1 + (ulong)d.m[1] * k2);
    const ulong padn = (ulong)d.pad_size;
    const ulong slot = padn * (ulong)L;

    int s = 0;
    for (; s + 1 < ncoil; s += 2) {          /* coil pairs: one toep read */
        device float2 *g0 = pad + slot * (ulong)s + (ulong)tid;
        device float2 *g1 = g0 + slot;
        float2 v0[L], v1[L], y0[L], y1[L];
        for (int l = 0; l < L; ++l) { v0[l] = g0[padn * l]; v1[l] = g1[padn * l]; }
        mr_block_apply2<L>(v0, v1, y0, y1, toep + pc, padn, d.tri_upper != 0);
        for (int l = 0; l < L; ++l) { g0[padn * l] = y0[l]; g1[padn * l] = y1[l]; }
    }
    if (s < ncoil) {                          /* odd coil out */
        device float2 *g = pad + slot * (ulong)s + (ulong)tid;
        float2 v[L], y[L];
        for (int l = 0; l < L; ++l) v[l] = g[padn * l];
        mr_block_apply<L>(v, y, toep + pc, padn, d.tri_upper != 0);
        for (int l = 0; l < L; ++l) g[padn * l] = y[l];
    }
}

/**
 * @brief Masked block-diagonal multiplication specialized on L (host names
 *        `mr_k_bd_masked_<L>`). Same contract as ::mr_k_blockdiag_masked,
 *        for a group of @p ncoil coils as in ::mr_k_bd_dense (`[num_masked,
 *        L]` compact slots and `[Mx, My, Mz, L]` grid slots, one per coil).
 */
template <int L>
kernel void mr_k_bd_masked(device float2       *compact    [[buffer(0)]],
                           device const float2 *pad        [[buffer(1)]],
                           device const float2 *toep       [[buffer(2)]],
                           device const long   *mask       [[buffer(3)]],
                           constant long       &num_masked [[buffer(4)]],
                           constant mr_dims    &d          [[buffer(5)]],
                           constant int        &ncoil      [[buffer(6)]],
                           uint tid [[thread_position_in_grid]])
{
    if ((long)tid >= num_masked) return;
    const ulong jj = mr_mask_to_natural(mask[tid], d);
    const ulong padn = (ulong)d.pad_size, nm = (ulong)num_masked;

    const ulong gslot = padn * (ulong)L, cslot = nm * (ulong)L;
    int s = 0;
    for (; s + 1 < ncoil; s += 2) {          /* coil pairs: one toep read */
        device const float2 *g0 = pad + gslot * (ulong)s + jj;
        device const float2 *g1 = g0 + gslot;
        device float2 *o0 = compact + cslot * (ulong)s + (ulong)tid;
        device float2 *o1 = o0 + cslot;
        float2 v0[L], v1[L], y0[L], y1[L];
        for (int l = 0; l < L; ++l) { v0[l] = g0[padn * l]; v1[l] = g1[padn * l]; }
        mr_block_apply2<L>(v0, v1, y0, y1, toep + tid, nm, d.tri_upper != 0);
        for (int l = 0; l < L; ++l) { o0[nm * l] = y0[l]; o1[nm * l] = y1[l]; }
    }
    if (s < ncoil) {                          /* odd coil out */
        device const float2 *g = pad + gslot * (ulong)s + jj;
        device float2 *o = compact + cslot * (ulong)s + (ulong)tid;
        float2 v[L], y[L];
        for (int l = 0; l < L; ++l) v[l] = g[padn * l];
        mr_block_apply<L>(v, y, toep + tid, nm, d.tri_upper != 0);
        for (int l = 0; l < L; ++l) o[nm * l] = y[l];
    }
}

/* Explicit instantiations, L = 1..MR_METAL_BD_MAX_L (mr_aha_metal.cpp). Larger ranks use the tiled kernels
 * (mr_k_bdt) below. */
#define MR_BD_INSTANTIATE(L)                                                     \
    template [[host_name("mr_k_bd_dense_" #L)]] [[kernel]] void                 \
    mr_k_bd_dense<L>(device float2 *pad [[buffer(0)]],                           \
                     device const float2 *toep [[buffer(1)]],                    \
                     constant mr_dims &d [[buffer(2)]],                          \
                     constant int &ncoil [[buffer(3)]],                          \
                     uint tid [[thread_position_in_grid]]);                      \
    template [[host_name("mr_k_bd_masked_" #L)]] [[kernel]] void                \
    mr_k_bd_masked<L>(device float2 *compact [[buffer(0)]],                      \
                      device const float2 *pad [[buffer(1)]],                    \
                      device const float2 *toep [[buffer(2)]],                   \
                      device const long *mask [[buffer(3)]],                     \
                      constant long &num_masked [[buffer(4)]],                   \
                      constant mr_dims &d [[buffer(5)]],                         \
                      constant int &ncoil [[buffer(6)]],                         \
                      uint tid [[thread_position_in_grid]]);
MR_BD_INSTANTIATE(1)  MR_BD_INSTANTIATE(2)  MR_BD_INSTANTIATE(3)  MR_BD_INSTANTIATE(4)
MR_BD_INSTANTIATE(5)
#undef MR_BD_INSTANTIATE

/* ------------------------------------------------------------------------ */
/* Tiled block-diagonal kernels for ranks above the specialized range       */
/* ------------------------------------------------------------------------ */

/**
 * @brief Largest tile width (locations per threadgroup) of ::mr_k_bdt; sizes
 *        the static `kidx` array. The host picks P in {4, 8, 16, 32}.
 */
#define MR_BDT_MAX_P 32

/** @brief Packed offset of the first stored element of column @p c. */
static inline int mr_bdt_colstart(int c, int L, int upper)
{
    return upper ? c * (c + 1) / 2 : c * L - c * (c - 1) / 2;
}

/** @brief Number of stored elements of column @p c. */
static inline int mr_bdt_collen(int c, int L, int upper)
{
    return upper ? c + 1 : L - c;
}

/**
 * @brief Tiled in-place block-diagonal multiplication for a group of coils
 *        (host names `mr_k_bdt_{dense,masked}_nc<NC>_opt<OPT>`).
 *
 * Port of `mr_k_bdt` in `src/cuda/mr_kernels_bdt.cu`; see that file for the
 * full derivation. The specialized kernels above keep a location's rank
 * vectors in registers and stop at L = 5; the generic ones stage one rank
 * vector per thread in threadgroup memory (few threads per threadgroup at
 * large L), fetch every off-diagonal element of `phi2_toep` twice and handle
 * one coil per dispatch. Here:
 *
 *  - a threadgroup of T threads owns a tile of P consecutive grid locations
 *    and the NC coils of a batch; the inputs are staged once in threadgroup
 *    memory as `v[k][l][p]` (location fastest, so the device loads are
 *    contiguous);
 *  - the stored triangle is walked in panels of whole stored columns (a
 *    contiguous range of packed offsets), each copied once from device
 *    memory into threadgroup memory, `tile[e][p]`, and used for all NC
 *    coils: `phi2_toep` is read from DRAM once per coil batch instead of
 *    twice per coil;
 *  - thread (g, p) = (lid / P, lid % P) accumulates the outputs
 *    `o = g, g + G, ...` (G = T / P) of location p in registers, for all
 *    coils of the batch, and writes them back in place.
 *
 * Masked operator: @p inv maps every grid location to its column of the
 * masked `phi2_toep` (-1 outside the mask), so the kernel works in place on
 * the grid like the dense one and writes zeros outside the mask: no compact
 * buffer, zeroing pass or scatter. A tile with no location in the mask only
 * writes zeros and reads nothing else.
 *
 * Differences from the CUDA kernel: no `cp.async` on Apple GPUs, so the
 * panel is fetched with ordinary loads issued four at a time per thread
 * (independent loads in flight before the threadgroup stores); and the
 * panel / tile sizes come from the 32 KB threadgroup-memory limit
 * (host `mr_bdt_pick` in mr_aha_metal.cpp).
 *
 * @param pad        `[pad_size, L, ncoil]` grids, transformed in place.
 * @param toep       `[Mx, My, Mz, (L^2+L)/2]` (centered) or
 *                   `[num_masked, (L^2+L)/2]` (masked).
 * @param inv        Masked only: grid-to-mask map (-1 outside the mask).
 * @param num_masked Masked only: number of retained locations.
 * @param geo        `(P, E)`: locations per tile and panel capacity in
 *                   stored elements (`E >= L`).
 * @param sh         Threadgroup memory: `v[NC][L][P]` then `tile[E][P]`.
 */
template <int NC, int OPT, bool MASKED>
kernel void mr_k_bdt(device float2       *pad        [[buffer(0)]],
                     device const float2 *toep       [[buffer(1)]],
                     device const int    *inv        [[buffer(2)]],
                     constant long       &num_masked [[buffer(3)]],
                     constant mr_dims    &d          [[buffer(4)]],
                     constant int        &ncoil      [[buffer(5)]],
                     constant int2       &geo        [[buffer(6)]],
                     threadgroup float2  *sh         [[threadgroup(0)]],
                     uint tg  [[threadgroup_position_in_grid]],
                     uint lid [[thread_index_in_threadgroup]],
                     uint T   [[threads_per_threadgroup]])
{
    threadgroup long kidx[MR_BDT_MAX_P];
    const int L = d.L;
    const int upper = d.tri_upper;
    const int P = geo.x;
    const int E = geo.y;
    const int G = (int)T / P;
    const int p = (int)lid % P;
    const int g = (int)lid / P;
    threadgroup float2 *v = sh;
    threadgroup float2 *tile = sh + NC * L * P;

    const long nloc = d.pad_size;
    const long first = (long)tg * P;

    /* 0. Where each location's block lives in toep: centered grid index
     * (dense) or mask column (masked); -1 = no work. */
    if ((int)lid < P) {
        const long i = first + (long)lid;
        long k = -1;
        if (i < nloc) {
            if (MASKED) {
                k = (long)inv[i];
            } else {
                const uint3 jc = mr_decompose3((uint)i, (uint)d.m[0], (uint)d.m[1]);
                const long k0 = mr_nat2cent((int)jc.x, d.m[0], d.c_m[0]);
                const long k1 = mr_nat2cent((int)jc.y, d.m[1], d.c_m[1]);
                const long k2 = mr_nat2cent((int)jc.z, d.m[2], d.c_m[2]);
                k = k0 + (long)d.m[0] * (k1 + (long)d.m[1] * k2);
            }
        }
        kidx[lid] = k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const long tid = first + p;
    const bool inrange = tid < nloc;
    const bool active = kidx[p] >= 0;
    const ulong padn = (ulong)d.pad_size;
    const ulong slot = padn * (ulong)L;
    const ulong kstride = MASKED ? (ulong)num_masked : padn;

    /* Masked tile entirely outside the mask: write its zeros and leave. The
     * test reads the same kidx values in every thread, so the whole
     * threadgroup leaves together and no barrier below is skipped by some
     * threads only. */
    if (MASKED) {
        bool has_work = false;
        for (int q = 0; q < P; ++q) has_work = has_work || (kidx[q] >= 0);
        if (!has_work) {
            if (inrange)
                for (int k = 0; k < ncoil; ++k)
                    for (int o = g; o < L; o += G)
                        pad[slot * (ulong)k + padn * (ulong)o + (ulong)tid] = float2(0.0f);
            return;
        }
    }

    for (int s0 = 0; s0 < ncoil; s0 += NC) {
        const int nb = min(ncoil - s0, NC);

        float2 y[NC][OPT];
#pragma unroll
        for (int k = 0; k < NC; ++k)
#pragma unroll
            for (int j = 0; j < OPT; ++j) y[k][j] = float2(0.0f);

        /* 1. Stage the inputs, idx = pp + P * (l + L * k); missing coils and
         * inactive locations as zeros. The barrier keeps the previous batch's
         * readers of v ahead of these writes. */
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = (int)lid; idx < NC * L * P; idx += (int)T) {
            const int pp = idx % P;
            const int kl = idx / P;
            const int k = kl / L, l = kl % L;
            float2 a = float2(0.0f);
            if (k < nb && kidx[pp] >= 0)
                a = pad[slot * (ulong)(s0 + k) + padn * (ulong)l + (ulong)(first + pp)];
            v[idx] = a;
        }

        /* 2.-3. Panels of whole stored columns [c0, c1), packed offsets
         * [t0, t0 + cnt). */
        for (int c0 = 0, c1 = 0; c0 < L; c0 = c1) {
            int cnt = 0;
            c1 = c0;
            while (c1 < L && cnt + mr_bdt_collen(c1, L, upper) <= E) {
                cnt += mr_bdt_collen(c1, L, upper);
                ++c1;
            }
            const int t0 = mr_bdt_colstart(c0, L, upper);

            threadgroup_barrier(mem_flags::mem_threadgroup); /* tile free */
            {
                /* Four independent loads per thread in flight, then the
                 * threadgroup stores. idx = pp + P * e. */
                const int n = cnt * P;
                for (int base = (int)lid; base < n; base += 4 * (int)T) {
                    float2 a[4];
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const int idx = base + u * (int)T;
                        a[u] = float2(0.0f);
                        if (idx < n) {
                            const long kk = kidx[idx % P];
                            if (kk >= 0)
                                a[u] = toep[(ulong)kk + (ulong)(t0 + idx / P) * kstride];
                        }
                    }
#pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        const int idx = base + u * (int)T;
                        if (idx < n) tile[idx] = a[u];
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup); /* panel landed */

            if (active) {
#pragma unroll
                for (int j = 0; j < OPT; ++j) {
                    const int o = g + j * G;
                    if (o >= L) break;
                    float2 acc[NC];
#pragma unroll
                    for (int k = 0; k < NC; ++k) acc[k] = float2(0.0f);

                    /* (a) Stored column o: y_o += sum_r A_ro v_r. */
                    if (o >= c0 && o < c1) {
                        const int rmin = upper ? 0 : o;
                        const int n = mr_bdt_collen(o, L, upper);
                        threadgroup const float2 *tp =
                            tile + (mr_bdt_colstart(o, L, upper) - t0) * P + p;
                        threadgroup const float2 *vp = v + rmin * P + p;
                        for (int r = 0; r < n; ++r) {
                            const float2 a = tp[r * P];
#pragma unroll
                            for (int k = 0; k < NC; ++k)
                                acc[k] += mr_cmul(a, vp[(k * L + r) * P]);
                        }
                    }

                    /* (b) Row o of the panel's other columns, conjugated:
                     * y_o += sum_c conj(A_oc) v_c (upper: c > o, lower: c < o). */
                    const int cb = upper ? max(c0, o + 1) : c0;
                    const int ce = upper ? c1 : min(c1, o);
                    if (cb < ce) {
                        int e = mr_bdt_colstart(cb, L, upper) - t0 + (upper ? o : o - cb);
                        for (int c = cb; c < ce; ++c) {
                            const float2 a = tile[e * P + p];
#pragma unroll
                            for (int k = 0; k < NC; ++k)
                                acc[k] += mr_cmul_conj(a, v[(k * L + c) * P + p]);
                            e += upper ? c + 1 : L - c - 1;
                        }
                    }
#pragma unroll
                    for (int k = 0; k < NC; ++k) y[k][j] += acc[k];
                }
            }
        }

        /* 4. Write back in place (zero outside the mask). */
        if (inrange) {
#pragma unroll
            for (int k = 0; k < NC; ++k) {
                if (k >= nb) break;
#pragma unroll
                for (int j = 0; j < OPT; ++j) {
                    const int o = g + j * G;
                    if (o >= L) break;
                    pad[slot * (ulong)(s0 + k) + padn * (ulong)o + (ulong)tid] =
                        active ? y[k][j] : float2(0.0f);
                }
            }
        }
    }
}

/* Instances: NC in {1, 2, 4} times even OPT in 2..16, dense and masked (the
 * host picks one per plan; NC and OPT size the register accumulators). */
#define MR_BDT_INST1(NAME, NC, OPT, MASKED)                                      \
    template [[host_name(NAME)]] [[kernel]] void mr_k_bdt<NC, OPT, MASKED>(     \
        device float2 *pad [[buffer(0)]],                                         \
        device const float2 *toep [[buffer(1)]],                                  \
        device const int *inv [[buffer(2)]],                                      \
        constant long &num_masked [[buffer(3)]],                                  \
        constant mr_dims &d [[buffer(4)]],                                        \
        constant int &ncoil [[buffer(5)]],                                        \
        constant int2 &geo [[buffer(6)]],                                         \
        threadgroup float2 *sh [[threadgroup(0)]],                                \
        uint tg [[threadgroup_position_in_grid]],                                 \
        uint lid [[thread_index_in_threadgroup]],                                 \
        uint T [[threads_per_threadgroup]]);
#define MR_BDT_INST(NC, OPT)                                                     \
    MR_BDT_INST1("mr_k_bdt_dense_nc" #NC "_opt" #OPT, NC, OPT, false)           \
    MR_BDT_INST1("mr_k_bdt_masked_nc" #NC "_opt" #OPT, NC, OPT, true)
#define MR_BDT_INST_NC(NC)                                                       \
    MR_BDT_INST(NC, 2) MR_BDT_INST(NC, 4) MR_BDT_INST(NC, 6) MR_BDT_INST(NC, 8)  \
    MR_BDT_INST(NC, 10) MR_BDT_INST(NC, 12) MR_BDT_INST(NC, 14) MR_BDT_INST(NC, 16)
MR_BDT_INST_NC(1)
MR_BDT_INST_NC(2)
MR_BDT_INST_NC(4)
#undef MR_BDT_INST_NC
#undef MR_BDT_INST
#undef MR_BDT_INST1

/**
 * @brief Grid-to-mask map of the tiled masked kernel: `inv[natural(mask[i])]
 *        = i`. The host first fills @p inv with -1 (::mr_k_fill_i32). Port
 *        of `mr_k_mask_inverse`.
 */
kernel void mr_k_mask_inverse(device int        *inv        [[buffer(0)]],
                              device const long *mask       [[buffer(1)]],
                              constant long     &num_masked [[buffer(2)]],
                              constant mr_dims  &d          [[buffer(3)]],
                              uint tid [[thread_position_in_grid]])
{
    if ((long)tid >= num_masked) return;
    inv[mr_mask_to_natural(mask[tid], d)] = (int)tid;
}

/** @brief Fill @p n ints with @p value. */
kernel void mr_k_fill_i32(device int    *p     [[buffer(0)]],
                          constant long &n     [[buffer(1)]],
                          constant int  &value [[buffer(2)]],
                          uint tid [[thread_position_in_grid]])
{
    if ((long)tid >= n) return;
    p[tid] = value;
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

/* ------------------------------------------------------------------------ */
/* calc_toep kernels (stage 2: point-spread function -> phi2_toep)          */
/* ------------------------------------------------------------------------ */

/**
 * @brief Extents and per-axis centering shifts of a reorder kernel.
 *
 * A shift of 0 leaves that axis untouched, which is how the kz axis of a
 * stack-of-stars kernel is carried through without a transform.
 */
struct mr_reorder {
    int m[3]; /**< Axis extents `[m0, m1, m2]`.                         */
    int s[3]; /**< Center offsets `floor(m/2)`, or 0 to skip that axis. */
};

/**
 * @brief Batch of volumes from centered to natural (FFT) order.
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
 * @brief Batch of volumes from natural to centered order, scaled.
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
 * neighborhood mask is bit-identical across backends.
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
 * @brief One axis of a separable, periodic max-filter (gray-scale dilation).
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
