/**
 * @file mr_kernels_bd.cu
 * @brief Block-diagonal kernels specialised on the rank @f$L@f$.
 *
 * The generic kernels ::mr_k_blockdiag_dense / ::mr_k_blockdiag_masked
 * (`mr_kernels.cu`) handle any rank, but
 *  - stage each thread's rank vector in shared memory (the rank is only known
 *    at run time), which limits the block size and the occupancy for large
 *    @f$L@f$;
 *  - evaluate the product one output component at a time, so every
 *    off-diagonal stored element of `phi2_toep` is fetched twice (once as @f$A_{ij}@f$, once as
 *    @f$\overline{A_{ji}}@f$);
 *  - process one coil per launch, so `phi2_toep` (by far the largest array of
 *    the operator) is read from memory once per coil.
 *
 * The kernels in this file are templates on @f$L@f$ (instantiated for
 * @f$L = 1 \ldots@f$ ::MR_BD_MAX_L) that fix all three:
 *  - the rank vectors and accumulators live in registers and every loop is
 *    fully unrolled;
 *  - the stored triangle is walked once in packed order and each element
 *    contributes to two outputs (::mr_block_apply);
 *  - in masked mode the multiplication is done in place on the padded grid
 *    (through a grid-to-mask index map), without the `[num_masked, L]`
 *    compact buffer, the memset and the scatter pass of the generic path;
 *  - a launch serves a *group* of coils whose padded grids follow one another
 *    in memory. The coils are processed two at a time by the same thread, so
 *    each loaded element is applied to both from registers, and the elements
 *    loaded for one pair are still in L2 for the next: `phi2_toep` is read
 *    from DRAM about once per group instead of once per coil.
 *
 * These are ports of the Metal kernels `mr_k_bd_dense<L>` / `mr_k_bd_masked<L>`
 * (`src/metal/mr_kernels.metal`) and produce the same results as the generic
 * kernels up to the summation order (single-precision rounding).
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_kernels.cuh"

#include <cstdlib>

/* ------------------------------------------------------------------------ */
/* Device helpers                                                           */
/* ------------------------------------------------------------------------ */

/** @brief @f$\overline{a}\,b@f$ for single-precision complex numbers. */
__device__ __forceinline__ mr_complex mr_cmul_conj(mr_complex a, mr_complex b)
{
    return make_cuFloatComplex(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);
}

/**
 * @brief @p NC row vectors times a Hermitian block, reading each stored
 *        element once.
 *
 * Gram-block convention: each rank vector is a row vector multiplied from
 * the right, @f$y = v A@f$, i.e. @f$y_c = \sum_r v_r A_{rc}@f$. Walks the
 * stored triangle in packed order (see ::mr_triangulation): packed component
 * `t` is entry `(r, c)` of the block. It contributes @f$v_r A_{rc}@f$ to
 * @f$y_c@f$ and, off the diagonal, @f$v_c A_{cr} = v_c \overline{A_{rc}}@f$ to
 * @f$y_r@f$ (Hermitian symmetry), for each of the @p NC vectors. The
 * triangle orientation is a run-time (but warp-uniform) flag; both walks are
 * fully unrolled so that every register index is a compile-time constant (a
 * run-time index would demote the arrays to local memory).
 *
 * @tparam L        Rank.
 * @tparam NC       Number of vectors (coils) sharing each loaded element, 1 or 2.
 * @param  v        Input vectors, `v[k][l]` = rank @p l of vector @p k.
 * @param  y        Output vectors (overwritten), same layout.
 * @param  base     Packed component 0 of this location.
 * @param  stride   Element stride between consecutive packed components.
 * @param  upper    Non-zero when the upper triangle is stored.
 */
template <int L, int NC>
__device__ __forceinline__ void mr_block_apply(const mr_complex (&v)[NC][L],
                                               mr_complex (&y)[NC][L],
                                               const mr_complex *base,
                                               long long stride, int upper)
{
#pragma unroll
    for (int k = 0; k < NC; ++k)
#pragma unroll
        for (int i = 0; i < L; ++i) y[k][i] = make_cuFloatComplex(0.0f, 0.0f);

    /* Register pressure. Both walks below visit the stored elements in
     * packed order (offset t = 0, 1, 2, ...), so the element pointer simply
     * advances by `stride`. Left to itself, the compiler computes the
     * addresses of, and issues the loads for, many elements ahead of their
     * use (one 64-bit address and one complex value each) and spills from
     * L = 11 on (sm_75); `__ldg` or `__restrict__` on the kernel array make it
     * worse, since invariant loads may be moved freely. An empty
     * `asm volatile` with a memory clobber at the start of every column stops
     * loads from moving across columns; within a column (at most L elements)
     * they still overlap. With -Xptxas -v: no spills for coil pairs up to
     * L = 16 on sm_75 and sm_90 (except L = 14 dense on sm_75, 84 bytes). */
    const mr_complex *bp = base;

    /* One stored element: its (r, c) position is a compile-time constant
     * once the loops below are unrolled. */
#define MR_BD_ELEMENT(r, c)                                                    \
    do {                                                                       \
        const mr_complex a = *bp;                                              \
        bp += stride;                                                          \
        _Pragma("unroll") for (int k = 0; k < NC; ++k) {                       \
            y[k][c] = cuCaddf(y[k][c], cuCmulf(a, v[k][r]));                   \
            if ((r) != (c)) y[k][r] = cuCaddf(y[k][r], mr_cmul_conj(a, v[k][c])); \
        }                                                                      \
    } while (0)

    if (upper) {
        /* Upper: column c holds rows 0..c (packed offset c(c+1)/2 + r). */
#pragma unroll
        for (int c = 0; c < L; ++c) {
            asm volatile("" ::: "memory");
#pragma unroll
            for (int r = 0; r <= c; ++r) MR_BD_ELEMENT(r, c);
        }
    } else {
        /* Lower: column c holds rows c..L-1 (offset cL - c(c-1)/2 + r - c). */
#pragma unroll
        for (int c = 0; c < L; ++c) {
            asm volatile("" ::: "memory");
#pragma unroll
            for (int r = c; r < L; ++r) MR_BD_ELEMENT(r, c);
        }
    }
#undef MR_BD_ELEMENT
}

/* ------------------------------------------------------------------------ */
/* Kernels                                                                  */
/* ------------------------------------------------------------------------ */

/**
 * @brief In-place dense block-diagonal multiplication for a group of coils,
 *        specialised on @p L.
 *
 * Same contract as ::mr_k_blockdiag_dense, without shared memory, applied to
 * the @p ncoil coils whose `[Mx, My, Mz, L]` grids follow one another in
 * @p d_pad (`L * Mx*My*Mz` elements apart). One thread per k-space location
 * processes the coils in pairs (one `phi2_toep` read per pair) when @p PAIR is
 * set, else one at a time (used for large @p L, where two coils' vectors would
 * not fit the register file).
 *
 * The kernel can also be restricted to a contiguous range of grid locations
 * (@p rg), with the data of that range stored compactly: this is how the
 * multi-GPU plan applies the block-diagonal stage to its slab of k-space when
 * a coil is split across devices by rank (`mr_op.cu`).
 *
 * @tparam L     Rank.
 * @tparam PAIR  Process two coils per triangle walk.
 * @param[in,out] d_pad       `[nloc, L, ncoil]` values with rank stride
 *                            `rg.vstride` (`[Mx, My, Mz, L, ncoil]`, natural
 *                            order, for the whole grid).
 * @param[in]     d_phi2_toep `[Mx, My, Mz, (L^2+L)/2]`, centred order.
 * @param[in]     dims        Geometry descriptor (`dims.L == L`).
 * @param[in]     ncoil       Number of coils in the group.
 * @param[in]     rg          Locations `[loc0, loc0 + nloc)` (natural-order
 *                            linear indices) and the rank stride of
 *                            @p d_pad.
 */
template <int L, bool PAIR>
__global__ void __launch_bounds__(MR_BD_THREADS)
mr_k_bd_dense_t(mr_complex *__restrict__ d_pad,
                const mr_complex *d_phi2_toep, mr_dims dims,
                int ncoil, mr_bd_range rg)
{
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= rg.nloc) return;
    const long long loc = rg.loc0 + tid;

    /* Natural (FFT-order) grid index -> centred kernel index. */
    const int j0 = (int)(loc % dims.m[0]);
    const long long q = loc / dims.m[0];
    const int j1 = (int)(q % dims.m[1]);
    const int j2 = (int)(q / dims.m[1]);
    const long long pc =
        mr_nat2cent(j0, dims.m[0], dims.c_m[0]) +
        (long long)dims.m[0] * (mr_nat2cent(j1, dims.m[1], dims.c_m[1]) +
                                (long long)dims.m[1] *
                                    mr_nat2cent(j2, dims.m[2], dims.c_m[2]));
    /* Data stride (the range's) vs kernel stride (always the whole grid). */
    const long long padn = rg.vstride;
    const long long slot = padn * L;
    const mr_complex *base = d_phi2_toep + pc;

    int s = 0;
    if (PAIR) {
        for (; s + 1 < ncoil; s += 2) {      /* coil pairs: one toep read */
            mr_complex *g = d_pad + slot * s + tid;
            mr_complex v[2][L], y[2][L];
#pragma unroll
            for (int l = 0; l < L; ++l) {
                v[0][l] = g[padn * l];
                v[1][l] = g[slot + padn * l];
            }
            mr_block_apply<L, 2>(v, y, base, dims.pad_size, dims.tri_upper);
#pragma unroll
            for (int l = 0; l < L; ++l) {
                g[padn * l] = y[0][l];
                g[slot + padn * l] = y[1][l];
            }
        }
    }
    for (; s < ncoil; ++s) {                 /* odd coil out, or all coils */
        mr_complex *g = d_pad + slot * s + tid;
        mr_complex v[1][L], y[1][L];
#pragma unroll
        for (int l = 0; l < L; ++l) v[0][l] = g[padn * l];
        mr_block_apply<L, 1>(v, y, base, dims.pad_size, dims.tri_upper);
#pragma unroll
        for (int l = 0; l < L; ++l) g[padn * l] = y[0][l];
    }
}

/**
 * @brief In-place masked block-diagonal multiplication for a group of coils,
 *        specialised on @p L.
 *
 * Same result as ::mr_k_blockdiag_masked followed by the memset and
 * ::mr_k_scatter_masked, in a single in-place pass without the
 * `[num_masked, L]` compact buffer: one thread per natural-order grid location
 * looks up its position in the mask through @p d_inv (built by
 * ::mr_k_mask_inverse). A location outside the mask is set to zero for every
 * rank and coil; a location inside is multiplied by its packed block, which
 * sits at column `d_inv[tid]` of the `[num_masked, (L^2+L)/2]` kernel. For a
 * sorted mask, consecutive threads mostly read consecutive kernel columns, so
 * the kernel reads stay coalesced. Coils are processed as in
 * ::mr_k_bd_dense_t.
 *
 * Like ::mr_k_bd_dense_t, the kernel can be restricted to a range of grid
 * locations (@p rg).
 *
 * @tparam L     Rank.
 * @tparam PAIR  Process two coils per triangle walk.
 * @param[in,out] d_pad       `[nloc, L, ncoil]` values with rank stride
 *                            `rg.vstride` (`[Mx, My, Mz, L, ncoil]`, natural
 *                            order, for the whole grid).
 * @param[in]     d_phi2_toep `[num_masked, (L^2+L)/2]`.
 * @param[in]     d_inv       `[Mx, My, Mz]`, natural order: mask position of
 *                            each location, or -1 outside the mask.
 * @param[in]     num_masked  Number of retained locations.
 * @param[in]     dims        Geometry descriptor (`dims.L == L`).
 * @param[in]     ncoil       Number of coils in the group.
 * @param[in]     rg          Location range and rank stride, see
 *                            ::mr_k_bd_dense_t.
 */
template <int L, bool PAIR>
__global__ void __launch_bounds__(MR_BD_THREADS)
mr_k_bd_masked_t(mr_complex *__restrict__ d_pad,
                 const mr_complex *d_phi2_toep,
                 const int *__restrict__ d_inv, long long num_masked,
                 mr_dims dims, int ncoil, mr_bd_range rg)
{
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= rg.nloc) return;

    const long long padn = rg.vstride, nm = num_masked;
    const long long slot = padn * L;
    const int m = d_inv[rg.loc0 + tid];

    if (m < 0) {                             /* outside the mask: zero */
        const mr_complex z = make_cuFloatComplex(0.0f, 0.0f);
        for (int s = 0; s < ncoil; ++s)
#pragma unroll
            for (int l = 0; l < L; ++l) d_pad[slot * s + padn * l + tid] = z;
        return;
    }
    const mr_complex *base = d_phi2_toep + m;

    int s = 0;
    if (PAIR) {
        for (; s + 1 < ncoil; s += 2) {      /* coil pairs: one toep read */
            mr_complex *g = d_pad + slot * s + tid;
            mr_complex v[2][L], y[2][L];
#pragma unroll
            for (int l = 0; l < L; ++l) {
                v[0][l] = g[padn * l];
                v[1][l] = g[slot + padn * l];
            }
            mr_block_apply<L, 2>(v, y, base, nm, dims.tri_upper);
#pragma unroll
            for (int l = 0; l < L; ++l) {
                g[padn * l] = y[0][l];
                g[slot + padn * l] = y[1][l];
            }
        }
    }
    for (; s < ncoil; ++s) {                 /* odd coil out, or all coils */
        mr_complex *g = d_pad + slot * s + tid;
        mr_complex v[1][L], y[1][L];
#pragma unroll
        for (int l = 0; l < L; ++l) v[0][l] = g[padn * l];
        mr_block_apply<L, 1>(v, y, base, nm, dims.tri_upper);
#pragma unroll
        for (int l = 0; l < L; ++l) g[padn * l] = y[0][l];
    }
}

/* ------------------------------------------------------------------------ */
/* Host-side lookup                                                         */
/* ------------------------------------------------------------------------ */

/**
 * @brief Whether rank @p L uses coil pairs.
 *
 * Two coils' input and output vectors take `8 L` 32-bit registers; above
 * ::MR_BD_PAIR_MAX_L that no longer fits next to the addressing within the
 * 255-register limit, and the kernels would spill to local memory. Larger
 * ranks therefore walk the triangle once per coil (the group still shares the
 * L2-resident `phi2_toep` values between consecutive coils).
 */
#define MR_BD_PAIR(L) ((L) <= MR_BD_PAIR_MAX_L)

/** @brief One `case` of ::mr_bd_lookup. */
#define MR_BD_CASE(L)                                                          \
    case L:                                                                    \
        k.dense  = mr_k_bd_dense_t<L, MR_BD_PAIR(L)>;                          \
        k.masked = mr_k_bd_masked_t<L, MR_BD_PAIR(L)>;                         \
        k.pair   = MR_BD_PAIR(L);                                              \
        break;

mr_bd_kernels mr_bd_lookup(int L)
{
    mr_bd_kernels k = {nullptr, nullptr, 0, 0};
    const char *e = std::getenv("MR_AHA_CUDA_TILED_BD");
    const bool force_tiled = e && e[0] && atoi(e) != 0;
    if (!force_tiled) switch (L) {
        MR_BD_CASE(1)  MR_BD_CASE(2)  MR_BD_CASE(3)  MR_BD_CASE(4)
        MR_BD_CASE(5)  MR_BD_CASE(6)  MR_BD_CASE(7)  MR_BD_CASE(8)
        MR_BD_CASE(9)  MR_BD_CASE(10) MR_BD_CASE(11) MR_BD_CASE(12)
        MR_BD_CASE(13) MR_BD_CASE(14) MR_BD_CASE(15) MR_BD_CASE(16)
        MR_BD_CASE(17) MR_BD_CASE(18) MR_BD_CASE(19) MR_BD_CASE(20)
        MR_BD_CASE(21) MR_BD_CASE(22) MR_BD_CASE(23) MR_BD_CASE(24)
        MR_BD_CASE(25) MR_BD_CASE(26) MR_BD_CASE(27) MR_BD_CASE(28)
        MR_BD_CASE(29) MR_BD_CASE(30)
        /* Keep in sync with MR_BD_MAX_L. */
        default: break;
    }
    /* Tiled kernels (mr_kernels_bdt.cu) for the other ranks. */
    if (k.dense == nullptr && L >= 1 && L <= MR_BDT_MAX_L) k.tiled = 1;
    return k;
}
