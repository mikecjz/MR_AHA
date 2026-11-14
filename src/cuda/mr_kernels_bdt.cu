/**
 * @file mr_kernels_bdt.cu
 * @brief Tiled block-diagonal kernels for ranks without a specialised kernel.
 *
 * The specialised kernels (`mr_kernels_bd.cu`) keep a location's rank vectors
 * in registers and stop at ::MR_BD_MAX_L, where those no longer fit. Above
 * that, the generic kernels ::mr_k_blockdiag_dense / ::mr_k_blockdiag_masked
 * (`mr_kernels.cu`) used to take over. They are slow on large ranks for three
 * reasons, which the kernels in this file remove:
 *
 *  - **Occupancy.** A generic thread stages its own length-@f$L@f$ vector in
 *    shared memory (576 bytes at @f$L = 72@f$), so few threads fit on an SM:
 *    about 12 warps on an H100, about 5 on an L40S. Too few loads are in
 *    flight to use HBM bandwidth (H100, `2d72`: about 1.1 TB/s masked,
 *    0.6 TB/s dense, of 3.35 TB/s).
 *  - **Each stored element is read twice.** The generic kernels evaluate one
 *    output at a time and rebuild the full @f$L \times L@f$ block, so every
 *    off-diagonal element of `phi2_toep` comes from DRAM twice (as
 *    @f$A_{rc}@f$ and as @f$\overline{A_{rc}}@f$).
 *  - **One coil per launch.** `phi2_toep`, the largest array of the operator,
 *    is read once per coil; the masked path also needs a compact buffer, a
 *    memset and a scatter pass.
 *
 * ### Algorithm
 *
 * A thread block owns a *tile* of @f$P@f$ consecutive grid locations (32 for
 * @f$L \le 128@f$) and the @f$NC@f$ coils of a batch:
 *
 *  1. The @f$NC \times L@f$ input values of the tile are staged in shared
 *     memory, `v[k][l][p]` (location fastest, so both the global loads and
 *     the shared-memory reads are coalesced / free of bank conflicts).
 *  2. The stored triangle is walked in *panels* of whole stored columns
 *     (packed order, so a panel is a contiguous range of packed offsets).
 *     Each panel is copied once from DRAM into shared memory for all
 *     @f$P@f$ locations, `tile[e][p]`; consecutive locations sit next to one
 *     another in `phi2_toep`, so these loads are coalesced too.
 *  3. Every thread owns one location @f$p@f$ (`threadIdx.x % P`) and the
 *     outputs @f$o = g, g + G, g + 2G, \ldots@f$ (@f$g@f$ =
 *     `threadIdx.x / P`, @f$G@f$ = threads / @f$P@f$), with one register
 *     accumulator per output and coil. For each output it adds every
 *     contribution held by the panel (see ::mr_k_bdt), reading the panel and
 *     the inputs from shared memory. Each stored element is thus read from
 *     DRAM once per coil batch, and from shared memory by the (at most) two
 *     threads whose outputs it feeds.
 *  4. After the last panel the outputs are written back in place.
 *
 * The convention is the library's: each rank vector is a row vector
 * multiplied from the right, @f$y_o = \sum_r v_r A_{ro}@f$, with
 * @f$A_{ro} = \overline{A_{or}}@f$ for the unstored half.
 *
 * ### Launch configuration
 *
 * ::mr_bdt_pick chooses, per device and launch: @f$P@f$ (16 locations per
 * tile by default), the coil batch @f$NC \in \{1, 2, 4\}@f$ (template
 * parameter; as large as the coil group and the shared memory allow), the
 * outputs per thread (template parameter, @f$\lceil L / G \rceil@f$ rounded
 * up to an even number), the panel capacity @f$E@f$ (stored elements per
 * panel, at least @f$L@f$ so that every column fits) and one or two resident
 * blocks per SM. `MR_AHA_CUDA_BDT_P`, `MR_AHA_CUDA_BDT_NC` and
 * `MR_AHA_CUDA_BDT_BLOCKS` override the choice, for tuning.
 *
 * ### Performance
 *
 * `2d72` (320^2 image, 640^2 grid, L = 72, 12 coils), whole A^H A on one
 * GPU, generic -> tiled kernels: H100 335 -> 36 ms unmasked, 166 -> 33 ms
 * masked; L40S 309 -> 90 ms and 288 -> 86 ms; TITAN RTX 663 -> 224 ms and
 * 575 -> 198 ms. On the H100 the block stage drops from about 150 to 22 ms
 * (masked) and is no longer limited by DRAM bandwidth.
 *
 * The kernels have the contract of the specialised ones (in place on the
 * padded grids, masked mode through the index map of ::mr_k_mask_inverse,
 * coil groups, location ranges) and are launched through
 * ::mr_bd_launch_dense / ::mr_bd_launch_masked.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#include "mr_kernels.cuh"

#include <cuda_pipeline.h>

#include <cstdlib>

/** @brief Threads per block of the tiled kernels (also `__launch_bounds__`). */
#define MR_BDT_THREADS 256

/** @brief Largest number of outputs (per coil) one thread accumulates. */
#define MR_BDT_MAX_OPT 16

/* ------------------------------------------------------------------------ */
/* Device helpers                                                           */
/* ------------------------------------------------------------------------ */

/** @brief @f$\overline{a}\,b@f$ for single-precision complex numbers. */
__device__ __forceinline__ mr_complex mr_bdt_cmul_conj(mr_complex a, mr_complex b)
{
    return make_cuFloatComplex(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);
}

/**
 * @brief Packed offset of the first stored element of column @p c.
 *
 * Upper: column @p c holds rows `0..c`, from offset `c(c+1)/2`.
 * Lower: column @p c holds rows `c..L-1`, from offset `cL - c(c-1)/2`.
 * (See ::mr_tri_offset.)
 */
__host__ __device__ __forceinline__ int mr_bdt_colstart(int c, int L, int upper)
{
    return upper ? c * (c + 1) / 2 : c * L - c * (c - 1) / 2;
}

/** @brief Number of stored elements of column @p c (`c + 1` upper, `L - c`
 *         lower). */
__host__ __device__ __forceinline__ int mr_bdt_collen(int c, int L, int upper)
{
    return upper ? c + 1 : L - c;
}

/**
 * @brief The panel that starts at stored column @p c0: whole columns
 *        `[c0, *c1)` holding `*cnt <= E` stored elements.
 */
__device__ __forceinline__ void mr_bdt_panel(int c0, int L, int upper, int E,
                                             int *c1, int *cnt)
{
    int c = c0, n = 0;
    while (c < L && n + mr_bdt_collen(c, L, upper) <= E) {
        n += mr_bdt_collen(c, L, upper);
        ++c;
    }
    *c1 = c;
    *cnt = n;
}

/**
 * @brief Copy the panel of packed offsets `[t0, t0 + cnt)` of the tile's
 *        locations into @p dst (`[cnt][P]`, location fastest), and wait for
 *        this thread's copies.
 *
 * Consecutive threads load consecutive locations of one packed offset, which
 * sit next to one another in `phi2_toep`: the loads are coalesced. Locations
 * without work (@p kidx < 0) get zeros. The copies use the CUDA pipeline
 * primitives: on sm_80+ they are asynchronous copies (`cp.async`) that go
 * from L2 to shared memory without passing through registers, so every
 * thread can have all of its (dozens of) loads in flight at once; ordinary
 * loads into registers measured 25-35 % slower on the whole operator (H100
 * 45 vs 34 ms, L40S 101 vs 86 ms, `2d72` masked). On older GPUs the
 * primitives fall back to ordinary loads. (Double buffering the panels with
 * the same copies brought nothing on the H100, the L40S or the TITAN RTX:
 * two resident blocks per SM already overlap one block's loads with the
 * other's arithmetic.) The caller synchronises the block before reading.
 */
__device__ __forceinline__ void mr_bdt_fetch(mr_complex *dst, const mr_complex *d_toep,
                                             const long long *kidx, long long kstride,
                                             int t0, int cnt, int P)
{
    for (int idx = threadIdx.x; idx < cnt * P; idx += blockDim.x) {
        const int pp = idx % P, e = idx / P;
        const long long kk = kidx[pp];
        if (kk >= 0)
            __pipeline_memcpy_async(dst + idx, d_toep + kk + (long long)(t0 + e) * kstride,
                                    sizeof(mr_complex));
        else
            dst[idx] = make_cuFloatComplex(0.0f, 0.0f);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
}

/* ------------------------------------------------------------------------ */
/* Kernel                                                                   */
/* ------------------------------------------------------------------------ */

/**
 * @brief Tiled in-place block-diagonal multiplication for a group of coils.
 *
 * Contract of ::mr_k_bd_dense_t (@p MASKED false) or ::mr_k_bd_masked_t
 * (@p MASKED true) for any rank `dims.L <= OPT * (blockDim.x / P)`; see the
 * file comment for the algorithm. Launch with ::MR_BDT_THREADS threads,
 * `ceil(rg.nloc / P)` blocks and the dynamic shared memory reported by
 * ::mr_bdt_pick.
 *
 * Contributions to output @f$o@f$ held by a panel of stored columns
 * @f$[c_0, c_1)@f$:
 *  - (a) when @f$c_0 \le o < c_1@f$, stored column @f$o@f$ itself:
 *    @f$A_{ro} v_r@f$ for its rows @f$r@f$;
 *  - (b) the element @f$(o, c)@f$ of every other column @f$c@f$ of the panel
 *    that holds row @f$o@f$ (upper: @f$c > o@f$, lower: @f$c < o@f$):
 *    @f$\overline{A_{oc}}\, v_c@f$.
 *  Over all panels, (a) and (b) together visit @f$A_{ro}@f$ for every
 *  @f$r@f$ exactly once (the diagonal through (a) only).
 *
 * @tparam NC     Coils per batch (each staged panel serves all of them).
 * @tparam OPT    Outputs per thread and coil (`>= ceil(L / G)`).
 * @tparam MASKED Masked kernel layout and index map.
 * @param[in,out] d_pad       `[nloc, L, ncoil]` values with rank stride
 *                            `rg.vstride`.
 * @param[in]     d_toep      `[Mx, My, Mz, (L^2+L)/2]` (centred) or
 *                            `[num_masked, (L^2+L)/2]` (masked).
 * @param[in]     d_inv       Masked only: grid-to-mask map of
 *                            ::mr_k_mask_inverse (-1 outside the mask).
 * @param[in]     num_masked  Masked only: number of retained locations.
 * @param[in]     dims        Geometry descriptor.
 * @param[in]     ncoil       Coils in the group.
 * @param[in]     rg          Location range and rank stride (::mr_bd_range).
 * @param[in]     P           Locations per tile; divides `blockDim.x`.
 * @param[in]     E           Panel capacity in stored elements (`>= L`).
 */
template <int NC, int OPT, bool MASKED>
__global__ void __launch_bounds__(MR_BDT_THREADS)
mr_k_bdt(mr_complex *__restrict__ d_pad, const mr_complex *__restrict__ d_toep,
         const int *__restrict__ d_inv, long long num_masked, mr_dims dims,
         int ncoil, mr_bd_range rg, int P, int E)
{
    extern __shared__ __align__(16) unsigned char mr_bdt_smem[];
    const int L = dims.L;
    const int upper = dims.tri_upper;
    const int T = blockDim.x;
    const int G = T / P;
    const int p = threadIdx.x % P;
    const int g = threadIdx.x / P;

    /* Shared memory: kernel index of each tile location (-1 = no work),
     * staged inputs v[NC][L][P], panel tile[E][P]. */
    long long *kidx = reinterpret_cast<long long *>(mr_bdt_smem);
    mr_complex *v = reinterpret_cast<mr_complex *>(kidx + P);
    mr_complex *tile = v + (size_t)NC * L * P;

    const mr_complex zero = make_cuFloatComplex(0.0f, 0.0f);
    const long long first = (long long)blockIdx.x * P; /* range-relative */

    if (threadIdx.x < P) {
        const long long i = first + threadIdx.x;
        long long k = -1;
        if (i < rg.nloc) {
            const long long loc = rg.loc0 + i;
            if (MASKED) {
                k = d_inv[loc];
            } else {
                /* Natural (FFT-order) grid index -> centred kernel index. */
                const int j0 = (int)(loc % dims.m[0]);
                const long long q = loc / dims.m[0];
                const int j1 = (int)(q % dims.m[1]);
                const int j2 = (int)(q / dims.m[1]);
                k = mr_nat2cent(j0, dims.m[0], dims.c_m[0]) +
                    (long long)dims.m[0] *
                        (mr_nat2cent(j1, dims.m[1], dims.c_m[1]) +
                         (long long)dims.m[1] *
                             mr_nat2cent(j2, dims.m[2], dims.c_m[2]));
            }
        }
        kidx[threadIdx.x] = k;
    }
    __syncthreads();

    const long long tid = first + p;
    const bool inrange = tid < rg.nloc;
    const bool active = kidx[p] >= 0;   /* in range and, masked, in the mask */
    const long long padn = rg.vstride;
    const long long slot = padn * L;
    const long long kstride = MASKED ? num_masked : dims.pad_size;

    for (int s0 = 0; s0 < ncoil; s0 += NC) {
        const int nb = (ncoil - s0 < NC) ? ncoil - s0 : NC;

        mr_complex y[NC][OPT];
#pragma unroll
        for (int k = 0; k < NC; ++k)
#pragma unroll
            for (int j = 0; j < OPT; ++j) y[k][j] = zero;

        /* 1. Stage the inputs (the previous batch must be done with v). */
        __syncthreads();
        for (int idx = threadIdx.x; idx < NC * L * P; idx += T) {
            const int pp = idx % P;
            const int kl = idx / P;
            const int k = kl / L, l = kl % L;
            mr_complex a = zero;
            if (k < nb && kidx[pp] >= 0)
                a = d_pad[slot * (s0 + k) + padn * l + first + pp];
            v[idx] = a;
        }

        /* 2.-3. Walk the triangle in panels of whole stored columns. */
        for (int c0 = 0, c1, cnt; c0 < L; c0 = c1) {
            mr_bdt_panel(c0, L, upper, E, &c1, &cnt);
            const int t0 = mr_bdt_colstart(c0, L, upper);
            __syncthreads();   /* tile free, inputs staged */
            mr_bdt_fetch(tile, d_toep, kidx, kstride, t0, cnt, P);
            __syncthreads();

            if (active) {
#pragma unroll
                for (int j = 0; j < OPT; ++j) {
                    const int o = g + j * G;
                    if (o >= L) break;
                    mr_complex acc[NC];
#pragma unroll
                    for (int k = 0; k < NC; ++k) acc[k] = zero;

                    /* (a) Stored column o, when it is in this panel. */
                    if (o >= c0 && o < c1) {
                        const int rmin = upper ? 0 : o;
                        const int n = mr_bdt_collen(o, L, upper);
                        const mr_complex *tp =
                            tile + (mr_bdt_colstart(o, L, upper) - t0) * P + p;
                        const mr_complex *vp = v + rmin * P + p;
                        #pragma unroll 4
                        for (int r = 0; r < n; ++r) {
                            const mr_complex a = tp[r * P];
#pragma unroll
                            for (int k = 0; k < NC; ++k)
                                acc[k] = cuCaddf(acc[k],
                                                 cuCmulf(a, vp[(k * L + r) * P]));
                        }
                    }

                    /* (b) Row o of the panel's other columns, conjugated. */
                    const int cb = upper ? max(c0, o + 1) : c0;
                    const int ce = upper ? c1 : min(c1, o);
                    if (cb < ce) {
                        /* Tile row of element (o, cb); then step by the
                         * distance to (o, c + 1): c + 1 (upper), L - c - 1
                         * (lower). */
                        int e = mr_bdt_colstart(cb, L, upper) - t0 +
                                (upper ? o : o - cb);
                        #pragma unroll 4
                        for (int c = cb; c < ce; ++c) {
                            const mr_complex a = tile[e * P + p];
#pragma unroll
                            for (int k = 0; k < NC; ++k)
                                acc[k] = cuCaddf(
                                    acc[k],
                                    mr_bdt_cmul_conj(a, v[(k * L + c) * P + p]));
                            e += upper ? c + 1 : L - c - 1;
                        }
                    }
#pragma unroll
                    for (int k = 0; k < NC; ++k) y[k][j] = cuCaddf(y[k][j], acc[k]);
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
                    d_pad[slot * (s0 + k) + padn * o + tid] = active ? y[k][j] : zero;
                }
            }
        }
    }
}

/* ------------------------------------------------------------------------ */
/* Host side: configuration and dispatch                                    */
/* ------------------------------------------------------------------------ */

/** @brief Pointer to a tiled kernel instance. */
typedef void (*mr_bdt_fn_t)(mr_complex *, const mr_complex *, const int *, long long,
                            mr_dims, int, mr_bd_range, int, int);

/** @brief Launch configuration of a tiled kernel on the current device. */
struct mr_bdt_cfg {
    int       P;       /**< Locations per tile.                             */
    int       NC;      /**< Coils per batch (1, 2 or 4).                    */
    int       OPT;     /**< Outputs per thread (even, <= ::MR_BDT_MAX_OPT). */
    int       E;       /**< Panel capacity, stored elements.                */
    size_t    shared;  /**< Dynamic shared memory per block, bytes.         */
    mr_bdt_fn_t fn;    /**< Kernel instance.                                */
};

/** @brief Read an integer from the environment, or @p def when unset. */
static int mr_bdt_env(const char *name, int def)
{
    const char *e = std::getenv(name);
    return (e && e[0]) ? atoi(e) : def;
}

/** @brief Instance for (NC, OPT, MASKED), or NULL. */
template <bool MASKED>
static mr_bdt_fn_t mr_bdt_select(int nc, int opt)
{
#define MR_BDT_OPT(NC)                                                         \
    switch (opt) {                                                             \
    case 2: return mr_k_bdt<NC, 2, MASKED>;                                    \
    case 4: return mr_k_bdt<NC, 4, MASKED>;                                    \
    case 6: return mr_k_bdt<NC, 6, MASKED>;                                    \
    case 8: return mr_k_bdt<NC, 8, MASKED>;                                    \
    case 10: return mr_k_bdt<NC, 10, MASKED>;                                  \
    case 12: return mr_k_bdt<NC, 12, MASKED>;                                  \
    case 14: return mr_k_bdt<NC, 14, MASKED>;                                  \
    case 16: return mr_k_bdt<NC, 16, MASKED>;                                  \
    default: return nullptr;                                                   \
    }
    switch (nc) {
    case 1: MR_BDT_OPT(1)
    case 2: MR_BDT_OPT(2)
    case 4: MR_BDT_OPT(4)
    default: return nullptr;
    }
#undef MR_BDT_OPT
}

/**
 * @brief Choose the tiled launch configuration for rank @p L and a group of
 *        @p ncoil coils on the current device.
 *
 * Measured on `2d72` (L = 72, 12 coils in groups of 4, masked; whole
 * operator), the coil batch matters most (each panel read from DRAM serves
 * @f$NC@f$ coils: H100 49 ms with 2 coils, 34 ms with 4), then 16 locations
 * per tile, then two resident blocks per SM where they fit with a reasonable
 * panel (H100: 34 ms with two blocks, 43 ms with one; L40S: 86 ms with one,
 * 89 ms with two and a panel of only 93 elements). Hence:
 *
 * - @f$P@f$ = 16 locations per tile (`MR_AHA_CUDA_BDT_P` = 8, 16 or 32 for
 *   tuning), halved while @f$\lceil L / G \rceil >@f$ ::MR_BDT_MAX_OPT outputs
 *   per thread (@f$G@f$ = ::MR_BDT_THREADS / @f$P@f$): up to @f$L = 256@f$
 *   with 16, 512 with 8.
 * - @f$NC@f$ = the smallest of 1, 2, 4 that covers @p ncoil (4 at most;
 *   `MR_AHA_CUDA_BDT_NC` for tuning), halved while the inputs and a
 *   one-column panel do not fit the shared memory of a block.
 * - Two resident blocks per SM when both the registers of the kernel
 *   instance and the shared memory allow it with a panel of at least
 *   @f$2L@f$ elements (or the whole triangle), else one
 *   (`MR_AHA_CUDA_BDT_BLOCKS` = 1 or 2 is tried first, for tuning). The
 *   panel takes the rest of the shared-memory budget, up to the whole
 *   triangle.
 *
 * @param[in]  L      Rank.
 * @param[in]  ncoil  Coils in the group.
 * @param[in]  masked Masked kernel.
 * @param[out] c      Configuration.
 * @return ::MR_SUCCESS, ::MR_ERR_CUDA, or ::MR_ERR_UNSUPPORTED when @p L is
 *         too large or the device's shared memory too small.
 */
static int mr_bdt_pick(int L, int ncoil, bool masked, mr_bdt_cfg *c)
{
    int dev = 0, sm_smem = 0, optin = 0, deflt = 0, sm_regs = 0;
    MR_CUDA_CHECK(cudaGetDevice(&dev));
    MR_CUDA_CHECK(cudaDeviceGetAttribute(
        &sm_smem, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
    MR_CUDA_CHECK(cudaDeviceGetAttribute(
        &optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    MR_CUDA_CHECK(cudaDeviceGetAttribute(
        &deflt, cudaDevAttrMaxSharedMemoryPerBlock, dev));
    MR_CUDA_CHECK(cudaDeviceGetAttribute(
        &sm_regs, cudaDevAttrMaxRegistersPerMultiprocessor, dev));
    if (optin < deflt) optin = deflt;

    int P = mr_bdt_env("MR_AHA_CUDA_BDT_P", 16);
    if (P != 8 && P != 32) P = 16;
    while (P >= 8 && (L + MR_BDT_THREADS / P - 1) / (MR_BDT_THREADS / P) > MR_BDT_MAX_OPT)
        P /= 2;
    if (P < 8)
        MR_FAIL(MR_ERR_UNSUPPORTED, "tiled block-diagonal kernel: L = %d is too large", L);
    const int G = MR_BDT_THREADS / P;
    int opt = (L + G - 1) / G;
    opt += opt & 1;

    int nc = mr_bdt_env("MR_AHA_CUDA_BDT_NC", 0);
    if (nc != 1 && nc != 2 && nc != 4) nc = (ncoil <= 1) ? 1 : (ncoil <= 2) ? 2 : 4;
    const int force_bps = mr_bdt_env("MR_AHA_CUDA_BDT_BLOCKS", 0);
    /* Blocks per SM to try, in order (a forced count first). */
    const int order[2] = {force_bps == 1 ? 1 : 2, force_bps == 1 ? 2 : 1};

    const size_t c8 = sizeof(mr_complex);
    const size_t tri = (size_t)L * (L + 1) / 2;
    /* Shared memory the driver reserves per resident block (sm_80+: 1 KB). */
    const size_t reserve = 1024;
    for (; nc >= 1; nc /= 2) {
        mr_bdt_fn_t f = masked ? mr_bdt_select<true>(nc, opt) : mr_bdt_select<false>(nc, opt);
        if (f == nullptr)
            MR_FAIL(MR_ERR_UNSUPPORTED, "tiled block-diagonal kernel: no instance for "
                                        "NC = %d, OPT = %d", nc, opt);
        /* Resident blocks the registers allow (allocated per warp in units
         * of 256). */
        cudaFuncAttributes fa;
        MR_CUDA_CHECK(cudaFuncGetAttributes(&fa, f));
        const int warp_regs = (fa.numRegs * 32 + 255) / 256 * 256;
        const int reg_bps = sm_regs / (warp_regs * (MR_BDT_THREADS / 32));

        const size_t fixed = (size_t)P * sizeof(long long) + (size_t)nc * L * P * c8;
        for (int oi = 0; oi < 2; ++oi) {
            const int bps = order[oi];
            if (bps > reg_bps && bps > 1) continue;
            size_t budget = (size_t)sm_smem / bps;
            budget = (budget > reserve) ? budget - reserve : 0;
            if (budget > (size_t)optin) budget = (size_t)optin;
            if (budget < fixed + (size_t)L * P * c8) continue;
            size_t E = (budget - fixed) / ((size_t)P * c8);
            if (E > tri) E = tri;
            if (bps == 2 && force_bps != 2 && E < 2 * (size_t)L && E < tri) continue;
            c->P = P;
            c->NC = nc;
            c->OPT = opt;
            c->E = (int)E;
            c->shared = fixed + E * P * c8;
            c->fn = f;
            return MR_SUCCESS;
        }
    }
    MR_FAIL(MR_ERR_UNSUPPORTED,
            "tiled block-diagonal kernel: L = %d does not fit %d bytes of shared "
            "memory per block",
            L, optin);
}

/** @brief Configure and launch one tiled kernel on stream @p s. */
static int mr_bdt_launch(bool masked, mr_complex *d_pad, const mr_complex *d_toep,
                         const int *d_inv, long long num_masked, const mr_dims &dims,
                         int ncoil, mr_bd_range rg, cudaStream_t s)
{
    if (rg.nloc <= 0 || ncoil <= 0) return MR_SUCCESS;
    mr_bdt_cfg c;
    int rc = mr_bdt_pick(dims.L, ncoil, masked, &c);
    if (rc != MR_SUCCESS) return rc;
    /* Opt in to the shared memory (per device and kernel; cheap to repeat). */
    MR_CUDA_CHECK(cudaFuncSetAttribute(c.fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       (int)c.shared));
    MR_CUDA_CHECK(cudaFuncSetAttribute(c.fn, cudaFuncAttributePreferredSharedMemoryCarveout,
                                       (int)cudaSharedmemCarveoutMaxShared));
    const long long grid = (rg.nloc + c.P - 1) / c.P;
    c.fn<<<(unsigned)grid, MR_BDT_THREADS, c.shared, s>>>(d_pad, d_toep, d_inv, num_masked,
                                                          dims, ncoil, rg, c.P, c.E);
    MR_KERNEL_CHECK(masked ? "mr_k_bdt (masked)" : "mr_k_bdt (dense)");
    return MR_SUCCESS;
}

/* ------------------------------------------------------------------------ */
/* Launchers shared by the specialised and the tiled kernels                */
/* ------------------------------------------------------------------------ */

int mr_bd_launch_dense(const mr_bd_kernels &k, int threads, mr_complex *d_pad,
                       const mr_complex *d_toep, const mr_dims &dims, int ncoil,
                       mr_bd_range rg, cudaStream_t s)
{
    if (k.tiled)
        return mr_bdt_launch(false, d_pad, d_toep, nullptr, 0, dims, ncoil, rg, s);
    const int grid = (int)((rg.nloc + threads - 1) / threads);
    k.dense<<<grid, threads, 0, s>>>(d_pad, d_toep, dims, ncoil, rg);
    MR_KERNEL_CHECK("mr_k_bd_dense_t");
    return MR_SUCCESS;
}

int mr_bd_launch_masked(const mr_bd_kernels &k, int threads, mr_complex *d_pad,
                        const mr_complex *d_toep, const int *d_inv,
                        long long num_masked, const mr_dims &dims, int ncoil,
                        mr_bd_range rg, cudaStream_t s)
{
    if (k.tiled)
        return mr_bdt_launch(true, d_pad, d_toep, d_inv, num_masked, dims, ncoil, rg, s);
    const int grid = (int)((rg.nloc + threads - 1) / threads);
    k.masked<<<grid, threads, 0, s>>>(d_pad, d_toep, d_inv, num_masked, dims, ncoil, rg);
    MR_KERNEL_CHECK("mr_k_bd_masked_t");
    return MR_SUCCESS;
}
