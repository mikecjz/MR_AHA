/**
 * @file mr_kernels.cuh
 * @brief Declarations of the CUDA kernels backing the MR_AHA operators.
 *
 * Not part of the public API. All kernels take their geometry by value through
 * ::mr_dims (or explicit scalars) so that no shared device state exists and the
 * library stays re-entrant across plans, streams and host threads.
 *
 * @author Junzhou Chen (junzhou@chen.engineer)
 */

#ifndef MR_AHA_MR_KERNELS_CUH
#define MR_AHA_MR_KERNELS_CUH

#include "mr_internal.cuh"

/* ------------------------------------------------------------------------ */
/* AhA kernels                                                              */
/* ------------------------------------------------------------------------ */

/**
 * @brief Multiply by one coil sensitivity map and scatter onto the padded grid.
 *
 * Writes the whole `[Mx, My, Mz, L]` working buffer in a single pass: locations
 * covered by the image receive @f$S_c \odot x@f$, every other location receives
 * zero, so no separate `cudaMemset` is needed.
 *
 * The destination is stored in **natural** (FFT) order: image index @f$x_d@f$
 * maps to padded index @f$(x_d - \lfloor N_d/2 \rfloor) \bmod M_d@f$. Folding
 * the centring shift into the scatter makes the subsequent unnormalised cuFFT
 * an exact *centred* DFT, with no phase modulation and no half-sample bias for
 * odd matrix sizes.
 *
 * @param[out] d_pad  Padded working buffer `[Mx, My, Mz, L]`, fully written.
 * @param[in]  d_x    Spatial coefficients `[Nx, Ny, Nz, L]`.
 * @param[in]  d_SEs  Sensitivity maps `[Nx, Ny, Nz, Ncoils]`, one per coil.
 * @param[in]  coil   Index of the coil being processed.
 * @param[in]  dims   Geometry descriptor.
 */
__global__ void mr_k_pad_mul_se(mr_complex *d_pad, const mr_complex *d_x,
                                const mr_complex *d_SEs, int coil, mr_dims dims);

/**
 * @brief Crop the padded volume, apply @f$S_c^H@f$ and accumulate into `AhAx`.
 *
 * Exact adjoint of ::mr_k_pad_mul_se: the same index remap is used, so the
 * pad/crop pair is @f$\mathcal{P}@f$ and @f$\mathcal{P}^H@f$ and the composed
 * operator is Hermitian whenever the block-diagonal multiplier is.
 *
 * @param[in,out] d_out      Output `[Nx, Ny, Nz, L]`.
 * @param[in]     d_pad      Padded volume `[Mx, My, Mz, L]` in natural order.
 * @param[in]     d_SEs      Sensitivity maps `[Nx, Ny, Nz, Ncoils]`.
 * @param[in]     coil       Index of the coil being processed.
 * @param[in]     accumulate Non-zero to add into @p d_out, zero to overwrite
 *                           (used for the first coil, avoiding a memset).
 * @param[in]     dims       Geometry descriptor.
 */
__global__ void mr_k_crop_mul_conj_se(mr_complex *d_out, const mr_complex *d_pad,
                                      const mr_complex *d_SEs, int coil,
                                      int accumulate, mr_dims dims);

/**
 * @brief In-place Hermitian block-diagonal multiply over the whole grid.
 *
 * At every upsampled k-space location the length-@f$L@f$ vector of rank
 * coefficients is multiplied by the Hermitian @f$L\times L@f$ block stored in
 * packed triangular form, @f$y_i = \sum_j A_{ij} v_j@f$, with the unstored half
 * reconstructed as @f$A_{ij} = \overline{A_{ji}}@f$.
 *
 * The working buffer is in natural FFT order while `phi2_toep` is centred, so
 * the kernel converts indices on the fly; this is what removes the need for any
 * explicit fftshift pass between the cuFFTs.
 *
 * Requires `blockDim.x * L * sizeof(mr_complex)` bytes of dynamic shared
 * memory to stage each thread's input vector.
 *
 * @param[in,out] d_pad       Padded k-space `[Mx, My, Mz, L]`, natural order.
 * @param[in]     d_phi2_toep Packed blocks `[Mx, My, Mz, (L^2+L)/2]`, centred.
 * @param[in]     dims        Geometry descriptor.
 */
__global__ void mr_k_blockdiag_dense(mr_complex *d_pad,
                                     const mr_complex *d_phi2_toep, mr_dims dims);

/**
 * @brief Hermitian block-diagonal multiply restricted to masked locations.
 *
 * Gathers the rank vectors at the ::mr_config::num_masked retained locations,
 * applies the corresponding packed blocks, and writes the result to a compact
 * `[num_masked, L]` buffer. The caller then zeroes the padded grid and calls
 * ::mr_k_scatter_masked, which realises the "zero outside the mask" semantics.
 *
 * Requires `blockDim.x * L * sizeof(mr_complex)` bytes of dynamic shared memory.
 *
 * @param[out] d_compact    Result buffer `[num_masked, L]`.
 * @param[in]  d_pad        Padded k-space `[Mx, My, Mz, L]`, natural order.
 * @param[in]  d_phi2_toep  Packed blocks `[num_masked, (L^2+L)/2]`.
 * @param[in]  d_mask       0-based centred linear indices, length @p num_masked.
 * @param[in]  num_masked   Number of retained locations.
 * @param[in]  dims         Geometry descriptor.
 */
__global__ void mr_k_blockdiag_masked(mr_complex *d_compact,
                                      const mr_complex *d_pad,
                                      const mr_complex *d_phi2_toep,
                                      const int64_t *d_mask,
                                      long long num_masked, mr_dims dims);

/**
 * @brief Build the natural-order grid-to-mask index map used by the
 *        specialised masked block-diagonal kernels.
 *
 * Writes `d_inv[j] = i` for every mask entry `i`, where `j` is the natural
 * (FFT-order) linear index of the centred location `d_mask[i]`. The caller
 * must first fill @p d_inv with -1 (`cudaMemset` to `0xFF`), which then marks
 * the locations outside the mask.
 *
 * @param[out] d_inv      `[Mx, My, Mz]` int32 map, natural order.
 * @param[in]  d_mask     0-based centred linear indices.
 * @param[in]  num_masked Number of retained locations (< 2^31).
 * @param[in]  dims       Geometry descriptor.
 */
__global__ void mr_k_mask_inverse(int *d_inv, const int64_t *d_mask,
                                  long long num_masked, mr_dims dims);

/**
 * @brief Scatter a compact `[num_masked, L]` buffer back onto the padded grid.
 *
 * The padded grid must have been zeroed beforehand; locations outside the mask
 * therefore stay zero, which is the intended behaviour of a masked Toeplitz
 * operator.
 *
 * @param[out] d_pad      Padded k-space `[Mx, My, Mz, L]`, natural order.
 * @param[in]  d_compact  Result buffer `[num_masked, L]`.
 * @param[in]  d_mask     0-based centred linear indices.
 * @param[in]  num_masked Number of retained locations.
 * @param[in]  dims       Geometry descriptor.
 */
__global__ void mr_k_scatter_masked(mr_complex *d_pad, const mr_complex *d_compact,
                                    const int64_t *d_mask, long long num_masked,
                                    mr_dims dims);

/* ------------------------------------------------------------------------ */
/* Block-diagonal kernels specialised on L (mr_kernels_bd.cu)               */
/* ------------------------------------------------------------------------ */

/** @brief Threads per block of the specialised block-diagonal kernels (also
 *         their `__launch_bounds__`). */
#define MR_BD_THREADS 256

/** @brief Largest rank with a specialised block-diagonal kernel. Larger ranks
 *         use the generic shared-memory kernels: above about L = 30 one coil's
 *         vectors no longer fit the register file and the specialised kernels
 *         spill (TITAN RTX, 384^2 2D, 8 coils: L = 28 73 vs 79 ms, L = 30
 *         88 vs 89 ms, L = 32 105 vs 102 ms, specialised vs generic). */
#define MR_BD_MAX_L 30

/** @brief Largest rank whose specialised kernels process coils in pairs (see
 *         `MR_BD_PAIR` in mr_kernels_bd.cu). */
#define MR_BD_PAIR_MAX_L 16

/**
 * @brief Range of grid locations processed by a specialised block-diagonal
 *        kernel, and the layout of its data.
 *
 * The kernels process the natural-order locations `[loc0, loc0 + nloc)`;
 * rank `l` of coil `s` of location `loc0 + i` is at
 * `d_pad[i + vstride * (l + L * s)]`. The whole grid is
 * `{0, Mx*My*Mz, Mx*My*Mz}` (see ::mr_bd_whole).
 */
struct mr_bd_range {
    long long loc0;    /**< First location (natural-order linear index).   */
    long long nloc;    /**< Number of locations.                           */
    long long vstride; /**< Element stride between consecutive ranks.      */
};

/** @brief ::mr_bd_range covering the whole upsampled grid of @p dims. */
inline mr_bd_range mr_bd_whole(const mr_dims &dims)
{
    return mr_bd_range{0, dims.pad_size, dims.pad_size};
}

/** @brief Signature of the specialised dense kernel `mr_k_bd_dense_t<L>`:
 *         `(d_pad, d_phi2_toep, dims, ncoil, range)`. */
typedef void (*mr_bd_dense_fn)(mr_complex *, const mr_complex *, mr_dims, int,
                               mr_bd_range);

/** @brief Signature of the specialised masked kernel `mr_k_bd_masked_t<L>`:
 *         `(d_pad, d_phi2_toep, d_inv, num_masked, dims, ncoil, range)`; it
 *         works in place on the padded grids through the index map of
 *         ::mr_k_mask_inverse. */
typedef void (*mr_bd_masked_fn)(mr_complex *, const mr_complex *, const int *,
                                long long, mr_dims, int, mr_bd_range);

/** @brief The specialised block-diagonal kernels of one rank. */
struct mr_bd_kernels {
    mr_bd_dense_fn  dense;   /**< Dense kernel, or NULL if no specialisation. */
    mr_bd_masked_fn masked;  /**< Masked kernel, or NULL likewise.            */
    int             pair;    /**< Non-zero when coils are processed in pairs. */
};

/**
 * @brief Look up the block-diagonal kernels specialised on rank @p L.
 *
 * The kernels (`mr_k_bd_dense_t<L>` / `mr_k_bd_masked_t<L>`, defined in
 * mr_kernels_bd.cu) differ from ::mr_k_blockdiag_dense /
 * ::mr_k_blockdiag_masked in three ways: the rank vectors live in registers
 * (no shared memory), each stored element of `phi2_toep` is read once per
 * triangle walk instead of twice, and one launch serves a group of `ncoil`
 * coils whose grids (and, masked, compact buffers) are stored back to back,
 * so `phi2_toep` is read from DRAM about once per group. The masked kernel
 * works in place on the padded grids (see ::mr_k_mask_inverse). Launch with
 * at most ::MR_BD_THREADS threads per block, no dynamic shared memory, and one
 * thread per location of the upsampled grid.
 *
 * @param[in] L Rank.
 * @return The kernels, with NULL pointers when @p L is outside
 *         `1..`::MR_BD_MAX_L.
 */
mr_bd_kernels mr_bd_lookup(int L);

/* ------------------------------------------------------------------------ */
/* calc_toep kernels                                                        */
/* ------------------------------------------------------------------------ */

/**
 * @brief Pack the dense @f$L\times L@f$ Gram blocks into triangular storage.
 *
 * @param[out] d_tri    Packed output `[nbatch, (L^2+L)/2]`.
 * @param[in]  d_phi2   Dense input `[nbatch, L, L]` (column-major blocks).
 * @param[in]  nbatch   `Ntrajs` (2D/3D) or `Ntrajs*Nkz` (stack).
 * @param[in]  L        Rank.
 * @param[in]  tri_upper Non-zero to keep the upper triangle.
 */
__global__ void mr_k_extract_tri(mr_complex *d_tri, const mr_complex *d_phi2,
                                 long long nbatch, int L, int tri_upper);

/**
 * @brief Broadcast packed Gram blocks onto the non-uniform samples of a chunk.
 *
 * Each trajectory contributes one Gram block that is shared by all `Nkx`
 * readout samples along it, so the strengths handed to cufinufft are simply a
 * broadcast along the readout axis.
 *
 * @param[out] d_c      Strength buffer `[Ntrajs*Nkx, Nkz_eff*chunk]`.
 * @param[in]  d_tri    Packed blocks `[Ntrajs, Nkz_eff, (L^2+L)/2]`.
 * @param[in]  Ntrajs   Number of trajectories.
 * @param[in]  Nkx      Samples per trajectory.
 * @param[in]  Nkz_eff  `Mz` for a stack trajectory, otherwise 1.
 * @param[in]  t0       First triangular component of this chunk.
 * @param[in]  chunk    Number of triangular components in this chunk. The last
 *                      chunk may run past @p ntri; those components are filled
 *                      with zeros so that a single cufinufft plan can be reused
 *                      for every chunk.
 * @param[in]  ntri     Total number of triangular components, `(L^2+L)/2`.
 */
__global__ void mr_k_fill_strengths(mr_complex *d_c, const mr_complex *d_tri,
                                    long long Ntrajs, long long Nkx,
                                    int Nkz_eff, long long t0, int chunk,
                                    long long ntri);

/**
 * @brief Copy a batch of volumes from centred to natural (FFT) index order.
 *
 * A per-axis shift of zero leaves that axis untouched, which is how the kz axis
 * of a stack-of-stars kernel is carried through without a transform.
 *
 * @param[out] dst    Destination `[m0, m1, m2, batch]`, natural order.
 * @param[in]  src    Source `[m0, m1, m2, batch]`, centred order.
 * @param[in]  m0,m1,m2 Axis extents.
 * @param[in]  s0,s1,s2 Per-axis centre offsets `floor(m/2)`, or 0 to skip.
 * @param[in]  batch  Number of volumes.
 */
__global__ void mr_k_to_natural(mr_complex *dst, const mr_complex *src,
                                int m0, int m1, int m2, int s0, int s1, int s2,
                                long long batch);

/**
 * @brief Copy a batch of volumes from natural to centred order, with scaling.
 *
 * @param[out] dst    Destination `[m0, m1, m2, batch]`, centred order.
 * @param[in]  src    Source `[m0, m1, m2, batch]`, natural order.
 * @param[in]  m0,m1,m2 Axis extents.
 * @param[in]  s0,s1,s2 Per-axis centre offsets `floor(m/2)`, or 0 to skip.
 * @param[in]  batch  Number of volumes.
 * @param[in]  scale  Real factor applied to every element (the @f$1/\prod M@f$
 *                    of the unnormalised inverse cuFFT).
 */
__global__ void mr_k_to_centered_scale(mr_complex *dst, const mr_complex *src,
                                       int m0, int m1, int m2,
                                       int s0, int s1, int s2,
                                       long long batch, float scale);

/**
 * @brief Gather masked locations from a natural-order batch, with scaling.
 *
 * Fuses the centring shift, the inverse-FFT scaling and the mask gather into a
 * single pass, writing directly into the `[num_masked, (L^2+L)/2]` output.
 *
 * @param[out] dst        Output `[num_masked, ntri]`; this chunk fills columns
 *                        `[t0, t0+batch)`.
 * @param[in]  src        Source `[m0, m1, m2, batch]`, natural order.
 * @param[in]  d_mask     0-based centred linear indices.
 * @param[in]  num_masked Number of retained locations.
 * @param[in]  m0,m1,m2   Axis extents.
 * @param[in]  s0,s1,s2   Per-axis centre offsets, or 0 to skip.
 * @param[in]  t0         First triangular component of this chunk.
 * @param[in]  batch      Number of volumes in this chunk.
 * @param[in]  scale      Real factor applied to every element.
 */
__global__ void mr_k_gather_masked_scale(mr_complex *dst, const mr_complex *src,
                                         const int64_t *d_mask,
                                         long long num_masked,
                                         int m0, int m1, int m2,
                                         int s0, int s1, int s2,
                                         long long t0, long long batch,
                                         float scale);

#endif /* MR_AHA_MR_KERNELS_CUH */
