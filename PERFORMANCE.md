# How the CUDA $A^H A$ operator was made faster

This note explains the mechanisms behind the 2026-11 performance work on the CUDA backend:
what each change does, why it saves time, and what was tried and rejected. The measurements are
in [`benchmarks/README.md`](benchmarks/README.md#cuda-operator-optimisation-and-multi-gpu-2026-11);
the user-facing description is in the [README](README.md#implementation-notes) and the
[Multi-GPU section](README.md#multi-gpu-cuda). The work followed two plans, one for a single GPU
and one for several GPUs.

| | before | after |
|---|---|---|
| stack-of-stars 320² × 56, L = 12, 12 coils, one H100 | 254.5 ms | 178.9 ms |
| the same, three H100s (NVLink) | — | 65.3 ms |
| 3D kooshball 160³, L = 12, 12 coils, one H100 | 362.6 ms | 254.6 ms |
| the same, three H100s | — | 88.2 ms |

---

## 1. Where the time went

The operator is, per coil $c$,

```
pad × S_c  →  forward FFT  →  block-diagonal multiply  →  inverse FFT  →  crop × conj(S_c)
```

and the per-stage GPU timer added for this work (`MR_AHA_CUDA_PROFILE=1`, CUDA events between the
stages, no extra synchronisation) showed on the TITAN RTX, stack problem:

| stage | before | share |
|---|---|---|
| pad × S_c | 66 ms | 5 % |
| forward FFT | 324 ms | 24 % |
| **block-diagonal** | **613 ms** | **45 %** |
| inverse FFT | 324 ms | 24 % |
| crop × conj(S_c) | 35 ms | 3 % |

The block-diagonal stage multiplies, at every upsampled k-space location, the length-$L$ vector of
rank coefficients by a Hermitian $L \times L$ block stored as its $(L^2+L)/2$ upper or lower
triangle in `phi2_toep`. It does very little arithmetic per byte, so it is limited by memory
bandwidth, and the bytes are dominated by `phi2_toep` itself: 78 complex values per location at
$L = 12$, against 12 values of data read and 12 written. For the stack problem `phi2_toep` is
14.3 GB, and the old operator streamed it from DRAM for every one of the 12 coils.

The old kernel had three inefficiencies, each addressed below:

1. each off-diagonal stored element was read **twice** (once as $A_{ij}$ for row $i$, once as
   $\overline{A_{ji}}$ for row $j$);
2. the whole kernel was read **once per coil**;
3. because $L$ was a run-time value, each thread staged its vector in shared memory, which limited
   the block size and the occupancy for large $L$.

---

## 2. Single GPU

### 2.1 Read each stored element once

Instead of evaluating the matrix-vector product row by row, the kernel walks the stored triangle
once, in packed order. Packed component $t$ is the element $(r, c)$, and its single load contributes
to two outputs:

$$
y_r \mathrel{+}= A_{rc}\, v_c, \qquad
y_c \mathrel{+}= \overline{A_{rc}}\, v_r \quad (r \ne c).
$$

This halves the reads of `phi2_toep` for the off-diagonal part. The order in which each $y_i$
accumulates its terms ($j = 0, 1, \ldots, L-1$) is the same as in the row-by-row kernel, so the
result is **bitwise identical** (checked by `tests/check_bd_kernels.py` for $L = 1 \ldots 33$, both
triangles, masked and unmasked).

### 2.2 Specialise on $L$ so everything lives in registers

The kernels `mr_k_bd_dense_t<L, PAIR>` and `mr_k_bd_masked_t<L, PAIR>` (`src/cuda/mr_kernels_bd.cu`)
are templates on the rank, instantiated for $L = 1 \ldots 30$ and selected at plan creation through a
`switch` (`mr_bd_lookup`). With $L$ a compile-time constant, every loop is fully unrolled, every
array index is a constant, and the input and output vectors are plain registers: no shared memory,
no occupancy limit from it.

**Register pressure was the hard part.** Fully unrolled, the walk is a straight line of ~78 loads
(at $L = 12$), and the compiler likes to issue loads early to hide latency. Left alone, it
computed the addresses of, and loaded, many elements ahead of their use. Each such element held a
64-bit address and a complex value in registers, and the kernel spilled to local memory from
$L = 11$ on. Measured with `-Xptxas -v`:

| variant | L = 12, coil pair, sm_75 |
|---|---|
| `__ldg` (read-only cache) loads | 255 registers, spills |
| plain loads, `phi2_toep[t * stride]` | 216 registers, no spill (but spills again once a location-range parameter was added) |
| **running pointer `p += stride`, empty `asm volatile("" ::: "memory")` at each column, no `__restrict__`** | **206 registers, no spill** |

`__ldg` and `__restrict__` make it worse because they mark the loads as invariant, which allows
the compiler to move them anywhere. The empty `asm` with a memory clobber is a compiler barrier:
loads cannot move across it, so at most one column's loads (≤ $L$) are in flight at once. That is
still plenty of memory-level parallelism. On the TITAN RTX the spilling version cost 16 % in the
block-diagonal stage (282 vs 243 ms for the stack problem).

Above $L \approx 30$, a single coil's two vectors ($4L$ registers) plus addressing no longer fit in
255 registers and the specialised kernel spills whatever the code shape. Measured (TITAN RTX, 384² 2D,
8 coils), the specialised kernel is 7 % faster at $L = 28$, equal at $L = 30$, and 3 % slower at
$L = 32$. So $L > 30$ uses the generic kernel (`MR_BD_MAX_L = 30`).

### 2.3 Coil groups: read `phi2_toep` once per group, not once per coil

The plan now processes coils in groups (default 4, `MR_AHA_CUDA_COIL_GROUP`):

- the padded grids of a group sit side by side in one buffer `[Mx, My, Mz, L, group]`;
- **one batched cuFFT** (`cufftMakePlanMany64` over `group · L` volumes) transforms all of them;
- **one block-diagonal launch** serves the whole group. Each thread owns one k-space location
  and processes the group's coils two at a time (`PAIR`, for $L \le 16$). Every element of
  `phi2_toep` it loads is applied to both coils' vectors from registers, so a coil pair costs one
  read. The elements loaded for one pair are still in L2 when the same thread handles the next pair,
  so `phi2_toep` comes from DRAM about **once per group**.

For the stack problem on the TITAN RTX, this took the block-diagonal stage from 408 ms (specialised,
one coil at a time) to 243 ms (groups of 2). The kernel then moved about 570 GB/s, against 672 GB/s
peak.

**Memory decides the group size.** Each extra coil in a group costs one padded grid (2.2 GB for the
stack problem, 3.1 GB for 3D), and the Toeplitz kernel already occupies most of the GPU. The plan
queries the cuFFT work-area size without allocating (`cufftSetAutoAllocation(0)`), adds the grids
and index map, and shrinks the group until everything fits `mem_fraction` (default 0.5) of the free
memory at plan creation. A group of 1 (the old coil-by-coil operator) is always attempted. Both cuFFT
plans (full group and the smaller last group) share one work area. On the 80 GB H100 groups of 4 fit
for every benchmark problem. Groups of 6 gain only 1–2 % more, so the default stays at 4.

### 2.4 Masked operator: in place, through an index map

A masked `phi2_toep` keeps only selected locations. The old masked path took four steps:

1. gather the selected locations into a compact `[num_masked, L]` buffer and multiply;
2. `memset` the whole padded grid;
3. scatter the results back.

That cost an extra 1.8–2 GB buffer per coil and three full passes.

The new masked kernel works like the dense one, with one thread per grid location. It looks up its
position in the mask through an `int32` map `inv[Mx, My, Mz]`:

- `inv[j] = -1`: outside the mask. The thread writes zeros for every rank and coil, which is the
  "zero outside the mask" semantics, in the same pass.
- `inv[j] = m`: the location's block is column `m` of the `[num_masked, (L^2+L)/2]` kernel, so the
  thread multiplies in place.

The map is rebuilt at every execution (a memset plus one pass over the mask, about 0.2 ms) because
the caller may pass a different mask each time. It costs 4 bytes per grid location (92 MB for the
stack problem) instead of $8L$ bytes per masked location *per coil*. For a sorted mask, consecutive
threads read consecutive kernel columns, so the reads stay coalesced.

The effect: masked operators went from slightly *slower* than unmasked to 3–11 % *faster*, and
the masked 3D problem now fits on the 24 GB TITAN RTX.

### 2.5 Things that did not help

- **Threads per block 128 vs 256** for the specialised kernels: identical times. 256 is kept, with
  `__launch_bounds__(256)`.
- **Groups larger than 4:** 1–2 % at best, for 50 % more memory.
- **Specialising $L > 30$:** see 2.2.
- **Fusing pad/crop into cuFFT callbacks** and **CUDA graphs:** not attempted. After these changes
  the two FFTs are about 70 % of the GPU time on the H100, so further gains have to come from the
  FFTs themselves.

---

## 3. Several GPUs

### 3.1 Decomposition by coil

$A^H A x = \sum_c (\ldots)_c$, and each coil's pipeline is independent. A multi-GPU plan
(`src/cuda/mr_mgpu.cu`) gives each GPU a contiguous range of whole coils and an ordinary single-GPU
plan for them, which keeps its own coil groups, specialised kernels and stream. At creation every
GPU receives a copy of `phi2_toep` (and the mask) and the sensitivity maps of its own coils. One
application then only moves `x` in and a partial result out:

```
caller's stream ── event ──┬─▶ GPU k: copy x ─ its coils ─ copy partial to the primary ─ event
                           └─▶ primary: its coils, straight into the output ─ event
caller's stream waits for all events ─▶ one kernel adds the staged partials into the output
```

A single host thread issues all of this asynchronously, with cross-device ordering through events
only. No host thread blocks, and the call returns immediately, like the single-GPU one. Every entry
point saves and restores the caller's current device, so MATLAB's "one GPU per session" model and
CuPy's current device are unaffected.

**How many GPUs.** The time is set by the busiest GPU, i.e. $\lceil C/G \rceil$ coils. The plan
uses the smallest $G' \le G$ with the same $\lceil C/G' \rceil$ (12 coils on 5 GPUs use 4), and
keeps problems below $2^{22}$ upsampled values per coil on one GPU (the 2D benchmark is
launch-bound at 0.5 ms).

**Scaling.** On three H100s the speed-up is ×2.76 (stack) to ×2.87 (3D). The rest is the copy of
`x` (≈ 1.5 ms over NVLink for the stack problem), the gather and the reduction.

### 3.2 Remainder split by rank

When $C$ is not a multiple of $G$, the last $r = C \bmod G$ coils would leave GPUs idle. Optionally
they are split across *all* GPUs by rank. Padding, FFTs and cropping are independent per rank, but
the block-diagonal stage needs all $L$ ranks of a location together, so the data is transposed
twice:

1. GPU $g$ pads and transforms its ranks $R_g$ (about $L/G$ of them);
2. **all-to-all:** GPU $g$ sends the k-space slab $K_h$ (a contiguous range of about
   $M_xM_yM_z/G$ locations) of its ranks to GPU $h$, which then holds all $L$ ranks of $K_h$;
3. GPU $h$ multiplies $K_h$ with its replicated kernel (the specialised kernels take a location
   range for this, `mr_bd_range`);
4. all-to-all back, inverse FFT and crop of $R_g$.

The copies are plain `cudaMemcpyPeerAsync` per rank. A single pitched `cudaMemcpy3DPeerAsync` would
do, but its source pitch (the whole grid) exceeds the 2 GB maximum pitch on large grids. Each GPU does
$1/G$ of the remaining coil's arithmetic, but moves its padded data twice, so the split pays only
over fast links.

**Cost model** (`MR_AHA_MGPU_SPLIT=auto`, the default). At plan creation the library measures the
GPU-to-GPU bandwidth (64 MB copies) and times the single-GPU operator on the primary over $n$ coils,
$T(n)$, *with its coil groups*. It then compares

$$
t_\text{whole} = T(\lceil C/G' \rceil), \qquad
t_\text{split} = T(\lfloor C/G \rfloor) + r\left(\frac{T(1)}{G} + \frac{2(G-1)}{G^2}\,
\frac{8 M_xM_yM_zL}{\text{bw}}\right),
$$

and splits only if $t_\text{split} < 0.95\, t_\text{whole}$. A first version timed one coil
without groups and scaled it linearly. That overstated $t_\text{whole}$, and it chose the split
where it lost. Measured on H100s, the split gains 4 % for 7 coils on 3 GPUs (3D) and loses 12 % for
8 coils on 3 GPUs, and the model picks correctly in both. On PCIe without peer access (~4 GB/s) it
never splits.

### 3.3 Robustness

- **Device resets under a live plan.** MATLAB resets a GPU on `reset(gpuDevice)` and whenever one is
  selected or deselected, which can destroy the CUDA context of the primary *and* of other GPUs of a
  plan. Using the plan's dead stream or event handles then crashes inside the driver. Freeing its old
  pointers could also free memory that a newer plan has since been given at the same addresses. Each
  device slot therefore records the unique ID of its context (`cuCtxGetId`). Release first checks that
  the primary context is active (`cuDevicePrimaryCtxGetState`, which does not create one) and that it
  is still the same instance, and skips the device otherwise. `mr_mgpu_plan_abandon` additionally
  skips the primary without probing it. The MATLAB gateway detects resets through a one-element
  sentinel gpuArray per plan.
- **Failure mid-issue:** if an execution fails after some work was issued, the call waits for the
  device streams before returning, so nothing keeps reading `x` or writing the output behind the
  caller's back.
- **Memory:** every GPU holds a full copy of the kernel, so multi-GPU does not raise the largest
  problem size. `MR_MGPU_SHARE_PRIMARY` (`share_primary` in Python/MATLAB) lets the primary GPU use
  the caller's arrays in place instead of a second copy.

---

## 4. Verification

- `tests/check_bd_kernels.py`: specialised vs generic kernels, bitwise identical (132
  combinations: $L$ = 1–33, both triangles, masked/unmasked, coil groups 1/2/4).
- `tests/check_reference.py` / `tests/check_reference_matlab.m`: unchanged reference data still met.
- `tests/check_mgpu.py` and `tests/check_mgpu_matlab.m` (run on H100s only): multi-GPU vs single-GPU
  to 6·10⁻⁸ in 192 combinations each (2D/stack/3D, 5 and 8 coils, split on/off, masked/unmasked,
  both triangles, shared/copied constants). The MATLAB test also covers device hygiene, leaks,
  `clear mex`, `reset(gpuDevice)` and switching `gpuDevice`.
- 20-iteration CG on the stack data: the solution moves by 2.5·10⁻⁶, which is the run-to-run noise
  of the unchanged NUFFT operator (2.8·10⁻⁶).
