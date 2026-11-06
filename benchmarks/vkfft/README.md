# VkFFT-Metal feasibility check for the Apple-silicon backend

The Toeplitz operator spends its FFT time on one forward and one inverse batched C2C transform per
coil, on the zero-padded grid `[Mx, My, (Mz), L]` (column-major, batch `L`, unnormalised in both
directions, exactly as `cufftExecC2C` in `src/core/mr_aha.cu`). Before porting the operator to
Metal, this folder checks that VkFFT's Metal backend computes that transform accurately and fast
on the Mac Studio, at the sizes of the three benchmark problems in [`../README.md`](../README.md).

| file | purpose |
|---|---|
| `vkfft_bench.cpp`, `build.sh` | VkFFT (Metal backend) on a shared `MTL::Buffer`: plan time, round-trip error, GPU time (command-buffer timestamps) of forward, inverse and forward+inverse |
| `check_vkfft.py` | runs `vkfft_bench` on fixed-seed data, compares its forward output against a double-precision SciPy FFT, and times MLX's `mx.fft` on the same data |
| `cufft_bench.py` | the same transform with cuFFT through CuPy, for the NVIDIA baseline |
| `results_AW650523HJY.json` | Mac Studio results (VkFFT default `coalescedMemory`) |
| `../results/pact400-recon3/cufft_bench.json` | TITAN RTX cuFFT results |

VkFFT and metal-cpp come from the `third_party/VkFFT` submodule (the `mikecjz/VkFFT` fork, see
[below](#known-vkfft-issue-double-free-with-autorelease-pools)). The results table was measured
with upstream v1.3.4.

```bash
# on the Mac, from the repository root
benchmarks/vkfft/build.sh
.venv/bin/python benchmarks/vkfft/check_vkfft.py            # all cases
benchmarks/vkfft/vkfft_bench --dims 640 640 56 --batch 12 --coalesced 32
# on a CUDA machine
python benchmarks/vkfft/cufft_bench.py
```

## Results (2026-11-05)

Mac Studio: Apple M3 Ultra, 512 GB, macOS 26.7.2, VkFFT v1.3.4, MLX 0.31.2.
NVIDIA baseline: TITAN RTX (`pact400-recon3`), cuFFT 11.3 via CuPy 14.2. Times are medians of 20
runs after warm-up, in ms. VkFFT and cuFFT times are GPU times; MLX times are wall-clock times of
`mx.eval`, so they include dispatch overhead.

| case | grid × L | size | cuFFT TITAN RTX fwd+inv | VkFFT M3 Ultra fwd+inv (default / `coalescedMemory=32`) | MLX `mx.fft` fwd+inv |
|---|---|---|---|---|---|
| 2d | 320² × 6 | 4.9 MB | 0.083 | 0.12 / – | 0.41 |
| stack | 640² × 56 × 12 | 2.2 GB | 54.1 | 51.0 / **37.9** | 73.6 |
| 3d | 320³ × 12 | 3.1 GB | 78.2 | 53.9 / 54.0 | 104.9 |

`coalescedMemory = 128` is worse (stack 51.0, 3d 91.5 ms), so the library will use 32 bytes,
the value mlx-nufft also measured best on Apple GPUs.

Accuracy of the forward transform against a double-precision SciPy FFT (relative L2 error) and the
round-trip error `||ifft(fft(x))/N − x|| / ||x||`:

| case | VkFFT forward | VkFFT round trip | MLX forward | MLX round trip | cuFFT round trip |
|---|---|---|---|---|---|
| 2d | 2.9e-7 | 3.1e-7 | 4.6e-7 | 6.6e-7 | 3.8e-7 |
| stack | 3.5e-7 | 3.7e-7 | 5.6e-7 | 6.9e-7 | 4.0e-7 |
| 3d | 3.8e-7 | 3.8e-7 | 6.1e-7 | 8.7e-7 | 4.9e-7 |

VkFFT plan creation (it generates and compiles its Metal kernels at run time) takes 2 ms (2d),
240 ms (stack) and 305 ms (3d); a plan is built once per `AhAPlan`.

**Conclusion.** VkFFT-Metal matches single-precision cuFFT accuracy and is 1.4–1.5× faster than
cuFFT on the TITAN RTX for the two large grids, and about 1.9× faster than MLX's FFT there. It is the FFT
backend for the Metal port. The 2D grid is too small to matter (0.1 ms per coil).

## Known VkFFT issue: double free with autorelease pools

VkFFT v1.3.4's Metal backend calls `release()` on objects it obtained autoreleased
(`NS::String::string`, `MTL::CommandQueue::commandBuffer`, the encoders) during
`initializeVkFFT`. Draining the autorelease pool that was active during plan creation therefore
double-frees them and segfaults. This is upstream issue
[DTolm/VkFFT#226](https://github.com/DTolm/VkFFT/issues/226), with an unmerged fix in PR
[#227](https://github.com/DTolm/VkFFT/pull/227). MATLAB and Python drain their pools, so the
library cannot live with it.

**Fix in use.** The `third_party/VkFFT` submodule points to the fork
[mikecjz/VkFFT](https://github.com/mikecjz/VkFFT), branch `mr_aha`: upstream `develop`
(`e687182`, two bug-fix releases after v1.3.4) plus only the Metal commit of PR #227
(`bd406e8`, authored by nweston), plus the push-constant fix below (`2f9fb8f`). The PR's second commit (an unrelated NVRTC change) is left out.
With the fix, VkFFT no longer releases autoreleased objects, so every call into VkFFT must run
inside an autorelease pool, which the library provides around each API entry point.
`vkfft_bench --replan N` exercises exactly that life cycle (create, run, destroy, drain) N times.
Drop the fork once upstream merges the fix.

**Second fork patch (`2f9fb8f`).** VkFFT's Metal backend kept each axis's push constants in one
shared `MTLBuffer` that it rewrote on the CPU before every dispatch. The GPU reads that buffer only
when the command buffer runs, so all dispatches encoded into one command buffer saw the values of
the last one. This broke launch-time offsets (`specifyOffsetsAtLaunch`) and would equally break
VkFFT's own split dispatches (`workGroupShift`) on very large grids. The fork now passes the push
constants with `setBytes`, which copies them at encode time.

**Size limit.** VkFFT switches to 64-bit addressing for buffers of 2^34 bytes (2^31 complex
values) or more, and that Metal code does not compile. An attempt to split such batches into
launches at buffer offsets (which is what exposed the push-constant bug) worked on small tests
but not at full size: the offsets enter VkFFT's 32-bit index arithmetic, and in a 20 GB 3D batch
(78 x 320^3) every volume past 2^31 values came out wrong. The library therefore rejects FFT
batches of 2^31 values or more (`src/metal/mr_metal_fft.cpp`) and `calc_toep` chunks below it.

Re-checked with the fork (`coalescedMemory=32`): `--replan 50` (2d) and `--replan 3` (stack, 3d)
pass; forward error vs SciPy 2.5e-7 / 2.9e-7 / 3.2e-7 and fwd+inv 0.12 / 38.1 / 54.1 ms
(2d / stack / 3d), the same as v1.3.4.
