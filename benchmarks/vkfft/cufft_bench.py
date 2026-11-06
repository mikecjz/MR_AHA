"""cuFFT timing of the batched FFTs inside :math:`A^HA` (baseline for VkFFT).

``mr_aha.cu`` applies, per coil, one forward and one inverse in-place batched
complex-to-complex FFT on the zero-padded grid: column-major ``[Mx, My,
(Mz)]`` batched over the ``L`` ranks, i.e. a row-major ``(L, [Mz,] My, Mx)``
array transformed over its last 2 (2D) or 3 (3D, stack) axes. The inverse is
unnormalised (``CUFFT_INVERSE``), exactly as in the library.

This script times that FFT with cuFFT through CuPy's ``PlanNd`` (the plan is
built once and reused, as ``mr_aha_plan_create`` does) for the three benchmark
problems, using CUDA events after a warm-up, and reports the median time of one
forward, one inverse and one forward + inverse. It also checks the round-trip
error ``||ifft(fft(a)) / N - a|| / ||a||``.

Cases (upsampled Toeplitz grids of ``benchmarks/README.md``):

=========  ==================  =====
name       grid (Mx, My, Mz)   batch
=========  ==================  =====
``2d``     320 x 320           6
``stack``  640 x 640 x 56      12
``3d``     320 x 320 x 320     12
=========  ==================  =====

Usage::

    CUDA_VISIBLE_DEVICES=1 python benchmarks/vkfft/cufft_bench.py \\
        [--out benchmarks/results/pact400-recon3/cufft_bench.json] [--reps 20]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cupy as cp
import cupyx.scipy.fft as cufft_sp
import numpy as np
from cupy.cuda import cufft

ROOT = Path(__file__).resolve().parents[2]

#: (name, column-major grid (Mx, My[, Mz]), batch L)
CASES = [
    ("2d", (320, 320), 6),
    ("stack", (640, 640, 56), 12),
    ("3d", (320, 320, 320), 12),
]


def time_ms(fn, reps: int, warmup: int = 3) -> float:
    """Median wall time of ``fn`` on the GPU in milliseconds (CUDA events).

    Args:
        fn: Callable that enqueues GPU work on the current stream.
        reps: Number of timed repetitions.
        warmup: Untimed calls before timing.

    Returns:
        Median elapsed time in ms.
    """
    for _ in range(warmup):
        fn()
    cp.cuda.Device().synchronize()
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    ts = []
    for _ in range(reps):
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        ts.append(cp.cuda.get_elapsed_time(start, stop))
    return float(np.median(ts))


def bench_case(name: str, grid, batch: int, reps: int) -> dict:
    """Time forward, inverse and forward + inverse for one case.

    Args:
        name: Case label.
        grid: Column-major grid ``(Mx, My[, Mz])``.
        batch: Number of batched transforms (``L``).
        reps: Timed repetitions per measurement.

    Returns:
        Dict with the case description, median times in ms and the round-trip
        relative error.
    """
    shape = (batch,) + tuple(reversed(grid))          # row-major (L, [Mz,] My, Mx)
    axes = tuple(range(-len(grid), 0))
    rng = cp.random.default_rng(0)
    a = (rng.standard_normal(shape, dtype=cp.float32)
         + 1j * rng.standard_normal(shape, dtype=cp.float32)).astype(cp.complex64)
    a0 = a.copy()
    plan = cufft_sp.get_fft_plan(a, axes=axes, value_type="C2C")

    def fwd():
        plan.fft(a, a, cufft.CUFFT_FORWARD)

    def inv():
        plan.fft(a, a, cufft.CUFFT_INVERSE)

    def both():
        fwd()
        inv()

    # Round-trip check before timing (the timed loops leave ``a`` scaled).
    both()
    n = int(np.prod(grid))
    err = float(cp.linalg.norm(a / n - a0) / cp.linalg.norm(a0))
    del a0
    res = dict(
        case=name, grid_colmajor=list(grid), batch=batch,
        rowmajor_shape=list(shape), bytes=int(a.nbytes),
        forward_ms=time_ms(fwd, reps), inverse_ms=time_ms(inv, reps),
        fwd_inv_ms=time_ms(both, reps), roundtrip_rel_err=err,
    )
    del a, plan
    cp.get_default_memory_pool().free_all_blocks()
    return res


def main() -> None:
    """Run all cases, print a table and write the JSON report."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path,
                    default=ROOT / "benchmarks/results/pact400-recon3/cufft_bench.json")
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    info = dict(gpu=props["name"].decode(), cupy=cp.__version__,
                cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                cufft=cufft.getVersion(), reps=args.reps,
                note="in-place C2C complex64, unnormalised inverse, plan reused; "
                     "median of CUDA-event timings after 3 warm-up calls")
    results = []
    print(f"{info['gpu']}  (cuFFT {info['cufft']})")
    print(f"{'case':6s} {'grid':>14s} {'L':>3s} {'GB':>6s} {'fwd ms':>8s} "
          f"{'inv ms':>8s} {'fwd+inv':>8s} {'roundtrip':>10s}")
    for name, grid, batch in CASES:
        r = bench_case(name, grid, batch, args.reps)
        results.append(r)
        print(f"{name:6s} {'x'.join(map(str, grid)):>14s} {batch:3d} "
              f"{r['bytes'] / 1e9:6.2f} {r['forward_ms']:8.2f} {r['inverse_ms']:8.2f} "
              f"{r['fwd_inv_ms']:8.2f} {r['roundtrip_rel_err']:10.1e}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(info=info, results=results), indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
