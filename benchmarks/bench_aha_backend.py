"""Time one Toeplitz :math:`A^H A` application on whichever backend is loaded.

Runs :class:`mr_aha.AhAPlan` at the sizes of the three benchmark problems of
``benchmarks/README.md`` (2D radial 160^2, stack-of-stars 320^2 x 56,
3D kooshball 160^3), unmasked and masked, and reports the plan-creation time
and the median time of one application. Only the speed is measured, so the
inputs are constant arrays (the operator has no data-dependent branches) and
the mask is a centred disc (2D, stack) or ball (3D) keeping the same fraction
of the upsampled grid as the benchmark's ``calc_mask`` masks (83 %, 81 %,
64 %).

Works for both backends: CuPy arrays and CUDA-event-free wall-clock timing
with a device synchronisation for ``'cuda'``, NumPy arrays for ``'metal'``
(whose calls are synchronous).

Usage (from the repository root)::

    python benchmarks/bench_aha_backend.py [--cases 2d stack 3d] [--reps 10]

Writes ``benchmarks/results/<host>/aha_<backend>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import mr_aha  # noqa: E402

#: name -> (traj_type, n_dim, L, n_coils, kept fraction of the masked run)
CASES = {
    "2d": ("2D", (160, 160), 6, 16, 0.831),
    "stack": ("stack", (320, 320, 56), 12, 12, 0.812),
    "3d": ("3D", (160, 160, 160), 12, 12, 0.637),
}


def centred_mask(cfg, keep: float) -> np.ndarray:
    """Linear indices of a centred disc/ball keeping ``keep`` of the grid.

    Args:
        cfg: Configuration; ``phi2_dim`` gives the upsampled grid.
        keep: Fraction of the grid to keep (per plane for a stack).

    Returns:
        Sorted ``int64`` 0-based column-major indices into ``[Mx, My, Mz]``.
    """
    M = [int(cfg.phi2_dim[i]) for i in range(3)]
    stack = cfg.traj_type == mr_aha.TrajType.STACK
    axes = 2 if (stack or cfg.n_spatial == 2) else 3
    # Normalised radius over the transformed axes, row-major (Mz, My, Mx).
    grids = np.meshgrid(*[(np.arange(m) - m // 2) / (m / 2) for m in reversed(M)],
                        indexing="ij")
    r2 = sum(g ** 2 for g in grids[-axes:])
    thr = np.quantile(r2[0] if axes == 2 else r2, keep)
    return np.flatnonzero((r2 <= thr).ravel()).astype(np.int64)


def run_case(name: str, masked: bool, reps: int, xp, sync) -> dict:
    """Time plan creation and ``reps`` applications of one configuration."""
    traj_type, n_dim, L, n_coils, keep = CASES[name]
    cfg = mr_aha.make_config(traj_type, n_dim, L, n_coils=n_coils,
                             is_masked=masked)
    mask = None
    if masked:
        mask_np = centred_mask(cfg, keep)
        cfg.num_masked = int(mask_np.size)
        mask = xp.asarray(mask_np)
    shape_x = (L, *reversed(n_dim))
    x = xp.full(shape_x, 0.01 + 0.02j, dtype=xp.complex64)
    SEs = xp.full((n_coils, *reversed(n_dim)), 0.5 - 0.25j, dtype=xp.complex64)
    toep = xp.full(mr_aha.phi2_toep_shape(cfg), 1e-3 + 1e-4j, dtype=xp.complex64)
    out = xp.empty_like(x)

    t0 = time.perf_counter()
    plan = mr_aha.AhAPlan(cfg)
    sync()
    t_plan = time.perf_counter() - t0
    for _ in range(2):
        plan.apply(x, SEs, toep, mask=mask, out=out)
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        plan.apply(x, SEs, toep, mask=mask, out=out)
        sync()
        ts.append(time.perf_counter() - t0)
    plan.close()
    res = dict(case=name, masked=masked, n_dim=n_dim, L=L, n_coils=n_coils,
               phi2_dim=[int(cfg.phi2_dim[i]) for i in range(cfg.n_spatial)],
               kept=(cfg.num_masked / np.prod([cfg.phi2_dim[i] for i in range(3)])
                     if masked else 1.0),
               kernel_gb=toep.nbytes / 1e9, plan_s=t_plan,
               median_s=float(np.median(ts)), min_s=float(np.min(ts)),
               reps=reps)
    print(f"{name:5s} {'masked  ' if masked else 'unmasked'} kernel "
          f"{res['kernel_gb']:5.1f} GB  plan {t_plan * 1e3:7.1f} ms  "
          f"AhA median {res['median_s'] * 1e3:8.1f} ms  min {res['min_s'] * 1e3:8.1f} ms",
          flush=True)
    del toep, x, SEs, out, mask
    return res


def main() -> None:
    """Parse arguments, run the cases, write the JSON report."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--no-masked", action="store_true")
    args = ap.parse_args()

    backend = mr_aha.backend()
    if backend == "cuda":
        import cupy as xp  # noqa: PLC0415
        sync = xp.cuda.Device().synchronize
        device = xp.cuda.runtime.getDeviceProperties(xp.cuda.Device().id)["name"].decode()
    else:
        xp = np
        sync = lambda: None  # noqa: E731 - Metal calls are synchronous
        import platform  # noqa: PLC0415
        device = platform.processor() or "Apple silicon"
        try:
            import subprocess  # noqa: PLC0415
            device = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                    capture_output=True, text=True).stdout.strip()
        except OSError:
            pass
    host = socket.gethostname().split(".")[0]
    print(f"backend {backend}, device {device}, host {host}")

    results = []
    for name in args.cases:
        for masked in ([False] if args.no_masked else [False, True]):
            try:
                results.append(run_case(name, masked, args.reps, xp, sync))
            except (MemoryError, RuntimeError) as err:
                # e.g. the 3D unmasked kernel (20 GB) on a 24 GB GPU
                print(f"{name:5s} {'masked  ' if masked else 'unmasked'} "
                      f"skipped: {type(err).__name__}: {str(err).splitlines()[0]}")
                if xp is not np:
                    xp.get_default_memory_pool().free_all_blocks()

    out_dir = Path(os.environ.get("MR_AHA_BENCH_RESULTS",
                           ROOT / "benchmarks" / "results")) / host
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"aha_{backend}.json"
    out.write_text(json.dumps(dict(backend=backend, device=device, host=host,
                                   results=results), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
