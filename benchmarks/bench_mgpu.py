"""Time one multi-GPU Toeplitz :math:`A^H A` application (CUDA backend).

Runs :class:`mr_aha.MultiGPUPlan` at the sizes of the benchmark problems of
``benchmarks/README.md`` (2D radial 160^2, stack-of-stars 320^2 x 56,
3D kooshball 160^3; constant inputs, as in ``bench_aha_backend.py``) on
``G`` = 1, 2, ... GPUs, with the remainder split (``MR_AHA_MGPU_SPLIT``) off
and in its automatic mode, and reports the median time of one application,
the speed-up over one GPU (:class:`mr_aha.AhAPlan`), the parallel efficiency
and the plan's choices (devices used, coils per device, split coils, measured
peer bandwidth).

The primary device's constants are shared in place (``share_primary``), so
the primary holds one copy of the kernel like a single-GPU run.

Usage (from the repository root)::

    python benchmarks/bench_mgpu.py [--cases stack 3d] [--gpus 1 2 4 8]
                                    [--split off auto on] [--reps 10]
                                    [--coils 8] [--out name.json]

Writes ``benchmarks/results/<host>/mgpu_cuda.json``.
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
sys.path.insert(0, str(ROOT / "benchmarks"))

import mr_aha  # noqa: E402
from bench_aha_backend import CASES, centred_mask  # noqa: E402


def timeit(fn, reps: int, sync) -> float:
    """Median wall-clock time of ``fn()`` over ``reps`` runs, after 2 warm-ups."""
    for _ in range(2):
        fn()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def run_case(name: str, masked: bool, gpus, splits, reps: int,
             coils: int = 0) -> list:
    """All GPU counts / split modes of one problem; returns result rows.

    ``coils`` > 0 replaces the problem's number of coils (e.g. a count that
    does not divide the number of GPUs, to exercise the remainder split).
    """
    import cupy as cp  # noqa: PLC0415

    traj_type, n_dim, L, n_coils, keep = CASES[name]
    n_coils = coils or n_coils
    cfg = mr_aha.make_config(traj_type, n_dim, L, n_coils=n_coils, is_masked=masked)
    mask = None
    if masked:
        mask_np = centred_mask(cfg, keep)
        cfg.num_masked = int(mask_np.size)
        mask = cp.asarray(mask_np)
    x = cp.full((L, *reversed(n_dim)), 0.01 + 0.02j, dtype=cp.complex64)
    SEs = cp.full((n_coils, *reversed(n_dim)), 0.5 - 0.25j, dtype=cp.complex64)
    toep = cp.full(mr_aha.phi2_toep_shape(cfg), 1e-3 + 1e-4j, dtype=cp.complex64)
    out = cp.empty_like(x)
    sync = cp.cuda.Device().synchronize
    label = f"{name:5s} {'masked  ' if masked else 'unmasked'} {n_coils:2d} coils"

    with mr_aha.AhAPlan(cfg) as plan:
        t1 = timeit(lambda: plan.apply(x, SEs, toep, mask=mask, out=out), reps, sync)
    print(f"{label} 1 GPU (AhAPlan)      {t1 * 1e3:8.1f} ms", flush=True)
    rows = [dict(case=name, masked=masked, n_coils=n_coils, gpus=1,
                 split="single", median_s=t1,
                 speedup=1.0, efficiency=1.0, info=None)]

    for G in gpus:
        for split in splits:
            os.environ["MR_AHA_MGPU_SPLIT"] = split
            try:
                t0 = time.perf_counter()
                op = mr_aha.MultiGPUPlan(cfg, SEs, toep, mask=mask,
                                         devices=list(range(G)), share_primary=True)
                t_plan = time.perf_counter() - t0
            except mr_aha.MRError as err:
                print(f"{label} {G} GPUs split={split:4s} failed: {err}", flush=True)
                continue
            finally:
                os.environ.pop("MR_AHA_MGPU_SPLIT", None)
            with op:
                t = timeit(lambda: op.apply(x, out=out), reps, sync)
                info = op.info
            used = len(info["devices"])
            row = dict(case=name, masked=masked, n_coils=n_coils, gpus=G,
                       split=split, median_s=t,
                       plan_s=t_plan, speedup=t1 / t, efficiency=t1 / t / used,
                       info=info)
            rows.append(row)
            print(f"{label} {G} GPUs split={split:4s} {t * 1e3:8.1f} ms  "
                  f"x{t1 / t:4.2f}  used {used}, coils {info['coil_count']}, "
                  f"split {info['n_split_coils']}, groups {info['group']}, "
                  f"peer {info['peer_access']} {info['peer_bandwidth_gbs']:.1f} GB/s, "
                  f"plan {t_plan:.1f} s", flush=True)
    del toep, x, SEs, out, mask
    cp.get_default_memory_pool().free_all_blocks()
    return rows


def main() -> None:
    """Parse arguments, run the cases, write the JSON report."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", nargs="+", default=["2d", "stack", "3d"],
                    choices=list(CASES))
    ap.add_argument("--gpus", type=int, nargs="+", default=None,
                    help="GPU counts to try (default 2..all visible)")
    ap.add_argument("--split", nargs="+", default=["off", "auto"],
                    choices=["off", "auto", "on"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--no-masked", action="store_true")
    ap.add_argument("--coils", type=int, default=0,
                    help="override the number of coils (0: the problem's)")
    ap.add_argument("--out", default=None, help="JSON file name (in results/<host>)")
    args = ap.parse_args()

    import cupy as cp  # noqa: PLC0415
    n_vis = cp.cuda.runtime.getDeviceCount()
    gpus = args.gpus or list(range(2, n_vis + 1))
    device = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    host = socket.gethostname().split(".")[0]
    print(f"device {device} x {n_vis}, host {host}")

    results = []
    for name in args.cases:
        for masked in ([False] if args.no_masked else [False, True]):
            results += run_case(name, masked, gpus, args.split, args.reps,
                                args.coils)

    out_dir = Path(os.environ.get("MR_AHA_BENCH_RESULTS",
                           ROOT / "benchmarks" / "results")) / host
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (args.out or "mgpu_cuda.json")
    out.write_text(json.dumps(dict(device=device, n_visible=n_vis, host=host,
                                   results=results), indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
