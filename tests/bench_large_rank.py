"""Time one A^H A at a large rank with each block-diagonal kernel family.

The problem has the shape of the ``2d72`` benchmark: a 320^2 image (640^2
upsampled grid), 12 coils, L = 72, a radial trajectory of 754 spokes x 320
samples, and the neighbor mask that :func:`mr_aha.calc_mask` derives from
it. The run time of the operator does not depend on the values of the
kernel, so ``phi2_toep`` is random and no NUFFT is needed: the script runs in
seconds on both backends.

For masked and unmasked operators it times the generic block-diagonal
kernels (``MR_AHA_<BACKEND>_GENERIC_BD=1``, the only ones above the
specialized range before the tiled kernels) and the default choice (the
tiled kernels at L > 32 on Metal, L > 30 on CUDA), reports the median of
``--reps`` applications after one warm-up, and checks that both families
give the same result.

Usage (from the repository root)::

    python tests/bench_large_rank.py [--L 72] [--coils 12] [--reps 5]

Extra variants can be timed by setting environment variables before the
run, e.g. ``MR_AHA_METAL_BDT_P=16 MR_AHA_METAL_BDT_NC=2`` or
``MR_AHA_METAL_COIL_GROUP=6``; ``MR_AHA_METAL_PROFILE=1`` prints the GPU time
of each stage on Metal.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import mr_aha  # noqa: E402


def radial_traj(n_spokes: int, n_read: int) -> np.ndarray:
    """``(2, n_read, n_spokes)`` golden-angle radial trajectory in radians."""
    k = (np.arange(n_read) - n_read // 2) * (2 * np.pi / n_read)
    ang = np.arange(n_spokes) * (np.pi * (np.sqrt(5) - 1) / 2)
    return np.stack([np.outer(k, np.cos(ang)), np.outer(k, np.sin(ang))]
                    ).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=320, help="image size N (N x N)")
    ap.add_argument("--L", type=int, default=72, help="rank")
    ap.add_argument("--coils", type=int, default=12)
    ap.add_argument("--spokes", type=int, default=754)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    be = mr_aha.backend()
    cuda = be == "cuda"
    if cuda:
        import cupy as cp  # noqa: PLC0415
        xp, sync = cp, cp.cuda.Device().synchronize
    else:
        xp, sync = np, (lambda: None)
    prefix = "MR_AHA_" + be.upper()
    rng = np.random.default_rng(0)
    crandn = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s)
                         ).astype(np.complex64)
    n, L, C = args.n, args.L, args.coils
    print(f"backend {be}: {n}^2 image, L = {L}, {C} coils, "
          f"{args.spokes} spokes x {n}, median of {args.reps}")

    x = xp.asarray(crandn(L, n, n))
    SEs = xp.asarray(crandn(C, n, n))
    traj = radial_traj(args.spokes, n)
    results = {}
    for masked in (False, True):
        cfg = mr_aha.make_config("2D", (n, n), L, n_coils=C, is_masked=masked)
        mask = mr_aha.calc_mask(cfg, xp.asarray(traj)) if masked else None
        if masked:
            cfg.num_masked = int(mask.size)
        toep = xp.asarray(crandn(*mr_aha.phi2_toep_shape(cfg)))
        for label, generic in (("generic", True), ("default", False)):
            os.environ[prefix + "_GENERIC_BD"] = "1" if generic else "0"
            try:
                op = mr_aha.create_op(cfg, SEs, None, None, mask, _phi2_toep=toep)
            finally:
                os.environ.pop(prefix + "_GENERIC_BD", None)
            with op:
                y = op(x)
                sync()
                ts = []
                for _ in range(args.reps):
                    t0 = time.perf_counter()
                    y = op(x)
                    sync()
                    ts.append(time.perf_counter() - t0)
            y = cp.asnumpy(y) if cuda else np.asarray(y)
            results[(masked, label)] = y
            name = "masked" if masked else "unmasked"
            print(f"  {name:9s} {label:8s} {1e3 * statistics.median(ts):9.1f} ms")
        a, b = results[(masked, "default")], results[(masked, "generic")]
        err = float(np.linalg.norm((a - b).astype(np.complex128))
                    / np.linalg.norm(b.astype(np.complex128)))
        print(f"  {name:9s} default vs generic: relative difference {err:.1e}")
        if not err < 1e-5:
            print("  MISMATCH")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
