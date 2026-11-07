"""Multi-GPU vs single-GPU :math:`A^H A` (CUDA backend).

:class:`mr_aha.MultiGPUPlan` distributes the coils over several GPUs and sums
the partial results on the primary device; only the order of the coil
summation differs from :class:`mr_aha.AhAPlan`, so the two must agree to
single-precision rounding. This test applies both on random inputs for 2D,
stack and 3D problems with coil counts that do and do not divide the number of
GPUs, unmasked and masked, both triangulations, the remainder split
(``MR_AHA_MGPU_SPLIT``) off and on, and the primary's constants copied or
shared in place, for 1 up to all visible GPUs. It also checks the device
hygiene: the current device is unchanged by every call and repeated plan
creation does not leak device memory.

The problems are small, so the size threshold below which a plan stays on the
primary device is disabled (``MR_AHA_MGPU_MIN_WORK=0``).

Usage (from the repository root; skips with fewer than 2 GPUs)::

    python tests/check_mgpu.py [--devices 0 1 2 3] [--quick]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import mr_aha  # noqa: E402

#: name -> (traj_type, n_dim, L)
PROBLEMS = {
    "2d": ("2D", (40, 36), 5),
    "stack": ("stack", (32, 30, 8), 4),
    "3d": ("3D", (20, 18, 16), 3),
}

TOL = 1e-6


def rel(a, b) -> float:
    """Relative 2-norm difference of two CuPy arrays, in double precision."""
    import cupy as cp  # noqa: PLC0415
    a = a.astype(cp.complex128)
    b = b.astype(cp.complex128)
    return float(cp.linalg.norm(a - b) / cp.linalg.norm(b))


def free_bytes(devices) -> list:
    """Free memory of each device in bytes."""
    import cupy as cp  # noqa: PLC0415
    out = []
    for d in devices:
        with cp.cuda.Device(d):
            out.append(cp.cuda.runtime.memGetInfo()[0])
    return out


def main() -> int:
    """Run every combination; return 1 on any failure."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--devices", type=int, nargs="+", default=None,
                    help="CUDA ordinals to use (default: all visible)")
    ap.add_argument("--quick", action="store_true",
                    help="fewer combinations (one triangulation, no split)")
    args = ap.parse_args()

    if mr_aha.backend() != "cuda":
        print("this test is for the CUDA backend")
        return 0
    import cupy as cp  # noqa: PLC0415

    devices = args.devices or list(range(cp.cuda.runtime.getDeviceCount()))
    if len(devices) < 2:
        print(f"skipped: needs at least 2 GPUs, {len(devices)} available")
        return 0
    os.environ["MR_AHA_MGPU_MIN_WORK"] = "0"
    primary = devices[0]
    cp.cuda.Device(primary).use()

    rng = np.random.default_rng(3)
    crandn = lambda *s: cp.asarray((rng.standard_normal(s) + 1j * rng.standard_normal(s)
                                    ).astype(np.complex64))
    worst, failed, count = 0.0, 0, 0
    tris = ("lower",) if args.quick else ("lower", "upper")
    splits = ("off",) if args.quick else ("off", "on")
    G_list = sorted({1, 2, 3, len(devices)} | ({4, 5} if len(devices) >= 5 else set()))
    G_list = [g for g in G_list if g <= len(devices)]

    for name, (tt, n_dim, L) in PROBLEMS.items():
        for n_coils, tri, masked in itertools.product((5, 8), tris, (False, True)):
            cfg = mr_aha.make_config(tt, n_dim, L, n_coils=n_coils,
                                     triangulation=tri, is_masked=masked)
            mask = None
            if masked:
                pad = int(np.prod([cfg.phi2_dim[i] for i in range(3)]))
                m = np.sort(rng.choice(pad, pad * 2 // 3, replace=False))
                mask = cp.asarray(m.astype(np.int64))
                cfg.num_masked = int(m.size)
            x = crandn(L, *reversed(n_dim))
            SEs = crandn(n_coils, *reversed(n_dim))
            toep = crandn(*mr_aha.phi2_toep_shape(cfg))
            with mr_aha.AhAPlan(cfg) as plan:
                ref = plan.apply(x, SEs, toep, mask=mask)
            for G, split, share in itertools.product(G_list, splits, (False, True)):
                os.environ["MR_AHA_MGPU_SPLIT"] = split
                dev_before = cp.cuda.runtime.getDevice()
                with mr_aha.MultiGPUPlan(cfg, SEs, toep, mask=mask,
                                         devices=devices[:G],
                                         share_primary=share) as op:
                    out = op.apply(x)
                    out2 = op.apply(x)          # plans are reusable
                    info = op.info
                cp.cuda.Device(primary).synchronize()
                err = max(rel(out, ref), rel(out2, ref))
                count += 1
                worst = max(worst, err)
                bad = []
                if not err < TOL:
                    bad.append(f"rel diff {err:.2e}")
                if cp.cuda.runtime.getDevice() != dev_before:
                    bad.append("current device changed")
                if info["devices"][0] != primary or sum(info["coil_count"]) + \
                        info["n_split_coils"] != n_coils:
                    bad.append(f"bad info {info}")
                if bad:
                    failed += 1
                    print(f"FAIL {name} coils={n_coils} {tri} masked={masked} "
                          f"G={G} split={split} share={share}: {', '.join(bad)} "
                          f"info={info}")
    os.environ.pop("MR_AHA_MGPU_SPLIT", None)

    # Leak check: repeated create/apply/destroy must return all memory.
    cfg = mr_aha.make_config("3D", (24, 24, 24), 4, n_coils=6)
    x = crandn(4, 24, 24, 24)
    SEs = crandn(6, 24, 24, 24)
    toep = crandn(*mr_aha.phi2_toep_shape(cfg))
    for _ in range(2):                  # warm-up: contexts, cuFFT, peer access
        with mr_aha.MultiGPUPlan(cfg, SEs, toep, devices=devices) as op:
            op.apply(x)
    cp.cuda.Device(primary).synchronize()
    before = free_bytes(devices)
    for _ in range(10):
        with mr_aha.MultiGPUPlan(cfg, SEs, toep, devices=devices) as op:
            op.apply(x)
    cp.cuda.Device(primary).synchronize()
    after = free_bytes(devices)
    leak = [b - a for a, b in zip(after, before)]
    if max(leak) > 8 << 20:             # allow allocator noise, not a leak
        failed += 1
        print(f"FAIL leak check: free memory dropped by {leak} bytes")

    print(f"{count} combinations on up to {len(devices)} GPUs {devices}, worst "
          f"relative difference {worst:.2e}, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
