"""Checks of the :math:`A^H A` operator object, :func:`mr_aha.create_op` (CUDA).

The operator is built from the Gram blocks, the trajectory and the maps, with
either method, on one GPU or several. This script checks it against the
stateless primitives and the independent reference, on random inputs for 2D,
stack and 3D problems (odd sizes, coil counts that do and do not divide the
number of GPUs, a rank above the specialised block-diagonal kernels' limit):

**One GPU** (always run):

* Toeplitz with an injected kernel (``_phi2_toep``) is **bit-identical** to
  :func:`mr_aha.aha_toeplitz` with that kernel (unmasked and masked, both
  triangulations, ``share_primary`` on and off).
* Toeplitz with its own kernel: ``op.kernel(0)`` matches
  :func:`mr_aha.calc_toep` and ``op(x)`` matches :func:`mr_aha.aha_toeplitz`
  to cufinufft's run-to-run noise.
* NUFFT matches the reference :class:`mr_aha.aha_nufft_plan` (CuPy + the
  cufinufft Python package, no library code) to ``1e-5``: the same transforms,
  only the summation order differs. NUFFT and Toeplitz agree to the NUFFT
  tolerance.
* Argument errors are reported clearly; the current device is unchanged;
  repeated creation does not leak device memory.

**Several GPUs** (skipped with fewer than two visible GPUs; never run on the
shared workstation, see the project notes): for G = 2 .. all, every
``toep_dist`` (and the remainder split by rank) against one GPU, the kernel
copies on every GPU, NUFFT on G GPUs against one GPU, leaks, and a forced
allocation failure on another GPU, which must raise an error naming it and
leak nothing. The problems are small, so the size threshold below which an
operator stays on the primary GPU is disabled (``MR_AHA_MGPU_MIN_WORK=0``).

Usage (from the repository root)::

    CUDA_VISIBLE_DEVICES=1 python tests/check_op.py            # one GPU
    python tests/check_op.py --devices 0 1 2 3 [--quick]       # HPC node

Exits with status 1 on any failure.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import mr_aha  # noqa: E402

#: name -> (traj_type, n_dim, L, n_coils, Ntrajs, Nkx)
PROBLEMS = {
    "2d": ("2D", (40, 36), 5, 3, 60, 48),
    "2d_odd_L33": ("2D", (33, 27), 33, 5, 40, 40),
    "stack": ("stack", (32, 30, 8), 4, 3, 40, 40),
    "stack_odd": ("stack", (31, 36, 9), 3, 2, 30, 40),
    "3d": ("3D", (20, 18, 16), 3, 4, 300, 24),
    "3d_odd": ("3D", (21, 19, 15), 4, 3, 300, 24),
}

TOL_SAME = 1e-6     # same method, different summation order / cufinufft noise
TOL_REF = 1e-5      # NUFFT op vs the Python reference
TOL_METHODS = 1e-4  # NUFFT vs Toeplitz (both within the NUFFT tolerance)


def rel(a, b) -> float:
    """Relative 2-norm difference of two CuPy arrays, in double precision."""
    import cupy as cp  # noqa: PLC0415
    a = cp.asarray(a).astype(cp.complex128)
    b = cp.asarray(b).astype(cp.complex128)
    nb = float(cp.linalg.norm(b))
    return float(cp.linalg.norm(a - b)) / (nb if nb else 1.0)


def free_bytes(devices) -> list:
    """Free memory of each device in bytes (after emptying CuPy's pool)."""
    import cupy as cp  # noqa: PLC0415
    cp.get_default_memory_pool().free_all_blocks()
    out = []
    for d in devices:
        with cp.cuda.Device(d):
            cp.cuda.runtime.deviceSynchronize()
            out.append(cp.cuda.runtime.memGetInfo()[0])
    return out


def synthetic(name: str, rng):
    """Random problem: config, x, SEs, Gram blocks and trajectory.

    The Gram blocks are genuine :math:`\\Phi\\Phi^H` blocks of a random
    rank x time basis (Hermitian positive semi-definite), as the library
    expects; for a stack they also vary with kz.

    Args:
        name: Key of :data:`PROBLEMS`.
        rng: NumPy random generator.

    Returns:
        ``(cfg_kwargs, x, SEs, phi2, traj)``; arrays are CuPy, row-major.
    """
    import cupy as cp  # noqa: PLC0415
    tt, n, L, C, Ntr, Nkx = PROBLEMS[name]
    nkz = n[2] if tt == "stack" else 1
    ndim = 3 if tt == "3D" else 2
    traj = rng.uniform(-np.pi, np.pi, (ndim, Nkx, Ntr)).astype(np.float32)
    phi2 = np.empty((L, L, nkz, Ntr), np.complex64)
    for t in range(Ntr):
        for z in range(nkz):
            P = rng.standard_normal((L, 3)) + 1j * rng.standard_normal((L, 3))
            # Row-major phi2[j, i] holds column-major entry (i, j) of P P^H.
            phi2[:, :, z, t] = (P @ P.conj().T).T / 3
    if tt != "stack":
        phi2 = phi2[:, :, 0, :]
    img = tuple(reversed(n))
    x = rng.standard_normal((L,) + img) + 1j * rng.standard_normal((L,) + img)
    S = rng.standard_normal((C,) + img) + 1j * rng.standard_normal((C,) + img)
    kw = dict(traj_type=tt, n_dim=n, L=L, n_coils=C)
    return (kw, cp.asarray(x.astype(np.complex64)), cp.asarray(S.astype(np.complex64)),
            cp.asarray(np.ascontiguousarray(phi2)), cp.asarray(traj))


def make_cfg(kw: dict, **over):
    """:func:`mr_aha.make_config` from :func:`synthetic`'s keywords."""
    k = dict(kw)
    tt, n, L = k.pop("traj_type"), k.pop("n_dim"), k.pop("L")
    k.update(over)
    return mr_aha.make_config(tt, n, L, **k)


class Report:
    """Collects check results and prints one line per check."""

    def __init__(self):
        self.failed = 0
        self.count = 0

    def ok(self, cond: bool, msg: str) -> None:
        """Record one check; print it (``FAIL`` lines always)."""
        self.count += 1
        if not cond:
            self.failed += 1
        print(f"  {'ok  ' if cond else 'FAIL'} {msg}", flush=True)

    def close_to(self, a, b, tol: float, msg: str) -> None:
        """Record ``rel(a, b) <= tol``."""
        e = rel(a, b)
        self.ok(e <= tol, f"{msg}: {e:.2e} (tol {tol:.0e})")

    def raises(self, exc, fn, msg: str, contains: str = "") -> None:
        """Record that ``fn()`` raises ``exc`` (whose text contains ``contains``)."""
        try:
            fn()
        except exc as e:  # noqa: BLE001
            self.ok(contains.lower() in str(e).lower(),
                    f"{msg}: {type(e).__name__}: {str(e)[:110]}")
            return
        except Exception as e:  # noqa: BLE001
            self.ok(False, f"{msg}: raised {type(e).__name__} instead: {e}")
            return
        self.ok(False, f"{msg}: no exception")


def single_gpu(rep: Report, rng, quick: bool) -> None:
    """The one-GPU checks (see the module docstring)."""
    import cupy as cp  # noqa: PLC0415
    names = ["2d", "stack", "3d"] if quick else list(PROBLEMS)
    dev0 = cp.cuda.runtime.getDevice()
    for name in names:
        kw, x, S, phi2, traj = synthetic(name, rng)
        print(f"[1 GPU] {name}", flush=True)
        for tri in ("lower", "upper"):
            cfg = make_cfg(kw, triangulation=tri)
            K = mr_aha.calc_toep(cfg, phi2, traj)
            ref = mr_aha.aha_toeplitz(cfg, x, S, K)
            for share in (True, False):
                with mr_aha.create_op(cfg, S, phi2, traj, _phi2_toep=K,
                                      share_primary=share) as op:
                    y = op(x)
                    rep.ok(bool(cp.all(y == ref)),
                           f"toeplitz {tri} share={share}: injected kernel "
                           f"bit-identical to aha_toeplitz")
                    rep.ok(op.info["toep_source"] == "supplied", "toep_source supplied")
            with mr_aha.create_op(cfg, S, phi2, traj) as op:
                rep.close_to(op.kernel(0), K, TOL_SAME, f"toeplitz {tri}: kernel(0) vs calc_toep")
                y = op(x)
                rep.close_to(y, ref, TOL_SAME, f"toeplitz {tri}: op(x) vs aha_toeplitz")
                inf = op.info
                rep.ok(inf["devices"] == [dev0] and inf["toep_source"] == "broadcast"
                       and inf["method"] == "toeplitz",
                       f"toeplitz info: {inf['devices']} {inf['toep_source']}")
                # out= and the call form.
                out = cp.empty_like(x)
                rep.ok(op.apply(x, out=out) is out and bool(cp.all(out == y)),
                       "apply(out=) writes into out")
            if not quick and tri == "lower" and name in ("2d", "stack", "3d"):
                # Masked Toeplitz.
                cfgm = make_cfg(kw, is_masked=True)
                mask = mr_aha.calc_mask(cfgm, traj)
                Km = mr_aha.calc_toep(cfgm, phi2, traj, mask)
                refm = mr_aha.aha_toeplitz(cfgm, x, S, Km, mask)
                with mr_aha.create_op(cfgm, S, phi2, traj, mask, _phi2_toep=Km) as op:
                    rep.ok(bool(cp.all(op(x) == refm)),
                           "masked toeplitz: injected kernel bit-identical")
                with mr_aha.create_op(cfgm, S, phi2, traj, mask) as op:
                    rep.close_to(op(x), refm, TOL_SAME, "masked toeplitz: own kernel")

        cfg = make_cfg(kw)
        with mr_aha.aha_nufft_plan(cfg, traj) as P:
            r = P.apply(x, S, phi2)
        for share in (True, False):
            with mr_aha.create_op(cfg, S, phi2, traj, method="nufft",
                                  share_primary=share) as op:
                y = op(x)
                rep.close_to(y, r, TOL_REF, f"nufft share={share} vs aha_nufft_plan")
                rep.ok(op.info["toep_source"] == "none" and op.info["method"] == "nufft",
                       "nufft info")
                rep.raises(mr_aha.MRError, lambda: op.kernel(0), "nufft kernel()",
                           "no kernel")
        K = mr_aha.calc_toep(cfg, phi2, traj)
        rep.close_to(y, mr_aha.aha_toeplitz(cfg, x, S, K), TOL_METHODS,
                     "nufft vs toeplitz")
        with mr_aha.create_op(cfg, S, phi2, traj, method="nufft",
                              nufft_opts={"upsampfac": 2.0, "gpu_method": 1}) as op:
            rep.close_to(op(x), r, TOL_REF, "nufft gpu_method=1 vs aha_nufft_plan")
        rep.ok(cp.cuda.runtime.getDevice() == dev0, "current device unchanged")

    # --- argument errors ------------------------------------------------------
    print("[1 GPU] argument errors", flush=True)
    kw, x, S, phi2, traj = synthetic("2d", rng)
    cfg = make_cfg(kw)
    rep.raises(ValueError, lambda: mr_aha.create_op(cfg, S, phi2, traj, method="dft"),
               "unknown method", "method")
    cfgm = make_cfg(kw, is_masked=True)
    mask = mr_aha.calc_mask(cfgm, traj)
    rep.raises(ValueError, lambda: mr_aha.create_op(cfgm, S, phi2, traj, mask,
                                                    method="nufft"),
               "masked nufft", "mask")
    rep.raises(ValueError, lambda: mr_aha.create_op(cfgm, S, phi2, traj),
               "masked without mask", "mask")
    rep.raises(ValueError, lambda: mr_aha.create_op(cfg, S, phi2, traj,
                                                    nufft_opts={"upsampfac": 2}),
               "nufft_opts with toeplitz", "nufft")
    rep.raises(ValueError, lambda: mr_aha.create_op(cfg, S, phi2, traj, method="nufft",
                                                    nufft_opts={"spread": 1}),
               "unknown nufft_opts", "nufft_opts")
    rep.raises(ValueError, lambda: mr_aha.create_op(cfg, S, phi2, traj, devices="some"),
               "devices string", "devices")
    rep.raises(mr_aha.MRError, lambda: mr_aha.create_op(cfg, S, phi2, traj, devices=[99]),
               "invalid device ordinal", "not visible")
    rep.raises(ValueError, lambda: mr_aha.create_op(cfg, S, phi2, traj, toep_dist="ring"),
               "unknown toep_dist", "toep_dist")
    rep.raises(ValueError, lambda: mr_aha.create_op(cfg, S[:1], phi2, traj),
               "wrong SEs size", "SEs")
    rep.raises(TypeError, lambda: mr_aha.AhAOp(cfg), "AhAOp() constructor", "create_op")
    with mr_aha.create_op(cfg, S, phi2, traj, devices=np.int64(dev0)) as op:
        rep.ok(op.info["devices"] == [dev0], "devices = NumPy integer")
    with mr_aha.create_op(cfg, S, phi2, traj) as op:
        L0 = cfg.L
        cfg.L = L0 + 1          # the operator keeps its own copy of cfg
        try:
            rep.ok(op.config.L == L0 and op(x).shape == x.shape,
                   "operator unaffected by a later change of the caller's cfg")
        finally:
            cfg.L = L0
    os.environ["MR_AHA_TOEP_SPLIT"] = "bogus"
    rep.raises(mr_aha.MRError, lambda: mr_aha.create_op(cfg, S, phi2, traj),
               "MR_AHA_TOEP_SPLIT=bogus", "MR_AHA_TOEP_SPLIT")
    os.environ.pop("MR_AHA_TOEP_SPLIT")
    op = mr_aha.create_op(cfg, S, phi2, traj, method="nufft")
    op.close()
    op.close()  # idempotent
    rep.raises(RuntimeError, lambda: op(x), "apply after close", "closed")
    rep.raises(ValueError, lambda: mr_aha.create_op(cfg, S, phi2, traj)(x[:1]),
               "wrong x size", "x")

    # --- leak check -----------------------------------------------------------
    print("[1 GPU] leak check", flush=True)
    for method in ("toeplitz", "nufft"):
        with mr_aha.create_op(cfg, S, phi2, traj, method=method) as op:
            op(x)
        before = free_bytes([dev0])
        for _ in range(5):
            with mr_aha.create_op(cfg, S, phi2, traj, method=method) as op:
                op(x)
        after = free_bytes([dev0])
        rep.ok(before[0] - after[0] <= 8 << 20,
               f"{method}: free memory change {(before[0] - after[0]) / 1e6:.1f} MB "
               f"over 5 creations")


def multi_gpu(rep: Report, rng, devices: list, quick: bool) -> None:
    """The multi-GPU checks (see the module docstring)."""
    import cupy as cp  # noqa: PLC0415
    os.environ["MR_AHA_MGPU_MIN_WORK"] = "0"
    primary = devices[0]
    cp.cuda.Device(primary).use()
    names = ["2d", "stack", "3d"] if quick else list(PROBLEMS)
    for name in names:
        kw, x, S, phi2, traj = synthetic(name, rng)
        C = kw["n_coils"]
        for masked in ((False,) if quick else (False, True)):
            cfg = make_cfg(kw, is_masked=masked)
            mask = mr_aha.calc_mask(cfg, traj) if masked else None
            K = mr_aha.calc_toep(cfg, phi2, traj, mask)
            ref = mr_aha.aha_toeplitz(cfg, x, S, K, mask)
            with mr_aha.aha_nufft_plan(make_cfg(kw), traj) as P:
                rnu = P.apply(x, S, phi2)
            for G in range(2, len(devices) + 1):
                devs = devices[:G]
                tag = f"[{G} GPUs] {name}{' masked' if masked else ''} C={C}"
                print(tag, flush=True)
                for dist in ("auto", "split", "replicate", "broadcast"):
                    with mr_aha.create_op(cfg, S, phi2, traj, mask, devices=devs,
                                          toep_dist=dist) as op:
                        inf = op.info
                        rep.close_to(op(x), ref, TOL_SAME,
                                     f"toeplitz {dist} ({inf['toep_source']}, "
                                     f"{len(inf['devices'])} GPUs) vs 1 GPU")
                        if dist in ("split", "replicate", "broadcast") and len(inf["devices"]) > 1:
                            rep.ok(inf["toep_source"] == dist,
                                   f"toep_source {inf['toep_source']} == {dist}")
                        k0 = op.kernel(0)
                        rep.close_to(k0, K, TOL_SAME, f"{dist}: primary kernel vs calc_toep")
                        for j in range(1, len(inf["devices"])):
                            with cp.cuda.Device(inf["devices"][j]):
                                kj = cp.asarray(op.kernel(j))
                            kj = cp.asarray(kj.get())
                            if inf["toep_source"] in ("split", "broadcast"):
                                rep.ok(bool(cp.all(kj == k0)),
                                       f"{dist}: kernel copy on device "
                                       f"{inf['devices'][j]} bit-identical")
                            else:
                                rep.close_to(kj, k0, TOL_SAME, f"{dist}: kernel copy {j}")
                        rep.ok(cp.cuda.runtime.getDevice() == primary,
                               "current device unchanged")
                if not quick and C % G != 0:
                    # Forced remainder split: every GPU keeps the kernel it
                    # helped compute (no later copy from the primary).
                    os.environ["MR_AHA_MGPU_SPLIT"] = "on"
                    for dist in ("auto", "split", "replicate"):
                        with mr_aha.create_op(cfg, S, phi2, traj, mask, devices=devs,
                                              toep_dist=dist) as op:
                            inf = op.info
                            rep.close_to(op(x), ref, TOL_SAME,
                                         f"remainder split {dist} "
                                         f"({inf['n_split_coils']} coil(s) by rank, "
                                         f"{len(inf['devices'])} GPUs) vs 1 GPU")
                            k0 = op.kernel(0)
                            for j in range(1, len(inf["devices"])):
                                with cp.cuda.Device(inf["devices"][j]):
                                    kj = cp.asarray(op.kernel(j).get())
                                rep.close_to(kj, k0, TOL_SAME,
                                             f"remainder split {dist}: kernel on "
                                             f"device {inf['devices'][j]}")
                    os.environ.pop("MR_AHA_MGPU_SPLIT")
                if not masked:
                    with mr_aha.create_op(cfg, S, phi2, traj, method="nufft",
                                          devices=devs) as op:
                        y = op(x)
                        rep.close_to(y, rnu, TOL_REF,
                                     f"nufft on {len(op.info['devices'])} GPUs vs "
                                     f"aha_nufft_plan")
                    with mr_aha.create_op(cfg, S, phi2, traj, method="nufft") as op1:
                        rep.close_to(y, op1(x), TOL_SAME, "nufft G GPUs vs 1 GPU")

    # --- 'too small for several GPUs' rule ------------------------------------
    # A NUFFT operator's work also grows with the sample count: with the
    # threshold just above the Toeplitz work (grid x L), a trajectory with
    # more samples than grid points still uses several GPUs for NUFFT.
    kw, x, S, phi2, traj = synthetic("2d", rng)
    traj = cp.ascontiguousarray(cp.concatenate([traj] * 4, axis=2))
    phi2 = cp.ascontiguousarray(cp.concatenate([phi2] * 4, axis=2))
    cfg = make_cfg(kw)
    grid = int(np.prod(list(cfg.phi2_dim)[:2]))
    rep.ok(traj.shape[1] * traj.shape[2] > grid, "test set-up: more samples than grid")
    os.environ["MR_AHA_MGPU_MIN_WORK"] = str(grid * kw["L"] + 1)
    with mr_aha.create_op(cfg, S, phi2, traj, devices=devices) as op:
        rep.ok(len(op.info["devices"]) == 1,
               f"small toeplitz stays on one GPU: {op.info['devices']}")
    with mr_aha.create_op(cfg, S, phi2, traj, method="nufft", devices=devices) as op:
        rep.ok(len(op.info["devices"]) > 1,
               f"nufft with many samples uses several GPUs: {op.info['devices']}")
    os.environ["MR_AHA_MGPU_MIN_WORK"] = "0"

    # --- 'all' and leaks --------------------------------------------------------
    kw, x, S, phi2, traj = synthetic("stack", rng)
    cfg = make_cfg(kw, n_coils=kw["n_coils"])
    with mr_aha.create_op(cfg, S, phi2, traj, devices="all") as op:
        rep.ok(len(op.info["devices"]) >= 1, f"devices='all': {op.info['devices']}")
    for method in ("toeplitz", "nufft"):
        with mr_aha.create_op(cfg, S, phi2, traj, method=method, devices=devices) as op:
            op(x)
        before = free_bytes(devices)
        for _ in range(4):
            with mr_aha.create_op(cfg, S, phi2, traj, method=method,
                                  devices=devices) as op:
                op(x)
        after = free_bytes(devices)
        leak = [b - a for a, b in zip(after, before)]
        rep.ok(max(leak) <= 8 << 20, f"{method} on {devices}: free memory change "
               f"{[round(v / 1e6, 1) for v in leak]} MB")

    # --- forced failure on another GPU ---------------------------------------
    # A device every operator below uses: with 3 coils on 4 GPUs the NUFFT
    # method uses only 3 GPUs (the Toeplitz method also lets the 4th help
    # compute the kernel), so the last device is not always involved.
    victim = devices[1]
    for method in ("toeplitz", "nufft"):
        before = free_bytes(devices)
        with cp.cuda.Device(victim):
            free = cp.cuda.runtime.memGetInfo()[0]
            hog = cp.cuda.runtime.malloc(max(free - (4 << 20), 0))
        try:
            rep.raises(mr_aha.MRError,
                       lambda: mr_aha.create_op(cfg, S, phi2, traj, method=method,
                                                devices=devices),
                       f"{method}: allocation failure on device {victim}",
                       f"device {victim}")
        finally:
            with cp.cuda.Device(victim):
                cp.cuda.runtime.free(hog)
        rep.ok(cp.cuda.runtime.getDevice() == primary, "current device unchanged")
        after = free_bytes(devices)
        leak = [b - a for a, b in zip(after, before)]
        rep.ok(max(leak) <= 8 << 20, f"{method}: no leak after the failure "
               f"{[round(v / 1e6, 1) for v in leak]} MB")
    os.environ.pop("MR_AHA_MGPU_MIN_WORK", None)


def main() -> int:
    """Run the checks; return 1 on any failure."""
    import cupy as cp  # noqa: PLC0415
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--devices", type=int, nargs="*", default=None,
                    help="CUDA ordinals for the multi-GPU checks (default: all "
                         "visible); the first is the primary")
    ap.add_argument("--quick", action="store_true", help="fewer problems")
    ap.add_argument("--single-only", action="store_true",
                    help="skip the multi-GPU checks")
    args = ap.parse_args()
    if mr_aha.backend() != "cuda":
        print("check_op.py needs the CUDA backend")
        return 1
    rng = np.random.default_rng(7)
    rep = Report()
    ndev = cp.cuda.runtime.getDeviceCount()
    devices = args.devices if args.devices else list(range(ndev))
    cp.cuda.Device(devices[0]).use()
    single_gpu(rep, rng, args.quick)
    if args.single_only or len(devices) < 2:
        print("multi-GPU checks skipped (fewer than 2 GPUs or --single-only)")
    else:
        multi_gpu(rep, rng, devices, args.quick)
    print(f"{rep.count - rep.failed}/{rep.count} checks passed")
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
