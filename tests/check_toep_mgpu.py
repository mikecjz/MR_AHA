"""Multi-GPU pre-computation of ``phi2_toep`` (CUDA backend).

Checks the two multi-GPU paths of the Toeplitz kernel against the
single-GPU :func:`mr_aha.calc_toep` and the stored references in
``data/ref_mlx/``:

``calc_toep(..., devices=...)`` (``mr_calc_toep_mgpu``)
    The components are split over the GPUs and gathered on the primary. The
    result must equal the single-GPU kernel up to cufinufft's run-to-run
    noise (~1e-7) and the stored reference within the ``toep`` tolerance of
    ``tests/check_reference.py``.
``MultiGPUPlan.from_gram`` (``mr_mgpu_plan_create_gram``)
    Every GPU computes a share into its own copy and the shares are
    exchanged. In ``split`` mode every copy must be **bit-identical** to the
    primary's (each component is computed once); in ``replicate`` mode the
    copies agree to the noise level. The operator must match a plan built
    from the single-GPU kernel.

Every case runs with ``MR_AHA_TOEP_SPLIT`` = ``split``, ``replicate`` and
``auto`` (``off`` for the plan as well), for 2 up to all given GPUs, on the
reference problems (2D / 3D / stack, odd and even sizes, both
triangulations, unmasked and both masks, L = 1 with fewer components than
GPUs) and on synthetic L = 12 and L = 72 problems. It also checks the device
hygiene: the current device is unchanged, repeated calls do not leak device
memory, and a failure on another GPU (its memory filled on purpose) raises a
clean error naming that GPU and leaks nothing.

Usage (from the repository root; skips with fewer than 2 GPUs; never on the
shared workstation, see ``benchmarks/hpc/run_mgpu_h100.sh``)::

    python tests/check_toep_mgpu.py [--devices 0 1 2 3] [--quick]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

import mr_aha  # noqa: E402
from check_reference import configs, make_cfg  # noqa: E402

#: Single- vs multi-GPU kernel: cufinufft's run-to-run noise is ~1e-7.
TOL_SAME = 1e-6
#: Multi-GPU kernel vs the stored reference (``check_reference.py``'s toep).
TOL_REF = 1e-5
#: Operator from the plan's kernel vs from the single-GPU kernel.
TOL_AHA = 1e-6


def rel(a, b) -> float:
    """Relative 2-norm difference of two CuPy arrays, in double precision."""
    import cupy as cp  # noqa: PLC0415
    a = a.astype(cp.complex128)
    b = b.astype(cp.complex128)
    return float(cp.linalg.norm(a - b) / cp.linalg.norm(b))


def free_bytes(devices) -> list:
    """Free memory of each device in bytes (CuPy's pool emptied first)."""
    import cupy as cp  # noqa: PLC0415
    out = []
    for d in devices:
        with cp.cuda.Device(d):
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.Device(d).synchronize()
            out.append(cp.cuda.runtime.memGetInfo()[0])
    return out


def synthetic(L: int, n_dim, n_coils: int, rng):
    """A random 2D problem with Hermitian Gram blocks.

    Args:
        L: Rank.
        n_dim: Image size ``(Nx, Ny)``.
        n_coils: Number of coils.
        rng: NumPy random generator.

    Returns:
        A dict shaped like a reference ``.npz`` (``phi2``, ``traj``, ``x``,
        ``SEs`` and the configuration keys), without stored kernels.
    """
    Ntrajs, Nkx = 24, 2 * n_dim[0]
    P = rng.standard_normal((Ntrajs, L, 3)) + 1j * rng.standard_normal((Ntrajs, L, 3))
    G = np.einsum("tla,tma->tlm", P, P.conj())           # Hermitian, [Ntrajs, L, L]
    phi2 = np.ascontiguousarray(G.transpose(2, 1, 0)).astype(np.complex64)
    ang = np.pi * np.arange(Ntrajs) / Ntrajs
    r = np.linspace(-np.pi, np.pi, Nkx, endpoint=False)
    traj = np.stack([np.outer(r, np.cos(ang)), np.outer(r, np.sin(ang))]).astype(np.float32)
    c = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s)).astype(np.complex64)
    return dict(traj_type=np.array("2D"), n_dim=np.array(n_dim), L=np.array(L),
                phi2_dim=np.array([2 * n for n in n_dim]), n_coils=np.array(n_coils),
                eps=np.array(1e-6), cfg_mask_radius=np.array(2),
                cfg_mask_threshold=np.array(0.05), phi2=phi2, traj=traj,
                x=c(L, *reversed(n_dim)), SEs=c(n_coils, *reversed(n_dim)))


def cases(rng, quick: bool):
    """Yield ``(name, data)`` for every problem: references, then synthetic."""
    for f in sorted(glob.glob(str(ROOT / "data" / "ref_mlx" / "*.npz"))):
        name = Path(f).stem
        if quick and not any(k in name for k in ("L1", "real", "3d_24", "stack_40")):
            continue
        yield name, dict(np.load(f))
    yield "synthetic_2d_L12", synthetic(12, (48, 40), 5, rng)
    if not quick:
        yield "synthetic_2d_L72", synthetic(72, (24, 20), 3, rng)


def variants(z):
    """Yield ``(triangulation, mask name or None)`` for a problem."""
    for tri in ("lower", "upper"):
        yield tri, None
        for m in ("neighbour", "threshold"):
            if f"mask_{m}" in z:
                yield tri, m


def main() -> int:
    """Run every combination; return 1 on any failure."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--devices", type=int, nargs="+", default=None,
                    help="CUDA ordinals to use (default: all visible)")
    ap.add_argument("--quick", action="store_true", help="fewer problems")
    args = ap.parse_args()

    if mr_aha.backend() != "cuda":
        print("this test is for the CUDA backend")
        return 0
    import cupy as cp  # noqa: PLC0415

    devices = args.devices or list(range(cp.cuda.runtime.getDeviceCount()))
    if len(devices) < 2:
        print(f"skipped: needs at least 2 GPUs, {len(devices)} available")
        return 0
    os.environ["MR_AHA_MGPU_MIN_WORK"] = "0"    # small problems on all GPUs too
    primary = devices[0]
    cp.cuda.Device(primary).use()
    rng = np.random.default_rng(5)
    G_list = sorted({2, len(devices)} | ({3} if len(devices) >= 3 else set()))

    failed, count = 0, 0
    worst = dict(same=0.0, ref=0.0, aha=0.0, rep=0.0)

    def fail(msg):
        nonlocal failed
        failed += 1
        print("FAIL", msg)

    for name, z in cases(rng, args.quick):
        c = configs(z)
        phi2, traj = cp.asarray(z["phi2"]), cp.asarray(z["traj"])
        x, SEs = cp.asarray(z["x"]), cp.asarray(z["SEs"])
        for tri, m in variants(z):
            cfg = make_cfg(c, triangulation=tri, is_masked=m is not None)
            mask = cp.asarray(z[f"mask_{m}"]) if m else None
            key = f"phi2_toep_{tri}" + (f"_{m}" if m else "")
            ref = cp.asarray(z[key]) if key in z else None
            single = mr_aha.calc_toep(cfg, phi2, traj, mask)
            y_single = mr_aha.aha_toeplitz(cfg, x, SEs, single, mask)
            tag = f"{name} {tri} mask={m}"

            for G in G_list:
                devs = devices[:G]
                for mode in ("split", "replicate", "auto"):
                    os.environ["MR_AHA_TOEP_SPLIT"] = mode
                    # --- approach D: gather on the primary ---------------
                    dev_before = cp.cuda.runtime.getDevice()
                    k = mr_aha.calc_toep(cfg, phi2, traj, mask, devices=devs)
                    cp.cuda.Device(primary).synchronize()
                    count += 1
                    e = rel(k, single)
                    worst["same"] = max(worst["same"], e)
                    if not e < TOL_SAME:
                        fail(f"{tag} G={G} {mode}: calc_toep_mgpu vs single {e:.2e}")
                    if ref is not None:
                        e = rel(k, ref)
                        worst["ref"] = max(worst["ref"], e)
                        if not e < TOL_REF:
                            fail(f"{tag} G={G} {mode}: calc_toep_mgpu vs ref {e:.2e}")
                    if cp.cuda.runtime.getDevice() != dev_before:
                        fail(f"{tag} G={G} {mode}: current device changed")
                    del k

                for mode in ("split", "replicate", "auto", "off"):
                    os.environ["MR_AHA_TOEP_SPLIT"] = mode
                    # --- approach C: plan from the Gram blocks -----------
                    dev_before = cp.cuda.runtime.getDevice()
                    with mr_aha.MultiGPUPlan.from_gram(cfg, SEs, phi2, traj, mask,
                                                       devices=devs) as op:
                        info = op.info
                        y = op.apply(x)
                        k0 = op.kernel(0)
                        e = rel(k0, single)
                        worst["same"] = max(worst["same"], e)
                        if not e < TOL_SAME:
                            fail(f"{tag} G={G} {mode}: plan kernel vs single {e:.2e}")
                        e = rel(y, y_single)
                        worst["aha"] = max(worst["aha"], e)
                        if not e < TOL_AHA:
                            fail(f"{tag} G={G} {mode}: from_gram AhA vs single {e:.2e}")
                        h0 = k0.get()
                        src = info["toep_source"]
                        for j in range(1, len(info["devices"])):
                            with cp.cuda.Device(info["devices"][j]):
                                hj = op.kernel(j).get()
                            if src == "split" or src == "broadcast":
                                if not np.array_equal(h0, hj):
                                    fail(f"{tag} G={G} {mode}: copy {j} differs "
                                         f"from the primary's ({src})")
                            else:
                                e = float(np.linalg.norm(hj - h0) / np.linalg.norm(h0))
                                worst["rep"] = max(worst["rep"], e)
                                if not e < TOL_SAME:
                                    fail(f"{tag} G={G} {mode}: copy {j} vs primary "
                                         f"{e:.2e} ({src})")
                        expect = {"split": "split", "replicate": "replicate",
                                  "off": "broadcast"}.get(mode)
                        if expect and src != expect and not (
                                len(info["devices"]) == 1 and src == "broadcast"):
                            fail(f"{tag} G={G} {mode}: toep_source {src}")
                        del k0
                    count += 1
                    if cp.cuda.runtime.getDevice() != dev_before:
                        fail(f"{tag} G={G} {mode}: current device changed")
            del single
        print(f"done {name}: worst so far {worst}", flush=True)
    os.environ.pop("MR_AHA_TOEP_SPLIT", None)

    # --- leak check: repeated multi-GPU pre-computation ----------------------
    z = dict(np.load(ROOT / "data" / "ref_mlx" / "3d_24x20x16_L4_c2.npz"))
    c = configs(z)
    cfg = make_cfg(c)
    phi2, traj = cp.asarray(z["phi2"]), cp.asarray(z["traj"])
    SEs = cp.asarray(z["SEs"])
    out = cp.empty(mr_aha.phi2_toep_shape(cfg), dtype=cp.complex64)
    for mode in ("split", "replicate"):
        os.environ["MR_AHA_TOEP_SPLIT"] = mode
        for _ in range(2):              # warm-up: contexts, plans, peer access
            mr_aha.calc_toep(cfg, phi2, traj, out=out, devices=devices)
            mr_aha.MultiGPUPlan.from_gram(cfg, SEs, phi2, traj, devices=devices).close()
        before = free_bytes(devices)
        for _ in range(5):
            mr_aha.calc_toep(cfg, phi2, traj, out=out, devices=devices)
            mr_aha.MultiGPUPlan.from_gram(cfg, SEs, phi2, traj, devices=devices).close()
        after = free_bytes(devices)
        leak = [b - a for a, b in zip(after, before)]
        if max(leak) > 8 << 20:         # allow allocator noise, not a leak
            fail(f"leak check ({mode}): free memory dropped by {leak} bytes")

    # --- failure on another GPU: fill its memory -----------------------------
    os.environ["MR_AHA_TOEP_SPLIT"] = "split"
    victim = devices[1]
    before = free_bytes(devices)
    with cp.cuda.Device(victim):
        hog = None
        size = cp.cuda.runtime.memGetInfo()[0] - (1 << 20)
        while hog is None and size > 0:
            try:
                hog = cp.cuda.runtime.malloc(size)
            except cp.cuda.runtime.CUDARuntimeError:
                size -= 16 << 20
    for what in ("calc_toep", "from_gram"):
        try:
            if what == "calc_toep":
                mr_aha.calc_toep(cfg, phi2, traj, out=out, devices=devices)
            else:
                mr_aha.MultiGPUPlan.from_gram(cfg, SEs, phi2, traj,
                                              devices=devices).close()
            fail(f"{what} succeeded although device {victim} has no free memory")
        except mr_aha.MRError as exc:
            if f"device {victim}" not in str(exc):
                fail(f"{what}: error does not name device {victim}: {exc}")
            else:
                print(f"ok   {what} with device {victim} full: {exc}")
        if cp.cuda.runtime.getDevice() != primary:
            fail(f"{what}: current device changed after the failure")
    with cp.cuda.Device(victim):
        cp.cuda.runtime.free(hog)
    after = free_bytes(devices)
    leak = [b - a for a, b in zip(after, before)]
    if max(leak) > 8 << 20:
        fail(f"leak after the forced failures: free memory dropped by {leak} bytes")
    os.environ.pop("MR_AHA_TOEP_SPLIT", None)

    print(f"{count} combinations on up to {len(devices)} GPUs {devices}; worst "
          f"relative difference vs single GPU {worst['same']:.2e}, vs reference "
          f"{worst['ref']:.2e}, AhA {worst['aha']:.2e}, replicated copies "
          f"{worst['rep']:.2e}; {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
