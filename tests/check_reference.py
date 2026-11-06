"""Check an MR_AHA build against the CUDA reference data in ``data/ref_mlx/``.

The reference files are written by ``benchmarks/make_reference.py`` from the
CUDA backend (see that script for the keys and the array layout). This script
re-runs the library that :mod:`mr_aha` loads (``$MR_AHA_LIBRARY`` selects a
specific build) on the stored inputs and compares every output. It works for
either backend: arrays are CuPy arrays for ``mr_aha.backend() == 'cuda'`` and
NumPy arrays for ``'metal'``.

Three independent stages can be checked, so that a backend can be validated
piece by piece while it is being ported:

``aha``
    :func:`mr_aha.aha_toeplitz` on the stored ``x``, ``SEs`` and each stored
    ``phi2_toep`` variant (lower/upper, unmasked and both masks), compared to
    the stored ``AhAx_*``. Independent of ``calc_toep`` and ``calc_mask``.
``toep``
    :func:`mr_aha.calc_toep` on the stored ``phi2`` and ``traj`` (and stored
    masks), compared to the stored ``phi2_toep_*``. When ``--nufft`` selects a
    different NUFFT than the references were made with (the reference uses
    single-precision cufinufft), the kernels legitimately differ by that
    NUFFT's error, about 1e-5; the stage then also applies the operator with
    each new kernel and compares it with the stored ``AhAx_*`` at the ``aha``
    tolerance, which is the accuracy that matters.
``mask``
    :func:`mr_aha.calc_mask_kernel` and :func:`mr_aha.calc_mask` (both
    methods), compared to the stored kernel and masks.

Usage (from the repository root)::

    MR_AHA_LIBRARY=build/lib/libmr_aha_c.so python tests/check_reference.py
    python tests/check_reference.py --stages aha --cases 2d_50x40_L3_c3

Exits with status 1 if any comparison exceeds its tolerance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import mr_aha  # noqa: E402

TRIS = ("lower", "upper")
MASKS = ("neighbour", "threshold")
STAGES = ("aha", "toep", "mask")


def rel(a: np.ndarray, b: np.ndarray) -> float:
    """Relative 2-norm difference ``||a - b|| / ||b||`` in double precision.

    Args:
        a: Array under test.
        b: Reference array.

    Returns:
        The relative difference, or ``inf`` when the shapes differ.
    """
    if a.shape != b.shape:
        return float("inf")
    a = a.astype(np.complex128 if np.iscomplexobj(a) else np.float64)
    b = b.astype(a.dtype)
    nb = np.linalg.norm(b.ravel())
    return float(np.linalg.norm((a - b).ravel()) / (nb if nb else 1.0))


class Backend:
    """Array module of the loaded library: CuPy for CUDA, NumPy for Metal."""

    def __init__(self):
        self.name = mr_aha.backend()
        if self.name == "cuda":
            import cupy  # noqa: PLC0415
            self.xp = cupy
        else:
            self.xp = np

    def put(self, a: np.ndarray):
        """Move a NumPy array to wherever the library expects its inputs."""
        return self.xp.asarray(a)

    def get(self, a) -> np.ndarray:
        """Bring a library output back as a NumPy array."""
        return a.get() if hasattr(a, "get") else np.asarray(a)


def configs(z) -> dict:
    """Rebuild the keyword arguments of :func:`mr_aha.make_config` from a file.

    Args:
        z: The loaded ``.npz`` reference file.

    Returns:
        Dict with the positional ``traj_type``, ``n_dim``, ``L`` and the common
        keyword arguments.
    """
    return dict(
        traj_type=str(z["traj_type"]),
        n_dim=[int(v) for v in z["n_dim"]],
        L=int(z["L"]),
        phi2_dim=[int(v) for v in z["phi2_dim"]],
        n_coils=int(z["n_coils"]),
        eps=float(z["eps"]),
        mask_radius=int(z["cfg_mask_radius"]),
        mask_threshold=float(z["cfg_mask_threshold"]),
    )


def make_cfg(c: dict, **over):
    """Call :func:`mr_aha.make_config` with the file's settings plus overrides."""
    kw = {k: v for k, v in c.items() if k not in ("traj_type", "n_dim", "L")}
    kw.update(over)
    return mr_aha.make_config(c["traj_type"], c["n_dim"], c["L"], **kw)


def variants():
    """Yield ``(suffix, triangulation, mask_name_or_None)`` of every stored kernel."""
    for tri in TRIS:
        yield tri, tri, None
        for m in MASKS:
            yield f"{tri}_{m}", tri, m


def check_aha(z, bk: Backend, c: dict, tol: float) -> list:
    """Stage ``aha``: apply the operator to every stored ``phi2_toep`` variant."""
    rows = []
    x, SEs = bk.put(z["x"]), bk.put(z["SEs"])
    for suf, tri, m in variants():
        mask = None
        if m is None:
            cfg = make_cfg(c, triangulation=tri)
        else:
            mask_np = z[f"mask_{m}"]
            cfg = make_cfg(c, triangulation=tri, is_masked=True,
                           num_masked=int(mask_np.size))
            mask = bk.put(mask_np)
        out = bk.get(mr_aha.aha_toeplitz(cfg, x, SEs,
                                         bk.put(z[f"phi2_toep_{suf}"]), mask=mask))
        rows.append((f"AhAx_{suf}", rel(out, z[f"AhAx_{suf}"]), tol))
    return rows


def check_toep(z, bk: Backend, c: dict, tol: float, nufft=None,
               tol_aha: float = 1e-5) -> list:
    """Stage ``toep``: recompute every stored ``phi2_toep`` variant.

    ``nufft`` is passed to :func:`mr_aha.calc_toep` (``None`` = backend
    default, ``'library'`` = the one-shot C path, ``'finufft'`` / ``'mlx'`` =
    the two-stage path). With an external NUFFT the operator is also applied
    with each new kernel and compared with the stored ``AhAx_*``.
    """
    rows = []
    external = nufft not in (None, "library") or bk.name != "cuda"
    phi2, traj = bk.put(z["phi2"]), bk.put(z["traj"])
    x, SEs = bk.put(z["x"]), bk.put(z["SEs"])
    for suf, tri, m in variants():
        mask = None
        if m is None:
            cfg = make_cfg(c, triangulation=tri)
        else:
            cfg = make_cfg(c, triangulation=tri, is_masked=True)
            mask = bk.put(z[f"mask_{m}"])
        out = mr_aha.calc_toep(cfg, phi2, traj, mask=mask, nufft=nufft)
        rows.append((f"phi2_toep_{suf}", rel(bk.get(out), z[f"phi2_toep_{suf}"]), tol))
        if external:
            aha = bk.get(mr_aha.aha_toeplitz(cfg, x, SEs, out, mask=mask))
            rows.append((f"AhAx_{suf} (new toep)", rel(aha, z[f"AhAx_{suf}"]), tol_aha))
    return rows


def check_mask(z, bk: Backend, c: dict, tol: float) -> list:
    """Stage ``mask``: kernel (relative error) and masks (fraction differing)."""
    rows = []
    traj = bk.put(z["traj"])
    base = make_cfg(c)
    kern = bk.get(mr_aha.calc_mask_kernel(base, traj))
    rows.append(("mask_kernel", rel(kern, z["mask_kernel"]), tol))
    got = {
        "neighbour": mr_aha.calc_mask(base, traj, radius=c["mask_radius"]),
        "threshold": mr_aha.calc_mask(base, traj, threshold=c["mask_threshold"]),
    }
    for m in MASKS:
        a, b = bk.get(got[m]), z[f"mask_{m}"]
        # Masks are index sets: report the share of indices in one but not the
        # other. The neighbour mask must match exactly; the threshold mask may
        # flip a few locations whose kernel value sits at the threshold.
        diff = np.setxor1d(a, b).size / max(b.size, 1)
        rows.append((f"mask_{m}", diff, 0.0 if m == "neighbour" else 1e-3))
    return rows


def main() -> int:
    """Run the requested stages on the requested reference files."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "ref_mlx")
    ap.add_argument("--cases", nargs="*", help="file stems to check (default all)")
    ap.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    ap.add_argument("--tol-aha", type=float, default=1e-5)
    ap.add_argument("--tol-toep", type=float, default=None,
                    help="default 1e-5 for the library's own NUFFT, 5e-5 "
                         "for an external one (see the toep stage)")
    ap.add_argument("--tol-kernel", type=float, default=1e-5)
    ap.add_argument("--nufft", default=None, choices=["library", "mlx", "finufft"],
                    help="NUFFT engine of the toep stage (default: backend's)")
    args = ap.parse_args()

    bk = Backend()
    files = sorted(args.data.glob("*.npz"))
    if args.cases:
        files = [f for f in files if f.stem in args.cases]
    if not files:
        print(f"no reference files found in {args.data}")
        return 1
    print(f"backend: {bk.name}; library: {mr_aha._lib.lib._name}")

    failed = 0
    for f in files:
        z = np.load(f)
        c = configs(z)
        rows = []
        if "aha" in args.stages:
            rows += check_aha(z, bk, c, args.tol_aha)
        if "toep" in args.stages:
            ext = args.nufft not in (None, "library") or bk.name != "cuda"
            tol_toep = args.tol_toep or (5e-5 if ext else 1e-5)
            rows += check_toep(z, bk, c, tol_toep, args.nufft, args.tol_aha)
        if "mask" in args.stages and "mask_kernel" in z.files:
            rows += check_mask(z, bk, c, args.tol_kernel)
        bad = [r for r in rows if not r[1] <= r[2]]
        failed += len(bad)
        worst = max(rows, key=lambda r: r[1] / r[2] if r[2] else r[1])
        print(f"{'FAIL' if bad else 'ok  '} {f.stem}: {len(rows)} checks, "
              f"worst {worst[0]} = {worst[1]:.2e}")
        for name, val, tol in bad:
            print(f"       {name}: {val:.3e} > {tol:.0e}")
    print(json.dumps({"backend": bk.name, "files": len(files), "failed": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
