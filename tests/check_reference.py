"""Check an MR_AHA build against the CUDA reference data in ``data/ref_mlx/``.

The reference files are written by ``tests/make_reference.py`` from the
CUDA backend (see that script for the keys and the array layout). This script
re-runs the library that :mod:`mr_aha` loads (``$MR_AHA_LIBRARY`` selects a
specific build) on the stored inputs and compares every output. It works for
either backend: arrays are CuPy arrays for ``mr_aha.backend() == 'cuda'`` and
NumPy arrays for ``'metal'``.

Five independent stages can be checked, so that a backend can be validated
piece by piece while it is being ported:

``aha``
    :func:`mr_aha.aha_toeplitz` on the stored ``x``, ``SEs`` and each stored
    ``phi2_toep`` variant (lower/upper, unmasked and masked), compared to
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
    :func:`mr_aha.calc_mask`, compared to the stored mask (which must match
    exactly).
``nufft``
    The NUFFT reference operator :class:`mr_aha.aha_nufft_plan` (cufinufft on
    CUDA, mlx-nufft on Metal) on the stored inputs, compared to the stored
    cufinufft ``AhAx_nufft``.
``op``
    The operator object :func:`mr_aha.create_op` on the stored inputs, with
    both methods: ``method='toeplitz'`` (the kernel computed by the operator,
    every triangulation and mask) against the stored ``AhAx_*``, and
    ``method='nufft'`` against the stored ``AhAx_nufft`` (the reference made
    by the independent :class:`mr_aha.aha_nufft_plan`) and against the
    Toeplitz ``AhAx_lower``, which agree to the NUFFT tolerance.

Tolerances (relative 2-norm error against the stored array, computed in
double precision):

* ``aha``, ``nufft`` and the Toeplitz part of ``op``: ``--tol-aha``
  (default ``1e-5``), single-precision FFT / NUFFT round-off on a different
  backend or summation order.
* ``toep``: ``--tol-toep``, default ``1e-5`` with the library's own
  cufinufft and ``5e-5`` with an external NUFFT (any ``--nufft`` other than
  ``library``, and always on Metal).
* ``mask``: ``0`` (the share of indices in only one of the two masks).
* NUFFT method vs Toeplitz in the ``op`` stage: ``1e-4``, since each is only
  accurate to the NUFFT tolerance.

Files whose ``config_json`` lacks ``gram_convention = "multitasking"`` were
written for the old Gram-block convention and are reported as failures
without being compared.

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

# Import the in-tree package so the library under test is the one selected by
# $MR_AHA_LIBRARY (or the default build of this checkout).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import mr_aha  # noqa: E402

# Stored kernel variants: both packings, each unmasked and with every mask.
TRIS = ("lower", "upper")
#: Suffix of the stored masked variants (``mask_neighbour``,
#: ``phi2_toep_<tri>_neighbour``, ``AhAx_<tri>_neighbour``).
MASKS = ("neighbour",)
STAGES = ("aha", "toep", "mask", "nufft", "op")


def rel(a: np.ndarray, b: np.ndarray) -> float:
    """Relative 2-norm difference ``||a - b|| / ||b||`` in double precision.

    Args:
        a: Array under test.
        b: Reference array.

    Returns:
        The relative difference, or ``inf`` when the shapes differ.
    """
    # A shape mismatch (e.g. a layout bug) is reported as an infinite error
    # rather than raising, so the remaining comparisons still run.
    if a.shape != b.shape:
        return float("inf")
    # Promote to double so the difference norm is not limited by float32.
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
    # n_dim / phi2_dim are stored column-major (Nx, Ny[, Nz]) as make_config
    # expects; the mask radius is stored as cfg_mask_radius (see
    # make_reference.py) to keep it apart from the mask_* arrays.
    return dict(
        traj_type=str(z["traj_type"]),
        n_dim=[int(v) for v in z["n_dim"]],
        L=int(z["L"]),
        phi2_dim=[int(v) for v in z["phi2_dim"]],
        n_coils=int(z["n_coils"]),
        eps=float(z["eps"]),
        mask_radius=int(z["cfg_mask_radius"]),
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
    # Each stage returns rows (name, relative error, tolerance); main() turns
    # them into ok / FAIL. Here only the block-diagonal application is under
    # test: the kernels and masks are the stored CUDA ones, not recomputed.
    rows = []
    x, SEs = bk.put(z["x"]), bk.put(z["SEs"])
    for suf, tri, m in variants():
        mask = None
        if m is None:
            cfg = make_cfg(c, triangulation=tri)
        else:
                # Masked variant: num_masked is set from the stored mask.
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
    # An "external" NUFFT (anything but the CUDA library's built-in
    # cufinufft path, and always on Metal) cannot reproduce the stored kernel
    # to the tight tolerance, so its kernels are additionally judged by the
    # A^H A x they produce.
    external = nufft not in (None, "library") or bk.name != "cuda"
    phi2, traj = bk.put(z["phi2"]), bk.put(z["traj"])
    x, SEs = bk.put(z["x"]), bk.put(z["SEs"])
    # Recompute each kernel from the stored Gram blocks, trajectory and (for
    # masked variants) the stored mask, so this stage is independent of the
    # mask stage.
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


def check_mask(z, bk: Backend, c: dict) -> list:
    """Stage ``mask``: the mask of :func:`mr_aha.calc_mask` vs the stored one.

    Masks are index sets, so the share of indices in one but not the other is
    reported. No NUFFT is involved and both backends round the samples the
    same way, so the mask must match exactly.
    """
    # Recompute the mask with the radius the reference used. The error is the
    # size of the symmetric difference relative to the reference size, with a
    # tolerance of exactly zero.
    traj = bk.put(z["traj"])
    got = bk.get(mr_aha.calc_mask(make_cfg(c), traj, radius=c["mask_radius"]))
    rows = []
    for m in MASKS:
        ref = z[f"mask_{m}"]
        rows.append((f"mask_{m}", np.setxor1d(got, ref).size / max(ref.size, 1), 0.0))
    return rows


def check_nufft(z, bk: Backend, c: dict, tol: float) -> list:
    """Stage ``nufft``: the NUFFT reference operator vs the stored one."""
    # The NUFFT reference is optional in the files (cufinufft may have been
    # unavailable when they were made); nothing to compare then.
    if "AhAx_nufft" not in z.files:
        return []
    with mr_aha.aha_nufft_plan(make_cfg(c), bk.put(z["traj"])) as plan:
        out = bk.get(plan.apply(bk.put(z["x"]), bk.put(z["SEs"]),
                                bk.put(z["phi2"])))
    return [("AhAx_nufft", rel(out, z["AhAx_nufft"]), tol)]


def check_op(z, bk: Backend, c: dict, tol: float, tol_methods: float) -> list:
    """Stage ``op``: :func:`mr_aha.create_op` with both methods.

    Toeplitz: the operator computes its own kernel from ``phi2`` and ``traj``
    (for every triangulation and stored mask) and is compared with the stored
    ``AhAx_*``. NUFFT: compared with the stored ``AhAx_nufft`` at ``tol`` and
    with the Toeplitz ``AhAx_lower`` at ``tol_methods``.
    """
    rows = []
    # Toeplitz method: the operator computes its own kernel (and uses the
    # stored mask), so this checks calc_toep + aha_toeplitz as wired inside
    # the operator object.
    x, SEs = bk.put(z["x"]), bk.put(z["SEs"])
    phi2, traj = bk.put(z["phi2"]), bk.put(z["traj"])
    for suf, tri, m in variants():
        mask = None
        if m is None:
            cfg = make_cfg(c, triangulation=tri)
        else:
            cfg = make_cfg(c, triangulation=tri, is_masked=True)
            mask = bk.put(z[f"mask_{m}"])
        with mr_aha.create_op(cfg, SEs, phi2, traj, mask) as op:
            rows.append((f"op toeplitz AhAx_{suf}", rel(bk.get(op(x)), z[f"AhAx_{suf}"]),
                         tol))
    # NUFFT method (unmasked default config): against the independent Python
    # NUFFT reference when stored, and always against the Toeplitz result,
    # which it matches only to the NUFFT accuracy (tol_methods).
    with mr_aha.create_op(make_cfg(c), SEs, phi2, traj, method="nufft") as op:
        out = bk.get(op(x))
    if "AhAx_nufft" in z.files:
        rows.append(("op nufft AhAx_nufft", rel(out, z["AhAx_nufft"]), tol))
    rows.append(("op nufft vs AhAx_lower", rel(out, z["AhAx_lower"]), tol_methods))
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
    ap.add_argument("--nufft", default=None, choices=["library", "mlx", "finufft"],
                    help="NUFFT engine of the toep stage (default: backend's)")
    args = ap.parse_args()

    # Select the reference files (all, or the requested stems).
    bk = Backend()
    files = sorted(args.data.glob("*.npz"))
    if args.cases:
        files = [f for f in files if f.stem in args.cases]
    if not files:
        print(f"no reference files found in {args.data}")
        return 1
    # Report which shared library was actually loaded, so a stale build is
    # easy to spot.
    print(f"backend: {bk.name}; library: {mr_aha._lib.lib._name}")

    failed = 0
    for f in files:
        # Refuse files made for the old Gram-block convention: their phi2 axes
        # (and hence every kernel) are transposed relative to the current one.
        z = np.load(f)
        if json.loads(str(z["config_json"])).get("gram_convention") != "multitasking":
            print(f"FAIL {f.stem}: written before the MR-Multitasking Gram convention "
                  "(no gram_convention in config_json); regenerate it with "
                  "tests/make_reference.py")
            failed += 1
            continue
        # Run the requested stages; the toep tolerance depends on whether an
        # external NUFFT computes the kernels (see the module docstring).
        c = configs(z)
        rows = []
        if "aha" in args.stages:
            rows += check_aha(z, bk, c, args.tol_aha)
        if "toep" in args.stages:
            ext = args.nufft not in (None, "library") or bk.name != "cuda"
            tol_toep = args.tol_toep or (5e-5 if ext else 1e-5)
            rows += check_toep(z, bk, c, tol_toep, args.nufft, args.tol_aha)
        if "mask" in args.stages:
            rows += check_mask(z, bk, c)
        if "nufft" in args.stages:
            rows += check_nufft(z, bk, c, args.tol_aha)
        if "op" in args.stages:
            rows += check_op(z, bk, c, args.tol_aha, 1e-4)
        # A row passes when error <= tolerance (a NaN error fails). The
        # summary names the worst row, ranked by error / tolerance (or by the
        # raw error for the zero-tolerance mask rows); every failing row is
        # listed after it.
        bad = [r for r in rows if not r[1] <= r[2]]
        failed += len(bad)
        worst = max(rows, key=lambda r: r[1] / r[2] if r[2] else r[1])
        print(f"{'FAIL' if bad else 'ok  '} {f.stem}: {len(rows)} checks, "
              f"worst {worst[0]} = {worst[1]:.2e}")
        for name, val, tol in bad:
            print(f"       {name}: {val:.3e} > {tol:.0e}")
    # Final machine-readable summary line for scripts and CI logs.
    print(json.dumps({"backend": bk.name, "files": len(files), "failed": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
