"""Specialized vs generic block-diagonal kernels (CUDA and Metal backends).

Both backends use block-diagonal kernels specialized on ``L`` (CUDA:
``mr_k_bd_dense_t<L>`` / ``mr_k_bd_masked_t<L>``; Metal: ``mr_k_bd_dense<L>``
/ ``mr_k_bd_masked<L>``; L = 1..16 on CUDA, 1..5 on Metal). Above that both
use tiled shared/threadgroup-memory kernels (``mr_k_bdt``); both backends
fall back to the generic shared/threadgroup-memory kernels otherwise or when
``MR_AHA_<BACKEND>_GENERIC_BD=1``. The reference data cover only a few ranks,
so this test applies the operator with both kernel families on random inputs
for many ranks (including the tiled kernels up to L = 130 with 32 and 16
locations per tile), both triangulations,
unmasked and masked, and checks that they agree to single-precision rounding.
The specialized kernels also process coils in groups
(``MR_AHA_<BACKEND>_COIL_GROUP``); with 5 coils, groups of 1, 2 and 4 (the last
group smaller) are all checked against the generic kernels, which handle one
coil at a time. The tiled kernels are also forced for every rank
(``MR_AHA_<BACKEND>_TILED_BD=1``), with coil batches of 1, 2 and 4
(``MR_AHA_<BACKEND>_BDT_NC``); on Metal also with 4, 8 and 16 locations per
tile (``MR_AHA_METAL_BDT_P``). The generic Metal kernels stop at L = 128
(one rank vector per thread in 32 KB), so L = 130 is checked against
L = 130's own tiled variants only on CUDA.

Reference and tolerance: the generic kernels (one coil at a time) are the
reference; every other kernel/grouping must agree to a relative 2-norm
difference below ``1e-5``. The kernel ``phi2_toep`` is random (not a genuine
Toeplitz kernel, and not Hermitian-derived from a Gram block): only the
block-diagonal arithmetic and the triangle unpacking are under test, so any
complex values will do. Exits with status 1 on any mismatch.

Usage (from the repository root)::

    python tests/check_bd_kernels.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# Import the in-tree package so the library just built is the one tested.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import mr_aha  # noqa: E402

# Ranks chosen around every dispatch boundary: small L, the end of the Metal
# specialized range (5) and just above it (6), the end of the CUDA specialized
# range (16) and just above it (17), and larger ranks served by the tiled
# kernels (up to 130).
RANKS = (1, 2, 5, 6, 7, 12, 16, 17, 20, 30, 31, 32, 33, 47, 72, 130)


def apply(cfg, x, SEs, toep, mask, generic: bool, group: int = 4,
          env: dict[str, str] | None = None) -> np.ndarray:
    """One operator application with the chosen kernel family and coil group.

    Inputs and output are NumPy arrays; on CUDA they are moved to the current
    device (CuPy) and back. @p env sets further environment variables for
    this application only.
    """
    # The kernel family and coil group are selected through environment
    # variables that the library reads when a plan is created. They are set
    # just for this call and removed in `finally`, so nothing leaks into the
    # next application.
    prefix = "MR_AHA_" + mr_aha.backend().upper()
    env = dict(env or {})
    env[prefix + "_GENERIC_BD"] = "1" if generic else "0"
    env[prefix + "_COIL_GROUP"] = str(group)
    os.environ.update(env)
    try:
        # CUDA takes device arrays: copy the NumPy inputs to the current GPU
        # and the result back. Metal takes host (unified-memory) NumPy arrays.
        if mr_aha.backend() == "cuda":
            import cupy as cp  # noqa: PLC0415
            dev = [None if a is None else cp.asarray(a) for a in (x, SEs, toep, mask)]
            return cp.asnumpy(mr_aha.aha_toeplitz(cfg, *dev[:3], mask=dev[3]))
        return mr_aha.aha_toeplitz(cfg, x, SEs, toep, mask=mask)
    finally:
        for k in env:
            os.environ.pop(k, None)


def main() -> int:
    """Run every combination; return 1 on any mismatch."""
    # Fixed seed; a small 2D problem (24 x 20 image, 5 coils) keeps the many
    # combinations fast. 5 coils with group 2 or 4 leaves a smaller last group.
    rng = np.random.default_rng(1)
    n_dim, n_coils = (24, 20), 5
    worst, failed, count = 0.0, 0, 0
    # Variants checked against the generic kernels: (label, coil group, env).
    variants = [(f"group={g}", g, {}) for g in (1, 2, 4)]
    if mr_aha.backend() == "cuda":
        variants += [(f"tiled nc={nc} group={g}", g,
                      {"MR_AHA_CUDA_TILED_BD": "1", "MR_AHA_CUDA_BDT_NC": str(nc)})
                     for nc in (1, 2, 4) for g in (1, 4)]
    else:
        variants += [(f"tiled nc={nc} P={P} group={g}", g,
                      {"MR_AHA_METAL_TILED_BD": "1", "MR_AHA_METAL_BDT_NC": str(nc),
                       "MR_AHA_METAL_BDT_P": str(P)})
                     for nc in (1, 2, 4) for P in (4, 8, 16) for g in (1, 4)]
    # Sweep rank x triangulation x masking.
    for L in RANKS:
        # Metal's generic reference kernels stop at L = 128.
        if L > 128 and mr_aha.backend() != "cuda":
            continue
        for tri in ("lower", "upper"):
            for masked in (False, True):
                cfg = mr_aha.make_config("2D", n_dim, L, n_coils=n_coils,
                                         triangulation=tri, is_masked=masked)
                # Masked case: a random, sorted, duplicate-free subset of two
                # thirds of the upsampled grid (0-based linear indices); any
                # subset exercises the masked kernels' index handling.
                mask = None
                if masked:
                    pad = int(np.prod([cfg.phi2_dim[i] for i in range(3)]))
                    mask = np.sort(rng.choice(pad, pad * 2 // 3, replace=False))
                    mask = mask.astype(np.int64)
                    cfg.num_masked = int(mask.size)
                # Random complex64 inputs in the binding's row-major layout:
                # x (L, Ny, Nx), maps (Ncoils, Ny, Nx) and a kernel of shape
                # phi2_toep_shape(cfg) (masked or unmasked).
                crandn = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s)
                                     ).astype(np.complex64)
                x = crandn(L, n_dim[1], n_dim[0])
                SEs = crandn(n_coils, n_dim[1], n_dim[0])
                toep = crandn(*mr_aha.phi2_toep_shape(cfg))
                # Reference: the generic kernels.
                b = apply(cfg, x, SEs, toep, mask, generic=True)
                for label, group, env in variants:
                    a = apply(cfg, x, SEs, toep, mask, generic=False, group=group,
                              env=env)
                    count += 1
                    # Relative difference in double precision; `not err < tol`
                    # also counts a NaN result as a failure.
                    err = float(np.linalg.norm((a - b).astype(np.complex128).ravel())
                                / np.linalg.norm(b.astype(np.complex128).ravel()))
                    worst = max(worst, err)
                    if not err < 1e-5:
                        failed += 1
                        print(f"FAIL L={L} {tri} masked={masked} {label}: {err:.2e}")
    print(f"{count} combinations, worst relative difference {worst:.2e}, "
          f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
