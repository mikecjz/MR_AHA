"""Specialised vs generic block-diagonal kernels of the Metal backend.

The Metal operator uses block-diagonal kernels specialised on ``L``
(``mr_k_bd_dense<L>`` / ``mr_k_bd_masked<L>``, L = 1..32) and falls back to
the generic threadgroup-memory kernels otherwise or when
``MR_AHA_METAL_GENERIC_BD=1``. The reference data cover only a few ranks, so
this test applies the operator with both kernel families on random inputs for
many ranks (including the fallback at L = 33), both triangulations, unmasked
and masked, and checks that they agree to single-precision rounding. The
specialised kernels also process coils in groups (``MR_AHA_METAL_COIL_GROUP``);
with 5 coils, groups of 1, 2 and 4 (the last group smaller) are all checked
against the generic kernels, which handle one coil at a time.

Usage (Metal backend, from the repository root)::

    python tests/check_bd_kernels.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

import mr_aha  # noqa: E402

RANKS = (1, 2, 5, 7, 12, 16, 20, 31, 32, 33)


def apply(cfg, x, SEs, toep, mask, generic: bool, group: int = 4) -> np.ndarray:
    """One operator application with the chosen kernel family and coil group."""
    os.environ["MR_AHA_METAL_GENERIC_BD"] = "1" if generic else "0"
    os.environ["MR_AHA_METAL_COIL_GROUP"] = str(group)
    try:
        return mr_aha.aha_toeplitz(cfg, x, SEs, toep, mask=mask)
    finally:
        os.environ.pop("MR_AHA_METAL_GENERIC_BD", None)
        os.environ.pop("MR_AHA_METAL_COIL_GROUP", None)


def main() -> int:
    """Run every combination; return 1 on any mismatch."""
    if mr_aha.backend() != "metal":
        print("this test is for the Metal backend")
        return 0
    rng = np.random.default_rng(1)
    n_dim, n_coils = (24, 20), 5
    worst, failed = 0.0, 0
    for L in RANKS:
        for tri in ("lower", "upper"):
            for masked in (False, True):
                cfg = mr_aha.make_config("2D", n_dim, L, n_coils=n_coils,
                                         triangulation=tri, is_masked=masked)
                mask = None
                if masked:
                    pad = int(np.prod([cfg.phi2_dim[i] for i in range(3)]))
                    mask = np.sort(rng.choice(pad, pad * 2 // 3, replace=False))
                    mask = mask.astype(np.int64)
                    cfg.num_masked = int(mask.size)
                crandn = lambda *s: (rng.standard_normal(s) + 1j * rng.standard_normal(s)
                                     ).astype(np.complex64)
                x = crandn(L, n_dim[1], n_dim[0])
                SEs = crandn(n_coils, n_dim[1], n_dim[0])
                toep = crandn(*mr_aha.phi2_toep_shape(cfg))
                b = apply(cfg, x, SEs, toep, mask, generic=True)
                for group in (1, 2, 4):
                    a = apply(cfg, x, SEs, toep, mask, generic=False, group=group)
                    err = float(np.linalg.norm((a - b).astype(np.complex128).ravel())
                                / np.linalg.norm(b.astype(np.complex128).ravel()))
                    worst = max(worst, err)
                    if not err < 1e-5:
                        failed += 1
                        print(f"FAIL L={L} {tri} masked={masked} group={group}: {err:.2e}")
    print(f"{len(RANKS) * 4 * 3} combinations, worst relative difference {worst:.2e}, "
          f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
