"""Export the ``data/ref_mlx/*.npz`` reference files to MATLAB ``.mat`` files.

``tests/make_reference.py`` stores every array in the row-major (NumPy)
order of the Python binding. MATLAB uses the column-major shapes of the
project documentation, which are the same memory with the dimensions
reversed, so each array is written transposed (``a.T``): ``savemat`` stores it
in Fortran order and MATLAB sees e.g. ``x`` as ``[Nx, Ny, Nz, L]``, ``traj`` as
``[Ntrajs, Nkx, ndim]`` and ``phi2`` as ``[Ntrajs, (Nkz,) L, L]``. Masks stay
0-based ``int64``; ``tests/check_reference_matlab.m`` adds one.

Usage (from the repository root)::

    python tests/export_reference_mat.py [--src data/ref_mlx] [--dst data/ref_mlx_mat]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io


def main() -> None:
    """Convert every reference file found in ``--src``."""
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, default=root / "data" / "ref_mlx")
    ap.add_argument("--dst", type=Path, default=root / "data" / "ref_mlx_mat")
    args = ap.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(args.src.glob("*.npz")):
        z = np.load(f)
        out = {}
        for k in z.files:
            a = z[k]
            if k == "config_json":
                # Carry the Gram-block convention marker over to MATLAB.
                conv = json.loads(str(a)).get("gram_convention")
                if conv:
                    out["gram_convention"] = conv
                continue
            if a.dtype.kind == "U":
                out[k] = str(a)
            elif a.ndim == 0:
                out[k] = a.item()
            elif k in ("n_dim", "phi2_dim"):
                out[k] = a.astype(np.float64)[None, :]     # row vector
            elif k.startswith("mask_") and a.dtype.kind == "i":
                out[k] = a.astype(np.int64)[:, None]       # column, 0-based
            else:
                out[k] = a.T
        scipy.io.savemat(args.dst / (f.stem + ".mat"), out, do_compression=False)
        print(f"wrote {args.dst / (f.stem + '.mat')}")


if __name__ == "__main__":
    main()
