"""Time :func:`mr_aha.calc_toep` on whichever backend is loaded.

Builds synthetic trajectories with the shapes of the three benchmark problems
of ``benchmarks/README.md`` -- 2D golden-angle radial 377 x 160 (L = 6),
stack-of-stars 280 x 320 x 56 kz (L = 12) and a golden-means 3D kooshball
15680 x 160 (L = 12) -- and times one unmasked ``calc_toep`` each. Only the
speed is measured; the Gram blocks are random Hermitian matrices.

On the CUDA backend this times the one-shot library path (cufinufft inside
``mr_calc_toep``); on Metal the two-stage path (mlx-nufft, then
``mr_calc_toep_from_psf``). ``--nufft`` overrides the engine.

Usage (from the repository root)::

    python benchmarks/bench_toep_backend.py [--cases 2d stack 3d] [--reps 2]

Writes ``benchmarks/results/<host>/toep_<backend>.json``.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import mr_aha  # noqa: E402

#: name -> (traj_type, n_dim, L, Ntrajs, Nkx)
CASES = {
    "2d": ("2D", (160, 160), 6, 377, 160),
    "stack": ("stack", (320, 320, 56), 12, 280, 320),
    "3d": ("3D", (160, 160, 160), 12, 15680, 160),
}


def radial_traj(ndim: int, n_traj: int, n_kx: int) -> np.ndarray:
    """Golden-angle (2D) or golden-means (3D) radial trajectory.

    Returns:
        ``float32`` array ``(ndim, Nkx, Ntrajs)`` in radians, ``[-pi, pi)``.
    """
    k = np.pi * (np.arange(n_kx) - n_kx // 2) / (n_kx // 2)
    if ndim == 2:
        a = np.arange(n_traj) * np.pi * (np.sqrt(5) - 1) / 2
        d = np.stack([np.cos(a), np.sin(a)])
    else:  # golden means (Chan et al.)
        phi1, phi2 = 0.4656, 0.6823
        m = np.arange(n_traj)
        z = (m * phi1) % 1.0 * 2 - 1
        az = 2 * np.pi * ((m * phi2) % 1.0)
        r = np.sqrt(1 - z ** 2)
        d = np.stack([r * np.cos(az), r * np.sin(az), z])
    return (d[:, None, :] * k[None, :, None]).astype(np.float32)


def gram(rng, L: int, n: int) -> np.ndarray:
    """Random Hermitian PSD blocks, row-major ``(L, L, n)``."""
    a = rng.standard_normal((n, 2, L)) + 1j * rng.standard_normal((n, 2, L))
    g = np.einsum("nki,nkj->jin", a.conj(), a)  # [j, i, n] = (A^H A)_{ij}
    return g.astype(np.complex64)


def main() -> None:
    """Run the requested cases and write the JSON report."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--nufft", default=None)
    args = ap.parse_args()

    backend = mr_aha.backend()
    if backend == "cuda":
        import cupy as xp  # noqa: PLC0415
        sync = xp.cuda.Device().synchronize
    else:
        xp, sync = np, (lambda: None)
    host = socket.gethostname().split(".")[0]
    rng = np.random.default_rng(0)
    results = []
    for name in args.cases:
        traj_type, n_dim, L, n_traj, n_kx = CASES[name]
        traj = radial_traj(3 if traj_type == "3D" else 2, n_traj, n_kx)
        g = gram(rng, L, n_traj)
        if traj_type == "stack":
            g = np.ascontiguousarray(np.repeat(g[:, :, None, :], n_dim[2], axis=2))
        cfg = mr_aha.make_config(traj_type, n_dim, L)
        traj_d, g_d = xp.asarray(traj), xp.asarray(g)
        ts = []
        for _ in range(args.reps):
            sync()
            t0 = time.perf_counter()
            out = mr_aha.calc_toep(cfg, g_d, traj_d, nufft=args.nufft)
            sync()
            ts.append(time.perf_counter() - t0)
            del out
        res = dict(case=name, n_dim=n_dim, L=L, Ntrajs=n_traj, Nkx=n_kx,
                   nufft=args.nufft or ("library" if backend == "cuda" else "mlx"),
                   first_s=ts[0], best_s=min(ts), reps=args.reps)
        results.append(res)
        print(f"{name:5s} calc_toep first {ts[0]:7.2f} s  best {min(ts):7.2f} s",
              flush=True)

    out_dir = ROOT / "benchmarks" / "results" / host
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"toep_{backend}.json"
    out.write_text(json.dumps(dict(backend=backend, host=host, results=results),
                              indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
