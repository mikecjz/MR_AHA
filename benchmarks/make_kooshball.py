"""Generate the synthetic 3D radial "kooshball" benchmark problem.

No measured 3D data is available, so a realistic stand-in is synthesised with
the sizes requested for the benchmark:

* ``Ntraj = 15680`` full-diameter radial spokes of ``Nkx = 160`` samples,
  ``[Nx, Ny, Nz] = [160, 160, 160]``, ``n_coils = 12``, ``L = 12``.
* Spoke directions follow the 2D golden-means pattern on the hemisphere
  (successive spokes are near-uniformly spread, as in a real golden-means
  kooshball), samples run over ``[-pi, pi)``.
* The temporal subspace is a smooth, complex, orthonormal basis ``U``
  (``Ntraj x L``) of DCT-like functions of acquisition time. Spoke ``t`` is
  acquired at one time point, so its Gram block is the rank-one
  ``phi2_t = conj(u_t) u_t^T`` with ``u_t`` row ``t`` of ``U``.
* Coil maps are smooth complex Gaussians around the object, normalised to unit
  root-sum-of-squares inside the object support.
* ``x_true`` is a sum of random ellipsoids per rank with energy decaying over
  the ranks, and the CG right-hand side is ``b = A^H A x_true`` evaluated with
  the NUFFT operator.

The result is written to ``benchmarks/data/kooshball_3d.mat`` (MATLAB v5,
single precision, column-major shapes) so that Python and MATLAB benchmark the
exact same problem.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BENCH_DATA, cp, mr_aha  # noqa: E402

N, NTRAJ, NKX, NC, L = 160, 15680, 160, 12, 12


def trajectory():
    """Golden-means kooshball, returned as ``[Ntraj, Nkx, 3]`` (float32)."""
    i = np.arange(NTRAJ)
    phi1, phi2 = 0.4656, 0.6823  # 2D golden means (Chan et al., MRM 2009)
    cos_pol = np.mod(i * phi1, 1.0)          # uniform on the hemisphere
    azi = 2 * np.pi * np.mod(i * phi2, 1.0)
    sin_pol = np.sqrt(1 - cos_pol**2)
    dirs = np.stack([sin_pol * np.cos(azi), sin_pol * np.sin(azi), cos_pol], 1)
    kr = np.pi * (np.arange(NKX) - NKX / 2) / (NKX / 2)  # [-pi, pi)
    return (dirs[:, None, :] * kr[None, :, None]).astype(np.float32)


def grid():
    """Normalised image coordinates in [-1, 1), row-major (z, y, x)."""
    r = (np.arange(N) - N // 2) / (N / 2)
    return np.meshgrid(r, r, r, indexing="ij")


def coil_maps(support):
    """Twelve smooth complex coil maps, row-major ``(Nc, Nz, Ny, Nx)``."""
    z, y, x = grid()
    maps = []
    for c in range(NC):
        ring, k = divmod(c, 6)
        ang = 2 * np.pi * k / 6 + ring * np.pi / 6
        cz = -0.4 if ring == 0 else 0.4
        cx, cy = 1.1 * np.cos(ang), 1.1 * np.sin(ang)
        d2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
        phase = ang + 0.8 * (x * np.cos(ang) + y * np.sin(ang)) + 0.3 * z
        maps.append(np.exp(-d2 / (2 * 0.7**2)) * np.exp(1j * phase))
    S = np.stack(maps)
    rss = np.sqrt((np.abs(S) ** 2).sum(0))
    return (S / np.maximum(rss, 1e-6) * support).astype(np.complex64)


def temporal_basis():
    """Smooth complex orthonormal temporal basis ``U`` of shape ``(Ntraj, L)``."""
    t = (np.arange(NTRAJ) + 0.5) / NTRAJ
    B = np.stack([np.cos(np.pi * l * t) * np.exp(1j * 2 * np.pi * 0.37 * l * t)
                  for l in range(L)], 1)
    Q, _ = np.linalg.qr(B)
    return (Q * np.sqrt(NTRAJ / L)).astype(np.complex64)


def phantom(rng, support):
    """Random-ellipsoid spatial coefficients, row-major ``(L, Nz, Ny, Nx)``."""
    z, y, x = grid()
    X = np.zeros((L, N, N, N), np.complex64)
    for _ in range(10):
        c = rng.uniform(-0.45, 0.45, 3)
        a = rng.uniform(0.1, 0.35, 3)
        inside = (((x - c[0]) / a[0]) ** 2 + ((y - c[1]) / a[1]) ** 2
                  + ((z - c[2]) / a[2]) ** 2) <= 1
        amp = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) / (1 + np.arange(L))
        X += amp[:, None, None, None].astype(np.complex64) * inside[None]
    return X * support[None]


def main():
    """Build the problem, evaluate ``b`` on the GPU and save the .mat file."""
    rng = np.random.default_rng(0)
    z, y, x = grid()
    support = ((x**2 + y**2 + z**2) <= 0.85**2).astype(np.float32)

    traj = trajectory()                                   # [Ntraj, Nkx, 3]
    S = coil_maps(support)                                # (Nc, Nz, Ny, Nx)
    U = temporal_basis()                                  # (Ntraj, L)
    phi2 = np.conj(U)[:, :, None] * U[:, None, :]         # [t, i, j]
    X = phantom(rng, support)                             # (L, Nz, Ny, Nx)

    cfg = mr_aha.make_config("3D", (N, N, N), L, n_coils=NC)
    with mr_aha.AhANufftPlan(cfg, cp.asarray(np.ascontiguousarray(traj.T))) as p:
        b = p.apply(cp.asarray(X), cp.asarray(S),
                    cp.asarray(np.ascontiguousarray(phi2.T))).get()

    BENCH_DATA.mkdir(parents=True, exist_ok=True)
    out = BENCH_DATA / "kooshball_3d.mat"
    # .T turns the row-major arrays into the column-major shapes of the docs.
    sio.savemat(out, dict(traj=traj, SEs=S.T, phi2=phi2.astype(np.complex64),
                          x_true=X.T, b=b.T, L=np.int32(L)),
                do_compression=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
