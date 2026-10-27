"""Shared helpers for the MR_AHA benchmarks.

Loads the three benchmark problems into row-major CuPy arrays (the
column-major shapes of the documentation, reversed), together with a ready
:class:`mr_aha.MRConfig`, and provides a conjugate-gradient solver and timing
utilities used by both ``bench_python.py`` and ``mask_threshold.py``.

Datasets
--------
``2d``
    ``data/human_C-35_2D_radial.mat``: 2D golden-angle radial, 160 x 160,
    16 coils, L = 6, 377 spokes x 160 samples. The right-hand side of the CG
    problem is the stored ``AhAx_ref`` (= A^H A x computed with CPU finufft),
    so the exact solution is the stored ``x``.
``stack``
    ``data/AhAx_test_toeplitz.mat``: stack-of-stars, 320 x 320 x 56,
    12 coils, L = 12, 280 spokes x 320 samples. ``Phi2`` is stored as
    ``[L, L, Ntraj, Nz]`` with kz in natural FFT order (DC first, as its trace
    shows); it is permuted to ``[Ntraj, Nz, L, L]`` and fftshifted along kz
    into the centred order the library expects. The CG right-hand side is the
    stored ``Ahb``.
``3d``
    Synthetic 3D kooshball, 160^3, 12 coils, L = 12, 15680 spokes x 160
    samples, produced by ``make_kooshball.py``. The CG right-hand side is
    ``A^H A x_true`` for a synthetic ``x_true``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import cupy as cp  # noqa: E402
import mr_aha  # noqa: E402

DATA = ROOT / "data"
BENCH_DATA = Path(os.environ.get("MR_AHA_BENCH_DATA", ROOT / "benchmarks" / "data"))
RESULTS = Path(os.environ.get("MR_AHA_BENCH_RESULTS", ROOT / "benchmarks" / "results"))


def _h5_complex(ds):
    """Read a MATLAB v7.3 complex dataset into a complex64 NumPy array."""
    a = ds[()]
    if a.dtype.names:
        return (a["real"] + 1j * a["imag"]).astype(np.complex64)
    return a.astype(np.complex64)


def load(name: str):
    """Load a benchmark problem onto the GPU.

    Args:
        name: ``'2d'``, ``'stack'`` or ``'3d'``.

    Returns:
        ``dict`` with keys ``cfg`` (unmasked :class:`mr_aha.MRConfig`),
        ``traj_type``, ``n_dim``, ``L``, ``n_coils``, ``SEs``, ``phi2``,
        ``traj``, ``b`` (CG right-hand side), ``x_true`` (or ``None``) and
        ``desc``. All arrays are row-major CuPy arrays.
    """
    if name == "2d":
        import scipy.io as sio

        d = sio.loadmat(DATA / "human_C-35_2D_radial.mat")
        L, Nt = 6, 377
        out = dict(
            traj_type="2D", n_dim=(160, 160), L=L,
            SEs=d["SEs"].T, phi2=d["phi2"].T.reshape(L, L, Nt),
            traj=d["traj"].T.astype(np.float32), b=d["AhAx_ref"].T,
            x_true=d["x"].T,
            desc="2D radial 160x160, 16 coils, L=6, 377x160 samples",
        )
    elif name == "stack":
        import h5py

        with h5py.File(DATA / "AhAx_test_toeplitz.mat", "r") as h:
            P = _h5_complex(h["Phi2"])              # [z, t, col, row]
            SEs = _h5_complex(h["SEs"])             # (Nc, Nz, Ny, Nx)
            b = _h5_complex(h["Ahb"])               # (L, Nz, Ny, Nx)
            om = h["st/om"][()].astype(np.float32)  # (2, Ntraj*Nkx)
        Nz, Nt, L = P.shape[0], P.shape[1], P.shape[2]
        # Row-major (L, L, Nkz, Ntraj) with phi2[j, i, z, t] = Phi_{t,z}(i, j);
        # then natural -> centred kz order.
        phi2 = np.fft.fftshift(P.transpose(2, 3, 0, 1), axes=2)
        Nkx = om.shape[1] // Nt
        out = dict(
            traj_type="stack", n_dim=(320, 320, Nz), L=L,
            SEs=SEs, phi2=phi2, traj=om.reshape(2, Nkx, Nt), b=b, x_true=None,
            desc=f"stack-of-stars 320x320x{Nz}, 12 coils, L={L}, {Nt}x{Nkx} samples x {Nz} kz",
        )
    elif name == "3d":
        import scipy.io as sio

        f = BENCH_DATA / "kooshball_3d.mat"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing; run benchmarks/make_kooshball.py")
        d = sio.loadmat(f)
        L = int(d["L"].item())
        Nt = int(d["traj"].shape[0])
        out = dict(
            traj_type="3D", n_dim=(160, 160, 160), L=L,
            SEs=d["SEs"].T, phi2=d["phi2"].T.reshape(L, L, Nt),
            traj=d["traj"].T.astype(np.float32), b=d["b"].T, x_true=d["x_true"].T,
            desc=f"3D kooshball 160^3, 12 coils, L={L}, {Nt}x160 samples (synthetic)",
        )
    else:
        raise ValueError(name)

    g = {k: (cp.asarray(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v)
         for k, v in out.items()}
    for k in ("SEs", "phi2", "b", "x_true"):
        if g.get(k) is not None:
            g[k] = g[k].astype(cp.complex64)
    g["n_coils"] = int(g["SEs"].shape[0])
    g["cfg"] = make_cfg(g)
    return g


def make_cfg(d, **kw):
    """Configuration for a loaded problem, with optional overrides."""
    return mr_aha.make_config(d["traj_type"], d["n_dim"], d["L"],
                              n_coils=d["n_coils"], **kw)


def sync():
    """Block until all queued GPU work is done."""
    cp.cuda.Device().synchronize()


def timeit(fn, n=10, warmup=2):
    """Time a GPU function with device synchronisation.

    Args:
        fn: Zero-argument callable.
        n: Timed repetitions.
        warmup: Untimed repetitions first (plan caches, allocator warm-up).

    Returns:
        ``dict`` with ``median``, ``mean``, ``min`` in seconds and ``n``.
    """
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return dict(median=float(np.median(ts)), mean=float(ts.mean()),
                min=float(ts.min()), n=n)


def timed_once(fn):
    """Run ``fn`` once with synchronisation; return ``(result, seconds)``."""
    sync()
    t0 = time.perf_counter()
    r = fn()
    sync()
    return r, time.perf_counter() - t0


def cg(op, b, n_iter=20):
    """Plain conjugate gradient for the Hermitian system ``op(u) = b``.

    Starts from ``u = 0`` and runs exactly ``n_iter`` iterations without a
    stopping test, so that different operators are compared on equal terms.
    The same algorithm is implemented in ``bench_matlab.m``.

    Args:
        op: Callable applying the (Hermitian, PSD) operator.
        b: Right-hand side.
        n_iter: Number of iterations.

    Returns:
        ``(u, residual_norms)`` where ``residual_norms[i]`` is ``||r_i|| /
        ||b||`` after iteration ``i`` (index 0 is the initial residual, 1).
    """
    from cupy import cublas

    def dot(a, c):
        # cuBLAS dotc: conj(a) . c without a full-size temporary (cp.vdot
        # materialises the elementwise product, which matters for 3D data).
        return complex(cublas.dotc(a.ravel(), c.ravel()))

    u = cp.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = dot(r, r).real
    bn = rs ** 0.5
    res = [1.0]
    for _ in range(n_iter):
        Ap = op(p)
        alpha = rs / dot(p, Ap).real
        u += np.complex64(alpha) * p
        r -= np.complex64(alpha) * Ap
        rs_new = dot(r, r).real
        p *= np.complex64(rs_new / rs)
        p += r
        rs = rs_new
        res.append(rs ** 0.5 / bn)
        del Ap
    return u, res


def rel(a, b):
    """Relative 2-norm difference ``||a - b|| / ||b||`` as a float."""
    return float(cp.linalg.norm((a - b).ravel()) / cp.linalg.norm(b.ravel()))


def gpu_info():
    """Name and memory of the current GPU, plus software versions."""
    import cufinufft

    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    return dict(
        gpu=props["name"].decode(),
        gpu_mem_gb=round(props["totalGlobalMem"] / 2**30, 1),
        cupy=cp.__version__, cufinufft=cufinufft.__version__,
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        host=os.uname().nodename.split(".")[0],
    )


def save_json(obj, path):
    """Write ``obj`` as pretty JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
