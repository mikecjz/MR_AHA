"""Experiment: how much memory the k-space mask saves for a sparse spiral.

Radial and kooshball trajectories cover most of the 2x-upsampled Toeplitz
grid, so masking only drops 20-35 % of it (see ``mask_threshold.py``). A
sparse trajectory such as a two-arm spiral (``make_spiral.py``: 2 arms x 1024
samples, one revolution, Archimedean) leaves most of the grid empty, and this
script measures how much of the Toeplitz kernel the mask removes and what that
costs in accuracy.

Part 1, mask size (no kernel is built)
    For several in-plane matrix sizes ``N`` and every threshold, the fraction
    of the ``2N x 2N`` grid kept by ``mr_aha.calc_mask`` and the resulting size
    of ``phi2_toep`` (+ the ``int64`` mask) for ``L = 12``, for a 2D problem
    and for a stack-of-spirals with ``Nz = 56`` partitions. The stack mask is
    the in-plane mask replicated over all ``Mz = Nz`` planes, so the kept
    fraction is the same.

Part 2, accuracy and measured memory (synthetic problems, ``L = 12``,
12 coils)
    ``2d``: ``N x N`` spiral; ``stack``: ``N x N x Nz`` stack-of-spirals. The
    spatial coefficients are random ellipses/ellipsoids with energy decaying
    over the ranks, the coil maps smooth complex Gaussians, and each readout's
    Gram block is a random rank-3 PSD matrix ``B^H B``. For every threshold
    (and unmasked, when it fits) the script records the device bytes held by
    ``phi2_toep`` and the mask, the peak device memory during ``calc_toep``
    (with the default ``mem_fraction = 0.5`` and with ``0.02``) and during
    ``AhAPlan`` creation plus one ``AhA`` (sampled by a polling thread, so a
    lower bound), the
    relative error of one ``AhA`` against the NUFFT operator for the image and
    for white noise, and the difference between 20 CG iterations (from zero)
    with the masked Toeplitz and with the NUFFT operator, for
    ``b = A^H A x_true`` evaluated with NUFFT. The error of the NUFFT CG
    solution against ``x_true`` is recorded as the scale of the latter.

Usage::

    python benchmarks/make_spiral.py -o benchmarks/data/spiral_trajectory.mat
    python benchmarks/spiral_mask.py [sizes] [2d] [stack]

Results go to ``benchmarks/results/<host>/spiral_mask.json``.
"""

from __future__ import annotations

import math
import sys
import threading

import numpy as np

from common import (BENCH_DATA, RESULTS, cg, cp, gpu_info, mr_aha, rel,
                    save_json, sync, timed_once, timeit)

THRESHOLDS = [1.0, 0.3, 0.1, 0.05, 0.03, 0.01, 0.003, 0.0]
SIZES = [128, 192, 256, 320, 384]
L, NC, RANK_PHI, NZ_STACK, N_ITER = 12, 12, 3, 56, 20
N_2D, N_STACK = 256, 320
LOW_MEM_FRACTION = 0.02  #: calc_toep run that shows its minimum footprint


def load_traj():
    """Spiral trajectory as row-major ``(2, Ns, Nc)`` float32 CuPy array."""
    import scipy.io as sio

    f = BENCH_DATA / "spiral_trajectory.mat"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing; run benchmarks/make_spiral.py "
                                f"-o {f}")
    d = sio.loadmat(f)
    nc, ns = int(d["Nc"].item()), int(d["Ns"].item())
    t = d["trajectory"].reshape(nc, ns, 2).transpose(2, 1, 0)
    return cp.asarray(np.ascontiguousarray(t, dtype=np.float32)), nc, ns


def kernel_bytes(n_kept, L):
    """Bytes of ``phi2_toep`` (complex64) plus, if masked, the int64 mask."""
    return 8 * n_kept * mr_aha.num_tri(L)


def mask_sizes(traj):
    """Part 1: kept fraction and kernel size versus ``N`` and threshold."""
    rows = []
    print("\n== mask size, L = 12 (kernel + mask; stack has Nz = 56)")
    print(f"{'N':>4} {'threshold':>9} {'kept':>7} {'2D MB':>9} {'2D unm MB':>9} "
          f"{'stack GB':>9} {'unm GB':>7}")
    for N in SIZES:
        pad = 4 * N * N
        unm = kernel_bytes(pad, L)
        for th in THRESHOLDS:
            c = mr_aha.make_config("2D", (N, N), 1, n_coils=1, is_masked=True,
                                   mask_method="threshold", mask_threshold=th)
            n = int(mr_aha.calc_mask(c, traj).size)
            masked = kernel_bytes(n, L) + 8 * n
            rows.append(dict(N=N, threshold=th, kept_fraction=n / pad,
                             bytes_2d=masked, bytes_2d_unmasked=unm,
                             bytes_stack=masked * NZ_STACK,
                             bytes_stack_unmasked=unm * NZ_STACK))
            print(f"{N:4d} {th:9g} {n / pad:7.4f} {masked / 2**20:9.2f} "
                  f"{unm / 2**20:9.2f} {masked * NZ_STACK / 2**30:9.3f} "
                  f"{unm * NZ_STACK / 2**30:7.2f}")
    return rows


def grid(shape):
    """Normalised coordinates in ``[-1, 1)`` for a row-major ``shape``."""
    axes = [(np.arange(n) - n // 2) / (n / 2) for n in shape]
    return np.meshgrid(*axes, indexing="ij")


def synthetic(kind, traj, nc, rng):
    """Synthetic spiral problem on the GPU.

    Args:
        kind: ``'2d'`` (``N_2D^2``) or ``'stack'`` (``N_STACK^2 x NZ_STACK``).
        traj: Spiral trajectory ``(2, Ns, Nc)``.
        nc: Number of spiral arms (= ``Ntrajs``).
        rng: NumPy random generator.

    Returns:
        ``dict`` with ``traj_type``, ``n_dim`` (column-major order), ``x``
        ``(L, [Nz,] Ny, Nx)``, ``SEs`` ``(NC, [Nz,] Ny, Nx)`` and ``phi2``
        ``(L, L, [Nkz,] Ntrajs)``.
    """
    if kind == "2d":
        shape, traj_type, n_dim = (N_2D, N_2D), "2D", (N_2D, N_2D)
    else:
        shape = (NZ_STACK, N_STACK, N_STACK)
        traj_type, n_dim = "stack", (N_STACK, N_STACK, NZ_STACK)
    g = grid(shape)
    r2 = sum(c ** 2 for c in g)
    support = (r2 <= 0.85 ** 2).astype(np.float32)

    # Ellipses / ellipsoids, energy decaying over the ranks.
    x = np.zeros((L,) + shape, np.complex64)
    for _ in range(10):
        c = rng.uniform(-0.45, 0.45, len(shape))
        a = rng.uniform(0.1, 0.35, len(shape))
        inside = sum(((gi - ci) / ai) ** 2 for gi, ci, ai in zip(g, c, a)) <= 1
        amp = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) / (1 + np.arange(L))
        x += amp.reshape((L,) + (1,) * len(shape)).astype(np.complex64) * inside[None]
    x *= support[None]

    # Smooth complex Gaussian coils around the object, unit RSS on the support.
    gy, gx = g[-2], g[-1]
    gz = g[0] if len(shape) == 3 else 0.0
    maps = []
    for k in range(NC):
        ang = 2 * np.pi * k / NC
        cz = (-0.4 if k % 2 == 0 else 0.4) if len(shape) == 3 else 0.0
        d2 = (gx - 1.1 * np.cos(ang)) ** 2 + (gy - 1.1 * np.sin(ang)) ** 2 + (gz - cz) ** 2
        phase = ang + 0.8 * (gx * np.cos(ang) + gy * np.sin(ang))
        maps.append(np.exp(-d2 / (2 * 0.7 ** 2)) * np.exp(1j * phase))
    S = np.stack(maps)
    S = (S / np.maximum(np.sqrt((np.abs(S) ** 2).sum(0)), 1e-6) * support).astype(np.complex64)
    del maps, g, r2

    # Random rank-3 PSD Gram block per readout (and per kz for the stack).
    lead = (nc,) if kind == "2d" else (NZ_STACK, nc)
    B = (rng.standard_normal(lead + (RANK_PHI, L))
         + 1j * rng.standard_normal(lead + (RANK_PHI, L))) / math.sqrt(2 * RANK_PHI)
    G = np.einsum("...ki,...kj->...ij", B.conj(), B)        # [.., i, j] = Phi(i, j)
    # Row-major phi2[j, i, (z,) t] = Phi_{(t,z)}(i, j).
    phi2 = np.moveaxis(G, (-1, -2), (0, 1)).astype(np.complex64)

    return dict(traj_type=traj_type, n_dim=n_dim,
                x=cp.asarray(x), SEs=cp.asarray(S),
                phi2=cp.asarray(np.ascontiguousarray(phi2)), traj=traj,
                desc=(f"{'2D spiral' if kind == '2d' else 'stack-of-spirals'} "
                      f"{' x '.join(map(str, n_dim))}, {NC} coils, L={L}, "
                      f"{nc} arms x {traj.shape[1]} samples"))


class PeakMemory:
    """Context manager sampling the device's used memory every ~0.5 ms.

    ``peak`` is the maximum of ``total - free`` minus the usage on entry, i.e.
    the peak additional device memory during the block (a lower bound, since
    short-lived allocations can fall between samples). CuPy's pool is emptied
    on entry so cached blocks do not hide new allocations.
    """

    def __enter__(self):
        sync()
        cp.get_default_memory_pool().free_all_blocks()
        free, self.total = cp.cuda.runtime.memGetInfo()
        self.base = self.total - free
        self.peak = 0
        self._stop = threading.Event()
        dev = cp.cuda.Device().id

        def poll():
            with cp.cuda.Device(dev):
                while not self._stop.is_set():
                    used = self.total - cp.cuda.runtime.memGetInfo()[0]
                    self.peak = max(self.peak, used - self.base)
                    self._stop.wait(5e-4)

        self._t = threading.Thread(target=poll, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        sync()
        self._stop.set()
        self._t.join()
        used = self.total - cp.cuda.runtime.memGetInfo()[0]
        self.peak = max(self.peak, used - self.base)
        return False


def accuracy(kind, traj, nc):
    """Part 2: masked vs unmasked Toeplitz against NUFFT for one problem."""
    d = synthetic(kind, traj, nc, np.random.default_rng(0))
    cfg = mr_aha.make_config(d["traj_type"], d["n_dim"], L, n_coils=NC)
    pad = math.prod(cfg.phi2_dim[:3])
    x, S, P = d["x"], d["SEs"], d["phi2"]
    cp.random.seed(0)
    x_rnd = (cp.random.standard_normal(x.shape)
             + 1j * cp.random.standard_normal(x.shape)).astype(cp.complex64)

    with mr_aha.AhANufftPlan(cfg, traj, type1_opts={"gpu_method": 1},
                             type2_opts={"gpu_method": 2}) as pn:
        b = pn.apply(x, S, P)
        ref_rnd = pn.apply(x_rnd, S, P)
        t_nufft = timeit(lambda: pn.apply(x, S, P), n=3, warmup=1)["median"]
        u_ref, _ = cg(lambda v: pn.apply(v, S, P), b, N_ITER)
    sync()
    cg_truth = rel(u_ref, x)

    print(f"\n== {kind}: {d['desc']}  (NUFFT A^H A {t_nufft:.4f} s, "
          f"NUFFT CG vs x_true {cg_truth:.3e})")
    print(f"{'threshold':>9} {'kept':>7} {'kernel MB':>10} {'toep pk':>8} "
          f"{'toep lo':>8} {'AhA pk':>8} {'err(img)':>9} {'err(rand)':>9} "
          f"{'err(CG)':>9} {'toep s':>7} {'lo s':>7} {'AhA s':>7}")
    rows = []
    for th in [None] + THRESHOLDS:
        row = dict(threshold=th)
        if th is None:
            c, mask, n = cfg, None, pad
        else:
            c = mr_aha.make_config(d["traj_type"], d["n_dim"], L, n_coils=NC,
                                   is_masked=True, mask_method="threshold", mask_threshold=th)
            mask = mr_aha.calc_mask(c, traj)
            n = int(mask.size)
        row["kept_fraction"] = n / pad
        need = 8 * (n * mr_aha.num_tri(L) + pad * L + (n * L if mask is not None else 0)
                    + 6 * x.size) + (1 << 30)
        if need > cp.cuda.runtime.memGetInfo()[0]:
            row["skipped"] = f"needs ~{need / 2**30:.1f} GB"
            print(f"{str(th):>9} {n / pad:7.4f}   (skipped: {row['skipped']})")
            rows.append(row)
            del mask
            continue
        # calc_toep sizes its chunks to mem_fraction (default 0.5) of the free
        # memory, so its peak is opportunistic; a run with a small fraction
        # shows the footprint it actually needs.
        c_low = mr_aha.make_config(d["traj_type"], d["n_dim"], L, n_coils=NC,
                                   is_masked=mask is not None,
                                   mask_method="threshold",
                                   mask_threshold=th if th is not None else 0.05,
                                   num_masked=n if mask is not None else 0,
                                   mem_fraction=LOW_MEM_FRACTION)
        with PeakMemory() as pm:
            Om, row["calc_toep_low_s"] = timed_once(
                lambda: mr_aha.calc_toep(c_low, P, traj, mask=mask))
        row["peak_calc_toep_low_bytes"] = pm.peak
        del Om
        with PeakMemory() as pm:
            Om, row["calc_toep_s"] = timed_once(
                lambda: mr_aha.calc_toep(c, P, traj, mask=mask))
        row["peak_calc_toep_bytes"] = pm.peak
        row["kernel_bytes"] = Om.nbytes + (mask.nbytes if mask is not None else 0)
        # Plan creation allocates the per-coil work buffers, so it is inside
        # the measurement; the peak includes the output array (x.nbytes).
        with PeakMemory() as pm:
            p = mr_aha.AhAPlan(c)
            y = p.apply(x, S, Om, mask=mask)
        row["peak_aha_bytes"] = pm.peak
        with p:
            row["err_img"] = rel(y, b)
            del y
            row["err_rand"] = rel(p.apply(x_rnd, S, Om, mask=mask), ref_rnd)
            row["aha_s"] = timeit(lambda: p.apply(x, S, Om, mask=mask),
                                  n=3, warmup=1)["median"]
            u, _ = cg(lambda v: p.apply(v, S, Om, mask=mask), b, N_ITER)
            row["err_cg"] = rel(u, u_ref)
            del u
        del Om, mask
        print(f"{str(th):>9} {n / pad:7.4f} {row['kernel_bytes'] / 2**20:10.1f} "
              f"{row['peak_calc_toep_bytes'] / 2**20:8.0f} "
              f"{row['peak_calc_toep_low_bytes'] / 2**20:8.0f} "
              f"{row['peak_aha_bytes'] / 2**20:8.0f} "
              f"{row['err_img']:9.2e} {row['err_rand']:9.2e} "
              f"{row['err_cg']:9.2e} {row['calc_toep_s']:7.2f} "
              f"{row['calc_toep_low_s']:7.2f} {row['aha_s']:7.4f}")
        rows.append(row)
    out = dict(desc=d["desc"], nufft_aha_s=t_nufft, nufft_cg_vs_truth=cg_truth,
               rows=rows)
    del d, x, S, P, x_rnd, b, ref_rnd, u_ref
    cp.get_default_memory_pool().free_all_blocks()
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["sizes", "2d", "stack"]
    traj, nc, ns = load_traj()
    info = gpu_info()
    f = RESULTS / info["host"] / "spiral_mask.json"
    import json
    res = json.loads(f.read_text()) if f.exists() else {}
    res.update(info=info, trajectory=dict(arms=nc, samples=ns, revolutions=1,
                                          spiral="archimedes"))
    if "sizes" in which:
        res["sizes"] = mask_sizes(traj)
    for kind in ("2d", "stack"):
        if kind in which:
            res[kind] = accuracy(kind, traj, nc)
    save_json(res, f)
