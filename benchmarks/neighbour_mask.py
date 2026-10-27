"""Experiment: a trajectory-neighbourhood mask instead of the kernel threshold.

Alternative to :func:`mr_aha.calc_mask`: keep every upsampled-grid location
within a few grid cells of a k-space sample, and nothing else.

Neighbourhood mask of radius ``r``
    Each sample ``k`` (radians) is mapped to its nearest location on the
    ``M = 2N`` upsampled Toeplitz grid, ``round(k * M / (2 pi)) + M // 2``
    (mod ``M``, the grid is periodic). Those locations are marked and the
    marks are dilated by a ``(2r + 1)^d`` box (Chebyshev distance ``<= r``,
    periodic). ``r = 0`` keeps only the nearest grid location of each sample.
    This study led to the method becoming the library default
    (``mask_method='neighbour'``, ``mask_radius=2``); :func:`neighbour_mask`
    here is a CuPy reference that gives bit-identical masks.

For each problem and radius the script records the kept fraction of the grid
and the error of one masked Toeplitz ``A^H A`` against the NUFFT operator, for
an image-like input and for complex white noise, normalised two ways:

* ``/out``: ``||T_mask x - A^H A x|| / ||A^H A x||`` (as in the other
  benchmarks);
* ``/in``: ``||T_mask x - A^H A x|| / (||A^H A|| ||x||)``, with the operator
  norm from power iteration on the NUFFT operator. This does not penalise
  inputs that the operator itself attenuates.

The threshold masks of ``calc_mask`` (0.05, 0.01, 0.003) and the unmasked
operator (where it fits) are evaluated the same way for reference.

Problems
--------
``spiral``
    Two-arm spiral (``make_spiral.py``), 256^2, 12 coils, L = 12, synthetic
    images / coils / Gram blocks (see ``spiral_mask.py``).
``radial64``
    64 of the 377 spokes of the 2D radial data
    (``data/human_C-35_2D_radial.mat``), 160^2, 16 coils, L = 6; input image
    = the stored coefficients ``x``. The data's spokes are stored in
    sequential angle order (180/377 degrees apart), so every 377/64-th spoke
    is taken to get 64 spokes uniformly spread over 180 degrees.
``radial280``
    In-plane slice of the stack-of-stars data (``data/AhAx_test_toeplitz.mat``):
    280 spokes x 320 samples, 320^2, 12 coils, L = 12, the Gram blocks of
    the kz = 0 plane, coil maps of the middle slice and the middle slice of
    ``Ahb`` as input image. The stack mask is this in-plane mask replicated
    over kz, so kept fractions carry over unchanged.
``kooshball``
    Synthetic 3D kooshball (``make_kooshball.py``), 15680 spokes x 160
    samples, 160^3, 12 coils, L = 12; input image = ``x_true``.

Usage::

    python benchmarks/neighbour_mask.py [spiral radial64 radial280 kooshball]

Results: ``benchmarks/results/<host>/neighbour_mask_<problem>.json`` and the
figure ``neighbour_mask_<problem>.png`` next to it.
"""

from __future__ import annotations

import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import (RESULTS, cp, gpu_info, load, mr_aha, rel,  # noqa: E402
                    save_json, sync)

RADII = [0, 1, 2, 3, 4, 6, 8]
THRESHOLDS = [0.05, 0.01, 0.003]
FIG_RADII = [0, 1, 2, 4]
INK, MUTED, KEPT, SAMPLE = "#1f1f1f", "#6b6b6b", "#2a5c8a", "#d9480f"


# ---------------------------------------------------------------- problems --
def problem(name):
    """Load one test problem as row-major CuPy arrays.

    Returns:
        ``dict`` with ``traj_type`` (``'2D'``/``'3D'``), ``n_dim``
        (column-major order), ``L``, ``n_coils``, ``traj`` ``(ndim, Nkx,
        Ntrajs)``, ``phi2``, ``SEs``, ``x`` (image-like input) and ``desc``.
    """
    if name == "spiral":
        from spiral_mask import NC, L, load_traj, synthetic

        traj, nc, ns = load_traj()
        d = synthetic("2d", traj, nc, np.random.default_rng(0))
        return dict(traj_type="2D", n_dim=d["n_dim"], L=L, n_coils=NC,
                    traj=traj, phi2=d["phi2"], SEs=d["SEs"], x=d["x"],
                    desc=f"two-arm spiral, {nc} x {ns} samples, 256^2, {NC} coils, L={L}")
    if name == "radial64":
        d = load("2d")
        nt = 64
        sel = cp.asarray(np.round(np.arange(nt) * d["traj"].shape[2] / nt).astype(np.int64))
        return dict(traj_type="2D", n_dim=d["n_dim"], L=d["L"],
                    n_coils=d["n_coils"],
                    traj=cp.ascontiguousarray(d["traj"][:, :, sel]),
                    phi2=cp.ascontiguousarray(d["phi2"][:, :, sel]),
                    SEs=d["SEs"], x=d["x_true"],
                    desc=f"radial, {nt} uniform spokes x 160 samples, "
                         f"160^2, {d['n_coils']} coils, L={d['L']}")
    if name == "radial280":
        d = load("stack")
        nz = d["phi2"].shape[2]
        zc = nz // 2  # kz = 0 in centred order; middle image slice
        out = dict(traj_type="2D", n_dim=d["n_dim"][:2], L=d["L"],
                   n_coils=d["n_coils"], traj=d["traj"],
                   phi2=cp.ascontiguousarray(d["phi2"][:, :, zc, :]),
                   SEs=cp.ascontiguousarray(d["SEs"][:, zc]),
                   x=cp.ascontiguousarray(d["b"][:, zc]),
                   desc=f"radial, 280 spokes x 320 samples (stack-of-stars in-plane), "
                        f"320^2, {d['n_coils']} coils, L={d['L']}")
        del d
        return out
    if name == "kooshball":
        d = load("3d")
        return dict(traj_type="3D", n_dim=d["n_dim"], L=d["L"],
                    n_coils=d["n_coils"], traj=d["traj"], phi2=d["phi2"],
                    SEs=d["SEs"], x=d["x_true"],
                    desc=f"3D kooshball, 15680 spokes x 160 samples, 160^3, "
                         f"{d['n_coils']} coils, L={d['L']}")
    raise ValueError(name)


# ------------------------------------------------------------------- masks --
def grid_shape(cfg):
    """Row-major shape of the upsampled grid, ``(My, Mx)`` or ``(Mz, My, Mx)``."""
    nd = 3 if cfg.traj_type == mr_aha.TrajType.THREE_D else 2
    return tuple(int(cfg.phi2_dim[i]) for i in reversed(range(nd)))


def neighbour_mask(cfg, traj, r):
    """Boolean neighbourhood mask of radius ``r`` on the upsampled grid.

    Args:
        cfg: Configuration (supplies the upsampled grid size).
        traj: ``(ndim, Nkx, Ntrajs)`` trajectory in radians.
        r: Chebyshev radius in grid cells; ``0`` keeps only the nearest grid
            location of every sample.

    Returns:
        Row-major boolean CuPy array of :func:`grid_shape`.
    """
    from cupyx.scipy.ndimage import maximum_filter

    shape = grid_shape(cfg)
    nd = len(shape)
    # traj[0] is kx, the fastest (last row-major) grid axis.
    idx = []
    for a in range(nd):
        M = shape[nd - 1 - a]
        i = cp.rint(traj[a].ravel() * (M / (2 * math.pi))).astype(cp.int64) + M // 2
        idx.append(i % M)
    marks = cp.zeros(shape, cp.uint8)
    marks[tuple(reversed(idx))] = 1
    if r > 0:
        marks = maximum_filter(marks, size=2 * r + 1, mode="wrap")
    return marks.astype(bool)


def threshold_mask(cfg, traj, th):
    """Boolean mask of :func:`mr_aha.calc_mask` at threshold ``th``."""
    nd = 3 if cfg.traj_type == mr_aha.TrajType.THREE_D else 2
    c = mr_aha.make_config(cfg.traj_type, tuple(cfg.n_dim[:nd]), 1, n_coils=1,
                           is_masked=True, mask_method="threshold", mask_threshold=th)
    m = cp.zeros(math.prod(grid_shape(cfg)), bool)
    m[mr_aha.calc_mask(c, traj)] = True
    return m.reshape(grid_shape(cfg))


# ---------------------------------------------------------------- accuracy --
def nufft_reference(d, cfg, x_rnd, n_power):
    """NUFFT outputs for both inputs and the operator norm (power iteration)."""
    with mr_aha.AhANufftPlan(cfg, d["traj"], type1_opts={"gpu_method": 1},
                             type2_opts={"gpu_method": 2}) as pn:
        def op(v):
            return pn.apply(v, d["SEs"], d["phi2"])
        ref_img, ref_rnd = op(d["x"]), op(x_rnd)
        v = x_rnd / cp.linalg.norm(x_rnd.ravel())
        lam = 0.0
        for _ in range(n_power):
            w = op(v)
            lam = float(cp.linalg.norm(w.ravel()))
            v = w / lam
            del w
        del v
    sync()
    return ref_img, ref_rnd, lam


def evaluate(d, cfg, mask_bool, x_rnd, ref_img, ref_rnd, lam):
    """Errors of the Toeplitz operator with ``mask_bool`` (``None`` = unmasked)."""
    pad = math.prod(grid_shape(cfg))
    L = d["L"]
    if mask_bool is None:
        c, mask, n = cfg, None, pad
    else:
        mask = cp.flatnonzero(mask_bool.ravel()).astype(cp.int64)
        n = int(mask.size)
        c = mr_aha.make_config(d["traj_type"], d["n_dim"], L,
                               n_coils=d["n_coils"], is_masked=True,
                               num_masked=n)
    row = dict(kept_fraction=n / pad)
    need = 8 * (n * mr_aha.num_tri(L) + pad * L + (n * L if mask is not None else 0)
                + 4 * d["x"].size) + (1 << 30)
    cp.get_default_memory_pool().free_all_blocks()
    if need > cp.cuda.runtime.memGetInfo()[0]:
        row["skipped"] = f"needs ~{need / 2**30:.1f} GB"
        return row
    Om = mr_aha.calc_toep(c, d["phi2"], d["traj"], mask=mask)
    nx, nr = float(cp.linalg.norm(d["x"].ravel())), float(cp.linalg.norm(x_rnd.ravel()))
    with mr_aha.AhAPlan(c) as p:
        e = float(cp.linalg.norm((p.apply(d["x"], d["SEs"], Om, mask=mask) - ref_img).ravel()))
        row["err_img_out"] = e / float(cp.linalg.norm(ref_img.ravel()))
        row["err_img_in"] = e / (lam * nx)
        e = float(cp.linalg.norm((p.apply(x_rnd, d["SEs"], Om, mask=mask) - ref_rnd).ravel()))
        row["err_rnd_out"] = e / float(cp.linalg.norm(ref_rnd.ravel()))
        row["err_rnd_in"] = e / (lam * nr)
    del Om, mask
    cp.get_default_memory_pool().free_all_blocks()
    return row


# ------------------------------------------------------------------ figure --
def plot(name, d, cfg, out):
    """Masks at ``FIG_RADII`` and the 0.05 threshold mask, full grid + zoom."""
    shape = grid_shape(cfg)
    M = shape[-1]
    three_d = len(shape) == 3
    t = cp.asnumpy(d["traj"])
    kx = (t[0] * M / (2 * np.pi) + M // 2).ravel()
    ky = (t[1] * M / (2 * np.pi) + M // 2).ravel()
    if three_d:  # samples within half a cell of the central kz plane
        kz = (t[2] * M / (2 * np.pi) + M // 2).ravel()
        near = np.abs(kz - M // 2) < 0.5
        kx, ky = kx[near], ky[near]
    # Zoom on a patch at 60 % of k_max along +kx, where sampling is sparse.
    z = max(M // 10, 24)
    x0, y0 = int(M // 2 + 0.6 * M / 2) - z // 2, M // 2 - z // 2

    panels = [(f"radius {r}", lambda r=r: neighbour_mask(cfg, d["traj"], r)) for r in FIG_RADII]
    panels.append(("threshold 0.05 (calc_mask)", lambda: threshold_mask(cfg, d["traj"], 0.05)))
    fig, axes = plt.subplots(2, len(panels), figsize=(3.3 * len(panels), 6.9),
                             constrained_layout=True)
    cmap = matplotlib.colors.ListedColormap(["#f4f4f4", KEPT])
    for j, (title, fn) in enumerate(panels):
        m = fn()
        kept = float(m.mean())
        img = cp.asnumpy(m[shape[0] // 2] if three_d else m)
        del m
        for i, ax in enumerate(axes[:, j]):
            ax.imshow(img, origin="lower", cmap=cmap, interpolation="nearest",
                      vmin=0, vmax=1)
            if i == 1:
                ax.set_xlim(x0 - 0.5, x0 + z - 0.5)
                ax.set_ylim(y0 - 0.5, y0 + z - 0.5)
                sel = (kx >= x0 - 1) & (kx <= x0 + z) & (ky >= y0 - 1) & (ky <= y0 + z)
                ax.plot(kx[sel], ky[sel], ".", ms=2.5, color=SAMPLE)
            else:
                ax.add_patch(matplotlib.patches.Rectangle(
                    (x0 - 0.5, y0 - 0.5), z, z, fill=False, ec=SAMPLE, lw=1))
                ax.set_title(f"{title}\nkept {kept:.1%}", color=INK, fontsize=10)
            ax.tick_params(colors=MUTED, labelsize=7)
            for s in ax.spines.values():
                s.set_color(MUTED)
    axes[0, 0].set_ylabel("full grid" + (" (central kz plane)" if three_d else ""),
                          color=MUTED)
    axes[1, 0].set_ylabel(f"zoom {z}x{z}, samples in orange", color=MUTED)
    fig.suptitle(f"{d['desc']}: {' x '.join(map(str, reversed(shape)))} upsampled grid, "
                 f"kept locations in blue", color=INK, fontsize=11)
    fig.savefig(out, dpi=130)
    plt.close(fig)


# -------------------------------------------------------------------- main --
def run(name):
    """Evaluate every mask for one problem, save JSON and figure."""
    d = problem(name)
    cfg = mr_aha.make_config(d["traj_type"], d["n_dim"], d["L"], n_coils=d["n_coils"])
    cp.random.seed(0)
    x_rnd = (cp.random.standard_normal(d["x"].shape)
             + 1j * cp.random.standard_normal(d["x"].shape)).astype(cp.complex64)
    ref_img, ref_rnd, lam = nufft_reference(d, cfg, x_rnd,
                                            10 if name == "kooshball" else 30)
    print(f"\n== {name}: {d['desc']}, ||A^H A|| ~ {lam:.4g}")
    print(f"{'mask':>16} {'kept':>7} {'img/out':>9} {'img/in':>9} {'rnd/out':>9} {'rnd/in':>9}")

    rows = []
    cases = ([("neighbourhood", r) for r in RADII]
             + [("threshold", th) for th in THRESHOLDS] + [("unmasked", None)])
    for kind, p in cases:
        if kind == "neighbourhood":
            m = neighbour_mask(cfg, d["traj"], p)
        elif kind == "threshold":
            m = threshold_mask(cfg, d["traj"], p)
        else:
            m = None
        row = dict(kind=kind, param=p, **evaluate(d, cfg, m, x_rnd, ref_img, ref_rnd, lam))
        del m
        label = f"{kind[:5]} {p}" if p is not None else kind
        if "skipped" in row:
            print(f"{label:>16} {row['kept_fraction']:7.4f}   (skipped: {row['skipped']})")
        else:
            print(f"{label:>16} {row['kept_fraction']:7.4f} {row['err_img_out']:9.2e} "
                  f"{row['err_img_in']:9.2e} {row['err_rnd_out']:9.2e} {row['err_rnd_in']:9.2e}")
        rows.append(row)

    info = gpu_info()
    outdir = RESULTS / info["host"]
    save_json(dict(problem=name, desc=d["desc"], info=info, op_norm=lam, rows=rows),
              outdir / f"neighbour_mask_{name}.json")
    del ref_img, ref_rnd, x_rnd
    cp.get_default_memory_pool().free_all_blocks()
    plot(name, d, cfg, outdir / f"neighbour_mask_{name}.png")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["spiral", "radial64", "radial280", "kooshball"]):
        run(nm)
        cp.get_default_memory_pool().free_all_blocks()
