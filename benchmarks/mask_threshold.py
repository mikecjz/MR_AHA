"""Experiment: choose the default threshold of ``mr_aha.calc_mask``.

For every dataset and every threshold this records

* the fraction of the upsampled grid that is retained,
* the time of ``calc_mask``, of the masked ``calc_toep`` and of one masked
  ``AhA`` application,
* the relative difference between the solution of 20 CG iterations (from
  zero) with the masked Toeplitz operator and with the NUFFT operator,
* the relative error of the masked Toeplitz ``AhA`` against the NUFFT
  ``AhA`` (the exact operator up to the NUFFT tolerance), for an image-like
  input and for complex white noise. The image-like input is the true
  coefficient image where one exists (2D, 3D) and the stored ``A^H b`` for the
  stack data. White noise has a flat spectrum, so it is the worst case for
  dropping k-space locations; images concentrate their energy at low spatial
  frequencies and are representative of what CG actually sees.

Thresholds are absolute, in samples (see ``mr_aha.calc_mask``). The unmasked
operator (threshold "none") is included whenever its kernel fits in device
memory.

Usage::

    python benchmarks/mask_threshold.py [2d stack 3d]

Results go to ``benchmarks/results/<host>/mask_threshold_<dataset>.json``.
"""

from __future__ import annotations

import math
import sys

from common import (RESULTS, cg, cp, gpu_info, load, make_cfg, mr_aha, rel,
                    save_json, sync, timed_once, timeit)

THRESHOLDS = [1.0, 0.3, 0.1, 0.05, 0.03, 0.01, 0.003, 0.0]
N_ITER = 20


def free_bytes():
    """Free device memory after releasing CuPy's cached blocks."""
    cp.get_default_memory_pool().free_all_blocks()
    return cp.cuda.runtime.memGetInfo()[0]


def run(name):
    """Sweep the thresholds for one dataset and save the results."""
    d = load(name)
    cfg = d["cfg"]
    pad = math.prod(cfg.phi2_dim[:3])
    ntri = mr_aha.num_tri(d["L"])
    x_img = d["x_true"] if d["x_true"] is not None else d["b"]
    cp.random.seed(0)
    x_rnd = (cp.random.standard_normal(x_img.shape)
             + 1j * cp.random.standard_normal(x_img.shape)).astype(cp.complex64)

    with mr_aha.AhANufftPlan(cfg, d["traj"], type1_opts={"gpu_method": 1},
                             type2_opts={"gpu_method": 2}) as pn:
        ref_img = pn.apply(x_img, d["SEs"], d["phi2"])
        ref_rnd = pn.apply(x_rnd, d["SEs"], d["phi2"])
        u_ref, _ = cg(lambda v: pn.apply(v, d["SEs"], d["phi2"]), d["b"], N_ITER)
    sync()

    rows = []
    print(f"\n== {name}: {d['desc']}")
    print(f"{'threshold':>9} {'kept':>7} {'err(img)':>9} {'err(rand)':>9} "
          f"{'err(CG)':>9} {'mask s':>7} {'toep s':>7} {'AhA s':>7}")
    for th in THRESHOLDS + [None]:
        row = dict(threshold=th)
        if th is None:  # unmasked reference Toeplitz operator
            c = cfg
            mask = None
            n = pad
            row["mask_s"] = 0.0
        else:
            c = make_cfg(d, is_masked=True, mask_method="threshold", mask_threshold=th)
            mask, row["mask_s"] = timed_once(lambda: mr_aha.calc_mask(c, d["traj"]))
            n = int(mask.size)
        row["kept_fraction"] = n / pad
        # Kernel + padded buffer + compact scratch + CG vectors + a margin.
        need = 8 * (n * ntri + pad * d["L"] + (n * d["L"] if n < pad else 0)
                     + 5 * d["b"].size) + (1 << 30)
        if need > free_bytes():
            row["skipped"] = "insufficient device memory"
            print(f"{str(th):>9} {n / pad:7.4f}   (skipped: needs {need / 2**30:.1f} GB)")
            rows.append(row)
            del mask
            continue
        Om, row["calc_toep_s"] = timed_once(
            lambda: mr_aha.calc_toep(c, d["phi2"], d["traj"], mask=mask))
        with mr_aha.AhAPlan(c) as p:
            row["err_img"] = rel(p.apply(x_img, d["SEs"], Om, mask=mask), ref_img)
            row["err_rand"] = rel(p.apply(x_rnd, d["SEs"], Om, mask=mask), ref_rnd)
            row["aha_s"] = timeit(lambda: p.apply(x_img, d["SEs"], Om, mask=mask),
                                  n=3, warmup=1)["median"]
            u, _ = cg(lambda v: p.apply(v, d["SEs"], Om, mask=mask), d["b"], N_ITER)
            row["err_cg"] = rel(u, u_ref)
            del u
        del Om, mask
        print(f"{str(th):>9} {row['kept_fraction']:7.4f} {row['err_img']:9.2e} "
              f"{row['err_rand']:9.2e} {row['err_cg']:9.2e} {row['mask_s']:7.3f} "
              f"{row['calc_toep_s']:7.2f} {row['aha_s']:7.3f}")
        rows.append(row)

    info = gpu_info()
    save_json(dict(dataset=name, desc=d["desc"], info=info, rows=rows),
              RESULTS / info["host"] / f"mask_threshold_{name}.json")


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["2d", "stack", "3d"]):
        run(ds)
        cp.get_default_memory_pool().free_all_blocks()
