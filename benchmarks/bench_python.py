"""Speed and CG benchmark of the Toeplitz and NUFFT ``A^H A`` (Python).

For every dataset (see :mod:`common`) this measures

* **NUFFT** (:class:`mr_aha.AhANufftPlan`): plan time and the time of one
  ``A^H A`` with cufinufft's default ``gpu_method`` and with the fastest
  per-type method found by a small search;
* **Toeplitz, unmasked** (:func:`mr_aha.create_op`, one GPU): ``calc_toep``
  time (the kernel computed at creation), plan time (the rest of the
  creation) and ``A^H A`` time, whenever the kernel fits on the device (see
  :func:`common.toeplitz_op`);
* **Toeplitz, masked** with the library's default mask (neighbourhood of
  radius 2; results produced before that change used the kernel threshold
  0.05, see ``mask_method`` in the JSON): ``calc_mask``, ``calc_toep`` and
  ``A^H A`` time;
* **CG**: 20 iterations of plain conjugate gradient from zero with each
  operator, saving the solutions (``.npy``) and residual histories, and the
  relative differences between them (and to the ground truth if known).

On the Metal backend (Apple silicon) the NUFFT operator is mlx-nufft (no
``gpu_method`` search), ``calc_toep`` is the two-stage mlx-nufft path, and
the CG solutions are additionally compared with those of a CUDA run when
``$MR_AHA_CG_REF`` names a folder holding its ``cg_<dataset>_<operator>.npy``.

Usage::

    python benchmarks/bench_python.py [2d stack 3d]

Results: ``benchmarks/results/<host>/python_<dataset>.json`` plus
``cg_<dataset>_<operator>.npy``.
"""

from __future__ import annotations

import math
import sys
import time

import numpy as np

import os
from pathlib import Path

from common import (BACKEND, RESULTS, cg, cp, free_bytes, gpu_info, load,
                    make_cfg, mr_aha, rel, save_json, sync, timed_once, timeit,
                    toeplitz_op)

N_ITER = 20
N_REP = {"2d": 20, "stack": 5, "3d": 5}
METHODS = [(1, 1), (1, 2), (2, 1), (2, 2)]  # (type-1 method, type-2 method)


def nufft_plan(d, methods=None):
    """NUFFT plan with optional ``(type-1, type-2)`` gpu_method override."""
    kw = {}
    if methods is not None:
        kw = dict(type1_opts={"gpu_method": methods[0]},
                  type2_opts={"gpu_method": methods[1]})
    return mr_aha.AhANufftPlan(d["cfg"], d["traj"], **kw)


def bench_nufft(d, out, n_rep):
    """Time the NUFFT operator; return the fastest ``(t1, t2)`` methods."""
    x = d["b"]
    p, t_plan = timed_once(lambda: nufft_plan(d))
    out["nufft_default"] = dict(
        plan_s=t_plan,
        aha=timeit(lambda: p.apply(x, d["SEs"], d["phi2"]), n=n_rep, warmup=1))
    p.close()
    if BACKEND != "cuda":
        # mlx-nufft has no gpu_method to tune.
        out["nufft_tuned"] = out["nufft_default"]
        return None

    search = {}
    for m in METHODS:
        try:
            p = nufft_plan(d, m)
            search[f"{m[0]}{m[1]}"] = timeit(
                lambda: p.apply(x, d["SEs"], d["phi2"]), n=2, warmup=1)["median"]
            p.close()
        except Exception as exc:  # e.g. shared-memory limits of a method
            search[f"{m[0]}{m[1]}"] = f"failed: {exc}"
    ok = {k: v for k, v in search.items() if isinstance(v, float)}
    best = min(ok, key=ok.get)
    best_m = (int(best[0]), int(best[1]))
    p, t_plan = timed_once(lambda: nufft_plan(d, best_m))
    out["nufft_tuned"] = dict(
        gpu_method_type1=best_m[0], gpu_method_type2=best_m[1], search=search,
        plan_s=t_plan,
        aha=timeit(lambda: p.apply(x, d["SEs"], d["phi2"]), n=n_rep, warmup=1))
    p.close()
    return best_m


def bench_toeplitz(d, out, key, masked, n_rep):
    """Create and time a Toeplitz operator (kernel computed at creation).

    Returns:
        The :class:`mr_aha.AhAOp` (holding its kernel), or ``None`` when it
        does not fit in memory.
    """
    ntri = mr_aha.num_tri(d["L"])
    pad = math.prod(d["cfg"].phi2_dim[:3])
    rec = {}
    if masked:
        cfg = make_cfg(d, is_masked=True)
        mask, rec["calc_mask_s"] = timed_once(lambda: mr_aha.calc_mask(cfg, d["traj"]))
        n = int(mask.size)
        rec["mask_method"] = ("threshold" if cfg.mask_method == mr_aha.MaskMethod.THRESHOLD
                              else "neighbour")
        rec["mask_radius"] = cfg.mask_radius
        rec["mask_threshold"] = cfg.mask_threshold
    else:
        cfg, mask, n = d["cfg"], None, pad
    rec["kept_fraction"] = n / pad
    rec["kernel_gb"] = n * ntri * 8 / 2**30
    # Kernel + padded buffer + compact scratch + CG vectors + a margin.
    need = 8 * (n * ntri + pad * d["L"] + (n * d["L"] if n < pad else 0)
                 + 5 * d["b"].size) + (1 << 30)
    if need > free_bytes():
        rec["skipped"] = f"needs ~{need / 2**30:.1f} GB of device memory"
        out[key] = rec
        print(f"  {key}: skipped ({rec['skipped']})")
        return None
    op, rec["calc_toep_s"], rec["plan_s"] = toeplitz_op(
        cfg, d["SEs"], d["phi2"], d["traj"], mask)
    del mask  # the operator keeps its own reference
    rec["aha"] = timeit(lambda: op(d["b"]), n=n_rep, warmup=1)
    out[key] = rec
    return op


def run(name):
    """Run every measurement for one dataset."""
    d = load(name)
    host = gpu_info()["host"]
    res_dir = RESULTS / host
    res_dir.mkdir(parents=True, exist_ok=True)
    n_rep = N_REP[name]
    out = dict(dataset=name, desc=d["desc"], info=gpu_info(), n_iter=N_ITER)
    print(f"\n== {name}: {d['desc']} on {out['info']['gpu']}")

    best_m = bench_nufft(d, out, n_rep)
    print(f"  NUFFT  default {out['nufft_default']['aha']['median']:.4f} s, "
          f"tuned {best_m} {out['nufft_tuned']['aha']['median']:.4f} s", flush=True)

    cg_out = {}

    def run_cg(key, op):
        sync()
        t0 = time.perf_counter()
        u, res = cg(op, d["b"], N_ITER)
        sync()
        cg_out[key] = dict(total_s=time.perf_counter() - t0, residuals=res)
        np.save(res_dir / f"cg_{name}_{key}.npy", u.get() if BACKEND == "cuda" else u)
        return u

    # NUFFT CG (tuned methods).
    with nufft_plan(d, best_m) as p:
        u_nufft = run_cg("nufft", lambda v: p.apply(v, d["SEs"], d["phi2"]))
    print(f"  CG NUFFT       {cg_out['nufft']['total_s']:.2f} s")

    sols = {"nufft": u_nufft}
    for key, masked in (("toeplitz", False), ("toeplitz_masked", True)):
        op = bench_toeplitz(d, out, key, masked, n_rep)
        if op is None:
            continue
        rec = out[key]
        print(f"  {key:15s} calc_toep {rec['calc_toep_s']:.2f} s, "
              f"AhA {rec['aha']['median']:.4f} s, kept {rec['kept_fraction']:.3f}")
        sols[key] = run_cg(key, op)
        print(f"  CG {key:15s} {cg_out[key]['total_s']:.2f} s")
        op.close()  # releases the kernel
        del op
        free_bytes()

    # Differences between the CG solutions (and to the truth, if known).
    diffs = {}
    keys = list(sols)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            diffs[f"{a}_vs_{b}"] = rel(sols[a], sols[b])
        if d["x_true"] is not None:
            diffs[f"{a}_vs_truth"] = rel(sols[a], d["x_true"])
    out["cg"] = cg_out
    out["cg_diffs"] = diffs
    # Against a CUDA run of the same benchmark (Metal validation at full size).
    ref_dir = os.environ.get("MR_AHA_CG_REF")
    if ref_dir:
        for k, u in sols.items():
            f = Path(ref_dir) / f"cg_{name}_{k}.npy"
            if f.exists():
                diffs[f"{k}_vs_cuda_{k}"] = rel(u, cp.asarray(np.load(f)))
        f = Path(ref_dir) / f"cg_{name}_toeplitz.npy"
        if f.exists() and "nufft" in sols:
            diffs["nufft_vs_cuda_toeplitz"] = rel(sols["nufft"], cp.asarray(np.load(f)))
    for k, v in diffs.items():
        print(f"  {k:36s} {v:.3e}")
    save_json(out, res_dir / f"python_{name}.json")


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["2d", "stack", "3d"]):
        run(ds)
        free_bytes()
