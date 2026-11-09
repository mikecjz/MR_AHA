"""Accuracy of one Toeplitz ``A^H A`` against the NUFFT ``A^H A``.

For every benchmark problem (see :mod:`common`) one application of the
unmasked and of the default-masked (neighbourhood, radius 2) Toeplitz operator
(:func:`mr_aha.create_op`, one GPU, kernel computed at creation)
is compared with the NUFFT operator (cufinufft, default ``gpu_method``, on
CUDA; mlx-nufft on Metal; ``eps = 1e-6``) on the same input as ``mask_threshold.py``: the true
coefficient image where it is known, the CG right-hand side otherwise. The
error is ``||T x - A^H A x|| / ||A^H A x||``. A Toeplitz operator that does
not fit in device memory is recorded as ``skipped``.

Usage::

    python benchmarks/op_error.py [2d stack 3d]

Results: ``benchmarks/results/<host>/op_error_<problem>.json``.
"""

from __future__ import annotations

import math
import sys

from common import (RESULTS, cp, free_bytes, gpu_info, load, make_cfg, mr_aha,
                    rel, save_json)


def run(name):
    """Measure both Toeplitz operator errors for one problem."""
    d = load(name)
    info = gpu_info()
    x = d["x_true"] if d["x_true"] is not None else d["b"]
    with mr_aha.AhANufftPlan(d["cfg"], d["traj"]) as p:
        ref = p.apply(x, d["SEs"], d["phi2"])
    free_bytes()
    out = dict(problem=name, desc=d["desc"], info=info)
    pad = math.prod(d["cfg"].phi2_dim[:3])
    ntri = mr_aha.num_tri(d["L"])
    for key, masked in (("toeplitz", False), ("toeplitz_masked", True)):
        cfg = make_cfg(d, is_masked=masked)
        mask = mr_aha.calc_mask(cfg, d["traj"]) if masked else None
        n = int(mask.size) if masked else pad
        rec = dict(kept_fraction=n / pad, kernel_gb=8 * n * ntri / 2**30)
        need = 8 * (n * ntri + pad * d["L"] + 4 * x.size) + (2 << 30)
        if need > free_bytes():
            rec["skipped"] = f"out of memory (needs ~{need / 2**30:.1f} GB)"
        else:
            with mr_aha.create_op(cfg, d["SEs"], d["phi2"], d["traj"], mask) as op:
                rec["op_error"] = rel(op(x), ref)
        del mask
        free_bytes()
        out[key] = rec
        print(f"  {name} {key}: {rec.get('op_error', rec.get('skipped'))}", flush=True)
    save_json(out, RESULTS / info["host"] / f"op_error_{name}.json")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["2d", "stack", "3d"]):
        run(nm)
        free_bytes()
