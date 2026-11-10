"""Speed and accuracy of ``calc_toep`` at cufinufft upsampling factor 1.25 vs 2.0.

``calc_toep`` computes the Toeplitz kernel with one batched cufinufft type-1
transform per triangular component. cufinufft's internal upsampling factor
sigma (not the Toeplitz grid ``phi2_dim``, which stays 2N) sets the size of its
fine grid and the width of its spreading kernel: sigma = 1.25 shrinks the fine
grid by (1.25/2)^D (0.39x in 2D, 0.24x in 3D) but widens the kernel by about
50%. This script switches sigma through the library's experimental environment
variable ``MR_AHA_TOEP_UPSAMPFAC`` (read when the operator is created) and,
for each problem, tolerance ``eps`` and type-1 spreading method
(``MR_AHA_TOEP_GPU_METHOD``: 0 = cufinufft's choice, 1 = GM-sort), records

* ``toep_s``: the kernel computation time inside :func:`mr_aha.create_op`
  (the operator's ``info['toep_ms']``), the minimum over ``N_REP`` creations;
* ``op_error``: ``||T x - A^H A x|| / ||A^H A x||`` of one application of the
  Toeplitz operator against the NUFFT operator (cufinufft on CUDA, mlx-nufft
  on Metal; sigma = 2, eps = 1e-6), on the same input as ``op_error.py``.

On Metal the kernel's NUFFT is mlx-nufft (``mr_aha._psf``), which reads the
same ``MR_AHA_TOEP_UPSAMPFAC``; it has no spreading-method switch, so only
method 0 is run there.

The 3D problem uses the default (neighbourhood) mask, since its unmasked
kernel does not fit on a 24 GB GPU; the others are unmasked.

Usage::

    python benchmarks/toep_upsampfac.py [2d stack 3d]

Results: ``benchmarks/results/<host>/toep_upsampfac_<problem>.json``.
"""

from __future__ import annotations

import os
import sys

from common import (BACKEND, RESULTS, free_bytes, gpu_info, load, make_cfg, mr_aha,
                    rel, save_json, toeplitz_op)

N_REP = 2
SIGMAS = (2.0, 1.25)
EPSS = (1e-6, 1e-5, 1e-4)
METHODS = (0, 1) if BACKEND == "cuda" else (0,)


def run(name):
    """Sweep sigma, eps and spreading method for one problem."""
    d = load(name)
    info = gpu_info()
    x = d["x_true"] if d["x_true"] is not None else d["b"]
    with mr_aha.AhANufftPlan(d["cfg"], d["traj"]) as p:
        ref = p.apply(x, d["SEs"], d["phi2"])
    free_bytes()
    masked = name == "3d"
    mask = mr_aha.calc_mask(make_cfg(d, is_masked=True), d["traj"]) if masked else None
    rows = []
    for eps in EPSS:
        cfg = make_cfg(d, is_masked=masked, eps=eps)
        for method in METHODS:
            for sigma in SIGMAS:
                os.environ["MR_AHA_TOEP_UPSAMPFAC"] = str(sigma)
                os.environ["MR_AHA_TOEP_GPU_METHOD"] = str(method)
                ts, err = [], None
                for _ in range(N_REP):
                    op, t_kernel, _ = toeplitz_op(cfg, d["SEs"], d["phi2"],
                                                  d["traj"], mask)
                    with op:
                        ts.append(t_kernel)
                        if err is None:
                            err = rel(op(x), ref)
                    free_bytes()
                row = dict(eps=eps, gpu_method=method, upsampfac=sigma,
                           toep_s=min(ts), toep_all_s=ts, op_error=err)
                rows.append(row)
                print(f"  {name} eps={eps:.0e} method={method} sigma={sigma}: "
                      f"toep {min(ts):7.3f} s  op_error {err:.3e}", flush=True)
    os.environ.pop("MR_AHA_TOEP_UPSAMPFAC", None)
    os.environ.pop("MR_AHA_TOEP_GPU_METHOD", None)
    out = dict(problem=name, desc=d["desc"], info=info, masked=masked, rows=rows)
    save_json(out, RESULTS / info["host"] / f"toep_upsampfac_{name}.json")


if __name__ == "__main__":
    for nm in (sys.argv[1:] or ["2d", "stack", "3d"]):
        run(nm)
        free_bytes()
