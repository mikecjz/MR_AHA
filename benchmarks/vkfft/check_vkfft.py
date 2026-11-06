"""Accuracy and speed check of VkFFT-Metal (and MLX's FFT) for the MR_AHA grids.

For each case this script

1. draws a fixed-seed complex64 grid of row-major shape ``(L, [Mz,] My, Mx)``,
   which is the column-major ``[Mx, My, (Mz), L]`` padded grid of
   ``src/core/mr_aha.cu`` seen from Python,
2. runs ``vkfft_bench`` on it (built by ``build.sh``), which writes VkFFT's
   unnormalised forward transform and prints its own timings,
3. compares that output against a double-precision SciPy FFT over the
   spatial axes,
4. times and checks ``mlx.core.fft.fftn`` / ``ifftn`` on the same data, as
   the fallback FFT that needs no extra dependency.

Run on the Mac from the repository root, with the project venv::

    benchmarks/vkfft/build.sh
    .venv/bin/python benchmarks/vkfft/check_vkfft.py [--cases 2d stack 3d]

Results are printed and written as JSON next to this script
(``results_<hostname>.json``).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import socket
import subprocess
import tempfile
import time

import numpy as np
import scipy.fft

#: Benchmark cases: name -> (spatial FFT extents fastest-first, batch L).
#: These are the upsampled Toeplitz grids of the three benchmark problems in
#: ``benchmarks/README.md``: 2D 160^2 (L=6), stack 320^2 x 56 (L=12, no
#: upsampling along z) and 3D 160^3 (L=12).
CASES = {
    "2d": ((320, 320), 6),
    "stack": ((640, 640, 56), 12),
    "3d": ((320, 320, 320), 12),
}

HERE = pathlib.Path(__file__).resolve().parent


def rel_err(a: np.ndarray, ref: np.ndarray) -> float:
    """Relative L2 error ``||a - ref|| / ||ref||`` computed in double precision.

    Args:
        a: Array under test.
        ref: Reference array of the same shape.

    Returns:
        The relative error as a Python float.
    """
    num = np.linalg.norm((a.astype(np.complex128) - ref).ravel())
    return float(num / np.linalg.norm(ref.ravel()))


def run_vkfft(binary: pathlib.Path, dims, batch: int, x: np.ndarray,
              iters: int, coalesced: int) -> tuple[np.ndarray, dict]:
    """Run ``vkfft_bench`` on ``x`` and return its forward output and timings.

    Args:
        binary: Path to the built ``vkfft_bench`` executable.
        dims: Spatial FFT extents, fastest-varying first (``Mx, My[, Mz]``).
        batch: Number of batched transforms ``L``.
        x: Input of row-major shape ``(L, [Mz,] My, Mx)``, complex64.
        iters: Timed repetitions inside the benchmark.
        coalesced: VkFFT ``coalescedMemory`` in bytes (0 keeps VkFFT's default).

    Returns:
        ``(forward, info)`` where ``forward`` has the shape of ``x`` and
        ``info`` holds the timings parsed from the benchmark's output.
    """
    with tempfile.TemporaryDirectory() as tmp:
        fin = pathlib.Path(tmp) / "in.c64"
        fout = pathlib.Path(tmp) / "out.c64"
        x.tofile(fin)
        cmd = [str(binary), "--dims", *map(str, dims), "--batch", str(batch),
               "--iters", str(iters), "--coalesced", str(coalesced),
               "--in", str(fin), "--out", str(fout)]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(res.stdout, end="")
        if res.returncode != 0:
            raise RuntimeError(f"vkfft_bench failed:\n{res.stderr}")
        fwd = np.fromfile(fout, dtype=np.complex64).reshape(x.shape)
    info = {}
    m = re.search(r"plan creation ([\d.]+) ms", res.stdout)
    info["plan_ms"] = float(m.group(1))
    m = re.search(r"round-trip rel. error ([\d.e+-]+)", res.stdout)
    info["roundtrip_err"] = float(m.group(1))
    m = re.search(r"forward ([\d.]+)\s+inverse ([\d.]+)\s+fwd\+inv ([\d.]+)",
                  res.stdout)
    info["fwd_ms"], info["inv_ms"], info["fwdinv_ms"] = map(float, m.groups())
    return fwd, info


def run_mlx(x: np.ndarray, rank: int, iters: int) -> tuple[np.ndarray, dict]:
    """Time ``mx.fft.fftn`` / ``ifftn`` over the last ``rank`` axes of ``x``.

    MLX's inverse is normalised by ``1/N``; the output returned here is the
    forward transform only, which is unnormalised like cuFFT's.

    Args:
        x: Input of row-major shape ``(L, [Mz,] My, Mx)``, complex64.
        rank: Number of trailing spatial axes to transform (2 or 3).
        iters: Timed repetitions.

    Returns:
        ``(forward, info)`` with the forward transform and median timings in ms.
    """
    import mlx.core as mx  # noqa: PLC0415

    axes = tuple(range(-rank, 0))
    a = mx.array(x)
    mx.eval(a)
    fwd = mx.fft.fftn(a, axes=axes)
    mx.eval(fwd)
    out = np.array(fwd)
    rt = mx.fft.ifftn(fwd, axes=axes)
    mx.eval(rt)
    info = {"roundtrip_err": rel_err(np.array(rt), x.astype(np.complex128))}
    del fwd, rt

    def timed(fn):
        ts = []
        for _ in range(3):
            mx.eval(fn())
        for _ in range(iters):
            t0 = time.perf_counter()
            mx.eval(fn())
            ts.append((time.perf_counter() - t0) * 1e3)
        return float(np.median(ts))

    info["fwd_ms"] = timed(lambda: mx.fft.fftn(a, axes=axes))
    info["inv_ms"] = timed(lambda: mx.fft.ifftn(a, axes=axes))
    info["fwdinv_ms"] = timed(
        lambda: mx.fft.ifftn(mx.fft.fftn(a, axes=axes), axes=axes))
    return out, info


def main() -> None:
    """Parse arguments, run every requested case and write the JSON report."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cases", nargs="+", default=list(CASES),
                    choices=list(CASES))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--coalesced", type=int, default=0,
                    help="VkFFT coalescedMemory in bytes (0 = VkFFT default)")
    ap.add_argument("--binary", type=pathlib.Path, default=HERE / "vkfft_bench")
    ap.add_argument("--no-mlx", action="store_true")
    args = ap.parse_args()

    report = {"host": socket.gethostname(), "coalesced": args.coalesced,
              "cases": {}}
    for name in args.cases:
        dims, batch = CASES[name]
        rank = len(dims)
        shape = (batch, *reversed(dims))
        print(f"== {name}: dims {dims}, batch {batch}, row-major {shape}")
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(shape, dtype=np.float32)
             + 1j * rng.standard_normal(shape, dtype=np.float32)
             ).astype(np.complex64)

        ref = scipy.fft.fftn(x.astype(np.complex128),
                             axes=tuple(range(-rank, 0)), workers=-1)
        entry = {"dims": dims, "batch": batch}

        fwd, info = run_vkfft(args.binary, dims, batch, x, args.iters,
                              args.coalesced)
        info["fwd_err"] = rel_err(fwd, ref)
        entry["vkfft"] = info
        del fwd
        print(f"   vkfft forward rel. error vs SciPy fp64: {info['fwd_err']:.3e}")

        if not args.no_mlx:
            fwd, info = run_mlx(x, rank, args.iters)
            info["fwd_err"] = rel_err(fwd, ref)
            entry["mlx"] = info
            del fwd
            print(f"   mlx   forward rel. error {info['fwd_err']:.3e}, "
                  f"round trip {info['roundtrip_err']:.3e}, median ms: "
                  f"fwd {info['fwd_ms']:.3f} inv {info['inv_ms']:.3f} "
                  f"fwd+inv {info['fwdinv_ms']:.3f}")
        report["cases"][name] = entry
        del ref, x

    out = HERE / f"results_{report['host'].split('.')[0]}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
