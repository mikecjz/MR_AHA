"""Generate CUDA reference data for validating ports of MR_AHA (e.g. Metal/MLX).

Every case runs the CUDA library through the Python binding and stores the
inputs and outputs of each public entry point in one ``.npz`` file under
``data/ref_mlx/``, so that another backend can be checked against them without
a CUDA GPU:

* :func:`mr_aha.calc_mask`,
* :func:`mr_aha.calc_toep` for ``'lower'`` and ``'upper'`` triangulation,
  unmasked and with that mask,
* :func:`mr_aha.aha_toeplitz` for every one of those ``phi2_toep`` variants,
* the NUFFT reference :class:`mr_aha.aha_nufft_plan` (cufinufft) when the
  ``cufinufft`` package is importable, as an independent check of the
  Toeplitz results.

The synthetic cases are small (a few MB each) and cover all three trajectory
types, odd and even image sizes, odd upsampled sizes (prime / non-2-3-5 FFT
lengths), odd ``Nz`` for a stack, and ``L`` in {1, 3, 4}. One extra case runs
the real 2D radial benchmark data (``data/human_C-35_2D_radial.mat``).

Array layout
------------
All arrays are saved in the **row-major (NumPy) order used by the Python
binding**, i.e. the column-major shapes of the documentation reversed. The
memory layout is therefore identical to the column-major layout of the C API:

====================  ===============================  =========================
key                   row-major shape (saved)          column-major (C API docs)
====================  ===============================  =========================
``traj``              ``(ndim, Nkx, Ntrajs)`` float32  ``[Ntrajs, Nkx, ndim]``
``phi2`` (2D/3D)      ``(L, L, Ntrajs)``               ``[Ntrajs, L, L]``
``phi2`` (stack)      ``(L, L, Nkz, Ntrajs)``          ``[Ntrajs, Nkz, L, L]``
``x``, ``AhAx_*``     ``(L, [Nz,] Ny, Nx)``            ``[Nx, Ny, [Nz,] L]``
``SEs``               ``(Nc, [Nz,] Ny, Nx)``           ``[Nx, Ny, [Nz,] Nc]``
``phi2_toep_<tri>``   ``(T, [Mz,] My, Mx)``            ``[Mx, My, [Mz,] T]``
``phi2_toep_<tri>_neighbour`` ``(T, num_masked)``      ``[num_masked, T]``
``mask_neighbour``    ``(num_masked,)`` int64          0-based linear indices
====================  ===============================  =========================

``T = L (L + 1) / 2``. ``phi2[j, i, ...]`` holds entry ``(i, j)`` of the
Hermitian Gram block ``Phi Phi^H`` (rank x time ``Phi``, MR-Multitasking
convention, recorded as ``gram_convention`` in ``config_json``). Mask indices are 0-based linear indices into the
centred upsampled grid ``[Mx, My, Mz]`` in column-major order, which equals
the flat index of the row-major ``(Mz, My, Mx)`` array. For a stack the mask
repeats the in-plane selection over all ``Mz = Nz`` planes. The masked
variants carry the suffix ``neighbour``, after the neighbourhood construction
of :func:`mr_aha.calc_mask`. All k-space arrays have the centre in the middle
of the array (index ``M // 2``).

Scalar metadata is stored both as individual keys (``traj_type``, ``n_dim``,
``phi2_dim``, ``L``, ``n_coils``, ``eps``, ``cfg_mask_radius``;
``n_dim``/``phi2_dim`` in column-major order ``(Nx, Ny, [Nz])``; the ``cfg_``
prefix keeps the scalar apart from the ``mask_*`` index arrays) and as one
JSON string ``config_json`` that also records the full
``MRConfig`` of every call, the layout note, the GPU and the software
versions.

Verification
------------
After generating, every file in ``data/ref_mlx/`` is reloaded and checked by
:func:`verify`. These are *self-consistency* checks of the CUDA output (the
consumers of the files, ``tests/check_reference.py`` and
``tests/check_reference_matlab.m``, compare another backend against them):

* every floating-point array is finite;
* each masked ``phi2_toep`` equals the unmasked kernel gathered at the mask
  (``rel <= 1e-6``; the same transform, only the output gather differs);
* ``AhAx`` from the ``'upper'`` and ``'lower'`` packings agree
  (``rel <= 1e-5``; the same Hermitian block, reconstructed from the other
  triangle);
* Toeplitz agrees with the cufinufft NUFFT reference (``rel <= 1e-4``, the
  NUFFT accuracy) when that reference was computed.

The masked-vs-unmasked ``AhAx`` gap and, for the real case, the gap to the
stored CPU finufft result are only printed (they depend on the mask radius
and on the data, so no fixed tolerance applies).

Requires the CUDA backend, CuPy and, for the NUFFT reference, the
``cufinufft`` Python package; the real case additionally needs SciPy.

Usage::

    CUDA_VISIBLE_DEVICES=1 python tests/make_reference.py          # generate + verify
    python tests/make_reference.py --verify-only                   # re-check existing files
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Repository root; the in-tree Python package is put first on sys.path so the
# library built in this checkout is the one exercised.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

import cupy as cp  # noqa: E402
import mr_aha  # noqa: E402

# Output directory of the reference files (one .npz per case).
OUT_DIR = ROOT / "data" / "ref_mlx"

#: Gram-block convention recorded in every file's ``config_json``
#: (``gram_convention = "multitasking"``); ``tests/check_reference.py`` refuses
#: files without it.
GRAM_NOTE = (
    "phi2 blocks are Phi*Phi' with a rank x time Phi (MR-Multitasking) and every "
    "block (phi2 and phi2_toep) is applied to the rank row vector from the right, "
    "v <- v*B."
)

#: Layout note copied into every file's ``config_json``.
LAYOUT_NOTE = (
    "Arrays are row-major NumPy order as used by the mr_aha Python binding "
    "(column-major shapes of the documentation reversed; identical memory "
    "layout). Mask indices are 0-based column-major linear indices into the "
    "centred upsampled grid [Mx, My, Mz]. k-space centre at index M//2."
)

# Each case: image size, rank L, coil count, spoke count / samples per spoke
# and the RNG seed. Together they cover every trajectory family, L = 1 (scalar
# blocks), odd image sizes (exact centring for odd N), an explicit odd
# upsampled grid, and odd Nz for a stack (Mz = Nz must stay odd).
#: The synthetic cases. ``phi2_dim=None`` means the library default
#: (``2 * n_dim``, ``Mz = Nz`` for a stack).
CASES = [
    dict(name="2d_50x40_L3_c3", traj_type="2D", n_dim=(50, 40), L=3, n_coils=3,
         n_traj=61, n_kx=100, seed=1),
    dict(name="2d_33x27_L1_c2_odd", traj_type="2D", n_dim=(33, 27), L=1, n_coils=2,
         n_traj=45, n_kx=66, seed=2),
    # Odd upsampled sizes: 101 is prime, 81 = 3^4 (non-2-3-5 / Bluestein paths).
    dict(name="2d_50x40_L4_c2_M101x81", traj_type="2D", n_dim=(50, 40), L=4,
         n_coils=2, n_traj=61, n_kx=100, seed=3, phi2_dim=(101, 81)),
    dict(name="3d_24x20x16_L4_c2", traj_type="3D", n_dim=(24, 20, 16), L=4,
         n_coils=2, n_traj=400, n_kx=48, seed=4),
    dict(name="3d_25x21x15_L3_c2_odd", traj_type="3D", n_dim=(25, 21, 15), L=3,
         n_coils=2, n_traj=350, n_kx=50, seed=5),
    dict(name="stack_40x36x10_L3_c3", traj_type="stack", n_dim=(40, 36, 10), L=3,
         n_coils=3, n_traj=40, n_kx=80, seed=6),
    dict(name="stack_31x36x9_L4_c2_odd", traj_type="stack", n_dim=(31, 36, 9), L=4,
         n_coils=2, n_traj=35, n_kx=62, seed=7),
]

# Both packings of the Hermitian L x L blocks are generated for every case.
TRIS = ("lower", "upper")
#: Suffix of the masked variants (the neighbourhood mask of
#: :func:`mr_aha.calc_mask`).
MASKS = ("neighbour",)
GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0  # golden-angle fraction


# --------------------------------------------------------------------------
# Synthetic inputs
# --------------------------------------------------------------------------

def radial_readout(n_kx: int) -> np.ndarray:
    """Radial readout positions ``pi * (2 s / n_kx - 1)``, in ``[-pi, pi)``.

    Args:
        n_kx: Number of samples per spoke.

    Returns:
        ``float64`` array of length ``n_kx``.
    """
    return np.pi * (2.0 * np.arange(n_kx) / n_kx - 1.0)


def make_traj(traj_type: str, n_traj: int, n_kx: int) -> np.ndarray:
    """Golden-angle radial (2D, stack) or golden-means kooshball (3D) spokes.

    Args:
        traj_type: ``'2D'``, ``'3D'`` or ``'stack'``.
        n_traj: Number of spokes ``Ntrajs``.
        n_kx: Samples per spoke ``Nkx``.

    Returns:
        ``float32`` array ``(ndim, Nkx, Ntrajs)`` in radians per image-grid
        unit, within ``[-pi, pi)``.
    """
    # Readout positions as a column and spoke indices as a row, so the
    # products below broadcast to (Nkx, Ntrajs).
    r = radial_readout(n_kx)[:, None]                      # (Nkx, 1)
    t = np.arange(n_traj)[None, :]                          # (1, Ntrajs)
    # 2D / stack: in-plane spokes at successive golden-angle increments of pi
    # (the kz axis of a stack is Cartesian and not part of the trajectory).
    if traj_type in ("2D", "stack"):
        th = np.pi * GOLDEN * t
        traj = np.stack([r * np.cos(th), r * np.sin(th)])
    else:
        # Golden-means directions on the half sphere (as make_kooshball.py).
        phi1, phi2 = 0.4656, 0.6823
        z = np.mod(t * phi1, 1.0)
        az = 2.0 * np.pi * np.mod(t * phi2, 1.0)
        s = np.sqrt(1.0 - z ** 2)
        traj = np.stack([r * s * np.cos(az), r * s * np.sin(az), r * z])
    # traj is (ndim, Nkx, Ntrajs) row-major, the C API's [Ntrajs, Nkx, ndim].
    return np.ascontiguousarray(traj, dtype=np.float32)


def make_phi2(rng, L: int, n_traj: int, n_kz: int | None) -> np.ndarray:
    """Random Hermitian positive semi-definite Gram blocks ``Phi Phi^H``.

    ``Phi = A^T`` is a random rank x time basis (MR-Multitasking convention,
    see :mod:`mr_aha.api`), so ``B = A^T conj(A)``.

    Args:
        rng: NumPy random generator.
        L: Number of ranks.
        n_traj: Number of trajectories.
        n_kz: Number of kz planes for a stack, else ``None``.

    Returns:
        ``complex64`` array ``(L, L, Ntrajs)`` or ``(L, L, Nkz, Ntrajs)`` with
        ``phi2[j, i, ...] = B[..., i, j]``.
    """
    # One block per readout (and per kz plane for a stack): A has shape
    # (lead..., 2, L), i.e. a time x rank matrix with two time points, so
    # Phi = A^T is rank x time.
    lead = (n_traj,) if n_kz is None else (n_traj, n_kz)
    A = (rng.standard_normal(lead + (2, L)) + 1j * rng.standard_normal(lead + (2, L)))
    # B[..., i, j] = sum_k A[k, i] conj(A[k, j]) = (Phi Phi^H)_{ij}, divided by
    # the number of time points.
    B = np.einsum("...ki,...kj->...ij", A, A.conj()) / 2.0   # (..., L, L) = Phi Phi^H
    # Reverse all axes so that the row-major array equals the column-major
    # [Ntrajs, (Nkz,) L, L] the library expects: phi2[j, i, (z,) t] = B[t, (z,) i, j].
    phi2 = B.transpose(tuple(range(B.ndim))[::-1])         # [j, i, (z,) t]
    return np.ascontiguousarray(phi2, dtype=np.complex64)


def crandn(rng, shape) -> np.ndarray:
    """Standard complex Gaussian ``complex64`` array of the given shape."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)


# --------------------------------------------------------------------------
# Running the CUDA library
# --------------------------------------------------------------------------

def cfg_dict(cfg) -> dict:
    """Plain-Python snapshot of an :class:`mr_aha.MRConfig` (no stream)."""
    out = {}
    # Walk the ctypes fields of the configuration struct; the stream pointer
    # is skipped (not meaningful, not JSON-serialisable) and fixed-size arrays
    # (n_dim, phi2_dim) become lists.
    for name, _ in cfg._fields_:
        if name == "stream":
            continue
        v = getattr(cfg, name)
        out[name] = list(v) if hasattr(v, "__len__") else v
    return out


def run_case(traj_type, n_dim, L, n_coils, traj, phi2, x, SEs, phi2_dim=None,
             with_nufft=True) -> tuple[dict, dict]:
    """Run every entry point of the CUDA library on one problem.

    Args:
        traj_type: ``'2D'``, ``'3D'`` or ``'stack'``.
        n_dim: Image size ``(Nx, Ny[, Nz])`` (column-major order).
        L: Number of ranks.
        n_coils: Number of coils.
        traj, phi2, x, SEs: NumPy inputs in the row-major layout of the
            module docstring.
        phi2_dim: Optional upsampled size override ``(Mx, My[, Mz])``.
        with_nufft: Also evaluate the cufinufft NUFFT reference operator.

    Returns:
        ``(arrays, meta)``: the outputs as NumPy arrays keyed as in the module
        docstring, and a dict of per-call configurations for ``config_json``.
    """
    # Upload the NumPy inputs to the GPU (the CUDA binding takes CuPy arrays).
    g_traj, g_phi2 = cp.asarray(traj), cp.asarray(phi2)
    g_x, g_SEs = cp.asarray(x), cp.asarray(SEs)
    # Keywords shared by every configuration built below.
    kw = dict(n_coils=n_coils)
    if phi2_dim is not None:
        kw["phi2_dim"] = phi2_dim
    out, meta = {}, {"calls": {}}

    # The unmasked default configuration ('lower' triangulation) is used for
    # the mask and the NUFFT reference. The neighbourhood mask is computed
    # with the configuration's default radius and saved as mask_neighbour.
    base = mr_aha.make_config(traj_type, n_dim, L, **kw)
    meta["base_config"] = cfg_dict(base)
    masks = {
        "neighbour": mr_aha.calc_mask(base, g_traj, radius=int(base.mask_radius)),
    }
    for m, v in masks.items():
        out[f"mask_{m}"] = v.get()

    # For each packing: the unmasked kernel and its A^H A x, then for each
    # mask the masked kernel and its A^H A x. Every configuration used is
    # recorded under meta["calls"]. Kernels are released as soon as they have
    # been copied to the host to keep GPU memory low.
    for tri in TRIS:
        cfg = mr_aha.make_config(traj_type, n_dim, L, triangulation=tri, **kw)
        Om = mr_aha.calc_toep(cfg, g_phi2, g_traj)
        out[f"phi2_toep_{tri}"] = Om.get()
        out[f"AhAx_{tri}"] = mr_aha.aha_toeplitz(cfg, g_x, g_SEs, Om).get()
        meta["calls"][tri] = cfg_dict(cfg)
        del Om
        for m in MASKS:
            cfgm = mr_aha.make_config(traj_type, n_dim, L, triangulation=tri,
                                      is_masked=True, **kw)
            Om = mr_aha.calc_toep(cfgm, g_phi2, g_traj, mask=masks[m])  # sets num_masked
            out[f"phi2_toep_{tri}_{m}"] = Om.get()
            out[f"AhAx_{tri}_{m}"] = mr_aha.aha_toeplitz(
                cfgm, g_x, g_SEs, Om, mask=masks[m]).get()
            meta["calls"][f"{tri}_{m}"] = cfg_dict(cfgm)
            del Om

    # Independent check: the pure Python NUFFT operator (CuPy + cufinufft),
    # which shares no code with the Toeplitz path. It is optional; when it
    # cannot run, the reason is recorded in meta["nufft"] instead.
    if with_nufft:
        try:
            with mr_aha.aha_nufft_plan(base, g_traj) as plan:
                out["AhAx_nufft"] = plan.apply(g_x, g_SEs, g_phi2).get()
            meta["nufft"] = "aha_nufft_plan (cufinufft), eps = cfg.eps, upsampfac 2"
        except Exception as e:  # cufinufft missing or libcudart not found
            meta["nufft"] = f"skipped: {type(e).__name__}: {e}"
    # Release CuPy's cached blocks before the next case.
    cp.get_default_memory_pool().free_all_blocks()
    return out, meta


def environment() -> dict:
    """GPU name and software versions recorded in ``config_json``."""
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    env = dict(gpu=props["name"].decode(), cupy=cp.__version__,
               cuda_runtime=cp.cuda.runtime.runtimeGetVersion())
    try:
        import cufinufft
        env["cufinufft"] = cufinufft.__version__
    except Exception:
        pass
    return env


def save_case(name: str, inputs: dict, outputs: dict, meta: dict) -> Path:
    """Write one case to ``data/ref_mlx/<name>.npz`` (uncompressed).

    Args:
        name: Case name (file stem).
        inputs: Input arrays and scalar metadata.
        outputs: Output arrays from :func:`run_case`.
        meta: JSON-serialisable metadata, stored as ``config_json``.

    Returns:
        Path of the written file.
    """
    # The metadata gains the case name, the layout and Gram-convention notes
    # (check_reference.py rejects files without gram_convention) and the
    # environment; it is stored as a 0-d string array next to the arrays.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.npz"
    meta = dict(meta, case=name, layout=LAYOUT_NOTE, env=environment(),
                gram_convention="multitasking", gram_convention_note=GRAM_NOTE)
    np.savez(path, config_json=np.array(json.dumps(meta)), **inputs, **outputs)
    return path


def scalar_inputs(traj_type, n_dim, L, n_coils, phi2_dim, base_cfg) -> dict:
    """Scalar metadata keys stored next to the arrays."""
    # phi2_dim is taken from the configuration the library actually used
    # (resolving the default 2*N / Mz = Nz), truncated to the spatial
    # dimensions; eps and mask_radius are the configuration defaults.
    nsp = len(n_dim)
    return dict(
        traj_type=np.array(traj_type),
        n_dim=np.array(n_dim, dtype=np.int64),
        phi2_dim=np.array(base_cfg["phi2_dim"][:nsp], dtype=np.int64),
        L=np.int64(L), n_coils=np.int64(n_coils),
        eps=np.float64(base_cfg["eps"]),
        cfg_mask_radius=np.int64(base_cfg["mask_radius"]),
    )


def generate_synthetic(case: dict) -> Path:
    """Build, run and save one synthetic case from :data:`CASES`."""
    # Per-case seed, so each file can be regenerated independently. The
    # trajectory is deterministic; Gram blocks, x and the maps are random.
    rng = np.random.default_rng(case["seed"])
    tt, n_dim, L, nc = case["traj_type"], case["n_dim"], case["L"], case["n_coils"]
    img = tuple(reversed(n_dim))                           # ([Nz,] Ny, Nx)
    traj = make_traj(tt, case["n_traj"], case["n_kx"])
    phi2 = make_phi2(rng, L, case["n_traj"], n_dim[2] if tt == "stack" else None)
    x = crandn(rng, (L,) + img)
    SEs = crandn(rng, (nc,) + img)
    # Run the library, then store the case definition, the inputs and the
    # scalar metadata together with the outputs.
    outputs, meta = run_case(tt, n_dim, L, nc, traj, phi2, x, SEs,
                             phi2_dim=case.get("phi2_dim"))
    meta["synthetic"] = {k: v for k, v in case.items()}
    inputs = dict(traj=traj, phi2=phi2, x=x, SEs=SEs,
                  **scalar_inputs(tt, n_dim, L, nc, case.get("phi2_dim"),
                                  meta["base_config"]))
    return save_case(case["name"], inputs, outputs, meta)


def generate_real_2d() -> Path | None:
    """Run the real 2D radial benchmark (160^2, 16 coils, L = 6, 377 spokes).

    Uses the stored ``x`` and the stored CPU-finufft ``AhAx_ref`` (saved as
    ``AhAx_finufft_cpu``) from ``data/human_C-35_2D_radial.mat``.

    Returns:
        Path of the written file, or ``None`` if the data file is missing.
    """
    # The real data set is optional (not part of the repository checkout).
    f = ROOT / "data" / "human_C-35_2D_radial.mat"
    if not f.exists():
        print(f"skip real 2D: {f} missing")
        return None
    import scipy.io as sio

    # loadmat returns MATLAB's column-major arrays as Fortran-ordered NumPy
    # arrays; .T reverses the axes, giving the row-major layout of the binding
    # with the same memory order.
    d = sio.loadmat(f)
    L, Nt = 6, 377
    traj = np.ascontiguousarray(d["traj"].T, dtype=np.float32)
    # The file's phi2 predates the MR-Multitasking convention: swap the L axes.
    phi2 = np.ascontiguousarray(np.swapaxes(d["phi2"].T.reshape(L, L, Nt), 0, 1),
                                dtype=np.complex64)
    x = np.ascontiguousarray(d["x"].T, dtype=np.complex64)
    SEs = np.ascontiguousarray(d["SEs"].T, dtype=np.complex64)
    ref = np.ascontiguousarray(d["AhAx_ref"].T, dtype=np.complex64)
    # Run every entry point on the real data; the coil count comes from the
    # maps. The stored CPU finufft result is kept as AhAx_finufft_cpu.
    n_dim = (160, 160)
    outputs, meta = run_case("2D", n_dim, L, SEs.shape[0], traj, phi2, x, SEs)
    meta["source"] = "data/human_C-35_2D_radial.mat (AhAx_finufft_cpu = stored AhAx_ref)"
    inputs = dict(traj=traj, phi2=phi2, x=x, SEs=SEs, AhAx_finufft_cpu=ref,
                  **scalar_inputs("2D", n_dim, L, SEs.shape[0], None, meta["base_config"]))
    return save_case("2d_real_160x160_L6_c16", inputs, outputs, meta)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------

def rel(a, b) -> float:
    """Relative 2-norm difference ``||a - b|| / ||b||``."""
    return float(np.linalg.norm((a - b).ravel()) / np.linalg.norm(b.ravel()))


def verify(path: Path) -> list[str]:
    """Reload one file and check it is complete and self-consistent.

    Checks: every array is finite; each masked ``phi2_toep`` equals the
    unmasked one gathered at the mask; ``'lower'`` and ``'upper'`` give the
    same ``AhAx``; and reports the masked-vs-unmasked and Toeplitz-vs-NUFFT
    differences.

    Args:
        path: ``.npz`` file written by :func:`save_case`.

    Returns:
        List of failure messages (empty if all hard checks pass). A one-line
        summary of the soft metrics is printed.
    """
    # Hard check 1: no NaN or Inf in any floating-point array.
    z = np.load(path)
    meta = json.loads(str(z["config_json"]))
    fails = []
    for k in z.files:
        a = z[k]
        if a.dtype.kind in "fc" and not np.isfinite(a).all():
            fails.append(f"{k} has non-finite values")
    # Hard check 2: masking only gathers the kernel at the mask locations, so
    # the masked kernel must equal the unmasked one indexed by the mask (a
    # 0-based column-major index is the flat index of the row-major grid).
    for tri in TRIS:
        full = z[f"phi2_toep_{tri}"]
        flat = full.reshape(full.shape[0], -1)
        for m in MASKS:
            mk = z[f"mask_{m}"]
            d = rel(z[f"phi2_toep_{tri}_{m}"], flat[:, mk])
            if d > 1e-6:
                fails.append(f"phi2_toep_{tri}_{m} != gathered unmasked (rel {d:.2e})")
    # Hard check 3: both packings describe the same Hermitian block, so the
    # outputs agree up to single-precision rounding.
    d_tri = rel(z["AhAx_upper"], z["AhAx_lower"])
    if d_tri > 1e-5:
        fails.append(f"AhAx upper vs lower differ (rel {d_tri:.2e})")
    # Soft metrics (printed only): fraction of the upsampled grid the mask
    # keeps and the masked-vs-unmasked A^H A x gap.
    pad = int(np.prod(z["phi2_dim"]))
    parts = [f"upper/lower {d_tri:.1e}"]
    for m in MASKS:
        parts.append(f"{m}: keep {z[f'mask_{m}'].size / pad:.0%}, "
                     f"AhA vs unmasked {rel(z[f'AhAx_lower_{m}'], z['AhAx_lower']):.1e}")
    # Hard check 4 (when present): Toeplitz vs the cufinufft NUFFT operator,
    # within the NUFFT accuracy. The CPU finufft comparison of the real case is
    # informational.
    if "AhAx_nufft" in z.files:
        d_nu = rel(z["AhAx_lower"], z["AhAx_nufft"])
        parts.append(f"toep vs nufft {d_nu:.1e}")
        if d_nu > 1e-4:
            fails.append(f"Toeplitz vs NUFFT differ (rel {d_nu:.2e})")
    else:
        parts.append(f"nufft {meta.get('nufft', 'absent')}")
    if "AhAx_finufft_cpu" in z.files:
        parts.append(f"toep vs CPU finufft {rel(z['AhAx_lower'], z['AhAx_finufft_cpu']):.1e}")
    print(f"  {path.name}: " + "; ".join(parts))
    return fails


def main() -> int:
    """Command-line entry point; returns the process exit code."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--verify-only", action="store_true",
                    help="only re-check the files already in data/ref_mlx")
    args = ap.parse_args()

    # Generate every synthetic case, then the real case (skipped when its
    # data file is missing). --verify-only re-checks existing files only.
    if not args.verify_only:
        for case in CASES:
            print(f"generating {case['name']} ...", flush=True)
            generate_synthetic(case)
        print("generating 2d_real_160x160_L6_c16 ...", flush=True)
        generate_real_2d()

    # Verify every file present in the output directory (including files from
    # earlier runs) and report all failures together.
    print("verifying:")
    fails = []
    for p in sorted(OUT_DIR.glob("*.npz")):
        fails += [f"{p.name}: {m}" for m in verify(p)]
    for f in fails:
        print("FAIL", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
