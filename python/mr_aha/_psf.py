"""Two-stage ``calc_toep``: type-1 NUFFT in Python, the rest in the library.

The Metal backend has no C/C++ NUFFT to call, so :func:`mr_aha.calc_toep`
builds ``phi2_toep`` in two stages there:

1. **Point-spread functions (this module).** The Gram blocks are packed into
   the stored triangle (:class:`~mr_aha.Triangulation`), each stored element is
   broadcast along its readout to give the non-uniform strengths, and a type-1
   NUFFT with sign -1, centered (``modeord = 0``) modes ``phi2_dim`` and
   upsampling factor 2 (see :func:`toep_upsampfac`) evaluates the
   point-spread function of each triangular component. This is exactly what the CUDA ``mr_calc_toep`` does with
   cufinufft before its inverse FFT.
2. **The library** (``mr_calc_toep_from_psf``): centered inverse FFT,
   ``1/prod(M)`` scaling and the optional mask gather.

The components are processed in chunks that bound the transient memory. Two
NUFFT engines are available:

``'mlx'``
    `mlx-nufft <https://github.com/martinlachaine/mlx-nufft>`_ on the Apple
    GPU (the ``third_party/mlx-nufft`` submodule). The default on Metal.
``'finufft'``
    CPU FINUFFT, for validation (e.g. of this module on the CUDA backend
    against the one-shot ``mr_calc_toep``).

Array layout follows :mod:`mr_aha.api`: ``phi2`` is row-major
``(L, L, Ntrajs)`` or ``(L, L, Nkz, Ntrajs)`` with ``phi2[j, i, ...]`` the
block entry ``(i, j)``; ``traj`` is ``(ndim, Nkx, Ntrajs)``.

Data flow of one chunk of components ``[t0, t0 + n)`` in
:func:`calc_toep_two_stage`::

    phi2 (host)  --_strengths-->  cs (n*Nkz, Ntrajs*Nkx) complex64, host
    cs           --_Type1------>  psf (n*Nkz, *modes_rev), engine output
    psf          --xp.asarray-->  flat complex64 backend array
    psf          --mr_calc_toep_from_psf-->  out[t0 : t0+n] (in place)

Because the two inputs are first copied to the host (``.get()`` on CuPy),
the NUFFT engines always see NumPy arrays, whatever the backend.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

from ._lib import MRConfig, TrajType, Triangulation, check, lib

__all__ = ["ENGINES", "tri_pairs", "calc_toep_two_stage", "toep_upsampfac"]

#: NUFFT engines accepted by :func:`calc_toep_two_stage`.
ENGINES = ("mlx", "finufft")

#: Experimental environment variable that overrides the NUFFT upsampling
#: factor of the kernel computation; the same name is read by the CUDA
#: ``mr_calc_toep``.
UPSAMPFAC_ENV = "MR_AHA_TOEP_UPSAMPFAC"


def toep_upsampfac() -> float:
    """Upsampling factor sigma of the type-1 NUFFT that computes the kernel.

    **Experimental.** The default is 2.0, the standard (FINUFFT / cufinufft)
    setting. The environment variable ``MR_AHA_TOEP_UPSAMPFAC``, read at every
    kernel computation, overrides it; 1.25 is the other value worth trying.
    sigma only sizes the NUFFT's internal fine grid and its spreading kernel:
    the Toeplitz grid ``phi2_dim`` is unchanged. At the same ``eps``,
    sigma = 1.25 needs a wider kernel, so it pays off only at a looser
    ``eps`` (see https://mikecjz.github.io/MR_AHA_docs/benchmarks/upsampfac.html).
    mlx-nufft caps that kernel at
    width 8 in 3D (fp32), which bounds the achievable error to about 5e-5.

    Returns:
        sigma, a float > 1.

    Raises:
        ValueError: The environment variable is set but is not a number > 1.
    """
    # Unset or blank variable: the standard sigma = 2.
    env = os.environ.get(UPSAMPFAC_ENV, "").strip()
    if not env:
        return 2.0
    # Parse the override; a non-numeric value is mapped to 0.0 so that it
    # falls into the same "must be > 1" error below (``not sigma > 1.0`` also
    # rejects NaN).
    try:
        sigma = float(env)
    except ValueError:
        sigma = 0.0
    if not sigma > 1.0:
        raise ValueError(f"{UPSAMPFAC_ENV} must be a number > 1 (e.g. 2.0 or "
                         f"1.25), got {env!r}")
    return sigma


def tri_pairs(L: int, triangulation: int) -> list:
    """Block entries ``(i, j)`` of the stored triangle, in packed order.

    The packing is the column-major order of MATLAB's ``A(tril(true(L)))`` /
    ``A(triu(true(L)))``, as documented for ``mr_triangulation``.

    Example for ``L = 3``: lower gives ``(0,0) (1,0) (2,0) (1,1) (2,1) (2,2)``;
    upper gives ``(0,0) (0,1) (1,1) (0,2) (1,2) (2,2)``.

    Args:
        L: Rank.
        triangulation: :class:`~mr_aha.Triangulation` value.

    Returns:
        List of ``(L^2 + L) / 2`` pairs; entry ``t`` is the element stored at
        packed offset ``t``.
    """
    # Column-major traversal: the column index j is the outer loop, the row
    # index i the inner one. Upper keeps rows 0..j of column j.
    if triangulation == Triangulation.UPPER:
        return [(i, j) for j in range(L) for i in range(j + 1)]
    # Lower (the default for any other value) keeps rows j..L-1 of column j.
    return [(i, j) for j in range(L) for i in range(j, L)]


def _strengths(phi2: np.ndarray, pairs: list, comps: range, Nkx: int,
               is_stack: bool) -> np.ndarray:
    """Non-uniform strengths of a range of triangular components.

    One Gram block is shared by every sample of a readout, so the strength
    of component ``t`` at sample ``(kx, traj)`` is simply block entry
    ``pairs[t]`` of that readout's block, repeated ``Nkx`` times.

    Args:
        phi2: Host Gram blocks, ``(L, L, Ntrajs)`` or ``(L, L, Nkz, Ntrajs)``.
        pairs: Output of :func:`tri_pairs`.
        comps: Components to build.
        Nkx: Samples per readout.
        is_stack: Whether ``phi2`` carries a kz axis.

    Returns:
        ``complex64`` array ``(len(comps) * Nkz, Ntrajs * Nkx)``: one row per
        transform, component-major then kz, each row in the sample order of
        ``traj[d].ravel()`` (readout sample slowest, trajectory fastest).
    """
    rows = []
    for t in comps:
        # Row-major phi2[j, i, ...] holds column-major block entry (i, j).
        i, j = pairs[t]
        v = phi2[j, i]                                  # (Ntrajs,) or (Nkz, Ntrajs)
        # Give 2D/3D data a singleton kz axis so both cases share one path.
        v = v if is_stack else v[None, :]               # (Nkz, Ntrajs)
        # Repeat each readout's value over its Nkx samples (a zero-copy
        # broadcast view), then flatten to (Nkz, Nkx*Ntrajs) in the
        # traj[d].ravel() order; reshape materializes the copy here.
        rows.append(np.broadcast_to(v[:, None, :], (v.shape[0], Nkx, v.shape[1]))
                    .reshape(v.shape[0], -1))
    # Stack all components (component-major, kz within) into one contiguous
    # complex64 batch, the layout the NUFFT engines expect.
    return np.ascontiguousarray(np.concatenate(rows, axis=0), dtype=np.complex64)


class _Type1:
    """Type-1 NUFFT over a fixed point set, producing ``(B, *modes_rev)``.

    The coordinates and mode counts are passed in reversed axis order so that
    the C-ordered output has ``x`` fastest, i.e. it is the column-major
    ``[Mx, My(, Mz)]`` grid of the C API.

    The transform has sign -1 and centered mode order, matching what
    ``mr_calc_toep`` computes with cufinufft and what
    ``mr_calc_toep_from_psf`` expects as input.

    Args:
        engine: ``'mlx'`` (mlx-nufft plan, built once here) or ``'finufft'``
            (CPU FINUFFT, a stateless call per batch).
        traj: Host trajectory ``(ndim, Nkx, Ntrajs)`` in radians.
        modes_rev: Mode counts in reversed order, ``(My, Mx)`` or
            ``(Mz, My, Mx)``.
        eps: Requested tolerance (``cfg.eps``).

    Raises:
        ImportError: ``engine='mlx'`` and mlx-nufft is not installed (or
            ``engine='finufft'`` and finufft is not installed).
        ValueError: Unknown ``engine``.
    """

    def __init__(self, engine: str, traj: np.ndarray, modes_rev: tuple,
                 eps: float):
        """Prepare the point set and, for mlx-nufft, the plan. See the class
        docstring for the arguments."""
        # Non-uniform coordinates, one float64 vector per axis in reversed
        # order (y, x) / (z, y, x), flattened in traj[d].ravel() order to
        # match the strengths of _strengths.
        coords = tuple(np.ascontiguousarray(traj[d].ravel(), dtype=np.float64)
                       for d in reversed(range(traj.shape[0])))
        self.engine = engine
        self.modes = modes_rev
        #: NUFFT upsampling factor, see :func:`toep_upsampfac`.
        self.upsampfac = toep_upsampfac()
        if engine == "mlx":
            # mlx-nufft: build the plan (point sorting, kernel set-up) once
            # and reuse it for every chunk. The tolerance is floored at 1e-6,
            # the limit of its fp32 arithmetic.
            try:
                from mlx_nufft import Type1PlanND  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError(
                    "calc_toep on the Metal backend needs mlx-nufft; install "
                    "it from the submodule: `pip install -e third_party/mlx-nufft`"
                ) from exc
            self._plan = Type1PlanND(coords, modes_rev, eps=max(eps, 1e-6),
                                     isign=-1, upsampfac=self.upsampfac)
        elif engine == "finufft":
            # CPU FINUFFT: pick the 2D or 3D simple interface by the number of
            # mode axes; it has no persistent plan here, so keep the inputs.
            import finufft  # noqa: PLC0415
            self._fn = {2: finufft.nufft2d1, 3: finufft.nufft3d1}[len(modes_rev)]
            self._coords = coords
            self._eps = eps
        else:
            raise ValueError(f"unknown NUFFT engine {engine!r}; use one of {ENGINES}")

    def __call__(self, cs: np.ndarray) -> np.ndarray:
        """Transform the strength rows ``cs`` (B, P); returns ``(B, *modes)``.

        Args:
            cs: ``complex64`` strengths, one row of ``P = Ntrajs * Nkx``
                values per transform (output of :func:`_strengths`).

        Returns:
            The point-spread functions. For ``'finufft'`` a ``complex64``
            NumPy array; for ``'mlx'`` whatever ``execute_batch`` returns
            (converted by the caller with ``xp.asarray``).
        """
        # mlx-nufft: one batched GPU call over all B rows.
        if self.engine == "mlx":
            return self._plan.execute_batch(cs)
        # FINUFFT runs in double precision (complex128 strengths, float64
        # coordinates); the result is reshaped to (B, *modes) and cast back
        # to the library's complex64.
        out = self._fn(*self._coords, cs.astype(np.complex128), self.modes,
                       eps=self._eps, isign=-1, upsampfac=self.upsampfac)
        return out.reshape(cs.shape[0], *self.modes).astype(np.complex64)


def _chunk(cfg: MRConfig, pad: int, ntri: int, xp) -> int:
    """Triangular components per chunk.

    Each component costs about three upsampled volumes at a time (its
    point-spread function, the library's scratch, the NUFFT's working set);
    the chunk keeps them within ``mem_fraction`` of a quarter of the memory
    (unified memory on Metal, free device memory on CUDA). On Metal a chunk
    also stays below 2^31 complex values, the limit of one VkFFT transform
    (see ``src/metal/mr_metal_fft.cpp``).

    Note: as implemented, the CUDA budget is ``mem_fraction`` of *half* the
    free device memory (``free / 2``), not a quarter; the Metal budget is
    ``mem_fraction`` of a quarter of the physical RAM.

    Args:
        cfg: Configuration (``mem_fraction`` is read; <= 0 means 0.5).
        pad: Upsampled grid size ``Mx*My*Mz`` (one component's volume).
        ntri: Total number of triangular components, ``(L^2+L)/2``.
        xp: :mod:`numpy` (Metal / host) or :mod:`cupy` (CUDA).

    Returns:
        Number of components per chunk, between 1 and ``ntri``.
    """
    # Same default as the library when mem_fraction was left unset.
    frac = cfg.mem_fraction if cfg.mem_fraction > 0 else 0.5
    # Memory budget: physical RAM (unified memory) on the NumPy path, the
    # current CUDA device's free memory on the CuPy path. os.sysconf is
    # POSIX-only, which is fine since this path is used on macOS.
    if xp is np:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        budget = frac * total / 4
    else:
        free, _ = xp.cuda.runtime.memGetInfo()
        budget = frac * free / 2
    # Three complex64 (8-byte) volumes per component; at least one component
    # per chunk, never more than there are.
    n = int(max(1, min(ntri, budget // (3 * pad * 8))))
    # VkFFT limit on Metal: one batch must hold fewer than 2^31 values.
    if xp is np:
        n = max(1, min(n, ((1 << 31) - 1) // pad))
    return n


def calc_toep_two_stage(cfg: MRConfig, phi2, traj, mask, out, engine: str,
                        xp, ptr) -> None:
    """Fill ``out`` with ``phi2_toep`` via external type-1 NUFFTs.

    Called by :func:`mr_aha.calc_toep` after it has validated the sizes and
    completed ``cfg``. Each chunk of components is transformed by the NUFFT
    engine and immediately handed to ``mr_calc_toep_from_psf``, which writes
    only those components of ``out``; so only one chunk of point-spread
    functions is alive at a time.

    Args:
        cfg: Validated configuration (``num_masked`` already set if masked).
        phi2: Gram blocks (backend array), see the module docstring.
        traj: Trajectory (backend array).
        mask: Mask (backend array) or ``None``.
        out: Pre-allocated output (backend array) of ``mr_phi2_toep_size``.
        engine: One of :data:`ENGINES`.
        xp: Array module of the backend (:mod:`numpy` or :mod:`cupy`).
        ptr: ``mr_aha.api._ptr``, to pass arrays to the library.

    Raises:
        ValueError: Unknown ``engine`` or invalid ``MR_AHA_TOEP_UPSAMPFAC``.
        ImportError: The selected NUFFT engine is not installed.
        TypeError: ``mask`` or ``out`` has the wrong array type or dtype
            (raised by ``ptr``).
        MRError: If ``mr_calc_toep_from_psf`` fails.
    """
    # --- Host copies of the inputs ----------------------------------------
    # The NUFFT engines take NumPy arrays: copy device arrays to the host
    # (CuPy .get()), or just view them on the NumPy (Metal) path.
    host = (lambda a: a.get()) if xp is not np else np.asarray
    phi2_h, traj_h = host(phi2), host(traj)

    # --- Geometry derived from the configuration ---------------------------
    is_stack = cfg.traj_type == TrajType.STACK
    is3d = cfg.traj_type == TrajType.THREE_D
    M = [int(cfg.phi2_dim[d]) for d in range(3)]
    # NUFFT modes in reversed (row-major) order. A stack uses a 2D in-plane
    # transform repeated for each of the Mz (== Nkz) kz planes.
    modes_rev = (M[2], M[1], M[0]) if is3d else (M[1], M[0])
    nkz = M[2] if is_stack else 1
    # Values per component (one upsampled volume) and number of components.
    pad = M[0] * M[1] * M[2]
    ntri = int(lib.mr_num_tri(cfg.L))
    Nkx = int(traj_h.shape[1])

    # --- One-time set-up: packing order, NUFFT engine, chunk size ----------
    pairs = tri_pairs(cfg.L, cfg.triangulation)
    nufft = _Type1(engine, traj_h, modes_rev, float(cfg.eps))
    chunk = _chunk(cfg, pad, ntri, xp)

    # --- Chunk loop over the triangular components -------------------------
    for t0 in range(0, ntri, chunk):
        n = min(chunk, ntri - t0)
        # Stage 1a: strengths of components [t0, t0 + n) on the host.
        cs = _strengths(phi2_h, pairs, range(t0, t0 + n), Nkx, is_stack)
        # Stage 1b: their point-spread functions on the upsampled grid.
        psf = nufft(cs)                                 # (n*nkz, *modes_rev)
        # Move the result to the backend's array type (device upload on
        # CUDA), as one flat contiguous complex64 vector: the column-major
        # [Mx, My, Mz, n] layout mr_calc_toep_from_psf expects.
        psf = xp.ascontiguousarray(xp.asarray(psf, dtype=xp.complex64)
                                   .reshape(n * nkz * int(np.prod(modes_rev))))
        # Stage 2: library finishes these components (centered inverse FFT,
        # 1/prod(M) scaling, optional mask gather) directly into `out`.
        check(
            lib.mr_calc_toep_from_psf(
                ctypes.byref(cfg), ptr(psf, "psf", "complex64"),
                ctypes.c_int64(t0), ctypes.c_int64(n),
                ptr(mask, "mask", "int64", allow_none=True),
                ptr(out, "out", "complex64"),
            ),
            "mr_calc_toep_from_psf",
        )
        # Release this chunk's buffers before building the next one, so that
        # peak memory stays at one chunk. This is safe on CUDA because
        # mr_calc_toep_from_psf synchronizes its stream before returning.
        del psf, cs
