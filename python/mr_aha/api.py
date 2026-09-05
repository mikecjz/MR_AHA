"""High-level, CuPy-friendly interface to MR_AHA.

MR_AHA evaluates the forward normal operator :math:`A^H A` for non-Cartesian
MRI with low-rank (subspace / MR-Multitasking) modelling, using a Toeplitz
embedding so that no non-uniform FFT is needed inside the iteration loop.

Array layout
------------
The C library, and the project documentation, describe every array in
**column-major** (MATLAB) order. NumPy and CuPy are row-major, so *the
dimension order is reversed here*, which leaves the underlying memory layout
identical and therefore requires no data movement:

===========================  =======================================
column-major (docs, MATLAB)  row-major (this module, CuPy)
===========================  =======================================
``x[Nx, Ny, Nz, L]``         ``x[L, Nz, Ny, Nx]``
``SEs[Nx, Ny, Nz, Nc]``      ``SEs[Nc, Nz, Ny, Nx]``
``phi2_toep[Mx, My, Mz, T]`` ``phi2_toep[T, Mz, My, Mx]``
``phi2[Ntrajs, L*L]``        ``phi2[L*L, Ntrajs]``
``traj[Ntrajs, Nkx, ndim]``  ``traj[ndim, Nkx, Ntrajs]``
===========================  =======================================

Every array must be a C-contiguous CuPy array resident on the GPU; ``complex64``
for the data arrays, ``float32`` for ``traj`` and ``int64`` for ``mask``.
Masks use **0-based** linear indices into the centred upsampled grid.
"""

from __future__ import annotations

import ctypes
from typing import Optional, Sequence

from ._lib import MRConfig, MRError, TrajType, Triangulation, check, lib

__all__ = [
    "MRConfig",
    "MRError",
    "TrajType",
    "Triangulation",
    "make_config",
    "num_tri",
    "phi2_toep_shape",
    "calc_toep",
    "aha_toeplitz",
    "AhAPlan",
]


def _cupy():
    """Import CuPy lazily so that ``import mr_aha`` works without a GPU.

    Returns:
        The imported :mod:`cupy` module.

    Raises:
        ImportError: If CuPy is not installed.
    """
    try:
        import cupy  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "mr_aha's Python interface needs CuPy for GPU array management; "
            "install the wheel matching your CUDA toolkit, e.g. "
            "`pip install cupy-cuda12x`."
        ) from exc
    return cupy


def _ptr(array, name: str, dtype, allow_none: bool = False):
    """Validate a CuPy array and return its raw device pointer.

    Args:
        array: The CuPy array, or ``None`` when ``allow_none``.
        name: Argument name, used in error messages.
        dtype: Required dtype, as a NumPy dtype specifier.
        allow_none: Whether ``None`` is acceptable.

    Returns:
        The device pointer as an ``int``, or ``None``.

    Raises:
        TypeError: If the array is not a CuPy array of the required dtype.
        ValueError: If the array is not C-contiguous.
    """
    if array is None:
        if allow_none:
            return None
        raise TypeError(f"{name} must not be None")
    cupy = _cupy()
    if not isinstance(array, cupy.ndarray):
        raise TypeError(f"{name} must be a cupy.ndarray, got {type(array).__name__}")
    if array.dtype != cupy.dtype(dtype):
        raise TypeError(f"{name} must have dtype {cupy.dtype(dtype)}, got {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError(
            f"{name} must be C-contiguous; call cupy.ascontiguousarray first"
        )
    return array.data.ptr


def make_config(
    traj_type,
    n_dim: Sequence[int],
    L: int,
    *,
    phi2_dim: Optional[Sequence[int]] = None,
    triangulation="lower",
    is_masked: bool = False,
    num_masked: int = 0,
    n_coils: int = 1,
    eps: float = 1e-6,
    gpu_device_id: int = -1,
    stream=None,
    mem_fraction: float = 0.5,
    verbose: bool = False,
) -> MRConfig:
    """Build a validated :class:`MRConfig`.

    Args:
        traj_type: ``'2D'``, ``'3D'`` or ``'stack'`` (or a :class:`TrajType`).
        n_dim: Image matrix size, ``(Nx, Ny)`` or ``(Nx, Ny, Nz)``, given in
            **column-major order** just as in the documentation.
        L: Number of ranks.
        phi2_dim: Upsampled Toeplitz size ``(Mx, My[, Mz])``. Defaults to
            ``2 * n_dim``, except along kz of a stack trajectory where
            ``Mz = Nz`` because that direction is already Cartesian.
        triangulation: ``'lower'`` (default) or ``'upper'``.
        is_masked: Whether ``phi2_toep`` is stored only at masked locations.
        num_masked: Number of masked locations; inferred from the mask when
            left at 0.
        n_coils: Number of receive coils. ``SEs`` holds one map per coil,
            shared by every rank.
        eps: cufinufft tolerance used by :func:`calc_toep`.
        gpu_device_id: CUDA device, or ``-1`` (default) to use whichever device
            is current, which is what CuPy callers want.
        stream: A :class:`cupy.cuda.Stream`, a raw stream handle, or ``None``.
        mem_fraction: Share of the free device memory :func:`calc_toep` may use.
        verbose: Print per-stage progress.

    Returns:
        A populated and validated :class:`MRConfig`.

    Raises:
        ValueError: If ``n_dim`` or ``phi2_dim`` has the wrong length.
        MRError: If the library rejects the configuration.
    """
    tt = TrajType.parse(traj_type)
    cfg = MRConfig()
    check(lib.mr_config_defaults(ctypes.byref(cfg), tt), "mr_config_defaults")

    n_spatial = cfg.n_spatial
    n_dim = [int(v) for v in n_dim]
    if len(n_dim) != n_spatial:
        raise ValueError(
            f"n_dim must have {n_spatial} entries for this trajectory type, "
            f"got {len(n_dim)}"
        )

    if phi2_dim is None:
        phi2_dim = [2 * v for v in n_dim]
        if tt == TrajType.STACK:
            phi2_dim[2] = n_dim[2]
    else:
        phi2_dim = [int(v) for v in phi2_dim]
    if len(phi2_dim) != n_spatial:
        raise ValueError(
            f"phi2_dim must have {n_spatial} entries, got {len(phi2_dim)}"
        )

    for i in range(n_spatial):
        cfg.n_dim[i] = n_dim[i]
        cfg.phi2_dim[i] = phi2_dim[i]

    cfg.L = int(L)
    cfg.triangulation = Triangulation.parse(triangulation)
    cfg.is_masked = 1 if is_masked else 0
    cfg.num_masked = int(num_masked)
    cfg.n_coils = int(n_coils)
    cfg.eps = float(eps)
    cfg.gpu_device_id = int(gpu_device_id)
    cfg.mem_fraction = float(mem_fraction)
    cfg.verbose = 1 if verbose else 0
    cfg.stream = _stream_ptr(stream)

    check(lib.mr_config_validate(ctypes.byref(cfg)), "mr_config_validate")
    return cfg


def _stream_ptr(stream):
    """Extract a raw ``cudaStream_t`` from whatever the caller passed.

    Args:
        stream: ``None``, an ``int`` handle, or an object exposing ``.ptr``
            (as :class:`cupy.cuda.Stream` does).

    Returns:
        The stream handle as an ``int``, or ``None`` for the default stream.
    """
    if stream is None:
        return None
    if isinstance(stream, int):
        return stream or None
    ptr = getattr(stream, "ptr", None)
    if ptr is None:
        raise TypeError("stream must be None, an int handle, or expose a .ptr")
    return int(ptr) or None


def num_tri(L: int) -> int:
    """Number of stored entries of one packed Hermitian block.

    Args:
        L: Rank.

    Returns:
        ``L * (L + 1) // 2``.
    """
    return int(lib.mr_num_tri(int(L)))


def phi2_toep_shape(cfg: MRConfig) -> tuple:
    """Row-major shape of the ``phi2_toep`` array for a configuration.

    Args:
        cfg: A configuration from :func:`make_config`.

    Returns:
        ``(ntri, num_masked)`` when masked, else ``(ntri, Mz, My, Mx)`` for a
        3D/stack operator or ``(ntri, My, Mx)`` for a 2D one -- i.e. the
        column-major shapes of the documentation, reversed.
    """
    ntri = num_tri(cfg.L)
    if cfg.is_masked:
        return (ntri, int(cfg.num_masked))
    dims = [cfg.phi2_dim[i] for i in range(cfg.n_spatial)]
    return tuple([ntri] + list(reversed(dims)))


def _check_size(array, expected: int, name: str) -> None:
    """Verify an array's element count.

    Args:
        array: The CuPy array to check.
        expected: Expected number of elements.
        name: Argument name, used in the error message.

    Raises:
        ValueError: If the sizes disagree.
    """
    if array.size != expected:
        raise ValueError(
            f"{name} has {array.size} elements but {expected} were expected "
            f"(remember that Python shapes are the reverse of the "
            f"column-major shapes in the documentation)"
        )


def calc_toep(cfg: MRConfig, phi2, traj, mask=None, out=None):
    """Pre-compute the block-diagonal Toeplitz operator ``phi2_toep``.

    Args:
        cfg: Configuration from :func:`make_config`. ``cfg.phi2_dim`` selects
            the upsampled grid the kernel is evaluated on.
        phi2: ``complex64`` CuPy array of Gram blocks, shaped
            ``(L*L, Ntrajs)`` for ``'2D'``/``'3D'`` or ``(L*L, Nkz, Ntrajs)``
            for ``'stack'``. The ``L*L`` axis is the column-major ``L x L``
            matrix, i.e. ``Phi2.reshape(L * L, order='F')``.
        traj: ``float32`` CuPy array of non-Cartesian coordinates shaped
            ``(ndim, Nkx, Ntrajs)``, in radians per unit of the un-padded image
            grid (the usual ``[-pi, pi)`` range). ``ndim`` is 2 for ``'2D'``
            and ``'stack'``, 3 for ``'3D'``.
        mask: ``int64`` CuPy array of 0-based linear indices into the centred
            upsampled grid. Required when ``cfg.is_masked``.
        out: Optional pre-allocated ``complex64`` output of the right size.

    Returns:
        The ``complex64`` CuPy array ``phi2_toep``, shaped as described by
        :func:`phi2_toep_shape`.

    Raises:
        ValueError: If an array has the wrong size or a mask is missing.
        MRError: If the library reports a failure.
    """
    cupy = _cupy()

    ndim_traj = 3 if cfg.traj_type == TrajType.THREE_D else 2
    if traj.ndim != 3 or traj.shape[0] != ndim_traj:
        raise ValueError(
            f"traj must be shaped (ndim, Nkx, Ntrajs) with ndim = {ndim_traj}, "
            f"got {traj.shape}"
        )
    _, Nkx, Ntrajs = traj.shape

    nkz = cfg.phi2_dim[2] if cfg.traj_type == TrajType.STACK else 1
    _check_size(phi2, cfg.L * cfg.L * nkz * Ntrajs, "phi2")

    if cfg.is_masked:
        if mask is None:
            raise ValueError("cfg.is_masked is set, so a mask is required")
        if cfg.num_masked == 0:
            cfg.num_masked = int(mask.size)
            check(lib.mr_config_validate(ctypes.byref(cfg)), "mr_config_validate")
        elif int(mask.size) != int(cfg.num_masked):
            raise ValueError(
                f"mask has {mask.size} elements but cfg.num_masked is "
                f"{cfg.num_masked}"
            )

    shape = phi2_toep_shape(cfg)
    expected = int(lib.mr_phi2_toep_size(ctypes.byref(cfg)))
    if out is None:
        out = cupy.empty(shape, dtype=cupy.complex64)
    else:
        _check_size(out, expected, "out")

    check(
        lib.mr_calc_toep(
            ctypes.byref(cfg),
            _ptr(phi2, "phi2", "complex64"),
            _ptr(traj, "traj", "float32"),
            ctypes.c_int64(int(Ntrajs)),
            ctypes.c_int64(int(Nkx)),
            _ptr(mask, "mask", "int64", allow_none=True),
            _ptr(out, "out", "complex64"),
        ),
        "mr_calc_toep",
    )
    return out


class AhAPlan:
    """Reusable :math:`A^H A` operator.

    Creating a plan allocates the padded working buffers and the cuFFT plan
    once, so an iterative reconstruction can apply the operator many times
    without re-planning. Use it as a context manager, or call :meth:`close`.

    Args:
        cfg: Configuration from :func:`make_config`.

    Raises:
        MRError: If the plan cannot be created.
    """

    def __init__(self, cfg: MRConfig):
        self._cfg = cfg
        self._handle = ctypes.c_void_p()
        check(
            lib.mr_aha_plan_create(ctypes.byref(cfg), ctypes.byref(self._handle)),
            "mr_aha_plan_create",
        )

    @property
    def config(self) -> MRConfig:
        """The configuration this plan was built from."""
        return self._cfg

    def apply(self, x, SEs, phi2_toep, mask=None, out=None):
        """Apply :math:`A^H A` to a set of spatial coefficients.

        Args:
            x: ``complex64`` CuPy array shaped ``(L, Ny, Nx)`` (2D) or
                ``(L, Nz, Ny, Nx)`` (3D/stack).
            SEs: ``complex64`` CuPy array of sensitivity maps shaped
                ``(Ncoils, Nz, Ny, Nx)``, or ``(Ncoils, Ny, Nx)`` in 2D. One
                map per coil, shared by every rank.
            phi2_toep: ``complex64`` CuPy array from :func:`calc_toep`.
            mask: ``int64`` CuPy array of 0-based linear indices; required when
                ``cfg.is_masked``.
            out: Optional pre-allocated ``complex64`` output shaped like ``x``.

        Returns:
            The ``complex64`` CuPy array ``AhAx``, shaped like ``x``.

        Raises:
            RuntimeError: If the plan has already been closed.
            ValueError: If an array has the wrong size or a mask is missing.
            MRError: If the library reports a failure.
        """
        if not self._handle:
            raise RuntimeError("this AhAPlan has already been closed")
        cupy = _cupy()
        cfg = self._cfg

        _check_size(x, int(lib.mr_x_size(ctypes.byref(cfg))), "x")
        _check_size(SEs, int(lib.mr_se_size(ctypes.byref(cfg))), "SEs")
        _check_size(
            phi2_toep, int(lib.mr_phi2_toep_size(ctypes.byref(cfg))), "phi2_toep"
        )
        if cfg.is_masked and mask is None:
            raise ValueError("cfg.is_masked is set, so a mask is required")

        if out is None:
            out = cupy.empty(x.shape, dtype=cupy.complex64)
        else:
            _check_size(out, x.size, "out")

        check(
            lib.mr_aha_plan_execute(
                self._handle,
                _ptr(x, "x", "complex64"),
                _ptr(SEs, "SEs", "complex64"),
                _ptr(phi2_toep, "phi2_toep", "complex64"),
                _ptr(mask, "mask", "int64", allow_none=True),
                _ptr(out, "out", "complex64"),
            ),
            "mr_aha_plan_execute",
        )
        return out

    def close(self) -> None:
        """Release the plan's cuFFT plan and device buffers. Idempotent."""
        if self._handle:
            lib.mr_aha_plan_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __enter__(self) -> "AhAPlan":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter shutdown
            pass


def aha_toeplitz(cfg: MRConfig, x, SEs, phi2_toep, mask=None, out=None):
    """One-shot :math:`A^H A` application.

    Convenience wrapper that creates an :class:`AhAPlan`, applies it once and
    destroys it. Prefer the plan inside an iterative loop.

    Args:
        cfg: Configuration from :func:`make_config`.
        x: See :meth:`AhAPlan.apply`.
        SEs: See :meth:`AhAPlan.apply`.
        phi2_toep: See :meth:`AhAPlan.apply`.
        mask: See :meth:`AhAPlan.apply`.
        out: See :meth:`AhAPlan.apply`.

    Returns:
        The ``complex64`` CuPy array ``AhAx``, shaped like ``x``.
    """
    with AhAPlan(cfg) as plan:
        return plan.apply(x, SEs, phi2_toep, mask=mask, out=out)
