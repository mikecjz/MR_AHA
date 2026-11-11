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
``phi2[Ntrajs, L, L]``        ``phi2[L, L, Ntrajs]``
``phi2[Ntrajs, Nkz, L, L]``  ``phi2[L, L, Nkz, Ntrajs]`` (stack)
``traj[Ntrajs, Nkx, ndim]``  ``traj[ndim, Nkx, Ntrajs]``
``mask_kernel[Mx, My, Mz]``  ``mask_kernel[Mz, My, Mx]``
===========================  =======================================

Every array must be a C-contiguous CuPy array resident on the GPU; ``complex64``
for the data arrays, ``float32`` for ``traj`` and ``int64`` for ``mask``.
Masks use **0-based** linear indices into the centred upsampled grid.

Gram blocks (MR-Multitasking convention)
----------------------------------------
The temporal basis :math:`\\Phi` is **rank x time** (MATLAB ``Phi_rt``,
``[L, T]``) and the spatial coefficients keep the rank as their last
column-major dimension. The Gram block of a readout (and kz encode) is

.. math::

    \\Phi_2 = \\sum_{t \\in \\mathcal{T}} \\Phi(:,t)\\,\\Phi(:,t)^H ,

MATLAB ``Phi_rt(:, idx) * Phi_rt(:, idx)'``, and every block-diagonal
multiplication treats the rank vector :math:`v` of a k-space location as a
**row vector multiplied from the right**, :math:`v \\leftarrow v\\,\\Phi_2`
(MR-Multitasking's ``reshape(k, [], L) * Phi2``). The same holds for the
Toeplitz blocks in ``phi2_toep``. In NumPy terms, with a row-major
``phi2[a, b, ...]`` holding column-major entry ``(b, a)``::

    out[l] = sum_m v[m] * Phi2[m, l] = sum_m phi2[l, m] * v[m]

A MR-Multitasking ``Phi2`` ``[L, L, Ntrajs, Nz]`` read with h5py (row-major
``(Nz, Ntrajs, L, L)``) is passed as ``P.transpose(2, 3, 0, 1)`` (plus a kz
fftshift if its kz order is natural). Earlier versions of the library
multiplied ``Phi2 @ v`` instead (time x rank :math:`\\Phi`,
:math:`\\Phi^H\\Phi`); blocks prepared for them need their two ``L`` axes
swapped.
"""

from __future__ import annotations

import ctypes
import numbers
from typing import Optional, Sequence

from ._lib import (
    DEFAULT_MASK_RADIUS,
    DEFAULT_MASK_THRESHOLD,
    OP_ALL_DEVICES,
    OP_SHARE_PRIMARY,
    OP_TOEP_SOURCES,
    MaskMethod,
    MRConfig,
    MRError,
    MROpInfo,
    MROpInputs,
    MROpOptions,
    OpMethod,
    ToepDist,
    TrajType,
    Triangulation,
    backend,
    check,
    lib,
)

__all__ = [
    "DEFAULT_MASK_RADIUS",
    "DEFAULT_MASK_THRESHOLD",
    "MaskMethod",
    "MRConfig",
    "MRError",
    "TrajType",
    "Triangulation",
    "make_config",
    "num_tri",
    "phi2_toep_shape",
    "mask_kernel_shape",
    "calc_mask_kernel",
    "calc_mask",
    "calc_toep",
    "aha_toeplitz",
    "AhAOp",
    "create_op",
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


def _xp():
    """Array module matching the loaded backend.

    Returns:
        :mod:`cupy` for the CUDA backend (arrays live in GPU memory) or
        :mod:`numpy` for the Metal backend (Apple silicon: arrays are ordinary
        host arrays in unified memory, which the GPU reads in place).
    """
    if backend() == "cuda":
        return _cupy()
    import numpy  # noqa: PLC0415
    return numpy


def _ptr(array, name: str, dtype, allow_none: bool = False):
    """Validate an array and return the raw pointer the C API expects.

    For the CUDA backend the array must be a CuPy array and the pointer is a
    device pointer; for the Metal backend it must be a NumPy array and the
    pointer is its host address (unified memory).

    Args:
        array: The array, or ``None`` when ``allow_none``.
        name: Argument name, used in error messages.
        dtype: Required dtype, as a NumPy dtype specifier.
        allow_none: Whether ``None`` is acceptable.

    Returns:
        The pointer as an ``int``, or ``None``.

    Raises:
        TypeError: If the array is not of the backend's array type or does not
            have the required dtype.
        ValueError: If the array is not C-contiguous.
    """
    if array is None:
        if allow_none:
            return None
        raise TypeError(f"{name} must not be None")
    xp = _xp()
    if not isinstance(array, xp.ndarray):
        raise TypeError(
            f"{name} must be a {xp.__name__}.ndarray for the {backend()} "
            f"backend, got {type(array).__name__}"
        )
    if array.dtype != xp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {xp.dtype(dtype)}, got {array.dtype}")
    if not array.flags.c_contiguous:
        raise ValueError(
            f"{name} must be C-contiguous; call {xp.__name__}.ascontiguousarray first"
        )
    if xp.__name__ == "numpy":
        return array.ctypes.data
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
    mask_method="neighbour",
    mask_radius: int = DEFAULT_MASK_RADIUS,
    mask_threshold: float = DEFAULT_MASK_THRESHOLD,
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
        num_masked: Number of masked locations; inferred from the mask by
            :func:`calc_toep` when left at 0.
        n_coils: Number of receive coils. ``SEs`` holds one map per coil,
            shared by every rank.
        eps: cufinufft tolerance used by :func:`calc_toep`.
        gpu_device_id: CUDA device, or ``-1`` (default) to use whichever device
            is current, which is what CuPy callers want.
        stream: A :class:`cupy.cuda.Stream`, a raw stream handle, or ``None``.
        mem_fraction: Share of the free device memory :func:`calc_toep` may use.
        verbose: Print per-stage progress.
        mask_method: How :func:`calc_mask` builds the mask: ``'neighbour'``
            (default) keeps the locations within ``mask_radius`` cells of a
            sample; ``'threshold'`` thresholds the unit-weight Toeplitz kernel
            at ``mask_threshold``. See :func:`calc_mask`.
        mask_radius: Neighbourhood radius in upsampled grid cells for the
            ``'neighbour'`` method (>= 0).
        mask_threshold: Absolute threshold for the ``'threshold'`` method, in
            units of samples; a location is kept when the (locally
            max-filtered) unit-weight Toeplitz kernel there exceeds it. One
            isolated on-grid sample contributes exactly 1.

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
    cfg.mask_threshold = float(mask_threshold)
    cfg.mask_method = MaskMethod.parse(mask_method)
    cfg.mask_radius = int(mask_radius)
    cfg.stream = _stream_ptr(stream)

    # A masked configuration may be created before its mask exists (e.g. to
    # call calc_mask); num_masked is then filled in by calc_toep.
    probe = MRConfig.from_buffer_copy(cfg)
    if probe.is_masked and probe.num_masked == 0:
        probe.is_masked = 0
    check(lib.mr_config_validate(ctypes.byref(probe)), "mr_config_validate")
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


def _check_traj(cfg: MRConfig, traj):
    """Validate a trajectory array and return ``(Ntrajs, Nkx)``.

    Args:
        cfg: Configuration that fixes the number of coordinates.
        traj: ``float32`` CuPy array shaped ``(ndim, Nkx, Ntrajs)``.

    Returns:
        Tuple ``(Ntrajs, Nkx)``.

    Raises:
        ValueError: If ``traj`` is not shaped ``(ndim, Nkx, Ntrajs)``.
    """
    ndim_traj = 3 if cfg.traj_type == TrajType.THREE_D else 2
    if traj.ndim != 3 or traj.shape[0] != ndim_traj:
        raise ValueError(
            f"traj must be shaped (ndim, Nkx, Ntrajs) with ndim = {ndim_traj}, "
            f"got {traj.shape}"
        )
    _, Nkx, Ntrajs = traj.shape
    return int(Ntrajs), int(Nkx)


def mask_kernel_shape(cfg: MRConfig) -> tuple:
    """Row-major shape of the unit-weight kernel of :func:`calc_mask_kernel`.

    Args:
        cfg: A configuration from :func:`make_config`.

    Returns:
        ``(Mz, My, Mx)`` for a 3D trajectory, else ``(My, Mx)``: the kernel of
        a stack trajectory is the same for every kz plane, so only the in-plane
        kernel is returned.
    """
    if cfg.traj_type == TrajType.THREE_D:
        return (int(cfg.phi2_dim[2]), int(cfg.phi2_dim[1]), int(cfg.phi2_dim[0]))
    return (int(cfg.phi2_dim[1]), int(cfg.phi2_dim[0]))


def calc_mask_kernel(cfg: MRConfig, traj, nufft: Optional[str] = None):
    """Magnitude of the unit-weight Toeplitz kernel of a trajectory.

    This is what :func:`calc_toep` would return for ``L = 1`` with every Gram
    block set to one -- the Toeplitz multiplier of the plain normal operator of
    the trajectory -- in absolute value. It is large where the trajectory
    samples k-space and decays to (nearly) zero away from it, and is the
    quantity :func:`calc_mask` thresholds.

    Args:
        cfg: Configuration from :func:`make_config`. ``L`` and the masking
            fields are ignored.
        traj: ``float32`` trajectory shaped ``(ndim, Nkx, Ntrajs)``.
        nufft: As for :func:`calc_toep`: ``None`` uses the library's own
            cufinufft on CUDA and mlx-nufft (two-stage :func:`calc_toep` with
            ``L = 1`` and unit Gram blocks) on Metal; ``'finufft'`` selects
            CPU FINUFFT for the two-stage path.

    Returns:
        A ``float32`` array (CuPy on CUDA, NumPy on Metal) shaped as
        :func:`mask_kernel_shape`, in centred order.

    Raises:
        ValueError: If ``traj`` is malformed.
        MRError: If the library reports a failure.
    """
    xp = _xp()
    Ntrajs, Nkx = _check_traj(cfg, traj)
    if nufft is None:
        nufft = "library" if backend() == "cuda" else "mlx"
    if nufft != "library":
        return _unit_kernel(cfg, traj, Ntrajs, nufft, xp)
    out = xp.empty(mask_kernel_shape(cfg), dtype=xp.float32)
    check(
        lib.mr_calc_mask_kernel(
            ctypes.byref(cfg),
            _ptr(traj, "traj", "float32"),
            ctypes.c_int64(Ntrajs),
            ctypes.c_int64(Nkx),
            _ptr(out, "out", "float32"),
        ),
        "mr_calc_mask_kernel",
    )
    return out


def _unit_kernel(cfg: MRConfig, traj, Ntrajs: int, nufft: str, xp):
    """Unit-weight kernel through the two-stage :func:`calc_toep`.

    Mirrors ``mr_unit_config`` in ``src/cuda/mr_calc_mask.cu``: rank one,
    unmasked, every Gram block equal to one, and a stack trajectory reduced to
    its in-plane 2D problem.
    """
    stack = cfg.traj_type == TrajType.STACK
    ns = 2 if (stack or cfg.traj_type == TrajType.TWO_D) else 3
    ucfg = make_config(
        TrajType.TWO_D if stack else cfg.traj_type,
        [cfg.n_dim[i] for i in range(ns)], 1,
        phi2_dim=[cfg.phi2_dim[i] for i in range(ns)],
        eps=cfg.eps, gpu_device_id=cfg.gpu_device_id,
        mem_fraction=cfg.mem_fraction, verbose=bool(cfg.verbose))
    ones = xp.ones((1, 1, Ntrajs), dtype=xp.complex64)
    T = calc_toep(ucfg, ones, traj, nufft=nufft)
    return xp.abs(T).astype(xp.float32).reshape(mask_kernel_shape(cfg))


def calc_mask(cfg: MRConfig, traj, threshold: Optional[float] = None,
              radius: Optional[int] = None, nufft: Optional[str] = None):
    """Derive the masked-operator sample set from the trajectory alone.

    Two methods are available (``cfg.mask_method``):

    ``'neighbour'`` (default)
        Every sample is mapped to its nearest point on the upsampled grid
        (``rint(k * M / (2 pi)) + M // 2`` per axis, periodic) and every
        location within ``radius`` cells of such a point (Chebyshev distance,
        i.e. a ``(2r+1)^d`` box) is kept. No NUFFT is needed. The default
        radius 2 matches the unmasked accuracy on densely sampled radial and 3D
        trajectories while keeping roughly the sampled disc or ball.
    ``'threshold'``
        The unit-weight kernel of :func:`calc_mask_kernel` is max-filtered
        over the immediate 3x3 (3x3x3 in 3D) neighbourhood of every location
        and compared with the absolute ``threshold``. The kernel is measured
        in samples (one isolated on-grid sample contributes exactly 1), so
        0.05 drops locations receiving less than 5 % of one sample's worth.
        It follows the kernel's slowly decaying, axis-aligned tails, which
        makes it more accurate per retained location for very sparse
        trajectories such as few-arm spirals.

    For a stack trajectory the in-plane mask is repeated over all kz planes.
    https://mikecjz.github.io/MR_AHA_docs/benchmarks/neighbour-mask.html
    compares the two methods.

    Typical use::

        cfg  = mr_aha.make_config('3D', (160,) * 3, L=12, n_coils=12,
                                  is_masked=True)
        mask = mr_aha.calc_mask(cfg, traj)
        with mr_aha.create_op(cfg, SEs, phi2, traj, mask) as op:  # sets num_masked
            y = op(x)

    Args:
        cfg: Configuration from :func:`make_config`. Not modified.
        traj: ``float32`` CuPy trajectory shaped ``(ndim, Nkx, Ntrajs)``.
        threshold: If given, use the ``'threshold'`` method with this
            absolute threshold (>= 0, in samples) regardless of
            ``cfg.mask_method``.
        radius: If given, use the ``'neighbour'`` method with this radius
            (>= 0, in grid cells) regardless of ``cfg.mask_method``.
        nufft: NUFFT of the ``'threshold'`` method's kernel, see
            :func:`calc_mask_kernel`. On Metal (or with an external engine)
            the kernel is computed in Python and the library's
            ``mr_calc_mask_from_kernel`` finishes the mask; the
            ``'neighbour'`` method needs no NUFFT and always runs in the
            library.

    Returns:
        An ``int64`` array (CuPy on CUDA, NumPy on Metal) of sorted, 0-based
        linear indices into the centred upsampled grid ``[Mx, My, Mz]``
        (column-major).

    Raises:
        ValueError: If ``traj`` is malformed or both ``threshold`` and
            ``radius`` are given.
        MRError: If the library reports a failure.
    """
    xp = _xp()
    Ntrajs, Nkx = _check_traj(cfg, traj)
    if threshold is not None and radius is not None:
        raise ValueError("pass at most one of threshold and radius")
    c = MRConfig.from_buffer_copy(cfg)  # private copy; the caller's is untouched
    if threshold is not None:
        c.mask_method = MaskMethod.THRESHOLD
        c.mask_threshold = float(threshold)
    if radius is not None:
        c.mask_method = MaskMethod.NEIGHBOUR
        c.mask_radius = int(radius)

    buf = xp.empty(int(lib.mr_pad_size(ctypes.byref(c))), dtype=xp.int64)
    n = ctypes.c_int64(0)
    if nufft is None:
        nufft = "library" if backend() == "cuda" else "mlx"
    if c.mask_method == MaskMethod.THRESHOLD and nufft != "library":
        kernel = calc_mask_kernel(c, traj, nufft=nufft)
        check(
            lib.mr_calc_mask_from_kernel(
                ctypes.byref(c),
                _ptr(kernel, "kernel", "float32"),
                _ptr(buf, "mask", "int64"),
                ctypes.byref(n),
            ),
            "mr_calc_mask_from_kernel",
        )
        mask = buf[: n.value].copy()
        del buf, kernel
        return mask
    check(
        lib.mr_calc_mask(
            ctypes.byref(c),
            _ptr(traj, "traj", "float32"),
            ctypes.c_int64(Ntrajs),
            ctypes.c_int64(Nkx),
            _ptr(buf, "mask", "int64"),
            ctypes.byref(n),
        ),
        "mr_calc_mask",
    )
    mask = buf[: n.value].copy()  # shrink to fit and release the worst case
    del buf
    return mask


def _check_gram(cfg: MRConfig, phi2, traj, mask):
    """Validate the inputs of :func:`calc_toep` and complete ``cfg``.

    Checks the sizes of ``phi2`` and ``traj`` against ``cfg`` and, in masked
    mode, that a mask is given; fills in ``cfg.num_masked`` from the mask when
    it is still 0.

    Args:
        cfg: Configuration; ``num_masked`` may be updated.
        phi2: Gram blocks, see :func:`calc_toep`.
        traj: Trajectory, see :func:`calc_toep`.
        mask: Mask or ``None``.

    Returns:
        ``(Ntrajs, Nkx)``.

    Raises:
        ValueError: If an array has the wrong size or a mask is missing.
    """
    Ntrajs, Nkx = _check_traj(cfg, traj)
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
    return Ntrajs, Nkx


def calc_toep(cfg: MRConfig, phi2, traj, mask=None, out=None,
              nufft: Optional[str] = None):
    """Pre-compute the block-diagonal Toeplitz operator ``phi2_toep``.

    Args:
        cfg: Configuration from :func:`make_config`. ``cfg.phi2_dim`` selects
            the upsampled grid the kernel is evaluated on.
        phi2: ``complex64`` CuPy array of Gram blocks
            :math:`\\Phi_2 = \\Phi\\Phi^H` (rank x time :math:`\\Phi`, see
            the module docstring), shaped ``(L, L, Ntrajs)`` for
            ``'2D'``/``'3D'`` or ``(L, L, Nkz, Ntrajs)`` for ``'stack'`` (the
            column-major ``[Ntrajs, L, L]`` / ``[Ntrajs, Nkz, L, L]`` of the
            documentation, reversed). Element ``phi2[j, i, ...]`` is entry
            ``(i, j)`` of the block (the column-major layout of the C API).
        traj: ``float32`` CuPy array of non-Cartesian coordinates shaped
            ``(ndim, Nkx, Ntrajs)``, in radians per unit of the un-padded image
            grid (the usual ``[-pi, pi)`` range). ``ndim`` is 2 for ``'2D'``
            and ``'stack'``, 3 for ``'3D'``.
        mask: ``int64`` CuPy array of 0-based linear indices into the centred
            upsampled grid, typically from :func:`calc_mask`. Required when
            ``cfg.is_masked``.
        out: Optional pre-allocated ``complex64`` output of the right size.
        nufft: Where the type-1 NUFFT runs. ``None`` (default) picks
            ``'library'`` on the CUDA backend (cufinufft inside the C library,
            one call) and ``'mlx'`` on the Metal backend (mlx-nufft in Python,
            then the C library finishes; see :mod:`mr_aha._psf`).
            ``'finufft'`` uses CPU FINUFFT for the same two-stage path, for
            validation.
    Returns:
        The ``complex64`` array ``phi2_toep`` (CuPy on CUDA, NumPy on Metal),
        shaped as described by :func:`phi2_toep_shape`.

    Raises:
        ValueError: If an array has the wrong size or a mask is missing.
        MRError: If the library reports a failure.
    """
    xp = _xp()
    Ntrajs, Nkx = _check_gram(cfg, phi2, traj, mask)

    shape = phi2_toep_shape(cfg)
    expected = int(lib.mr_phi2_toep_size(ctypes.byref(cfg)))
    if out is None:
        out = xp.empty(shape, dtype=xp.complex64)
    else:
        _check_size(out, expected, "out")

    if nufft is None:
        nufft = "library" if backend() == "cuda" else "mlx"
    if nufft != "library":
        # Validate the pointers' types up front, then run the two stages.
        _ptr(phi2, "phi2", "complex64")
        _ptr(traj, "traj", "float32")
        from ._psf import calc_toep_two_stage  # noqa: PLC0415
        calc_toep_two_stage(cfg, phi2, traj, mask, out, nufft, xp, _ptr)
        return out

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


def aha_toeplitz(cfg: MRConfig, x, SEs, phi2_toep, mask=None, out=None):
    """One-shot, stateless Toeplitz :math:`A^H A` (``mr_aha_toeplitz``).

    The low-level primitive: applies a finished kernel once, on the current
    device, creating and destroying the working buffers on every call. Use
    :func:`create_op` for anything applied more than once (it builds the
    kernel from the Gram blocks, keeps every buffer and can use several GPUs).

    Args:
        cfg: Configuration from :func:`make_config`.
        x: ``complex64`` array shaped ``(L, Ny, Nx)`` (2D) or
            ``(L, Nz, Ny, Nx)`` (3D/stack).
        SEs: ``complex64`` sensitivity maps shaped ``(Ncoils, Nz, Ny, Nx)``,
            or ``(Ncoils, Ny, Nx)`` in 2D. One map per coil, shared by every
            rank.
        phi2_toep: ``complex64`` kernel from :func:`calc_toep`.
        mask: ``int64`` 0-based linear indices; required when
            ``cfg.is_masked``.
        out: Optional pre-allocated ``complex64`` output shaped like ``x``.

    Returns:
        The ``complex64`` array ``AhAx`` (CuPy on CUDA, NumPy on Metal),
        shaped like ``x``.

    Raises:
        ValueError: If an array has the wrong size or a mask is missing.
        MRError: If the library reports a failure.
    """
    xp = _xp()
    _check_size(x, int(lib.mr_x_size(ctypes.byref(cfg))), "x")
    _check_size(SEs, int(lib.mr_se_size(ctypes.byref(cfg))), "SEs")
    _check_size(phi2_toep, int(lib.mr_phi2_toep_size(ctypes.byref(cfg))),
                "phi2_toep")
    if cfg.is_masked and mask is None:
        raise ValueError("cfg.is_masked is set, so a mask is required")
    if out is None:
        out = xp.empty(x.shape, dtype=xp.complex64)
    else:
        _check_size(out, x.size, "out")
    check(
        lib.mr_aha_toeplitz(
            ctypes.byref(cfg),
            _ptr(x, "x", "complex64"),
            _ptr(SEs, "SEs", "complex64"),
            _ptr(phi2_toep, "phi2_toep", "complex64"),
            _ptr(mask, "mask", "int64", allow_none=True),
            _ptr(out, "out", "complex64"),
        ),
        "mr_aha_toeplitz",
    )
    return out


def _devices_arg(devices):
    """Convert the ``devices`` argument of :func:`create_op` for the C API.

    Args:
        devices: ``None`` (the current device only), ``'all'`` (every visible
            device), an ``int`` ordinal or a sequence of ordinals.

    Returns:
        ``(ctypes int array or None, n_devices)`` for ``mr_op_options``.

    Raises:
        ValueError: If ``devices`` is a string other than ``'all'``.
    """
    if devices is None:
        return None, 0
    if isinstance(devices, str):
        if devices.lower() != "all":
            raise ValueError(f"devices must be None, 'all', an int or a list; "
                             f"got {devices!r}")
        return None, OP_ALL_DEVICES
    if isinstance(devices, numbers.Integral):  # also NumPy integer scalars
        devices = [devices]
    devices = [int(d) for d in devices]
    if not devices:
        return None, 0
    return (ctypes.c_int * len(devices))(*devices), len(devices)


#: Options accepted in ``nufft_opts`` of :func:`create_op`.
_NUFFT_OPTS = ("upsampfac", "gpu_method")


class AhAOp:
    """The :math:`A^H A` operator: one object, built from the Gram blocks.

    Create it with :meth:`create_op` (or the module-level :func:`create_op`),
    then call :meth:`apply` (or the object itself) as often as needed:

    >>> with mr_aha.create_op(cfg, SEs, phi2, traj) as op:   # doctest: +SKIP
    ...     y = op(x)

    The operator holds everything it needs: the sensitivity maps and either
    the Toeplitz kernel (``method='toeplitz'``, computed at creation) or the
    NUFFT plans and Gram blocks (``method='nufft'``). Whether it runs on one
    GPU or several is decided at creation (``devices``); on several GPUs the
    coils are split over the GPUs and the partial results summed on the
    current (primary) device. Inputs and outputs live on the primary device.

    Use it as a context manager, or call :meth:`close`, to release the GPU
    memory promptly.
    """

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "AhAOp is created with mr_aha.create_op(cfg, SEs, phi2, traj, ...) "
            "(or AhAOp.create_op), not by calling the class"
        )

    @classmethod
    def create_op(cls, cfg: MRConfig, SEs, phi2, traj, mask=None, *,
                  method: str = "toeplitz", devices=None,
                  toep_dist: str = "auto", share_primary: bool = True,
                  nufft_opts: Optional[dict] = None,
                  _phi2_toep=None) -> "AhAOp":
        """Create the :math:`A^H A` operator (``mr_op_create``).

        Args:
            cfg: Configuration from :func:`make_config`. ``n_coils`` maps are
                used; ``cfg.stream`` is the stream :meth:`apply` is ordered
                on. ``cfg.num_masked`` is filled in from ``mask`` when 0.
            SEs: ``complex64`` sensitivity maps ``(Ncoils, Nz, Ny, Nx)``, or
                ``(Ncoils, Ny, Nx)`` in 2D; one map per coil, shared by every
                rank.
            phi2: ``complex64`` Gram blocks :math:`\\Phi_2 = \\Phi\\Phi^H`
                (rank x time :math:`\\Phi`), ``(L, L, Ntrajs)`` or, for a
                stack, ``(L, L, Nkz, Ntrajs)``; see :func:`calc_toep`.
            traj: ``float32`` trajectory ``(ndim, Nkx, Ntrajs)`` in radians;
                see :func:`calc_toep`.
            mask: ``int64`` mask from :func:`calc_mask`; required when
                ``cfg.is_masked`` (Toeplitz method only).
            method: ``'toeplitz'`` (default): the kernel ``phi2_toep`` is
                computed here and every application is FFT + block-diagonal
                multiplication. ``'nufft'``: every application runs, per coil,
                a type-2 NUFFT, the Gram multiplication on the samples and a
                type-1 NUFFT (cufinufft); no kernel, so far less memory but
                more time per application. Both evaluate the same operator
                (to within the NUFFT tolerance ``cfg.eps``). ``'nufft'``
                rejects masked configurations.
            devices: ``None`` (default): the current device only. ``'all'``:
                every visible GPU. An ``int`` or a list of CUDA ordinals: those
                GPUs (the current device always takes part). The library may
                use fewer: with ``C`` coils on ``G`` GPUs, the smallest number
                with the same ``ceil(C/G)`` coils per GPU, and one GPU for
                problems too small to gain; :attr:`info` reports the choice.
                On Metal only ``None``, ``0``, ``[0]`` and ``'all'`` (the one
                GPU) are accepted.
            toep_dist: Toeplitz on several GPUs: how every GPU gets its copy
                of the kernel. ``'auto'`` (default) chooses between
                ``'split'`` (every GPU computes a share, the shares are
                exchanged) and ``'replicate'`` (every GPU computes the whole
                kernel, no transfers) with a cost model; ``'broadcast'``: the
                primary computes it and copies it to the others. The
                environment variable ``MR_AHA_TOEP_SPLIT`` overrides it.
            share_primary: Use ``SEs`` and ``mask`` in place on the primary
                device instead of copying them (default; saves a copy of the
                maps). The operator then keeps references to these arrays,
                which must not be modified in place while it is open.
            nufft_opts: ``method='nufft'`` only: ``{'upsampfac': 2.0,
                'gpu_method': 0}`` (cufinufft's upsampling factor, default 2,
                and cufinufft's ``gpu_method`` for both transforms, 1 or 2;
                0, the default, uses method 1 for the type-1 transform and
                cufinufft's choice for the type-2 one).
            _phi2_toep: Private: a finished kernel to use instead of
                computing one (tests, benchmarks, and the Metal backend
                internally). ``phi2`` and ``traj`` may then be ``None``.

        Returns:
            The operator.

        Raises:
            ValueError: On inconsistent arguments (sizes, method, devices,
                masked NUFFT).
            MRError: If the library reports a failure.
        """
        self = cls.__new__(cls)
        self._handle = ctypes.c_void_p()
        self._keep = ()
        self._impl = None
        self._method = OpMethod._TO_STR[OpMethod.parse(method)]
        is_nufft = self._method == "nufft"
        if is_nufft and cfg.is_masked:
            raise ValueError("method='nufft' has no kernel to mask; build the "
                             "configuration with is_masked=False")
        nufft_opts = dict(nufft_opts or {})
        if not is_nufft and nufft_opts:
            raise ValueError("nufft_opts is only used with method='nufft'")
        unknown = set(nufft_opts) - set(_NUFFT_OPTS)
        if unknown:
            raise ValueError(f"unknown nufft_opts {sorted(unknown)}; accepted: "
                             f"{list(_NUFFT_OPTS)}")
        _check_size(SEs, int(lib.mr_se_size(ctypes.byref(cfg))), "SEs")
        if _phi2_toep is not None and phi2 is None and traj is None and not is_nufft:
            # Injected kernel: the Gram blocks are not needed.
            Ntrajs, Nkx = 0, 0
            if cfg.is_masked:
                if mask is None:
                    raise ValueError("cfg.is_masked is set, so a mask is required")
                if cfg.num_masked == 0:
                    cfg.num_masked = int(mask.size)
        else:
            if phi2 is None or traj is None:
                raise ValueError("phi2 and traj are required (the operator is built "
                                 "from the Gram blocks)")
            Ntrajs, Nkx = _check_gram(cfg, phi2, traj, mask)
        if _phi2_toep is not None:
            if is_nufft:
                raise ValueError("_phi2_toep is only used with method='toeplitz'")
            _check_size(_phi2_toep, int(lib.mr_phi2_toep_size(ctypes.byref(cfg))),
                        "_phi2_toep")
        devs, ndev = _devices_arg(devices)
        # The operator's own copy of the (now complete) configuration: a later
        # change to the caller's cfg must not change this operator's sizes.
        self._cfg = MRConfig.from_buffer_copy(cfg)

        if backend() != "cuda":
            self._create_metal(cfg, SEs, phi2, traj, mask, devices, ndev,
                               nufft_opts, _phi2_toep, share_primary)
            return self

        opt = MROpOptions()
        check(lib.mr_op_options_defaults(ctypes.byref(opt)), "mr_op_options_defaults")
        opt.method = OpMethod.parse(method)
        opt.devices = ctypes.cast(devs, ctypes.POINTER(ctypes.c_int)) if devs else None
        opt.n_devices = ndev
        opt.toep_dist = ToepDist.parse(toep_dist)
        opt.flags = OP_SHARE_PRIMARY if share_primary else 0
        opt.nufft_upsampfac = float(nufft_opts.get("upsampfac", 0.0))
        opt.nufft_gpu_method = int(nufft_opts.get("gpu_method", 0))
        inp = MROpInputs(
            _ptr(SEs, "SEs", "complex64"),
            _ptr(phi2, "phi2", "complex64", allow_none=_phi2_toep is not None),
            _ptr(traj, "traj", "float32", allow_none=_phi2_toep is not None),
            int(Ntrajs), int(Nkx),
            _ptr(mask, "mask", "int64", allow_none=True),
            _ptr(_phi2_toep, "_phi2_toep", "complex64", allow_none=True),
        )
        # Arrays used in place must outlive the operator.
        if share_primary:
            self._keep = (SEs, mask, _phi2_toep)
        check(lib.mr_op_create(ctypes.byref(cfg), ctypes.byref(inp),
                               ctypes.byref(opt), ctypes.byref(self._handle)),
              "mr_op_create")
        return self

    def _create_metal(self, cfg, SEs, phi2, traj, mask, devices, ndev,
                      nufft_opts, phi2_toep, share_primary):
        """Metal backend: one GPU; the kernel or the NUFFT run through MLX.

        Toeplitz: the kernel is computed by :func:`calc_toep` (mlx-nufft +
        the library) unless supplied, and handed to the library's
        ``mr_op_create``. NUFFT: the MLX implementation of the reference
        operator (:mod:`mr_aha.nufft`) behind the same interface.
        """
        single = (ndev in (0, OP_ALL_DEVICES)) or \
            (ndev == 1 and list(devices if not isinstance(devices, numbers.Integral)
                                else [devices]) == [0])
        if not single:
            raise ValueError(f"the {backend()} backend has a single GPU; devices "
                             f"must be None, 0, [0] or 'all', got {devices!r}")
        if self._method == "nufft":
            from .nufft import _MlxNufft  # noqa: PLC0415
            self._impl = _MlxNufft(cfg, traj, cfg.eps)
            self._keep = (SEs, phi2)
            self._info = dict(method="nufft", devices=[0], coil_begin=[0],
                              coil_count=[int(cfg.n_coils)], group=[1],
                              n_split_coils=0, peer_access=False,
                              peer_bandwidth_gbs=0.0, toep_source="none",
                              toep_devices=0, toep_ms=0.0,
                              toep_component_ms=0.0, toep_gather_gbs=0.0,
                              create_ms=0.0)
            return
        self._toep_ms = 0.0
        if phi2_toep is None:
            # The library cannot time this part on Metal: time it here.
            import time  # noqa: PLC0415
            t0 = time.perf_counter()
            phi2_toep = calc_toep(cfg, phi2, traj, mask)
            self._toep_ms = (time.perf_counter() - t0) * 1e3
        opt = MROpOptions()
        check(lib.mr_op_options_defaults(ctypes.byref(opt)), "mr_op_options_defaults")
        opt.flags = OP_SHARE_PRIMARY if share_primary else 0
        inp = MROpInputs(_ptr(SEs, "SEs", "complex64"), None, None, 0, 0,
                         _ptr(mask, "mask", "int64", allow_none=True),
                         _ptr(phi2_toep, "phi2_toep", "complex64"))
        self._keep = (SEs, mask, phi2_toep)
        check(lib.mr_op_create(ctypes.byref(cfg), ctypes.byref(inp),
                               ctypes.byref(opt), ctypes.byref(self._handle)),
              "mr_op_create")

    @property
    def config(self) -> MRConfig:
        """The configuration this operator was built from."""
        return self._cfg

    @property
    def method(self) -> str:
        """``'toeplitz'`` or ``'nufft'``."""
        return self._method

    def _check_open(self) -> None:
        """Raise :class:`RuntimeError` when the operator has been closed."""
        if not self._handle and self._impl is None:
            raise RuntimeError("this AhAOp has already been closed")

    def apply(self, x, out=None):
        """Apply :math:`A^H A` to a set of spatial coefficients.

        Args:
            x: ``complex64`` array on the primary device, ``(L, Ny, Nx)``
                (2D) or ``(L, Nz, Ny, Nx)`` (3D/stack).
            out: Optional pre-allocated ``complex64`` output shaped like
                ``x``; must not be ``x``.

        Returns:
            The ``complex64`` array ``AhAx`` (CuPy on CUDA, NumPy on Metal),
            shaped like ``x``. On CUDA the result is ordered on
            ``cfg.stream`` (complete once that stream reaches this point).

        Raises:
            RuntimeError: If the operator has been closed.
            ValueError: If ``x`` or ``out`` has the wrong size.
            MRError: If the library reports a failure.
        """
        self._check_open()
        xp = _xp()
        _check_size(x, int(lib.mr_x_size(ctypes.byref(self._cfg))), "x")
        if out is not None:
            _check_size(out, x.size, "out")
        if self._impl is not None:
            SEs, phi2 = self._keep
            return self._impl.apply(x, SEs, phi2, out=out)
        if out is None:
            out = xp.empty(x.shape, dtype=xp.complex64)
        check(
            lib.mr_op_execute(self._handle, _ptr(x, "x", "complex64"),
                              _ptr(out, "out", "complex64")),
            "mr_op_execute",
        )
        return out

    __call__ = apply

    @property
    def info(self) -> dict:
        """How the operator was built and distributes the work.

        Returns:
            A dictionary with ``method``; ``devices`` (CUDA ordinals, primary
            first), ``coil_begin`` / ``coil_count`` (whole coils per device),
            ``group`` (Toeplitz: coils per group on each device; NUFFT: 1),
            ``n_split_coils`` (Toeplitz remainder coils split by rank),
            ``peer_access``, ``peer_bandwidth_gbs``; how the kernel was
            obtained: ``toep_source`` (``'broadcast'``: the primary computed
            it, and copied it to any other GPU, ``'split'``: every GPU
            computed a share and the shares were exchanged, ``'replicate'``:
            every GPU computed it itself, ``'supplied'``, or ``'none'`` for
            the NUFFT method), ``toep_devices``, ``toep_ms`` (time of
            computing and distributing it), ``toep_component_ms`` (GPU time
            per triangular component), ``toep_gather_gbs`` (measured
            concurrent receive bandwidth); and ``create_ms`` (whole
            creation).
        """
        self._check_open()
        if self._impl is not None:
            return dict(self._info)
        metal_ms = getattr(self, "_toep_ms", 0.0)
        inf = MROpInfo()
        check(lib.mr_op_info(self._handle, ctypes.byref(inf)), "mr_op_info")
        n = inf.n_devices
        return dict(
            method=OpMethod._TO_STR.get(int(inf.method), "unknown"),
            devices=list(inf.devices[:n]),
            coil_begin=list(inf.coil_begin[:n]),
            coil_count=list(inf.coil_count[:n]),
            group=list(inf.group[:n]),
            n_split_coils=int(inf.n_split_coils),
            peer_access=bool(inf.peer_access),
            peer_bandwidth_gbs=float(inf.peer_bandwidth_gbs),
            toep_source=("broadcast" if metal_ms else
                         OP_TOEP_SOURCES.get(int(inf.toep_source), "unknown")),
            toep_devices=1 if metal_ms else int(inf.toep_devices),
            toep_ms=metal_ms or float(inf.toep_ms),
            toep_component_ms=float(inf.toep_component_ms),
            toep_gather_gbs=float(inf.toep_gather_gbs),
            create_ms=float(inf.create_ms),
        )

    def kernel(self, k: int = 0):
        """The Toeplitz kernel ``phi2_toep`` the operator holds on its
        ``k``-th device (``mr_op_kernel``).

        The array is a read-only *view* of memory owned by the operator: it
        keeps the operator object alive, but becomes invalid when the
        operator is closed, and must not be modified. Copy it to keep it.

        Args:
            k: Device index, ``0 <= k < len(info['devices'])``; 0 is the
                primary device. The array lives on ``info['devices'][k]``.

        Returns:
            A ``complex64`` array shaped like :func:`phi2_toep_shape`.

        Raises:
            RuntimeError: If the operator has been closed.
            MRError: For a NUFFT operator (no kernel) or ``k`` out of range.
        """
        self._check_open()
        if self._impl is not None:
            raise MRError(-7, "mr_op_kernel: a NUFFT operator has no kernel")
        ptr = ctypes.c_void_p()
        check(lib.mr_op_kernel(self._handle, int(k), ctypes.byref(ptr)),
              "mr_op_kernel")
        shape = phi2_toep_shape(self._cfg)
        size = int(lib.mr_phi2_toep_size(ctypes.byref(self._cfg)))
        if backend() != "cuda":
            import numpy  # noqa: PLC0415
            buf = (ctypes.c_char * (size * 8)).from_address(ptr.value)
            arr = numpy.frombuffer(buf, dtype=numpy.complex64).reshape(shape)
            arr.flags.writeable = False
            return arr
        cp = _cupy()
        dev = self.info["devices"][int(k)]
        mem = cp.cuda.UnownedMemory(ptr.value, size * 8, self, device_id=dev)
        return cp.ndarray(shape, dtype=cp.complex64,
                          memptr=cp.cuda.MemoryPointer(mem, 0))

    def close(self) -> None:
        """Release the operator and everything it holds on every GPU.
        Idempotent."""
        if self._handle:
            lib.mr_op_destroy(self._handle)
            self._handle = ctypes.c_void_p()
        if self._impl is not None:
            self._impl.close()
            self._impl = None
        self._keep = ()

    def __enter__(self) -> "AhAOp":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # pragma: no cover - interpreter shutdown
            pass


def create_op(cfg: MRConfig, SEs, phi2, traj, mask=None, **kwargs) -> AhAOp:
    """Create the :math:`A^H A` operator; alias of :meth:`AhAOp.create_op`.

    Args:
        cfg: Configuration from :func:`make_config`.
        SEs: Sensitivity maps, see :meth:`AhAOp.create_op`.
        phi2: Gram blocks, see :meth:`AhAOp.create_op`.
        traj: Trajectory, see :meth:`AhAOp.create_op`.
        mask: Mask (Toeplitz, masked configurations only).
        **kwargs: ``method``, ``devices``, ``toep_dist``, ``share_primary``,
            ``nufft_opts``; see :meth:`AhAOp.create_op`.

    Returns:
        The operator, an :class:`AhAOp`.
    """
    return AhAOp.create_op(cfg, SEs, phi2, traj, mask, **kwargs)
