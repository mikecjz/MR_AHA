"""Low-level ``ctypes`` binding to the MR_AHA shared library.

This module is an almost literal transcription of ``include/mr_aha/mr_aha.h``.
It deliberately contains no array handling: it only locates ``libmr_aha_c``,
mirrors the C ``mr_config`` struct, and declares the prototypes. The friendly,
array-aware layer lives in :mod:`mr_aha.api`.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes.util import find_library
from pathlib import Path

__all__ = [
    "MRConfig",
    "TrajType",
    "Triangulation",
    "MaskMethod",
    "MRError",
    "lib",
    "check",
    "DEFAULT_MASK_THRESHOLD",
    "DEFAULT_MASK_RADIUS",
]


# --------------------------------------------------------------------------
# Enumerations, mirroring the C header
# --------------------------------------------------------------------------


class TrajType:
    """Trajectory families accepted by :class:`MRConfig`."""

    TWO_D = 0
    """Planar 2D trajectory; the image is ``[Nx, Ny, L]`` (column-major)."""

    THREE_D = 1
    """Fully 3D trajectory; the image is ``[Nx, Ny, Nz, L]`` (column-major)."""

    STACK = 2
    """Stack of identical 2D trajectories along kz, with ``Mz == Nz == Nkz``."""

    _FROM_STR = {"2d": TWO_D, "3d": THREE_D, "stack": STACK}

    @classmethod
    def parse(cls, value) -> int:
        """Convert ``'2D'``/``'3D'``/``'stack'`` (or an int) to an enum value.

        Args:
            value: A string naming the trajectory family, or an int already in
                the enum.

        Returns:
            The integer enum value.

        Raises:
            ValueError: If ``value`` names no known trajectory family.
        """
        if isinstance(value, int):
            return value
        try:
            return cls._FROM_STR[str(value).lower()]
        except KeyError:
            raise ValueError(
                f"traj_type must be one of '2D', '3D', 'stack'; got {value!r}"
            ) from None


class Triangulation:
    """Which triangle of each Hermitian ``L x L`` block is stored."""

    LOWER = 0
    """Lower triangle, column-major packed (MATLAB ``A(tril(true(L)))``)."""

    UPPER = 1
    """Upper triangle, column-major packed (MATLAB ``A(triu(true(L)))``)."""

    _FROM_STR = {"lower": LOWER, "upper": UPPER}

    @classmethod
    def parse(cls, value) -> int:
        """Convert ``'lower'``/``'upper'`` (or an int) to an enum value.

        Args:
            value: A string naming the triangle, or an int already in the enum.

        Returns:
            The integer enum value.

        Raises:
            ValueError: If ``value`` names no known triangle.
        """
        if isinstance(value, int):
            return value
        try:
            return cls._FROM_STR[str(value).lower()]
        except KeyError:
            raise ValueError(
                f"triangulation must be 'lower' or 'upper'; got {value!r}"
            ) from None


class MaskMethod:
    """How :func:`mr_aha.calc_mask` chooses the retained k-space locations."""

    NEIGHBOUR = 0
    """Keep every upsampled-grid location within ``mask_radius`` cells of the
    grid point nearest to a sample (default; no NUFFT needed)."""

    THRESHOLD = 1
    """Keep every location where the max-filtered unit-weight Toeplitz kernel
    exceeds ``mask_threshold`` (better for very sparse trajectories)."""

    _FROM_STR = {"neighbour": NEIGHBOUR, "neighbor": NEIGHBOUR,
                 "threshold": THRESHOLD}

    @classmethod
    def parse(cls, value) -> int:
        """Convert ``'neighbour'``/``'threshold'`` (or an int) to an enum value.

        Args:
            value: A string naming the method (``'neighbor'`` is accepted
                too), or an int already in the enum.

        Returns:
            The integer enum value.

        Raises:
            ValueError: If ``value`` names no known method.
        """
        if isinstance(value, int):
            return value
        try:
            return cls._FROM_STR[str(value).lower()]
        except KeyError:
            raise ValueError(
                f"mask_method must be 'neighbour' or 'threshold'; got {value!r}"
            ) from None


class MRConfig(ctypes.Structure):
    """``ctypes`` mirror of the C ``mr_config`` struct.

    Field order, types and therefore the natural alignment match the C
    declaration exactly. Use :func:`mr_aha.api.make_config` rather than
    populating this by hand.
    """

    _fields_ = [
        ("traj_type", ctypes.c_int),
        ("n_spatial", ctypes.c_int),
        ("n_dim", ctypes.c_int * 3),
        ("phi2_dim", ctypes.c_int * 3),
        ("L", ctypes.c_int),
        ("triangulation", ctypes.c_int),
        ("is_masked", ctypes.c_int),
        ("num_masked", ctypes.c_int64),
        ("n_coils", ctypes.c_int),
        ("eps", ctypes.c_double),
        ("gpu_device_id", ctypes.c_int),
        ("stream", ctypes.c_void_p),
        ("mem_fraction", ctypes.c_double),
        ("verbose", ctypes.c_int),
        ("mask_threshold", ctypes.c_double),
        ("mask_method", ctypes.c_int),
        ("mask_radius", ctypes.c_int),
    ]


class MRError(RuntimeError):
    """Raised when an MR_AHA entry point returns a non-zero status.

    Attributes:
        status: The numeric :c:enum:`mr_status` value.
    """

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------
# Library discovery
# --------------------------------------------------------------------------

_LIB_BASENAMES = {
    "win32": ["mr_aha_c.dll"],
    "darwin": ["libmr_aha_c.dylib"],
}.get(sys.platform, ["libmr_aha_c.so"])


def _candidate_paths():
    """Yield plausible locations of the MR_AHA shared library.

    The search order is: ``$MR_AHA_LIBRARY``, the installed package
    directory, the in-tree CMake build directories, and finally the platform
    loader's own search path.

    Yields:
        ``str`` or :class:`pathlib.Path` candidates, most specific first.
    """
    override = os.environ.get("MR_AHA_LIBRARY")
    if override:
        yield override

    here = Path(__file__).resolve().parent
    roots = [
        here,                              # installed alongside the package
        here.parent,                       # python/
        here.parent.parent / "build" / "lib",
        here.parent.parent / "build",
    ]
    for root in roots:
        for name in _LIB_BASENAMES:
            yield root / name

    for name in _LIB_BASENAMES:
        yield name
    found = find_library("mr_aha_c")
    if found:
        yield found


def _load_library() -> ctypes.CDLL:
    """Load ``libmr_aha_c`` from the first candidate that works.

    Returns:
        The loaded :class:`ctypes.CDLL`.

    Raises:
        OSError: If no candidate could be loaded, listing what was tried.
    """
    tried = []
    for cand in _candidate_paths():
        cand = str(cand)
        tried.append(cand)
        try:
            return ctypes.CDLL(cand)
        except OSError:
            continue
    raise OSError(
        "Could not load the MR_AHA shared library. Build it with\n"
        "    cmake -S . -B build -DMR_AHA_BUILD_PYTHON=ON && cmake --build build\n"
        "and point $MR_AHA_LIBRARY at the result if it lives elsewhere.\n"
        "Tried:\n  " + "\n  ".join(tried)
    )


lib = _load_library()

# --------------------------------------------------------------------------
# Prototypes
# --------------------------------------------------------------------------

_c_ptr = ctypes.c_void_p
_cfg_p = ctypes.POINTER(MRConfig)

lib.mr_status_string.restype = ctypes.c_char_p
lib.mr_status_string.argtypes = [ctypes.c_int]

lib.mr_last_error.restype = ctypes.c_char_p
lib.mr_last_error.argtypes = []

lib.mr_config_defaults.restype = ctypes.c_int
lib.mr_config_defaults.argtypes = [_cfg_p, ctypes.c_int]

lib.mr_config_validate.restype = ctypes.c_int
lib.mr_config_validate.argtypes = [_cfg_p]

lib.mr_num_tri.restype = ctypes.c_int64
lib.mr_num_tri.argtypes = [ctypes.c_int]

for _name in ("mr_image_size", "mr_pad_size", "mr_phi2_toep_size", "mr_x_size",
              "mr_se_size", "mr_mask_kernel_size"):
    _fn = getattr(lib, _name)
    _fn.restype = ctypes.c_int64
    _fn.argtypes = [_cfg_p]

lib.mr_aha_plan_create.restype = ctypes.c_int
lib.mr_aha_plan_create.argtypes = [_cfg_p, ctypes.POINTER(ctypes.c_void_p)]

lib.mr_aha_plan_execute.restype = ctypes.c_int
lib.mr_aha_plan_execute.argtypes = [
    ctypes.c_void_p, _c_ptr, _c_ptr, _c_ptr, _c_ptr, _c_ptr]

lib.mr_aha_plan_destroy.restype = ctypes.c_int
lib.mr_aha_plan_destroy.argtypes = [ctypes.c_void_p]

lib.mr_aha_toeplitz.restype = ctypes.c_int
lib.mr_aha_toeplitz.argtypes = [_cfg_p, _c_ptr, _c_ptr, _c_ptr, _c_ptr, _c_ptr]

lib.mr_calc_toep.restype = ctypes.c_int
lib.mr_calc_toep.argtypes = [
    _cfg_p, _c_ptr, _c_ptr, ctypes.c_int64, ctypes.c_int64, _c_ptr, _c_ptr]

lib.mr_calc_mask_kernel.restype = ctypes.c_int
lib.mr_calc_mask_kernel.argtypes = [
    _cfg_p, _c_ptr, ctypes.c_int64, ctypes.c_int64, _c_ptr]

lib.mr_calc_mask.restype = ctypes.c_int
lib.mr_calc_mask.argtypes = [
    _cfg_p, _c_ptr, ctypes.c_int64, ctypes.c_int64, _c_ptr,
    ctypes.POINTER(ctypes.c_int64)]

#: Default absolute threshold of :func:`mr_aha.calc_mask` with the
#: ``'threshold'`` method (in samples), mirroring ``MR_DEFAULT_MASK_THRESHOLD``.
DEFAULT_MASK_THRESHOLD = 0.05

#: Default neighbourhood radius of :func:`mr_aha.calc_mask` with the default
#: ``'neighbour'`` method (in upsampled grid cells), mirroring
#: ``MR_DEFAULT_MASK_RADIUS``.
DEFAULT_MASK_RADIUS = 2


def check(status: int, what: str) -> None:
    """Raise :class:`MRError` when ``status`` reports a failure.

    Args:
        status: Value returned by an MR_AHA entry point.
        what: Short description of the call, used in the message.

    Raises:
        MRError: If ``status`` is non-zero.
    """
    if status == 0:
        return
    summary = lib.mr_status_string(status).decode("utf-8", "replace")
    detail = lib.mr_last_error().decode("utf-8", "replace")
    raise MRError(status, f"{what}: {summary} ({status}): {detail}")
