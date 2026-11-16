"""Low-level ``ctypes`` binding to the MR_AHA shared library.

This module is an almost literal transcription of ``include/mr_aha/mr_aha.h``.
It deliberately contains no array handling: it only locates ``libmr_aha_c``,
mirrors the C ``mr_config`` struct, and declares the prototypes. The friendly,
array-aware layer lives in :mod:`mr_aha.api`.

What the module does, in order, when it is first imported:

1. Defines Python mirrors of the C enumerations (:class:`TrajType`,
   :class:`Triangulation`, :class:`OpMethod`, :class:`ToepDist`) and of the
   C structs (:class:`MRConfig`, :class:`MROpInputs`, :class:`MROpOptions`,
   :class:`MROpInfo`). The struct mirrors must match the C declarations field
   for field, because ``ctypes`` lays them out with the same natural
   alignment rules as the C compiler and the library reads them by address.
2. Loads the shared library (:data:`lib`) from the first location that works
   (see :func:`_candidate_paths`). Failure raises :class:`OSError` at import
   time, so every other module of the package can assume :data:`lib` exists.
3. Declares ``restype`` / ``argtypes`` of every entry point used by the
   package. Without these declarations ``ctypes`` would assume ``int``
   arguments and an ``int`` return value, which truncates 64-bit pointers and
   ``int64_t`` sizes on LP64 platforms.

Pointer convention: every array argument is declared ``c_void_p`` and is
passed as a plain Python ``int`` address (or ``None`` for ``NULL``). On the
CUDA backend that address is a device pointer (``cupy.ndarray.data.ptr``);
on the Metal backend it is a host address in unified memory
(``numpy.ndarray.ctypes.data``). See ``@section mr_memory`` of the header.

Every C entry point returns an ``mr_status`` (0 on success, negative on
failure); :func:`check` turns a failure into :class:`MRError`.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes.util import find_library
from pathlib import Path

# Names re-exported to the rest of the package. The operator-object mirrors
# (MROpInputs, MROpOptions, MROpInfo, OpMethod, ToepDist, OP_*) are imported
# explicitly by mr_aha.api and are not part of the public surface.
__all__ = [
    "MRConfig",
    "TrajType",
    "Triangulation",
    "MRError",
    "lib",
    "check",
    "backend",
    "DEFAULT_MASK_RADIUS",
]


# --------------------------------------------------------------------------
# Enumerations, mirroring the C header
# --------------------------------------------------------------------------
# The C enums are plain ``int``s in every struct and prototype, so they are
# mirrored as classes of integer constants (not :class:`enum.IntEnum`): the
# values can be assigned straight into ``ctypes.c_int`` fields. Each class
# carries a ``parse`` helper that accepts the user-facing string spelling.


class TrajType:
    """Trajectory families accepted by :class:`MRConfig` (C ``mr_traj_type``).

    The trajectory family decides the number of spatial dimensions
    (``n_spatial``: 2 for :attr:`TWO_D`, 3 otherwise), the number of
    coordinates per trajectory sample (``ndim``: 3 for :attr:`THREE_D`, 2
    otherwise), and whether the Gram blocks carry a kz axis (:attr:`STACK`).
    """

    TWO_D = 0
    """Planar 2D trajectory; the image is ``[Nx, Ny, L]`` (column-major)."""

    THREE_D = 1
    """Fully 3D trajectory; the image is ``[Nx, Ny, Nz, L]`` (column-major)."""

    STACK = 2
    """Stack of identical 2D trajectories along kz, with ``Mz == Nz == Nkz``."""

    # Case-insensitive lookup table used by :meth:`parse`; keys are the
    # lower-cased spellings accepted from users (``'2D'`` -> ``'2d'``).
    _FROM_STR = {"2d": TWO_D, "3d": THREE_D, "stack": STACK}

    @classmethod
    def parse(cls, value) -> int:
        """Convert ``'2D'``/``'3D'``/``'stack'`` (or an int) to an enum value.

        Matching is case-insensitive. An ``int`` is passed through unchecked;
        an out-of-range integer is rejected later by ``mr_config_validate``
        (``mr_config_defaults`` does not check it).

        Args:
            value: A string naming the trajectory family, or an int already in
                the enum.

        Returns:
            The integer enum value.

        Raises:
            ValueError: If ``value`` names no known trajectory family.
        """
        # Integers are assumed to already be C enum values.
        if isinstance(value, int):
            return value
        # Anything else is stringified and looked up case-insensitively;
        # ``from None`` hides the internal KeyError from the traceback.
        try:
            return cls._FROM_STR[str(value).lower()]
        except KeyError:
            raise ValueError(
                f"traj_type must be one of '2D', '3D', 'stack'; got {value!r}"
            ) from None


class Triangulation:
    """Which triangle of each Hermitian ``L x L`` block is stored
    (C ``mr_triangulation``).

    ``phi2_toep`` keeps only ``(L^2 + L) / 2`` of the ``L^2`` entries of each
    block; the other half is reconstructed from Hermitian symmetry,
    ``A[i, j] = conj(A[j, i])``. Both packings are column-major:

    * :attr:`LOWER`: element ``(i, j)``, ``i >= j``, is stored at
      ``t = j*L - j*(j-1)/2 + (i - j)``.
    * :attr:`UPPER`: element ``(i, j)``, ``i <= j``, is stored at
      ``t = j*(j+1)/2 + i``.

    :func:`mr_aha._psf.tri_pairs` enumerates the same order in Python.
    """

    LOWER = 0
    """Lower triangle, column-major packed (MATLAB ``A(tril(true(L)))``)."""

    UPPER = 1
    """Upper triangle, column-major packed (MATLAB ``A(triu(true(L)))``)."""

    # Case-insensitive lookup table used by :meth:`parse`.
    _FROM_STR = {"lower": LOWER, "upper": UPPER}

    @classmethod
    def parse(cls, value) -> int:
        """Convert ``'lower'``/``'upper'`` (or an int) to an enum value.

        Matching is case-insensitive. An ``int`` is passed through unchecked
        (``mr_config_validate`` rejects values other than 0 and 1).

        Args:
            value: A string naming the triangle, or an int already in the enum.

        Returns:
            The integer enum value.

        Raises:
            ValueError: If ``value`` names no known triangle.
        """
        # Integers are assumed to already be C enum values.
        if isinstance(value, int):
            return value
        # Strings (or str()-able objects) are looked up case-insensitively.
        try:
            return cls._FROM_STR[str(value).lower()]
        except KeyError:
            raise ValueError(
                f"triangulation must be 'lower' or 'upper'; got {value!r}"
            ) from None


class MRConfig(ctypes.Structure):
    """``ctypes`` mirror of the C ``mr_config`` struct.

    Field order, types and therefore the natural alignment match the C
    declaration exactly. Use :func:`mr_aha.api.make_config` rather than
    populating this by hand.

    The struct is passed to the library by reference
    (``ctypes.byref(cfg)``); the library copies what it needs, so a
    configuration need not outlive a call. :class:`mr_aha.AhAOp` nevertheless
    keeps its own copy (``MRConfig.from_buffer_copy``) so that later edits to
    the caller's object cannot change the sizes it assumes.

    Fields (all sizes column-major, as in the C header):

    * ``traj_type`` -- :class:`TrajType` value.
    * ``n_spatial`` -- 2 or 3; set from ``traj_type`` by
      ``mr_config_defaults``.
    * ``n_dim`` -- image size ``[Nx, Ny, Nz]``; unused entries are 1.
    * ``phi2_dim`` -- upsampled Toeplitz grid ``[Mx, My, Mz]``; unused
      entries are 1, and ``Mz == Nz`` for :attr:`TrajType.STACK`.
    * ``L`` -- number of ranks (subspace coefficients).
    * ``triangulation`` -- :class:`Triangulation` value of ``phi2_toep``.
    * ``is_masked`` / ``num_masked`` -- whether ``phi2_toep`` is stored only
      at ``num_masked`` masked grid locations (``int64_t`` in C).
    * ``n_coils`` -- number of receive coils.
    * ``eps`` -- NUFFT tolerance of the kernel computation (``double``).
    * ``gpu_device_id`` -- CUDA ordinal, or -1 for the caller's current
      device (ignored on Metal).
    * ``stream`` -- ``cudaStream_t`` as ``void *``; ``None`` is the default
      stream (ignored on Metal).
    * ``mem_fraction`` -- share of free device memory the library may use
      for working buffers (clamped to ``(0, 0.95]`` by the library).
    * ``verbose`` -- non-zero to print per-stage timing.
    * ``mask_radius`` -- neighbourhood radius of ``mr_calc_mask``.
    """

    # Order and C types must stay identical to ``struct mr_config`` in
    # mr_aha.h: the library reads the struct by offset. Note the int64
    # ``num_masked`` after eleven ints (44 bytes): ctypes inserts the same
    # 4-byte alignment padding the C compiler does.
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
        ("mask_radius", ctypes.c_int),
    ]


# --------------------------------------------------------------------------
# Operator object (mr_op): constants, enums and struct mirrors
# --------------------------------------------------------------------------

#: Mirror of ``MR_OP_MAX_DEVICES``: most devices one operator can use. Also
#: the length of the fixed-size arrays of :class:`MROpInfo`.
OP_MAX_DEVICES = 16

#: Mirror of ``MR_OP_ALL_DEVICES`` (``n_devices`` selecting every visible
#: device).
OP_ALL_DEVICES = -1

#: Mirror of ``MR_OP_SHARE_PRIMARY``: ``mr_op_options.flags`` bit that makes
#: the primary device use the caller's ``SEs``, ``mask`` and supplied
#: ``phi2_toep`` in place instead of copying them (so they must stay alive and
#: unchanged until the operator is destroyed).
OP_SHARE_PRIMARY = 1


class OpMethod:
    """How :func:`mr_aha.create_op` evaluates :math:`A^H A` (``mr_op_method``)."""

    TOEPLITZ = 0
    """Toeplitz embedding with a kernel pre-computed at creation (default)."""

    NUFFT = 1
    """Type-2 NUFFT, Gram multiplication, type-1 NUFFT per coil; no kernel."""

    # String <-> enum tables. _TO_STR doubles as the set of valid integers in
    # :meth:`parse` and is used by mr_aha.api to normalise ``method`` and to
    # name ``mr_op_info_t.method``.
    _FROM_STR = {"toeplitz": TOEPLITZ, "nufft": NUFFT}
    _TO_STR = {TOEPLITZ: "toeplitz", NUFFT: "nufft"}

    @classmethod
    def parse(cls, value) -> int:
        """Convert ``'toeplitz'``/``'nufft'`` to an enum value.

        Unlike :meth:`TrajType.parse`, an integer is range-checked: only
        values present in ``_TO_STR`` pass through unchanged; any other
        integer is stringified, fails the lookup and raises.

        Args:
            value: A string naming the method, or an int already in the enum.

        Returns:
            The integer enum value.

        Raises:
            ValueError: If ``value`` names no known method.
        """
        # A valid integer enum value is accepted as is.
        if isinstance(value, int) and value in cls._TO_STR:
            return value
        # Otherwise a case-insensitive string lookup.
        try:
            return cls._FROM_STR[str(value).lower()]
        except KeyError:
            raise ValueError(
                f"method must be 'toeplitz' or 'nufft'; got {value!r}"
            ) from None


class ToepDist:
    """How a multi-GPU Toeplitz operator puts the kernel on every GPU
    (``mr_toep_dist``). See :func:`mr_aha.create_op`.

    Only relevant to ``method='toeplitz'`` on more than one device; with one
    device every mode computes the kernel in place. The environment variable
    ``MR_AHA_TOEP_SPLIT`` (read by the library) can override the choice.
    """

    AUTO = 0
    """Cost model: ``'split'`` or ``'replicate'``."""

    SPLIT = 1
    """Every GPU computes a share of the components; shares exchanged."""

    REPLICATE = 2
    """Every GPU computes the whole kernel itself, no transfers."""

    BROADCAST = 3
    """The primary computes the kernel alone and copies it to the others."""

    # Case-insensitive lookup table used by :meth:`parse`.
    _FROM_STR = {"auto": AUTO, "split": SPLIT, "replicate": REPLICATE,
                 "broadcast": BROADCAST}

    @classmethod
    def parse(cls, value) -> int:
        """Convert ``'auto'``/``'split'``/``'replicate'``/``'broadcast'``.

        Integers in ``0..3`` pass through; other integers fail the string
        lookup and raise.

        Args:
            value: A string naming the mode, or an int already in the enum.

        Returns:
            The integer enum value.

        Raises:
            ValueError: If ``value`` names no known mode.
        """
        # A valid integer enum value is accepted as is.
        if isinstance(value, int) and 0 <= value <= 3:
            return value
        # Otherwise a case-insensitive string lookup.
        try:
            return cls._FROM_STR[str(value).lower()]
        except KeyError:
            raise ValueError(
                "toep_dist must be 'auto', 'split', 'replicate' or 'broadcast'; "
                f"got {value!r}"
            ) from None


#: Names of the ``mr_op_info_t.toep_source`` values (``MR_OP_TOEP_*``): how
#: an operator obtained ``phi2_toep``. Used by :attr:`mr_aha.AhAOp.info` to
#: report the source as a string. Note that the numeric values do *not*
#: coincide with :class:`ToepDist` (``MR_OP_TOEP_BROADCAST`` is 1 here but
#: ``BROADCAST`` is 3 there).
OP_TOEP_SOURCES = {0: "supplied", 1: "broadcast", 2: "split", 3: "replicate",
                   4: "none"}


class MROpInputs(ctypes.Structure):
    """``ctypes`` mirror of the C ``mr_op_inputs`` struct (arrays as raw
    device or unified-memory pointers).

    Fields:

    * ``d_SEs`` -- sensitivity maps, ``n_coils`` of them. Required.
    * ``d_phi2`` -- Gram blocks; may be ``NULL`` only for the Toeplitz
      method with ``d_phi2_toep`` supplied.
    * ``d_traj`` -- ``float32`` trajectory in radians; required whenever
      ``d_phi2`` is.
    * ``Ntrajs`` / ``Nkx`` -- number of readouts and samples per readout
      (``int64_t``).
    * ``d_mask`` -- ``int64`` mask of ``num_masked`` 0-based indices, or
      ``NULL`` when unmasked.
    * ``d_phi2_toep`` -- optional finished kernel (used by the Metal path and
      by tests), or ``NULL`` to let the operator compute it.

    All arrays live on the primary device. :class:`mr_aha.AhAOp` fills the
    struct positionally, so the field order matters here too.
    """

    _fields_ = [
        ("d_SEs", ctypes.c_void_p),
        ("d_phi2", ctypes.c_void_p),
        ("d_traj", ctypes.c_void_p),
        ("Ntrajs", ctypes.c_int64),
        ("Nkx", ctypes.c_int64),
        ("d_mask", ctypes.c_void_p),
        ("d_phi2_toep", ctypes.c_void_p),
    ]


class MROpOptions(ctypes.Structure):
    """``ctypes`` mirror of the C ``mr_op_options`` struct. Fill with
    ``mr_op_options_defaults`` before overriding fields.

    Fields:

    * ``method`` -- :class:`OpMethod` value.
    * ``devices`` / ``n_devices`` -- ``int *`` array of extra CUDA ordinals
      and its length. With ``devices = NULL``, ``n_devices = 0`` means the
      primary only and :data:`OP_ALL_DEVICES` every visible device.
    * ``toep_dist`` -- :class:`ToepDist` value.
    * ``flags`` -- bit set of :data:`OP_SHARE_PRIMARY`.
    * ``nufft_upsampfac`` -- NUFFT method: cufinufft upsampling factor, 0
      for the default 2.0.
    * ``nufft_gpu_method`` -- NUFFT method: cufinufft ``gpu_method`` (0 =
      library default).

    The ``devices`` field is a raw C pointer: the ``ctypes`` array it points
    to must stay alive until ``mr_op_create`` returns.
    """

    _fields_ = [
        ("method", ctypes.c_int),
        ("devices", ctypes.POINTER(ctypes.c_int)),
        ("n_devices", ctypes.c_int),
        ("toep_dist", ctypes.c_int),
        ("flags", ctypes.c_int),
        ("nufft_upsampfac", ctypes.c_double),
        ("nufft_gpu_method", ctypes.c_int),
    ]


class MROpInfo(ctypes.Structure):
    """``ctypes`` mirror of the C ``mr_op_info_t`` struct; see
    :attr:`mr_aha.AhAOp.info` for a dictionary view.

    The per-device arrays have the fixed length :data:`OP_MAX_DEVICES`; only
    the first ``n_devices`` entries are meaningful, and entry 0 is the
    primary device.
    """

    _fields_ = [
        ("method", ctypes.c_int),
        ("n_devices", ctypes.c_int),
        ("devices", ctypes.c_int * OP_MAX_DEVICES),
        ("coil_begin", ctypes.c_int * OP_MAX_DEVICES),
        ("coil_count", ctypes.c_int * OP_MAX_DEVICES),
        ("group", ctypes.c_int * OP_MAX_DEVICES),
        ("n_split_coils", ctypes.c_int),
        ("peer_access", ctypes.c_int),
        ("peer_bandwidth_gbs", ctypes.c_double),
        ("toep_source", ctypes.c_int),
        ("toep_devices", ctypes.c_int),
        ("toep_ms", ctypes.c_double),
        ("toep_component_ms", ctypes.c_double),
        ("toep_gather_gbs", ctypes.c_double),
        ("create_ms", ctypes.c_double),
    ]


class MRError(RuntimeError):
    """Raised when an MR_AHA entry point returns a non-zero status.

    The message has the form ``"<call>: <mr_status_string> (<status>):
    <mr_last_error>"`` (see :func:`check`).

    Attributes:
        status: The numeric :c:enum:`mr_status` value (negative), e.g. -1
            ``MR_ERR_INVALID_ARG``, -2 ``MR_ERR_INVALID_CONFIG``, -3
            ``MR_ERR_ALLOC``, -7 ``MR_ERR_UNSUPPORTED``.
    """

    def __init__(self, status: int, message: str):
        """Store the status code alongside the formatted message.

        Args:
            status: The ``mr_status`` value returned by the library.
            message: Human-readable description (becomes ``str(exc)``).
        """
        super().__init__(message)
        self.status = status


# --------------------------------------------------------------------------
# Library discovery
# --------------------------------------------------------------------------

# Platform-specific file name of the C-ABI shared library produced by the
# CMake target ``mr_aha_c``: a DLL on Windows, a dylib on macOS, and a
# ``.so`` on Linux and every other platform.
_LIB_BASENAMES = {
    "win32": ["mr_aha_c.dll"],
    "darwin": ["libmr_aha_c.dylib"],
}.get(sys.platform, ["libmr_aha_c.so"])


def _candidate_paths():
    """Yield plausible locations of the MR_AHA shared library.

    The search order is: ``$MR_AHA_LIBRARY``, the installed package
    directory, the in-tree CMake build directories, and finally the platform
    loader's own search path.

    The bare file names (no directory) are handed to the platform loader as
    is, so ``LD_LIBRARY_PATH`` / ``DYLD_LIBRARY_PATH`` / ``PATH`` and the
    system library directories are honoured; :func:`ctypes.util.find_library`
    is tried last as a further fallback.

    Yields:
        ``str`` or :class:`pathlib.Path` candidates, most specific first.
    """
    # 1. Explicit override: a full path to the library file.
    override = os.environ.get("MR_AHA_LIBRARY")
    if override:
        yield override

    # 2. Directories relative to this file: the package directory itself
    #    (library copied next to the Python sources by an install), then the
    #    source tree layout python/mr_aha/ -> <repo>/build[/lib] of an
    #    in-tree CMake build (`pip install -e python` development setup).
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

    # 3. Bare names: resolved by the platform's dynamic loader search path.
    for name in _LIB_BASENAMES:
        yield name
    # 4. ctypes' own platform-specific lookup (ldconfig cache, etc.).
    found = find_library("mr_aha_c")
    if found:
        yield found


def _load_library() -> ctypes.CDLL:
    """Load ``libmr_aha_c`` from the first candidate that works.

    Candidates come from :func:`_candidate_paths`; a candidate that does not
    exist or fails to load (missing CUDA runtime, wrong architecture, ...)
    is skipped silently and recorded for the error message.

    Returns:
        The loaded :class:`ctypes.CDLL`.

    Raises:
        OSError: If no candidate could be loaded, listing what was tried.
    """
    # Try every candidate in order, remembering each one for the diagnostic.
    tried = []
    for cand in _candidate_paths():
        cand = str(cand)
        tried.append(cand)
        try:
            return ctypes.CDLL(cand)
        except OSError:
            continue
    # Nothing loaded: tell the user how to build the library and where we
    # looked. (The individual loader errors are not kept.)
    raise OSError(
        "Could not load the MR_AHA shared library. Build it with\n"
        "    cmake -S . -B build -DMR_AHA_BUILD_PYTHON=ON && cmake --build build\n"
        "and point $MR_AHA_LIBRARY at the result if it lives elsewhere.\n"
        "Tried:\n  " + "\n  ".join(tried)
    )


#: The loaded shared library. Loaded once, at import time of this module.
lib = _load_library()

# --------------------------------------------------------------------------
# Prototypes
# --------------------------------------------------------------------------
# Each block below transcribes one group of declarations of mr_aha.h. All
# entry points return ``int`` (an mr_status) unless noted. Array arguments
# are typed ``c_void_p`` so that a Python int address (CuPy device pointer or
# NumPy host address) or ``None`` (NULL) can be passed directly; the scalar
# ``int64_t`` arguments are ``c_int64`` so they are never truncated.

# Shorthands: an untyped array pointer, and ``const mr_config *``.
_c_ptr = ctypes.c_void_p
_cfg_p = ctypes.POINTER(MRConfig)

# --- Error reporting -------------------------------------------------------
# const char *mr_status_string(int status): static description of a code.
lib.mr_status_string.restype = ctypes.c_char_p
lib.mr_status_string.argtypes = [ctypes.c_int]

# const char *mr_last_error(void): thread-local detail of the last failure.
lib.mr_last_error.restype = ctypes.c_char_p
lib.mr_last_error.argtypes = []

# const char *mr_backend(void): "cuda" or "metal". Guarded because libraries
# built before the backend split do not export it (see backend()).
if hasattr(lib, "mr_backend"):
    lib.mr_backend.restype = ctypes.c_char_p
    lib.mr_backend.argtypes = []

# --- Configuration ---------------------------------------------------------
# int mr_config_defaults(mr_config *cfg, mr_traj_type traj_type)
lib.mr_config_defaults.restype = ctypes.c_int
lib.mr_config_defaults.argtypes = [_cfg_p, ctypes.c_int]

# int mr_config_validate(const mr_config *cfg)
lib.mr_config_validate.restype = ctypes.c_int
lib.mr_config_validate.argtypes = [_cfg_p]

# --- Derived-size helpers (return element counts, not status) -------------
# int64_t mr_num_tri(int L) = L*(L+1)/2
lib.mr_num_tri.restype = ctypes.c_int64
lib.mr_num_tri.argtypes = [ctypes.c_int]

# int64_t mr_<x>_size(const mr_config *cfg): image (Nx*Ny*Nz), padded grid
# (Mx*My*Mz), phi2_toep, x (Nx*Ny*Nz*L) and SEs (Nx*Ny*Nz*Ncoils) element
# counts. All share one signature, so they are declared in a loop.
for _name in ("mr_image_size", "mr_pad_size", "mr_phi2_toep_size", "mr_x_size",
              "mr_se_size"):
    _fn = getattr(lib, _name)
    _fn.restype = ctypes.c_int64
    _fn.argtypes = [_cfg_p]

# --- Single-GPU Toeplitz plan (stateful primitive) -------------------------
# Declared for completeness of the C API; mr_aha.api itself goes through
# mr_aha_toeplitz (one-shot) and the mr_op_* operator object instead.
# int mr_aha_plan_create(const mr_config *cfg, mr_aha_plan *plan_out):
# the opaque plan handle is returned through a void ** out-parameter.
lib.mr_aha_plan_create.restype = ctypes.c_int
lib.mr_aha_plan_create.argtypes = [_cfg_p, ctypes.POINTER(ctypes.c_void_p)]

# int mr_aha_plan_execute(plan, d_x, d_SEs, d_phi2_toep, d_mask, d_AhAx)
lib.mr_aha_plan_execute.restype = ctypes.c_int
lib.mr_aha_plan_execute.argtypes = [
    ctypes.c_void_p, _c_ptr, _c_ptr, _c_ptr, _c_ptr, _c_ptr]

# int mr_aha_plan_destroy(mr_aha_plan plan); NULL is a no-op.
lib.mr_aha_plan_destroy.restype = ctypes.c_int
lib.mr_aha_plan_destroy.argtypes = [ctypes.c_void_p]

# int mr_aha_toeplitz(cfg, d_x, d_SEs, d_phi2_toep, d_mask, d_AhAx):
# one-shot create + execute + destroy (used by mr_aha.api.aha_toeplitz).
lib.mr_aha_toeplitz.restype = ctypes.c_int
lib.mr_aha_toeplitz.argtypes = [_cfg_p, _c_ptr, _c_ptr, _c_ptr, _c_ptr, _c_ptr]

# --- Operator object (mr_op) -----------------------------------------------
# Present in every backend that implements the operator object; guarded so
# that an older library without it still imports (AhAOp then fails on use).
if hasattr(lib, "mr_op_create"):  # every backend that has the operator object
    # int mr_op_options_defaults(mr_op_options *opt)
    lib.mr_op_options_defaults.restype = ctypes.c_int
    lib.mr_op_options_defaults.argtypes = [ctypes.POINTER(MROpOptions)]
    # int mr_op_create(cfg, const mr_op_inputs *in, const mr_op_options *opt,
    #                  mr_op *op_out)
    lib.mr_op_create.restype = ctypes.c_int
    lib.mr_op_create.argtypes = [
        _cfg_p, ctypes.POINTER(MROpInputs), ctypes.POINTER(MROpOptions),
        ctypes.POINTER(ctypes.c_void_p)]
    # int mr_op_execute(mr_op op, const mr_complex *d_x, mr_complex *d_AhAx)
    lib.mr_op_execute.restype = ctypes.c_int
    lib.mr_op_execute.argtypes = [ctypes.c_void_p, _c_ptr, _c_ptr]
    # int mr_op_info(mr_op op, mr_op_info_t *info)
    lib.mr_op_info.restype = ctypes.c_int
    lib.mr_op_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(MROpInfo)]
    # int mr_op_kernel(mr_op op, int k, const mr_complex **d_phi2_toep):
    # returns a borrowed pointer to device k's copy of the kernel.
    lib.mr_op_kernel.restype = ctypes.c_int
    lib.mr_op_kernel.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
    # int mr_op_destroy(mr_op op); NULL is a no-op.
    lib.mr_op_destroy.restype = ctypes.c_int
    lib.mr_op_destroy.argtypes = [ctypes.c_void_p]
    # int mr_op_abandon(mr_op op): release after the primary device was reset
    # (declared for completeness; not used by mr_aha.api).
    lib.mr_op_abandon.restype = ctypes.c_int
    lib.mr_op_abandon.argtypes = [ctypes.c_void_p]

# --- Toeplitz kernel pre-computation ---------------------------------------
# int mr_calc_toep(cfg, d_phi2, d_traj, int64 Ntrajs, int64 Nkx, d_mask,
#                  d_phi2_toep): one-shot kernel (type-1 cufinufft inside).
lib.mr_calc_toep.restype = ctypes.c_int
lib.mr_calc_toep.argtypes = [
    _cfg_p, _c_ptr, _c_ptr, ctypes.c_int64, ctypes.c_int64, _c_ptr, _c_ptr]

# int mr_calc_toep_from_psf(cfg, d_psf, int64 t0, int64 n_comp, d_mask,
#                           d_phi2_toep): second stage of the two-stage path
# of mr_aha._psf. Guarded because older libraries lack it.
if hasattr(lib, "mr_calc_toep_from_psf"):
    lib.mr_calc_toep_from_psf.restype = ctypes.c_int
    lib.mr_calc_toep_from_psf.argtypes = [
        _cfg_p, _c_ptr, ctypes.c_int64, ctypes.c_int64, _c_ptr, _c_ptr]

# --- Automatic k-space mask ------------------------------------------------
# int mr_calc_mask(cfg, d_traj, int64 Ntrajs, int64 Nkx, int64 *d_mask,
#                  int64 *num_masked_out): the count is written to a *host*
# int64, hence the typed POINTER(c_int64) last argument.
lib.mr_calc_mask.restype = ctypes.c_int
lib.mr_calc_mask.argtypes = [
    _cfg_p, _c_ptr, ctypes.c_int64, ctypes.c_int64, _c_ptr,
    ctypes.POINTER(ctypes.c_int64)]

#: Default neighbourhood radius of :func:`mr_aha.calc_mask` (in upsampled grid
#: cells), mirroring ``MR_DEFAULT_MASK_RADIUS``.
DEFAULT_MASK_RADIUS = 2


def backend() -> str:
    """Name of the GPU backend the loaded library was built for.

    The answer decides the array type of the whole Python API: see
    :func:`mr_aha.api._xp`.

    Returns:
        ``'cuda'`` (NVIDIA GPUs; arrays are CuPy device arrays) or
        ``'metal'`` (Apple silicon; arrays are NumPy arrays in unified
        memory). Libraries built before the backend split report ``'cuda'``.
    """
    # Older libraries do not export mr_backend; they were CUDA-only.
    if not hasattr(lib, "mr_backend"):
        return "cuda"
    # The C function returns a static ASCII string ("cuda" / "metal").
    return lib.mr_backend().decode("ascii")


def check(status: int, what: str) -> None:
    """Raise :class:`MRError` when ``status`` reports a failure.

    Must be called immediately after the failing call and on the same thread,
    because ``mr_last_error`` is thread-local and overwritten by the next
    failure.

    Args:
        status: Value returned by an MR_AHA entry point.
        what: Short description of the call, used in the message.

    Raises:
        MRError: If ``status`` is non-zero.
    """
    # MR_SUCCESS == 0: nothing to report.
    if status == 0:
        return
    # Combine the generic text of the code with the library's thread-local
    # detail message; undecodable bytes are replaced rather than raising.
    summary = lib.mr_status_string(status).decode("utf-8", "replace")
    detail = lib.mr_last_error().decode("utf-8", "replace")
    raise MRError(status, f"{what}: {summary} ({status}): {detail}")
