"""MR_AHA -- GPU forward normal operator for low-rank non-Cartesian MRI.

The package is built around one object, mirroring the MATLAB interface:

``create_op`` / ``AhAOp``
    The :math:`A^H A` operator, created from the Gram blocks, the trajectory
    and the sensitivity maps. ``method='toeplitz'`` (default) pre-computes the
    Toeplitz kernel at creation; ``method='nufft'`` applies type-2 / type-1
    NUFFTs per coil instead. ``devices=`` makes it multi-GPU on one node
    (coils split over the GPUs). ``op(x)`` applies it.

Low-level, stateless, single-GPU primitives:

``calc_toep``
    Pre-computes the block-diagonal Toeplitz kernel from the non-Cartesian
    Gram blocks and the trajectory.

``aha_toeplitz``
    Applies :math:`A^H A` once with a given kernel.

``calc_mask``
    Derives the masked-operator sample set from the trajectory alone.

Independent reference (no library code):

``aha_nufft`` / ``aha_nufft_plan``
    :math:`A^H A` evaluated with the cufinufft Python package and CuPy (MLX on
    Metal), used by the tests to validate both methods of the operator.

Every array is a CuPy array resident on the GPU (NumPy on the Metal backend).
Because the project documents arrays in column-major (MATLAB) order while
NumPy and CuPy are row-major, all shapes are reversed here; see
:mod:`mr_aha.api` for the full table.

Package layout
--------------
:mod:`mr_aha._lib`
    The raw ``ctypes`` binding: locates and loads ``libmr_aha_c`` (at import
    time, so ``import mr_aha`` fails with :class:`OSError` when the shared
    library cannot be found), mirrors the C structs/enums of
    ``include/mr_aha/mr_aha.h`` and declares every prototype.
:mod:`mr_aha.api`
    The array-aware layer: shape/dtype/contiguity validation, conversion of
    arrays to raw pointers, CuPy (CUDA) vs NumPy (Metal) dispatch, and the
    :class:`AhAOp` object that owns an ``mr_op`` handle.
:mod:`mr_aha._psf`
    The first stage of the two-stage ``calc_toep`` used on Metal (type-1
    NUFFT of the Gram blocks in Python, then ``mr_calc_toep_from_psf``).
:mod:`mr_aha.nufft`
    The library-independent NUFFT reference operator.

Importing this package does not import CuPy, cufinufft or MLX; those are
imported lazily by the functions that need them.

Example:
    >>> import cupy, mr_aha
    >>> cfg = mr_aha.make_config('stack', (256, 256, 64), L=16, n_coils=8)
    >>> with mr_aha.create_op(cfg, SEs, phi2, traj) as op:   # doctest: +SKIP
    ...     y = op(x)
"""

# Public, array-aware API (validation, pointer marshaling, operator object).
# Importing .api imports ._lib, which loads the shared library immediately.
from .api import (
    DEFAULT_MASK_RADIUS,
    AhAOp,
    MRConfig,
    MRError,
    TrajType,
    Triangulation,
    aha_toeplitz,
    calc_mask,
    calc_toep,
    create_op,
    make_config,
    num_tri,
    phi2_toep_shape,
)
# Runtime query of the backend ('cuda' or 'metal') the loaded library was
# built for; decides whether arrays are CuPy or NumPy.
from ._lib import backend
# Independent NUFFT reference operator (no library numerics involved). The
# module itself imports no GPU package at import time.
from .nufft import aha_nufft_plan, aha_nufft

# Names exported by ``from mr_aha import *``. The lower-level enums of
# ``_lib`` (``OpMethod``, ``ToepDist``) are deliberately not re-exported: the
# API takes them as strings ('toeplitz'/'nufft', 'auto'/'split'/...).
__all__ = [
    "DEFAULT_MASK_RADIUS",
    "aha_nufft_plan",
    "AhAOp",
    "MRConfig",
    "MRError",
    "TrajType",
    "Triangulation",
    "aha_nufft",
    "aha_toeplitz",
    "backend",
    "calc_mask",
    "calc_toep",
    "create_op",
    "make_config",
    "num_tri",
    "phi2_toep_shape",
]

#: Version of the Python package (currently equal to the CMake ``project``
#: version and to ``python/pyproject.toml``; update all three together).
__version__ = "1.0.0"
