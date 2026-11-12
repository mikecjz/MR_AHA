"""MR_AHA -- GPU forward normal operator for low-rank non-Cartesian MRI.

The package is built around one object, mirroring the MATLAB interface:

``create_op`` / ``AhAOp``
    The :math:`A^H A` operator, created from the Gram blocks, the trajectory
    and the sensitivity maps. ``method='toeplitz'`` (default) pre-computes the
    Toeplitz kernel at creation; ``method='nufft'`` applies type-2 / type-1
    NUFFTs per coil instead. ``devices=`` puts it on several GPUs of one node
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

``aha_nufft`` / ``AhANufftPlan``
    :math:`A^H A` evaluated with the cufinufft Python package and CuPy (MLX on
    Metal), used by the tests to validate both methods of the operator.

Every array is a CuPy array resident on the GPU (NumPy on the Metal backend).
Because the project documents arrays in column-major (MATLAB) order while
NumPy and CuPy are row-major, all shapes are reversed here; see
:mod:`mr_aha.api` for the full table.

Example:
    >>> import cupy, mr_aha
    >>> cfg = mr_aha.make_config('stack', (256, 256, 64), L=16, n_coils=8)
    >>> with mr_aha.create_op(cfg, SEs, phi2, traj) as op:   # doctest: +SKIP
    ...     y = op(x)
"""

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
from ._lib import backend
from .nufft import AhANufftPlan, aha_nufft

__all__ = [
    "DEFAULT_MASK_RADIUS",
    "AhANufftPlan",
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

__version__ = "1.0.0"
