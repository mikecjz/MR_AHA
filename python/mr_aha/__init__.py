"""MR_AHA -- GPU forward normal operator for low-rank non-Cartesian MRI.

The package exposes two operations, mirroring the MATLAB interface:

``calc_toep``
    Pre-computes the block-diagonal Toeplitz operator from the non-Cartesian
    Gram blocks and the trajectory.

``calc_mask`` / ``calc_mask_kernel``
    Derive the masked-operator sample set from the trajectory alone.

``aha_toeplitz`` / ``AhAPlan``
    Applies :math:`A^H A` to a set of spatial coefficients using that operator.

``aha_nufft`` / ``AhANufftPlan``
    Reference :math:`A^H A` evaluated directly with type-2 / type-1 cufinufft
    transforms (no Toeplitz embedding), for validation and benchmarking.

Every array is a CuPy array resident on the GPU. Because the project documents
arrays in column-major (MATLAB) order while NumPy and CuPy are row-major, all
shapes are reversed here; see :mod:`mr_aha.api` for the full table.

Example:
    >>> import cupy, mr_aha
    >>> cfg = mr_aha.make_config('stack', (256, 256, 64), L=16, n_coils=8)
    >>> Om = mr_aha.calc_toep(cfg, phi2, traj)          # doctest: +SKIP
    >>> with mr_aha.AhAPlan(cfg) as plan:               # doctest: +SKIP
    ...     y = plan.apply(x, SEs, Om)
"""

from .api import (
    DEFAULT_MASK_RADIUS,
    DEFAULT_MASK_THRESHOLD,
    AhAPlan,
    MaskMethod,
    MRConfig,
    MRError,
    TrajType,
    Triangulation,
    aha_toeplitz,
    calc_mask,
    calc_mask_kernel,
    calc_toep,
    make_config,
    mask_kernel_shape,
    num_tri,
    phi2_toep_shape,
)
from .nufft import AhANufftPlan, aha_nufft

__all__ = [
    "DEFAULT_MASK_RADIUS",
    "DEFAULT_MASK_THRESHOLD",
    "AhANufftPlan",
    "AhAPlan",
    "MaskMethod",
    "MRConfig",
    "MRError",
    "TrajType",
    "Triangulation",
    "aha_nufft",
    "aha_toeplitz",
    "calc_mask",
    "calc_mask_kernel",
    "calc_toep",
    "make_config",
    "mask_kernel_shape",
    "num_tri",
    "phi2_toep_shape",
]

__version__ = "1.0.0"
