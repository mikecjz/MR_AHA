"""MT_CUDA -- GPU forward normal operator for low-rank non-Cartesian MRI.

The package exposes two operations, mirroring the MATLAB interface:

``calc_toep``
    Pre-computes the block-diagonal Toeplitz operator from the non-Cartesian
    Gram blocks and the trajectory.

``aha_toeplitz`` / ``AhAPlan``
    Applies :math:`A^H A` to a set of spatial coefficients using that operator.

Every array is a CuPy array resident on the GPU. Because the project documents
arrays in column-major (MATLAB) order while NumPy and CuPy are row-major, all
shapes are reversed here; see :mod:`mt_cuda.api` for the full table.

Example:
    >>> import cupy, mt_cuda
    >>> cfg = mt_cuda.make_config('stack', (256, 256, 64), L=16, n_coils=8)
    >>> Om = mt_cuda.calc_toep(cfg, phi2, traj)          # doctest: +SKIP
    >>> with mt_cuda.AhAPlan(cfg) as plan:               # doctest: +SKIP
    ...     y = plan.apply(x, SEs, Om)
"""

from .api import (
    AhAPlan,
    MTConfig,
    MTError,
    TrajType,
    Triangulation,
    aha_toeplitz,
    calc_toep,
    make_config,
    num_tri,
    phi2_toep_shape,
)

__all__ = [
    "AhAPlan",
    "MTConfig",
    "MTError",
    "TrajType",
    "Triangulation",
    "aha_toeplitz",
    "calc_toep",
    "make_config",
    "num_tri",
    "phi2_toep_shape",
]

__version__ = "1.0.0"
