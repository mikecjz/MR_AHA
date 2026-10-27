"""Reference :math:`A^H A` evaluated directly with cufinufft (no Toeplitz).

This module implements the "vanilla" forward normal operator of low-rank
non-Cartesian MRI,

.. math::

    A^H A x \\;=\\; \\sum_c S_c^H \\, F_{+}\\, \\mathcal{B}\\, F_{-}\\, S_c\\, x ,

where :math:`F_{-}` is a type-2 NUFFT (uniform image to non-uniform k-space,
sign :math:`-1`), :math:`F_{+} = F_{-}^H` is the matching type-1 NUFFT
(non-uniform k-space to uniform image, sign :math:`+1`) and
:math:`\\mathcal{B}` multiplies the length-:math:`L` vector of every
non-uniform sample by the Gram block :math:`\\Phi^H\\Phi` of its readout. Both
NUFFTs are unnormalised, which is precisely the convention the Toeplitz kernel
of :func:`mr_aha.calc_toep` reproduces, so ``AhANufftPlan.apply`` and
``AhAPlan.apply`` agree to within the NUFFT tolerance.

No C/CUDA code is involved: the NUFFTs come from the ``cufinufft`` Python
package and everything else is CuPy, so this path doubles as an independent
check of the Toeplitz library and as the baseline of the benchmarks.

For a stack-of-stars trajectory, kz is Cartesian: the image is first Fourier
transformed along z (centred, unnormalised), every kz plane then goes through
the same 2D NUFFT pair, and the Gram block depends on both the readout and
the kz encode. The z transform pair is unnormalised in both directions, again
matching the Toeplitz implementation.

The centring of the z transform costs no data movement. With
:math:`c = \\lfloor N_z/2 \\rfloor`, the centred DFT is

.. math::

    X_c[k] = e^{-2\\pi i c^2/N_z}\\, e^{2\\pi i k c/N_z}\\,
             \\mathrm{FFT}\\big(e^{2\\pi i c n/N_z} x[n]\\big)[k],

i.e. a plain FFT of the image modulated by :math:`e^{2\\pi i c n/N_z}`,
followed by a phase that depends only on :math:`k`. The adjoint applies the
conjugate phases. Because the Gram multiplication is diagonal in kz, the
:math:`k`-dependent phases (and the constant) of the forward and adjoint
transforms cancel exactly, so only the image-domain modulation remains, and it
is folded into the coil sensitivity.

Array layout
------------
Shapes are row-major, i.e. the column-major shapes of the project
documentation reversed (see :mod:`mr_aha.api`):

====================  ==================================================
argument              row-major shape (2D / 3D / stack)
====================  ==================================================
``x``                 ``(L, Ny, Nx)`` / ``(L, Nz, Ny, Nx)``
``SEs``               ``(Nc, Ny, Nx)`` / ``(Nc, Nz, Ny, Nx)``
``phi2``              ``(L, L, Ntrajs)`` / same / ``(L, L, Nkz, Ntrajs)``
``traj``              ``(ndim, Nkx, Ntrajs)``
====================  ==================================================
"""

from __future__ import annotations

import math
from typing import Optional

from ._lib import MRConfig, TrajType

__all__ = ["AhANufftPlan", "aha_nufft"]


def _modules():
    """Import CuPy and cufinufft lazily.

    Returns:
        Tuple ``(cupy, cufinufft)``.

    Raises:
        ImportError: If either package is missing.
    """
    try:
        import cupy  # noqa: PLC0415
        import cufinufft  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "mr_aha.nufft needs CuPy and the cufinufft Python package, e.g. "
            "`pip install cupy-cuda12x cufinufft`."
        ) from exc
    return cupy, cufinufft


class AhANufftPlan:
    """Reusable NUFFT-based :math:`A^H A` operator.

    The constructor builds one type-2 and one type-1 cufinufft plan for the
    trajectory and sorts the non-uniform points once, so that
    :meth:`apply` can be called many times (e.g. inside conjugate gradient)
    without any planning cost. Each plan transforms all :math:`L` ranks (and,
    for a stack, all kz planes) of one coil in a single batched call.

    Args:
        cfg: Configuration from :func:`mr_aha.make_config`. Uses
            ``traj_type``, ``n_dim``, ``L``, ``n_coils`` and ``eps``;
            ``phi2_dim`` and the masking fields are irrelevant here.
        traj: ``float32`` CuPy array ``(ndim, Nkx, Ntrajs)`` in radians per
            unit of the image grid (``[-pi, pi)``), the same array that is
            passed to :func:`mr_aha.calc_toep`.
        eps: NUFFT tolerance. Defaults to ``cfg.eps``.
        type1_opts: cufinufft options for the type-1 (adjoint) plan only,
            e.g. ``{'gpu_method': 1}``; they override ``nufft_opts``.
        type2_opts: Same for the type-2 (forward) plan.
        **nufft_opts: Extra cufinufft options forwarded to both plans, e.g.
            ``gpu_method=2`` or ``upsampfac=1.25`` (default 2). ``modeord``
            is always 0 (centred modes).

    Raises:
        ValueError: If ``traj`` does not match the configuration.
    """

    def __init__(self, cfg: MRConfig, traj, eps: Optional[float] = None,
                 type1_opts: Optional[dict] = None,
                 type2_opts: Optional[dict] = None, **nufft_opts):
        cupy, cufinufft = _modules()
        self._cfg = cfg
        self._tt = int(cfg.traj_type)
        self._L = int(cfg.L)
        n = [int(cfg.n_dim[i]) for i in range(3)]

        ndim_traj = 3 if self._tt == TrajType.THREE_D else 2
        if traj.ndim != 3 or traj.shape[0] != ndim_traj:
            raise ValueError(
                f"traj must be shaped (ndim, Nkx, Ntrajs) with ndim = {ndim_traj}, "
                f"got {traj.shape}"
            )
        self._Nkx = int(traj.shape[1])
        self._Ntrajs = int(traj.shape[2])
        self._M = self._Nkx * self._Ntrajs

        # Image shape without the rank axis, row-major, and the NUFFT modes.
        if self._tt == TrajType.TWO_D:
            self._img_shape = (n[1], n[0])
            modes = (n[1], n[0])
            self._ntr = self._L
        elif self._tt == TrajType.THREE_D:
            self._img_shape = (n[2], n[1], n[0])
            modes = (n[2], n[1], n[0])
            self._ntr = self._L
        else:  # stack: 2D NUFFT of every kz plane of every rank
            self._img_shape = (n[2], n[1], n[0])
            modes = (n[1], n[0])
            self._ntr = self._L * n[2]
        self._Nz = n[2]

        # cufinufft maps its first point argument onto the first (slowest)
        # array axis, so the coordinates go in reverse order: (y, x) in 2D and
        # (z, y, x) in 3D. Each slice is a contiguous view of traj, and the
        # plans keep references to them.
        pts = [cupy.ascontiguousarray(traj[d].reshape(-1))
               for d in reversed(range(ndim_traj))]

        tol = float(cfg.eps if eps is None else eps)
        # Centred (CMCL) mode order, like the library; upsampfac pinned to 2 so
        # that the MATLAB and Python references run the same transform.
        opts = dict(upsampfac=2.0)
        opts.update(nufft_opts)
        opts["modeord"] = 0
        o2 = dict(opts, **(type2_opts or {}), modeord=0)
        o1 = dict(opts, **(type1_opts or {}), modeord=0)
        self._t2 = cufinufft.Plan(2, modes, n_trans=self._ntr, eps=tol, isign=-1,
                                  dtype="complex64", **o2)
        self._t2.setpts(*pts)
        self._t1 = cufinufft.Plan(1, modes, n_trans=self._ntr, eps=tol, isign=+1,
                                  dtype="complex64", **o1)
        self._t1.setpts(*pts)
        self._pts = pts

        # Stack: image-domain modulation that turns the plain z FFT into the
        # centred one (see the module docstring), shaped to broadcast over a
        # (Nz, Ny, Nx) coil map.
        self._zmod = None
        if self._tt == TrajType.STACK:
            nz = self._Nz
            ph = 2.0 * math.pi * (nz // 2) * cupy.arange(nz) / nz
            self._zmod = cupy.exp(1j * ph).astype(cupy.complex64)[:, None, None]

    @property
    def config(self) -> MRConfig:
        """The configuration this plan was built from."""
        return self._cfg

    def _check_phi2(self, phi2):
        """Validate the Gram blocks and view them with explicit axes.

        Args:
            phi2: ``complex64`` CuPy array ``(L, L, Ntrajs)`` or, for a stack,
                ``(L, L, Nkz, Ntrajs)``; ``phi2[j, i, ...]`` is entry
                ``(i, j)`` of the block.

        Returns:
            ``phi2`` reshaped to exactly that shape (a view, no copy).

        Raises:
            ValueError: If ``phi2`` has the wrong number of elements.
        """
        L = self._L
        nkz = self._Nz if self._tt == TrajType.STACK else 1
        expected = L * L * nkz * self._Ntrajs
        if phi2.size != expected:
            raise ValueError(
                f"phi2 has {phi2.size} elements but {expected} were expected "
                f"(row-major (L, L{', Nkz' if nkz > 1 else ''}, Ntrajs))"
            )
        if nkz > 1:
            return phi2.reshape(L, L, nkz, self._Ntrajs)
        return phi2.reshape(L, L, self._Ntrajs)

    def _blockdiag(self, k, phi2):
        """Apply the per-readout Gram blocks to non-uniform k-space data.

        For every readout ``t`` (and kz encode ``z`` of a stack) this computes
        ``k[i, kx] <- sum_j Phi_t[i, j] k[j, kx]`` with a single
        :func:`cupy.einsum`, which CuPy maps onto a batched GEMM without
        materialising any broadcast intermediate. On the 3D test case this was
        about 1.5x faster than an explicit transpose + :func:`cupy.matmul`
        and 8x faster than a broadcast-and-accumulate loop.

        Args:
            k: Type-2 output ``(ntr, M)`` with ``M = Nkx * Ntrajs`` (trajectory
                index fastest) and ``ntr = L`` or ``L * Nkz``.
            phi2: Output of :meth:`_check_phi2`.

        Returns:
            Contiguous array shaped like ``k`` holding the multiplied data.
        """
        cupy, _ = _modules()
        L, Nkx, Nt = self._L, self._Nkx, self._Ntrajs
        if self._tt == TrajType.STACK:
            kk = k.reshape(L, self._Nz, Nkx, Nt)
            out = cupy.einsum("jizt,jzkt->izkt", phi2, kk)
        else:
            kk = k.reshape(L, Nkx, Nt)
            out = cupy.einsum("jit,jkt->ikt", phi2, kk)
        return cupy.ascontiguousarray(out).reshape(self._ntr, self._M)

    def apply(self, x, SEs, phi2, out=None):
        """Apply :math:`A^H A` to a set of spatial coefficients.

        Args:
            x: ``complex64`` CuPy array ``(L, Ny, Nx)`` (2D) or
                ``(L, Nz, Ny, Nx)`` (3D/stack).
            SEs: ``complex64`` CuPy array of coil maps ``(Nc, Ny, Nx)`` or
                ``(Nc, Nz, Ny, Nx)``; one map per coil, shared by every rank.
            phi2: ``complex64`` Gram blocks ``(L, L, Ntrajs)`` or, for a
                stack, ``(L, L, Nkz, Ntrajs)``.
            out: Optional pre-allocated ``complex64`` output shaped like ``x``.

        Returns:
            The ``complex64`` CuPy array ``AhAx``, shaped like ``x``.

        Raises:
            ValueError: If an array has the wrong size.
        """
        cupy, _ = _modules()
        L = self._L
        img = self._img_shape
        x = x.reshape((L,) + img)
        nc = int(SEs.size) // math.prod(img)
        SEs = SEs.reshape((nc,) + img)
        phi2 = self._check_phi2(phi2)

        if out is None:
            out = cupy.zeros((L,) + img, dtype=cupy.complex64)
        else:
            out = out.reshape((L,) + img)
            out.fill(0)

        is_stack = self._tt == TrajType.STACK
        for c in range(nc):
            # 1. Coil sensitivity, broadcast over the ranks. For a stack it
            #    carries the z modulation that centres the z FFT.
            se = SEs[c] * self._zmod if is_stack else SEs[c]
            im = x * se

            # 2. Stack only: unnormalised FFT along z (centred via the
            #    modulation above; the k-side phases cancel).
            if is_stack:
                im = cupy.fft.fft(im, axis=1)

            # 3. Type-2 NUFFT: uniform image -> non-uniform k-space.
            k = self._t2.execute(im.reshape((self._ntr,) + im.shape[-2:])
                                 if is_stack else im)
            del im

            # 4. Block-diagonal Gram multiplication, one block per readout.
            k = self._blockdiag(k, phi2)

            # 5. Type-1 NUFFT: non-uniform k-space -> uniform image.
            im = self._t1.execute(k)
            del k
            im = im.reshape((L,) + img)

            # 6. Stack only: unnormalised inverse FFT along z.
            if is_stack:
                im = cupy.fft.ifft(im, axis=1, norm="forward")

            # 7. Adjoint coil sensitivity (and z demodulation), accumulated.
            out += im * cupy.conj(se)
            del im, se
        return out

    def close(self) -> None:
        """Destroy the cufinufft plans. Idempotent."""
        # Dropping the last reference runs cufinufft's own destructor.
        self._t1 = None
        self._t2 = None

    def __enter__(self) -> "AhANufftPlan":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def aha_nufft(cfg: MRConfig, x, SEs, phi2, traj, out=None, **nufft_opts):
    """One-shot NUFFT-based :math:`A^H A`.

    Builds an :class:`AhANufftPlan`, applies it once and releases it. Prefer
    the plan inside an iterative loop, since planning sorts the trajectory.

    Args:
        cfg: Configuration from :func:`mr_aha.make_config`.
        x: See :meth:`AhANufftPlan.apply`.
        SEs: See :meth:`AhANufftPlan.apply`.
        phi2: See :meth:`AhANufftPlan.apply`.
        traj: See :class:`AhANufftPlan`.
        out: See :meth:`AhANufftPlan.apply`.
        **nufft_opts: Forwarded to :class:`AhANufftPlan`.

    Returns:
        The ``complex64`` CuPy array ``AhAx``, shaped like ``x``.
    """
    with AhANufftPlan(cfg, traj, **nufft_opts) as plan:
        return plan.apply(x, SEs, phi2, out=out)
