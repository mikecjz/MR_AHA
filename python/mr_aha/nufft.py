"""Reference :math:`A^H A` evaluated directly with cufinufft (no Toeplitz).

This module implements the "vanilla" forward normal operator of low-rank
non-Cartesian MRI,

.. math::

    A^H A x \\;=\\; \\sum_c S_c^H \\, F_{+}\\, \\mathcal{B}\\, F_{-}\\, S_c\\, x ,

where :math:`F_{-}` is a type-2 NUFFT (uniform image to non-uniform k-space,
sign :math:`-1`), :math:`F_{+} = F_{-}^H` is the matching type-1 NUFFT
(non-uniform k-space to uniform image, sign :math:`+1`) and
:math:`\\mathcal{B}` multiplies the length-:math:`L` rank vector of every
non-uniform sample, as a row vector from the right (MATLAB
``reshape(k, [], L) * Phi2``), by the Gram block
:math:`\\Phi_2 = \\Phi\\Phi^H` of its readout, :math:`\\Phi` being the
rank x time temporal basis (see :mod:`mr_aha.api`). Both
NUFFTs are unnormalised, which is precisely the convention the Toeplitz kernel
of :func:`mr_aha.calc_toep` reproduces, so ``aha_nufft_plan.apply`` and the
operator of :func:`mr_aha.create_op` (either method) agree to within the
NUFFT tolerance.

This is the **independent reference**: no C/CUDA/Metal code of the library is
involved (it uses the cufinufft Python package and CuPy), so the tests use it
to validate both methods of :func:`mr_aha.create_op`, in particular
``method='nufft'``, which runs the same algorithm inside the library (in C,
single- or multi-GPU). For reconstructions use :func:`mr_aha.create_op`;
this module is kept for validation and as the baseline of the
benchmarks. The engine follows the loaded backend (:func:`mr_aha.backend`):

* **CUDA**: ``cufinufft`` plans, everything else in CuPy; arrays are CuPy.
* **Metal**: `mlx-nufft <https://github.com/martinlachaine/mlx-nufft>`_ plans
  (the ``third_party/mlx-nufft`` submodule), everything else in MLX on the
  Apple GPU; arrays are NumPy. mlx-nufft's type-2 plan transforms one vector
  per call, so the type-2 step loops over the ``L`` ranks (and kz planes); the
  type-1 step is one batched call.

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

The number of coils is taken from ``SEs.size`` at every :meth:`apply`
call, not from ``cfg.n_coils``.

Per-coil pipeline of :meth:`aha_nufft_plan.apply` (``ntr`` transforms per
NUFFT call: ``L``, or ``L * Nz`` for a stack)::

    x (L, *img) --* S_c [* zmod]--> --FFT_z (stack)--> --type-2--> k (ntr, M)
    k --Gram blocks--> --type-1--> --IFFT_z * Nz (stack)--> --* conj(S_c ...)-->
    accumulate into out

with ``M = Nkx * Ntrajs`` non-uniform samples per transform.
"""

from __future__ import annotations

import math
from typing import Optional

# Only the configuration mirror and the backend query are needed from the
# library binding; no library numerics are called by this module.
from ._lib import MRConfig, TrajType, backend

__all__ = ["aha_nufft_plan", "aha_nufft"]


class _MlxNufft:
    """Metal implementation of :class:`aha_nufft_plan` (MLX + mlx-nufft).

    Same algorithm, conventions and array layout as the CuPy/cufinufft
    implementation in :class:`aha_nufft_plan`; see that class and the module
    docstring. Inputs and outputs are NumPy arrays; the work runs on the GPU
    as MLX arrays.

    Differences from the CUDA implementation: mlx-nufft's type-2 plan takes
    one vector per call, so the type-2 step loops over the ``ntr`` transforms;
    the upsampling factor is fixed at 2 and no cufinufft options apply; the
    tolerance is floored at ``1e-6``; and the type-2 output is copied back to
    NumPy before the batched type-1 call. :class:`mr_aha.AhAOp` also uses
    this class directly for ``method='nufft'`` on Metal.

    Args:
        cfg: Configuration (uses ``traj_type``, ``n_dim`` and ``L``).
        traj: ``float32``/``float64`` array-like ``(ndim, Nkx, Ntrajs)`` in
            radians.
        eps: NUFFT tolerance (floored at ``1e-6``).

    Raises:
        ImportError: If MLX or mlx-nufft is missing.
        ValueError: If ``traj`` is not shaped ``(ndim, Nkx, Ntrajs)``.
    """

    def __init__(self, cfg: MRConfig, traj, eps: float):
        """Validate the trajectory and build both mlx-nufft plans. See the
        class docstring for the arguments."""
        # --- Lazy imports: MLX and mlx-nufft exist only on Apple silicon. ---
        import numpy as np  # noqa: PLC0415
        try:
            import mlx.core as mx  # noqa: PLC0415
            from mlx_nufft import Type1PlanND, Type2PlanND  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "mr_aha.nufft on the Metal backend needs mlx and mlx-nufft: "
                "`pip install -e third_party/mlx-nufft`."
            ) from exc
        self._np, self._mx = np, mx
        # --- Geometry from the configuration and trajectory validation. ---
        self._tt = int(cfg.traj_type)
        self._L = int(cfg.L)
        n = [int(cfg.n_dim[i]) for i in range(3)]
        ndim_traj = 3 if self._tt == TrajType.THREE_D else 2
        traj = np.asarray(traj)
        if traj.ndim != 3 or traj.shape[0] != ndim_traj:
            raise ValueError(
                f"traj must be shaped (ndim, Nkx, Ntrajs) with ndim = {ndim_traj}, "
                f"got {traj.shape}"
            )
        self._Nkx, self._Ntrajs = int(traj.shape[1]), int(traj.shape[2])
        self._M = self._Nkx * self._Ntrajs
        self._Nz = n[2]
        # Row-major image shape (no rank axis), NUFFT modes and the number of
        # vectors per coil: 2D and 3D transform the whole image (L vectors);
        # a stack transforms each of the Nz kz planes in 2D (L * Nz vectors).
        if self._tt == TrajType.TWO_D:
            self._img_shape, self._modes, self._ntr = (n[1], n[0]), (n[1], n[0]), self._L
        elif self._tt == TrajType.THREE_D:
            self._img_shape = self._modes = (n[2], n[1], n[0])
            self._ntr = self._L
        else:
            self._img_shape, self._modes = (n[2], n[1], n[0]), (n[1], n[0])
            self._ntr = self._L * n[2]
        # Coordinates reversed so that the C-ordered modes have x fastest.
        coords = tuple(np.ascontiguousarray(traj[d].ravel(), dtype=np.float64)
                       for d in reversed(range(ndim_traj)))
        # --- NUFFT plans: type-2 forward (sign -1) and type-1 adjoint
        #     (sign +1) over the same points, centred modes, sigma = 2. ---
        tol = max(float(eps), 1e-6)  # mlx-nufft's fp32 floor
        self._t2 = Type2PlanND(coords, self._modes, eps=tol, isign=-1,
                               upsampfac=2.0)
        self._t1 = Type1PlanND(coords, self._modes, eps=tol, isign=+1,
                               upsampfac=2.0)
        # --- Stack only: image-domain modulation exp(2 pi i c n / Nz),
        #     c = Nz // 2, that turns the plain z FFT into the centred one
        #     (see the module docstring); shaped (Nz, 1, 1) to broadcast over
        #     a (Nz, Ny, Nx) map. ---
        self._zmod = None
        if self._tt == TrajType.STACK:
            nz = self._Nz
            ph = 2.0 * math.pi * (nz // 2) * np.arange(nz) / nz
            self._zmod = mx.array(np.exp(1j * ph).astype(np.complex64)[:, None, None])

    def apply(self, x, SEs, phi2, out=None):
        """See :meth:`aha_nufft_plan.apply` (NumPy arrays in and out).

        Args:
            x: ``complex64`` array-like ``(L, *img)`` (any shape with that
                many elements; it is reshaped).
            SEs: ``complex64`` coil maps; the coil count is
                ``SEs.size // prod(img)``.
            phi2: Gram blocks ``(L, L[, Nkz], Ntrajs)``.
            out: Optional NumPy output; the result is written into
                ``out.reshape(res.shape)`` and ``out`` is returned.

        Returns:
            ``complex64`` NumPy array ``(L, *img)`` (or ``out``).

        Raises:
            ValueError: If ``phi2`` has the wrong number of elements.
        """
        np, mx = self._np, self._mx
        L, img = self._L, self._img_shape
        stack = self._tt == TrajType.STACK
        nkz = self._Nz if stack else 1
        # --- Convert the NumPy inputs to MLX arrays with explicit shapes
        #     (complex64, contiguous), validating the Gram-block size. ---
        x = mx.array(np.ascontiguousarray(x, dtype=np.complex64).reshape((L,) + img))
        nc = int(np.asarray(SEs).size) // math.prod(img)
        SEs = mx.array(np.ascontiguousarray(SEs, dtype=np.complex64)
                       .reshape((nc,) + img))
        expected = L * L * nkz * self._Ntrajs
        if np.asarray(phi2).size != expected:
            raise ValueError(f"phi2 has {np.asarray(phi2).size} elements but "
                             f"{expected} were expected")
        ph = mx.array(np.ascontiguousarray(phi2, dtype=np.complex64).reshape(
            (L, L, nkz, self._Ntrajs) if stack else (L, L, self._Ntrajs)))

        # --- Coil loop: A^H A x = sum_c S_c^H F+ B F- S_c x, accumulated. ---
        acc = mx.zeros((L,) + img, dtype=mx.complex64)
        for c in range(nc):
            # Coil map (with the z modulation for a stack), broadcast over L.
            se = SEs[c] * self._zmod if stack else SEs[c]
            im = x * se
            if stack:
                im = mx.fft.fft(im, axis=1)          # unnormalised, centred via zmod
            im = im.reshape((self._ntr,) + self._modes)
            # Type-2, one vector per call.
            k = mx.stack([self._t2.execute(im[i], return_np=False)
                          for i in range(self._ntr)])
            del im
            # Block-diagonal Gram multiplication, one block per readout.
            if stack:
                kk = k.reshape(L, self._Nz, self._Nkx, self._Ntrajs)
                k = mx.einsum("ijzt,jzkt->izkt", ph, kk)
            else:
                kk = k.reshape(L, self._Nkx, self._Ntrajs)
                k = mx.einsum("ijt,jkt->ikt", ph, kk)
            k = k.reshape(self._ntr, self._M)
            # Force MLX's lazy graph to run before handing k to NumPy.
            mx.eval(k)
            # Type-1, all vectors in one batched call.
            im = self._t1.execute_batch(np.asarray(k), return_np=False)
            del k, kk
            im = im.reshape((L,) + img)
            if stack:
                # Unnormalised inverse along z (MLX's ifft divides by Nz).
                im = mx.fft.ifft(im, axis=1) * float(self._Nz)
            # Adjoint coil map (and z demodulation), accumulated; evaluating
            # here keeps the lazy graph from growing across coils.
            acc = acc + im * mx.conj(se)
            mx.eval(acc)
        # --- Back to NumPy; honour a caller-supplied output buffer. ---
        res = np.array(acc)
        if out is not None:
            out.reshape(res.shape)[...] = res
            return out
        return res

    def close(self) -> None:
        """Drop the mlx-nufft plans (their memory is freed with the last
        reference). The object cannot be used afterwards."""
        self._t1 = self._t2 = None


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


# Lowercase class name kept for parity with the MATLAB reference class.
class aha_nufft_plan:  # noqa: N801  (same name as the MATLAB class)
    """Reusable NUFFT-based :math:`A^H A` operator.

    The constructor builds one type-2 and one type-1 cufinufft plan for the
    trajectory and sorts the non-uniform points once, so that
    :meth:`apply` can be called many times (e.g. inside conjugate gradient)
    without any planning cost. :meth:`execute` is an alias of :meth:`apply`.
    Each plan transforms all :math:`L` ranks (and, for a stack, all kz
    planes) of one coil in a single batched call.

    Args:
        cfg: Configuration from :func:`mr_aha.make_config`. Uses
            ``traj_type``, ``n_dim``, ``L``, ``n_coils`` and ``eps``;
            ``phi2_dim`` and the masking fields are irrelevant here. (As
            implemented, :meth:`apply` infers the coil count from
            ``SEs.size`` rather than reading ``n_coils``.)
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

    On the Metal backend the constructor delegates everything to
    :class:`_MlxNufft` (``type1_opts``, ``type2_opts`` and ``nufft_opts``
    are then ignored) and the arrays are NumPy instead of CuPy.

    Raises:
        ValueError: If ``traj`` does not match the configuration.
        ImportError: If CuPy/cufinufft (CUDA) or MLX/mlx-nufft (Metal) is
            missing.
    """

    def __init__(self, cfg: MRConfig, traj, eps: Optional[float] = None,
                 type1_opts: Optional[dict] = None,
                 type2_opts: Optional[dict] = None, **nufft_opts):
        """Build the plans; see the class docstring for the arguments."""
        self._cfg = cfg
        # _impl is the Metal delegate; None means the CuPy/cufinufft path.
        self._impl = None
        # --- Backend dispatch. ---
        if backend() != "cuda":
            # Metal: MLX + mlx-nufft; the cufinufft options do not apply.
            self._impl = _MlxNufft(cfg, traj, cfg.eps if eps is None else eps)
            return
        # --- CUDA path: lazy imports, then geometry from the config. ---
        cupy, cufinufft = _modules()
        self._tt = int(cfg.traj_type)
        self._L = int(cfg.L)
        n = [int(cfg.n_dim[i]) for i in range(3)]

        # --- Trajectory validation: (ndim, Nkx, Ntrajs), ndim from type. ---
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

        # --- Plan options. Precedence (lowest to highest): upsampfac=2
        #     default, **nufft_opts, type{1,2}_opts; modeord is forced to 0
        #     last, whatever the caller passed. ---
        tol = float(cfg.eps if eps is None else eps)
        # Centred (CMCL) mode order, like the library; upsampfac pinned to 2 so
        # that the MATLAB and Python references run the same transform.
        opts = dict(upsampfac=2.0)
        opts.update(nufft_opts)
        opts["modeord"] = 0
        o2 = dict(opts, **(type2_opts or {}), modeord=0)
        o1 = dict(opts, **(type1_opts or {}), modeord=0)
        # --- Plans: type-2 forward (sign -1) and type-1 adjoint (sign +1),
        #     each batched over ntr vectors; setpts sorts the points once
        #     here so apply() pays no planning cost. ---
        self._t2 = cufinufft.Plan(2, modes, n_trans=self._ntr, eps=tol, isign=-1,
                                  dtype="complex64", **o2)
        self._t2.setpts(*pts)
        self._t1 = cufinufft.Plan(1, modes, n_trans=self._ntr, eps=tol, isign=+1,
                                  dtype="complex64", **o1)
        self._t1.setpts(*pts)
        # Keep the coordinate arrays alive: the plans reference their memory.
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
        # Expected element count from the plan's geometry.
        L = self._L
        nkz = self._Nz if self._tt == TrajType.STACK else 1
        expected = L * L * nkz * self._Ntrajs
        if phi2.size != expected:
            raise ValueError(
                f"phi2 has {phi2.size} elements but {expected} were expected "
                f"(row-major (L, L{', Nkz' if nkz > 1 else ''}, Ntrajs))"
            )
        # Give the blocks explicit axes for the einsum of _blockdiag.
        if nkz > 1:
            return phi2.reshape(L, L, nkz, self._Ntrajs)
        return phi2.reshape(L, L, self._Ntrajs)

    def _blockdiag(self, k, phi2):
        """Apply the per-readout Gram blocks to non-uniform k-space data.

        For every readout ``t`` (and kz encode ``z`` of a stack) this computes
        the product ``k[:, kx] <- k[:, kx] * Phi2_t``, i.e.
        ``k[i, kx] <- sum_j k[j, kx] Phi2_t[j, i]`` (row-major
        ``phi2[i, j, ...]`` is ``Phi2_t[j, i]``), with a single
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
        # Unflatten the sample axis to (Nkx, Ntrajs) (plus kz for a stack) so
        # the block index t lines up with phi2's last axis, then contract
        # over the rank index j: out[i] = sum_j phi2[i, j] * k[j].
        if self._tt == TrajType.STACK:
            kk = k.reshape(L, self._Nz, Nkx, Nt)
            out = cupy.einsum("ijzt,jzkt->izkt", phi2, kk)
        else:
            kk = k.reshape(L, Nkx, Nt)
            out = cupy.einsum("ijt,jkt->ikt", phi2, kk)
        # Back to the flat (ntr, M) layout the type-1 plan consumes.
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
                It is zeroed and then accumulated into; it must not alias
                ``x``.

        Returns:
            The ``complex64`` CuPy array ``AhAx``, shaped like ``x``. When
            ``out`` is given, the returned array is ``out.reshape((L,) +
            img)`` (a view of ``out`` when ``out`` is contiguous).

        Raises:
            ValueError: If an array has the wrong size.
        """
        # --- Metal delegate. ---
        if self._impl is not None:
            return self._impl.apply(x, SEs, phi2, out=out)
        # --- Normalise the inputs to explicit shapes (views, no copies);
        #     the coil count is inferred from SEs.size. ---
        cupy, _ = _modules()
        L = self._L
        img = self._img_shape
        x = x.reshape((L,) + img)
        nc = int(SEs.size) // math.prod(img)
        SEs = SEs.reshape((nc,) + img)
        phi2 = self._check_phi2(phi2)

        # --- Output buffer: allocate, or zero the caller's, since the coil
        #     loop accumulates into it. ---
        if out is None:
            out = cupy.zeros((L,) + img, dtype=cupy.complex64)
        else:
            out = out.reshape((L,) + img)
            out.fill(0)

        # --- Coil loop: A^H A x = sum_c S_c^H F+ B F- S_c x. Temporaries are
        #     deleted eagerly so CuPy's pool can reuse them for the next coil.
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

            # 3. Type-2 NUFFT: uniform image -> non-uniform k-space. A stack's
            #    (L, Nz, Ny, Nx) image is viewed as L*Nz 2D planes.
            k = self._t2.execute(im.reshape((self._ntr,) + im.shape[-2:])
                                 if is_stack else im)
            del im

            # 4. Block-diagonal Gram multiplication, one block per readout.
            k = self._blockdiag(k, phi2)

            # 5. Type-1 NUFFT: non-uniform k-space -> uniform image.
            im = self._t1.execute(k)
            del k
            im = im.reshape((L,) + img)

            # 6. Stack only: unnormalised inverse FFT along z (norm="forward"
            #    puts the 1/Nz on the forward transform, so ifft is unscaled).
            if is_stack:
                im = cupy.fft.ifft(im, axis=1, norm="forward")

            # 7. Adjoint coil sensitivity (and z demodulation), accumulated.
            out += im * cupy.conj(se)
            del im, se
        return out

    def execute(self, x, SEs, phi2, out=None):
        """Alias of :meth:`apply`."""
        return self.apply(x, SEs, phi2, out=out)

    def close(self) -> None:
        """Destroy the NUFFT plans. Idempotent.

        The plan cannot be applied afterwards.
        """
        # Metal: delegate to the MLX implementation.
        if self._impl is not None:
            self._impl.close()
            return
        # Dropping the last reference runs cufinufft's own destructor.
        self._t1 = None
        self._t2 = None

    def __enter__(self) -> "aha_nufft_plan":
        """Context-manager entry: return the plan itself."""
        return self

    def __exit__(self, *exc) -> None:
        """Context-manager exit: release the plans; exceptions propagate."""
        self.close()


def aha_nufft(cfg: MRConfig, x, SEs, phi2, traj, out=None, **nufft_opts):
    """One-shot NUFFT-based :math:`A^H A`.

    Builds an :class:`aha_nufft_plan`, applies it once and releases it. Prefer
    the plan inside an iterative loop, since planning sorts the trajectory.

    Args:
        cfg: Configuration from :func:`mr_aha.make_config`.
        x: See :meth:`aha_nufft_plan.apply`.
        SEs: See :meth:`aha_nufft_plan.apply`.
        phi2: See :meth:`aha_nufft_plan.apply`.
        traj: See :class:`aha_nufft_plan`.
        out: See :meth:`aha_nufft_plan.apply`.
        **nufft_opts: Forwarded to :class:`aha_nufft_plan`.

    Returns:
        The ``complex64`` CuPy array ``AhAx``, shaped like ``x`` (NumPy on
        the Metal backend).
    """
    # Plan, apply once, and release the plans on exit (also on error).
    with aha_nufft_plan(cfg, traj, **nufft_opts) as plan:
        return plan.apply(x, SEs, phi2, out=out)
