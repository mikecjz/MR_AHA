#!/usr/bin/env python
"""Generate a 2D spiral k-space trajectory with mrinufft and save it as a .mat file.

Uses mrinufft.initialize_2D_spiral to build the trajectory. The output trajectory
has shape (Nc, Ns, 2) with coordinates normalized to [-0.5, 0.5].

Used by ``benchmarks/spiral_mask.py`` (two-arm spiral, the defaults)::

    python benchmarks/make_spiral.py -o benchmarks/data/spiral_trajectory.mat

Requires ``pip install mri-nufft``.
"""

import argparse

import numpy as np
from scipy.io import savemat

from mrinufft import initialize_2D_spiral


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Nc", type=int, default=2,
                        help="Number of shots / spiral arms (default: 2)")
    parser.add_argument("--Ns", type=int, default=1024,
                        help="Number of samples per shot (default: 1024)")
    parser.add_argument("--tilt", default="uniform",
                        help="Tilt between consecutive shots (default: uniform)")
    parser.add_argument("--in-out", action="store_true",
                        help="Make the spiral go in-and-out (center-out by default)")
    parser.add_argument("--nb-revolutions", type=float, default=1.0,
                        help="Number of revolutions per spiral arm (default: 1)")
    parser.add_argument("--spiral", default="archimedes",
                        help="Spiral type: archimedes, fermat, galilean, or a float power (default: archimedes)")
    parser.add_argument("-o", "--output", default="spiral_trajectory.mat",
                        help="Output .mat file path (default: spiral_trajectory.mat)")
    args = parser.parse_args()

    # allow a numeric spiral power to be passed on the command line
    try:
        spiral = float(args.spiral)
    except ValueError:
        spiral = args.spiral

    trajectory = initialize_2D_spiral(
        Nc=args.Nc,
        Ns=args.Ns,
        tilt=args.tilt,
        in_out=args.in_out,
        nb_revolutions=args.nb_revolutions,
        spiral=spiral,
    )
    trajectory = np.asarray(trajectory, dtype=np.float32)
    print(f"Generated spiral trajectory with shape {trajectory.shape}")

    # flatten the shots -> (Nsamples, 2) and rescale from [-0.5, 0.5] to [-pi, pi]
    trajectory = trajectory.reshape(-1, trajectory.shape[-1]) * (2 * np.pi)
    print(f"Flattened and rescaled trajectory to shape {trajectory.shape} in [-pi, pi]")

    savemat(args.output, {
        "trajectory": trajectory,
        "Nc": args.Nc,
        "Ns": args.Ns,
        "tilt": args.tilt,
        "in_out": args.in_out,
        "nb_revolutions": args.nb_revolutions,
        "spiral": args.spiral,
    })
    print(f"Saved trajectory to {args.output}")


if __name__ == "__main__":
    main()
