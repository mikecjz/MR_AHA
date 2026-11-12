function mask = calc_mask(traj, config)
%CALC_MASK  Derive the masked-operator sample set from the trajectory alone.
%
%   mask = CALC_MASK(traj, config)
%
%   A masked A'A operator (config.is_masked = true) stores the Toeplitz
%   kernel only at selected locations of the upsampled Cartesian k-space
%   grid. CALC_MASK chooses those locations from the trajectory: every
%   sample is mapped to its nearest point on the upsampled grid
%   (round(k*M/(2*pi)) + floor(M/2) per axis, periodic) and every location
%   within config.mask_radius cells of such a point (a (2r+1)^d box) is
%   kept. No NUFFT is needed. The default radius 2 matches the unmasked
%   accuracy on densely sampled radial and 3D trajectories while keeping
%   roughly the sampled disc or ball of k-space; sparse trajectories need a
%   larger radius.
%
%   For a 'stack' trajectory the in-plane mask is repeated on every kz plane.
%   https://mikecjz.github.io/MR_AHA_docs/benchmarks/neighbour-mask.html
%   has the measurements.
%
%   INPUTS
%     traj       real single gpuArray [Ntrajs, Nkx, ndim], radians in
%                [-pi, pi), as for CALC_TOEP.
%     config     struct as for CALC_TOEP / AHA_TOEPLITZ (see
%                MR_DEFAULT_CONFIG). Uses config.mask_radius (default 2).
%
%   OUTPUT
%     mask       int64 gpuArray column of sorted, 1-based linear indices into
%                the centred upsampled grid config.phi2_dim. Pass it to
%                MR_AHA.CREATE_OP ('mask'), or to CALC_TOEP and
%                AHA_TOEPLITZ. (Both MR_AHA.CREATE_OP and CALC_TOEP call
%                CALC_MASK themselves when a masked config comes without a
%                mask.)
%
%   EXAMPLE
%     config = mr_default_config('3D', [160 160 160], 12, 'is_masked', true);
%     mask   = calc_mask(traj, config);
%     op     = mr_aha.create_op(SEs, phi2, traj, config, 'mask', mask);
%     AhAx   = op * x;                  % or, with the low-level primitives:
%     Om     = calc_toep(phi2, traj, config, mask);
%     AhAx   = AhA_Toeplitz(x, SEs, Om, config, mask);
%
%   BACKENDS
%     CUDA   gpuArrays.
%     Metal  (Apple silicon, see MR_BACKEND) ordinary arrays.
%     Both run entirely in the MEX file and give the same mask.
%
%   See also MR_AHA, CALC_TOEP, AHA_TOEPLITZ, MR_DEFAULT_CONFIG.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    narginchk(2, 2);

    if strcmp(mr_backend(), 'metal')
        if isa(traj, 'gpuArray'), traj = gather(traj); end
        traj = single(traj);
        if ~isreal(traj)
            error('MR_AHA:calc_mask:complexTraj', 'traj must be real-valued.');
        end
        mask = calc_mask_mex(traj, config) + 1;
        return
    end

    if ~isa(traj, 'gpuArray'), traj = gpuArray(traj); end
    if ~strcmp(underlyingType(traj), 'single'), traj = single(traj); end
    if ~isreal(traj)
        error('MR_AHA:calc_mask:complexTraj', 'traj must be real-valued.');
    end

    % The library indexes from 0. (int64 arithmetic on gpuArrays needs
    % R2023a+; double is exact for any index below 2^53.)
    mask = int64(double(calc_mask_mex(traj, config)) + 1);
end
