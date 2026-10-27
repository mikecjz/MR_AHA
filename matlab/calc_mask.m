function [mask, kernel] = calc_mask(traj, config, threshold)
%CALC_MASK  Derive the masked-operator sample set from the trajectory alone.
%
%   mask = CALC_MASK(traj, config)
%   mask = CALC_MASK(traj, config, threshold)
%   [mask, kernel] = CALC_MASK(...)
%
%   A masked A'A operator (config.is_masked = true) stores the Toeplitz
%   kernel only at selected locations of the upsampled Cartesian k-space
%   grid. CALC_MASK chooses those locations from the trajectory, with the
%   method in config.mask_method:
%
%   'neighbour' (default)
%     every sample is mapped to its nearest point on the upsampled grid
%     (round(k*M/(2*pi)) + floor(M/2) per axis, periodic) and every location
%     within config.mask_radius cells of such a point (a (2r+1)^d box) is
%     kept. No NUFFT is needed. The default radius 2 matches the unmasked
%     accuracy on densely sampled radial and 3D trajectories while keeping
%     roughly the sampled disc or ball of k-space.
%
%   'threshold'
%     1. every Gram block is replaced by one, which turns CALC_TOEP into the
%        Toeplitz kernel T of the plain normal operator of the trajectory
%        ("Toeplitz weights with all weights set to one");
%     2. |T| is max-filtered over the 3x3 (2D, stack) or 3x3x3 (3D)
%        neighbourhood of each location;
%     3. locations where the result exceeds the absolute threshold
%        config.mask_threshold are retained. The kernel is measured in
%        samples: one isolated sample on the upsampled grid contributes
%        exactly 1, so 0.05 drops locations that receive less than 5% of one
%        sample's worth.
%     It follows the kernel's slowly decaying, axis-aligned tails, which
%     makes it more accurate per retained location for very sparse
%     trajectories such as few-arm spirals.
%
%   For a 'stack' trajectory the in-plane mask is repeated on every kz plane.
%   benchmarks/neighbour_mask.md compares the two methods.
%
%   INPUTS
%     traj       real single gpuArray [Ntrajs, Nkx, ndim], radians in
%                [-pi, pi), as for CALC_TOEP.
%     config     struct as for CALC_TOEP / AHA_TOEPLITZ (see
%                MR_DEFAULT_CONFIG). Uses config.mask_method (default
%                'neighbour'), config.mask_radius (default 2) and
%                config.mask_threshold (default 0.05).
%     threshold  (optional) if given, use the 'threshold' method with this
%                absolute threshold >= 0 (in samples), whatever
%                config.mask_method says.
%
%   OUTPUTS
%     mask       int64 gpuArray column of sorted, 1-based linear indices into
%                the centred upsampled grid config.phi2_dim. Pass it to
%                CALC_TOEP and AHA_TOEPLITZ.
%     kernel     single gpuArray |T| (the unit-weight kernel of the
%                'threshold' method, before the max-filter), [Mx, My] for
%                '2D'/'stack' or [Mx, My, Mz] for '3D', centred. Computed for
%                either method.
%
%   EXAMPLE
%     config = mr_default_config('3D', [160 160 160], 12, 'is_masked', true);
%     mask   = calc_mask(traj, config);
%     Om     = calc_toep(phi2, traj, config, mask);
%     AhAx   = AhA_Toeplitz(x, SEs, Om, config, mask);
%
%   See also CALC_TOEP, AHA_TOEPLITZ, MR_DEFAULT_CONFIG.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    narginchk(2, 3);

    if ~isa(traj, 'gpuArray'), traj = gpuArray(traj); end
    if ~strcmp(classUnderlying(traj), 'single'), traj = single(traj); end
    if ~isreal(traj)
        error('MR_AHA:calc_mask:complexTraj', 'traj must be real-valued.');
    end

    if nargin >= 3 && ~isempty(threshold)
        config.mask_method    = 'threshold';
        config.mask_threshold = double(threshold);
    end

    mask = calc_mask_mex(traj, config) + 1;   % the library indexes from 0
    if nargout > 1
        kernel = calc_mask_mex(traj, config, 'kernel');
    end
end
