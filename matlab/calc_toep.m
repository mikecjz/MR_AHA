function [phi2_toep, mask] = calc_toep(phi2, traj, config, mask)
%CALC_TOEP  Pre-compute the block-diagonal Toeplitz operator for A'A.
%
%   phi2_toep = CALC_TOEP(phi2, traj, config)
%   phi2_toep = CALC_TOEP(phi2, traj, config, mask)
%   [phi2_toep, mask] = CALC_TOEP(phi2, traj, config)    % masked, auto mask
%
%   Turns the non-Cartesian Gram blocks Phi'*Phi into the Cartesian
%   block-diagonal multiplier that AHA_TOEPLITZ applies between its two FFTs.
%
%   INPUTS
%     phi2       complex single gpuArray of Gram blocks, one L-by-L matrix per
%                trajectory (per kz encode for a stack):
%                  [Ntrajs, L, L]        for config.traj_type = '2D' or '3D'
%                  [Ntrajs, Nkz, L, L]   for 'stack', with Nkz == Nz == Mz
%                i.e. phi2(itraj, :, :) is the L-by-L matrix of trajectory
%                itraj. Data stored as [L, L, Ntrajs] (or [L, L, Ntrajs, Nz]
%                for a stack) must be permuted first, e.g.
%                permute(Phi2, [3 1 2]) or permute(Phi2, [3 4 1 2]).
%
%     traj       real single gpuArray of non-Cartesian coordinates,
%                  [Ntrajs, Nkx, ndim]
%                with ndim = 2 for '2D' and 'stack' and 3 for '3D'. Units are
%                radians per unit of the un-padded image grid, i.e. the usual
%                [-pi, pi) range.
%
%     config     struct as documented in AHA_TOEPLITZ. config.phi2_dim gives
%                the upsampled grid the kernel is evaluated on (typically
%                2*config.n_dim, except along kz of a stack where Mz == Nz).
%                Optional: .eps (NUFFT tolerance, default 1e-6),
%                .mem_fraction (share of free GPU memory the chunking may use,
%                default 0.5), .mask_method, .mask_radius, .mask_threshold
%                (see CALC_MASK), .verbose.
%
%     mask       (optional) 1-based linear indices into the centred upsampled
%                grid config.phi2_dim at which the operator is retained. Only
%                used when config.is_masked is true; when it is omitted the
%                mask is derived from the trajectory with CALC_MASK and
%                returned as the second output.
%
%   OUTPUTS
%     phi2_toep  complex single gpuArray:
%                  [Mx, My, (L^2+L)/2]        '2D',  unmasked
%                  [Mx, My, Mz, (L^2+L)/2]    '3D'/'stack', unmasked
%                  [numel(mask), (L^2+L)/2]   masked
%                Only the triangle named by config.triangulation is stored;
%                the other half follows from Hermitian symmetry.
%     mask       the mask that was used (int64 gpuArray, 1-based), or [] for
%                an unmasked operator. Pass it on to AHA_TOEPLITZ.
%
%   The result already absorbs the 1/prod(config.phi2_dim) normalisation of the
%   FFT pair used by AHA_TOEPLITZ, so no further scaling is required anywhere.
%
%   See also AHA_TOEPLITZ, CALC_MASK, MR_DEFAULT_CONFIG, AHA_CUFINUFFT.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    narginchk(3, 4);

    if ~isa(phi2, 'gpuArray'), phi2 = gpuArray(phi2); end
    if ~strcmp(classUnderlying(phi2), 'single'), phi2 = single(phi2); end
    if isreal(phi2), phi2 = complex(phi2); end

    if ~isa(traj, 'gpuArray'), traj = gpuArray(traj); end
    if ~strcmp(classUnderlying(traj), 'single'), traj = single(traj); end
    if ~isreal(traj)
        error('MR_AHA:calc_toep:complexTraj', 'traj must be real-valued.');
    end

    is_masked = isfield(config, 'is_masked') && config.is_masked;
    if ~is_masked
        mask = [];
        phi2_toep = calc_toep_mex(phi2, traj, config);
        return
    end

    if nargin < 4 || isempty(mask)
        mask = calc_mask(traj, config);       % derived from the trajectory
    end
    mask0 = int64(gpuArray(mask(:))) - 1;    % the library indexes from 0
    config.num_masked = numel(mask0);
    phi2_toep = calc_toep_mex(phi2, traj, config, mask0);
end
