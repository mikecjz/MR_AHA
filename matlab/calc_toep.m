function phi2_toep = calc_toep(phi2, traj, config, mask)
%CALC_TOEP  Pre-compute the block-diagonal Toeplitz operator for A'A.
%
%   phi2_toep = CALC_TOEP(phi2, traj, config)
%   phi2_toep = CALC_TOEP(phi2, traj, config, mask)
%
%   Turns the non-Cartesian Gram blocks Phi'*Phi into the Cartesian
%   block-diagonal multiplier that AHA_TOEPLITZ applies between its two FFTs.
%
%   INPUTS
%     phi2       complex single gpuArray of Gram blocks, one L-by-L matrix per
%                trajectory (per kz encode for a stack):
%                  [Ntrajs, L*L]        for config.traj_type = '2D' or '3D'
%                  [Ntrajs, Nkz, L*L]   for 'stack', with Nkz == Nz == Mz
%                The trailing L*L axis is the column-major L-by-L matrix, i.e.
%                reshape(Phi2(:,:,itraj), 1, []) for each trajectory.
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
%                default 0.5), .verbose.
%
%     mask       (optional, required when config.is_masked is true) 1-based
%                linear indices into the centred upsampled grid config.phi2_dim
%                at which the operator is retained.
%
%   OUTPUT
%     phi2_toep  complex single gpuArray:
%                  [Mx, My, (L^2+L)/2]        '2D',  unmasked
%                  [Mx, My, Mz, (L^2+L)/2]    '3D'/'stack', unmasked
%                  [numel(mask), (L^2+L)/2]   masked
%                Only the triangle named by config.triangulation is stored;
%                the other half follows from Hermitian symmetry.
%
%   The result already absorbs the 1/prod(config.phi2_dim) normalisation of the
%   FFT pair used by AHA_TOEPLITZ, so no further scaling is required anywhere.
%
%   See also AHA_TOEPLITZ, MT_DEFAULT_CONFIG.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    narginchk(3, 4);

    if ~isa(phi2, 'gpuArray'), phi2 = gpuArray(phi2); end
    if ~strcmp(classUnderlying(phi2), 'single'), phi2 = single(phi2); end
    if isreal(phi2), phi2 = complex(phi2); end

    if ~isa(traj, 'gpuArray'), traj = gpuArray(traj); end
    if ~strcmp(classUnderlying(traj), 'single'), traj = single(traj); end
    if ~isreal(traj)
        error('MT_CUDA:calc_toep:complexTraj', 'traj must be real-valued.');
    end

    if nargin < 4 || isempty(mask)
        if isfield(config, 'is_masked') && config.is_masked
            error('MT_CUDA:calc_toep:missingMask', ...
                  'config.is_masked is true, so a mask must be supplied.');
        end
        phi2_toep = calc_toep_mex(phi2, traj, config);
    else
        mask = int64(gpuArray(mask(:))) - 1;   % the library indexes from 0
        config.num_masked = numel(mask);
        phi2_toep = calc_toep_mex(phi2, traj, config, mask);
    end
end
