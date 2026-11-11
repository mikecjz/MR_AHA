function [phi2_toep, mask] = calc_toep(phi2, traj, config, varargin)
%CALC_TOEP  Pre-compute the block-diagonal Toeplitz operator for A'A.
%
%   phi2_toep = CALC_TOEP(phi2, traj, config)
%   phi2_toep = CALC_TOEP(phi2, traj, config, mask)
%   [phi2_toep, mask] = CALC_TOEP(phi2, traj, config)    % masked, auto mask
%
%   Turns the non-Cartesian Gram blocks Phi*Phi' into the Cartesian
%   block-diagonal multiplier that AHA_TOEPLITZ applies between its two FFTs.
%   The kernel is computed on MATLAB's current GPU.
%
%   CALC_TOEP and AHA_TOEPLITZ are the stateless low-level primitives. To
%   apply A'A repeatedly (iterative reconstruction), or on several GPUs, use
%   MR_AHA_OP.CREATE_OP instead: it computes the kernel itself from the same
%   phi2 and traj (on every GPU it uses) and keeps it, with the sensitivity
%   maps, for every application.
%
%   Gram blocks follow the MR-Multitasking convention: the temporal basis
%   Phi_rt is rank-by-time [L, T], the Gram block of a readout (and kz
%   encode) is Phi_rt(:,idx) * Phi_rt(:,idx)', and the operator multiplies
%   the rank coefficients of every k-space location as a row vector from the
%   right, k(1,:) <- k(1,:) * Phi2, exactly like MR-Multitasking's
%   reshape(k, [], L) * Phi2. Its Phi2 can therefore be passed unchanged
%   apart from the dimension order (see below).
%
%   INPUTS
%     phi2       complex single gpuArray of Gram blocks, one L-by-L matrix per
%                trajectory (per kz encode for a stack):
%                  [Ntrajs, L, L]        for config.traj_type = '2D' or '3D'
%                  [Ntrajs, Nkz, L, L]   for 'stack', with Nkz == Nz == Mz
%                i.e. phi2(itraj, :, :) is the L-by-L matrix
%                Phi_rt(:,idx) * Phi_rt(:,idx)' of trajectory itraj. Data
%                stored as [L, L, Ntrajs] (or [L, L, Ntrajs, Nz] for a stack,
%                as MR-Multitasking does) must be permuted first, e.g.
%                permute(Phi2, [3 1 2]) or permute(Phi2, [3 4 1 2]); a stack
%                in natural kz order also needs fftshift(.., 2).
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
%   BACKENDS
%     CUDA   gpuArrays; the type-1 NUFFT is cufinufft inside the MEX file.
%     Metal  (Apple silicon, see MR_BACKEND) ordinary arrays; the type-1 NUFFT
%            runs in Python with mlx-nufft through MATLAB's Python interface
%            (see MR_PYTHON), and the library finishes the operator on the
%            Apple GPU. The outputs are ordinary arrays (mask: int64 column).
%
%   See also AHA_TOEPLITZ, CALC_MASK, MR_DEFAULT_CONFIG, AHA_CUFINUFFT,
%   MR_AHA_OP.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    narginchk(3, 4);
    mask = [];
    if ~isempty(varargin)
        mask = varargin{1};
    end

    if strcmp(mr_backend(), 'metal')
        [phi2_toep, mask] = mr_calc_toep_metal(phi2, traj, config, mask);
        return
    end

    if ~isa(phi2, 'gpuArray'), phi2 = gpuArray(phi2); end
    if ~strcmp(underlyingType(phi2), 'single'), phi2 = single(phi2); end
    if isreal(phi2), phi2 = complex(phi2); end

    if ~isa(traj, 'gpuArray'), traj = gpuArray(traj); end
    if ~strcmp(underlyingType(traj), 'single'), traj = single(traj); end
    if ~isreal(traj)
        error('MR_AHA:calc_toep:complexTraj', 'traj must be real-valued.');
    end

    is_masked = isfield(config, 'is_masked') && config.is_masked;
    if ~is_masked
        mask = [];
        phi2_toep = calc_toep_mex(phi2, traj, config);
        return
    end

    if isempty(mask)
        mask = calc_mask(traj, config);       % derived from the trajectory
    end
    % The library indexes from 0. (int64 arithmetic on gpuArrays needs
    % R2023a+; double is exact for any index below 2^53.)
    mask0 = int64(double(gpuArray(mask(:))) - 1);
    config.num_masked = numel(mask0);
    phi2_toep = calc_toep_mex(phi2, traj, config, mask0);
end

function [phi2_toep, mask] = mr_calc_toep_metal(phi2, traj, config, mask)
%MR_CALC_TOEP_METAL  Metal path: mlx-nufft in Python, then the library.
    if isa(phi2, 'gpuArray'), phi2 = gather(phi2); end
    if isa(traj, 'gpuArray'), traj = gather(traj); end
    phi2 = complex(single(phi2));
    traj = single(traj);
    if ~isreal(traj)
        error('MR_AHA:calc_toep:complexTraj', 'traj must be real-valued.');
    end

    stack = strcmpi(config.traj_type, 'stack');
    mr = mr_python();
    phi2_py = mr_to_numpy(phi2, 3 + stack);    % [Ntrajs,(Nkz),L,L] -> (L,L,(Nkz),Ntrajs)
    traj_py = mr_to_numpy(traj, 3);            % [Ntrajs,Nkx,ndim]  -> (ndim,Nkx,Ntrajs)

    is_masked = isfield(config, 'is_masked') && config.is_masked;
    mask_py = py.None;
    if is_masked
        if isempty(mask)
            mask = calc_mask(traj, config);
        end
        mask = int64(gather(mask(:)));
        config.num_masked = numel(mask);
        mask_py = mr_to_numpy(mask - 1).ravel();   % 0-based, 1-D
    else
        mask = [];
    end
    out = mr.calc_toep(mr_py_config(config), phi2_py, traj_py, pyargs('mask', mask_py));
    phi2_toep = mr_from_numpy(out);
end
