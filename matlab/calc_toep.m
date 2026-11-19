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
%   apply A'A repeatedly (iterative reconstruction), or multi-GPU, use
%   MR_AHA.CREATE_OP instead: it computes the kernel itself from the same
%   phi2 and traj (on every GPU it uses) and keeps it, with the sensitivity
%   maps, for every application.
%
%   Gram block convention: the temporal basis
%   Phi_rt is rank-by-time [L, T], the Gram block of a readout (and kz
%   encode) is Phi_rt(:,idx) * Phi_rt(:,idx)', and the operator multiplies
%   the rank coefficients of every k-space location as a row vector from the
%   right, k(1,:) <- k(1,:) * Phi2, i.e. reshape(k, [], L) * Phi2. Blocks
%   stored [L, L, Ntrajs(, Nz)] are passed unchanged apart from the
%   dimension order (see below).
%
%   INPUTS
%     phi2       complex single gpuArray of Gram blocks, one L-by-L matrix per
%                trajectory (per kz encode for a stack):
%                  [Ntrajs, L, L]        for config.traj_type = '2D' or '3D'
%                  [Ntrajs, Nkz, L, L]   for 'stack', with Nkz == Nz == Mz
%                i.e. phi2(itraj, :, :) is the L-by-L matrix
%                Phi_rt(:,idx) * Phi_rt(:,idx)' of trajectory itraj. Data
%                stored as [L, L, Ntrajs] (or [L, L, Ntrajs, Nz] for a stack)
%                must be permuted first, e.g.
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
%                default 0.5), .mask_radius (see CALC_MASK), .verbose.
%
%     mask       (optional) 1-based linear indices into the centered upsampled
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
%   The result already absorbs the 1/prod(config.phi2_dim) normalization of the
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
%   MR_AHA.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    % ---- Arguments -------------------------------------------------------
    % The optional fourth argument is a 1-based mask; [] means "none given",
    % in which case a masked config derives one with CALC_MASK below.
    narginchk(3, 4);
    mask = [];
    if ~isempty(varargin)
        mask = varargin{1};
    end

    % ---- Backend dispatch -------------------------------------------------
    % The Metal build has no cufinufft: its type-1 NUFFT runs in Python
    % (mlx-nufft), so the whole computation is delegated to a local function.
    if strcmp(mr_backend(), 'metal')
        [phi2_toep, mask] = mr_calc_toep_metal(phi2, traj, config, mask);
        return
    end

    % ---- CUDA path: coerce inputs to what calc_toep_mex accepts -----------
    % phi2 must be a complex single gpuArray. Each step is skipped when the
    % input already has that form, so correctly prepared data is not copied.
    % complex() adds an explicit zero imaginary part, because the library
    % reads interleaved complex storage.
    if ~isa(phi2, 'gpuArray'), phi2 = gpuArray(phi2); end
    if ~strcmp(underlyingType(phi2), 'single'), phi2 = single(phi2); end
    if isreal(phi2), phi2 = complex(phi2); end

    % traj must be a real single gpuArray (coordinates in radians).
    if ~isa(traj, 'gpuArray'), traj = gpuArray(traj); end
    if ~strcmp(underlyingType(traj), 'single'), traj = single(traj); end
    if ~isreal(traj)
        error('MR_AHA:calc_toep:complexTraj', 'traj must be real-valued.');
    end

    % ---- Unmasked kernel ---------------------------------------------------
    % The full upsampled grid is stored; any mask argument is discarded and
    % the second output is [].
    is_masked = isfield(config, 'is_masked') && config.is_masked;
    if ~is_masked
        mask = [];
        phi2_toep = calc_toep_mex(phi2, traj, config);
        return
    end

    % ---- Masked kernel -----------------------------------------------------
    % Without a caller-supplied mask, derive it from the trajectory. The
    % 1-based mask is returned as the second output so it can be passed to
    % AHA_TOEPLITZ unchanged.
    if isempty(mask)
        mask = calc_mask(traj, config);       % derived from the trajectory
    end
    % The library indexes from 0. (int64 arithmetic on gpuArrays needs
    % R2023a+; double is exact for any index below 2^53.)
    mask0 = int64(double(gpuArray(mask(:))) - 1);
    % num_masked is set from the mask so the config always matches it.
    config.num_masked = numel(mask0);
    phi2_toep = calc_toep_mex(phi2, traj, config, mask0);
end

function [phi2_toep, mask] = mr_calc_toep_metal(phi2, traj, config, mask)
%MR_CALC_TOEP_METAL  Metal path: mlx-nufft in Python, then the library.

    % ---- Inputs as ordinary arrays -----------------------------------------
    % Metal works on host memory (unified memory on Apple silicon), so
    % gpuArrays are gathered. phi2 becomes complex single, traj real single.
    if isa(phi2, 'gpuArray'), phi2 = gather(phi2); end
    if isa(traj, 'gpuArray'), traj = gather(traj); end
    phi2 = complex(single(phi2));
    traj = single(traj);
    if ~isreal(traj)
        error('MR_AHA:calc_toep:complexTraj', 'traj must be real-valued.');
    end

    % ---- Hand the arrays to the Python package -----------------------------
    % mr_to_numpy reverses the dimension order: MATLAB column-major
    % [a, b, c] has the same memory layout as row-major (c, b, a), so NumPy
    % sees the arrays without data movement. The second argument pins the
    % number of dimensions so trailing singleton axes (e.g. L = 1) survive.
    stack = strcmpi(config.traj_type, 'stack');
    mr = mr_python();
    phi2_py = mr_to_numpy(phi2, 3 + stack);    % [Ntrajs,(Nkz),L,L] -> (L,L,(Nkz),Ntrajs)
    traj_py = mr_to_numpy(traj, 3);            % [Ntrajs,Nkx,ndim]  -> (ndim,Nkx,Ntrajs)

    % ---- Mask --------------------------------------------------------------
    % Masked: use or derive the 1-based mask, keep it as an int64 column for
    % the second output, and pass the 0-based 1-D copy to Python. Unmasked:
    % Python receives None and the second output is [].
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
    % ---- Compute ---------------------------------------------------------
    % mr_aha.calc_toep (Python) runs the NUFFT and the library's
    % mr_calc_toep_from_psf; mr_from_numpy converts the NumPy result back to a
    % MATLAB array with the original column-major shape.
    out = mr.calc_toep(mr_py_config(config), phi2_py, traj_py, pyargs('mask', mask_py));
    phi2_toep = mr_from_numpy(out);
end
