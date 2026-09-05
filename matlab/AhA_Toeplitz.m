function AhAx = AhA_Toeplitz(x, SEs, phi2_toep, config, mask)
%AHA_TOEPLITZ  Forward normal operator A'A for low-rank non-Cartesian MRI.
%
%   AhAx = AHA_TOEPLITZ(x, SEs, phi2_toep, config)
%   AhAx = AHA_TOEPLITZ(x, SEs, phi2_toep, config, mask)
%
%   Evaluates A'*A*x on the GPU using a Toeplitz embedding, so that no
%   non-uniform FFT is needed inside an iterative reconstruction.
%
%   INPUTS
%     x          complex single gpuArray of spatial coefficients.
%                  [Nx, Ny, L]        for config.traj_type = '2D'
%                  [Nx, Ny, Nz, L]    for '3D' and 'stack'
%
%     SEs        complex single gpuArray of coil sensitivity maps.
%                  [Nx, Ny, Ncoils]        for '2D'
%                  [Nx, Ny, Nz, Ncoils]    for '3D' and 'stack'
%                One map per coil, shared by every rank.
%
%     phi2_toep  complex single gpuArray holding the block-diagonal Toeplitz
%                operator in packed triangular form, as returned by CALC_TOEP.
%                  [Mx, My, (L^2+L)/2]          '2D',  unmasked
%                  [Mx, My, Mz, (L^2+L)/2]      '3D'/'stack', unmasked
%                  [numel(mask), (L^2+L)/2]     masked
%
%     config     struct describing the operator:
%                  .traj_type      '2D' | '3D' | 'stack'
%                  .n_dim          [Nx Ny] or [Nx Ny Nz]
%                  .L              number of ranks
%                  .triangulation  'upper' | 'lower'
%                  .phi2_dim       [Mx My] or [Mx My Mz]
%                  .is_masked      logical
%                Optional fields: .n_coils, .gpu_device_id, .verbose.
%                See MR_DEFAULT_CONFIG.
%
%     mask       (optional, required when config.is_masked is true) linear
%                indices, in MATLAB's 1-based convention, into the centred
%                upsampled Cartesian grid of size config.phi2_dim.
%
%   OUTPUT
%     AhAx       complex single gpuArray, the same size as x.
%
%   All computation happens on the GPU; inputs that are not already gpuArrays
%   are moved there, and inputs that are real or double precision are cast, so
%   passing data in the exact expected form avoids all copies.
%
%   See also CALC_TOEP, MR_DEFAULT_CONFIG.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    narginchk(4, 5);

    x         = mr_to_complex_single_gpu(x,         'x');
    SEs       = mr_to_complex_single_gpu(SEs,       'SEs');
    phi2_toep = mr_to_complex_single_gpu(phi2_toep, 'phi2_toep');

    config = mr_fill_n_coils(config, SEs);

    if nargin < 5 || isempty(mask)
        if isfield(config, 'is_masked') && config.is_masked
            error('MR_AHA:AhA_Toeplitz:missingMask', ...
                  'config.is_masked is true, so a mask must be supplied.');
        end
        AhAx = AhA_Toeplitz_mex(x, SEs, phi2_toep, config);
    else
        % The library indexes from 0; MATLAB indexes from 1.
        mask = int64(gpuArray(mask(:))) - 1;
        config.num_masked = numel(mask);
        AhAx = AhA_Toeplitz_mex(x, SEs, phi2_toep, config, mask);
    end
end

function a = mr_to_complex_single_gpu(a, name)
%MR_TO_COMPLEX_SINGLE_GPU  Coerce an array to a complex single gpuArray.
%   Emits no copy when the input already has the required form.
    if ~isa(a, 'gpuArray')
        a = gpuArray(a);
    end
    if ~strcmp(classUnderlying(a), 'single')
        a = single(a);
    end
    if isreal(a)
        % The library always reads interleaved complex storage.
        a = complex(a);
    end
    if isempty(a)
        error('MR_AHA:AhA_Toeplitz:emptyInput', '%s must not be empty.', name);
    end
end

function config = mr_fill_n_coils(config, SEs)
%MR_FILL_N_COILS  Derive .n_coils from the trailing dimension of SEs.
%   An explicit value already present in config is never overwritten.
    if isfield(config, 'n_coils') && ~isempty(config.n_coils)
        return
    end
    if strcmpi(config.traj_type, '2D')
        n_spatial = 2;
    else
        n_spatial = 3;
    end
    sz = size(SEs);
    sz(end+1:n_spatial + 1) = 1;   % restore MATLAB's dropped trailing ones
    config.n_coils = sz(n_spatial + 1);
end
