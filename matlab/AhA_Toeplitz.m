function AhAx = AhA_Toeplitz(x, SEs, phi2_toep, config, mask)
%AHA_TOEPLITZ  Forward normal operator A'A for low-rank non-Cartesian MRI.
%
%   AhAx = AHA_TOEPLITZ(x, SEs, phi2_toep, config)
%   AhAx = AHA_TOEPLITZ(x, SEs, phi2_toep, config, mask)
%
%   Evaluates A'*A*x on the GPU using a Toeplitz embedding, so that no
%   non-uniform FFT is needed inside an iterative reconstruction.
%
%   AHA_TOEPLITZ is the stateless low-level primitive: it applies a finished
%   kernel once, on the current GPU, and allocates its working buffers on
%   every call. For repeated application (iterative reconstruction), or on
%   several GPUs, build the operator once with MR_AHA.CREATE_OP, which
%   computes the kernel itself from the Gram blocks and keeps it, the maps
%   and the buffers: op = mr_aha.create_op(SEs, phi2, traj, config);
%   AhAx = op * x.
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
%                At every upsampled k-space location the rank coefficients
%                are multiplied as a row vector from the right by that
%                location's Hermitian block B, k(1,:) <- k(1,:) * B
%                (MR-Multitasking convention, see CALC_TOEP).
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
%   All computation happens on the GPU. With the CUDA backend inputs that are
%   not already gpuArrays are moved there; with the Metal backend (Apple
%   silicon, see MR_BACKEND) arrays stay ordinary MATLAB arrays, which the GPU
%   reads in place through unified memory, and gpuArrays are gathered. Inputs
%   that are real or double precision are cast, so passing data in the exact
%   expected form (complex single) avoids all copies.
%
%   See also MR_AHA, CALC_TOEP, MR_DEFAULT_CONFIG.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    narginchk(4, 5);
    metal = strcmp(mr_backend(), 'metal');

    x         = mr_to_complex_single(x,         'x',         metal);
    SEs       = mr_to_complex_single(SEs,       'SEs',       metal);
    phi2_toep = mr_to_complex_single(phi2_toep, 'phi2_toep', metal);

    config = mr_fill_n_coils(config, SEs);

    if nargin < 5 || isempty(mask)
        if isfield(config, 'is_masked') && config.is_masked
            error('MR_AHA:AhA_Toeplitz:missingMask', ...
                  'config.is_masked is true, so a mask must be supplied.');
        end
        AhAx = AhA_Toeplitz_mex(x, SEs, phi2_toep, config);
    else
        % The library indexes from 0; MATLAB indexes from 1.
        if metal
            mask = int64(gather(mask(:))) - 1;
        else
            % (int64 arithmetic on gpuArrays needs R2023a+; double is exact
            % for any index below 2^53)
            mask = int64(double(gpuArray(mask(:))) - 1);
        end
        config.num_masked = numel(mask);
        AhAx = AhA_Toeplitz_mex(x, SEs, phi2_toep, config, mask);
    end
end

function a = mr_to_complex_single(a, name, metal)
%MR_TO_COMPLEX_SINGLE  Coerce an array to complex single on the backend's side:
%   a gpuArray for CUDA, an ordinary array for Metal. Emits no copy when the
%   input already has the required form.
    if metal
        if isa(a, 'gpuArray'), a = gather(a); end
    elseif ~isa(a, 'gpuArray')
        a = gpuArray(a);
    end
    if ~strcmp(underlyingType(a), 'single')
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
