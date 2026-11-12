classdef aha_nufft_plan < handle
%AHA_NUFFT_PLAN  Reusable NUFFT-based A'A reference operator (no Toeplitz).
%
%   plan = AHA_NUFFT_PLAN(traj, config)
%   plan = AHA_NUFFT_PLAN(traj, config, 'Name', Value, ...)
%   AhAx = plan.apply(x, SEs, phi2)
%   AhAx = plan.execute(x, SEs, phi2)      % alias of apply
%
%   Creates one type-2 (image -> k-space, sign -1) and one type-1 (k-space ->
%   image, sign +1) single-precision cufinufft plan for the trajectory and
%   sorts the non-uniform points once, so that APPLY can be called many times
%   (e.g. inside conjugate gradient) without any planning cost. Each plan
%   transforms all L ranks of one coil in a single batched call; for a
%   'stack' trajectory the batch is L*Nz, one 2D transform per kz plane and
%   rank.
%
%   APPLY evaluates the forward normal operator coil by coil:
%
%       im = x .* SEs(:,:,..,c)                 % broadcast over the L ranks
%       k  = type-2 NUFFT(im)                   % image -> k-space, sign -1
%       k  = k * phi2                           % one L-by-L block per readout,
%                                               % rank row vector from the right
%       im = type-1 NUFFT(k)                    % k-space -> image, sign +1
%       AhAx = AhAx + im .* conj(SEs(:,:,..,c))
%
%   Both NUFFTs are unnormalised, the convention reproduced by the Toeplitz
%   kernel of CALC_TOEP, so APPLY and AHA_TOEPLITZ agree to within the NUFFT
%   tolerance. For a 'stack' trajectory the image is first transformed along
%   z (centred, unnormalised FFT), the 2D NUFFT pair is applied to every kz
%   plane with the Gram block of that kz, and the result is transformed back
%   (unnormalised inverse FFT). The centring is realised by a phase
%   modulation folded into the coil map rather than by fftshift copies.
%
%   ROLE: this is the independent REFERENCE of the NUFFT route, not the
%   production implementation. No MR_AHA library code is involved: the
%   NUFFTs come from cufinufft's own MATLAB interface (CUFINUFFT_PLAN) and
%   everything else is gpuArray code. The tests use it to validate
%   MR_AHA.CREATE_OP(..., 'method', 'nufft'), which evaluates the same
%   operator inside the library (cufinufft's C API, multi-GPU if asked, no
%   MATLAB per-coil loop). Use that operator in reconstructions.
%
%   With the Metal backend (Apple silicon, see MR_BACKEND) the plan wraps the
%   Python reference operator mr_aha.aha_nufft_plan (MLX + mlx-nufft), APPLY
%   takes ordinary arrays, and the cufinufft options are ignored.
%
%   CONSTRUCTOR INPUTS
%     traj    real gpuArray [Ntrajs, Nkx, ndim], radians in [-pi, pi), the same
%             array passed to CALC_TOEP. ndim = 2 for '2D'/'stack', 3 for '3D'.
%     config  struct from MR_DEFAULT_CONFIG; uses .traj_type, .n_dim, .L and
%             .eps (default 1e-6).
%
%   NAME-VALUE PAIRS
%     'type1_opts'  struct of cufinufft options for the type-1 (adjoint) plan
%                   only, e.g. struct('gpu_method', 1).
%     'type2_opts'  same for the type-2 (forward) plan.
%     Any other name is a cufinufft option (e.g. 'gpu_method', 'upsampfac',
%     'gpu_maxbatchsize') forwarded to both plans. 'upsampfac' defaults to 2
%     and 'modeord' is always 0 (centred modes, the library convention).
%
%   APPLY INPUTS
%     x       complex single gpuArray, [Nx, Ny, L] ('2D') or [Nx, Ny, Nz, L].
%     SEs     complex single gpuArray, [Nx, Ny, Ncoils] or [Nx, Ny, Nz, Ncoils].
%     phi2    complex single gpuArray of Gram blocks Phi*Phi' (Phi is the
%             rank-by-time basis Phi_rt, and the block of a readout is
%             Phi_rt(:,idx)*Phi_rt(:,idx)'),
%               [Ntrajs, L, L]        '2D' / '3D'
%               [Ntrajs, Nkz, L, L]   'stack' (centred kz order, Nkz == Nz)
%             Blocks stored [L, L, Ntrajs, Nz] are permute(Phi2, [3 4 1 2])
%             (fftshift along kz if needed).
%   Host inputs are moved to the GPU.
%
%   APPLY OUTPUT
%     AhAx    complex single gpuArray, the same size as x.
%
%   The block-diagonal product: the L rank coefficients of a k-space sample
%   form a row vector multiplied from the right by the Gram block,
%   k(1,:) <- k(1,:) * Phi2, i.e. k_l <- sum_m k_m Phi2(m, l). It is
%   evaluated with PAGEMTIMES, one Nkx-by-L times L-by-L product per readout,
%   which avoids materialising the [Ntrajs, Nkx, L, L] broadcast intermediate.
%
%   DELETE(plan), or clearing the last reference, frees the cufinufft plans.
%
%   Example:
%     plan = aha_nufft_plan(traj, config);          % once per trajectory
%     for it = 1:n_iter
%         AhAx = plan.apply(x, SEs, phi2);          % every application
%     end
%
%   See also AHA_CUFINUFFT, MR_AHA, AHA_TOEPLITZ, CALC_TOEP, CUFINUFFT_PLAN.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    properties (SetAccess = private)
        traj_type   % '2d', '3d' or 'stack'
        L           % number of ranks
        img_dim     % image grid, [Nx Ny] or [Nx Ny Nz]
        Ntrajs      % number of readouts
        Nkx         % samples per readout
    end

    properties (Access = private)
        backend = 'cuda'
        ntr         % transforms per NUFFT call
        t1          % type-1 cufinufft plan
        t2          % type-2 cufinufft plan
        zmod        % stack: z modulation that centres the z FFT
        py          % Metal: the Python mr_aha.aha_nufft_plan
    end

    methods
        function obj = aha_nufft_plan(traj, config, varargin)
            obj.traj_type = lower(config.traj_type);
            obj.L         = double(config.L);

            if strcmp(mr_backend(), 'metal')
                % Apple silicon: the Python reference operator (MLX + mlx-nufft).
                if isa(traj, 'gpuArray'), traj = gather(traj); end
                mr = mr_python();
                obj.backend = 'metal';
                obj.py = mr.aha_nufft_plan(mr_py_config(config), ...
                                           mr_to_numpy(single(traj), 3));
                return
            end

            if exist('cufinufft_plan', 'file') ~= 2
                error('MR_AHA:nufft:noCufinufft', ...
                      ['cufinufft''s MATLAB interface is not on the path. Build it ' ...
                       'with -DMR_AHA_BUILD_CUFINUFFT_MATLAB=ON and addpath the ' ...
                       'folder holding cufinufft.mexa64.']);
            end

            opts = struct('floatprec', 'single', 'modeord', 0, 'upsampfac', 2.0);
            o1 = struct();
            o2 = struct();
            for i = 1:2:numel(varargin)
                switch varargin{i}
                    case 'type1_opts', o1 = varargin{i+1};
                    case 'type2_opts', o2 = varargin{i+1};
                    otherwise,         opts.(varargin{i}) = varargin{i+1};
                end
            end
            opts1 = mr_merge(opts, o1);
            opts2 = mr_merge(opts, o2);

            if ~isa(traj, 'gpuArray'), traj = gpuArray(traj); end
            traj = single(traj);

            n_dim = double(config.n_dim(:)).';
            tol   = 1e-6;
            if isfield(config, 'eps') && ~isempty(config.eps), tol = config.eps; end

            obj.Ntrajs = size(traj, 1);
            obj.Nkx    = size(traj, 2);
            switch obj.traj_type
                case '2d'
                    obj.img_dim = n_dim(1:2);
                    modes       = n_dim(1:2);
                    obj.ntr     = obj.L;
                    ndim        = 2;
                case '3d'
                    obj.img_dim = n_dim(1:3);
                    modes       = n_dim(1:3);
                    obj.ntr     = obj.L;
                    ndim        = 3;
                case 'stack'
                    % kz is Cartesian: a 2D NUFFT for every kz plane of every rank.
                    obj.img_dim = n_dim(1:3);
                    modes       = n_dim(1:2);
                    obj.ntr     = obj.L * n_dim(3);
                    ndim        = 2;
                otherwise
                    error('MR_AHA:nufft:trajType', ...
                          'config.traj_type must be ''2D'', ''3D'' or ''stack''.');
            end
            if size(traj, 3) ~= ndim
                error('MR_AHA:nufft:traj', 'traj must be [Ntrajs, Nkx, %d].', ndim);
            end

            % The points are ordered with the trajectory index fastest, matching
            % the [Ntrajs, Nkx] layout of the k-space data and of phi2.
            xj = reshape(traj(:, :, 1), [], 1);
            yj = reshape(traj(:, :, 2), [], 1);
            if ndim == 3, zj = reshape(traj(:, :, 3), [], 1); else, zj = []; end

            obj.t2 = cufinufft_plan(2, modes, -1, obj.ntr, tol, opts2);
            obj.t2.setpts(xj, yj, zj);
            obj.t1 = cufinufft_plan(1, modes, +1, obj.ntr, tol, opts1);
            obj.t1.setpts(xj, yj, zj);

            % Stack: image-domain modulation exp(2i*pi*c*n/Nz), c = floor(Nz/2),
            % which turns a plain FFT along z into the centred one. The
            % k-dependent phases of the centred transform cancel between the
            % forward and adjoint directions because phi2 is diagonal in kz, so
            % this is all that is needed; APPLY folds it into the coil map.
            if strcmp(obj.traj_type, 'stack')
                Nz = n_dim(3);
                obj.zmod = gpuArray(reshape(single(exp(2i * pi * floor(Nz / 2) * ...
                                            (0:Nz-1) / Nz)), 1, 1, Nz));
            end
        end

        function AhAx = apply(obj, x, SEs, phi2)
            %APPLY  AhAx = plan.apply(x, SEs, phi2); see AHA_NUFFT_PLAN.
            if strcmp(obj.backend, 'metal')
                % Apple silicon: Python reference operator, ordinary arrays.
                cs = @(a) complex(single(gather(a)));
                stack = strcmp(obj.traj_type, 'stack');
                AhAx = mr_from_numpy(obj.py.apply(mr_to_numpy(cs(x)), ...
                                                  mr_to_numpy(cs(SEs)), ...
                                                  mr_to_numpy(cs(phi2), 3 + stack)));
                AhAx = reshape(AhAx, size(x));
                return
            end

            L      = obj.L;
            img    = obj.img_dim;
            Ntrajs = obj.Ntrajs;
            Nkx    = obj.Nkx;
            nvox   = prod(img);
            stack  = strcmp(obj.traj_type, 'stack');

            % Host inputs are moved to the GPU (cufinufft's MATLAB interface
            % accepts gpuArrays only), as AHA_TOEPLITZ does.
            x    = reshape(gpuArray(single(x)), [img, L]);
            nc   = numel(SEs) / nvox;
            SEs  = reshape(gpuArray(single(SEs)), [nvox, nc]);
            phi2 = gpuArray(single(phi2));

            % Rearrange the Gram blocks once into L-by-L pages so that
            % pagemtimes(k_page, P_page) = k_page * Phi2, with the rank
            % coefficients of each sample along the rows of k_page.
            if stack
                Nz = img(3);
                P  = permute(reshape(phi2, [Ntrajs, Nz, L, L]), [3 4 1 2]); % [m l t z]
            else
                P  = permute(reshape(phi2, [Ntrajs, L, L]), [2 3 1]);       % [m l t]
            end

            AhAx = zeros([img, L], 'like', x);
            for c = 1:nc
                % 1. Coil sensitivity, broadcast over the ranks. For a stack it
                %    also carries the z modulation that centres the z FFT, so
                %    no fftshift copies are needed.
                se = reshape(SEs(:, c), img);
                if stack, se = se .* obj.zmod; end
                im = x .* se;

                % 2. Stack only: unnormalised FFT along z.
                if stack
                    im = fft(im, [], 3);
                end

                % 3. Type-2 NUFFT: [Ntrajs*Nkx, ntr]. Only numel(im) is
                %    checked, and [Nx,Ny,Nz,L] is already [Nx,Ny,Nz*L] in
                %    memory for a stack.
                k = obj.t2.execute(im);
                clear im

                % 4. Block-diagonal multiplication, one L-by-L block per readout.
                if stack
                    k = permute(reshape(k, [Ntrajs, Nkx, Nz, L]), [2 4 1 3]);  % [kx m t z]
                    k = pagemtimes(k, P);                                       % [kx l t z]
                    k = permute(k, [3 1 4 2]);                                  % [t kx z l]
                else
                    k = permute(reshape(k, [Ntrajs, Nkx, L]), [2 3 1]);         % [kx m t]
                    k = pagemtimes(k, P);                                       % [kx l t]
                    k = permute(k, [3 1 2]);                                    % [t kx l]
                end

                % 5. Type-1 NUFFT back onto the image grid.
                im = reshape(obj.t1.execute(reshape(k, [], obj.ntr)), [img, L]);
                clear k

                % 6. Stack only: unnormalised inverse FFT along z.
                if stack
                    im = ifft(im, [], 3) * Nz;
                end

                % 7. Adjoint coil sensitivity (and z demodulation), accumulated.
                AhAx = AhAx + im .* conj(se);
                clear im se
            end
        end

        function AhAx = execute(obj, x, SEs, phi2)
            %EXECUTE  Alias of APPLY.
            AhAx = obj.apply(x, SEs, phi2);
        end

        function delete(obj)
            %DELETE  Free the cufinufft plans.
            if ~isempty(obj.t1) && isvalid(obj.t1), delete(obj.t1); end
            if ~isempty(obj.t2) && isvalid(obj.t2), delete(obj.t2); end
            if ~isempty(obj.py), obj.py.close(); end
        end
    end
end

function o = mr_merge(o, extra)
%MR_MERGE  Overlay the fields of struct extra onto struct o.
    f = fieldnames(extra);
    for i = 1:numel(f)
        o.(f{i}) = extra.(f{i});
    end
    o.floatprec = 'single';
    o.modeord   = 0;
end
