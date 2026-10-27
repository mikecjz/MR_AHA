function plan = mr_nufft_plan(traj, config, varargin)
%MR_NUFFT_PLAN  Build reusable cufinufft plans for AHA_CUFINUFFT.
%
%   plan = MR_NUFFT_PLAN(traj, config)
%   plan = MR_NUFFT_PLAN(traj, config, 'Name', Value, ...)
%
%   Creates one type-2 (image -> k-space, sign -1) and one type-1 (k-space ->
%   image, sign +1) single-precision cufinufft plan for the trajectory and
%   sorts the non-uniform points once. Pass the result to AHA_CUFINUFFT so an
%   iterative reconstruction pays the planning cost only once.
%
%   Each plan transforms all L ranks of one coil in a single batched call; for
%   a 'stack' trajectory the batch is L*Nz, one 2D transform per kz plane and
%   rank.
%
%   INPUTS
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
%   OUTPUT
%     plan    struct with the two cufinufft_plan handles (.t1, .t2) and the
%             geometry AHA_CUFINUFFT needs.
%
%   See also AHA_CUFINUFFT, CUFINUFFT_PLAN.

%   Author: Junzhou Chen (junzhou@chen.engineer)

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
    opts.floatprec = 'single';
    opts.modeord   = 0;
    opts1 = mr_merge(opts, o1);
    opts2 = mr_merge(opts, o2);

    if ~isa(traj, 'gpuArray'), traj = gpuArray(traj); end
    traj = single(traj);

    n_dim = double(config.n_dim(:)).';
    L     = double(config.L);
    tol   = 1e-6;
    if isfield(config, 'eps') && ~isempty(config.eps), tol = config.eps; end

    plan.traj_type = lower(config.traj_type);
    plan.L         = L;
    plan.Ntrajs    = size(traj, 1);
    plan.Nkx       = size(traj, 2);
    switch plan.traj_type
        case '2d'
            plan.img_dim = n_dim(1:2);
            modes        = n_dim(1:2);
            plan.ntr     = L;
            ndim         = 2;
        case '3d'
            plan.img_dim = n_dim(1:3);
            modes        = n_dim(1:3);
            plan.ntr     = L;
            ndim         = 3;
        case 'stack'
            % kz is Cartesian: a 2D NUFFT for every kz plane of every rank.
            plan.img_dim = n_dim(1:3);
            modes        = n_dim(1:2);
            plan.ntr     = L * n_dim(3);
            ndim         = 2;
        otherwise
            error('MR_AHA:nufft:trajType', ...
                  'config.traj_type must be ''2D'', ''3D'' or ''stack''.');
    end
    if size(traj, 3) ~= ndim
        error('MR_AHA:nufft:traj', 'traj must be [Ntrajs, Nkx, %d].', ndim);
    end

    % The points are ordered with the trajectory index fastest, matching the
    % [Ntrajs, Nkx] layout of the k-space data and of phi2.
    xj = reshape(traj(:, :, 1), [], 1);
    yj = reshape(traj(:, :, 2), [], 1);
    if ndim == 3, zj = reshape(traj(:, :, 3), [], 1); else, zj = []; end

    plan.t2 = cufinufft_plan(2, modes, -1, plan.ntr, tol, opts2);
    plan.t2.setpts(xj, yj, zj);
    plan.t1 = cufinufft_plan(1, modes, +1, plan.ntr, tol, opts1);
    plan.t1.setpts(xj, yj, zj);

    % Stack: image-domain modulation exp(2i*pi*c*n/Nz), c = floor(Nz/2),
    % which turns a plain FFT along z into the centred one. The k-dependent
    % phases of the centred transform cancel between the forward and adjoint
    % directions because phi2 is diagonal in kz, so this is all that is
    % needed; AHA_CUFINUFFT folds it into the coil sensitivity.
    plan.zmod = [];
    if strcmp(plan.traj_type, 'stack')
        Nz = n_dim(3);
        plan.zmod = gpuArray(reshape(single(exp(2i * pi * floor(Nz / 2) * ...
                                        (0:Nz-1) / Nz)), 1, 1, Nz));
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
