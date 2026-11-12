classdef mr_aha < handle
%MR_AHA  The forward normal operator A'A as a reusable object.
%
%   op = MR_AHA.CREATE_OP(SEs, phi2, traj, config)
%   op = MR_AHA.CREATE_OP(SEs, phi2, traj, config, 'Name', Value, ...)
%
%   AhAx = op.apply(x);      % or  AhAx = op * x
%   s    = op.info();
%   Om   = op.kernel();      % Toeplitz method: the kernel phi2_toep
%   delete(op);              % or  clear op
%
%   Builds the low-rank non-Cartesian normal operator x -> A'*A*x once, from
%   the Gram blocks phi2, the trajectory traj and the sensitivity maps SEs,
%   and applies it as often as needed. The object holds everything it needs
%   (the maps and either the Toeplitz kernel or the NUFFT plans), so APPLY
%   only takes x. CREATE_OP is the only way to build one; the constructor is
%   not public.
%
%   The class has the name of the Python package, so the operator is created
%   by the same call in both languages: MR_AHA.CREATE_OP here,
%   mr_aha.create_op in Python (which takes the configuration first:
%   mr_aha.create_op(cfg, SEs, phi2, traj, ...), and returns an AhAOp).
%
%   Two methods evaluate the same operator (to within the NUFFT tolerance
%   config.eps):
%     'toeplitz' (default)  The Toeplitz embedding. The kernel phi2_toep is
%                computed at creation (as CALC_TOEP does) and every
%                application is, per coil, a zero-padded FFT, the
%                block-diagonal multiplication by phi2_toep and an inverse
%                FFT (as AHA_TOEPLITZ does). No non-uniform FFT per
%                application.
%     'nufft'    The direct route: per coil a type-2 NUFFT (image to the
%                trajectory), the Gram multiplication of every readout's
%                samples by its block phi2 and a type-1 NUFFT back, with
%                cufinufft inside the library. No kernel, so far less memory
%                but more time per application. Masked configurations are
%                rejected (there is no kernel to mask).
%   AHA_CUFINUFFT is an independent implementation of the 'nufft' route
%   (finufft's own MATLAB interface, no library code); the tests use it to
%   validate this object.
%
%   Whether the operator uses one GPU or several is decided at creation
%   ('devices'). A'A is a sum over coils, so on several GPUs each GPU
%   applies a contiguous range of whole coils and the partial sums are added
%   on the primary GPU; x is copied to every GPU at each APPLY. With the
%   Toeplitz method every GPU holds the whole kernel, so several GPUs make
%   the operator faster but do not raise the largest problem size; 'toep_dist'
%   chooses how the copies are produced.
%
%   The PRIMARY GPU is MATLAB's current gpuDevice when the operator is
%   created. It holds the inputs and outputs (gpuArrays) and always takes
%   part in the computation. MATLAB's current gpuDevice is never changed by
%   MR_AHA, although other GPUs are used internally.
%
%   INPUTS
%     SEs     complex single sensitivity maps, one per coil, shared by every
%             rank (see AHA_TOEPLITZ):
%               [Nx, Ny, Ncoils]        for '2D'
%               [Nx, Ny, Nz, Ncoils]    for '3D' and 'stack'
%     phi2    complex single Gram blocks Phi*Phi' (MR-Multitasking
%             convention, see CALC_TOEP):
%               [Ntrajs, L, L]          for '2D' and '3D'
%               [Ntrajs, Nkz, L, L]     for 'stack' (centred kz order)
%     traj    real single trajectory [Ntrajs, Nkx, ndim] in radians, as for
%             CALC_TOEP (ndim = 2 for '2D'/'stack', 3 for '3D').
%     config  struct from MR_DEFAULT_CONFIG, as for AHA_TOEPLITZ and
%             CALC_TOEP. config.n_coils defaults to the last dimension of
%             SEs. config.gpu_device_id is ignored: the primary GPU is always
%             the current gpuDevice.
%     Arrays that are not gpuArrays, not single or not complex are converted
%     like AHA_TOEPLITZ does; passing complex single gpuArrays (real single
%     for traj) avoids copies. phi2 and traj are only needed during
%     CREATE_OP: the operator keeps its own (transformed) copies, so they
%     may be cleared afterwards.
%
%   NAME-VALUE PAIRS
%     'method'         'toeplitz' (default) or 'nufft', see above.
%     'devices'        GPUs to use:
%                        []      (default) the current gpuDevice only;
%                        'all'   every visible GPU;
%                        vector  those GPUs, as 1-based gpuDevice indices
%                                (the numbering of gpuDevice(k) and
%                                gpuDeviceTable). The primary GPU always
%                                takes part, listed or not; duplicates are
%                                ignored.
%                      The library may use fewer GPUs than given: with C
%                      coils and G GPUs it uses the smallest number with the
%                      same ceil(C/G) coils per GPU (e.g. 12 coils on 5 GPUs
%                      use 4), and only the primary GPU for problems too
%                      small to benefit (fewer than 2^22 upsampled values per
%                      coil; set the environment variable
%                      MR_AHA_MGPU_MIN_WORK with SETENV before creating the
%                      operator to change this, 0 disables the check). INFO
%                      reports the choice.
%     'mask'           'toeplitz' with config.is_masked only: 1-based linear
%                      indices into the centred upsampled grid
%                      config.phi2_dim (as for CALC_TOEP and AHA_TOEPLITZ).
%                      When omitted, the mask is derived from traj with
%                      CALC_MASK, as CALC_TOEP does.
%     'toep_dist'      'toeplitz' on several GPUs: how every GPU gets its
%                      copy of the kernel. The (L^2+L)/2 triangular
%                      components of phi2_toep are independent.
%                        'auto'       (default) 'split' or 'replicate',
%                                     chosen by a cost model measured at
%                                     creation (a timed first chunk on every
%                                     GPU and a bandwidth probe);
%                        'split'      every GPU computes a share of the
%                                     components into its own copy and the
%                                     shares are exchanged, overlapped with
%                                     the computation; the copies are
%                                     identical;
%                        'replicate'  every GPU computes the whole kernel,
%                                     without transfers (best without fast
%                                     peer access); the copies agree to
%                                     cufinufft's run-to-run noise (~1e-7);
%                        'broadcast'  the primary computes the kernel and
%                                     copies it to the other GPUs (slowest,
%                                     for comparison).
%                      The environment variable MR_AHA_TOEP_SPLIT
%                      ('auto', 'split', 'replicate', 'broadcast'; SETENV
%                      before creating) overrides this option. Ignored on
%                      one GPU. INFO reports the choice (toep_source).
%     'share_primary'  true (default) or false. When true the primary GPU
%                      uses SEs and the mask in place instead of copying
%                      them, which saves one copy of the maps (memory as for
%                      AHA_TOEPLITZ). The operator then keeps references to
%                      these arrays until it is deleted, so their memory is
%                      not released by clearing the caller's variables. The
%                      caller's variables can still be changed freely:
%                      gpuArrays are copy-on-write, so an assignment creates
%                      new storage and the operator keeps the old values.
%                      The other GPUs always receive copies.
%     'nufft_opts'     'nufft' only: struct of cufinufft options,
%                        .upsampfac   upsampling factor (default 2, the
%                                     setting of AHA_CUFINUFFT)
%                        .gpu_method  cufinufft's gpu_method (1: non-uniform
%                                     points driven, 2: shared memory,
%                                     for both transforms; default 0:
%                                     method 1 for type 1, cufinufft's
%                                     choice for type 2)
%                      Default struct() (all defaults). A non-empty struct
%                      with the Toeplitz method is an error.
%
%   METHODS
%     AhAx = apply(op, x)  A'A x. x is complex single [Nx, Ny, L] ('2D') or
%                          [Nx, Ny, Nz, L] ('3D', 'stack') on the primary GPU
%                          (host arrays are moved there); AhAx is a gpuArray
%                          of the same shape. op * x is the same.
%     Om = kernel(op)      'toeplitz' only: a copy of the kernel phi2_toep the
%     Om = kernel(op, k)   operator holds on the primary GPU (or on its k-th
%                          GPU, the k-th entry of info().devices), as a
%                          complex single gpuArray on the primary GPU shaped
%                          like CALC_TOEP's output, e.g. to save it. Costs one
%                          more kernel's memory. A 'nufft' operator raises
%                          'MR_AHA:mr_aha:noKernel'.
%     s = info(op)         Struct describing how the operator was built:
%                            method              'toeplitz' or 'nufft'
%                            n_devices           GPUs actually used
%                            devices             their gpuDevice indices
%                                                (1-based), primary first
%                            cuda_devices        the same as CUDA ordinals
%                            primary             gpuDevice index of the
%                                                primary GPU
%                            coil_begin          first coil (1-based) of each
%                                                GPU
%                            coil_count          number of coils of each GPU
%                            group               'toeplitz': coils per group
%                                                within each GPU's plan;
%                                                'nufft': 1
%                            n_split_coils       'toeplitz': coils split by
%                                                rank across all GPUs
%                                                (MR_AHA_MGPU_SPLIT)
%                            peer_access         direct peer access enabled
%                                                between the primary and every
%                                                other GPU used
%                            peer_bandwidth_gbs  measured bandwidth or 0 (only
%                                                measured for the remainder
%                                                split's cost model)
%                            n_coils             number of coils
%                            share_primary       see above
%                            valid               false once the primary GPU
%                                                was reset or deselected
%                            toep_source         how phi2_toep was obtained:
%                                                'broadcast' (the primary
%                                                computed it, and copied it to
%                                                any other GPU; always the case
%                                                on one GPU), 'split',
%                                                'replicate' (see
%                                                'toep_dist'), or 'none'
%                                                ('nufft')
%                            toep_devices        GPUs that computed it
%                            toep_ms             time of computing and
%                                                distributing it (ms)
%                            toep_component_ms   GPU time per component (ms)
%                            toep_gather_gbs     measured concurrent receive
%                                                bandwidth (GB/s)
%                            create_ms           time of the whole CREATE_OP
%                                                in the library (ms)
%     tf = is_alive(op)    True until the operator has been deleted.
%     delete(op)           Release all GPU memory of the operator on every
%                          GPU. Also happens automatically when the last
%                          reference to op is cleared.
%
%   DEVICE CHANGES
%     The operator is bound to its primary GPU. reset(gpuDevice) and
%     selecting another GPU with gpuDevice(k) free the operator's data on
%     the primary GPU (MATLAB destroys the GPU context); APPLY then raises
%     the error 'MR_AHA:mr_aha:deviceReset' (or ':deviceChanged' while a
%     different GPU is selected) instead of computing. Delete the operator
%     and create a new one; deleting it then warns
%     ('MR_AHA:mr_aha:staleDestroy') and releases what it holds on the
%     other GPUs. Delete operators before reset(gpuDevice) to avoid the
%     warning.
%
%   CLEAR MEX
%     While any operator exists the gateway MEX file mr_aha_mex is locked
%     (mexLock), so 'clear mex' / 'clear functions' do not unload it and the
%     existing operators keep working. The lock is released when the last
%     operator is deleted. 'clear all' deletes the operators first.
%     Operators still alive when MATLAB exits are released.
%
%   BACKENDS
%     CUDA   gpuArrays; everything runs in the library (MEX file
%            mr_aha_mex).
%     Metal  (Apple silicon, see MR_BACKEND) ordinary arrays; one GPU only
%            ('devices' must be [], 1 or 'all'). The operator is the Python
%            mr_aha.create_op object (see MR_PYTHON): the Toeplitz kernel is
%            computed with mlx-nufft and applied by the library, the NUFFT
%            method runs in MLX. The operator works on its own copies of SEs
%            and the mask, so 'share_primary' saves no memory there.
%            'toep_dist' and 'nufft_opts' are accepted and validated but have
%            no effect, and the reset / device-change checks do not apply.
%
%   EXAMPLE
%     gpuDevice(1);                                   % primary GPU
%     config = mr_default_config('stack', [256 256 64], 16);
%     op = mr_aha.create_op(SEs, phi2, traj, config);
%     AhAx = op * x;                % = AhA_Toeplitz(x, SEs, calc_toep(phi2, traj, config), config)
%     disp(op.info())
%     delete(op)
%
%     % The same operator on four GPUs, and the NUFFT route on all GPUs:
%     op4 = mr_aha.create_op(SEs, phi2, traj, config, 'devices', 1:4);
%     opn = mr_aha.create_op(SEs, phi2, traj, config, 'method', 'nufft', ...
%                               'devices', 'all');
%
%   See also AHA_TOEPLITZ, CALC_TOEP, CALC_MASK, AHA_CUFINUFFT,
%   MR_DEFAULT_CONFIG, GPUDEVICE.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    properties (SetAccess = private)
        %CONFIG  Configuration the operator was built with (with .n_coils
        %   and, for a masked operator, .num_masked filled in).
        Config
        %METHOD  'toeplitz' or 'nufft'.
        Method = ''
    end

    properties (Access = private, Transient)
        %HANDLE  uint64 operator handle of mr_aha_mex (CUDA); 0 once
        %   destroyed.
        Handle = uint64(0)
        %PYOP  The Python mr_aha.AhAOp object (Metal); [] once destroyed.
        PyOp = []
        %SHAREPRIMARY  The 'share_primary' argument (reported by INFO on
        %   Metal).
        SharePrimary = true
    end

    methods
        function op = mr_aha(varargin)
            %MR_AHA  Not public: use MR_AHA.CREATE_OP.
            if nargin ~= 1 || ~ischar(varargin{1}) || ...
                    ~strcmp(varargin{1}, mr_op_token())
                error('MR_AHA:mr_aha:useCreateOp', ...
                      ['mr_aha objects are created with ' ...
                       'mr_aha.create_op(SEs, phi2, traj, config, ...); ' ...
                       'the constructor is not public.']);
            end
        end

        function AhAx = apply(op, x)
            %APPLY  A'A x; see the class help.
            op.check_alive();
            if ~isempty(op.PyOp)
                % Metal: the Python operator on ordinary arrays.
                x = complex(single(gather(x)));
                y = op.PyOp.apply(mr_to_numpy(x, mr_op_n_spatial(op.Config) + 1));
                AhAx = reshape(mr_from_numpy(y), size(x));
                return
            end
            x = mr_op_complex_single(x, 'x');
            AhAx = mr_aha_mex('apply', op.Handle, x);
        end

        function AhAx = mtimes(op, x)
            %MTIMES  op * x is the same as op.apply(x).
            if ~isa(op, 'mr_aha')
                error('MR_AHA:mr_aha:mtimes', ...
                      'Only op * x is supported (the operator on the left).');
            end
            AhAx = op.apply(x);
        end

        function s = info(op)
            %INFO  How the operator was built; see the class help.
            op.check_alive();
            if ~isempty(op.PyOp)
                s = mr_op_info_metal(op);
            else
                s = mr_aha_mex('info', op.Handle);
            end
            s.cuda_devices = s.devices;
            s.devices = s.devices + 1;            % gpuDevice indices
            s.primary = s.primary + 1;
            s.coil_begin = s.coil_begin + 1;      % 1-based coils
            s = orderfields(s, {'method', 'n_devices', 'devices', ...
                'cuda_devices', 'primary', 'coil_begin', 'coil_count', ...
                'group', 'n_split_coils', 'peer_access', ...
                'peer_bandwidth_gbs', 'n_coils', 'share_primary', 'valid', ...
                'toep_source', 'toep_devices', 'toep_ms', ...
                'toep_component_ms', 'toep_gather_gbs', 'create_ms'});
        end

        function Om = kernel(op, k)
            %KERNEL  Copy of the operator's phi2_toep (Toeplitz method); see
            %   the class help. k (default 1) selects the GPU, as an index
            %   into info().devices.
            op.check_alive();
            if nargin < 2 || isempty(k)
                k = 1;
            end
            if ~(isnumeric(k) && isscalar(k) && isreal(k) && k == round(k) && k >= 1)
                error('MR_AHA:mr_aha:kernel', ...
                      'k must be a positive integer (an index into info().devices).');
            end
            if strcmp(op.Method, 'nufft')
                error('MR_AHA:mr_aha:noKernel', ...
                      ['This operator uses the NUFFT method, which has no ' ...
                       'Toeplitz kernel.']);
            end
            if ~isempty(op.PyOp)
                if k ~= 1
                    error('MR_AHA:mr_aha:kernel', ...
                          'The Metal backend has a single GPU: k must be 1.');
                end
                Om = mr_from_numpy(op.PyOp.kernel(int64(0)));
                return
            end
            Om = mr_aha_mex('kernel', op.Handle, double(k) - 1);
        end

        function tf = is_alive(op)
            %IS_ALIVE  True until the operator has been deleted (independent
            %   of device resets; see INFO's .valid for that).
            tf = op.Handle ~= 0 || ~isempty(op.PyOp);
        end

        function delete(op)
            %DELETE  Release the operator on every GPU. Never throws.
            if op.Handle ~= 0
                h = op.Handle;
                op.Handle = uint64(0);
                try
                    mr_aha_mex('destroy', h);
                catch err
                    warning('MR_AHA:mr_aha:destroy', ...
                            'Releasing the operator failed: %s', err.message);
                end
            end
            if ~isempty(op.PyOp)
                p = op.PyOp;
                op.PyOp = [];
                try
                    p.close();
                catch err
                    warning('MR_AHA:mr_aha:destroy', ...
                            'Releasing the operator failed: %s', err.message);
                end
            end
        end
    end

    methods (Static)
        function op = create_op(SEs, phi2, traj, config, varargin)
            %CREATE_OP  Build the A'A operator from the Gram blocks; see the
            %   class help.
            %
            %   op = mr_aha.create_op(SEs, phi2, traj, config)
            %   op = mr_aha.create_op(..., 'method', 'toeplitz', ...
            %            'devices', [], 'mask', [], 'toep_dist', 'auto', ...
            %            'share_primary', true, 'nufft_opts', struct())
            narginchk(4, inf);
            % '_phi2_toep' (private, for tests): a finished kernel used
            % instead of computing one. Not a valid inputParser name.
            [varargin, phi2_toep] = mr_op_take(varargin, '_phi2_toep');

            p = inputParser();
            p.FunctionName = 'mr_aha.create_op';
            p.addParameter('method', 'toeplitz');
            p.addParameter('devices', []);
            p.addParameter('mask', []);
            p.addParameter('toep_dist', 'auto');
            p.addParameter('share_primary', true);
            p.addParameter('nufft_opts', struct());
            p.parse(varargin{:});
            opt = p.Results;

            % ---- Arguments independent of the backend ----------------------
            method = mr_op_choice(opt.method, {'toeplitz', 'nufft'}, ...
                                  'method', 'MR_AHA:mr_aha:method');
            toep_dist = mr_op_choice(opt.toep_dist, ...
                {'auto', 'split', 'replicate', 'broadcast'}, 'toep_dist', ...
                'MR_AHA:mr_aha:toepDist');
            nufft = strcmp(method, 'nufft');
            nopts = mr_op_nufft_opts(opt.nufft_opts, nufft);
            if ~(isscalar(opt.share_primary) && ...
                 (islogical(opt.share_primary) || isnumeric(opt.share_primary)))
                error('MR_AHA:mr_aha:sharePrimary', ...
                      'share_primary must be a logical scalar.');
            end
            share = logical(opt.share_primary);
            is_masked = isfield(config, 'is_masked') && config.is_masked;
            if nufft && is_masked
                error('MR_AHA:mr_aha:maskedNufft', ...
                      ['method ''nufft'' has no kernel to mask; use ' ...
                       'config.is_masked = false (or the Toeplitz method).']);
            end
            if nufft && ~isempty(phi2_toep)
                error('MR_AHA:mr_aha:phi2Toep', ...
                      '_phi2_toep is only used with method ''toeplitz''.');
            end

            op = mr_aha(mr_op_token());
            op.Method = method;
            op.SharePrimary = share;

            if strcmp(mr_backend(), 'metal')
                op.create_metal(SEs, phi2, traj, config, opt.devices, ...
                                opt.mask, toep_dist, share, nopts, phi2_toep);
                return
            end

            % ---- CUDA: same conversions as calc_toep.m and AhA_Toeplitz.m --
            devices = mr_op_devices(opt.devices);
            SEs  = mr_op_complex_single(SEs,  'SEs');
            % With a supplied kernel the Gram blocks and the trajectory are
            % not needed and may both be empty (passed on as empty).
            if isempty(phi2_toep) || ~isempty(phi2) || ~isempty(traj)
                phi2 = mr_op_complex_single(phi2, 'phi2');
                traj = mr_op_real_single(traj);
            end
            if ~isempty(phi2_toep)
                phi2_toep = mr_op_complex_single(phi2_toep, '_phi2_toep');
            end
            config = mr_op_fill_n_coils(config, SEs);
            mask = opt.mask;
            if is_masked && isempty(mask)
                if isempty(traj)
                    error('MR_AHA:mr_aha:missingMask', ...
                          ['A masked operator from a supplied kernel ' ...
                           'needs the mask (no trajectory to compute it).']);
                end
                mask = calc_mask(traj, config);   % as calc_toep.m does
            end
            [mask, config] = mr_op_mask(mask, config);

            mex_opts = struct('method', method, 'devices', devices, ...
                'toep_dist', toep_dist, 'share_primary', share, ...
                'nufft_upsampfac', nopts.upsampfac, ...
                'nufft_gpu_method', nopts.gpu_method);
            mex_opts.phi2_toep = phi2_toep;     % may be [] (not computed)
            op.Handle = mr_aha_mex('create', SEs, phi2, traj, config, ...
                                      mask, mex_opts);
            op.Config = config;
        end
    end

    methods (Access = private)
        function check_alive(op)
            %CHECK_ALIVE  Error if the operator was deleted or not created
            %   in this session (e.g. loaded from a MAT-file).
            if op.Handle == 0 && isempty(op.PyOp)
                error('MR_AHA:mr_aha:invalidHandle', ...
                      'This mr_aha has been deleted (or was loaded from a file).');
            end
        end

        function create_metal(op, SEs, phi2, traj, config, devices, mask, ...
                              toep_dist, share, nopts, phi2_toep)
            %CREATE_METAL  Metal backend: wrap Python's mr_aha.create_op.
            %
            %   One GPU only. Arrays become NumPy arrays in the Python
            %   package's row-major layout (MR_TO_NUMPY). The arrays the Python
            %   operator keeps (SEs, the mask, a supplied kernel and, for
            %   'nufft', phi2) are passed as private copies, so later
            %   in-place changes of the caller's MATLAB arrays cannot reach
            %   the operator.
            % 'all' resolves to the one GPU, so that the same script runs on
            % CUDA and Metal (as in Python).
            if ~(isempty(devices) || (ischar(devices) || isstring(devices)) ...
                     && strcmpi(devices, 'all') || ...
                 (isnumeric(devices) && isscalar(devices) && devices == 1))
                error('MR_AHA:mr_aha:devices', ...
                      ['The metal backend has a single GPU: devices must be ' ...
                       '[], 1 or ''all''.']);
            end
            host = @(a) complex(single(gather(a)));
            SEs  = host(SEs);
            phi2 = host(phi2);
            traj = single(gather(traj));
            if ~isreal(traj)
                error('MR_AHA:mr_aha:complexTraj', 'traj must be real-valued.');
            end
            config = mr_op_fill_n_coils(config, SEs);
            nsp = mr_op_n_spatial(config);
            stack = strcmpi(config.traj_type, 'stack');

            mask_py = py.None;
            if isfield(config, 'is_masked') && config.is_masked
                if isempty(mask)
                    mask = calc_mask(traj, config);
                end
                mask = int64(gather(mask(:)));
                config.num_masked = numel(mask);
                mask_py = mr_to_numpy(mask - 1).ravel().copy();   % 0-based, 1-D
            end

            mr = mr_python();
            kw = py.dict(pyargs('method', op.Method, 'devices', py.None, ...
                'toep_dist', toep_dist, 'share_primary', share));
            if ~isempty(fieldnames(nopts.given))
                nd = py.dict();
                for f = fieldnames(nopts.given).'
                    v = nopts.given.(f{1});
                    if strcmp(f{1}, 'gpu_method'), v = int64(v); end
                    py.operator.setitem(nd, f{1}, v);
                end
                py.operator.setitem(kw, 'nufft_opts', nd);
            end
            if ~isempty(phi2_toep)
                Om = host(phi2_toep);
                py.operator.setitem(kw, '_phi2_toep', ...
                                    mr_to_numpy(Om, ndims(Om)).copy());
            end
            phi2_py = mr_to_numpy(phi2, 3 + stack);
            if strcmp(op.Method, 'nufft')
                phi2_py = phi2_py.copy();   % kept by the NUFFT operator
            end
            args = py.tuple({mr_py_config(config), ...
                             mr_to_numpy(SEs, nsp + 1).copy(), phi2_py, ...
                             mr_to_numpy(traj, 3), mask_py});
            % f(*args, **kwargs): MATLAB has no keyword splat, and pyargs
            % cannot name the private '_phi2_toep' keyword.
            splat = py.eval('lambda f, a, k: f(*a, **k)', py.dict());
            op.PyOp = splat(mr.create_op, args, kw);
            op.Config = config;
        end
    end
end

% ----------------------------------------------------------------------
% Local functions
% ----------------------------------------------------------------------

function t = mr_op_token()
%MR_OP_TOKEN  Argument by which CREATE_OP calls the non-public constructor.
    t = 'mr_aha.create_op';
end

function [args, value] = mr_op_take(args, name)
%MR_OP_TAKE  Remove the name-value pair NAME from ARGS; VALUE is [] when
%   absent.
    value = [];
    k = 1;
    while k < numel(args)
        if (ischar(args{k}) || isstring(args{k})) && strcmpi(args{k}, name)
            value = args{k + 1};
            args(k:k + 1) = [];
        else
            k = k + 2;
        end
    end
end

function v = mr_op_choice(v, choices, name, id)
%MR_OP_CHOICE  Validate a string option (case-insensitive); returns it in
%   lower case.
    if ~((ischar(v) && (isrow(v) || isempty(v))) || (isstring(v) && isscalar(v))) ...
            || ~any(strcmpi(v, choices))
        error(id, '%s must be one of %s.', name, ...
              strjoin(strcat('''', choices, ''''), ', '));
    end
    v = lower(char(v));
end

function n = mr_op_nufft_opts(s, nufft)
%MR_OP_NUFFT_OPTS  Validate 'nufft_opts'; returns .upsampfac and
%   .gpu_method (0 for the library defaults) and .given (the fields set).
    if isempty(s) && ~isstruct(s)
        s = struct();
    end
    if ~(isstruct(s) && isscalar(s))
        error('MR_AHA:mr_aha:nufftOpts', 'nufft_opts must be a scalar struct.');
    end
    f = fieldnames(s);
    if ~nufft && ~isempty(f)
        error('MR_AHA:mr_aha:nufftOpts', ...
              'nufft_opts is only used with method ''nufft''.');
    end
    bad = setdiff(f, {'upsampfac', 'gpu_method'});
    if ~isempty(bad)
        error('MR_AHA:mr_aha:nufftOpts', ...
              'Unknown nufft_opts field(s) %s; accepted: upsampfac, gpu_method.', ...
              strjoin(bad, ', '));
    end
    n = struct('upsampfac', 0, 'gpu_method', 0, 'given', s);
    for k = 1:numel(f)
        v = s.(f{k});
        if ~(isnumeric(v) && isscalar(v) && isreal(v))
            error('MR_AHA:mr_aha:nufftOpts', ...
                  'nufft_opts.%s must be a real scalar.', f{k});
        end
        n.(f{k}) = double(v);
    end
end

function devices = mr_op_devices(devices)
%MR_OP_DEVICES  'devices' argument -> what mr_aha_mex expects: [] (the
%   primary only), 'all', or an int32 row of CUDA ordinals (from 1-based
%   gpuDevice indices).
    if isempty(devices) && ~ischar(devices) && ~isstring(devices)
        devices = int32([]);
        return
    end
    if (ischar(devices) || isstring(devices))
        if strcmpi(devices, 'all')
            devices = 'all';
            return
        end
        error('MR_AHA:mr_aha:devices', ...
              'devices must be [], ''all'' or a vector of gpuDevice indices.');
    end
    n_gpu = gpuDeviceCount();
    if ~(isnumeric(devices) && isreal(devices) && isvector(devices))
        error('MR_AHA:mr_aha:devices', ...
              'devices must be [], ''all'' or a vector of gpuDevice indices.');
    end
    devices = double(devices(:)).';
    if any(devices ~= round(devices)) || any(devices < 1) || any(devices > n_gpu)
        error('MR_AHA:mr_aha:devices', ...
              'devices must be gpuDevice indices between 1 and %d.', n_gpu);
    end
    devices = int32(devices - 1);
end

function [mask, config] = mr_op_mask(mask, config)
%MR_OP_MASK  Mask handling identical to AhA_Toeplitz.m: required in masked
%   mode, converted to 0-based int64 indices (config.num_masked filled in).
    if isempty(mask)
        if isfield(config, 'is_masked') && config.is_masked
            error('MR_AHA:mr_aha:missingMask', ...
                  'config.is_masked is true, so a mask must be supplied.');
        end
        mask = [];
        return
    end
    % The library indexes from 0; MATLAB indexes from 1. The subtraction is
    % done in double (exact for any grid below 2^53 points) because gpuArray
    % does not support 64-bit integer arithmetic in every MATLAB release
    % (R2022b: "Input arguments of type 'int64' are not supported for
    % 'minus'").
    mask = int64(double(gpuArray(mask(:))) - 1);
    config.num_masked = numel(mask);
end

function a = mr_op_complex_single(a, name)
%MR_OP_COMPLEX_SINGLE  Coerce an array to a complex single gpuArray (the
%   CUDA branch of AhA_Toeplitz.m's conversion). No copy when the input
%   already has that form.
    if ~isa(a, 'gpuArray')
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
        error('MR_AHA:mr_aha:emptyInput', '%s must not be empty.', name);
    end
end

function traj = mr_op_real_single(traj)
%MR_OP_REAL_SINGLE  Coerce the trajectory to a real single gpuArray (as
%   calc_toep.m does).
    if ~isa(traj, 'gpuArray'), traj = gpuArray(traj); end
    if ~strcmp(underlyingType(traj), 'single'), traj = single(traj); end
    if ~isreal(traj)
        error('MR_AHA:mr_aha:complexTraj', 'traj must be real-valued.');
    end
end

function n = mr_op_n_spatial(config)
%MR_OP_N_SPATIAL  2 for '2D', 3 for '3D' and 'stack'.
    if strcmpi(config.traj_type, '2D'), n = 2; else, n = 3; end
end

function config = mr_op_fill_n_coils(config, SEs)
%MR_OP_FILL_N_COILS  Derive .n_coils from the trailing dimension of SEs, as
%   AhA_Toeplitz.m does. An explicit value already in config is kept.
    if isfield(config, 'n_coils') && ~isempty(config.n_coils)
        return
    end
    n_spatial = mr_op_n_spatial(config);
    sz = size(SEs);
    sz(end+1:n_spatial + 1) = 1;   % restore MATLAB's dropped trailing ones
    config.n_coils = sz(n_spatial + 1);
end

function s = mr_op_info_metal(op)
%MR_OP_INFO_METAL  INFO on Metal: the Python operator's info dictionary in
%   the (0-based) form mr_aha_mex('info', h) returns on CUDA.
    d = struct(op.PyOp.info);
    f = fieldnames(d);
    for k = 1:numel(f)
        d.(f{k}) = mr_op_from_py(d.(f{k}));
    end
    s = struct();
    s.method = d.method;
    s.n_devices = numel(d.devices);
    s.devices = d.devices;
    s.coil_begin = d.coil_begin;
    s.coil_count = d.coil_count;
    s.group = d.group;
    s.n_split_coils = d.n_split_coils;
    s.peer_access = logical(d.peer_access);
    s.peer_bandwidth_gbs = d.peer_bandwidth_gbs;
    s.primary = d.devices(1);
    s.n_coils = double(op.Config.n_coils);
    s.share_primary = op.SharePrimary;
    s.valid = true;
    s.toep_source = d.toep_source;
    s.toep_devices = d.toep_devices;
    s.toep_ms = d.toep_ms;
    s.toep_component_ms = d.toep_component_ms;
    s.toep_gather_gbs = d.toep_gather_gbs;
    s.create_ms = d.create_ms;
end

function v = mr_op_from_py(v)
%MR_OP_FROM_PY  Python scalar / string / list value -> MATLAB (double row,
%   char, logical or double).
    if isa(v, 'py.list') || isa(v, 'py.tuple')
        v = cellfun(@double, cell(v));
    elseif isa(v, 'py.str')
        v = char(v);
    elseif isa(v, 'py.bool')
        v = logical(v);
    elseif isa(v, 'py.int') || isa(v, 'py.float')
        v = double(v);
    end
end
