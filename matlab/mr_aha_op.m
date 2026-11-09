classdef mr_aha_op < handle
%MR_AHA_OP  Multi-GPU forward normal operator A'A (CUDA backend).
%
%   op = MR_AHA_OP(SEs, phi2_toep, config)
%   op = MR_AHA_OP(SEs, phi2_toep, config, 'Name', Value, ...)
%   op = MR_AHA_OP.FROM_GRAM(SEs, phi2, traj, config, 'Name', Value, ...)
%
%   AhAx = op.apply(x);      % or  AhAx = op * x
%   info = op.info();
%   delete(op);              % or  clear op
%
%   Builds a reusable operator that evaluates the same A'*A*x as
%   AHA_TOEPLITZ, but spreads the coils over several GPUs of this machine.
%   A'A is a sum over coils, so every GPU processes a subset of the coils and
%   the partial sums are added on the primary GPU. The constants are uploaded
%   once, at construction: every GPU receives a copy of phi2_toep (and of the
%   mask) and the sensitivity maps of its own coils; each APPLY then only
%   broadcasts x and gathers the partial results. Multi-GPU therefore makes
%   the operator faster but does not raise the largest problem size: the
%   whole kernel must fit on every GPU.
%
%   MR_AHA_OP.FROM_GRAM builds the same operator from the Gram blocks and the
%   trajectory instead of a finished kernel, and is several times faster
%   than CALC_TOEP followed by MR_AHA_OP: every GPU computes a share of the
%   (L^2+L)/2 independent triangular components of phi2_toep directly into
%   its own copy, and the GPUs then exchange their shares (overlapped with
%   the computation), instead of one GPU computing the whole kernel and
%   copying it to every other GPU. Every component is computed once, so the
%   copies on all GPUs are identical. When the GPUs cannot exchange data
%   quickly (no peer access), each computes the whole kernel itself; a cost
%   model measured at creation decides, and the environment variable
%   MR_AHA_TOEP_SPLIT ('auto' default, 'split', 'replicate', 'off'; SETENV
%   before creating) overrides it. INFO reports the choice (toep_source).
%   Arguments: SEs, config, 'devices', 'mask' and 'share_primary' as below;
%   phi2 and traj as for CALC_TOEP. With config.is_masked and no 'mask', the
%   mask is derived from traj with CALC_MASK, as CALC_TOEP does.
%
%   The PRIMARY GPU is MATLAB's current gpuDevice when the operator is
%   created. It holds the inputs and outputs (gpuArrays) and always takes
%   part in the computation.
%
%   INPUTS
%     SEs        complex single sensitivity maps, see AHA_TOEPLITZ.
%                  [Nx, Ny, Ncoils]        for '2D'
%                  [Nx, Ny, Nz, Ncoils]    for '3D' and 'stack'
%     phi2_toep  complex single packed Toeplitz kernel from CALC_TOEP.
%     config     struct from MR_DEFAULT_CONFIG, exactly as for AHA_TOEPLITZ
%                (config.gpu_device_id is ignored: the primary GPU is always
%                the current gpuDevice).
%     Arrays that are not gpuArrays, not single or not complex are converted
%     like AHA_TOEPLITZ does; passing complex single gpuArrays avoids copies.
%
%   NAME-VALUE PAIRS
%     'devices'        GPUs to use, as 1-based gpuDevice indices (the
%                      numbering of gpuDevice(k) and gpuDeviceTable).
%                      Default: every visible GPU. The primary GPU always
%                      takes part, listed or not; duplicates are ignored.
%                      The library may use fewer GPUs than given: with C
%                      coils and G GPUs it uses the smallest number with the
%                      same ceil(C/G) coils per GPU (e.g. 12 coils on 5 GPUs
%                      use 4), and only the primary GPU for problems too small
%                      to benefit (fewer than 2^22 upsampled values per coil;
%                      set the environment variable MR_AHA_MGPU_MIN_WORK with
%                      SETENV before creating the operator to change this,
%                      0 disables the check). INFO reports the choice.
%     'mask'           1-based linear indices into the centred upsampled grid
%                      (as for AHA_TOEPLITZ); required when config.is_masked
%                      is true.
%     'share_primary'  false (default) or true. When true the primary GPU
%                      uses SEs, phi2_toep and the mask in place instead of
%                      copying them, which saves one copy of the kernel on
%                      the primary GPU. The operator then keeps references to
%                      these arrays until it is deleted, so their memory is
%                      not released by clearing the caller's variables. The
%                      caller's variables can still be changed freely:
%                      gpuArrays are copy-on-write, so an assignment creates
%                      new storage and the operator keeps the old values.
%
%   METHODS
%     AhAx = apply(op, x)  A'A x. x is complex single [Nx, Ny, L] ('2D') or
%                          [Nx, Ny, Nz, L] ('3D', 'stack') on the primary GPU
%                          (host arrays are moved there); AhAx is a gpuArray
%                          of the same shape. op * x is the same.
%     Om = kernel(op)      A copy of the operator's phi2_toep on the
%                          primary GPU (complex single gpuArray, shaped like
%                          CALC_TOEP's output), e.g. to save the kernel built
%                          by FROM_GRAM. Costs one more kernel's memory.
%     info = info(op)      Struct describing how the work is distributed:
%                            n_devices           GPUs actually used
%                            devices             their gpuDevice indices
%                                                (1-based), primary first
%                            cuda_devices        the same as CUDA ordinals
%                            primary             gpuDevice index of the
%                                                primary GPU
%                            coil_begin          first coil (1-based) of each
%                                                GPU
%                            coil_count          number of coils of each GPU
%                            group               coils per group within each
%                                                GPU's single-GPU plan
%                            n_split_coils       coils split by rank across
%                                                all GPUs (MR_AHA_MGPU_SPLIT)
%                            peer_access         direct peer access enabled
%                                                between the primary and every
%                                                other GPU used
%                            peer_bandwidth_gbs  measured bandwidth or 0
%                            n_coils             number of coils
%                            share_primary       see above
%                            valid               false once the primary GPU
%                                                was reset or deselected
%                            toep_source         how phi2_toep was obtained:
%                                                'supplied' (constructor),
%                                                'split' (shares computed on
%                                                all GPUs and exchanged),
%                                                'replicate' (every GPU
%                                                computed it), 'broadcast'
%                                                (the primary computed it and
%                                                copied it)
%                            toep_devices        GPUs that computed it
%                            toep_ms             time of computing and
%                                                distributing it (ms)
%                            toep_component_ms   GPU time per component (ms)
%                            toep_gather_gbs     measured concurrent receive
%                                                bandwidth (GB/s)
%     delete(op)           Release all GPU memory of the operator on every
%                          GPU. Also happens automatically when the last
%                          reference to op is cleared.
%
%   DEVICE CHANGES
%     The operator is bound to its primary GPU. reset(gpuDevice) and
%     selecting another GPU with gpuDevice(k) free the operator's data on
%     the primary GPU (MATLAB destroys the GPU context); APPLY then raises
%     the error 'MR_AHA:mr_aha_op:deviceReset' (or ':deviceChanged' while a
%     different GPU is selected) instead of computing. Delete the operator
%     and create a new one; deleting it then warns
%     ('MR_AHA:mr_aha_op:staleDestroy') and releases what it holds on the
%     other GPUs (mr_mgpu_plan_abandon). Delete operators before
%     reset(gpuDevice) to avoid the warning. MATLAB's current gpuDevice is
%     never changed by MR_AHA_OP, although other GPUs are used internally.
%
%   CLEAR MEX
%     While any operator exists the gateway MEX file mr_aha_op_mex is locked
%     (mexLock), so 'clear mex' / 'clear functions' do not unload it and the
%     existing operators keep working. The lock is released when the last
%     operator is deleted. 'clear all' deletes the operators first. Plans
%     still alive when MATLAB exits are released.
%
%   EXAMPLE
%     gpuDevice(1);                                   % primary GPU
%     config = mr_default_config('stack', [256 256 64], 16);
%     Om = calc_toep(phi2, traj, config);
%     op = mr_aha_op(SEs, Om, config, 'devices', 1:4);
%     clear Om                                        % op holds a copy
%     AhAx = op * x;                                  % = AhA_Toeplitz(x, SEs, Om, config)
%     disp(op.info())
%     delete(op)
%
%     % Faster set-up: compute the kernel on all GPUs at once.
%     op = mr_aha_op.from_gram(SEs, phi2, traj, config, 'devices', 1:4);
%
%   See also AHA_TOEPLITZ, CALC_TOEP, MR_DEFAULT_CONFIG, GPUDEVICE.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    properties (SetAccess = private)
        %CONFIG  Configuration the operator was built with (with .n_coils and,
        %   for a masked operator, .num_masked filled in).
        Config
    end

    properties (Access = private, Transient)
        %HANDLE  uint64 plan handle of mr_aha_op_mex; 0 once destroyed.
        Handle = uint64(0)
    end

    methods
        function op = mr_aha_op(SEs, phi2_toep, config, varargin)
            %MR_AHA_OP  Construct the operator; see the class help.
            if nargin == 0
                return      % empty object, filled in by FROM_GRAM
            end
            narginchk(3, inf);
            if strcmp(mr_backend(), 'metal')
                error('MR_AHA:mr_aha_op:backend', ...
                      'mr_aha_op requires the CUDA backend.');
            end
            p = inputParser();
            p.FunctionName = 'mr_aha_op';
            p.addParameter('devices', []);
            p.addParameter('mask', []);
            p.addParameter('share_primary', false);
            p.parse(varargin{:});
            opt = p.Results;

            % Same conversions as AhA_Toeplitz.m.
            SEs       = mr_op_complex_single(SEs,       'SEs');
            phi2_toep = mr_op_complex_single(phi2_toep, 'phi2_toep');
            config    = mr_op_fill_n_coils(config, SEs);
            devices   = mr_op_ordinals(opt.devices);
            [mask, config] = mr_op_mask(opt.mask, config);

            op.Handle = mr_aha_op_mex('create', SEs, phi2_toep, config, ...
                                      mask, devices, ...
                                      logical(opt.share_primary));
            op.Config = config;
        end

        function AhAx = apply(op, x)
            %APPLY  A'A x on the operator's GPUs; see the class help.
            op.check_alive();
            x = mr_op_complex_single(x, 'x');
            AhAx = mr_aha_op_mex('apply', op.Handle, x);
        end

        function AhAx = mtimes(op, x)
            %MTIMES  op * x is the same as op.apply(x).
            if ~isa(op, 'mr_aha_op')
                error('MR_AHA:mr_aha_op:mtimes', ...
                      'Only op * x is supported (the operator on the left).');
            end
            AhAx = op.apply(x);
        end

        function s = info(op)
            %INFO  Work distribution of the operator; see the class help.
            op.check_alive();
            s = mr_aha_op_mex('info', op.Handle);
            s.cuda_devices = s.devices;
            s.devices = s.devices + 1;            % gpuDevice indices
            s.primary = s.primary + 1;
            s.coil_begin = s.coil_begin + 1;      % 1-based coils
            s = orderfields(s, {'n_devices', 'devices', 'cuda_devices', ...
                'primary', 'coil_begin', 'coil_count', 'group', ...
                'n_split_coils', 'peer_access', 'peer_bandwidth_gbs', ...
                'n_coils', 'share_primary', 'valid', 'toep_source', ...
                'toep_devices', 'toep_ms', 'toep_component_ms', ...
                'toep_gather_gbs'});
        end

        function Om = kernel(op)
            %KERNEL  Copy of the operator's phi2_toep on the primary GPU;
            %   see the class help.
            op.check_alive();
            Om = mr_aha_op_mex('kernel', op.Handle);
        end

        function tf = is_alive(op)
            %IS_ALIVE  True until the operator has been deleted (independent
            %   of device resets; see INFO's .valid for that).
            tf = op.Handle ~= 0;
        end

        function delete(op)
            %DELETE  Release the operator on every GPU. Never throws.
            if op.Handle ~= 0
                h = op.Handle;
                op.Handle = uint64(0);
                try
                    mr_aha_op_mex('destroy', h);
                catch err
                    warning('MR_AHA:mr_aha_op:destroy', ...
                            'Releasing the operator failed: %s', err.message);
                end
            end
        end
    end

    methods (Static)
        function op = from_gram(SEs, phi2, traj, config, varargin)
            %FROM_GRAM  Build the operator from the Gram blocks, computing
            %   phi2_toep on all GPUs at once; see the class help.
            %
            %   op = mr_aha_op.from_gram(SEs, phi2, traj, config)
            %   op = mr_aha_op.from_gram(..., 'devices', 1:4, 'mask', mask, ...
            %                            'share_primary', false)
            narginchk(4, inf);
            if strcmp(mr_backend(), 'metal')
                error('MR_AHA:mr_aha_op:backend', ...
                      'mr_aha_op requires the CUDA backend.');
            end
            p = inputParser();
            p.FunctionName = 'mr_aha_op.from_gram';
            p.addParameter('devices', []);
            p.addParameter('mask', []);
            p.addParameter('share_primary', false);
            p.parse(varargin{:});
            opt = p.Results;

            % Same conversions as calc_toep.m and AhA_Toeplitz.m.
            SEs    = mr_op_complex_single(SEs,  'SEs');
            phi2   = mr_op_complex_single(phi2, 'phi2');
            if ~isa(traj, 'gpuArray'), traj = gpuArray(traj); end
            if ~strcmp(underlyingType(traj), 'single'), traj = single(traj); end
            if ~isreal(traj)
                error('MR_AHA:mr_aha_op:complexTraj', 'traj must be real-valued.');
            end
            config  = mr_op_fill_n_coils(config, SEs);
            devices = mr_op_ordinals(opt.devices);
            mask = opt.mask;
            if isempty(mask) && isfield(config, 'is_masked') && config.is_masked
                mask = calc_mask(traj, config);   % as calc_toep.m does
            end
            [mask, config] = mr_op_mask(mask, config);

            op = mr_aha_op();
            op.Handle = mr_aha_op_mex('create_gram', SEs, phi2, traj, ...
                                      config, mask, devices, ...
                                      logical(opt.share_primary));
            op.Config = config;
        end
    end

    methods (Access = private)
        function check_alive(op)
            %CHECK_ALIVE  Error if the operator was deleted or not created
            %   in this session (e.g. loaded from a MAT-file).
            if op.Handle == 0
                error('MR_AHA:mr_aha_op:invalidHandle', ...
                      'This mr_aha_op has been deleted (or was loaded from a file).');
            end
        end
    end
end

function devices = mr_op_ordinals(devices)
%MR_OP_ORDINALS  1-based gpuDevice indices -> int32 CUDA ordinals; [] (every
%   visible GPU) stays [].
    if isempty(devices)
        devices = int32([]);
        return
    end
    devices = double(devices(:)).';
    n_gpu = gpuDeviceCount();
    if any(devices ~= round(devices)) || any(devices < 1) || any(devices > n_gpu)
        error('MR_AHA:mr_aha_op:devices', ...
              'devices must be gpuDevice indices between 1 and %d.', n_gpu);
    end
    devices = int32(devices - 1);
end

function [mask, config] = mr_op_mask(mask, config)
%MR_OP_MASK  Mask handling identical to AhA_Toeplitz.m: required in masked
%   mode, converted to 0-based int64 indices (config.num_masked filled in).
    if isempty(mask)
        if isfield(config, 'is_masked') && config.is_masked
            error('MR_AHA:mr_aha_op:missingMask', ...
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
        error('MR_AHA:mr_aha_op:emptyInput', '%s must not be empty.', name);
    end
end

function config = mr_op_fill_n_coils(config, SEs)
%MR_OP_FILL_N_COILS  Derive .n_coils from the trailing dimension of SEs, as
%   AhA_Toeplitz.m does. An explicit value already in config is kept.
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
