function check_op_matlab(varargin)
%CHECK_OP_MATLAB  Validate the A'A operator object MR_AHA.
%
%   CHECK_OP_MATLAB()
%   CHECK_OP_MATLAB('multi', true)              % also the multi-GPU sections
%   CHECK_OP_MATLAB('multi', true, 'device_sets', {[1 2], 1:4})
%
%   Builds operators with MR_AHA.CREATE_OP on small random problems and
%   errors if any check fails, so `matlab -batch check_op_matlab` exits
%   non-zero on failure. The MEX files, the matlab/ folder and, for the NUFFT
%   reference, cufinufft's MATLAB interface (<build>/finufft/matlab, built
%   with -DMR_AHA_BUILD_CUFINUFFT_MATLAB=ON) must be on the path (see
%   RUN_MATLAB_TESTS).
%
%   SINGLE-GPU CHECKS (always run, on the current gpuDevice)
%     toeplitz     for '2D', 'stack', '3D' (even and odd sizes), 5 and 3
%                  coils, 'lower' / 'upper', unmasked / masked,
%                  share_primary false / true:
%                    - with the kernel of CALC_TOEP passed through the
%                      private '_phi2_toep' argument, op * x and op.apply(x)
%                      are bit-identical to AHA_TOEPLITZ with that kernel;
%                    - built from the Gram blocks, op * x agrees with it to
%                      'tol' (1e-6) and op.kernel() with the CALC_TOEP kernel
%                      to 1e-6 (cufinufft's run-to-run noise);
%                    - info(): method, one GPU (the current one), all coils,
%                      toep_source 'supplied' / 'broadcast'.
%     nufft        method 'nufft' on the same cases (unmasked): against
%                  AHA_CUFINUFFT (the independent reference: finufft's own
%                  MATLAB interface, no library code; skipped when it is not
%                  on the path), relative difference <= 1e-5; against the
%                  Toeplitz operator (<= 1e-4); op * x against apply (<= 1e-6;
%                  cufinufft's type 1 is not bit-reproducible); nufft_opts
%                  gpu_method 1 and 2 against the defaults (<= 1e-5); info().
%     errors       unknown method, masked config with 'nufft', bad
%                  'devices', 'nufft_opts' with 'toeplitz' or unknown
%                  fields, bad 'toep_dist', '_phi2_toep' with 'nufft', the
%                  non-public constructor, kernel() of a NUFFT operator,
%                  kernel(k) out of range, a wrongly sized x; none leaves an
%                  operator in the registry.
%     device       MATLAB's current gpuDevice is unchanged by create /
%                  apply / info / kernel / delete; 'devices', 'all' and
%                  'devices', <current> work.
%     share        with share_primary, the operator keeps the shared arrays
%                  alive after the caller clears them, and later in-place
%                  assignments to the caller's arrays do not affect it (both
%                  methods).
%     handles      apply on a deleted operator errors cleanly; destroying an
%                  unknown handle is harmless; clearing the variable destroys
%                  the operator.
%     clear mex    with an operator alive, `clear mex` leaves the (locked)
%                  MEX file loaded and the operator usable; deleting the last
%                  operator unlocks it.
%     leaks        repeated create / apply / delete (both methods) leaves the
%                  memory used by this MATLAB process on every GPU unchanged
%                  (nvidia-smi; falls back to the primary's AvailableMemory).
%     reset        after reset(gpuDevice), apply raises
%                  'MR_AHA:mr_aha:deviceReset', info().valid is false and
%                  delete warns 'MR_AHA:mr_aha:staleDestroy' and releases
%                  the operator (mr_op_abandon), for both methods.
%
%   MULTI-GPU CHECKS ('multi'; skipped by default unless more than one GPU is
%   visible)
%     toeplitz     for every device set, MR_AHA_MGPU_SPLIT 'off' / 'on',
%                  'toep_dist' 'auto' / 'split' / 'replicate' / 'broadcast',
%                  unmasked / masked, 'lower' / 'upper', 5 and 8 coils:
%                  op * x against AHA_TOEPLITZ on the primary alone (< tol),
%                  the kernel copy of every GPU (kernel(k)) against CALC_TOEP
%                  (< 1e-6; identical copies with 'split' / 'broadcast'),
%                  toep_source, the work really spread over several GPUs, all
%                  coils assigned, the primary first; with '_phi2_toep'
%                  bit-for-bit the same kernel on every GPU.
%     nufft        G GPUs against the single-GPU NUFFT operator (<= 1e-6).
%     device change after gpuDevice(k) with another k, apply raises
%                  ':deviceChanged'; back on the primary, ':deviceReset'.
%     leaks        as above, on every GPU of the largest set.
%
%   NAME-VALUE PAIRS
%     'multi'          'auto' (default: run the multi-GPU checks when more
%                      than one GPU is visible), true (run them; with one
%                      visible GPU the device sets degenerate to the primary,
%                      which only exercises the code paths) or false.
%     'primary'        gpuDevice index (1-based) of the primary GPU. Default:
%                      the current gpuDevice.
%     'device_sets'    cell array of gpuDevice index lists (1-based) for the
%                      multi-GPU checks. Default: {[primary, one other GPU],
%                      all GPUs}.
%     'device_change'  run the gpuDevice(k) check (multi-GPU only, selects a
%                      second GPU), default true.
%     'tol'            relative tolerance of the operator checks, default
%                      1e-6.
%
%   Tiny problems normally stay on the primary GPU; the multi-GPU checks set
%   MR_AHA_MGPU_MIN_WORK=0 so that they are distributed. MR_AHA_TOEP_SPLIT,
%   which would override 'toep_dist', is cleared for the duration. All three
%   environment variables are restored on return.
%
%   REFERENCES. Every reference is computed on the fly from the same random
%   inputs (seeded with rng(1234)): CALC_TOEP + AHA_TOEPLITZ (the stateless
%   primitives) for the Toeplitz method, AHA_CUFINUFFT (independent of the
%   library) and the Toeplitz operator for the NUFFT method, and the
%   single-GPU operator for every multi-GPU result. Bit-identity is required
%   wherever the arithmetic is the same (a supplied kernel, kernel copies
%   that are transferred rather than recomputed).
%
%   On the Metal backend only the 'toeplitz', 'nufft' and argument checks
%   run (one GPU; the operator is Python's mr_aha.create_op).

%   Author: Junzhou Chen (junzhou@chen.engineer)

    % ---- Options (see NAME-VALUE PAIRS above) ---------------------------
    p = inputParser();
    p.addParameter('multi', 'auto');
    p.addParameter('primary', []);
    p.addParameter('device_sets', {});
    p.addParameter('device_change', true);
    p.addParameter('tol', 1e-6);
    p.parse(varargin{:});
    opt = p.Results;

    % Primary GPU: on CUDA the requested (or current) gpuDevice, selected
    % now so that every array created below lives there. Metal has one GPU.
    metal = strcmp(mr_backend(), 'metal');
    if metal
        n_gpu = 1;
        opt.primary = 1;
    else
        n_gpu = gpuDeviceCount();
        if isempty(opt.primary)
            opt.primary = gpuDevice().Index;
        end
        gpuDevice(opt.primary);
    end
    % 'multi': 'auto' runs the multi-GPU part only with several visible GPUs;
    % true / false force it on / off. Never on Metal.
    if ischar(opt.multi) || isstring(opt.multi)
        run_multi = ~metal && n_gpu > 1;
    else
        run_multi = ~metal && logical(opt.multi);
    end
    % The independent NUFFT reference AHA_CUFINUFFT needs cufinufft's MATLAB
    % interface on CUDA (on Metal it uses the Python package).
    has_nufft_ref = metal || exist('cufinufft_plan', 'file') == 2;
    fprintf('check_op_matlab: backend %s, primary gpuDevice(%d), %d GPU(s) visible\n', ...
            mr_backend(), opt.primary, n_gpu);

    % Save the three environment variables the checks modify and restore them
    % when this function exits (onCleanup also runs on error). The kernel
    % distribution override is cleared so that 'toep_dist' is honoured.
    env = {'MR_AHA_MGPU_MIN_WORK', 'MR_AHA_MGPU_SPLIT', 'MR_AHA_TOEP_SPLIT'};
    old_env = cellfun(@getenv, env, 'UniformOutput', false);
    restore_env = onCleanup(@() cellfun(@setenv, env, old_env));
    setenv('MR_AHA_TOEP_SPLIT', '');

    % Fixed seed: repeatable problems. nfail accumulates failed checks.
    rng(1234);
    nfail = 0;

    % Problems: traj_type, image size, rank. Even and odd sizes.
    % Odd sizes test the exact centring remap for odd dimensions; every
    % problem is run with 5 and 3 coils (with the default coil group of 4,
    % one full group plus a remainder, and a single partial group).
    cases = {'2D',    [32 32],    3;
             'stack', [24 24 8],  3;
             '3D',    [16 16 12], 2;
             '2D',    [33 27],    4;
             'stack', [21 19 7],  2;
             '3D',    [15 13 11], 3};

    % ------------------------------------------------------------------
    % Toeplitz, one GPU
    % ------------------------------------------------------------------
    % Tallies: variants run / passed, worst errors, and how many supplied-
    % kernel variants were bit-identical.
    n_run = 0; n_pass = 0; worst = 0; worst_k = 0; n_ident = 0;
    for ic = 1:size(cases, 1)
        for nc = [5 3]
            % One random problem per (case, coil count); every variant below
            % reuses it.
            g = make_problem(cases{ic, :}, nc);
            for tri = {'lower', 'upper'}
                for masked = [false true]
                    % Reference for this variant: the kernel from CALC_TOEP
                    % (which also returns the mask when cfg.is_masked) applied
                    % by AHA_TOEPLITZ.
                    cfg = g.cfg;
                    cfg.triangulation = tri{1};
                    cfg.is_masked = masked;
                    if masked
                        [Om, mask] = calc_toep(g.phi2, g.traj, cfg);
                    else
                        Om = calc_toep(g.phi2, g.traj, cfg);
                        mask = [];
                    end
                    ref = AhA_Toeplitz(g.x, g.SEs, Om, cfg, mask);
                    % Every variant once with the maps (and mask) copied and
                    % once used in place (share_primary).
                    for share = [false true]
                        bad = {};
                        % Supplied kernel: bit-identical to AhA_Toeplitz.
                        % The operator then runs exactly the plan code of
                        % AhA_Toeplitz, so both call forms (op.apply and the
                        % mtimes overload op * x) must reproduce ref bit for
                        % bit; info() must report the 'supplied' kernel.
                        op = mr_aha.create_op(g.SEs, g.phi2, g.traj, cfg, ...
                            'mask', mask, 'share_primary', share, '_phi2_toep', Om);
                        y1 = op.apply(g.x);
                        y2 = op * g.x;
                        s = op.info();
                        delete(op);
                        ident = isequal(gather(y1), gather(ref)) && ...
                                isequal(gather(y2), gather(ref));
                        n_ident = n_ident + ident;
                        if ~ident
                            bad{end+1} = sprintf('supplied kernel not bit-identical (%.2e)', ...
                                                 rel(y1, ref)); %#ok<AGROW>
                        end
                        bad = [bad, check_info(s, 'toeplitz', nc, opt.primary, ...
                                               'supplied', share, metal)]; %#ok<AGROW>
                        if share && ~metal
                            % Supplied kernel without the Gram blocks.
                            % (CUDA only) With '_phi2_toep' the Gram blocks
                            % and trajectory are not needed and may be [].
                            op = mr_aha.create_op(g.SEs, [], [], cfg, ...
                                'mask', mask, '_phi2_toep', Om);
                            y3 = op * g.x;
                            delete(op);
                            if ~isequal(gather(y3), gather(ref))
                                bad{end+1} = 'supplied kernel, empty phi2/traj: not bit-identical'; %#ok<AGROW>
                            end
                        end
                        % Kernel computed by the operator.
                        % The operator recomputes the kernel with cufinufft,
                        % which is not bit-reproducible: op * x is compared to
                        % ref at opt.tol, op.kernel() to Om at 1e-6, and info()
                        % must report 'broadcast' (computed on the primary).
                        op = mr_aha.create_op(g.SEs, g.phi2, g.traj, cfg, ...
                            'mask', mask, 'share_primary', share);
                        y = op * g.x;
                        ek = rel(op.kernel(), Om);
                        s = op.info();
                        delete(op);
                        e = rel(y, ref);
                        worst = max(worst, e); worst_k = max(worst_k, ek);
                        if ~(e < opt.tol), bad{end+1} = sprintf('AhA %.2e', e); end %#ok<AGROW>
                        if ~(ek < 1e-6), bad{end+1} = sprintf('kernel %.2e', ek); end %#ok<AGROW>
                        bad = [bad, check_info(s, 'toeplitz', nc, opt.primary, ...
                                               'broadcast', share, metal)]; %#ok<AGROW>
                        % One pass/fail per variant; a failure prints the
                        % variant and every problem found.
                        n_run = n_run + 1;
                        if isempty(bad)
                            n_pass = n_pass + 1;
                        else
                            fprintf('  FAIL toeplitz %s [%s] L=%d coils=%d %s masked=%d share=%d: %s\n', ...
                                    cases{ic, 1}, num2str(cases{ic, 2}), cases{ic, 3}, ...
                                    nc, tri{1}, masked, share, strjoin(bad, ', '));
                        end
                    end
                end
            end
        end
    end
    nfail = nfail + (n_run - n_pass);
    fprintf(['toeplitz (1 GPU): %d / %d passed; supplied kernel bit-identical ' ...
             'in %d / %d; computed kernel: worst AhA difference %.3e, worst ' ...
             'kernel difference %.3e\n'], n_pass, n_run, n_ident, n_run, worst, worst_k);

    % ------------------------------------------------------------------
    % NUFFT, one GPU, against AhA_cufinufft and the Toeplitz operator
    % ------------------------------------------------------------------
    % NUFFT method on every case (unmasked: it has no kernel to mask).
    if ~has_nufft_ref
        fprintf(['nufft (1 GPU): cufinufft''s MATLAB interface is not on the ' ...
                 'path, the comparison with AhA_cufinufft is SKIPPED\n']);
    end
    n_run = 0; n_pass = 0; worst = 0; worst_t = 0; worst_o = 0;
    for ic = 1:size(cases, 1)
        for nc = [5 3]
            g = make_problem(cases{ic, :}, nc);
            % Independent reference (finufft's MATLAB interface, no library
            % code), when available.
            ref = [];
            if has_nufft_ref
                ref = AhA_cufinufft(g.x, g.SEs, g.phi2, g.traj, g.cfg);
            end
            % Default nufft_opts for every case; for the two 2D cases also
            % gpu_method 1 and gpu_method 2 (with an explicit upsampfac),
            % which must agree with the defaults to 1e-5.
            opt_sets = {struct()};
            if ic == 1 || ic == 4
                opt_sets = [opt_sets, {struct('gpu_method', 1), ...
                    struct('gpu_method', 2, 'upsampfac', 2)}]; %#ok<AGROW>
            end
            y0 = [];
            for io = 1:numel(opt_sets)
                bad = {};
                op = mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg, ...
                    'method', 'nufft', 'nufft_opts', opt_sets{io});
                y1 = op.apply(g.x);
                y2 = op * g.x;
                s = op.info();
                delete(op);
                % NUFFT operator vs AhA_cufinufft: same transforms, different
                % summation order, hence 1e-5.
                if ~isempty(ref)
                    e = rel(y1, ref);
                    worst = max(worst, e);
                    if ~(e <= 1e-5), bad{end+1} = sprintf('vs AhA_cufinufft %.2e', e); end %#ok<AGROW>
                end
                % Run-to-run: cufinufft's type 1 sums with atomics, so two
                % applications agree to rounding, not bit for bit.
                e2 = rel(y2, y1);
                if ~(e2 <= 1e-6), bad{end+1} = sprintf('op * x vs apply %.2e', e2); end %#ok<AGROW>
                bad = [bad, check_info(s, 'nufft', nc, opt.primary, 'none', ...
                                       true, metal)]; %#ok<AGROW>
                % Default options: compare with the Toeplitz operator, which
                % evaluates the same A'A to within the NUFFT tolerance (1e-4).
                % Other options: compare with the default-options result.
                if io == 1
                    y0 = y1;
                    opt_t = mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg);
                    et = rel(y1, opt_t * g.x);
                    delete(opt_t);
                    worst_t = max(worst_t, et);
                    if ~(et <= 1e-4), bad{end+1} = sprintf('vs toeplitz %.2e', et); end %#ok<AGROW>
                else
                    eo = rel(y1, y0);
                    worst_o = max(worst_o, eo);
                    if ~(eo <= 1e-5), bad{end+1} = sprintf('vs default options %.2e', eo); end %#ok<AGROW>
                end
                n_run = n_run + 1;
                if isempty(bad)
                    n_pass = n_pass + 1;
                else
                    fprintf('  FAIL nufft %s [%s] L=%d coils=%d opts#%d: %s\n', ...
                            cases{ic, 1}, num2str(cases{ic, 2}), cases{ic, 3}, ...
                            nc, io, strjoin(bad, ', '));
                end
            end
        end
    end
    nfail = nfail + (n_run - n_pass);
    if has_nufft_ref
        ref_msg = sprintf('worst difference to AhA_cufinufft %.3e (tol 1e-5), ', worst);
    else
        ref_msg = 'AhA_cufinufft not run, ';
    end
    fprintf(['nufft (1 GPU): %d / %d passed, %sto the Toeplitz operator %.3e ' ...
             '(tol 1e-4), nufft_opts variants to the defaults %.3e (tol 1e-5)\n'], ...
            n_pass, n_run, ref_msg, worst_t, worst_o);

    % ------------------------------------------------------------------
    % Argument errors
    % ------------------------------------------------------------------
    % Every entry is {description, expected error identifier, call}; each
    % call must fail with exactly that identifier. n0 records the number of
    % live operators so that the end of the section can check that failed
    % creations leave nothing in the MEX registry.
    g = make_problem('2D', [32 32], 3, 5);
    cfgm = g.cfg; cfgm.is_masked = true;
    n0 = op_count(metal);
    errs = {
        'unknown method', 'MR_AHA:mr_aha:method', ...
            @() mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg, 'method', 'direct');
        'masked config with nufft', 'MR_AHA:mr_aha:maskedNufft', ...
            @() mr_aha.create_op(g.SEs, g.phi2, g.traj, cfgm, 'method', 'nufft');
        'devices 0', 'MR_AHA:mr_aha:devices', ...
            @() mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg, 'devices', 0);
        'devices beyond the last GPU', 'MR_AHA:mr_aha:devices', ...
            @() mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg, 'devices', n_gpu + 1);
        'devices 1.5', 'MR_AHA:mr_aha:devices', ...
            @() mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg, 'devices', 1.5);
        'devices ''some''', 'MR_AHA:mr_aha:devices', ...
            @() mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg, 'devices', 'some');
        'nufft_opts with toeplitz', 'MR_AHA:mr_aha:nufftOpts', ...
            @() mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg, ...
                                    'nufft_opts', struct('upsampfac', 2));
        'unknown nufft_opts field', 'MR_AHA:mr_aha:nufftOpts', ...
            @() mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg, 'method', 'nufft', ...
                                    'nufft_opts', struct('modeord', 1));
        'bad toep_dist', 'MR_AHA:mr_aha:toepDist', ...
            @() mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg, 'toep_dist', 'gather');
        '_phi2_toep with nufft', 'MR_AHA:mr_aha:phi2Toep', ...
            @() mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg, 'method', 'nufft', ...
                                    '_phi2_toep', g.SEs);
        'public constructor', 'MR_AHA:mr_aha:useCreateOp', ...
            @() mr_aha(g.SEs, g.phi2, g.cfg);
        'constructor without arguments', 'MR_AHA:mr_aha:useCreateOp', ...
            @() mr_aha();
        };
    for ie = 1:size(errs, 1)
        id = error_id(errs{ie, 3});
        nfail = nfail + report(['error: ' errs{ie, 1}], strcmp(id, errs{ie, 2}), id);
    end
    % The removed MR_AHA.FROM_GRAM must fail (any identifier).
    id = error_id(@() mr_aha.from_gram(g.SEs, g.phi2, g.traj, g.cfg));
    nfail = nfail + report('error: from_gram no longer exists', ~isempty(id), id);
    % Errors raised on a live operator: kernel() of a NUFFT operator (it has
    % none), kernel(k) for a device index beyond the operator's GPUs, and an
    % x with the wrong size.
    op = mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg, 'method', 'nufft');
    id = error_id(@() op.kernel());
    nfail = nfail + report('error: kernel() of a NUFFT operator', ...
                           strcmp(id, 'MR_AHA:mr_aha:noKernel'), id);
    delete(op);
    op = mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg);
    id = error_id(@() op.kernel(op.info().n_devices + 1));
    nfail = nfail + report('error: kernel(k) beyond the operator''s GPUs', ~isempty(id), id);
    id = error_id(@() op * g.x(:, 1:end-1, :));
    nfail = nfail + report('error: wrongly sized x', ~isempty(id), id);
    delete(op);
    nfail = nfail + report('errors leave no operator behind', op_count(metal) == n0, '');

    % The remaining sections (gpuDevice handling, the MEX registry, device
    % reset, multi-GPU) are specific to the CUDA MEX gateway.
    if metal
        finish(nfail, 'single-GPU checks on Metal');
        return
    end

    % ------------------------------------------------------------------
    % Device hygiene: current gpuDevice unchanged; 'all' and the primary
    % ------------------------------------------------------------------
    % MATLAB tracks its own current gpuDevice; the library must leave it as
    % it was after every call (create, apply, info, kernel, delete), and the
    % result must still be correct.
    g = make_problem('3D', [16 16 12], 2, 8);
    Om = calc_toep(g.phi2, g.traj, g.cfg);
    ref = AhA_Toeplitz(g.x, g.SEs, Om, g.cfg);
    idx0 = current_index();
    op = mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg);
    i1 = current_index();
    y = op.apply(g.x);
    i2 = current_index();
    op.info(); op.kernel();
    i3 = current_index();
    delete(op);
    i4 = current_index();
    nfail = nfail + report('device unchanged by create/apply/info/kernel/delete', ...
        all([i1 i2 i3 i4] == idx0) && rel(y, ref) < opt.tol, ...
        sprintf('indices %s', num2str([idx0 i1 i2 i3 i4])));
    % Naming the primary explicitly in 'devices' is the same as the default:
    % one GPU, the primary.
    op = mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg, 'devices', opt.primary);
    s = op.info(); y = op * g.x; delete(op);
    nfail = nfail + report('''devices'', <current gpuDevice>', ...
        s.n_devices == 1 && s.devices == opt.primary && rel(y, ref) < opt.tol, '');
    % With a single visible GPU, 'all' resolves to that GPU.
    if n_gpu == 1
        op = mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg, 'devices', 'all');
        s = op.info(); y = op * g.x; delete(op);
        nfail = nfail + report('''devices'', ''all'' with one visible GPU', ...
            s.n_devices == 1 && rel(y, ref) < opt.tol, '');
    end

    % ------------------------------------------------------------------
    % share_primary keeps the shared arrays alive and is copy-on-write safe
    % ------------------------------------------------------------------
    % share_primary makes the operator use the caller's arrays in place, so
    % it must keep them alive itself: (1) build it from temporaries that go
    % out of scope (make_shared_op), overwrite recycled GPU memory, and check
    % that the operator still gives the reference result; (2) build it from
    % copies, then modify the caller's copies in place: MATLAB's
    % copy-on-write must give the caller new storage, never altering the data
    % the operator reads.
    for method = {'toeplitz', 'nufft'}
        m = method{1};
        if strcmp(m, 'nufft')
            refm = mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg, 'method', 'nufft');
            refy = refm * g.x;
            delete(refm);
        else
            refy = ref;
        end
        % Allocate and fill a large array so the freed temporaries' memory
        % would be reused (and overwritten) if the operator had not kept them.
        op = make_shared_op(g, Om, m, opt.primary);   % temporaries only
        junk = gpuArray.rand(4e6, 1, 'single');           % recycle freed memory
        junk(:) = 7; %#ok<NASGU>
        clear junk
        ok_alive = rel(op.apply(g.x), refy) < opt.tol;
        % Copy-on-write check: SEs2 / Om2 share storage with g until written.
        SEs2 = g.SEs; Om2 = Om;
        if strcmp(m, 'nufft')
            op2 = mr_aha.create_op(SEs2, g.phi2, g.traj, g.cfg, 'method', 'nufft');
        else
            op2 = mr_aha.create_op(SEs2, g.phi2, g.traj, g.cfg, '_phi2_toep', Om2);
        end
        SEs2(:) = 0; Om2(1:10) = 1e6;                     %#ok<NASGU> caller modifies
        clear SEs2 Om2
        ok_cow = rel(op2.apply(g.x), refy) < opt.tol;
        delete(op); delete(op2);
        nfail = nfail + report(['share_primary (' m '): arrays kept alive after clear'], ...
                               ok_alive, '');
        nfail = nfail + report(['share_primary (' m '): caller assignments do not leak in'], ...
                               ok_cow, '');
    end

    % ------------------------------------------------------------------
    % Handles: deleted operator, unknown handle, live-operator count
    % ------------------------------------------------------------------
    % A deleted operator and a handle that never existed must raise clean
    % errors (no crash); destroying an unknown handle is a silent no-op.
    op = mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg);
    delete(op);
    % MATLAB itself rejects methods of a deleted handle object ...
    id = error_id(@() op.apply(g.x));
    % ... and the gateway rejects handles that are not live operators.
    id2 = error_id(@() mr_aha_mex('apply', uint64(12345), g.x));
    ok_del = strcmp(id, 'MATLAB:class:InvalidHandle') && ...
             strcmp(id2, 'MR_AHA:mr_aha:invalidHandle');
    id = [id ' / ' id2];
    mr_aha_mex('destroy', uint64(12345));            % unknown handle: no-op
    % Clearing the last reference to an operator must destroy it: the
    % registry is empty and the MEX file is no longer locked.
    op = mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg, 'method', 'nufft'); %#ok<NASGU>
    clear op
    nfail = nfail + report('apply after delete / on a bad handle errors', ok_del, id);
    nfail = nfail + report('clear op destroys the operator', ...
        mr_aha_mex('count') == 0 && ~mislocked('mr_aha_mex'), ...
        sprintf('count=%d', mr_aha_mex('count')));

    % ------------------------------------------------------------------
    % clear mex with a live operator
    % ------------------------------------------------------------------
    % The MEX file locks itself while operators exist, so `clear mex` must
    % not unload it (which would orphan GPU memory and handles); the
    % operator stays usable, and deleting it releases the lock.
    op = mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg);
    locked_before = mislocked('mr_aha_mex');
    clear mex %#ok<CLMEX>
    locked_after = mislocked('mr_aha_mex');
    y = op.apply(g.x);
    ok_clear = locked_before && locked_after && rel(y, ref) < opt.tol && ...
               mr_aha_mex('count') == 1;
    delete(op);
    ok_unlock = ~mislocked('mr_aha_mex') && mr_aha_mex('count') == 0;
    nfail = nfail + report('clear mex keeps the locked MEX and the operator usable', ...
                           ok_clear, '');
    nfail = nfail + report('deleting the last operator unlocks the MEX', ok_unlock, '');

    % ------------------------------------------------------------------
    % Leaks: repeated create / apply / delete on the primary
    % ------------------------------------------------------------------
    % See LEAK_CHECK: warm-up, then 10 x (toeplitz + nufft) cycles.
    nfail = nfail + leak_check(opt.primary, []);

    % ------------------------------------------------------------------
    % Device reset (both methods)
    % ------------------------------------------------------------------
    for method = {'toeplitz', 'nufft'}
        % reset(gpuDevice) destroys every CUDA object the operator held. x is
        % gathered beforehand because the reset also invalidates gpuArrays.
        % Applying must raise deviceReset, info() must mark the operator
        % invalid, and delete must only warn (staleDestroy) and release it
        % through mr_op_abandon, leaving the registry empty and the MEX
        % unlocked.
        g = make_problem('3D', [16 16 12], 2, 5);
        op = mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg, 'method', method{1});
        xh = gather(g.x);
        reset(gpuDevice());
        id = error_id(@() op.apply(xh));
        s = op.info();
        lastwarn('');
        delete(op);                         % must not crash; warns staleDestroy
        [~, wid] = lastwarn();
        nfail = nfail + report(['reset(gpuDevice) then apply errors (' method{1} ')'], ...
            strcmp(id, 'MR_AHA:mr_aha:deviceReset') && ~s.valid, id);
        nfail = nfail + report(['delete after reset is safe (' method{1} ')'], ...
            strcmp(wid, 'MR_AHA:mr_aha:staleDestroy') && ...
            mr_aha_mex('count') == 0 && ~mislocked('mr_aha_mex'), wid);
    end
    % After the reset, a freshly created operator must work normally.
    g = make_problem('2D', [32 32], 3, 5);
    op = mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg);
    Om = calc_toep(g.phi2, g.traj, g.cfg);
    e = rel(op * g.x, AhA_Toeplitz(g.x, g.SEs, Om, g.cfg));
    delete(op);
    nfail = nfail + report('a new operator works after the reset', e < opt.tol, ...
                           sprintf('%.2e', e));

    % ------------------------------------------------------------------
    % Multi-GPU
    % ------------------------------------------------------------------
    if run_multi
        nfail = nfail + check_multi(opt, n_gpu);
    else
        fprintf(['multi-GPU checks: SKIPPED (%d GPU(s) visible; pass ''multi'', ' ...
                 'true to force)\n'], n_gpu);
    end

    % Final invariants: every operator created above was released. Restore
    % the primary as the current gpuDevice, then error if anything failed.
    nfail = nfail + report('registry empty and MEX unlocked at the end', ...
        mr_aha_mex('count') == 0 && ~mislocked('mr_aha_mex'), '');
    gpuDevice(opt.primary);
    finish(nfail, 'all checks');
end

% ----------------------------------------------------------------------
% Multi-GPU section
% ----------------------------------------------------------------------

function nfail = check_multi(opt, n_gpu)
%CHECK_MULTI  The multi-GPU checks (see the help of CHECK_OP_MATLAB).
    nfail = 0;
    % Default device sets: the primary plus one other GPU, and all GPUs.
    if isempty(opt.device_sets)
        others = setdiff(1:n_gpu, opt.primary);
        if isempty(others)
            opt.device_sets = {opt.primary};   % forced on a 1-GPU machine
        else
            opt.device_sets = {[opt.primary others(1)], 1:n_gpu};
        end
    end
    % Skip (rather than fail) if a requested set names a GPU that is not
    % visible.
    need = max([opt.primary, cellfun(@max, opt.device_sets)]);
    if need > n_gpu
        fprintf('multi-GPU checks: need GPUs up to index %d, %d visible: SKIPPED\n', ...
                need, n_gpu);
        return
    end
    big_set = opt.device_sets{end};
    fprintf('multi-GPU checks: primary gpuDevice(%d), device sets %s\n', ...
            opt.primary, strjoin(cellfun(@(d) ['[' num2str(d) ']'], ...
            opt.device_sets, 'UniformOutput', false), ' '));
    % These problems are tiny; disable the "too small for several GPUs"
    % rule so the work is really distributed.
    setenv('MR_AHA_MGPU_MIN_WORK', '0');
    gpuDevice(opt.primary);

    % ---- Toeplitz: G GPUs against the primary alone ----------------------
    cases = {'2D', [40 36], 3; 'stack', [32 30 8], 3; '3D', [20 18 16], 2};
    % toep_source expected for each requested distribution when more than
    % one GPU is used ('' = auto, which may resolve to split or replicate).
    expect = struct('split', 'split', 'replicate', 'replicate', ...
                    'broadcast', 'broadcast', 'auto', '');
    n_run = 0; n_pass = 0; worst = 0; worst_k = 0;
    for ic = 1:size(cases, 1)
        for nc = [5 8]
            g = make_problem(cases{ic, :}, nc);
            for tri = {'lower', 'upper'}
                for masked = [false true]
                    cfg = g.cfg;
                    cfg.triangulation = tri{1};
                    cfg.is_masked = masked;
                    if masked
                        [Om, mask] = calc_toep(g.phi2, g.traj, cfg);
                    else
                        Om = calc_toep(g.phi2, g.traj, cfg);
                        mask = [];
                    end
                    ref = AhA_Toeplitz(g.x, g.SEs, Om, cfg, mask);
                    for ids = 1:numel(opt.device_sets)
                        devs = opt.device_sets{ids};
                        % Whether the operator should end up on several GPUs.
                        multi = numel(unique([devs opt.primary])) > 1;
                        % MR_AHA_MGPU_SPLIT: 'on' forces the coils that do not
                        % divide evenly over the GPUs to be split by rank.
                        for split = {'off', 'on'}
                            setenv('MR_AHA_MGPU_SPLIT', split{1});
                            for dist = {'auto', 'split', 'replicate', 'broadcast'}
                                share = strcmp(dist{1}, 'split');   % vary it cheaply
                                k0 = current_index();
                                bad = {};
                                op = mr_aha.create_op(g.SEs, g.phi2, g.traj, cfg, ...
                                    'devices', devs, 'mask', mask, ...
                                    'toep_dist', dist{1}, 'share_primary', share);
                                % Result against the single-GPU reference.
                                y = op * g.x;
                                s = op.info();
                                e = rel(y, ref); worst = max(worst, e);
                                if ~(e < opt.tol), bad{end+1} = sprintf('AhA %.2e', e); end %#ok<AGROW>
                                % Each device's kernel copy against CALC_TOEP;
                                % 'split' and 'broadcast' transfer components
                                % rather than recompute them, so their copies
                                % must equal the primary's bit for bit.
                                K1 = op.kernel(1);
                                for k = 1:s.n_devices
                                    Kk = op.kernel(k);
                                    ek = rel(Kk, Om); worst_k = max(worst_k, ek);
                                    if ~(ek < 1e-6)
                                        bad{end+1} = sprintf('kernel(%d) %.2e', k, ek); %#ok<AGROW>
                                    end
                                    if any(strcmp(s.toep_source, {'split', 'broadcast'})) && ...
                                            ~isequal(gather(Kk), gather(K1))
                                        bad{end+1} = sprintf('kernel(%d) differs from kernel(1)', k); %#ok<AGROW>
                                    end
                                end
                                % Reported distribution: the requested mode
                                % (auto: split or replicate) on several GPUs;
                                % 'broadcast' when it stayed on one.
                                if s.n_devices > 1
                                    want = expect.(dist{1});
                                    if isempty(want)
                                        okk = any(strcmp(s.toep_source, {'split', 'replicate'}));
                                    else
                                        okk = strcmp(s.toep_source, want);
                                    end
                                else
                                    okk = strcmp(s.toep_source, 'broadcast');
                                end
                                if ~okk, bad{end+1} = ['toep_source ' s.toep_source]; end %#ok<AGROW>
                                % Work distribution: several GPUs exactly when
                                % expected, primary first and starting at coil
                                % 1, every coil assigned once (whole or split
                                % by rank), only requested GPUs used, and the
                                % current gpuDevice unchanged.
                                if (s.n_devices > 1) ~= multi || s.primary ~= opt.primary || ...
                                        s.devices(1) ~= opt.primary || s.coil_begin(1) ~= 1 || ...
                                        sum(s.coil_count) + s.n_split_coils ~= nc || ...
                                        ~all(ismember(s.devices, [devs opt.primary]))
                                    bad{end+1} = sprintf('distribution: devices [%s], coils [%s], split %d', ...
                                        num2str(s.devices), num2str(s.coil_count), s.n_split_coils); %#ok<AGROW>
                                end
                                delete(op);
                                if current_index() ~= k0, bad{end+1} = 'gpuDevice changed'; end %#ok<AGROW>
                                n_run = n_run + 1;
                                if isempty(bad)
                                    n_pass = n_pass + 1;
                                else
                                    fprintf('  FAIL toeplitz %s coils=%d %s masked=%d devices=[%s] split=%s dist=%s: %s\n', ...
                                            cases{ic, 1}, nc, tri{1}, masked, num2str(devs), ...
                                            split{1}, dist{1}, strjoin(bad, ', '));
                                end
                            end
                            % Supplied kernel: copied as is to every GPU.
                            % Every GPU must hold a bit-identical copy of the
                            % supplied kernel, and toep_source is 'supplied'.
                            op = mr_aha.create_op(g.SEs, g.phi2, g.traj, cfg, ...
                                'devices', devs, 'mask', mask, '_phi2_toep', Om);
                            e = rel(op * g.x, ref); worst = max(worst, e);
                            s = op.info();
                            same = true;
                            for k = 1:s.n_devices
                                same = same && isequal(gather(op.kernel(k)), gather(Om));
                            end
                            delete(op);
                            n_run = n_run + 1;
                            if e < opt.tol && same && strcmp(s.toep_source, 'supplied')
                                n_pass = n_pass + 1;
                            else
                                fprintf('  FAIL toeplitz supplied %s coils=%d %s masked=%d devices=[%s] split=%s: rel %.2e, identical copies %d, %s\n', ...
                                        cases{ic, 1}, nc, tri{1}, masked, num2str(devs), ...
                                        split{1}, e, same, s.toep_source);
                            end
                        end
                    end
                end
            end
        end
    end
    % Back to the library's default rank-split policy.
    setenv('MR_AHA_MGPU_SPLIT', '');
    nfail = nfail + (n_run - n_pass);
    fprintf(['multi-GPU toeplitz: %d / %d passed, worst AhA difference %.3e (tol ' ...
             '%.0e), worst kernel difference %.3e\n'], n_pass, n_run, worst, opt.tol, worst_k);

    % ---- NUFFT: G GPUs against one GPU ----------------------------------
    n_run = 0; n_pass = 0; worst = 0;
    for ic = 1:size(cases, 1)
        for nc = [5 8]
            g = make_problem(cases{ic, :}, nc);
            % Reference: the library's single-GPU NUFFT operator. Splitting
            % by coils only changes the order of the final coil sum (1e-6).
            op1 = mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg, 'method', 'nufft');
            ref = op1 * g.x;
            delete(op1);
            for ids = 1:numel(opt.device_sets)
                devs = opt.device_sets{ids};
                multi = numel(unique([devs opt.primary])) > 1;
                for share = [false true]
                    op = mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg, ...
                        'method', 'nufft', 'devices', devs, 'share_primary', share);
                    y = op * g.x;
                    s = op.info();
                    delete(op);
                    e = rel(y, ref); worst = max(worst, e);
                    % Also: several GPUs exactly when expected, method and
                    % source reported correctly, all coils assigned, primary
                    % first.
                    ok = e <= 1e-6 && (s.n_devices > 1) == multi && ...
                         strcmp(s.method, 'nufft') && strcmp(s.toep_source, 'none') && ...
                         sum(s.coil_count) == nc && s.devices(1) == opt.primary;
                    n_run = n_run + 1;
                    n_pass = n_pass + ok;
                    if ~ok
                        fprintf('  FAIL nufft %s coils=%d devices=[%s] share=%d: rel %.2e, used [%s], coils [%s]\n', ...
                                cases{ic, 1}, nc, num2str(devs), share, e, ...
                                num2str(s.devices), num2str(s.coil_count));
                    end
                end
            end
        end
    end
    nfail = nfail + (n_run - n_pass);
    fprintf('multi-GPU nufft: %d / %d passed, worst difference to 1 GPU %.3e (tol 1e-6)\n', ...
            n_pass, n_run, worst);

    % 'all' uses every visible GPU (up to the ceil(C/G) rule).
    g = make_problem('stack', [32 30 8], 3, 8);
    op = mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg, 'devices', 'all');
    s = op.info();
    disp(s);
    delete(op);
    nfail = nfail + report('''devices'', ''all''', s.n_devices >= min(n_gpu, 2) || n_gpu == 1, ...
                           sprintf('%d GPUs used', s.n_devices));

    % ---- Leaks on every GPU of the largest set --------------------------
    % Leak check across every GPU of the largest device set.
    nfail = nfail + leak_check(opt.primary, big_set);

    % ---- Device change ---------------------------------------------------
    % A multi-GPU operator belongs to the primary. With another gpuDevice
    % current, apply must refuse (deviceChanged). Switching gpuDevice in
    % MATLAB resets the previously selected device, so back on the primary
    % the operator's resources are gone and apply must report deviceReset.
    if opt.device_change && n_gpu > 1
        gpuDevice(opt.primary);
        g = make_problem('3D', [16 16 12], 2, 8);
        op = mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg, 'devices', big_set);
        xh = gather(g.x);
        other = setdiff(1:n_gpu, opt.primary);
        gpuDevice(other(1));
        id1 = error_id(@() op.apply(xh));
        gpuDevice(opt.primary);
        id2 = error_id(@() op.apply(xh));
        delete(op);
        nfail = nfail + report('gpuDevice(k) then apply errors (deviceChanged)', ...
            strcmp(id1, 'MR_AHA:mr_aha:deviceChanged'), id1);
        nfail = nfail + report('back on the primary: apply errors (deviceReset)', ...
            strcmp(id2, 'MR_AHA:mr_aha:deviceReset'), id2);
    end
    % (onCleanup in CHECK_OP_MATLAB restores the original values at exit.)
    setenv('MR_AHA_MGPU_MIN_WORK', '');
end

% ----------------------------------------------------------------------
% Helpers
% ----------------------------------------------------------------------

function g = make_problem(traj_type, n_dim, L, nc)
%MAKE_PROBLEM  Random Hermitian Gram blocks, a radial trajectory
%   (golden-angle in-plane, or golden-means 3D), x and SEs; gpuArrays on
%   CUDA, ordinary arrays on Metal.
    % Default configuration (upsampled grid 2*N, Mz = Nz for a stack) with
    % nc coils.
    cfg = mr_default_config(traj_type, n_dim, L);
    cfg.n_coils = nc;
    % 24 spokes of 2*Nx samples each, readout positions in [-pi, pi).
    % Directions: golden-means for 3D, golden-angle in-plane otherwise. The
    % trajectory is [Nt, Nkx, ndim] (spoke direction times readout radius).
    Nt = 24; Nkx = 2 * n_dim(1);
    k = pi * ((0:Nkx-1) - Nkx/2) / (Nkx/2);
    m = (0:Nt-1).';
    if strcmpi(traj_type, '3D')
        z = mod(m * 0.4656, 1) * 2 - 1; az = 2 * pi * mod(m * 0.6823, 1);
        d = [sqrt(1 - z.^2) .* cos(az), sqrt(1 - z.^2) .* sin(az), z];
    else
        a = m * pi * (sqrt(5) - 1) / 2;
        d = [cos(a), sin(a)];
    end
    traj = zeros(Nt, Nkx, size(d, 2), 'single');
    for c = 1:size(d, 2), traj(:, :, c) = d(:, c) * k; end
    % Gram blocks [Nt, (Nkz,) L, L]: block (t, z) is P*P' for a random
    % rank x time P with two time points, i.e. Hermitian positive
    % semi-definite in the library's convention (a stack has one per kz).
    nkz = 1;
    if strcmpi(traj_type, 'stack'), nkz = n_dim(3); end
    phi2 = complex(zeros([Nt, nkz, L, L], 'single'));
    for t = 1:Nt
        for z = 1:nkz
            P = complex(randn(L, 2), randn(L, 2));
            phi2(t, z, :, :) = reshape(single(P * P'), [1 1 L L]);
        end
    end
    if nkz == 1, phi2 = reshape(phi2, [Nt, L, L]); end
    % Random complex single x [n_dim, L] and maps [n_dim, nc], then move all
    % inputs to the GPU (CUDA) or leave them as ordinary arrays (Metal).
    crand = @(sz) complex(randn(sz, 'single'), randn(sz, 'single'));
    if strcmp(mr_backend(), 'metal')
        dev = @(a) a;
    else
        dev = @gpuArray;
    end
    g.cfg = cfg;
    g.phi2 = dev(phi2);
    g.traj = dev(traj);
    g.x = dev(crand([n_dim L]));
    g.SEs = dev(crand([n_dim nc]));
end

function bad = check_info(s, method, nc, primary, source, share, metal)
%CHECK_INFO  Consistency of a single-GPU operator's info(); returns the
%   list of problems.
    % Expected for a single-GPU operator: the requested method, exactly the
    % primary GPU, all nc coils on it starting at coil 1, the expected kernel
    % source, the share_primary setting as requested, a valid operator and
    % (CUDA) a measured creation time.
    bad = {};
    if ~strcmp(s.method, method), bad{end+1} = ['method ' s.method]; end
    if s.n_devices ~= 1 || s.devices ~= primary || s.primary ~= primary
        bad{end+1} = sprintf('devices [%s]', num2str(s.devices));
    end
    if s.coil_begin ~= 1 || s.coil_count ~= nc || s.n_coils ~= nc
        bad{end+1} = sprintf('coils %d+%d', s.coil_begin, s.coil_count);
    end
    if ~strcmp(s.toep_source, source), bad{end+1} = ['toep_source ' s.toep_source]; end
    if s.share_primary ~= share || ~s.valid, bad{end+1} = 'share_primary/valid'; end
    if ~metal && ~(s.create_ms > 0), bad{end+1} = 'create_ms'; end
end

function op = make_shared_op(g, Om, method, primary)
%MAKE_SHARED_OP  share_primary operator built from temporaries that go out
%   of scope on return, so only the operator keeps them alive.
    % "+ 0" / "* 1" force new GPU storage, so these variables own the only
    % references once the function returns.
    SEs = g.SEs + 0;              % fresh storage
    if strcmp(method, 'nufft')
        op = mr_aha.create_op(SEs, g.phi2, g.traj, g.cfg, 'method', 'nufft', ...
                                 'devices', primary, 'share_primary', true);
    else
        toep = Om * 1;
        op = mr_aha.create_op(SEs, g.phi2, g.traj, g.cfg, 'devices', primary, ...
                                 'share_primary', true, '_phi2_toep', toep);
    end
end

function nfail = leak_check(primary, devs)
%LEAK_CHECK  Repeated create / apply / delete of both methods on DEVS (or
%   the primary only) must not change this process's memory on any GPU.
    nfail = 0;
    % A larger 3D problem, so a per-cycle leak would be visible above the
    % allocator noise.
    big = make_problem('3D', [32 32 32], 4, 8);        % ~4 MB x, ~26 MB kernel
    cycle = @(m, share) leak_once(big, m, devs, share);
    % Warm-up: CUDA contexts on every GPU, cuFFT / cufinufft state and
    % MATLAB's memory pool reach their steady state before the baseline.
    for it = 1:2                                       % warm up (contexts, plans)
        cycle('toeplitz', true); cycle('nufft', true);
    end
    wait(gpuDevice());
    % Baseline, then 20 cycles alternating share_primary, then measure again.
    m0 = process_gpu_memory();
    a0 = gpuDevice().AvailableMemory;
    for it = 1:10
        cycle('toeplitz', mod(it, 2) == 0);
        cycle('nufft', mod(it, 2) == 1);
    end
    wait(gpuDevice());
    m1 = process_gpu_memory();
    a1 = gpuDevice().AvailableMemory;
    where = sprintf('gpuDevice(%d)', primary);
    if ~isempty(devs), where = ['[' num2str(devs) ']']; end
    % Without nvidia-smi only the primary's AvailableMemory can be compared
    % (8 MiB allowance); otherwise this process's usage on every GPU, which
    % may grow by at most 4 MiB.
    if isempty(m0)
        d_mb = (a0 - a1) / 2^20;
        nfail = nfail + report(['no leak over 20 create/apply/delete on ' where ...
                                ' (primary AvailableMemory)'], d_mb < 8, ...
                               sprintf('%.1f MB', d_mb));
    else
        keys = union(m0.keys, m1.keys);
        msg = ''; ok = true;
        for k = 1:numel(keys)
            u0 = get_or0(m0, keys{k}); u1 = get_or0(m1, keys{k});
            msg = [msg sprintf('gpu%s %d->%d MiB  ', keys{k}, u0, u1)]; %#ok<AGROW>
            ok = ok && (u1 - u0) <= 4;
        end
        nfail = nfail + report(['no leak over 20 create/apply/delete on ' where], ok, msg);
    end
end

function leak_once(g, method, devs, share)
%LEAK_ONCE  One create / apply / delete cycle.
    op = mr_aha.create_op(g.SEs, g.phi2, g.traj, g.cfg, 'method', method, ...
                             'devices', devs, 'share_primary', share);
    op.apply(g.x);
    delete(op);
end

function n = op_count(metal)
%OP_COUNT  Live operators of the MEX registry (0 on Metal).
    if metal, n = 0; else, n = mr_aha_mex('count'); end
end

function finish(nfail, what)
%FINISH  Error with the number of failures, or report success.
    if nfail > 0
        error('MR_AHA:test', 'check_op_matlab: %d check(s) failed.', nfail);
    end
    fprintf('check_op_matlab: %s passed\n', what);
end

function r = rel(a, b)
%REL  Relative l2 difference ||a - b|| / ||b|| in double precision.
    a = double(gather(a(:))); b = double(gather(b(:)));
    if numel(a) ~= numel(b), r = inf; return, end
    r = norm(a - b) / norm(b);
end

function k = current_index()
%CURRENT_INDEX  Index of MATLAB's current gpuDevice (does not change it).
    k = gpuDevice().Index;
end

function id = error_id(f)
%ERROR_ID  Identifier of the error raised by f(), or '' when none.
    id = '';
    try
        f();
    catch err
        id = err.identifier;
        if isempty(id), id = '(no identifier)'; end
    end
end

function bad = report(name, ok, detail)
%REPORT  Print one check result; returns 1 on failure.
    if ok, tag = 'ok  '; else, tag = 'FAIL'; end
    fprintf('  %s %s %s\n', tag, name, detail);
    bad = double(~ok);
end

function m = process_gpu_memory()
%PROCESS_GPU_MEMORY  MiB used by this MATLAB process on every GPU (CUDA
%   ordinal -> MiB), from nvidia-smi; empty when nvidia-smi is unavailable.
    m = [];
    % Map each GPU UUID to nvidia-smi's index, then pick this process's rows
    % from the compute-apps list. Note: the keys are nvidia-smi indices,
    % which follow PCI order, not necessarily CUDA_VISIBLE_DEVICES order.
    pid = feature('getpid');
    [s1, gpus] = system('nvidia-smi --query-gpu=index,uuid --format=csv,noheader');
    [s2, apps] = system(['nvidia-smi --query-compute-apps=pid,gpu_uuid,' ...
                         'used_memory --format=csv,noheader,nounits']);
    if s1 ~= 0 || s2 ~= 0, return, end
    uuid2idx = containers.Map();
    for line = splitlines(strtrim(gpus)).'
        f = strtrim(split(line{1}, ','));
        if numel(f) >= 2, uuid2idx(f{2}) = f{1}; end
    end
    m = containers.Map();
    for line = splitlines(strtrim(apps)).'
        f = strtrim(split(line{1}, ','));
        if numel(f) >= 3 && str2double(f{1}) == pid && isKey(uuid2idx, f{2})
            m(uuid2idx(f{2})) = str2double(f{3});
        end
    end
end

function v = get_or0(m, k)
%GET_OR0  m(k), or 0 when the key is absent.
    if isKey(m, k), v = m(k); else, v = 0; end
end
