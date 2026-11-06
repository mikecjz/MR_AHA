function check_mgpu_matlab(varargin)
%CHECK_MGPU_MATLAB  Validate the multi-GPU operator MR_AHA_OP (CUDA backend).
%
%   CHECK_MGPU_MATLAB()
%   CHECK_MGPU_MATLAB('device_sets', {[1 2], [1 2 3 4]})
%   CHECK_MGPU_MATLAB('device_sets', {[1 2], 1:4}, 'split_modes', {'off', 'on'})
%   CHECK_MGPU_MATLAB('device_sets', {3}, 'allow_single', true, ...
%                     'device_change', false)       % one GPU only
%
%   Runs on small random problems and errors if any check fails, so
%   `matlab -batch check_mgpu_matlab` exits non-zero on failure. Needs at
%   least two GPUs and skips (with a message) otherwise, unless
%   'allow_single' is true. The MEX files and the matlab/ folder must be on
%   the path (see RUN_MATLAB_TESTS).
%
%   CHECKS
%     equivalence  op.apply(x) and op * x against AHA_TOEPLITZ on the primary
%                  GPU alone, relative difference < tol, for every
%                  combination of
%                    traj_type      '2D', 'stack', '3D'
%                    coils          5, 8 (one divides 2 and 4 GPUs, one
%                                   does not)
%                    devices        each entry of 'device_sets'
%                    split          each entry of 'split_modes'
%                                   (MR_AHA_MGPU_SPLIT; multi-GPU sets only)
%                    mask           unmasked, masked (random 1-based mask)
%                    triangulation  'lower', 'upper'
%                    share_primary  false, true
%                  and that the operator really uses more than one GPU when
%                  the set has more than one, its primary is the current
%                  gpuDevice and the coils are all assigned.
%     device       MATLAB's current gpuDevice is unchanged by
%                  create / apply / info / delete.
%     share        with share_primary, the operator keeps the shared arrays
%                  alive after the caller clears them, and later in-place
%                  assignments to the caller's arrays do not affect it.
%     handles      apply on a deleted operator errors cleanly; destroying
%                  an unknown handle is harmless; clearing the variable
%                  destroys the plan.
%     clear mex    with an operator alive, `clear mex` leaves the (locked)
%                  MEX file loaded and the operator usable; deleting the last
%                  operator unlocks it.
%     leaks        repeated create / apply / delete leaves the memory used by
%                  this MATLAB process on every GPU unchanged (nvidia-smi;
%                  falls back to the primary's AvailableMemory).
%     reset        after reset(gpuDevice), apply raises
%                  'MR_AHA:mr_aha_op:deviceReset' and delete still works.
%     device change (optional) after gpuDevice(k) with another k, apply
%                  raises ':deviceChanged'; back on the primary,
%                  ':deviceReset' (selecting a GPU resets it).
%
%   NAME-VALUE PAIRS
%     'primary'        gpuDevice index (1-based) of the primary GPU. Default:
%                      the current gpuDevice.
%     'device_sets'    cell array of gpuDevice index lists (1-based). Default:
%                      {[primary, one other GPU], all GPUs}.
%     'split_modes'    values of MR_AHA_MGPU_SPLIT to test with multi-GPU
%                      sets, default {'off', 'on'}.
%     'allow_single'   run even with one GPU / single-GPU sets (exercises
%                      the MEX lifecycle only), default false.
%     'device_change'  run the gpuDevice(k) check, which selects a second
%                      GPU, default true.
%     'tol'            relative tolerance, default 1e-6.
%
%   Tiny problems normally stay on the primary GPU; this test sets
%   MR_AHA_MGPU_MIN_WORK=0 so that they are distributed. Both environment
%   variables are restored on return.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    p = inputParser();
    p.addParameter('primary', []);
    p.addParameter('device_sets', {});
    p.addParameter('split_modes', {'off', 'on'});
    p.addParameter('allow_single', false);
    p.addParameter('device_change', true);
    p.addParameter('tol', 1e-6);
    p.parse(varargin{:});
    opt = p.Results;

    if strcmp(mr_backend(), 'metal')
        fprintf('check_mgpu_matlab: Metal backend, multi-GPU is CUDA only: skipped\n');
        return
    end
    n_gpu = gpuDeviceCount();
    if n_gpu < 2 && ~opt.allow_single
        fprintf('check_mgpu_matlab: needs at least 2 GPUs, %d visible: skipped\n', n_gpu);
        return
    end
    if isempty(opt.primary)
        opt.primary = gpuDevice().Index;
    end
    if isempty(opt.device_sets)
        others = setdiff(1:n_gpu, opt.primary);
        if isempty(others)
            opt.device_sets = {opt.primary};     % allow_single on a 1-GPU box
        else
            opt.device_sets = {[opt.primary others(1)], 1:n_gpu};
        end
    end
    need = max([opt.primary, cellfun(@max, opt.device_sets)]);
    if need > n_gpu
        fprintf('check_mgpu_matlab: needs GPUs up to index %d, %d visible: skipped\n', ...
                need, n_gpu);
        return
    end
    if ~opt.allow_single && any(cellfun(@(d) numel(unique([d opt.primary])) < 2, ...
                                        opt.device_sets))
        error('MR_AHA:test', ['every device set must span 2+ GPUs ' ...
              '(or pass ''allow_single'', true)']);
    end
    big_set = opt.device_sets{end};
    fprintf('check_mgpu_matlab: primary gpuDevice(%d), device sets %s\n', ...
            opt.primary, strjoin(cellfun(@(d) ['[' num2str(d) ']'], ...
            opt.device_sets, 'UniformOutput', false), ' '));

    old_min_work = getenv('MR_AHA_MGPU_MIN_WORK');
    old_split = getenv('MR_AHA_MGPU_SPLIT');
    setenv('MR_AHA_MGPU_MIN_WORK', '0');
    restore_env = onCleanup(@() restore_envs(old_min_work, old_split)); %#ok<NASGU>

    gpuDevice(opt.primary);
    rng(1234);
    nfail = 0;
    worst = 0;

    % ------------------------------------------------------------------
    % Equivalence with the single-GPU AhA_Toeplitz
    % ------------------------------------------------------------------
    cases = {'2D',    [32 32],    3;
             'stack', [24 24 8],  3;
             '3D',    [16 16 12], 2};
    n_pass = 0; n_run = 0;
    for ic = 1:size(cases, 1)
        [traj, n_dim, L] = cases{ic, :};
        for nc = [5 8]
            prob = make_problem(traj, n_dim, L, nc);
            for tri = {'lower', 'upper'}
                for masked = [false true]
                    cfg = prob.cfg;
                    cfg.triangulation = tri{1};
                    cfg.is_masked = masked;
                    if masked
                        toep = prob.toep_masked;
                        mask = prob.mask;
                        % AhA_Toeplitz.m's own conversion of the mask,
                        % int64(gpuArray(mask)) - 1, needs 64-bit integer
                        % gpuArray arithmetic, which R2022b lacks; call the
                        % gateway with the 0-based int64 mask directly.
                        cfgr = cfg;
                        cfgr.num_masked = numel(mask);
                        ref = AhA_Toeplitz_mex(prob.x, prob.SEs, toep, cfgr, ...
                                               gpuArray(int64(mask) - 1));
                    else
                        toep = prob.toep;
                        mask = [];
                        ref = AhA_Toeplitz(prob.x, prob.SEs, toep, cfg);
                    end
                    for ids = 1:numel(opt.device_sets)
                        devs = opt.device_sets{ids};
                        multi = numel(unique([devs opt.primary])) > 1;
                        if multi, modes = opt.split_modes; else, modes = {'off'}; end
                        for im = 1:numel(modes)
                            setenv('MR_AHA_MGPU_SPLIT', modes{im});
                            for share = [false true]
                                n_run = n_run + 1;
                                op = mr_aha_op(prob.SEs, toep, cfg, 'devices', devs, ...
                                               'mask', mask, 'share_primary', share);
                                y1 = op.apply(prob.x);
                                y2 = op * prob.x;
                                in = op.info();
                                delete(op);
                                r = max(rel(y1, ref), rel(y2, ref));
                                worst = max(worst, r);
                                ok = r < opt.tol && ...
                                     (in.n_devices > 1) == multi && ...
                                     in.primary == opt.primary && ...
                                     in.devices(1) == opt.primary && ...
                                     sum(in.coil_count) + in.n_split_coils == nc && ...
                                     in.coil_begin(1) == 1 && ...
                                     all(ismember(in.devices, [devs opt.primary]));
                                n_pass = n_pass + ok;
                                if ~ok
                                    nfail = nfail + 1;
                                    fprintf(['  FAIL %-5s coils=%d %s masked=%d ' ...
                                             'devices=[%s] split=%s share=%d: ' ...
                                             'rel=%.2e, used [%s], coils [%s], ' ...
                                             'split coils %d\n'], traj, nc, ...
                                            tri{1}, masked, num2str(devs), ...
                                            modes{im}, share, r, ...
                                            num2str(in.devices), ...
                                            num2str(in.coil_count), ...
                                            in.n_split_coils);
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    setenv('MR_AHA_MGPU_SPLIT', old_split);
    fprintf('equivalence: %d / %d passed, worst relative difference %.3e (tol %.0e)\n', ...
            n_pass, n_run, worst, opt.tol);

    % A representative info() for the log.
    prob = make_problem('stack', [24 24 8], 3, 5);
    op = mr_aha_op(prob.SEs, prob.toep, prob.cfg, 'devices', big_set);
    disp(op.info());
    delete(op);

    % ------------------------------------------------------------------
    % Device hygiene: current gpuDevice unchanged
    % ------------------------------------------------------------------
    prob = make_problem('3D', [16 16 12], 2, 8);
    ref = AhA_Toeplitz(prob.x, prob.SEs, prob.toep, prob.cfg);
    idx0 = current_index();
    op = mr_aha_op(prob.SEs, prob.toep, prob.cfg, 'devices', big_set);
    i1 = current_index();
    y = op.apply(prob.x);
    i2 = current_index();
    op.info();
    i3 = current_index();
    delete(op);
    i4 = current_index();
    nfail = nfail + report('device unchanged by create/apply/info/delete', ...
        all([i1 i2 i3 i4] == idx0) && rel(y, ref) < opt.tol, ...
        sprintf('indices %s', num2str([idx0 i1 i2 i3 i4])));

    % ------------------------------------------------------------------
    % share_primary keeps the shared arrays alive and is copy-on-write safe
    % ------------------------------------------------------------------
    op = make_shared_op(prob, big_set);   % temporaries only
    junk = gpuArray.rand(4e6, 1, 'single');            % recycle freed memory
    junk(:) = 7; %#ok<NASGU>
    clear junk
    y = op.apply(prob.x);
    ok_alive = rel(y, ref) < opt.tol;
    SEs2 = prob.SEs; toep2 = prob.toep;
    op2 = mr_aha_op(SEs2, toep2, prob.cfg, 'devices', big_set, ...
                    'share_primary', true);
    SEs2(:) = 0; toep2(1:10) = 1e6;                    %#ok<NASGU> caller modifies
    clear SEs2 toep2
    y2 = op2.apply(prob.x);
    ok_cow = rel(y2, ref) < opt.tol;
    delete(op); delete(op2);
    nfail = nfail + report('share_primary: arrays kept alive after clear', ok_alive, '');
    nfail = nfail + report('share_primary: caller assignments do not leak in', ok_cow, '');

    % ------------------------------------------------------------------
    % Handles: deleted operator, double destroy, live-plan count
    % ------------------------------------------------------------------
    op = mr_aha_op(prob.SEs, prob.toep, prob.cfg, 'devices', opt.device_sets{1});
    delete(op);
    % MATLAB itself rejects methods of a deleted handle object ...
    id = error_id(@() op.apply(prob.x));
    % ... and the gateway rejects handles that are not live plans.
    id2 = error_id(@() mr_aha_op_mex('apply', uint64(12345), prob.x));
    ok_del = strcmp(id, 'MATLAB:class:InvalidHandle') && ...
             strcmp(id2, 'MR_AHA:mr_aha_op:invalidHandle');
    id = [id ' / ' id2];
    mr_aha_op_mex('destroy', uint64(12345));            % unknown handle: no-op
    op = mr_aha_op(prob.SEs, prob.toep, prob.cfg, 'devices', opt.device_sets{1});
    clear op
    nfail = nfail + report('apply after delete / on a bad handle errors', ok_del, id);
    nfail = nfail + report('clear op destroys the plan', ...
        mr_aha_op_mex('count') == 0 && ~mislocked('mr_aha_op_mex'), ...
        sprintf('count=%d', mr_aha_op_mex('count')));

    % ------------------------------------------------------------------
    % clear mex with a live operator
    % ------------------------------------------------------------------
    op = mr_aha_op(prob.SEs, prob.toep, prob.cfg, 'devices', big_set);
    locked_before = mislocked('mr_aha_op_mex');
    clear mex %#ok<CLMEX>
    locked_after = mislocked('mr_aha_op_mex');
    y = op.apply(prob.x);
    ok_clear = locked_before && locked_after && rel(y, ref) < opt.tol && ...
               mr_aha_op_mex('count') == 1;
    delete(op);
    ok_unlock = ~mislocked('mr_aha_op_mex') && mr_aha_op_mex('count') == 0;
    nfail = nfail + report('clear mex keeps the locked MEX and the operator usable', ...
                           ok_clear, '');
    nfail = nfail + report('deleting the last operator unlocks the MEX', ok_unlock, '');

    % ------------------------------------------------------------------
    % Leaks: repeated create / apply / delete
    % ------------------------------------------------------------------
    big = make_problem('3D', [32 32 32], 4, 8);         % ~5 MB x, ~26 MB kernel
    for it = 1:2                                        % warm up (contexts, cuFFT)
        op = mr_aha_op(big.SEs, big.toep, big.cfg, 'devices', big_set);
        op.apply(big.x); delete(op);
    end
    wait(gpuDevice());
    m0 = process_gpu_memory();
    a0 = gpuDevice().AvailableMemory;
    for it = 1:20
        op = mr_aha_op(big.SEs, big.toep, big.cfg, 'devices', big_set, ...
                       'share_primary', mod(it, 2) == 0);
        op.apply(big.x); delete(op);
    end
    wait(gpuDevice());
    m1 = process_gpu_memory();
    a1 = gpuDevice().AvailableMemory;
    if isempty(m0)
        d_mb = (a0 - a1) / 2^20;
        nfail = nfail + report('no leak (primary AvailableMemory)', d_mb < 8, ...
                               sprintf('%.1f MB', d_mb));
    else
        keys = union(m0.keys, m1.keys);
        msg = ''; ok = true;
        for k = 1:numel(keys)
            u0 = get_or0(m0, keys{k}); u1 = get_or0(m1, keys{k});
            msg = [msg sprintf('gpu%s %d->%d MiB  ', keys{k}, u0, u1)]; %#ok<AGROW>
            ok = ok && (u1 - u0) <= 4;
        end
        nfail = nfail + report('no leak over 20 create/apply/delete (per GPU)', ok, msg);
    end

    % ------------------------------------------------------------------
    % Device reset / device change
    % ------------------------------------------------------------------
    op = mr_aha_op(prob.SEs, prob.toep, prob.cfg, 'devices', big_set);
    xh = gather(prob.x);
    reset(gpuDevice());
    id = error_id(@() op.apply(xh));
    s = op.info();
    lastwarn('');
    delete(op);                         % must not crash; warns staleDestroy
    [~, wid] = lastwarn();
    nfail = nfail + report('reset(gpuDevice) then apply errors (deviceReset)', ...
        strcmp(id, 'MR_AHA:mr_aha_op:deviceReset') && ~s.valid, id);
    nfail = nfail + report('delete after reset is safe (staleDestroy warning)', ...
        strcmp(wid, 'MR_AHA:mr_aha_op:staleDestroy') && ...
        mr_aha_op_mex('count') == 0 && ~mislocked('mr_aha_op_mex'), wid);

    if opt.device_change && gpuDeviceCount() > 1
        gpuDevice(opt.primary);
        prob = make_problem('3D', [16 16 12], 2, 8);
        op = mr_aha_op(prob.SEs, prob.toep, prob.cfg, 'devices', big_set);
        xh = gather(prob.x);
        other = setdiff(1:gpuDeviceCount(), opt.primary);
        gpuDevice(other(1));
        id1 = error_id(@() op.apply(xh));
        gpuDevice(opt.primary);
        id2 = error_id(@() op.apply(xh));
        delete(op);
        nfail = nfail + report('gpuDevice(k) then apply errors (deviceChanged)', ...
            strcmp(id1, 'MR_AHA:mr_aha_op:deviceChanged'), id1);
        nfail = nfail + report('back on the primary: apply errors (deviceReset)', ...
            strcmp(id2, 'MR_AHA:mr_aha_op:deviceReset'), id2);
    end
    nfail = nfail + report('registry empty and MEX unlocked at the end', ...
        mr_aha_op_mex('count') == 0 && ~mislocked('mr_aha_op_mex'), '');

    gpuDevice(opt.primary);
    if nfail > 0
        error('MR_AHA:test', 'check_mgpu_matlab: %d check(s) failed.', nfail);
    end
    fprintf('check_mgpu_matlab: all checks passed\n');
end

% ----------------------------------------------------------------------
% Helpers
% ----------------------------------------------------------------------

function prob = make_problem(traj, n_dim, L, nc)
%MAKE_PROBLEM  Random x, SEs, unmasked and masked kernels on the current GPU.
    cfg = mr_default_config(traj, n_dim, L);
    cfg.n_coils = nc;
    ntri = L * (L + 1) / 2;
    crand = @(varargin) complex(gpuArray.randn(varargin{:}, 'single'), ...
                                gpuArray.randn(varargin{:}, 'single'));
    prob.cfg = cfg;
    prob.x = crand([n_dim L]);
    prob.SEs = crand([n_dim nc]);
    prob.toep = crand([cfg.phi2_dim ntri]);
    npad = prod(cfg.phi2_dim);
    prob.mask = sort(randperm(npad, round(0.4 * npad))).';   % 1-based
    prob.toep_masked = crand([numel(prob.mask) ntri]);
end

function op = make_shared_op(prob, devs)
%MAKE_SHARED_OP  share_primary operator built from temporaries that go out
%   of scope on return, so only the operator keeps them alive.
    SEs = prob.SEs + 0;              % fresh storage
    toep = prob.toep * 1;
    op = mr_aha_op(SEs, toep, prob.cfg, 'devices', devs, 'share_primary', true);
end

function r = rel(a, b)
%REL  Relative l2 difference ||a - b|| / ||b|| in double precision.
    a = double(gather(a(:))); b = double(gather(b(:)));
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

function restore_envs(min_work, split)
%RESTORE_ENVS  Restore MR_AHA_MGPU_MIN_WORK and MR_AHA_MGPU_SPLIT.
    setenv('MR_AHA_MGPU_MIN_WORK', min_work);
    setenv('MR_AHA_MGPU_SPLIT', split);
end

function v = get_or0(m, k)
%GET_OR0  m(k), or 0 when the key is absent.
    if isKey(m, k), v = m(k); else, v = 0; end
end
