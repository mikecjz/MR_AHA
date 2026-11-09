function check_reference_matlab(ref_dir)
%CHECK_REFERENCE_MATLAB  Check the MATLAB interface against the CUDA references.
%
%   CHECK_REFERENCE_MATLAB() runs the public MATLAB functions on every file of
%   data/ref_mlx_mat (written by tests/export_reference_mat.py from the CUDA
%   reference data of benchmarks/make_reference.py) and compares:
%
%     AhA_Toeplitz   with each stored phi2_toep (lower/upper, unmasked and both
%                    masks) against the stored AhAx;
%     calc_toep      against the stored phi2_toep, and the operator built from
%                    it against the stored AhAx (the operator-level check that
%                    matters when the NUFFT differs from the reference's);
%     calc_mask      'neighbour' (must match exactly) and 'threshold';
%     mr_aha_op      MR_AHA_OP.CREATE_OP with method 'toeplitz' (kernel
%                    computed by the operator; lower/upper, unmasked and both
%                    masks) against the stored AhAx, and with method 'nufft'
%                    against the stored NUFFT result and against AHA_CUFINUFFT;
%     AhA_cufinufft  (the independent NUFFT reference operator: finufft's own
%                    MATLAB interface, no library code) against the stored
%                    one.
%
%   Works with either backend (see MR_BACKEND): on CUDA the arrays go to the
%   GPU, on Metal they stay ordinary arrays. Errors if any check fails, so
%   `matlab -batch check_reference_matlab` exits non-zero.
%
%   The MEX files (build/matlab) and the matlab/ folder must be on the path.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    if nargin < 1
        ref_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
                           'data', 'ref_mlx_mat');
    end
    files = dir(fullfile(ref_dir, '*.mat'));
    if isempty(files)
        error('MR_AHA:test', 'No reference files in %s.', ref_dir);
    end
    fprintf('backend: %s\n', mr_backend());
    if ~strcmp(mr_backend(), 'metal') && exist('cufinufft_plan', 'file') ~= 2
        fprintf('cufinufft MATLAB interface not on the path: AhA_cufinufft is skipped\n');
    end
    tol_aha = 1e-5; tol_toep_ext = 5e-5;
    metal = strcmp(mr_backend(), 'metal');
    nfail = 0;
    for f = files'
        z = load(fullfile(f.folder, f.name));
        if ~isfield(z, 'gram_convention') || ~strcmp(z.gram_convention, 'multitasking')
            fprintf('FAIL %s: written before the MR-Multitasking Gram convention; regenerate it\n', f.name);
            nfail = nfail + 1;
            continue
        end
        base = mr_default_config(z.traj_type, z.n_dim, double(z.L), ...
            'phi2_dim', z.phi2_dim, 'n_coils', double(z.n_coils), ...
            'eps', double(z.eps), 'mask_radius', double(z.cfg_mask_radius), ...
            'mask_threshold', double(z.cfg_mask_threshold));
        rows = {};
        for tri = {'lower', 'upper'}
            t = tri{1};
            cfg = base; cfg.triangulation = t;
            % AhA with the stored kernel, then calc_toep + AhA.
            rows(end+1, :) = {['AhAx_' t], rel(AhA_Toeplitz(z.x, z.SEs, ...
                z.(['phi2_toep_' t]), cfg), z.(['AhAx_' t])), tol_aha}; %#ok<AGROW>
            Om = calc_toep(z.phi2, z.traj, cfg);
            rows(end+1, :) = {['phi2_toep_' t], rel(Om, z.(['phi2_toep_' t])), ...
                              ifelse(metal, tol_toep_ext, 1e-5)}; %#ok<AGROW>
            rows(end+1, :) = {['AhAx_' t ' (new toep)'], rel(AhA_Toeplitz( ...
                z.x, z.SEs, Om, cfg), z.(['AhAx_' t])), tol_aha}; %#ok<AGROW>
            % The operator object, kernel computed from the Gram blocks.
            op = mr_aha_op.create_op(z.SEs, z.phi2, z.traj, cfg);
            rows(end+1, :) = {['AhAx_' t ' (create_op)'], rel(op * z.x, ...
                z.(['AhAx_' t])), tol_aha}; %#ok<AGROW>
            delete(op);
            for m = {'neighbour', 'threshold'}
                mm = m{1};
                cfgm = cfg; cfgm.is_masked = true;
                mask = z.(['mask_' mm]) + 1;
                suf = [t '_' mm];
                rows(end+1, :) = {['AhAx_' suf], rel(AhA_Toeplitz(z.x, z.SEs, ...
                    z.(['phi2_toep_' suf]), cfgm, mask), z.(['AhAx_' suf])), tol_aha}; %#ok<AGROW>
                Om = calc_toep(z.phi2, z.traj, cfgm, mask);
                rows(end+1, :) = {['AhAx_' suf ' (new toep)'], rel(AhA_Toeplitz( ...
                    z.x, z.SEs, Om, cfgm, mask), z.(['AhAx_' suf])), tol_aha}; %#ok<AGROW>
                op = mr_aha_op.create_op(z.SEs, z.phi2, z.traj, cfgm, 'mask', mask);
                rows(end+1, :) = {['AhAx_' suf ' (create_op)'], rel(op * z.x, ...
                    z.(['AhAx_' suf])), tol_aha}; %#ok<AGROW>
                delete(op);
            end
        end
        % Masks.
        cn = base; cn.mask_method = 'neighbour';
        ct = base; ct.mask_method = 'threshold';
        mn = gather(calc_mask(z.traj, cn)); mt = gather(calc_mask(z.traj, ct));
        rows(end+1, :) = {'mask_neighbour', setdiff_frac(mn, z.mask_neighbour + 1), 0}; %#ok<AGROW>
        rows(end+1, :) = {'mask_threshold', setdiff_frac(mt, z.mask_threshold + 1), 1e-3}; %#ok<AGROW>
        % NUFFT operator (library) and the independent NUFFT reference
        % (CUDA: needs cufinufft's MATLAB interface).
        has_nufft = metal || exist('cufinufft_plan', 'file') == 2;
        op = mr_aha_op.create_op(z.SEs, z.phi2, z.traj, base, 'method', 'nufft');
        y_op = op * z.x;
        delete(op);
        if isfield(z, 'AhAx_nufft')
            rows(end+1, :) = {'AhAx_nufft (create_op)', rel(y_op, z.AhAx_nufft), ...
                              tol_aha}; %#ok<AGROW>
        end
        if has_nufft
            y_ref = AhA_cufinufft(z.x, z.SEs, z.phi2, z.traj, base);
            rows(end+1, :) = {'create_op nufft vs AhA_cufinufft', rel(y_op, y_ref), ...
                              tol_aha}; %#ok<AGROW>
            if isfield(z, 'AhAx_nufft')
                rows(end+1, :) = {'AhAx_nufft', rel(y_ref, z.AhAx_nufft), ...
                                  tol_aha}; %#ok<AGROW>
            end
        end

        vals = cell2mat(rows(:, 2)); tols = cell2mat(rows(:, 3));
        bad = find(~(vals <= tols));
        [~, w] = max(vals ./ max(tols, eps));
        fprintf('%s %s: %d checks, worst %s = %.2e\n', ifelse(isempty(bad), 'ok  ', 'FAIL'), ...
                f.name(1:end-4), size(rows, 1), rows{w, 1}, vals(w));
        for b = bad'
            fprintf('       %s: %.3e > %.0e\n', rows{b, 1}, vals(b), tols(b));
        end
        nfail = nfail + numel(bad);
    end
    if nfail > 0
        error('MR_AHA:test', '%d check(s) failed.', nfail);
    end
    fprintf('all %d files passed\n', numel(files));
end

function r = rel(a, b)
%REL  Relative 2-norm difference ||a - b|| / ||b|| in double precision.
    a = double(gather(a(:))); b = double(b(:));
    if numel(a) ~= numel(b), r = inf; return, end
    r = norm(a - b) / max(norm(b), realmin);
end

function d = setdiff_frac(a, b)
%SETDIFF_FRAC  Share of indices in exactly one of two index sets.
    d = numel(setxor(double(a(:)), double(b(:)))) / max(numel(b), 1);
end

function v = ifelse(c, a, b)
%IFELSE  Inline conditional.
    if c, v = a; else, v = b; end
end
