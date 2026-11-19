function check_reference_matlab(ref_dir)
%CHECK_REFERENCE_MATLAB  Check the MATLAB interface against the CUDA references.
%
%   CHECK_REFERENCE_MATLAB() runs the public MATLAB functions on every file of
%   data/ref_mlx_mat (written by tests/export_reference_mat.py from the CUDA
%   reference data of tests/make_reference.py) and compares:
%
%     AhA_Toeplitz   with each stored phi2_toep (lower/upper, unmasked and
%                    masked) against the stored AhAx;
%     calc_toep      against the stored phi2_toep, and the operator built from
%                    it against the stored AhAx (the operator-level check that
%                    matters when the NUFFT differs from the reference's);
%     calc_mask      against the stored mask (must match exactly);
%     mr_aha         MR_AHA.CREATE_OP with method 'toeplitz' (kernel
%                    computed by the operator; lower/upper, unmasked and
%                    masked) against the stored AhAx, and with method 'nufft'
%                    against the stored NUFFT result and against AHA_CUFINUFFT;
%     AhA_cufinufft  (the independent NUFFT reference operator: finufft's own
%                    MATLAB interface, no library code) against the stored
%                    one.
%
%   Works with either backend (see MR_BACKEND): on CUDA the arrays go to the
%   GPU, on Metal they stay ordinary arrays. Errors if any check fails, so
%   `matlab -batch check_reference_matlab` exits non-zero.
%
%   Tolerances (relative 2-norm error in double precision): 1e-5 for every
%   AhAx comparison and for the NUFFT results; 1e-5 for the recomputed
%   phi2_toep on CUDA, 5e-5 on Metal (whose kernel uses a different NUFFT
%   than the cufinufft reference); 0 for the mask (share of indices in only
%   one of the two masks). Files exported before the MR-Multitasking Gram
%   convention (no gram_convention field) count as failures and are skipped.
%
%   Masks are stored 0-based (as the C API uses them) and converted to the
%   1-based indices of the MATLAB interface by adding one.
%
%   The MEX files (build/matlab) and the matlab/ folder must be on the path.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    % Default reference directory: data/ref_mlx_mat next to tests/. An empty
    % directory is an error, so a missing export is not mistaken for a pass.
    if nargin < 1
        ref_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
                           'data', 'ref_mlx_mat');
    end
    files = dir(fullfile(ref_dir, '*.mat'));
    if isempty(files)
        error('MR_AHA:test', 'No reference files in %s.', ref_dir);
    end
    % Without cufinufft's MATLAB interface on CUDA, the AhA_cufinufft
    % comparisons are skipped (announced once here). The tolerances are
    % explained in the help block above.
    fprintf('backend: %s\n', mr_backend());
    if ~strcmp(mr_backend(), 'metal') && exist('cufinufft_plan', 'file') ~= 2
        fprintf('cufinufft MATLAB interface not on the path: AhA_cufinufft is skipped\n');
    end
    tol_aha = 1e-5; tol_toep_ext = 5e-5;
    metal = strcmp(mr_backend(), 'metal');
    nfail = 0;
    % ---- One reference file per iteration ------------------------------
    for f = files'
        z = load(fullfile(f.folder, f.name));
        % Refuse files made for the old Gram-block convention: their phi2
        % blocks are transposed relative to the current one.
        if ~isfield(z, 'gram_convention') || ~strcmp(z.gram_convention, 'multitasking')
            fprintf('FAIL %s: written before the MR-Multitasking Gram convention; regenerate it\n', f.name);
            nfail = nfail + 1;
            continue
        end
        % Rebuild the configuration the references were made with. Every
        % check appends one row {name, relative error, tolerance} to rows.
        base = mr_default_config(z.traj_type, z.n_dim, double(z.L), ...
            'phi2_dim', z.phi2_dim, 'n_coils', double(z.n_coils), ...
            'eps', double(z.eps), 'mask_radius', double(z.cfg_mask_radius));
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
            op = mr_aha.create_op(z.SEs, z.phi2, z.traj, cfg);
            rows(end+1, :) = {['AhAx_' t ' (create_op)'], rel(op * z.x, ...
                z.(['AhAx_' t])), tol_aha}; %#ok<AGROW>
            delete(op);
            % Masked variants: the stored keys carry the suffix 'neighbor'
            % (the neighborhood mask of CALC_MASK).
            % Masked kernel, recomputed kernel and operator object; the
            % recomputed masked kernel is judged through its AhAx only.
            cfgm = cfg; cfgm.is_masked = true;
            mask = z.mask_neighbor + 1;
            suf = [t '_neighbor'];
            rows(end+1, :) = {['AhAx_' suf], rel(AhA_Toeplitz(z.x, z.SEs, ...
                z.(['phi2_toep_' suf]), cfgm, mask), z.(['AhAx_' suf])), tol_aha}; %#ok<AGROW>
            Om = calc_toep(z.phi2, z.traj, cfgm, mask);
            rows(end+1, :) = {['AhAx_' suf ' (new toep)'], rel(AhA_Toeplitz( ...
                z.x, z.SEs, Om, cfgm, mask), z.(['AhAx_' suf])), tol_aha}; %#ok<AGROW>
            op = mr_aha.create_op(z.SEs, z.phi2, z.traj, cfgm, 'mask', mask);
            rows(end+1, :) = {['AhAx_' suf ' (create_op)'], rel(op * z.x, ...
                z.(['AhAx_' suf])), tol_aha}; %#ok<AGROW>
            delete(op);
        end
        % Mask.
        % calc_mask with the reference's radius (from base); index sets must
        % match exactly.
        mn = gather(calc_mask(z.traj, base));
        rows(end+1, :) = {'mask_neighbor', setdiff_frac(mn, z.mask_neighbor + 1), 0}; %#ok<AGROW>
        % NUFFT operator (library) and the independent NUFFT reference
        % (CUDA: needs cufinufft's MATLAB interface).
        % The library's NUFFT operator is always built; it is compared with
        % the stored cufinufft result when present, and with AhA_cufinufft
        % (plus AhA_cufinufft with the stored result) when that is available.
        has_nufft = metal || exist('cufinufft_plan', 'file') == 2;
        op = mr_aha.create_op(z.SEs, z.phi2, z.traj, base, 'method', 'nufft');
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

        % A row passes when error <= tolerance (NaN fails). The worst row is
        % ranked by error / tolerance, with eps guarding the zero-tolerance
        % mask row; every failing row is then listed.
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
    % Raise an error so `matlab -batch` exits non-zero on any failure.
    if nfail > 0
        error('MR_AHA:test', '%d check(s) failed.', nfail);
    end
    fprintf('all %d files passed\n', numel(files));
end

function r = rel(a, b)
%REL  Relative 2-norm difference ||a - b|| / ||b|| in double precision.
    % gather works on gpuArrays (CUDA) and is a no-op on ordinary arrays
    % (Metal). A size mismatch is reported as an infinite error.
    a = double(gather(a(:))); b = double(b(:));
    if numel(a) ~= numel(b), r = inf; return, end
    r = norm(a - b) / max(norm(b), realmin);
end

function d = setdiff_frac(a, b)
%SETDIFF_FRAC  Share of indices in exactly one of two index sets.
    % Normalized by the size of the reference set b.
    d = numel(setxor(double(a(:)), double(b(:)))) / max(numel(b), 1);
end

function v = ifelse(c, a, b)
%IFELSE  Inline conditional.
    if c, v = a; else, v = b; end
end
