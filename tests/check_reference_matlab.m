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
%     AhA_cufinufft  (the NUFFT reference operator) against the stored one.
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
    tol_aha = 1e-5; tol_toep_ext = 5e-5;
    metal = strcmp(mr_backend(), 'metal');
    nfail = 0;
    for f = files'
        z = load(fullfile(f.folder, f.name));
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
            end
        end
        % Masks.
        cn = base; cn.mask_method = 'neighbour';
        ct = base; ct.mask_method = 'threshold';
        mn = gather(calc_mask(z.traj, cn)); mt = gather(calc_mask(z.traj, ct));
        rows(end+1, :) = {'mask_neighbour', setdiff_frac(mn, z.mask_neighbour + 1), 0}; %#ok<AGROW>
        rows(end+1, :) = {'mask_threshold', setdiff_frac(mt, z.mask_threshold + 1), 1e-3}; %#ok<AGROW>
        % NUFFT reference operator.
        if isfield(z, 'AhAx_nufft')
            rows(end+1, :) = {'AhAx_nufft', rel(AhA_cufinufft(z.x, z.SEs, z.phi2, ...
                z.traj, base), z.AhAx_nufft), tol_aha}; %#ok<AGROW>
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
