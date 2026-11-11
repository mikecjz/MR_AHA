function cfg = mr_py_config(config)
%MR_PY_CONFIG  MATLAB config struct -> Python mr_aha.MRConfig.
%
%   cfg = MR_PY_CONFIG(config) builds the Python configuration through
%   mr_aha.make_config from the fields of a MATLAB config struct (see
%   MR_DEFAULT_CONFIG). Missing optional fields keep mr_aha's defaults.
%
%   See also MR_DEFAULT_CONFIG, MR_PYTHON.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    mr = mr_python();
    ints = @(v) py.list(num2cell(int64(v(:).')));
    args = {'phi2_dim', ints(config.phi2_dim)};
    if isfield(config, 'triangulation'), args = [args, {'triangulation', char(config.triangulation)}]; end
    if isfield(config, 'is_masked'),     args = [args, {'is_masked', logical(config.is_masked)}]; end
    if isfield(config, 'num_masked') && ~isempty(config.num_masked)
        args = [args, {'num_masked', int64(config.num_masked)}];
    end
    if isfield(config, 'n_coils') && ~isempty(config.n_coils)
        args = [args, {'n_coils', int64(config.n_coils)}];
    end
    if isfield(config, 'eps'),            args = [args, {'eps', double(config.eps)}]; end
    if isfield(config, 'mem_fraction'),   args = [args, {'mem_fraction', double(config.mem_fraction)}]; end
    if isfield(config, 'verbose'),        args = [args, {'verbose', logical(config.verbose)}]; end
    if isfield(config, 'mask_method'),    args = [args, {'mask_method', char(config.mask_method)}]; end
    if isfield(config, 'mask_radius'),    args = [args, {'mask_radius', int64(config.mask_radius)}]; end
    if isfield(config, 'mask_threshold'), args = [args, {'mask_threshold', double(config.mask_threshold)}]; end
    cfg = mr.make_config(char(config.traj_type), ints(config.n_dim), ...
                         int64(config.L), pyargs(args{:}));
end
