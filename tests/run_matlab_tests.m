%RUN_MATLAB_TESTS  Put MR_AHA on the path and run the MATLAB checks.
%
%   From the repository root:
%       matlab -batch "run('tests/run_matlab_tests.m')"
%
%   Uses the MEX files in build/matlab (or $MR_AHA_MEX_DIR) and the
%   reference files in data/ref_mlx_mat (tests/export_reference_mat.py).
%   With the environment variable MR_AHA_TEST_CUFINUFFT set to 1, cufinufft's
%   MATLAB interface in <build>/finufft/matlab (built with
%   -DMR_AHA_BUILD_CUFINUFFT_MATLAB=ON) is put on the path as well, so the
%   checks also compare against the independent NUFFT reference
%   AHA_CUFINUFFT (opt-in: on some machines that MEX file aborts MATLAB on
%   some of the reference trajectories). Runs
%
%     CHECK_REFERENCE_MATLAB  the public functions and MR_AHA_OP.CREATE_OP
%                             (both methods) against the stored references;
%     CHECK_OP_MATLAB         the operator object MR_AHA_OP on the current
%                             gpuDevice. Its multi-GPU checks run only when
%                             the environment variable MR_AHA_TEST_MGPU is 1
%                             (opt-in because they occupy every visible GPU
%                             of a possibly shared machine) and more than one
%                             GPU is visible.

%   Author: Junzhou Chen (junzhou@chen.engineer)

mr_root = fileparts(fileparts(mfilename('fullpath')));
mex_dir = getenv('MR_AHA_MEX_DIR');
if isempty(mex_dir)
    mex_dir = fullfile(mr_root, 'build', 'matlab');
end
addpath(mex_dir, fullfile(mr_root, 'matlab'), fullfile(mr_root, 'tests'));
cufinufft_dir = fullfile(fileparts(mex_dir), 'finufft', 'matlab');
if strcmp(getenv('MR_AHA_TEST_CUFINUFFT'), '1') && ...
        isfile(fullfile(cufinufft_dir, 'cufinufft_plan.m'))
    addpath(cufinufft_dir);
end
check_reference_matlab(fullfile(mr_root, 'data', 'ref_mlx_mat'));
if strcmp(getenv('MR_AHA_TEST_MGPU'), '1')
    check_op_matlab('multi', 'auto');
else
    check_op_matlab('multi', false);
end
