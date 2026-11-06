%RUN_MATLAB_TESTS  Put MR_AHA on the path and run the MATLAB reference check.
%
%   From the repository root:
%       matlab -batch "run('tests/run_matlab_tests.m')"
%
%   Uses the MEX files in build/matlab (or $MR_AHA_MEX_DIR) and the reference
%   files in data/ref_mlx_mat (tests/export_reference_mat.py). See
%   CHECK_REFERENCE_MATLAB.

%   Author: Junzhou Chen (junzhou@chen.engineer)

mr_root = fileparts(fileparts(mfilename('fullpath')));
mex_dir = getenv('MR_AHA_MEX_DIR');
if isempty(mex_dir)
    mex_dir = fullfile(mr_root, 'build', 'matlab');
end
addpath(mex_dir, fullfile(mr_root, 'matlab'), fullfile(mr_root, 'tests'));
check_reference_matlab(fullfile(mr_root, 'data', 'ref_mlx_mat'));
