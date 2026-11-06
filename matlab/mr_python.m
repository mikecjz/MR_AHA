function mr = mr_python()
%MR_PYTHON  The Python mr_aha package, used by the Metal backend.
%
%   mr = MR_PYTHON() returns the Python module mr_aha, loaded in MATLAB's own
%   process. On Apple silicon there is no C/C++ NUFFT the MEX files could
%   call, so CALC_TOEP, the 'threshold' method of CALC_MASK and AHA_CUFINUFFT
%   run their non-uniform FFTs through mlx-nufft in Python, then hand the
%   rest to the same library (libmr_aha_c) the MEX files link.
%
%   Python selection, unless PYENV has already loaded an interpreter:
%     1. the interpreter named by the environment variable MR_AHA_PYTHON;
%     2. otherwise <repository>/.venv/bin/python.
%   It must provide numpy, mlx and mlx-nufft (see python/README.md) and be a
%   version MATLAB's PYENV supports (3.9-3.12 for R2024b). It is loaded
%   in-process (ExecutionMode "InProcess"), so arrays are exchanged without
%   inter-process copies.
%
%   The repository's python/ folder is put on Python's sys.path, and
%   mr_aha finds the library in <repository>/build/lib (or through
%   MR_AHA_LIBRARY).
%
%   See also PYENV, CALC_TOEP, CALC_MASK.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    persistent mod
    if ~isempty(mod)
        mr = mod;
        return
    end

    root = fileparts(fileparts(mfilename('fullpath')));   % repository root
    pe = pyenv;
    if pe.Status == "NotLoaded"
        exe = getenv('MR_AHA_PYTHON');
        if isempty(exe)
            exe = fullfile(root, '.venv', 'bin', 'python');
        end
        if isfile(exe)
            pyenv(Version=exe, ExecutionMode="InProcess");
        end
    end

    pyroot = fullfile(root, 'python');
    syspath = py.sys.path;
    if ~any(string(cell(syspath)) == string(pyroot))
        insert(syspath, int32(0), pyroot);
    end
    try
        mod = py.importlib.import_module('mr_aha');
    catch err
        error('MR_AHA:python', ['Could not import the Python package mr_aha ' ...
              '(%s). Check pyenv, MR_AHA_PYTHON and python/README.md.'], ...
              err.message);
    end
    mr = mod;
end
