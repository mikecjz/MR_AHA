function b = mr_backend()
%MR_BACKEND  GPU backend of the MR_AHA MEX files on the path.
%
%   b = MR_BACKEND() returns 'cuda' (NVIDIA GPUs: arrays are gpuArrays) or
%   'metal' (Apple silicon: arrays are ordinary MATLAB arrays in unified
%   memory, which the GPU reads in place). The answer comes from
%   MR_BACKEND_MEX, built with the library, and is cached.
%
%   See also AHA_TOEPLITZ, CALC_TOEP, CALC_MASK.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    % The backend is fixed at build time, so it is queried once per session.
    % exist(...) == 3 means a MEX file of that name is on the path.
    persistent cached
    if isempty(cached)
        if exist('mr_backend_mex', 'file') == 3
            cached = mr_backend_mex();
        else
            % MEX files built before mr_backend_mex existed are CUDA builds.
            cached = 'cuda';
        end
    end
    b = cached;
end
