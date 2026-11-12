function A = mr_from_numpy(p)
%MR_FROM_NUMPY  NumPy array in Python mr_aha's row-major layout -> MATLAB array.
%
%   A = MR_FROM_NUMPY(p) is the inverse of MR_TO_NUMPY: a C-contiguous NumPy
%   (L, Nz, Ny, Nx) array becomes a MATLAB [Nx, Ny, Nz, L] array with the same
%   element order. complex64 and float32 become single, int64 stays int64.
%
%   See also MR_TO_NUMPY, MR_PYTHON.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    t = p.T;                                   % F-ordered, MATLAB's shape
    switch char(t.dtype.kind)
        case {'c', 'f'}
            A = single(t);
        case {'i', 'u'}
            A = int64(t);
        otherwise
            error('MR_AHA:numpy', 'Unsupported NumPy dtype %s.', char(py.str(t.dtype)));
    end
end
