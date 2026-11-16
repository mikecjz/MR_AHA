function p = mr_to_numpy(A, nd)
%MR_TO_NUMPY  MATLAB array -> NumPy array in the row-major layout of mr_aha.
%
%   p = MR_TO_NUMPY(A) returns a C-contiguous NumPy array holding A's
%   elements in A's own (column-major) memory order, with the dimensions
%   reversed: a MATLAB [Nx, Ny, Nz, L] array becomes a NumPy (L, Nz, Ny, Nx)
%   array. That is exactly the convention of the Python mr_aha package (and of
%   the C library), so no permutation of the data takes place.
%
%   p = MR_TO_NUMPY(A, nd) pads A's size with trailing ones to nd dimensions
%   first, restoring the singleton dimensions MATLAB drops (e.g. L = 1).
%
%   See also MR_FROM_NUMPY, MR_PYTHON.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    % Pad the size vector with trailing ones when nd is given, so the NumPy
    % array keeps every axis the C layout expects.
    sz = size(A);
    if nargin > 1 && numel(sz) < nd
        sz(end+1:nd) = 1;
    end
    % asarray wraps the MATLAB array as Fortran-ordered with MATLAB's shape.
    % ravel('F') walks it in memory order, and reshaping that flat vector to
    % the reversed shape gives a C-contiguous array whose layout matches the
    % original column-major buffer element for element.
    p = py.numpy.asarray(A);                        % F-ordered, MATLAB's shape
    p = p.ravel('F').reshape(int64(fliplr(sz)));    % C-ordered, reversed shape
end
