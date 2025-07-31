function AhAx = AhA_ps3D_toep(x, st, SEs, Phi2, dim)
% AhA_ps3D_toep: wrapper for AhA_cufinufft_toep

try

    if dim == 2
        AhAx = AhA_cufinufft_toep(x, st, SEs, Phi2) ;
    elseif dim == 3
        AhAx = AhA_cufinufft_toep_3D(x, st, SEs, Phi2) ;
    else
        error('dim must be 2 or 3');
    end

    AhAx = AhAx(:);
    fprintf('.')

catch errormsg

    fprintf(2,'AhA_ps3D_toep failed. \n');
    fprintf(2,'%s\n', errormsg.message);

end
