function AhAx = AhA_ps3D_toep(x, st, SEs, Phi2)
% AhA_ps3D_toep: wrapper for AhA_cufinufft_toep

try

    AhAx = AhA_cufinufft_toep(x, st, SEs, Phi2) ;

catch errormsg

    fprintf(2,'AhA_ps3D_toep failed. \n');
    fprintf(2,'%s\n', errormsg.message);

end

AhAx = AhAx(:);
fprintf('.')
