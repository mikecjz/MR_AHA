function d_AhAx = AhA_toep_nodeconv(d_im, d_SEs, d_Phi2Cartesian)

    Nx = size(d_im, 1);
    Ny = size(d_im, 2);
    Nz = size(d_im, 3);
    Ncoils = size(d_SEs, 4);

    d_AhAx = gpuArray(zeros(Nx, Ny, Nz,'like',d_im));

    for i = 1:Ncoils

        d_im_SE_padded = gpuArray(zeros(2*Nx, 2*Ny, 2*Nz));

        d_im_SE_padded(Nx/2+1:3*Nx/2, Ny/2+1:3*Ny/2, Nz/2+1:3*Nz/2) = d_im .* d_SEs(:,:,:,i);

        %fft
        d_k = ft1d(ft1d(ft1d(d_im_SE_padded, 1),2),3);

        %times Phi2Cartesian
        d_k = d_k .* d_Phi2Cartesian;

        %ifft
        d_im_SE_padded = ift1d(ift1d(ift1d(d_k, 1),2),3);

        %crop
        d_im_SE = d_im_SE_padded(Nx/2+1:3*Nx/2, Ny/2+1:3*Ny/2, Nz/2+1:3*Nz/2);

        %sum over coils complex conjugate
        d_AhAx = d_AhAx + d_im_SE .* conj(d_SEs(:,:,:,i));

    end

end