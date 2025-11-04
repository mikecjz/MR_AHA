function d_Phi2Cartesian = toeplitz_baron(st, is_gpu)

    if nargin < 2
        is_gpu = true;
    end

    M = st.M;
    N = st.Nd(1);

    if is_gpu
        d_kx = gpuArray(st.om(:,1));
        d_ky = gpuArray(st.om(:,2));
        d_kz = gpuArray(st.om(:,3));

        d_Phi2 = gpuArray(single(1i * ones(st.M, 1)));

        d_PSF = cufinufft3d1(d_kx, d_ky, d_kz, d_Phi2, -1, 1e-6, 2*N, 2*N,2*N);

        d_Phi2Cartesian = ft1d(ft1d(ft1d(d_PSF,1),2),3);


        d_Phi2Cartesian = d_Phi2Cartesian ./ single(1i);
    else

        kx = st.om(:,1);
        ky = st.om(:,2);
        kz = st.om(:,3);

        Phi2 = single(1i * ones(st.M, 1));

        PSF = finufft3d1(kx, ky, kz, Phi2, -1, 1e-6, 2*N, 2*N,2*N);

        Phi2Cartesian = ft1d(ft1d(ft1d(PSF,1),2),3);

        Phi2Cartesian = Phi2Cartesian ./ single(1i);

        Phi2Cartesian = Phi2Cartesian / (8*N^3);

        d_Phi2Cartesian = gpuArray(Phi2Cartesian);
    end

end