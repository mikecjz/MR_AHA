function d_Phi2Cartesian = get_toep(kx, ky, kz, N)

    d_kx = gpuArray(single(kx(:)));
    d_ky = gpuArray(single(ky(:)));
    d_kz = gpuArray(single(kz(:)));

    opts.gpu_spreadinterponly=1;

    kspace_ones = gpuArray(single(1i)*ones(2*N, 2*N, 2*N,'like', single(1i)));

    d_c = cufinufft3d2(d_kx, d_ky, d_kz, 1, 1e-6, kspace_ones, opts);

    d_Phi2Cartesian = cufinufft3d1(d_kx, d_ky, d_kz, d_c, 1, 1e-6, 2*N, 2*N, 2*N, opts);

    d_Phi2Cartesian = d_Phi2Cartesian ./single(1i);

end