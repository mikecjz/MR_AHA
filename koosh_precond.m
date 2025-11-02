function M_inv_x = koosh_precond(x, st, DCF)

M = st.M;
N = st.Nd(1);

d_kx = gpuArray(st.om(:,1));
d_ky = gpuArray(st.om(:,2));
d_kz = gpuArray(st.om(:,3));

x = reshape(x, st.Nd);


kspace = gpuArray(zeros(M,1,'like', single(1i)));



kspace = cufinufft3d2(d_kx, d_ky, d_kz, 1, 1e-6, x);



%     FhFx(:,:,:,i) = cufinufft3d1(d_kx, d_ky, d_kz, kspace(:,i) .* DCF(:), -1, 1e-6, N, N,N);
AhAx = cufinufft3d1(d_kx, d_ky, d_kz, kspace(:) .* DCF(:) .* DCF(:), -1, 1e-6, N, N,N);


M_inv_x = AhAx(:) / (N^3) * 8;

fprintf(',')


end