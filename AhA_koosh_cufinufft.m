function AhAx = AhA_koosh_cufinufft(x, st, SEs)

Ncoils = size(SEs, 4);
M = st.M;
N = st.Nd(1);

d_kx = gpuArray(st.om(:,1));
d_ky = gpuArray(st.om(:,2));
d_kz = gpuArray(st.om(:,3));

x = reshape(x, st.Nd);

x_SEs = x.* SEs;

kspace = gpuArray(zeros(M, Ncoils,'like', single(1i)));

for i = 1:Ncoils

kspace(:,i) = cufinufft3d2(d_kx, d_ky, d_kz, 1, 1e-6, x_SEs(:,:,:,i));

end

FhFx = gpuArray(zeros(N, N, N, Ncoils,'like', single(1i)));


for i = 1:Ncoils

    FhFx(:,:,:,i) = cufinufft3d1(d_kx, d_ky, d_kz, kspace(:,i), -1, 1e-6, N, N,N);

end

AhAx = sum(FhFx .* conj(SEs), 4);

AhAx = AhAx(:);

fprintf('.')


end