close all hidden
load('./Data/FSU_cu_test_rank.mat');
%%
Ntrajs = 280;
Nkx = 320;
Ny = 320;
Nx = 320;
Nz = 56;
Ncoils = 12;
NM = st.M;
%%
% tempom = single(zeros(Ntrajs, Nkx)) + 1i * zeros(Ntrajs,Nkx);
% traj = 1:70;
% tempom(traj,81:240) = 1 - 1i;

tempom = single(randn(Ntrajs, Nkx) + 1i *randn(Ntrajs, Nkx));

% a = cufinufftfspread2d1(single(st.om(:,1)), single(st.om(:,2)), tempom, +1,1e-6,Ny,Nx,NM,1);
% b = cufinufftf2d1(single(st.om(:,1)), single(st.om(:,2)), tempom, +1,1e-6,Ny,Nx,NM,1);
% figure;imshow(abs(permute(a,[1,2])),[])
% figure;imshow(angle(permute(a,[1,2])),[-pi,pi])

% figure;imshow(abs(permute(b,[1,2])),[])
% figure;plot(st.om(traj:280:end,2),st.om(traj:280:end,1),'.');xlim([-pi,pi]);ylim([-pi,pi]);pbaspect([1 1 1]);
%% using spread

FSU_spread = cufinufftfspread2d1(single(st.om(:,1)), single(st.om(:,2)), filtFSU_rank1, +1,1e-6,Ny,Nx,NM, 56 * 12);
FSU_spread = reshape(FSU_spread, size(FSU_spread,1), size(FSU_spread,2), 56, 12);
FSU_spread = ifftshift(ifftshift(FSU_spread,1),2);

filtAhb_coils = ifft(ifft(ifft(FSU_spread,[],1),[],2),[],3);
filtAhb_coils = fftshift(fftshift(fftshift(filtAhb_coils,1),2),3);

filtAhb = sqrt(sum(filtAhb_coils .* conj(filtAhb_coils),4));
filtAhb = gather(filtAhb);

%% using cufinufftf2d1
filtAhb_rank1_coils = cufinufftf2d1(single(st.om(:,1)), single(st.om(:,2)), filtFSU_rank1, +1,1e-6,Ny,Nx,NM, 56 * 12);

filtAhb_rank1_coils = reshape(filtAhb_rank1_coils,Nx, Ny, Nz, Ncoils);

filtAhb_rank1_coils = ifft(filtAhb_rank1_coils,[],3);

filtAhb_rank1 = sqrt(sum(filtAhb_rank1_coils .* conj(filtAhb_rank1_coils),4));
