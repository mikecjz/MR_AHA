load ./Data/AhAx_test_toeplitz.mat;
Phi2 = single(Phi2);
%% Spread Phi2
L = size(Phi2,1);
Ntrajs = size(Phi2,3);
M = st.M;
Nkx = M/Ntrajs;
Nx = st.Nd(1);
Ny = st.Nd(2);
Nz = st.Nz;
Phi2_permuted = permute(Phi2,[3,4,1,2]); % [Ntraj, Nz, L, L]

%LxL coordinates
[ROW_L, COL_L] = ndgrid(1:L, 1:L);

%index of lower coordinates
lower_indices = find(COL_L<=ROW_L);

%extract the lower triangular parts of Phi2, including the diagonal
Phi2_lower = Phi2_permuted(:,:,lower_indices); %[Ntraj, Nz, (L-1)*L/2 +L]
num_lower_coordinates = (L-1)*L/2 +L;

%extend the Nkx dimension
Phi2_lower = reshape(Phi2_lower, Ntrajs, 1, Nz, num_lower_coordinates);

%repeat along the readout dimension
Phi2_lower = repmat(Phi2_lower, [1, Nkx, 1, 1]);

%spread the Phi2_lower to cartedian grid
Phi2Cartesian = cufinufftfspread2d1(single(st.om(:,1)), single(st.om(:,2)), Phi2_lower, +1,1e-6,Ny, Nx, M, Nz * num_lower_coordinates);


%%
tic;
AhAx = AhA_ps_cufinufft(filtAhb, st, SEs, Phi2);
AhAx = reshape(AhAx, st.Nd(1), st.Nd(2), st.Nz, []);
toc;

%%
% U0 = AhA_ps_cufinufft(filtAhb, st, SEs, Phi2);
% c2 = mean(abs(Ahb(:)),'all')./mean(abs(U0),'all');%real(pinv(U0(:))*Ahb(:));
% U0 = filtAhb*c2;
% 
% tic;
% U = pcg(@(x) AhA_ps_cufinufft(x, st, SEs, Phi2), Ahb(:), [], 10, [], [], U0(:));
% U = reshape(U, st.Nd(1), st.Nd(2), st.Nz, []);
% toc;

%%
% load('FSU_cu_test_rank2.mat');
% tic;
% SU_rank2= cufinufftf2d1(single(st.om(:,1)),single(st.om(:,2)), filtFSU_rank2, 1, 1e-6, 320,320, 280*320, 56*12 * 2); 
% SU_rank2 = ifft(reshape(SU_rank2, 320,320,56,12,2),[],3);
% toc;
% 
% U_rank2 = squeeze(sum(SU_rank2 .*conj(SEs),4));
% 
% L = size(U_rank2,4);
% 
% Phi2 = zeros(L,L,280,56,'like',single(1));
% st.om = single(st.om);
% 
% U_SE_rank2 = AhA_ps_cufinufft(U_rank2, st, SEs, Phi2);
