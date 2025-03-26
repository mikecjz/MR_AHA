%% Load test data

load ./Data/AhAx_test_toeplitz.mat

%% Transfer input data to GPU
d_filtAhb = gpuArray(filtAhb);
d_SEs = gpuArray(SEs);
d_Ahb = gpuArray(Ahb);
%% Spread Phi2
L = size(Phi2,1);
Ntrajs = size(Phi2,3);
M = st.M;
Nkx = M/Ntrajs;
Nx = st.Nd(1);
Ny = st.Nd(2);
Nz = st.Nz;
st.L = L;
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
% Phi2_lower = 1i*ones(size(Phi2_lower),'like',single(1i));

%spread the Phi2_lower to cartedian grid, note the outpt will be on GPU
d_Phi2Cartesian = cufinufftfspread2d1(single(st.om(:,1)), single(st.om(:,2)), Phi2_lower, +1,1e-6,Ny, Nx, M, Nz * num_lower_coordinates);
d_Phi2Cartesian = reshape(d_Phi2Cartesian,2 * Nx, 2 * Ny, Nz, []);


%% Test AhA

tic;

d_AhAx = AhA_cufinufft_toep(d_filtAhb, st, d_SEs, d_Phi2Cartesian);

toc;

AhAx = reshape(gather(d_AhAx),[Nx, Ny, Nz, L]);
%% Least Squares Reconstruction
d_U0 = AhA_ps3D_toep(d_filtAhb, st, d_SEs, d_Phi2Cartesian);
c2 = mean(abs(Ahb(:)),'all')./mean(abs(d_U0),'all');%real(pinv(U0(:))*Ahb(:));
d_U0 = d_filtAhb * c2;

tic;
d_U = pcg(@(x) AhA_ps3D_toep(x, st, d_SEs, d_Phi2Cartesian), d_Ahb(:), [], 10, [], [], d_U0(:));
d_U = reshape(d_U, st.Nd(1), st.Nd(2), st.Nz, []);
toc;

U = gather(d_U);

figure;
imshow(abs(U(:,:,1,1)),[])
