%% Load data

load('./Data/toep3D_test.mat')

%% Setup GPU arrays

d_filtAhb1 = gpuArray(filtAhb1);
d_SEs = gpuArray(SEs);
d_Phi2Cartesian = gpuArray(Phi2Cartesian);
% d_SEs = gpuArray(1i * ones(N, N, N, 2,'like', single(1i)));


%% AhA test Toep
tic;
d_AhAx = AhA_cufinufft_toep_3D(d_filtAhb1, st, d_SEs, d_Phi2Cartesian);
% d_AhAx = AhA_toep_nodeconv(d_filtAhb1, d_SEs, d_Phi2Cartesian);
toc;

AhAx_toep = gather(d_AhAx);
AhAx_toep = reshape(AhAx_toep, N, N, N);

%% AhA test Toep 2x
tic;
d_AhAx = AhA_cufinufft_toep_3D(d_filtAhb1, st, d_SEs, d_Phi2Cartesian);
% d_AhAx = AhA_toep_nodeconv(d_filtAhb1, d_SEs, d_Phi2Cartesian);
toc;

AhAx_toep = gather(d_AhAx);
AhAx_toep = reshape(AhAx_toep, N, N, N);