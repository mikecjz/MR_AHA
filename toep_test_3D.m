% toep_test_3D.m
CURRENTDIR = pwd;


addpath('/home/chenj6/Toolboxes/MT_CUDA')
addpath('/home/chenj6/Toolboxes/finufft_toep/matlab')
addpath('/home/chenj6/projects/coronary/SENSE')
load('./Data/SENSE_test.mat');

BART_DIR = '/home/chenj6/BART/bart-0.9.00';
cd(BART_DIR);
startup;
cd(CURRENTDIR);

%%
% get DCF
load('./Data/SENSE_test.mat', 'kx','ky','kz');
N = size(kx,1);
Ntrajs = numel(kx) / N;

kx = single(kx(:)/N * 2*pi);
ky = single(ky(:)/N * 2*pi);
kz = single(kz(:)/N * 2*pi);

d_kx = gpuArray(kx);
d_ky = gpuArray(ky);
d_kz = gpuArray(kz);

% DCF = dcf_cufinufft3d(kx, ky, kz, N, Ntrajs, 20);
DCF = gather(dcf_pipe(kx, ky, kz, 1e-6, N, N, N, numel(kx),5));

% DCF = sdc3_MAT(crds,numIter,effMtx,verbose,osf);
% DCF = dcf_func([kx(:), ky(:), kz(:)],1);
% load('temp_DCF.mat')
% DCF = temp_DCF;

load('/home/chenj6/Desktop/DCF.mat')

DCF = reshape(DCF,N,[]);

channels = size(Raw_data1,2);
img_grid1 = zeros(N,N,N,channels, 'like', single(1i));
img_grid2 = zeros(N,N,N,channels, 'like', single(1i));
        
img_bp1 = zeros(N,N,N,channels, 'like', single(1i));
img_bp2 = zeros(N,N,N,channels, 'like', single(1i));

for ii = 1:channels

    disp(['Adjoint NUFFT for channel: ', num2str(ii)])
    noncart_kspace_1 = permute(squeeze(single(Raw_data1(:,ii,:))),[2 1]);
    noncart_kspace_2 = permute(squeeze(single(Raw_data2(:,ii,:))),[2 1]);

    d_filtered_kspace_1 = gpuArray(noncart_kspace_1 .* DCF);
    d_filtered_kspace_2 = gpuArray(noncart_kspace_2 .* DCF);

    d_kspace_1 = gpuArray(noncart_kspace_1);
    d_kspace_2 = gpuArray(noncart_kspace_2);
    
    
    d_image_fbp = cufinufft3d1(d_kx,d_ky,d_kz,d_filtered_kspace_1,-1,1e-6,N,N,N);
    img_grid1(:,:,:,ii) = gather(d_image_fbp);

    d_image_bp = cufinufft3d1(d_kx,d_ky,d_kz,d_kspace_1,-1,1e-6,N,N,N);
    img_bp1(:,:,:,ii) = gather(d_image_bp);

    d_image_fbp = cufinufft3d1(d_kx,d_ky,d_kz,d_filtered_kspace_2,-1,1e-6,N,N,N);
    img_grid2(:,:,:,ii) = gather(d_image_fbp);

    d_image_bp = cufinufft3d1(d_kx,d_ky,d_kz,d_kspace_2,-1,1e-6,N,N,N);
    img_bp2(:,:,:,ii) = gather(d_image_bp);
end

filtAhb1 = sum(img_grid1 .* conj(SEs), 4);
filtAhb2 = sum(img_grid2 .* conj(SEs), 4);

Ahb1 = sum(img_bp1 .* conj(SEs), 4);
Ahb2 = sum(img_bp2 .* conj(SEs), 4);

gpuDevice(1);
%% Sensitivity map

% kspace_1 = ft1d(ft1d(ft1d(img_grid1,1),2),3);
% calib_size  = [24, 24, 24];
% 
% fbp_k = kspace_1; %permute(kspace_1,[2, 1, 3, 4]);
% % calib_size = permute(calib_size, [2 1 3]);
% 
% 
% % evalstring = ['[SEs, emaps] = bart(''ecalib -g -m 1 -r ', num2str(calib_size(1)),':',num2str(calib_size(2)),':',num2str(calib_size(3)) , ' -c 0', ''', fbp_k);'];
% SEs = bart('ecalib -g -m 1 -r 24:24:24 -c 0', fbp_k);
% 
% tic;eval(evalstring);toc;
% thresh_eig = 0.7; %Eigenvalue Threshold
% image_support = single((emaps >= thresh_eig));
% image_non_support = 1-image_support;
% SEs = SEs.*image_support; % Crop coils maps based on eigen value threshold
% SEs = SEs + (image_non_support / sqrt(Ncoils));% Enforce non image area to be non-zero and sum of ShS to be ones
% SEs = permute(SEs,[2 1 3 4 5]);

%% st



st.om = [kx(:), ky(:), kz(:)];
st.Nd = [N, N, N];
st.M = numel(kx);
st.L = 1;

%% Spread to produce diagonal operator for Toeplitz operations

Phi2 = single(1i * ones(st.M, 1));

d_Phi2Cartesian = cufinufftfspread3d1(single(st.om(:,1)), single(st.om(:,2)), single(st.om(:,3)), Phi2, +1,1e-6, N, N, N, st.M, 1);
% d_Phi2Cartesian = gpuArray(single(1i* ones(2*N, 2*N, 2*N)));

d_Phi2Cartesian = d_Phi2Cartesian ./ single(1i);

%% Spread using native interface

d_Phi2Cartesian = get_toep(kx, ky, kz, N);

%% Setup GPU arrays

d_Ahb1 = gpuArray(Ahb1);
d_filtAhb1 = gpuArray(filtAhb1);
d_SEs = gpuArray(SEs);
% d_SEs = gpuArray(1i * ones(N, N, N, 2,'like', single(1i)));

range = 1:288;
delta = zeros(N, N, N, 'like', single(1i));
delta(range, range, range) = 1 + 1i;
d_delta = gpuArray(delta);
%% AhA test Toep
tic;
d_AhAx = AhA_cufinufft_toep_3D(d_filtAhb1, st, d_SEs, d_Phi2Cartesian);
% d_AhAx = AhA_toep_nodeconv(d_filtAhb1, d_SEs, d_Phi2Cartesian);
toc;

AhAx_toep = gather(d_AhAx);
AhAx_toep = reshape(AhAx_toep, N, N, N);

%% CG Test
d_img0 = AhA_ps3D_toep(d_filtAhb1, st, d_SEs, d_Phi2Cartesian, 3);

c2 = mean(abs(Ahb1(:)),'all')./mean(abs(d_img0),'all');%real(pinv(U0(:))*Ahb(:));
d_img0 = d_filtAhb1 * c2 * 5;


tic;
d_img = pcg(@(x) AhA_ps3D_toep(x, st, d_SEs, d_Phi2Cartesian, 3), d_Ahb1(:), [], 30, [], [], []);
d_img = reshape(d_img, st.Nd(1), st.Nd(2), st.Nd(3), []);
toc;

img_recon = gather(d_img);

%% AhA test cufinufft

tic;
d_AhAx = AhA_koosh_cufinufft(d_filtAhb1, st, d_SEs, DCF);
toc;

AhAx = gather(d_AhAx);
AhAx = reshape(AhAx, N, N, N);

%% CG Test
d_img0 = AhA_koosh_cufinufft(d_filtAhb1, st, d_SEs, DCF);

c2 = mean(abs(Ahb1(:)),'all')./mean(abs(d_img0),'all');%real(pinv(U0(:))*Ahb(:));
d_img0 = d_filtAhb1 * c2 * 5;


tic;
d_img = pcg(@(x) AhA_koosh_cufinufft(x, st, d_SEs, DCF), d_filtAhb1(:), [], 4, @(x)koosh_precond(x, st, DCF), [], d_filtAhb1(:));
d_img = reshape(d_img, st.Nd(1), st.Nd(2), st.Nd(3), []);
toc;

img_recon = gather(d_img);



