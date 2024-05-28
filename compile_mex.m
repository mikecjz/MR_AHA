%% Set The following paths according to your server environments
cuda_lib_path   = '/usr/local/cuda-12.0/lib64';

% mainpath should point to the MR-Multitasking Unified Recon folder
% Assuming cufinufft is properly compiled there
finufft_include = [mainpath '/supporting/nufft/finufft/include'];
finufft_build   = [mainpath '/supporting/nufft/finufft/build'];

%% Realease compilation

mexcuda('-v','AhA_ps_cufinufft.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft')
mexcuda('-v','cufinufftf2d1.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft')

%% Debug compilation

% mexcuda('-v','-G','-g','AhA_ps_cufinufft.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft')
% mexcuda('-v','-G','-g','cufinufftf2d1.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft')