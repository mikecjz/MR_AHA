%% Set The following paths according to your server environments
cuda_lib_path   = '/usr/local/cuda-12.0/lib64';

finufft_path = '/home/jc_350/Toolboxes/finufft';

% mainpath should point to the MR-Multitasking Unified Recon folder
% Assuming cufinufft is properly compiled there
finufft_include = fullfile(finufft_path ,'include');
finufft_build   = fullfile(finufft_path ,'build');

%% Realease compilation

mexcuda('-v','./src/AhA_cufinufft_toep.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft','-lcublas')
mexcuda('-v','./src/cufinufftf2d1.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft','-lcublas')
mexcuda('-v','./src/cufinufftfspread2d1.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft','-lcublas')

%% Debug compilation

% mexcuda('-v','-G','-g','AhA_cufinufft_toep.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft','-lcublas')
% mexcuda('-v','-G','-g','cufinufftf2d1.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft','-lcublas')
% mexcuda('-v','-G','-g','cufinufftfspread2d1.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft','-lcublas')