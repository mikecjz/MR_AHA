%% Set The following paths according to your server environments
cuda_lib_path   = '/usr/local/cuda-12/lib64';

finufft_path = '/home/chenj6/Toolboxes/finufft_toep';

% mainpath should point to the MR-Multitasking Unified Recon folder
% Assuming cufinufft is properly compiled there
finufft_include = fullfile(finufft_path ,'include');
finufft_build   = fullfile(finufft_path ,'build');
gpu_arch = '75';   

%% Realease compilation

mexcuda('-v',['NVCCFLAGS=-gencode=arch=compute_' gpu_arch ',code=sm_' gpu_arch],'./src/AhA_ps_cufinufft.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft','-lcublas')
mexcuda('-v',['NVCCFLAGS=-gencode=arch=compute_' gpu_arch ',code=sm_' gpu_arch],'./src/AhA_cufinufft_toep.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft','-lcublas')
mexcuda('-v',['NVCCFLAGS=-gencode=arch=compute_' gpu_arch ',code=sm_' gpu_arch],'./src/AhA_cufinufft_toep_3D.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft','-lcublas')
mexcuda('-v',['NVCCFLAGS=-gencode=arch=compute_' gpu_arch ',code=sm_' gpu_arch],'./src/cufinufftf2d1.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft','-lcublas')
mexcuda('-v',['NVCCFLAGS=-gencode=arch=compute_' gpu_arch ',code=sm_' gpu_arch],'./src/cufinufftfspread2d1.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft','-lcublas')
mexcuda('-v',['NVCCFLAGS=-gencode=arch=compute_' gpu_arch ',code=sm_' gpu_arch],'./src/cufinufftfspread3d1.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft','-lcublas')
mexcuda('-v',['NVCCFLAGS=-gencode=arch=compute_' gpu_arch ',code=sm_' gpu_arch],'./src/dcf_pipe.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft','-lcublas')

%% Debug compilation

% mexcuda('-v','-G','-g',['NVCCFLAGS=-gencode=arch=compute_' gpu_arch ',code=sm_' gpu_arch],'./src/AhA_ps_cufinufft.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft','-lcublas')
% mexcuda('-v','-G','-g',['NVCCFLAGS=-gencode=arch=compute_' gpu_arch ',code=sm_' gpu_arch],'./src/AhA_cufinufft_toep.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft','-lcublas')
% mexcuda('-v','-G','-g',['NVCCFLAGS=-gencode=arch=compute_' gpu_arch ',code=sm_' gpu_arch],'./src/AhA_cufinufft_toep_3D.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft','-lcublas')
% mexcuda('-v','-G','-g',['NVCCFLAGS=-gencode=arch=compute_' gpu_arch ',code=sm_' gpu_arch],'./src/cufinufftf2d1.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft','-lcublas')
% mexcuda('-v','-G','-g',['NVCCFLAGS=-gencode=arch=compute_' gpu_arch ',code=sm_' gpu_arch],'./src/cufinufftfspread2d1.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft','-lcublas')
% mexcuda('-v','-G','-g',['NVCCFLAGS=-gencode=arch=compute_' gpu_arch ',code=sm_' gpu_arch],'./src/dcf_pipe.cu','-R2018a',['-I' finufft_include],['-L' finufft_build],['-L' cuda_lib_path],'-lcufinufft','-lcufft','-lcublas')