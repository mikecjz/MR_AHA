%% Set The following paths according to your server environments

finufft_include = '/home/jc_350/Toolboxes/finufft/include';
finufft_build   = '/home/jc_350/Toolboxes/finufft/build';

cuda_lib_path   = '/usr/local/cuda-12.0/lib64';

%% Realease compilation

compile_command_r = ['mexcuda -R2018a -I',finufft_include, ' -L',finufft_build, ' -lcufinufft -L',cuda_lib_path,...
    ' -lcufft AhA_ps_cufinufft.cu']
eval(compile_command_r);

%% Debug compilation

% compile_command_d = ['mexcuda -R2018a -g -G -I',finufft_include, ' -L',finufft_build, ' -lcufinufft -L',cuda_lib_path,...
%     ' -lcufft AhA_ps_cufinufft.cu']
% eval(compile_command_d);
