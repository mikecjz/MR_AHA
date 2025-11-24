function dcf = dcf_cufinufft3d(kx, ky, kz, N, ntrajs, num_iterations)

if nargin == 5
    num_iterations = 10;
end

%Density compensation function using the James Pipe method
% https://github.com/mind-inria/mri-nufft/blob/19c673d1b2718a3042f3dee33ea43a3fe05ca361/src/mrinufft/operators/interfaces/cufinufft.py#L852
    opts.gpu_spreadinterponly=1;
%     opts.gpu_kerevalmeth=0;

    d_kx = gpuArray(single(kx(:)));
    d_ky = gpuArray(single(ky(:)));
    d_kz = gpuArray(single(kz(:)));

    d_density_comp = gpuArray(1i * ones(N *ntrajs, 1, 'like', single(1i)));

    for i = 1:num_iterations
        d_temp_image = cufinufft3d1(d_kx, d_ky, d_kz, d_density_comp, -1, 1e-6, 2*N, 2*N, 2*N, opts);
        d_temp_density  = cufinufft3d2(d_kx, d_ky, d_kz, 1, 1e-6, d_temp_image, opts);

        d_density_comp = d_density_comp ./ abs(d_temp_density);
    end

    density_comp = reshape(abs(gather(d_density_comp)), N, []);

%     dcf = density_comp;

    % interpolate the first 1:60 and last end-59 : end using values from 61:end-60
    % coordiantes 
    x = 10:(N-9); % coordiantes

    v = density_comp(x,:); % values

    xq = 1:N;

%     dcf = interp1(x, v, xq, 'linear', 'extrap');
    dcf = density_comp;

    % Initialize Uniform Kspace

%     d_kspace = 1i * gpuArray(single(ones(N*ntrajs,1, 'like', single(1i))));
% 
%     d_image = cufinufft3d1(d_kx, d_ky, d_kz, d_kspace, -1, 1e-6, N, N, N);
% 
%     d_dcf_inv = cufinufft3d2(d_kx, d_ky, d_kz, 1, 1e-6, d_image);
% 
%     d_dcf = 1./d_dcf_inv;
% 
%     dcf = gather(reshape(d_dcf, N, []));
% 

end