% toep_test_3D.m

disp('Loading Data')

load('./Data/SENSE_test.mat', 'kx','ky','kz');

disp('Loading Data Complete')



%% st
N = size(kx,1);

kx = single(kx(:)/N * 2*pi);
ky = single(ky(:)/N * 2*pi);
kz = single(kz(:)/N * 2*pi);


st.om = [kx(:), ky(:), kz(:)];
st.Nd = [N, N, N];
st.M = numel(kx);
st.L = 1;
Ntrajs = numel(kx) / N;

disp('Setting Up GPU')


% gpuDevice(1);

disp('Setting Up GPU Complete')


%%

for i = 1:5
DCF_pipe = gather(dcf_pipe(kx, ky, kz, 1e-6, N, N, N, numel(kx),20));
end


DCF_pipe = abs(reshape((DCF_pipe),N, []));

close all hidden

figure;plot(DCF(:,line));
figure;plot(DCF_pipe(:,line));