%% Load test data

load ./Data/AhAx_test_toeplitz.mat;
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

%spread the Phi2_lower to cartedian grid
Phi2Cartesian = cufinufftfspread2d1(single(st.om(:,1)), single(st.om(:,2)), Phi2_lower, +1,1e-6,Ny, Nx, M, Nz * num_lower_coordinates);


%%

temp_input = zeros(size(filtAhb),'like',single(1i));
temp_input(161,161,1,:) = single(complex(0,1));
for i = 1:1
    tic;
    % d_AhAx = AhA_ps_cufinufft(filtAhb, st, SEs, Phi2Cartesian);
    d_AhAx = AhA_ps_cufinufft(temp_input, st, SEs, Phi2Cartesian);
    toc;
end

AhAx = gather(d_AhAx);
%%
% U0 = AhA_ps_cufinufft(filtAhb, st, SEs, Phi2);
% c2 = mean(abs(Ahb(:)),'all')./mean(abs(U0),'all');%real(pinv(U0(:))*Ahb(:));
% U0 = filtAhb*c2;
%
% tic;
% U = pcg(@(x) AhA_ps_cufinufft(x, st, SEs, Phi2), Ahb(:), [], 10, [], [], U0(:));
% U = reshape(U, st.Nd(1), st.Nd(2), st.Nz, []);
% toc;
