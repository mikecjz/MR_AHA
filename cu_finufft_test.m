load AhAx_test.mat;

AhAx = AhA_ps_cufinufft(fftshift(fftshift(filtAhb,1),3), st, SEs, Phi2);

% FSU = reshape(AhAx, 280,320,56,12,16);
% SU= cufinufftf2d1(single(st.om(:,1)),single(st.om(:,2)), FSU, 1, 1e-5, 320,320, 280*320, 56*12 * 16); 
% SU = ifft(reshape(SU, 320,320,56,12,16),[],3);

%%
% tic;
% SU_rank2= cufinufftf2d1(single(st.om(:,1)),single(st.om(:,2)), filtFSU_rank2, 1, 1e-5, 320,320, 280*320, 56*12 * 2); 
% SU_rank2 = ifft(reshape(SU_rank2, 320,320,56,12,2),[],3);
% toc;
% 
% U_rank2 = squeeze(sum(SU_rank2 .*conj(SEs),4));
% 
% L = size(U_rank2,4);
% 
% Phi2 = zeros(L,L,280,56,'like',single(1));
% st.om = single(st.om);
% 
% U_SE_rank2 = AhA_ps_cufinufft(U_rank2, st, SEs, Phi2);