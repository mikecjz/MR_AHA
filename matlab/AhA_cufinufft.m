function AhAx = AhA_cufinufft(x, SEs, phi2, traj, config, plan)
%AHA_CUFINUFFT  Reference A'A evaluated directly with cufinufft (no Toeplitz).
%
%   AhAx = AHA_CUFINUFFT(x, SEs, phi2, traj, config)
%   AhAx = AHA_CUFINUFFT(x, SEs, phi2, traj, config, plan)
%
%   Evaluates the forward normal operator of low-rank non-Cartesian MRI on the
%   GPU the "vanilla" way, coil by coil:
%
%       im = x .* SEs(:,:,..,c)                 % broadcast over the L ranks
%       k  = type-2 NUFFT(im)                   % image -> k-space, sign -1
%       k  = k * phi2                           % one L-by-L block per readout,
%                                               % rank row vector from the right
%       im = type-1 NUFFT(k)                    % k-space -> image, sign +1
%       AhAx = AhAx + im .* conj(SEs(:,:,..,c))
%
%   Both NUFFTs are unnormalised, the convention reproduced by the Toeplitz
%   kernel of CALC_TOEP, so AHA_CUFINUFFT and AHA_TOEPLITZ agree to within the
%   NUFFT tolerance. For a 'stack' trajectory the image is first transformed
%   along z (centred, unnormalised FFT), the 2D NUFFT pair is applied to every
%   kz plane with the Gram block of that kz, and the result is transformed
%   back (unnormalised inverse FFT). The centring is realised by a phase
%   modulation folded into the coil map rather than by fftshift copies.
%
%   ROLE: this is the independent REFERENCE of the NUFFT route, not the
%   production implementation. No MR_AHA library code is involved: the
%   NUFFTs come from cufinufft's own MATLAB interface (CUFINUFFT_PLAN) and
%   everything else is gpuArray code. The tests use it to validate
%   MR_AHA.CREATE_OP(..., 'method', 'nufft'), which evaluates the same
%   operator inside the library (cufinufft's C API, several GPUs if asked,
%   no MATLAB per-coil loop). Use that operator in reconstructions. With the
%   Metal backend (Apple silicon) this reference runs through the Python
%   mr_aha.AhANufftPlan (MLX + mlx-nufft) on ordinary arrays.
%
%   INPUTS
%     x       complex single gpuArray, [Nx, Ny, L] ('2D') or [Nx, Ny, Nz, L].
%     SEs     complex single gpuArray, [Nx, Ny, Ncoils] or [Nx, Ny, Nz, Ncoils].
%     phi2    complex single gpuArray of Gram blocks Phi*Phi' (MR-Multitasking
%             convention: Phi is the rank-by-time basis Phi_rt, and the block
%             of a readout is Phi_rt(:,idx)*Phi_rt(:,idx)'),
%               [Ntrajs, L, L]        '2D' / '3D'
%               [Ntrajs, Nkz, L, L]   'stack' (centred kz order, Nkz == Nz)
%             A MR-Multitasking Phi2 [L, L, Ntrajs, Nz] is
%             permute(Phi2, [3 4 1 2]) (fftshift along kz if needed).
%     traj    real single gpuArray [Ntrajs, Nkx, ndim], radians in [-pi, pi).
%             Ignored when plan is given.
%     config  struct from MR_DEFAULT_CONFIG.
%     plan    (optional) output of MR_NUFFT_PLAN(traj, config). Build it once
%             and reuse it inside iterative solvers; otherwise every call
%             re-plans and re-sorts the trajectory.
%
%   OUTPUT
%     AhAx    complex single gpuArray, the same size as x.
%
%   The block-diagonal product follows MR-Multitasking: the L rank
%   coefficients of a k-space sample form a row vector multiplied from the
%   right by the Gram block, k(1,:) <- k(1,:) * Phi2, i.e.
%   k_l <- sum_m k_m Phi2(m, l). It is evaluated with PAGEMTIMES, one
%   Nkx-by-L times L-by-L product per readout, which avoids materialising the
%   [Ntrajs, Nkx, L, L] broadcast intermediate.
%
%   See also MR_AHA, MR_NUFFT_PLAN, AHA_TOEPLITZ, CALC_TOEP.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    narginchk(5, 6);
    if nargin < 6 || isempty(plan)
        plan = mr_nufft_plan(traj, config);
    end

    if isfield(plan, 'backend') && strcmp(plan.backend, 'metal')
        % Apple silicon: Python reference operator, ordinary arrays.
        cs = @(a) complex(single(gather(a)));
        stack = strcmp(plan.traj_type, 'stack');
        AhAx = mr_from_numpy(plan.py.apply(mr_to_numpy(cs(x)), ...
                                           mr_to_numpy(cs(SEs)), ...
                                           mr_to_numpy(cs(phi2), 3 + stack)));
        AhAx = reshape(AhAx, size(x));
        return
    end

    L       = plan.L;
    img     = plan.img_dim;
    Ntrajs  = plan.Ntrajs;
    Nkx     = plan.Nkx;
    nvox    = prod(img);
    stack   = strcmp(plan.traj_type, 'stack');

    % Host inputs are moved to the GPU (cufinufft's MATLAB interface accepts
    % gpuArrays only), as AHA_TOEPLITZ does.
    x    = reshape(gpuArray(single(x)), [img, L]);
    nc   = numel(SEs) / nvox;
    SEs  = reshape(gpuArray(single(SEs)), [nvox, nc]);
    phi2 = gpuArray(single(phi2));

    % Rearrange the Gram blocks once into L-by-L pages so that
    % pagemtimes(k_page, P_page) = k_page * Phi2 (MR-Multitasking), with the
    % rank coefficients of each sample along the rows of k_page.
    if stack
        Nz = img(3);
        P  = permute(reshape(phi2, [Ntrajs, Nz, L, L]), [3 4 1 2]); % [m l t z]
    else
        P  = permute(reshape(phi2, [Ntrajs, L, L]), [2 3 1]);       % [m l t]
    end

    AhAx = zeros([img, L], 'like', x);
    for c = 1:nc
        % 1. Coil sensitivity, broadcast over the ranks. For a stack it also
        %    carries the z modulation that centres the z FFT (see
        %    MR_NUFFT_PLAN), so no fftshift copies are needed.
        se = reshape(SEs(:, c), img);
        if stack, se = se .* plan.zmod; end
        im = x .* se;

        % 2. Stack only: unnormalised FFT along z.
        if stack
            im = fft(im, [], 3);
        end

        % 3. Type-2 NUFFT: [Ntrajs*Nkx, ntr]. Only numel(im) is checked, and
        %    [Nx,Ny,Nz,L] is already [Nx,Ny,Nz*L] in memory for a stack.
        k = plan.t2.execute(im);
        clear im

        % 4. Block-diagonal multiplication, one L-by-L block per readout.
        if stack
            k = permute(reshape(k, [Ntrajs, Nkx, Nz, L]), [2 4 1 3]);  % [kx m t z]
            k = pagemtimes(k, P);                                       % [kx l t z]
            k = permute(k, [3 1 4 2]);                                  % [t kx z l]
        else
            k = permute(reshape(k, [Ntrajs, Nkx, L]), [2 3 1]);         % [kx m t]
            k = pagemtimes(k, P);                                       % [kx l t]
            k = permute(k, [3 1 2]);                                    % [t kx l]
        end

        % 5. Type-1 NUFFT back onto the image grid.
        im = reshape(plan.t1.execute(reshape(k, [], plan.ntr)), [img, L]);
        clear k

        % 6. Stack only: unnormalised inverse FFT along z.
        if stack
            im = ifft(im, [], 3) * Nz;
        end

        % 7. Adjoint coil sensitivity (and z demodulation), accumulated.
        AhAx = AhAx + im .* conj(se);
        clear im se
    end
end
