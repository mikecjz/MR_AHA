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
%       k  = phi2 * k                           % one L-by-L block per readout
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
%   No custom C/CUDA code is involved: the NUFFTs come from cufinufft's own
%   MATLAB interface (CUFINUFFT_PLAN) and everything else is gpuArray code.
%
%   INPUTS
%     x       complex single gpuArray, [Nx, Ny, L] ('2D') or [Nx, Ny, Nz, L].
%     SEs     complex single gpuArray, [Nx, Ny, Ncoils] or [Nx, Ny, Nz, Ncoils].
%     phi2    complex single gpuArray of Gram blocks Phi'*Phi,
%               [Ntrajs, L, L]        '2D' / '3D'
%               [Ntrajs, Nkz, L, L]   'stack' (centred kz order, Nkz == Nz)
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
%   The block-diagonal product is evaluated with PAGEMTIMES, one L-by-L times
%   L-by-Nkx product per readout, which avoids materialising the
%   [Ntrajs, Nkx, L, L] broadcast intermediate.
%
%   See also MR_NUFFT_PLAN, AHA_TOEPLITZ, CALC_TOEP.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    narginchk(5, 6);
    if nargin < 6 || isempty(plan)
        plan = mr_nufft_plan(traj, config);
    end

    L       = plan.L;
    img     = plan.img_dim;
    Ntrajs  = plan.Ntrajs;
    Nkx     = plan.Nkx;
    nvox    = prod(img);
    stack   = strcmp(plan.traj_type, 'stack');

    x    = reshape(single(x), [img, L]);
    nc   = numel(SEs) / nvox;
    SEs  = reshape(single(SEs), [nvox, nc]);
    phi2 = single(phi2);

    % Rearrange the Gram blocks once into pages of Phi.' so that
    % pagemtimes(k_page, P_page) = k_page * Phi.' = (Phi * k_page.').'.
    if stack
        Nz = img(3);
        P  = permute(reshape(phi2, [Ntrajs, Nz, L, L]), [4 3 1 2]); % [j i t z]
    else
        P  = permute(reshape(phi2, [Ntrajs, L, L]), [3 2 1]);       % [j i t]
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
            k = permute(reshape(k, [Ntrajs, Nkx, Nz, L]), [2 4 1 3]);  % [kx j t z]
            k = pagemtimes(k, P);                                       % [kx i t z]
            k = permute(k, [3 1 4 2]);                                  % [t kx z i]
        else
            k = permute(reshape(k, [Ntrajs, Nkx, L]), [2 3 1]);         % [kx j t]
            k = pagemtimes(k, P);                                       % [kx i t]
            k = permute(k, [3 1 2]);                                    % [t kx i]
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
