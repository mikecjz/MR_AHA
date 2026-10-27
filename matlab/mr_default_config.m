function config = mr_default_config(traj_type, n_dim, L, varargin)
%MR_DEFAULT_CONFIG  Build a config struct for AHA_TOEPLITZ and CALC_TOEP.
%
%   config = MR_DEFAULT_CONFIG(traj_type, n_dim, L)
%   config = MR_DEFAULT_CONFIG(traj_type, n_dim, L, 'Name', Value, ...)
%
%   INPUTS
%     traj_type  '2D', '3D' or 'stack'.
%     n_dim      [Nx Ny] for '2D', [Nx Ny Nz] for '3D' and 'stack'.
%     L          number of ranks.
%
%   NAME-VALUE PAIRS
%     'phi2_dim'       upsampled Toeplitz size. Defaults to 2*n_dim, except
%                      along kz of a 'stack' trajectory where Mz = Nz because
%                      that direction is already Cartesian.
%     'triangulation'  'lower' (default) or 'upper'.
%     'is_masked'      false (default) or true.
%     'n_coils'        left empty by default and inferred from size(SEs).
%     'eps'            NUFFT tolerance for CALC_TOEP and AHA_CUFINUFFT,
%                      default 1e-6.
%     'mask_method'    how CALC_MASK builds the mask: 'neighbour' (default,
%                      keep locations within mask_radius grid cells of a
%                      sample) or 'threshold' (threshold the unit-weight
%                      Toeplitz kernel at mask_threshold).
%     'mask_radius'    neighbourhood radius of CALC_MASK in upsampled grid
%                      cells, default 2.
%     'mask_threshold' absolute threshold of CALC_MASK's 'threshold' method
%                      (in samples), default 0.05.
%     'mem_fraction'   share of free GPU memory CALC_TOEP may use, default 0.5.
%     'gpu_device_id'  -1 (default) keeps MATLAB's currently selected GPU.
%     'verbose'        false (default).
%
%   EXAMPLE
%     config = mr_default_config('stack', [256 256 64], 16);
%     Om     = calc_toep(phi2, traj, config);
%     AhAx   = AhA_Toeplitz(x, SEs, Om, config);
%
%     % masked operator, mask derived from the trajectory:
%     config = mr_default_config('stack', [256 256 64], 16, 'is_masked', true);
%     [Om, mask] = calc_toep(phi2, traj, config);
%     AhAx   = AhA_Toeplitz(x, SEs, Om, config, mask);
%
%   See also AHA_TOEPLITZ, CALC_TOEP, CALC_MASK, AHA_CUFINUFFT.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    p = inputParser();
    p.addParameter('phi2_dim', []);
    p.addParameter('triangulation', 'lower');
    p.addParameter('is_masked', false);
    p.addParameter('n_coils', []);
    p.addParameter('eps', 1e-6);
    p.addParameter('mask_method', 'neighbour');
    p.addParameter('mask_radius', 2);
    p.addParameter('mask_threshold', 0.05);
    p.addParameter('mem_fraction', 0.5);
    p.addParameter('gpu_device_id', -1);
    p.addParameter('verbose', false);
    p.parse(varargin{:});
    opt = p.Results;

    n_dim = double(n_dim(:)).';
    switch lower(traj_type)
        case '2d',    n_spatial = 2;
        case '3d',    n_spatial = 3;
        case 'stack', n_spatial = 3;
        otherwise
            error('MR_AHA:config:trajType', ...
                  'traj_type must be ''2D'', ''3D'' or ''stack''.');
    end
    if numel(n_dim) ~= n_spatial
        error('MR_AHA:config:nDim', ...
              'n_dim must have %d entries for traj_type ''%s''.', ...
              n_spatial, traj_type);
    end

    if isempty(opt.phi2_dim)
        phi2_dim = 2 * n_dim;
        if strcmpi(traj_type, 'stack')
            % kz is fully sampled and Cartesian, so it is never upsampled.
            phi2_dim(3) = n_dim(3);
        end
    else
        phi2_dim = double(opt.phi2_dim(:)).';
    end

    config = struct( ...
        'traj_type',       lower(traj_type), ...
        'n_dim',           n_dim, ...
        'L',               double(L), ...
        'triangulation',   lower(opt.triangulation), ...
        'phi2_dim',        phi2_dim, ...
        'is_masked',       logical(opt.is_masked), ...
        'eps',             double(opt.eps), ...
        'mask_method',     lower(opt.mask_method), ...
        'mask_radius',     double(opt.mask_radius), ...
        'mask_threshold',  double(opt.mask_threshold), ...
        'mem_fraction',    double(opt.mem_fraction), ...
        'gpu_device_id',   double(opt.gpu_device_id), ...
        'verbose',         double(opt.verbose));

    if ~isempty(opt.n_coils), config.n_coils = double(opt.n_coils); end
end
