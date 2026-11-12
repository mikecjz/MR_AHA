function AhAx = AhA_cufinufft(x, SEs, phi2, traj, config, plan)
%AHA_CUFINUFFT  One-shot reference A'A evaluated with cufinufft (no Toeplitz).
%
%   AhAx = AHA_CUFINUFFT(x, SEs, phi2, traj, config)
%   AhAx = AHA_CUFINUFFT(x, SEs, phi2, traj, config, plan)
%
%   Builds an AHA_NUFFT_PLAN for traj, applies it once and releases it. With
%   a plan (from AHA_NUFFT_PLAN(traj, config)), it is plan.apply(x, SEs,
%   phi2) and traj is ignored. Inside an iterative loop, build the plan once
%   and call plan.apply directly: planning sorts the trajectory.
%
%   This is the independent REFERENCE of the NUFFT route (no MR_AHA library
%   code); see AHA_NUFFT_PLAN for the inputs, the algorithm and the Metal
%   backend. Use MR_AHA.CREATE_OP in reconstructions.
%
%   See also AHA_NUFFT_PLAN, MR_AHA, AHA_TOEPLITZ, CALC_TOEP.

%   Author: Junzhou Chen (junzhou@chen.engineer)

    narginchk(5, 6);
    if nargin < 6 || isempty(plan)
        plan = aha_nufft_plan(traj, config);
    end
    AhAx = plan.apply(x, SEs, phi2);
end
