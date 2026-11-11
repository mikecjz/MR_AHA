# Licensing

Copyright (C) 2024-2026 Junzhou Chen

mr_aha is **dual-licensed**. You may use it under either of the following:

1. **The GNU Affero General Public License, version 3 only (AGPL-3.0-only)**,
   with the additional permission below. The full license text is in
   [`LICENSE`](../LICENSE).
2. **A commercial license** from the copyright holder, on terms agreed in a
   separate written agreement.

SPDX expression: `AGPL-3.0-only OR LicenseRef-MR-AHA-Commercial`

## Which license applies to me?

Under the AGPL you may use, study, modify and redistribute mr_aha free of
charge, for any purpose, including commercial use. The condition is that if you
**distribute** a work based on mr_aha, or **let users interact with it over a
network** (for example a hosted or cloud reconstruction service), you must make
the complete Corresponding Source of that work available to them under the
AGPL.

Some examples:

| Use | License |
|---|---|
| Academic or non-profit research | AGPL, free |
| Internal use inside a company, not distributed and not offered as a service to others | AGPL, free |
| An open-source project released under the AGPL (or GPLv3) | AGPL, free |
| A closed-source or proprietary product, scanner software, or hosted service that includes or is built on mr_aha | **Commercial license required** |

This table is a summary for convenience. Where it differs from the license
text, the license text governs.

## Commercial licensing

To include mr_aha in a proprietary product or service without the AGPL's
source-release obligations, contact the copyright holder for a commercial
license:

Junzhou Chen — <junzhou@chen.engineer>

## Additional permission under GNU AGPL version 3 section 7

If you modify this Program, or any covered work, by linking or combining it
with the NVIDIA CUDA Toolkit libraries (including, but not limited to, the
CUDA runtime, cuFFT and cuBLAS) or with MATLAB and its MEX/MATLAB Data API
libraries (or modified versions of any of these), containing parts covered by
the terms of the licenses of those libraries, the licensors of this Program
grant you additional permission to convey the resulting work. Corresponding
Source for a non-source form of such a combination shall include the source
code for the parts of this Program used as well as that of the covered work,
but need not include the source code of those libraries.

## Third-party components

The submodules under [`third_party/`](../third_party/) are separate works under
their own licenses. They are not covered by the license above:

| Component | License |
|---|---|
| [FINUFFT / cuFINUFFT](../third_party/finufft) | Apache-2.0 |
| [mlx-nufft](../third_party/mlx-nufft) | Apache-2.0 |
| [VkFFT](../third_party/VkFFT) | MIT |

## Contributions

To keep dual licensing possible, every contributor signs the
[Contributor License Agreement](CLA.md) (CLA) once, before their first pull
request is merged. Contributors keep their copyright. The CLA lets the copyright
holder distribute contributions under both the AGPL and commercial licenses.
Signing is a single comment on the pull request and is checked automatically;
see [`CONTRIBUTING.md`](../.github/CONTRIBUTING.md).
