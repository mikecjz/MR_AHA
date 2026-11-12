# Contributing to mr_aha

Thank you for your interest in improving mr_aha! Bug reports, questions,
benchmarks on new hardware and pull requests are all welcome.

## Before you start: the Contributor License Agreement

mr_aha is dual-licensed: it is published under the GNU AGPL v3, and the
copyright holder also offers commercial licenses (see
[`LICENSING.md`](../licenses/LICENSING.md)). To keep that possible, **every contributor
must sign the [Contributor License Agreement](../licenses/CLA.md) (CLA)** before their first
pull request is merged.

* You keep the copyright in your work. The CLA grants the project a license to
  distribute it under the AGPL, a commercial license, or other terms.
* You sign **once**, and the CLA then covers all of your later contributions.
* Signing is automatic. When you open your first pull request, a bot posts a
  comment with a link to the CLA. To sign, reply with the exact sentence

  > I have read the CLA Document and I hereby sign the CLA

  The check then turns green. If it does not, comment `recheck`.
* **Contributing as part of your job?** Your employer (company, hospital or
  university) probably owns what you write at work. Ask them for permission
  first. They may need to sign a Corporate CLA: contact <junzhou@chen.engineer>.

Issues and discussions do not need a CLA, only code and documentation that are
merged into the repository.

## Reporting bugs

Please open an issue that includes:

* the backend (CUDA or Metal), GPU, driver/CUDA version, OS, and the
  interface you used (C, Python, MATLAB);
* the `config` (traj_type, n_dim, L, phi2_dim, triangulation, is_masked) and,
  for operator objects, the method and devices;
* a minimal script that reproduces the problem, and the full error message.

## Making a change

1. For anything larger than a small fix, open an issue first so we can agree
   on the approach.
2. Fork the repository and create a branch from `main`.
3. Make your change, following the conventions below.
4. Build and run the tests (see below) on the backend(s) you changed.
5. Open a pull request against `main`. Describe what changed, why, and how
   you tested it (GPU, tests run, benchmark numbers for performance changes).

### Building

See [Installation](https://mikecjz.github.io/MR_AHA_docs/getting-started/install.html) in the documentation. In short:

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMR_AHA_BUILD_PYTHON=ON
cmake --build build -j
```

### Testing

| Interface | Command |
|---|---|
| Python, public functions against the stored references | `python tests/check_reference.py` |
| Python, operator object (`create_op`) | `python tests/check_op.py` |
| MATLAB | `matlab -batch "run('tests/run_matlab_tests.m')"` |

The multi-GPU checks run only when two or more GPUs are visible (MATLAB:
only with `MR_AHA_TEST_MGPU=1`). Please do not run them on shared machines
without asking their users.

## Conventions

* **Array layout and Gram blocks.** Follow the conventions in the
  [documentation](https://mikecjz.github.io/MR_AHA_docs/guide/conventions.html):
  column-major shapes in the documentation, centred k-space and image space,
  and the Gram-block convention (`v <- v * Phi2`).
* **Documentation in the code.** The comments are published as the API
  documentation, so write them generously: Doxygen comments (`/** ... */`,
  `@param`, `@return`) for C, C++ and CUDA, and docstrings for Python and
  MATLAB help text. Document every public function's array shapes, types and
  units.
* **Memory.** Free intermediate buffers as soon as they are no longer needed.
  GPU memory is the limiting resource for large 3D problems.
* **Accuracy.** Changes to the numerics must keep the existing checks passing
  at their tolerances. Say so explicitly in the pull request if a tolerance
  had to change, and why.
* **Performance.** For changes aimed at speed, include before/after timings
  (the scripts of [MR_AHA_benchmarks](https://github.com/mikecjz/MR_AHA_benchmarks)
  make that a one-liner) and the GPU they were measured on.

## Third-party code

Do not copy code from other projects into your contribution without saying
so in the pull request. Third-party code is accepted only if its license
allows use in both the AGPL release and closed-source commercial licenses,
for example MIT, BSD or Apache-2.0. **GPL, LGPL, AGPL and non-commercial
licenses cannot be accepted**, nor can code without a license. Keep the
original copyright and license notices, and add the component to the
third-party table in [`LICENSING.md`](../licenses/LICENSING.md).

---

## For maintainers: one-off CLA set-up

The CLA check is [`.github/workflows/cla.yml`](workflows/cla.yml)
(CLA Assistant Lite). To enable it:

1. Merge the workflow, `licenses/CLA.md` and this file into `main`. Workflows
   triggered by pull requests only run from the default branch.
2. Create the branch that stores the signatures, empty and unprotected:

   ```bash
   git switch --orphan cla-signatures
   git commit --allow-empty -m "CLA signatures"
   git push origin cla-signatures
   git switch main
   ```

   The bot creates `signatures/version1/cla.json` on it with the first
   signature. Do not create that file by hand.
3. In **Settings → Actions → General → Workflow permissions**, allow
   "Read and write permissions".
4. In the branch protection rule for `main`, make the `CLAAssistant` check
   required.

If the CLA text ever changes in substance, bump its version and change
`path-to-signatures` to `signatures/version2/cla.json`, so that everyone
signs the new version.
