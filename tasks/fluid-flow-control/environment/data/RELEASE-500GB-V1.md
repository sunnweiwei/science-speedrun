# HydroGym public 500 GB release v1

Historical specification. Its canonical manifest is now
[`release-500gb-v1.json`](release-500gb-v1.json). See [`RELEASE.md`](RELEASE.md)
for the current speed-aware selection. Commands and counts below describe
the historical experiment, not an instruction to switch current data back.

Data selection: **frozen**, `hydrogym-public-500gb-v1`, 2026-09-06.
Execution status: **not yet a CFD-certified/fully runnable benchmark**.

`release.json` is the authoritative inclusion/exclusion list and common
training/test membership. It contains **47 resource variants, 730 source files,
488,770,756,805 bytes (488.77 GB)** under a **500,000,000,000-byte** source-data
budget. Training and evaluation share one copy. This is our benchmark scope, not a claim
to reproduce the original paper's complete 61-environment suite. The earlier55
count describes parameter rows, not the number of solver/fidelity/objective
variants. This user-approved selection supersedes `hydrogym-public-resources-v1`
(62 variants, 1.09 TB); old local data roots and experiments are preserved.
Unrelated or excluded Hub environments are not additional training data.

The 500 GB budget applies only to fixed source assets. It excludes container
images, download caches, generated trajectories, model checkpoints, logs and
solver scratch. It does **not** set a training-transition count or a research
duration. Every selected variant retains all its published source files;
no restart fields or initial-state samples have been truncated or quantized.

| Family | Resource variants | Source GB |
| --- | ---: | ---: |
| Cylinder | 9 | 67.410 |
| Pinball | 11 | 34.689 |
| Cavity | 3 | 3.242 |
| Square cylinder | 2 | 0.157 |
| NACA0012, including gust mitigation | 12 | 186.257 |
| Cube | 1 | 46.191 |
| Sphere | 2 | 54.408 |
| DRA2303 jet, Ma0.2 | 1 | 93.473 |
| Kolmogorov | 1 | <0.001 |
| Differentiable turbulent channel | 1 | 0.009 |
| Nek turbulent channel | 1 | 0.020 |
| Divergent nozzle, 2D coarse/fine and 3D coarse | 3 | 2.914 |
| **Total** | **47** | **488.771** |

The selection retains all 30 previously included 2D variants and these 17 3D
variants, without introducing a held-out-task split:

- Cylinder Re200/3900; sphere Re300/3700; cube Re3700.
- Pinball Re30/75/100/150.
- NACA0012 Re10000/50000, AOA12, both lift-to-drag and gust-control objectives.
- JAX channel Re_tau180 and Nek TCFmini Re180.
- Coarse 3D nozzle and DRA2303 jet Re400000, Ma0.2.

Backends: 31 MAIA lattice-Boltzmann, 1 structured MAIA, 9 Firedrake, 2 JAX,
3 JAX-Fluids, and 1 Nek5000 v19 variant. An unimplemented evaluator is not a data
exclusion; those cases remain in the frozen set and block full grading until
integrated. No benchmark model score has been produced by this release.

## What is excluded, and why

All exclusions are explicit records in `release.json`, not selected according
to candidate quality. There are two separate kinds of exclusion.

**Budget/coverage choice: 15 resource-complete variants**, recorded with
`reason: not_selected_500gb_scope` and `resource_status: required_payloads_present`:

- Cube Re300 and Cylinder Re1000 (3D): retain the selected representative/pairs.
- Eight 3D NACA variants at Re100/1000 and AOA20/40: retain the paired
  Re10000/50000, AOA12, two-objective grid instead; the 2D variants remain.
- NACA0012 3D Re200000, AOA0, Nek v17: 174.53 GB single resource package.
- Three 3D square-cylinder variants: retain the selected 3D geometries instead.
- Fine 3D nozzle: retain the public coarse 3D and both 2D fidelities.

**Missing, incomplete or mismatched resources**, unchanged from the prior audit:

- The9 paper parameter rows without payloads:3D Cavity (2), stenotic pipe (2),
  flat-plate turbulent boundary layer (4), and DRA2303 Ma0.7 (1).
- Paper2D Cavity Re4200: public Re4140 is not silently relabelled Re4200.
  The organizer's reconstructed Re4200 baseline remains separate; it is not
  automatically a certified replacement asset. Re7500 remains included.
- The two distinct Cylinder MARL rows have no corresponding multi-agent
  actuation/configuration binding in their candidate YAMLs (which declare one
  agent). Their single-agent configurations remain included.
- DRA2303 Ma0.2 surface-wave variant: marker only; the jet variant is included.
- `SquareCylinder_2D_Re3900`: the published package has a grid but lacks
  `restart_.Netcdf`. Its two complete 2D variants remain included; its three
  complete 3D variants are outside the approved budget/coverage selection.

Two previously investigated packaging issues are **not missing physical data**.
These variants are now outside the 500 GB scope, for the explicit reasons above:

- `Nozzle3D_fine` supplies `restart_proc0/1/2.h5`, each containing four domain
  partitions, matching the declared12 partitions. Native runtime currently
  expects `restart.h5`; a lossless layout adaptation is still required. The
  files' grids and dataset shapes were inspected, but this is not a native
  solver restart/replay certification.
- Nek v17's native `NEK_INIT.get_Case_Files` uses `.map`, not `.ma2`, and removes
  `.sch` at startup. Those overstrict generic data-profile requirements must be
  corrected in its adapter, not used to drop the complete NACA input package.

No new geometry or initial fluid state is synthesized in v1. Firedrake's9
configuration identities are decoded from the public directory names and
checkpoint lists, with provenance to the pinned flow code. Their executable
observation/action/time-step binding remains to validate. Any later physical
reconstruction must have reproducible code, source hashes and native physical
checks; inclusion requires a new release, not silent changes to frozen tests.

## Training set versus test set

This is **known-task interactive training**, not a hidden-task split:

- Training may generate trajectories in all 47 declared environments, using
  their supplied initial states. There is no fixed offline sample count, and no
  new unrelated fluid corpus is included.
- Test membership is the same 47 cases. Independent replay uses one shared
  checkpoint with fixed parameters and learned normalization. A second seed
  label on a deterministic restart is not a new held-out flow.
- Development evaluation is visible and can inform global checkpoint selection;
  scoring episodes themselves cannot update the model or its replay buffer.
- Per-case measurement is the mean undiscounted native episode return. Rewards
  with different units are not blindly averaged into a supposed official score.
  Normalized tracking requires separately measured same-protocol anchors.

`suite.json` remains the required executable evaluation specification: exact
sensor coordinates, actuation, reward mapping, initialization selection,
horizons and backend launch settings must be bound and checked for every case.
It must declare `data_release_id` and `data_release_sha256`, cover exactly the
frozen case IDs, and use their original resource identities. The evaluator now
rejects a smaller/mutated suite against this release. Neither a data receipt nor
the `selection_status` field asserts that these native checks have passed.

## Reproducible materialization

From the task repository root, on a new machine:

```bash
python3 fluid-flow-control/environment/data/public_release.py --check
python3 fluid-flow-control/environment/data/materialize.py /large-disk/hydrogym-500gb-v1
python3 fluid-flow-control/environment/data/materialize.py /large-disk/hydrogym-500gb-v1 --download
```

The first materializer invocation is a download-free plan. The second downloads
only the fixed selection using `hf download` at the pinned commit, verifies
every file against `source-files.json`, and installs `release.json` plus a
materialization receipt. It does not install an unvalidated `suite.json`.
`--verify-only` is fully offline. `--case NAME` prepares a diagnostic subset;
`--inventory` explicitly selects the broader archive, not a larger training set.

To reuse an existing local data archive without copying another 488.77 GB:

```bash
python3 fluid-flow-control/environment/data/materialize.py /large-disk/hydrogym-500gb-v1 \
  --from-local /large-disk/existing-hydrogym
```

This creates only the selected files as hard links, requires the same filesystem,
refuses conflicting files and verifies all bytes. It never creates escaping
symlinks or silently copies a terabyte across filesystems. Hard-linked originals
and the release view must remain immutable; mount the release read-only. It
does not remove or alter the old archive or ongoing experiment outputs.

The standard release view on the development host is
`/mnt/workspace/hydrogym/releases/hydrogym-public-500gb-v1`.
Bind that directory using `TASK_DATA_DIR`. Git contains only small manifests,
code and documentation; large source assets remain hosted upstream.
On 2026-09-06, all 730 source files in this view passed full offline hash
verification. They are hard-linked to unchanged source payloads in the older
release root, with no second 488.77 GB physical copy. The receipt binds manifest
SHA256 `226b7059b4b6b34bb4a8f169374f2a6bc74bc4f780b22ccbc3171c06b34778f3`.
No executable `suite.json` is published by this resource verification.

## Remaining release acceptance

Membership, source bytes and the delivery interface are now fixed. Before
advertising a fully runnable task, the organizer must finish native bindings,
validate every included solver and replay the same portable model on all 47
cases. Baseline reproduction follows the same fixed protocol. No AutoResearch
agent is launched by constructing this data release, and eight H100s are not a
proven full-suite capacity guarantee. Full replay time and peak runtime disk
usage are not yet measured. These are execution/resource requirements, not
grounds to quietly change the case selection or drop DRA2303 from grading.
