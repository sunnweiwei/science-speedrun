# Native replay acceptance — 2026-09-08

Historical protocol acceptance record. The return values below predate the
fixed physical-metric reporting contract and are not the current primary
benchmark table. See [current PR validation](../../docs/VALIDATION.md) and
[verified physical results](../references/specialists/verified/README.md).

The **submission → independent per-case evaluator** implementation is usable.
This is not a declaration that the 39-case benchmark, paper scores or a jointly
trained research baseline have passed acceptance.

## Implemented and checked

- Frozen `hydrogym-policy/v1`: one portable inference/code/checkpoint package;
  resident reset/act/close protocol; no evaluation-time parameter updates.
- `evaluate_native.sh`: one/repeated/all case IDs, backend runtime dispatch,
  isolated package snapshot, source/profile identity checks, explicit outcomes,
  native horizons, finite signals, timeouts and diagnostic/progress reports.
- All **39** source/profile bindings passed preflight against the fully hashed
  Research v2 data. No membership or source bytes were changed.
- **126** tests passed without skips in an offline dependency-complete container.
  The host run skips the six NumPy/ML-dependent tests. Synthetic tests cover
  protocol failures, full-case accounting, snapshot relocation/mutation,
  resource/code changes, native completion, GPU allocation and process cleanup;
  they do not produce CFD benchmark results.
- A verifier image was built from the checked-in Dockerfile. A clean-image
  command replayed Cylinder and Channel sequentially using the same snapshot,
  with no task-code mount, no network, and read-only data/candidate mounts.
- A deliberately malformed, immediately exiting policy was also tested against
  real native Cylinder CFD. It produced `policy_error: Policy closed stdin`,
  cleaned up the solver, exited 2 in 4.73 seconds and wrote no reward file.

## Real completed native episodes

The same diagnostic checkpoint/package was used throughout:
`e7b1aceb0aefff1b27b23f3686d2579ecc9c2a924925e1ac656c9d535df55097`.
It is the earlier 2,000-control Channel PPO pilot (279,345 parameters), not a
newly trained 39-case controller or a reproduced paper baseline. Unknown task
names use the starter's zero task conditioning, not different checkpoints.
Its finite shape capacity covers the selected interfaces; successful plumbing
does not establish good control or compliant joint-training provenance.

| Case | Completed native controls | Undiscounted native return | End-to-end case time |
| --- | ---: | ---: | ---: |
| Cylinder_2D_Re200 | 201 | -288.8581635368488 | 27.54 s |
| Channel_3D_Retau180 | 100 | -27.89094513654709 | 40.33 s |
| Nozzle2D_coarse | 667 | -74.30931243874754 | 618.49 s |
| NACA0012Gust_3D_Re100_AOA20 (240 GiB host-RAM retry) | 201 | -58.71962346949299 | 1,578.35 s |

These raw returns have different physical meanings/scales and must not be
averaged into a benchmark score. Nozzle also reported terminal thrust angle
-0.01610023566285 and target angle 0.12244033185694. The clean-image two-case
repeat produced exactly the same Cylinder/Channel returns on this machine;
that is not cross-hardware or statistical repeatability certification.

Initial individual runs used the pinned native runtime plus the current task
source mounted read-only. The clean two-case repeat used built image ID
`sha256:91947d2cd4a0b6ff59d2dac691c26c568ef58179e5fef49ca546a78e67009248`.
Later evaluator hardening is additionally covered by the tests above.

The final implementation also passed a clean two-case repeat in
`final-cross-backend/run/results.json`: Cylinder -288.8581635368488 (201 controls,
26.38 s), Channel -27.890992492437363 (100 controls, 34.49 s), same snapshot.
The small Channel difference from the earlier run is retained, not rounded into
a bitwise-repeatability claim. This run includes the final GPU allocation and
process-cleanup code. Its evaluator implementation SHA-256 is
`217f68836c4f74862c51ee860ed858333b09c1b5000b72b2d46aaacb73d7a332`.

Final built image `ai4sci-hydrogym-native-evaluator:20260908`:
`sha256:e9883069cb24f7e6953bb209e19b7f42f1c5a50231042498e22237e770ecd018`.
All 39 source/profile preflights passed inside it and its implementation hash
matches the local source. The final native repeat used its immediately preceding
image layer (`sha256:cc205f8b04321a67f590a3905e8420fb2b5ff212f8948848b350596fec207aa1`),
whose complete filesystem layer list was checked equal to the final image;
only the default CMD metadata differs, and the test supplied an explicit command.

## Random initialization: actual numeric outputs

`starter/init_random.py --seed 0` now creates a real, randomly initialized shared
SB3 actor/critic: **289,585 parameters**, hidden widths [256, 256], capacities
128 observations / 24 actions and all 39 task names. It uses no pretrained
checkpoint, training data or optimizer updates. The seed was fixed before
evaluation, not selected by performance. Inference remains frozen, deterministic
and observation-conditioned. It is an untrained fixture, not a trained baseline.

The package contains `infer.sh`, `README.md` and `policy.pyz`; the archive carries
`checkpoint.pt`, config and inference code. It was created in a separate offline
process and then given to the clean final verifier image, mounting only the
package and source data read-only plus a fresh output directory. No training or
development source directory was mounted in the verifier.

Package SHA-256:
`883fed5d1fa1c35250481fd587f50df6c0e548843454921860c39883cb569085`.
The first two cases were run automatically by one command using the same snapshot.
Coarse and fine Nozzle subsequently completed with the same unchanged package:

| Case | Complete controls | Numeric `mean_return` | End-to-end case time |
| --- | ---: | ---: | ---: |
| Cylinder_2D_Re200 | 201 | -288.4614663414012 | 27.74 s |
| Channel_3D_Retau180 | 100 | -26.415240168571472 | 36.82 s |
| Nozzle2D_coarse | 667 | -71.87954324694597 | 623.79 s |
| Nozzle2D_fine | 667 | -72.0097395071455 | 1957.73 s |

The evaluator, not the model, computes these numbers from real native rewards.
They are already JSON numbers at `results.json.cases[CASE].mean_return`, with
the native episode length/return and unchanged package identity recorded beside
them. The new unit test additionally verifies reproducible random weights at
the same seed, different weights at a different seed, all 39 declared interface
shapes, a new environment ID, observation-dependent actions, relocation and
unchanged checkpoint tensors after inference.

Local random-test records:
`/mnt/workspace/hydrogym/reproduction/random-submission-20260908-tHTyNJfW/`.
The portable package is under `submission/`; the completed two-case report is
`maia-jax/run/results.json`; the additional reports are
`nozzle-coarse/run/results.json` and `nozzle-fine/run/results.json`.
Kolmogorov is still running at this update; only a terminal report establishes its outcome.
This proves a random contract-compatible model can obtain per-case numeric
returns. It does not establish an automatically available 39-case normalized
total, stability for every possible policy, or full-benchmark acceptance.

## Failures retained, not scored or silently omitted

- **Nozzle2D_fine:** the legal actions from this candidate made native outputs
  nonfinite. A direct native comparison, bypassing the replay adapter, reproduced
  the failure at control 9. Five zero-action controls were finite; this does not
  establish a full zero-action episode or stability for all admissible actions.
  No source restart, action bounds, grid or timestep was changed to hide it.
- **Kolmogorov_2D:** this candidate produced nonfinite native outputs after the
  last progress receipt at control 60. No finite full-episode return is reported.
  A separate **100-control zero-action diagnostic completed with finite outputs**;
  it is not a 1,000-control complete native episode or a learned-controller result.
- **NACA0012Gust_3D_Re100_AOA20:** the initial test's **60 GiB host-RAM** container
  limit killed the MAIA solver during initialization (`OOMKilled: true`). This
  was not GPU VRAM exhaustion or missing data. The retry uses 240 GiB host RAM
  with the same data, profile and checkpoint **completed all 201 controls** with
  finite outputs, return -58.71962346949299 and end-to-end time 1,578.35 seconds.
  This closes that run's memory-allocation issue, not full-suite certification.

The evaluator correctly returned explicit infrastructure errors and no invented
score for these failures. Diagnosis and full native acceptance of those routes
remain separate from the delivered single-case interface. In particular, the
direct Nozzle reproduction is evidence that its failure is not caused by the
JSON-lines transport; it is not proof of a particular numerical root cause.

## Reproduction records and remaining acceptance

Organizer records on this machine:
`/mnt/workspace/hydrogym/reproduction/native-evaluator-20260908-bq5rcm53/`.
Each case directory contains original native logs and replay outputs; the
`clean-image-two-case` directory contains the two-case single-snapshot report.
`maia3d-240g` holds the bounded retry. Diagnostic scripts are kept beside these
records. Read terminal results, not the mere existence of a container, as proof
of completion. These absolute paths are evidence locations, not dependencies
of the task, submission or evaluator.

Before declaring full benchmark acceptance: resolve native stability issues,
complete all 39 episodes with one package (including two-GPU cases), validate
physical behavior and repeatability, then freeze the certified suite. Measured
same-protocol reference anchors are additionally required for normalized Harbor
grading. No `suite.json`, reference scores or benchmark certification were
fabricated by this implementation.
