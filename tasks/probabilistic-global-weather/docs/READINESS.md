# WN2 delivery validation

## Layout check after TASK_STANDARD.md (2026-09-27, task reviewers)

The starter moved to `environment/baseline/starter/`, the task-root documents, `validation/` and
`compute-profiles.toml` to `docs/`, and the data README to `environment/data/MATERIALIZE.md` (a
short agent-facing README replaces it). Development data are mounted under `/workspace/data`
(`TASK_DATA_DIR`), the verifier reads `TASK_PRIVATE_DIR/evaluation-resources`, and
`/workspace/research` is an empty writable mount; the starter now packages next to
`/workspace/submission` and swaps the package in with one rename. Checked: both compose files pass
`docker compose config`; the rebuilt development image starts healthy through the task compose file
with the training and development trees mounted read-only; 3 starter and 17 data unit tests pass.

The reviewers' 2026-09-26 Docker evaluation of their reproduced 580-update package stopped in its
first window: `bwrap: Failed to make / slave: Permission denied` (Ubuntu 24.04 host,
`kernel.apparmor_restrict_unprivileged_userns=1`), so all cases would have scored 0. A probe with the
verifier image showed Bubblewrap needs `seccomp`, `apparmor` and `systempaths` unconfined plus
`SYS_ADMIN` there; `tests/docker-compose.yaml` now sets them and the same probe passes through the
compose file. The full evaluation has not been re-run.

The main starter uses the 580-update extended AR curriculum. Its independent
submission evaluation uses eight dates, 56 members and 15 days, including core,
spatial pooling, spectra and cyclone scoring. Normalized CRPS is **0.469183**
and reward is **0.680651**. The [final report](validation/results-summary/README.md)
compares this model with original Seed 0, GenCast and simple references.

The recipe retains the measured training prefix through AR2 and uses four
updates at each AR3–AR8 stage. Its composed training cost is 87.51 RTX PRO 6000
GPU-hours; the envelope allows four GPUs for 24 hours and 30 minutes for
packaging. A fresh uninterrupted 580-update training run has not been timed.

## Validation evidence

- Starter/preparation tests cover the delivered schedule and 24.5-hour task
  timeout. Evaluator/comparison tests cover the independent scoring protocol.
- Dockerfile-derived Apptainer runtime tests and isolated submission-entry
  inference cover all 30 leads. Extended inference and scoring use the same
  numerical model and evaluator code as the validated original submission.
- The dedicated resource bundle includes training-only statistics, isolated
  scoring truth and the frozen IBTrACS snapshot, now included in the public HF release.
- GPU inference and CPU output validation/scoring are separate phases.
- [Extended evaluation record](validation/extended-evaluation.json) and
  [original acceptance record](validation/delivery-acceptance-20260912.json)
  preserve the two experiments separately.

The 13 starter/preparation tests and 76 evaluator/comparison tests passed.
Current evaluator source was mounted into the existing validated image; the
updated staged starter schedule was checked in that runtime. Rebuilding the
image containing the new schedule remains unverified.

## Standalone delivery checks

A clean Git clone resolves all three internal symlinks and both Dockerfile COPY
contexts. Both Apptainer definitions render, both Compose configurations parse,
and the standalone guide's shell blocks pass syntax checks. The clean source
passes 13 starter/preparation tests and 76 evaluator/comparison tests in the
existing validated runtime. Source-preflight success, changed-source and
unreachable-source handling were checked separately. See the
[check record](validation/delivery-preflight.json) and [standalone guide](RUNNING.md).

The live source preflight and the actual materializer source gate pass with
the declared ARCO rolling-attribute policy. The original raw metadata pin is
retained for existing receipts; all 32 training source receipts remain valid.
Current validation includes 13 starter/preparation and 80 evaluator/comparison
tests (including four metadata-policy tests). Two historical timestamps, each
with both 100 m wind components, were re-read upstream and matched stored
values exactly. This sample does not validate all historical chunks or replace
an end-to-end materialization run. See the [ARCO validation record](validation/arco-metadata-policy.json).

## Prepared HF data release

The public dataset contains 47 shards and 35,273 extracted files (205.244 GiB).
The default operator materializer now downloads the pinned HF release and
verifies archive and per-file hashes; upstream rebuilding is opt-in.

Release checks include a full offline rehash against the original preparation
receipts, all 47 remote shard hashes, anonymous download/extraction samples
covering each of the seven component types, and original source validators for
the sampled training window, cyclone labels, normalizers, contexts and development
precipitation. The 7 archive/download tests, 10 existing preparation tests and
3 starter tests pass. This is not a fresh full-release download or new model
training/evaluation. See [release evidence](validation/hf-release.json).

## Evaluation within a configurable disk allowance

The default verifier now uses v3 lossless continuation with a 1 TB allowance and four concurrent cases.
Its planner counts prepared inputs, model copies, two checkpoint generations,
cyclone fields, precipitation boundary state and a runtime reserve. It chooses
consecutive lead windows using uncompressed float32 sizes. Every forecast step
is computed once. A measured allowance violation stops subprocesses and records
a failed evaluation; polling is not a filesystem quota.

All eight dates, 56 members, 30 leads and metric endpoints remain required. A
window is scored before its arrays are released. Native, pooled, REV, SEEPS and
spectral rows are merged in canonical order, and the full cyclone trajectory is
reconstructed from only three fields. Legacy retained/split modes opt out and
still require their larger storage provision.

Existing stock weights are upgraded with `environment/weathernext/wn2/upgrade_inference.py`; training is
not rerun. The tool retains the original package and records unchanged weights
hashes. The delivered starter includes the continuation helper, and the default
`tests/test.sh` passes the disk allowance and reserve to the complete verifier.

Validation covers checkpoint identity and lossless round trips, precipitation
boundaries with lazy arrays, window completeness, all-endpoint cleanup, all eight
cases through the default entry point, preservation of existing weights and a
real over-budget subprocess terminated by the monitor. The 98 focused regression tests pass. A real trained package's initial and
resumed windows match the original package bitwise for two members at hours
12 and 24 (44 field comparisons); its weights hash is unchanged. Native-grid
56-member checks at hours 12 and 24 match historical core, pooling and spectral
rows, including REV and SEEPS. Two complete 30-lead cyclone trajectories split
into windows of 1, 7 and 22 leads reproduce original input arrays bitwise and
original detector/track file hashes. The actual container `tests/test.sh` also
rejects an undersized allowance before inference. See the
[complete validation record](validation/windowed-evaluation.json). The full eight-date evaluation
with four concurrent cases was later run end to end in Docker (5.1 h on four H100s; see
[parallel-evaluation-20260930.json](validation/parallel-evaluation-20260930.json)). Lower allowances may need a longer evaluation budget and runner timeout.
See [the operator workflow](../wn2/WINDOWED_EVALUATION.md).

The earlier case-level cleanup evidence remains in
[bounded-evaluation-storage.json](validation/bounded-evaluation-storage.json);
the initial no-recompute estimate is superseded by the executable planner.

## Runtime limitation

Actual Docker/Compose execution, GPU passthrough and namespace behavior remain
unverified. Delivery uses the accepted best-effort evidence from Apptainer,
isolated inference and Compose configuration checks. These checks do not
establish Docker runtime equivalence.

The eight-date comparison is a limited paper-method subset. Recipe promotion
used its test results, so future independent confirmation needs new held-out
cases. Full-year 2023 reproduction and additional seeds are separate experiments.
