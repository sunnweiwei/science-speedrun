# Integration record

Integrators' checks on PR #68 (head `dfd5dcf`) and the v3 evaluation built during integration
(2026-09-30 to 2026-10-01). The host had 8× H100 80 GB. Every run used the task's own images,
with the `task.toml` limits for training and verification (1 GPU, 10 CPUs, 22 GiB).

## Data

- **Release.** The pinned development release `f7da2d0` was downloaded with a credentialed token.
  All 13 inventoried files match `runtime-manifest.json`, whose SHA-256
  (`10ce6beb…d3be`) matches `docs/release.json`.
- **Counts.** 68,827 train and 14,687 validation patients, and 262 diseases, as documented. Train
  and validation share no sample IDs.
- **Cutoff.** Of 42.34 M train and 9.04 M validation events, none has an event or availability time
  after the cutoff.
- **Era.** In the train and validation pool, 45,346 index admissions are ICD-9-era, 38,149 are
  ICD-10-era, and 19 are mixed or have no diagnoses. The outcome gap between eras and the shortcut
  it creates are in [EVALUATION_DESIGN.md](EVALUATION_DESIGN.md).
- **Split build.** `build_heldout_split.py` was run twice into fresh directories, and the
  inventories were byte-identical: development `758bc59e…b4bb`, private `ae727505…8c50`.
- **Derived boundaries.** On the release's own validation rows, the derivation reproduces the
  cutoff, death date, death days and both observation ends for all 14,687 samples. It differs only
  in the diagnostic `any_future_disease_record_365` flag (874 rows), because the release counts
  every disease there.

## Images and tests

- All four images build: the dev image and the verifier's `main`, `grader` and `unit-tests`
  targets. Both images now set `OMP_NUM_THREADS=10`.
- **Why.** Without the setting, PyTorch used 104 threads under a 10-CPU quota on this 208-core
  host. The reference GRU then ran at about 5 min per epoch with the GPU at 2 %. With it, an epoch
  takes about 20 s.
- **Unit tests: 42 passed.** Five are new and cover the reward:
  - tie handling, so that a constant prediction scores chance;
  - the alert-budget arithmetic;
  - family exclusion;
  - the 20-case minimum;
  - that the endpoint set is independent of predictions.

## v2 reproduction (before the redesign)

- **Reference GRU.** Retrained with the documented commands (seed 20260929, 12 epochs, 233 s):
  validation macro time-averaged AUROC **0.8311172**, documented 0.83112.
- **Two-phase verifier.** Run on a validation-as-test tree (the v2 test is not distributed): reward
  0.8311172096, identical to the development score to 8e-10, in 123 s.
- **Red-team packages** all scored 0:
  - empty: `invalid_submission`;
  - `infer.sh` exiting 3: `inference_failed`;
  - NaN predictions: `invalid_prediction`;
  - a probe package: it ran as uid 10001, `/trusted` was absent, `/sealed` was permission-denied,
    `/workspace/data` was absent, and the network was unreachable.

## v3 (the evaluated task)

Agent-side smoke test through `environment/docker-compose.yaml` with the v3 development tree:
- the container sees only `public/validation`, `trusted/validation`, `train` and the normalization artifact;
- `publish.py --check` packaged and installed the GRU, and re-publishing over it also works;
- `score.py` on validation gives 0.2112, as below.

`publish.py` runs `infer.sh` without `MKL_THREADING_LAYER`: once numpy is imported through mkl-service, that variable makes a PyTorch child abort.


Each baseline was trained by its documented recipe on the v3 train split and scored by
`environment/evaluation/score.py`:

| model | validation reward | test reward | test v2-metric AUROC |
|---|---:|---:|---:|
| statistical strata | 0.1464 | 0.1848 | 0.676 |
| temporal GRU | 0.2112 | **0.2363** | 0.767 |
| LightGBM (maintainer reference) | 0.2351 | 0.2682 | 0.746 |

- **GRU through the full verifier.** `package.sh` built the package, then the task compose ran
  `up grader` and `run main bash /tests/test.sh` against the real private test tree:
  - reward **0.2362567160**, identical to `score.py`;
  - `report.json` status `scored`, `reward_metric` `panel_recall_at_5pct`;
  - 105 s.
- **Red-team packages:**

  | package | reward | status |
  |---|---:|---|
  | constant prediction | 0.0501 | `scored` (chance, by the tie rule) |
  | empty | 0 | `invalid_submission` |
  | NaN | 0 | `invalid_prediction` |

## Accepted, not reproduced

The author's v2 hidden-test numbers (GRU 0.8316, LightGBM 0.8243) and the Codex agent campaign
(validation 0.8315) were accepted from the author's report. They used the v2 splits and reward and
are superseded as task results by v3.

## Integration changes

- **Layout.** The reference GRU became `environment/baseline/temporal-gru/`, with `package.sh`,
  `publish.py` and RESULTS.md. Delphi moved to `environment/references/delphi/`, and tests and
  paths were updated.
- **Instruction and README.** The instruction was rewritten into the five standard sections. The
  README was rewritten into the eight standard sections.
- **New files.** `environment/data/MATERIALIZE.md`, this record, and
  [EVALUATION_DESIGN.md](EVALUATION_DESIGN.md).
- **`task.toml`.** Authors added. `gpu_types = ["RTX 3070"]` removed, because it pinned both
  containers to one consumer GPU. The report schema is now v3.
- **Compose.** `network_mode` was removed from the development compose file; `task.toml` owns it.
- **Evaluation v3.** `build_heldout_split.py`, `panel_scoring.py`, and the reward switch in
  `score.py`, `grader.py` and the `risk_scoring` report (`reward_metric`, `panel`). The contract,
  the data README and a note in the protocol were updated to match.
