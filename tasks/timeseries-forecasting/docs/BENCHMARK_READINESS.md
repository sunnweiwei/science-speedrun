# Benchmark readiness

Organizer-facing status of the `timeseries-forecasting` task; last updated
2026-09-22 after the full quality audit (Harbor overlays rewritten, verifier
image completed, uid fallback for isolation, stale numbers fixed).
Contributor machine: one shared A100-40GB, no Docker, Python 3.11 (the images
run Python 3.12 with the same pins).

| item | status | evidence |
|---|---|---|
| Sources pinned to commit SHAs, licenses recorded | done | `environment/data/sources.lock.json`, `docs/UPSTREAM_LICENSES.md` |
| Corpus checked for GIFT-Eval content, not just names | done: six leaking directories excluded, each pair re-derived by correlation | `docs/validation/corpus_leakage_verified.json`, `docs/validation/corpus_overlap.json` |
| Every file verified on real downloads | done: 786/786 in the 2026-09-17 receipt (gifteval 175, corpus 344 including the six directories excluded afterwards, leaderboard 267; LFS SHA-256 or git blob SHA-1); the current lock lists 768 files (175 + 326 + 267), all in that receipt | `docs/validation/materialize-receipt-2026-09-17.json` |
| Protocol frozen and cross-checked with the benchmark's code | done: 97 configs, windows / horizons / seasonalities from gift-eval `data.py` | `environment/data/split.json`, unit tests |
| Grader equals the leaderboard's metric implementation | done: max rel diff 1.9e-7 vs gluonts on 326,490 instances | `docs/validation/golden_gluonts_full.json` |
| Normalizer equals the board's Seasonal Naive | done: taken from the pinned CSV; contract replay reproduces MASE on 97/97, WQL on 63/97 (cap effect documented) | `docs/validation/seasonal_naive_replay.json` |
| Agent cannot see the test region | mount discipline: the development overlay mounts only `gifteval_pretrain/`, `valid/` and the receipt; `entrypoint.sh` refuses to start if `/data/gifteval` is present | `environment/docker-compose.yaml`, `environment/entrypoint.sh` |
| Inference cannot see labels or data at verification | done: Bubblewrap namespace with runtime, package, input and output only (the verifier overlay grants the user namespace Docker's default profile denies); where a container still cannot create one, the package runs as uid 10001 with the labels root-owned; `reward.json` names the mode; canary probe passed | `tests/isolation.py`, `tests/test_isolation.py`, `tests/check_local_isolation.py`, `docs/validation/isolation_probe.json`, `tests/docker-compose.yaml` |
| No instance's label is visible in another instance's context at verification | done: one `infer.sh` call per cutoff cohort; call c holds each timeline's c-th cutoff, a timeline being a dataset directory or directories recording one system (`bizitobs_application` with `bizitobs_service`), 82 calls on the test split; `work/input` is root-only and a call sees one cohort's copy. Measured on the full split: a single call would expose 95.6% of label points to another instance on the same timeline, cohort-wise calls 0% on the directories with real timestamps; unrelated timelines in one call share calendar time (34.7% of those label points fall inside some unrelated series' context period, recorded per directory; a single global timeline would take 594 calls). M4 has placeholder timestamps: the 722 pairs of M4 series where one holds the other's test label verbatim (533 labels) or both are one series under two directories never share a call | `docs/validation/cohort_leakage.json` (`docs/validation/check_cohort_leakage.py`), `evaluation/make_cases.py`, `tests/main_verifier.py`, unit test |
| Rules: from scratch, no pretrained weights, synthetic data allowed and declared, parameters fixed at inference | done | `instruction.md`, `environment/contract.md` |
| Task-owned from-scratch baseline trained and replayed through the sealed verifier | done: reward 0.153 on all 97 (82 cohort calls, Bubblewrap), trained on the 81 non-leaking directories; recipe recorded in the starter README | `docs/validation/task_trained_baseline.json`, `docs/baseline_score.json` |
| Scaled starter as a calibration of the headroom claim | done: the same recipe at 41.8M parameters and 2M steps (28 GPU-hours) replays at 0.207 (starter 0.153, TimesFM-3 0.447); its validation skill peaked at 28% of the run, so the gap is the recipe, not compute | `docs/validation/task_starter_scaled_41m.json` |
| External targets replayed under the contract | done: TimesFM-3 (0.447), TimesFM-2.5 (0.412, reproduces the board to 4.3e-6 per config), Chronos-2 (0.418, to 0.04%), Chronos-Bolt-base (0.325) | `docs/validation/timesfm3_zero_shot.json` and the other `*_zero_shot.json` |
| Leaderboard table reproducible | done: generated from the pinned snapshot by a committed script | `docs/validation/make_leaderboard_snapshot.py`, `docs/validation/leaderboard_snapshot.json` |
| Rules enforced by the verifier | declared pretrained assets → reward 0 (unit-tested); determinism recorded; undeclared pretrained weights are not detectable and need review | `tests/main_verifier.py`, `tests/test_verifier_failures.py` |
| Verifier: sealed copy, timeout, determinism check, failure isolation | done, exercised locally | `tests/main_verifier.py`, `tests/test_verifier_failures.py` |
| Trivial-predictor control scored | done: context mean and deciles, reward 0.000 (unclipped skill −0.310), so the metric is not measuring compliance | `docs/validation/trivial_control.json` |
| Agent-visible validation cannot contain held-out labels | done: 151,170 instances staged before the earliest test region of every config that shares the series (alias copies of one M4 series share it); the 632 M4 series whose values hold another series' test label verbatim get no window; the checker reports 0 overlaps in time (11,745 before this rule) and 0 such series staged | `docs/validation/valid_split_leakage.json` (`docs/validation/check_valid_split.py`) |
| Checkpoint selection on the corrected validation split | the superseded v2 run was re-selected on the corrected split with the same choice (`docs/validation/checkpoint_selection_recheck.json`); the shipped v3 run was trained with the corrected split from the start (its validation curve is in `docs/validation/task_trained_baseline.json`, best at step 25,000) | `docs/validation/checkpoint_selection_recheck.json`, `docs/validation/task_trained_baseline.json` |
| Package cannot reach outside itself | done: a symlink escaping the package is a rule violation (reward 0); packages carry their own reader | `tests/main_verifier.py`, unit tests |
| Unit tests | 75 passing and 3 skipped as a regular user (the three uid-mode tests need root); 78 passing as root, which is the integrator's uid mode; no data or GPU needed (the end-to-end verifier tests build a synthetic GIFT-Eval directory); pytest only | `bash tests/run_unit_tests.sh` |
| Normalised 0-1 benchmark score emitted by the verifier | done: `reward.json` carries `normalised_score` beside `reward`, mapped between Seasonal Naive (0.000) and the best of 130 leaderboard systems (1.000); the bounds file is baked into the verifier image beside the grader | `environment/evaluation/score_bounds.json`, `tests/test_score_bounds.py`, `tests/Dockerfile` |
| Republication of the corpus assessed | done: no file describes the held-out windows, the 81-directory allow-list excludes the six leaking directories, and the licence position is recorded (release would be CC BY-NC 4.0; four Kaggle-sourced directories, 1.9%, unresolved); nothing republished | `docs/validation/republication_readiness.json` |
| Science introduction figure | done | `docs/science_intro.png`, `docs/make_science_intro.py` |
| Harbor images and overlays | written to Harbor's conventions: `main` service, task-directory build context, `security_opt` for the namespace, no network, sub-directory mounts (the verifier's under root-only `/private`, so the uid fallback cannot regenerate labels), healthchecks; verifier image bakes `isolation.py`, both manifests, the grader, the normalizer and `score_bounds.json`, installs bubblewrap and the runner uid; base image cu126 with the PEP 668 override, after the integrator's review of pde-surrogate's twin images (2026-09-23: cu130 needs driver ≥ 580, the system Python refuses pip, Docker refuses bubblewrap's network namespace, `/data` had to be root-only) | `environment/{Dockerfile,docker-compose.yaml}`, `tests/{Dockerfile,docker-compose.yaml}`, `task.toml` |
| Inference budget | starter: 10 min across the 82 cohort calls on a shared A100 (7 s mean, 45 s max per call); limit 4 h | `docs/validation/task_trained_baseline.json` |

## Verifier hardening (2026-09-24, adversarial pre-review after the integration review of `aeef792`)

A read-only multi-agent review of this head, each finding checked by independent
skeptics, found what the integration review had not reached:

- The verifier read the package's output with its own privileges and followed
  HDF5 virtual datasets and external links: a package that maps each row of
  `mean` and `quantiles` onto `work/truth/cases/<config>.h5` by `source_index`
  scored 0.999 without reading the labels. Package output is now read only from
  regular files whose datasets are plain (`ts_io.open_package_h5`, also in the
  determinism compare); `contract.md` states the rule.
- A FIFO in place of a prediction hung the verifier; one invalid UTF-8 byte on
  a package's stdout, a manifest that is not a JSON object or a deep tree in
  `HOME` made it exit 2 with no reward. All are now a failed config or reward 0.
- In uid mode a call could leave a detached process, files in `/tmp` or the
  runner's home, SysV IPC objects, runner-owned outputs of earlier calls next to
  later ones, and a group-writable sealed package shared by all 82 calls.
  `isolation.after_uid_call` ends every call: every runner process killed,
  output and home back to root (0700), runner-writable directories and IPC
  emptied; the seal drops write and set-id bits; logs are root-only.
- Cohorts: `bizitobs_service` (the services of the application
  `bizitobs_application` measures, one 10-second step later) now shares its
  timeline; 722 pairs of M4 series where one holds the other's test label
  verbatim, or both are one series under two directories, never share a call,
  and the 632 M4 series holding another's test label get no validation window.
  Copies of one M4 series share the earliest validation boundary.

`tests/test_verifier_endtoend.py` runs the whole verifier on a synthetic GIFT-Eval
directory with each attack; against the previous head the virtual-dataset
package scores 0.999, the FIFO hangs and the other cases crash or fail. The
starter replays under the new verifier and cohorts at 0.15301884551 (recorded
0.15301884549).

## Out of scope for this contribution

Agreed with the maintainers as not blocking review.

| item | state |
|---|---|
| Docker images (dev, verifier) built and run | written and pinned as above; no Docker on the contributor's machine, so unbuilt |
| Agent pilot | protocol in `AGENT_EVALUATION.md`; needs Docker |
| H100 timing | unmeasured; every number here is from one shared A100-40GB |

## Pending

Nothing.

## Known limitations

- The corpus/eval leakage boundary was checked by content, not by name, and the authors' non-leaking claim did not hold: six directories carried GIFT-Eval series up to an affine transform (`m4_hourly` is `pedestrian_counts` + 10, correlation 1.000000 over the whole series) and are excluded, leaving 81 directories and 41.14 GB. The check is fingerprint-based, so it cannot see a series that was resampled to another frequency, subsampled, differenced or mixed; that residual risk remains.
- Targets only: the 24 configs with past covariates are harder here than on the board, equally for every package (`docs/validation/timesfm3_zero_shot.json` quantifies it on a model that can use them).
- Context cap of 16,384 observations per instance (needed so instances are self-contained); documented in the contract, no listed deep model exceeds it.
- Leaderboard leak flags are self-declared by submitters.
- 48 GPU-hours is far below the targets' training budgets. The reward clips at Seasonal Naive (0); the starter sits at 0.153 after 16 minutes and TimesFM-3 replays here at 0.447 (its own board row is 0.449); the unclipped skill score is reported for progress below the floor.
