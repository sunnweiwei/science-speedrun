# Validation artifacts

Every number in the task README, `task.toml` and `docs/baseline_score.json` is in one of these files, each with a `protocol` field.

| file | what |
|---|---|
| `materialize-receipt-2026-09-17.json` | per-file digest check of `gifteval` (175), `gifteval_pretrain` (344), `leaderboard` (267) |
| `golden_gluonts_full.json` | `evaluation/score.py` vs gluonts 0.15.1 + gift-eval on the same forecasts, all 97 configs (max rel diff 1.9e-7); `golden_gluonts_check.py` reproduces it in the pinned reference environment |
| `seasonal_naive_replay.json` | `environment/baseline/seasonal_naive` through the sealed contract vs the board's Seasonal Naive CSV, per config |
| `task_trained_baseline_v1.json` | the superseded first starter run (reward 0.026; positional table only partly trained) |
| `task_trained_baseline_v2.json` | the superseded second run, trained on all 87 directories before the six leaking ones were excluded (reward 0.138) |
| `task_starter_scaled_41m.json` | the same recipe at 41.8M parameters and 2M steps (28 GPU-hours), best-validation checkpoint replayed on all 97 configs; a calibration of the headroom claim, not the baseline |
| `task_trained_baseline.json` | the shipped from-scratch starter (v3, 81 directories): training curve, package hashes, verifier result on all 97 configs, per-config metrics |
| `timesfm3_zero_shot.json`, `timesfm25_zero_shot.json`, `chronos2_zero_shot.json`, `chronos_bolt_zero_shot.json` | external-target replays under the contract vs their published per-config CSVs |
| `leaderboard_snapshot.json` | every published system's aggregate and task reward, recomputed from the pinned Space (`make_leaderboard_snapshot.py`) |
| `isolation_probe.json` | which verifier-side paths a canary package could reach, with the targets baked into the package: Bubblewrap as a user and as root, uid mode as root (data under a root-only directory), and a no-isolation control that must reach the data and labels (`tests/check_local_isolation.py`) |
| `trivial_control.json` | the context-distribution control: reward 0.000, unclipped skill −0.310 |
| `checkpoint_selection_recheck.json` | the v2 run's packaged checkpoint re-selected on the corrected validation split (same choice); the shipped v3 run used the corrected split from the start |
| `republication_readiness.json` | what a corpus release would need (held-out-free, allow-listed, licence position); nothing is republished |
| `valid_split_leakage.json` | overlap between the agent-visible validation windows and any held-out test region, alias copies of one M4 series included, and the M4 series holding another's test label that were staged (both must be zero; `check_valid_split.py`) |
| `corpus_leakage_verified.json` | the six corpus directories excluded for containing GIFT-Eval series, each re-derived by correlation (`check_corpus_leakage.py`) |
| `corpus_overlap.json` | the broader survey behind it: content overlap between GIFT-Eval and six public corpora, including the ones this task forbids |
| `cohort_leakage.json` | share of held-out label points readable from another instance's context, with and without the cohort split, per series and per timeline; M4 content conflicts sharing a call; and the calendar-time overlap across unrelated timelines (`check_cohort_leakage.py --data --manifest`) |
| `compare_with_leaderboard.py` | builds the replay comparison JSONs from a grader report and a leaderboard results directory |

Protocols: "task test split, sealed contract" = `tests/main_verifier.py --manifest environment/data/split.json`, 97 configs, 326,490 instances, 16,384-observation context cap. "board protocol" = values from the leaderboard's own CSVs (full history, models run by their authors).
