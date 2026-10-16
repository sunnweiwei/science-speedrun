# temporal-gru: results

The commands in [README.md](README.md) on the v3 splits: train 68,175, validation 3,915, hidden
test 11,424. The run used seed 20260929, early stopping on the internal fold, and the
calibrated checkpoint, on one H100 with 10 CPUs (`OMP_NUM_THREADS=10`). Caching took about 4 min
and training plus calibration about 4 min.

| split | `panel_recall_at_5pct` (reward) | recall@1 % | PPV@5 % | panel AUROC | panel AUPRC | calibration slope (median) | predicted/observed (median) | macro time-avg AUROC, 262 diseases |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation (14 endpoints) | 0.2112 | 0.0704 | 0.068 | 0.764 | 0.072 | 2.70 | 1.07 | 0.747 |
| **test** (17 endpoints) | **0.2363** | 0.0705 | 0.068 | 0.782 | 0.064 | 2.92 | 1.15 | 0.767 |

The test row was scored through the task's own two-phase verifier, after packaging with
`package.sh`. The verifier reward, 0.2362567160, is identical to `score.py` on the same predictions.

## Anchors

| system | validation reward | test reward |
|---|---:|---:|
| constant prediction (chance) | 0.05 | 0.0501 |
| statistical: empirical-Bayes sex/age/burden/history strata | 0.146 | 0.185 |
| **temporal-gru (this baseline)** | 0.211 | **0.236** |
| LightGBM on 567 hand-built features (reference; not an eligible submission) | 0.235 | **0.268** |

## Reading the numbers

- The baseline finds about a quarter of next-year cases of serious new disease when 5 % of patients
  are flagged. A tree on hand-built features finds a few more. Beating that tree with a learned
  representation is the open problem.
- The baseline ranks well but its risks are compressed: a calibration slope near 3 means the
  predicted spread is about a third of what it should be. That is not rewarded, but it is reported
  and worth fixing.
- Validation is small (3,915 patients, 14 endpoints with ≥ 20 cases). A single model's interval is about
  ±0.03. Treat validation differences below about 0.03 as noise,
  or score on an internal fold of the ICD-10-era training patients as well.
- These are single-seed numbers; seed SD on a closely related metric was about 0.004.
