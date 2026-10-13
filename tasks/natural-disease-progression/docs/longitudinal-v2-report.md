# Longitudinal-v2 AUROC report

> Historical (v2). These numbers use the v2 splits, where both coding eras are mixed, and the v2
> reward. On those splits a coding-era indicator alone scores 0.653. For the evaluated task, see
> [EVALUATION_DESIGN.md](EVALUATION_DESIGN.md) and `environment/baseline/temporal-gru/RESULTS.md`.

## Decision

The frozen v2 cohort contains usable longitudinal and multimodal signal. On validation, the prespecified three-seed temporal GRU ensemble scores 0.83718 disease-first macro time-averaged AUROC, versus 0.82406 for same-information LightGBM and 0.82513 for the unordered neural control. Patient-paired bootstrap intervals exclude zero for the temporal comparisons. The conclusion is narrower than “deep learning wins”: rich non-linear features account for most of the improvement over the 0.70787 statistical model, while time/order adds a smaller reproducible increment in this experiment. A corrected diagnosis-only ablation scores 0.82155, supporting additional multimodal value.

These are development-stage findings. Earlier test results influenced the decision to replace normalized Brier with AUROC, so the existing test set is no longer an independent blind confirmation set. The one-time frozen-rule test comparison below is consistency evidence only; a new sealed split or external cohort is required for a confirmatory claim.

## Frozen task and data audit

V2 is unchanged: the prediction time is the second eligible live adult discharge, there is one sample per patient, splits are patient-level, and the 262-disease vocabulary is selected from train only. The target is the first later **inpatient record** of a disease not previously recorded, not biological disease onset. Lack of a later record means no record before the documented observation boundary, not proven absence. Inputs require both `event_time <= cutoff` and `available_time <= cutoff`. The primary cohort continues to require a complete 365-day diagnosis and death observation window; partial follow-up remains unknown and is not converted to a negative.

| Audit | Train | Validation | Test |
|---|---:|---:|---:|
| Patients / samples | 68,827 | 14,687 | 14,735 |
| Complete-year primary patients | — | 12,842 | 12,825 |
| Median / p90 events | 410 / 1,244 | 408 / 1,264.4 | 413 / 1,239 |
| At least two stay keys | 68,787 (99.942%) | 14,676 (99.925%) | 14,731 (99.973%) |
| Numeric lab rows | 28,243,748 | 6,045,046 | 6,150,164 |
| History days p10 / p50 / p90 / p99 | 20 / 374 / 2,255.4 / 4,324 | 19 / 369 / 2,225.4 / 4,382 | 20 / 376 / 2,293.2 / 4,343.7 |
| First-to-second eligible discharge days p10 / p50 / p90 / p99 | 8 / 158 / 1,725 / 3,856 | 8 / 155 / 1,716.4 / 3,831.1 | 8 / 165 / 1,780 / 3,904.6 |

The old v1 first-discharge cohort had only 101/152,601 train, 13/32,699 validation, and 22/32,725 test samples with two stay keys. That was a task-design limitation, not evidence that longitudinal models cannot help. V2 regeneration preserved all sample and disease keys relative to the preceding v2 release.

The real-data validator passed 98,249 stable anonymous patient keys with no cross-split overlap; it no longer compares split-prefixed `sample_id` strings. It also checked no event or availability time after cutoff, no label at/before cutoff, monotone labels, unique `(sample_id,disease_code)` label keys, complete key coverage, 3,406 normalization cells, and no public raw patient/encounter/source identifiers. `working.duckdb` is a build intermediate and is not staged.

## Metric and implementation corrections

The sole reward is now AUROC. For each frozen disease/horizon cell, AUROC is computed only on official valid, at-risk, complete-year samples. A cell with one observed class is unavailable—not 0 or 0.5. Available horizons are averaged equally within disease, then available diseases are averaged equally. This is not micro AUROC and the valid set cannot depend on predictions. Validation has 262 valid diseases and 3,396/3,406 valid cells (156,713 positive and 41,887,982 negative cell observations); test has 262 and 3,401/3,406 (158,352 positive and 41,887,951 negative).

The report retains 365-day/per-horizon macro AUROC, disease-first macro AUPRC, raw and frozen age/sex-normalized Brier, calibration, support, and death metrics. Harbor writes `{"reward": <macro_time_averaged_auroc>}`; invalid submissions receive zero.

The starter fixes the following experiment-changing defects:

- diagnosis-only collation snapshots the original modality mask before changing tensors, filters code/modality/value/missing/time together, and recomputes gaps after filtering;
- numeric parse failures are represented by train-fit code/unit mean imputation in standardized space plus an explicit missingness channel, never silently changed into a real zero before normalization;
- event ordering and gaps both use `event_time` (then `available_time` as a stable tie-break), rather than sorting by availability and assuming event time is monotone;
- label loading aligns by unique keys and rejects duplicate, unknown, missing, or shuffled coverage instead of relying on row order and reshape;
- early stopping uses the full train-internal early-stop fold's new primary metric, with a disjoint train-only calibration fold; test never selects epoch, seed, hyperparameters, or ensemble weights.

LightGBM receives 567 train-fit features from the same cutoff-bounded inputs: demographics, diagnosis counts/A–Z chapters/hashes, visits, history span, event/modality recency, medication hashes, and recent/mean/std/min/max/count/missing summaries for 48 train-selected labs. Thus the neural models do not receive a privileged modality.

## Validation experiments

Hardware was one NVIDIA RTX 3070 8 GB; seeds were fixed in advance as 20260929, 20260930, and 20260931. Neural caches contain all 42.34M train events and deterministic history-spanning sequences. Unordered models have 1,560,395 parameters and ran 406.6/423.6/432.3 seconds for 15 epochs. GRUs have 1,667,915 parameters and ran 412.3/396.5/407.9 seconds, stopping after 12 epochs. LightGBM fit 262 fixed-recipe 48-tree models in 490.4 seconds. The corrected diagnosis-only run took 504.2 seconds and stopped after 12 epochs. These runs fit well within the 24-hour agent budget and are not the earlier two-epoch/one-hour limit.

| Model | Primary AUROC | Macro AUPRC | 365d AUROC | Raw Brier | Normalized Brier |
|---|---:|---:|---:|---:|---:|
| All zero | 0.50000 | 0.00393 | 0.50000 | 0.003924 | 0.86802 |
| Train prevalence | 0.50000 | 0.00393 | 0.50000 | 0.003855 | 0.86411 |
| Age/sex | 0.62977 | 0.00603 | 0.63258 | 0.003844 | 0.86315 |
| Stratified statistical | 0.70787 | 0.01089 | 0.71236 | 0.003824 | 0.86162 |
| LightGBM, same information | 0.82406 | **0.03554** | 0.81879 | **0.003748** | **0.85484** |
| Diagnosis-only GRU, corrected seed 20260929 | 0.82155 | 0.03008 | 0.81257 | 0.003837 | 0.86136 |
| Unordered NN seed 20260929 | 0.82321 | 0.02227 | 0.81832 | — | — |
| Unordered NN seed 20260930 | 0.82457 | 0.02462 | 0.81910 | — | — |
| Unordered NN seed 20260931 | 0.81955 | 0.02333 | 0.81523 | — | — |
| Unordered NN, 3-seed calibrated ensemble | 0.82513 | 0.02478 | 0.82005 | 0.003841 | 0.86259 |
| Temporal GRU seed 20260929 | 0.83112 | 0.02960 | 0.82385 | — | — |
| Temporal GRU seed 20260930 | 0.82413 | 0.02975 | 0.81930 | — | — |
| Temporal GRU seed 20260931 | 0.82998 | 0.02793 | 0.82352 | — | — |
| Temporal GRU, 3-seed calibrated ensemble | **0.83718** | 0.03149 | **0.83094** | 0.003835 | 0.86113 |

Unordered seed mean ± sample SD is 0.82244 ± 0.00260; GRU is 0.82841 ± 0.00375. The paired seed differences are +0.00791, -0.00044, and +0.01043: seed variation is real, and no claim rests on picking the best seed. The fixed ensemble is the validation-selected default.

Calibration uses only the reserved train fold and one predeclared global convex coefficient. Every neural disease model selected alpha 0.02, shrinking toward train-fit cell prevalence while preserving ranking; the temporal ensemble Brier becomes 0.003835. This demonstrates that strong neural ranking had initially been hidden by miscalibration. LightGBM remains better calibrated and has higher AUPRC, while GRU has higher primary AUROC.

Actual validation history-span quartiles are 69/369/1,107 days. Temporal AUROC by quartile is 0.8513/0.8450/0.8292/0.8179 versus unordered 0.8450/0.8347/0.8193/0.7961 and LightGBM 0.8420/0.8280/0.8211/0.7975. The temporal increment occurs in every quartile, while absolute performance falls with longer/likely more complex histories. For train 365-day disease-frequency quartiles 79.25/150.5/305 positives, temporal scores 0.8532/0.8395/0.8382/0.8179; the conclusion is not created by selecting only common diseases or one horizon. Full per-horizon/support output is produced by `analyze_subgroups.py`.

## Patient-paired uncertainty

The exact patient bootstrap uses 1,000 prespecified draws and the same complete-year patient multiplicities for both models. AUROC is recomputed in every draw. If a resampled disease/horizon cell becomes single-class it is unavailable for both models, followed by the frozen time-within-disease and disease averaging. This interval measures patient sampling only; neural seed SD above measures optimization variation.

| Candidate minus reference | Difference | Patient-bootstrap 95% CI | P(candidate > reference) |
|---|---:|---:|---:|
| Unordered ensemble − statistical | +0.11727 | [0.10799, 0.12572] | 1.000 |
| Unordered ensemble − LightGBM | +0.00107 | [-0.00471, 0.00728] | 0.693 |
| Temporal ensemble − unordered ensemble | +0.01205 | [0.00770, 0.01644] | 1.000 |
| Temporal ensemble − LightGBM | +0.01312 | [0.00772, 0.01923] | 1.000 |

## Development-stage test consistency check

After freezing the three-seed ensembles and calibration rule, the sealed evaluator was run once. Because test had already influenced the metric decision, these are not independent confirmatory estimates.

| Model | Primary AUROC | Macro AUPRC | 365d AUROC | Raw Brier | Normalized Brier |
|---|---:|---:|---:|---:|---:|
| Age/sex | 0.62914 | 0.00607 | 0.63383 | 0.003892 | 0.88838 |
| Statistical | 0.70586 | 0.01143 | 0.71138 | 0.003871 | 0.88653 |
| LightGBM | 0.82428 | **0.03587** | 0.82098 | **0.003797** | **0.87992** |
| Unordered calibrated ensemble | 0.82335 | 0.02744 | 0.82130 | 0.003890 | 0.88714 |
| Temporal calibrated ensemble | **0.84000** | 0.03240 | **0.83606** | 0.003884 | 0.88564 |

## Verifier and isolation acceptance

A real compose main+grader run—not pytest—executed submissions as uid 10001. The inference log recorded `submission_uid=10001` and `test_truth_visible=false`; `/trusted` was absent and mount inspection found no test truth in main. Beyond the prevalence smoke (`reward=0.5`), the canonical runtime verifier loaded and scored the actual 567-feature LightGBM package at `0.8242798909464752` and the calibrated temporal-GRU seed-20260929 package at `0.8316238634686397`. The GRU rebuilt its 9,198,927-event test cache inside verifier main; the tree package independently rebuilt its 567 features. A run with an empty submission produced `{"reward":0.0}` and `status=invalid_submission`. Development staging contains train, public validation, validation truth/scorer, and train-only normalization only. Private test inputs and truth are mounted into separate verifier services; neither the materialized root, raw MIMIC, `working.duckdb`, nor controller patient map is mounted into the agent or verifier main. Verifier main now installs the same pinned PyTorch, LightGBM 4.6.0, joblib 1.5.3, NumPy, scikit-learn, DuckDB, and PyArrow inference dependencies required by the official starters.

## Reproduction

```bash
python environment/data/materialize_longitudinal_v2.py \
  --source-materialized build-intermediate --output materialized
python environment/data/validate_longitudinal_v2.py --materialized materialized
python environment/data/stage_runtime_data.py \
  --materialized materialized --destination runtime

python environment/baseline/temporal-gru/longitudinal.py cache \
  --input runtime/development/train \
  --labels runtime/development/train/label_windows.parquet \
  --cache research/train.pt
python environment/baseline/temporal-gru/longitudinal.py cache \
  --input runtime/development/public/validation \
  --adapter research/train.adapter.json --cache research/validation.pt
python environment/baseline/temporal-gru/longitudinal.py train \
  --materialized runtime/development --cache research/train.pt \
  --kind temporal --epochs 15 --batch-size 64 --seed 20260929 \
  --device cuda --output research/temporal-20260929.pt
python environment/baseline/temporal-gru/longitudinal.py calibrate \
  --cache research/train.pt --checkpoint research/temporal-20260929.pt \
  --device cuda --output research/temporal-20260929-calibrated.pt
python environment/baseline/temporal-gru/longitudinal.py predict \
  --cache research/validation.pt --checkpoint research/temporal-20260929-calibrated.pt \
  --device cuda --output research/temporal-validation/predictions.parquet
python environment/evaluation/score.py \
  --input runtime/development/public/validation \
  --trusted runtime/development/trusted/validation \
  --normalization runtime/development/trusted/risk_normalization.parquet \
  --predictions research/temporal-validation/predictions.parquet \
  --output research/temporal-validation
```

Repeat the fixed train command for all three seeds and `--kind unordered`; use `--kind diagnosis` for the modality ablation. `tabular_baseline.py cache/fit/predict`, `ensemble_predictions.py`, `analyze_subgroups.py`, and `paired_bootstrap.py` reproduce the remaining tables and keyed comparisons.

```bash
python maintenance/baselines/tabular_baseline.py cache \
  --input runtime/development/train --output research/tabular-train.npz
python maintenance/baselines/tabular_baseline.py cache \
  --input runtime/development/public/validation \
  --adapter research/tabular-train.adapter.json --output research/tabular-validation.npz
python maintenance/baselines/tabular_baseline.py fit \
  --materialized runtime/development --features research/tabular-train.npz \
  --seed 20260929 --output research/lightgbm.joblib
python maintenance/baselines/tabular_baseline.py predict \
  --features research/tabular-validation.npz --checkpoint research/lightgbm.joblib \
  --output research/lightgbm-validation/predictions.parquet

python environment/evaluation/ensemble_predictions.py \
  --input research/temporal-20260929-validation/predictions.parquet \
  --input research/temporal-20260930-validation/predictions.parquet \
  --input research/temporal-20260931-validation/predictions.parquet \
  --output research/temporal-ensemble-validation/predictions.parquet
python environment/evaluation/paired_bootstrap.py \
  --input runtime/development/public/validation \
  --trusted runtime/development/trusted/validation \
  --reference research/lightgbm-validation/predictions.parquet \
  --candidate research/temporal-ensemble-validation/predictions.parquet \
  --replicates 1000 --seed 20260929 --device cuda \
  --output research/temporal-vs-lightgbm-bootstrap.json
python environment/evaluation/analyze_subgroups.py \
  --input runtime/development/public/validation \
  --trusted runtime/development/trusted/validation \
  --prediction temporal=research/temporal-ensemble-validation/predictions.parquet \
  --output research/subgroups.json
```

## Publication and remaining limits

The private Hugging Face repository `main` now resolves to the clean development-only revision `f7da2d06295f1ed614de0ceb5619b3f72dbaedd7`. Its 13-file, 1,075,921,290-byte runtime inventory was matched against remote file sizes and LFS SHA-256 metadata, and a pinned download reproduced the root and runtime manifests. It excludes test input/truth, controller identifiers, raw source artifacts, and `working.duckdb`. The superseded revision `6b9c2cd9c92a0dfd795b51e04451a2b9507a027a` remains addressable through repository history pending explicit authorization for an irreversible history purge, so repository access must not be granted before that cleanup is complete.

Every accessor must independently satisfy current PhysioNet credentialing, the required CITI course, and the MIMIC-IV DUA; private repository visibility alone is insufficient. No test labels may be published. CPU (10 cores), memory (22,528 MB), disk (131,072 MB), one RTX 3070-class 8 GB GPU, and 24 agent-hours are recorded in task metadata and cover the measured recipes.

The evidence now supports reproducible model improvement space: non-linear rich features deliver most of the gain over simple statistics, multimodal values add signal beyond diagnoses, and temporal order contributes a smaller validation-supported increment. It does not establish biological onset prediction, transportability, or an independently confirmed test effect.
