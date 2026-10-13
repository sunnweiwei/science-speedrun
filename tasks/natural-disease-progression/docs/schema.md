# Data and prediction schema

## Public input

| File | Fields | Visibility |
| --- | --- | --- |
| `samples.parquet` | `sample_id`, sex, approximate age, high-age flag, cutoff | Model-visible |
| `events.parquet` | `sample_id`, optional stay ID, event type/code/value/unit, event and availability time, source, simultaneous group, mapping status, relative time | Model-visible; all rows cutoff-bounded |
| `manifest.json` | schema/task, split, ordered scored diseases, `horizons_days` | Model-visible |

`sample_id` aligns output and is not a feature.

## Training-only exports

| File | Purpose |
| --- | --- |
| `train/timeline.parquet` | Complete normalized structured timeline for all train patients |
| `train/admissions.parquet` | Every train stay and complete diagnosis set under anonymous patient/stay keys |
| `train/patients.parquet` | Demographics, death and observation boundaries under anonymous keys |
| `train/labels.parquet` | Compact first-future-record state for each sample/disease |
| `train/label_windows.parquet` | Direct Boolean label and validity columns at all 13 horizons |

Admissions are retained as training context only; they are not a validation/test prediction target.

## Trusted evaluator

`trusted/{validation,test}/labels.parquet` and `trusted/{validation,test}/label_windows.parquet` mirror the target state and direct labels for exactly one split. Each matching `sample_boundaries.parquet` stores cutoff, separate diagnosis/death observation ends, death state, and a diagnostic any-future-disease flag. `trusted/risk_normalization.parquet` is split-independent and stores one frozen train-only age/sex reference Brier, effective denominator, reciprocal weight, and floor flag for every scored disease/time cell. `trusted/subject_split.parquet` is the frozen raw-ID split and never enters either runtime tree.

## Submission output

`predictions.parquet` has exactly one row per public sample:

| Field | Type |
| --- | --- |
| `sample_id` | string |
| `disease_risk` | fixed list of float32, length `D*13`, disease-major then time |
| `death_risk` | fixed list of float32, length 13 |

Every entry must be finite and in `[0,1]`; each disease and death curve must be nondecreasing. The grid must exactly match the manifest. There is no `trajectories` column.

The v2 trusted report contains `primary.macro_time_averaged_auroc`, disease/cell/support counts, `primary.macro_time_averaged_auprc`, auxiliary raw and frozen age/sex-normalized Brier, separate mortality results, per-horizon and per-disease metrics with calibration/counts/reasons, fixed-cohort sizes, and strata. `reward.json` contains exactly the available primary AUROC (or zero for an invalid submission). Optional patient-bootstrap intervals are disabled by default.
