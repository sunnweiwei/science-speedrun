# Natural Disease Progression protocol

> **v3 evaluation (2026-10-01).** Cohort, inputs, labels and the prediction interface below are
> unchanged. The splits and the reward changed. Validation and test are ICD-10-era index
> admissions, and the reward is `panel_recall_at_5pct` (case detection for serious new-onset
> disease at a 5 % alert budget). The disease-first macro time-averaged AUROC defined below is
> still computed and reported, but no longer rewarded. See [EVALUATION_DESIGN.md](EVALUATION_DESIGN.md).

This document defines the only supported endpoint and scorer. The `longitudinal-v2` wire identifier in `environment/data/protocols/longitudinal-v2.json` uses the second eligible live adult discharge, one sample per patient, and requires at least 50 train 365-day events. The patient split, target semantics, time grid, censoring logic, vocabulary, and AUROC aggregation are frozen. First-discharge artifacts are historical audit evidence only and are rejected by the active launcher.

## Cohort and information boundary

MIMIC-IV v3.1 adult inpatients are frozen by `subject_id` into 70% train, 15% validation, and 15% test. Existing split rows are reused and verified. The cutoff is the second qualifying live adult discharge and does not require a later admission or future diagnosis. Public history includes demographics, diagnoses recorded by cutoff, labs whose result availability is no later than cutoff, prescriptions already ordered, and administrations already documented with an available code source. It contains no text, raw patient ID, future event, label, follow-up boundary, or future-use metadata. All raw encounter and source-row identifiers are replaced with opaque salted keys.

Diagnosis record time is discharge. Diagnoses within a stay are an unordered simultaneous set. ICD-10 is reduced to validated three-character categories; ICD-9 uses the fixed CMS mapping, with ambiguous candidates masked rather than expanded. Approximate age follows MIMIC anchors and flags age de-identification. The final index-stay diagnosis set is allowed by the retrospective discharge protocol; this is not a validated real-time deployment protocol.

## Target and masks

For each patient and frozen scored disease, the target is the first clearly mapped MIMIC inpatient diagnosis record after cutoff. Multiple first records from one admission share its discharge date and are all events. A disease already recorded by cutoff is outside its risk set. A historical ambiguous ICD-9 mapping that could denote the disease also masks that pair. The task does not model recurrence, remission, severity, or future admission sequences.

Death is a separate terminal event. A disease record on or before the death day can be positive; no event after death is positive. If death occurs before a horizon with no prior disease record, that patient-disease label is a known terminal negative, not ordinary censoring. Otherwise a negative is valid only when the patient-specific diagnosis observation boundary covers the horizon. Death labels use their separate death observation boundary. Unknown follow-up is masked, never set to zero.

The primary cohort is fixed before model scoring and requires both observation boundaries to cover day 365. Early death or future disease does not change membership. Patients with no subsequent admission or disease remain included. Partial-observation samples remain in data with masks but are outside the primary full-year result.

## Prediction interface

`predict_risk(history,horizons)` must directly return cumulative disease risk `[batch,D,13]` and death risk `[batch,13]` at `30,60,...,360,365` days. Probabilities are finite, in `[0,1]`, and nondecreasing. Four-point output followed by evaluator interpolation is invalid: the finer grid is part of what is assessed. Diseases are not mutually exclusive and their probabilities need not sum to one. There is no required `sample_trajectory` interface.

## Scores

For disease `d` and time `t`, let `V(d,t)` be the frozen set of complete-year primary-cohort patients for whom the label is valid and the disease remains in its risk set. If both a positive and a negative occur in `V(d,t)`, the evaluator computes ordinary patient-level `AUROC(d,t)`. A no-positive or no-negative cell is unavailable: it is never replaced by 0 or 0.5.

The primary score is

`Primary = mean_valid_d [ mean_valid_t AUROC(d,t) ]`.

Thus time points have equal weight within a disease and diseases have equal weight in the final score. Patients, diseases, and times are never pooled into a micro AUROC. The valid set depends only on frozen labels and observation rules, not on the submitted predictions. The report records valid/frozen disease and cell counts plus positive and negative supports. In v2 Harbor wrote `{"reward": Primary}`; v3 writes `panel_recall_at_5pct` instead. Invalid predictions receive zero.

The same disease-first/time-first aggregation is reported for AUPRC. The report also retains day-365 and per-time macro AUROC, per-time macro AUPRC, ordinary raw disease-time Brier, ten-bin calibration error, and the prior frozen age/sex-normalized Brier. That normalization artifact remains train-only and fixed for calibration continuity; it is no longer the reward. Death AUROC/AUPRC/Brier and death time Brier are reported separately and never weighted into disease reward. Age, sex, and future-record strata are diagnostics on the same fixed cohort.

Patient-paired comparisons resample complete-year patients with replacement and recompute every AUROC. A disease/time cell that becomes single-class in a bootstrap sample is unavailable for both compared models in that replicate, followed by the same time-within-disease and disease averaging. The same resampled patients are used for both models. This interval quantifies patient sampling only; the spread over prespecified training seeds is reported separately.

This benchmark measures **MIMIC first-recorded inpatient diagnoses, not true onset**. Its predictions necessarily combine disease evolution with observation, care-seeking, and MIMIC capture.
