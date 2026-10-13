# Evaluation design (v3): why the reward and the splits changed

The longitudinal-v2 cohort, inputs, labels and prediction contract are unchanged. Two things were
redesigned during integration (2026-10-01), so that a higher reward means clinically meaningful
progress: which patients are scored, and what is scored. This page records the evidence. The
operator procedure is in [../environment/data/MATERIALIZE.md](../environment/data/MATERIALIZE.md);
the scorer is [../environment/evaluation/panel_scoring.py](../environment/evaluation/panel_scoring.py).
Prototype numbers are in [evaluation-design/prototype-results.json](evaluation-design/prototype-results.json).

## 1. A coding-era shortcut in v2

MIMIC-IV coded diagnoses in ICD-9 until October 2015 and in ICD-10 afterwards. Event dates are
shifted per patient, but each diagnosis keeps its `mapping_status` (`direct` is native ICD-10). The
index (second) admission therefore dates every sample to before or after the switch:

| index admission | patients (train + validation of v2) |
|---|---:|
| ICD-9-era | 45,346 |
| ICD-10-era | 38,149 |
| mixed or no diagnoses | 19 |

v2 counts an ICD-9 outcome record only when its category mapping is unambiguous (`docs/protocol.md`:
"ambiguous candidates masked rather than expanded"). After an ICD-9-era index admission, most first
records of many diseases are therefore missing and appear as negatives:

| one-year incidence | ICD-9-era index | ICD-10-era index |
|---|---:|---:|
| sepsis A41 | 0.15 % | 3.29 % |
| heart failure I50 | 0.13 % | 2.53 % |
| acute MI I21 | 0.06 % | 1.26 % |
| atrial fibrillation I48 | 0.12 % | 1.93 % |
| acute kidney injury N17 (unambiguous mapping) | 5.73 % | 5.61 % |
| all 262 diseases | 0.38 % | 0.89 % |

v2 mixes both eras in every split, so recognising the era earns reward. On v2 validation, the
disease-first macro time-averaged AUROC (the v2 reward) is:

| predictor | v2 reward |
|---|---:|
| age only | 0.570 |
| age/sex baseline (documented) | 0.630 |
| **"index admission coded in ICD-10" (0/1), no clinical content** | **0.653** |
| reference temporal GRU | 0.831 |

When the same models are scored on ICD-10-era patients only, the v2 reward falls by about 0.07
(prototype: GRU 0.816 → 0.760, LightGBM 0.811 → 0.749).

**Fix.** Validation and test hold only ICD-10-era index admissions, whose outcomes are completely
coded. Training keeps the ICD-9-era patients; their incomplete labels are disclosed in the
instruction, and using them well is part of the problem. In the prototype, adding them helped the
GRU (0.198 → 0.228 on the new reward) and slightly hurt LightGBM (0.256 → 0.236).

**Rejected alternative.** Training on the ICD-9 era and testing on the ICD-10 era looks like a
temporal split, but it mostly measures the label gap. Every model's predicted risk fell to
0.09–0.25× the observed rate. A genuine prospective split needs `anchor_year_group`, which the
release does not carry.

## 2. The reward: case detection for serious new disease

The v2 reward averaged AUROC over all 262 diseases. Three things made it a weak signal of progress:

- **Saturation.** Most of the 0.5 → 0.83 range comes from history any tabular model reads. The
  model-dependent band was about 0.01–0.03 wide, against a seed SD of about 0.004.
- **Easy endpoints.** The GRU's best diseases were O69 0.99, F19 0.98, O99 0.98 and F15 0.97:
  obstetric and substance-use codes that largely continue recorded care.
- **Rank only.** The GRU's macro AUPRC was 0.030, and nothing rewarded calibrated risk.

**v3 reward: `panel_recall_at_5pct`.** The panel is 26 prespecified serious, actionable endpoints:
heart failure, AF, MI, ischaemic stroke, ICH, PE, venous thrombosis, aortic disease, type 2
diabetes, CKD, AKI, sepsis, pneumonia, respiratory failure, cirrhosis, hepatic failure, dementia,
COPD, ILD, colon, liver, pancreas, prostate and bladder cancer, and two metastasis groups.

For each endpoint:
- **Eligibility.** Patients are complete-year with a valid 365-day label, and their history must
  carry none of the endpoint's precursor or recoding codes (for example I11/I13 for heart failure,
  I12/I13/N19 for CKD, or F01/F02/G30/G31 for dementia). This is strict new onset.
- **Score.** Patients are ranked by predicted 365-day risk, and the endpoint score is the share of
  cases found in the top 5 %. Chance is 0.05 and perfect is 1.0.

The reward is the mean over endpoints with at least 20 eligible cases. That set is fixed by the
labels: 17 endpoints on test and 14 on validation. Ties are credited at their expectation, so a
constant prediction scores 0.0501 on test.

Recall at 1 %, PPV, AUROC, AUPRC and calibration are reported per endpoint, together with the v2
metrics.

Why recall at an alert budget:
- It is the question a follow-up programme asks: of the patients we can call back, how many will
  be next year's cases?
- It is bounded in [0, 1], so it fits `reward.json` directly.
- It leaves room to improve. The best baseline finds about a quarter of next-year cases.
- Calibration is reported but not rewarded. The GRU's calibration slope is about 3, so its risks
  are far too compressed. Whether calibration should gate the reward is left to the score map.

## 3. Measured on the v3 splits (one seed each)

Train 68,175 / validation 3,915 / test 11,424. Each model was trained by its documented recipe and
scored by the task's own `score.py`. The GRU row was also scored through the full two-phase
verifier, with an identical reward.

| model | validation reward | **test reward** | test recall@1 % | test panel AUROC | test calibration slope | test v2-metric AUROC |
|---|---:|---:|---:|---:|---:|---:|
| statistical (empirical-Bayes strata) | 0.146 | 0.185 | 0.034 | 0.728 | 0.64 | 0.676 |
| LightGBM, 567 features (maintainer reference) | 0.235 | **0.268** | 0.095 | 0.786 | 0.72 | 0.746 |
| temporal GRU (`baseline/temporal-gru`) | 0.211 | 0.236 | 0.071 | 0.782 | 2.92 | 0.767 |

In the prototype, a patient-paired bootstrap gave a single-model interval of about ±0.03:
differences of 0.05 or more are resolved, 0.01-level differences are not. LightGBM leads the GRU
here, which is the point of the track: a learned representation should overtake hand-built
features.

## Open items

- A genuine prospective split (`anchor_year_group`) needs the controller's raw build.
- Fixing ICD-9 outcome labelling, by counting ambiguous GEM candidates, would make the ICD-9-era
  patients fully usable.
- The panel and its family exclusions should be reviewed by a clinician.
- Decide whether calibration enters the score map.
