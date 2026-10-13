# MIMIC-IV v3.1 audit

This report records observed local aggregates, not estimates from Delphi-2M. Source was the authorized MIMIC-IV v3.1 directory `/home/zifeng/datasets/mimic-iv/3.1`; all eight used tables were checked against the release manifest. The converter does not mix releases. Reviewable source aggregates are in [audit.json](audit.json), and frozen disease counts are in [scored_diseases.json](scored_diseases.json). Detailed per-disease split counts remain in ignored credentialed `materialized/risk_audit.json` because rare cells are not redistributed.

## Population and coverage

| Quantity | Observed |
| --- | ---: |
| Adult inpatients | 223,452 |
| Adult admissions | 546,028 |
| Exactly one admission | 123,289 patients (55.17%) |
| Multiple admissions | 100,163 patients (44.83%) |
| Qualifying live-discharge samples | 218,025 |
| Samples with only one admission | 117,888 |
| Frozen patients: train / validation / test | 156,416 / 33,518 / 33,518 |
| Cutoff samples: train / validation / test | 152,601 / 32,699 / 32,725 |
| Complete diagnosis year: train / validation / test | 131,697 / 28,115 / 28,197 |

All 218,025 samples have the separately defined 365-day death observation window. Diagnosis coverage is 213,527 / 208,877 / 201,774 / 188,009 at 30 / 90 / 180 / 365 days. The fixed primary cohort requires both day-365 boundaries; future events and early death do not determine membership. The main exclusion is conservative diagnosis-calendar shortfall (20,904 train, 4,584 validation, 4,528 test). Patients without later admission or disease record are retained.

Historical cutoff inputs contain a median of 8 diagnosis records and 159/160/161 total events in train/validation/test; the corresponding 90th percentiles are 615/631/613. The train cutoff-input diagnosis vocabulary contains 7,192 tokens and the full train timeline contains 8,017, including retained ambiguous/unmapped ICD-9 source tokens. The frozen scored output contains 346 ICD-10 three-character diseases, chosen only from train positives at the existing threshold of 50 in the A–Q/U family. It is distinct from the input vocabulary.

## First-record target scale

The following descriptive counts are record-time groups of first newly recorded mapped categories during the first 365 days. They do not define cohort eligibility and are not an admission-trajectory target.

| Split | 0 groups | 1 group | 2 groups | ≥3 groups | Multiple new diseases in one admission | New diseases across ≥2 later admissions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 114,914 (75.30%) | 24,619 (16.13%) | 7,411 (4.86%) | 5,657 (3.71%) | 30,996 | 13,068 |
| Validation | 24,605 (75.25%) | 5,257 (16.08%) | 1,542 (4.72%) | 1,295 (3.96%) | 6,667 | 2,837 |
| Test | 24,667 (75.38%) | 5,183 (15.84%) | 1,624 (4.96%) | 1,251 (3.82%) | 6,618 | 2,875 |

The approximately 75% zero-group fraction remains in the primary population. Same-admission diseases share a discharge time, while cross-admission groups can differ. The formal output is each scored disease's cumulative first-record probability, not the number or contents of future admissions.

`risk_audit.json` reports, for every one of the 346 frozen diseases and all 13 time points, train/validation/test positives and valid risk-set counts. It also records actual complete-year coverage and vocabulary sizes. Validation and test results never alter the frozen list.

## Frozen score normalization

The primary-score reference was fit on the 131,697 complete-year train samples only, using deterministic five-fold out-of-fold age-bin/sex rates with Laplace smoothing. It covers all 346 diseases at all 13 direct horizons (4,498 cells). Of these, 4,330 cells met the predeclared reliability condition of at least 20 positives and 20 negatives for estimating the floor; no small-fixture fallback was used.

The reliable-cell 5th percentile reference Brier was 0.0002137677. The predeclared maximum inverse weight of 1000 set the effective denominator floor to 0.001, which applies to 2,620 cells. Raw reference Brier had median 0.0007407, 90th percentile 0.0043774, 99th percentile 0.0245356, and maximum 0.0424113. These values and every disease/time denominator are frozen artifacts; validation and test labels are not inputs to their construction.

## Coding and modalities

There are 6,364,488 diagnosis rows: 2,908,741 ICD-9 (45.70%) and 3,455,747 ICD-10 (54.30%). With CMS 2018 diagnosis GEMs, 1,046,121 ICD-9 rows (35.96%) have a clear unique exact non-combination mapping, 1,813,659 are ambiguous, and 48,961 are unmapped. Ambiguous candidates mask patient-disease pairs rather than becoming multiple positives. The CMS archive SHA-256 is `1c5e5f14026ace48437a0d1c485d282fb5d171c357c6ab4f7798fc0a1e3624a2`.

Raw coverage is 158,374,764 lab rows / 313,442 patients, 20,292,611 prescription rows / 196,738 patients, and 42,808,593 eMAR rows / 195,717 patients. The normalized timeline contains 6,362,171 diagnoses, 148,328,263 labs, 17,718,866 prescriptions, and 24,067,476 administrations after conservative availability and medication-code provenance checks. These are repeated event rows, not target counts.

The earlier paired cutoff audit found admission-time historical lab coverage of 85.74%. This was rechecked: counted labs require both `charttime` and `storetime` no later than admission and `storetime >= charttime`; 745,938 charted-before but stored-after rows were excluded. The median storage-to-admission lead was 1,179 hours, so this reflects any historical lab, not immediate pre-admission testing. The formal cohort remains discharge-based.

## Timestamp and death checks

Of 546,028 admissions, 175 have discharge earlier than admission. Labs have 2,568,511 missing `storetime` values and 1,372 store-before-chart conflicts; observed chart-to-store delay has median 1 hour, 90th percentile 5 hours, and 99th percentile 20 hours. Prescriptions have 21,890 missing starts, 31,436 missing stops, and 816,994 stop-before-start conflicts. eMAR has 1,457,991 store-before-chart rows. Conservative rules exclude unreliable rows and never interpolate across cutoff.

There are 36,882 adult inpatients with recorded death date. Twenty-two adult admission rows have `dod` before admission, 32 before discharge, and one has a date disagreement between hospital `deathtime` and patient `dod`. Same-day terminal diagnoses may tie death; diagnoses after death are excluded. A total of 5,427 adults have no qualifying live-discharge index.

These facts reinforce the interpretation boundary: an unobserved future diagnosis is not proof of no biological onset. The task predicts **MIMIC first-recorded inpatient diagnosis time**, conditional on its explicit observation rules.
