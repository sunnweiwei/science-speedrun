# Verifier and regression tests

`test.sh` is the Harbor entry point. It runs `main_verifier.py`, which executes
untrusted `infer.sh` as uid 10001 with only `public/test` mounted, kills remaining
uid-10001 processes, and seals `predictions.parquet` with a run ID and content
hashes. `grader.py` is a separate labels-only service. It mounts exactly
`trusted/test`, the fixed train-only normalization parquet, and the same public
test input; it rejects a validation manifest or non-test boundaries. The grader
writes its result through the sealed volume, then main copies `report.json` and
`reward.json` to `/logs/verifier`.

Verifier main includes the pinned runtime dependencies used by the task image.
Package presence does not determine architecture compliance: the grader remains
an objective scorer, while the submitted neural model, training artifacts, and
actual `infer.sh` path are reviewed separately. The grader executes only trusted
scoring code against the sealed prediction file.

The primary score (`reward.json["score"]` and `reward.json["reward"]`) is
`panel_recall_at_5pct`: the equal-weighted mean of 365-day `Recall@5%` across the
17 eligible serious new-onset disease endpoints (`>= 20` complete-year cases after
precursor/recoding family-history exclusion) evaluated on all `11,424` held-out
ICD-10-era test patients (`8,269` complete-year primary patients). Valid
predictions receive a finite value in `[0, 1]` matching `ssi_index_v3.disease()`
within `< 1e-12` (constant/chance predictions score `~0.0501`). Missing,
malformed, timed-out, or failed submissions receive `0.0`. Trusted data, mount,
and scorer failures are infrastructure errors and produce no reward. There is no
trajectory field or sampler budget.

`test_risk_protocol.py` covers split reproducibility, competing death and censoring, feature leakage, direct-grid enforcement, probability constraints, disease-first/time-first aggregation, single-class handling, frozen denominator diagnostics, fixed-cohort scoring, and risk-only output. `test_longitudinal_v2.py` locks keyed label alignment, missing-value handling, event-time gaps, and diagnosis-only tensor compaction. `test_v4_tabpfn_features_and_score.py` verifies the sealed `comparators/v4_tabpfn_clinical_replay.py` collision-free feature dictionary, full 7-statistic lab retention, un-truncated support sampling with prior-odds rank invariance, 2,000-draw paired patient-level bootstrap CI, and `< 1e-12` parity with `ssi_index_v3.disease()`. `smoke_verifier.py` builds a tiny MIMIC-shaped fixture through the complete two-phase path. Patient bootstrap remains optional and disabled by default in `grader.py`.

Unit tests are deliberately separate from the Harbor score path:

```bash
docker build -f tests/Dockerfile --target unit-tests -t natural-disease-progression-tests .
docker run --rm natural-disease-progression-tests
```

For an end-to-end replay, set `TASK_PRIVATE_DIR` to the private tree produced by
`stage_runtime_data.py`, place a legal package at `/workspace/submission` through
Harbor, and start both services from `tests/docker-compose.yaml`. The main service
cannot mount `trusted/test`, its parent, `working.duckdb`, or raw MIMIC.

---

## Scientific & Literature Justification of Verifier Design

### 1. Primary Metric & Upstream Paper Provenance

- **Metric Definition**: For each endpoint $e$ in the prespecified 26-condition serious new-onset clinical panel (`environment/evaluation/panel_scoring.py:PANEL`), the eligible risk set $\mathcal{E}_e$ comprises held-out test patients with complete 365-day follow-up (`cutoff + 365d <= min(diagnosis_observation_end, death_observation_end)`), a valid 365-day incident label (`risk_set == True` and `ambiguous_prior == False`), and zero prior diagnosis codes in endpoint $e$'s precursor/recoding family (matching three-character ICD-10 categories exactly and historical ICD-9 prefixes). Patients in $\mathcal{E}_e$ ($n_e = |\mathcal{E}_e|$) are ranked by predicted 365-day cumulative risk $\hat{p}_{i,e,365}$, and top-$5\%$ case detection is computed with exact expectation over ties at threshold $k_e = \max(1, \lceil 0.05 \, n_e \rceil)$:
  $$\text{Recall@5\%}_e = \frac{1}{P_e}\left(\sum_{i:\,\hat{p}_i > \tau_e} y_{i,e} + \bigl(k_e - |\{i:\,\hat{p}_i > \tau_e\}|\bigr)\frac{\sum_{i:\,\hat{p}_i = \tau_e} y_{i,e}}{|\{i:\,\hat{p}_i = \tau_e\}|}\right), \quad S_{10} = \frac{1}{|\mathcal{E}^*|} \sum_{e \in \mathcal{E}^*} \text{Recall@5\%}_e$$
  where $\mathcal{E}^* = \{e \in \text{PANEL} : P_e \ge 20\}$ is the label-fixed set of **17** eligible endpoints on the `11,424`-patient held-out MIMIC-IV ICD-10 test cohort (`ssi_index_v3.disease`).
- **Upstream Literature Provenance**:
  - **Multi-disease longitudinal generative & hazard modeling**: Directly grounded in *Delphi-2M* (Shmatko et al., *Nature* 2025, DOI: `10.1038/s41586-025-09529-3`), which models multi-morbidity progression across hundreds of ICD chapters from longitudinal patient event histories, and *ETHOS* (Renc et al., *npj Digital Medicine* 2024), which evaluates tokenized EHR timelines on MIMIC-IV inpatient cohorts.
  - **Fixed-capacity clinical alert budget (`Recall@5%`)**: Grounded in post-discharge care-management and clinical decision support workload literature (*JAMA Network Open* 2024 clinical alert workload & post-discharge outreach studies), where transitional-care teams can realistically intervene on at most ~5% (1 in 20) of live hospital discharges. Under rare incident prevalence (0.3%–4%), global AUROC overweights distinctions in the bottom 80% of low-risk discharges that never trigger clinical outreach, whereas `Recall@5%` directly measures the fraction of true next-year cases captured within a fixed hospital outreach capacity while excluding prevalent/recoded chronic carry-forward diagnoses.
  - **Sealed Tabular Foundation Comparator (`tests/comparators/v4_tabpfn_clinical_replay.py`)**: Benchmark external reference calibration uses *TabPFN-3.5 / TabPFN v2* (Hollmann et al., *Nature* 2025) on cutoff-bounded multimodal clinical summaries. In V4, the sealed replay script eliminates both V3 audit artifacts—replacing `hash(code) % 128` / `% 64` bucket collisions with an explicit collision-free train-only prevalence vocabulary (top 256 exact ICD codes + top 128 exact medication codes + 26 ICD chapters + all 7 lab summary statistics `last, mean, std, min, max, log_count, missing`) and removing the `3,072`-row / `512`-positive support cap in favor of full-positive + stratified-negative support (`--max-support 10000` or full `~19k` cohort with exact Bayes prior-odds adjustment) paired via a 2,000-draw patient-level bootstrap on all `11,424` test patients.

### 2. Why Additional Ad-Hoc Constraints Are Intentionally Omitted

- **No Calibration Penalty Mixed into the Primary Ranking Score**: Probability calibration (`ECE`, calibration slope, prediction-to-observed ratio, and age-sex normalized multi-horizon Brier) is sensitive to historical ICD-9-to-ICD-10 secular prevalence shifts between pre-2015 training admissions and post-2015 ICD-10 test admissions. Operational post-discharge triage acts on top-$k\%$ risk orderings, so mixing an arbitrary Brier/ECE weight into `panel_recall_at_5pct` would conflate secular prevalence shift with patient-ranking discrimination. Monotonicity across the 13 horizons (`[30, ..., 360, 365]` days) and `[0, 1]` bounds are enforced as hard mathematical validity checks (`risk_scoring.py:read_predictions`), while calibration metrics remain unpenalized diagnostics.
- **No Ad-Hoc Heuristic Static Code Inspection in `grader.py`**: As documented in `main_verifier.py`, shallow filename or AST regex gates cannot reliably distinguish genuine learned neural architectures from disguised tabular wrappers and risk false-rejecting legitimate PyTorch modules; architectural compliance with the neural track contract is verified by reproducible training/inference artifact audit rather than brittle heuristic penalties in the scorer.
- **No Prediction-Dependent Endpoint Filtering**: Endpoint inclusion (`positives >= 20`, yielding 17 endpoints) and precursor/family-history exclusions depend strictly on frozen public history and trusted ground-truth labels, never on model outputs.

### 3. Diagnostic Signals Retained in `report.json`

Although `reward.json` exports only the unified scalar V3/V4 score (`score` and `reward`), `report.json` retains comprehensive clinical, temporal, and subgroup diagnostics for scientific analysis:
- ** stricter & Secondary Alert-Budget Metrics**: `panel.panel_recall_at_1pct`, `panel.panel_ppv_at_5pct`, `panel.panel_auroc`, `panel.panel_auprc`, and per-endpoint `recall_at_5pct`, `recall_at_1pct`, `ppv_at_5pct`, `ppv_at_1pct`, `auroc`, `auprc`, `eligible`, `positives`, and `family_history_excluded`.
- **Full-Vocabulary Multi-Horizon Discrimination**: `primary.macro_time_averaged_auroc` and `primary.macro_time_averaged_auprc` across all scored diseases and all 13 horizons (`30` to `365` days), plus per-horizon summaries in `horizons["30".."365"]`.
- **Proper Scoring & Calibration Diagnostics**: `auxiliary.raw_disease_time_brier_macro`, `auxiliary.age_sex_normalized_disease_time_brier` (against frozen train-only out-of-fold age/sex reference denominators), `panel.panel_calibration_slope_median`, `panel.panel_prediction_to_observed_median`, and `panel.panel_ece_decile`.
- **Competing Mortality & Demographic Subgroup Audits**: `mortality.death_time_brier` and per-horizon mortality AUROC/AUPRC/Brier, plus `strata` breakdowns across `age_18_64`, `age_65_plus`, `sex_F`, `sex_M`, `no_future_disease_record`, and `future_disease_record`.
