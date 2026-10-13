# Baseline results

These are legacy first-discharge v1 results retained for continuity. They are not the current longitudinal-v2 research comparison, and the old two-epoch neural result is not an estimate of temporal-model capacity. See [the longitudinal-v2 report](longitudinal-v2-report.md) for the audited experiment.

The risk-only v1 requires fresh direct predictions at 13 time points. Scores from the superseded trajectory protocol are intentionally not carried forward. These validation point estimates use the regenerated MIMIC-IV v3.1 artifacts, the same 28,115-patient complete-year cohort, 346 frozen diseases, and no bootstrap calculation.

| Baseline | Input | Primary normalized Brier | Raw disease-time Brier | Death time Brier | 365d Macro-AUROC | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Full | demographics + diagnosis + labs + medications | 0.698399 | 0.00212438 | 0.0379916 | 0.703565 | complete; existing checkpoint rescored |
| Diagnosis-only | demographics + diagnosis | — | — | — | — | stopped by user; no completed checkpoint |
| Age/sex | train-only demographic strata | 0.672234 | 0.00208798 | 0.0436750 | 0.666506 | complete; existing predictions rescored |
| Prevalence | train-only disease rates | 0.672964 | 0.00209440 | 0.0463517 | 0.500000 | complete; existing predictions rescored |

At day 365 the full model has normalized Macro-Brier 0.825522, raw Macro-Brier 0.00293503, Macro-AUPRC 0.00845625, and macro ten-bin ECE 0.00081540; 345 of 346 diseases have both classes and therefore a defined AUC. The full model ranks risk better than the two rate baselines but has worse normalized and raw Brier, so this minimal neural baseline is not adequately calibrated. Normalization makes its primary-score gap from the demographic model about 3.89%, versus about 1.74% under raw macro Brier, while retaining the proper unweighted patient error.

The normalization reference was cross-fit only on train. Across 4,498 disease/time cells, the reliable-cell 5th percentile was 0.000213768; the predeclared 1000 maximum weight therefore fixed the effective floor at 0.001. The floor applies to 2,620 cells. Because of this floor and the out-of-fold reference, the demographic validation score is not expected to equal 1.

The full run used two epochs over 305,202 epoch-samples on an RTX 3070, took 1,876.39 seconds, and ended at training loss 0.178840. The age/sex and prevalence fits took 83.76 and 16.46 seconds. No model was retrained for this metric change: saved direct-grid predictions were rescored. Diagnosis-only was interrupted at the user's request and no partial checkpoint was written.

Commands (replace `$NAME` with `full`, `diagnosis`, `demographic`, or `prevalence`):

```bash
python docs/maintenance/legacy-neural/train_baseline.py --materialized materialized \
  --output materialized/checkpoints/$NAME.pt --baseline $NAME --epochs 2 --device cuda
python docs/maintenance/legacy-neural/predict.py materialized/public/validation \
  materialized/validation-predictions/$NAME \
  --checkpoint materialized/checkpoints/$NAME.pt --device cuda
python environment/evaluation/score.py --input materialized/public/validation \
  --trusted materialized/trusted/validation \
  --normalization materialized/trusted/risk_normalization.parquet \
  --predictions materialized/validation-predictions/$NAME/predictions.parquet \
  --output materialized/validation-score/$NAME
```

Synthetic metric behavior checks are documented in [scoring_behavior.md](scoring_behavior.md). Baselines are deliberately small reference points, not claims of clinical usefulness.
