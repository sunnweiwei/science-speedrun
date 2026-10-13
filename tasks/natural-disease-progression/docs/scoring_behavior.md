# Scoring behavior acceptance

Deterministic synthetic tests exercise the formal score before any model comparison:

| Perturbation | Expected response |
| --- | --- |
| Improve ranking for one disease at multiple times | That disease's time-averaged AUROC improves before equal disease averaging |
| Pool many easy negatives across diseases | Cannot inflate the score; no micro AUROC is computed |
| Present a single-class disease/time cell | Reported unavailable, never scored as 0 or 0.5 |
| Keep ranking but make probabilities overconfident or underconfident | Primary AUROC can remain unchanged; raw/normalized Brier and calibration diagnose it |
| Supply only 30/90/180/365 values | Rejected for wrong dimensions; evaluator does not interpolate |
| Make cumulative risk decrease | Rejected before scoring |
| Change post-cutoff data | Public history and therefore model input remain unchanged |

The synthetic aggregation regression uses two diseases and two times, including one single-class cell. It verifies the exact disease-first result and proves it differs from pooled micro AUROC. A separate tie regression matches scikit-learn AUROC/AUPRC, and the legacy normalization test remains as an auxiliary-calibration regression. Production Brier denominators come only from the frozen training artifact, never from evaluated predictions or labels.

The tests also verify competing-death labels, censor masks, patient-level fixed cohort membership, keyed label alignment, absence of trajectory outputs, exact normalization coverage, and separate event/no-event diagnostic strata. They do not prove causal disease-onset modeling or eliminate hospital observation effects; those remain explicit limitations.
