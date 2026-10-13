# B2 STATE zero-shot: reproduced results

Pinned `arc-state` 9bbfe78, `state_sm`, gene-space ST with ESM-2 perturbation features, 20k steps
on a 7,330-gene common panel, trained on all four training contexts (about 1.5 h on one uncontended
H100, 4.5 h under load). Release v1, cell-eval 0.8.2, isolated verifier. Record:
`../evaluation_history.jsonl`, candidate `B2-state-zeroshot`.

| reward (final) | leaderboard | near final | mid final | far final | LOCO k562 | jurkat | hct116 | hek293t |
|---|---|---|---|---|---|---|---|---|
| 0.046 | 0.034 | 0.063 | 0.033 | 0.043 | 0.011 | 0.063 | not run | not run |

Per-metric normalised scores on the final subsets:

| context | DES | PDS | MAE | pearson_delta | spearman_lfc | auprc_de | spearman_effect |
|---|---|---|---|---|---|---|---|
| near | 0.061 | 0.006 | 0.000 | 0.010 | 0.027 | 0.009 | 0.325 |
| mid | 0.085 | 0.006 | 0.000 | 0.000 | 0.000 | 0.002 | 0.138 |
| far | 0.188 | 0.000 | 0.000 | 0.038 | 0.011 | 0.064 | 0.000 |

STATE beats B0 everywhere but loses to the mean-delta table overall (0.046 vs 0.103), with two
exceptions on the far context: DES 0.19 vs 0.07 for B1, and 0.12 on the withheld targets where B1
scores 0 (ESM-2 perturbation features generalise to genes never perturbed in training). It gives
up PDS and MAE. This is the SOTA architecture at a small budget, not a trained frontier model.
Inference takes 2–5 min per context.
