# B1 / B1g mean delta: reproduced results

Release v1, cell-eval 0.8.2, isolated verifier (three test contexts) and `run_loco.py` (four
training-context folds). Records: `../evaluation_history.jsonl`, candidates `B1-mean-delta` and
`B1g-gated-mean-delta`.

| variant | reward (final) | leaderboard | near final | mid final | far final | LOCO k562 | jurkat | hct116 | hek293t |
|---|---|---|---|---|---|---|---|---|---|
| B1 mean delta | **0.103** | 0.095 | 0.150 | 0.118 | 0.042 | 0.089 | 0.153 | 0.039 | 0.042 |
| B1g gated mean delta | 0.102 | 0.094 | 0.150 | 0.118 | 0.039 | 0.089 | 0.153 | 0.026 | 0.028 |

Per-metric normalised scores on the final subsets:

| variant | context | DES | PDS | MAE | pearson_delta | spearman_lfc | auprc_de | spearman_effect |
|---|---|---|---|---|---|---|---|---|
| B1 | near | 0.069 | 0.097 | 0.006 | 0.147 | 0.249 | 0.000 | 0.480 |
| B1 | mid | 0.079 | 0.037 | 0.002 | 0.128 | 0.219 | 0.000 | 0.359 |
| B1 | far | 0.063 | 0.024 | 0.000 | 0.074 | 0.132 | 0.000 | 0.000 |
| B1g | near | 0.069 | 0.097 | 0.006 | 0.147 | 0.249 | 0.000 | 0.480 |
| B1g | mid | 0.079 | 0.037 | 0.002 | 0.128 | 0.219 | 0.000 | 0.359 |
| B1g | far | 0.069 | 0.024 | 0.000 | 0.075 | 0.100 | 0.007 | 0.000 |

The withheld targets score 0 in every context (the table has nothing to say about them).
`conserved-strong` targets transfer better than `context-variable` ones (all 300 requests: 0.129
vs 0.057 near, 0.106 vs 0.060 mid, 0.051 vs 0.044 far). Gating changes nothing on the 3' v3 panels
and slightly hurts on the far context and the Orion folds. Runtime: table build < 2 min, inference
8–30 s per context.
