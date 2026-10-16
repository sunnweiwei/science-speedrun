# Research report

## Current best candidate
- candidate_id: (none yet) · submission SHA-256: · published (UTC):
- leaderboard scores (ctx_near / ctx_mid / ctx_far): · dev LOCO mean:

## Model, data, training, reproducibility
(architecture; perturbation and context representation; objective; data view and caps used;
training contexts; seeds; commands to reproduce from this directory)

## Candidate table (chronological, every published candidate)
| candidate | parent | hypothesis | LOCO k562 | jurkat | hct116 | hek293t | LB near | LB mid | LB far | status |
|---|---|---|---|---|---|---|---|---|---|---|

## Baselines on the same folds
| baseline | LOCO k562 | jurkat | hct116 | hek293t | LB near | LB mid | LB far |
|---|---|---|---|---|---|---|---|
| B0 no-effect | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B1 mean delta | 0.089 | 0.153 | 0.039 | 0.042 | 0.162 | 0.083 | 0.041 |
| B1g gated mean delta | 0.089 | 0.153 | 0.026 | 0.028 | 0.162 | 0.083 | 0.036 |
| B2 STATE zero-shot (gene-space ST) | 0.011 | 0.063 | – | – | 0.038 | 0.013 | 0.051 |

## Findings by question
### Q1 — effect existence / strength in a new context
### Q2 — conserved vs. context-specific response
### Q3 — never-seen targets

## Ablations and negative results

## Error analysis (per stratum, per metric)

## Compute use

## Limitations

## Next hypotheses (ranked)
1.
