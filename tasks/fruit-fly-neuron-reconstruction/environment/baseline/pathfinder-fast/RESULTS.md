# Selected starter: measured development results

The selected FFN is legacy step 6750 plus one production-rule update, using
corrected production inference. See [README.md](README.md) for the recipe and
checkpoint hashes. The checkpoint is a measured reference; trained weights
are not supplied to agents and a fresh training run must be evaluated.

| Evaluation scope | Selected starter nERL | Corrected-from-start step 16300 comparator |
|---|---:|---:|
| Reduced-context 311-cube proxy | 0.0722669 | 0.0349967 |
| Full-context single window | 0.0707893 | 0.0406383 |
| Stitched eight windows | 0.0350171 | 0.0190529 |

Compare models only within a row. The comparator is not the selected recipe;
its inclusion shows the measured selection basis. Eight-window nERL is
recomputed on the original reference graph, not averaged across windows.
The selected starter's eight-window reference-node omission is 52.26%.
Lower omission or better seed Dice alone does not establish higher nERL.

These are native GPU FFN development evaluations, not the complete private-test
SENSE/SHAPE reward. Proxy ranking requires larger-window and stitched
confirmation. The comparisons do not identify a single cause of the selected
recipe's advantage: training settings and exposure also differ.

## Evaluation cost

Runtime depends on the checkpoint and evaluation scope. A completed four-GPU
eight-window run took 90 minutes (6 GPU hours); the step-16300 comparator took
1h29m11s for one window on one GPU and 4h31m50s for eight windows on four GPUs.
These are measured examples, not a selected-starter timing guarantee. They
exclude training, queueing and earlier failed attempts. No universal ≤2.5-hour
completion claim follows from them. The four-checkpoint proxy calibration was
3.46–5.50x faster than full-window evaluation, with score differences up to 26%.

## Verification scope

Native training, inference and scoring completed within the scopes above.
Independent fresh replay of the entire selected two-stage recipe and complete
GPU execution through the exact formal container entry are not recorded.
Docker is unavailable on the author host; deployment coverage is reported
separately from algorithm results.

Maintainer evidence in the repository: `docs/validation/random-start-final.json`,
`random-start-multiblock.json`, and `contender-runtime-compatibility.json` in the
same directory. The task README and `docs/READINESS.md` summarize that evidence.
