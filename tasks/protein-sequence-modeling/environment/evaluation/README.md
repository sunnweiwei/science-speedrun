# Self-service development evaluation

Score yourself whenever you want. There is **no evaluation service and no
wrapper** — `score.py` is the same code the verifier runs, so a number you get
here is the number the grader would give on the same predictions.

```bash
# after writing predictions (and optional swapped PPI predictions) for the tune split
python3 /workspace/evaluation/score.py \
    --predictions OUTPUT_ROOT/predictions \
    --labels /workspace/data/tune \
    --stats /workspace/data/stats.json
```

If you also run inference on swapped `(sequence_b, sequence_a)` inputs for
`yeast_ppi` and `human_ppi`, pass `--swapped-predictions SWAPPED_OUT/predictions`
(or place `yeast_ppi_swapped.jsonl` / `human_ppi_swapped.jsonl` inside
`OUTPUT_ROOT/predictions`) to reproduce the verifier's exact two-orientation
symmetric PPI probability averaging `p_sym = 0.5 * (p_ab + p_ba)`.

`score.py` prints a per-task and four-pillar report plus the primary `score`
(`reward`), and flags any task whose predictions violate the contract with an
`invalid_reason` instead of silently hiding the failure — read those, they mean
the package would zero that task in the real verifier.

## What is here

| File | Purpose |
| --- | --- |
| `metrics.py` | The scoring implementation: `score_task(...)` computes chance-calibrated V3 task `skill` (`reward`) and symmetric PPI diagnostics; `aggregate_pillars(details)` computes the 4 equal-weight capability pillars (`Function`, `Localization`, `Structure`, `Protein interactions`), `signed_score`, and `score = clip(signed_score, 0, 1)`. |
| `score.py` | A thin driver over `metrics.py` for a directory of predictions and labels. |

`metrics.py` is kept **byte-identical** with `tests/metrics.py`, which the
trusted grader imports; a unit test enforces it. That is the whole point: the
metric you optimize against is the metric you are graded on, so there is no gap
to be surprised by at replay time.

## Using it during training, not just at the end

`starter/train.py` imports `metrics.py` directly, evaluates both PPI orientations
on `tune` periodically, and selects checkpoints using `metrics.aggregate_pillars`.
Model selection is part of what the contract asks a submission to document, and
doing it against the real four-pillar chance-calibrated metric avoids selecting
on a proxy loss that ignores class imbalance or pillar balance.

## What you cannot evaluate here

`fit` and `tune` are labeled and yours to use. The held-out eval split is not
mounted in this container: its inputs and labels live only in the verifier and
grader environments. Scores on `tune` are therefore an estimate, and `tune` is
small for some tasks — `yeast_ppi` has 95 rows, `human_ppi` 315, `ppi_affinity`
203 — so treat single-task movements on those as noisy and prefer the four-pillar
aggregate when deciding what to keep. Note also that `SEQ_K["fold"] = 136` is
calibrated to the 136 observed fold classes in `test_fold_holdout`.
