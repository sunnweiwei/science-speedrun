# Self-service development evaluation

Score yourself whenever you want. There is **no evaluation service and no
wrapper** — `score.py` is the same code the verifier runs, so a number you get
here is the number the grader would give on the same predictions.

```bash
# after writing predictions for the tune split
python3 /workspace/evaluation/score.py \
    --predictions OUTPUT_ROOT/predictions \
    --labels /workspace/data/tune \
    --stats /workspace/data/stats.json
```

It prints a per-task report and the primary reward, and flags any task whose
predictions violate the contract with an `invalid_reason` instead of silently
scoring zero — read those, they mean the package would fail the real verifier.

## What is here

| File | Purpose |
| --- | --- |
| `metrics.py` | The scoring implementation: `score_task(task, truth, found, ppi_affinity_std)` returns the per-task reward and its components. |
| `score.py` | A thin driver over `metrics.py` for a directory of predictions and labels. |

`metrics.py` is kept **byte-identical** with `tests/metrics.py`, which the
trusted grader imports; a unit test enforces it. That is the whole point: the
metric you optimize against is the metric you are graded on, so there is no gap
to be surprised by at replay time.

## Using it during training, not just at the end

`starter/train.py` imports `metrics.py` directly and scores the `tune` split
periodically, keeping the best checkpoint. Model selection is part of what the
contract asks a submission to document, and doing it against the real metric
avoids the classic failure of selecting on a proxy loss that does not track the
reward — the eleven tasks are aggregated with an equal-weight mean, so a
per-task loss average is *not* a good proxy for it.

## What you cannot evaluate here

`fit` and `tune` are labeled and yours to use. The held-out eval split is not
mounted in this container: its inputs and labels live only in the verifier and
grader environments. Scores on `tune` are therefore an estimate, and `tune` is
small for some tasks — `yeast_ppi` has 95 rows, `human_ppi` 315, `ppi_affinity`
203 — so treat single-task movements on those as noisy and prefer the aggregate
when deciding what to keep.
