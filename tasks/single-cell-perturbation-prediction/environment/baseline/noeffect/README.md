# B0: no effect

Resamples `n_cells` basal cells per request, unchanged. It is the normalisation baseline of the
score: every normalised metric of this package is 0 by construction, so it checks the contract and
the scoring pipeline, not a model. No training step.

```bash
python /workspace/evaluation/check_submission.py --package /workspace/baseline/noeffect \
  --input /workspace/data/test/ctx_far/public
```

Results: [RESULTS.md](RESULTS.md).
