# Development evaluation

Scoring is the verifier's own code (`qecbench.scoring` in `/workspace/baseline/alphaqubit`, on
`PYTHONPATH`). Validation is yours to inspect: select hyperparameters and checkpoints on
training and validation only.

```bash
cd /workspace/baseline/alphaqubit
# a label-free validation input view in the contract's format
python -m qecbench.stage_validation --source /workspace/data/validation --output /workspace/research/validation-inputs
bash /workspace/submission/infer.sh /workspace/research/validation-inputs /workspace/research/validation-predictions
python -m qecbench.scoring --input /workspace/research/validation-inputs \
  --predictions /workspace/research/validation-predictions --labels /workspace/data/validation \
  --config /workspace/baseline/alphaqubit/score_config.json --validation \
  --output /workspace/research/validation-score.json
```

`/workspace/data/reference/results/validation_report.json` is the reference model's validation
report under the same scorer.
