# Baseline: trivial fixed-statistic control

A predictor that trains nothing: per-task constants (regression means, majority classes) from
`/workspace/data/stats.json`. It is the output-format demo and the control. It scores 0.2874,
so a model that does not clearly beat it has learned nothing.

```bash
cd /workspace/baseline/trivial
python3 build_stats.py                           # copies /workspace/data/stats.json next to predict.py
stage=/workspace/research/trivial-package; rm -rf $stage; mkdir -p $stage
cp infer.sh predict.py stats.json $stage/ && echo "Trivial control." > $stage/README.md
python3 /workspace/baseline/starter/publish.py $stage --to /workspace/submission
```
