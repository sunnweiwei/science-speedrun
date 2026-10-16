# Baseline: hold-current anchor

A protocol-conforming trivial policy: `V = R I` holds the coil currents. It is the "no control"
anchor. It loses the plasma in every sealed episode, so its
`reward` is 0. A randomly initialized actor also loses the plasma within a few dozen steps.

```bash
python /workspace/evaluation/harbor_verify.py --submission /workspace/baseline/hold_current --output /workspace/research/hold-current-check --workers 6
```

Results: [RESULTS.md](RESULTS.md).
