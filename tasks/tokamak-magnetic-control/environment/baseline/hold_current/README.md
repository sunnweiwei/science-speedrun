# Baseline: hold-current anchor

A protocol-conforming trivial policy: `V = R I` holds the coil currents. It is the "no control"
anchor. Under v0.2 it loses the plasma in all 14 sealed episodes (seeds 1001-1002, seven cases)
after 29-117 steps, so its `reward` is 0. Under v0.1 (seeds 1001–1003) it lost every episode after
48–362 steps (the task author's run, artifacts checked by the integrator). A randomly initialized
actor loses the plasma after 15–52 steps.

```bash
python /workspace/evaluation/harbor_verify.py --submission /workspace/baseline/hold_current --output /workspace/research/hold-current-check --workers 6
```

Results: [RESULTS.md](RESULTS.md).
