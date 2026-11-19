# Two-wide geometry beam

Keep two survivors per independent group. At steps 20 and 40, each survivor branches twice. The native sampler keeps the two strongest clean predictions among the four children, ranked by the fraction of clashing atoms. Each output slot consumes 2 × 20 × 2 + 2 × 20 × 2 + 20 × 1 = 180 trunk passes. This control tests repeated pruning with more than one surviving trajectory.

```bash
cd /workspace
bash baseline/publish.sh baseline/two-wide-beam
run=$(mktemp -d /workspace/research/two-wide-beam.XXXXXX)
python evaluation/run.py --work "$run/work" --output "$run/score.json"
```

Use `--paper-size` for the public 150-slot protocol. RESULTS.md lists the locally measured cohorts.
