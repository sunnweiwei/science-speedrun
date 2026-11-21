# Greedy geometry branching (beam width one)

Run 60 stochastic TSR steps with two trunk passes each. At step 30 score the clean estimate of each of two siblings by steric clashes and keep the better one. Each returned slot costs 2 × 30 × 2 + 30 × 2 = 180 trunk passes. Final quality and diversity are measured with separate AF3 and Foldseek calls. RESULTS.md and reproduction/ contain the local development measurements.

```bash
cd /workspace
candidate=$(mktemp -d /workspace/research/geometry-beam-package.XXXXXX)
python evaluation/reference.py geometry-beam "$candidate"
bash baseline/publish.sh "$candidate"
run=$(mktemp -d /workspace/research/geometry-beam.XXXXXX)
python evaluation/run.py --work "$run/work" --output "$run/score.json"
```

Add `--paper-size` to the last command for 50 slots per length on public seeds.
