# stochastic-tsr

Run the frozen release’s 60-step stochastic TSR sampler with its default 180-pass front-loaded recycle schedule, independently for each slot. Final quality and diversity are measured with separate AF3 and Foldseek calls. RESULTS.md and reproduction/ contain the local development measurements.

```bash
cd /workspace
bash baseline/publish.sh baseline/stochastic-tsr
run=$(mktemp -d /workspace/research/stochastic-tsr.XXXXXX)
python evaluation/run.py --work "$run/work" --output "$run/score.json"
```

Add `--paper-size` to the last command for 50 slots per length on public seeds.
