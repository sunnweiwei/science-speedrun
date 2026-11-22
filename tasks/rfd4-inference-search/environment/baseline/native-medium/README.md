# Native Medium baseline

The shared executable starter uses the released medium recycling preset across all four conditioned design tracks (`medium/front_loaded` for protein and DNA binding, `medium/midpeak` for small-molecule binding and enzyme scaffolding, and `medium/front_loaded` for unconditional monomers). The package builder copies the preset and the unified `infer.py` entry point into a relocatable submission directory. [Measured results](RESULTS.md)

```bash
cd /workspace
candidate=$(mktemp -d /workspace/research/native-medium.XXXXXX)
python evaluation/reference.py native-medium "$candidate"
bash baseline/publish.sh "$candidate"
run=$(mktemp -d /workspace/research/native-medium-check.XXXXXX)
python evaluation/evaluate_tracks.py --submission "$candidate" --method-name cand-medium \
  --panel dev --attempts 20 --work "$run/work" --output "$run/report"
```
