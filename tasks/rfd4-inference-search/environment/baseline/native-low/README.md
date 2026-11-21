# Native Low monomer baseline

The shared executable starter uses the released low front-loaded preset (120 trunk passes per output). The package builder copies the preset and the inference program into a relocatable directory. [Public measurements](RESULTS.md)

```bash
cd /workspace
candidate=$(mktemp -d /workspace/research/native-low.XXXXXX)
python evaluation/reference.py native-low "$candidate"
bash baseline/publish.sh "$candidate"
run=$(mktemp -d /workspace/research/native-low-check.XXXXXX)
python evaluation/run.py --work "$run/work" --output "$run/score.json"
```
