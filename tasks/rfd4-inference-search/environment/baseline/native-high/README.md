# Native High monomer baseline

The shared executable starter uses the released high front-loaded preset (300 trunk passes per output). The package builder copies the preset and the inference program into a relocatable directory. [Public measurements](RESULTS.md)

```bash
cd /workspace
candidate=$(mktemp -d /workspace/research/native-high.XXXXXX)
python evaluation/reference.py native-high "$candidate"
bash baseline/publish.sh "$candidate"
run=$(mktemp -d /workspace/research/native-high-check.XXXXXX)
python evaluation/run.py --work "$run/work" --output "$run/score.json"
```
