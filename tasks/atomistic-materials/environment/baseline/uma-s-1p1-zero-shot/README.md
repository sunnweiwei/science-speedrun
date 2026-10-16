# Baseline: UMA-S-1.1 as released (no training)

The starting checkpoint itself, evaluated through the same contract: `calculator.py` loads
`model.pt` with FAIRChem's `omat` task and default inference settings.

```bash
mkdir -p /workspace/research/zero-shot
cp /workspace/data/models/uma-s-1p1.pt /workspace/research/zero-shot/model.pt
cp calculator.py /workspace/research/zero-shot/
# score every validation suite with the commands in /workspace/contract.md
```

Its `omat` head is trained on OMat24, whose energies are not on the MP scale. The stability suite
applies MP2020 corrections to those energies anyway, so its discovery numbers in `RESULTS.md` show
what an un-adapted model gets there. They are not a bug in the evaluation.
