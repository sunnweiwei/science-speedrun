# Reproduction record: OMatG starter

The development evaluation behind `../RESULTS.md`. The starter has already been trained and scored;
these files are its record, so there is no need to train it again to see what it does.

The model is the OMatG starter trained 2,000 epochs from scratch on six H100s (5.4 h), with the
best-`val_loss_total` checkpoint near epoch 700 published (`model.safetensors` SHA-256
`e298e7a6…940b`). The training log of that run was not kept. One unconditional request of
2,500 structures was generated, relaxed and scored as `/workspace/evaluation/score.sh` does, with
reward 0.208 (SUN + MetaSUN). The verifier's request uses a different, private seed.

| file | content |
| --- | --- |
| `requests.jsonl` | the request: 2,500 unconditional samples, seed 20260919 |
| `generated_2500.tgz` | the package's output: `output/manifest.csv` and `output/structures/*.cif` |
| `relaxed_2500.tgz` | the UMA-200 relaxation: `relaxation/generated/` (relaxed CIFs and manifest), `relaxation/samples/NNNNNN.json` (per sample: convergence, steps, energies, fallback), `summary.json`, `inventory.json` |
| `score_report.json` | the LeMat-GenBench report on the relaxed set: the counts (valid, unique, novel, stable, metastable, SUN, MetaSUN) and every upstream benchmark result (validity, distribution, diversity, novelty, uniqueness, HHI, SUN, stability) |

Extract the archives under `/workspace/research` with `tar xzf`.
