# B2 — STATE zero-shot (ST, gene space)

The STATE paper's held-out-context setting with the pinned `arc-state` (9bbfe78), in the
configuration STATE's own VCC recipe uses: the ST set-to-set transformer reads and predicts
log1p CP10k expression on the common gene panel, with ESM-2 perturbation features. The
SE-600M-latent variant (ST in SE space + SE counts decoder) is not shipped because the pinned
arc-state hard-codes the decoder's SE checkpoint paths (`tx/models/decoders.py`); the shipped SE
embeddings are for the agent's own models.

1. `prepare_data.py` writes one h5ad per training context (`.X` = log1p CP10k on the genes
   common to the listed contexts, `obsm["X_state"]` = shipped SE-600M embeddings, `obs.gene`,
   `obs.cell_type`, `obs.gem_group`), ESM-2 perturbation features for every gene, and
   `zeroshot.toml`.
2. `train.sh [steps]` runs `state tx train` with `model=state_sm`, `output_space=all`, ESM-2
   perturbation features (so withheld targets get a real feature). 20k steps ≈ 1.5 h on one H100.
3. `infer.sh IN OUT` creates `n_cells` virtual copies of basal cells per request, runs
   `state tx infer`, and writes count-scale predictions on the input panel (genes outside the
   model's panel keep the basal cell's value = no effect).

Work in a copy under `/workspace/research`: `st_data/` is about 51 GB for the four contexts and
`runs/` about 2 GB, too much for the container's own filesystem.

```bash
cp -r /workspace/baseline/state_zeroshot /workspace/research/b2 && cd /workspace/research/b2
python prepare_data.py --data /workspace/data --contexts k562 jurkat hct116 hek293t
bash train.sh 20000
```

LOCO: `python prepare_data.py --contexts <three contexts>` then `train.sh`, then
`evaluation/run_loco.py --package . --holdout <fourth>`. The published package needs
`runs/zeroshot/checkpoints/final.ckpt` and `st_data/output_genes.json`; `st_data/*.h5ad` and
`last.ckpt` are training-only (a 1.6 GB package instead of about 53 GB):

```bash
python /workspace/baseline/publish.py /workspace/research/b2 \
  --exclude 'runs/*/checkpoints/last.ckpt' --check-input /workspace/data/test/ctx_far/public
```

Reproduced numbers: [RESULTS.md](RESULTS.md).
