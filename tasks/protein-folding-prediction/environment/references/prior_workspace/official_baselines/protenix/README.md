# Protenix official-code 50M baseline

This baseline registers a scaled config in a pinned upstream Protenix checkout.
It does not copy or reimplement Pairformer, MSA/template processing, all-atom
diffusion, permutation matching, losses, confidence, checkpointing, or
inference. Those paths execute directly from upstream revision
`c3bfc365b3e1341a11935eddfe7bfdc308092147`.

The config starts from the official `protenix_tiny_default_v0.5.0` recipe and
changes only channel dimensions. The verified folding-model parameter count is
48,681,151. External sequence embedding models are not included in this count.

The complete one-step training/inference/scoring smoke has passed. See
[`RESULTS.md`](RESULTS.md) for the exact budget and real sampled FoldBench
per-track results. Zero scores in that file are expected from a one-step model;
they validate the evaluator path, not model quality.

Run the source/provenance audit in Docker:

```bash
docker run --rm --network none \
  -e PYTHONPATH=/workspace:/official -e LAYERNORM_TYPE=torch \
  -v /mnt/localssd/${USER}/workspaces/protein-folding:/workspace:ro \
  -v /mnt/localssd/${USER}/workspaces/references/Protenix:/official:ro \
  -w /official \
  ai4s-share-public-cn-beijing.cr.volces.com/release/protenix:1.0.0.4 \
  python -m official_baselines.protenix.verify
```

Training and inference use `official_baselines.protenix.train` and
`official_baselines.protenix.inference`, respectively. Every remaining CLI
argument is parsed by the upstream runner. Training should mount only the
training-data root; FoldBench ground truth must be mounted only into the
separate scoring container.

Training also requires `--unified-native-view-audit <path>`, explicit
`--data.train_sets`, and an explicit native index for every train set. The
wrapper verifies each index SHA-256 against an audit produced by
`audit_native_view.py`; an unaudited default/paper index is rejected before the
upstream trainer starts.

Training examples come from the same frozen FoldingTrain split used by every
other baseline. The Protenix native-data preparation is a schema adapter; its
original-paper data mixture is not part of the synchronized Recipe. The smoke
selector now requires `unified_split=train` so validation structures cannot be
selected accidentally.
The adapter applies the released weighted-PDB ion exclusion before writing its
audit, so `accepted_examples` is the exact row count visible to the upstream
loader rather than a pre-filter count.
Validation and FoldBench inference use a declared 2,048-token capability
limit. Larger full assemblies are reported as unsupported; they are never
silently cropped for scoring.

The controlled campaign disables protein and RNA MSA plus templates. Upstream
still eagerly opens the configured RNA sequence index while RNA MSA is off, so
`no_msa_empty_index.json` is passed as that index. It is an explicit empty
sentinel—not an MSA feature source—and prevents the no-MSA route from requiring
the unrelated `rna_msa` download.

Contract glue is deliberately kept outside the model:

- `build_foldbench_inputs.py` converts the coordinate-free common input to the
  current upstream Protenix JSON schema and expands assembly copy suffixes.
- `postprocess_foldbench.py` applies FoldBench's official Protenix CIF
  normalization and emits `prediction_reference.csv`.
- `build_foldbench_target_subset.py` creates an auditable sampled target set.
