# PEER protein modeling contract

The input schema is `peer-protein-input/v1`. There are eleven task files:
`fluorescence`, `stability`, `beta_lactamase`, `solubility`,
`subcellular_localization`, `binary_localization`, `fold`,
`secondary_structure`, `yeast_ppi`, `human_ppi`, and `ppi_affinity`.
Development uses `fit/<task>.jsonl` and `tune/<task>.jsonl`; evaluation uses
`<task>.jsonl`.

Every row has exactly `example_id`, `task`, and `features`; development rows
also have `label`. For the pair tasks (`yeast_ppi`, `human_ppi`,
`ppi_affinity`), `features` is `{"sequence_a": "...", "sequence_b": "..."}`;
for all other tasks it is `{"sequence": "..."}`. Sequences are uppercase
amino-acid strings and may contain nonstandard letters (`X`, `U`, `B`, `Z`,
`O`). IDs are opaque. Do not hard-code the visible row counts, IDs, or
ordering; evaluation may use a different set with the same schema.

Submissions must use one shared learned backbone or representation across all
eleven tasks, with task-specific heads as needed. Storage format and
checkpoint topology are unrestricted. The submission README must identify the
shared component, the joint/multitask training procedure, and the head for
every task. Independent per-task models without a learned shared component are
out of scope.

## Permitted training resources

Three sources, all mounted read-only under `/workspace/data`:

- `fit/<task>.jsonl` — labeled supervised data (219,325 examples / 262,360
  sequences).
- `tune/<task>.jsonl` — labeled data for model selection (23,395 examples /
  24,008 sequences).
- `pretrain/uniref50_subset.fasta` — unlabeled UniRef50 for self-supervised
  pretraining: every cluster of 30–1024 residues, already decontaminated
  against the held-out eval sequences at 50% identity.

All learned parameters must be initialized randomly and trained in this
session. **External pretrained protein models or embeddings (ESM, ProtT5,
ProtBERT, Ankh, any checkpoint trained elsewhere) are not permitted**, and
neither is downloading additional sequence data — the environment is offline
and the mounted corpus is the whole training pool. Inference runs offline with
no network access.

## Location

The package is the directory `/workspace/submission`. It may be read at any moment, so replace it
as a whole: assemble the new package elsewhere and swap it in with one rename
(`/workspace/baseline/starter/publish.py` does this after checking the package the way the
verifier will).

## Invocation

The verifier calls `bash infer.sh INPUT_ROOT OUTPUT_ROOT` from the submission
directory. `INPUT_ROOT` contains the eleven `<task>.jsonl` input files. Create
eleven `OUTPUT_ROOT/predictions/<task>.jsonl` files. Every line has exactly:

```json
{"example_id":"...","prediction":{}}
```

## Prediction formats

- Classification tasks use `{"scores":[...]}` with 2/10/2/1195/2/2 finite
  nonnegative values with positive sum for `solubility`,
  `subcellular_localization`, `binary_localization`, `fold`, `yeast_ppi`, and
  `human_ppi`. The argmax is the predicted class.
- Regression tasks (`fluorescence`, `stability`, `beta_lactamase`,
  `ppi_affinity`) use `{"value":0.0}` with a finite number.
- `secondary_structure` uses `{"ss3_scores":[[...],[...],...]}` — one 3-value
  score vector per residue, list length exactly the length of the input
  `sequence`, each vector finite, nonnegative, positive-sum.

A malformed, missing, duplicate, or unknown prediction zeros only that
top-level task.

## Scoring

- Classification reward is accuracy (balanced accuracy and macro-F1 are also
  reported). `secondary_structure` accuracy counts only residues where the
  ground-truth `valid_mask` is set, pooled over all residues.
- `fluorescence`, `stability`, and `beta_lactamase` reward is
  `max(spearman_rho, 0)`.
- `ppi_affinity` reward is `exp(-RMSE / fit_standard_deviation)` on the
  original label scale; RMSE and Spearman ρ are also reported.
- The primary reward is the equal-weight mean of all eleven task rewards.

The trusted grader writes both a nested `report.json` and a flat
`reward.json`. The nested report contains every task and metric. The flat file
repeats all numeric metrics using underscore-separated names so orchestration
summaries can expose more than the primary reward.
