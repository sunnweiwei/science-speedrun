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
ordering; evaluation replays conforming input directories with different row
counts and shuffled/swapped pair orientations.

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
  `human_ppi`. Scores are normalized to probabilities by dividing by their sum,
  and the argmax is the predicted class.
- Regression tasks (`fluorescence`, `stability`, `beta_lactamase`,
  `ppi_affinity`) use `{"value":0.0}` with a finite number.
- `secondary_structure` uses `{"ss3_scores":[[...],[...],...]}` — one 3-value
  score vector per residue, list length exactly the length of the input
  `sequence`, each vector finite, nonnegative, positive-sum.

A malformed, missing, duplicate, or unknown prediction zeros only that
top-level task (`skill = 0.0`).

## Scoring

Each task is converted into a chance-calibrated signed skill `v_k` matching
`ssi_index_v3.sequence()`:

- **Classification tasks**: `v_k = (balanced_accuracy - 1/K) / (1 - 1/K)` with
  `K = 2` for `solubility`, `binary_localization`, `yeast_ppi`, and `human_ppi`,
  `K = 10` for `subcellular_localization`, and `K = 136` for `fold` (accuracy
  and macro-F1 are also reported). Although `fold` outputs 1,195 SCOPe class
  scores, `balanced_accuracy` averages per-class recall over classes present in
  the ground truth, and the held-out `test_fold_holdout` split contains `136`
  observed classes with positive support; thus `1/136` is the exact constant/uniform
  chance baseline on the evaluation split.
- **Symmetric undirected PPI evaluation (`yeast_ppi`, `human_ppi`)**: physical
  protein-protein interaction is undirected. The label-free verifier evaluates
  every test pair in both `(sequence_a, sequence_b)` and `(sequence_b, sequence_a)`
  orientations and computes `p_sym = 0.5 * (p_ab + p_ba)` before argmax
  thresholding and `balanced_accuracy` computation. Raw `p_ab`, `p_ba`, and
  `asymmetry_mae` are retained in `report.json` as diagnostics.
- **`secondary_structure`**: `v_k = (accuracy - 0.4306464618) / (1 - 0.4306464618)`,
  where accuracy counts only residues where ground-truth `valid_mask` is set
  pooled over all residues, and `0.4306464618` is the CB513 majority-class coil
  baseline.
- **`fluorescence`, `stability`, `beta_lactamase`**: signed Spearman rank
  correlation `v_k = spearman_rho`.
- **`ppi_affinity`**: `v_k = 1 - rmse / 3.083206134` on the original label scale
  (calibrated against the constant-mean test RMSE anchor `3.083206134`; RMSE and
  Spearman ρ are also reported).
- **Four-pillar aggregation**: tasks are averaged within four equal-weight (`1/4`)
  capability pillars:
  - `Function`: `("fluorescence", "stability", "beta_lactamase", "solubility")`
  - `Localization`: `("subcellular_localization", "binary_localization")`
  - `Structure`: `("fold", "secondary_structure")`
  - `Protein interactions`: `("yeast_ppi", "human_ppi", "ppi_affinity")`

  Let `signed_score` be the mean of the four pillar scores. The primary scalar
  `score` (and `reward` alias) is `clip(signed_score, 0.0, 1.0)`.

The trusted grader writes both a nested `report.json` and a flat
`reward.json` (containing both `"score"` and `"reward"`). The nested report
contains every pillar, task, and diagnostic metric; the flat file repeats all
scalar numeric metrics using underscore-separated names.
