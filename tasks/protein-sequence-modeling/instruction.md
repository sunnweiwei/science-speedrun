# Continuously improve one unified protein sequence model

## Task and goal

Design, train, evaluate and improve a single model across all eleven PEER protein sequence
understanding tasks grouped into four equal-weight (`1/4`) capability pillars:

- **Function (`1/4`)**: `fluorescence`, `stability`, `beta_lactamase`, `solubility`;
- **Localization (`1/4`)**: `subcellular_localization`, `binary_localization`;
- **Structure (`1/4`)**: `fold`, `secondary_structure`;
- **Protein interactions (`1/4`)**: `yeast_ppi`, `human_ppi`, `ppi_affinity`.

Evaluation replays your frozen package offline on the held-out PEER test splits. Each task is
scored by a chance-calibrated signed skill `v_k` matching the SSI V3 benchmark index:

- Classification tasks (`solubility` `K=2`, `subcellular_localization` `K=10`,
  `binary_localization` `K=2`, `fold` `K=136`, `yeast_ppi` `K=2`, `human_ppi` `K=2`):
  `v_k = (balanced_accuracy - 1/K) / (1 - 1/K)`. Note that `fold` emits 1,195 SCOPe fold logits,
  while the held-out `test_fold_holdout` split contains `136` observed classes with positive test
  support; because `balanced_accuracy` averages recall over classes present in ground truth, any
  constant or uniform predictor achieves `1/136`, making `K=136` the exact zero-skill baseline.
- `secondary_structure`: `v_k = (accuracy - 0.4306464618) / (1 - 0.4306464618)` (calibrated against
  the CB513 3-state coil majority fraction).
- `ppi_affinity`: `v_k = 1 - rmse / 3.083206134` (calibrated against the constant-mean test RMSE
  anchor).
- Fitness/stability regressions (`fluorescence`, `stability`, `beta_lactamase`): signed Spearman
  rank correlation `v_k = spearman_rho`.

For the two undirected physical interaction classification tasks (`yeast_ppi` and `human_ppi`),
the verifier evaluates every test pair in **both input orientations** `(sequence_a, sequence_b)`
and `(sequence_b, sequence_a)` and computes `balanced_accuracy` from the symmetrized positive
probability `p_sym = 0.5 * (p_ab + p_ba)`.

Each pillar score is the arithmetic mean of its member task skills `v_k`, `signed_score` is the
equal-weight mean of the four pillars, and the primary `score` (`reward`) is
`clip(signed_score, 0.0, 1.0)`.

Predictions for all eleven tasks must depend substantially on the same jointly learned
parameters. Eleven independently trained task models behind a common interface do not satisfy
the objective. These are all fine:

- shared components with task-specific heads;
- task or instruction conditioning;
- input-dependent computation inside one jointly trained model (including symmetric pair pooling
  or pair cross-attention for undirected PPI tasks).

| system | V4/V3 4-pillar score | notes |
| --- | ---: | --- |
| trivial fixed-statistic control | 0.0002 | chance-calibrated near zero (legacy uncalibrated 11-task mean: 0.2874) |
| ESM-2 8M starter baseline | see `/workspace/baseline/starter/` | pretrained + jointly fine-tuned from scratch (legacy 11-task mean: 0.6695 / 0.6774) |

## Workspace

- `/workspace/contract.md`: the delivery interface the verifier replays, and the rules on
  permitted training resources.
- `/workspace/data/`:
  - `README.md` describes the data;
  - `fit/` holds labeled training rows;
  - `tune/` holds labeled rows for model selection;
  - `pretrain/uniref50_subset.fasta` is unlabeled UniRef50, decontaminated against the held-out
    sequences at 50% identity (how it was built: `pretrain/manifest.json`);
  - `stats.json` holds fit-split label statistics.

  All of these are read-only.
- `/workspace/baseline/`: verified reference implementations with their results:
  - `starter/` pretrains an ESM-2 encoder, fine-tunes it jointly on all eleven tasks, selects on
    `tune`, and packages the model. The published size ladder from 8M to 650M is in `esm2.py`.
  - `trivial/` is the control. A model near `0.0` skill has learned nothing above constant chance.
- `/workspace/references/`: further measured runs (a 37M generic transformer, throughput per
  model size) and model specifications.
- `/workspace/evaluation/`: score yourself on `tune` with the grader's own metric.
- `/workspace/research/`: writable space for experiments, checkpoints and resume state.

## Rules

- All learned parameters must originate from random initialization and training in this
  session. Do not use external pretrained protein models or embeddings: ESM, ProtT5, ProtBERT,
  Ankh, or any checkpoint trained elsewhere. `esm2.py` is an architecture, not a checkpoint.
- The mounted data are the entire training pool. `fit` and `tune` are yours to use however you
  like, including as additional pretraining text. The held-out split is not here.
- Do not fit parameters at test time, and do not build ID or sequence lookup tables keyed to
  evaluation inputs.
- There is no network.

## Submission

`/workspace/submission` is the currently selected replay package, never an experiment archive.
At every publication it contains exactly one candidate's inference-time parameters, plus only the
code, configuration and assets needed to replay it. The entry point is `infer.sh`, documented in
the contract. It must:

- load the package independently of the live trainer;
- use one parameter set for every task;
- handle unseen conforming inputs (including arbitrary row counts and swapped pair orientations).

Keep a concise README in the package: the shared backbone, exact parameter count, from-scratch
provenance, data used, model-selection policy and known limitations.

Publish a usable candidate early; the first trained starter checkpoint is enough. Train into
`/workspace/research` and swap the package in atomically with `baseline/starter/publish.py`,
because `/workspace/submission` may be read at any moment. Keep research history, checkpoints
and resume state under `/workspace/research`.

## Continuous research

The baselines have already been trained and scored in this setup: `RESULTS.md` in each
`/workspace/baseline/<name>/` records their per-task measurements and, where present,
`reproduction/` their training logs and validation outputs. Do not retrain a baseline to confirm
its numbers; check your own pipeline with a short run instead.

This is a continuing research task, not a finite deliverable. Repeatedly:

1. establish or update a measured reference and form a hypothesis;
2. implement the change: architecture, objective, data mixture, tokenization, task weighting,
   optimizer, schedule, model selection or inference-time computation;
3. train from scratch, or resume only checkpoints from your own candidate lineage
   (`train.py --checkpoint-dir ... --resume ...`; `--skip-pretrain` reuses a pretrained encoder
   for a new downstream recipe);
4. evaluate on `tune` and analyze every affected task and capability pillar, not just the aggregate;
5. publish the strongest complete, compliant, independently replayable candidate whenever it
   improves the evidence;
6. immediately begin the next experiment.

Do not return a final answer or declare the task complete. A finished training run, a published
model, an empty todo list or a reached score is only a milestone.

Questions worth investigating:

- How to split effort between self-supervised pretraining and supervised training, and at what
  model size. A 5× larger model is roughly 2.5× slower per token.
- Whether the eleven tasks help or interfere with each other across the four capability pillars,
  and how class-imbalanced tasks respond to balanced-accuracy optimization (`fold` stays far
  behind most tasks).
- What the pooled representation throws away. The starter mean-pools and, for pair tasks,
  concatenates `[pool(sequence_a), pool(sequence_b)]` asymmetrically rather than using an
  order-invariant or symmetric pair architecture.
- Sequences are truncated to 1024 tokens, so long chains lose their tail, while
  `secondary_structure` is scored per residue over the whole chain.
