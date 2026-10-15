# Continuously improve one unified protein sequence model

## Task and goal

Design, train, evaluate and improve a single model across all eleven PEER protein sequence
understanding tasks:

- function prediction: fluorescence, stability, β-lactamase;
- localization: subcellular and binary;
- structure: fold and secondary structure;
- solubility;
- protein-protein interaction: yeast, human, affinity.

Evaluation replays your frozen package offline on the PEER test splits. The reward is the
equal-weight mean of the eleven task rewards. Improve it across tasks, not only in the mean.

Predictions for all eleven tasks must depend substantially on the same jointly learned
parameters. Eleven independently trained task models behind a common interface do not satisfy
the objective. These are all fine:

- shared components with task-specific heads;
- task or instruction conditioning;
- input-dependent computation inside one jointly trained model.

| system | reward |
| --- | ---: |
| trivial fixed-statistic control | 0.2874 |
| ESM-2 8M starter baseline (pretrained and fine-tuned from scratch here) | 0.6695 / 0.6774 |

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
  - `trivial/` is the control. A model not clearly above 0.2874 has learned nothing.
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
- handle unseen conforming inputs.

Keep a concise README in the package: the shared backbone, exact parameter count, from-scratch
provenance, data used, model-selection policy and known limitations.

Publish a usable candidate early; the first trained starter checkpoint is enough. Train into
`/workspace/research` and swap the package in atomically with `baseline/starter/publish.py`,
because `/workspace/submission` may be read at any moment. Keep research history, checkpoints
and resume state under `/workspace/research`.

## Continuous research

The baselines have already been trained and scored in this setup: `RESULTS.md` in each
`/workspace/baseline/<name>/` has their numbers and, where present, `reproduction/` their
training logs and validation outputs. Do not retrain a baseline to confirm its numbers; check your
own pipeline with a short run instead.

This is a continuing research task, not a finite deliverable. Repeatedly:

1. establish or update a measured reference and form a hypothesis;
2. implement the change: architecture, objective, data mixture, tokenization, task weighting,
   optimizer, schedule, model selection or inference-time computation;
3. train from scratch, or resume only checkpoints from your own candidate lineage
   (`train.py --checkpoint-dir ... --resume ...`; `--skip-pretrain` reuses a pretrained encoder
   for a new downstream recipe);
4. evaluate on `tune` and analyze every affected task, not just the aggregate;
5. publish the strongest complete, compliant, independently replayable candidate whenever it
   improves the evidence;
6. immediately begin the next experiment.

Do not return a final answer or declare the task complete. A finished training run, a published
model, an empty todo list or a reached score is only a milestone.

Questions worth investigating:

- How to split effort between self-supervised pretraining and supervised training, and at what
  model size. A 5× larger model is roughly 2.5× slower per token.
- Whether the eleven tasks help or interfere with each other. They count equally, and `fold` is
  far behind everything else.
- What the pooled representation throws away. The starter mean-pools and, for pair tasks,
  concatenates two pooled sequences.
- Sequences are truncated to 1024 tokens, so long chains lose their tail, while
  `secondary_structure` is scored per residue over the whole chain.
