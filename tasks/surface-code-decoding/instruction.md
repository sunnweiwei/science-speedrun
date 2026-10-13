# Decode logical errors in a quantum memory

## Task and goal

Develop a model that predicts whether the logical observable flipped during a Sycamore
surface-code memory experiment. Each example contains packed detection events across repeated
stabilizer-measurement rounds, the ideal circuit geometry and an opaque sample ID. The target is
the official logical-path observable flip, a binary variable. Better decoding lowers the logical
error rate, helping quantum processors preserve encoded information despite physical noise.

The primary performance P is `1 - balanced_logical_failure`, computed in three steps:

1. average the 13 round-count error fractions within each geometry/basis group;
2. average the groups within each distance;
3. give d3 and d5 equal weight.

The reward is P itself.

| decoder | P (reward) |
| --- | ---: |
| MWPM | 0.71605 |
| independent AlphaQubit reproduction, trained from scratch (baseline) | 0.75569 |
| tensor network (released predictions rescored on the same shots; a reference, not a ceiling) | 0.76152 |

Logical error per round, error
suppression and calibration are reported separately. Optimize the full benchmark distribution
rather than dropping conditions or reporting only the easiest duration or patch.

## Workspace

- `/workspace/contract.md`: the input format (`surface-code-decoding-input/v1`), the output format,
  the package and the inference limits.
- `/workspace/data/`:
  - `README.md` describes the data;
  - `train/` holds 2,584,400 shots and `validation/` holds 665,600 shots. Each covers the d3 and
    d5 patches, X/Z memory bases and odd round counts 1 through 25, per condition with its ideal
    circuit and an even-fitted detector error model for streaming Stim simulation;
  - `reference/` holds the reference AlphaQubit's results and a weightless package template.

  All of these are read-only.
- `/workspace/baseline/alphaqubit/`: the independent AlphaQubit reproduction, with its complete
  training code, packaging and scoring library, recipe and reproduced results. It is a Python
  project on `PYTHONPATH`.
- `/workspace/references/`: the anchors and the published work behind them.
- `/workspace/evaluation/`: scoring on validation with the verifier's own code.
- `/workspace/research/`: writable space for models, simulations and experiments.

## Rules

- **No trained weights are provided.** Every model you submit is trained from random
  initialization in this session. You may reproduce the reference, improve it or build something
  else. You may change the architecture, features, loss, training schedule, simulation,
  calibration or distillation, or combine learned and matching decoders.
- The even-fitted detector error models use all even syndromes, including validation syndromes,
  but no test syndromes or logical labels.
- Do not use any of the following:
  - pretrained or external checkpoints;
  - historical test records;
  - released per-shot decoder answers;
  - test labels;
  - models trained or selected on the test split.

  Do not search the public upstream dataset for test answers. The published historical test is
  public; runtime isolation does not make it a new secret dataset.
- Select hyperparameters and checkpoints on training and validation only.
- There is no network.

## Submission

Keep a self-contained model at `/workspace/submission`:

- `infer.sh`, `README.md`, `model_manifest.json`, `model_config.json`, `training_provenance.json`
  and `requirements.lock`;
- code, and trained weights or serialized decoder state.

Only this directory survives into evaluation. `infer.sh INPUT_ROOT OUTPUT_ROOT` must infer on
the provided examples; precomputed prediction files are not a model submission. Relative file
paths in the manifest must stay valid after relocation. Record actual initialization, training
data and selection history. The runtime lock records dependencies, and the verifier does not
install them.

For each input condition, write one `<condition-name>.npz` with exactly `sample_ids: uint64[N]` and
`probabilities: float32[N]`:

- the IDs match the input exactly, without duplicates; output order may differ;
- probabilities are finite, in [0, 1], and refer to observable flips; the decision is
  `probability >= 0.5`;
- the decoder must support arbitrary N, order and subsets within supported conditions, and must not
  infer original row indices from IDs.

The package may be read at any moment, so replace it as a whole:
`python -m qecbench.package ... --output /workspace/submission` assembles elsewhere and swaps it in
with one rename. Publish a first trained package early.

## Continuous research

The baselines have already been trained and scored in this setup: `RESULTS.md` in each
`/workspace/baseline/<name>/` has their numbers and, where present, `reproduction/` their
training logs and validation outputs. Do not retrain a baseline to confirm its numbers; check your
own pipeline with a short run instead.

There is no end point. Analyze where the current decoder fails (by distance, basis, geometry and
round count), form a hypothesis, train and validate a candidate, and publish it when it is better.
Then start the next experiment. Do not stop to hand in a final answer. A published model or a
reached score is a milestone, not the end of the task.
