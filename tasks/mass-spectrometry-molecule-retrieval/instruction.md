# Improve molecular retrieval from tandem mass spectra

## Task and goal

Develop a model that ranks a supplied set of candidate molecular structures for each
experimental MS/MS spectrum of MassSpecGym v1.5. The same trained model must handle every test
query and unseen conforming queries.

Hit rate at 1 on the 17,556 test queries is the primary objective. Hit rates at 5 and 20 and
MCES distance at 1 are secondary. The reward is the raw test Hit@1.

| system | Hit@1 (reward) |
| --- | ---: |
| Fourier-DeepSets starter | 0.0400 |
| native GLACIER trained from scratch, with the DeepSets fallback (baseline) | 0.4045 |
| official-weight GLACIER/MIST hybrid (reference; its weights are not available to you) | 0.4939 |

The reference is reproduced, not an optimum. Aim for `hit_rate_at_1 > 0.4939052175894281` while
monitoring Hit@5, Hit@20 and MCES@1.

## Workspace

- `/workspace/contract.md`: the package, the ranking interface, replay limits and metrics.
- `/workspace/data/`:
  - `README.md` describes the data;
  - the MassSpecGym v1.5 training and validation folds with candidate pools;
  - `glacier/`: train/validation-only MAGMa fragment supervision;
  - `synthetic_augmentation/`: 2,333,485 held-out-clean public ICEBERG simulations, optional
    training data available equally to every agent.

  All of these are read-only.
- `/workspace/baseline/starter/`: verified reference implementations with their reproduced
  results:
  - the Fourier-DeepSets starter;
  - `glacier/`, the native GLACIER forward simulator trained from random initialization.

  The native GLACIER source is at `/opt/ms-pred` and its pinned runtime at
  `/opt/glacier/bin/python`.
- `/workspace/references/`: the official-weight target, the SOTA and protocol audits, and the
  author's reproduction scripts, not re-run here.
- `/workspace/evaluation/`: scoring and diagnostics on validation.
- `/workspace/research/`: writable space for experiments, checkpoints and records.

## Rules

- Use training labels for fitting and validation labels for model selection and error analysis.
  Do not fit parameters on validation examples or encode their answers in the model.
- Test inputs and labels are not available. Do not obtain, reconstruct or hard-code test
  answers. The upstream benchmark is public, so this is a benchmark-visible evaluation, not a
  previously secret test set. Candidate SMILES are legitimate model inputs at inference time.
- The only operator-supplied pretrained assets are the ones documented as such, and none are
  supplied in this release. Published GLACIER weights are not authorized initialization assets.
- Deliver one fixed package that runs on the verifier's single GPU. These are forbidden:
  - query-specific fitted parameters;
  - true test formulae;
  - candidate-position shortcuts;
  - answer tables.
- Everything you may use is here: research data and runtime dependencies are supplied locally.
  There is no network.

Directions worth trying:

- train a forward spectrum simulator with train-only negatives;
- extend a forward simulator to every adduct and instrument condition;
- replace the fallback;
- train a formula-free joint embedding;
- distill candidate simulation into a fast reranker;
- use another hybrid.

## Submission

Keep a complete package at `/workspace/submission`, starting early: `infer.sh`,
`model_manifest.json`, `README.md`, model code, configuration, weights and every runtime asset
not supplied by the verifier image. It must run without network, the live trainer, development
data or an undeclared cache, within the replay limits in the contract. First make the starter
path train, publish, relocate and replay successfully.

The package may be read at any moment, so switch it atomically, as the starter's versioned
publication does. Keep `EXPERIMENTS.md` in the package: each hypothesis, changed factor, seed,
measured compute, validation query-set hash, checkpoint hash and result, including failed or
negative experiments. Preserve the selected training source, configuration and logs in the
package's `research/` directory. Select using validation retrieval metrics, and compare
uncertainty using molecule groups. Validation scores are not evidence of an improvement over the
reference's test score.

## Continuous research

The baselines have already been trained and scored in this setup: `RESULTS.md` in each
`/workspace/baseline/<name>/` has their numbers and, where present, `reproduction/` their
training logs and validation outputs. Do not retrain a baseline to confirm its numbers; check your
own pipeline with a short run instead.

There is no end point. Analyze where the current model fails (by adduct, instrument, collision
energy and molecule group), form a hypothesis, train and validate a candidate, and publish it when
it is better. Then start the next experiment. Do not stop to hand in a final answer. A published
model or a reached score is a milestone, not the end of the task.
