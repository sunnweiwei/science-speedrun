## Task and goal

Develop a useful geospatial representation from scratch using the supplied
reduced public pretraining dataset. The same frozen encoder must serve African
Wildlife Foundation (AWF) land-cover classification and live fuel moisture
content (LFMC) regression.

The final reward is
`0.5 * (AWF_validation_accuracy + max(0, 1 - LFMC_test_MAE / max(constant_test_MAE, 1e-12)))`.
Accuracy is a fraction, and `constant_test_MAE` is the test MAE of the LFMC
training-label mean predictor. During development, use AWF validation accuracy
and LFMC validation MAE; LFMC test scoring is reserved for final evaluation.
AWF has no independent test split in this release.

## Workspace

- `/workspace/contract.md`: the portable encoder package and inference contract.
- `/workspace/data/`: the data guide, provenance and read-only pretraining and
  downstream train/validation data.
- `/workspace/baseline/starter/`: starter code, commands in `README.md` and
  recorded validation evidence in `RESULTS.md`.
- `/workspace/evaluation/`: development scoring with the fixed frozen probes.
- `/workspace/submission/`: the current complete encoder package.
- `/workspace/research/`: writable space for derived data, checkpoints,
  experiments and research notes.

## Rules

You may change encoder size, training objective, auxiliary targets, schedule and
implementation efficiency while preserving the inference contract and recording
those changes. WorldCover maps and moisture targets derived from the declared
pretraining imagery may supervise representation learning. Preserve the declared
geographic exclusions and training-data provenance. Write derived data under
`/workspace/research`; do not modify mounted datasets. Data preparation is
CPU-only. The workspace has no network access; use its installed runtime and
provided data.

Do not initialize or supervise any candidate model component using public
pretrained model weights. Official fixed normalization statistics are permitted.
Use downstream training labels only for frozen probe fitting and validation
labels only for model/probe selection. AWF and LFMC labels must not supervise
the encoder or training-only heads. Test data are off limits: never use LFMC
test labels for representation learning, hyperparameter selection, early
stopping or feature selection.

Keep the official splits, full validation populations, input preprocessing,
missingness handling, pooling, probe grids and primary metrics fixed. Do not
replace spatial splits with random row splits or reduce held-out rows to speed
up evaluation. Compare the same selected encoder on both tasks with an
architecture-matched random initialization, raw spectral features and trivial
predictors under the same evaluation settings. Report uncertainty, selection
history and secondary-metric regressions honestly. Validation gains alone do
not establish generalization to an untouched test set.

## Submission

Keep a complete package that follows `/workspace/contract.md` at
`/workspace/submission` from early in development. Stage each replacement in a
separate directory, check that it contains all required files and passes a short
inference check, then publish the complete package atomically. Preserve the
previous complete submission until its replacement is ready; do not write
individual checkpoint or configuration files into the live submission. Use
`/workspace/baseline/starter/publish.py` for the checked atomic publication
procedure documented in the baseline README.

## Continuous research

Keep improving the model and replacing the submission when you have a better
candidate. There is no end point and no final answer to hand in.

The starter uses a Large encoder with 6,000 WorldCover updates followed by 1,000
updates of WorldCover plus NDMI mean/standard-deviation prediction with
newest-first inputs. Its recorded scores are 77.034884% AWF validation accuracy
and 21.459503 LFMC validation MAE from one adaptively selected training seed.
These are existing validation measurements, not a reproduction in this agent
container or a measured final reward. Read `baseline/starter/RESULTS.md` for the
evidence and limitations. Check your own pipeline with a short run, then pursue
improvements; a full baseline retraining solely to confirm those numbers is not
required.
