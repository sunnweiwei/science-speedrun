# Train the weights of a fruit-fly neuron reconstruction pipeline

## Task and goal

Develop the trained weights of a PATHFINDER-style pipeline that reconstructs
neurons from public 16 nm electron microscopy of the male fruit-fly central
nervous system (MaleCNS). Automated segmentation errors (splits, merges and
omissions) are the main cost of building connectomes. The pipeline grows
segments with one FFN, proposes merges with SENSE and accepts them with SHAPE;
its architectures and inference protocol are fixed, so progress comes from
training. The package is judged by local sparse normalized expected run length
(nERL, higher is better) of the SHAPE-merged reconstruction of a held-out
region. Report split/merge errors, omissions and model coverage alongside nERL.

## Workspace

- `/workspace/contract.md`: the model package, its fixed architectures and inference limits.
- `/workspace/data/`: the data guide and read-only training and validation data (image, reference
  suite, trusted runtime).
- `/workspace/baseline/pathfinder-fast/`: commands in `README.md`, measured results in `RESULTS.md`
  and an editable copy of the reference implementation.
- `/workspace/references/`: pointers to papers, source pins and unverified extensions.
- `/workspace/evaluation/`: development evaluation on the validation region and its runtime.
- `/workspace/submission/`: the current complete model package.
- `/workspace/research/`: writable space for derived data, checkpoints, experiments and notes.

Read `/workspace/contract.md` before exporting a package. The current native
baseline uses one FFN checkpoint, ascending seed order, no consensus, shared
SENSE proposals and disjoint-pair SHAPE merging. The independent verifier uses this same fixed protocol.

## Rules

Use only supplied training data for training and development data for selection.
Test labels and references are off limits for model development. Do not use
evaluation GT to choose seeds, fragments, candidates, scores or merge decisions.
Public availability does not grant permission to train on held-out labels.
Preserve the frozen MaleCNS train/validation/test regions and shared-ID exclusions.
Unknown tissue and unlabelled neurons are not negative labels.

Retain the supplied FFN, full128 SENSE and 8192-point/10µm SHAPE architectures.
Report any protocol reduction explicitly. Negative and zero gains are valid
outcomes. Do not equate local sparse-reference scores with official PATHFINDER
paper results or move scores between datasets. There is no network access,
and no training happens during inference. Write derived data, caches and outputs
under `/workspace/research`; do not modify mounted datasets.

## Submission

Keep a complete `pathfinder-model-package/v1` at `/workspace/submission` from
early in development. Export it with the baseline's `run.sh export` (weights,
configuration and file hashes; see the contract) into a separate directory, then
swap it in atomically with `run.sh publish`. The verifier uses trusted inference
code and native extensions; submitted executable code is not evaluated, and
predictions, cached answers, scores and reports are not model submissions.

## Continuous research

Keep improving the model and the current package. Use development evaluation
to identify failures, compare methods on identical data, retain reproducible
commands and provenance, and replace the package when a better valid model is
available. There is no final hand-in or stopping point within the research run.

The baseline in `/workspace/baseline/pathfinder-fast` has already been trained and
evaluated; its measured results are in its `RESULTS.md`. Its weights are not supplied,
so a trained package needs your own training run. Check your pipeline with a short
run, then work on improvements rather than retraining the baseline recipe only to
confirm its numbers.
