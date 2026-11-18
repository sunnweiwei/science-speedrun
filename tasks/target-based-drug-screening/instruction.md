# Continuously improve a target-based drug-screening model

## Task and goal

Develop a learned model that ranks candidate small molecules for a protein
binding pocket. Better early retrieval helps prioritize compounds for
experimental screening. The primary score (`score` / `reward` in `reward.json`)
is BEDROC(alpha=80.5), averaged across targets within DUD-E and LIT-PCBA, then
averaged equally between the two datasets:

    score = 0.5 * mean_DUDE(BEDROC_80.5) + 0.5 * mean_LIT_PCBA(BEDROC_80.5)

It emphasizes early retrieval and ranges from 0 to 1; higher is better. Both
datasets must be complete (102 DUD-E and 15 LIT-PCBA targets). ROC-AUC and EF
at 0.5%, 1%, 2%, and 5% are reported separately for diagnosis. This is a
ranking score, not accuracy or the fraction of the task solved.

Keep one shared model package and one fixed learned parameter set for both
benchmarks and all targets. The supplied DrugCLIP reference has 95,305,418
parameters; this task does not impose a 50M parameter limit. You may change
the architecture, representations, objectives, sampling, optimizer, training
recipe, and inference method.

## Workspace

- /workspace/contract.md: portable model package and inference interface.
- /workspace/data/: read-only ProFSA pretraining and BioLiP train/dev LMDBs,
  split IDs, and the fixed Uni-Mol initialization assets. README.md describes
  the records and permitted uses.
- /workspace/baseline/drugclip/: the dual Uni-Mol implementation, short-run
  commands, and RESULTS.md with the reproduced experiments and their scope.
- /workspace/references/minimal-starter/: a small atom-histogram interface
  example. It demonstrates packaging, not strong screening performance.
- /workspace/evaluation/: BioLiP development retrieval evaluation and the
  screening metric implementation. Hidden benchmark data are not supplied.
- /workspace/research/: writable experiments, logs, checkpoints, and packages
  waiting to be published.
- /workspace/publish.py: validate and atomically publish a complete package.
- /workspace/submission/: your current independently replayable candidate.

## Rules

- Learn from the supplied ProFSA and BioLiP training records. Use BioLiP dev
  for evaluation and checkpoint selection; do not train on dev records.
  Derived training features, augmentations, and targets belong in research/.
- You may use the supplied, checksum-pinned generic Uni-Mol molecule
  checkpoint for initialization. Its dictionaries are provided alongside it.
  The pocket encoder in the reference starts randomly. Other pretrained
  weights, including author DUD-E/LIT-PCBA benchmark checkpoints, are not
  allowed. You may resume your own experiments.
- DUD-E and LIT-PCBA are held-out tests. Do not acquire their inputs, labels,
  benchmark-trained weights, prediction caches, or answer tables. Do not fit
  parameters during inference or use target-specific learned checkpoints.
- Develop a general predictor that accepts unseen conforming pocket and
  molecule records. IDs, input ordering, and file paths are not labels.
- There is no network. Do not download additional data or model assets.
- Keep mounted data unchanged. Write all derived artifacts under research/.

## Submission

Publish a working candidate early. First use a short training run to check
your data, optimization, development evaluation, package loading, and replay.
The reference's train.sh exports an experiment package under research/.
Publish it into submission with:

    python /workspace/publish.py /workspace/research/candidate

The package must follow contract.md, including infer.sh, owned learned
weights, configuration, a model manifest, and a concise README explaining
training provenance and limitations. Inference receives label-free records
and must score every ligand. It must work without the training workspace or
original Uni-Mol checkpoint.

Keep only the selected inference package in submission/. Store optimizer
states, earlier candidates, and ongoing experiments in research/. Publish
each improved complete candidate atomically with publish.py.

## Continuous research

The dev-selected epoch-14 DrugCLIP reference has independently verified
full-suite reward 0.166385 under this definition (mean EF1% 16.108 is also
reported). Its training recipe, 34-epoch
development curve, replay evidence, and limitations are in RESULTS.md and
reproduction/. Its trained weights are not candidate initialization. Start
from the supplied generic Uni-Mol weights and check your own pipeline with
a short run before scaling up.

Repeat hypothesis formation, implementation, training, BioLiP development
evaluation, error analysis, and publication of the best supported candidate.
Explore scientific improvements rather than only reproducing a launcher.
Keep improving the model and the current submission. A completed experiment,
a published checkpoint, or a reached score is a milestone. Do not stop to
hand in a final answer or declare the research complete.
